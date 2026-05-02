#!/usr/bin/env python3
"""
Compute theoretical Operational Intensity (OI) for nn.Conv2d layers.

Target use:
  IEEE Access resubmission Experiment A preprocessing on Jetson Orin Nano.

Examples:
  python3 compute_theoretical_oi.py --model resnet18 --batch 8
  python3 compute_theoretical_oi.py --model efficientnet_b7 --batch 8
  python3 compute_theoretical_oi.py --model all --batch 8 --output-dir ./oi_csv

Notes:
  - This script calculates Theoretical OI only.
  - Measured OI must be calculated later from Nsight Compute DRAM traffic.
  - BatchNorm, activation, pooling, linear layers, and bias are excluded.
  - FP32 is assumed: 4 bytes per element.
  - Weight elements are counted once per batch; cache reuse is not modeled.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_RIDGE_POINT = 32.37
BYTES_PER_ELEMENT_FP32 = 4
SUPPORTED_TARGET_MODELS = ("resnet18", "efficientnet_b7")

# Keep a reference alive if the local environment needs the torchvision::nms
# import workaround. This is not normally used on Jetson; it only runs after a
# failed torchvision import caused by a mismatched torchvision build.
_TORCHVISION_LIBRARY_PATCH = None


CSV_COLUMNS = [
    "model_name",
    "target_batch_size",
    "layer_index",
    "layer_name",
    "conv_type_basic",
    "layer_type_for_experiment",
    "input_shape",
    "output_shape",
    "N",
    "Cin",
    "Cout",
    "Hin",
    "Win",
    "Hout",
    "Wout",
    "Kh",
    "Kw",
    "stride_h",
    "stride_w",
    "padding_h",
    "padding_w",
    "dilation_h",
    "dilation_w",
    "groups",
    "weight_elements",
    "input_elements",
    "output_elements",
    "theoretical_FLOPs",
    "theoretical_memory_bytes",
    "theoretical_OI_FLOPs_per_byte",
    "ridge_point_used",
    "roofline_class_theoretical",
    "notes",
]


@dataclass(frozen=True)
class ShapeInfo:
    """Input/output tensor shape for one Conv2d layer."""

    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    source: str


def _clean_failed_torchvision_import() -> None:
    """Remove partially imported torchvision modules after an import failure."""

    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            del sys.modules[name]


def _install_torchvision_nms_import_workaround() -> None:
    """
    Work around local CPU/container environments where torchvision import fails
    before model creation with: operator torchvision::nms does not exist.

    The generated script should normally run on Jetson without this path, but the
    workaround keeps the script testable in environments with a torchvision build
    mismatch. It defines only the operator schema needed for import-time fake-op
    registration; it does not implement or use NMS.
    """

    global _TORCHVISION_LIBRARY_PATCH

    if _TORCHVISION_LIBRARY_PATCH is not None:
        return

    schema = "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"

    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define(schema)
        _TORCHVISION_LIBRARY_PATCH = lib
        return
    except Exception:
        pass

    try:
        lib = torch.library.Library("torchvision", "FRAGMENT")
        try:
            lib.define(schema)
        except Exception:
            # The schema may already be defined. Keeping the library reference is
            # harmless and avoids garbage-collection surprises.
            pass
        _TORCHVISION_LIBRARY_PATCH = lib
    except Exception:
        # The follow-up import will raise the original/real error if this did not
        # help, which is clearer for the user than failing here.
        return


def import_torchvision_models():
    """Import torchvision.models, with a narrow fallback for one known issue."""

    try:
        from torchvision import models

        return models
    except RuntimeError as exc:
        message = str(exc)
        if "torchvision::nms" not in message and "operator torchvision" not in message:
            raise

        _clean_failed_torchvision_import()
        _install_torchvision_nms_import_workaround()
        from torchvision import models

        return models


def create_model(model_name: str) -> nn.Module:
    """Create a torchvision model with randomly initialized weights."""

    models = import_torchvision_models()

    if not hasattr(models, model_name):
        available_hint = ", ".join(SUPPORTED_TARGET_MODELS)
        raise ValueError(
            f"torchvision.models has no model named '{model_name}'. "
            f"Expected one of: {available_hint}, or another torchvision model name."
        )

    factory = getattr(models, model_name)

    # Newer torchvision API.
    try:
        return factory(weights=None)
    except TypeError:
        pass

    # Older torchvision API.
    try:
        return factory(pretrained=False)
    except TypeError:
        pass

    # Fallback for custom factories with no weight arguments.
    return factory()


def normalize_hw(value: Any, name: str) -> Tuple[int, int]:
    """Normalize int/list/tuple kernel-size-like values to (h, w)."""

    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Could not parse {name}={value!r} as a 2-D parameter")


def tensor_shape_4d(x: Any) -> Tuple[int, int, int, int]:
    """Return a 4-D tensor shape tuple from hook input/output objects."""

    if isinstance(x, torch.Tensor):
        shape = tuple(int(v) for v in x.shape)
    elif isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        shape = tuple(int(v) for v in x[0].shape)
    else:
        raise TypeError(f"Expected a Tensor or a tuple/list containing a Tensor, got {type(x)!r}")

    if len(shape) != 4:
        raise ValueError(f"Expected a 4-D NCHW tensor, got shape={shape}")
    return shape  # type: ignore[return-value]


def fallback_output_hw(
    hin: int,
    win: int,
    kh: int,
    kw: int,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
) -> Tuple[int, int]:
    """Conv2d output spatial-size formula."""

    hout = math.floor((hin + 2 * padding_h - dilation_h * (kh - 1) - 1) / stride_h + 1)
    wout = math.floor((win + 2 * padding_w - dilation_w * (kw - 1) - 1) / stride_w + 1)
    return int(hout), int(wout)


def collect_conv_shapes(
    model: nn.Module,
    input_size: int,
    shape_batch: int,
    device: torch.device,
) -> Dict[str, ShapeInfo]:
    """Run a forward pass with hooks and collect Conv2d input/output shapes."""

    shapes: Dict[str, ShapeInfo] = {}
    hooks = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, hook_input: Any, hook_output: Any) -> None:
            in_shape = tensor_shape_4d(hook_input)
            out_shape = tensor_shape_4d(hook_output)
            shapes[layer_name] = ShapeInfo(in_shape, out_shape, "forward_hook")

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval().to(device)
        with torch.no_grad():
            dummy = torch.zeros(shape_batch, 3, input_size, input_size, device=device)
            _ = model(dummy)
    finally:
        for hook in hooks:
            hook.remove()
        model.to(torch.device("cpu"))

    return shapes


def classify_conv(module: nn.Conv2d) -> Tuple[str, str]:
    """Return (conv_type_basic, layer_type_for_experiment)."""

    cin = int(module.in_channels)
    cout = int(module.out_channels)
    groups = int(module.groups)
    kh, kw = normalize_hw(module.kernel_size, "kernel_size")

    is_depthwise = groups == cin and cout % cin == 0
    is_pointwise = (kh, kw) == (1, 1) and groups == 1

    if is_depthwise:
        basic = "depthwise"
        experiment = "depthwise"
    elif is_pointwise:
        basic = "pointwise"
        if cout > cin:
            experiment = "pointwise_expand"
        elif cout < cin:
            experiment = "pointwise_project"
        else:
            experiment = "pointwise_unknown"
    elif groups > 1:
        basic = "grouped"
        experiment = "grouped"
    else:
        basic = "standard"
        experiment = "standard"

    return basic, experiment


def classify_roofline(theoretical_oi: float, ridge_point: float) -> str:
    """Classify a layer using the pre-fixed theoretical roofline thresholds."""

    if theoretical_oi < 0.8 * ridge_point:
        return "memory_bound"
    if theoretical_oi <= 1.25 * ridge_point:
        return "transition"
    return "compute_bound"


def format_shape(shape: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(v)) for v in shape) + ")"


def infer_row_for_conv(
    model_name: str,
    target_batch_size: int,
    layer_index: int,
    layer_name: str,
    module: nn.Conv2d,
    shape_info: Optional[ShapeInfo],
    input_size: int,
    ridge_point: float,
) -> Dict[str, Any]:
    """Calculate all CSV fields for one Conv2d layer."""

    cin = int(module.in_channels)
    cout = int(module.out_channels)
    groups = int(module.groups)
    kh, kw = normalize_hw(module.kernel_size, "kernel_size")
    stride_h, stride_w = normalize_hw(module.stride, "stride")
    padding_h, padding_w = normalize_hw(module.padding, "padding")
    dilation_h, dilation_w = normalize_hw(module.dilation, "dilation")

    notes = []

    if shape_info is not None:
        _, hook_cin, hin, win = shape_info.input_shape
        _, hook_cout, hout, wout = shape_info.output_shape
        if hook_cin != cin:
            notes.append(f"warning_hook_Cin={hook_cin}_module_Cin={cin}")
        if hook_cout != cout:
            notes.append(f"warning_hook_Cout={hook_cout}_module_Cout={cout}")
        notes.append("shape_source=forward_hook")
    else:
        # Conservative fallback. For normal torchvision models this should not
        # be used; hooks should see every Conv2d layer in the forward pass.
        hin = input_size
        win = input_size
        hout, wout = fallback_output_hw(
            hin,
            win,
            kh,
            kw,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        )
        notes.append("shape_source=fallback_formula_assuming_input_image_size")

    n = int(target_batch_size)
    input_shape_target = (n, cin, int(hin), int(win))
    output_shape_target = (n, cout, int(hout), int(wout))

    cin_per_group = cin // groups
    theoretical_flops = int(2 * n * cout * int(hout) * int(wout) * cin_per_group * kh * kw)

    weight_elements = int(module.weight.numel())
    input_elements = int(n * cin * int(hin) * int(win))
    output_elements = int(n * cout * int(hout) * int(wout))
    theoretical_memory_bytes = int((input_elements + weight_elements + output_elements) * BYTES_PER_ELEMENT_FP32)
    theoretical_oi = float(theoretical_flops) / float(theoretical_memory_bytes) if theoretical_memory_bytes else float("nan")

    basic, experiment_type = classify_conv(module)
    roofline_class = classify_roofline(theoretical_oi, ridge_point)

    return {
        "model_name": model_name,
        "target_batch_size": n,
        "layer_index": layer_index,
        "layer_name": layer_name,
        "conv_type_basic": basic,
        "layer_type_for_experiment": experiment_type,
        "input_shape": format_shape(input_shape_target),
        "output_shape": format_shape(output_shape_target),
        "N": n,
        "Cin": cin,
        "Cout": cout,
        "Hin": int(hin),
        "Win": int(win),
        "Hout": int(hout),
        "Wout": int(wout),
        "Kh": kh,
        "Kw": kw,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "padding_h": padding_h,
        "padding_w": padding_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "groups": groups,
        "weight_elements": weight_elements,
        "input_elements": input_elements,
        "output_elements": output_elements,
        "theoretical_FLOPs": theoretical_flops,
        "theoretical_memory_bytes": theoretical_memory_bytes,
        "theoretical_OI_FLOPs_per_byte": f"{theoretical_oi:.10f}",
        "ridge_point_used": f"{ridge_point:.10f}",
        "roofline_class_theoretical": roofline_class,
        "notes": ";".join(notes),
    }


def compute_model_rows(
    model_name: str,
    target_batch_size: int,
    input_size: int,
    shape_batch: int,
    ridge_point: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Create model, collect shapes, and compute all Conv2d rows."""

    model = create_model(model_name)
    shapes = collect_conv_shapes(model, input_size=input_size, shape_batch=shape_batch, device=device)

    rows: List[Dict[str, Any]] = []
    conv_index = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        row = infer_row_for_conv(
            model_name=model_name,
            target_batch_size=target_batch_size,
            layer_index=conv_index,
            layer_name=name,
            module=module,
            shape_info=shapes.get(name),
            input_size=input_size,
            ridge_point=ridge_point,
        )
        rows.append(row)
        conv_index += 1

    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, Any]], output_path: str) -> str:
    oi_values = [float(row["theoretical_OI_FLOPs_per_byte"]) for row in rows]
    basic_counts: Dict[str, int] = {}
    for row in rows:
        key = str(row["conv_type_basic"])
        basic_counts[key] = basic_counts.get(key, 0) + 1

    standard_count = basic_counts.get("standard", 0)
    depthwise_count = basic_counts.get("depthwise", 0)
    pointwise_count = basic_counts.get("pointwise", 0)
    grouped_count = basic_counts.get("grouped", 0)

    if oi_values:
        min_oi = min(oi_values)
        max_oi = max(oi_values)
        mean_oi = statistics.mean(oi_values)
    else:
        min_oi = max_oi = mean_oi = float("nan")

    return "\n".join(
        [
            f"Output CSV: {output_path}",
            f"number of Conv2d layers: {len(rows)}",
            f"number of standard layers: {standard_count}",
            f"number of depthwise layers: {depthwise_count}",
            f"number of pointwise layers: {pointwise_count}",
            f"number of grouped layers: {grouped_count}",
            f"min theoretical_OI: {min_oi:.6f} FLOPs/byte",
            f"max theoretical_OI: {max_oi:.6f} FLOPs/byte",
            f"mean theoretical_OI: {mean_oi:.6f} FLOPs/byte",
        ]
    )


def sanity_check_rows(model_name: str, rows: List[Dict[str, Any]]) -> List[str]:
    """Return warning strings. Empty list means the basic checks passed."""

    warnings: List[str] = []

    for row in rows:
        flops = int(row["theoretical_FLOPs"])
        mem = int(row["theoretical_memory_bytes"])
        oi = float(row["theoretical_OI_FLOPs_per_byte"])
        expected_oi = flops / mem if mem else float("nan")
        if not math.isclose(oi, expected_oi, rel_tol=1e-9, abs_tol=1e-9):
            warnings.append(f"{row['layer_name']}: theoretical_OI does not match FLOPs/memory_bytes")

        expected_mem = (
            int(row["input_elements"]) + int(row["weight_elements"]) + int(row["output_elements"])
        ) * BYTES_PER_ELEMENT_FP32
        if int(row["theoretical_memory_bytes"]) != expected_mem:
            warnings.append(f"{row['layer_name']}: theoretical_memory_bytes is not input+weight+output times 4")

        if row["conv_type_basic"] == "depthwise" and int(row["groups"]) != int(row["Cin"]):
            warnings.append(f"{row['layer_name']}: depthwise row does not have groups == Cin")

        if row["conv_type_basic"] == "pointwise" and (int(row["Kh"]) != 1 or int(row["Kw"]) != 1):
            warnings.append(f"{row['layer_name']}: pointwise row does not have Kh == Kw == 1")

    if model_name == "resnet18":
        conv1 = next((row for row in rows if row["layer_name"] == "conv1"), None)
        if conv1 is None:
            warnings.append("resnet18 conv1 was not found")
        else:
            expected = {
                "Cin": 3,
                "Hin": 224,
                "Win": 224,
                "Kh": 7,
                "Kw": 7,
                "stride_h": 2,
                "stride_w": 2,
                "padding_h": 3,
                "padding_w": 3,
                "Hout": 112,
                "Wout": 112,
            }
            for key, value in expected.items():
                if int(conv1[key]) != value:
                    warnings.append(f"resnet18 conv1: expected {key}={value}, got {conv1[key]}")

    return warnings


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute theoretical OI for Conv2d layers in torchvision models."
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        help="Model name: resnet18, efficientnet_b7, all, or another torchvision.models name.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Target batch size N used for FLOPs and memory calculations. Default: 8.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image height/width. Default: 224.",
    )
    parser.add_argument(
        "--shape-batch",
        type=int,
        default=1,
        help="Batch size used only for the forward hook shape extraction pass. Default: 1.",
    )
    parser.add_argument(
        "--ridge-point",
        type=float,
        default=DEFAULT_RIDGE_POINT,
        help="Measured ridge point in FLOPs/byte. Default: 32.37.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output CSV files. Default: current directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Device for the one dummy forward pass. Default: cpu.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help=(
            "CPU thread count for the shape-extraction forward pass. "
            "Default: 1, which avoids OpenMP oversubscription on small devices."
        ),
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip built-in validation warnings.",
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    return torch.device(device_arg)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    if args.shape_batch <= 0:
        raise ValueError("--shape-batch must be positive")
    if args.ridge_point <= 0:
        raise ValueError("--ridge-point must be positive")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")

    torch.set_num_threads(args.num_threads)
    try:
        torch.set_num_interop_threads(args.num_threads)
    except RuntimeError:
        # PyTorch only allows setting interop threads before parallel work starts.
        # In normal script execution this should not trigger; ignore if embedded.
        pass

    if args.model == "all":
        model_names = list(SUPPORTED_TARGET_MODELS)
    else:
        model_names = [args.model]

    device = resolve_device(args.device)

    for model_name in model_names:
        rows = compute_model_rows(
            model_name=model_name,
            target_batch_size=args.batch,
            input_size=args.input_size,
            shape_batch=args.shape_batch,
            ridge_point=args.ridge_point,
            device=device,
        )

        output_path = os.path.join(args.output_dir, f"{model_name}_theoretical_oi_b{args.batch}.csv")
        write_csv(rows, output_path)

        print("=" * 72)
        print(summarize_rows(rows, output_path))

        if not args.skip_sanity_check:
            warnings = sanity_check_rows(model_name, rows)
            if warnings:
                print("Sanity check: WARN")
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print("Sanity check: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
