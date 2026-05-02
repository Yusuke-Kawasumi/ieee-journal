#!/usr/bin/env python3
"""
Run exactly one torchvision nn.Conv2d layer for Nsight Compute profiling.

This script is for Experiment A single-layer profiling. It intentionally does
not run the full model, BatchNorm, activation, pooling, or Linear layers.

Typical use under ncu:
  sudo -E /usr/local/cuda/bin/ncu \
    --target-processes all \
    --profile-from-start off \
    --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
    --csv --log-file logs/ncu_csv/resnet18_conv1.csv \
    --export profiles/ncu_reports/resnet18_conv1 --force-overwrite \
    python3 scripts/profile_single_layer.py \
      --model resnet18 --layer-name conv1 --batch 8 --warmup 50 --iters 1 \
      --theoretical-csv input/resnet18_theoretical_oi_b8.csv \
      --use-profiler-api
"""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

SUPPORTED_MODELS = ("resnet18", "efficientnet_b7")
_TORCHVISION_LIBRARY_PATCH = None


def _clean_failed_torchvision_import() -> None:
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            del sys.modules[name]


def _install_torchvision_nms_import_workaround() -> None:
    """Narrow import workaround for environments with a broken torchvision::nms registration."""
    global _TORCHVISION_LIBRARY_PATCH
    if _TORCHVISION_LIBRARY_PATCH is not None:
        return

    schema = "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    for kind in ("DEF", "FRAGMENT"):
        try:
            lib = torch.library.Library("torchvision", kind)
            try:
                lib.define(schema)
            except Exception:
                pass
            _TORCHVISION_LIBRARY_PATCH = lib
            return
        except Exception:
            continue


def import_torchvision_models():
    try:
        from torchvision import models
        return models
    except RuntimeError as exc:
        msg = str(exc)
        if "torchvision::nms" not in msg and "operator torchvision" not in msg:
            raise
        _clean_failed_torchvision_import()
        _install_torchvision_nms_import_workaround()
        from torchvision import models
        return models


def create_model(model_name: str) -> nn.Module:
    models = import_torchvision_models()
    if not hasattr(models, model_name):
        raise ValueError(
            f"torchvision.models has no model named {model_name!r}. "
            f"Expected one of: {', '.join(SUPPORTED_MODELS)}."
        )
    factory = getattr(models, model_name)
    try:
        return factory(weights=None)
    except TypeError:
        pass
    try:
        return factory(pretrained=False)
    except TypeError:
        pass
    return factory()


def get_conv_layer(model: nn.Module, layer_name: str) -> nn.Conv2d:
    modules = dict(model.named_modules())
    module = modules.get(layer_name)
    if module is None:
        conv_names = [name for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)]
        preview = ", ".join(conv_names[:20])
        more = "" if len(conv_names) <= 20 else f" ... ({len(conv_names)} conv layers total)"
        raise KeyError(f"Layer {layer_name!r} was not found. Conv2d candidates: {preview}{more}")
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"Layer {layer_name!r} exists, but it is {type(module).__name__}, not nn.Conv2d")
    return module


def normalize_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected int or length-2 tuple/list, got {value!r}")


def parse_shape(shape_text: str) -> Tuple[int, int, int, int]:
    text = str(shape_text).strip()
    if not text:
        raise ValueError("Empty shape string")
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (tuple, list)) and len(value) == 4:
            return tuple(int(v) for v in value)  # type: ignore[return-value]
    except Exception:
        pass
    parts = [p.strip() for p in text.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")]
    parts = [p for p in parts if p]
    if len(parts) != 4:
        raise ValueError(f"Could not parse shape {shape_text!r}; expected N,C,H,W")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def default_theoretical_csv_candidates(model_name: str, batch: int) -> List[str]:
    filename = f"{model_name}_theoretical_oi_b{batch}.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        filename,
        os.path.join("input", filename),
        os.path.join("experiment_A", "input", filename),
        os.path.join(script_dir, "..", "input", filename),
    ]


def resolve_theoretical_csv(path_arg: Optional[str], model_name: str, batch: int) -> Optional[str]:
    candidates: List[str] = []
    if path_arg:
        candidates.append(path_arg)
    candidates.extend(default_theoretical_csv_candidates(model_name, batch))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def read_input_shape_from_theoretical_csv(csv_path: str, layer_name: str, batch: int) -> Tuple[int, int, int, int]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("layer_name") == layer_name:
                shape = parse_shape(row.get("input_shape", ""))
                if shape[0] != batch:
                    shape = (int(batch), int(shape[1]), int(shape[2]), int(shape[3]))
                return shape
    raise KeyError(f"Layer {layer_name!r} was not found in theoretical CSV: {csv_path}")


def cuda_profiler_call(method_name: str) -> str:
    """Call cudaProfilerStart/cudaProfilerStop and return the backend used."""
    errors: List[str] = []

    try:
        cudart = torch.cuda.cudart()
        func = getattr(cudart, method_name)
        ret = func()
        if ret is None:
            return "torch.cuda.cudart"
        if isinstance(ret, tuple):
            code = int(ret[0])
        else:
            code = int(ret)
        if code == 0:
            return "torch.cuda.cudart"
        errors.append(f"torch.cuda.cudart returned {code}")
    except Exception as exc:
        errors.append(f"torch.cuda.cudart failed: {exc}")

    for lib_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so.10.2"):
        try:
            lib = ctypes.CDLL(lib_name)
            func = getattr(lib, method_name)
            ret = int(func())
            if ret == 0:
                return f"ctypes:{lib_name}"
            errors.append(f"{lib_name} returned {ret}")
        except Exception as exc:
            errors.append(f"{lib_name} failed: {exc}")

    joined = " | ".join(errors)
    raise RuntimeError(f"Could not call {method_name}: {joined}")


def conv_metadata(module: nn.Conv2d) -> Dict[str, Any]:
    kh, kw = normalize_pair(module.kernel_size)
    sh, sw = normalize_pair(module.stride)
    ph, pw = normalize_pair(module.padding)
    dh, dw = normalize_pair(module.dilation)
    return {
        "Cin": int(module.in_channels),
        "Cout": int(module.out_channels),
        "Kh": kh,
        "Kw": kw,
        "stride_h": sh,
        "stride_w": sw,
        "padding_h": ph,
        "padding_w": pw,
        "dilation_h": dh,
        "dilation_w": dw,
        "groups": int(module.groups),
        "bias": bool(module.bias is not None),
        "weight_shape": [int(v) for v in module.weight.shape],
    }


def parse_bool_text(value: str) -> bool:
    value_l = value.strip().lower()
    if value_l in ("1", "true", "yes", "y", "on"):
        return True
    if value_l in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean text, got {value!r}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Conv2d layer for ncu profiling.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--layer-name", required=True, help="Exact Conv2d layer name from theoretical OI CSV.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--input-shape", default=None, help="Explicit N,C,H,W, e.g. 8,64,56,56")
    parser.add_argument("--theoretical-csv", default=None, help="Path to theoretical OI CSV.")
    parser.add_argument("--use-profiler-api", action="store_true")
    parser.add_argument("--cudnn-benchmark", type=parse_bool_text, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the Jetson CUDA environment.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.set_grad_enabled(False)

    theoretical_csv = resolve_theoretical_csv(args.theoretical_csv, args.model, args.batch)
    if args.input_shape:
        input_shape = parse_shape(args.input_shape)
        if input_shape[0] != args.batch:
            input_shape = (args.batch, input_shape[1], input_shape[2], input_shape[3])
    else:
        if theoretical_csv is None:
            searched = default_theoretical_csv_candidates(args.model, args.batch)
            raise FileNotFoundError(
                "--input-shape was not provided and theoretical CSV was not found. "
                f"Searched: {searched}"
            )
        input_shape = read_input_shape_from_theoretical_csv(theoretical_csv, args.layer_name, args.batch)

    model = create_model(args.model)
    model.eval()
    layer = get_conv_layer(model, args.layer_name).eval().to("cuda")

    # Allocate after moving the layer so cudnn/cuda setup happens outside the profiled region.
    x = torch.randn(*input_shape, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = layer(x)
        torch.cuda.synchronize()

        profiler_backend_start = "not_used"
        profiler_backend_stop = "not_used"
        if args.use_profiler_api:
            profiler_backend_start = cuda_profiler_call("cudaProfilerStart")

        y = None
        for _ in range(args.iters):
            y = layer(x)

        torch.cuda.synchronize()

        if args.use_profiler_api:
            profiler_backend_stop = cuda_profiler_call("cudaProfilerStop")

    assert y is not None
    output_shape = tuple(int(v) for v in y.shape)

    metadata: Dict[str, Any] = {
        "script": "profile_single_layer.py",
        "model_name": args.model,
        "layer_name": args.layer_name,
        "input_shape": [int(v) for v in input_shape],
        "output_shape": [int(v) for v in output_shape],
        "batch": int(args.batch),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "theoretical_csv": theoretical_csv or "",
        "use_profiler_api": bool(args.use_profiler_api),
        "profiler_backend_start": profiler_backend_start,
        "profiler_backend_stop": profiler_backend_stop,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "conv": conv_metadata(layer),
        "notes": "single Conv2d forward only; no full model, BatchNorm, activation, pooling, or Linear layers",
    }
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
