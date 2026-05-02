#!/usr/bin/env python3
"""
Phase A0 smoke-test Conv2d runner for Nsight Compute.

This script creates a synthetic Conv2d layer and runs only that layer. It is used
only to check that ncu, sudo Python, CUDA profiler API, and metric collection work
on the Jetson. Do not use these smoke-test values in the paper.

Examples:
  python3 scripts/smoke_test_conv_layer.py --case standard --warmup 5 --iters 1
  python3 scripts/smoke_test_conv_layer.py --case depthwise --warmup 5 --iters 1 --use-profiler-api
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def parse_shape(shape_text: str) -> Tuple[int, int, int, int]:
    text = str(shape_text).strip()
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


def cuda_profiler_call(method_name: str) -> str:
    errors: List[str] = []
    try:
        cudart = torch.cuda.cudart()
        func = getattr(cudart, method_name)
        ret = func()
        if ret is None:
            return "torch.cuda.cudart"
        code = int(ret[0]) if isinstance(ret, tuple) else int(ret)
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
    raise RuntimeError(f"Could not call {method_name}: " + " | ".join(errors))


def parse_bool_text(value: str) -> bool:
    value_l = value.strip().lower()
    if value_l in ("1", "true", "yes", "y", "on"):
        return True
    if value_l in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean text, got {value!r}")


def make_conv(case: str, input_shape: Tuple[int, int, int, int]) -> nn.Conv2d:
    _, cin, _, _ = input_shape
    if case == "standard":
        return nn.Conv2d(in_channels=cin, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
    if case == "depthwise":
        return nn.Conv2d(in_channels=cin, out_channels=cin, kernel_size=3, stride=1, padding=1, groups=cin, bias=False)
    raise ValueError(f"Unknown case: {case}")


def theoretical_flops(conv: nn.Conv2d, input_shape: Tuple[int, int, int, int], output_shape: Tuple[int, int, int, int]) -> int:
    n, cin, _, _ = input_shape
    _, cout, hout, wout = output_shape
    kh, kw = conv.kernel_size
    return int(2 * n * cout * hout * wout * (cin // conv.groups) * kh * kw)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic Conv2d smoke test for ncu.")
    parser.add_argument("--case", choices=("standard", "depthwise"), default="standard")
    parser.add_argument("--input-shape", default="8,64,56,56")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--use-profiler-api", action="store_true")
    parser.add_argument("--cudnn-benchmark", type=parse_bool_text, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this smoke test on the Jetson CUDA environment.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.set_grad_enabled(False)

    input_shape = parse_shape(args.input_shape)
    conv = make_conv(args.case, input_shape).eval().to("cuda")
    x = torch.randn(*input_shape, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = conv(x)
        torch.cuda.synchronize()

        profiler_backend_start = "not_used"
        profiler_backend_stop = "not_used"
        if args.use_profiler_api:
            profiler_backend_start = cuda_profiler_call("cudaProfilerStart")

        y = None
        for _ in range(args.iters):
            y = conv(x)

        torch.cuda.synchronize()

        if args.use_profiler_api:
            profiler_backend_stop = cuda_profiler_call("cudaProfilerStop")

    assert y is not None
    output_shape = tuple(int(v) for v in y.shape)
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    ph, pw = conv.padding
    dh, dw = conv.dilation

    metadata: Dict[str, Any] = {
        "script": "smoke_test_conv_layer.py",
        "phase": "A0_smoke_test",
        "case": args.case,
        "input_shape": [int(v) for v in input_shape],
        "output_shape": [int(v) for v in output_shape],
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "use_profiler_api": bool(args.use_profiler_api),
        "profiler_backend_start": profiler_backend_start,
        "profiler_backend_stop": profiler_backend_stop,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "conv": {
            "Cin": int(conv.in_channels),
            "Cout": int(conv.out_channels),
            "Kh": int(kh),
            "Kw": int(kw),
            "stride_h": int(sh),
            "stride_w": int(sw),
            "padding_h": int(ph),
            "padding_w": int(pw),
            "dilation_h": int(dh),
            "dilation_w": int(dw),
            "groups": int(conv.groups),
            "bias": bool(conv.bias is not None),
            "weight_shape": [int(v) for v in conv.weight.shape],
        },
        "theoretical_FLOPs_for_reference_only": theoretical_flops(conv, input_shape, output_shape),
        "notes": "Phase A0 only; do not use smoke-test values in the paper",
    }
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
