import torch
import time
import statistics
from datetime import datetime
import subprocess

M = N = K = 4096
WARMUP = 10
REPEAT = 100

OUTPUT_TXT = "peak_fp32_pytorch_result.txt"
OUTPUT_CSV = "peak_fp32_pytorch_iterations.csv"


def get_tegrastats_once():
    try:
        out = subprocess.check_output(
            "timeout 2s tegrastats --interval 1000 | head -n 1",
            shell=True,
            stderr=subprocess.STDOUT,
            timeout=4,
            executable="/bin/bash",
        )
        return out.decode(errors="replace").strip()
    except Exception as e:
        return f"Error: {e}"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda")

    print("Allocating matrices...")
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)

    torch.cuda.synchronize()

    start_temp_raw = get_tegrastats_once()

    print("Warmup...")
    for _ in range(WARMUP):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    flops = 2 * M * N * K
    results = []

    print("Measurement...")
    for i in range(REPEAT):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        C = torch.matmul(A, B)
        end_event.record()

        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        gflops = flops / elapsed_s / 1e9

        results.append((i + 1, elapsed_ms, gflops))
        print(f"{i+1:03d}: {elapsed_ms:.3f} ms, {gflops:.2f} GFLOPS")

    end_temp_raw = get_tegrastats_once()

    gflops_values = [x[2] for x in results]
    mean_gflops = statistics.mean(gflops_values)
    std_gflops = statistics.stdev(gflops_values) if len(gflops_values) > 1 else 0.0

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("iteration,elapsed_ms,gflops\n")
        for iteration, elapsed_ms, gflops in results:
            f.write(f"{iteration},{elapsed_ms:.6f},{gflops:.6f}\n")

    report = f"""=== PyTorch FP32 SGEMM-like Peak Measurement ===
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Method: torch.matmul
Matrix size: M={M}, N={N}, K={K}
Precision: FP32
TF32 disabled: True
Warmup: {WARMUP}
Measurement repeats: {REPEAT}

FLOPs per iteration: {flops}
Mean FP32 performance: {mean_gflops:.6f} GFLOPS
Std FP32 performance: {std_gflops:.6f} GFLOPS

Start tegrastats:
{start_temp_raw}

End tegrastats:
{end_temp_raw}

Per-iteration CSV:
{OUTPUT_CSV}
"""

    print()
    print(report)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved: {OUTPUT_TXT}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()