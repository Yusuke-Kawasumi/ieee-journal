# IEEE再投稿のための追加実験
# Jetson Orin Nano 環境情報取得スクリプト

import torch
import subprocess
from datetime import datetime

# 保存モード:
# "print"   : 標準出力のみ。動作確認用
# "append"  : env_log.txt に追記
# "newfile" : タイムスタンプ付きファイルを新規作成
SAVE_MODE = "print"

LOG_FILE = "env_log.txt"


def get_sys_info(cmd, timeout=5):
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            executable="/bin/bash",
        ).decode(errors="replace").strip()
    except Exception as e:
        return f"Error/Not Found: {str(e)}"


def is_maxn_mode(power_mode: str) -> bool:
    text = power_mode.lower()

    return (
        "maxn" in text
        or "mode 0" in text
        or "mode: 0" in text
        or "id: 0" in text
    )


def make_report_text(info: dict) -> str:
    lines = []
    lines.append("=" * 40)
    lines.append("JETSON ENVIRONMENT REPORT")
    lines.append("=" * 40)

    for k, v in info.items():
        lines.append(f"{k}: {v}")

    lines.append("=" * 40)
    return "\n".join(lines)


def save_report(report_text: str, timestamp_for_file: str):
    if SAVE_MODE == "print":
        return None

    if SAVE_MODE == "append":
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + report_text + "\n")
        return LOG_FILE

    if SAVE_MODE == "newfile":
        filename = f"env_info_{timestamp_for_file}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        return filename

    raise ValueError(f"Unknown SAVE_MODE: {SAVE_MODE}")


def log_environment():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_for_file = now.strftime("%Y%m%d_%H%M%S")

    gpu_stat = get_sys_info(
        "timeout 2s tegrastats --interval 1000 | head -n 1",
        timeout=4,
    )

    # 要望により sudo パスワード待ち問題は一旦無視
    power_mode = get_sys_info("sudo nvpmodel -q", timeout=5)

    info = {
        "Timestamp": timestamp,
        "JetPack": get_sys_info("cat /etc/nv_tegra_release"),
        "CUDA": get_sys_info("/usr/local/cuda/bin/nvcc --version | grep release"),
        "cuDNN": str(torch.backends.cudnn.version()),
        "PyTorch": torch.__version__,
        "Torch CUDA available": str(torch.cuda.is_available()),
        "Torch CUDA version": str(torch.version.cuda),
        "TensorRT": get_sys_info("dpkg -l | grep nvinfer | awk '{print $3}' | head -n 1"),
        "Power Mode": power_mode,
        "GPU Status (tegrastats)": gpu_stat,
    }

    report_text = make_report_text(info)

    print("\n" + report_text + "\n")

    saved_path = save_report(report_text, timestamp_for_file)

    if not is_maxn_mode(power_mode):
        print("[WARNING] 電源モードが MAXN ではない可能性があります。")
        print("          必要なら `sudo nvpmodel -m 0` を実行してください。")

    if saved_path is not None:
        print(f"Environment info saved to: {saved_path}")
    else:
        print("SAVE_MODE='print': ファイル保存はしていません。")


if __name__ == "__main__":
    log_environment()