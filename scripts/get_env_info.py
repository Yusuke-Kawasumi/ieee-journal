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


# --- nvpmodel パース ---
def parse_power_mode(power_mode_str: str):
    lines = power_mode_str.splitlines()

    name = lines[0] if len(lines) > 0 else "Unknown"
    mode_id = lines[1] if len(lines) > 1 else "Unknown"

    return name, mode_id


def is_max_performance_mode(mode_id: str) -> bool:
    return mode_id.strip() == "0"


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
    power_mode_raw = get_sys_info("sudo nvpmodel -q", timeout=20)
    power_name, power_id = parse_power_mode(power_mode_raw)

    info = {
        "Timestamp": timestamp,
        "JetPack": get_sys_info("cat /etc/nv_tegra_release"),
        "CUDA": get_sys_info("/usr/local/cuda/bin/nvcc --version | grep release"),
        "cuDNN": str(torch.backends.cudnn.version()),
        "PyTorch": torch.__version__,
        "Torch CUDA available": str(torch.cuda.is_available()),
        "Torch CUDA version": str(torch.version.cuda),
        "TensorRT": get_sys_info("dpkg -l | grep nvinfer | awk '{print $3}' | head -n 1"),
        
        # --- Power mode ---
        "Power Mode Name": power_name,
        "Power Mode ID": power_id,
        "Power Mode Raw": power_mode_raw,

        "GPU Status (tegrastats)": gpu_stat,
    }

    report_text = make_report_text(info)

    print("\n" + report_text + "\n")

    saved_path = save_report(report_text, timestamp_for_file)

    # --- 最大性能モードチェック ---
    if not is_max_performance_mode(power_id):
        print("[WARNING] Not in maximum performance mode.")
        print("          Expected: ID=0 (15W on this device)")

    if saved_path is not None:
        print(f"Environment info saved to: {saved_path}")
    else:
        print("SAVE_MODE='print': no file was saved.")


if __name__ == "__main__":
    log_environment()