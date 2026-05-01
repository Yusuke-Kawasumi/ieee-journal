# IEEE再投稿のための追加実験
# Jetson Orin Nano 環境情報取得スクリプト

import torch
import subprocess
from datetime import datetime
import re

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

# --- tegrastats パース ---
def parse_tegrastats(tegrastats_str: str) -> dict:
    parsed = {}

    # RAM 1915/7451MB
    m = re.search(r"RAM\s+(\d+)/(\d+)MB", tegrastats_str)
    if m:
        parsed["RAM Used MB"] = m.group(1)
        parsed["RAM Total MB"] = m.group(2)

    # SWAP 0/3726MB
    m = re.search(r"SWAP\s+(\d+)/(\d+)MB", tegrastats_str)
    if m:
        parsed["SWAP Used MB"] = m.group(1)
        parsed["SWAP Total MB"] = m.group(2)

    # CPU [0%@883, ...]
    m = re.search(r"CPU\s+\[([^\]]+)\]", tegrastats_str)
    if m:
        parsed["CPU Status"] = m.group(1)

    # EMC_FREQ 0%
    m = re.search(r"EMC_FREQ\s+(\d+)%", tegrastats_str)
    if m:
        parsed["EMC Utilization %"] = m.group(1)

    # GR3D_FREQ 0%@[0]
    m = re.search(r"GR3D_FREQ\s+(\d+)%", tegrastats_str)
    if m:
        parsed["GPU Utilization %"] = m.group(1)

    # GPU@41.437C
    m = re.search(r"GPU@([-\d.]+)C", tegrastats_str)
    if m:
        parsed["GPU Temperature C"] = m.group(1)

    # CPU@42.437C
    m = re.search(r"CPU@([-\d.]+)C", tegrastats_str)
    if m:
        parsed["CPU Temperature C"] = m.group(1)

    # tj@42.437C
    m = re.search(r"tj@([-\d.]+)C", tegrastats_str)
    if m:
        parsed["Thermal Junction Temperature C"] = m.group(1)

    return parsed


# --- jetson_clocks パース ---
def parse_jetson_clocks(clock_str: str) -> dict:
    import re

    parsed = {}

    # GPU
    m = re.search(r"GPU MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)", clock_str)
    if m:
        parsed["GPU MinFreq"] = m.group(1)
        parsed["GPU MaxFreq"] = m.group(2)
        parsed["GPU CurrentFreq"] = m.group(3)

    # EMC
    m = re.search(r"EMC MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)", clock_str)
    if m:
        parsed["EMC MinFreq"] = m.group(1)
        parsed["EMC MaxFreq"] = m.group(2)
        parsed["EMC CurrentFreq"] = m.group(3)

    # CPU (cpu0だけ代表として取得)
    m = re.search(r"cpu0:.*MaxFreq=(\d+).*CurrentFreq=(\d+)", clock_str)
    if m:
        parsed["CPU MaxFreq"] = m.group(1)
        parsed["CPU CurrentFreq"] = m.group(2)

    # Power mode再確認
    m = re.search(r"NV Power Mode:\s*(.+)", clock_str)
    if m:
        parsed["Power Mode (clock)"] = m.group(1)

    return parsed


def get_onnx_version():
    try:
        import onnx
        return onnx.__version__
    except Exception as e:
        return f"Error/Not Found: {str(e)}"    


def get_jetson_clocks_status():
    return get_sys_info("sudo jetson_clocks --show", timeout=10)


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

    gpu_stat_raw = get_sys_info(
        "timeout 2s tegrastats --interval 1000 | head -n 1",
        timeout=4,
    )

    gpu_stat_parsed = parse_tegrastats(gpu_stat_raw)

    # 要望により sudo パスワード待ち問題は一旦無視
    power_mode_raw = get_sys_info("sudo nvpmodel -q", timeout=20)
    power_name, power_id = parse_power_mode(power_mode_raw)

    clock_raw = get_sys_info("sudo jetson_clocks --show", timeout=10)
    clock_parsed = parse_jetson_clocks(clock_raw)

    info = {
        "Timestamp": timestamp,
        "JetPack": get_sys_info("cat /etc/nv_tegra_release"),
        "CUDA": get_sys_info("/usr/local/cuda/bin/nvcc --version | grep release"),
        "cuDNN": str(torch.backends.cudnn.version()),
        "PyTorch": torch.__version__,
        "Torch CUDA available": str(torch.cuda.is_available()),
        "Torch CUDA version": str(torch.version.cuda),
        "TensorRT": get_sys_info("dpkg -l | grep nvinfer | awk '{print $3}' | head -n 1"),
        "ONNX": get_onnx_version(),

        # --- Power mode ---
        "Power Mode Name": power_name,
        "Power Mode ID": power_id,
        "Power Mode Raw": power_mode_raw,
        
        # --- jetson clocks ---
        "GPU MinFreq": clock_parsed.get("GPU MinFreq", "Unknown"),
        "GPU MaxFreq": clock_parsed.get("GPU MaxFreq", "Unknown"),
        "GPU CurrentFreq": clock_parsed.get("GPU CurrentFreq", "Unknown"),

        "EMC MaxFreq": clock_parsed.get("EMC MaxFreq", "Unknown"),
        "EMC CurrentFreq": clock_parsed.get("EMC CurrentFreq", "Unknown"),

        "CPU MaxFreq": clock_parsed.get("CPU MaxFreq", "Unknown"),
        "CPU CurrentFreq": clock_parsed.get("CPU CurrentFreq", "Unknown"),

        "jetson_clocks Raw": clock_raw,

        # --- GPU status ---
        "GPU Temperature C": gpu_stat_parsed.get("GPU Temperature C", "Unknown"),
        "CPU Temperature C": gpu_stat_parsed.get("CPU Temperature C", "Unknown"),
        "Thermal Junction Temperature C": gpu_stat_parsed.get("Thermal Junction Temperature C", "Unknown"),
        "GPU Utilization %": gpu_stat_parsed.get("GPU Utilization %", "Unknown"),
        "EMC Utilization %": gpu_stat_parsed.get("EMC Utilization %", "Unknown"),
        "RAM Used MB": gpu_stat_parsed.get("RAM Used MB", "Unknown"),
        "RAM Total MB": gpu_stat_parsed.get("RAM Total MB", "Unknown"),
        "SWAP Used MB": gpu_stat_parsed.get("SWAP Used MB", "Unknown"),
        "SWAP Total MB": gpu_stat_parsed.get("SWAP Total MB", "Unknown"),

        "GPU Status Raw": gpu_stat_raw,
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