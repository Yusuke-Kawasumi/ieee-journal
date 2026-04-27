# IEEE再投稿のための追加実験
# 環境情報取得コマンド

import torch
import subprocess
import os

def get_sys_info(cmd):
    try:
        # タイムアウトを設定して、コマンドが止まらないようにする
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=5).decode().strip()
    except Exception as e:
        return f"Error/Not Found: {str(e)}"

def log_environment():
    # tegrastats の取得方法を変更 (1秒間実行して最初の1行を取得)
    gpu_stat = get_sys_info("tegrastats --interval 1000 | head -n 1 & sleep 1.5; kill $! 2>/dev/null || true")
    
    # 電源モードの取得
    power_mode = get_sys_info("sudo nvpmodel -q")
    
    info = {
        "JetPack": get_sys_info("cat /etc/nv_tegra_release"),
        "CUDA": get_sys_info("/usr/local/cuda/bin/nvcc --version | grep release"),
        "cuDNN": torch.backends.cudnn.version(),
        "PyTorch": torch.__version__,
        "TensorRT": get_sys_info("dpkg -l | grep nvinfer | awk '{print $3}' | head -n 1"),
        "Power Mode": power_mode,
        "GPU Status (tegrastats)": gpu_stat
    }
    
    # チェック: MAXN になっていない場合の警告
    if "MAXN" not in power_mode:
        print("\n[WARNING] 電源モードが MAXN ではありません！ 'sudo nvpmodel -m 0' を実行してください。")

    # 実行結果が安定するまでコメントアウト（無為なtxtファイル増加の防止）
    """
    with open("env_info.txt", "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    
    print("\nEnvironment info saved to env_info.txt")
    print(f"Current Status: {gpu_stat}")
    """

    # 実行結果が安定したらコメントアウト推奨
    print("\n" + "="*30)
    print("JETSON ENVIRONMENT REPORT")
    print("="*30)
    for k, v in info.items():
        print(f"{k}: {v}")
    print("="*30 + "\n")


if __name__ == "__main__":
    log_environment()