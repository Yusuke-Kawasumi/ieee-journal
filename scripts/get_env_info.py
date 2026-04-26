# IEEE再投稿のための追加実験
# 環境情報取得コマンド

import torch
import subprocess
import os

def get_sys_info(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except:
        return "Not Found"

def log_environment():
    info = {
        "JetPack": get_sys_info("cat /etc/nv_tegra_release"),
        "CUDA": get_sys_info("/usr/local/cuda/bin/nvcc --version | grep release"),
        "cuDNN": torch.backends.cudnn.version(),
        "PyTorch": torch.__version__,
        "TensorRT": get_sys_info("dpkg -l | grep nvinfer | awk '{print $3}' | head -n 1"),
        "Power Mode": get_sys_info("sudo nvpmodel -q"),
        "GPU Status": get_sys_info("tegrastats --interval 1000 --count 1")
    }
    
    with open("env_info.txt", "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print("Environment info saved to env_info.txt")

if __name__ == "__main__":
    log_environment()