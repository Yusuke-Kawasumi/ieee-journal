#!/usr/bin/env bash
# Phase A0 environment check for Jetson Orin Nano Experiment A.
# Run from experiment_A/ or from anywhere. This does not profile layers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NCU_BIN="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${ROOT_DIR}/logs"
NCU_LOG_DIR="${ROOT_DIR}/logs/ncu_csv"
mkdir -p "${LOG_DIR}" "${NCU_LOG_DIR}"

echo "[A0] root dir: ${ROOT_DIR}"
echo "[A0] python: ${PYTHON_BIN}"
echo "[A0] ncu: ${NCU_BIN}"

echo "[A0] Checking power mode. Expected: 15W, ID=0, maximum available mode on this device."
sudo nvpmodel -q || true

echo "[A0] Checking jetson_clocks status."
sudo jetson_clocks --show || true

echo "[A0] Checking sudo -E Python imports."
sudo -E "${PYTHON_BIN}" - <<'PY'
import torch
import torchvision
import pandas
print("sudo_python_import_check: PASS")
print("torch:", torch.__version__)
print("torch_cuda_available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("pandas:", pandas.__version__)
PY

echo "[A0] Checking Nsight Compute version."
"${NCU_BIN}" --version || true

echo "[A0] Querying ncu metrics to logs/ncu_csv/ncu_query_metrics.txt"
"${NCU_BIN}" --query-metrics > "${NCU_LOG_DIR}/ncu_query_metrics.txt" 2>&1 || true

echo "[A0] If get_env_info.py exists, append environment report."
if [ -f "${ROOT_DIR}/get_env_info.py" ]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/get_env_info.py" || true
elif [ -f "${ROOT_DIR}/../get_env_info.py" ]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/../get_env_info.py" || true
elif [ -f "get_env_info.py" ]; then
  "${PYTHON_BIN}" "get_env_info.py" || true
else
  echo "[A0] get_env_info.py not found; skipped."
fi

echo "[A0] Environment check finished."
