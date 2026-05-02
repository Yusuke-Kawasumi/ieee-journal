#!/usr/bin/env bash
# Run Phase A0 Nsight Compute smoke test for synthetic Conv2d layers.
# Usage:
#   bash scripts/run_ncu_smoke_test.sh standard
#   bash scripts/run_ncu_smoke_test.sh depthwise
#   bash scripts/run_ncu_smoke_test.sh both
#
# Useful overrides:
#   METRIC_LIST="dram__bytes.sum,gpu__time_duration.sum" bash scripts/run_ncu_smoke_test.sh standard
#   USE_PROFILER_API=0 PROFILE_FROM_START=on bash scripts/run_ncu_smoke_test.sh standard
#   WARMUP=0 ITERS=1 bash scripts/run_ncu_smoke_test.sh standard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NCU_BIN="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CASE="${1:-standard}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-1}"
INPUT_SHAPE="${INPUT_SHAPE:-8,64,56,56}"
USE_PROFILER_API="${USE_PROFILER_API:-1}"
PROFILE_FROM_START="${PROFILE_FROM_START:-off}"
SKIP_SUDO_PY_CHECK="${SKIP_SUDO_PY_CHECK:-0}"

METRIC_LIST_DEFAULT="dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,gpu__time_duration.sum"
METRIC_LIST="${METRIC_LIST:-${METRIC_LIST_DEFAULT}}"

mkdir -p "${ROOT_DIR}/logs/ncu_csv" "${ROOT_DIR}/profiles/ncu_reports"

usage() {
  sed -n '1,16p' "$0"
}

if [ "${CASE}" = "-h" ] || [ "${CASE}" = "--help" ]; then
  usage
  exit 0
fi

if [ "${USE_PROFILER_API}" != "1" ] && [ "${PROFILE_FROM_START}" = "off" ]; then
  echo "[A0] USE_PROFILER_API is not 1 while PROFILE_FROM_START=off; switching PROFILE_FROM_START=on."
  PROFILE_FROM_START="on"
fi

if [ "${SKIP_SUDO_PY_CHECK}" != "1" ]; then
  echo "[A0] Checking sudo -E Python environment."
  sudo -E "${PYTHON_BIN}" - <<'PY'
import torch
import torchvision
import pandas
print("sudo_python_import_check: PASS")
print("torch:", torch.__version__)
print("torch_cuda_available:", torch.cuda.is_available())
PY
fi

run_case() {
  local case_name="$1"
  if [ "${case_name}" != "standard" ] && [ "${case_name}" != "depthwise" ]; then
    echo "[A0] Unknown case: ${case_name}" >&2
    exit 2
  fi

  local log_file="${ROOT_DIR}/logs/ncu_csv/smoke_${case_name}.csv"
  local export_base="${ROOT_DIR}/profiles/ncu_reports/smoke_${case_name}"
  local profiler_arg=()
  if [ "${USE_PROFILER_API}" = "1" ]; then
    profiler_arg=(--use-profiler-api)
  fi

  echo "[A0] Running ncu smoke test: ${case_name}"
  echo "[A0] log_file: ${log_file}"
  echo "[A0] export: ${export_base}.ncu-rep"

  sudo -E "${NCU_BIN}" \
    --target-processes all \
    --profile-from-start "${PROFILE_FROM_START}" \
    --metrics "${METRIC_LIST}" \
    --csv \
    --log-file "${log_file}" \
    --export "${export_base}" \
    --force-overwrite \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_test_conv_layer.py" \
      --case "${case_name}" \
      --input-shape "${INPUT_SHAPE}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      "${profiler_arg[@]}"

  echo "[A0] Finished smoke test: ${case_name}"
}

if [ "${CASE}" = "both" ]; then
  run_case standard
  run_case depthwise
else
  run_case "${CASE}"
fi

echo "[A0] Smoke test command completed. Check logs/ncu_csv/*.csv for metric availability."
