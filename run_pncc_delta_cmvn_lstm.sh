#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/pncc_delta_cmvn_best_checkpoint.pt}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SAVE_CM_DIR="${SAVE_CM_DIR:-./figures}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/pncc_delta_cmvn_lstm.log}"

python3 feature_dev_lstm.py \
  --feature-source pncc_delta_cmvn \
  --use-xgb-feature-selection false \
  --combine-mfcc false \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --skip-train "${SKIP_TRAIN}" \
  --save-cm-dir "${SAVE_CM_DIR}" \
  --log-file "${LOG_FILE}" \
  --device cuda \
  "$@"
