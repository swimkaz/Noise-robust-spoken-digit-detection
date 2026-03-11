#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/largescaleml/dsp_final_project
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/e/largescaleml/dsp_final_project/checkpoints/opensmile_mfcc_best_checkpoint.pt}"
SKIP_TRAIN=false
SAVE_CM_DIR="${SAVE_CM_DIR:-./figures}"
LOG_FILE="${LOG_FILE:-/mnt/e/largescaleml/dsp_final_project/logs/opensmile_mfcc_lstm.log}"

python opensmile_lstm.py \
  --use-xgb-feature-selection true \
  --combine-mfcc false \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --skip-train "${SKIP_TRAIN}" \
  --save-cm-dir "${SAVE_CM_DIR}" \
  --log-file "${LOG_FILE}" \
  --device cuda \
  "$@"
