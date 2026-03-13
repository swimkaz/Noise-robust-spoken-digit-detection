#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/pncc_delta_cmvn_aug_best_checkpoint2.pt}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SAVE_CM_DIR="${SAVE_CM_DIR:-./figures}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/pncc_delta_cmvn_aug_lstm2.log}"
AUG_PROB="${AUG_PROB:-0.6}"
AUG_SNR_MIN="${AUG_SNR_MIN:-3}"
AUG_SNR_MAX="${AUG_SNR_MAX:-15}"

python3 feature_dev_lstm.py \
  --feature-source pncc_delta_cmvn \
  --use-xgb-feature-selection false \
  --augment-train-noise true \
  --train-aug-prob "${AUG_PROB}" \
  --train-aug-snr-min "${AUG_SNR_MIN}" \
  --train-aug-snr-max "${AUG_SNR_MAX}" \
  --combine-mfcc false \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --skip-train "${SKIP_TRAIN}" \
  --save-cm-dir "${SAVE_CM_DIR}" \
  --log-file "${LOG_FILE}" \
  --device cuda \
  "$@"
