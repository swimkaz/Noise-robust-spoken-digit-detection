# Session Notes

Date: 2026-03-11

## Context recovered

- User referred to a prior conversation about `deep_final_project`, specifically `opensmile_lstm.py`.
- Workspace inspection showed the active project folder is `dsp_final_project`, and the main file is `opensmile_lstm.py`.

## Files reviewed

- `opensmile_lstm.py`
- `run_opensmile_lstm.sh`
- `M214A_26_Winter_proj.pdf`

## Project understanding

- The project is a noise-robust spoken digit recognition assignment for UCLA ECEM214A Digital Speech Processing.
- Main task constraints from the PDF:
  - train on clean data only
  - evaluate on clean, 5 dB babble, and 10 dB babble noisy test sets
  - report accuracy and additional metrics such as confusion matrices
  - for main results, do not change the baseline LSTM/classifier, learning rate, or batch size
  - focus on robust feature design rather than classifier changes

## Current code summary

- `opensmile_lstm.py` implements:
  - dataset extraction and setup
  - OpenSMILE eGeMAPSv02 feature extraction
  - optional MFCC concatenation
  - optional XGBoost-based feature selection using clean-train data only
  - BiLSTM digit classifier training
  - evaluation on clean, 5 dB, and 10 dB test sets
  - checkpoint saving/loading
  - confusion matrix plotting
  - training curve plotting
  - per-digit feature summary CSV export

- `run_opensmile_lstm.sh` launches the script with CUDA and project-specific paths.

## Important observation

- The shell script currently passes `--combine-mfcc false`.
- That means the launcher is running OpenSMILE-only features by default, even though the checkpoint/log naming includes `mfcc`.

## Status

- No code changes were made yet beyond saving this session note.
- The next sensible step is either:
  - explain the existing pipeline in more detail, or
  - make the next requested code change directly.
