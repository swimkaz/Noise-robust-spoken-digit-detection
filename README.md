# M214A Group 4 Final Project

Noise-robust spoken digit recognition using handcrafted acoustic features and a BiLSTM classifier.

## Repository Layout

- `feature_dev_lstm.py`: main training and evaluation script
- `create_conda_env.sh`: environment setup
- `run_*.sh`: experiment launchers
- `M214_project_data/`: extracted dataset
- `logs/`: training and evaluation logs
- `checkpoints/`: saved model checkpoints
- `figures/`: confusion matrices and training curves
- `features/`: exported metrics and feature summaries

## Dataset Layout

The code expects the extracted dataset directory:

```text
M214_project_data/
  train_clean/
  test_clean/
  test_snr_5db_babble/
  test_snr_10db_babble/
```

If only `M214_project_data.zip` is present, the script can extract it automatically.

## Environment Setup

Create a conda environment:

```bash
bash create_conda_env.sh audio_env
conda activate audio_env
```

This installs the required packages, including:
- `torch`
- `torchaudio`
- `librosa`
- `opensmile`
- `spafe`
- `xgboost`
- `umap-learn`

## Main Entry Point

All experiments run through:

```bash
python3 feature_dev_lstm.py
```

Useful arguments:

```bash
--feature-source
--num-epochs
--device
--checkpoint-path
--log-file
--save-cm-dir
--skip-train
--augment-train-noise
--train-aug-prob
--train-aug-snr-min
--train-aug-snr-max
```

## Best Results

### Best clean-train-only result

Feature set:
- `PNCC + delta + delta-delta + CMVN`

Run:

```bash
bash run_pncc_delta_cmvn_lstm.sh
```

Current reference result:
- clean accuracy: `0.9800`
- 5 dB accuracy: `0.7633`
- 10 dB accuracy: `0.9000`
- average accuracy: `0.8811`

Outputs:
- log: `logs/pncc_delta_cmvn_lstm.log`
- checkpoint: `checkpoints/pncc_delta_cmvn_best_checkpoint.pt`

### Best overall result with training augmentation

Feature set:
- `PNCC + delta + delta-delta + CMVN`

Training augmentation:
- waveform-level additive noise during training only
- default probability: `0.6`
- default SNR range: `3` to `15 dB`

Run:

```bash
bash run_pncc_delta_cmvn_aug_lstm.sh
```

Current reference result:
- clean accuracy: `0.9800`
- 5 dB accuracy: `0.8233`
- 10 dB accuracy: `0.9400`
- average accuracy: `0.9144`

Outputs:
- log: `logs/pncc_delta_cmvn_aug_lstm2.log`
- checkpoint: `checkpoints/pncc_delta_cmvn_aug_best_checkpoint2.pt`
- detailed metrics: `features/pncc_delta_cmvn_aug_best_metrics*.csv`

## Reproducing the Best Augmented Result

From the project directory:

```bash
conda activate audio_env
bash run_pncc_delta_cmvn_aug_lstm.sh
```

To change the augmentation strength:

```bash
AUG_PROB=0.5 AUG_SNR_MIN=5 AUG_SNR_MAX=12 bash run_pncc_delta_cmvn_aug_lstm.sh
```

To run on CPU:

```bash
bash run_pncc_delta_cmvn_aug_lstm.sh --device cpu
```

To evaluate from an existing checkpoint without retraining:

```bash
SKIP_TRAIN=true bash run_pncc_delta_cmvn_aug_lstm.sh
```

## Notes
- It takes around 15 minutes to finish training when including data augmentation on Nvidia 4090 GPU.
- The official project setting is clean-train-only evaluation against clean and noisy test sets.
- The augmented run should be presented as a supplementary experiment because it trains with synthetic noise.
- Per-digit metrics are exported under `features/`.
- Confusion matrices and training curves are exported under `figures/`.
