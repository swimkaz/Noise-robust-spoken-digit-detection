#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-audio_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
PYTORCH_CUDA_TAG="${PYTORCH_CUDA_TAG:-cu121}"
REQUIRE_CUDA="${REQUIRE_CUDA:-false}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating conda env '${ENV_NAME}' for ${SCRIPT_DIR}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists." >&2
  echo "Remove it first with: conda env remove -n ${ENV_NAME}" >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
  --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}" \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

python -m pip install \
  numpy \
  scipy \
  pandas \
  scikit-learn \
  matplotlib \
  librosa \
  soundfile \
  xgboost \
  ipykernel \
  jupyter \
  opensmile \
  spafe \
  umap-learn

export REQUIRE_CUDA
python - <<'PY'
import os
import sys
import torch
import torchaudio
import librosa
import opensmile
import sklearn
import matplotlib
import xgboost

require_cuda = os.getenv("REQUIRE_CUDA", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
if require_cuda:
    if torch.version.cuda is None:
        raise SystemExit(
            "Environment verification failed: installed torch build has no CUDA support "
            "(torch.version.cuda is None). Set REQUIRE_CUDA=false to allow CPU-only usage."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "Environment verification failed: CUDA build installed, but torch.cuda.is_available() is False. "
            "Set REQUIRE_CUDA=false to allow CPU-only usage."
        )

print("Environment verification passed:")
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("require_cuda:", require_cuda)
print("torchaudio:", torchaudio.__version__)
print("librosa:", librosa.__version__)
print("opensmile:", opensmile.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("xgboost:", xgboost.__version__)
PY

cat <<EOF

Environment '${ENV_NAME}' is ready.

Activate it with:
  conda activate ${ENV_NAME}

Then run:
  cd "${SCRIPT_DIR}"
  bash run_opensmile_lstm.sh

Optional:
  REQUIRE_CUDA=true ./create_conda_env.sh ${ENV_NAME}
EOF
