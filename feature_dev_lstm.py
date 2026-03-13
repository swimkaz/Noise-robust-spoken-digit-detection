import argparse
import csv
import copy
import logging
import os
import random
import time
import zipfile
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from opensmile import FeatureLevel, FeatureSet, Smile
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Set before CUDA ops for deterministic CUDA kernels.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def add_file_logging(log_file: str, level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM digit classifier with OpenSMILE features")
    parser.add_argument(
        "--dataset-zip",
        type=str,
        default=project_path("M214_project_data.zip"),
        help="Path to dataset zip file",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=project_path("M214_project_data"),
        help="Expected extracted dataset directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--log-file",
        type=str,
        default=project_path("logs", "run_local.log"),
        help="Path to log file (default enabled).",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--feature-source",
        type=str,
        default="opensmile",
        choices=[
            "mfcc",
            "mfcc_delta_cmvn",
            "pncc",
            "pncc_cmvn",
            "pncc_delta_only",
            "pncc_delta_cmvn_only",
            "pncc_delta",
            "pncc_delta2_cmvn",
            "pncc_delta_cmvn",
            "opensmile",
            "opensmile_mfcc",
            "opensmile_pncc",
        ],
        help=(
            "Input feature set: MFCC baseline, MFCC+delta+delta-delta with CMVN, PNCC baseline, PNCC+CMVN, PNCC+delta, PNCC+delta+CMVN, "
            "PNCC+delta+delta-delta, PNCC+delta-delta+CMVN, PNCC+delta+delta-delta with CMVN, "
            "OpenSMILE only, OpenSMILE+MFCC, or OpenSMILE+PNCC."
        ),
    )
    parser.add_argument(
        "--feature-level",
        type=str,
        default="lld",
        choices=["lld", "functionals"],
        help="OpenSMILE feature level",
    )
    parser.add_argument(
        "--opensmile-feature-set",
        type=str,
        default="eGeMAPSv02",
        choices=["eGeMAPSv02", "ComParE_2016"],
        help="OpenSMILE feature set to use for OpenSMILE-based modes.",
    )
    parser.add_argument(
        "--use-xgb-feature-selection",
        type=str2bool,
        default=True,
        help="Whether to enable XGBoost-based robust feature selection (true/false)",
    )
    parser.add_argument("--xgb-val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--augment-train-noise",
        type=str2bool,
        default=False,
        help="Enable waveform-level noisy augmentation on training batches only (true/false).",
    )
    parser.add_argument(
        "--train-aug-prob",
        type=float,
        default=0.5,
        help="Probability of applying noisy augmentation to each training utterance.",
    )
    parser.add_argument(
        "--train-aug-snr-min",
        type=float,
        default=3.0,
        help="Minimum SNR in dB for train-time noisy augmentation.",
    )
    parser.add_argument(
        "--train-aug-snr-max",
        type=float,
        default=15.0,
        help="Maximum SNR in dB for train-time noisy augmentation.",
    )
    parser.add_argument(
        "--combine-mfcc",
        type=str2bool,
        default=False,
        help="Whether to concatenate MFCC features with OpenSMILE features (true/false)",
    )
    parser.add_argument(
        "--skip-train",
        type=str2bool,
        default=False,
        help="Skip training and directly load checkpoint for evaluation/figures (true/false)",
    )
    parser.add_argument(
        "--save-cm-dir",
        type=str,
        default="./figures",
        help="Directory to save confusion matrix images (default: ./figures).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=project_path("checkpoints", "best_checkpoint.pt"),
        help="Path to save/load the best checkpoint automatically",
    )
    parser.add_argument(
        "--save-feature-summary",
        type=str2bool,
        default=True,
        help="Save selected-feature list and clean-set per-digit feature summary (true/false)",
    )
    parser.add_argument(
        "--feature-summary-max-features",
        type=int,
        default=30,
        help="Max number of features to include in clean-set per-digit summary CSV",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.getLogger(__name__).info("Random seed set to %d", seed)


def ensure_dataset_ready(dataset_zip: str, extract_dir: str) -> str:
    logger = logging.getLogger(__name__)
    expected_train_dir = os.path.join(extract_dir, "train_clean")
    if os.path.isdir(expected_train_dir):
        logger.info("Dataset already extracted at %s", extract_dir)
        return extract_dir

    nested_extract_dir = os.path.join(extract_dir, "M214_project_data")
    nested_train_dir = os.path.join(nested_extract_dir, "train_clean")
    if os.path.isdir(nested_train_dir):
        logger.warning("Detected nested extracted path. Using %s", nested_extract_dir)
        return nested_extract_dir

    if not os.path.isfile(dataset_zip):
        raise FileNotFoundError(f"Dataset zip not found: {dataset_zip}")

    parent_dir = os.path.dirname(extract_dir)
    if not parent_dir:
        parent_dir = "."
    os.makedirs(parent_dir, exist_ok=True)

    logger.info("Extracting dataset zip %s -> %s", dataset_zip, parent_dir)
    t0 = time.time()
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(parent_dir)
    logger.info("Dataset extraction completed in %.2f seconds", time.time() - t0)

    if os.path.isdir(expected_train_dir):
        logger.info("Dataset extracted to %s", extract_dir)
        return extract_dir
    if os.path.isdir(nested_train_dir):
        logger.info("Dataset extracted to nested path %s", nested_extract_dir)
        return nested_extract_dir

    raise RuntimeError(
        "Dataset extraction completed, but expected folders were not found under "
        f"{extract_dir} or {nested_extract_dir}."
    )


def load_audio(audio_file):
    try:
        audio, fs = torchaudio.load(audio_file)
    except ImportError as exc:
        raise ImportError(
            "MFCC baseline mode follows Copy_of_baseline.ipynb and requires "
            "`torchaudio.load(...)` to work in your environment. Install the "
            "missing torchaudio backend dependency (for your current setup this is likely "
            "`torchcodec`) or use an environment matching the notebook."
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            "Failed to load audio with `torchaudio.load(...)`. "
            "This is usually an audio-backend/runtime issue in the current environment "
            f"(file: {audio_file}). Original error: {exc}"
        ) from exc
    audio = audio.numpy().reshape(-1)
    return audio, int(fs)


def add_waveform_gaussian_noise_by_snr(audio, snr_db, rng):
    """Add waveform-domain Gaussian noise at a target SNR (in dB)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    signal_power = np.mean(audio ** 2, dtype=np.float64) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def extract_feature(audio, fs):
    """Extract MFCC feature from raw audio. Returns 2-D array (F, T)."""
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "MFCC baseline mode follows Copy_of_baseline.ipynb and requires `librosa`."
        ) from exc

    n_mfcc = 13
    win_length = 200
    hop_length = 80
    n_fft = 256
    return librosa.feature.mfcc(
        y=audio,
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )


def extract_feature_from_file(audio_file):
    audio, fs = load_audio(audio_file)
    return extract_feature(audio, fs)


def extract_mfcc_delta_cmvn(audio, fs):
    mfcc = extract_feature(audio, fs).astype(np.float32)
    mfcc = apply_cmvn(mfcc)
    return np.concatenate(
        [
            mfcc,
            compute_delta_features(mfcc, order=1),
            compute_delta_features(mfcc, order=2),
        ],
        axis=0,
    ).astype(np.float32)


def extract_mfcc_delta_cmvn_from_file(audio_file):
    audio, fs = load_audio(audio_file)
    return extract_mfcc_delta_cmvn(audio, fs)


def extract_pncc(audio, fs):
    """Extract PNCC features from raw audio. Returns 2-D array (F, T)."""
    try:
        from spafe.features.pncc import pncc as spafe_pncc
    except ImportError as exc:
        raise ImportError(
            "PNCC mode requires `spafe`. Install it with `pip install spafe` "
            "or recreate the project environment with the updated setup script."
        ) from exc

    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    call_attempts = [
        lambda: spafe_pncc(sig=audio, fs=fs, num_ceps=13),
        lambda: spafe_pncc(audio, fs=fs, num_ceps=13),
        lambda: spafe_pncc(sig=audio, fs=fs),
        lambda: spafe_pncc(audio, fs=fs),
    ]

    last_exc = None
    features = None
    for attempt in call_attempts:
        try:
            features = attempt()
            break
        except TypeError as exc:
            last_exc = exc

    if features is None:
        raise RuntimeError(
            "Failed to call `spafe.features.pncc.pncc(...)` with expected signatures. "
            f"Last error: {last_exc}"
        )

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        values = np.atleast_2d(values)
    return values.T


def extract_pncc_from_file(audio_file):
    audio, fs = load_audio(audio_file)
    return extract_pncc(audio, fs)


def apply_cmvn(feature_map):
    """Apply per-utterance cepstral mean and variance normalization over time."""
    mean = feature_map.mean(axis=1, keepdims=True)
    std = feature_map.std(axis=1, keepdims=True)
    return ((feature_map - mean) / np.clip(std, 1e-6, None)).astype(np.float32)


def compute_delta_features(feature_map, order=1):
    """Compute delta-like features for a (F, T) feature map."""
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Delta-feature PNCC modes require `librosa`. Install it or use an environment "
            "created with the provided setup script."
        ) from exc

    base = np.asarray(feature_map, dtype=np.float32)
    return librosa.feature.delta(base, order=order).astype(np.float32)


def build_pncc_variant_feature_map(audio, fs, use_cmvn=False, add_delta=False, add_delta2=False):
    pncc = extract_pncc(audio, fs)
    if use_cmvn:
        pncc = apply_cmvn(pncc)
    parts = [pncc]
    if add_delta:
        parts.append(compute_delta_features(pncc, order=1))
    if add_delta2:
        parts.append(compute_delta_features(pncc, order=2))
    return np.concatenate(parts, axis=0).astype(np.float32)


def extract_pncc_variant_from_file(audio_file, use_cmvn=False, add_delta=False, add_delta2=False):
    audio, fs = load_audio(audio_file)
    return build_pncc_variant_feature_map(
        audio,
        fs,
        use_cmvn=use_cmvn,
        add_delta=add_delta,
        add_delta2=add_delta2,
    )


def extract_features_from_audio(audio, fs, feature_source, smile=None):
    if feature_source == "mfcc":
        return extract_feature(audio, fs).astype(np.float32)
    if feature_source == "mfcc_delta_cmvn":
        return extract_mfcc_delta_cmvn(audio, fs).astype(np.float32)
    if feature_source == "pncc":
        return extract_pncc(audio, fs).astype(np.float32)
    if feature_source == "pncc_cmvn":
        return build_pncc_variant_feature_map(audio, fs, use_cmvn=True, add_delta=False, add_delta2=False)
    if feature_source == "pncc_delta_only":
        return build_pncc_variant_feature_map(audio, fs, use_cmvn=False, add_delta=True, add_delta2=False)
    if feature_source == "pncc_delta_cmvn_only":
        return build_pncc_variant_feature_map(audio, fs, use_cmvn=True, add_delta=True, add_delta2=False)
    if feature_source == "pncc_delta":
        return build_pncc_variant_feature_map(audio, fs, use_cmvn=False, add_delta=True, add_delta2=True)
    if feature_source == "pncc_delta2_cmvn":
        return build_pncc_variant_feature_map(audio, fs, use_cmvn=True, add_delta=False, add_delta2=True)
    if feature_source == "pncc_delta_cmvn":
        return build_pncc_variant_feature_map(audio, fs, use_cmvn=True, add_delta=True, add_delta2=True)

    raise ValueError(
        "Waveform-level on-the-fly extraction is currently supported only for MFCC/PNCC-based modes, "
        f"got feature_source={feature_source}"
    )


def build_smile(feature_level_name: str, feature_set_name: str) -> Smile:
    feature_level = (
        FeatureLevel.LowLevelDescriptors
        if feature_level_name == "lld"
        else FeatureLevel.Functionals
    )
    feature_set = getattr(FeatureSet, feature_set_name)
    return Smile(feature_set=feature_set, feature_level=feature_level)


def extract_egemaps_features(audio_file, smile: Smile):
    """Extract eGeMAPSv02 features and return shape (F, T)."""
    features = smile.process_file(audio_file)
    values = features.values.astype(np.float32)
    if values.ndim != 2:
        values = np.atleast_2d(values)
    return values.T


def get_opensmile_mfcc_channel_indices(smile: Smile):
    names = []
    if hasattr(smile, "feature_names"):
        try:
            names = list(smile.feature_names)
        except Exception:
            names = []
    return [
        i for i, name in enumerate(names) if str(name).lower().startswith("mfcc") and str(name).lower().endswith("_sma3")
    ]


def extract_combined_features(audio_file, smile: Smile, combine_mfcc: bool):
    """Extract OpenSMILE features and optionally concatenate MFCCs along feature axis."""
    egemaps = extract_egemaps_features(audio_file, smile)
    if not combine_mfcc:
        return egemaps

    opensmile_mfcc_idx = get_opensmile_mfcc_channel_indices(smile)
    if opensmile_mfcc_idx:
        keep_idx = [i for i in range(egemaps.shape[0]) if i not in opensmile_mfcc_idx]
        egemaps = egemaps[keep_idx, :]

    mfcc = extract_feature_from_file(audio_file).astype(np.float32)
    t_min = min(egemaps.shape[1], mfcc.shape[1])
    if t_min <= 0:
        raise ValueError(f"Invalid frame length while combining features: {audio_file}")
    if egemaps.shape[1] != mfcc.shape[1]:
        logging.getLogger(__name__).debug(
            "Frame length mismatch for %s (OpenSMILE=%d, MFCC=%d). Truncating to %d.",
            os.path.basename(audio_file),
            egemaps.shape[1],
            mfcc.shape[1],
            t_min,
        )
    return np.concatenate([egemaps[:, :t_min], mfcc[:, :t_min]], axis=0).astype(np.float32)


def extract_opensmile_pncc_features(audio_file, smile: Smile):
    """Extract OpenSMILE features with OpenSMILE MFCC channels removed, then append PNCC."""
    egemaps = extract_egemaps_features(audio_file, smile)
    opensmile_mfcc_idx = get_opensmile_mfcc_channel_indices(smile)
    if opensmile_mfcc_idx:
        keep_idx = [i for i in range(egemaps.shape[0]) if i not in opensmile_mfcc_idx]
        egemaps = egemaps[keep_idx, :]

    pncc = extract_pncc_from_file(audio_file).astype(np.float32)
    t_min = min(egemaps.shape[1], pncc.shape[1])
    if t_min <= 0:
        raise ValueError(f"Invalid frame length while combining OpenSMILE and PNCC: {audio_file}")
    if egemaps.shape[1] != pncc.shape[1]:
        logging.getLogger(__name__).debug(
            "Frame length mismatch for %s (OpenSMILE=%d, PNCC=%d). Truncating to %d.",
            os.path.basename(audio_file),
            egemaps.shape[1],
            pncc.shape[1],
            t_min,
        )
    return np.concatenate([egemaps[:, :t_min], pncc[:, :t_min]], axis=0).astype(np.float32)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, x_list, y):
        self.X = x_list
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        return x, int(self.y[idx]), x.shape[1]


class AudioFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_files,
        y,
        feature_source,
        smile=None,
        augment_noise=False,
        aug_prob=0.5,
        aug_snr_min=3.0,
        aug_snr_max=15.0,
        seed=0,
    ):
        self.audio_files = list(audio_files)
        self.y = y
        self.feature_source = feature_source
        self.smile = smile
        self.augment_noise = augment_noise
        self.aug_prob = float(aug_prob)
        self.aug_snr_min = float(aug_snr_min)
        self.aug_snr_max = float(aug_snr_max)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, fs = load_audio(self.audio_files[idx])
        if self.augment_noise and self.aug_prob > 0.0 and self.rng.random() < self.aug_prob:
            snr_low = min(self.aug_snr_min, self.aug_snr_max)
            snr_high = max(self.aug_snr_min, self.aug_snr_max)
            snr_db = float(self.rng.uniform(snr_low, snr_high))
            audio = add_waveform_gaussian_noise_by_snr(audio, snr_db, self.rng)
        x_np = extract_features_from_audio(audio, fs, self.feature_source, smile=self.smile)
        x = torch.from_numpy(x_np)
        return x, int(self.y[idx]), x.shape[1]


def collate_pad(batch):
    xs, ys, lens = zip(*batch)
    b = len(xs)
    f = xs[0].shape[0]
    t_max = max(lens)
    xb = torch.zeros(b, 1, f, t_max, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        xb[i, 0, :, : x.shape[1]] = x
    return xb, torch.tensor(ys, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def get_label(file_name):
    base = os.path.splitext(os.path.basename(file_name))[0]
    return int(base.split("_")[0])


def list_audio_files(data_dir):
    return sorted(glob(os.path.join(data_dir, "*.wav")))


def load_dir(data_dir, desc="Loading"):
    logger = logging.getLogger(__name__)
    files = list_audio_files(data_dir)
    if not files:
        logger.warning("No wav files found in %s", data_dir)
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(extract_feature_from_file(wav))
        labels.append(get_label(wav))
    logger.info("%s: %d files loaded", desc, len(feats))
    return feats, labels


def load_dir_with_mfcc_delta_cmvn(data_dir, desc="Loading"):
    logger = logging.getLogger(__name__)
    files = list_audio_files(data_dir)
    if not files:
        logger.warning("No wav files found in %s", data_dir)
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(extract_mfcc_delta_cmvn_from_file(wav))
        labels.append(get_label(wav))
    logger.info("%s: %d files loaded (MFCC+delta+delta-delta+CMVN)", desc, len(feats))
    return feats, labels


def load_dir_with_pncc(data_dir, desc="Loading"):
    logger = logging.getLogger(__name__)
    files = list_audio_files(data_dir)
    if not files:
        logger.warning("No wav files found in %s", data_dir)
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(extract_pncc_from_file(wav))
        labels.append(get_label(wav))
    logger.info("%s: %d files loaded (PNCC)", desc, len(feats))
    return feats, labels


def load_dir_with_pncc_variant(data_dir, desc="Loading", use_cmvn=False, add_delta=False, add_delta2=False):
    logger = logging.getLogger(__name__)
    files = list_audio_files(data_dir)
    if not files:
        logger.warning("No wav files found in %s", data_dir)
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(
            extract_pncc_variant_from_file(
                wav,
                use_cmvn=use_cmvn,
                add_delta=add_delta,
                add_delta2=add_delta2,
            )
        )
        labels.append(get_label(wav))
    mode_parts = ["PNCC"]
    if use_cmvn:
        mode_parts.append("CMVN")
    if add_delta:
        mode_parts.append("delta")
    if add_delta2:
        mode_parts.append("delta-delta")
    mode = "+".join(mode_parts)
    logger.info("%s: %d files loaded (%s)", desc, len(feats), mode)
    return feats, labels


def load_dir_with_egemaps(data_dir, smile, combine_mfcc=False, desc="Loading"):
    logger = logging.getLogger(__name__)
    files = list_audio_files(data_dir)
    if not files:
        logger.warning("No wav files found in %s", data_dir)
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(extract_combined_features(wav, smile, combine_mfcc=combine_mfcc))
        labels.append(get_label(wav))
    mode = "eGeMAPS+MFCC" if combine_mfcc else "eGeMAPS"
    logger.info("%s: %d files loaded (%s)", desc, len(feats), mode)
    return feats, labels


def load_dir_with_opensmile_pncc(data_dir, smile, desc="Loading"):
    logger = logging.getLogger(__name__)
    files = list_audio_files(data_dir)
    if not files:
        logger.warning("No wav files found in %s", data_dir)
        return [], []
    feats, labels = [], []
    for wav in files:
        feats.append(extract_opensmile_pncc_features(wav, smile))
        labels.append(get_label(wav))
    logger.info("%s: %d files loaded (eGeMAPS+PNCC)", desc, len(feats))
    return feats, labels


def load_feature_split(data_dir, feature_source, smile=None, desc="Loading"):
    if feature_source == "mfcc":
        return load_dir(data_dir, desc=desc)
    if feature_source == "mfcc_delta_cmvn":
        return load_dir_with_mfcc_delta_cmvn(data_dir, desc=desc)
    if feature_source == "pncc":
        return load_dir_with_pncc(data_dir, desc=desc)
    if feature_source == "pncc_cmvn":
        return load_dir_with_pncc_variant(data_dir, desc=desc, use_cmvn=True, add_delta=False, add_delta2=False)
    if feature_source == "pncc_delta_only":
        return load_dir_with_pncc_variant(data_dir, desc=desc, use_cmvn=False, add_delta=True, add_delta2=False)
    if feature_source == "pncc_delta_cmvn_only":
        return load_dir_with_pncc_variant(data_dir, desc=desc, use_cmvn=True, add_delta=True, add_delta2=False)
    if feature_source == "pncc_delta":
        return load_dir_with_pncc_variant(data_dir, desc=desc, use_cmvn=False, add_delta=True, add_delta2=True)
    if feature_source == "pncc_delta2_cmvn":
        return load_dir_with_pncc_variant(data_dir, desc=desc, use_cmvn=True, add_delta=False, add_delta2=True)
    if feature_source == "pncc_delta_cmvn":
        return load_dir_with_pncc_variant(data_dir, desc=desc, use_cmvn=True, add_delta=True, add_delta2=True)
    if feature_source == "opensmile_pncc":
        if smile is None:
            raise ValueError("OpenSMILE+PNCC loading requires a Smile extractor instance.")
        return load_dir_with_opensmile_pncc(data_dir, smile, desc=desc)
    if smile is None:
        raise ValueError("OpenSMILE feature loading requires a Smile extractor instance.")
    return load_dir_with_egemaps(
        data_dir,
        smile,
        combine_mfcc=(feature_source == "opensmile_mfcc"),
        desc=desc,
    )


def summarize_sequence_features(x):
    """Convert one (F, T) feature map into robust utterance-level statistics."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    p10 = np.percentile(x, 10, axis=1)
    p90 = np.percentile(x, 90, axis=1)
    return np.concatenate([mean, std, p10, p90]).astype(np.float32)


def summarize_feature_list(feats):
    return np.stack([summarize_sequence_features(x) for x in feats], axis=0)


def add_gaussian_noise_by_snr(x, snr_db, rng):
    """Add per-channel Gaussian noise at target SNR (in dB)."""
    signal_power = np.mean(x ** 2, axis=1, keepdims=True).astype(np.float32) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def select_columns_from_summary(x_summary, selected_idx, feat_dim):
    parts = [x_summary[:, selected_idx + i * feat_dim] for i in range(4)]
    return np.concatenate(parts, axis=1)


def build_xgb_k_candidates(feat_dim: int):
    if feat_dim <= 64:
        k_candidates = [8, 12, 16, 20, 24, 32, 48, 64]
    elif feat_dim <= 256:
        k_candidates = [16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
    else:
        # For very high-dimensional OpenSMILE sets such as ComParE_2016, search a
        # broader but still bounded subset ladder instead of only tiny k values.
        ratio_candidates = [
            int(round(feat_dim * r))
            for r in [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0]
        ]
        absolute_candidates = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        k_candidates = absolute_candidates + ratio_candidates

    return sorted(set(k for k in k_candidates if 1 <= k <= feat_dim))


def select_robust_features_with_xgb(train_feat, y_train, seed=0, val_ratio=0.2):
    """
    Rank OpenSMILE channels with XGBoost and choose top-k by noisy validation score
    built from train_clean only (no test leakage).
    """
    logger = logging.getLogger(__name__)

    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed. Install it with `pip install xgboost` "
            "or disable XGBoost feature selection."
        )

    n_samples = len(train_feat)
    if n_samples < 20:
        feat_dim = train_feat[0].shape[0]
        logger.warning("Too few samples for stable XGB FS split. Keeping all %d features.", feat_dim)
        return np.arange(feat_dim, dtype=np.int64)
    logger.info(
        "[XGB FS] Starting feature selection with n_samples=%d, feat_dim=%d, val_ratio=%.3f",
        n_samples,
        train_feat[0].shape[0],
        val_ratio,
    )

    if not (0.0 < val_ratio < 1.0):
        logger.warning("Invalid xgb-val-ratio=%.4f. Falling back to 0.2.", val_ratio)
        val_ratio = 0.2

    indices = np.arange(n_samples)
    try:
        tr_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, stratify=y_train
        )
    except ValueError as exc:
        logger.warning(
            "Stratified split failed for XGB FS (%s). Falling back to non-stratified split.",
            exc,
        )
        tr_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, stratify=None
        )

    tr_feat = [train_feat[i] for i in tr_idx]
    tr_y = y_train[tr_idx]
    val_feat = [train_feat[i] for i in val_idx]
    val_y = y_train[val_idx]

    feat_dim = tr_feat[0].shape[0]
    x_tr = summarize_feature_list(tr_feat)
    x_val_clean = summarize_feature_list(val_feat)

    rng = np.random.default_rng(seed)
    val_feat_5db = [add_gaussian_noise_by_snr(x, 5.0, rng) for x in val_feat]
    val_feat_10db = [add_gaussian_noise_by_snr(x, 10.0, rng) for x in val_feat]
    x_val_5db = summarize_feature_list(val_feat_5db)
    x_val_10db = summarize_feature_list(val_feat_10db)
    logger.info(
        "[XGB FS] Split sizes: train=%d, val=%d | summary dims: train=%s, val=%s",
        len(tr_feat),
        len(val_feat),
        tuple(x_tr.shape),
        tuple(x_val_clean.shape),
    )

    ranker = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        n_estimators=240 if feat_dim > 128 else 300,
        max_depth=4 if feat_dim > 128 else 5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6 if feat_dim > 128 else 0.8,
        reg_lambda=1.5 if feat_dim > 128 else 1.0,
        reg_alpha=0.1 if feat_dim > 128 else 0.0,
        random_state=seed,
        n_jobs=4,
    )
    ranker.fit(x_tr, tr_y)
    importances = ranker.feature_importances_.astype(np.float32)

    per_channel_importance = np.zeros(feat_dim, dtype=np.float32)
    for i in range(4):
        per_channel_importance += importances[i * feat_dim : (i + 1) * feat_dim]
    ranked_idx = np.argsort(-per_channel_importance)

    k_candidates = build_xgb_k_candidates(feat_dim)
    logger.info("[XGB FS] Candidate subset sizes: %s", k_candidates)

    best_score = -1.0
    best_k = feat_dim
    best_metrics = (0.0, 0.0, 0.0)

    for k in k_candidates:
        sel = np.sort(ranked_idx[:k])
        xtr_k = select_columns_from_summary(x_tr, sel, feat_dim)
        xval_clean_k = select_columns_from_summary(x_val_clean, sel, feat_dim)
        xval_5db_k = select_columns_from_summary(x_val_5db, sel, feat_dim)
        xval_10db_k = select_columns_from_summary(x_val_10db, sel, feat_dim)

        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            eval_metric="mlogloss",
            n_estimators=180 if feat_dim > 128 else 220,
            max_depth=3 if feat_dim > 128 else 4,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.6 if feat_dim > 128 else 0.8,
            reg_lambda=1.5 if feat_dim > 128 else 1.0,
            reg_alpha=0.1 if feat_dim > 128 else 0.0,
            random_state=seed,
            n_jobs=4,
        )
        clf.fit(xtr_k, tr_y)
        p_clean = clf.predict(xval_clean_k)
        p_5db = clf.predict(xval_5db_k)
        p_10db = clf.predict(xval_10db_k)

        a_clean = metrics.accuracy_score(val_y, p_clean)
        a_5db = metrics.accuracy_score(val_y, p_5db)
        a_10db = metrics.accuracy_score(val_y, p_10db)
        robust_score = (a_clean + a_5db + a_10db) / 3.0

        logger.info(
            "[XGB FS] k=%3d  clean=%.4f  5dB=%.4f  10dB=%.4f  avg=%.4f",
            k,
            a_clean,
            a_5db,
            a_10db,
            robust_score,
        )

        if robust_score > best_score:
            best_score = robust_score
            best_k = k
            best_metrics = (a_clean, a_5db, a_10db)

    selected = np.sort(ranked_idx[:best_k]).astype(np.int64)
    logger.info(
        "[XGB FS] Selected %d/%d channels (val clean=%.4f, 5dB=%.4f, 10dB=%.4f, avg=%.4f)",
        best_k,
        feat_dim,
        best_metrics[0],
        best_metrics[1],
        best_metrics[2],
        best_score,
    )
    logger.debug("[XGB FS] First selected channel indices: %s", selected[: min(20, len(selected))].tolist())
    return selected


def apply_selected_channels(feats, selected_idx):
    return [x[selected_idx, :] for x in feats]


class SimpleLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 10),
        )

    def forward(self, x, lengths):
        x = x.squeeze(1).permute(0, 2, 1).contiguous()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        _, t_max, _ = out.shape
        device = out.device

        mask = torch.arange(t_max, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        out_sum = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean = out_sum / denom

        return self.classifier(mean)


@torch.no_grad()
def evaluate(model, loader, device, plot_cm=False, class_names=None, title=None, save_path=None):
    model.eval()
    if loader is None:
        return (0.0, None) if plot_cm else 0.0

    all_preds = []
    all_labels = []

    correct, total = 0, 0
    for xb, yb, lengths in loader:
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        logits = model(xb, lengths)
        preds = logits.argmax(dim=1)

        correct += (preds == yb).sum().item()
        total += yb.size(0)

        if plot_cm:
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    if not plot_cm:
        return acc

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)

    labels = class_names if class_names is not None else None
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for confusion-matrix plotting.") from exc

    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names if class_names is not None else None,
    )
    disp.plot(values_format="d")
    plt.title(title or "Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return acc, cm


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    """Compute multiclass metrics for one loader."""
    if loader is None:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "per_digit": [],
        }

    model.eval()
    all_preds = []
    all_labels = []
    for xb, yb, lengths in loader:
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        logits = model(xb, lengths)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(yb.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "per_digit": [],
        }

    labels = list(range(10))
    precision_per_digit, recall_per_digit, f1_per_digit, support_per_digit = metrics.precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_digit_metrics = []
    for digit, precision, recall, f1, support in zip(
        labels,
        precision_per_digit,
        recall_per_digit,
        f1_per_digit,
        support_per_digit,
    ):
        per_digit_metrics.append(
            {
                "digit": int(digit),
                "accuracy": float(recall),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
        )

    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision_macro": metrics.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": metrics.recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_digit": per_digit_metrics,
    }


@torch.no_grad()
def evaluate_loss_and_accuracy(model, loader, device, criterion):
    """Compute average loss and accuracy for one loader."""
    if loader is None:
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb, lengths in loader:
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
        logits = model(xb, lengths)
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * xb.size(0)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def save_training_curves(history, save_dir, prefix):
    """Save training/validation loss+accuracy curves."""
    if not save_dir:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for training-curve plotting.") from exc

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_training_curves.png")
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    noisy5_loss = history.get("noisy5_loss", [])
    noisy10_loss = history.get("noisy10_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    noisy5_acc = history.get("noisy5_acc", [])
    noisy10_acc = history.get("noisy10_acc", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="train_loss", marker="o", linewidth=1.6)
    axes[0].plot(epochs, val_loss, label="val_loss(clean)", marker="o", linewidth=1.6)
    if noisy5_loss:
        axes[0].plot(epochs, noisy5_loss, label="val_loss(5dB)", marker="o", linewidth=1.4)
    if noisy10_loss:
        axes[0].plot(epochs, noisy10_loss, label="val_loss(10dB)", marker="o", linewidth=1.4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc", marker="o", linewidth=1.6)
    axes[1].plot(epochs, val_acc, label="val_acc(clean)", marker="o", linewidth=1.6)
    if noisy5_acc:
        axes[1].plot(epochs, noisy5_acc, label="val_acc(5dB)", marker="o", linewidth=1.4)
    if noisy10_acc:
        axes[1].plot(epochs, noisy10_acc, label="val_acc(10dB)", marker="o", linewidth=1.4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_feature_names(smile: Smile, combine_mfcc: bool, feat_dim: int):
    """Build feature names aligned to current feature dimension."""
    opensmile_names = []
    if hasattr(smile, "feature_names"):
        try:
            opensmile_names = list(smile.feature_names)
        except Exception:
            opensmile_names = []

    if not opensmile_names:
        opensmile_names = [f"opensmile_{i}" for i in range(feat_dim)]

    if combine_mfcc:
        mfcc_idx = set(get_opensmile_mfcc_channel_indices(smile))
        opensmile_names = [name for i, name in enumerate(opensmile_names) if i not in mfcc_idx]

    names = opensmile_names
    if combine_mfcc:
        mfcc_names = [f"mfcc_{i+1}" for i in range(13)]
        names = names + mfcc_names

    if len(names) < feat_dim:
        names = names + [f"feat_{i}" for i in range(len(names), feat_dim)]
    elif len(names) > feat_dim:
        names = names[:feat_dim]
    return names


def build_mfcc_feature_names(feat_dim: int):
    return [f"mfcc_{i+1}" for i in range(feat_dim)]


def build_mfcc_delta_feature_names(feat_dim: int):
    if feat_dim % 3 != 0:
        return [f"mfcc_feat_{i+1}" for i in range(feat_dim)]
    base_dim = feat_dim // 3
    names = [f"mfcc_{i+1}" for i in range(base_dim)]
    names += [f"mfcc_delta_{i+1}" for i in range(base_dim)]
    names += [f"mfcc_delta2_{i+1}" for i in range(base_dim)]
    return names


def build_pncc_feature_names(feat_dim: int):
    return [f"pncc_{i+1}" for i in range(feat_dim)]


def build_pncc_delta_feature_names(feat_dim: int):
    if feat_dim % 3 != 0:
        return [f"pncc_feat_{i+1}" for i in range(feat_dim)]
    base_dim = feat_dim // 3
    names = [f"pncc_{i+1}" for i in range(base_dim)]
    names += [f"pncc_delta_{i+1}" for i in range(base_dim)]
    names += [f"pncc_delta2_{i+1}" for i in range(base_dim)]
    return names


def build_pncc_variant_feature_names(feat_dim: int, add_delta=False, add_delta2=False):
    base_divisor = 1 + int(add_delta) + int(add_delta2)
    if feat_dim % base_divisor != 0:
        return [f"pncc_feat_{i+1}" for i in range(feat_dim)]
    base_dim = feat_dim // base_divisor
    names = [f"pncc_{i+1}" for i in range(base_dim)]
    if add_delta:
        names += [f"pncc_delta_{i+1}" for i in range(base_dim)]
    if add_delta2:
        names += [f"pncc_delta2_{i+1}" for i in range(base_dim)]
    return names


def log_opensmile_features(
    logger,
    smile: Smile,
    feature_level: str,
    combine_mfcc: bool,
    opensmile_feature_set: str,
):
    """Log OpenSMILE feature configuration and names."""
    names = []
    if hasattr(smile, "feature_names"):
        try:
            names = list(smile.feature_names)
        except Exception:
            names = []

    logger.info(
        "OpenSMILE config | feature_set=%s feature_level=%s opensmile_feature_count=%d combine_mfcc=%s",
        opensmile_feature_set,
        feature_level,
        len(names),
        combine_mfcc,
    )
    logger.info("OpenSMILE feature names: %s", names)


def log_and_save_selected_features(logger, selected_feature_names, original_idx, output_path):
    selected_names = list(selected_feature_names)
    saved_indices = np.array(original_idx, dtype=np.int64).tolist()
    logger.info("Selected feature count: %d", len(selected_names))
    logger.info("Selected features (first 40): %s", selected_names[:40])

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("selected_rank,original_index,feature_name\n")
        for rank, (idx, name) in enumerate(zip(saved_indices, selected_names)):
            f.write(f"{rank},{idx},{name}\n")
    logger.info("Saved selected feature list to %s", output_path)


def save_clean_digit_feature_summary(
    logger,
    clean_feats,
    clean_labels,
    feature_names,
    selected_idx,
    output_csv,
    max_features=30,
):
    """Save a compact per-digit summary CSV from clean dataset features."""
    if not clean_feats or not clean_labels:
        logger.warning("Clean feature summary skipped: empty clean dataset.")
        return

    chosen_idx = np.array(selected_idx, dtype=np.int64)
    if chosen_idx.size == 0:
        logger.warning("Clean feature summary skipped: no selected features.")
        return
    if chosen_idx.size > max_features:
        chosen_idx = chosen_idx[:max_features]

    # Per-utterance mean over time, then per-digit mean over utterances.
    per_utt_means = [x.mean(axis=1) for x in clean_feats]
    by_digit = {}
    for vec, lab in zip(per_utt_means, clean_labels):
        d = int(lab)
        by_digit.setdefault(d, []).append(vec)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    feature_cols = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in chosen_idx]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["digit"] + feature_cols)
        for digit in sorted(by_digit.keys()):
            mat = np.stack(by_digit[digit], axis=0)  # (N, F)
            digit_mean = mat.mean(axis=0)
            row = [digit] + [float(digit_mean[i]) for i in chosen_idx]
            writer.writerow(row)

    logger.info(
        "Saved clean per-digit feature summary to %s (digits=%d, features=%d)",
        output_csv,
        len(by_digit),
        len(chosen_idx),
    )


def save_digit_feature_summary(
    logger,
    split_name,
    feats,
    labels,
    feature_names,
    selected_idx,
    output_csv,
    max_features=30,
):
    """Save a compact per-digit summary CSV for any split."""
    if not feats or not labels:
        logger.warning("%s feature summary skipped: empty dataset.", split_name)
        return

    chosen_idx = np.array(selected_idx, dtype=np.int64)
    if chosen_idx.size == 0:
        logger.warning("%s feature summary skipped: no selected features.", split_name)
        return
    if chosen_idx.size > max_features:
        chosen_idx = chosen_idx[:max_features]

    per_utt_means = [x.mean(axis=1) for x in feats]
    by_digit = {}
    for vec, lab in zip(per_utt_means, labels):
        d = int(lab)
        by_digit.setdefault(d, []).append(vec)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    feature_cols = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in chosen_idx]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["digit"] + feature_cols)
        for digit in sorted(by_digit.keys()):
            mat = np.stack(by_digit[digit], axis=0)
            digit_mean = mat.mean(axis=0)
            row = [digit] + [float(digit_mean[i]) for i in chosen_idx]
            writer.writerow(row)

    logger.info(
        "Saved %s per-digit feature summary to %s (digits=%d, features=%d)",
        split_name,
        output_csv,
        len(by_digit),
        len(chosen_idx),
    )


def log_split_metrics(logger, split_name, loss_value, metric_dict):
    logger.info(
        "[%s] loss=%.4f acc=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f "
        "precision_weighted=%.4f recall_weighted=%.4f f1_weighted=%.4f",
        split_name,
        loss_value,
        metric_dict["accuracy"],
        metric_dict["precision_macro"],
        metric_dict["recall_macro"],
        metric_dict["f1_macro"],
        metric_dict["precision_weighted"],
        metric_dict["recall_weighted"],
        metric_dict["f1_weighted"],
    )
    for digit_metrics in metric_dict.get("per_digit", []):
        logger.info(
            "[%s][digit=%d] accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f support=%d",
            split_name,
            digit_metrics["digit"],
            digit_metrics["accuracy"],
            digit_metrics["precision"],
            digit_metrics["recall"],
            digit_metrics["f1"],
            digit_metrics["support"],
        )


def save_per_digit_metrics_csv(logger, split_name, metric_dict, output_csv):
    per_digit_metrics = metric_dict.get("per_digit", [])
    if not per_digit_metrics:
        logger.warning("%s per-digit metrics CSV skipped: no per-digit metrics available.", split_name)
        return

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["digit", "accuracy", "precision", "recall", "f1", "support"])
        for digit_metrics in per_digit_metrics:
            writer.writerow(
                [
                    digit_metrics["digit"],
                    digit_metrics["accuracy"],
                    digit_metrics["precision"],
                    digit_metrics["recall"],
                    digit_metrics["f1"],
                    digit_metrics["support"],
                ]
            )

    logger.info("Saved %s per-digit metrics CSV to %s", split_name, output_csv)


def save_metrics_summary_csv(logger, split_metrics, output_csv):
    if not split_metrics:
        logger.warning("Metrics summary CSV skipped: no split metrics available.")
        return

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    split_names = list(split_metrics.keys())
    overall_fields = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scope",
                "split",
                "digit",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "per_digit_accuracy",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        )

        overall_avg = {
            field: float(np.mean([split_metrics[name][field] for name in split_names]))
            for field in overall_fields
        }
        writer.writerow(
            [
                "overall_average",
                "all",
                "all",
                overall_avg["accuracy"],
                overall_avg["precision_macro"],
                overall_avg["recall_macro"],
                overall_avg["f1_macro"],
                overall_avg["precision_weighted"],
                overall_avg["recall_weighted"],
                overall_avg["f1_weighted"],
                "",
                "",
                "",
                "",
                "",
            ]
        )

        for split_name in split_names:
            metric_dict = split_metrics[split_name]
            writer.writerow(
                [
                    "split_summary",
                    split_name,
                    "all",
                    metric_dict["accuracy"],
                    metric_dict["precision_macro"],
                    metric_dict["recall_macro"],
                    metric_dict["f1_macro"],
                    metric_dict["precision_weighted"],
                    metric_dict["recall_weighted"],
                    metric_dict["f1_weighted"],
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        per_digit_by_digit = {}
        for split_name in split_names:
            for digit_metrics in split_metrics[split_name].get("per_digit", []):
                digit = int(digit_metrics["digit"])
                per_digit_by_digit.setdefault(digit, []).append(digit_metrics)
                writer.writerow(
                    [
                        "per_digit",
                        split_name,
                        digit,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        digit_metrics.get("accuracy", digit_metrics["recall"]),
                        digit_metrics["precision"],
                        digit_metrics["recall"],
                        digit_metrics["f1"],
                        digit_metrics["support"],
                    ]
                )

        for digit in sorted(per_digit_by_digit.keys()):
            rows = per_digit_by_digit[digit]
            writer.writerow(
                [
                    "per_digit_average",
                    "all",
                    digit,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    float(np.mean([row.get("accuracy", row["recall"]) for row in rows])),
                    float(np.mean([row["precision"] for row in rows])),
                    float(np.mean([row["recall"] for row in rows])),
                    float(np.mean([row["f1"] for row in rows])),
                    int(np.sum([row["support"] for row in rows])),
                ]
            )

    logger.info("Saved metrics summary CSV to %s", output_csv)


def load_checkpoint_payload(checkpoint_path, device):
    """Load checkpoint payload and return (state_dict, metadata)."""
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload
    elif isinstance(payload, dict):
        state_dict = payload
        metadata = {}
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    return state_dict, metadata


def resolve_device(device_arg: str) -> torch.device:
    logger = logging.getLogger(__name__)
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_dataloaders(
    train_feat,
    train_label,
    test_feat,
    test_label,
    noisy5_feat,
    noisy5_label,
    noisy10_feat,
    noisy10_label,
    batch_size,
    seed,
    train_files=None,
    feature_source="mfcc",
    smile=None,
    augment_train_noise=False,
    train_aug_prob=0.5,
    train_aug_snr_min=3.0,
    train_aug_snr_max=15.0,
):
    logger = logging.getLogger(__name__)

    y_train = np.array(train_label, dtype=np.int64)
    y_test = np.array(test_label, dtype=np.int64)
    y_noisy5 = np.array(noisy5_label, dtype=np.int64)
    y_noisy10 = np.array(noisy10_label, dtype=np.int64)
    logger.info(
        "Class distribution | train=%s test_clean=%s",
        np.bincount(y_train, minlength=10).tolist() if len(y_train) else [],
        np.bincount(y_test, minlength=10).tolist() if len(y_test) else [],
    )

    waveform_aug_supported = feature_source in {
        "mfcc",
        "mfcc_delta_cmvn",
        "pncc",
        "pncc_cmvn",
        "pncc_delta_only",
        "pncc_delta_cmvn_only",
        "pncc_delta",
        "pncc_delta2_cmvn",
        "pncc_delta_cmvn",
    }
    use_audio_train_dataset = (
        augment_train_noise and waveform_aug_supported and train_files is not None
    )
    if augment_train_noise and not waveform_aug_supported:
        logger.warning(
            "Waveform-level train augmentation is currently unsupported for feature_source=%s. "
            "Training will proceed without augmentation.",
            feature_source,
        )

    loader_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        AudioFeatureDataset(
            train_files,
            y_train,
            feature_source=feature_source,
            smile=smile,
            augment_noise=augment_train_noise,
            aug_prob=train_aug_prob,
            aug_snr_min=train_aug_snr_min,
            aug_snr_max=train_aug_snr_max,
            seed=seed,
        )
        if use_audio_train_dataset
        else FeatureDataset(train_feat, y_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad,
        generator=loader_g,
    )
    test_loader = DataLoader(
        FeatureDataset(test_feat, y_test),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_pad,
    )

    noisy5_loader = None
    if noisy5_feat:
        noisy5_loader = DataLoader(
            FeatureDataset(noisy5_feat, y_noisy5),
            batch_size=16,
            shuffle=False,
            collate_fn=collate_pad,
        )

    noisy10_loader = None
    if noisy10_feat:
        noisy10_loader = DataLoader(
            FeatureDataset(noisy10_feat, y_noisy10),
            batch_size=16,
            shuffle=False,
            collate_fn=collate_pad,
        )

    logger.info(
        "DataLoaders ready | train_batches=%d test_clean_batches=%d train_dataset_mode=%s",
        len(train_loader),
        len(test_loader),
        "audio_onthefly" if use_audio_train_dataset else "precomputed_features",
    )
    if noisy5_loader is not None:
        logger.info("Noisy 5dB loader batches=%d", len(noisy5_loader))
    if noisy10_loader is not None:
        logger.info("Noisy 10dB loader batches=%d", len(noisy10_loader))

    return train_loader, test_loader, noisy5_loader, noisy10_loader


def train_model(
    net,
    train_loader,
    test_loader,
    noisy5_loader,
    noisy10_loader,
    device,
    num_epochs,
    lr,
    checkpoint_path,
):
    logger = logging.getLogger(__name__)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "noisy5_loss": [],
        "noisy10_loss": [],
        "train_acc": [],
        "val_acc": [],
        "noisy5_acc": [],
        "noisy10_acc": [],
    }

    best_clean, best_clean_ep = 0.0, -1
    best_5db, best_5db_ep = 0.0, -1
    best_10db, best_10db_ep = 0.0, -1
    saved_checkpoint = None

    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    last_clean_acc, last_acc_5db, last_acc_10db = 0.0, 0.0, 0.0

    for epoch in range(1, num_epochs + 1):
        net.train()
        epoch_start = time.time()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb, lengths in train_loader:
            xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits = net(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_loss, val_acc = evaluate_loss_and_accuracy(net, test_loader, device, criterion)
        noisy5_loss, noisy5_acc = evaluate_loss_and_accuracy(net, noisy5_loader, device, criterion)
        noisy10_loss, noisy10_acc = evaluate_loss_and_accuracy(net, noisy10_loader, device, criterion)
        clean_acc = evaluate(net, test_loader, device)
        acc_5db = evaluate(net, noisy5_loader, device)
        acc_10db = evaluate(net, noisy10_loader, device)
        last_clean_acc, last_acc_5db, last_acc_10db = clean_acc, acc_5db, acc_10db
        history["epochs"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["noisy5_loss"].append(noisy5_loss)
        history["noisy10_loss"].append(noisy10_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["noisy5_acc"].append(noisy5_acc)
        history["noisy10_acc"].append(noisy10_acc)

        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss(clean)=%.4f | val_loss(5dB)=%.4f | "
            "val_loss(10dB)=%.4f | train_acc=%.4f | val_acc(clean)=%.4f | val_acc(5dB)=%.4f | "
            "val_acc(10dB)=%.4f | time=%.2fs",
            epoch,
            num_epochs,
            avg_loss,
            val_loss,
            noisy5_loss,
            noisy10_loss,
            train_acc,
            val_acc,
            noisy5_acc,
            noisy10_acc,
            time.time() - epoch_start,
        )

        if clean_acc > best_clean:
            best_clean, best_clean_ep = clean_acc, epoch
            logger.info("New best clean accuracy: %.4f at epoch %d", best_clean, best_clean_ep)
        if acc_5db > best_5db:
            best_5db, best_5db_ep = acc_5db, epoch
            logger.info("New best 5dB accuracy: %.4f at epoch %d", best_5db, best_5db_ep)
        if acc_10db > best_10db:
            best_10db, best_10db_ep = acc_10db, epoch
            saved_checkpoint = copy.deepcopy(net.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": saved_checkpoint,
                    "clean_acc": clean_acc,
                    "acc_5db": acc_5db,
                    "acc_10db": acc_10db,
                    "history": history,
                },
                checkpoint_path,
            )
            logger.info("Saved best checkpoint to %s (epoch %d)", checkpoint_path, epoch)

    if saved_checkpoint is None:
        logger.warning("No checkpoint was selected during training; using final trained model weights.")
        saved_checkpoint = copy.deepcopy(net.state_dict())
        torch.save(
            {
                "epoch": max(num_epochs, 0),
                "model_state_dict": saved_checkpoint,
                "clean_acc": last_clean_acc,
                "acc_5db": last_acc_5db,
                "acc_10db": last_acc_10db,
                "history": history,
            },
            checkpoint_path,
        )
        logger.info("Saved fallback checkpoint to %s", checkpoint_path)
    elif os.path.isfile(checkpoint_path):
        try:
            saved_checkpoint, loaded_meta = load_checkpoint_payload(checkpoint_path, device)
            logger.info("Loaded best checkpoint from %s", checkpoint_path)
        except Exception as exc:
            logger.warning(
                "Failed to load checkpoint from %s (%s). Using in-memory checkpoint.",
                checkpoint_path,
                exc,
            )

    return {
        "best_clean": best_clean,
        "best_clean_ep": best_clean_ep,
        "best_5db": best_5db,
        "best_5db_ep": best_5db_ep,
        "best_10db": best_10db,
        "best_10db_ep": best_10db_ep,
        "state_dict": saved_checkpoint,
        "history": history,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    add_file_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    run_start = time.time()
    logger.info("Starting run_local.py")
    logger.info(
        "Args | dataset_zip=%s extract_dir=%s seed=%d batch_size=%d num_epochs=%d lr=%g "
        "device=%s feature_source=%s feature_level=%s opensmile_feature_set=%s "
        "use_xgb_feature_selection=%s xgb_val_ratio=%.3f "
        "augment_train_noise=%s train_aug_prob=%.3f train_aug_snr_min=%.2f train_aug_snr_max=%.2f "
        "combine_mfcc=%s skip_train=%s checkpoint_path=%s save_cm_dir=%s "
        "save_feature_summary=%s feature_summary_max_features=%d",
        args.dataset_zip,
        args.extract_dir,
        args.seed,
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.device,
        args.feature_source,
        args.feature_level,
        args.opensmile_feature_set,
        args.use_xgb_feature_selection,
        args.xgb_val_ratio,
        args.augment_train_noise,
        args.train_aug_prob,
        args.train_aug_snr_min,
        args.train_aug_snr_max,
        args.combine_mfcc,
        args.skip_train,
        args.checkpoint_path,
        args.save_cm_dir if args.save_cm_dir else "(show)",
        args.save_feature_summary,
        args.feature_summary_max_features,
    )

    set_seed(args.seed)

    data_root = ensure_dataset_ready(args.dataset_zip, args.extract_dir)
    train_dir = os.path.join(data_root, "train_clean")
    test_clean_dir = os.path.join(data_root, "test_clean")
    test_noisy_5db_dir = os.path.join(data_root, "test_snr_5db_babble")
    test_noisy_10db_dir = os.path.join(data_root, "test_snr_10db_babble")
    train_files = list_audio_files(train_dir)

    feature_source = args.feature_source
    if feature_source == "opensmile" and args.combine_mfcc:
        logger.warning(
            "`--combine-mfcc true` with `--feature-source opensmile` is deprecated. "
            "Using `opensmile_mfcc` behavior for backward compatibility."
        )
        feature_source = "opensmile_mfcc"
    if feature_source == "mfcc" and args.combine_mfcc:
        logger.warning("Ignoring `--combine-mfcc` because `--feature-source mfcc` is active.")

    smile = None
    if feature_source not in {"mfcc", "mfcc_delta_cmvn", "pncc", "pncc_cmvn", "pncc_delta_only", "pncc_delta_cmvn_only", "pncc_delta", "pncc_delta2_cmvn", "pncc_delta_cmvn"}:
        smile = build_smile(args.feature_level, args.opensmile_feature_set)
        log_opensmile_features(
            logger,
            smile,
            args.feature_level,
            feature_source in {"opensmile_mfcc", "opensmile_pncc"},
            args.opensmile_feature_set,
        )
    elif feature_source == "mfcc":
        logger.info("Using MFCC-only baseline features.")
    elif feature_source == "mfcc_delta_cmvn":
        logger.info("Using MFCC+delta+delta-delta with CMVN features.")
    elif feature_source == "pncc":
        logger.info("Using PNCC-only baseline features.")
    elif feature_source == "pncc_cmvn":
        logger.info("Using PNCC+CMVN features.")
    elif feature_source == "pncc_delta_only":
        logger.info("Using PNCC+delta features.")
    elif feature_source == "pncc_delta_cmvn_only":
        logger.info("Using PNCC+delta with CMVN features.")
    elif feature_source == "pncc_delta":
        logger.info("Using PNCC+delta+delta-delta features.")
    elif feature_source == "pncc_delta2_cmvn":
        logger.info("Using PNCC+delta-delta with CMVN features.")
    else:
        logger.info("Using PNCC+delta+delta-delta with CMVN features.")

    train_feat, train_label = load_feature_split(train_dir, feature_source, smile=smile, desc="Train")
    test_feat, test_label = load_feature_split(test_clean_dir, feature_source, smile=smile, desc="Test clean")
    noisy5_feat, noisy5_label = load_feature_split(
        test_noisy_5db_dir, feature_source, smile=smile, desc="Test noisy 5dB"
    )
    noisy10_feat, noisy10_label = load_feature_split(
        test_noisy_10db_dir, feature_source, smile=smile, desc="Test noisy 10dB"
    )

    if not train_feat:
        logger.error("No training features loaded. Check dataset paths and extraction.")
        return 1
    logger.info(
        "Loaded datasets | train=%d test_clean=%d test_5db=%d test_10db=%d",
        len(train_feat),
        len(test_feat),
        len(noisy5_feat),
        len(noisy10_feat),
    )
    selected_idx_original = np.arange(train_feat[0].shape[0], dtype=np.int64)
    selected_idx_for_summary = np.arange(train_feat[0].shape[0], dtype=np.int64)
    if feature_source == "mfcc":
        feature_names = build_mfcc_feature_names(train_feat[0].shape[0])
    elif feature_source == "mfcc_delta_cmvn":
        feature_names = build_mfcc_delta_feature_names(train_feat[0].shape[0])
    elif feature_source == "pncc":
        feature_names = build_pncc_feature_names(train_feat[0].shape[0])
    elif feature_source == "pncc_cmvn":
        feature_names = build_pncc_feature_names(train_feat[0].shape[0])
    elif feature_source == "pncc_delta_only":
        feature_names = build_pncc_variant_feature_names(train_feat[0].shape[0], add_delta=True, add_delta2=False)
    elif feature_source == "pncc_delta_cmvn_only":
        feature_names = build_pncc_variant_feature_names(train_feat[0].shape[0], add_delta=True, add_delta2=False)
    elif feature_source in {"pncc_delta", "pncc_delta_cmvn"}:
        feature_names = build_pncc_delta_feature_names(train_feat[0].shape[0])
    elif feature_source == "pncc_delta2_cmvn":
        feature_names = build_pncc_variant_feature_names(train_feat[0].shape[0], add_delta=False, add_delta2=True)
    else:
        feature_names = build_feature_names(
            smile,
            feature_source in {"opensmile_mfcc", "opensmile_pncc"},
            train_feat[0].shape[0],
        )

    if args.use_xgb_feature_selection and feature_source not in {"mfcc", "mfcc_delta_cmvn", "pncc", "pncc_cmvn", "pncc_delta_only", "pncc_delta_cmvn_only", "pncc_delta", "pncc_delta2_cmvn", "pncc_delta_cmvn"}:
        if XGBClassifier is None:
            logger.warning("XGBoost not installed; skipping feature selection.")
        else:
            y_train_for_fs = np.array(train_label, dtype=np.int64)
            selected_idx_original = select_robust_features_with_xgb(
                train_feat,
                y_train_for_fs,
                seed=args.seed,
                val_ratio=args.xgb_val_ratio,
            )
            train_feat = apply_selected_channels(train_feat, selected_idx_original)
            test_feat = apply_selected_channels(test_feat, selected_idx_original) if test_feat else test_feat
            noisy5_feat = (
                apply_selected_channels(noisy5_feat, selected_idx_original) if noisy5_feat else noisy5_feat
            )
            noisy10_feat = (
                apply_selected_channels(noisy10_feat, selected_idx_original) if noisy10_feat else noisy10_feat
            )
            feature_names = [
                feature_names[i] if i < len(feature_names) else f"feat_{i}"
                for i in selected_idx_original
            ]
            logger.info("[XGB FS] Applied selected channels. New feature dim: %d", train_feat[0].shape[0])
            selected_idx_for_summary = np.arange(len(feature_names), dtype=np.int64)
    elif args.use_xgb_feature_selection and feature_source in {"mfcc", "mfcc_delta_cmvn", "pncc", "pncc_cmvn", "pncc_delta_only", "pncc_delta_cmvn_only", "pncc_delta", "pncc_delta2_cmvn", "pncc_delta_cmvn"}:
        logger.info(
            "Skipping XGBoost feature selection because %s-only baseline mode is active.",
            feature_source.upper(),
        )

    if args.save_feature_summary:
        summary_dir = "./features"
        ckpt_prefix = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        os.makedirs(summary_dir, exist_ok=True)
        selected_feature_path = os.path.join(summary_dir, f"{ckpt_prefix}_selected_features.csv")
        clean_summary_path = os.path.join(summary_dir, f"{ckpt_prefix}_clean_per_digit_feature_summary.csv")
        log_and_save_selected_features(logger, feature_names, selected_idx_original, selected_feature_path)
        save_digit_feature_summary(
            logger,
            "test_clean",
            test_feat,
            test_label,
            feature_names,
            selected_idx_for_summary,
            clean_summary_path,
            max_features=max(1, args.feature_summary_max_features),
        )
        noisy5_summary_path = os.path.join(summary_dir, f"{ckpt_prefix}_noisy5db_per_digit_feature_summary.csv")
        noisy10_summary_path = os.path.join(summary_dir, f"{ckpt_prefix}_noisy10db_per_digit_feature_summary.csv")
        save_digit_feature_summary(
            logger,
            "test_noisy_5db",
            noisy5_feat,
            noisy5_label,
            feature_names,
            selected_idx_for_summary,
            noisy5_summary_path,
            max_features=max(1, args.feature_summary_max_features),
        )
        save_digit_feature_summary(
            logger,
            "test_noisy_10db",
            noisy10_feat,
            noisy10_label,
            feature_names,
            selected_idx_for_summary,
            noisy10_summary_path,
            max_features=max(1, args.feature_summary_max_features),
        )

    feat_dim = train_feat[0].shape[0]
    logger.info(
        "Feature dim: %d | Train: %d | Test clean: %d | Test noisy 5dB: %d | Test noisy 10dB: %d",
        feat_dim,
        len(train_feat),
        len(test_feat),
        len(noisy5_feat),
        len(noisy10_feat),
    )

    for name, flist in [
        ("Train", train_feat),
        ("Test clean", test_feat),
        ("Noisy 5dB", noisy5_feat),
        ("Noisy 10dB", noisy10_feat),
    ]:
        if flist:
            lengths = [f.shape[1] for f in flist]
            logger.info(
                "%s frames: min=%d, max=%d, mean=%.1f",
                name,
                min(lengths),
                max(lengths),
                float(np.mean(lengths)),
            )

    train_loader, test_loader, noisy5_loader, noisy10_loader = build_dataloaders(
        train_feat,
        train_label,
        test_feat,
        test_label,
        noisy5_feat,
        noisy5_label,
        noisy10_feat,
        noisy10_label,
        batch_size=args.batch_size,
        seed=args.seed,
        train_files=train_files,
        feature_source=feature_source,
        smile=smile,
        augment_train_noise=args.augment_train_noise,
        train_aug_prob=args.train_aug_prob,
        train_aug_snr_min=args.train_aug_snr_min,
        train_aug_snr_max=args.train_aug_snr_max,
    )

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    net = SimpleLSTM(input_size=feat_dim).to(device)
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Model initialized | input_size=%d trainable_params=%d", feat_dim, trainable_params)

    if args.skip_train:
        if not os.path.isfile(args.checkpoint_path):
            logger.error("Checkpoint not found for --skip-train: %s", args.checkpoint_path)
            return 1
        saved_checkpoint, ckpt_meta = load_checkpoint_payload(args.checkpoint_path, device)
        best_clean = float(ckpt_meta.get("clean_acc", 0.0)) if isinstance(ckpt_meta, dict) else 0.0
        best_5db = float(ckpt_meta.get("acc_5db", 0.0)) if isinstance(ckpt_meta, dict) else 0.0
        best_10db = float(ckpt_meta.get("acc_10db", 0.0)) if isinstance(ckpt_meta, dict) else 0.0
        best_clean_ep = int(ckpt_meta.get("epoch", -1)) if isinstance(ckpt_meta, dict) else -1
        best_5db_ep = int(ckpt_meta.get("epoch", -1)) if isinstance(ckpt_meta, dict) else -1
        best_10db_ep = int(ckpt_meta.get("epoch", -1)) if isinstance(ckpt_meta, dict) else -1
        history = ckpt_meta.get("history", {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}) if isinstance(ckpt_meta, dict) else {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        logger.info("Skip-train enabled. Loaded checkpoint from %s (epoch=%d)", args.checkpoint_path, best_10db_ep)
    else:
        train_summary = train_model(
            net,
            train_loader,
            test_loader,
            noisy5_loader,
            noisy10_loader,
            device=device,
            num_epochs=args.num_epochs,
            lr=args.lr,
            checkpoint_path=args.checkpoint_path,
        )

        best_clean = train_summary["best_clean"]
        best_clean_ep = train_summary["best_clean_ep"]
        best_5db = train_summary["best_5db"]
        best_5db_ep = train_summary["best_5db_ep"]
        best_10db = train_summary["best_10db"]
        best_10db_ep = train_summary["best_10db_ep"]
        saved_checkpoint = train_summary["state_dict"]
        history = train_summary["history"]

    net = SimpleLSTM(input_size=feat_dim).to(device)
    net.load_state_dict(saved_checkpoint)

    class_names = list(range(10))
    figure_prefix = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    if args.save_cm_dir:
        os.makedirs(args.save_cm_dir, exist_ok=True)
        clean_path = os.path.join(args.save_cm_dir, f"{figure_prefix}_cm_clean.png")
        noisy5_path = os.path.join(args.save_cm_dir, f"{figure_prefix}_cm_noisy5db.png")
        noisy10_path = os.path.join(args.save_cm_dir, f"{figure_prefix}_cm_noisy10db.png")
        logger.info(
            "Saving confusion matrices to %s with prefix '%s'",
            args.save_cm_dir,
            figure_prefix,
        )
        curve_path = save_training_curves(history, args.save_cm_dir, figure_prefix)
        if curve_path:
            logger.info("Saved training curves to %s", curve_path)
    else:
        clean_path = noisy5_path = noisy10_path = None

    clean_acc, _ = evaluate(
        net,
        test_loader,
        device,
        plot_cm=True,
        class_names=class_names,
        title="Confusion Matrix - Clean",
        save_path=clean_path,
    )
    acc_5db, _ = evaluate(
        net,
        noisy5_loader,
        device,
        plot_cm=True,
        class_names=class_names,
        title="Confusion Matrix - Noisy 5 dB",
        save_path=noisy5_path,
    )
    acc_10db, _ = evaluate(
        net,
        noisy10_loader,
        device,
        plot_cm=True,
        class_names=class_names,
        title="Confusion Matrix - Noisy 10 dB",
        save_path=noisy10_path,
    )

    metrics_clean = evaluate_metrics(net, test_loader, device)
    metrics_5db = evaluate_metrics(net, noisy5_loader, device)
    metrics_10db = evaluate_metrics(net, noisy10_loader, device)
    eval_criterion = nn.CrossEntropyLoss()
    clean_loss, _ = evaluate_loss_and_accuracy(net, test_loader, device, eval_criterion)
    loss_5db, _ = evaluate_loss_and_accuracy(net, noisy5_loader, device, eval_criterion)
    loss_10db, _ = evaluate_loss_and_accuracy(net, noisy10_loader, device, eval_criterion)

    logger.info("Best clean accuracy: %.4f (epoch %d)", best_clean, best_clean_ep)
    logger.info("Best 5dB accuracy: %.4f (epoch %d)", best_5db, best_5db_ep)
    logger.info("Best 10dB accuracy: %.4f (epoch %d)", best_10db, best_10db_ep)
    logger.info("Final loaded-checkpoint accuracy | clean=%.4f 5dB=%.4f 10dB=%.4f", clean_acc, acc_5db, acc_10db)
    logger.info("Final evaluation metrics on all test datasets:")
    log_split_metrics(logger, "test_clean", clean_loss, metrics_clean)
    log_split_metrics(logger, "test_noisy_5db", loss_5db, metrics_5db)
    log_split_metrics(logger, "test_noisy_10db", loss_10db, metrics_10db)

    metrics_dir = "./features"
    os.makedirs(metrics_dir, exist_ok=True)
    save_per_digit_metrics_csv(
        logger,
        "test_clean",
        metrics_clean,
        os.path.join(metrics_dir, f"{figure_prefix}_test_clean_per_digit_metrics.csv"),
    )
    save_per_digit_metrics_csv(
        logger,
        "test_noisy_5db",
        metrics_5db,
        os.path.join(metrics_dir, f"{figure_prefix}_test_noisy_5db_per_digit_metrics.csv"),
    )
    save_per_digit_metrics_csv(
        logger,
        "test_noisy_10db",
        metrics_10db,
        os.path.join(metrics_dir, f"{figure_prefix}_test_noisy_10db_per_digit_metrics.csv"),
    )
    save_metrics_summary_csv(
        logger,
        {
            "test_clean": metrics_clean,
            "test_noisy_5db": metrics_5db,
            "test_noisy_10db": metrics_10db,
        },
        os.path.join(metrics_dir, f"{figure_prefix}_metrics_summary.csv"),
    )
    logger.info("Run completed in %.2f minutes", (time.time() - run_start) / 60.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
