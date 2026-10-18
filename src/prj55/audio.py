"""Audio loading, validation, and preprocessing pipeline for PRJ_55.

Follows official ASVspoof and AASIST specifications:
- Sample rate: 16,000 Hz
- Channels: Mono (downmixed by mean if stereo/multichannel)
- Target length: 64,600 samples (~4.0375 seconds) via repeat-padding or truncation
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

TARGET_SAMPLE_RATE: int = 16000
TARGET_SAMPLES: int = 64600


class AudioValidationError(Exception):
    """Raised when audio fails integrity, format, or dimensional validation."""


def load_audio(
    audio_path: str | Path,
    expected_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Load an audio file, validate integrity, and downmix to mono float32.

    Args:
        audio_path: Path to .flac or .wav audio file.
        expected_sr: Required sample rate (default 16000 Hz).

    Returns:
        1D numpy array of shape (samples,) with float32 values.

    Raises:
        FileNotFoundError: If audio file does not exist.
        AudioValidationError: If file is empty, corrupted, non-finite, or has incorrect sample rate.
    """
    path_obj = Path(audio_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if path_obj.stat().st_size == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): {audio_path}")

    try:
        data, sr = sf.read(str(path_obj), dtype="float32")
    except Exception as e:
        msg = f"Corrupt or unsupported audio file '{path_obj.name}': {e}"
        raise AudioValidationError(msg) from e

    if data.size == 0:
        raise AudioValidationError(f"Decoded audio stream contains 0 samples: {audio_path}")

    if sr != expected_sr:
        raise AudioValidationError(
            f"Sample rate mismatch for '{path_obj.name}': expected {expected_sr} Hz, got {sr} Hz"
        )

    # Validate numeric integrity (no NaNs or Infs)
    if not np.all(np.isfinite(data)):
        raise AudioValidationError(
            f"Audio file contains non-finite samples (NaN/Inf): {audio_path}"
        )

    # Downmix multi-channel audio to mono
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    elif data.ndim > 2:
        raise AudioValidationError(f"Unsupported audio channel dimension: {data.ndim}")

    return data


def pad_or_truncate_waveform(
    waveform: np.ndarray | Tensor,
    target_length: int = TARGET_SAMPLES,
) -> Tensor:
    """Apply upstream AASIST repeat-padding or truncation to reach exact sample length.

    Args:
        waveform: 1D array or tensor of shape (samples,).
        target_length: Required sample length (default 64,600).

    Returns:
        Tensor of shape `(1, target_length)` in float32.
    """
    if isinstance(waveform, np.ndarray):
        tensor = torch.from_numpy(waveform).float()
    else:
        tensor = waveform.float()

    if tensor.dim() == 0 or tensor.numel() == 0:
        raise AudioValidationError("Cannot process empty audio waveform")

    if tensor.dim() > 1:
        tensor = tensor.flatten()

    cur_len = tensor.size(0)

    if cur_len < target_length:
        num_repeats = math.ceil(target_length / cur_len)
        tensor = tensor.repeat(num_repeats)[:target_length]
    elif cur_len > target_length:
        tensor = tensor[:target_length]

    return tensor.unsqueeze(0)  # Shape: [1, 64600]


def preprocess_audio(
    audio_path: str | Path,
    expected_sr: int = TARGET_SAMPLE_RATE,
    target_length: int = TARGET_SAMPLES,
) -> Tensor:
    """Full audio pipeline: load, validate, mono-convert, and pad/truncate to tensor.

    Args:
        audio_path: Path to audio file.
        expected_sr: Target sample rate in Hz.
        target_length: Target frame length in samples.

    Returns:
        Tensor of shape `(1, target_length)` ready for baseline model ingestion.
    """
    raw_samples = load_audio(audio_path, expected_sr=expected_sr)
    return pad_or_truncate_waveform(raw_samples, target_length=target_length)
