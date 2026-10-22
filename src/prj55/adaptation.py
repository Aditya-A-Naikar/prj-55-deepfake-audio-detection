"""Front-end signal adaptation and channel mitigation algorithms for PRJ_55.

Provides lightweight, non-parametric signal-processing filters that operate directly
on 1D waveforms prior to AASIST-L neural feature extraction. These filters mitigate
restricted-bandwidth telephony artifacts and additive background noise without altering
the ultra-lightweight <100k parameter budget of the neural model (REQ-01, REQ-10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from prj55.audio import TARGET_SAMPLE_RATE


class AdaptationError(Exception):
    """Raised when audio adaptation or filtering fails numeric validation."""


@dataclass(frozen=True)
class MitigationConfig:
    """Configuration parameters for front-end channel mitigation."""

    denoise_enabled: bool = True
    spectral_subtraction_factor: float = 1.5
    spectral_floor: float = 0.05
    bandpass_enabled: bool = True
    bandpass_low_hz: float = 120.0
    bandpass_high_hz: float = 7600.0
    normalize_energy: bool = True
    target_rms: float = 0.1


def apply_spectral_subtraction(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    alpha: float = 1.5,
    floor: float = 0.05,
) -> np.ndarray:
    """Apply spectral subtraction to suppress stationary background noise.

    Estimates the noise floor from initial low-energy spectral frames and subtracts
    the estimated noise magnitude with a spectral floor to avoid musical noise.

    Args:
        waveform: 1D float32 numpy array.
        sr: Sample rate in Hz (default 16000).
        alpha: Over-subtraction factor (higher values remove more noise).
        floor: Spectral floor parameter (prevents negative/zero values).

    Returns:
        Filtered 1D float32 numpy array of identical length.
    """
    if waveform.ndim != 1 or waveform.size == 0:
        raise AdaptationError("Waveform must be a non-empty 1D array")
    if not np.all(np.isfinite(waveform)):
        raise AdaptationError("Waveform contains non-finite samples")

    # Short-Time Fourier Transform
    n_fft = 512
    hop_length = 128
    win_length = 512
    window = signal.windows.hann(win_length)

    f, t_bins, zxx = signal.stft(
        waveform,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    # Estimate noise spectrum from the lowest 10% energy frames (or first 5 frames)
    frame_energy = np.sum(magnitude**2, axis=0)
    n_noise_frames = max(5, int(0.10 * magnitude.shape[1]))
    noise_frame_indices = np.argsort(frame_energy)[:n_noise_frames]
    noise_estimate = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)

    # Spectral subtraction with floor
    subtracted_mag = magnitude - (alpha * noise_estimate)
    clean_mag = np.maximum(subtracted_mag, floor * magnitude)

    # Inverse STFT
    complex_spectrum = clean_mag * np.exp(1j * phase)
    _, recovered = signal.istft(
        complex_spectrum,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
    )

    # Ensure output length matches input exactly
    target_len = len(waveform)
    if len(recovered) > target_len:
        recovered = recovered[:target_len]
    elif len(recovered) < target_len:
        recovered = np.pad(recovered, (0, target_len - len(recovered)))

    recovered = recovered.astype(np.float32)
    if not np.all(np.isfinite(recovered)):
        raise AdaptationError("Spectral subtraction produced non-finite values")

    return recovered


def apply_adaptive_bandpass(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    lowcut: float = 120.0,
    highcut: float = 7600.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter to eliminate sub-bass rumble and Nyquist artifacts.

    Args:
        waveform: 1D float32 numpy array.
        sr: Sample rate in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered 1D float32 numpy array.
    """
    if waveform.ndim != 1 or waveform.size == 0:
        raise AdaptationError("Waveform must be a non-empty 1D array")
    if not np.all(np.isfinite(waveform)):
        raise AdaptationError("Waveform contains non-finite samples")

    nyq = 0.5 * sr
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)

    if low >= high:
        raise AdaptationError(f"Invalid bandpass cutoffs: low={lowcut}Hz, high={highcut}Hz")

    b, a = signal.butter(order, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, waveform).astype(np.float32)

    if not np.all(np.isfinite(filtered)):
        raise AdaptationError("Bandpass filtering produced non-finite values")

    return filtered


def normalize_signal_energy(
    waveform: np.ndarray,
    target_rms: float = 0.1,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize waveform RMS energy to a standardized reference level.

    Args:
        waveform: 1D float32 numpy array.
        target_rms: Target root-mean-square amplitude.
        eps: Epsilon to prevent divide-by-zero on silent signals.

    Returns:
        Normalized 1D float32 numpy array.
    """
    if waveform.ndim != 1 or waveform.size == 0:
        raise AdaptationError("Waveform must be a non-empty 1D array")

    current_rms = float(np.sqrt(np.mean(waveform**2)))
    if current_rms < eps:
        # Near-silent signal, return as-is
        return waveform.copy().astype(np.float32)

    scale = target_rms / current_rms
    normalized = (waveform * scale).astype(np.float32)

    # Soft clip if peaks exceed [-1.0, 1.0]
    peak = float(np.max(np.abs(normalized)))
    if peak > 1.0:
        normalized = normalized / peak

    return normalized


def apply_front_end_mitigation(
    waveform: np.ndarray,
    config: MitigationConfig | None = None,
    sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Apply the complete front-end channel mitigation pipeline to a waveform.

    Executes bandpass filtering, spectral subtraction denoising, and RMS energy
    normalization in sequence to recover speech acoustic cues before neural scoring.

    Args:
        waveform: 1D float32 numpy array.
        config: Optional MitigationConfig dataclass.
        sr: Sample rate in Hz (default 16000).

    Returns:
        Mitigated 1D float32 numpy array preserving original length.
    """
    cfg = config or MitigationConfig()
    output = waveform.copy().astype(np.float32)

    if cfg.bandpass_enabled:
        output = apply_adaptive_bandpass(
            output,
            sr=sr,
            lowcut=cfg.bandpass_low_hz,
            highcut=cfg.bandpass_high_hz,
        )

    if cfg.denoise_enabled:
        output = apply_spectral_subtraction(
            output,
            sr=sr,
            alpha=cfg.spectral_subtraction_factor,
            floor=cfg.spectral_floor,
        )

    if cfg.normalize_energy:
        output = normalize_signal_energy(
            output,
            target_rms=cfg.target_rms,
        )

    return output
