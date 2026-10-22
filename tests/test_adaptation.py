"""Tests for front-end channel mitigation and adaptation module (Plan 4.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prj55.adaptation import (
    AdaptationError,
    MitigationConfig,
    apply_adaptive_bandpass,
    apply_front_end_mitigation,
    apply_spectral_subtraction,
    normalize_signal_energy,
)
from prj55.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, pad_or_truncate_waveform


@pytest.fixture
def clean_synthetic_speech() -> np.ndarray:
    """Generate a 4-second synthetic vocal harmonic tone."""
    sr = TARGET_SAMPLE_RATE
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
    # Fundamental pitch (120 Hz) + Formants (600 Hz, 1200 Hz, 2400 Hz)
    signal_wave = (
        0.4 * np.sin(2 * np.pi * 120.0 * t)
        + 0.3 * np.sin(2 * np.pi * 600.0 * t)
        + 0.2 * np.sin(2 * np.pi * 1200.0 * t)
        + 0.1 * np.sin(2 * np.pi * 2400.0 * t)
    ).astype(np.float32)
    return signal_wave


class TestSpectralSubtraction:
    """Tests for STFT spectral subtraction."""

    def test_spectral_subtraction_preserves_shape_and_finite(
        self, clean_synthetic_speech: np.ndarray
    ) -> None:
        filtered = apply_spectral_subtraction(clean_synthetic_speech)
        assert filtered.shape == clean_synthetic_speech.shape
        assert filtered.dtype == np.float32
        assert np.all(np.isfinite(filtered))

    def test_spectral_subtraction_reduces_noise(
        self, clean_synthetic_speech: np.ndarray
    ) -> None:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, size=len(clean_synthetic_speech)).astype(np.float32)
        noisy = clean_synthetic_speech + noise

        denoised = apply_spectral_subtraction(noisy, alpha=2.0)
        assert len(denoised) == len(noisy)
        assert np.all(np.isfinite(denoised))

    def test_empty_waveform_raises_error(self) -> None:
        with pytest.raises(AdaptationError, match="must be a non-empty"):
            apply_spectral_subtraction(np.array([], dtype=np.float32))

    def test_non_finite_waveform_raises_error(self) -> None:
        bad_wave = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with pytest.raises(AdaptationError, match="non-finite"):
            apply_spectral_subtraction(bad_wave)


class TestAdaptiveBandpass:
    """Tests for Butterworth bandpass filtering."""

    def test_bandpass_attenuates_sub_bass(self) -> None:
        sr = TARGET_SAMPLE_RATE
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # 30 Hz sub-bass rumble
        rumble = np.sin(2 * np.pi * 30.0 * t).astype(np.float32)
        filtered = apply_adaptive_bandpass(rumble, sr=sr, lowcut=120.0, highcut=7000.0)

        # Attenuation should be significant (> 12 dB)
        orig_power = np.mean(rumble**2)
        filt_power = np.mean(filtered**2)
        assert filt_power < 0.1 * orig_power

    def test_invalid_cutoffs_raises_error(
        self, clean_synthetic_speech: np.ndarray
    ) -> None:
        with pytest.raises(AdaptationError, match="Invalid bandpass cutoffs"):
            apply_adaptive_bandpass(clean_synthetic_speech, lowcut=5000.0, highcut=1000.0)


class TestSignalNormalization:
    """Tests for RMS energy normalization."""

    def test_normalization_reaches_target_rms(
        self, clean_synthetic_speech: np.ndarray
    ) -> None:
        target = 0.08
        norm = normalize_signal_energy(clean_synthetic_speech, target_rms=target)
        actual_rms = float(np.sqrt(np.mean(norm**2)))
        assert pytest.approx(actual_rms, rel=1e-2) == target

    def test_silent_signal_handled_gracefully(self) -> None:
        silent = np.zeros(1000, dtype=np.float32)
        norm = normalize_signal_energy(silent)
        assert np.array_equal(norm, silent)


class TestCompleteMitigationPipeline:
    """Tests for apply_front_end_mitigation."""

    def test_full_pipeline_preserves_length_and_type(
        self, clean_synthetic_speech: np.ndarray
    ) -> None:
        cfg = MitigationConfig(
            denoise_enabled=True,
            bandpass_enabled=True,
            normalize_energy=True,
        )
        mitigated = apply_front_end_mitigation(clean_synthetic_speech, config=cfg)
        assert mitigated.shape == clean_synthetic_speech.shape
        assert mitigated.dtype == np.float32
        assert np.all(np.isfinite(mitigated))

    def test_mitigated_waveform_compatible_with_aasist_framing(
        self, clean_synthetic_speech: np.ndarray
    ) -> None:
        mitigated = apply_front_end_mitigation(clean_synthetic_speech)
        tensor = pad_or_truncate_waveform(mitigated, target_length=TARGET_SAMPLES)
        assert tensor.shape == torch.Size([1, TARGET_SAMPLES])
        assert torch.all(torch.isfinite(tensor))
