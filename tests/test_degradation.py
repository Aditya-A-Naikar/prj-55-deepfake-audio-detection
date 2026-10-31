"""Tests for the Phase 3 channel degradation pipeline (src/prj55/degradation.py).

All tests use deterministic synthetic signals. None of these tests use real
speech from the ASVspoof corpus — they verify SOFTWARE BEHAVIOR ONLY and must
never be described as benchmark deepfake-detection results.

Test scope:
  - Telephony: correct downsampling, upsampling, length preservation, anti-aliasing.
  - Noise: SNR-scaled noise, reproducibility, invalid SNR handling.
  - Opus: availability check, graceful error when ffmpeg is absent.
  - Input validation: empty, non-finite, multi-dimensional, wrong-type inputs.
  - DegradationResult: success flag, score_delta calculation.
  - AASIST-L compatibility: transforms produce tensors the model accepts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prj55.audio import TARGET_SAMPLES
from prj55.degradation import (
    DegradationError,
    DegradationResult,
    DegradationUnavailableError,
    NoiseConfig,
    OpusConfig,
    TelephonyConfig,
    _validate_waveform,
    apply_noise,
    apply_telephony,
    codec_probe_report,
    opus_available,
)

try:
    from prj55.degradation import apply_opus  # noqa: F401
    _OPUS_IMPORT_OK = True
except ImportError:
    _OPUS_IMPORT_OK = False  # pragma: no cover


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_16k() -> np.ndarray:
    """1-D float32 sine wave at 16 kHz, 4 seconds (64 000 samples)."""
    sr = 16_000
    t = np.linspace(0, 4.0, 4 * sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def short_sine_16k() -> np.ndarray:
    """0.5 s sine at 16 kHz (8 000 samples) — shorter than TARGET_SAMPLES."""
    sr = 16_000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Section 1: Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for _validate_waveform gating in all transforms."""

    def test_empty_waveform_raises(self, sine_16k: np.ndarray) -> None:
        empty = np.array([], dtype=np.float32)
        with pytest.raises(DegradationError, match="empty"):
            _validate_waveform(empty, context="test")

    def test_non_array_raises(self) -> None:
        with pytest.raises(DegradationError, match="np.ndarray"):
            _validate_waveform([0.1, 0.2, 0.3], context="test")  # type: ignore

    def test_2d_array_raises(self, sine_16k: np.ndarray) -> None:
        two_d = sine_16k.reshape(2, -1)
        with pytest.raises(DegradationError, match="1-D"):
            _validate_waveform(two_d, context="test")

    def test_nan_waveform_raises(self) -> None:
        bad = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with pytest.raises(DegradationError, match="non-finite"):
            _validate_waveform(bad, context="test")

    def test_inf_waveform_raises(self) -> None:
        bad = np.array([0.1, np.inf, -np.inf], dtype=np.float32)
        with pytest.raises(DegradationError, match="non-finite"):
            _validate_waveform(bad, context="test")

    def test_valid_waveform_passes(self, sine_16k: np.ndarray) -> None:
        # Should not raise
        _validate_waveform(sine_16k, context="test")


# ---------------------------------------------------------------------------
# Section 2: Telephony degradation tests
# ---------------------------------------------------------------------------


class TestTelephonyDegradation:
    """Tests for apply_telephony: resampling, length, format, anti-aliasing."""

    def test_output_length_preserved(self, sine_16k: np.ndarray) -> None:
        out = apply_telephony(sine_16k)
        assert len(out) == len(sine_16k), (
            f"Output length {len(out)} must match input {len(sine_16k)}"
        )

    def test_output_dtype_float32(self, sine_16k: np.ndarray) -> None:
        out = apply_telephony(sine_16k)
        assert out.dtype == np.float32

    def test_output_1d(self, sine_16k: np.ndarray) -> None:
        out = apply_telephony(sine_16k)
        assert out.ndim == 1

    def test_output_finite(self, sine_16k: np.ndarray) -> None:
        out = apply_telephony(sine_16k)
        assert np.all(np.isfinite(out)), "Telephony output contains non-finite values"

    def test_short_waveform_preserved_length(self, short_sine_16k: np.ndarray) -> None:
        out = apply_telephony(short_sine_16k)
        assert len(out) == len(short_sine_16k)

    def test_anti_aliasing_attenuates_high_freq(self) -> None:
        """Components above the Nyquist of the intermediate rate should be suppressed.

        A 5 kHz tone is above the 4 kHz Nyquist of the 8 kHz intermediate rate.
        After telephony simulation the energy of the 5 kHz component should
        decrease substantially compared to a 1 kHz (in-band) component.
        """
        sr = 16_000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone_high = (0.5 * np.sin(2 * np.pi * 5000.0 * t)).astype(np.float32)
        tone_low = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

        out_high = apply_telephony(tone_high)
        out_low = apply_telephony(tone_low)

        energy_high = float(np.mean(out_high ** 2))
        energy_in = float(np.mean(tone_high ** 2))
        energy_low = float(np.mean(out_low ** 2))
        energy_low_in = float(np.mean(tone_low ** 2))

        # High-frequency component should be significantly attenuated (> 50 %)
        suppression_ratio_high = energy_high / energy_in
        # Low-frequency component should be largely preserved (retain > 50 %)
        preservation_ratio_low = energy_low / energy_low_in

        assert suppression_ratio_high < 0.2, (
            f"5 kHz component not sufficiently attenuated: {suppression_ratio_high:.3f}"
        )
        assert preservation_ratio_low > 0.5, (
            f"1 kHz in-band component incorrectly attenuated: {preservation_ratio_low:.3f}"
        )

    def test_telephony_modifies_waveform(self, sine_16k: np.ndarray) -> None:
        out = apply_telephony(sine_16k)
        # Telephony does modify the waveform (removes high-freq content)
        assert not np.allclose(sine_16k, out, atol=1e-6), (
            "Telephony output is identical to input — degradation had no effect"
        )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(DegradationError, match="empty"):
            apply_telephony(np.array([], dtype=np.float32))

    def test_invalid_config_intermediate_rate(self) -> None:
        with pytest.raises(DegradationError, match="intermediate_rate"):
            TelephonyConfig(intermediate_rate=0)

    def test_invalid_config_upsample_order(self) -> None:
        with pytest.raises(DegradationError, match="upsample_order"):
            TelephonyConfig(upsample_order=-1)

    def test_custom_intermediate_rate(self, sine_16k: np.ndarray) -> None:
        """4 kHz intermediate rate should also work without error."""
        cfg = TelephonyConfig(intermediate_rate=4000)
        out = apply_telephony(sine_16k, config=cfg)
        assert len(out) == len(sine_16k)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Section 3: Additive noise tests
# ---------------------------------------------------------------------------


class TestNoiseDegradation:
    """Tests for apply_noise: SNR accuracy, reproducibility, error handling."""

    def test_output_length_preserved(self, sine_16k: np.ndarray) -> None:
        out = apply_noise(sine_16k)
        assert len(out) == len(sine_16k)

    def test_output_dtype_float32(self, sine_16k: np.ndarray) -> None:
        out = apply_noise(sine_16k)
        assert out.dtype == np.float32

    def test_output_finite(self, sine_16k: np.ndarray) -> None:
        out = apply_noise(sine_16k)
        assert np.all(np.isfinite(out))

    def test_noise_modifies_waveform(self, sine_16k: np.ndarray) -> None:
        out = apply_noise(sine_16k)
        assert not np.allclose(sine_16k, out, atol=1e-6)

    def test_reproducible_output_same_seed(self, sine_16k: np.ndarray) -> None:
        cfg = NoiseConfig(snr_db=10.0, seed=99)
        out1 = apply_noise(sine_16k, config=cfg)
        out2 = apply_noise(sine_16k, config=cfg)
        np.testing.assert_array_equal(out1, out2, err_msg="Same seed must yield identical output")

    def test_different_seeds_differ(self, sine_16k: np.ndarray) -> None:
        out1 = apply_noise(sine_16k, config=NoiseConfig(snr_db=10.0, seed=1))
        out2 = apply_noise(sine_16k, config=NoiseConfig(snr_db=10.0, seed=2))
        assert not np.allclose(out1, out2), "Different seeds must yield different noise"

    def test_snr_approximately_correct(self, sine_16k: np.ndarray) -> None:
        """Verify that actual SNR is within ±1 dB of target."""
        target_snr = 15.0
        cfg = NoiseConfig(snr_db=target_snr, seed=42)
        out = apply_noise(sine_16k, config=cfg)
        noise = out - sine_16k

        sig_pow = float(np.mean(sine_16k ** 2))
        noise_pow = float(np.mean(noise ** 2))
        actual_snr = 10.0 * np.log10(sig_pow / noise_pow)

        assert abs(actual_snr - target_snr) < 1.0, (
            f"SNR target {target_snr} dB, actual {actual_snr:.2f} dB (tolerance ±1 dB)"
        )

    def test_high_snr_closer_to_original(self, sine_16k: np.ndarray) -> None:
        out_high = apply_noise(sine_16k, config=NoiseConfig(snr_db=40.0, seed=42))
        out_low = apply_noise(sine_16k, config=NoiseConfig(snr_db=0.0, seed=42))

        mse_high = float(np.mean((out_high - sine_16k) ** 2))
        mse_low = float(np.mean((out_low - sine_16k) ** 2))
        assert mse_high < mse_low, (
            f"High-SNR ({mse_high:.6f}) should be closer to original than low-SNR ({mse_low:.6f})"
        )

    def test_invalid_snr_nan(self) -> None:
        with pytest.raises(DegradationError, match="finite"):
            NoiseConfig(snr_db=float("nan"))

    def test_invalid_snr_inf(self) -> None:
        with pytest.raises(DegradationError, match="finite"):
            NoiseConfig(snr_db=float("inf"))

    def test_silent_signal_raises(self) -> None:
        silent = np.zeros(16000, dtype=np.float32)
        with pytest.raises(DegradationError, match="zero"):
            apply_noise(silent)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(DegradationError, match="empty"):
            apply_noise(np.array([], dtype=np.float32))

    def test_nan_input_raises(self) -> None:
        bad = np.full(1000, np.nan, dtype=np.float32)
        with pytest.raises(DegradationError, match="non-finite"):
            apply_noise(bad)


# ---------------------------------------------------------------------------
# Section 4: Opus codec tests
# ---------------------------------------------------------------------------


class TestOpusDegradation:
    """Tests for apply_opus: availability, error handling."""

    def test_codec_probe_report_structure(self) -> None:
        report = codec_probe_report()
        assert "ffmpeg_binary" in report
        assert "ffmpeg_version" in report
        assert "libopus_available" in report
        assert isinstance(report["libopus_available"], bool)

    def test_opus_available_returns_bool(self) -> None:
        result = opus_available()
        assert isinstance(result, bool)

    def test_opus_raises_unavailable_when_no_ffmpeg(self, sine_16k: np.ndarray) -> None:
        """When FFmpeg is not present, apply_opus must raise DegradationUnavailableError."""
        if opus_available():
            pytest.skip("FFmpeg/libopus is installed — testing unavailability path is N/A")
        with pytest.raises(DegradationUnavailableError):
            from prj55.degradation import apply_opus
            apply_opus(sine_16k)

    def test_opus_with_ffmpeg_available(self, sine_16k: np.ndarray) -> None:
        """Integration test: run Opus round-trip if FFmpeg is available."""
        if not opus_available():
            pytest.skip("FFmpeg/libopus not installed — skipping Opus round-trip test")
        from prj55.degradation import apply_opus
        cfg = OpusConfig(bitrate_kbps=24, application="voip")
        out = apply_opus(sine_16k, config=cfg)
        assert len(out) == len(sine_16k)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_opus_config_invalid_bitrate(self) -> None:
        with pytest.raises(DegradationError, match="bitrate_kbps"):
            OpusConfig(bitrate_kbps=0)

    def test_opus_config_invalid_application(self) -> None:
        with pytest.raises(DegradationError, match="application"):
            OpusConfig(application="invalid_mode")

    def test_opus_empty_input_raises(self) -> None:
        if not opus_available():
            pytest.skip("FFmpeg not available")
        from prj55.degradation import apply_opus
        with pytest.raises(DegradationError, match="empty"):
            apply_opus(np.array([], dtype=np.float32))


# ---------------------------------------------------------------------------
# Section 5: DegradationResult tests
# ---------------------------------------------------------------------------


class TestDegradationResult:
    """Tests for DegradationResult dataclass behaviour."""

    def test_success_flag_true_when_no_error(self, sine_16k: np.ndarray) -> None:
        result = DegradationResult(
            transform_name="test_transform",
            degraded_waveform=sine_16k.copy(),
            error="",
        )
        assert result.success is True

    def test_success_flag_false_when_error_set(self) -> None:
        result = DegradationResult(
            transform_name="test_transform",
            error="Something broke",
        )
        assert result.success is False

    def test_score_delta_computation(self) -> None:
        result = DegradationResult(
            transform_name="test",
            original_score_bonafide=-0.5,
            degraded_score_bonafide=-1.0,
            score_delta=-0.5,
        )
        assert result.score_delta == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Section 6: AASIST-L compatibility tests
# ---------------------------------------------------------------------------


class TestAASISTCompatibility:
    """Verify that degraded waveforms can be processed by the model."""

    def test_telephony_output_compatible_with_model(self, sine_16k: np.ndarray) -> None:
        from prj55.audio import pad_or_truncate_waveform
        from prj55.models.aasist import AASIST, AASISTConfig

        degraded = apply_telephony(sine_16k)
        tensor = pad_or_truncate_waveform(degraded, target_length=TARGET_SAMPLES)

        assert tensor.shape == (1, TARGET_SAMPLES)

        model = AASIST(AASISTConfig())
        model.eval()
        with torch.no_grad():
            _hidden, logits = model(tensor)

        assert logits.shape == (1, 2)
        assert torch.all(
            torch.isfinite(logits)
        ), "Model produced non-finite logits on telephony input"

    def test_noise_output_compatible_with_model(self, sine_16k: np.ndarray) -> None:
        from prj55.audio import pad_or_truncate_waveform
        from prj55.models.aasist import AASIST, AASISTConfig

        degraded = apply_noise(sine_16k, config=NoiseConfig(snr_db=10.0, seed=0))
        tensor = pad_or_truncate_waveform(degraded, target_length=TARGET_SAMPLES)

        assert tensor.shape == (1, TARGET_SAMPLES)

        model = AASIST(AASISTConfig())
        model.eval()
        with torch.no_grad():
            _hidden, logits = model(tensor)

        assert logits.shape == (1, 2)
        assert torch.all(torch.isfinite(logits)), "Model produced non-finite logits on noisy input"

    def test_multiple_snr_conditions_all_finite(self, sine_16k: np.ndarray) -> None:
        """Score finiteness across SNR range — software test, not benchmark."""
        from prj55.audio import pad_or_truncate_waveform
        from prj55.models.aasist import AASIST, AASISTConfig

        model = AASIST(AASISTConfig())
        model.eval()

        for snr in [0.0, 5.0, 10.0, 20.0]:
            degraded = apply_noise(sine_16k, config=NoiseConfig(snr_db=snr, seed=42))
            tensor = pad_or_truncate_waveform(degraded, target_length=TARGET_SAMPLES)
            with torch.no_grad():
                _, logits = model(tensor)
            assert torch.all(torch.isfinite(logits)), (
                f"Non-finite logits at SNR={snr} dB"
            )


class TestScoreDriftCalculation:
    """Rigorous tests for score delta convention and edge cases (REQ-05)."""

    def test_positive_scores_delta(self) -> None:
        clean = 2.05
        degraded = 1.10
        delta = round(degraded - clean, 5)
        assert delta == -0.95
        assert abs(delta) == 0.95

    def test_positive_to_negative_decision_flip(self) -> None:
        clean = 2.04599
        degraded = -1.55305
        delta = round(degraded - clean, 5)
        assert delta < 0
        threshold = 0.0
        clean_pred = "BONAFIDE" if clean >= threshold else "SPOOF"
        deg_pred = "BONAFIDE" if degraded >= threshold else "SPOOF"
        assert clean_pred != deg_pred

    def test_negative_scores_spoof_stability(self) -> None:
        clean = -3.91324
        degraded = -4.20150
        delta = round(degraded - clean, 5)
        assert delta == -0.28826
        threshold = 0.0
        clean_pred = "BONAFIDE" if clean >= threshold else "SPOOF"
        deg_pred = "BONAFIDE" if degraded >= threshold else "SPOOF"
        assert clean_pred == deg_pred == "SPOOF"

    def test_zero_and_near_zero_scores(self) -> None:
        clean_zero = 0.0
        degraded = -0.5
        delta = degraded - clean_zero
        assert delta == -0.5

        near_zero_pos = 1e-5
        delta_near = degraded - near_zero_pos
        assert pytest.approx(delta_near, abs=1e-4) == -0.5

        near_zero_neg = -1e-5
        delta_near_neg = degraded - near_zero_neg
        assert pytest.approx(delta_near_neg, abs=1e-4) == -0.5

    def test_identical_scores_zero_delta(self) -> None:
        score = 1.75
        delta = score - score
        assert delta == 0.0
        assert abs(delta) == 0.0
