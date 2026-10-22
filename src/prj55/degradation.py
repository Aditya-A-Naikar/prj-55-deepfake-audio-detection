"""Channel degradation pipeline for PRJ_55 robustness analysis.

This module implements controlled audio-signal-processing transformations used
to study how degraded recording conditions affect AASIST-L deepfake detection.

Three degradation families are provided:

1. **Telephony** — Restricted-bandwidth simulation via 8 kHz downsampling and
   8 kHz→16 kHz upsampling with anti-alias filtering (SciPy resample_poly).
   This simulates G.711/PSTN bandwidth restriction (300–3400 Hz) only; it does
   not reproduce every codec artefact of a real telephone channel.

2. **Additive Noise** — Synthetic white Gaussian noise scaled to a configurable
   SNR (dB) relative to the source waveform power.  The noise is synthetic and
   should not be confused with real environmental or babble noise.

3. **Opus Codec** — Encode-decode round-trip via FFmpeg + libopus.  This
   transformation is CONDITIONALLY AVAILABLE: the module probes for a working
   ``ffmpeg`` executable with libopus support at import time.  If not found the
   module remains fully usable for telephony and noise; the Opus functions raise
   ``DegradationUnavailableError`` with an actionable message.

Input / output contract (all public functions):
- Input: 1-D ``np.ndarray`` of float32, sample rate ``16_000 Hz``.
- Output: 1-D ``np.ndarray`` of float32, sample rate ``16_000 Hz``.
- All outputs are validated for finiteness, length, and dtype before returning.

No audio files from the ASVspoof corpus are touched by this module; tests use
only synthetic deterministic signals.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

INPUT_SAMPLE_RATE: int = 16_000
"""Expected input and output sample rate of all degradation transforms (Hz)."""

TELEPHONY_RATE: int = 8_000
"""Intermediate sample rate used for telephony bandwidth simulation (Hz)."""

DEFAULT_SNR_DB: float = 10.0
"""Default signal-to-noise ratio for additive-noise degradation (dB)."""

# ---------------------------------------------------------------------------
# Module-level Opus availability probe
# ---------------------------------------------------------------------------

_FFMPEG_BIN: str | None = shutil.which("ffmpeg")
_FFMPEG_OPUS_AVAILABLE: bool = False
_FFMPEG_VERSION: str = "not found"
_FFMPEG_PROBE_DETAIL: str = ""

if _FFMPEG_BIN is not None:
    try:
        _ver_result = subprocess.run(
            [_FFMPEG_BIN, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _FFMPEG_VERSION = _ver_result.stdout.splitlines()[0] if _ver_result.stdout else "unknown"
        # Check for libopus support in the encoder list
        _enc_result = subprocess.run(
            [_FFMPEG_BIN, "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _FFMPEG_OPUS_AVAILABLE = "libopus" in _enc_result.stdout or "opus" in _enc_result.stdout
        _FFMPEG_PROBE_DETAIL = (
            f"ffmpeg at {_FFMPEG_BIN}; opus={'yes' if _FFMPEG_OPUS_AVAILABLE else 'no'}"
        )
    except Exception as exc:  # pragma: no cover
        _FFMPEG_PROBE_DETAIL = f"ffmpeg probe failed: {exc}"


def opus_available() -> bool:
    """Return True if FFmpeg with libopus is detected on the current system."""
    return _FFMPEG_OPUS_AVAILABLE


def codec_probe_report() -> dict[str, Any]:
    """Return a structured report of Opus/FFmpeg availability for documentation."""
    return {
        "ffmpeg_binary": _FFMPEG_BIN,
        "ffmpeg_version": _FFMPEG_VERSION,
        "libopus_available": _FFMPEG_OPUS_AVAILABLE,
        "detail": _FFMPEG_PROBE_DETAIL,
    }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DegradationError(ValueError):
    """Raised when a degradation transform fails due to invalid inputs."""


class DegradationUnavailableError(RuntimeError):
    """Raised when a degradation requires an external dependency that is absent."""


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TelephonyConfig:
    """Configuration for telephony-style bandwidth-restriction transform.

    Attributes:
        intermediate_rate: Intermediate sample rate (Hz) used to simulate
            bandwidth restriction.  Default 8000 Hz approximates G.711 / PSTN.
        upsample_order: Polyphase filter order passed to
            ``scipy.signal.resample_poly``.  Higher values trade speed for
            sharper band-limiting.
    """

    intermediate_rate: int = TELEPHONY_RATE
    upsample_order: int = 16

    def __post_init__(self) -> None:
        if self.intermediate_rate <= 0 or self.intermediate_rate > INPUT_SAMPLE_RATE:
            raise DegradationError(
                f"intermediate_rate must be in (0, {INPUT_SAMPLE_RATE}], "
                f"got {self.intermediate_rate}"
            )
        if self.upsample_order <= 0:
            raise DegradationError(f"upsample_order must be > 0, got {self.upsample_order}")


@dataclass
class NoiseConfig:
    """Configuration for additive synthetic white-noise transform.

    Attributes:
        snr_db: Target signal-to-noise ratio in decibels.  Must be finite.
        seed: Random seed for reproducibility.  ``None`` uses a non-seeded RNG
            (non-reproducible across runs).
    """

    snr_db: float = DEFAULT_SNR_DB
    seed: int | None = 42

    def __post_init__(self) -> None:
        if not np.isfinite(self.snr_db):
            raise DegradationError(
                f"snr_db must be a finite number, got {self.snr_db!r}"
            )


@dataclass
class OpusConfig:
    """Configuration for Opus encode-decode round-trip codec transform.

    Attributes:
        bitrate_kbps: Target bitrate in kilobits per second (e.g. 16, 24, 32).
        application: Opus application profile: ``"voip"`` or ``"audio"``.
        ffmpeg_bin: Override FFmpeg executable path.  ``None`` uses the
            system-detected binary.
    """

    bitrate_kbps: int = 24
    application: str = "voip"
    ffmpeg_bin: str | None = None

    def __post_init__(self) -> None:
        if self.bitrate_kbps <= 0:
            raise DegradationError(f"bitrate_kbps must be > 0, got {self.bitrate_kbps}")
        if self.application not in ("voip", "audio", "lowdelay"):
            raise DegradationError(
                f"application must be 'voip', 'audio', or 'lowdelay', "
                f"got '{self.application}'"
            )


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_waveform(
    waveform: np.ndarray,
    context: str = "input",
    expected_sr: int | None = INPUT_SAMPLE_RATE,
) -> None:
    """Raise ``DegradationError`` if *waveform* violates basic audio contracts.

    Checks performed:
    - ndarray with ``dtype`` coercible to float32
    - Exactly 1-D
    - Non-empty (> 0 samples)
    - All values finite (no NaN or Inf)
    """
    if not isinstance(waveform, np.ndarray):
        raise DegradationError(f"Waveform ({context}) must be np.ndarray, got {type(waveform)}")
    if waveform.ndim != 1:
        raise DegradationError(
            f"Waveform ({context}) must be 1-D, got shape {waveform.shape}"
        )
    if waveform.size == 0:
        raise DegradationError(f"Waveform ({context}) is empty (0 samples)")
    if not np.all(np.isfinite(waveform)):
        n_bad = int(np.sum(~np.isfinite(waveform)))
        raise DegradationError(
            f"Waveform ({context}) contains {n_bad} non-finite sample(s) (NaN/Inf)"
        )


# ---------------------------------------------------------------------------
# Transform 1: Telephony bandwidth restriction
# ---------------------------------------------------------------------------


def apply_telephony(
    waveform: np.ndarray,
    config: TelephonyConfig | None = None,
) -> np.ndarray:
    """Simulate restricted-bandwidth telephony-channel processing.

    The transform:
    1. Validates the 16 kHz input waveform.
    2. Downsamples from 16 kHz → ``config.intermediate_rate`` (default 8 kHz)
       using a polyphase anti-alias filter (``scipy.signal.resample_poly``).
    3. Upsamples back to 16 kHz with the same filter order.
    4. Validates and returns the output waveform.

    **Important limitations**: This approximates the frequency-band truncation
    of a G.711/PSTN channel (effective bandwidth ≈ 0–4 kHz).  It does not
    reproduce µ-law/A-law quantisation noise, packet-loss artefacts, or
    network jitter.

    Args:
        waveform: 1-D float32 array at 16 000 Hz.
        config: Telephony configuration.  Defaults to ``TelephonyConfig()``.

    Returns:
        1-D float32 array at 16 000 Hz with restricted bandwidth.

    Raises:
        DegradationError: On invalid input or non-finite output.
    """
    cfg = config or TelephonyConfig()
    _validate_waveform(waveform, context="telephony input")

    wf = waveform.astype(np.float32)
    rate = INPUT_SAMPLE_RATE  # 16_000

    # Compute coprime up/down factors
    from math import gcd as _gcd
    common = _gcd(cfg.intermediate_rate, rate)
    down_factor = rate // common          # e.g. 2 for 8000 Hz
    up_factor = cfg.intermediate_rate // common  # e.g. 1 for 8000 Hz

    # Downsample 16k → intermediate (anti-aliased)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        downsampled = resample_poly(wf, up_factor, down_factor)

    # Upsample intermediate → 16k (band-limited reconstruction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored = resample_poly(downsampled, down_factor, up_factor)

    # Trim or zero-pad to original length (polyphase filter may add edge samples)
    n_orig = len(wf)
    if len(restored) > n_orig:
        restored = restored[:n_orig]
    elif len(restored) < n_orig:
        restored = np.pad(restored, (0, n_orig - len(restored)), mode="constant")

    result = restored.astype(np.float32)
    _validate_waveform(result, context="telephony output")
    return result


# ---------------------------------------------------------------------------
# Transform 2: Additive synthetic Gaussian noise
# ---------------------------------------------------------------------------


def apply_noise(
    waveform: np.ndarray,
    config: NoiseConfig | None = None,
) -> np.ndarray:
    """Add synthetic white Gaussian noise at a controlled SNR.

    Noise amplitude is scaled such that:

        SNR (dB) = 10 * log10(P_signal / P_noise)

    where P is mean-squared power of the signal / noise waveforms.

    **Important**: The noise is computationally generated white Gaussian noise.
    It does not represent recorded babble, street noise, or any environmental
    recording.  Refer to the MUSAN corpus for real-world noise stimuli.

    Args:
        waveform: 1-D float32 array at 16 000 Hz.
        config: Noise configuration.  Defaults to ``NoiseConfig()``.

    Returns:
        1-D float32 array (signal + noise) at the same length and sample rate.

    Raises:
        DegradationError: On invalid input, near-silent signal (power < 1e-10),
            or non-finite output.
    """
    cfg = config or NoiseConfig()
    _validate_waveform(waveform, context="noise input")

    wf = waveform.astype(np.float32)
    signal_power = float(np.mean(wf ** 2))

    if signal_power < 1e-10:
        raise DegradationError(
            f"Signal power is effectively zero ({signal_power:.2e}); "
            "cannot compute SNR-scaled noise for a silent signal."
        )

    rng = np.random.default_rng(cfg.seed)
    noise = rng.standard_normal(len(wf)).astype(np.float32)

    # Scale noise to achieve target SNR
    # P_noise = P_signal / 10^(SNR_dB / 10)
    noise_power_target = signal_power / (10.0 ** (cfg.snr_db / 10.0))
    noise_power_actual = float(np.mean(noise ** 2))
    if noise_power_actual < 1e-30:  # degenerate RNG outcome — extremely unlikely
        raise DegradationError("Generated noise waveform has effectively zero power.")

    scale = float(np.sqrt(noise_power_target / noise_power_actual))
    result = (wf + scale * noise).astype(np.float32)

    _validate_waveform(result, context="noise output")
    return result


# ---------------------------------------------------------------------------
# Transform 3: Opus encode-decode round-trip (conditional on FFmpeg)
# ---------------------------------------------------------------------------


def apply_opus(
    waveform: np.ndarray,
    config: OpusConfig | None = None,
) -> np.ndarray:
    """Encode waveform to Opus and immediately decode back (lossy round-trip).

    Requires: ``ffmpeg`` with ``libopus`` support installed and accessible in
    ``$PATH``.  Use :func:`opus_available` to check at runtime.

    The pipeline:
    1. Write input as a 16-bit PCM WAV to a temporary file.
    2. Run ``ffmpeg -i input.wav -c:a libopus -b:a {bitrate}k output.opus``.
    3. Decode ``output.opus`` back to WAV with ffmpeg.
    4. Read the decoded WAV via soundfile, validate, and return as float32.

    Codec parameters are recorded in the returned metadata via
    :func:`opus_transform_with_metadata`.

    Args:
        waveform: 1-D float32 array at 16 000 Hz.
        config: Opus configuration.  Defaults to ``OpusConfig()``.

    Returns:
        1-D float32 array at 16 000 Hz after encode-decode round-trip.

    Raises:
        DegradationUnavailableError: If FFmpeg/libopus is not found.
        DegradationError: On invalid input, codec failure, or non-finite output.
    """
    cfg = config or OpusConfig()
    ffmpeg = cfg.ffmpeg_bin or _FFMPEG_BIN

    if not ffmpeg or not _FFMPEG_OPUS_AVAILABLE:
        raise DegradationUnavailableError(
            "Opus degradation requires FFmpeg with libopus support. "
            f"Current status: {_FFMPEG_PROBE_DETAIL}. "
            "Install ffmpeg (e.g. 'brew install ffmpeg') to enable this transform."
        )

    _validate_waveform(waveform, context="opus input")

    with tempfile.TemporaryDirectory(prefix="prj55_opus_") as tmpdir:
        tmp = Path(tmpdir)
        wav_in = tmp / "input.wav"
        opus_out = tmp / "encoded.opus"
        wav_out = tmp / "decoded.wav"

        # Write input WAV
        sf.write(str(wav_in), waveform.astype(np.float32), INPUT_SAMPLE_RATE, subtype="PCM_16")

        # Encode to Opus
        encode_cmd = [
            ffmpeg, "-y",
            "-i", str(wav_in),
            "-c:a", "libopus",
            "-b:a", f"{cfg.bitrate_kbps}k",
            "-application", cfg.application,
            "-vbr", "off",
            "-ar", str(INPUT_SAMPLE_RATE),
            "-ac", "1",
            str(opus_out),
        ]
        enc_result = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=60)
        if enc_result.returncode != 0 or not opus_out.is_file():
            raise DegradationError(
                f"Opus encoding failed (returncode={enc_result.returncode}): "
                f"{enc_result.stderr[-500:]}"
            )

        # Decode back to WAV
        decode_cmd = [
            ffmpeg, "-y",
            "-i", str(opus_out),
            "-ar", str(INPUT_SAMPLE_RATE),
            "-ac", "1",
            str(wav_out),
        ]
        dec_result = subprocess.run(decode_cmd, capture_output=True, text=True, timeout=60)
        if dec_result.returncode != 0 or not wav_out.is_file():
            raise DegradationError(
                f"Opus decoding failed (returncode={dec_result.returncode}): "
                f"{dec_result.stderr[-500:]}"
            )

        decoded, sr = sf.read(str(wav_out), dtype="float32")

    # Soundfile may return stereo on some ffmpeg builds; downmix if needed
    if decoded.ndim == 2:
        decoded = np.mean(decoded, axis=1).astype(np.float32)

    # Align length to original
    n_orig = len(waveform)
    if len(decoded) > n_orig:
        decoded = decoded[:n_orig]
    elif len(decoded) < n_orig:
        decoded = np.pad(decoded, (0, n_orig - len(decoded)), mode="constant")

    result = decoded.astype(np.float32)
    _validate_waveform(result, context="opus output")
    return result


# ---------------------------------------------------------------------------
# Combined transform runner
# ---------------------------------------------------------------------------


@dataclass
class DegradationResult:
    """Structured output of a single degradation + inference pass.

    Attributes:
        transform_name: Human-readable name of the applied transform.
        transform_params: Dict of configuration parameters applied.
        original_waveform: Source waveform before transform (may be ``None``
            if caller did not preserve it to save memory).
        degraded_waveform: Waveform after transform.
        original_score_bonafide: Raw bonafide logit from AASIST-L on the
            original waveform, or ``None`` if inference was not run.
        degraded_score_bonafide: Raw bonafide logit after degradation, or
            ``None`` if inference was not run.
        score_delta: ``degraded - original`` bonafide logit, or ``None``.
        inference_latency_ms: Inference wall-time for the degraded input, or
            ``None`` if not measured.
        error: Non-empty string if the transform or inference step failed.
        metadata: Additional free-form audit fields.
    """

    transform_name: str
    transform_params: dict[str, Any] = field(default_factory=dict)
    original_waveform: np.ndarray | None = None
    degraded_waveform: np.ndarray | None = None
    original_score_bonafide: float | None = None
    degraded_score_bonafide: float | None = None
    score_delta: float | None = None
    inference_latency_ms: float | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if the transform completed without error."""
        return self.error == "" and self.degraded_waveform is not None
