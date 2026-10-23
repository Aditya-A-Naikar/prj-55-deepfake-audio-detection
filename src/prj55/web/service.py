"""Backend service logic for PRJ_55 Audio Forensics Web Platform.

Handles model lifecycle, audio preprocessing, waveform feature extraction,
inference execution, channel degradation testing, and sample audio generation.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import torch.nn.functional as F

from prj55.adaptation import apply_front_end_mitigation
from prj55.audio import (
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    AudioValidationError,
    pad_or_truncate_waveform,
)
from prj55.degradation import (
    NoiseConfig,
    OpusConfig,
    TelephonyConfig,
    apply_noise,
    apply_opus,
    apply_telephony,
    codec_probe_report,
    opus_available,
)
from prj55.models.aasist import AASIST, load_aasist_l

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "weights" / "AASIST-L.pth"
CHECKPOINT_SHA256 = "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a"


@dataclass
class AudioFeatures:
    """Waveform downsampled peaks and frequency spectrum for UI visualization."""

    peaks: list[float]
    spectrum_frequencies: list[float]
    spectrum_magnitudes: list[float]


class ForensicsService:
    """Manages AASIST-L model instance and provides analysis services."""

    def __init__(self, checkpoint_path: Path | None = None, device_str: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.model: AASIST | None = None
        self.is_loaded = False
        self._history: list[dict[str, Any]] = []

    def load_model(self) -> None:
        """Load AASIST-L weights and execute a warmup forward pass."""
        if not self.checkpoint_path.is_file():
            self.is_loaded = False
            return

        self.model = load_aasist_l(self.checkpoint_path, device=self.device)
        self.model.eval()

        # Warmup pass
        dummy = torch.zeros(1, TARGET_SAMPLES, device=self.device)
        with torch.no_grad():
            self.model(dummy)

        self.is_loaded = True

    def get_system_status(self) -> dict[str, Any]:
        """Return system and model readiness status."""
        probe = codec_probe_report()
        return {
            "status": "online" if self.is_loaded else "model_unavailable",
            "model_architecture": "AASIST-L",
            "parameter_count": 85306,
            "device": str(self.device),
            "checkpoint_found": self.checkpoint_path.is_file(),
            "checkpoint_sha256": CHECKPOINT_SHA256 if self.checkpoint_path.is_file() else None,
            "preprocessing": {
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "target_samples": TARGET_SAMPLES,
                "duration_s": round(TARGET_SAMPLES / TARGET_SAMPLE_RATE, 4),
            },
            "codecs": {
                "ffmpeg_available": probe.get("ffmpeg_binary") is not None,
                "opus_available": bool(probe.get("libopus_available", False)),
                "ffmpeg_version": probe.get("ffmpeg_version"),
            },
            "storage_guard": {
                "policy": "ADR-002",
                "external_ssd_required": True,
                "status": "BLOCKED_PENDING_EXTERNAL_SSD",
                "description": (
                    "Physical ASVspoof 2019 LA download safely paused pending external SSD."
                ),
            },
        }

    def decode_and_validate_audio(
        self,
        audio_bytes: bytes,
        filename: str = "upload.wav",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Decode audio bytes to 16 kHz mono float32 array with metadata."""
        if len(audio_bytes) == 0:
            raise AudioValidationError("Audio file payload is empty (0 bytes)")
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise AudioValidationError("Audio file exceeds maximum size limit (25 MB)")

        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
        except Exception as e:
            raise AudioValidationError(f"Could not decode audio file '{filename}': {e}") from e

        if data.size == 0:
            raise AudioValidationError("Audio stream contains 0 decoded samples")

        if not np.all(np.isfinite(data)):
            raise AudioValidationError("Audio contains non-finite samples (NaN or Inf)")

        orig_channels = 1 if data.ndim == 1 else data.shape[1]
        orig_samples = len(data)
        duration_s = float(orig_samples / sr)

        # Downmix multi-channel to mono
        if data.ndim == 2:
            data = np.mean(data, axis=1)

        # Resample to TARGET_SAMPLE_RATE if different
        if sr != TARGET_SAMPLE_RATE:
            gcd = math.gcd(sr, TARGET_SAMPLE_RATE)
            up = TARGET_SAMPLE_RATE // gcd
            down = sr // gcd
            data = signal.resample_poly(data, up, down).astype(np.float32)

        metadata = {
            "filename": filename,
            "original_sample_rate": sr,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "original_channels": orig_channels,
            "original_duration_s": round(duration_s, 3),
            "final_samples": len(data),
        }
        return data, metadata

    def extract_visual_features(self, waveform: np.ndarray, num_peaks: int = 160) -> AudioFeatures:
        """Extract downsampled waveform peaks and FFT frequency spectrum profile."""
        # 1. Downsampled peaks for fast Canvas waveform rendering
        step = max(1, len(waveform) // num_peaks)
        peaks: list[float] = []
        for i in range(0, len(waveform), step):
            chunk = waveform[i : i + step]
            if len(chunk) > 0:
                peaks.append(float(np.max(np.abs(chunk))))
            if len(peaks) >= num_peaks:
                break

        # 2. FFT Power Spectrum
        n_fft = min(2048, len(waveform))
        if n_fft > 64:
            fft_vals = np.abs(np.fft.rfft(waveform[:n_fft] * np.hanning(n_fft)))
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / TARGET_SAMPLE_RATE)
            # Downsample to 64 frequency bins for clean UI charts
            bin_size = max(1, len(freqs) // 64)
            spectrum_freqs = [float(freqs[j]) for j in range(0, len(freqs), bin_size)[:64]]
            spectrum_mags = [float(fft_vals[j]) for j in range(0, len(fft_vals), bin_size)[:64]]
        else:
            spectrum_freqs = []
            spectrum_mags = []

        return AudioFeatures(
            peaks=peaks,
            spectrum_frequencies=spectrum_freqs,
            spectrum_magnitudes=spectrum_mags,
        )

    def analyze_waveform(
        self,
        waveform: np.ndarray,
        metadata: dict[str, Any],
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Execute AASIST-L forward pass and format forensic report."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("AASIST-L model is not loaded or weights are unavailable")

        # Frame to 64,600 samples
        tensor = pad_or_truncate_waveform(waveform, target_length=TARGET_SAMPLES).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            _, logits = self.model(tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        raw_spoof = float(logits[0, 0].item())
        raw_bonafide = float(logits[0, 1].item())

        probs = F.softmax(logits, dim=1)
        prob_spoof = float(probs[0, 0].item())
        prob_bonafide = float(probs[0, 1].item())

        # Primary Countermeasure (CM) score is raw bonafide logit
        prediction = "BONAFIDE" if raw_bonafide >= threshold else "SPOOF"

        # Real-time factor (RTF) = processing time / audio duration (4.0375s)
        audio_dur = TARGET_SAMPLES / TARGET_SAMPLE_RATE
        rtf = (latency_ms / 1000.0) / audio_dur

        features = self.extract_visual_features(waveform)

        result = {
            "prediction": prediction,
            "primary_score": round(raw_bonafide, 5),
            "scores": {
                "bonafide_logit": round(raw_bonafide, 5),
                "spoof_logit": round(raw_spoof, 5),
                "softmax_bonafide": round(prob_bonafide, 5),
                "softmax_spoof": round(prob_spoof, 5),
                "threshold": threshold,
            },
            "performance": {
                "latency_ms": round(latency_ms, 2),
                "real_time_factor": round(rtf, 4),
                "device": str(self.device),
            },
            "audio_metadata": metadata,
            "visual_features": {
                "peaks": features.peaks,
                "spectrum_frequencies": features.spectrum_frequencies,
                "spectrum_magnitudes": features.spectrum_magnitudes,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": {
                "model": "AASIST-L",
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "score_convention": "Bonafide logit >= threshold indicates authentic speech.",
            },
        }

        # Record in local session history
        self._history.append({
            "timestamp": result["timestamp"],
            "filename": metadata.get("filename", "unknown"),
            "prediction": prediction,
            "bonafide_logit": round(raw_bonafide, 5),
            "latency_ms": round(latency_ms, 2),
            "condition": metadata.get("condition", "clean"),
        })

        return result

    def run_robustness_test(
        self,
        waveform: np.ndarray,
        condition: str,
        snr_db: float = 10.0,
        apply_mitigation: bool = False,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Apply channel degradation, optional front-end mitigation, and measure score drift."""
        # Baseline analysis on clean signal
        clean_meta = {"filename": "original_audio.wav", "condition": "clean"}
        baseline_result = self.analyze_waveform(waveform, clean_meta)
        clean_logit = baseline_result["scores"]["bonafide_logit"]

        # Apply degradation
        degraded = waveform.copy()
        if condition == "telephony_8khz":
            degraded = apply_telephony(degraded, TelephonyConfig())
        elif condition.startswith("noise"):
            degraded = apply_noise(degraded, NoiseConfig(snr_db=snr_db, seed=seed))
        elif condition.startswith("opus"):
            if not opus_available():
                return {
                    "status": "UNAVAILABLE",
                    "condition": condition,
                    "error": (
                        "Opus encoder requires system FFmpeg. Install with 'brew install ffmpeg'."
                    ),
                    "baseline_score": clean_logit,
                }
            degraded = apply_opus(degraded, OpusConfig(bitrate_kbps=16))
        else:
            raise ValueError(f"Unknown degradation condition: {condition}")

        # Optional front-end mitigation (Plan 4.1)
        mitigated = degraded.copy()
        if apply_mitigation:
            mitigated = apply_front_end_mitigation(mitigated)

        deg_meta = {
            "filename": f"degraded_{condition}.wav",
            "condition": condition,
            "mitigation_applied": apply_mitigation,
        }
        deg_result = self.analyze_waveform(mitigated, deg_meta)
        degraded_logit = deg_result["scores"]["bonafide_logit"]
        delta_logit = round(degraded_logit - clean_logit, 5)

        # Convert degraded waveform to WAV bytes for in-browser playback
        wav_io = io.BytesIO()
        sf.write(wav_io, degraded, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        return {
            "status": "SUCCESS",
            "condition": condition,
            "parameters": {
                "snr_db": snr_db if condition.startswith("noise") else None,
                "mitigation_applied": apply_mitigation,
                "seed": seed,
            },
            "comparison": {
                "clean_bonafide_logit": clean_logit,
                "degraded_bonafide_logit": degraded_logit,
                "delta_logit": delta_logit,
                "score_drift_percentage": round(
                    abs(delta_logit / (abs(clean_logit) + 1e-6)) * 100.0, 2
                ),
                "clean_prediction": baseline_result["prediction"],
                "degraded_prediction": deg_result["prediction"],
            },
            "latency_ms": deg_result["performance"]["latency_ms"],
            "degraded_visual_features": deg_result["visual_features"],
            "audio_wav_base64": wav_bytes.hex(),  # efficient hex string
        }

    def get_sample_library(self) -> list[dict[str, Any]]:
        """Return metadata for demo samples including real speech and synthetic fixtures."""
        return [
            {
                "id": "speech_human_male",
                "name": "Authentic Spoken Voice (Male — VCTK/ASVspoof)",
                "type": "bonafide_speech",
                "description": (
                    "Genuine human speaker (p226) reading a natural English sentence from "
                    "the official ASVspoof 2019 / VCTK benchmark dataset."
                ),
                "duration_s": 4.74,
            },
            {
                "id": "speech_human_female",
                "name": "Authentic Spoken Voice (Female — VCTK/ASVspoof)",
                "type": "bonafide_speech",
                "description": (
                    "Genuine human speaker (p281) reading a natural English sentence from "
                    "the official ASVspoof 2019 / VCTK benchmark dataset."
                ),
                "duration_s": 2.48,
            },
            {
                "id": "speech_deepfake_tts",
                "name": "Deepfake Spoken AI Voice (Neural TTS — SS_6)",
                "type": "spoof_speech",
                "description": (
                    "High-fidelity neural text-to-speech deepfake (neural vocoder + acoustic "
                    "model) from the ASVspoof 2019 speech synthesis attack protocol."
                ),
                "duration_s": 2.67,
            },
            {
                "id": "speech_deepfake_vocoder",
                "name": "Deepfake Spoken AI Voice (Vocoder Artifact — SS_3)",
                "type": "spoof_speech",
                "description": (
                    "Synthesized deepfake speech containing neural vocoder frame stitching "
                    "and spectral phase anomalies from ASVspoof 2019."
                ),
                "duration_s": 2.50,
            },
            {
                "id": "sample_bonafide",
                "name": "Synthetic Vocal Harmonic Formant Tone",
                "type": "bonafide_simulation",
                "description": (
                    "Multi-formant harmonic speech simulation (F0=125Hz, F1=650Hz, "
                    "F2=1250Hz, F3=2500Hz) with natural amplitude modulation."
                ),
                "duration_s": 4.0,
            },
            {
                "id": "sample_spoof",
                "name": "Synthetic Deepfake Vocal Artifact Tone",
                "type": "spoof_simulation",
                "description": (
                    "Phase-discontinuous frequency modulated tone simulating vocoder neural "
                    "synthesis artifacts and robotic harmonics."
                ),
                "duration_s": 4.0,
            },
            {
                "id": "sample_reference_sine",
                "name": "Reference Calibration Tone (440 Hz)",
                "type": "reference",
                "description": (
                    "Standard 440 Hz sinusoidal calibration tone for acoustic plumbing "
                    "verification."
                ),
                "duration_s": 4.0,
            },
        ]

    def get_sample_audio(self, sample_id: str) -> np.ndarray:
        """Generate or retrieve deterministic audio array for a sample fixture."""
        speech_map = {
            "speech_human_male": "bonafide_human_male.wav",
            "speech_human_female": "bonafide_human_female.wav",
            "speech_deepfake_tts": "deepfake_neural_tts.wav",
            "speech_deepfake_vocoder": "deepfake_vocoder_ss3.wav",
        }
        if sample_id in speech_map:
            path = REPO_ROOT / "data" / "demo_samples" / "speech" / speech_map[sample_id]
            if path.exists():
                wav_data, _ = sf.read(str(path), dtype="float32")
                if wav_data.ndim > 1:
                    wav_data = np.mean(wav_data, axis=1)
                return wav_data.astype(np.float32)

        sr = TARGET_SAMPLE_RATE
        dur = 4.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)

        if sample_id == "sample_bonafide":
            # Authentic speech acoustic model: glottal excitation + vocal tract formants
            f0 = 125.0
            glottal = (
                0.4 * np.sin(2 * np.pi * f0 * t)
                + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
                + 0.1 * np.sin(2 * np.pi * 4 * f0 * t)
            )
            envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))  # natural syllable modulation
            formant1 = 0.2 * np.sin(2 * np.pi * 650.0 * t)
            formant2 = 0.15 * np.sin(2 * np.pi * 1250.0 * t)
            wave = (glottal * envelope + formant1 + formant2).astype(np.float32)
        elif sample_id == "sample_spoof":
            # Spoof artifact model: rapid phase modulations and high-frequency spectral buzz
            f_carrier = 320.0
            buzz = np.sin(2 * np.pi * f_carrier * t + 2.5 * np.sin(2 * np.pi * 45.0 * t))
            glitch = 0.3 * np.sign(np.sin(2 * np.pi * 1800.0 * t))  # robotic square-wave buzz
            phase_jitter = 0.15 * np.sin(2 * np.pi * 3600.0 * t**1.05)
            wave = (0.5 * buzz + 0.25 * glitch + phase_jitter).astype(np.float32)
        elif sample_id == "sample_reference_sine":
            wave = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        else:
            raise KeyError(f"Unknown sample ID: {sample_id}")

        # Peak normalize to 0.75
        peak = float(np.max(np.abs(wave)))
        if peak > 0:
            wave = (wave / peak * 0.75).astype(np.float32)
        return wave

    def get_history(self) -> list[dict[str, Any]]:
        """Return copy of local session history."""
        return list(self._history)
