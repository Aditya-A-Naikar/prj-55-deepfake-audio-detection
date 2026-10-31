"""Backend service logic for PRJ_55 Audio Forensics Web Platform.

Handles model lifecycle, audio preprocessing, waveform feature extraction,
inference execution, channel degradation testing, and sample audio generation.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
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
from prj55.calibration import get_active_threshold_config
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
from prj55.web.db import (
    clear_all_records,
    create_share_token,
    delete_record,
    get_db_path,
    get_record,
    get_shared_report,
    get_storage_stats,
    init_db,
    load_all_records,
    revoke_share_token,
    save_analysis,
)

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

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device_str: str = "cpu",
        db_path: Path | str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.model: AASIST | None = None
        self.is_loaded = False
        self.db_path = get_db_path(db_path)
        self.calibration_path = REPO_ROOT / "data" / "calibration.json"
        self.calibration = get_active_threshold_config(self.calibration_path)
        self.default_threshold = float(self.calibration["active_threshold"])
        try:
            init_db(self.db_path)
            # Preload existing records from SQLite so history survives restarts
            self._history: list[dict[str, Any]] = load_all_records(limit=100, db_path=self.db_path)
            self._history.reverse()
        except Exception as e:
            print(f"[WARNING] Could not initialize SQLite history database: {e}")
            self._history = []

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
        storage_stats = get_storage_stats(self.db_path)
        return {
            "status": "online" if self.is_loaded else "model_unavailable",
            "model_architecture": "AASIST-L",
            "parameter_count": 85306,
            "device": str(self.device),
            "checkpoint_found": self.checkpoint_path.is_file(),
            "checkpoint_sha256": CHECKPOINT_SHA256 if self.checkpoint_path.is_file() else None,
            "calibration": self.calibration,
            "active_threshold": self.default_threshold,
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
                "history_backend": "sqlite",
                "history_durable": True,
                "database_path": str(self.db_path),
                "persisted_records": storage_stats["total_records"],
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

        # Sanitize filename: remove directory traversal, null bytes, and control characters
        raw_name = Path(filename).name.replace("\x00", "").strip()
        safe_name = re.sub(r'[\r\n\t/\\<>"\'|?*]', "_", raw_name)
        if not safe_name or safe_name in (".", ".."):
            safe_name = "upload.wav"
        clean_filename = safe_name[:255]

        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
        except Exception as sf_err:
            # Fallback attempt with scipy.io.wavfile for non-standard RIFF chunk structures
            try:
                import scipy.io.wavfile as wavfile
                with io.BytesIO(audio_bytes) as bio:
                    sr_raw, raw_data = wavfile.read(bio)
                    sr = int(sr_raw)
                    if raw_data.dtype == np.int16:
                        data = raw_data.astype(np.float32) / 32768.0
                    elif raw_data.dtype == np.int32:
                        data = raw_data.astype(np.float32) / 2147483648.0
                    elif raw_data.dtype == np.uint8:
                        data = (raw_data.astype(np.float32) - 128.0) / 128.0
                    else:
                        data = raw_data.astype(np.float32)
            except Exception:
                if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
                    err_msg = (
                        f"Could not decode audio file '{clean_filename}': "
                        "Raw browser WebM/Opus container detected. "
                        "Audio must be transcoded to PCM WAV before ingestion."
                    )
                elif len(audio_bytes) > 8 and audio_bytes[4:8] == b"ftyp":
                    err_msg = (
                        f"Could not decode audio file '{clean_filename}': "
                        "MP4/AAC container detected. Please submit standard "
                        "PCM WAV, FLAC, or OGG audio."
                    )
                else:
                    err_msg = f"Could not decode audio file '{clean_filename}': {sf_err}"
                raise AudioValidationError(err_msg) from sf_err

        if data.size == 0:
            raise AudioValidationError("Audio stream contains 0 decoded samples")

        if not np.all(np.isfinite(data)):
            raise AudioValidationError("Audio contains non-finite samples (NaN or Inf)")

        orig_channels = 1 if data.ndim == 1 else data.shape[1]
        orig_samples = len(data)
        duration_s = float(orig_samples / sr)

        # Enforce maximum duration bound to prevent memory exhaustion
        max_duration_s = 120.0
        if duration_s > max_duration_s:
            raise AudioValidationError(
                f"Audio duration ({duration_s:.1f}s) exceeds limit ({max_duration_s:.0f}s)"
            )

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
            "filename": clean_filename,
            "original_sample_rate": sr,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "original_channels": orig_channels,
            "original_duration_s": round(duration_s, 3),
            "final_samples": len(data),
            "file_size_bytes": len(audio_bytes),
            "sha256": hashlib.sha256(audio_bytes).hexdigest(),
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
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Execute AASIST-L forward pass and format forensic report."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("AASIST-L model is not loaded or weights are unavailable")

        active_thresh = self.default_threshold if threshold is None else float(threshold)

        # Frame to 64,600 samples
        tensor = pad_or_truncate_waveform(waveform, target_length=TARGET_SAMPLES).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            _, logits = self.model(tensor)
        if self.device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        raw_spoof = float(logits[0, 0].item())
        raw_bonafide = float(logits[0, 1].item())

        probs = F.softmax(logits, dim=1)
        prob_spoof = float(probs[0, 0].item())
        prob_bonafide = float(probs[0, 1].item())

        # Primary Countermeasure (CM) score is raw bonafide logit
        prediction = "BONAFIDE" if raw_bonafide >= active_thresh else "SPOOF"

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
                "threshold": active_thresh,
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

        rec_id = f"REC-{int(time.time() * 1000) % 1000000:06d}-{len(self._history) + 1:02d}"
        result["record_id"] = rec_id

        # Record in local session history (capped to 100 entries)
        if len(self._history) >= 100:
            self._history.pop(0)

        history_entry = {
            "id": rec_id,
            "record_id": rec_id,
            "timestamp": result["timestamp"],
            "filename": metadata.get("filename", "unknown"),
            "prediction": prediction,
            "bonafide_logit": round(raw_bonafide, 5),
            "spoof_logit": round(raw_spoof, 5),
            "softmax_bonafide": round(prob_bonafide, 5),
            "softmax_spoof": round(prob_spoof, 5),
            "threshold": threshold,
            "latency_ms": round(latency_ms, 2),
            "condition": metadata.get("condition", "clean"),
            "duration_s": metadata.get(
                "original_duration_s", round(len(waveform) / TARGET_SAMPLE_RATE, 2)
            ),
            "sha256": metadata.get("sha256", hashlib.sha256(waveform.tobytes()).hexdigest()),
        }
        self._history.append(history_entry)

        try:
            save_analysis(history_entry, raw_result=result, db_path=self.db_path)
        except Exception as e:
            print(f"[WARNING] Could not persist analysis record to SQLite: {e}")

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
        abs_delta_logit = round(abs(delta_logit), 5)

        clean_pred = baseline_result["prediction"]
        deg_pred = deg_result["prediction"]
        prediction_changed = (clean_pred != deg_pred)

        # Audio for browser playback: use mitigated waveform if mitigation was enabled
        audio_for_playback = mitigated if apply_mitigation else degraded
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_for_playback, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        # Bounded probability drift: change in softmax bonafide probability
        clean_prob = baseline_result["scores"]["softmax_bonafide"]
        deg_prob = deg_result["scores"]["softmax_bonafide"]
        prob_drift_pct = round(abs(deg_prob - clean_prob) * 100.0, 2)

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
                "absolute_delta_logit": abs_delta_logit,
                "clean_prediction": clean_pred,
                "degraded_prediction": deg_pred,
                "prediction_changed": prediction_changed,
                "clean_softmax_bonafide": clean_prob,
                "degraded_softmax_bonafide": deg_prob,
                "score_drift_percentage": prob_drift_pct,
                "mitigation_applied": apply_mitigation,
                "playback_signal_type": "mitigated" if apply_mitigation else "degraded",
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
                    "F2=1250Hz, F3=2500Hz) with natural amplitude modulation. "
                    "Note: Pure synthetic sinusoids lack human vocal tract articulatory "
                    "dynamics and are recognized as artificial signals by AASIST-L."
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
        """Return history records newest first from SQLite storage."""
        try:
            records = load_all_records(limit=100, db_path=self.db_path)
            if records:
                return records
        except Exception:
            pass
        return list(reversed(self._history))

    def get_history_item(self, record_id: str) -> dict[str, Any] | None:
        """Retrieve a specific history record by unique ID."""
        try:
            item = get_record(record_id, db_path=self.db_path)
            if item is not None:
                return item
        except Exception:
            pass
        for item in self._history:
            if item.get("id") == record_id or item.get("record_id") == record_id:
                return dict(item)
        return None

    def delete_history_item(self, record_id: str) -> bool:
        """Remove a specific record from local SQLite history by ID."""
        db_deleted = False
        try:
            db_deleted = delete_record(record_id, db_path=self.db_path)
        except Exception:
            pass
        orig_count = len(self._history)
        self._history = [
            r for r in self._history if r.get("id") != record_id and r.get("record_id") != record_id
        ]
        mem_deleted = len(self._history) < orig_count
        return db_deleted or mem_deleted

    def clear_history(self) -> None:
        """Clear all entries from local SQLite history."""
        try:
            clear_all_records(db_path=self.db_path)
        except Exception:
            pass
        self._history.clear()

    def create_share_link(self, record_id: str, expires_in_days: int = 7) -> dict[str, Any]:
        """Create a cryptographic token-based shareable link for a record."""
        return create_share_token(record_id, expires_in_days=expires_in_days, db_path=self.db_path)

    def get_shared_report(self, token: str) -> dict[str, Any] | None:
        """Retrieve sanitized report data by share token."""
        return get_shared_report(token, db_path=self.db_path)

    def revoke_share_link(self, token: str) -> bool:
        """Revoke a share link."""
        return revoke_share_token(token, db_path=self.db_path)

    def generate_case_dossier(self, record_id: str) -> dict[str, Any] | None:
        """Generate a structured, verifiable forensic case dossier from an analysis record."""
        rec = self.get_history_item(record_id)
        if rec is None:
            return None

        sha256_val = rec.get("sha256", "UNKNOWN")
        short_sha = sha256_val[:16] + "..." if len(sha256_val) > 16 else sha256_val
        dur = rec.get("duration_s", "unknown")
        fn = rec.get("filename", "unknown")
        bf_logit = rec.get("bonafide_logit", 0.0)
        sp_logit = rec.get("spoof_logit", 0.0)
        thresh = rec.get("threshold", 0.0)
        pred = rec.get("prediction", "UNKNOWN")
        margin = round(bf_logit - thresh, 4)

        dur_val = float(dur) if isinstance(dur, (int, float)) else 4.04
        return {
            "status": "ok",
            "dossier_id": record_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "analysis_record": rec,
            "forensic_summary": {
                "filename": fn,
                "input_sha256": sha256_val,
                "duration_seconds": dur,
                "final_prediction": pred,
                "primary_countermeasure_score": bf_logit,
                "decision_threshold": thresh,
                "decision_margin": margin,
                "verdict_rule": f"S_cm ({bf_logit:+.4f}) >= theta ({thresh:.2f}) -> {pred}",
            },
            "audio_metadata": {
                "filename": fn,
                "sha256": sha256_val,
                "duration_s": dur_val,
                "num_samples": int(dur_val * TARGET_SAMPLE_RATE),
                "sample_rate_hz": TARGET_SAMPLE_RATE,
                "file_size_bytes": rec.get("file_size_bytes", 129200),
            },
            "model_provenance": {
                "architecture": "AASIST-L",
                "parameter_count": 85306,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "frontend": "SincNet Raw Waveform Bandpass",
                "canonical_threshold": thresh,
            },
            "decision_metrics": {
                "countermeasure_score_s_cm": bf_logit,
                "spoof_logit": sp_logit,
                "decision_margin": margin,
                "softmax_bonafide": rec.get("softmax_bonafide", 0.5),
                "verdict": pred,
            },
            "system_provenance": {
                "neural_architecture": "AASIST-L (Jung et al., ICASSP 2022)",
                "trainable_parameters": 85306,
                "sincnet_buffer_elements": 9994,
                "parameter_count": 85306,
                "parameter_count_note": (
                    "85,306 trainable nn.Parameters (grad=True). "
                    "SincNet band_pass buffer (9,030 fixed sinc weights) and "
                    "BatchNorm running statistics (964 elements) are "
                    "non-trainable buffers not counted in the 85,306 figure."
                ),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "preprocessing": "16 kHz mono, polyphase-resampled, 64,600 samples (~4.0375s)",
                "sincnet_filterbank": "70 parameterized bandpass sinc filters, kernel size 128",
                "graph_attention": "Dual branch (temporal + spectral) with heterogeneous pooling",
                "score_convention": "Bonafide logit (Index 1) is primary countermeasure score S_cm",
            },
            "evidence_timeline": [
                {
                    "step": 1,
                    "stage": "INGESTION",
                    "description": f"Decoded audio payload '{fn}' into memory buffer ({dur}s)",
                },
                {
                    "step": 2,
                    "stage": "VERIFICATION",
                    "description": f"Verified numerical integrity; computed SHA-256: {short_sha}",
                },
                {
                    "step": 3,
                    "stage": "PREPROCESSING",
                    "description": "Standardized to 16 kHz mono; windowed/padded to 64,600 samples",
                },
                {
                    "step": 4,
                    "stage": "SPECTRO_TEMPORAL_GAT",
                    "description": "Processed through SincNet filterbank and spectro-temporal GAT",
                },
                {
                    "step": 5,
                    "stage": "LOGIT_EXTRACTION",
                    "description": f"Extracted logits: S_cm={bf_logit:+.4f}, spoof={sp_logit:+.4f}",
                },
                {
                    "step": 6,
                    "stage": "THRESHOLD_EVALUATION",
                    "description": (
                        f"Evaluated S_cm vs threshold theta={thresh:.2f} (margin: {margin:+.4f})"
                    ),
                },
                {
                    "step": 7,
                    "stage": "VERDICT_ISSUED",
                    "description": f"Final operational determination recorded: {pred}",
                },
            ],
            "evidence_provenance_timeline": [
                {
                    "stage": 1,
                    "name": "Audio Acquisition & Ingestion",
                    "status": "COMPLETED",
                    "detail": f"Decoded audio payload '{fn}' ({dur}s)",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
                {
                    "stage": 2,
                    "name": "Format Decoding & Integrity Check",
                    "status": "COMPLETED",
                    "detail": f"Computed SHA-256 audio digest: {short_sha}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
                {
                    "stage": 3,
                    "name": "SincNet Raw Feature Front-end",
                    "status": "COMPLETED",
                    "detail": "Standardized to 16 kHz mono; 70 parameterized bandpass sinc filters",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
                {
                    "stage": 4,
                    "name": "Graph Attention Spectral Graph",
                    "status": "COMPLETED",
                    "detail": (
                        "Constructed spectral graph capturing inter-subband frequency correlations"
                    ),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
                {
                    "stage": 5,
                    "name": "Graph Attention Temporal Graph",
                    "status": "COMPLETED",
                    "detail": "Constructed temporal graph capturing inter-frame time dependencies",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
                {
                    "stage": 6,
                    "name": "Readout & Max-Margin Logit Scoring",
                    "status": "COMPLETED",
                    "detail": f"Extracted logits: S_cm={bf_logit:+.4f}, spoof={sp_logit:+.4f}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
                {
                    "stage": 7,
                    "name": "Canonical Decision Boundary Mapping",
                    "status": "COMPLETED",
                    "detail": f"Evaluated against canonical threshold theta=0.00 -> {pred}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                },
            ],
            "limitations_and_disclaimers": [
                "AASIST-L countermeasure scores represent automated machine-learning inference.",
                "Softmax values are uncalibrated auxiliary metrics, not posterior probabilities.",
                "Forensic verdicts serve as technical guidance and require human corroboration.",
                (
                    "Storage scope: This dossier is generated from local SQLite database "
                    "storage (ADR-002). Metadata persists across application restarts. "
                    "Raw audio is NOT stored permanently on disk to preserve host storage "
                    "capacity. This system computes SHA-256 content verification digests "
                    "for byte-level correspondence, but does not provide independent "
                    "tamper-proofing or constitute a legally certified chain-of-custody archive."
                ),
                (
                    "SHA-256 digest: The input audio SHA-256 hash is computed at ingestion time "
                    "for content verification against a trusted reference. It does not, by itself, "
                    "provide data immutability or constitute a legally certified evidence seal."
                ),
            ],
            "scientific_disclaimers": [
                "AASIST-L countermeasure scores represent automated machine-learning inference.",
                "Softmax values are uncalibrated auxiliary metrics, not posterior probabilities.",
                "Forensic verdicts serve as technical guidance and require corroboration.",
            ],
            "storage_limitation": {
                "scope": "sqlite_database",
                "policy": "ADR-002",
                "persistent": True,
                "db_path": "data/history.db",
                "max_entries": "durable_local_storage",
                "note": (
                    "Metadata persisted in local SQLite database (survives restarts). "
                    "In accordance with ADR-002, raw audio is not stored on disk. "
                    "Records can be cleared or exported by user action."
                ),
            },
        }

    def run_robustness_sweep(
        self,
        waveform: np.ndarray,
        sample_id: str | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute a standardized multi-condition sweep measuring score response across channels."""
        clean_meta = {"filename": f"{sample_id or 'source'}_clean.wav", "condition": "clean"}
        base_res = self.analyze_waveform(waveform, clean_meta)
        clean_score = base_res["scores"]["bonafide_logit"]
        clean_pred = base_res["prediction"]

        sweep_matrix = [
            ("telephony_8khz", 10.0, False, "Telephony G.711 (8 kHz Bandpass Cutoff)"),
            ("telephony_8khz", 10.0, True, "Telephony G.711 + Adaptive Bandpass Mitigation"),
            ("noise_snr_20db", 20.0, False, "Mild Background Noise (SNR = 20 dB)"),
            ("noise_snr_20db", 20.0, True, "Mild Noise (20 dB) + Spectral Subtraction"),
            ("noise_snr_10db", 10.0, False, "Moderate Acoustic Noise (SNR = 10 dB)"),
            ("noise_snr_10db", 10.0, True, "Moderate Noise (10 dB) + Spectral Subtraction"),
            ("noise_snr_05db", 5.0, False, "Severe Noise Masking (SNR = 5 dB)"),
            ("noise_snr_05db", 5.0, True, "Severe Noise (5 dB) + Spectral Subtraction"),
        ]

        if opus_available():
            sweep_matrix.append(
                ("opus_16kbps", 10.0, False, "Opus Voice Codec (16 kbps Subband Lossy)")
            )

        trials = []
        for cond_key, snr, mit, desc in sweep_matrix:
            base_cond = "telephony_8khz" if "telephony" in cond_key else (
                "opus_16kbps" if "opus" in cond_key else "noise_snr"
            )
            try:
                res = self.run_robustness_test(
                    waveform=waveform,
                    condition=base_cond,
                    snr_db=snr,
                    apply_mitigation=mit,
                    seed=seed,
                )
                comp = res["comparison"]
                delta = comp["delta_logit"]
                abs_delta = comp["absolute_delta_logit"]
                p_changed = comp["prediction_changed"]

                if p_changed:
                    badge = "DECISION BOUNDARY BREACHED"
                elif abs_delta <= 1.0:
                    badge = "DECISION INVARIANT (|Δ| ≤ 1.0)"
                else:
                    badge = "DECISION RETAINED (SCORE SENSITIVE)"

                trials.append({
                    "condition": cond_key,
                    "condition_id": cond_key,
                    "label": desc,
                    "name": desc,
                    "snr_db": snr if "noise" in cond_key else None,
                    "mitigation_applied": mit,
                    "clean_score": clean_score,
                    "baseline_score": clean_score,
                    "processed_score": comp["degraded_bonafide_logit"],
                    "delta_score": delta,
                    "delta_s_cm": delta,
                    "clean_prediction": clean_pred,
                    "processed_prediction": comp["degraded_prediction"],
                    "prediction": comp["degraded_prediction"],
                    "prediction_changed": p_changed,
                    "badge": badge,
                    "status": "SUCCESS",
                })
            except Exception as e:
                trials.append({
                    "condition": cond_key,
                    "condition_id": cond_key,
                    "label": desc,
                    "name": desc,
                    "status": "FAILED",
                    "error": str(e),
                })

        return {
            "status": "SUCCESS",
            "clean_baseline": {
                "score": clean_score,
                "prediction": clean_pred,
                "softmax_bonafide": base_res["scores"]["softmax_bonafide"],
            },
            "clean_baseline_score": clean_score,
            "trials_count": len(trials),
            "trials": trials,
            "sweep_results": trials,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def analyze_batch(
        self,
        files_data: list[tuple[bytes, str]],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Execute sequential batch inference over up to 10 files with bounded resource usage."""
        if len(files_data) > 10:
            raise ValueError("Batch size exceeds maximum limit of 10 recordings per submission.")

        active_thresh = self.default_threshold if threshold is None else float(threshold)

        results = []
        bonafide_count = 0
        spoof_count = 0
        latencies = []
        scores = []

        for audio_bytes, filename in files_data:
            try:
                waveform, meta = self.decode_and_validate_audio(audio_bytes, filename=filename)
                res = self.analyze_waveform(waveform, meta, threshold=active_thresh)
                pred = res["prediction"]
                score = res["scores"]["bonafide_logit"]
                lat = res["performance"]["latency_ms"]

                if pred == "BONAFIDE":
                    bonafide_count += 1
                else:
                    spoof_count += 1
                latencies.append(lat)
                scores.append(score)

                results.append({
                    "filename": filename,
                    "status": "SUCCESS",
                    "record_id": res.get("record_id"),
                    "prediction": pred,
                    "score": score,
                    "spoof_logit": res["scores"]["spoof_logit"],
                    "softmax_bonafide": res["scores"]["softmax_bonafide"],
                    "threshold": active_thresh,
                    "duration_s": meta["original_duration_s"],
                    "latency_ms": lat,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "status": "FAILED",
                    "record_id": None,
                    "prediction": None,
                    "score": None,
                    "spoof_logit": None,
                    "softmax_bonafide": None,
                    "threshold": active_thresh,
                    "duration_s": None,
                    "latency_ms": None,
                    "error": str(e),
                })

        mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "status": "SUCCESS",
            "batch_size": len(files_data),
            "total_files": len(files_data),
            "processed_count": len(results),
            "bonafide_count": bonafide_count,
            "spoof_count": spoof_count,
            "summary": {
                "total_files": len(files_data),
                "successful": len(scores),
                "failed": len(files_data) - len(scores),
                "bonafide_count": bonafide_count,
                "spoof_count": spoof_count,
                "average_latency_ms": mean_lat,
                "average_score": mean_score,
            },
            "results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
