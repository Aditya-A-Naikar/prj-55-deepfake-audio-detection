"""Test suite for live microphone audio ingestion and resilient decoding.

Verifies:
1. Valid 16-bit PCM RIFF WAV decoding in service.
2. WebM/Opus magic byte diagnostic detection (root cause of browser mic capture bug).
3. MP4 container diagnostic detection.
4. End-to-end endpoint response when submitting genuine PCM WAV audio.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prj55.audio import AudioValidationError
from prj55.web.app import app, service


def make_pcm16_riff_wav(
    samples: np.ndarray, sample_rate: int = 16000
) -> bytes:
    """Helper to synthesize genuine 16-bit mono RIFF WAV bytes."""
    int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    data_bytes = int_samples.tobytes()
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(data_bytes)
    riff_chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size
        1,   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + data_bytes


def test_decode_pcm16_riff_wav() -> None:
    """Verifies that genuine 16-bit PCM WAV decodes cleanly in service."""
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_bytes = make_pcm16_riff_wav(tone, sample_rate=16000)

    waveform, meta = service.decode_and_validate_audio(
        wav_bytes, filename="mic_capture_test.wav"
    )
    assert len(waveform) == 16000
    assert meta["original_sample_rate"] == 16000
    assert meta["target_sample_rate"] == 16000
    assert meta["original_channels"] == 1
    assert np.all(np.isfinite(waveform))


def test_webm_container_detection_informative_error() -> None:
    """Verifies that raw WebM bytes (produced by MediaRecorder without transcoding)
    are caught with clear diagnostic feedback rather than crashing unhandled.
    """
    webm_raw_payload = b"\x1a\x45\xdf\xa3" + b"\x00" * 256

    with pytest.raises(AudioValidationError) as exc_info:
        service.decode_and_validate_audio(webm_raw_payload, filename="mic_capture.wav")

    assert "Could not decode audio file 'mic_capture.wav'" in str(exc_info.value)
    assert "WebM/Opus container detected" in str(exc_info.value)


def test_mp4_container_detection_informative_error() -> None:
    """Verifies that MP4/AAC container bytes produce an informative diagnostic error."""
    mp4_raw_payload = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 256

    with pytest.raises(AudioValidationError) as exc_info:
        service.decode_and_validate_audio(mp4_raw_payload, filename="recording.mp4")

    assert "Could not decode audio file 'recording.mp4'" in str(exc_info.value)
    assert "MP4/AAC container detected" in str(exc_info.value)


def test_api_analyze_with_mic_pcm_wav() -> None:
    """Simulates submitting browser-transcoded PCM WAV to /api/analyze."""
    service.load_model()
    t = np.linspace(0, 1.5, 24000, endpoint=False, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 300 * t)
    wav_bytes = make_pcm16_riff_wav(tone, sample_rate=16000)

    with TestClient(app) as client:
        files = {"file": ("mic_capture_1730000000.wav", wav_bytes, "audio/wav")}
        resp = client.post("/api/analyze", files=files, data={"threshold": "0.0"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_metadata"]["filename"] == "mic_capture_1730000000.wav"
        assert "prediction" in data
        assert "primary_score" in data
