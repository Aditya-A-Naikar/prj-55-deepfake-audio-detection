"""Automated tests for PRJ_55 Audio Forensics Web API (Plan 4.3 / 4.4)."""

from __future__ import annotations

import io
from collections.abc import Generator

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from prj55.audio import TARGET_SAMPLE_RATE
from prj55.web.app import app, service


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """TestClient fixture with service model loaded."""
    service.load_model()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_wav_bytes() -> bytes:
    """Generate 1-second clean synthetic WAV bytes."""
    sr = TARGET_SAMPLE_RATE
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    wave = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, wave, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


class TestSystemStatusEndpoints:
    """Tests for system status and readiness."""

    def test_status_endpoint_returns_online(self, client: TestClient) -> None:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_architecture"] == "AASIST-L"
        assert data["parameter_count"] == 85306
        assert data["storage_guard"]["policy"] == "ADR-002"
        assert data["status"] in ("online", "model_unavailable")

    def test_samples_endpoint_lists_authorized_fixtures(self, client: TestClient) -> None:
        resp = client.get("/api/samples")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) >= 3
        sample_ids = [s["id"] for s in samples]
        assert "sample_bonafide" in sample_ids
        assert "sample_spoof" in sample_ids

    def test_sample_audio_streaming(self, client: TestClient) -> None:
        resp = client.get("/api/samples/sample_bonafide/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 1000

    def test_nonexistent_sample_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/samples/nonexistent_xyz/audio")
        assert resp.status_code == 404


class TestAnalysisEndpoints:
    """Tests for AASIST-L inference endpoints."""

    def test_analyze_with_preset_sample_id(self, client: TestClient) -> None:
        resp = client.post(
            "/api/analyze", data={"sample_id": "sample_bonafide", "threshold": "0.0"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("BONAFIDE", "SPOOF")
        assert "bonafide_logit" in data["scores"]
        assert "spoof_logit" in data["scores"]
        assert data["performance"]["latency_ms"] > 0
        assert len(data["visual_features"]["peaks"]) > 0

    def test_analyze_with_json_sample_payload(self, client: TestClient) -> None:
        payload = {"sample_id": "sample_spoof", "threshold": 0.0}
        resp = client.post("/api/analyze/sample", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_score" in data
        assert data["prediction"] in ("BONAFIDE", "SPOOF")

    def test_analyze_with_uploaded_file(
        self, client: TestClient, dummy_wav_bytes: bytes
    ) -> None:
        files = {"file": ("test_tone.wav", dummy_wav_bytes, "audio/wav")}
        resp = client.post("/api/analyze", files=files, data={"threshold": "0.0"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_metadata"]["filename"] == "test_tone.wav"
        assert data["audio_metadata"]["target_sample_rate"] == 16000

    def test_analyze_with_empty_payload_raises_400(self, client: TestClient) -> None:
        resp = client.post("/api/analyze", data={})
        assert resp.status_code == 400

    def test_analyze_with_corrupt_file_raises_422(self, client: TestClient) -> None:
        files = {"file": ("broken.wav", b"NOT_A_VALID_WAV_HEADER", "audio/wav")}
        resp = client.post("/api/analyze", files=files)
        assert resp.status_code == 422
        assert "Could not decode" in resp.json()["detail"]


class TestRobustnessEndpoints:
    """Tests for channel degradation simulation endpoints."""

    def test_robustness_telephony_condition(self, client: TestClient) -> None:
        resp = client.post(
            "/api/robustness",
            data={"sample_id": "sample_bonafide", "condition": "telephony_8khz"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "delta_logit" in data["comparison"]
        assert len(data["audio_wav_base64"]) > 0

    def test_robustness_noise_with_mitigation(self, client: TestClient) -> None:
        resp = client.post(
            "/api/robustness",
            data={
                "sample_id": "sample_bonafide",
                "condition": "noise_snr_10db",
                "snr_db": "10.0",
                "mitigation_enabled": "true",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["parameters"]["mitigation_applied"] is True


class TestEvaluationAndHistoryEndpoints:
    """Tests for research benchmarks and session history."""

    def test_evaluation_benchmark_data(self, client: TestClient) -> None:
        resp = client.get("/api/evaluation")
        assert resp.status_code == 200
        data = resp.json()
        assert "published_official_baselines" in data
        assert data["physical_corpus_status"]["storage_guard"] == "ADR-002"

    def test_session_history_records_actions(self, client: TestClient) -> None:
        resp = client.get("/api/history")
        assert resp.status_code == 200
        history = resp.json()
        assert isinstance(history, list)
        assert len(history) > 0  # previous analyze calls recorded
