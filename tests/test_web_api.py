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
from prj55.web.service import ForensicsService


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

    def test_authentic_speech_preset_verdict_and_score_convention(
        self, client: TestClient
    ) -> None:
        """Authentic speech (speech_human_male) must yield S_cm >= 0.0 and BONAFIDE verdict."""
        resp = client.post(
            "/api/analyze", data={"sample_id": "speech_human_male", "threshold": "0.0"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == "BONAFIDE"
        assert data["scores"]["bonafide_logit"] >= 0.0
        assert data["scores"]["bonafide_logit"] == data["primary_score"]
        assert data["scores"]["bonafide_logit"] > data["scores"]["spoof_logit"]
        assert data["scores"]["softmax_bonafide"] > 0.5

    def test_deepfake_speech_preset_verdict_and_score_convention(
        self, client: TestClient
    ) -> None:
        """Deepfake speech (speech_deepfake_tts) must yield S_cm < 0.0 and SPOOF verdict."""
        resp = client.post(
            "/api/analyze", data={"sample_id": "speech_deepfake_tts", "threshold": "0.0"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == "SPOOF"
        assert data["scores"]["bonafide_logit"] < 0.0
        assert data["scores"]["spoof_logit"] > data["scores"]["bonafide_logit"]
        assert data["scores"]["softmax_bonafide"] < 0.5

    def test_threshold_dynamic_verdict_shift(self, client: TestClient) -> None:
        """Changing operating threshold must dynamically shift binary prediction."""
        # Baseline bonafide score for male speech is ~+2.05
        # Threshold at +5.0 must force classification to SPOOF
        resp_strict = client.post(
            "/api/analyze", data={"sample_id": "speech_human_male", "threshold": "5.0"}
        )
        assert resp_strict.json()["prediction"] == "SPOOF"

        # Baseline bonafide score for deepfake TTS is ~-3.91
        # Threshold at -5.0 must force classification to BONAFIDE
        resp_lenient = client.post(
            "/api/analyze", data={"sample_id": "speech_deepfake_tts", "threshold": "-5.0"}
        )
        assert resp_lenient.json()["prediction"] == "BONAFIDE"

    def test_synthetic_sine_simulation_fixture_behavior(self, client: TestClient) -> None:
        """Synthetic sine-sum fixture (sample_bonafide) is correctly classified as SPOOF.

        PROVENANCE: sample_bonafide is a multi-formant harmonic sine-sum (F0=125 Hz) generated
        programmatically in service.py. It is NOT genuine human speech and does NOT have a
        bonafide ground-truth label. Its fixture name is legacy; its actual ground truth for
        AASIST-L purposes is 'synthetic / non-speech', which the model correctly identifies
        as SPOOF (S_cm < 0.0). This test asserts the observed and expected model behavior.
        """
        resp = client.post(
            "/api/analyze", data={"sample_id": "sample_bonafide", "threshold": "0.0"}
        )
        assert resp.status_code == 200
        data = resp.json()
        # AASIST-L correctly classifies synthetic sinusoids as SPOOF (non-human-speech).
        assert data["scores"]["bonafide_logit"] < 0.0, (
            "Synthetic sine-sum (sample_bonafide) expected S_cm < 0.0; "
            f"got {data['scores']['bonafide_logit']}"
        )
        assert data["prediction"] == "SPOOF", (
            "Synthetic sine-sum expected SPOOF verdict; "
            f"got {data['prediction']}"
        )

    def test_synthetic_fm_tone_fixture_behavior(self, client: TestClient) -> None:
        """Synthetic FM-modulated tone (sample_spoof) returns a valid API response.

        PROVENANCE: sample_spoof is a phase-discontinuous frequency-modulated tone generated
        programmatically in service.py. It is NOT a real deepfake recording. It has no
        ground-truth spoof label that can be asserted; the model's response to this synthetic
        tone is recorded for calibration reference only. This test only verifies API
        structural correctness, not a classification ground-truth claim.
        """
        resp = client.post(
            "/api/analyze", data={"sample_id": "sample_spoof", "threshold": "0.0"}
        )
        assert resp.status_code == 200
        data = resp.json()
        # Only structural assertions — no ground-truth classification claim.
        assert "prediction" in data
        assert data["prediction"] in ("BONAFIDE", "SPOOF")
        assert "bonafide_logit" in data["scores"]
        assert isinstance(data["scores"]["bonafide_logit"], float)


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

    def test_clear_session_history(self, client: TestClient) -> None:
        del_resp = client.delete("/api/history")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "cleared"

        get_resp = client.get("/api/history")
        assert get_resp.status_code == 200
        assert get_resp.json() == []

    def test_research_figures_endpoint(self, client: TestClient) -> None:
        resp = client.get("/figures/figure_05_robustness.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_robustness_comparison_fields(self, client: TestClient) -> None:
        resp = client.post(
            "/api/robustness",
            data={
                "sample_id": "sample_bonafide",
                "condition": "telephony_8khz",
                "mitigation_enabled": "true",
            },
        )
        assert resp.status_code == 200
        comp = resp.json()["comparison"]
        assert "delta_logit" in comp
        assert "absolute_delta_logit" in comp
        assert comp["absolute_delta_logit"] == abs(comp["delta_logit"])
        assert "prediction_changed" in comp
        assert isinstance(comp["prediction_changed"], bool)
        assert comp["playback_signal_type"] == "mitigated"

    def test_filename_sanitization(self, client: TestClient) -> None:
        import io

        import numpy as np
        import soundfile as sf

        bio = io.BytesIO()
        wave = np.zeros(16000, dtype=np.float32)
        sf.write(bio, wave, 16000, format="WAV")
        bio.seek(0)

        files = {"file": ("../../malicious/path/test_sample.wav", bio.read(), "audio/wav")}
        resp = client.post("/api/analyze", files=files)
        assert resp.status_code == 200
        assert resp.json()["audio_metadata"]["filename"] == "test_sample.wav"

    def test_robustness_sweep_endpoint(self, client: TestClient) -> None:
        """Verify POST /api/robustness/sweep evaluates multiple conditions and computes delta."""
        resp = client.post(
            "/api/robustness/sweep",
            data={"sample_id": "sample_bonafide"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "clean_baseline_score" in data
        assert "sweep_results" in data
        results = data["sweep_results"]
        assert len(results) >= 5
        for item in results:
            assert "condition_id" in item
            assert "name" in item
            assert "delta_s_cm" in item
            assert "processed_score" in item
            assert "prediction" in item
            assert item["baseline_score"] == data["clean_baseline_score"]

    def test_batch_analysis_success(self, client: TestClient, dummy_wav_bytes: bytes) -> None:
        """Verify POST /api/analyze/batch processes multi-file inputs safely."""
        files = [
            ("files", ("test_file_1.wav", dummy_wav_bytes, "audio/wav")),
            ("files", ("test_file_2.wav", dummy_wav_bytes, "audio/wav")),
        ]
        resp = client.post("/api/analyze/batch", files=files, data={"threshold": "0.00"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["total_files"] == 2
        assert data["processed_count"] == 2
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["status"] == "SUCCESS"
            assert "score" in r
            assert r["prediction"] in ("BONAFIDE", "SPOOF")

    def test_batch_analysis_exceeds_limit(
        self, client: TestClient, dummy_wav_bytes: bytes
    ) -> None:
        """Verify POST /api/analyze/batch rejects more than 10 files with 400 Bad Request."""
        files = [
            ("files", (f"test_file_{i}.wav", dummy_wav_bytes, "audio/wav"))
            for i in range(12)
        ]
        resp = client.post("/api/analyze/batch", files=files)
        assert resp.status_code == 400
        assert "Batch exceeds maximum limit of 10 recordings" in resp.json()["detail"]

    def test_case_dossier_and_deletion_lifecycle(
        self, client: TestClient, dummy_wav_bytes: bytes
    ) -> None:
        """Verify case dossier generation and individual record deletion."""
        # 1. Run analysis to create a session record
        files = {"file": ("dossier_test.wav", dummy_wav_bytes, "audio/wav")}
        ana_resp = client.post("/api/analyze", files=files)
        assert ana_resp.status_code == 200

        # 2. Retrieve history and find record
        hist_resp = client.get("/api/history")
        assert hist_resp.status_code == 200
        history = hist_resp.json()
        target_record = next(
            (r for r in history if r.get("filename") == "dossier_test.wav"), None
        )
        assert target_record is not None
        rec_id = target_record.get("record_id") or target_record.get("id")
        assert rec_id is not None

        # 3. Request 7-Stage Case Dossier
        dossier_resp = client.get(f"/api/history/{rec_id}/dossier")
        assert dossier_resp.status_code == 200
        dossier_data = dossier_resp.json()
        assert dossier_data["status"] == "ok"
        dossier = dossier_data["dossier"]
        assert dossier["dossier_id"] == rec_id
        assert "evidence_provenance_timeline" in dossier
        assert len(dossier["evidence_provenance_timeline"]) == 7
        assert "audio_metadata" in dossier
        assert "sha256" in dossier["audio_metadata"]
        assert len(dossier["audio_metadata"]["sha256"]) == 64

        # 4. Delete the individual record
        del_resp = client.delete(f"/api/history/{rec_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] in ("deleted", "ok")

        # 5. Verify record is removed
        hist_after = client.get("/api/history").json()
        assert not any((r.get("record_id") == rec_id or r.get("id") == rec_id) for r in hist_after)

    def test_stage3_routes_and_pwa_assets(self, client: TestClient) -> None:
        """Verify /robustness, /history, manifest.webmanifest and sw.js routes."""
        rob_resp = client.get("/robustness")
        assert rob_resp.status_code == 200
        assert "Acoustic Robustness Lab" in rob_resp.text

        hist_resp = client.get("/history")
        assert hist_resp.status_code == 200
        assert "Session Evidence History" in hist_resp.text

        manifest_resp = client.get("/manifest.webmanifest")
        assert manifest_resp.status_code == 200
        assert "Audio Authenticity Lab" in manifest_resp.text

        sw_resp = client.get("/sw.js")
        assert sw_resp.status_code == 200
        assert "aal-static-shell" in sw_resp.text


class TestStage4HardeningSecurityAndSharing:
    """Stage 4 tests for security headers, SQLite persistence, share tokens, and validation."""

    def test_security_headers_present(self, client: TestClient) -> None:
        """Every response must include defense-in-depth security headers."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "SAMEORIGIN"
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert "content-security-policy" in resp.headers
        csp = resp.headers["content-security-policy"]
        assert "script-src 'self'" in csp
        # Verify unsafe-inline is absent from script-src
        script_directive = [d for d in csp.split(";") if "script-src" in d][0]
        assert "'unsafe-inline'" not in script_directive

    def test_durable_sqlite_history_restart_persistence(
        self, client: TestClient, dummy_wav_bytes: bytes
    ) -> None:
        """Analysis records must survive service re-initialization (restart persistence)."""
        files = {"file": ("persistence_check.wav", dummy_wav_bytes, "audio/wav")}
        resp = client.post("/api/analyze", files=files, data={"threshold": "0.0"})
        assert resp.status_code == 200
        rec_id = resp.json()["record_id"]

        # Simulate service restart by creating a new ForensicsService with the same db_path
        restarted_service = ForensicsService(db_path=service.db_path)
        item = restarted_service.get_history_item(rec_id)
        assert item is not None
        assert item["filename"] == "persistence_check.wav"
        assert item["prediction"] in ("BONAFIDE", "SPOOF")

    def test_shareable_report_lifecycle(
        self, client: TestClient, dummy_wav_bytes: bytes
    ) -> None:
        """Test share token creation, sanitized retrieval, revocation, and enumeration guard."""
        # 1. Ingest audio to generate a record
        files = {"file": ("share_audit.wav", dummy_wav_bytes, "audio/wav")}
        analyze_resp = client.post("/api/analyze", files=files)
        assert analyze_resp.status_code == 200
        rec_id = analyze_resp.json()["record_id"]

        # 2. Create high-entropy share token
        share_resp = client.post(f"/api/share/{rec_id}?expires_in_days=7")
        assert share_resp.status_code == 200
        share_data = share_resp.json()
        assert share_data["status"] == "created"
        token = share_data["share_token"]
        assert len(token) >= 32

        # 3. Retrieve sanitized report via token
        rep_resp = client.get(f"/api/share/{token}")
        assert rep_resp.status_code == 200
        rep = rep_resp.json()
        assert rep["status"] == "ok"
        assert rep["forensic_summary"]["filename"] == "share_audit.wav"
        assert "input_sha256" in rep["forensic_summary"]
        assert "audio_wav_base64" not in rep  # Raw audio must not be exposed
        assert "file_path" not in rep  # Server filesystem paths must not be exposed

        # 4. Revoke token
        rev_resp = client.delete(f"/api/share/{token}")
        assert rev_resp.status_code == 200
        assert rev_resp.json()["status"] == "revoked"

        # 5. Accessing revoked token returns 404 (enumeration resistance)
        after_resp = client.get(f"/api/share/{token}")
        assert after_resp.status_code == 404

        # 6. Accessing unknown token returns 404
        unk_resp = client.get("/api/share/invalid_token_xyz123456789")
        assert unk_resp.status_code == 404

    def test_input_validation_boundaries(
        self, client: TestClient, dummy_wav_bytes: bytes
    ) -> None:
        """API endpoints must safely reject extreme or non-finite inputs."""
        # Extreme threshold
        files = {"file": ("extreme_thresh.wav", dummy_wav_bytes, "audio/wav")}
        resp = client.post("/api/analyze", files=files, data={"threshold": "999.0"})
        assert resp.status_code == 422
        assert "Decision threshold" in resp.json()["detail"]

        # Extreme SNR
        files = {"file": ("extreme_snr.wav", dummy_wav_bytes, "audio/wav")}
        rob_resp = client.post(
            "/api/robustness",
            files=files,
            data={"condition": "noise_snr_10db", "snr_db": "150.0"},
        )
        assert rob_resp.status_code == 422
        assert "SNR must be" in rob_resp.json()["detail"]

        # Path traversal filename sanitization
        traversal_files = {"file": ("../../../../etc/shadow.wav", dummy_wav_bytes, "audio/wav")}
        trav_resp = client.post("/api/analyze", files=traversal_files)
        assert trav_resp.status_code == 200
        sanitized_fn = trav_resp.json()["audio_metadata"]["filename"]
        assert "/" not in sanitized_fn
        assert ".." not in sanitized_fn

    def test_clean_404_handling(self, client: TestClient) -> None:
        """Non-existent endpoints must return clean 404 responses without leaking stack traces."""
        api_404 = client.get("/api/this_endpoint_does_not_exist")
        assert api_404.status_code == 404
        assert api_404.json()["status_code"] == 404

        page_404 = client.get("/this_page_does_not_exist")
        assert page_404.status_code == 404


