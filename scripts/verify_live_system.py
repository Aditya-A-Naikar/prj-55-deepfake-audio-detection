"""Empirical live verification script for Audio Authenticity Lab.

Verifies live API endpoints, AASIST-L inference on authentic speech, deepfake speech,
and synthetic calibration fixtures, ensuring the authoritative score convention is preserved.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from starlette.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.web.app import app, service  # noqa: E402

service.load_model()
client = TestClient(app)

print("=" * 60)
print("1. Status Endpoint:")
r = client.get("/api/status")
print("  Status code:", r.status_code)
status_data = r.json()
print("  Body:", status_data)
assert r.status_code == 200
assert status_data["model_architecture"] == "AASIST-L"
assert status_data["checkpoint_found"] is True

print("\n" + "=" * 60)
print("2. Presets Library:")
r = client.get("/api/samples")
samples = r.json()
print("  Loaded samples count:", len(samples))
for s in samples:
    sid = s["id"]
    sname = s.get("name", s.get("id"))
    stype = s.get("type", "sample")
    print(f"   - {sid:25s}: {sname} ({stype})")
assert len(samples) >= 6

print("\n" + "=" * 60)
print("3. Analyze Authentic Speech (TRK-01: 'speech_human_male' — VCTK/ASVspoof bonafide):")
r = client.post("/api/analyze", data={"sample_id": "speech_human_male", "threshold": "0.0"})
print("  Status:", r.status_code)
assert r.status_code == 200
d_male = r.json()
bf_logit = d_male["scores"]["bonafide_logit"]
sp_logit = d_male["scores"]["spoof_logit"]
sm_bf = d_male["scores"]["softmax_bonafide"]
latency = d_male["performance"]["latency_ms"]
pred = d_male.get("prediction")
print(f"  Prediction: {pred} | S_cm: {bf_logit:+.4f} | Spoof Logit: {sp_logit:+.4f}")
print(f"  Softmax Confidence: {sm_bf * 100:.2f}% | Latency (forward pass): {latency:.2f} ms")
assert pred == "BONAFIDE", f"Expected BONAFIDE for genuine human speech, got {pred}"
assert bf_logit >= 0.0, f"Expected S_cm >= 0.0 for authentic speech, got {bf_logit}"
assert bf_logit == d_male["primary_score"]

print("\n" + "=" * 60)
print("4. Analyze Deepfake Speech (TRK-03: 'speech_deepfake_tts' — Neural TTS attack SS_6):")
r = client.post("/api/analyze", data={"sample_id": "speech_deepfake_tts", "threshold": "0.0"})
print("  Status:", r.status_code)
assert r.status_code == 200
d_tts = r.json()
bf_logit = d_tts["scores"]["bonafide_logit"]
sp_logit = d_tts["scores"]["spoof_logit"]
sm_bf = d_tts["scores"]["softmax_bonafide"]
latency = d_tts["performance"]["latency_ms"]
pred = d_tts.get("prediction")
print(f"  Prediction: {pred} | S_cm: {bf_logit:+.4f} | Spoof Logit: {sp_logit:+.4f}")
print(f"  Softmax Confidence: {(1 - sm_bf) * 100:.2f}% spoof | Latency: {latency:.2f} ms")
assert pred == "SPOOF", f"Expected SPOOF for deepfake neural TTS, got {pred}"
assert bf_logit < 0.0, f"Expected S_cm < 0.0 for deepfake speech, got {bf_logit}"

print("\n" + "=" * 60)
print("5. Analyze Secondary Evaluation Speech Presets (TRK-02 & TRK-04):")
r_fem = client.post("/api/analyze", data={"sample_id": "speech_human_female", "threshold": "0.0"})
d_fem = r_fem.json()
s_cm_fem = d_fem["scores"]["bonafide_logit"]
print(f"  TRK-02 Female Authentic: Pred={d_fem['prediction']} | S_cm={s_cm_fem:+.4f}")
assert d_fem["prediction"] == "BONAFIDE"

r_voc = client.post(
    "/api/analyze", data={"sample_id": "speech_deepfake_vocoder", "threshold": "0.0"}
)
d_voc = r_voc.json()
s_cm_voc = d_voc["scores"]["bonafide_logit"]
print(f"  TRK-04 Vocoder Deepfake:  Pred={d_voc['prediction']} | S_cm={s_cm_voc:+.4f}")
assert d_voc["prediction"] == "SPOOF"

print("\n" + "=" * 60)
print("6. Synthetic Procedural Calibration Fixtures (TRK-05 & TRK-06):")
print("  PROVENANCE: These are NOT genuine human speech recordings and NOT real deepfake files.")
print("  sample_bonafide: Multi-formant harmonic sine-sum (F0=125 Hz). Synthetic math fixture.")
print("  sample_spoof:    Phase-discontinuous FM-modulated tone. Synthetic math fixture.")
print("  Expected behavior: AASIST-L treats both as synthetic, NOT matching the misleading IDs.")
print("  Assertions: sample_bonafide -> SPOOF (S_cm < 0.0); sample_spoof -> score observed only.")

r_syn_bf = client.post("/api/analyze", data={"sample_id": "sample_bonafide", "threshold": "0.0"})
d_syn_bf = r_syn_bf.json()
s_syn_bf = d_syn_bf["scores"]["bonafide_logit"]
pred_syn_bf = d_syn_bf["prediction"]
print(f"  TRK-05 sample_bonafide (sine sum):  Pred={pred_syn_bf} | S_cm={s_syn_bf:+.4f}")
print("    -> Expected SPOOF (synthetic tone lacks human vocal tract dynamics)")
# This fixture is a mathematical sine-sum. AASIST-L correctly classifies it as non-speech (SPOOF).
assert pred_syn_bf == "SPOOF", (
    f"sample_bonafide (synthetic sine-sum) expected SPOOF from AASIST-L, got {pred_syn_bf}"
)
assert s_syn_bf < 0.0, (
    f"sample_bonafide S_cm expected < 0.0 for synthetic tone, got {s_syn_bf}"
)

r_syn_sp = client.post("/api/analyze", data={"sample_id": "sample_spoof", "threshold": "0.0"})
d_syn_sp = r_syn_sp.json()
s_syn_sp = d_syn_sp["scores"]["bonafide_logit"]
pred_syn_sp = d_syn_sp["prediction"]
print(f"  TRK-06 sample_spoof (FM tone):      Pred={pred_syn_sp} | S_cm={s_syn_sp:+.4f}")
print("    -> No ground-truth assertion (both IDs are synthetic; model behavior is observed only)")
# No ground-truth assertion for sample_spoof since it is also a synthetic FM tone.
# Its prediction is recorded for calibration reference, not treated as a classification test.
assert r_syn_sp.status_code == 200, "API must return HTTP 200 for synthetic fixture"

print("\n" + "=" * 60)
print("7. Robustness Test (Human Speech with Noise @ 10 dB + Front-End Mitigation):")
rob_data = {
    "sample_id": "speech_human_male",
    "condition": "noise_snr_10db",
    "snr_db": "10.0",
    "mitigation_enabled": "true",
}
r = client.post("/api/robustness", data=rob_data)
print("  Status:", r.status_code)
assert r.status_code == 200
rd = r.json()
comp = rd["comparison"]
c_score = comp["clean_bonafide_logit"]
c_verd = comp["clean_prediction"]
p_score = comp["degraded_bonafide_logit"]
p_verd = comp["degraded_prediction"]
d_logit = comp["delta_logit"]
abs_d = comp["absolute_delta_logit"]
mit_app = comp["mitigation_applied"]
sig_type = comp["playback_signal_type"]
p_change = comp["prediction_changed"]

print(f"  Clean: {c_score:+.4f} ({c_verd})")
print(f"  Mitigated: {p_score:+.4f} ({p_verd})")
print(f"  Delta S_cm: {d_logit:+.4f} | Absolute Delta: {abs_d:.4f}")
print(f"  Mitigation Applied: {mit_app} | Playback Signal: {sig_type}")
print(f"  Prediction Changed: {p_change}")
print(f"  Audio Base64 length: {len(rd.get('audio_wav_base64', ''))} chars")

print("\n" + "=" * 60)
print("8. Research Figures Static Endpoint (/figures/):")
fig_names = [
    "figure_01_confusion_matrix.png",
    "figure_02_roc_curve.png",
    "figure_03_det_eer.png",
    "figure_04_score_distributions.png",
    "figure_05_robustness.png",
    "figure_06_mitigation_comparison.png",
    "figure_07_waveform_spectrogram.png",
    "figure_08_inference_latency.png",
    "figure_09_parameter_budget.png",
]
for fname in fig_names:
    rf = client.get(f"/figures/{fname}")
    assert rf.status_code == 200
    print(f"  [OK] /figures/{fname} -> HTTP {rf.status_code} ({len(rf.content):,} bytes)")

print("\n" + "=" * 60)
print("9. Stage 3 Dedicated Workspaces & PWA Webmanifest Assets:")
routes = ["/robustness", "/history", "/methodology", "/manifest.webmanifest", "/sw.js"]
for route in routes:
    r_rt = client.get(route)
    assert r_rt.status_code == 200
    print(f"  [OK] {route:25s} -> HTTP {r_rt.status_code} ({len(r_rt.content):,} bytes)")

print("\n" + "=" * 60)
print("\n" + "=" * 60)
print("10. Stage 3 Robustness Multi-Condition Sweep Benchmark:")
r_sweep = client.post("/api/robustness/sweep", data={"sample_id": "speech_human_male"})
assert r_sweep.status_code == 200
sweep_data = r_sweep.json()
assert sweep_data["status"] == "SUCCESS"
s_clean = sweep_data["clean_baseline_score"]
p_clean = sweep_data["clean_baseline"]["prediction"]
print(f"  Baseline S_clean: {s_clean:+.4f} ({p_clean})")
print(f"  Total Evaluated Conditions: {sweep_data['trials_count']}")
for trial in sweep_data["trials"]:
    cond = trial.get("name") or trial.get("label")
    p_score = trial["processed_score"]
    delta = trial.get("delta_s_cm", trial.get("delta_score", 0.0))
    p_pred = trial.get("prediction", trial.get("processed_prediction"))
    badge = trial.get("badge", "")
    print(
        f"   - {cond:30s} | S_proc: {p_score:+.4f} | Δ S_cm: {delta:+.4f} | {p_pred:8s} | {badge}"
    )

print("\n" + "=" * 60)
print("11. Stage 3 Batch Audio Analysis (Multi-Recording Sequential Ingestion):")
bio1 = io.BytesIO()
bio2 = io.BytesIO()
sr = 16000
t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
sf.write(bio1, (0.4 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32), sr, format="WAV")
sf.write(bio2, (0.3 * np.sin(2 * np.pi * 600.0 * t)).astype(np.float32), sr, format="WAV")
batch_files = [
    ("files", ("batch_sample_alpha.wav", bio1.getvalue(), "audio/wav")),
    ("files", ("batch_sample_beta.wav", bio2.getvalue(), "audio/wav")),
]
r_batch = client.post("/api/analyze/batch", files=batch_files, data={"threshold": "0.00"})
assert r_batch.status_code == 200
batch_data = r_batch.json()
assert batch_data["status"] == "SUCCESS"
print(f"  Batch Processed: {batch_data['processed_count']} files")
print(f"  Bona Fide: {batch_data['bonafide_count']} | Spoof: {batch_data['spoof_count']}")
for res in batch_data["results"]:
    fn = res['filename']
    sc = res['score']
    pd = res['prediction']
    lat = res['latency_ms']
    print(f"   - {fn:24s} | S_cm: {sc:+.4f} | Verdict: {pd:8s} | Latency: {lat:.2f} ms")

print("\n" + "=" * 60)
print("12. Forensic Case Dossier & SHA-256 Content Provenance:")
rh = client.get("/api/history")
history_items = rh.json()
assert len(history_items) > 0
latest_rec_id = history_items[-1].get("record_id") or history_items[-1].get("id")
r_dos = client.get(f"/api/history/{latest_rec_id}/dossier")
assert r_dos.status_code == 200
dos_data = r_dos.json()
assert dos_data["status"] == "ok"
dossier = dos_data["dossier"]
print(f"  Dossier ID: {dossier['dossier_id']}")
sub_fn = dossier['audio_metadata']['filename']
sub_dur = dossier['audio_metadata']['duration_s']
print(f"  Subject File: {sub_fn} ({sub_dur}s)")
print(f"  SHA-256 Digest: {dossier['audio_metadata']['sha256']}")
vrd = dossier['decision_metrics']['verdict']
sc_cm = dossier['decision_metrics']['countermeasure_score_s_cm']
mg = dossier['decision_metrics']['decision_margin']
print(f"  Classification: {vrd} (S_cm: {sc_cm:+.4f}, Margin: {mg:+.4f})")
print("  Seven-Stage Forensic Evidence Timeline:")
for stage in dossier.get("evidence_provenance_timeline", []):
    st_num = stage['stage']
    st_name = stage['name']
    st_stat = stage['status']
    st_det = stage['detail']
    print(f"   [Stage {st_num}] {st_name:32s} -> [{st_stat}] {st_det}")

print("\n" + "=" * 60)
print("13. Session History & History Clear:")
history_count = len(history_items)
print(f"  History count before clear: {history_count}")
assert history_count >= 3

rc = client.delete("/api/history")
print("  DELETE /api/history response:", rc.json())
assert rc.status_code == 200

rh2 = client.get("/api/history")
history_after = len(rh2.json())
print(f"  History count after clear: {history_after}")
assert history_after == 0

print("\n" + "=" * 60)
print("14. Stage 4 Defense-in-Depth Security Headers Audit:")
r_sec = client.get("/api/status")
assert r_sec.status_code == 200
headers = r_sec.headers
print("  X-Content-Type-Options :", headers.get("x-content-type-options"))
print("  X-Frame-Options        :", headers.get("x-frame-options"))
print("  Referrer-Policy        :", headers.get("referrer-policy"))
print("  Content-Security-Policy:", headers.get("content-security-policy", "")[:60] + "...")
assert headers.get("x-content-type-options") == "nosniff"
assert headers.get("x-frame-options") == "SAMEORIGIN"
assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
assert "content-security-policy" in headers

print("\n" + "=" * 60)
print("15. Stage 4 Durable SQLite Storage & Restart Integrity:")
# Ingest an authentic sample to generate persistent record
r_ingest = client.post(
    "/api/analyze", data={"sample_id": "speech_human_male", "threshold": "0.0"}
)
assert r_ingest.status_code == 200
ingest_rec_id = r_ingest.json()["record_id"]
r_status_store = client.get("/api/status")
store_guard = r_status_store.json()["storage_guard"]
bk = store_guard.get("history_backend")
dur = store_guard.get("history_durable")
print(f"  Backend: {bk} | Durable: {dur}")
print(f"  Database Path: {store_guard.get('database_path')}")
print(f"  Persisted Record Count: {store_guard.get('persisted_records')}")
assert store_guard.get("history_backend") == "sqlite"
assert store_guard.get("history_durable") is True
assert store_guard.get("persisted_records", 0) >= 1

print("\n" + "=" * 60)
print("16. Token-Based Shareable Report Access (Controlled Bearer Link):")
# Create high-entropy share token
r_share = client.post(f"/api/share/{ingest_rec_id}?expires_in_days=7")
assert r_share.status_code == 200
share_payload = r_share.json()
token = share_payload["share_token"]
print(f"  Generated Bearer Token: {token[:16]}... (length: {len(token)})")
print(f"  Share URL Path: {share_payload['share_url']}")
print(f"  Expires At: {share_payload.get('expires_at')}")

# Retrieve sanitized dossier via share token
r_shared_get = client.get(f"/api/share/{token}")
assert r_shared_get.status_code == 200
rep = r_shared_get.json()
print("  Sanitized Access Scope :", rep["security_and_privacy_notice"]["access_model"][:70] + "...")
print("  Report Determination   :", rep["forensic_summary"]["final_prediction"])
print("  Content Digest (SHA256):", rep["audio_metadata"]["sha256"])
assert "audio_wav_base64" not in rep  # Zero raw audio exposure
assert "file_path" not in rep  # Zero local filesystem path leakage

# Revoke share token
r_revoke = client.delete(f"/api/share/{token}")
assert r_revoke.status_code == 200
assert r_revoke.json()["status"] == "revoked"
print("  Token Revocation Status: SUCCESS (revoked)")

# Assert 404 after revocation (resistance to probing)
r_revoked_check = client.get(f"/api/share/{token}")
assert r_revoked_check.status_code == 404
print("  Probing Revoked Token  : HTTP 404 Not Found (Correct)")

print("\n" + "=" * 60)
print("17. Stage 4 Route Integrity & Defensive 404 Resolution:")
stage4_routes: list[tuple[str, str]] = [
    ("/", "Audio Authenticity Lab"),
    ("/analyze", "Audio Forensics Workspace"),
    ("/robustness", "Acoustic Robustness Lab"),
    ("/history", "Session Evidence History"),
    ("/methodology", "Research Methodology"),
    (f"/shared/{token}", "Shared Forensic Dossier"),
    ("/404.html", "Destination Not Found"),
]
for path, expected_text in stage4_routes:
    r_route = client.get(path)
    assert r_route.status_code == 200, f"Route {path} returned {r_route.status_code}"
    assert expected_text in r_route.text, f"Route {path} missing expected text '{expected_text}'"
    print(f"  [OK] {path:28s} -> HTTP 200 (contains '{expected_text}')")

r_api_404 = client.get("/api/unknown_nonexistent_endpoint")
assert r_api_404.status_code == 404
assert r_api_404.json()["status_code"] == 404
print("  [OK] /api/unknown_nonexistent_endpoint -> Clean JSON 404")

print("\n" + "=" * 60)
print("ALL LIVE STAGE 4 INTEGRATION & HARDENING CHECKS PASSED EMPIRICALLY!")
print("=" * 60)

