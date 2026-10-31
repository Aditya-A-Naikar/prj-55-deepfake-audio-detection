# Audio Authenticity Lab (PRJ_55)
### Robust and Compute-Efficient Speech Anti-Spoofing & Acoustic Forensics Platform

**Academic Context:** B.Tech Computer Science & Engineering Final Year Main Project  
**Target Architecture:** Apple Silicon M2 (macOS, ARM64) with CPU Fallback  
**Verification Date:** October 31, 2026  
**Status:** **Production Platform Verified & Hardened** — 265 Automated Tests Passing (2 Skipped, 0 Failed)

---

## 1. Executive Summary

**Audio Authenticity Lab (PRJ_55)** is an on-device, compute-efficient speech deepfake detection and acoustic forensics intelligence platform. While contemporary foundation models frequently exceed hundreds of millions of parameters, require datacenter GPUs, and suffer catastrophic accuracy collapse when audio passes through lossy communications channels, this project establishes a hardened, lightweight forensic workstation engineered around the **AASIST-L** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention — Lightweight) architecture.

### Key Capabilities

1. **Ultra-Compact Neural Footprint**: AASIST-L operates with **85,306 trainable parameters** (~334 KB parameters; 426 KB complete checkpoint including SincNet buffers), processing raw 16 kHz audio directly via parameterized bandpass sinc filters and dual heterogeneous spectro-temporal graph attention layers.
2. **Sub-50ms On-Device Inference**:
   - **Apple Silicon Metal GPU (MPS)**: Median **~36 ms** isolated model forward pass; median **~41 ms** complete ForensicsService pipeline (~24.6 analyses/second).
   - **Apple Silicon M2 CPU**: Median **~163 ms** isolated model forward pass; median **~181 ms** complete ForensicsService pipeline (~5.5 analyses/second).
3. **Six Integrated Application Routes**: Dedicated landing page, forensic inspection workspace, acoustic robustness lab, persistent session history, research methodology reference, and token-based shareable case dossiers.
4. **Live Microphone Ingestion & Container Transcoding**: In-browser PCM audio transcoding to standard 16 kHz 16-bit mono RIFF WAV, rejecting compressed WebM/MP4 streams to guarantee pristine time-domain framing.
5. **Acoustic Robustness Sweeps & Front-End DSP Mitigation**: Multi-condition degradation simulator (telephony 300–3,400 Hz bandpass, additive white Gaussian noise at 0–20 dB SNR, lossy Opus compression) paired with non-parametric spectral subtraction recovering classification margins under adverse SNR without retraining.
6. **ACID-Compliant Persistent History & ADR-002 Research Governance**: SQLite WAL ledger (`data/history.db`) persisting SHA-256 digests, mathematical logits, and metadata across restarts with strict data privacy guaranteeing raw audio is never stored on disk.
7. **Controlled Bearer Token Sharing**: 256-bit URL-safe unguessable bearer tokens (`secrets.token_urlsafe(32)`) with time-limited expiration, instant revocation, and uniform HTTP 404 anti-enumeration defense.

---

## 2. Application Routes & Workspace Architecture

The web platform is organized into six dedicated routes served by FastAPI with zero client-side framework bloat:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUDIO AUTHENTICITY LAB (PRJ_55)                       │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ Route             │ Functional Scope & Capabilities                         │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ /                 │ Product Landing: 3D harmonic particle ribbon, hardware  │
│                   │ telemetry, model specifications, and ADR-002 policies.  │
│ /analyze          │ Forensic Workspace: Drag-and-drop file upload, bundled  │
│                   │ speech smoke-test presets, live microphone recording,   │
│                   │ waveform scrubber, 0–8 kHz spectrogram, batch queue.    │
│ /robustness       │ Acoustic Robustness Lab: Channel degradation sweeps,    │
│                   │ spectral subtraction DSP mitigation, and A/B audio.     │
│ /history          │ Session Evidence History: Persistent SQLite audit log,  │
│                   │ JSON export, single-record purging, and history clear.  │
│ /methodology      │ Research Methodology: AASIST-L architecture breakdown,  │
│                   │ decision threshold mathematics, and IEEE citations.     │
│ /shared/{token}   │ Controlled Case Dossier: High-entropy bearer link with   │
│                   │ time-based expiration and instant revocation.           │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

### Detailed Route Specifications

- **`/` — Overview & 3D Wavefield**: Canvas-based interactive 3D harmonic signal sculpture responding to cursor motion and real-time inference pulses, accompanied by live operational telemetry and storage governance disclosures.
- **`/analyze` — Forensic Analysis Workspace**:
  - **Dual Presentation Modes**: Segmented toggle between **Simple View** (plain-language executive findings and decision margin) and **Expert View** (unconstrained logits $S_{\text{bonafide}}$, $S_{\text{spoof}}$, decision margin, latency, RTF, and architectural tensor specs).
  - **Bundled Speech Demonstration Presets**: Neutral smoke-test recordings (`TRK-01` Male Authentic, `TRK-02` Female Authentic, `TRK-03` Neural TTS Deepfake, `TRK-04` Vocoder Artifact Deepfake) and procedural mathematical calibration fixtures (`TRK-05` Harmonic Sine-Sum, `TRK-06` FM Tone).
  - **Interactive Waveform Scrubber**: Click/drag time-domain waveform scrubbing synchronized with Web Audio API playback.
  - **0–8,000 Hz FFT Spectrogram**: Real-time canvas heatmap with calibrated frequency markings (`0 Hz`, `2 kHz`, `4 kHz`, `6 kHz`, `8 kHz`) and accessible color legend swatches.
  - **Batch Queue**: Sequential ingestion of up to 10 WAV/FLAC files preventing GPU/CPU memory thrashing while generating consolidated forensic scoring.
  - **Live Microphone Recording**: Client-side PCM capture and in-browser transcoding to standard 16 kHz 16-bit mono RIFF WAV.
- **`/robustness` — Acoustic Robustness Lab**:
  - **Configurable Degradation**: Polyphase 8 kHz telephony resample (G.711 300–3,400 Hz bandpass), additive white Gaussian noise (0–20 dB SNR), and lossy Opus compression.
  - **Front-End DSP Mitigation**: Spectral subtraction ($\alpha=2.0$, $\beta=0.01$; Boll, 1979) and adaptive bandpass compensation.
  - **Systematic Drift Classification**: Identifies outcomes as *Decision Invariant* ($|\Delta S_{\text{cm}}| \le 1.0$), *Decision Retained (Score Sensitive)* ($|\Delta S_{\text{cm}}| > 1.0$, classification unchanged), or *Decision Boundary Breached* (sign flipped across threshold $\theta$).
  - **A/B Audio Auditioning**: Seamless playback switching between pristine clean speech and degraded/mitigated signals.
- **`/history` — Session Evidence Audit Trail**:
  - Displays persistent analysis records loaded from `data/history.db`.
  - Filterable by verdict (`ALL`, `BONAFIDE`, `SPOOF`, `DEGRADED`).
  - One-click JSON dossier export, individual record deletion, and complete session history clearing.
- **`/methodology` — Research Reference & Citations**: Complete mathematical formulations of SincNet filterbanks, Max-Feature-Map (MFM) activations, Heterogeneous Spectro-Temporal Graph Attention (HS-GAL), ASVspoof 2019 baseline tables, and IEEE bibliographic references.
- **`/shared/{token}` — Shareable Evidence Dossier**: Clean, read-only forensic case dossier accessible via a 256-bit URL-safe bearer token, featuring a 7-stage processing timeline, SHA-256 audio verification digest, and instant token revocation.

---

## 3. Scientific Methodology & Decision Logic

### Countermeasure Score Convention

The AASIST-L readout head produces an unprojected 2-dimensional logit vector:

$$\mathbf{z} = [z_0, z_1] = [S_{\text{spoof}}, S_{\text{bonafide}}]$$

Following official ASVspoof protocol conventions, the countermeasure score $S_{\text{cm}}$ is defined by the bona fide logit:

$$S_{\text{cm}} = z_1 = \text{logits}[0, 1]$$

The classification decision rule is:

$$\text{Verdict} = \begin{cases} \text{BONAFIDE}, & \text{if } S_{\text{cm}} \ge \theta \\ \text{SPOOF}, & \text{if } S_{\text{cm}} < \theta \end{cases}$$

### Threshold Calibration Status

- **Canonical Baseline Default ($\theta = 0.00$)**: In the absence of an empirical calibration file generated on labelled development data, the platform operates at the standard uncalibrated baseline threshold $\theta = 0.00$.
- **Automated Calibration Architecture**: The system includes a dedicated calibration engine (`src/prj55/calibration.py`, `scripts/calibrate_threshold.py`) supporting **Equal Error Rate (EER)** and **Target False Alarm Rate (Target FAR)** optimization. The engine strictly enforces partition hygiene: evaluating against evaluation/test partitions raises `CalibrationPartitionError` to prevent data leakage. Full development-set empirical calibration is safely paused pending external SSD acquisition under ADR-002 storage governance.

### Published Upstream Benchmarks vs. Local Demonstration Presets

To maintain complete research integrity, published academic benchmarks are strictly distinguished from local demonstration smoke-tests:

| Architecture | Trainable Parameters | ASVspoof 2019 LA Partition | min t-DCF | EER (%) | Published Upstream Reference |
|:---|:---:|:---:|:---:|:---:|:---|
| **AASIST-L** (PRJ_55 Deployed) | **85,306** | Evaluation (71,237 utts) | **0.0309** | **0.99%** | Jung et al., IEEE ICASSP 2022 [1] |
| **Full AASIST** (Best Run) | 297,446 | Evaluation (71,237 utts) | **0.0275** | **0.83%** | Jung et al., IEEE ICASSP 2022 [1] |
| **Full AASIST** (3-Run Mean) | 297,446 | Evaluation (71,237 utts) | 0.0347 | 0.83% | Jung et al., IEEE ICASSP 2022 [1] |
| **Full AASIST** | 297,446 | Development (24,844 utts) | 0.0345 | 0.99% | Jung et al., IEEE ICASSP 2022 [1] |
| **RawNet2** (Baseline) | 16,600,000 | Evaluation (71,237 utts) | 0.0782 | 1.13% | Jung et al., IEEE ICASSP 2022 [1] |

*Evaluation Scope Disclaimer:* Bundled presets `TRK-01` through `TRK-04` are curated demonstration and smoke-test recordings from the ASVspoof/VCTK corpus. They verify that the end-to-end inference pipeline functions correctly, but do not establish full-dataset statistical detection accuracy. Procedural fixtures `TRK-05` and `TRK-06` are synthetic non-speech tones; the model correctly identifies `TRK-05` as artificial ($S_{\text{cm}} = -4.94$).

---

## 4. Hardware Benchmarks & Performance Metrics

Hardware latency and throughput were empirically benchmarked on an Apple Silicon M2 workstation (8-core CPU, 16 GB unified RAM, PyTorch 2.2.2 with `torch.mps.synchronize()`, $N=30$ iterations, 5 warmups):

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   EMPIRICAL LATENCY & THROUGHPUT BENCHMARKS                 │
├────────────────────────────────────────┬──────────────────┬─────────────────┤
│ Benchmark Condition                    │ Latency (Median) │ Throughput      │
├────────────────────────────────────────┼──────────────────┼─────────────────┤
│ Model Forward Pass (Apple Metal MPS)   │ 36.32 ms         │ 27.5 inf/sec    │
│ ForensicsService Pipeline (MPS)        │ 40.61 ms         │ 24.6 req/sec    │
│ Model Forward Pass (Apple M2 CPU)      │ 162.73 ms        │ 6.15 inf/sec    │
│ ForensicsService Pipeline (Apple M2)   │ 180.82 ms        │ 5.53 req/sec    │
│ HTTP Loopback (Sample-ID, MPS)         │ 41.62 ms         │ ~24.0 req/sec   │
│ HTTP Loopback (Multipart WAV, MPS)     │ 44.36 ms         │ ~22.5 req/sec   │
│ Concurrent Model Forward (4 Workers)   │ —                │ 5.43–6.63 inf/s │
│ Concurrent ForensicsService (4 Workers)│ —                │ 2.06 req/sec    │
└────────────────────────────────────────┴──────────────────┴─────────────────┘
```

*Concurrency Workload Distinction:* Isolated multi-worker forward passes achieve 5.43–6.63 inferences/second on CPU without disk persistence. When full end-to-end service execution includes SHA-256 digest computation, preprocessing, and persistent SQLite disk writes, throughput measures 2.06 analyses/second. Potential contributing factors include SQLite disk write serialization under WAL mode and PyTorch multi-thread CPU contention under concurrent worker load.

---

## 5. Storage Governance & Corpus Management (ADR-002)

- **Corpus Footprint**: The official ASVspoof 2019 Logical Access archive (`LA.zip`) is **7.64 GB compressed**, expanding to **~25–30 GB** across 121,420 uncompressed FLAC audio files.
- **ADR-002 Research Storage Policy**: Downloading the uncompressed physical dataset to the internal host system drive is strictly blocked by automated storage safety guards (`TestStorageSafetyGuard`). This prevents excessive NAND wear, OS swap exhaustion, and disk space depletion.
- **External Storage Workflow**: All acquisition, protocol parsing, and scoring tools are fully implemented and verified. When dedicated external storage media ($\ge 50\text{ GB}$) is mounted, set `$ASVSPOOF_DATA_ROOT` to point to the external volume:
  ```bash
  export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalStorageVolume>/asvspoof2019"
  python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"
  python scripts/generate_scores.py --data-dir "$ASVSPOOF_DATA_ROOT" --split dev
  python scripts/evaluate_scores.py --scores scores/cm_scores_dev.txt \
      --protocol "$ASVSPOOF_DATA_ROOT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
  ```

---

## 6. Security, Privacy & Research Limitations

1. **Zero Raw Audio Retention**: Under ADR-002, uploaded audio files and microphone recordings reside strictly in volatile RAM buffers during inference and are deallocated immediately upon score extraction. Zero raw audio waveforms are stored on disk or in the database.
2. **Persistent Metadata Only**: `data/history.db` records only high-level forensic artifacts: SHA-256 content verification hash, filename, duration, sample rate, mathematical logits ($S_{\text{bonafide}}$, $S_{\text{spoof}}$), decision margin, verdict, and timestamp.
3. **Defense-in-Depth HTTP Headers**: The web service enforces strict security headers:
   - `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'` (strictly zero inline `<script>` tags).
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: SAMEORIGIN`
   - `Referrer-Policy: strict-origin-when-cross-origin`
4. **Token Security & Revocation**: Shareable dossier tokens are high-entropy 256-bit URL-safe strings (`secrets.token_urlsafe(32)`). They carry a 7-day default lifespan, support instant one-click revocation, and resist enumeration via constant-structure HTTP 404 responses.
5. **Research & Legal Disclaimer**:
   - The SHA-256 digest provides byte-level content verification against tampering; it does **not** constitute an immutable legal chain of custody or cryptographically certified evidentiary seal.
   - The model operates on a global 4.0375-second window (64,600 samples) and cannot localize sub-second phoneme splicing or localized temporal edits.
   - Zero-shot transfer to heavily compressed social media voice notes (e.g. WhatsApp, cellular telephony) exhibits score drift, requiring front-end DSP mitigation.

---

## 7. Installation & Quick Start

### 7.1 Environment Setup

Requirements: Python 3.10+ (macOS Apple Silicon or Linux x86_64).

```bash
# Clone the repository:
git clone https://github.com/Aditya-A-Naikar/prj-55-deepfake-audio-detection.git
cd prj-55-deepfake-audio-detection

# Create and activate Python virtual environment:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

*Model Weights Note:* Pretrained AASIST-L weights are verified and located at `models/weights/AASIST-L.pth` (426,428 bytes, SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).

### 7.2 Launching the Web Platform

```bash
# Start the web platform on port 8000:
python scripts/run_demo_app.py --port 8000

# Open in browser:
# http://127.0.0.1:8000
```

### 7.3 CLI Verification Tools

```bash
# Baseline forward pass smoke test (synthetic tone fixture):
python scripts/smoke_test_inference.py --synthetic

# Baseline forward pass on an audio file:
python scripts/smoke_test_inference.py --audio path/to/sample.wav

# Run channel degradation experiment suite (telephony, AWGN noise):
python scripts/run_degradation_experiment.py --synthetic

# Run empirical latency and throughput benchmark:
python scripts/benchmark_performance.py

# Inspect threshold calibration CLI:
python scripts/calibrate_threshold.py --help
```

---

## 8. Automated Quality Gates

All quality gates pass with zero defects (Verified: October 31, 2026):

```bash
# 1. Run Python code linter (0 errors across codebase):
ruff check .

# 2. Run static type checker (0 issues across 40 source files):
mypy src tests scripts

# 3. Run complete automated test suite (265 passed, 2 skipped, 0 failed):
pytest -v

# 4. Run live system integration & security harness (17/17 passed):
python scripts/verify_live_system.py
```

---

## 9. Repository Layout

```text
prj-55-deepfake-audio-detection/
├── README.md                          ← Project documentation & system specification
├── pyproject.toml                     ← Build configuration, ruff, mypy, pytest settings
├── requirements.txt                   ← Production runtime dependencies
├── requirements-dev.txt               ← Test runner, linter, static typing tools
├── configs/                           ← Degradation & dataset YAML configurations
├── models/
│   └── weights/AASIST-L.pth          ← Pretrained weights (SHA-256 verified)
├── src/prj55/
│   ├── adaptation.py                  ← Front-end DSP mitigation (spectral subtraction, bandpass)
│   ├── audio.py                       ← 16 kHz audio loader, downmix, 64.6k framing
│   ├── calibration.py                 ← Automated threshold calibration module (EER, Target FAR)
│   ├── degradation.py                 ← Telephony, AWGN noise, Opus codec probe
│   ├── metrics.py                     ← ASVspoof EER, min t-DCF, ROC-AUC calculation
│   ├── models/aasist.py               ← AASIST-L architecture (SincNet + HS-GAL attention)
│   ├── protocol.py                    ← ASVspoof 2019 protocol parsing & validation
│   └── web/
│       ├── app.py                     ← FastAPI web service & typed REST endpoints
│       ├── db.py                      ← ACID SQLite persistent history & bearer sharing
│       ├── service.py                 ← ForensicsService pipeline & timeline generators
│       └── static/                    ← Redesigned multi-route frontend
│           ├── index.html             ← / Route (Landing Page & 3D Wavefield)
│           ├── analyze.html           ← /analyze Route (Forensic Analysis Workspace)
│           ├── robustness.html        ← /robustness Route (Acoustic Robustness Lab)
│           ├── history.html           ← /history Route (Persistent Evidence History)
│           ├── methodology.html       ← /methodology Route (Research Methodology)
│           ├── shared.html            ← /shared/{token} Route (Case Dossier)
│           ├── 404.html               ← Defensive Anti-Enumeration 404 Page
│           ├── manifest.webmanifest   ← PWA Web App Manifest
│           ├── sw.js                  ← Service Worker Cache Strategy
│           ├── css/style.css          ← Unified Dark Forensic Design System
│           └── js/                    ← Modular client logic (app, landing, robustness, history, etc.)
├── scripts/
│   ├── run_demo_app.py                ← Local web platform launcher
│   ├── smoke_test_inference.py        ← AASIST-L forward pass verification CLI
│   ├── run_degradation_experiment.py  ← Channel robustness experiment runner CLI
│   ├── benchmark_performance.py       ← Hardware latency & concurrency benchmark CLI
│   ├── calibrate_threshold.py         ← Decision threshold calibration CLI
│   ├── verify_live_system.py          ← Live system integration & security harness
│   ├── acquire_asvspoof2019.py        ← Dataset acquisition with storage safety guards
│   ├── generate_scores.py             ← ASVspoof evaluation trial score generator CLI
│   └── evaluate_scores.py             ← ASVspoof baseline metric evaluator CLI
├── tests/                             ← 267 collected test cases (265 passed, 2 skipped)
│   ├── test_acquire.py                ← Storage safety guard & zip-slip tests
│   ├── test_adaptation.py             ← Spectral subtraction & bandpass mitigation tests
│   ├── test_baseline_inference.py     ← Audio preprocessing & AASIST-L inference tests
│   ├── test_calibration.py            ← Threshold calibration & partition hygiene tests
│   ├── test_degradation.py            ← Telephony & acoustic noise degradation tests
│   ├── test_figures.py                ← Research figure generator contract tests
│   ├── test_metrics.py                ← EER, min t-DCF, and ROC-AUC calculation tests
│   ├── test_microphone_audio.py       ← Microphone PCM decoding & container rejection tests
│   ├── test_protocol_validation.py    ← ASVspoof 2019 protocol parsing tests
│   ├── test_ui_contracts.py           ← 83 automated DOM, CSS & navigation contract tests
│   └── test_web_api.py                ← REST API, security headers & SQLite restart tests
└── docs/reports/ui-review/            ← 13 verified browser screenshot PNGs & contact sheet
```

---

## 10. Primary Academic References

```text
[1] J. Jung, H. S. Heo, H. Tak, H. J. Shim, J. S. Chung, B. J. Lee, H. J. Yu, and N. Evans,
    "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks,"
    Proceedings of IEEE ICASSP, 2022, pp. 6367–6371, doi: 10.1109/ICASSP43922.2022.9747766.

[2] M. Todisco, X. Wang, V. Vestman, M. Sahidullah, H. Delgado, A. Nautsch, J. Yamagishi,
    N. Evans, T. Kinnunen, and K. A. Lee, "ASVspoof 2019: Future Horizons in Spoofed and
    Fake Audio Detection," Proceedings of Interspeech 2019, pp. 1008–1012,
    doi: 10.21437/Interspeech.2019-2249.

[3] M. Ravanelli and Y. Bengio, "Speaker Recognition from Raw Waveform with SincNet,"
    IEEE Spoken Language Technology Workshop (SLT), 2018, pp. 1021–1028,
    doi: 10.1109/SLT.2018.8639585.

[4] S. F. Boll, "Suppression of Acoustic Noise in Speech Using Spectral Subtraction,"
    IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 27, no. 2,
    pp. 113–120, Apr. 1979, doi: 10.1109/TASSP.1979.1163209.
```

---

*PRJ_55 Audio Authenticity Lab — Final Year Computer Science & Engineering Undergraduate Research Project*
