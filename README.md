# PRJ_55 — Deepfake Audio Detection & Forensics Platform

**Working Research Title:**  
> *Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation*

**Academic Context:** B.Tech CSE Final Year Main Project  
**Target Hardware:** Apple Silicon M2 (MacBook Air/Pro, macOS 26.x, ARM64)  
**Status:** **Phase 4 Complete — Production-Quality Audio Forensics Platform Operational**

---

## 1. Executive Overview

PRJ_55 is an on-device, compute-efficient audio authenticity and deepfake countermeasure platform. While heavy foundation models require GPU clusters and often degrade when audio passes through cellular telephony or VoIP compression, this project establishes:

1. **Ultra-Lightweight Neural Architecture**: Official **AASIST-L** (Jung et al., ICASSP 2022) with SincNet learnable bandpass filterbanks and spectro-temporal heterogeneous graph attention (85,306 parameters, ~420 KB footprint).
2. **Sub-300ms Real-Time Inference**: Operates locally on Apple Silicon (M2 CPU: ~181 ms pipeline / ~163 ms model forward; Metal Performance Shaders MPS: ~41 ms pipeline / ~36 ms model forward; Real-Time Factor < 0.01x on 4-second audio).
3. **Channel Degradation Simulator**: Evaluates model vulnerability under restricted-bandwidth telephony (8 kHz polyphase resample + anti-aliasing), additive white Gaussian noise (0–20 dB SNR), and lossy Opus compression.
4. **Front-End Scoped Mitigation**: Lightweight, non-parametric spectral subtraction and adaptive bandpass normalization recovering ~35% of noise-induced score drift without increasing model parameters.
5. **Interactive Audio Forensics Web Platform**: Refined local web application featuring an interactive 3D harmonic signal sculpture, Web Audio waveform scrubbing, FFT spectrogram heatmap, side-by-side robustness comparisons, and session audit history.

---

## 2. Interactive Web Platform

The local demonstration platform provides an end-to-end forensic instrument for university defense and research demonstrations:

```bash
# Launch the platform locally:
.venv/bin/python scripts/run_demo_app.py

# Access the interface in your browser:
# http://127.0.0.1:8000
```

### Key Capabilities
- **Overview & 3D Wavefield**: Interactive Canvas-based 3D harmonic particle ribbon responding to cursor rotation, clicks, and real-time inference pulses.
- **Analysis Workspace**: Drag-and-drop WAV/FLAC upload or one-click testing using calibrated sample fixtures (`Authentic Vocal Formants`, `Synthetic Vocoder Tone`, `440 Hz Reference`).
- **Forensic Evidence & Spectrogram**: Decision verdict pill (`BONAFIDE` in electric blue vs `SPOOF` in neon red), countermeasure logit score meter relative to decision threshold ($\theta = 0.00$), and 0–8000 Hz FFT spectrogram.
- **Robustness Laboratory**: Configurable channel degradation with SNR slider (0–20 dB), front-end mitigation filter toggle, side-by-side $\Delta S_{\text{cm}}$ drift analysis, and playback of degraded audio.
- **Research Benchmarks & Governance**: Official ASVspoof 2019 LA targets, quality gate audit, and explicit storage policy documentation (ADR-002).
- **Session Audit Trail**: Local in-memory experiment history with one-click JSON export.

---

## 3. Quick Start & CLI Workflows

### 3.1 Environment Activation
```bash
# Activate existing Python 3.10 virtual environment:
source .venv/bin/activate
```

### 3.2 Single-Sample Inference Smoke Test
```bash
# Synthetic test tone:
python scripts/smoke_test_inference.py --synthetic

# Real speech recording:
python scripts/smoke_test_inference.py --audio /path/to/recording.flac
```

### 3.3 Channel Degradation Experiment Suite
```bash
# Run degradation suite (Telephony 8 kHz, Additive Noise at 0, 10, 20 dB SNR):
python scripts/run_degradation_experiment.py --synthetic
```

### 3.4 Automated Quality Gates
```bash
# Run full automated test suite (159 passed, 90% statement coverage):
pytest -v

# Run linting and static typing checks:
ruff check .
mypy src tests scripts

# Run all GSD validator checks:
bash scripts/validate-all.sh
```

---

## 4. Architecture & Technical Specifications

| Parameter | Specification | Compliance / Measurement |
|---|---|---|
| **Neural Backbone** | AASIST-L (Jung et al., ICASSP 2022) | Verified (`src/prj55/models/aasist.py`) |
| **Parameter Count** | 85,306 parameters ($< 100\text{k}$ budget) | **PASS** (REQ-01) |
| **Model Footprint** | 426 KB (`models/weights/AASIST-L.pth`) | SHA-256: `814331d088032bb...` verified |
| **Input Specifications** | 16,000 Hz, Mono float32, 64,600 samples (~4.0375s) | **PASS** (REQ-03) |
| **Inference Latency** | Apple M2 CPU: ~181 ms • MPS Metal: ~41 ms pipeline (~36 ms forward) ($< 500\text{ ms}$ budget) | **PASS** (REQ-02) |
| **Real-Time Factor** | $\text{RTF} \approx 0.009\text{x}$ (Metal MPS) / $0.045\text{x}$ (Apple M2 CPU) | Real-time capable |
| **Channel Simulator** | Telephony (8 kHz resample_poly), AWGN noise, Opus | **PASS** (REQ-05) |
| **Front-End Mitigation** | Spectral Subtraction + Bandpass Normalization | **PASS** (REQ-10) |
| **Demonstration Interface** | FastAPI REST API + Vanilla CSS/JS 3D Workspace | **PASS** (REQ-09) |
| **Automated Test Suite** | 159 tests passing, 2 skipped (conditional Opus) | **PASS** (REQ-07, 90% coverage) |

---

## 5. Storage Governance & Physical Corpus Policy (ADR-002)

- **Hardware Reality**: The target Apple Silicon host currently has 61 GiB free space on its internal SSD partition (`/dev/disk3s5`), but **0 external USB drives are attached**.
- **Corpus Footprint**: The official ASVspoof 2019 Logical Access archive (`LA.zip`) is **7.64 GB compressed**, expanding to **~25–30 GB** across 121,420 FLAC audio files.
- **Integrity Enforcement**: Per **ADR-002**, downloading the corpus to the internal drive is strictly prohibited to preserve host disk longevity and operating system swap margins.
- **Physical Evaluation Readiness**: All dataset ingestion (`scripts/acquire_asvspoof2019.py`), protocol verification (`scripts/validate_protocol.py`), and baseline evaluation tools (`scripts/generate_scores.py`, `scripts/evaluate_scores.py`) are fully implemented and verified with synthetic fixtures. When an external SSD ($\ge 50\text{ GiB}$) is attached:
  ```bash
  export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalVolume>/asvspoof2019"
  python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"
  python scripts/generate_scores.py --data-dir "$ASVSPOOF_DATA_ROOT" --split dev
  python scripts/evaluate_scores.py --scores scores/cm_scores_dev.txt --protocol "$ASVSPOOF_DATA_ROOT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
  ```

---

## 6. Repository Layout

```text
prj-55-deepfake-audio-detection/
├── README.md                          ← Project documentation & quick start
├── pyproject.toml                     ← Build configuration, ruff, mypy, pytest settings
├── requirements.txt                   ← Runtime dependencies (fastapi, uvicorn, numpy, soundfile)
├── requirements-dev.txt               ← Testing & linting tools (pytest, ruff, mypy, httpx)
├── requirements-research.txt          ← PyTorch runtime specification
├── configs/                           ← Degradation & dataset YAML configs
├── models/
│   └── weights/AASIST-L.pth          ← Pretrained weights (SHA-256 verified)
├── src/prj55/
│   ├── adaptation.py                  ← Front-end mitigation (spectral subtraction, bandpass)
│   ├── audio.py                       ← 16 kHz audio loader, downmix, 64.6k framing
│   ├── degradation.py                 ← Telephony, AWGN noise, Opus codec probe
│   ├── metrics.py                     ← ASVspoof EER, min-tDCF, ROC-AUC, score alignment
│   ├── models/aasist.py               ← AASIST-L SincNet + Heterogeneous Graph Attention
│   ├── protocol.py                    ← ASVspoof 2019 protocol parsing & validation
│   └── web/
│       ├── app.py                     ← FastAPI web service & typed endpoints
│       ├── service.py                 ← Forensics business logic, features, audio streaming
│       └── static/                    ← Frontend (index.html, style.css, sculpture.js, app.js)
├── scripts/
│   ├── run_demo_app.py                ← Local web platform launcher
│   ├── smoke_test_inference.py        ← AASIST-L forward pass verification CLI
│   ├── run_degradation_experiment.py  ← Channel robustness experiment runner CLI
│   ├── generate_scores.py             ← ASVspoof trial score generator CLI
│   ├── evaluate_scores.py             ← ASVspoof baseline metric evaluator CLI
│   └── acquire_asvspoof2019.py        ← Autonomous dataset acquisition tool
├── tests/                             ← 159 automated unit and integration tests
└── reports/                           ← Milestone reports (Phase 1, 2, 3, 4)
```

---

*PRJ_55 Audio Forensics Lab — B.Tech CSE Main Project (Apple Silicon M2)*
