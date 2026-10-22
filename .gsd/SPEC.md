---
status: FINALIZED
created: 2026-10-16
finalized: 2026-10-22
version: 1.0.0
project_id: PRJ_55
---

# SPEC.md — Project Specification: Deepfake Audio Detection (PRJ_55)

> **Status**: `FINALIZED`  
> **Project ID**: PRJ_55  
> **Project Name**: Deepfake Audio Detection  
> **Working Title**: "Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation"  
> **Academic Context**: B.Tech CSE Final Year Main Project (30-day timeline)  
> **Hardware Target**: Apple Silicon M2 (MacBook Air/Pro, 8 GB Unified RAM, macOS 26.x)  

---

## 1. Vision & Core Objectives

PRJ_55 investigates and implements an on-device, compute-efficient countermeasure against synthetic speech and voice cloning attacks. Unlike commercial cloud-based detectors that demand heavy GPU clusters and often degrade when subjected to real-world audio channels (telephony, WhatsApp/VoIP codecs, acoustic noise), this project establishes an ultra-lightweight detector (**AASIST-L**, ~85,300 parameters, 416 KB) that executes in real-time on consumer laptops and studies its resilience under controlled channel degradation.

---

## 2. Project State & Evidence Audit

### 2.1 Verified Completed Software Work
- **Repository Scaffold & Architecture**: Clean modular layout (`src/prj55/`, `tests/`, `scripts/`, `configs/`, `docs/`, `reports/`).
- **Reproducible Python Environment**: Python 3.10.0 ARM64 in `.venv/` with tiered dependency specifications (`requirements.txt`, `requirements-dev.txt`, `requirements-research.txt`).
- **Baseline Neural Architecture**: Official **AASIST-L** (Jung et al., ICASSP 2022) implemented in `src/prj55/models/aasist.py` using SincNet bandpass filterbanks, 6-block residual encoder, heterogeneous spectro-temporal graph attention, and top-k graph pooling.
- **Pretrained Checkpoint Provenance**: Official Clova AI / NAVER Corp weights (`AASIST-L.pth`, 426,428 bytes, SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`) verified and excluded from git tracking.
- **Audio Preprocessing Engine**: 16 kHz mono enforcement, channel downmixing, and repeat/truncate framing to 64,600 samples (~4.0375s) implemented in `src/prj55/audio.py`.
- **Inference Smoke Test**: Automated forward-pass execution script `scripts/smoke_test_inference.py` operational on CPU (latency: ~279 ms) and Apple Metal MPS (~3.5s initial warmup), producing finite logits `[1, 2]`.
- **Protocol Validator Engine**: ASVspoof 2019 LA trial parser and validator (`src/prj55/protocol.py`, `scripts/validate_protocol.py`) with support for `$ASVSPOOF_DATA_ROOT` and multi-split discovery.
- **Automated Test Suite**: **48 passing unit and integration tests** with **95% code coverage** (`pytest --cov=src`).
- **Code Quality Gates**: Clean static linting (`ruff`) and strict typing (`mypy`) across 11 source files.

### 2.2 Implemented but Incompletely Validated Functionality
- **Full Corpus Protocol Validation**: Multi-split protocol validator is implemented and verified using synthetic fixtures, but has not yet scanned the full 121,420 physical audio files because corpus acquisition is blocked by local storage constraints.
- **Benchmark Evaluation Metrics**: Metric calculation scripts for Equal Error Rate (EER) and minimum tandem Detection Cost Function (min-tDCF) await physical corpus trial scoring.

### 2.3 Current Physical Blocker (Hardware Constraint)
- **Constraint**: Internal storage partition (`/dev/disk3s5`) has only **~2.9 GiB free space** (99% capacity utilized).
- **Dataset Footprint**: The official ASVspoof 2019 Logical Access archive (`LA.zip`) is **7.64 GB (7.12 GiB) compressed**, unpacking to **~25–30 GiB** across 121,420 FLAC audio files.
- **Policy Enforcement**: Downloading the full corpus to the internal drive is strictly prohibited to prevent operating system failure. An external SSD/USB-C drive with $\ge 50\text{ GiB}$ free space mapped to `ASVSPOOF_DATA_ROOT` is required.

---

## 3. Goals & Non-Goals

### 3.1 Primary Goals
1. **Benchmark Baseline Verification**: Evaluate pretrained AASIST-L on the official ASVspoof 2019 LA development and evaluation partitions to verify published baseline performance (EER ~1.00% on Dev).
2. **Channel Degradation Simulation**: Develop a controlled audio perturbation pipeline simulating:
   - Telephony bandpass filtering (G.711 / 8 kHz downsampling).
   - Social media / VoIP lossy compression (Opus at 16–32 kbps, AAC, MP3).
   - Background acoustic noise and additive white Gaussian noise (AWGN at varying SNRs: 0 dB, 10 dB, 20 dB).
3. **Robustness Characterization**: Quantify performance degradation ($\Delta \text{EER}$, $\Delta \text{t-DCF}$) under each channel perturbation to establish vulnerability boundaries.
4. **Targeted Adaptation / Mitigation**: Implement lightweight data augmentation or front-end filtering to recover detection performance under degraded channels without ballooning parameter count.
5. **Interactive Local Demonstration**: Provide a lightweight local web interface (or CLI demo) where users can inspect audio waveforms, apply real-time channel degradation, and observe detector confidence scores.

### 3.2 Non-Goals (Explicitly Out of Scope)
- Training massive transformer/foundation models from scratch (e.g. Whisper-based or WavLM multi-gigabyte models).
- Large-scale multi-GPU distributed cluster training.
- Commercial production cloud API hosting or multi-tenant database infrastructure.
- Bypassing academic dataset licensing or downloading unverified third-party mirrors.
- Developing deepfake video or facial manipulation detectors (audio speech focus only).

---

## 4. Technical Requirements & Specifications

| ID | Requirement | Priority | Target / Metric |
|---|---|---|---|
| **REQ-01** | Architecture Parameter Limit | Must-Have | $< 100,000$ parameters (AASIST-L: 85,306) |
| **REQ-02** | CPU Inference Latency | Must-Have | $< 500\text{ ms}$ on Apple M2 (Currently ~279 ms) |
| **REQ-03** | Audio Preprocessing | Must-Have | 16 kHz sample rate, mono float32, 64,600 samples |
| **REQ-04** | Protocol Split Integrity | Must-Have | Disjoint speakers across Train (20), Dev (10), Eval (48); zero eval leakage |
| **REQ-05** | Degradation Simulator | Must-Have | Telephony (8 kHz G.711), Opus/AAC codec, additive noise |
| **REQ-06** | Official Metric Scoring | Must-Have | ASVspoof EER and min-tDCF implementations |
| **REQ-07** | Automated Test Coverage | Must-Have | $\ge 90\%$ test coverage (Currently 95%) |
| **REQ-08** | External Storage Support | Must-Have | Portable path configuration via `$ASVSPOOF_DATA_ROOT` |
| **REQ-09** | Interactive Demo Interface | Should-Have | Lightweight local UI (FastAPI/Gradio/Streamlit) running on CPU |
| **REQ-10** | Robustness Fine-Tuning | Should-Have | Scoped adaptation recovering $\ge 30\%$ degraded EER loss |

---

## 5. Success Criteria

- [x] Week 1 foundation verified (scaffold, virtual environment, AASIST-L architecture, audio pipeline, 48 tests passing).
- [ ] External SSD connected and official ASVspoof 2019 LA corpus acquired and MD5 checksum verified.
- [ ] Official development and evaluation split protocols validated against 121,420 physical FLAC files.
- [ ] Baseline EER and min-tDCF reproduced on the development partition.
- [ ] Channel degradation benchmark executed and documented in a comparative performance matrix.
- [ ] Working local demonstration app operational without cloud dependencies.
- [ ] Comprehensive academic project report and defense presentation finalized.

---

*Last updated: 2026-10-22*
