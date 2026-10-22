---
milestone: PRJ_55 B.Tech CSE Main Project
version: 1.0.0
updated: 2026-10-22T20:55:00+05:30
---

# Roadmap: PRJ_55 Deepfake Audio Detection

> **Current Phase:** Phase 2 — Dataset Acquisition, Protocol Validation & Baseline Benchmark  
> **Status:** ⏸️ Blocked by Storage (Software Ready; External SSD Pending)  
> **Working Title:** Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation  

---

## Must-Haves (from SPEC)

- [x] **REQ-01**: Ultra-lightweight model architecture ($< 100\text{k}$ parameters; AASIST-L has 85,306 parameters).
- [x] **REQ-02**: Real-time CPU inference on Apple M2 ($< 500\text{ ms}$; measured at ~279 ms).
- [x] **REQ-03**: Audio preprocessing pipeline (16 kHz mono float32, 64,600 samples fixed length).
- [x] **REQ-04**: ASVspoof 2019 protocol parsing and validation engine with zero split leakage.
- [x] **REQ-05**: Degradation simulator (Telephony 8 kHz, Opus codec with ffmpeg probe, additive noise at 0, 10, 20 dB SNR).
- [x] **REQ-07**: High automated test coverage ($\ge 90\%$; current: 136 passing tests, 2 skipped, 0 failed, 92% statement coverage).
- [x] **REQ-08**: Dynamic dataset path resolution via `$ASVSPOOF_DATA_ROOT`.
- [ ] **REQ-06**: Baseline scoring and replication of ASVspoof 2019 LA EER and min-tDCF on physical corpus.
- [ ] **REQ-09**: Interactive local demonstration app (FastAPI/Gradio/Streamlit) for real-time waveform degradation & detection.
- [ ] **REQ-10**: Scoped robustness adaptation recovering performance on degraded speech channels.

---

## Phases

### Phase 1: Foundation, Architecture & Inference Feasibility (Week 1)
**Status:** ✅ Complete  
**Objective:** Establish reproducible Python 3.10 ARM64 environment, implement modular AASIST-L architecture, verify official weights provenance, implement audio preprocessing, protocol parser, automated tests, and inference smoke tests.  
**Requirements:** REQ-01, REQ-02, REQ-03, REQ-04, REQ-07, REQ-08  

**Plans & Deliverables:**
- [x] **Plan 1.1**: Repository layout, environment setup (`requirements.txt`, `.venv/`), and dependency sanity checks.
- [x] **Plan 1.2**: Implement AASIST-L PyTorch neural architecture (`src/prj55/models/aasist.py`) with SincNet and Graph Attention.
- [x] **Plan 1.3**: Audio preprocessor (`src/prj55/audio.py`) enforcing 16 kHz, mono, 64,600-sample framing.
- [x] **Plan 1.4**: Protocol validation engine (`src/prj55/protocol.py`) parsing ASVspoof 2019 LA train/dev/eval splits.
- [x] **Plan 1.5**: Checkpoint provenance verification (`models/weights/AASIST-L.pth`, SHA-256 verified) and smoke test (`scripts/smoke_test_inference.py`).
- [x] **Plan 1.6**: Comprehensive unit and integration test suite (`tests/`, 48 tests passing, 95% coverage, ruff/mypy clean).

---

### Phase 2: Dataset Acquisition, Protocol Validation & Baseline Benchmark (Week 2)
**Status:** 🔄 Software Pipeline Complete / ⏸️ Physical Benchmark Blocked by External SSD  
**Objective:** Acquire the official ASVspoof 2019 LA corpus on an external SSD ($\ge 50\text{ GiB}$), verify MD5 checksum, validate protocols against 121,420 audio files, compute baseline EER and min-tDCF on the development partition.  
**Requirements:** REQ-04, REQ-06, REQ-08  
**Depends on:** Phase 1  

**Plans & Deliverables:**
- [x] **Plan 2.1**: Autonomous acquisition script (`scripts/acquire_asvspoof2019.py`) and protocol validation engine (`scripts/validate_protocol.py`, `src/prj55/protocol.py`) with storage safety checks and Zip Slip protection.
- [ ] **Plan 2.2**: Physical download of `LA.zip` (7.64 GB) onto external storage, verify MD5 (`30c98f11d8b2bc21f2c257bfd78bb5c5`), and extract ~15–20 GiB. *(Blocked by external media availability; internal disk protected per ADR-002)*.
- [ ] **Plan 2.3**: Execute official multi-split protocol validation across all 121,420 FLAC files using `scripts/validate_protocol.py`.
- [x] **Plan 2.4**: Implement standard ASVspoof scoring module (`src/prj55/metrics.py`) for EER, Operating Point metrics, ROC-AUC, and official min-tDCF tandem guards. Implement score generator (`scripts/generate_scores.py`) and score evaluator (`scripts/evaluate_scores.py`).
- [ ] **Plan 2.5**: Run baseline evaluation on physical ASVspoof 2019 LA development split to reproduce published benchmark metrics. *(Pending external SSD mounting)*.

---

### Phase 3: Controlled Channel Degradation & Robustness Profiling (Week 3)
**Status:** 🔄 Software Complete / ⏸️ Physical Benchmark Pending External SSD  
**Objective:** Implement modular audio degradation pipeline, apply controlled telephony, lossy codecs, and acoustic noise perturbations, and quantify degradation impact on detection EER and min-tDCF.  
**Requirements:** REQ-05, REQ-06  
**Depends on:** Phase 2  

**Plans & Deliverables:**
- [x] **Plan 3.1**: Implement degradation pipeline (`src/prj55/degradation.py`) supporting telephony bandpass (8 kHz resample_poly), lossy codecs (Opus via ffmpeg probe), and additive noise (AWGN at 0, 10, 20 dB SNR).
- [x] **Plan 3.2**: Unit test degradation transforms with deterministic synthetic signals and verify output format compliance (16 kHz, 64,600 samples). 63 degradation & metric tests passing.
- [x] **Plan 3.3**: Implement degradation configuration and CLI experiment runner (`configs/degradation_config.yaml`, `scripts/run_degradation_experiment.py`) with AASIST-L inference comparison and JSON reporting.
- [ ] **Plan 3.4**: Run batch evaluation of baseline AASIST-L on degraded physical ASVspoof 2019 LA development split and compute empirical robustness curves ($\Delta\text{EER}$ across perturbation types). *(Pending external SSD mounting)*.

---

### Phase 4: Scoped Adaptation, Application Demonstration & Project Defense (Week 4)
**Status:** ⬜ Not Started  
**Objective:** Implement scoped data augmentation or front-end filtering if justified, build an interactive local demonstration app, finalize project documentation, and prepare presentation defense materials.  
**Requirements:** REQ-09, REQ-10  
**Depends on:** Phase 3  

**Plans & Deliverables:**
- [ ] **Plan 4.1**: Evaluate lightweight mitigation strategies (e.g., front-end spectral normalization, augmentation fine-tuning of SincNet or Graph layers without exceeding 100k parameters).
- [ ] **Plan 4.2**: Implement lightweight local web demonstration application (FastAPI + Vanilla JS/CSS or Gradio) allowing users to upload audio, inject real-time degradation, and visualize AASIST-L spoof scores and attention weights.
- [ ] **Plan 4.3**: End-to-end integration testing and user acceptance validation of the demonstration app.
- [ ] **Plan 4.4**: Compile final academic project report, methodology charts, ablation tables, and slide deck for university defense.

---

## Progress Summary

| Phase | Description | Status | Plans Completed | Target Window |
|---|---|---|---|---|
| **Phase 1** | Foundation, Architecture & Feasibility | ✅ Complete | 6 / 6 | Week 1 |
| **Phase 2** | Dataset Acquisition & Baseline Benchmark | ⏸️ Blocked | 2 / 5 | Week 2 |
| **Phase 3** | Channel Degradation & Robustness | 🔄 Software Complete | 3 / 4 | Week 3 |
| **Phase 4** | Adaptation, Demo & Final Report | ⬜ Not Started | 0 / 4 | Week 4 |

---

## Timeline & Milestone Tracking

| Phase | Milestone Objective | Target Start | Target Completion | Actual Completion |
|---|---|---|---|---|
| **Phase 1** | Week 1 Foundation & Smoke Test | Day 1 | Day 7 | Day 7 (Verified) |
| **Phase 2** | Week 2 Corpus & Baseline Benchmark | Day 8 | Day 14 | Pending External SSD |
| **Phase 3** | Week 3 Degradation Experiments | Day 15 | Day 21 | Pending Phase 2 |
| **Phase 4** | Week 4 Local App & Project Defense | Day 22 | Day 30 | Pending Phase 3 |

---

## Status Icons Legend

- ⬜ **Not Started**: Planned work scheduled for subsequent phases.
- 🔄 **In Progress**: Work currently being actively implemented.
- ⏸️ **Paused / Blocked**: Temporarily halted due to external resource constraints (e.g. disk space).
- ✅ **Complete**: Fully implemented, verified with tests, and committed.
- ❌ **Failed / Deprecated**: Abandoned or replaced approach.
