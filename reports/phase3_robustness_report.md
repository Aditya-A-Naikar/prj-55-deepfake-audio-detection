# Phase 3 Completion Report: Channel Degradation Simulator and Robustness Evaluation Framework

**Project ID:** PRJ_55  
**Project Name:** Deepfake Audio Detection  
**Working Research Title:** Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation  
**Date:** 2026-10-22  
**Author:** Lead Audio ML & Research Engineer  
**Status:** Software Implementation & Verification Complete (`VERIFIED_SOFTWARE_ONLY`); Physical Benchmark Evaluation Pending External Storage (`PENDING_DATASET_ACQUISITION`)

---

## 1. Executive Summary & Purpose

The objective of Phase 3 is to implement and validate an end-to-end, reproducible framework to investigate how common real-world channel degradation affects the predictions of the AASIST-L deepfake speech detection model.

In real-world deployment (telephony fraud detection, social media monitoring, messaging platform security), audio signals rarely arrive in pristine 16 kHz uncompressed FLAC format. Instead, they undergo bandwidth limiting (e.g., PSTN/cellular networks), lossy compression (e.g., Opus, AAC in VoIP and messaging apps), and acoustic background noise.

This phase implemented:
1. A modular, reproducible signal processing degradation pipeline (`src/prj55/degradation.py`).
2. An official ASVspoof evaluation and robustness comparison module (`src/prj55/metrics.py`) implementing EER, min-tDCF, and score alignment with strict data-availability guards.
3. An experiment runner CLI and YAML configuration (`scripts/run_degradation_experiment.py`, `configs/degradation_config.yaml`).
4. A 63-test automated test suite for degradation transforms, AASIST-L compatibility, and evaluation metrics (`tests/test_degradation.py`, `tests/test_metrics.py`).

**Critical Integrity Note:** The official ASVspoof 2019 LA dataset is currently unmounted due to internal storage constraints (requiring an external SSD of $\ge 50\text{ GiB}$). Therefore, all software verification has been conducted using deterministic synthetic test fixtures. No synthetic test score is reported as a benchmark detection result.

---

## 2. Transformations Implemented

The degradation pipeline in `src/prj55/degradation.py` adheres to a strict contract:
- **Input:** 1-D `np.ndarray` of float32, sampled at 16,000 Hz.
- **Output:** 1-D `np.ndarray` of float32, sampled at 16,000 Hz, with identical duration/length preserved, fully verified for finiteness (`np.isfinite`) and absence of NaN/Inf.
- **Source Protection:** Source waveforms and files are never mutated or overwritten in-place.

### 2.1 Telephony Simulation (`apply_telephony`)
- **Mechanism:** Polyphase resampling (`scipy.signal.resample_poly`) to an intermediate 8 kHz sample rate followed by upsampling back to 16 kHz.
- **Anti-Aliasing:** Polyphase filtering automatically applies an anti-aliasing low-pass filter with cutoff at Nyquist (4 kHz) prior to downsampling and post-interpolation.
- **Scope & Modeling Bounds:** Simulates narrow-band telephony frequency restriction (nominal passband ~300–3,400 Hz). It serves as a controlled frequency-truncation benchmark rather than simulating every acoustic or packet-loss artifact of cellular/VoIP networks.

### 2.2 Additive Acoustic Noise (`apply_noise`)
- **Mechanism:** Generates zero-mean Gaussian white noise scaled to achieve an exact target Signal-to-Noise Ratio (SNR in dB):
  $$\text{Noise Power} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}} / 10}}$$
- **Reproducibility:** Accepts an explicit `seed` parameter (default: 42) creating an isolated `np.random.default_rng(seed)`.
- **Validation:** Handles SNR ranges from severe degradation (0 dB) to moderate (10 dB, 20 dB). Validates that silent inputs or non-finite inputs raise explicit errors.
- **Distinction:** Documents clearly that synthetic white Gaussian noise differs from structured non-stationary babble or environmental acoustic noise.

### 2.3 Lossy Opus Compression (`apply_opus`)
- **Mechanism:** Implements an encode-decode round-trip via subprocess invoking `ffmpeg` with `-c:a libopus` at configurable bitrates (e.g., 16 kbps, 24 kbps, 32 kbps) and application profiles (`voip`, `audio`, `lowdelay`).
- **Conditional Availability & Guarding:** Probes system environment at import time (`codec_probe_report()`, `opus_available()`). If `ffmpeg` or `libopus` is absent, the module raises `DegradationUnavailableError` with clear instructions (`brew install ffmpeg`) while keeping telephony and noise transforms 100% operational.

---

## 3. Codec Availability and Environment Audit

At the time of this execution on the target Apple Silicon host:

| Component | Status | Details |
|---|---|---|
| Python Version | 3.10.0 (ARM64) | Active in `.venv/` |
| PyTorch | 2.1.2 (Apple Silicon) | MPS available; CPU used for single-sample inference |
| SciPy | 1.15.3 | Polyphase filtering (`scipy.signal.resample_poly`) operational |
| SoundFile | 0.13.1 | libsndfile backend operational |
| FFmpeg Binary | `not found` | Not installed on host PATH |
| Opus Codec | `unavailable` | `DegradationUnavailableError` triggered as designed |

**Resolution Path for Opus:** When lossy codec evaluation is needed, install FFmpeg with libopus:
```bash
brew install ffmpeg
```
Once installed, `opus_available()` dynamically returns `True` with zero code changes required.

---

## 4. ASVspoof Metrics & Score Alignment Framework

The evaluation module `src/prj55/metrics.py` implements the official ASVspoof 2019 scoring logic:

1. **Equal Error Rate (EER):**
   - Implemented in `compute_eer(cm_scores, labels)`.
   - Computes False Alarm Rate (FAR) and False Rejection Rate (FRR) across candidate thresholds.
   - Identifies the operating point minimizing $|FAR - FRR|$ and reports the threshold and error rate.
2. **Minimum tandem Detection Cost Function (min-tDCF):**
   - Implemented in `compute_min_tdcf(cm_scores, labels, cost_model)`.
   - Uses official ASVspoof 2019 LA parameters: $P_{\text{spoof}} = 0.05, C_{\text{miss,cm}} = 1.0, C_{\text{fa,cm}} = 10.0$.
   - Normalizes by trivial system cost.
3. **Score Alignment Tool:**
   - Implemented in `align_score_files(clean_scores, degraded_scores)`.
   - Reconciles trial dictionaries by utterance ID.
   - Detects and logs missing records and unmatched trials.
4. **Robustness Metric Comparator:**
   - Implemented in `compare_robustness(clean_scores, degraded_scores, labels)`.
   - Computes $\Delta\text{EER}_{\text{abs}} = \text{EER}_{\text{deg}} - \text{EER}_{\text{clean}}$ and $\Delta\text{EER}_{\text{rel}} = \frac{\Delta\text{EER}_{\text{abs}}}{\text{EER}_{\text{clean}}} \times 100\%$.
5. **Data Unavailability Guards:**
   - If labels are missing, empty, or single-class, `MetricUnavailableError` is raised with a descriptive message rather than producing misleading metrics.

---

## 5. Model Integration & Software Verification Results

The experiment runner CLI `scripts/run_degradation_experiment.py` was executed with the verified AASIST-L checkpoint (`models/weights/AASIST-L.pth`, SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).

### 5.1 Verification Run Output
```text
========================================================================
  PRJ_55 Phase 3 — Channel Degradation Experiment Summary
========================================================================
  Source:   SYNTHETIC_SIGNAL — software verification only, not benchmark
  Time:     2026-10-22T15:45:47.158504+00:00
  FFmpeg/Opus: False
  Baseline bonafide logit: -0.3512 (444.6 ms)
------------------------------------------------------------------------
  Condition                            BF logit   Δ logit Status      
------------------------------------------------------------------------
  telephony_8khz                        -0.3179   +0.0333 OK
  noise_snr_20db                        -0.3515   -0.0003 OK
  noise_snr_10db                        -0.3132   +0.0381 OK
  noise_snr_0db                         -0.2840   +0.0672 OK
========================================================================
```

### 5.2 Observations
- **Inference Latency:** Baseline CPU forward pass took 444.6 ms (warmup); degraded conditions executed in 231.7–308.0 ms on Apple Silicon M2, well within real-time budgets for 4-second audio (Real-Time Factor $\approx 0.06$).
- **Software Pipeline Integrity:** All degraded waveforms passed strictly into the AASIST-L architecture without shape mismatch, numeric instability, or dimension errors.
- **Structured Artifacts:** Results were output to structured JSON with full transformation parameters, timestamp, latency, and status codes.

---

## 6. Automated Quality Assurance & Test Verification

All project quality gates were executed and passed cleanly:

### 6.1 Unit & Integration Test Suite (`pytest`)
- **Total Tests:** 138 (136 passed, 2 skipped conditionally due to missing system `ffmpeg`).
- **Execution Time:** ~11.6 seconds.
- **Coverage Breakdown:**
  - `tests/test_degradation.py`: 43 tests covering signal resampling, anti-aliasing, SNR scaling, noise reproducibility, exception paths, and AASIST-L tensor compatibility (41 passed, 2 skipped).
  - `tests/test_metrics.py`: 30 tests covering EER mathematical accuracy, min-tDCF, score alignment, ROC-AUC, operating-point metrics, and unavailable data guards.
  - `tests/test_protocol_validation.py`: 17 tests verifying ASVspoof protocol parsing, validation reports, and dataset root discovery.
  - `tests/test_project_scaffold.py`: 14 tests verifying project architecture and gitignore rules.
  - `tests/test_baseline_inference.py`: 12 tests verifying AASIST-L weights provenance, architecture, and forward pass.
  - `tests/test_evaluate_scores.py`: 7 tests verifying score loading, alignment, and evaluation logic.
  - `tests/test_acquire.py`: 6 tests verifying dataset acquisition logic and storage pre-flight guards.
  - `tests/test_environment.py`: 5 tests verifying Apple Silicon runtime and dependencies.
  - `tests/test_generate_scores.py`: 4 tests verifying score generation CLI and synthetic streaming.

### 6.2 Code Quality & Static Analysis
- **`pip check`:** No broken requirements found.
- **`ruff check .`:** All checks passed (zero lint errors).
- **`mypy src tests scripts`:** Success: no issues found in 22 source files.

---

## 7. Reproduction Instructions

To reproduce the Phase 3 verification on any workstation:

1. **Run full automated test suite:**
   ```bash
   .venv/bin/pytest -v
   ```

2. **Run code quality checks:**
   ```bash
   .venv/bin/python -m pip check
   .venv/bin/ruff check .
   .venv/bin/mypy src tests scripts
   ```

3. **Run channel degradation experiment on synthetic signal:**
   ```bash
   .venv/bin/python scripts/run_degradation_experiment.py --synthetic
   ```

4. **Run channel degradation experiment on real audio (when available):**
   ```bash
   .venv/bin/python scripts/run_degradation_experiment.py \
       --audio /path/to/speech.flac \
       --checkpoint models/weights/AASIST-L.pth \
       --output-dir reports/phase3_experiment_results
   ```

---

## 8. Remaining Limitations & Transition to Benchmark Evaluation

| Area | Current Status | Blocker / Requirement | Action Required |
|---|---|---|---|
| Channel Degradation Simulator | **Complete & Verified** | None | Operational |
| Metric Evaluator (EER / tDCF) | **Complete & Verified** | None | Operational |
| Opus Codec Support | **Software Implemented** | Host `ffmpeg` not installed | Optional: `brew install ffmpeg` |
| ASVspoof 2019 LA Audio | **Unmounted** | Internal SSD capacity (< 3 GB free) | Mount $\ge 50\text{ GiB}$ external SSD |
| Benchmark Robustness Scoring | **Pending Data Acquisition** | Requires physical FLAC files and labels | Run benchmark runner once external SSD is mounted |

The software foundation for Phase 3 is robust, tested, and ready. Once external storage is connected, official corpus evaluation can proceed immediately without further code development.
