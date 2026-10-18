# Baseline Inference Smoke Test Report — PRJ_55: Deepfake Audio Detection

**Date:** 2026-10-18  
**Auditor / Engineer:** Antigravity AI Assistant  
**Repository:** `prj-55-deepfake-audio-detection`  
**Milestone:** Week 1, Task 6 — Baseline Inference Smoke Test  
**Status:** PASS (Pipeline & Model Architecture Feasibility Verified)  

---

## 1. Executive Summary

This report documents the verification and smoke test of the primary deepfake audio detection candidate model, **AASIST-L** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention - Light). 

Inference execution feasibility was confirmed on the local Apple M2 development environment. Forward-pass execution with verified pretrained weights was successfully performed on CPU and Apple Silicon Metal Performance Shaders (MPS), achieving numerically stable outputs matching the expected tensor dimensions and documented preprocessing pipeline.

> **CRITICAL SCIENTIFIC GUARD:**  
> The tests documented below utilize synthetic audio test signals and synthetic file fixtures exclusively to verify software, data flow, and tensor plumbing. **No benchmark metric (e.g., EER or min-tDCF) is claimed or measured here.** Full benchmark scoring will occur only once official ASVspoof 2019 LA evaluation splits are processed.

---

## 2. Model & Checkpoint Provenance

| Dimension | Specification |
|---|---|
| Architecture | AASIST-L (Jung et al., ICASSP 2022) |
| Upstream Repository | `https://github.com/clovaai/aasist` |
| Primary Upstream Commit | `3788ff3` (main) |
| Author / Organization | NAVER Corp / Clova AI |
| Licence | MIT License |
| Architecture Implementation | `src/prj55/models/aasist.py` |
| Checkpoint Source | `https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth` |
| Checkpoint Path | `models/weights/AASIST-L.pth` (excluded from git via `.gitignore`) |
| Checkpoint Size | 426,428 bytes (416.4 KiB) |
| Checkpoint SHA-256 | `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a` |
| Parameter Count | 85,306 trainable parameters |

---

## 3. Preprocessing Specifications

The input pipeline (`src/prj55/audio.py`) was implemented in exact accordance with the official AASIST research specification:

1. **Sampling Rate:** 16,000 Hz. Audio with other sample rates is rejected with an `AudioValidationError`.
2. **Channel Format:** Mono float32 in range `[-1.0, 1.0]`. Multi-channel audio is averaged across channels (`mean(axis=1)`).
3. **Audio Framing & Target Length:** Exact length of 64,600 samples (~4.0375 seconds).
   - If shorter than 64,600 samples: waveform is repeated $\lceil 64600 / L \rceil$ times and truncated to 64,600.
   - If longer than 64,600 samples: truncated to the initial 64,600 samples.
4. **Numerical Validation:** Verifies that all samples are finite ($\ne \text{NaN}, \ne \pm\infty$). Non-finite or corrupt files are rejected.

---

## 4. Execution Evidence & Commands

### 4.1 CPU Smoke Test Execution

**Command:**
```bash
.venv/bin/python scripts/smoke_test_inference.py --synthetic --device cpu
```

**Observed Output:**
```
[INFO] Initializing AASIST-L baseline model (device: cpu)...
[INFO] Generating synthetic audio fixture (4.0s sine wave at 16 kHz)...
[INFO] Executing forward pass (torch.no_grad, eval mode)...

============================================================
           BASELINE INFERENCE SMOKE TEST RESULTS
============================================================
Status:                      PASS
Model Architecture:          AASIST-L
Parameter Count:             85,306
Checkpoint Loaded:           True (models/weights/AASIST-L.pth)
Compute Device:              cpu
Inference Latency:           279.16 ms
Input Source:                SYNTHETIC_SIGNAL (Input pipeline test only - not real benchmark)
Input Shape:                 [1, 64600]
Logits Shape:                [1, 2]
Numerical Integrity (Finite):True

Model Output Scores:
  Raw Logit (Spoof [0]):     0.38662
  Raw Logit (Bonafide [1]):  0.05245
  Softmax Prob (Spoof):      0.58278
  Softmax Prob (Bonafide):   0.41722
============================================================
```

### 4.2 Hardware-Accelerated (MPS) Smoke Test Execution

**Command:**
```bash
.venv/bin/python scripts/smoke_test_inference.py --synthetic --device mps
```

**Observed Output:**
```
[INFO] Initializing AASIST-L baseline model (device: mps)...
[INFO] Generating synthetic audio fixture (4.0s sine wave at 16 kHz)...
[INFO] Executing forward pass (torch.no_grad, eval mode)...

============================================================
           BASELINE INFERENCE SMOKE TEST RESULTS
============================================================
Status:                      PASS
Model Architecture:          AASIST-L
Parameter Count:             85,306
Checkpoint Loaded:           True (models/weights/AASIST-L.pth)
Compute Device:              mps
Inference Latency:           3595.46 ms (includes Apple Metal kernel initialization)
Input Source:                SYNTHETIC_SIGNAL (Input pipeline test only - not real benchmark)
Input Shape:                 [1, 64600]
Logits Shape:                [1, 2]
Numerical Integrity (Finite):True

Model Output Scores:
  Raw Logit (Spoof [0]):     0.03132
  Raw Logit (Bonafide [1]):  -0.35458
  Softmax Prob (Spoof):      0.5953
  Softmax Prob (Bonafide):   0.4047
============================================================
```

---

## 5. Input Validation & Fault Injection Test Suite

A comprehensive test suite was implemented in `tests/test_baseline_inference.py` to ensure robust error handling under diverse input conditions:

| Test Case | Condition Tested | Result |
|---|---|---|
| `test_load_valid_flac` | Valid 16 kHz mono FLAC file | PASSED |
| `test_downmix_stereo_to_mono` | 2-channel audio averaged to 1D array | PASSED |
| `test_wrong_sample_rate_rejected` | 44.1 kHz audio triggers `AudioValidationError` | PASSED |
| `test_empty_audio_file_rejected` | 0-byte file rejected | PASSED |
| `test_corrupt_audio_file_rejected` | Corrupted header rejected | PASSED |
| `test_nonexistent_audio_file_raises` | Missing file triggers `FileNotFoundError` | PASSED |
| `test_pad_or_truncate_short_waveform` | 16,000 samples repeated to 64,600 | PASSED |
| `test_pad_or_truncate_long_waveform` | 80,000 samples truncated to 64,600 | PASSED |
| `test_pad_or_truncate_empty_waveform` | Empty waveform rejected | PASSED |
| `test_preprocess_audio_end_to_end` | Validates end-to-end tensor dimensions `(1, 64600)` | PASSED |
| `test_model_forward_pass_shapes` | Model produces `[2, 160]` hidden and `[2, 2]` logits | PASSED |
| `test_load_aasist_l_with_verified_checkpoint` | Loads real checkpoint and computes forward pass | PASSED |

**Full Test Suite Run:** 45 tests passed across all modules in 1.98s.

---

## 6. Score Interpretation & ASVspoof Conventions

- **Index 0:** Spoof logit score.
- **Index 1:** Bonafide logit score.
- **ASVspoof Convention:** In official ASVspoof challenge evaluations, the countermeasure score submitted for trial scoring is the **raw logit score of the bonafide class (Index 1)**. Higher values indicate higher confidence of genuine human speech, while lower values indicate deepfake / synthetic speech.
- **Softmax Probabilities:** Calibrated probabilities are computed via `softmax` for inspection, but raw logits are preserved for downstream t-DCF metric calculation to avoid log-ratio distortions.

---

## 7. Exit Decision

**Task 6 Status: PASS**  
The baseline model AASIST-L runs reliably on CPU and MPS with verified weights, produces valid finite outputs of the expected shapes in under 300 ms on CPU, and passes all 12 audio and inference unit tests.
