---
updated: 2026-10-22T20:55:00+05:30
---

# Project State: PRJ_55 Deepfake Audio Detection

## Current Position

**Milestone:** PRJ_55 B.Tech CSE Final Year Main Project  
**Phase:** Phase 2 — Dataset Acquisition, Protocol Validation & Baseline Benchmark  
**Status:** `blocked` (Internal disk storage exhaustion; external SSD required)  
**Plan:** Plan 2.2 — Acquire ASVspoof 2019 LA corpus onto external storage and verify MD5 checksum  

---

## Last Action

- Executed preflight audit: verified 48 passing automated unit and integration tests (95% code coverage, clean `ruff` and `mypy` checks).
- Verified AASIST-L PyTorch implementation (`src/prj55/models/aasist.py`), audio preprocessor (`src/prj55/audio.py`), and inference smoke test (`scripts/smoke_test_inference.py`).
- Initialized GSD specification (`.gsd/SPEC.md`, status: `FINALIZED`).
- Initialized project roadmap (`.gsd/ROADMAP.md`) covering Phases 1–4 across the 30-day timeline.
- Fixed line endings and validated GSD tool manifests (`scripts/validate-all.sh` passed).

---

## Next Steps

1. **Resolve Dataset Storage Blocker**:
   - Connect external SSD/USB-C storage volume with $\ge 50\text{ GiB}$ free space.
   - Set environment variable: `export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalVolume>/asvspoof2019"`.
   - Run autonomous acquisition: `python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"`.
   - Validate MD5 checksum against Edinburgh DataShare record (`30c98f11d8b2bc21f2c257bfd78bb5c5`).
2. **Execute Full Corpus Protocol Validation**:
   - Run `python scripts/validate_protocol.py --data-dir "$ASVSPOOF_DATA_ROOT"` to verify 121,420 physical FLAC audio files and guarantee zero split leakage across train/dev/eval partitions.
3. **Run Baseline Benchmark Scoring**:
   - Implement official metric evaluator (`src/prj55/metrics.py`) for Equal Error Rate (EER) and minimum tandem Detection Cost Function (min-tDCF).
   - Evaluate AASIST-L on ASVspoof 2019 LA development split to reproduce published baseline (EER ~1.00%).
4. **Alternative Immediate Action (if SSD pending)**:
   - Begin Phase 3 (Plan 3.1): Implement controlled channel degradation simulator (`src/prj55/degradation.py`) using synthetic audio signals, decoupling pipeline development from dataset download.

---

## Active Decisions

| Decision | Choice | Made | Affects |
|---|---|---|---|
| **ADR-001: Architecture Selection** | Official AASIST-L (Jung et al., ICASSP 2022, 85,306 parameters, SincNet + Heterogeneous Graph Attention) | Week 1 | All phases; strictly satisfies sub-100k parameter limit and real-time CPU budget |
| **ADR-002: Storage & Corpus Management** | Strict prohibition of dataset download to internal SSD; resolve paths via `$ASVSPOOF_DATA_ROOT` | Week 2 | Phase 2 acquisition, protocol validation, benchmark scripts |
| **ADR-003: Modular Degradation Design** | Discrete transform pipelines (Telephony 8 kHz, Opus/AAC lossy compression, acoustic SNR noise) | Week 2 | Phase 3 robustness experiments and Phase 4 local application |

---

## Blockers

- [ ] **Physical Internal Storage Exhaustion**:
  - *Context*: macOS partition `/dev/disk3s5` has only ~2.9 GiB free disk space (99% utilized). Official ASVspoof 2019 LA archive (`LA.zip`) is 7.64 GB compressed, unpacking to ~25–30 GiB.
  - *Resolution Approach*: Mount external SSD ($\ge 50\text{ GiB}$), configure `$ASVSPOOF_DATA_ROOT`, and run dataset acquisition directly to the external drive.

---

## Concerns

- **30-Day Project Timeline**: Delays in acquiring physical dataset files reduce available time for running channel degradation experiments and model adaptation.
- *Mitigation*: Develop degradation simulation pipelines and unit tests using synthetic fixtures immediately while external drive provisioning is underway.
- **Metal Performance Shaders (MPS) Latency**: Inference smoke test showed MPS warmup overhead (~3.5s) compared to single-thread CPU execution (~279 ms). CPU inference is preferred for single-sample real-time evaluation.

---

## Session Context

- Python environment: `.venv/` (Python 3.10.0 ARM64 on Apple Silicon M2).
- Pretrained weights: `models/weights/AASIST-L.pth` (verified SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).
- Test suite: 48 passed, 0 failed, 95% coverage (`pytest`).
- Lint & Type check: 0 errors (`ruff check .`, `mypy src tests scripts`).
