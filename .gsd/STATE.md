---
updated: 2026-10-22T22:45:00+05:30
---

# Project State: PRJ_55 Deepfake Audio Detection

## Current Position

**Milestone:** PRJ_55 B.Tech CSE Final Year Main Project  
**Phase:** Phase 2 — Dataset Acquisition, Protocol Validation & Baseline Benchmark  
**Status:** `pipeline_verified` (Evaluation pipeline, score generator, score evaluator, metrics, and safety guards complete; physical dataset benchmark blocked by external SSD media separation)  
**Plan:** Plan 2.1 & 2.4 Complete; Plan 2.2 / 2.3 / 2.5 Pending External Storage  

---

## Last Action

- Conducted exhaustive hardware and filesystem storage audit: discovered that while **61 GiB of free space** is now available on `/System/Volumes/Data` (`/dev/disk3s5`), it is located entirely on the Mac's internal Apple SSD (`/dev/disk0`), with **0 external volumes attached**.
- Enforced ADR-002 and prompt directives: prevented downloading or unpacking the ~15–20 GiB ASVspoof 2019 LA corpus on the internal disk to protect system swap and host disk longevity.
- Implemented autonomous acquisition tool (`scripts/acquire_asvspoof2019.py`) with pre-flight storage safety checks, DataShare REST API endpoints, streaming MD5 verification (`30c98f11d8b2bc21f2c257bfd78bb5c5`), and Zip Slip safe extraction.
- Enhanced evaluation metrics in `src/prj55/metrics.py`: added `compute_operating_point_metrics` (confusion matrix, FPR, FNR, precision, recall, F1, balanced accuracy with threshold governance), `compute_roc_auc` (Wilcoxon-Mann-Whitney U statistic with rank ties), and `compute_official_min_tdcf` (strict tandem ASV score requirement guard).
- Implemented score generator CLI (`scripts/generate_scores.py`): streams AASIST-L bonafide logits to formatted text files with metadata headers, supporting official protocol loading and synthetic dry-run verification.
- Implemented score evaluator CLI (`scripts/evaluate_scores.py`): aligns scores and protocol labels, computes EER, operating point, ROC-AUC, min-tDCF guards, attack breakdowns, and outputs structured JSON and reports.
- Implemented comprehensive test suites (`tests/test_acquire.py`, `tests/test_generate_scores.py`, `tests/test_evaluate_scores.py`, `tests/test_metrics.py`). Verified **136 passed, 2 skipped, 0 failed** with **92% code coverage** across `src/prj55`.
- Verified quality gates: `pip check` (clean), `ruff check .` (0 errors), `mypy src tests scripts` (0 errors), `bash scripts/validate-all.sh` (all passed).
- Generated comprehensive Week 2 report (`reports/week2_completion_report.md`).

---

## Next Steps

1. **Resolve Dataset Storage Blocker (External Media)**:
   - Connect external SSD/USB-C storage volume with $\ge 50\text{ GiB}$ free space.
   - Set environment variable: `export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalVolume>/asvspoof2019"`.
   - Run autonomous acquisition: `python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"`.
   - Validate MD5 checksum against Edinburgh DataShare record (`30c98f11d8b2bc21f2c257bfd78bb5c5`).
2. **Execute Full Corpus Protocol Validation**:
   - Run `python scripts/validate_protocol.py --data-dir "$ASVSPOOF_DATA_ROOT"` to verify 121,420 physical FLAC audio files and guarantee zero split leakage across train/dev/eval partitions.
3. **Execute Physical Benchmark Scoring**:
   - Run baseline score generation on development split:
     `python scripts/generate_scores.py --data-dir "$ASVSPOOF_DATA_ROOT" --split dev`
   - Run baseline evaluation:
     `python scripts/evaluate_scores.py --scores scores/cm_scores_dev.txt --protocol "$ASVSPOOF_DATA_ROOT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"`
   - Derive development threshold and evaluate held-out evaluation split.
4. **Advance to Phase 4 (Demonstration & Mitigation)**:
   - Implement interactive local web application demonstration (e.g., FastAPI + frontend) for real-time degradation simulation and spoof score visualization.

---

## Active Decisions

| Decision | Choice | Made | Affects |
|---|---|---|---|
| **ADR-001: Architecture Selection** | Official AASIST-L (Jung et al., ICASSP 2022, 85,306 parameters, SincNet + Heterogeneous Graph Attention) | Week 1 | All phases; strictly satisfies sub-100k parameter limit and real-time CPU budget |
| **ADR-002: Storage & Corpus Management** | Strict prohibition of dataset download to internal SSD; resolve paths via `$ASVSPOOF_DATA_ROOT` | Week 2 | Phase 2 acquisition, protocol validation, benchmark scripts |
| **ADR-003: Modular Degradation Design** | Discrete transform pipelines (Telephony 8 kHz, Opus lossy compression, acoustic SNR noise) | Week 2 | Phase 3 robustness experiments and Phase 4 local application |
| **ADR-004: Strict Metric Availability Guarding** | Throw `MetricUnavailableError` when labels/data are absent; never report synthetic tests as benchmark metrics | Week 3 | Metrics module, experiment runner, reporting integrity |

---

## Blockers

- [ ] **External Storage Media Unmounted**:
  - *Context*: macOS partition `/dev/disk3s5` now has 61 GiB free space, but is on the internal disk. No external SSD is attached.
  - *Resolution Approach*: Mount external SSD ($\ge 50\text{ GiB}$), configure `$ASVSPOOF_DATA_ROOT`, and run dataset acquisition directly to the external drive.

---

## Concerns

- **30-Day Project Timeline**: Physical dataset benchmarking requires external drive mounting.
- *Mitigation*: All acquisition, score generation, and evaluation pipelines are 100% written, tested, and validated with synthetic fixtures. When external storage is attached, executing the benchmark will require zero new code.

---

## Session Context

- Python environment: `.venv/` (Python 3.10.0 ARM64 on Apple Silicon M2).
- Pretrained weights: `models/weights/AASIST-L.pth` (verified SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).
- Test suite: 136 passed, 2 skipped, 0 failed (`pytest --cov=prj55 tests/`).
- Code coverage: 92% statement coverage on `src/prj55`.
- Lint & Type check: 0 errors across 22 source files (`ruff check .`, `mypy src tests scripts`).
- Dependencies: Clean (`pip check` passed).
