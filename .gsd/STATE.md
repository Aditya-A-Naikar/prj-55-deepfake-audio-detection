---
updated: 2026-10-22T21:20:00+05:30
---

# Project State: PRJ_55 Deepfake Audio Detection

## Current Position

**Milestone:** PRJ_55 B.Tech CSE Final Year Main Project  
**Phase:** Phase 3 — Controlled Channel Degradation & Robustness Profiling  
**Status:** `software_verified` (Signal processing, metrics, and runner verified; physical dataset benchmark blocked by external SSD)  
**Plan:** Plan 3.3 Complete / Plan 2.2 Pending External Storage  

---

## Last Action

- Implemented channel degradation pipeline (`src/prj55/degradation.py`) supporting telephony simulation (8 kHz polyphase resample with anti-aliasing), additive white Gaussian noise at configurable SNR (0, 10, 20 dB), and Opus codec encode-decode probe with graceful unavailability handling.
- Implemented official ASVspoof scoring and robustness evaluation framework (`src/prj55/metrics.py`) implementing EER, min-tDCF (official ASVspoof 2019 LA cost model), score alignment by utterance ID, and strict data-unavailability error guards (`MetricUnavailableError`).
- Implemented YAML experiment configuration (`configs/degradation_config.yaml`) and CLI experiment runner (`scripts/run_degradation_experiment.py`) executing AASIST-L inference across degradation conditions, logging inference latencies, score deltas, and outputting structured JSON artifacts.
- Implemented test suites (`tests/test_degradation.py`, `tests/test_metrics.py`). Verified 109 passing automated tests (2 conditionally skipped for ffmpeg, 0 failures).
- Verified zero errors across quality gates (`ruff check .`, `mypy src tests scripts`, `pip check`).
- Generated comprehensive Phase 3 report (`reports/phase3_robustness_report.md`).
- Added `reports/phase3_experiment_results/` to `.gitignore` to prevent generated experiment artifacts from polluting Git.

---

## Next Steps

1. **Resolve Dataset Storage Blocker**:
   - Connect external SSD/USB-C storage volume with $\ge 50\text{ GiB}$ free space.
   - Set environment variable: `export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalVolume>/asvspoof2019"`.
   - Run autonomous acquisition: `python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"`.
   - Validate MD5 checksum against Edinburgh DataShare record (`30c98f11d8b2bc21f2c257bfd78bb5c5`).
2. **Execute Full Corpus Protocol Validation**:
   - Run `python scripts/validate_protocol.py --data-dir "$ASVSPOOF_DATA_ROOT"` to verify 121,420 physical FLAC audio files and guarantee zero split leakage across train/dev/eval partitions.
3. **Execute Physical Benchmark Scoring**:
   - Run baseline evaluation of AASIST-L on official ASVspoof 2019 LA development split using `src/prj55/metrics.py` (reproducing published EER ~1.00%).
   - Execute batch channel degradation on ASVspoof 2019 LA development split and generate empirical robustness degradation curves ($\Delta\text{EER}$).
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

- [ ] **Physical Internal Storage Exhaustion**:
  - *Context*: macOS partition `/dev/disk3s5` has only ~2.9 GiB free disk space (99% utilized). Official ASVspoof 2019 LA archive (`LA.zip`) is 7.64 GB compressed, unpacking to ~25–30 GiB.
  - *Resolution Approach*: Mount external SSD ($\ge 50\text{ GiB}$), configure `$ASVSPOOF_DATA_ROOT`, and run dataset acquisition directly to the external drive.

---

## Concerns

- **30-Day Project Timeline**: Delays in acquiring physical dataset files reduce available time for running empirical channel degradation benchmarks.
- *Mitigation*: All software transformations, metrics, and runners are 100% pre-tested and verified with synthetic signals. When the drive is mounted, benchmark execution will run seamlessly with zero pending software work.
- **Metal Performance Shaders (MPS) Latency**: Single-thread CPU forward pass execution takes ~230–300 ms on Apple Silicon M2, easily satisfying real-time constraints without MPS warmup overhead.

---

## Session Context

- Python environment: `.venv/` (Python 3.10.0 ARM64 on Apple Silicon M2).
- Pretrained weights: `models/weights/AASIST-L.pth` (verified SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).
- Test suite: 109 passed, 2 skipped, 0 failed (`pytest tests/`).
- Lint & Type check: 0 errors across 16 source files (`ruff check .`, `mypy src tests scripts`).
- Dependencies: Clean (`pip check` passed).
