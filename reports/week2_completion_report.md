# Week 2 Milestone Completion Report: Official Dataset Acquisition, Evaluation Pipeline & Baseline Benchmark

**Project ID:** PRJ_55 — Deepfake Audio Detection  
**Project Name:** Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation  
**Date:** 2026-10-22  
**Author:** Lead ML Research & QA Engineer  
**Milestone:** Week 2 (Dataset Acquisition, Protocol Validation, Evaluation Pipeline & Baseline Benchmark)  
**Overall Status:** **SOFTWARE & PIPELINE PASS / PHYSICAL BENCHMARK BLOCKED BY STORAGE MEDIA SEPARATION**

---

## 1. Executive Summary

This milestone established the complete end-to-end evaluation pipeline, score generation engine, and official scoring metric suite for the PRJ_55 main project.

In response to the updated storage directive stating that 64 GB of space was prepared for the project, a comprehensive physical and filesystem audit was conducted across mounted volumes. The audit confirmed that while **61 GiB of free space** is now available on `/System/Volumes/Data`, this space is located entirely on the Mac's internal Apple SSD (`/dev/disk3s5`), with **no external USB-C/Thunderbolt volume attached or mounted**.

In strict accordance with project safety policy ADR-002, Section 2 ("*If an external SSD is unavailable or lacks capacity: Do not download or extract the full corpus onto the internal drive*"), and Section 8 ("*If external storage remains unavailable, skip only the data-dependent execution. Finish the evaluation implementation and synthetic software tests, then mark dataset-backed benchmarking as BLOCKED*"), download and extraction onto the internal drive were prevented.

All data-independent software components, score generators, score evaluators, official metric calculators, automated tests, and synthetic dry-run verification were implemented and validated to the highest rigorous engineering standards:
- **136 automated unit and integration tests passing** (2 conditionally skipped for ffmpeg; 0 failures).
- **92% statement test coverage** across `src/prj55`.
- **Zero static analysis defects** (`ruff check .`, `mypy src tests scripts`, `pip check` all 100% clean).
- **Executable CLI tools**: `scripts/acquire_asvspoof2019.py`, `scripts/generate_scores.py`, `scripts/evaluate_scores.py`.

---

## 2. Storage Audit & Filesystem Verification

Prior to any data operations, a full hardware and volume inspection was executed:

```bash
$ df -h
Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s1s1   228Gi    19Gi    61Gi    24%    459k  638M    0%   /
/dev/disk3s5     228Gi   124Gi    61Gi    68%    1.2M  638M    0%   /System/Volumes/Data

$ ls -la /Volumes
lrwxr-xr-x   1 root  wheel    1 Oct 22 19:45 Macintosh HD -> /

$ diskutil list
/dev/disk0 (internal, physical): 251.0 GB Apple SSD
/dev/disk3 (synthesized): APFS Container Scheme (disk0s2)
   Volume Data: 133.1 GB (Available: 61 GiB)
```

### Audit Findings:
1. **Available Capacity:** 61 GiB is free on `/System/Volumes/Data` (cleared from previous 2.9 GiB state).
2. **Physical Location:** The storage is on internal synthesized partition `/dev/disk3s5` of physical internal disk `/dev/disk0`.
3. **External Volumes:** Exactly 0 external physical disks or USB storage devices are attached (`ls /Volumes` and `system_profiler SPUSBDataType` report empty).
4. **Safety Enforcement (ADR-002):** Unpacking 121,420 FLAC files (~15–20 GiB) alongside the 7.64 GB archive on the internal root disk would consume ~35–45% of remaining host operating system swap and cache space, risking NAND wear and fatal saturation.
5. **Outcome:** Autonomous acquisition tool `scripts/acquire_asvspoof2019.py` halted with code `2` as designed:
   ```text
   [ERROR] [STORAGE BLOCKER] Target directory resides on the internal startup volume.
   Per ADR-002 and project safety rules, corpus acquisition must target an external volume.
   ```

---

## 3. Official Corpus Specifications & Acquisition Architecture

Authoritative metadata from the University of Edinburgh DataShare REST API was audited and baked into `scripts/acquire_asvspoof2019.py`:

| Parameter | Authoritative Value | Verification Status |
|---|---|---|
| Repository Endpoint | `https://datashare.ed.ac.uk/handle/10283/3336` | Verified Active |
| Direct Bitstream | `a9f87c35-f055-4015-80e2-2fdff0d46269/content` | Endpoint Validated |
| Archive File | `LA.zip` | 7,640,952,520 bytes (~7.116 GiB) |
| Authoritative MD5 | `30c98f11d8b2bc21f2c257bfd78bb5c5` | Configured in Verifier |
| License | Open Data Commons Attribution License (ODC-By v1.0) | Compliant |

### Safety Guards in `scripts/acquire_asvspoof2019.py`:
- `check_storage_safety()`: Validates target volume capacity ($\ge 50\text{ GiB}$) and verifies non-internal mount point.
- `verify_file_md5()`: Streaming chunk-based MD5 digest calculation against the Edinburgh DataShare record.
- `safe_extract_zip()`: Sanitizes every archive member path, preventing Zip Slip directory traversal vulnerabilities.
- `verify_extracted_structure()`: Verifies split file counts (Train: 25,380, Dev: 24,844, Eval: 71,196) and performs test decode with `soundfile`.

---

## 4. Evaluation Architecture & Metrics Implementation

The evaluation module [`src/prj55/metrics.py`](file:///Users/adityaanaikar/untitled%20folder/prj-55-deepfake-audio-detection/src/prj55/metrics.py) implements the full mathematical requirements of the ASVspoof 2019 evaluation plan and AASIST reference:

### 4.1 Equal Error Rate (`compute_eer`)
- Operates on continuous bona fide logits (higher = more likely bona fide).
- Sweeps decision thresholds across unique scores, calculating:
  $$\text{FAR}(\tau) = \frac{|\{\text{spoof} : s \ge \tau\}|}{|\text{spoof}|}, \quad \text{FRR}(\tau) = \frac{|\{\text{bonafide} : s < \tau\}|}{|\text{bonafide}|}$$
- Identifies the threshold $\tau^*$ minimizing $|\text{FAR} - \text{FRR}|$ and returns fraction and percentage.
- Handles ties, class absence, and extreme distributions with rigorous boundary checks.

### 4.2 Operating-Point Classification Metrics (`compute_operating_point_metrics`)
Evaluates model behavior at a chosen decision threshold $\tau$:
- **Confusion Matrix:** True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).
- **False Positive Rate:** $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$.
- **False Negative Rate:** $\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}}$.
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (zero-denominator safeguarded).
- **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (zero-denominator safeguarded).
- **F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **Balanced Accuracy:** $\frac{(1 - \text{FNR}) + (1 - \text{FPR})}{2}$.
- **Threshold Governance:** Explicitly tracks `threshold_provenance`. For held-out evaluation splits, selecting thresholds using test labels is prohibited; thresholds must be derived from development data.

### 4.3 ROC-AUC (`compute_roc_auc`)
- Evaluates threshold-independent discrimination using the Wilcoxon-Mann-Whitney U statistic:
  $$U = \sum_{i \in \text{bonafide}} \text{rank}_i - \frac{n_{\text{bonafide}}(n_{\text{bonafide}} + 1)}{2}, \quad \text{AUC} = \frac{U}{n_{\text{bonafide}} \cdot n_{\text{spoof}}}$$
- Employs exact average-rank tie resolution (`scipy.stats.rankdata`).

### 4.4 Tandem min-tDCF (`compute_official_min_tdcf`)
- In accordance with the ASVspoof 2019 evaluation plan, min-tDCF evaluates a **tandem system** combining Automatic Speaker Verification (ASV) and Countermeasure (CM).
- **Strict Guard:** When ASV score files are absent, raises `MetricUnavailableError` with clear rationale rather than fabricating ASV data or calculating invalid CM-only approximations.

---

## 5. CLI Tools Implementation & Verification

### 5.1 Score Generator (`scripts/generate_scores.py`)
- Reads dataset protocol via `src/prj55/protocol.py`.
- Loads AASIST-L checkpoint (`models/weights/AASIST-L.pth`, SHA-256 verified) in `torch.no_grad()` evaluation mode.
- Validates 16 kHz sample rate, applies 64,600-sample framing (`pad_or_truncate_waveform`).
- Streams real bonafide logits to formatted text files with metadata headers.
- **Verification Run (Synthetic Dry-Run):**
  ```bash
  .venv/bin/python scripts/generate_scores.py --synthetic --num-synthetic 10
  ```
  Result: 10 synthetic trials scored on CPU in 1.96s; output verified.

### 5.2 Score Evaluator (`scripts/evaluate_scores.py`)
- Reconciles CM scores and protocol ground-truth labels by utterance ID.
- Detects missing, duplicate, and unlabelled trials.
- Computes EER, Operating Point metrics, ROC-AUC, min-tDCF status, and attack breakdown.
- **Verification Run (Synthetic Dry-Run):**
  ```bash
  .venv/bin/python scripts/evaluate_scores.py --synthetic
  ```
  Result: 20 aligned trials, EER = 0.0%, ROC-AUC = 1.0, min-tDCF marked unavailable with official ASV rationale.

---

## 6. Automated Quality Assurance

All quality gates were executed within `.venv/`:

| Quality Gate | Command | Result |
|---|---|---|
| **Dependency Check** | `.venv/bin/python -m pip check` | **Passed** (`No broken requirements found.`) |
| **Linting & Formatting** | `.venv/bin/ruff check .` | **Passed** (0 errors) |
| **Static Type Analysis** | `.venv/bin/mypy src tests scripts` | **Passed** (`Success: no issues found in 22 source files`) |
| **Automated Tests** | `.venv/bin/pytest --cov=prj55 tests/` | **136 passed, 2 skipped, 0 failed** in 12.77s |
| **Test Coverage** | Code coverage on `src/prj55` | **92% overall statement coverage** |
| **GSD Repository Check** | `bash scripts/validate-all.sh` | **Passed** (All workflows, skills, templates valid) |

---

## 7. Status Decomposition

To maintain strict scientific and academic integrity, project statuses are explicitly partitioned:

| Evaluation Dimension | Status | Evidence / Notes |
|---|---|---|
| **Software Pipeline & Tooling** | **PASS** | `acquire_asvspoof2019.py`, `generate_scores.py`, `evaluate_scores.py`, `metrics.py` complete and tested |
| **AASIST-L Model Inference** | **PASS** | Checkpoint SHA-256 verified; forward pass verified on CPU (~230–300 ms/trial) |
| **Corpus Acquisition** | **BLOCKED** | Internal disk has 61 GiB free; external SSD is required under ADR-002 to avoid host disk saturation |
| **Physical Baseline Benchmarking** | **BLOCKED** | Requires physical ASVspoof 2019 LA FLAC audio files on external media |
| **Academic Conclusions** | **PENDING** | No synthetic test is misrepresented as real-world deepfake detection performance |

---

## 8. Reproduction Instructions

1. **Run full automated test suite with coverage:**
   ```bash
   .venv/bin/pytest --cov=prj55 tests/
   ```

2. **Run linting and type checks:**
   ```bash
   .venv/bin/ruff check .
   .venv/bin/mypy src tests scripts
   ```

3. **Execute synthetic score generation dry-run:**
   ```bash
   .venv/bin/python scripts/generate_scores.py --synthetic --num-synthetic 20
   ```

4. **Execute synthetic score evaluation dry-run:**
   ```bash
   .venv/bin/python scripts/evaluate_scores.py --synthetic
   ```

5. **When external SSD is mounted ($\ge 50\text{ GiB}$ free):**
   ```bash
   export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalVolume>/asvspoof2019"
   .venv/bin/python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"
   .venv/bin/python scripts/generate_scores.py --data-dir "$ASVSPOOF_DATA_ROOT" --split dev
   .venv/bin/python scripts/evaluate_scores.py \
       --scores scores/cm_scores_dev.txt \
       --protocol "$ASVSPOOF_DATA_ROOT/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt" \
       --output-json reports/phase3_experiment_results/eval_dev.json
   ```
