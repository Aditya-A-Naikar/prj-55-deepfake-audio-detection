# Dataset Access and Protocol Audit — PRJ_55: Deepfake Audio Detection

**Date:** 2026-10-17  
**Auditor:** Antigravity AI Assistant  
**Repository:** `prj-55-deepfake-audio-detection`  
**Milestone:** Week 1, Task 5 — Dataset Access and Protocol Validation  
**Overall Status:** PARTIAL (Software & Protocol Pipeline: PASS; Local Data Acquisition: BLOCKED by Disk Capacity)  

---

## 1. Local Data Availability Inspection

A comprehensive filesystem scan was conducted across the project tree and parent workspace for existing copies of the ASVspoof dataset or FLAC audio files.

| Check | Target | Result | Status |
|---|---|---|---|
| Local Raw Audio Files | `data/raw/`, `*.flac`, `*.wav` | 0 files found | No pre-existing dataset |
| Protocol Text Files | `*ASVspoof2019*.txt` | 0 files found | No pre-existing protocols |
| Primary Volume Storage | `/dev/disk3s5` | **2.9 GiB free** (99% capacity) | **INSUFFICIENT STORAGE** |
| ASVspoof 2019 LA Download Size | Compressed ZIP archives | ~25–30 GiB | Exceeds free disk by ~10x |
| Uncompressed Dataset Footprint | FLAC audio + protocols | ~35–40 GiB | Cannot unpack on laptop |

### Finding
No authorized local copy of ASVspoof 2019 LA is present. Furthermore, downloading the official corpus directly to this laptop's internal disk is completely blocked by storage constraints until external storage is mounted or disk space is reclaimed.

---

## 2. Official Access Instructions & Licensing Conditions

- **Primary Repository:** University of Edinburgh DataShare (`https://datashare.ed.ac.uk/handle/10283/3336`).
- **Official Evaluation Portal:** ASVspoof Consortium (`https://www.asvspoof.org/index2019.html`).
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Access Procedure:**
  1. Access Edinburgh DataShare directly or register on `asvspoof.org`.
  2. Download official archives:
     - `LA.zip` (containing `ASVspoof2019_LA_train/`, `ASVspoof2019_LA_dev/`, `ASVspoof2019_LA_eval/`, and `ASVspoof2019_LA_cm_protocols/`).
  3. Verify MD5 / SHA-256 checksums published on Edinburgh DataShare.
  4. Store the raw dataset outside git tracking (configured via `configs/dataset_protocol.example.yaml` or `$ASVSPOOF_DATA_ROOT`).

---

## 3. Protocol Architecture & Validation Pipeline

To ensure rigorous validation of protocol files prior to model consumption, a dedicated protocol validator module and CLI tool have been designed and implemented.

### 3.1 Software Artifacts Implemented

1. **`src/prj55/protocol.py`:**
   - Parses ASVspoof 2019 Logical Access protocol records:
     - Standard 5-column format: `SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY LABEL`
     - Blind evaluation trial format: `SPEAKER_ID AUDIO_FILE_NAME [SYSTEM_ID] [KEY] [LABEL]`
   - Detects malformed lines (insufficient tokens, invalid labels, unparseable entries).
   - Detects duplicate utterance identifiers to prevent trial corruption.
   - Accurately tracks label distributions (`bonafide` vs `spoof`) and spoof attack algorithms (`A01`–`A06` in train/dev; `A07`–`A19` in evaluation).
   - Verifies the physical presence of associated audio files (`.flac`) when an audio directory is provided.
   - Generates machine-readable JSON summaries (`to_json()`) and human-readable Markdown reports (`generate_markdown_report()`).

2. **`scripts/validate_protocol.py`:**
   - Standalone CLI utility for validating protocol files and verifying file existence without starting training.

3. **`configs/dataset_protocol.example.yaml`:**
   - Portable configuration template using environment variables (`${ASVSPOOF_DATA_ROOT:-data/raw/LA}`) to prevent hardcoding personal absolute paths into source repositories.

---

## 4. Test Suite and Synthetic Fixtures

Per Week 1 testing rules, software validation was verified using synthetic fixtures created exclusively inside temporary pytest directories. Synthetic fixtures are isolated and never reported as benchmark dataset results.

### 4.1 Test Coverage (`tests/test_protocol_validation.py`)

- `TestProtocolParsing`: Single line parsing for bonafide, spoof, whitespace, comments, malformed labels, and truncated lines.
- `TestProtocolValidationFile`: Full file parsing on valid train protocols, blind evaluation protocols, malformed files, duplicate IDs, and non-existent files.
- `TestAudioPathValidation`: Verifies audio existence check, detects missing audio files, and handles absent audio directories.
- `TestReportOutputs`: Validates Markdown and JSON serialization integrity.
- `TestSplitAutodetection`: Validates automatic classification of train, dev, and eval files from path names.

### 4.2 Test Results

```
============================== 33 passed in 1.64s ==============================
```
All 33 project tests (scaffold + environment + protocol validation) pass cleanly.

---

## 5. Protocol Split Integrity Rules

To preserve scientific rigor and prevent data leakage:

1. **Training Partition (`train`):**
   - Strictly reserved for model parameter fitting and feature representation learning.
   - Contains attacks `A01`–`A06`.
2. **Development Partition (`dev`):**
   - Reserved exclusively for hyperparameter tuning, model selection, and score calibration threshold tuning.
   - Contains attacks `A01`–`A06`.
3. **Evaluation Partition (`eval`):**
   - Must remain completely held-out and untouched until final evaluation.
   - Contains unseen synthesis attacks `A07`–`A19`.
   - **Critical Rule:** Never fit normalization (mean/variance cepstral normalization, PCA, or whitening) on evaluation data. Any normalization statistics must be computed strictly on training data and frozen.

---

## 6. Exact Files Changed

- `src/prj55/protocol.py` (New: core protocol parsing and validation engine)
- `scripts/validate_protocol.py` (New: CLI validation utility)
- `tests/test_protocol_validation.py` (New: 14 comprehensive unit tests)
- `configs/dataset_protocol.example.yaml` (New: portable protocol configuration template)
- `pyproject.toml` (Updated: configured pytest `pythonpath = ["src"]`)
- `reports/dataset_protocol_audit.md` (New: this audit report)

---

## 7. Exit Decision & Recommendations

| Item | Status | Evidence |
|---|---|---|
| Protocol Validator Engine | **PASS** | `src/prj55/protocol.py` implemented, 14 unit tests green, type checked with mypy |
| CLI Validation Tool | **PASS** | `scripts/validate_protocol.py` operational with JSON/Markdown reporting |
| Split Integrity Rules | **PASS** | Documented in `reports/dataset_protocol_audit.md` and configuration schema |
| Local Data Acquisition | **BLOCKED** | Available disk is 2.9 GiB; dataset requires ~35 GiB; external storage required |

**Overall Milestone Assessment:** PARTIAL (Software ready; external disk allocation needed for full dataset download).  
**Recommended Next Task:** Week 1, Task 6 — Baseline Inference Smoke Test.
