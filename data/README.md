# Data Management & Storage Governance — PRJ_55

This document outlines storage conventions, access procedures, and directory structures for dataset files in PRJ_55.

**All audio corpora, archive packages, and protocol files are strictly excluded from Git** (enforced by `.gitignore`). Only documentation and example configurations are tracked.

---

## 1. External Storage Policy (Hardware Constraint)

Due to development machine storage limitations (laptop internal disk has only ~2.9 GiB free), **the raw dataset must reside on an external volume** (e.g. external SSD / USB-C drive or network mount) with at least 50–60 GiB free capacity.

### Recommended External Drive Layout

```text
<EXTERNAL_VOLUME>/
└── data/
    ├── archives/
    │   └── LA.zip                  # Official archive (~7.12 GiB)
    ├── staging/                    # Temporary unpacking workspace
    └── raw/
        └── LA/                     # Extracted corpus (~25–30 GiB)
            ├── ASVspoof2019_LA_cm_protocols/
            │   ├── ASVspoof2019.LA.cm.train.trn.txt
            │   ├── ASVspoof2019.LA.cm.dev.trl.txt
            │   └── ASVspoof2019.LA.cm.eval.trl.txt
            ├── ASVspoof2019_LA_train/
            │   └── flac/           # 25,380 FLAC audio files (16 kHz)
            ├── ASVspoof2019_LA_dev/
            │   └── flac/           # 24,844 FLAC audio files (16 kHz)
            └── ASVspoof2019_LA_eval/
                └── flac/           # 71,196 FLAC audio files (16 kHz)
```

---

## 2. Configuring the Dataset Root (`ASVSPOOF_DATA_ROOT`)

Set the environment variable pointing to the extracted `LA` root directory without committing absolute local paths:

```bash
# Export the root path of the extracted Logical Access corpus:
export ASVSPOOF_DATA_ROOT="/Volumes/<YOUR_EXTERNAL_VOLUME>/data/raw/LA"

# Validate all splits using the project CLI:
python scripts/validate_protocol.py --all-splits
```

Alternatively, copy `configs/dataset_protocol.example.yaml` to an untracked local configuration `configs/dataset_protocol.yaml` and configure custom paths.

---

## 3. Official Source & Authoritative Checksums

- **Repository:** University of Edinburgh DataShare
- **Handle:** `http://hdl.handle.net/10283/3336` (Item UUID: `31074a11-b6f6-4e92-a4ad-07093f8c0c45`)
- **Direct Bitstream Download URL:**  
  `https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content`
- **File:** `LA.zip`
- **Official Size:** `7,640,952,520` bytes (~7.12 GiB)
- **Official Checksum:** `MD5: 30c98f11d8b2bc21f2c257bfd78bb5c5`
- **License:** Open Data Commons Attribution License (ODC-By) v1.0

---

## 4. Split Integrity Governance

- **Train:** 20 speakers (8 male, 12 female); known attacks A01–A06. Used exclusively for model parameter fitting.
- **Dev:** 10 speakers (4 male, 6 female); known attacks A01–A06. Used for hyperparameter tuning and score threshold calibration.
- **Eval:** 48 speakers (21 male, 27 female); unseen attacks A07–A19. Strictly held out until final benchmark evaluation. Normalization statistics must never be estimated on evaluation data.
