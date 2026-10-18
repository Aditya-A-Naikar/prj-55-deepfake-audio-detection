# Data Governance Policy — PRJ_55: Deepfake Audio Detection

**Document status:** Active  
**Created:** 2026-10-16  
**Scope:** This policy applies to all datasets, audio recordings, model weights, and generated artefacts used or produced in this project.

---

## 1. Core Principles

1. **Reproducibility over convenience.** Every dataset version and model checkpoint used must be precisely identified so that any result can be reproduced from the documented source.
2. **No silent inclusions.** Nothing may be added to the repository or to the local data directory without recording its source, version, and licence/access status in `docs/source_registry.md`.
3. **Honesty about what is committed.** Only source code, documentation, and small configuration text files belong in Git. Data and weights live outside the repository.

---

## 2. What Must NOT Be Committed to Git

### 2.1 Raw and processed datasets
- All audio recordings, regardless of format (`.wav`, `.flac`, `.mp3`, etc.).
- All feature matrices, trial lists derived from protected sources, and intermediate processing outputs.
- This includes data downloaded from ASVspoof, any publicly available speech corpus, and any privately collected recordings.

### 2.2 Model weights and checkpoints
- Pretrained weights (`.pt`, `.pth`, `.ckpt`, `.bin`, `.onnx`, `.safetensors`, etc.) must **not** be committed unless:
  1. Their file size has been verified to be small enough for Git (< 5 MB as a guideline).
  2. Their redistribution licence explicitly permits inclusion in a public or private repository.
  3. Their inclusion is technically necessary and documented in an ADR.
- If all three conditions are met, the checkpoint's SHA-256 hash must be recorded in `docs/source_registry.md` before committing.

### 2.3 Secrets and credentials
- API keys, access tokens, login credentials, and `.env` files must never be committed.
- If a service requires credentials, document the procedure for obtaining them without embedding the credentials in code or config files.

### 2.4 Private recordings
- Audio recordings of individuals, including researcher voices used for testing, must not be uploaded or redistributed without the explicit informed consent of each person recorded.

---

## 3. Local and External Data Directory Convention

Due to local storage constraints on development hardware (~2.9 GiB free on internal disk), large raw corpora (such as the ~35 GiB ASVspoof 2019 LA partition) must reside on external storage or remote server volumes rather than the internal disk.

The project resolves data locations via the `ASVSPOOF_DATA_ROOT` environment variable or portable configuration files (`configs/dataset_protocol.example.yaml`).

Expected external layout:

```text
$ASVSPOOF_DATA_ROOT/
├── ASVspoof2019_LA_cm_protocols/  ← Official trial and key protocol files
├── ASVspoof2019_LA_train/flac/    ← 16 kHz FLAC audio
├── ASVspoof2019_LA_dev/flac/      ← 16 kHz FLAC audio
└── ASVspoof2019_LA_eval/flac/     ← 16 kHz FLAC audio
```

For small local tests, synthetic fixtures in `tests/` are used exclusively and isolated within temporary directories. Every external dataset root must correspond to a verified entry in `docs/source_registry.md`.

---

## 4. Local Models Directory Convention

```text
models/
├── README.md          ← Tracked
├── checkpoints/       ← NOT tracked; downloaded weight files
└── exports/           ← NOT tracked; ONNX or other exports
```

Every file placed in `models/checkpoints/` must be recorded in `docs/source_registry.md` with its SHA-256 hash, source URL, and licence.

---

## 5. Licence and Access Compliance

| Source type | Requirement |
|---|---|
| ASVspoof 2019 / 2021 datasets | Must be accessed only through official registration; data must not be redistributed |
| Pretrained model weights | Must be used within the terms stated by the weight author; redistribution requires explicit permission |
| Third-party code copied into `src/` | Must retain original copyright headers and licence notices |
| All other external materials | Must be cited; licence must be confirmed before use |

---

## 6. Enforcement

- The `.gitignore` file is the primary technical control.
- Developers must run `git ls-files` before each push to confirm no forbidden file types are tracked.
- The scaffold test (`tests/test_project_scaffold.py`) verifies `.gitignore` patterns at every CI run.
- Any accidental commit of a forbidden file must be removed from history using `git filter-repo` (not just `git rm`) before the next push. **Do not force-push to a shared remote without team agreement.**
