# data/

This directory holds all dataset files used by PRJ_55.

**This directory is excluded from Git** (see `.gitignore`). Only this README is tracked.

## Convention

```text
data/
├── README.md        ← This file (tracked)
├── raw/             ← Original downloaded files; never modify after download
│   └── asvspoof2019/
└── processed/       ← Feature files and derived outputs; reproducible from raw/
```

## Rules

- Never commit audio files, feature matrices, or trial lists to Git.
- Every dataset placed here must have an entry in `docs/source_registry.md`.
- Record the SHA-256 hash of any downloaded archive before unpacking it.
- Do not mix datasets from different sources in the same subdirectory.

## Current status

No data has been downloaded yet. See `docs/source_registry.md` for planned sources and their verification status.
