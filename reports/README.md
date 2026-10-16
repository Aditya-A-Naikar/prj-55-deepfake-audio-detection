# reports/

This directory holds generated outputs: figures, result tables, and summary PDFs.

Generated outputs are not tracked by Git (see `.gitignore` — `reports/figures/`, `reports/outputs/`).

## Convention

```text
reports/
├── README.md        ← This file (tracked)
├── figures/         ← NOT tracked; generated plots (PNG, SVG)
├── outputs/         ← NOT tracked; CSV/JSON result summaries
└── week1_summary/   ← Will be added at end of Week 1
```

## Rules

- Never commit generated figures or result files; reproduce them from code instead.
- Every reported number must correspond to a completed experiment entry in `docs/experiment_log.md`.

## Current status

No reports have been generated yet. This stage is bootstrap only.
