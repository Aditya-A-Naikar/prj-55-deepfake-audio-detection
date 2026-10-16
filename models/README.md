# models/

This directory holds pretrained model checkpoints and exported model files.

**This directory is excluded from Git** (see `.gitignore`). Only this README is tracked.

## Convention

```text
models/
├── README.md        ← This file (tracked)
├── checkpoints/     ← Downloaded or trained weight files (.pt, .pth, .ckpt, etc.)
└── exports/         ← Converted or exported models (e.g., ONNX)
```

## Rules

- Never commit weight files to Git unless explicitly authorised by an ADR.
- Every file placed here must be recorded in `docs/source_registry.md` with its provenance URL, SHA-256 hash, and licence status.
- Verify the licence of any pretrained checkpoint before loading it.

## Current status

No model checkpoints have been downloaded yet. See `docs/source_registry.md` (SR-003, SR-004) and `docs/decisions/ADR-001-compute-strategy.md` for the checkpoint policy.
