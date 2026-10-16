# scripts/

This directory holds utility and automation scripts.

Scripts should be short, single-purpose, and documented with a usage comment at the top.

## Convention

```text
scripts/
├── README.md                  ← This file
├── verify_environment.sh      ← (planned) Check Python version and key packages
├── download_asvspoof.sh       ← (planned) Dataset download helper with checksum verification
└── score_eval.py              ← (planned) Wrap the official ASVspoof scoring script
```

## Rules

- Scripts must not embed absolute paths, credentials, or dataset download tokens.
- Each script must print a usage message when run with `--help` or no arguments.
- Add a corresponding entry to `docs/experiment_log.md` whenever a script is used to produce results.

## Current status

No scripts have been created yet.
