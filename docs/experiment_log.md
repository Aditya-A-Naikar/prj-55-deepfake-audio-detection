# Experiment Log — PRJ_55: Deepfake Audio Detection

**Template version:** 1.0  
**Created:** 2026-10-16  
**Status:** No experiments have been run yet. This file contains the blank template only.

> Fill one section per experiment. Never edit a completed experiment entry; add a new entry if corrections are needed and reference the original by ID.

---

## How to Use This Log

1. Copy the template below for each new experiment.
2. Fill every field **before** starting the experiment.
3. Record results **immediately** after they are produced — do not rely on memory.
4. Record failures and partial results with the same rigour as successes.
5. Leave result fields as `N/A` only when a field genuinely does not apply; never leave them blank to conceal a failure.

---

## Experiment Template

---

### EXP-000 — [Short descriptive title]

| Field | Value |
|---|---|
| **Experiment ID** | EXP-000 |
| **Date** | YYYY-MM-DD |
| **Experimenter** | [Name] |

#### Environment

| Field | Value |
|---|---|
| OS | e.g., macOS 14.5, Python 3.11.9 |
| CPU | e.g., Apple M1, 8-core, 8 GB RAM |
| GPU | None / [model if available] |
| Python | [exact version, e.g., 3.11.9] |
| Key packages | [e.g., torch 2.3.0+cpu, numpy 1.26.4] |
| Virtual environment | [path or name] |

#### Dataset

| Field | Value |
|---|---|
| Dataset name | e.g., ASVspoof 2019 LA |
| Version / release | e.g., V1.0, downloaded from [URL] |
| Partition used | e.g., `eval`, `dev`, `train` |
| Exact trial list | e.g., `ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt` |
| Total utterances | [number] |
| Bona fide / Spoofed split | [number bona fide] / [number spoofed] |

#### Model

| Field | Value |
|---|---|
| Model name | e.g., LFCC-GMM, AASIST-L |
| Checkpoint / weights | [filename, SHA-256 hash, and provenance URL] |
| Training status | Pretrained / Trained from scratch |
| Training data (if trained here) | N/A or [details] |
| Architecture notes | [key hyperparameters or configuration file] |

#### Preprocessing

| Field | Value |
|---|---|
| Feature type | e.g., LFCC (60-dim), raw waveform |
| Sampling rate | e.g., 16 000 Hz |
| Duration handling | e.g., padded/truncated to 4 s |
| Augmentation | None / [details] |
| Config file | [path to config file, if any] |

#### Reproducibility

| Field | Value |
|---|---|
| Random seed | [integer or N/A] |
| Deterministic mode | Yes / No / Not applicable |
| Reproduction command | `python scripts/run_eval.py --config configs/exp000.yaml` |

---

#### Results

> **Leave this section blank until results are genuinely available.**

| Metric | Value | Notes |
|---|---|---|
| EER (%) | — | — |
| FAR (%) @ threshold | — | — |
| MDR (%) @ threshold | — | — |
| min-tDCF | — | — |
| Inference time | — | — |

#### Failures and Limitations

_None reported / [Description of any errors, instabilities, or incomplete runs]_

#### Observations and Next Steps

_[Brief notes on what was learned and what should be done next]_

---

<!-- Append new experiments below this line. Do not delete the template above. -->

---

*End of experiment log. No experiments have been run at Stage 0.*
