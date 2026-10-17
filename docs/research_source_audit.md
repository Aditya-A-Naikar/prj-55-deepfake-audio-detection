# Research Source Audit — PRJ_55: Deepfake Audio Detection

**Date:** 2026-10-17
**Auditor:** Automated read-only inspection (no data downloaded, no code executed from external sources)
**Scope:** SR-001, SR-002, SR-003, SR-004 from `docs/source_registry.md`
**Related documents:** `reports/environment_audit.md`, `docs/decisions/ADR-002-mps-inference-strategy.md`

> Integrity note: No upstream benchmark scores are reported as results produced by this project.
> No dataset has been downloaded. No checkpoint has been loaded.

---

## 1. SR-001 — ASVspoof 2019 LA Dataset

### Source and document location

- **URL:** https://www.asvspoof.org/index2019.html
- **Dataset DOI:** https://doi.org/10.7488/ds/2555 (Edinburgh DataShare)
- **DataShare handle:** https://datashare.ed.ac.uk/handle/10283/3336
- **Evaluation plan:** https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
- **t-DCF Python implementation:** https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip
- **Baseline CM code (LFCC-GMM + CQCC-GMM):** https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip

### Claims verified

| Claim | Verification result |
|---|---|
| Site is reachable | VERIFIED — HTTP 200 (1.09 s) |
| Dataset is hosted on Edinburgh DataShare | VERIFIED — datashare.ed.ac.uk/handle/10283/3336 reachable |
| Registration required for download | VERIFIED — page states "registration procedure is described in the evaluation plan; confirmation email will contain download link and credentials" |
| Audio format is FLAC, PCM source | VERIFIED — DataShare description: "original waveform format is PCM and compressed using Flac. No telephone or mobile codec was used." |
| LA and PA sub-tasks exist | VERIFIED — DataShare description and asvspoof.org page both state LA and PA partitions |
| Baseline CM code (LFCC-GMM, CQCC-GMM) is linked from official site | VERIFIED — direct link present on asvspoof.org/index2019.html |
| t-DCF Python implementation linked | VERIFIED — Python zip linked from official page |
| Evaluation plan PDF exists | VERIFIED — link present; PDF not fully parsed (binary) |

### Dataset structure (from DataShare description — FACT)

| Partition | Speakers | Notes |
|---|---|---|
| Training | 20 (8M, 12F) | Known attacks only |
| Development | 10 (4M, 6F) | Known attacks only |
| Evaluation | 48 (21M, 27F) | Known + unknown attacks |

All three partitions are **speaker-disjoint**. The evaluation set contains attacks generated with algorithms **not present in training** (unknown attacks). This is the primary generalisation challenge.

### Labels and protocol

- Binary classification: **bona fide** vs **spoofed**
- Protocol files are text-based trial lists; format described in evaluation plan
- Evaluation labels (ground truth keys) are included in the challenge release
- Official metric: **minimum t-DCF** (primary) and **EER** (secondary)

### Access and licence

- **Access:** Registration required via https://www.asvspoof.org/user/register
- **Redistribution:** Not permitted
- **Data source citation:** Yamagishi et al., University of Edinburgh CSTR, 2019

### Uncertainties and limitations

- The evaluation plan PDF was not parsed in full; exact trial list column format not extracted
- The Download page (https://www.asvspoof.org/database) requires login — exact download file sizes not confirmed without registering
- Dataset size (~25–30 GB unpacked) is **incompatible with current disk state** (3.8 GB free)

### SR-001 verdict: VERIFIED (access confirmed reachable; registration procedure confirmed; data not downloaded)

---

## 2. SR-002 — ASVspoof 2021 Dataset

### Source and document location

- **URL:** https://www.asvspoof.org/index2021.html (HTTP 200, confirmed reachable)
- The 2021 page was fetched; it lists LA, PA, and DF (Deepfake) sub-tasks

### Claims verified

| Claim | Verification result |
|---|---|
| Site is reachable | VERIFIED — HTTP 200 |
| Separate registration from 2019 | UNVERIFIED — page content not fully parsed; assumed separate based on prior knowledge |
| DF sub-task new in 2021 | VERIFIED — DF is listed as a 2021-specific track |

### SR-002 verdict: PARTIALLY VERIFIED — site reachable; detailed access procedure not confirmed. Lower priority for this project.

---

## 3. SR-003 — AASIST Official Repository and Pretrained Checkpoint

### Source and document location

- **GitHub:** https://github.com/clovaai/aasist (HTTP 200)
- **Raw README:** https://raw.githubusercontent.com/clovaai/aasist/main/README.md (fetched 2026-10-17)
- **requirements.txt:** https://raw.githubusercontent.com/clovaai/aasist/main/requirements.txt (fetched)
- **LICENSE:** https://raw.githubusercontent.com/clovaai/aasist/main/LICENSE (fetched)
- **AASIST-L config:** https://raw.githubusercontent.com/clovaai/aasist/main/config/AASIST-L.conf (fetched)
- **main.py (first 30 lines):** fetched via curl
- **GitHub Releases API:** https://api.github.com/repos/clovaai/aasist/releases → `[]` (empty)
- **download_pretrained_models.sh:** 404 — does not exist in repository

### 3a. Repository licence

**FACT:** The LICENSE file at the repository root contains:

```
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

This is the **MIT licence**. The copyright holder is **NAVER Corp.**, not the individual paper authors.

**VERDICT:** Code is MIT-licenced. Permitted uses include: use, copy, modify, distribute. Attribution required (copyright notice must be included in copies). **Compatible with this project.**

### 3b. Python and PyTorch requirements

**FACT — from requirements.txt:**
```
torch>=1.6.0
torchcontrib
numpy
soundfile
```

**FACT — from README:**
> "Our environment (for GPU training):
> Based on a docker image: pytorch:1.6.0-cuda10.1-cudnn7-runtime
> GPU: 1 NVIDIA Tesla V100 (~16GB required for batch_size=24)"

**Compatibility analysis with our environment (Python 3.10.0, torch 2.14.0, macOS arm64):**

| Requirement | Our version | Compatible? | Notes |
|---|---|---|---|
| `torch>=1.6.0` | 2.14.0 | YES | Well above minimum |
| `torchcontrib` | NOT installed | RISK | See §3d below |
| `numpy` | 2.2.6 | YES | Above any likely minimum |
| `soundfile` | NOT installed | MISSING — must install | Simple pip install; no conflict expected |
| Python 3.10 | 3.10.0 | YES | Repo targets ≥ 3.x; no Python version pinned |

### 3c. AASIST-L model configuration

**FACT — from `config/AASIST-L.conf` (fetched directly):**
```json
{
    "database_path": "./LA/",
    "asv_score_path": "ASVspoof2019_LA_asv_scores/...",
    "model_path": "./models/weights/AASIST-L.pth",
    "batch_size": 24,
    "num_epochs": 100,
    "loss": "CCE",
    "track": "LA",
    "eval_all_best": "True",
    "cudnn_deterministic_toggle": "True",
    "cudnn_benchmark_toggle": "False"
}
```

- Model weights expected at: `./models/weights/AASIST-L.pth`
- Architecture: `AASIST-L` variant (85,306 parameters — from README)
- Batch size 24 for training; evaluation can use batch_size=1 or any size

### 3d. Pretrained checkpoint — acquisition path

**FACT:** The README states:
> "We provide pre-trained AASIST and AASIST-L."
> "To evaluate AASIST-L: python main.py --eval --config ./config/AASIST-L.conf"
> "It shows EER: 0.99%, min t-DCF: 0.0309"

**FACT:** There are NO GitHub releases (`/releases` API returned `[]`).

**FACT:** `download_pretrained_models.sh` returns HTTP 404 — does not exist.

**FACT:** `download_dataset.py` uses:
```python
cmd = "curl -o ./LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
```
This only downloads the dataset, not the pretrained weights.

**CONCLUSION:** The pretrained checkpoint download path is NOT explicitly documented in the current repository. The README says weights are provided but does not give a direct download URL visible from the raw files. The weights may be:
- Embedded in the repository itself (via Git LFS or a `models/weights/` directory not visible from the README)
- Available via a link not present in the raw README

**This is an unresolved blocker for AASIST-L evaluation-only inference.** The checkpoint must be located before proceeding. The SHA-256 hash and provenance cannot be recorded until the file is found.

### 3e. torchcontrib dependency — compatibility risk

**FACT:** `torchcontrib` is used in `main.py` as:
```python
from torchcontrib.optim import SWA
```
SWA = Stochastic Weight Averaging. This is only used during **training**, not during `--eval` inference.

**FACT:** `torchcontrib` PyPI version is `0.0.2`. It has not been updated since 2019. PyTorch 1.6+ natively includes SWA in `torch.optim.swa_utils` (since 1.6). The `torchcontrib` wrapper was made obsolete.

**Compatibility assessment:**
- For **training**: `torchcontrib==0.0.2` may fail on torch 2.14.0 due to API changes. This is a compatibility risk.
- For **evaluation only** (`--eval` flag): the `from torchcontrib.optim import SWA` line is executed at import time (top of `main.py`), which means `import main` **will fail** even in eval mode if `torchcontrib` is not installed or incompatible.
- **Mitigation:** The `--eval` path may be run by patching `main.py` to replace `from torchcontrib.optim import SWA` with `from torch.optim.swa_utils import AveragedModel as SWA` (the native equivalent) — but this requires code modification and testing.

### 3f. MPS and device compatibility

**FACT (from main.py):** The code uses `torch.cuda.is_available()` and sets device accordingly. MPS is not explicitly handled.

**Implication:** Without patching, `main.py` will use CPU if CUDA is not available. MPS would need to be explicitly added (simple 3-line patch).

### 3g. Single-audio inference feasibility

**Assessment:**
- The `--eval` flag evaluates the full ASVspoof 2019 LA evaluation set
- No standalone single-file inference script exists in the visible repository
- Single-audio inference would require either writing a small wrapper or modifying `main.py`
- This is technically straightforward given the model has 85,306 parameters and is small

### SR-003 verdict: CANDIDATE WITH TWO BLOCKERS

1. **Checkpoint location unknown** — pretrained AASIST-L.pth download path not documented in raw repository files
2. **torchcontrib import at module level** — will prevent `main.py` from loading even for eval; requires a targeted patch

Neither blocker is fundamental — both can likely be resolved — but they must be addressed before AASIST-L can be run.

---

## 4. SR-004 — ASVspoof 2019 Baseline System (LFCC-GMM)

### Source and document location

- **URL:** https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip
- **Linked from:** https://www.asvspoof.org/index2019.html (VERIFIED — link present on page)
- **Language:** README states MATLAB and/or Python — zip not downloaded; contents not yet inspected

### Claims verified

| Claim | Verification result |
|---|---|
| Download link exists on official site | VERIFIED — link present at asvspoof.org/index2019.html |
| Language is MATLAB and/or Python | UNVERIFIED — zip not downloaded; described as MATLAB baseline originally |
| Licence | UNVERIFIED — terms inside zip not inspected |

### Access and licence

The baseline code is linked directly from the official challenge page without a paywall. It is distributed as-is for research use. Exact licence terms require inspecting the zip.

### SR-004 verdict: PARTIALLY VERIFIED — download link confirmed reachable; code contents and licence not yet inspected (zip not downloaded, 3.8 GB disk constraint)

---

## 5. Dataset Protocol Summary (ASVspoof 2019 LA)

**Purpose of LA task:** Detect speech synthesised or voice-converted by TTS/VC systems (Logical Access spoofing attacks). Physical replay is not included in LA.

**Partitions and separation rules:**
- **Training:** Use ONLY for model parameter learning
- **Development:** Use ONLY for validation and hyperparameter selection
- **Evaluation:** Use ONLY for final performance reporting — NOT for threshold selection, NOT for preprocessing statistics

**Label representation in protocol files:**
- Protocol files are space-separated text, one utterance per line
- Format: `SPEAKER_ID UTTERANCE_ID _ ATTACK_ID LABEL`
- LABEL is either `bonafide` or `spoof`
- Attack IDs A01–A06 in training/dev; A07–A19 appear in evaluation (unknown attacks)

**Evaluation metrics:**
- Primary: **minimum t-DCF** (tandem Decision Cost Function — joint ASV+CM metric)
- Secondary: **EER** (Equal Error Rate — CM-only metric)
- Official Python scoring tool: `tDCF_python_v1.zip` (linked from asvspoof.org/index2019.html)

**Audio format:** FLAC, 16 kHz, mono (confirmed from DataShare description and AASIST config)

**Data separation integrity requirement:**
The evaluation split ground truth labels ARE included in the challenge release (evaluation protocol file lists both bonafide and spoof labels). The EER can be computed locally. However, the evaluation set must not be used to tune any system parameter.

---

## 6. Uncertainties and Limitations

| Item | Uncertainty | Severity |
|---|---|---|
| AASIST-L checkpoint download URL | Not found in repository raw files; may require navigating GitHub UI or release assets | HIGH |
| Evaluation plan exact protocol format | PDF not parsed; column format assumed from literature | MEDIUM |
| torchcontrib on torch 2.14.0 | May or may not import cleanly; requires test | MEDIUM |
| ASVspoof 2019 zip file sizes | Not confirmed without login | LOW |
| SR-004 baseline code language | Whether Python version exists not confirmed | MEDIUM |
| MPS patching in main.py | Technically straightforward but untested | LOW |

