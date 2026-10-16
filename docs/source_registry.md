# Source Registry — PRJ_55: Deepfake Audio Detection

**Document status:** Active  
**Created:** 2026-10-16  
**Purpose:** Record every external dataset, code repository, and model checkpoint used or planned for this project. Each entry must be verified before the resource is used.

> **Verification status legend:**
> - `PENDING` — Not yet inspected; must not be used until verified.
> - `VERIFIED` — URL confirmed reachable; licence and access conditions confirmed; terms acceptable.
> - `BLOCKED` — Inaccessible, licence prevents use, or access was denied. See notes.
> - `ACCESS_REQUESTED` — Registration or request submitted; awaiting approval.

---

## SR-001 — ASVspoof 2019 Dataset (Logical Access)

| Field | Value |
|---|---|
| **Registry ID** | SR-001 |
| **Type** | Dataset |
| **Name** | ASVspoof 2019 (Logical Access sub-task) |
| **Official site** | https://www.asvspoof.org/index2019.html |
| **Paper** | Nautsch et al., "ASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized, Converted and Replayed Speech," IEEE TASLP, 2021 |
| **Expected contents** | Training, development, and evaluation audio (`.flac`); CM protocol trial lists; scoring scripts |
| **Licence / access** | Restricted — requires registration via the official site; redistribution not permitted |
| **Verification status** | **PENDING** |
| **Verified by** | — |
| **Verified date** | — |
| **Notes** | Registration URL must be confirmed in Week 1, Day 2. Do not download before access is confirmed. |

---

## SR-002 — ASVspoof 2021 Dataset

| Field | Value |
|---|---|
| **Registry ID** | SR-002 |
| **Type** | Dataset |
| **Name** | ASVspoof 2021 (Logical Access and Deepfake sub-tasks) |
| **Official site** | https://www.asvspoof.org/index2021.html |
| **Paper** | Liu et al., "ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild," IEEE TASLP, 2023 |
| **Expected contents** | Evaluation audio; protocol files; official baseline models |
| **Licence / access** | Restricted — requires separate registration; different from ASVspoof 2019 access |
| **Verification status** | **PENDING** |
| **Verified by** | — |
| **Verified date** | — |
| **Notes** | Lower priority than SR-001 for this project. Confirm access conditions before relying on this dataset. |

---

## SR-003 — AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks

| Field | Value |
|---|---|
| **Registry ID** | SR-003 |
| **Type** | Code repository + pretrained weights |
| **Name** | AASIST (official implementation) |
| **Official repository** | https://github.com/clovaai/aasist |
| **Paper** | Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022 |
| **Expected contents** | PyTorch model code; training/evaluation scripts; pretrained checkpoints for AASIST and AASIST-L |
| **Licence / access** | MIT Licence (as of publication; must be confirmed at time of use) |
| **Verification status** | **PENDING** |
| **Verified by** | — |
| **Verified date** | — |
| **Pretrained checkpoint** | Expected at repository releases or linked in README; SHA-256 hash not yet recorded |
| **Notes** | Confirm that (a) the MIT licence text is present in the repository, (b) the checkpoint is released under the same licence or a compatible one, and (c) CPU-only inference is supported. Record checkpoint SHA-256 here once verified. |

---

## SR-004 — ASVspoof 2019 Baseline System (LFCC-GMM)

| Field | Value |
|---|---|
| **Registry ID** | SR-004 |
| **Type** | Code / baseline system |
| **Name** | ASVspoof 2019 Official Baseline (LFCC-GMM) |
| **Location** | Expected to be bundled with the ASVspoof 2019 dataset release or linked from https://www.asvspoof.org/index2019.html |
| **Language** | MATLAB and/or Python (version to be confirmed) |
| **Licence / access** | To be confirmed; likely tied to dataset registration |
| **Verification status** | **PENDING** |
| **Verified by** | — |
| **Verified date** | — |
| **Notes** | Confirm whether a Python reimplementation exists and is citable. Record the exact commit or version used. |

---

## Changelog

| Date | Change | Author |
|---|---|---|
| 2026-10-16 | Initial registry created with SR-001 through SR-004 as PENDING | PRJ_55 team |
