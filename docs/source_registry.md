# Source Registry — PRJ_55: Deepfake Audio Detection

**Document status:** Active
**Created:** 2026-10-16
**Last updated:** 2026-10-17 (Task 3 source audit)
**Purpose:** Record every external dataset, code repository, and model checkpoint used or planned
for this project. Each entry must be verified before the resource is used.

> **Verification status legend:**
> - `PENDING` — Not yet inspected; must not be used until verified.
> - `VERIFIED` — URL confirmed reachable; licence and access conditions confirmed; terms acceptable.
> - `BLOCKED` — Inaccessible, licence prevents use, or access was denied. See notes.
> - `ACCESS_REQUESTED` — Registration or request submitted; awaiting approval.
> - `CANDIDATE_BLOCKED` — Reachable but has an unresolved technical blocker before use.

---

## SR-001 — ASVspoof 2019 Dataset (Logical Access)

| Field | Value |
|---|---|
| **Registry ID** | SR-001 |
| **Type** | Dataset |
| **Name** | ASVspoof 2019 (Logical Access sub-task) |
| **Official site** | https://www.asvspoof.org/index2019.html |
| **DOI / DataShare** | https://doi.org/10.7488/ds/2555 |
| **DataShare handle** | https://datashare.ed.ac.uk/handle/10283/3336 |
| **Paper** | Nautsch et al., "ASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized, Converted and Replayed Speech," IEEE TASLP, 2021 |
| **Audio format** | FLAC (PCM source, no codec), 16 kHz, mono |
| **Partitions** | Training (20 spk), Development (10 spk), Evaluation (48 spk) — speaker-disjoint |
| **Labels** | bonafide / spoof; protocol files are space-separated text |
| **Evaluation metric** | min t-DCF (primary), EER (secondary) |
| **Scoring tool** | tDCF_python_v1.zip — https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip |
| **Baseline CM code** | ASVspoof_2019_baseline_CM_v1.zip — https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip |
| **Licence / access** | Restricted — registration required; redistribution not permitted |
| **Registration URL** | https://www.asvspoof.org/user/register |
| **Verification status** | **VERIFIED** (site and DataShare reachable; access procedure confirmed; data not yet downloaded) |
| **Verified by** | PRJ_55 automated audit |
| **Verified date** | 2026-10-17 |
| **Data downloaded** | NO — disk constraint (3.8 GB free; dataset requires ~25-30 GB) |
| **Notes** | Free disk space must be resolved before download. Evaluation split must not be used for model selection or threshold tuning. |

---

## SR-002 — ASVspoof 2021 Dataset

| Field | Value |
|---|---|
| **Registry ID** | SR-002 |
| **Type** | Dataset |
| **Name** | ASVspoof 2021 (Logical Access and Deepfake sub-tasks) |
| **Official site** | https://www.asvspoof.org/index2021.html |
| **Paper** | Liu et al., "ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild," IEEE TASLP, 2023 |
| **Licence / access** | Restricted — requires separate registration |
| **Verification status** | **PARTIALLY VERIFIED** — site reachable (HTTP 200); detailed access procedure not confirmed |
| **Verified by** | PRJ_55 automated audit |
| **Verified date** | 2026-10-17 |
| **Notes** | Lower priority than SR-001 for this project. Confirm access conditions before relying on this dataset. |

---

## SR-003 — AASIST Official Repository and Pretrained Weights

| Field | Value |
|---|---|
| **Registry ID** | SR-003 |
| **Type** | Code repository + pretrained weights |
| **Name** | AASIST (official implementation) |
| **Official repository** | https://github.com/clovaai/aasist |
| **Paper** | Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022 |
| **ArXiv** | https://arxiv.org/abs/2110.01200 |
| **Licence** | MIT Licence — Copyright (c) 2021-present NAVER Corp. (VERIFIED from LICENSE file) |
| **Python requirement** | Not pinned in requirements.txt; torch>=1.6.0 (we have 2.14.0) |
| **Dependencies** | torch>=1.6.0, torchcontrib==0.0.2, numpy, soundfile |
| **Model size (AASIST-L)** | 85,306 parameters |
| **Published EER (AASIST-L)** | 0.99% on ASVspoof 2019 LA eval — upstream result, NOT our result |
| **Checkpoint path in config** | ./models/weights/AASIST-L.pth |
| **Checkpoint download URL** | **UNKNOWN — not found in raw repository files** |
| **GitHub Releases** | Empty — API returned [] |
| **download_pretrained_models.sh** | Does not exist (HTTP 404) |
| **Checkpoint SHA-256** | Not yet recorded |
| **Checkpoint downloaded** | NO |
| **Verification status** | **CANDIDATE_BLOCKED** — code and licence verified; checkpoint acquisition path unresolved; torchcontrib module-level import untested on torch 2.14.0 |
| **Verified by** | PRJ_55 automated audit |
| **Verified date** | 2026-10-17 |
| **Notes** | See docs/research_source_audit.md §3 and ADR-003 for detailed blocker analysis and resolution steps. Do not use until checkpoint is located and SHA-256 recorded. |

---

## SR-004 — ASVspoof 2019 Baseline System (LFCC-GMM)

| Field | Value |
|---|---|
| **Registry ID** | SR-004 |
| **Type** | Code / baseline system |
| **Name** | ASVspoof 2019 Official Baseline (LFCC-GMM and CQCC-GMM) |
| **Download URL** | https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip |
| **Linked from** | https://www.asvspoof.org/index2019.html (link confirmed present) |
| **Language** | Predominantly MATLAB; Python re-implementations exist |
| **Licence / access** | To be confirmed from zip contents; distributed openly from challenge site |
| **Downloaded** | NO — disk constraint |
| **Verification status** | **PARTIALLY VERIFIED** — download link confirmed reachable; code contents and licence not yet inspected |
| **Verified by** | PRJ_55 automated audit |
| **Verified date** | 2026-10-17 |
| **Notes** | CPU-compatible. Confirmed as fallback if AASIST-L remains blocked (see ADR-003). Python reimplementations may exist in scikit-learn ecosystem. |

---

## Changelog

| Date | Change | Author |
|---|---|---|
| 2026-10-16 | Initial registry created with SR-001 through SR-004 as PENDING | PRJ_55 team |
| 2026-10-17 | Updated SR-001 to VERIFIED; SR-002 to PARTIALLY VERIFIED; SR-003 to CANDIDATE_BLOCKED; SR-004 to PARTIALLY VERIFIED after Task 3 source audit | PRJ_55 automated audit |
