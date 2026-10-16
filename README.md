# PRJ_55 — Deepfake Audio Detection

**Working title:**
> Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation

---

## ⚠️ Status Notice

**No project-specific results have been produced yet.**  
This repository is at Stage 0: workspace and documentation bootstrap only.  
No experiments have been run, no models have been loaded, and no accuracy numbers are available.

---

## 1. Problem Statement

The task is **binary speech anti-spoofing**: classifying an utterance as either

| Label | Meaning |
|---|---|
| **Bona fide** | Genuine human speech |
| **Spoofed** | Synthesised, voice-converted, or replayed speech |

Modern text-to-speech (TTS) and voice-conversion (VC) systems produce output that is acoustically indistinguishable from genuine speech to human listeners. Automatic speaker verification (ASV) systems are therefore vulnerable to presentation attacks. This project investigates countermeasure (CM) systems designed to detect such attacks.

---

## 2. Preliminary Research Question

> Can a countermeasure system that is trained or evaluated on ASVspoof 2019 LA maintain acceptable detection performance when (a) the synthesis method is unseen at evaluation time and (b) the audio is degraded by realistic channel or environmental distortions, while remaining feasible to develop on a laptop without GPU access?

This question is **provisional** and will be refined after the literature and benchmark protocols have been reviewed in Week 1.

---

## 3. Timeline and Current Milestone

| Horizon | Period | Status |
|---|---|---|
| Full project | 30 days | Planned |
| **Week 1** | Days 1–7 | **Current milestone** |
| Week 2–4 | Days 8–30 | Not started |

**Week 1 objective:** Establish a reproducible research foundation — verify sources, confirm the development environment is functional on the available laptop, validate benchmark protocol access, and produce a feasibility assessment for the planned baselines.

---

## 4. Planned Candidate Approaches

The following approaches are **candidates only**, pending source verification and environment confirmation:

| Approach | Type | Compute profile | Source to verify |
|---|---|---|---|
| **LFCC-GMM** | Classical feature + generative model | CPU-compatible | ASVspoof 2019 baseline release |
| **AASIST-L** | Graph-attention neural network (light variant) | Pretrained inference may be CPU-compatible | Jung et al., ICASSP 2022 / official GitHub |

Neither approach will be used until its source, licence, dependencies, and loading procedure have been confirmed.

---

## 5. Planned Evaluation Dimensions

Subject to dataset access and feasibility:

1. **Equal Error Rate (EER)** — primary detection metric used in ASVspoof benchmarks.
2. **Threshold-dependent errors** — false alarm rate (FAR) and miss detection rate (MDR) at a chosen operating point.
3. **Controlled robustness experiment** — if feasible within the laptop constraint, evaluate performance under at least one form of controlled channel degradation (e.g., additive noise or codec distortion).

---

## 6. Repository Layout

```text
prj-55-deepfake-audio-detection/
├── README.md                        ← This file
├── .gitignore                       ← Excludes datasets, weights, secrets
├── .gitattributes                   ← Line-ending policy
├── pyproject.toml                   ← Package metadata and test config
├── docs/
│   ├── project_charter.md           ← Objectives, scope, risks, success criteria
│   ├── week1_execution_plan.md      ← Day-by-day task plan with exit criteria
│   ├── experiment_log.md            ← Blank template; filled when experiments run
│   ├── data_governance.md           ← Rules for dataset and weight handling
│   ├── source_registry.md           ← External sources with verification status
│   └── decisions/
│       └── ADR-001-compute-strategy.md
├── data/                            ← NOT tracked by Git (see .gitignore)
├── models/                          ← NOT tracked by Git (see .gitignore)
├── configs/                         ← Experiment configuration files
├── scripts/                         ← Utility and automation scripts
├── src/
│   └── prj55/                       ← Main Python package
│       └── __init__.py
├── tests/
│   └── test_project_scaffold.py     ← Scaffold integrity checks
└── reports/                         ← Generated outputs (figures, tables)
```

> **`data/` and `models/` are excluded from Git** via `.gitignore`. Only their `README.md` placeholder files are tracked.

---

## 7. Installation

> ⚠️ Installation instructions will be provided after the development environment has been verified on the target hardware and the correct dependency versions have been confirmed.

No Python dependencies should be installed from this README before completing Week 1, Day 4 (environment setup).

---

## 8. Contributing and Conventions

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- Every experiment must be logged in `docs/experiment_log.md` before results are reported.
- No dataset files, audio recordings, model checkpoints, or secrets may be committed to this repository.

---

## 9. Licence

Licence selection for code produced in this project is **deferred** pending review of dependency licences and the intended use of project outputs. No licence file has been created.

---

*Last updated: 2026-10-16 | Stage 0 bootstrap only.*
