# Project Charter — PRJ_55: Deepfake Audio Detection

**Document status:** Draft — provisional until Week 1 literature and protocol review is complete.  
**Created:** 2026-10-16  
**Last updated:** 2026-10-16

---

## 1. Project Identity

| Field | Value |
|---|---|
| Project ID | PRJ_55 |
| Full title | Robust and Compute-Efficient Deepfake Speech Detection Under Unseen Synthesis Attacks and Real-World Channel Degradation |
| Duration | 30 days |
| Computing constraint | Laptop-only; hardware capability unverified |
| Current stage | 0 — bootstrap |

---

## 2. Objectives

1. **Primary:** Implement and evaluate at least one anti-spoofing countermeasure on a recognised benchmark (ASVspoof 2019 LA), producing reproducible, honest results.
2. **Secondary:** Assess generalisation under one or more controlled degradation conditions (e.g., additive noise, codec distortion), if feasible within the laptop constraint.
3. **Foundation:** Deliver a clean, documented, reproducible codebase that could be extended or submitted as a university project deliverable.

---

## 3. Scope

### In scope
- Binary speech anti-spoofing classification: bona fide vs. spoofed.
- Use of the ASVspoof 2019 Logical Access (LA) protocol and its evaluation metrics.
- Evaluation of one or two baseline systems (LFCC-GMM and/or AASIST-L), subject to feasibility.
- A controlled robustness experiment, if laptop resources permit.
- Documentation sufficient to reproduce all results from scratch.

### Out of scope
- Full training of large neural models from scratch (compute prohibitive on laptop).
- Physical Access (PA) or Deepfake (DF) sub-tasks (may be noted but not evaluated).
- Real-time or deployment-ready inference systems.
- Custom TTS or voice-conversion system development.
- Automatic speaker verification (ASV) — only the countermeasure (CM) component.
- Any collection, storage, or use of private speech recordings without explicit permission.
- Any web or mobile front-end.

---

## 4. Research Questions

> **RQ-1 (Primary — provisional):** Does a system trained/evaluated on ASVspoof 2019 LA show meaningful degradation in EER when evaluated on utterances processed by unseen synthesis attacks compared to seen attacks?

> **RQ-2 (Secondary — provisional):** How does performance change under a single controlled channel degradation condition (e.g., additive babble noise at a fixed SNR) relative to the clean evaluation condition?

Both questions are provisional. They will be confirmed, revised, or replaced after the Week 1 literature and protocol review.

---

## 5. Hypotheses to Test

| ID | Hypothesis | Status |
|---|---|---|
| H-1 | LFCC-GMM, as implemented in the ASVspoof 2019 baseline, achieves an EER close to the published figure on the LA evaluation partition. | **Unverified** — requires environment and data access |
| H-2 | AASIST-L achieves lower EER than LFCC-GMM on the same partition using a pretrained checkpoint. | **Unverified** — requires checkpoint provenance and environment verification |
| H-3 | Both systems show measurable EER increase under at least one controlled degradation condition. | **Speculative** — depends on feasibility of robustness experiment |

---

## 6. Constraints

| Constraint | Detail |
|---|---|
| Hardware | Laptop CPU only; RAM and disk capacity not yet measured |
| Time | 30 days total; Week 1 is setup and feasibility only |
| No GPU | Training large models from scratch is not feasible |
| Dataset access | ASVspoof datasets require registration; access not confirmed |
| Budget | No paid compute or API budget available |
| Licence compliance | All datasets and models must be used within their stated terms |

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dataset access denied or delayed | Medium | High | Begin registration immediately in Week 1; identify mirror or alternative |
| Laptop insufficient for AASIST-L inference | Medium | Medium | Fall back to LFCC-GMM alone; document as finding |
| Pretrained checkpoint unavailable or incompatible | Medium | Medium | Document source verification failure; plan CPU re-training of small model |
| Benchmark protocol misunderstood | Low | High | Follow official scoring scripts; read ASVspoof evaluation plan carefully |
| Scope creep | Medium | Medium | Strict adherence to this charter; changes require ADR |

---

## 8. Success Criteria

The project succeeds if, at the end of 30 days, the following are all true:

1. At least one baseline system produces a reported EER on ASVspoof 2019 LA evaluation that is reproducible from this repository.
2. All results are accompanied by the exact software environment, random seed (where applicable), and command used to produce them.
3. The repository contains no tracked datasets, audio files, model checkpoints, or secrets.
4. The experiment log records every result honestly, including failures and limitations.
5. The research question has been answered or the reason it could not be answered has been documented.

Partial success is acceptable and must be documented honestly. Invented or unverified results are never acceptable.
