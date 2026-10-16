# ADR-001 — Compute Strategy: CPU-First, Pretrained-Checkpoint Policy

**Status:** Accepted  
**Date:** 2026-10-16  
**Deciders:** PRJ_55 team  
**Context document:** `docs/project_charter.md` §6 (Constraints)

---

## Context

The project must be developed entirely on a laptop without a discrete GPU.  
The available hardware has not yet been benchmarked (see Week 1, Day 3 task in `docs/week1_execution_plan.md`).

The following facts motivate this decision:

1. Training a large anti-spoofing neural network (e.g., full AASIST) from scratch requires GPU compute that is not available.
2. Reproducing published ASVspoof 2019 LA results with LFCC-GMM is achievable on CPU within reasonable time, based on prior literature (exact timing not yet measured on this hardware).
3. A pretrained AASIST-L checkpoint, if available and verified, may allow evaluation-only inference on CPU, but inference speed has not been measured.

---

## Decision

**Prefer reproducible, CPU-compatible development throughout the project.**

Specifically:

1. **LFCC-GMM is the primary fallback baseline.** It requires no GPU and is well-documented. If no neural approach is feasible, the project can still produce a valid result using LFCC-GMM alone.

2. **A pretrained neural checkpoint (e.g., AASIST-L) may be used for inference only**, subject to all of the following conditions being satisfied:
   - The source repository is confirmed reachable and the code is under a compatible licence.
   - The checkpoint file has a documented provenance URL and a recorded SHA-256 hash.
   - The loading procedure has been tested and confirmed to work in the project's virtual environment.
   - The inference time on the available CPU has been measured and is feasible for the evaluation set size.
   - No condition above may be assumed or estimated; each must be verified and recorded in `docs/source_registry.md`.

3. **Training a large model from scratch is out of scope** for this project unless a compute upgrade becomes available and is documented in a new ADR.

4. **This decision does not assert that CPU inference will be fast.** Speed measurements are required before committing to any neural approach. If inference is too slow to complete within the project timeline, the decision reverts to LFCC-GMM as sole baseline.

---

## Consequences

### Positive
- The project remains executable regardless of GPU availability.
- A clear fallback prevents complete project failure if neural approaches prove infeasible.
- All decisions are backed by measurements, not assumptions.

### Negative / Risks
- Results may be less competitive than GPU-trained models.
- Inference time for AASIST-L on CPU is unknown; this is an open risk until Day 6.
- If the pretrained checkpoint licence is incompatible, the neural approach must be abandoned.

### Neutral
- Any deviation from this strategy requires a new ADR (ADR-002 or higher).

---

## Review Trigger

This decision should be reviewed if:
- GPU access becomes available.
- The CPU inference time measurement (Week 1, Day 6) shows that AASIST-L inference is infeasible.
- The AASIST-L checkpoint source is blocked or unlicensed.
