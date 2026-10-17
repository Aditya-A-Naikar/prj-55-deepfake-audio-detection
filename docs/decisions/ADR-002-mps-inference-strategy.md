# ADR-002 — Revised Inference Device Strategy: MPS-Preferred on Apple M2

**Status:** Accepted
**Date:** 2026-10-17
**Supersedes / amends:** ADR-001 (`docs/decisions/ADR-001-compute-strategy.md`)
**Deciders:** PRJ_55 team
**Evidence source:** `reports/environment_audit.md`

---

## Context

ADR-001 established a "CPU-first" strategy on the grounds that GPU availability was unverified.

The Week 1 hardware audit (`reports/environment_audit.md`, 2026-10-17) confirmed:

- Hardware: Apple M2, Mac14,15, unified memory architecture.
- Metal Support: Metal 4.
- `torch.backends.mps.is_available()` — True
- `torch.backends.mps.is_built()` — True
- PyTorch version: 2.14.0 (already installed in system Python 3.10).

MPS (Metal Performance Shaders) provides hardware-accelerated tensor operations via the M2
integrated GPU. This is not a discrete GPU and does not support CUDA, but it is a meaningful
upgrade over CPU-only inference for neural network forward passes.

---

## Decision

**Revise the inference device preference from CPU-only to MPS-preferred, with CPU as fallback.**

Specifically:

1. **Neural model inference (e.g., AASIST-L):** Use `device='mps'` when MPS is available.
   Fall back to `device='cpu'` if MPS produces incorrect outputs or is unavailable in the venv.

2. **Feature extraction (LFCC and similar):** Remain on CPU. MPS provides no benefit for
   NumPy/SciPy-based signal processing pipelines not using PyTorch tensors throughout.

3. **Device selection must be runtime-detected, not hardcoded:**
   ```python
   import torch

   def get_device() -> torch.device:
       if torch.backends.mps.is_available():
           return torch.device("mps")
       return torch.device("cpu")
   ```

4. **Cold-start overhead must be handled:** The audit measured 259 ms for the first MPS
   matmul (Metal shader JIT compilation). Any benchmark must include a warm-up forward pass
   before timing measurements are recorded.

5. **MPS speed for AASIST-L has not yet been benchmarked** with real model shapes and audio
   inputs. This decision does NOT assert that MPS inference is fast enough to evaluate the full
   ASVspoof 2019 LA evaluation set within the project timeline. That measurement is required
   on Week 1 Day 6.

6. **ADR-001 condition 4 (inference time measurement) still applies.** If MPS inference is
   not fast enough to complete evaluation within the timeline, the fallback remains
   LFCC-GMM on CPU.

---

## Consequences

### Positive
- AASIST-L forward passes will likely be faster than CPU-only after warm-up.
- No additional hardware or configuration required — MPS is already enabled.
- PyTorch 2.14.0 has mature MPS support.

### Negative / Risks
- MPS does not support all PyTorch operations. An unsupported op triggers a runtime error;
  fallback is CPU.
- MPS results may differ numerically from CPU results at low precision. Verify EER matches
  within tolerance between devices before reporting final numbers.
- `torchaudio` is not yet installed. End-to-end MPS pipeline cannot be confirmed until the
  full audio environment is set up.

### Neutral
- CUDA-specific code (e.g., `tensor.cuda()`) will fail. All device references must use
  the `get_device()` pattern above.

---

## Review Trigger

This decision should be reviewed if:
- MPS produces NaN or incorrect outputs during AASIST-L inference.
- The Day 6 benchmark shows MPS is not materially faster than CPU for this model.
- A CUDA-capable GPU becomes available.
