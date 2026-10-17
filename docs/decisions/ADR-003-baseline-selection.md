# ADR-003 — Baseline Model Selection: AASIST-L as Primary Candidate with LFCC-GMM Fallback

**Status:** Accepted — provisional pending checkpoint location and torchcontrib resolution
**Date:** 2026-10-17
**Supersedes:** N/A (new decision)
**Relates to:** ADR-001 (compute strategy), ADR-002 (MPS inference)
**Evidence source:** `docs/research_source_audit.md`

---

## Context

The source audit (2026-10-17) has established the following verified facts:

**AASIST-L (SR-003):**
- Repository: https://github.com/clovaai/aasist
- Licence: MIT (NAVER Corp.) — confirmed from LICENSE file
- Model size: 85,306 parameters (from README — small neural model)
- Requirements: torch>=1.6.0 (we have 2.14.0), torchcontrib, numpy, soundfile
- Published EER: 0.99% on ASVspoof 2019 LA eval (upstream result — NOT our result)
- Checkpoint: README states pretrained weights are provided; exact download path NOT found in raw repository files (no GitHub releases, no download script for weights)
- Compatibility risk: torchcontrib is imported at module level in main.py; this will block eval mode even if training is not run

**LFCC-GMM (SR-004):**
- Baseline code linked directly from asvspoof.org/index2019.html as a zip file
- Confirmed reachable; zip not downloaded (disk constraint)
- Language: predominantly MATLAB; Python re-implementations exist in the wild
- Expected behaviour: well-documented, CPU-compatible, deterministic
- No GPU required; expected EER ~8-9% on ASVspoof 2019 LA eval (published range)

---

## Decision: Outcome B

**AASIST-L remains a primary candidate but has unresolved blockers.**

Specifically:

1. **Checkpoint location is unresolved.** The README claims pretrained weights are available,
   but no download URL was found in the raw repository files, the GitHub Releases API returned
   an empty list, and `download_pretrained_models.sh` does not exist (HTTP 404).
   The checkpoint must be located before AASIST-L inference can be attempted.
   Do not assume the checkpoint exists until its file path, size, and SHA-256 hash are recorded.

2. **torchcontrib import risk.** `main.py` imports `torchcontrib.optim.SWA` at module level.
   This will raise ImportError in eval mode if torchcontrib is not installed or is incompatible
   with torch 2.14.0. Resolution options in order of preference:
   a. Install torchcontrib==0.0.2 and test for import errors on Python 3.10/torch 2.14.0.
   b. If torchcontrib fails, patch the single import line in main.py to use
      `torch.optim.swa_utils.AveragedModel` (native since torch 1.6) — document the patch.

3. **Fallback is LFCC-GMM on CPU.** If both blockers cannot be resolved within the project
   timeline, LFCC-GMM is the accepted fallback. Results from LFCC-GMM must never be
   labelled or presented as AASIST results.

---

## Rationale

- AASIST-L has 85,306 parameters. This is very small by neural network standards and
  is plausibly runnable on CPU or MPS after warm-up.
- The MIT licence permits use, modification, and distribution with attribution.
- The blockers are practical (checkpoint location, one import line), not fundamental.
- LFCC-GMM provides a guaranteed CPU-compatible fallback with well-understood behaviour.

---

## Constraints

- Do NOT install the upstream requirements.txt wholesale. Install only what is needed,
  after compatibility testing. Specifically: `torchcontrib==0.0.2` must be tested before
  adding it to the project's requirements.
- Do NOT train AASIST-L from scratch. Inference only, using a verified pretrained checkpoint.
- All inference results must be obtained from the ASVspoof 2019 LA eval partition only,
  using the official scoring tool (tDCF_python_v1.zip or equivalent).
- Published benchmark numbers from the AASIST paper are NOT project results.

---

## Immediate Next Steps (for implementation task)

1. Locate the AASIST-L pretrained checkpoint:
   - Check if weights are included in the repo itself (not visible from raw README; may be in models/weights/)
   - Search GitHub UI for releases or assets
   - Check if a Google Drive / OneDrive link is mentioned in issues or elsewhere in the repo
2. Test `pip install torchcontrib==0.0.2` in the project venv and verify import success.
3. If checkpoint found and torchcontrib works: run single-audio synthetic inference test.
4. Record checkpoint SHA-256 in docs/source_registry.md before using it.
5. If AASIST-L is unresolvable within one additional day: proceed with LFCC-GMM.

---

## Review Trigger

- Move to Outcome A if: checkpoint is located, SHA-256 recorded, torchcontrib imports without error, and a synthetic forward pass succeeds.
- Move to Outcome C if: checkpoint cannot be located after reasonable effort, OR torchcontrib is fatally incompatible with torch 2.14.0, OR disk space cannot be resolved before the dataset is needed.
