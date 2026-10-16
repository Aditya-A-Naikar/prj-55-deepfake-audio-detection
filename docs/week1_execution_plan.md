# Week 1 Execution Plan — PRJ_55

**Period:** Days 1–7 (2026-10-16 to 2026-10-22)  
**Goal:** Establish a reproducible research foundation; produce a feasibility assessment.  
**Status:** In progress (Day 1 underway)

> All tasks in this plan are bounded by the laptop-only compute constraint.  
> No model training, dataset download, or frontend work is performed this week.

---

## Day 1 — Workspace and Repository Bootstrap

### Task
- Create the project directory and Git repository.
- Write all initial documentation (README, charter, this plan, governance, source registry, ADR-001).
- Establish `.gitignore` and `.gitattributes`.
- Create the scaffold test and verify it passes.
- Make the initial Git commit.

### Expected Artifact
- `prj-55-deepfake-audio-detection/` directory with full initial structure.
- Initial commit on `main` branch.
- Passing scaffold test (`test_project_scaffold.py`).

### Verification Method
- Run `python -m pytest tests/test_project_scaffold.py -v` and confirm all tests pass.
- Run `git log --oneline` and confirm the bootstrap commit exists.
- Run `git status` and confirm the working tree is clean.
- Run `git ls-files` and confirm no audio, weight, or credential files are tracked.

### Pass / Fail Exit Criterion
- **PASS:** All scaffold tests green, initial commit present, no forbidden files tracked.
- **FAIL:** Any test fails, commit missing, or forbidden file type appears in `git ls-files` output.

---

## Day 2 — Source and Literature Verification

### Task
- Locate and read the official ASVspoof 2019 evaluation plan and baseline paper.
- Locate the official ASVspoof 2021 evaluation plan.
- Locate the official AASIST repository and paper (Jung et al., ICASSP 2022).
- Verify that each source is reachable, identify the correct dataset registration URL.
- Update `docs/source_registry.md` with confirmed URLs and access requirements.
- Confirm licences and access conditions for ASVspoof 2019 data and AASIST code.

### Expected Artifact
- `docs/source_registry.md` with status updated from `PENDING` to `VERIFIED` or `BLOCKED` for each source.
- Notes added to `docs/experiment_log.md` header summarising literature findings.

### Verification Method
- Each source in the registry must have a confirmed URL that resolves, a licence/access status, and a responsible person or organisation.

### Pass / Fail Exit Criterion
- **PASS:** All primary sources either verified or documented as blocked with a reason.
- **FAIL:** Any source remains in `PENDING` status without a documented reason for the delay.

---

## Day 3 — Hardware Feasibility Assessment

### Task
- Measure available RAM, disk space, and CPU specifications on the development laptop.
- Run a CPU-only benchmark using only standard library or lightweight tools (e.g., time a NumPy matrix operation) to establish a baseline computation speed.
- Estimate approximate inference time for LFCC feature extraction on a 5-second audio clip, based on the literature or published reference implementations — without downloading audio yet.
- Document findings in `docs/decisions/` as a new ADR if any constraint changes the compute strategy.

### Expected Artifact
- Laptop hardware specification recorded (RAM GB, CPU cores/speed, free disk GB).
- A rough estimate of feasibility for LFCC-GMM inference and AASIST-L inference.
- ADR-002 if a change to the compute strategy is required.

### Verification Method
- Hardware specs reported from `system_profiler` or equivalent.
- Feasibility estimate backed by at least one published reference (cited in source registry).

### Pass / Fail Exit Criterion
- **PASS:** Hardware specs documented; feasibility estimate present with source.
- **FAIL:** No hardware characterisation performed.

---

## Day 4 — Python Environment Setup

### Task
- Create a virtual environment (`.venv`) using Python 3.10 or 3.11.
- Install only the confirmed-safe development dependencies listed in `pyproject.toml` `[dev]` extras.
- Verify that `pytest`, `ruff`, and `mypy` work correctly.
- Do NOT install PyTorch, torchaudio, or any ML framework until Day 5.

### Expected Artifact
- `.venv/` present locally (excluded from Git).
- `requirements-dev.txt` or equivalent lock file (optional but recommended).
- All scaffold tests still pass in the new environment.

### Verification Method
- `python --version` confirms Python ≥ 3.10 inside the venv.
- `pytest tests/test_project_scaffold.py -v` all green inside the venv.
- `ruff check src/ tests/` reports no errors.

### Pass / Fail Exit Criterion
- **PASS:** Scaffold tests pass in the venv; no ML libraries installed accidentally.
- **FAIL:** Tests fail, Python version wrong, or ML library present in the environment.

---

## Day 5 — Dataset Protocol Validation (No Download Yet)

### Task
- Read the ASVspoof 2019 LA evaluation plan carefully.
- Understand the exact evaluation partition, trial lists, and scoring protocol.
- Confirm the expected output format for the scoring script.
- Begin dataset registration/request process if not already started on Day 2.
- Document the data structure expected (file naming convention, partition layout) in `docs/data_governance.md`.

### Expected Artifact
- `docs/data_governance.md` updated with the ASVspoof 2019 LA expected data structure.
- Dataset registration submitted (or blocked, with reason documented).

### Verification Method
- The evaluation plan document has been read and the partition sizes, trial list format, and scoring metric (EER, min-tDCF) are correctly summarised in the governance doc.

### Pass / Fail Exit Criterion
- **PASS:** Protocol understood and documented; registration in progress.
- **FAIL:** Protocol not read; scoring metric or partition unclear.

---

## Day 6 — Baseline Inference Feasibility Probe

### Task
- If PyTorch can be installed on the laptop (CPU-only), install it and run a minimal forward-pass test on a synthetic tensor (no real data needed).
- Attempt to locate the AASIST-L pretrained checkpoint and verify its source and licence.
- If the checkpoint is locatable and its licence permits, document the download procedure (without downloading) in `docs/source_registry.md`.
- Attempt to import the LFCC baseline code (from ASVspoof 2019 baseline, if accessible) and confirm it runs on a synthetic waveform.

### Expected Artifact
- Notes in `docs/experiment_log.md` (template section) documenting what was attempted and what succeeded/failed.
- `docs/source_registry.md` updated with checkpoint provenance and licence status.

### Verification Method
- Any code run must be documented with the exact command used.
- Results recorded honestly — success and failure both documented.

### Pass / Fail Exit Criterion
- **PASS:** At least one baseline component (LFCC feature extraction OR AASIST-L forward pass) verified as runnable on CPU with a synthetic input.
- **FAIL:** Neither baseline can be run; this must be escalated and documented as a project blocker.

---

## Day 7 — Week 1 Acceptance Review

### Task
- Review all Week 1 deliverables against their pass/fail criteria.
- Update `docs/source_registry.md` to final verified/blocked status.
- Write a brief Week 1 summary (1–2 pages) appended to `docs/experiment_log.md`.
- Commit all changes with message: `docs: week 1 acceptance review`.
- Identify any blockers for Week 2 and document them.
- Draft the Week 2 plan (can be a stub at this point).

### Expected Artifact
- All docs updated and committed.
- Week 1 summary in `docs/experiment_log.md`.
- No untracked documentation changes outstanding.

### Verification Method
- `git status` shows clean working tree.
- `pytest tests/ -v` all green.
- All source registry entries have a non-`PENDING` status or a documented reason for delay.

### Pass / Fail Exit Criterion
- **PASS:** All six prior days' pass criteria met; clean commit produced; Week 2 blockers documented.
- **FAIL:** Any Day 1–6 criterion unmet and not documented as a known blocker with a mitigation plan.

---

## Summary Checklist

| Day | Task summary | Pass criterion |
|---|---|---|
| 1 | Bootstrap repository | Scaffold tests green; initial commit exists |
| 2 | Source & literature verification | All sources verified or documented as blocked |
| 3 | Hardware feasibility assessment | Hardware specs and feasibility estimate documented |
| 4 | Python environment setup | Scaffold tests pass in venv; no accidental ML installs |
| 5 | Dataset protocol validation | Protocol understood; registration submitted |
| 6 | Baseline inference feasibility probe | At least one baseline component runs on CPU with synthetic input |
| 7 | Week 1 acceptance review | All above criteria met; clean commit; Week 2 blockers noted |
