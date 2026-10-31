---
updated: 2026-10-28T23:30:00+05:30
---

# Project State: PRJ_55 Deepfake Audio Detection

## Current Position

**Milestone:** PRJ_55 B.Tech CSE Final Year Main Project  
**Phase:** Stage 2: Complete Analysis Workspace, Audio Evidence, Classification Results & Professional UX  
**Status:** `stage_2_complete` (Interactive waveform scrubber synchronized with Web Audio API, dual Simple/Expert presentation modes, single authoritative threshold slider, honest indeterminate processing, 0-8000 Hz FFT spectrogram canvas with high-visibility swatches, TRK-01..TRK-06 neutral sample library, 222 automated tests passing, 0 ruff errors, 0 mypy issues, 100% live system verification passed)  
**Plan:** Stage 2 Implementation and Empirical Validation Complete  

---

## Last Action

- **Executed Stage 2: Complete Analysis Workspace, Audio Evidence, Classification Results & Professional UX**:
  - **Analysis Workspace Redesign (`/analyze`, `style.css`, `app.js`)**:
    - Created dedicated, distraction-free analysis workspace layout separating landing page (`/`) from `/analyze`.
    - Added responsive desktop 40/60 split with collapsible input panel and `↺ Change Audio / Upload Another` trigger.
    - Updated navigation header with concise links (`Home`, `Analyze`, `Evidence`, `Robustness`, `Methodology`, `History`) and responsive mobile drawer navigation (`#mobileMenuBtn`, `#mobileNavDrawer`).
    - Added sample presets library with neutral `.track-badge` indicators (`TRK-01`..`TRK-06`) and objective descriptions preventing ground-truth leakage prior to inference.
    - Fixed disabled button contrast, eliminated duplicate threshold sliders, and ensured clean state resets when loading new audio.
  - **Dual Mode Result Presentation (Simple vs Expert)**:
    - Implemented `.view-mode-bar` segmented toggle between **Simple View** and **Expert View**.
    - **Simple View**: Executive plain-language findings card (`#simpleStatusTitle`, `#simpleStatusText`, `#simpleMarginText`, and scientific scope limitations).
    - **Expert View**: Telemetry row with raw logits ($S_{\text{bonafide}}$, $S_{\text{spoof}}$), decision margin, inference latency, RTF, model architecture specs grid (SincNet bandpass & graph attention parameters), and collapsible interpretation accordion.
  - **Interactive Waveform Scrubber & Accessible Spectrogram**:
    - Built interactive click/drag waveform scrubbing synchronized with HTML5 Web Audio playback.
    - 0–8000 Hz FFT spectrogram canvas with readable frequency markers (`0 Hz`, `2 kHz`, `4 kHz`, `6 kHz`, `8 kHz`) and accessible high-contrast legend swatches (`#A855F7`, `#60A5FA`, `#E2E8F0`).
    - 2-line balanced empty-state text preventing mobile clipping.
  - **Robustness Delta Scoring**:
    - Grounded robustness drift calculation on $\Delta S_{cm} = S_{\text{degraded}} - S_{\text{clean}}$, distinguishing score drift from categorical prediction flips.
  - **AASIST-L Score Convention & Sample Provenance Forensic Investigation**:
    - Confirmed upstream NAVER Clova AI convention: Index 1 is `bonafide` ($S_{cm} = \text{logits}[0, 1]$), Index 0 is `spoof`. Decision rule is $S_{cm} \ge \theta \implies \text{BONAFIDE}$.
    - Provenance audit: `sample_bonafide` and `sample_spoof` are synthetic mathematical test tones (sine-wave sums and FM buzzes), NOT human speech. AASIST-L correctly rejects `sample_bonafide` ($S_{cm} = -4.9437$) as an artificial signal lacking human vocal tract acoustics.
    - Verified real ASVspoof/VCTK speech evaluation tracks: `speech_human_male` ($S_{cm} = +2.0460$, `BONAFIDE`), `speech_human_female` ($S_{cm} = +1.1608$, `BONAFIDE`), `speech_deepfake_tts` ($S_{cm} = -3.9132$, `SPOOF`), `speech_deepfake_vocoder` ($S_{cm} = -1.0934$, `SPOOF`).
    - Added MPS GPU kernel synchronization (`torch.mps.synchronize()`) for fair, accurate latency measurement (36.95 ms synchronized vs async submission).
    - Added 4 new regression tests in `tests/test_web_api.py` covering authentic speech, deepfake speech, dynamic threshold shifting, and synthetic fixture behavior.
    - Ran all 12 core test suites: **226 passed, 2 skipped, 0 failed** in 10.96s.
    - Linter: `ruff check .` -> **0 errors**.
    - Static Typing: `mypy src tests scripts` -> **0 issues across 33 source files**.
    - Live Empirical Verification: `scripts/verify_live_system.py` -> **100% empirical pass**.

---

## Next Steps

1. **Academic Defense & Demonstration**:
   - Run local web application for technical presentation: `python scripts/run_demo_app.py`.
   - Walk through the 3D harmonic signal sculpture, authentic vs spoofed sample predictions, and the channel degradation simulator.
2. **Physical ASVspoof 2019 LA Dataset Benchmarking (When External Storage Available)**:
   - Connect external SSD ($\ge 50\text{ GiB}$) and set `export ASVSPOOF_DATA_ROOT="/Volumes/<ExternalVolume>/asvspoof2019"`.
   - Run `python scripts/acquire_asvspoof2019.py --data-dir "$ASVSPOOF_DATA_ROOT"`.
   - Run `python scripts/generate_scores.py` and `python scripts/evaluate_scores.py`.

---

## Active Decisions

| Decision | Choice | Made | Affects |
|---|---|---|---|
| **ADR-001: Architecture Selection** | Official AASIST-L (Jung et al., ICASSP 2022, 85,306 parameters, SincNet + Heterogeneous Graph Attention) | Week 1 | All phases; sub-100k parameter limit and real-time CPU budget |
| **ADR-002: Storage & Corpus Management** | Strict prohibition of dataset download to internal SSD; resolve paths via `$ASVSPOOF_DATA_ROOT` | Week 2 | Dataset acquisition, protocol validation, benchmark scripts |
| **ADR-003: Modular Degradation Design** | Discrete transform pipelines (Telephony 8 kHz, Opus lossy compression, acoustic SNR noise) | Week 2 | Robustness experiments and web demonstration lab |
| **ADR-004: Strict Metric Availability Guarding** | Throw `MetricUnavailableError` when labels/data are absent; never report synthetic tests as benchmark metrics | Week 3 | Metrics module, experiment runner, reporting integrity |
| **ADR-005: On-Device Forensics Web Architecture** | FastAPI + Vanilla CSS/JS SPA with Canvas 3D Sculpture & Web Audio API (zero cloud/npm bloat) | Week 4 | Local demonstration platform, latency (<100ms), offline usability |

---

## Blockers

- [ ] **External Storage Media Unmounted**:
  - *Context*: macOS partition `/dev/disk3s5` has 61 GiB free space on the internal Apple SSD. 0 external USB volumes are attached.
  - *Resolution Approach*: Mount external SSD ($\ge 50\text{ GiB}$), set `$ASVSPOOF_DATA_ROOT`, and execute autonomous acquisition directly to external media.

---

## Session Context

- Python environment: `.venv/` (Python 3.10.0 ARM64 on Apple Silicon M2).
- Pretrained weights: `models/weights/AASIST-L.pth` (verified SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).
- Web service: FastAPI running on `127.0.0.1:8000` via Uvicorn.
- Test suite: **188 passed, 2 skipped, 0 failed** in 11.80s (`pytest -v`).
- Code coverage: **90%+** statement coverage on `src/prj55`.
- Static typing & Lint: **0 errors** across entire codebase (`ruff check src tests scripts`, `mypy src`).
- Quality gates: Clean (`scripts/verify_live_system.py` passed 100%).
