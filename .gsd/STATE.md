---
updated: 2026-10-31T22:30:00+05:30
---

# Project State: PRJ_55 Deepfake Audio Detection

## Current Position

**Milestone:** PRJ_55 B.Tech CSE Final Year Main Project  
**Phase:** Phase 4 & Final Hardening Complete — Production-Quality Audio Authenticity Lab Fully Synchronized  
**Status:** `production_verified` (All six web application routes fully implemented and hardened; 265 automated tests passing, 2 skipped, 0 failed; zero Ruff linter errors; zero Mypy static typing issues across 40 source files; 17 of 17 live system integration stages verified empirically on October 31, 2026)  
**Plan:** All development phases, UI/UX transformations, security hardening, performance reconciliations, and bibliographic synchronizations complete.

---

## Technical Summary of Completed Work

1. **AASIST-L Neural Architecture & Audio Ingestion**:
   - Ultra-compact graph attention countermeasure with 85,306 trainable parameters (~334 KB parameters; 426 KB full checkpoint with SincNet buffers).
   - Standardized 16 kHz 64,600-sample (~4.0375s) audio framing with symmetric repeat padding and deterministic truncation.
   - Verified pretrained checkpoint loading from `models/weights/AASIST-L.pth` (SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).

2. **Empirical Performance & Latency Profile (Apple Silicon M2)**:
   - **Isolated Model Forward Pass**: Median **36.32 ms** (Apple Metal MPS) vs **162.73 ms** (Apple M2 CPU).
   - **ForensicsService Complete Pipeline**: Median **40.61 ms** (Apple Metal MPS, ~24.6 req/s) vs **180.82 ms** (Apple M2 CPU, ~5.5 req/s).
   - **Loopback HTTP Request**: Median **41.62 ms** (Sample ID payload) / **44.36 ms** (Multipart 4.0s WAV upload).
   - **Multi-Worker Concurrency (4 Workers, Apple M2 CPU)**: 5.43–6.63 inferences/second (isolated model forward pass) vs 2.06 analyses/second (complete ForensicsService pipeline with persistent SQLite disk writes).

3. **Complete Six-Route Web Application & Design System**:
   - `/` (Product Landing): Interactive 3D harmonic particle ribbon canvas, telemetry metrics, and ADR-002 governance.
   - `/analyze` (Forensic Workspace): Dual Simple/Expert presentation modes, dropzone, bundled demonstration speech presets (`TRK-01` to `TRK-04`) & synthetic calibration tones (`TRK-05`, `TRK-06`), interactive waveform scrubber, 0–8 kHz FFT spectrogram canvas, batch upload queue (up to 10 files), and live microphone recording with in-browser PCM-to-WAV transcoding.
   - `/robustness` (Acoustic Robustness Lab): Channel degradation simulator (telephony 8 kHz, AWGN noise 0–20 dB, Opus), non-parametric spectral subtraction DSP mitigation, side-by-side $\Delta S_{\text{cm}}$ drift analysis, and A/B audio comparison.
   - `/history` (Session Evidence History): Persistent SQLite audit log (`data/history.db`), structured JSON export, individual record deletion, and history clearing.
   - `/methodology` (Research Methodology): Mathematical formulations, ASVspoof 2019 baseline tables, and IEEE references.
   - `/shared/{token}` (Controlled Case Dossier): 256-bit URL-safe bearer token access (`secrets.token_urlsafe(32)`), time-limited expiration, one-click revocation, and uniform 404 anti-enumeration resistance.

4. **Security Hardening & Data Governance**:
   - Zero-inline-script Content Security Policy (`default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'`).
   - `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: strict-origin-when-cross-origin`.
   - ADR-002 research governance: zero raw audio stored on disk; uploaded audio is deallocated from memory immediately post-inference.

5. **Decision Logic & Calibration Hygiene**:
   - Canonical ASVspoof logit decision convention: $S_{\text{cm}} = \text{logits}[0, 1] \ge \theta \implies \text{BONAFIDE}$.
   - Uncalibrated baseline default $\theta = 0.00$; automated calibration module (`src/prj55/calibration.py`) enforces strict partition hygiene by rejecting evaluation/test partitions (`CalibrationPartitionError`).

---

## Documented Research Limitations

1. **Utterance-Level Evaluation**: The countermeasure produces a single scalar score $S_{\text{cm}}$ for a 4.0375-second temporal window. It cannot localize sub-second splicing, localized phoneme-level substitutions, or partially manipulated speech frames.
2. **Development-Set Calibration Pause**: The active production threshold is the canonical baseline $\theta = 0.00$. Empirical threshold calibration against the complete physical ASVspoof 2019 LA development partition is paused pending dedicated external storage acquisition under ADR-002.
3. **Unseen Domain & Channel Mismatch**: While front-end spectral subtraction effectively mitigates additive acoustic noise, zero-shot transfer to lossy social media compression codecs (e.g. WhatsApp, cellular telephony) exhibits noticeable score drift, requiring adaptive normalization.
4. **Legal Evidentiary Scope**: The SHA-256 digest provides mathematical content verification against bit-level file alteration. It does not constitute a legally certified digital signature or legally binding chain of custody.

---

## Active Decisions (ADR Summary)

| Decision | Choice | Status | Affects |
|---|---|---|---|
| **ADR-001: Architecture Selection** | Official AASIST-L (Jung et al., IEEE ICASSP 2022; 85,306 parameters) | Active | Core model, compute budget (<100k params), latency |
| **ADR-002: Storage & Corpus Policy** | Strict prohibition of physical dataset download to internal SSD; resolve paths via `$ASVSPOOF_DATA_ROOT` | Active | Dataset acquisition, disk protection, calibration |
| **ADR-003: Modular Degradation Design** | Discrete transform pipelines (Telephony 8 kHz, Opus, AWGN noise) | Active | Robustness lab, perturbation sweeps, DSP mitigation |
| **ADR-004: Strict Metric Availability Guarding** | Throw `MetricUnavailableError` when labels/data are absent; never report smoke-tests as full dataset benchmarks | Active | Metrics, evaluation CLI, reporting integrity |
| **ADR-005: On-Device Forensic Web Architecture** | FastAPI + Vanilla CSS/JS SPA with Canvas 3D Sculpture, Web Audio API, and SQLite persistence | Active | Demonstration platform, sub-50ms latency, zero cloud dependency |

---

## Quality Gate Verification (October 31, 2026)

- **Python Environment**: `.venv/` (Python 3.10.0 ARM64 on Apple Silicon M2).
- **Pretrained Weights**: `models/weights/AASIST-L.pth` (verified SHA-256: `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`).
- **Automated Test Suite**: **265 passed, 2 skipped, 0 failed** in 16.07s (`pytest -v`).
- **Linter**: `ruff check .` -> **All checks passed! (0 errors)**.
- **Static Typing**: `mypy src tests scripts` -> **Success: no issues found in 40 source files**.
- **Live System Integration**: `python scripts/verify_live_system.py` -> **17 of 17 sections passed empirically**.
- **Remote Synchronization**: Synchronized with GitHub `origin/main` at `https://github.com/Aditya-A-Naikar/prj-55-deepfake-audio-detection.git`.
