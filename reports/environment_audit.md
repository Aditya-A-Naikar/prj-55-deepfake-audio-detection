# Environment Audit — PRJ_55: Deepfake Audio Detection

**Date:** 2026-10-17
**Auditor:** Antigravity AI assistant (automated read-only commands)
**Repository:** `prj-55-deepfake-audio-detection` — branch `main`, commit `47c52f2`
**Status at audit start:** Working tree clean, no remote.

> This document distinguishes:
> - **FACT** — directly observed from command output.
> - **INFERRED** — reasonable conclusion from observed facts.
> - **RECOMMENDATION** — requires further testing before relying on it.

---

## 1. Operating System

| Field | Value | Source |
|---|---|---|
| OS | macOS 26.5.1 | `sw_vers` |
| Build | 25F80 | `sw_vers` |
| Kernel | Darwin 25.5.0 | `uname -srm` |
| Architecture | arm64 (Apple Silicon, native) | `uname -m`, `arch` |
| Rosetta active | No | `arch` returned `arm64` directly |

**FACT:** Native Apple Silicon macOS installation. No x86 emulation layer is active.

---

## 2. CPU

| Field | Value | Source |
|---|---|---|
| Model identifier | Mac14,15 | `sysctl hw.model` |
| CPU brand | Apple M2 | `sysctl machdep.cpu.brand_string` |
| Physical cores | 8 | `sysctl hw.physicalcpu` |
| Logical cores | 8 | `sysctl hw.logicalcpu` |

**FACT:** M2 has unified memory architecture — CPU, GPU, and Neural Engine share RAM. No discrete GPU is present; the integrated GPU is accessed via Apple Metal / MPS.

**INFERRED:** All 8 cores are available for multiprocessed data loading and feature extraction.

---

## 3. Memory (RAM)

| Field | Value | Source |
|---|---|---|
| Total RAM | 8 GB (8,589,934,592 bytes) | `sysctl hw.memsize` |
| Available RAM at audit time | Not measured | — |

**FACT:** 8 GB total unified RAM.

**RECOMMENDATION:** Before loading a pretrained AASIST checkpoint, measure actual free memory with `vm_stat | head -5`.

---

## 4. Disk Space — CRITICAL BLOCKER

Commands run:
```
df -h "/Users/adityaanaikar/untitled folder/prj-55-deepfake-audio-detection"
df -h "$HOME"
```

Observed output:
```
Filesystem      Size    Used   Avail Capacity
/dev/disk3s5   228Gi  181Gi   3.8Gi    98%     /System/Volumes/Data
```

| Field | Value |
|---|---|
| Total capacity | 228 GB |
| Used | 181 GB |
| **Available** | **3.8 GB** |
| Utilisation | **98%** |

> WARNING: DISK SPACE IS CRITICALLY LOW — 3.8 GB free on a 98%-full volume.
>
> The ASVspoof 2019 LA dataset (training + development + evaluation) is approximately
> 25–30 GB unpacked. This CANNOT be downloaded in the current state.
>
> This is the highest-priority blocker for the entire project.

**Required actions before dataset work can begin:**
1. Free at least 35 GB to safely accommodate the dataset plus working space.
2. Consider an external drive or network storage, symlinking `data/raw/` into the project.
3. Update `docs/data_governance.md` with the chosen storage strategy.

---

## 5. Python Installations

### 5a. Python versions found

| Executable | Version | Architecture | Location |
|---|---|---|---|
| `python3` (default) | 3.10.0 | arm64 | `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3` |
| `python3.12` | 3.12.4 | arm64 | `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` |
| `/usr/local/bin/python3` | — | Universal (x86_64 + arm64) | Homebrew/system path |

Commands run:
```
which python3 && python3 --version
ls /Library/Frameworks/Python.framework/Versions/
ls /opt/homebrew/bin/python*  (no matches)
ls /usr/local/bin/python*
pyenv versions  → not found
conda info      → not found
```

**FACT:** Python 3.10.0 is the shell default. Python 3.12.4 also installed. No pyenv, no conda.

### 5b. Packages in Python 3.10 system environment (full pip list, audit date)

```
annotated-types 0.8.0    anyio 4.15.1        certifi 2026.7.22
click 8.5.0              contourpy 1.3.2     easyocr 1.7.2
fastapi 0.141.1          filelock 3.32.7     fonttools 4.65.0
fsspec 2026.7.0          ImageIO 2.37.4      Jinja2 3.1.6
kiwisolver 1.5.1         lazy-loader 0.5     matplotlib 3.10.9
mpmath 1.3.0             networkx 3.4.2      ninja 1.13.2
numpy 2.2.6              nvidia-ml-py 13.610.43  opencv-python 5.0.0.93
packaging 26.3           pillow 12.3.0       pip 21.2.3
pluggy 1.7.0             polars 1.44.2       psutil 7.2.2
pyclipper 1.4.0          pydantic 2.13.5     Pygments 2.21.0
pytest 9.1.1             PyYAML 6.0.3        requests 2.34.2
scikit-image 0.25.2      scipy 1.15.3        setuptools 84.0.0
shapely 2.1.2            six 1.17.0          starlette 1.6.0
sympy 1.14.0             tifffile 2025.5.10  tomli 2.5.0
torch 2.14.0             torchvision 0.29.0  typing_extensions 4.16.0
ultralytics 8.4.123      urllib3 2.8.0       uvicorn 0.53.0
uvloop 0.22.1            watchfiles 1.2.0    websockets 16.1.1
```

**Selected relevant packages for this project:**

| Package | Version | Project relevance |
|---|---|---|
| `numpy` | 2.2.6 | Array ops, feature matrices — PRESENT |
| `scipy` | 1.15.3 | Signal processing, GMM — PRESENT |
| `torch` | **2.14.0** | Neural inference — **PRESENT, MPS-enabled** |
| `torchvision` | 0.29.0 | Less relevant for audio — present |
| `matplotlib` | 3.10.9 | Plotting results — PRESENT |
| `pytest` | 9.1.1 | Testing — PRESENT |
| `PyYAML` | 6.0.3 | Config loading — PRESENT |
| `torchaudio` | **NOT installed** | Waveform loading; AASIST pipeline — MISSING |
| `librosa` | **NOT installed** | LFCC feature extraction — MISSING |
| `soundfile` | **NOT installed** | Reading .flac/.wav — MISSING |
| `scikit-learn` | **NOT installed** | GMM fitting for baseline — MISSING |

**FACT:** PyTorch 2.14.0 is already installed in system Python. This is a very recent version.

**INFERRED:** The system Python has many unrelated packages (YOLO/ultralytics, FastAPI, EasyOCR). Installing project packages into this environment risks conflicts. A dedicated venv is required.

---

## 6. Hardware Acceleration — GPU / MPS

Commands run:
```
system_profiler SPDisplaysDataType  (filtered for GPU fields)
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

Observed output:
```
Chipset Model: Apple M2
Vendor: Apple (0x106b)
Metal Support: Metal 4

torch version: 2.14.0
MPS available: True
MPS built: True
```

MPS benchmark (observed):
```
Command: torch.randn(1000,1000) @ torch.randn(1000,1000) on 'mps' vs 'cpu'
MPS 1000x1000 matmul: 259.1 ms  (first call — includes Metal shader compilation)
CPU 1000x1000 matmul: 6.2 ms
```

| Field | Value | Source |
|---|---|---|
| GPU type | Apple M2 integrated (unified memory) | `system_profiler` |
| Metal version | Metal 4 | `system_profiler` |
| MPS available | **True** | `torch.backends.mps.is_available()` |
| MPS built in torch | **True** | `torch.backends.mps.is_built()` |
| Discrete NVIDIA/AMD GPU | Not present | `system_profiler` |
| CUDA available | No | No discrete GPU |

**FACT:** MPS is available and functional with torch 2.14.0.

**IMPORTANT — MPS benchmark interpretation:** The 259 ms MPS figure vs 6.2 ms CPU for a cold 1000×1000 matmul is misleading. The first MPS call always incurs Metal shader JIT compilation overhead. Steady-state MPS throughput for repeated operations is substantially faster. This benchmark does NOT indicate that CPU is faster for neural inference — a warm-up pass and model-specific benchmark (Day 6) are required.

**INFERRED:** MPS inference for AASIST-L is likely feasible and faster than CPU-only for the neural model component, once the cold-start overhead is paid. Audio feature extraction (LFCC) will likely remain on CPU.

**RECOMMENDATION:** ADR-001 should be updated (ADR-002) to record that MPS is confirmed available and should be the preferred device for neural inference, with CPU as fallback.

---

## 7. Git and GitHub CLI

Commands run:
```
git --version
which gh && gh --version
xcode-select -p
```

Observed output:
```
git version 2.50.1 (Apple Git-155)
gh: not installed
/Library/Developer/CommandLineTools
```

| Tool | Version | Status |
|---|---|---|
| `git` | 2.50.1 (Apple Git-155) | Installed and functional |
| `gh` (GitHub CLI) | Not installed | Remote repo creation blocked |
| Xcode Command Line Tools | Installed at `/Library/Developer/CommandLineTools` | Available |

**BLOCKER (carried from Stage 0):** `gh` not installed. Manual GitHub repo creation required.

---

## 8. Network Connectivity

Commands run:
```
curl -s -o /dev/null -w "... HTTP %{http_code} in %{time_total}s" --max-time 10 <URL>
```

| Destination | HTTP Status | Response time | Result |
|---|---|---|---|
| `https://www.asvspoof.org/` | 200 | 1.09 s | Reachable |
| `https://github.com/clovaai/aasist` | 200 | 1.01 s | Reachable |
| `https://pypi.org/` | 200 | 0.35 s | Reachable |
| `https://pytorch.org/` | 200 | 0.36 s | Reachable |

**FACT:** All four primary research sources are reachable from this network.

---

## 9. Existing Virtual Environments

Command run:
```
find project_root -name "pyvenv.cfg"
```
Output: (none found)

**FACT:** No virtual environment exists in the project directory. Week 1 Day 4 venv task is still pending.

---

## 10. CPU Baseline Benchmark

Command run:
```python
# Pure-Python 200x200 nested-list matrix multiply
Pure-Python 200x200 matmul: 0.532 s
```

This measures raw Python interpreter speed, not NumPy/BLAS performance.
NumPy-based LFCC computation will be substantially faster via the Accelerate framework on M2.

**RECOMMENDATION:** A representative benchmark (LFCC extraction on a synthetic 16 kHz waveform using the actual feature extractor) should be run on Day 6.

---

## 11. Summary Table

### Hardware (all FACT)

| Item | Value |
|---|---|
| OS | macOS 26.5.1, arm64, Darwin 25.5.0 |
| CPU | Apple M2, Mac14,15, 8 physical + 8 logical cores |
| RAM | 8 GB unified memory |
| GPU | M2 integrated, Metal 4, MPS=True in torch 2.14.0 |
| Disk free | **3.8 GB — CRITICAL BLOCKER** |

### Software (all FACT)

| Item | Value |
|---|---|
| Python default | 3.10.0 (arm64, Framework install) |
| Python available | 3.12.4 also installed |
| torch | 2.14.0, MPS-enabled — already installed |
| torchaudio | NOT installed — required |
| librosa | NOT installed — required |
| soundfile | NOT installed — required |
| scikit-learn | NOT installed — required |
| git | 2.50.1 — functional |
| gh | NOT installed |
| Network | All sources HTTP 200 |
| Existing venv | None |

---

## 12. Recommended Python Environment Strategy

**Step 1 — Resolve disk space first** (prerequisite for everything else).

**Step 2 — Create a dedicated project venv** using Python 3.10:
```bash
cd "prj-55-deepfake-audio-detection"
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Step 3 — Install project dependencies** (requires ~1.5 GB free after torch download, or ~600 MB if torch wheel cached):
```bash
pip install torch==2.14.0 torchaudio==2.14.0
pip install librosa soundfile scikit-learn
pip install pytest ruff mypy
```

**Do NOT use `--system-site-packages`** — the system Python contains unrelated packages (ultralytics, fastapi, easyocr) that may conflict.

**Step 4 — Verify scaffold tests still pass inside the venv:**
```bash
python -m pytest tests/test_project_scaffold.py -v
```

---

## 13. Major Compatibility Risks

| Risk | Severity | Detail |
|---|---|---|
| Disk space: 3.8 GB free | CRITICAL | Dataset (~25 GB) cannot be downloaded |
| torch re-download in venv | High | ~600 MB download needed; requires free disk |
| torchaudio not installed | High | Required for AASIST-L audio pipeline |
| MPS cold-start latency | Medium | 259 ms first call; warm-up required; steady state unknown |
| System Python pollution | Medium | Mixed packages; dedicated venv mitigates this |
| Python 3.10.0 age | Low | Still supported until Oct 2026; acceptable |
| gh not installed | Low | Remote repo creation blocked; manual workaround available |

---

## 14. Pass / Fail Status

| Check | Result |
|---|---|
| OS and CPU recorded | PASS |
| RAM recorded | PASS |
| Disk space recorded | PASS (finding: CRITICAL LOW) |
| Python versions identified | PASS |
| Package inventory taken | PASS |
| GPU / MPS status confirmed | PASS — MPS available and functional |
| MPS and CPU benchmarks run | PASS |
| Git version confirmed | PASS |
| Network connectivity to all four sources | PASS — all HTTP 200 |
| Virtual environment status | PASS (finding: none exists yet) |
| No sensitive data in report | PASS — no tokens, keys, env vars, credentials |
| No software installed during audit | PASS |
| No system files modified | PASS |

**Overall: 13/13 checks passed. One critical finding (disk space) requires user action before any dataset or venv work.**

---

## 15. Files Changed in This Task

| File | Change |
|---|---|
| `reports/environment_audit.md` | Created — this file |

No other project files were modified.

---

## 16. Recommended Next Task

**Immediate (before any other Week 1 task):**
Free at least 35 GB of disk space on `/dev/disk3s5`.

**After disk is resolved — Day 3/4 combined:**
```bash
# Create venv
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.0 torchaudio==2.14.0 librosa soundfile scikit-learn pytest ruff mypy

# Verify
python -m pytest tests/test_project_scaffold.py -v

# Commit updated docs
git add reports/environment_audit.md
git commit -m "docs: add environment audit report (Week 1 Task 2)"
```

**Also recommended:** Create `docs/decisions/ADR-002-mps-inference-strategy.md` to update ADR-001 with the confirmed MPS availability finding.
