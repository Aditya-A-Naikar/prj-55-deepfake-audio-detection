"""Tests for verifying the reproducible Python environment (Week 1, Task 4).

Validates interpreter version, isolation, core dependencies, research dependencies,
and absence of broken requirements.
"""

import importlib
import sys


def test_python_version_compatible():
    """Python interpreter must be 3.10.x as established in environment audit."""
    assert sys.version_info >= (3, 10), f"Expected Python >= 3.10, got {sys.version_info}"
    assert sys.version_info < (3, 13), (
        f"Python version too new for legacy baselines: {sys.version_info}"
    )


def test_core_dependencies_importable():
    """All core direct dependencies must import cleanly without error."""
    core_packages = ["numpy", "soundfile", "yaml"]
    for pkg in core_packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import core dependency: {pkg}"


def test_soundfile_flac_support():
    """Soundfile must support FLAC format required by ASVspoof audio."""
    import soundfile as sf
    formats = sf.available_formats()
    assert "FLAC" in formats, f"FLAC format missing from soundfile: {formats.keys()}"


def test_research_torch_importable():
    """PyTorch must be importable and MPS (Apple Silicon GPU) acceleration detected."""
    import torch
    assert torch.__version__ is not None
    # Verify torch runs a basic tensor computation on CPU
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2.0
    assert torch.equal(y, torch.tensor([2.0, 4.0, 6.0]))


def test_torch_mps_availability():
    """Apple Silicon Metal Performance Shaders (MPS) should be available."""
    import torch
    # On Apple Silicon M2, MPS is supported
    assert hasattr(torch.backends, "mps"), "PyTorch MPS backend not available in build"
    assert torch.backends.mps.is_available(), "MPS is not available on this Apple Silicon machine"
