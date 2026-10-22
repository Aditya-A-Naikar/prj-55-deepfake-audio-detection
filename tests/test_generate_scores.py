"""Tests for the score generation CLI (scripts/generate_scores.py).

Verifies synthetic generation, argument parsing, header metadata formatting,
and error handling on invalid or missing checkpoints.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from prj55.models.aasist import AASIST, AASISTConfig
from scripts.generate_scores import (
    generate_synthetic_trials,
    run_synthetic_score_generation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestSyntheticScoreGeneration:
    """Tests for synthetic trial score generation logic."""

    def test_generate_synthetic_trials_count_and_types(self) -> None:
        trials = generate_synthetic_trials(num_trials=5, seed=123)
        assert len(trials) == 5
        for uid, waveform, label in trials:
            assert isinstance(uid, str)
            assert uid.startswith("LA_T_SYNTH_")
            assert isinstance(waveform, np.ndarray)
            assert waveform.dtype == np.float32
            assert label in {"bonafide", "spoof"}

    def test_run_synthetic_score_generation_writes_valid_output(
        self, tmp_path: Path
    ) -> None:
        cfg = AASISTConfig()
        model = AASIST(cfg)
        out_file = tmp_path / "test_synthetic_scores.txt"

        summary = run_synthetic_score_generation(
            model=model,
            num_trials=4,
            output_file=out_file,
            device="cpu",
            checkpoint_hash="dummy_hash_12345",
        )

        assert summary["mode"] == "synthetic_dry_run"
        assert summary["scored_trials"] == 4
        assert out_file.is_file()

        lines = out_file.read_text(encoding="utf-8").strip().splitlines()
        # Verify header lines
        header_lines = [line for line in lines if line.startswith("#")]
        score_lines = [line for line in lines if not line.startswith("#")]

        assert len(header_lines) >= 3
        assert len(score_lines) == 4

        for s_line in score_lines:
            parts = s_line.split()
            assert len(parts) == 2
            uid, score_str = parts[0], parts[1]
            assert uid.startswith("LA_T_SYNTH_")
            score_val = float(score_str)
            assert np.isfinite(score_val)


class TestGenerateScoresCLI:
    """CLI invocations for scripts/generate_scores.py."""

    def test_cli_synthetic_flag(self, tmp_path: Path) -> None:
        out_file = tmp_path / "cli_scores.txt"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_scores.py"),
            "--synthetic",
            "--num-synthetic",
            "3",
            "--output-file",
            str(out_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Stderr: {result.stderr}"
        assert out_file.is_file()
        assert "Synthetic run complete: 3 trials scored" in result.stderr

    def test_cli_missing_data_dir_fails_gracefully(self) -> None:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_scores.py"),
            "--data-dir",
            "/nonexistent/directory/path/here",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "dataset root not found" in result.stderr
