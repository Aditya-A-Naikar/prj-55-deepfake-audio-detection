"""Tests for the score evaluation CLI (scripts/evaluate_scores.py).

Verifies loading score files, parsing protocol labels, computing metrics,
handling threshold provenance, and running synthetic dry-runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from prj55.metrics import MetricUnavailableError
from scripts.evaluate_scores import (
    evaluate_trial_population,
    generate_synthetic_evaluation,
    load_protocol_labels,
    load_score_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestScoreFileLoading:
    """Tests for load_score_file utility."""

    def test_load_score_file_with_headers_and_comments(self, tmp_path: Path) -> None:
        content = (
            "# PRJ_55 Countermeasure Scores\n"
            "# Split: dev\n"
            "# Checkpoint: dummy\n"
            "LA_0001 1.25\n"
            "LA_0002 -2.10\n"
            "LA_0003 0.05\n"
        )
        score_path = tmp_path / "scores.txt"
        score_path.write_text(content, encoding="utf-8")

        scores, meta = load_score_file(score_path)
        assert len(scores) == 3
        assert scores["LA_0001"] == 1.25
        assert scores["LA_0002"] == -2.10
        assert scores["LA_0003"] == 0.05
        assert meta.get("Split") == "dev"

    def test_load_nonexistent_score_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_score_file(tmp_path / "missing.txt")


class TestProtocolLabelLoading:
    """Tests for load_protocol_labels."""

    def test_load_protocol_labels(self, tmp_path: Path) -> None:
        proto_content = (
            "LA_0001 LA_D_1001 A01 - spoof\n"
            "LA_0002 LA_D_1002 - - bonafide\n"
            "LA_0003 LA_D_1003 A02 - spoof\n"
        )
        proto_path = tmp_path / "protocol.txt"
        proto_path.write_text(proto_content, encoding="utf-8")

        labels, system_ids, records = load_protocol_labels(proto_path)
        assert len(records) == 3
        assert labels["LA_D_1001"] == 0
        assert labels["LA_D_1002"] == 1
        assert labels["LA_D_1003"] == 0
        assert system_ids["LA_D_1001"] == "A01"
        assert system_ids["LA_D_1003"] == "A02"


class TestEvaluationLogic:
    """Tests for evaluate_trial_population."""

    def test_evaluate_aligned_trials_success(self) -> None:
        scores = {
            "u1": 2.0, "u2": 1.5, "u3": 3.0,
            "u4": -2.0, "u5": -1.5, "u6": -3.0,
        }
        labels = {
            "u1": 1, "u2": 1, "u3": 1,
            "u4": 0, "u5": 0, "u6": 0,
        }
        system_ids = {
            "u1": "-", "u2": "-", "u3": "-",
            "u4": "A01", "u5": "A01", "u6": "A02",
        }

        res = evaluate_trial_population(
            scores=scores,
            labels=labels,
            system_ids=system_ids,
            threshold=0.0,
            threshold_provenance="fixed_zero",
        )

        assert res["trial_alignment"]["aligned_trials"] == 6
        assert res["eer"]["percent"] == 0.0
        assert res["roc_auc"] == 1.0
        assert res["operating_point"]["precision"] == 1.0
        assert res["operating_point"]["recall"] == 1.0
        assert res["min_tdcf"]["status"] == "unavailable"

    def test_evaluate_unaligned_raises_unavailable(self) -> None:
        scores = {"s1": 1.0}
        labels = {"p1": 0}
        with pytest.raises(MetricUnavailableError, match="No common trials"):
            evaluate_trial_population(scores=scores, labels=labels)

    def test_synthetic_evaluation_dry_run(self) -> None:
        res = generate_synthetic_evaluation()
        assert "eer" in res
        assert "operating_point" in res
        assert "roc_auc" in res
        assert res["trial_alignment"]["aligned_trials"] == 20
        assert "SYNTHETIC DRY-RUN" in res["notice"]


class TestEvaluateScoresCLI:
    """CLI testing for scripts/evaluate_scores.py."""

    def test_cli_synthetic_flag(self, tmp_path: Path) -> None:
        out_json = tmp_path / "eval.json"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_scores.py"),
            "--synthetic",
            "--output-json",
            str(out_json),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert out_json.is_file()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert "eer" in data
        assert "roc_auc" in data
