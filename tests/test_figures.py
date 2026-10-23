"""Tests for PRJ_55 publication-quality figure generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
ANALYSIS_DIR = REPO_ROOT / "reports" / "analysis"


class TestFigureArtifacts:
    """Validate that all publication figures exist, are non-empty, and correctly formatted."""

    EXPECTED_FIGURE_BASES = [
        "figure_01_confusion_matrix",
        "figure_02_roc_curve",
        "figure_03_det_eer",
        "figure_04_score_distributions",
        "figure_05_robustness",
        "figure_06_mitigation_comparison",
        "figure_07_waveform_spectrogram",
        "figure_08_inference_latency",
        "figure_09_parameter_budget",
    ]

    @pytest.mark.parametrize("base_name", EXPECTED_FIGURE_BASES)
    def test_png_figures_exist_and_non_empty(self, base_name: str) -> None:
        png_path = FIGURES_DIR / f"{base_name}.png"
        assert png_path.is_file(), f"Expected PNG figure missing: {png_path}"
        assert png_path.stat().st_size > 5000, f"PNG figure too small (< 5 KB): {png_path}"

    @pytest.mark.parametrize("base_name", EXPECTED_FIGURE_BASES)
    def test_pdf_figures_exist_and_non_empty(self, base_name: str) -> None:
        pdf_path = FIGURES_DIR / f"{base_name}.pdf"
        assert pdf_path.is_file(), f"Expected vector PDF figure missing: {pdf_path}"
        assert pdf_path.stat().st_size > 5000, f"PDF figure too small (< 5 KB): {pdf_path}"


class TestFigureManifestAndMetrics:
    """Validate completeness and schema of figure manifest and metrics summary."""

    def test_figure_manifest_exists_and_contains_all_figures(self) -> None:
        manifest_path = ANALYSIS_DIR / "figure_manifest.md"
        assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"
        content = manifest_path.read_text(encoding="utf-8")

        for fig_num in range(1, 10):
            assert f"Figure {fig_num}" in content, f"Missing Figure {fig_num} entry in manifest"
        assert "BLOCKED_PENDING_PHYSICAL_DATASET" in content
        assert "AVAILABLE_MEASURED" in content
        assert "ADR-002" in content

    def test_metrics_summary_valid_json_and_contains_all_figures(self) -> None:
        metrics_path = ANALYSIS_DIR / "metrics_summary.json"
        assert metrics_path.is_file(), f"Metrics summary JSON missing: {metrics_path}"
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "figures" in data
        assert "generated_at" in data

        figures = data["figures"]
        for fig_num in range(1, 10):
            key = f"figure_{fig_num:02d}"
            assert key in figures, f"Missing {key} in metrics summary JSON"

        # Validate Figure 1 (Protocol Spec)
        f1 = figures["figure_01"]
        assert f1["status"] == "BLOCKED_PENDING_PHYSICAL_DATASET"
        assert f1["target_trials"] == 24844
        assert f1["n_bonafide"] == 2548
        assert f1["n_spoof"] == 22296

        # Validate Figure 2 (ROC Target)
        f2 = figures["figure_02"]
        assert 0.99 <= f2["published_target_auc"] <= 1.0

        # Validate Figure 5 (Robustness)
        f5 = figures["figure_05"]
        assert f5["status"] == "AVAILABLE_MEASURED"
        assert len(f5["conditions_tested"]) == 5

        # Validate Figure 8 (Latency Benchmark)
        f8 = figures["figure_08"]
        assert f8["status"] == "AVAILABLE_MEASURED"
        cpu_metrics = f8["cpu_metrics_ms"]
        assert cpu_metrics["median"] > 0
        assert cpu_metrics["real_time_factor"] < 0.2  # Must be faster than real time

        # Validate Figure 9 (Parameters)
        f9 = figures["figure_09"]
        assert f9["total_parameters"] == 85306
