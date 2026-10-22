"""Tests for the ASVspoof scoring and robustness metrics (src/prj55/metrics.py).

Verifies EER, min-tDCF, score alignment, and robustness comparison using
deterministic synthetic fixtures.  These tests verify software mathematical
integrity and error-handling guards; they do not use actual speech data.
"""

from __future__ import annotations

import numpy as np
import pytest

from prj55.metrics import (
    COST_MODEL,
    EERResult,
    MetricInputError,
    MetricUnavailableError,
    OperatingPointMetrics,
    RobustnessComparison,
    TDCFResult,
    align_score_files,
    compare_robustness,
    compute_eer,
    compute_min_tdcf,
    compute_official_min_tdcf,
    compute_operating_point_metrics,
    compute_roc_auc,
)

# ---------------------------------------------------------------------------
# Test validation helpers
# ---------------------------------------------------------------------------


class TestMetricValidation:
    """Tests for input validation in metrics functions."""

    def test_empty_scores_raises(self) -> None:
        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int32)
        with pytest.raises(MetricInputError, match="empty"):
            compute_eer(scores, labels)

    def test_mismatched_length_raises(self) -> None:
        scores = np.array([0.5, 0.2, -0.1], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int32)
        with pytest.raises(MetricInputError, match="same shape"):
            compute_eer(scores, labels)

    def test_nan_scores_raises(self) -> None:
        scores = np.array([0.5, np.nan, -0.1], dtype=np.float32)
        labels = np.array([1, 0, 1], dtype=np.int32)
        with pytest.raises(MetricInputError, match="non-finite"):
            compute_eer(scores, labels)

    def test_inf_scores_raises(self) -> None:
        scores = np.array([0.5, np.inf, -0.1], dtype=np.float32)
        labels = np.array([1, 0, 1], dtype=np.int32)
        with pytest.raises(MetricInputError, match="non-finite"):
            compute_eer(scores, labels)

    def test_invalid_label_values_raises(self) -> None:
        scores = np.array([0.5, 0.2, -0.1], dtype=np.float32)
        labels = np.array([1, 2, 0], dtype=np.int32)  # 2 is invalid
        with pytest.raises(MetricInputError, match="labels must be binary"):
            compute_eer(scores, labels)


# ---------------------------------------------------------------------------
# Test EER computation
# ---------------------------------------------------------------------------


class TestEERComputation:
    """Tests for compute_eer calculation accuracy and edge cases."""

    def test_perfect_separation_eer_zero(self) -> None:
        # Spoofs: [-5, -4, -3], Bonafide: [3, 4, 5]
        scores = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        result = compute_eer(scores, labels)

        assert isinstance(result, EERResult)
        assert result.eer == 0.0
        assert result.n_bonafide == 3
        assert result.n_spoof == 3

    def test_completely_inverted_scores_eer_one(self) -> None:
        # Spoofs higher than bonafide: Spoofs [3, 4, 5], Bonafide [-5, -4, -3]
        scores = np.array([3.0, 4.0, 5.0, -5.0, -4.0, -3.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        result = compute_eer(scores, labels)

        assert result.eer == 1.0

    def test_overlapping_distributions_eer(self) -> None:
        # Symmetric overlap:
        # Spoof: [-2, -1, 0, 1]
        # Bonafide: [-1, 0, 1, 2]
        scores = np.array([-2.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        result = compute_eer(scores, labels)

        assert 0.0 < result.eer < 1.0
        assert 0.0 <= result.far_at_threshold <= 1.0
        assert 0.0 <= result.frr_at_threshold <= 1.0

    def test_missing_bonafide_raises_unavailable(self) -> None:
        scores = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        labels = np.array([0, 0, 0], dtype=np.int32)  # All spoof
        with pytest.raises(MetricUnavailableError, match="No bonafide trials"):
            compute_eer(scores, labels)

    def test_missing_spoof_raises_unavailable(self) -> None:
        scores = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([1, 1, 1], dtype=np.int32)  # All bonafide
        with pytest.raises(MetricUnavailableError, match="No spoof trials"):
            compute_eer(scores, labels)


# ---------------------------------------------------------------------------
# Test min-tDCF computation
# ---------------------------------------------------------------------------


class TestTDCFComputation:
    """Tests for compute_min_tdcf calculation."""

    def test_perfect_separation_tdcf_zero(self) -> None:
        scores = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        result = compute_min_tdcf(scores, labels)

        assert isinstance(result, TDCFResult)
        assert result.min_tdcf == 0.0
        assert result.norm_factor > 0

    def test_tdcf_single_class_raises(self) -> None:
        scores = np.array([1.0, 2.0], dtype=np.float64)
        labels = np.array([1, 1], dtype=np.int32)
        with pytest.raises(MetricUnavailableError, match="both bonafide and spoof"):
            compute_min_tdcf(scores, labels)

    def test_tdcf_with_custom_cost_model(self) -> None:
        scores = np.array([-2.0, -1.0, 0.5, 1.5], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        custom_model = dict(COST_MODEL)
        custom_model["Pspoof"] = 0.10
        result = compute_min_tdcf(scores, labels, cost_model=custom_model)
        assert result.min_tdcf >= 0.0


# ---------------------------------------------------------------------------
# Test Score Alignment & Robustness Comparison
# ---------------------------------------------------------------------------


class TestScoreAlignmentAndComparison:
    """Tests for align_score_files and compare_robustness."""

    def test_align_score_files_identical_keys(self) -> None:
        clean = {"utt_01": 1.5, "utt_02": -2.0, "utt_03": 0.8}
        deg = {"utt_01": 1.2, "utt_02": -1.8, "utt_03": 0.5}
        clean_arr, deg_arr, uids = align_score_files(clean, deg)

        assert len(uids) == 3
        assert uids == ["utt_01", "utt_02", "utt_03"]
        np.testing.assert_allclose(clean_arr, [1.5, -2.0, 0.8])
        np.testing.assert_allclose(deg_arr, [1.2, -1.8, 0.5])

    def test_align_score_files_partial_overlap(self) -> None:
        clean = {"utt_01": 1.5, "utt_02": -2.0, "utt_03": 0.8}
        deg = {"utt_02": -1.8, "utt_03": 0.5, "utt_04": -3.0}
        clean_arr, deg_arr, uids = align_score_files(clean, deg, condition_name="noise_10db")

        assert uids == ["utt_02", "utt_03"]
        assert len(clean_arr) == 2
        assert len(deg_arr) == 2

    def test_align_score_files_no_overlap_raises(self) -> None:
        clean = {"utt_01": 1.5}
        deg = {"utt_99": -1.8}
        with pytest.raises(MetricUnavailableError, match="No common utterance IDs"):
            align_score_files(clean, deg)

    def test_align_score_files_empty_raises(self) -> None:
        with pytest.raises(MetricInputError, match="clean_scores dict is empty"):
            align_score_files({}, {"a": 1.0})

        with pytest.raises(MetricInputError, match="empty"):
            align_score_files({"a": 1.0}, {})

    def test_compare_robustness_success(self) -> None:
        clean = {
            "u1": 2.0, "u2": 1.5, "u3": 3.0,  # bonafide
            "u4": -2.0, "u5": -1.5, "u6": -3.0,  # spoof
        }
        deg = {
            "u1": 1.0, "u2": 0.5, "u3": 2.0,
            "u4": -1.0, "u5": -0.5, "u6": -2.0,
        }
        labels = {
            "u1": 1, "u2": 1, "u3": 1,
            "u4": 0, "u5": 0, "u6": 0,
        }
        res = compare_robustness(clean, deg, labels, condition_name="telephony")
        assert isinstance(res, RobustnessComparison)
        assert res.condition_name == "telephony"
        assert res.clean_eer == 0.0
        assert res.absolute_delta_eer == res.degraded_eer - res.clean_eer

    def test_compare_robustness_empty_labels_raises(self) -> None:
        clean = {"u1": 1.0, "u2": -1.0}
        deg = {"u1": 0.8, "u2": -0.9}
        with pytest.raises(MetricUnavailableError, match="Ground-truth labels are required"):
            compare_robustness(clean, deg, labels={})

    def test_compare_robustness_missing_label_for_utterance_raises(self) -> None:
        clean = {"u1": 1.0, "u2": -1.0}
        deg = {"u1": 0.8, "u2": -0.9}
        labels = {"u1": 1}  # Missing u2
        with pytest.raises(MetricUnavailableError, match="missing from labels dict"):
            compare_robustness(clean, deg, labels)


# ---------------------------------------------------------------------------
# Test Operating Point Metrics
# ---------------------------------------------------------------------------


class TestOperatingPointMetrics:
    """Tests for compute_operating_point_metrics calculation."""

    def test_perfect_separation_at_threshold_zero(self) -> None:
        # Spoofs: [-5, -3, -1], Bonafide: [1, 3, 5]
        scores = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        op = compute_operating_point_metrics(
            scores, labels, threshold=0.0, threshold_provenance="test_zero"
        )

        assert isinstance(op, OperatingPointMetrics)
        assert op.threshold == 0.0
        assert op.threshold_provenance == "test_zero"
        assert op.tp == 3
        assert op.tn == 3
        assert op.fp == 0
        assert op.fn == 0
        assert op.fpr == 0.0
        assert op.fnr == 0.0
        assert op.precision == 1.0
        assert op.recall == 1.0
        assert op.f1_score == 1.0
        assert op.balanced_accuracy == 1.0

    def test_all_classified_as_spoof(self) -> None:
        # Threshold above all scores -> all classified as spoof (pred_bonafide = False)
        scores = np.array([-5.0, -3.0, 1.0, 3.0], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        op = compute_operating_point_metrics(scores, labels, threshold=10.0)

        assert op.tp == 0
        assert op.tn == 2
        assert op.fp == 0
        assert op.fn == 2
        assert op.fpr == 0.0
        assert op.fnr == 1.0
        assert op.precision == 0.0  # Zero denominator handled
        assert op.recall == 0.0
        assert op.f1_score == 0.0
        assert op.balanced_accuracy == 0.5  # (0 + 1) / 2

    def test_non_finite_threshold_raises(self) -> None:
        scores = np.array([1.0, -1.0], dtype=np.float64)
        labels = np.array([1, 0], dtype=np.int32)
        with pytest.raises(MetricInputError, match="Threshold must be finite"):
            compute_operating_point_metrics(scores, labels, threshold=np.nan)


# ---------------------------------------------------------------------------
# Test ROC-AUC computation
# ---------------------------------------------------------------------------


class TestROCAUC:
    """Tests for compute_roc_auc calculation."""

    def test_roc_auc_perfect(self) -> None:
        scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        auc = compute_roc_auc(scores, labels)
        assert auc == 1.0

    def test_roc_auc_inverted(self) -> None:
        scores = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        auc = compute_roc_auc(scores, labels)
        assert auc == 0.0

    def test_roc_auc_tied_scores(self) -> None:
        # All identical scores -> completely random discrimination
        scores = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        auc = compute_roc_auc(scores, labels)
        assert auc == 0.5

    def test_roc_auc_single_class_raises(self) -> None:
        scores = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        labels = np.array([1, 1, 1], dtype=np.int32)
        with pytest.raises(MetricUnavailableError, match="both positive and negative"):
            compute_roc_auc(scores, labels)


# ---------------------------------------------------------------------------
# Test Official min-tDCF
# ---------------------------------------------------------------------------


class TestOfficialTDCF:
    """Tests for compute_official_min_tdcf."""

    def test_missing_asv_scores_raises_unavailable(self) -> None:
        scores = np.array([1.0, -1.0], dtype=np.float64)
        labels = np.array([1, 0], dtype=np.int32)
        with pytest.raises(MetricUnavailableError, match="aligned ASV system scores"):
            compute_official_min_tdcf(scores, asv_scores=None, labels=labels)

    def test_mismatched_asv_shape_raises(self) -> None:
        scores = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        asv_scores = np.array([0.2, 0.3], dtype=np.float64)
        labels = np.array([1, 0, 1], dtype=np.int32)
        with pytest.raises(MetricInputError, match="asv_scores shape"):
            compute_official_min_tdcf(scores, asv_scores=asv_scores, labels=labels)

    def test_valid_asv_scores_returns_result(self) -> None:
        scores = np.array([3.0, 2.0, -2.0, -3.0], dtype=np.float64)
        asv_scores = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        labels = np.array([1, 1, 0, 0], dtype=np.int32)
        res = compute_official_min_tdcf(scores, asv_scores=asv_scores, labels=labels)
        assert isinstance(res, TDCFResult)
        assert res.min_tdcf >= 0.0

