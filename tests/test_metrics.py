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
    RobustnessComparison,
    TDCFResult,
    align_score_files,
    compare_robustness,
    compute_eer,
    compute_min_tdcf,
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
