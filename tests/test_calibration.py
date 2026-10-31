"""Unit tests for automatic decision-threshold calibration module (PRJ_55)."""

import json
from pathlib import Path

import numpy as np
import pytest

from prj55.calibration import (
    CalibrationCriterion,
    CalibrationError,
    CalibrationPartitionError,
    calibrate_threshold,
    get_active_threshold_config,
    load_calibration,
    save_calibration,
)
from prj55.web.service import ForensicsService


class TestThresholdCalibrationCore:
    """Tests for threshold calibration calculation and optimization criteria."""

    def test_eer_calibration_on_synthetic_dev_partition(self) -> None:
        """Calibrating for EER must find threshold where |FAR - FRR| is minimized."""
        np.random.seed(123)
        # 500 bonafide trials centered at +2.0, 1000 spoof trials centered at -2.0
        bonafide = np.random.normal(loc=2.0, scale=0.8, size=500)
        spoof = np.random.normal(loc=-2.0, scale=0.8, size=1000)
        scores = np.concatenate([bonafide, spoof])
        labels = np.concatenate([np.ones(500, dtype=int), np.zeros(1000, dtype=int)])

        record = calibrate_threshold(
            scores,
            labels,
            partition="dev",
            criterion=CalibrationCriterion.EER,
            provenance="synthetic_dev_set_unit_test",
        )

        assert record.status == "EMPIRICALLY_CALIBRATED"
        assert record.criterion == "EER"
        assert record.partition == "dev"
        assert record.n_bonafide == 500
        assert record.n_spoof == 1000
        # Optimal threshold for symmetric distributions centered at -2.0 and +2.0 should be near 0.0
        assert -0.5 <= record.threshold <= 0.5
        assert record.measured_eer is not None
        assert record.measured_eer < 0.05  # Well separated distributions

    def test_target_far_calibration(self) -> None:
        """Calibrating for a target FAR must find a threshold that bounds False Acceptance."""
        np.random.seed(456)
        bonafide = np.random.normal(loc=1.0, scale=1.0, size=500)
        spoof = np.random.normal(loc=-1.0, scale=1.0, size=1000)
        scores = np.concatenate([bonafide, spoof])
        labels = np.concatenate([np.ones(500, dtype=int), np.zeros(1000, dtype=int)])

        target_far = 0.05  # Max 5% FAR
        record = calibrate_threshold(
            scores,
            labels,
            partition="validation",
            criterion=CalibrationCriterion.TARGET_FAR,
            target_far=target_far,
        )

        assert record.status == "EMPIRICALLY_CALIBRATED"
        assert record.criterion == "TARGET_FAR"
        assert record.target_far == 0.05
        assert record.measured_far is not None
        assert record.measured_far <= 0.055  # Near or below target FAR

    def test_evaluation_partition_strictly_prohibited(self) -> None:
        """Calibration must reject evaluation/test partitions to prevent target data leakage."""
        scores = np.array([1.0, -1.0])
        labels = np.array([1, 0])

        for forbidden in ["eval", "evaluation", "test", "EVAL_SET"]:
            with pytest.raises(CalibrationPartitionError, match="Prohibited partition"):
                calibrate_threshold(scores, labels, partition=forbidden)

    def test_missing_target_far_raises(self) -> None:
        """Criterion TARGET_FAR without valid target_far must raise CalibrationError."""
        scores = np.array([1.0, -1.0])
        labels = np.array([1, 0])

        with pytest.raises(CalibrationError, match="target_far must be a float"):
            calibrate_threshold(
                scores,
                labels,
                partition="dev",
                criterion=CalibrationCriterion.TARGET_FAR,
                target_far=None,
            )

    def test_serialization_round_trip(self, tmp_path: Path) -> None:
        """CalibrationRecord must serialize to JSON and reload identically."""
        scores = np.array([2.0, 1.5, -1.0, -2.5])
        labels = np.array([1, 1, 0, 0])

        record = calibrate_threshold(scores, labels, partition="dev", criterion="EER")
        out_file = tmp_path / "calibration.json"
        save_calibration(record, out_file)

        assert out_file.is_file()
        loaded = load_calibration(out_file)
        assert loaded is not None
        assert loaded.threshold == record.threshold
        assert loaded.criterion == record.criterion
        assert loaded.partition == record.partition
        assert loaded.status == record.status


class TestThresholdServiceIntegration:
    """Tests for threshold configuration integration in ForensicsService."""

    def test_uncalibrated_baseline_fallback(self, tmp_path: Path) -> None:
        """When no calibration file exists, system must default to baseline theta=0.00."""
        non_existent = tmp_path / "absent_calibration.json"
        config = get_active_threshold_config(non_existent)

        assert config["active_threshold"] == 0.0
        assert config["is_calibrated"] is False
        assert config["criterion"] == "CANONICAL_LOGIT_ZERO"
        assert "UNAVAILABLE" in config["status"]

    def test_calibrated_service_inference_default(self, tmp_path: Path) -> None:
        """ForensicsService must automatically use calibrated threshold when present."""
        cal_file = tmp_path / "cal.json"
        with cal_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold": 0.456,
                    "criterion": "EER",
                    "partition": "dev",
                    "status": "EMPIRICALLY_CALIBRATED",
                    "calibrated_at": "2026-10-30T00:00:00Z",
                    "provenance": "test_calibration",
                },
                f,
            )

        config = get_active_threshold_config(cal_file)
        assert config["active_threshold"] == 0.456
        assert config["is_calibrated"] is True
        assert config["criterion"] == "EER"

        service = ForensicsService()
        # Override calibration directly for test
        service.calibration = config
        service.default_threshold = 0.456

        # When threshold=None, service must use default_threshold
        status = service.get_system_status()
        assert status["calibration"]["active_threshold"] == 0.456
