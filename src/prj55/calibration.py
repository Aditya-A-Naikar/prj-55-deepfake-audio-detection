"""Automatic decision-threshold calibration module for PRJ_55.

Implements rigorous, evidence-backed decision threshold selection using labelled
development/validation data only. Strictly prohibits evaluating or tuning thresholds
on evaluation/test partitions or user upload samples.

Supported Criteria:
- EER (Equal Error Rate): Threshold where False Alarm Rate equals False Rejection Rate.
- TARGET_FAR: Threshold calibrated to achieve a maximum specified False Alarm Rate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from prj55.metrics import compute_eer, compute_operating_point_metrics


class CalibrationError(Exception):
    """Raised when threshold calibration fails validation or execution."""


class CalibrationPartitionError(CalibrationError):
    """Raised when an attempt is made to calibrate on an evaluation/test partition."""


class CalibrationCriterion(str, Enum):
    """Supported optimization criteria for automatic threshold calibration."""

    EER = "EER"
    TARGET_FAR = "TARGET_FAR"


@dataclass(frozen=True)
class CalibrationRecord:
    """Metadata record for a calibrated decision threshold."""

    threshold: float
    criterion: str
    partition: str
    target_far: float | None = None
    measured_eer: float | None = None
    measured_far: float | None = None
    measured_frr: float | None = None
    n_bonafide: int = 0
    n_spoof: int = 0
    status: str = "EMPIRICALLY_CALIBRATED"
    calibrated_at: str = ""
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return asdict(self)


def calibrate_threshold(
    cm_scores: np.ndarray,
    labels: np.ndarray,
    partition: str = "dev",
    criterion: CalibrationCriterion | str = CalibrationCriterion.EER,
    target_far: float | None = None,
    provenance: str = "",
) -> CalibrationRecord:
    """Calibrate decision threshold strictly from labelled development/validation trials.

    Args:
        cm_scores: 1D array of countermeasure scores (higher = more authentic).
        labels: 1D binary array (0 = spoof, 1 = bonafide).
        partition: Partition name. Must be 'dev', 'development', or 'validation'.
        criterion: Optimization criterion (EER or TARGET_FAR).
        target_far: Desired False Alarm Rate (required if criterion is TARGET_FAR).
        provenance: Description of source data file or experiment run.

    Returns:
        CalibrationRecord containing selected threshold and verification metrics.

    Raises:
        CalibrationPartitionError: If partition indicates evaluation/test data.
        CalibrationError: If input data is malformed or criterion requirements are unsatisfied.
    """
    clean_partition = partition.strip().lower()
    if any(term in clean_partition for term in ("eval", "test")):
        raise CalibrationPartitionError(
            f"Prohibited partition '{partition}' for threshold calibration. "
            "Calibration must strictly use validation/development data to prevent target leakage."
        )

    if clean_partition not in ("dev", "development", "validation", "synthetic_dev"):
        raise CalibrationError(
            f"Invalid partition '{partition}'. Must be 'dev', 'development', or 'validation'."
        )

    if not isinstance(criterion, CalibrationCriterion):
        try:
            criterion = CalibrationCriterion(str(criterion).upper())
        except ValueError:
            raise CalibrationError(f"Unsupported calibration criterion: {criterion}")

    if criterion == CalibrationCriterion.TARGET_FAR:
        if target_far is None or not (0.0 < target_far < 1.0):
            raise CalibrationError(
                f"target_far must be a float in (0.0, 1.0), got {target_far}"
            )

    # 1. EER Optimization
    if criterion == CalibrationCriterion.EER:
        eer_res = compute_eer(cm_scores, labels)
        op_metrics = compute_operating_point_metrics(
            cm_scores,
            labels,
            threshold=eer_res.threshold,
            threshold_provenance=f"dev_eer_calibration ({clean_partition})",
        )
        return CalibrationRecord(
            threshold=float(eer_res.threshold),
            criterion=criterion.value,
            partition=clean_partition,
            target_far=None,
            measured_eer=float(eer_res.eer),
            measured_far=float(op_metrics.fpr),
            measured_frr=float(op_metrics.fnr),
            n_bonafide=eer_res.n_bonafide,
            n_spoof=eer_res.n_spoof,
            status="EMPIRICALLY_CALIBRATED",
            calibrated_at=datetime.now(timezone.utc).isoformat(),
            provenance=provenance or f"EER minimum on {clean_partition} partition",
        )

    # 2. Target FAR Optimization
    spoof_scores = cm_scores[labels == 0]
    bonafide_scores = cm_scores[labels == 1]
    if len(spoof_scores) == 0 or len(bonafide_scores) == 0:
        raise CalibrationError("Both bonafide and spoof trials are required for calibration.")

    thresholds = np.sort(np.unique(cm_scores))
    far_values = np.array([np.mean(spoof_scores >= t) for t in thresholds], dtype=np.float64)

    # Find the lowest threshold that satisfies FAR <= target_far
    assert target_far is not None
    valid_indices = np.where(far_values <= target_far)[0]
    if len(valid_indices) == 0:
        # Extreme case: even highest score yields FAR > target_far
        chosen_idx = int(np.argmax(thresholds))
    else:
        chosen_idx = int(valid_indices[0])

    chosen_thresh = float(thresholds[chosen_idx])
    op_metrics = compute_operating_point_metrics(
        cm_scores,
        labels,
        threshold=chosen_thresh,
        threshold_provenance=f"dev_target_far_{target_far} ({clean_partition})",
    )

    return CalibrationRecord(
        threshold=chosen_thresh,
        criterion=criterion.value,
        partition=clean_partition,
        target_far=target_far,
        measured_eer=None,
        measured_far=float(op_metrics.fpr),
        measured_frr=float(op_metrics.fnr),
        n_bonafide=len(bonafide_scores),
        n_spoof=len(spoof_scores),
        status="EMPIRICALLY_CALIBRATED",
        calibrated_at=datetime.now(timezone.utc).isoformat(),
        provenance=provenance or f"Target FAR {target_far} on {clean_partition} partition",
    )


def save_calibration(record: CalibrationRecord, path: Path | str) -> None:
    """Save calibration record to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)


def load_calibration(path: Path | str) -> CalibrationRecord | None:
    """Load calibration record from a JSON file if present and valid."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return CalibrationRecord(
            threshold=float(data["threshold"]),
            criterion=str(data.get("criterion", "EER")),
            partition=str(data.get("partition", "dev")),
            target_far=data.get("target_far"),
            measured_eer=data.get("measured_eer"),
            measured_far=data.get("measured_far"),
            measured_frr=data.get("measured_frr"),
            n_bonafide=int(data.get("n_bonafide", 0)),
            n_spoof=int(data.get("n_spoof", 0)),
            status=str(data.get("status", "EMPIRICALLY_CALIBRATED")),
            calibrated_at=str(data.get("calibrated_at", "")),
            provenance=str(data.get("provenance", "")),
        )
    except Exception:
        return None


def get_active_threshold_config(calibration_file: Path | str | None = None) -> dict[str, Any]:
    """Retrieve active decision threshold and provenance configuration.

    If an empirical calibration file exists, loads the calibrated threshold.
    Otherwise, cleanly falls back to canonical baseline threshold 0.00 while
    transparently documenting that full-dataset calibration is pending external SSD storage.
    """
    if calibration_file is not None:
        rec = load_calibration(calibration_file)
        if rec is not None:
            return {
                "active_threshold": rec.threshold,
                "is_calibrated": True,
                "criterion": rec.criterion,
                "partition": rec.partition,
                "status": rec.status,
                "calibrated_at": rec.calibrated_at,
                "provenance": rec.provenance,
                "description": (
                    f"Threshold calibrated to {rec.threshold:.4f} via {rec.criterion} "
                    f"on labelled '{rec.partition}' partition."
                ),
            }

    # Canonical default when development dataset is not downloaded
    return {
        "active_threshold": 0.0,
        "is_calibrated": False,
        "criterion": "CANONICAL_LOGIT_ZERO",
        "partition": "none",
        "status": "UNAVAILABLE_PENDING_DEV_DATA",
        "calibrated_at": "",
        "provenance": "ASVspoof 2019 baseline default (S_cm >= 0.00 -> BONAFIDE)",
        "description": (
            "Operating at standard baseline threshold theta=0.00. Automatic empirical calibration "
            "using labelled ASVspoof 2019 LA development partition is safely paused pending "
            "external SSD storage acquisition under ADR-002."
        ),
    }
