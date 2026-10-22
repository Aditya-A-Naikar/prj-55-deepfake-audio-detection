"""ASVspoof scoring metrics: Equal Error Rate (EER) and min-tDCF.

This module implements the official ASVspoof 2019 evaluation metrics as described
in the evaluation plan:

    Todisco, M. et al. "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio
    Detection." Interspeech 2019.

**Availability notice**: EER and tDCF require labeled ground-truth data (bonafide /
spoof labels per utterance) aligned with model CM scores.  When the ASVspoof 2019
LA corpus is unavailable, this module will explicitly report that the metric cannot
be calculated rather than silently returning a placeholder.

Functions that cannot run due to missing labels/data raise
``MetricUnavailableError`` with a precise explanation.

Score convention (inherited from AASIST-L):
- ``cm_score`` = bonafide logit (index 1 from model output).
- Higher scores → more likely bonafide.
- Lower (more negative) scores → more likely spoof.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from official ASVspoof 2019 evaluation plan)
# ---------------------------------------------------------------------------

# tDCF cost parameters for ASVspoof 2019 LA condition
COST_MODEL = {
    "Pspoof": 0.05,          # Prior probability of spoofing attack
    "Cmiss_asv": 1.0,        # ASV cost of miss
    "Cfa_asv": 10.0,         # ASV cost of false alarm
    "Cmiss_cm": 1.0,         # CM cost of miss (spoof accepted as bona-fide)
    "Cfa_cm": 10.0,          # CM cost of false alarm (bona-fide rejected as spoof)
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MetricUnavailableError(RuntimeError):
    """Raised when a metric cannot be computed due to missing data or labels."""


class MetricInputError(ValueError):
    """Raised when metric inputs are invalid (empty, mismatched, non-finite)."""


# ---------------------------------------------------------------------------
# Named result types
# ---------------------------------------------------------------------------


class EERResult(NamedTuple):
    """Result of an EER computation.

    Attributes:
        eer: Equal Error Rate as a fraction in [0, 1].  Multiply by 100 for %.
        threshold: Decision threshold at which FAR ≈ FRR.
        far_at_threshold: False Acceptance Rate at the EER threshold.
        frr_at_threshold: False Rejection Rate at the EER threshold.
        n_bonafide: Number of bonafide trials used.
        n_spoof: Number of spoof trials used.
    """

    eer: float
    threshold: float
    far_at_threshold: float
    frr_at_threshold: float
    n_bonafide: int
    n_spoof: int


class TDCFResult(NamedTuple):
    """Result of a min-tDCF computation.

    Attributes:
        min_tdcf: Minimum normalised tandem Detection Cost Function.
        norm_factor: Normalisation denominator (cost of a trivial system).
        tdcf_threshold: CM threshold at which min-tDCF is achieved.
    """

    min_tdcf: float
    norm_factor: float
    tdcf_threshold: float


class RobustnessComparison(NamedTuple):
    """EER comparison between clean and degraded conditions.

    Attributes:
        condition_name: Label for the degraded condition.
        clean_eer: EER on clean recordings (fraction).
        degraded_eer: EER on degraded recordings (fraction).
        absolute_delta_eer: ``degraded_eer - clean_eer`` (fraction).
        relative_delta_eer: ``(degraded_eer - clean_eer) / clean_eer`` × 100 (%).
    """

    condition_name: str
    clean_eer: float
    degraded_eer: float
    absolute_delta_eer: float
    relative_delta_eer: float


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_score_label_arrays(
    cm_scores: np.ndarray,
    labels: np.ndarray,
    context: str = "",
) -> None:
    """Raise MetricInputError if score/label arrays are invalid."""
    if cm_scores.shape != labels.shape:
        raise MetricInputError(
            f"cm_scores and labels must have the same shape "
            f"({cm_scores.shape} vs {labels.shape}) [{context}]"
        )
    if cm_scores.size == 0:
        raise MetricInputError(f"cm_scores is empty [{context}]")
    if not np.all(np.isfinite(cm_scores)):
        n_bad = int(np.sum(~np.isfinite(cm_scores)))
        raise MetricInputError(
            f"cm_scores contains {n_bad} non-finite value(s) [{context}]"
        )
    unique_labels = set(np.unique(labels.astype(int)).tolist())
    if not {0, 1}.issuperset(unique_labels):
        raise MetricInputError(
            f"labels must be binary (0=spoof, 1=bonafide), found: {unique_labels} [{context}]"
        )


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------


def compute_eer(
    cm_scores: np.ndarray,
    labels: np.ndarray,
) -> EERResult:
    """Compute Equal Error Rate (EER) from CM scores and binary labels.

    Args:
        cm_scores: 1-D array of CM scores (bonafide logit or probability).
            Higher = more bonafide.
        labels: 1-D binary array (0 = spoof, 1 = bonafide), same length.

    Returns:
        :class:`EERResult` named tuple.

    Raises:
        MetricInputError: On empty, mismatched, or non-finite inputs.
        MetricUnavailableError: If one class has zero samples.

    Formula:
        Threshold sweeps from min→max score.  At each threshold τ:
            FAR(τ) = |{spoof trials : score ≥ τ}| / |spoof trials|
            FRR(τ) = |{bonafide trials : score < τ}| / |bonafide trials|
        EER ≈ FAR at the threshold where |FAR - FRR| is minimised.
    """
    _validate_score_label_arrays(cm_scores, labels, "EER")

    bonafide_scores = cm_scores[labels == 1]
    spoof_scores = cm_scores[labels == 0]

    if len(bonafide_scores) == 0:
        raise MetricUnavailableError(
            "No bonafide trials found in labels.  "
            "EER cannot be computed.  "
            "Check that labeled ground-truth data is available."
        )
    if len(spoof_scores) == 0:
        raise MetricUnavailableError(
            "No spoof trials found in labels.  "
            "EER cannot be computed.  "
            "Check that labeled ground-truth data is available."
        )

    # Sweep thresholds at every unique score value
    thresholds = np.sort(np.unique(cm_scores))

    far_values = np.array(
        [np.mean(spoof_scores >= t) for t in thresholds], dtype=np.float64
    )
    frr_values = np.array(
        [np.mean(bonafide_scores < t) for t in thresholds], dtype=np.float64
    )

    # Find the point of minimum |FAR - FRR|
    abs_diff = np.abs(far_values - frr_values)
    idx = int(np.argmin(abs_diff))

    eer_value = float((far_values[idx] + frr_values[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return EERResult(
        eer=eer_value,
        threshold=threshold,
        far_at_threshold=float(far_values[idx]),
        frr_at_threshold=float(frr_values[idx]),
        n_bonafide=len(bonafide_scores),
        n_spoof=len(spoof_scores),
    )


# ---------------------------------------------------------------------------
# min-tDCF computation
# ---------------------------------------------------------------------------


def compute_min_tdcf(
    cm_scores: np.ndarray,
    labels: np.ndarray,
    cost_model: dict[str, float] | None = None,
) -> TDCFResult:
    """Compute normalised minimum tandem Detection Cost Function (min-tDCF).

    This implements the normalised tDCF as used in the ASVspoof 2019 challenge.
    The normalisation makes the metric interpretable in [0, 1] for a properly
    functioning CM system.

    Note: This simplified implementation computes the CM-only tDCF cost without
    ASV score files.  Full tDCF requires both CM and ASV scores per trial.  When
    only CM scores are available, the formula degenerates to a cost weighted by
    the spoof prior and error rates.

    Args:
        cm_scores: 1-D array of CM scores.
        labels: 1-D binary label array (0=spoof, 1=bonafide).
        cost_model: Dict with tDCF cost parameters.  Defaults to the official
            ASVspoof 2019 LA parameters (see :data:`COST_MODEL`).

    Returns:
        :class:`TDCFResult` named tuple.

    Raises:
        MetricInputError: On invalid inputs.
        MetricUnavailableError: If one class has zero samples.

    Formula (simplified CM-only tDCF):
        tDCF(τ) = C_miss · P_spoof · FRR_cm(τ) + C_fa · (1-P_spoof) · FAR_cm(τ)
        min_tDCF = min_τ tDCF(τ) / C_trivial
        where C_trivial = min(C_miss · P_spoof, C_fa · (1-P_spoof))
    """
    _validate_score_label_arrays(cm_scores, labels, "min-tDCF")

    cm = cost_model or COST_MODEL
    p_spoof = cm["Pspoof"]
    c_miss = cm["Cmiss_cm"]
    c_fa = cm["Cfa_cm"]

    bonafide_scores = cm_scores[labels == 1]
    spoof_scores = cm_scores[labels == 0]

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        raise MetricUnavailableError(
            "min-tDCF requires both bonafide and spoof trials.  "
            "Ensure labeled ground-truth data from the official corpus is available."
        )

    thresholds = np.sort(np.unique(cm_scores))
    frr_values = np.array([np.mean(bonafide_scores < t) for t in thresholds], dtype=np.float64)
    far_values = np.array([np.mean(spoof_scores >= t) for t in thresholds], dtype=np.float64)

    tdcf_values = c_miss * p_spoof * frr_values + c_fa * (1.0 - p_spoof) * far_values

    # Normalise by trivial-system cost
    c_trivial = min(c_miss * p_spoof, c_fa * (1.0 - p_spoof))
    norm_factor = c_trivial if c_trivial > 0 else 1.0

    idx = int(np.argmin(tdcf_values))
    min_tdcf = float(tdcf_values[idx] / norm_factor)
    tdcf_threshold = float(thresholds[idx])

    return TDCFResult(
        min_tdcf=min_tdcf,
        norm_factor=norm_factor,
        tdcf_threshold=tdcf_threshold,
    )


# ---------------------------------------------------------------------------
# Score alignment and comparison tool
# ---------------------------------------------------------------------------


def align_score_files(
    clean_scores: dict[str, float],
    degraded_scores: dict[str, float],
    condition_name: str = "degraded",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align clean and degraded score dicts by utterance ID.

    Detects and reports missing, duplicate, and mismatched records.

    Args:
        clean_scores: Mapping of utterance_id → CM score (clean condition).
        degraded_scores: Mapping of utterance_id → CM score (degraded condition).
        condition_name: Label for the degraded condition (for error messages).

    Returns:
        Tuple of:
        - ``clean_arr``: 1-D float64 array of aligned clean scores.
        - ``degraded_arr``: 1-D float64 array of aligned degraded scores.
        - ``common_ids``: List of utterance IDs present in both dicts.

    Raises:
        MetricUnavailableError: If no common utterance IDs are found.
        MetricInputError: If either input dict is empty.
    """
    if not clean_scores:
        raise MetricInputError("clean_scores dict is empty")
    if not degraded_scores:
        raise MetricInputError(f"degraded_scores dict for '{condition_name}' is empty")

    clean_ids = set(clean_scores.keys())
    deg_ids = set(degraded_scores.keys())
    common_ids = sorted(clean_ids & deg_ids)

    only_clean = clean_ids - deg_ids
    only_deg = deg_ids - clean_ids

    if only_clean:
        logger.warning(
            "%d utterance(s) in clean_scores not found in %s: %s...",
            len(only_clean),
            condition_name,
            list(only_clean)[:5],
        )
    if only_deg:
        logger.warning(
            "%d utterance(s) in %s not found in clean_scores: %s...",
            len(only_deg),
            condition_name,
            list(only_deg)[:5],
        )

    if not common_ids:
        raise MetricUnavailableError(
            f"No common utterance IDs between clean_scores and {condition_name} scores. "
            "Cannot align for comparison.  Check that score files reference the same recordings."
        )

    clean_arr = np.array([clean_scores[uid] for uid in common_ids], dtype=np.float64)
    deg_arr = np.array([degraded_scores[uid] for uid in common_ids], dtype=np.float64)

    return clean_arr, deg_arr, common_ids


def compare_robustness(
    clean_scores: dict[str, float],
    degraded_scores: dict[str, float],
    labels: dict[str, int],
    condition_name: str = "degraded",
) -> RobustnessComparison:
    """Compare EER between clean and a single degraded condition.

    Args:
        clean_scores: Mapping of utterance_id → bonafide logit (clean).
        degraded_scores: Mapping of utterance_id → bonafide logit (degraded).
        labels: Mapping of utterance_id → 0 (spoof) or 1 (bonafide).
        condition_name: Human-readable label for the degraded condition.

    Returns:
        :class:`RobustnessComparison` named tuple.

    Raises:
        MetricUnavailableError: If labels or scores are missing or unaligned.
    """
    if not labels:
        raise MetricUnavailableError(
            "Ground-truth labels are required to compute EER.  "
            "This metric cannot be computed until the ASVspoof 2019 LA protocol "
            "files and official audio are available."
        )

    clean_arr, deg_arr, common_ids = align_score_files(
        clean_scores, degraded_scores, condition_name
    )

    # Extract labels for common utterances
    label_vals = []
    for uid in common_ids:
        if uid not in labels:
            raise MetricUnavailableError(
                f"Utterance '{uid}' found in scores but missing from labels dict.  "
                "Labels must be complete for all scored utterances."
            )
        label_vals.append(labels[uid])
    label_arr = np.array(label_vals, dtype=np.int32)

    clean_eer_result = compute_eer(clean_arr, label_arr)
    deg_eer_result = compute_eer(deg_arr, label_arr)

    clean_eer = clean_eer_result.eer
    deg_eer = deg_eer_result.eer
    abs_delta = deg_eer - clean_eer
    rel_delta = (abs_delta / clean_eer * 100.0) if clean_eer > 0 else float("inf")

    return RobustnessComparison(
        condition_name=condition_name,
        clean_eer=clean_eer,
        degraded_eer=deg_eer,
        absolute_delta_eer=abs_delta,
        relative_delta_eer=rel_delta,
    )
