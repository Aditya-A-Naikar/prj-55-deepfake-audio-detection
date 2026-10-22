#!/usr/bin/env python3
"""PRJ_55 — ASVspoof 2019 Score Evaluation CLI.

Evaluates precomputed countermeasure (CM) scores against official protocol labels.
Computes Equal Error Rate (EER), operating-point metrics (Confusion Matrix, FPR, FNR,
Precision, Recall, F1, Balanced Accuracy), ROC-AUC, and checks for min-tDCF inputs.

Threshold Governance:
    - For development split evaluation, the threshold can be chosen at the dev EER point.
    - For evaluation split evaluation, threshold selection using evaluation labels is
      strictly prohibited. A development score file (--dev-scores) or explicit threshold
      (--threshold) derived from development data must be supplied.

Usage Examples:
    # Run in synthetic dry-run mode (software-only verification):
    python scripts/evaluate_scores.py --synthetic

    # Evaluate development scores against official dev protocol:
    python scripts/evaluate_scores.py \
        --scores scores/cm_scores_dev.txt \
        --protocol /path/to/ASVspoof2019.LA.cm.dev.trl.txt \
        --output-json reports/phase3_experiment_results/eval_dev.json

    # Evaluate evaluation split with development-selected threshold:
    python scripts/evaluate_scores.py \
        --scores scores/cm_scores_eval.txt \
        --protocol /path/to/ASVspoof2019.LA.cm.eval.trl.txt \
        --dev-scores scores/cm_scores_dev.txt \
        --output-json reports/phase3_experiment_results/eval_heldout.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.metrics import (  # noqa: E402
    MetricUnavailableError,
    compute_eer,
    compute_official_min_tdcf,
    compute_operating_point_metrics,
    compute_roc_auc,
)
from prj55.protocol import (  # noqa: E402
    ProtocolRecord,
    parse_protocol_line,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_scores")


def load_score_file(score_file: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Parse a score file returning mapping of utterance_id -> score and header metadata."""
    if not score_file.is_file():
        raise FileNotFoundError(f"Score file not found: {score_file}")

    scores: dict[str, float] = {}
    metadata: dict[str, str] = {}
    duplicates = 0

    with open(score_file, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("#"):
                # Header line
                content = line_str.lstrip("#").strip()
                if ":" in content:
                    k, v = content.split(":", 1)
                    metadata[k.strip()] = v.strip()
                continue

            parts = line_str.split()
            if len(parts) < 2:
                logger.warning("Malformed score line %d: '%s'", idx, line_str)
                continue

            uid = parts[0]
            try:
                score = float(parts[1])
            except ValueError:
                logger.warning("Non-float score on line %d: '%s'", idx, parts[1])
                continue

            if uid in scores:
                duplicates += 1
            else:
                scores[uid] = score

    if duplicates > 0:
        logger.warning(
            "Detected %d duplicate utterance IDs in score file %s", duplicates, score_file
        )

    logger.info("Loaded %d unique scores from %s", len(scores), score_file.name)
    return scores, metadata


def load_protocol_labels(
    protocol_path: Path,
) -> tuple[dict[str, int], dict[str, str], list[ProtocolRecord]]:
    """Load labels (0=spoof, 1=bonafide) and system IDs from an ASVspoof protocol file."""
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    labels: dict[str, int] = {}
    system_ids: dict[str, str] = {}
    records: list[ProtocolRecord] = []

    with open(protocol_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            rec, _mal = parse_protocol_line(line, line_number=idx)
            if rec:
                records.append(rec)
                if rec.label:
                    labels[rec.utterance_id] = 1 if rec.label == "bonafide" else 0
                if rec.system_id != "-":
                    system_ids[rec.utterance_id] = rec.system_id

    logger.info("Loaded protocol %s: %d total trials, %d with ground-truth labels",
                protocol_path.name, len(records), len(labels))
    return labels, system_ids, records


def evaluate_trial_population(
    scores: dict[str, float],
    labels: dict[str, int],
    system_ids: dict[str, str] | None = None,
    threshold: float | None = None,
    threshold_provenance: str = "dev_eer",
    asv_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Align trial scores and labels, and compute all available ASVspoof metrics."""
    common_uids = sorted(set(scores.keys()) & set(labels.keys()))
    missing_scores = len(labels) - len(common_uids)
    unlabelled_scores = len(scores) - len(common_uids)

    if not common_uids:
        raise MetricUnavailableError(
            "No common trials found between score file and protocol labels. "
            "Cannot calculate benchmark metrics."
        )

    score_arr = np.array([scores[uid] for uid in common_uids], dtype=np.float64)
    label_arr = np.array([labels[uid] for uid in common_uids], dtype=np.int32)

    # 1. Equal Error Rate (EER)
    eer_res = compute_eer(score_arr, label_arr)

    # 2. Threshold determination
    if threshold is None:
        op_thresh = eer_res.threshold
        op_prov = threshold_provenance
    else:
        op_thresh = float(threshold)
        op_prov = threshold_provenance

    # 3. Operating Point Metrics
    op_metrics = compute_operating_point_metrics(
        score_arr, label_arr, threshold=op_thresh, threshold_provenance=op_prov
    )

    # 4. ROC-AUC
    auc = compute_roc_auc(score_arr, label_arr)

    # 5. min-tDCF check
    min_tdcf_val: float | None = None
    min_tdcf_status: str = "unavailable"
    min_tdcf_reason: str = ""

    if asv_scores is not None:
        asv_common = [asv_scores.get(uid) for uid in common_uids]
        if all(v is not None for v in asv_common):
            asv_arr = np.array(asv_common, dtype=np.float64)
            tdcf_res = compute_official_min_tdcf(score_arr, asv_arr, label_arr)
            min_tdcf_val = tdcf_res.min_tdcf
            min_tdcf_status = "computed"
        else:
            min_tdcf_reason = "ASV score file has missing trials for common utterances."
    else:
        min_tdcf_reason = (
            "ASV scores not supplied. Official min-tDCF requires tandem ASV system "
            "scores and cannot be computed from CM scores alone without fabricating data."
        )

    # 6. Attack Family Breakdown (if system_ids provided)
    per_attack: dict[str, Any] = {}
    if system_ids:
        attacks = sorted({system_ids[uid] for uid in common_uids if uid in system_ids})
        bonafide_scores = score_arr[label_arr == 1]
        for att in attacks:
            att_uids = [uid for uid in common_uids if system_ids.get(uid) == att]
            if not att_uids:
                continue
            att_scores = np.array([scores[uid] for uid in att_uids], dtype=np.float64)
            # Combine bonafide + this attack to compute attack-specific EER
            comb_scores = np.concatenate([bonafide_scores, att_scores])
            comb_labels = np.concatenate([np.ones(len(bonafide_scores), dtype=np.int32),
                                          np.zeros(len(att_scores), dtype=np.int32)])
            try:
                att_eer = compute_eer(comb_scores, comb_labels).eer
                per_attack[att] = {
                    "trials": len(att_uids),
                    "eer": att_eer,
                    "eer_pct": att_eer * 100.0,
                }
            except Exception as exc:
                per_attack[att] = {"trials": len(att_uids), "error": str(exc)}

    return {
        "trial_alignment": {
            "total_scored_in_file": len(scores),
            "total_protocol_trials": len(labels),
            "aligned_trials": len(common_uids),
            "missing_from_scores": missing_scores,
            "unlabelled_scores": unlabelled_scores,
            "n_bonafide": eer_res.n_bonafide,
            "n_spoof": eer_res.n_spoof,
        },
        "eer": {
            "fraction": eer_res.eer,
            "percent": eer_res.eer * 100.0,
            "threshold": eer_res.threshold,
            "far_at_threshold": eer_res.far_at_threshold,
            "frr_at_threshold": eer_res.frr_at_threshold,
        },
        "operating_point": {
            "threshold": op_metrics.threshold,
            "threshold_provenance": op_metrics.threshold_provenance,
            "tp": op_metrics.tp,
            "tn": op_metrics.tn,
            "fp": op_metrics.fp,
            "fn": op_metrics.fn,
            "fpr": op_metrics.fpr,
            "fnr": op_metrics.fnr,
            "precision": op_metrics.precision,
            "recall": op_metrics.recall,
            "f1_score": op_metrics.f1_score,
            "balanced_accuracy": op_metrics.balanced_accuracy,
        },
        "roc_auc": auc,
        "min_tdcf": {
            "status": min_tdcf_status,
            "value": min_tdcf_val,
            "reason": min_tdcf_reason,
        },
        "per_attack_breakdown": per_attack,
    }


def generate_synthetic_evaluation() -> dict[str, Any]:
    """Create a deterministic synthetic evaluation for software verification."""
    rng = np.random.default_rng(42)
    scores: dict[str, float] = {}
    labels: dict[str, int] = {}
    system_ids: dict[str, str] = {}

    for i in range(20):
        uid = f"LA_T_SYNTH_{i:05d}"
        is_bf = (i % 2 == 0)
        labels[uid] = 1 if is_bf else 0
        mean = 1.5 if is_bf else -1.5
        scores[uid] = float(rng.normal(mean, 0.5))
        system_ids[uid] = "bonafide" if is_bf else f"A{(i % 6) + 1:02d}"

    res = evaluate_trial_population(
        scores=scores,
        labels=labels,
        system_ids=system_ids,
        threshold=0.0,
        threshold_provenance="synthetic_zero",
    )
    res["notice"] = "SYNTHETIC DRY-RUN — Software verification only, not benchmark data."
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="PRJ_55 ASVspoof 2019 Score Evaluator")
    parser.add_argument("--scores", type=Path, default=None, help="Path to CM score file")
    parser.add_argument(
        "--protocol", type=Path, default=None, help="Path to official protocol file"
    )
    parser.add_argument(
        "--dev-scores",
        type=Path,
        default=None,
        help="Dev scores to derive operating threshold",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Explicit decision threshold"
    )
    parser.add_argument(
        "--asv-scores",
        type=Path,
        default=None,
        help="Optional ASV score file for tandem min-tDCF",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Destination JSON path"
    )
    parser.add_argument(
        "--output-report", type=Path, default=None, help="Destination Markdown report path"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run with synthetic fixtures for dry-run verification",
    )

    args = parser.parse_args()

    if args.synthetic:
        logger.info("Executing synthetic evaluation dry-run...")
        results = generate_synthetic_evaluation()
        output_json = json.dumps(results, indent=2)
        print(output_json)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(output_json, encoding="utf-8")
            logger.info("Synthetic results written to %s", args.output_json)
        return 0

    if not args.scores or not args.scores.is_file():
        logger.error("Valid --scores file required. (Or use --synthetic for dry run).")
        return 1

    if not args.protocol or not args.protocol.is_file():
        logger.error("Valid --protocol file required. (Or use --synthetic for dry run).")
        return 1

    scores, score_meta = load_score_file(args.scores)
    labels, system_ids, _records = load_protocol_labels(args.protocol)

    # Threshold selection policy
    selected_threshold = args.threshold
    threshold_prov = "explicit_cli" if args.threshold is not None else "dev_eer"

    if selected_threshold is None and args.dev_scores:
        logger.info("Deriving operating threshold from development scores: %s", args.dev_scores)
        dev_scores, _ = load_score_file(args.dev_scores)
        common_dev = sorted(set(dev_scores.keys()) & set(labels.keys()))
        if common_dev:
            s_arr = np.array([dev_scores[u] for u in common_dev])
            l_arr = np.array([labels[u] for u in common_dev])
            dev_eer_res = compute_eer(s_arr, l_arr)
            selected_threshold = dev_eer_res.threshold
            threshold_prov = f"dev_scores_eer ({args.dev_scores.name})"
            logger.info(
                "Selected operating threshold: %.6f (from %s)",
                selected_threshold,
                threshold_prov,
            )

    # Optional ASV scores
    asv_dict: dict[str, float] | None = None
    if args.asv_scores and args.asv_scores.is_file():
        asv_dict, _ = load_score_file(args.asv_scores)

    results = evaluate_trial_population(
        scores=scores,
        labels=labels,
        system_ids=system_ids,
        threshold=selected_threshold,
        threshold_provenance=threshold_prov,
        asv_scores=asv_dict,
    )

    results["score_metadata"] = score_meta
    results["evaluation_timestamp"] = datetime.now(timezone.utc).isoformat()

    # Print summary
    print("\n" + "=" * 65)
    print("  PRJ_55 ASVspoof 2019 Score Evaluation Summary")
    print("=" * 65)
    al = results["trial_alignment"]
    print(
        f"  Aligned Trials:    {al['aligned_trials']} "
        f"(Bonafide: {al['n_bonafide']}, Spoof: {al['n_spoof']})"
    )
    print(
        f"  Equal Error Rate:  {results['eer']['percent']:.4f}% "
        f"(threshold: {results['eer']['threshold']:.4f})"
    )
    print(f"  ROC-AUC:           {results['roc_auc']:.4f}")
    op = results["operating_point"]
    print(f"  Operating Point:   Threshold = {op['threshold']:.4f} ({op['threshold_provenance']})")
    print(f"  FPR / FNR:         {op['fpr']*100:.2f}% / {op['fnr']*100:.2f}%")
    print(f"  Precision / Recall:{op['precision']:.4f} / {op['recall']:.4f}")
    print(f"  F1 Score / BalAcc: {op['f1_score']:.4f} / {op['balanced_accuracy']:.4f}")
    tdcf = results["min_tdcf"]
    tdcf_str = (
        f"{tdcf['value']:.4f}"
        if tdcf["value"] is not None
        else f"N/A ({tdcf['reason']})"
    )
    print(f"  min-tDCF:          {tdcf_str}")
    print("=" * 65)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("JSON evaluation saved to %s", args.output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
