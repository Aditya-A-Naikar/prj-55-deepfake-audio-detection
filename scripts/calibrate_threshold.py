#!/usr/bin/env python3
"""CLI utility for calibrating the AASIST-L decision threshold on labelled validation data.

Usage:
    python scripts/calibrate_threshold.py --scores data/dev_scores.txt \\
        --protocol data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
    python scripts/calibrate_threshold.py --synthetic --criterion EER
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add repository root and src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from prj55.calibration import (  # noqa: E402
    CalibrationCriterion,
    calibrate_threshold,
    save_calibration,
)
from prj55.protocol import parse_protocol_file  # noqa: E402


def load_score_file(path: Path) -> dict[str, float]:
    """Load countermeasure scores from a space-separated or CSV file."""
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                utt_id = parts[0]
                try:
                    score = float(parts[1])
                    scores[utt_id] = score
                except ValueError:
                    continue
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate decision threshold on labelled development data."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Path to development scores file (utt_id score)",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Path to development protocol label file (ASVspoof format)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="dev",
        help="Partition name (must be 'dev' or 'validation')",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=["EER", "TARGET_FAR"],
        default="EER",
        help="Optimization criterion (default: EER)",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help="Desired False Acceptance Rate (e.g. 0.01 for 1%% FAR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "calibration.json",
        help="Destination path for calibration JSON",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run verification on synthetic development data (for testing/dry-run)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  PRJ_55 Decision Threshold Calibration Engine")
    print(f"  Partition:  {args.partition}")
    print(f"  Criterion:  {args.criterion}")
    print("=" * 65)

    if args.synthetic:
        print("[INFO] Generating synthetic development trial distribution...")
        np.random.seed(42)
        n_bonafide, n_spoof = 1000, 4000
        # Bonafide centered at +1.5, Spoof centered at -2.0
        bonafide_scores = np.random.normal(loc=1.5, scale=1.0, size=n_bonafide)
        spoof_scores = np.random.normal(loc=-2.0, scale=1.2, size=n_spoof)
        cm_scores = np.concatenate([bonafide_scores, spoof_scores])
        labels = np.concatenate([np.ones(n_bonafide, dtype=int), np.zeros(n_spoof, dtype=int)])
        provenance = "Synthetic development benchmark trials (N=5000)"
    else:
        if not args.scores or not args.protocol:
            print("[ERROR] Both --scores and --protocol are required (or use --synthetic).")
            return 1
        if not args.scores.is_file():
            print(f"[ERROR] Scores file not found: {args.scores}")
            return 1
        if not args.protocol.is_file():
            print(f"[ERROR] Protocol file not found: {args.protocol}")
            return 1

        print(f"[INFO] Loading scores from {args.scores}...")
        scores_dict = load_score_file(args.scores)
        records, _ = parse_protocol_file(args.protocol)
        labels_dict = {r.utterance_id: (1 if r.key == "bonafide" else 0) for r in records}

        common_keys = sorted(set(scores_dict.keys()) & set(labels_dict.keys()))
        if not common_keys:
            print("[ERROR] Zero matching trial IDs found between scores and protocol.")
            return 1

        cm_scores = np.array([scores_dict[k] for k in common_keys], dtype=np.float64)
        labels = np.array([labels_dict[k] for k in common_keys], dtype=int)
        provenance = f"Calibrated on {args.scores.name} ({len(common_keys)} aligned trials)"

    criterion_enum = (
        CalibrationCriterion.EER
        if args.criterion == "EER"
        else CalibrationCriterion.TARGET_FAR
    )

    try:
        record = calibrate_threshold(
            cm_scores=cm_scores,
            labels=labels,
            partition=args.partition,
            criterion=criterion_enum,
            target_far=args.target_far,
            provenance=provenance,
        )
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        return 1

    print("\n--- Calibration Results ---")
    print(f"  Calibrated Threshold (theta): {record.threshold:+.6f}")
    print(f"  Optimization Criterion:       {record.criterion}")
    print(f"  Source Partition:             {record.partition}")
    print(f"  Aligned Trials:               {record.n_bonafide + record.n_spoof} "
          f"({record.n_bonafide} bonafide, {record.n_spoof} spoof)")
    if record.measured_eer is not None:
        print(f"  Measured EER:                 {record.measured_eer * 100:.2f}%")
    print(f"  False Alarm Rate (FAR):       {record.measured_far * 100:.2f}%")
    print(f"  False Rejection Rate (FRR):   {record.measured_frr * 100:.2f}%")
    print(f"  Timestamp:                    {record.calibrated_at}")

    if not args.synthetic:
        save_calibration(record, args.output)
        print(f"\n[SUCCESS] Calibration saved to {args.output}")
    else:
        print("\n[INFO] Synthetic dry-run complete. Record not saved to production data path.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
