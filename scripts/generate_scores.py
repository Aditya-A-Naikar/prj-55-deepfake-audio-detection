#!/usr/bin/env python3
"""PRJ_55 — ASVspoof 2019 AASIST-L Baseline Score Generator.

Loads the pretrained AASIST-L countermeasure model and computes bona fide logits
(CM scores) for utterances listed in an ASVspoof 2019 Logical Access protocol.

Output Score File Format:
    Each line contains whitespace-separated fields:
        <UTTERANCE_ID> <CM_SCORE>
    Header lines prefixed with '#' record model provenance, checkpoint hash,
    and generation metadata.

Usage Examples:
    # Run in synthetic dry-run mode (software-only verification):
    python scripts/generate_scores.py --synthetic --num-synthetic 20

    # Generate scores on official development split:
    python scripts/generate_scores.py \\
        --data-dir /Volumes/ExternalSSD/asvspoof2019 \\
        --split dev \\
        --checkpoint models/weights/AASIST-L.pth \\
        --output-file scores/cm_scores_dev.txt

    # Generate scores on evaluation split with CPU batching:
    python scripts/generate_scores.py \\
        --data-dir /Volumes/ExternalSSD/asvspoof2019 \\
        --split eval \\
        --device cpu \\
        --output-file scores/cm_scores_eval.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, pad_or_truncate_waveform  # noqa: E402
from prj55.models.aasist import load_aasist_l  # noqa: E402
from prj55.protocol import (  # noqa: E402
    ProtocolRecord,
    find_dataset_paths,
    parse_protocol_file,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("generate_scores")


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_trials(
    num_trials: int = 20, seed: int = 42
) -> list[tuple[str, np.ndarray, str]]:
    """Generate synthetic test signals and pseudo-records for software verification."""
    rng = np.random.default_rng(seed)
    trials: list[tuple[str, np.ndarray, str]] = []
    for i in range(num_trials):
        utt_id = f"LA_T_SYNTH_{i:05d}"
        label = "bonafide" if (i % 2 == 0) else "spoof"
        freq = 440.0 if label == "bonafide" else 880.0
        t = np.linspace(0, 3.0, int(TARGET_SAMPLE_RATE * 3.0), endpoint=False)
        waveform = 0.4 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(len(t))
        trials.append((utt_id, waveform.astype(np.float32), label))
    return trials


def run_score_generation(
    model: torch.nn.Module,
    records: list[ProtocolRecord],
    audio_dir: Path,
    output_file: Path,
    device: str = "cpu",
    batch_size: int = 16,
    max_trials: int | None = None,
    checkpoint_hash: str = "unknown",
    split: str = "dev",
) -> dict[str, Any]:
    """Execute model inference over protocol records and stream scores to disk."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    target_records = records[:max_trials] if max_trials else records

    logger.info("Processing %d records on device %s...", len(target_records), device)
    model.eval()

    n_scored = 0
    n_missing = 0
    n_failed = 0
    n_duplicates = 0
    seen_ids: set[str] = set()

    t_start = time.perf_counter()

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Write metadata header
        out_f.write("# PRJ_55 ASVspoof 2019 LA Countermeasure Scores\n")
        out_f.write(f"# Split: {split}\n")
        out_f.write(f"# Checkpoint SHA-256: {checkpoint_hash}\n")
        out_f.write(f"# Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        out_f.write("# Score convention: Bonafide logit (higher = more bonafide)\n")
        out_f.write("# Columns: UTTERANCE_ID CM_SCORE\n")

        for idx, rec in enumerate(target_records):
            uid = rec.utterance_id
            if uid in seen_ids:
                logger.warning(
                    "Duplicate utterance ID: %s (line %d). Skipping.", uid, rec.line_number
                )
                n_duplicates += 1
                continue
            seen_ids.add(uid)

            # Resolve audio path
            audio_path = audio_dir / f"{uid}.flac"
            if not audio_path.is_file():
                logger.warning("Audio file missing for trial %s: %s", uid, audio_path)
                n_missing += 1
                continue

            try:
                data, sr = sf.read(str(audio_path), dtype="float32")
                if sr != TARGET_SAMPLE_RATE:
                    logger.warning(
                        "Resampling trial %s from %d Hz to %d Hz", uid, sr, TARGET_SAMPLE_RATE
                    )
                if data.ndim == 2:
                    data = np.mean(data, axis=1).astype(np.float32)

                tensor = pad_or_truncate_waveform(data, target_length=TARGET_SAMPLES).to(device)

                with torch.no_grad():
                    _feat, logits = model(tensor)

                bf_score = float(logits[0, 1].item())
                if not np.isfinite(bf_score):
                    logger.error("Non-finite score produced for %s: %s", uid, bf_score)
                    n_failed += 1
                    continue

                out_f.write(f"{uid} {bf_score:.6f}\n")
                n_scored += 1

                if n_scored % 1000 == 0 or (idx + 1) == len(target_records):
                    elapsed = time.perf_counter() - t_start
                    rate = n_scored / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Scored %d / %d trials (%.1f trials/sec)",
                        n_scored,
                        len(target_records),
                        rate,
                    )

            except Exception as exc:
                logger.error("Failed to process trial %s: %s", uid, exc)
                n_failed += 1

    total_time = time.perf_counter() - t_start

    summary = {
        "split": split,
        "checkpoint_hash": checkpoint_hash,
        "total_records": len(target_records),
        "scored_trials": n_scored,
        "missing_audio": n_missing,
        "failed_inferences": n_failed,
        "duplicate_trials": n_duplicates,
        "elapsed_seconds": total_time,
        "output_file": str(output_file),
    }

    logger.info(
        "Score generation complete: %d scored, %d missing, %d failed in %.2fs.",
        n_scored,
        n_missing,
        n_failed,
        total_time,
    )
    return summary


def run_synthetic_score_generation(
    model: torch.nn.Module,
    num_trials: int,
    output_file: Path,
    device: str = "cpu",
    checkpoint_hash: str = "unknown",
) -> dict[str, Any]:
    """Run score generation on synthetic waveforms for software verification."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    trials = generate_synthetic_trials(num_trials=num_trials)
    logger.info("Executing synthetic score generation on %d trials...", len(trials))

    model.eval()
    t_start = time.perf_counter()
    n_scored = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(
            "# PRJ_55 SYNTHETIC TEST SCORES (Software verification only — not benchmark data)\n"
        )
        out_f.write(f"# Checkpoint SHA-256: {checkpoint_hash}\n")
        out_f.write(f"# Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        out_f.write("# Columns: UTTERANCE_ID CM_SCORE\n")

        for uid, waveform, _label in trials:
            tensor = pad_or_truncate_waveform(waveform, target_length=TARGET_SAMPLES).to(device)
            with torch.no_grad():
                _feat, logits = model(tensor)
            bf_score = float(logits[0, 1].item())
            out_f.write(f"{uid} {bf_score:.6f}\n")
            n_scored += 1

    total_time = time.perf_counter() - t_start
    summary = {
        "mode": "synthetic_dry_run",
        "scored_trials": n_scored,
        "elapsed_seconds": total_time,
        "output_file": str(output_file),
    }
    logger.info("Synthetic run complete: %d trials scored in %.2fs.", n_scored, total_time)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="PRJ_55 ASVspoof 2019 Score Generator")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Root directory of ASVspoof 2019 LA corpus"
    )
    parser.add_argument(
        "--split", choices=["train", "dev", "eval"], default="dev", help="Dataset split"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "weights" / "AASIST-L.pth",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--output-file", type=Path, default=None, help="Output score file destination"
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Inference device"
    )
    parser.add_argument(
        "--max-trials", type=int, default=None, help="Optional maximum trials to process"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run with synthetic test signals (software verification)",
    )
    parser.add_argument(
        "--num-synthetic", type=int, default=20, help="Number of synthetic trials"
    )

    args = parser.parse_args()

    # Model checkpoint validation
    if not args.checkpoint.is_file():
        logger.error("AASIST-L checkpoint not found: %s", args.checkpoint)
        return 1

    ckpt_hash = _compute_file_sha256(args.checkpoint)
    logger.info("Loading AASIST-L (SHA-256: %s...)", ckpt_hash[:16])
    model = load_aasist_l(args.checkpoint, device=args.device)

    # Output path default
    default_dir = REPO_ROOT / "reports" / "phase3_experiment_results"
    default_out = default_dir / f"cm_scores_{args.split}.txt"
    output_file = args.output_file or default_out

    if args.synthetic:
        summary = run_synthetic_score_generation(
            model=model,
            num_trials=args.num_synthetic,
            output_file=output_file,
            device=args.device,
            checkpoint_hash=ckpt_hash,
        )
        print(json.dumps(summary, indent=2))
        return 0

    # Physical dataset path discovery
    env_root = os.environ.get("ASVSPOOF_DATA_ROOT")
    data_dir = args.data_dir or (Path(env_root) if env_root else None)
    if not data_dir or not data_dir.is_dir():
        logger.error(
            "ASVspoof 2019 dataset root not found or unmounted. "
            "Specify --data-dir or export ASVSPOOF_DATA_ROOT. "
            "To test software functionality without dataset, use --synthetic."
        )
        return 2

    # Discover official paths
    dataset_paths = find_dataset_paths(data_dir)
    if args.split not in dataset_paths:
        logger.error("Split '%s' not discovered in %s", args.split, data_dir)
        return 2

    split_info = dataset_paths[args.split]
    proto_path = split_info["protocol"]
    audio_dir = split_info["audio_dir"]

    logger.info("Parsing %s protocol: %s", args.split, proto_path)
    records, val_res = parse_protocol_file(proto_path, split=args.split, check_audio=False)
    logger.info("Parsed %d records (%d unique).", len(records), val_res.unique_utterances)

    summary = run_score_generation(
        model=model,
        records=records,
        audio_dir=audio_dir,
        output_file=output_file,
        device=args.device,
        max_trials=args.max_trials,
        checkpoint_hash=ckpt_hash,
        split=args.split,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
