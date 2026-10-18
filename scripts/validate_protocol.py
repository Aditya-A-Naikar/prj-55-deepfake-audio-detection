#!/usr/bin/env python3
"""Command-line script for validating ASVspoof protocol files (Week 1 Task 5 & Week 2 Task W2-1).

Usage:
    # Single protocol file:
    python scripts/validate_protocol.py --protocol /path/to/protocol.txt
    python scripts/validate_protocol.py --protocol /path/to/protocol.txt --audio-dir /path/to/flac/

    # Entire dataset root (via flag or ASVSPOOF_DATA_ROOT env var):
    python scripts/validate_protocol.py --data-root /path/to/LA/
    python scripts/validate_protocol.py --all-splits
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.protocol import (  # noqa: E402
    find_dataset_paths,
    validate_dataset_root,
    validate_protocol_file,
)


def main() -> int:
    env_root = os.environ.get("ASVSPOOF_DATA_ROOT")

    parser = argparse.ArgumentParser(
        description="Validate ASVspoof protocol files and verify data integrity."
    )
    parser.add_argument(
        "--protocol",
        "-p",
        type=Path,
        default=None,
        help="Path to an individual ASVspoof protocol file to validate.",
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=Path,
        default=Path(env_root) if env_root else None,
        help="Path to root dataset directory (defaults to $ASVSPOOF_DATA_ROOT if set).",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Validate all discovered splits (train, dev, eval) under data-root.",
    )
    parser.add_argument(
        "--audio-dir",
        "-a",
        type=Path,
        default=None,
        help="Optional path to directory containing associated audio files (.flac).",
    )
    parser.add_argument(
        "--split",
        "-s",
        choices=["train", "dev", "eval", "unknown"],
        default=None,
        help="Explicit dataset split designation (overrides autodetection).",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        type=Path,
        default=None,
        help="Optional path to write JSON audit summary.",
    )
    parser.add_argument(
        "--output-report",
        "-r",
        type=Path,
        default=None,
        help="Optional path to write Markdown audit report.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress printing report to standard output.",
    )

    args = parser.parse_args()

    # Mode 1: Multi-split / data-root validation
    if args.all_splits or (args.data_root and not args.protocol):
        if not args.data_root:
            print(
                "Error: --all-splits requires --data-root or $ASVSPOOF_DATA_ROOT.",
                file=sys.stderr,
            )
            return 1
        if not args.data_root.is_dir():
            print(f"Error: Dataset root does not exist: {args.data_root}", file=sys.stderr)
            return 1

        discovered = find_dataset_paths(args.data_root)
        reports = validate_dataset_root(args.data_root)

        if not reports:
            print(
                f"[WARN] No standard ASVspoof protocol files discovered under: {args.data_root}",
                file=sys.stderr,
            )
            for split, paths in discovered.items():
                print(f"  {split}: protocol={paths['protocol']}, audio_dir={paths['audio_dir']}")
            return 1

        all_valid = True
        combined_md: list[str] = [
            f"# ASVspoof 2019 LA Dataset Audit: `{args.data_root.name}`",
            "",
            f"- **Dataset Root:** `{args.data_root}`",
            f"- **Splits Validated:** {list(reports.keys())}",
            "",
        ]
        combined_json: dict[str, object] = {
            "dataset_root": str(args.data_root),
            "splits": {},
        }

        for split, res in reports.items():
            if not res.is_valid:
                all_valid = False
            combined_md.append(f"## Split: {split.upper()}")
            combined_md.append(res.generate_markdown_report())
            combined_md.append("\n---\n")
            splits_dict = combined_json["splits"]
            assert isinstance(splits_dict, dict)
            splits_dict[split] = res.to_dict()

        full_report_text = "\n".join(combined_md)
        if not args.quiet:
            print(full_report_text)

        if args.output_report:
            args.output_report.parent.mkdir(parents=True, exist_ok=True)
            args.output_report.write_text(full_report_text, encoding="utf-8")
            print(f"[INFO] Markdown audit report written to: {args.output_report}")

        if args.output_json:
            import json
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(combined_json, indent=2), encoding="utf-8")
            print(f"[INFO] JSON summary written to: {args.output_json}")

        return 0 if all_valid else 1

    # Mode 2: Single protocol validation
    if not args.protocol:
        print(
            "Error: Specify --protocol <path> or --data-root <path> (or set $ASVSPOOF_DATA_ROOT).",
            file=sys.stderr,
        )
        return 1

    try:
        result = validate_protocol_file(
            protocol_path=args.protocol,
            audio_dir=args.audio_dir,
            split_hint=args.split,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error during protocol validation: {e}", file=sys.stderr)
        return 2

    report_md = result.generate_markdown_report()
    json_str = result.to_json()

    if not args.quiet:
        print(report_md)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_str, encoding="utf-8")
        print(f"\n[INFO] JSON summary written to: {args.output_json}")

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report_md, encoding="utf-8")
        print(f"[INFO] Markdown report written to: {args.output_report}")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
