#!/usr/bin/env python3
"""Command-line script for validating ASVspoof protocol files (Week 1, Task 5).

Usage:
    python scripts/validate_protocol.py --protocol /path/to/protocol.txt
    python scripts/validate_protocol.py --protocol /path/to/protocol.txt --audio-dir /path/to/flac/
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.protocol import validate_protocol_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ASVspoof protocol files and verify data integrity."
    )
    parser.add_argument(
        "--protocol",
        "-p",
        required=True,
        type=Path,
        help="Path to the ASVspoof protocol file to validate.",
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
