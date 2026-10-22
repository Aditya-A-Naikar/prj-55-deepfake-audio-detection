"""Dataset protocol parsing and validation module for PRJ_55.

Supports ASVspoof 2019 Logical Access (LA) protocol formats:
- Train / Dev format: `SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY LABEL`
- Eval (blinded) format: `SPEAKER_ID AUDIO_FILE_NAME [SYSTEM_ID] [KEY] [LABEL]`
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_LABELS: set[str] = {"bonafide", "spoof"}
KNOWN_SPLITS: set[str] = {"train", "dev", "eval"}


@dataclass
class ProtocolRecord:
    """Represents a single parsed protocol line."""

    line_number: int
    speaker_id: str
    utterance_id: str
    system_id: str
    key: str
    label: str | None = None
    audio_path: str | None = None
    audio_exists: bool | None = None


@dataclass
class MalformedLineRecord:
    """Information regarding a malformed protocol line."""

    line_number: int
    raw_content: str
    reason: str


@dataclass
class ProtocolValidationResult:
    """Summary of validation findings for a protocol file."""

    protocol_path: str
    split: str
    is_valid: bool
    has_labels: bool
    total_lines_read: int
    total_valid_records: int
    unique_utterances: int
    unique_speakers: int
    duplicate_utterance_count: int
    duplicate_utterances: list[str] = field(default_factory=list)
    malformed_line_count: int = 0
    malformed_lines: list[MalformedLineRecord] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    system_distribution: dict[str, int] = field(default_factory=dict)
    audio_directory_checked: str | None = None
    audio_files_found: int = 0
    audio_files_missing: int = 0
    missing_audio_samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Export result as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def generate_markdown_report(self) -> str:
        """Generate human-readable Markdown summary report."""
        lines = [
            f"# Protocol Validation Audit: {Path(self.protocol_path).name}",
            "",
            f"- **Protocol File:** `{self.protocol_path}`",
            f"- **Inferred/Specified Split:** `{self.split.upper()}`",
            f"- **Validation Status:** `{'PASS' if self.is_valid else 'FAIL'}`",
            f"- **Labels Present:** `{'Yes' if self.has_labels else 'No (Blind Eval)'}`",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Count |",
            "|---|---|",
            f"| Total Lines Read | {self.total_lines_read} |",
            f"| Valid Records | {self.total_valid_records} |",
            f"| Unique Utterances | {self.unique_utterances} |",
            f"| Unique Speakers | {self.unique_speakers} |",
            f"| Duplicate Utterance IDs | {self.duplicate_utterance_count} |",
            f"| Malformed Lines | {self.malformed_line_count} |",
        ]

        if self.has_labels:
            lines.extend([
                "",
                "## Class Distribution",
                "",
                "| Class | Count | Percentage |",
                "|---|---|---|",
            ])
            total = self.total_valid_records
            for lbl, count in sorted(self.label_distribution.items()):
                pct = (count / total * 100) if total > 0 else 0.0
                lines.append(f"| `{lbl}` | {count} | {pct:.2f}% |")

        if self.system_distribution:
            lines.extend([
                "",
                "## System / Attack Distribution",
                "",
                "| System / Key | Count |",
                "|---|---|",
            ])
            for sys_id, count in sorted(self.system_distribution.items()):
                lines.append(f"| `{sys_id}` | {count} |")

        if self.audio_directory_checked:
            lines.extend([
                "",
                "## Audio File Verification",
                "",
                f"- **Checked Directory:** `{self.audio_directory_checked}`",
                f"- **Audio Files Present:** {self.audio_files_found}",
                f"- **Audio Files Missing:** {self.audio_files_missing}",
            ])
            if self.missing_audio_samples:
                lines.append("- **Missing Sample Excerpt (up to 10):**")
                for s in self.missing_audio_samples[:10]:
                    lines.append(f"  - `{s}`")

        if self.malformed_lines:
            lines.extend([
                "",
                "## Malformed Records Encountered",
                "",
            ])
            for m in self.malformed_lines[:10]:
                lines.append(f"- Line {m.line_number}: {m.reason} (`{m.raw_content}`)")
            if len(self.malformed_lines) > 10:
                lines.append(f"- ... and {len(self.malformed_lines) - 10} more.")

        return "\n".join(lines)


def detect_split_from_path(path: Path) -> str:
    """Infer dataset split name ('train', 'dev', 'eval') from file or parent name."""
    name_lower = str(path).lower()
    if "train" in name_lower:
        return "train"
    if "dev" in name_lower:
        return "dev"
    if "eval" in name_lower:
        return "eval"
    return "unknown"


def parse_protocol_line(
    raw_line: str, line_number: int
) -> tuple[ProtocolRecord | None, MalformedLineRecord | None]:
    """Parse a single line from an ASVspoof protocol file.

    Format expected (5 columns in standard ASVspoof 2019 LA):
        SPEAKER_ID UTTERANCE_ID SYSTEM_ID KEY LABEL
    Or blind evaluation format (3, 4, or 5 columns):
        SPEAKER_ID UTTERANCE_ID [SYSTEM_ID] [KEY] [LABEL]
    """
    cleaned = raw_line.strip()
    if not cleaned or cleaned.startswith("#"):
        return None, None

    tokens = cleaned.split()
    if len(tokens) < 2:
        return None, MalformedLineRecord(
            line_number=line_number,
            raw_content=cleaned,
            reason="Line contains fewer than 2 tokens (missing speaker/utterance)",
        )

    speaker_id = tokens[0]
    utterance_id = tokens[1]
    system_id = "-"
    key = "-"
    label: str | None = None

    if len(tokens) == 2:
        # Bare evaluation list: SPEAKER UTTERANCE
        pass
    elif len(tokens) == 3:
        # Blind format: SPEAKER UTTERANCE SYSTEM
        system_id = tokens[2]
    elif len(tokens) == 4:
        # Blind format: SPEAKER UTTERANCE SYSTEM KEY
        system_id = tokens[2]
        key = tokens[3]
    elif len(tokens) == 5:
        # Standard ASVspoof 2019 format: SPEAKER UTTERANCE SYSTEM KEY LABEL
        system_id = tokens[2]
        key = tokens[3]
        candidate_label = tokens[4].lower()
        if candidate_label in VALID_LABELS:
            label = candidate_label
        else:
            return None, MalformedLineRecord(
                line_number=line_number,
                raw_content=cleaned,
                reason=f"Invalid label '{tokens[4]}'. Expected one of {sorted(VALID_LABELS)}",
            )
    else:
        return None, MalformedLineRecord(
            line_number=line_number,
            raw_content=cleaned,
            reason=f"Line has unexpected token count ({len(tokens)} > 5)",
        )

    record = ProtocolRecord(
        line_number=line_number,
        speaker_id=speaker_id,
        utterance_id=utterance_id,
        system_id=system_id,
        key=key,
        label=label,
    )
    return record, None


def parse_protocol_file(
    protocol_path: str | Path,
) -> tuple[list[ProtocolRecord], ProtocolValidationResult]:
    """Parse an ASVspoof protocol file and return records and validation summary.

    Args:
        protocol_path: Path to the protocol text file.

    Returns:
        Tuple of (list of valid ProtocolRecord objects, ProtocolValidationResult).
    """
    path_obj = Path(protocol_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    val_res = validate_protocol_file(protocol_path)
    records: list[ProtocolRecord] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            rec, _mal = parse_protocol_line(line, line_number=idx)
            if rec:
                records.append(rec)
    return records, val_res


def validate_protocol_file(
    protocol_path: str | Path,
    audio_dir: str | Path | None = None,
    split_hint: str | None = None,
    audio_ext: str = ".flac",
) -> ProtocolValidationResult:
    """Validate an ASVspoof protocol file against protocol specifications.

    Args:
        protocol_path: Path to the protocol text file.
        audio_dir: Optional directory containing corresponding audio files to verify existence.
        split_hint: Explicit split override ('train', 'dev', 'eval').
        audio_ext: Extension expected for audio files (defaults to '.flac').

    Returns:
        ProtocolValidationResult summarizing statistics and validation errors.
    """
    path_obj = Path(protocol_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    split = split_hint.lower() if split_hint else detect_split_from_path(path_obj)

    total_lines = 0
    records: list[ProtocolRecord] = []
    malformed: list[MalformedLineRecord] = []
    seen_utterances: set[str] = set()
    duplicate_utterances: list[str] = []
    speakers: set[str] = set()
    label_distribution: dict[str, int] = {}
    system_distribution: dict[str, int] = {}

    with path_obj.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            total_lines += 1
            record, mal = parse_protocol_line(line, line_number=idx)
            if mal:
                malformed.append(mal)
            elif record:
                if record.utterance_id in seen_utterances:
                    duplicate_utterances.append(record.utterance_id)
                else:
                    seen_utterances.add(record.utterance_id)

                speakers.add(record.speaker_id)

                if record.label:
                    label_distribution[record.label] = (
                        label_distribution.get(record.label, 0) + 1
                    )

                if record.system_id != "-":
                    system_distribution[record.system_id] = (
                        system_distribution.get(record.system_id, 0) + 1
                    )
                elif record.key != "-":
                    system_distribution[record.key] = (
                        system_distribution.get(record.key, 0) + 1
                    )

                records.append(record)

    # Audio file existence checking
    audio_found = 0
    audio_missing = 0
    missing_samples: list[str] = []
    audio_dir_str: str | None = None

    if audio_dir:
        audio_dir_path = Path(audio_dir)
        audio_dir_str = str(audio_dir_path)
        if audio_dir_path.is_dir():
            for rec in records:
                audio_file = audio_dir_path / f"{rec.utterance_id}{audio_ext}"
                if audio_file.exists():
                    rec.audio_path = str(audio_file)
                    rec.audio_exists = True
                    audio_found += 1
                else:
                    rec.audio_path = str(audio_file)
                    rec.audio_exists = False
                    audio_missing += 1
                    if len(missing_samples) < 20:
                        missing_samples.append(rec.utterance_id)
        else:
            # Audio directory specified but not found on filesystem
            audio_missing = len(records)
            missing_samples = [r.utterance_id for r in records[:20]]

    has_labels = len(label_distribution) > 0
    is_valid = (len(malformed) == 0) and (len(duplicate_utterances) == 0) and (total_lines > 0)

    return ProtocolValidationResult(
        protocol_path=str(path_obj),
        split=split,
        is_valid=is_valid,
        has_labels=has_labels,
        total_lines_read=total_lines,
        total_valid_records=len(records),
        unique_utterances=len(seen_utterances),
        unique_speakers=len(speakers),
        duplicate_utterance_count=len(duplicate_utterances),
        duplicate_utterances=duplicate_utterances,
        malformed_line_count=len(malformed),
        malformed_lines=malformed,
        label_distribution=label_distribution,
        system_distribution=system_distribution,
        audio_directory_checked=audio_dir_str,
        audio_files_found=audio_found,
        audio_files_missing=audio_missing,
        missing_audio_samples=missing_samples,
    )


def find_dataset_paths(data_root: str | Path) -> dict[str, dict[str, Path | None]]:
    """Discover official ASVspoof 2019 LA protocol files and audio directories under data_root.

    Returns:
        Dictionary mapping split ('train', 'dev', 'eval') to:
            {'protocol': Path | None, 'audio_dir': Path | None}
    """
    root = Path(data_root)
    results: dict[str, dict[str, Path | None]] = {
        "train": {"protocol": None, "audio_dir": None},
        "dev": {"protocol": None, "audio_dir": None},
        "eval": {"protocol": None, "audio_dir": None},
    }

    if not root.is_dir():
        return results

    # Candidate protocol directory locations
    proto_dirs = [
        root / "ASVspoof2019_LA_cm_protocols",
        root / "protocols",
        root,
    ]

    for split in ("train", "dev", "eval"):
        # 1. Resolve protocol file
        exts = ["trn.txt", "trl.txt", "txt"] if split == "train" else ["trl.txt", "txt"]
        found_proto: Path | None = None
        for pdir in proto_dirs:
            if not pdir.is_dir():
                continue
            for ext in exts:
                candidate = pdir / f"ASVspoof2019.LA.cm.{split}.{ext}"
                if candidate.is_file():
                    found_proto = candidate
                    break
            if found_proto:
                break

        # Fallback wildcard search if standard names not matched directly
        if not found_proto:
            for cand in root.glob(f"**/*LA*cm*{split}*.txt"):
                if cand.is_file():
                    found_proto = cand
                    break

        results[split]["protocol"] = found_proto

        # 2. Resolve audio directory
        audio_candidates = [
            root / f"ASVspoof2019_LA_{split}" / "flac",
            root / f"ASVspoof2019_LA_{split}",
            root / split / "flac",
            root / "flac" / split,
        ]
        found_audio: Path | None = None
        for adir in audio_candidates:
            if adir.is_dir():
                found_audio = adir
                break

        results[split]["audio_dir"] = found_audio

    return results


def validate_dataset_root(
    data_root: str | Path,
    splits: tuple[str, ...] = ("train", "dev", "eval"),
) -> dict[str, ProtocolValidationResult]:
    """Validate all discovered ASVspoof 2019 LA splits under a dataset root directory."""
    discovered = find_dataset_paths(data_root)
    validation_reports: dict[str, ProtocolValidationResult] = {}

    for split in splits:
        proto_path = discovered.get(split, {}).get("protocol")
        audio_dir = discovered.get(split, {}).get("audio_dir")
        if proto_path and proto_path.is_file():
            validation_reports[split] = validate_protocol_file(
                protocol_path=proto_path,
                audio_dir=audio_dir,
                split_hint=split,
            )

    return validation_reports

