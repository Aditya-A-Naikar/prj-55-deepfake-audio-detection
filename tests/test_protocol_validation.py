"""Unit tests for dataset protocol validation (Week 1, Task 5).

Uses synthetic test fixtures exclusively for software verification.
Synthetic fixtures are not real audio or official benchmark labels.
"""

from pathlib import Path

import pytest

from prj55.protocol import (
    detect_split_from_path,
    find_dataset_paths,
    parse_protocol_line,
    validate_dataset_root,
    validate_protocol_file,
)


@pytest.fixture
def synthetic_train_protocol(tmp_path: Path) -> Path:
    """Fixture with valid ASVspoof 2019 train protocol lines."""
    content = (
        "LA_0079 LA_T_1000137 - - bonafide\n"
        "LA_0079 LA_T_1000185 - A01 spoof\n"
        "LA_0080 LA_T_1000210 - A02 spoof\n"
        "LA_0081 LA_T_1000300 - - bonafide\n"
        "LA_0082 LA_T_1000450 - A06 spoof\n"
    )
    p = tmp_path / "ASVspoof2019.LA.cm.train.trn.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def synthetic_blind_eval_protocol(tmp_path: Path) -> Path:
    """Fixture with blind evaluation trial lines (no labels)."""
    content = (
        "LA_0023 LA_E_1000001 - -\n"
        "LA_0023 LA_E_1000002 - -\n"
        "LA_0045 LA_E_1000003 - -\n"
    )
    p = tmp_path / "ASVspoof2019.LA.cm.eval.trl.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def synthetic_malformed_protocol(tmp_path: Path) -> Path:
    """Fixture containing various malformed protocol entries."""
    content = (
        "LA_0079 LA_T_1000137 - - bonafide\n"
        "INCOMPLETE_LINE_ONLY_ONE_TOKEN\n"
        "LA_0080 LA_T_1000210 - A02 unknown_label\n"
        "LA_0081 LA_T_1000300 - - bonafide TOO_MANY_TOKENS_EXTRA_FIELD\n"
        "   \n"  # empty line should be ignored cleanly
        "# Comment line should be ignored\n"
    )
    p = tmp_path / "malformed_protocol.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def synthetic_duplicate_protocol(tmp_path: Path) -> Path:
    """Fixture containing duplicate utterance IDs."""
    content = (
        "LA_0079 LA_T_DUPLICATE_ID - - bonafide\n"
        "LA_0080 LA_T_UNIQUE_ID - A01 spoof\n"
        "LA_0079 LA_T_DUPLICATE_ID - A02 spoof\n"
    )
    p = tmp_path / "duplicate_protocol.txt"
    p.write_text(content, encoding="utf-8")
    return p


class TestProtocolParsing:
    """Tests for single line parsing logic."""

    def test_parse_valid_bonafide(self):
        line = "LA_0079 LA_T_1000137 - - bonafide"
        record, malformed = parse_protocol_line(line, line_number=1)
        assert malformed is None
        assert record is not None
        assert record.speaker_id == "LA_0079"
        assert record.utterance_id == "LA_T_1000137"
        assert record.label == "bonafide"
        assert record.system_id == "-"
        assert record.key == "-"

    def test_parse_valid_spoof(self):
        line = "LA_0079 LA_T_1000185 - A01 spoof"
        record, malformed = parse_protocol_line(line, line_number=2)
        assert malformed is None
        assert record is not None
        assert record.speaker_id == "LA_0079"
        assert record.utterance_id == "LA_T_1000185"
        assert record.label == "spoof"
        assert record.key == "A01"

    def test_parse_blank_and_comment(self):
        assert parse_protocol_line("   ", 1) == (None, None)
        assert parse_protocol_line("# Header", 2) == (None, None)

    def test_parse_invalid_label(self):
        line = "LA_0079 LA_T_1000185 - A01 genuine"
        record, malformed = parse_protocol_line(line, line_number=5)
        assert record is None
        assert malformed is not None
        assert "Invalid label" in malformed.reason

    def test_parse_single_token(self):
        line = "ONLY_ONE_TOKEN"
        record, malformed = parse_protocol_line(line, line_number=3)
        assert record is None
        assert malformed is not None
        assert "fewer than 2 tokens" in malformed.reason


class TestProtocolValidationFile:
    """Tests for entire protocol file validation."""

    def test_validate_clean_train_protocol(self, synthetic_train_protocol: Path):
        res = validate_protocol_file(synthetic_train_protocol)
        assert res.is_valid is True
        assert res.split == "train"
        assert res.has_labels is True
        assert res.total_valid_records == 5
        assert res.unique_speakers == 4
        assert res.unique_utterances == 5
        assert res.label_distribution == {"bonafide": 2, "spoof": 3}
        assert res.duplicate_utterance_count == 0
        assert res.malformed_line_count == 0

    def test_validate_blind_eval_protocol(self, synthetic_blind_eval_protocol: Path):
        res = validate_protocol_file(synthetic_blind_eval_protocol)
        assert res.is_valid is True
        assert res.split == "eval"
        assert res.has_labels is False
        assert res.total_valid_records == 3
        assert res.unique_speakers == 2

    def test_validate_malformed_protocol(self, synthetic_malformed_protocol: Path):
        res = validate_protocol_file(synthetic_malformed_protocol)
        assert res.is_valid is False
        assert res.malformed_line_count == 3
        assert res.total_valid_records == 1

    def test_validate_duplicate_protocol(self, synthetic_duplicate_protocol: Path):
        res = validate_protocol_file(synthetic_duplicate_protocol)
        assert res.is_valid is False
        assert res.duplicate_utterance_count == 1
        assert "LA_T_DUPLICATE_ID" in res.duplicate_utterances

    def test_missing_protocol_file_raises(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            validate_protocol_file(non_existent)


class TestAudioPathValidation:
    """Tests for verifying audio existence against protocol records."""

    def test_audio_verification_with_existing_and_missing_files(
        self, synthetic_train_protocol: Path, tmp_path: Path
    ):
        audio_dir = tmp_path / "flac"
        audio_dir.mkdir()
        # Create 2 of the 5 audio files
        (audio_dir / "LA_T_1000137.flac").touch()
        (audio_dir / "LA_T_1000185.flac").touch()

        res = validate_protocol_file(synthetic_train_protocol, audio_dir=audio_dir)
        assert res.audio_files_found == 2
        assert res.audio_files_missing == 3
        assert len(res.missing_audio_samples) == 3
        assert "LA_T_1000210" in res.missing_audio_samples

    def test_nonexistent_audio_directory(
        self, synthetic_train_protocol: Path, tmp_path: Path
    ):
        non_dir = tmp_path / "absent_audio_dir"
        res = validate_protocol_file(synthetic_train_protocol, audio_dir=non_dir)
        assert res.audio_files_missing == 5
        assert res.audio_files_found == 0


class TestReportOutputs:
    """Tests for JSON and Markdown serialization."""

    def test_markdown_and_json_generation(self, synthetic_train_protocol: Path):
        res = validate_protocol_file(synthetic_train_protocol)
        md = res.generate_markdown_report()
        assert "# Protocol Validation Audit" in md
        assert "bonafide" in md
        assert "spoof" in md

        json_out = res.to_json()
        assert '"is_valid": true' in json_out
        assert '"unique_utterances": 5' in json_out


class TestSplitAutodetection:
    """Tests split autodetection heuristics."""

    def test_autodetect_splits(self):
        assert detect_split_from_path(Path("/data/ASVspoof2019_LA_cm_train.txt")) == "train"
        assert detect_split_from_path(Path("ASVspoof2019.LA.cm.dev.trl.txt")) == "dev"
        assert detect_split_from_path(Path("evaluation_trials.txt")) == "eval"
        assert detect_split_from_path(Path("random_file.txt")) == "unknown"


@pytest.fixture
def synthetic_asvspoof_root(tmp_path: Path) -> Path:
    """Fixture creating the official ASVspoof 2019 LA directory hierarchy."""
    root = tmp_path / "LA"
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True)

    # Protocols
    (proto_dir / "ASVspoof2019.LA.cm.train.trn.txt").write_text(
        "LA_0079 LA_T_1000137 - - bonafide\nLA_0079 LA_T_1000185 - A01 spoof\n",
        encoding="utf-8",
    )
    (proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt").write_text(
        "LA_0080 LA_D_1000210 - - bonafide\nLA_0080 LA_D_1000211 - A02 spoof\n",
        encoding="utf-8",
    )
    (proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt").write_text(
        "LA_0023 LA_E_1000001 - -\nLA_0023 LA_E_1000002 - -\n",
        encoding="utf-8",
    )

    # Audio directories
    train_audio = root / "ASVspoof2019_LA_train" / "flac"
    dev_audio = root / "ASVspoof2019_LA_dev" / "flac"
    eval_audio = root / "ASVspoof2019_LA_eval" / "flac"
    train_audio.mkdir(parents=True)
    dev_audio.mkdir(parents=True)
    eval_audio.mkdir(parents=True)

    (train_audio / "LA_T_1000137.flac").touch()
    (dev_audio / "LA_D_1000210.flac").touch()
    (eval_audio / "LA_E_1000001.flac").touch()

    return root


class TestDatasetRootDiscovery:
    """Tests for discovering and validating multi-split ASVspoof hierarchies."""

    def test_find_dataset_paths_official_structure(self, synthetic_asvspoof_root: Path):
        paths = find_dataset_paths(synthetic_asvspoof_root)
        for split in ("train", "dev", "eval"):
            assert paths[split]["protocol"] is not None
            assert paths[split]["protocol"].is_file()  # type: ignore[union-attr]
            assert paths[split]["audio_dir"] is not None
            assert paths[split]["audio_dir"].is_dir()  # type: ignore[union-attr]

    def test_validate_dataset_root_all_splits(self, synthetic_asvspoof_root: Path):
        reports = validate_dataset_root(synthetic_asvspoof_root)
        assert len(reports) == 3
        assert "train" in reports
        assert "dev" in reports
        assert "eval" in reports

        assert reports["train"].is_valid is True
        assert reports["train"].total_valid_records == 2
        assert reports["train"].audio_files_found == 1
        assert reports["train"].audio_files_missing == 1

        assert reports["dev"].is_valid is True
        assert reports["eval"].is_valid is True

    def test_find_dataset_paths_nonexistent(self, tmp_path: Path):
        paths = find_dataset_paths(tmp_path / "nonexistent")
        assert paths["train"]["protocol"] is None
        assert paths["train"]["audio_dir"] is None

