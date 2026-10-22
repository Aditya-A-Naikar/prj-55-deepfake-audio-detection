"""Tests for the ASVspoof dataset acquisition tool (scripts/acquire_asvspoof2019.py).

Verifies storage safety guards (internal disk prevention per ADR-002, free space check),
MD5 verification logic, and Zip Slip path traversal protection.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.acquire_asvspoof2019 import (
    check_storage_safety,
    safe_extract_zip,
    verify_file_md5,
)


class TestStorageSafetyGuard:
    """Tests for check_storage_safety."""

    def test_internal_disk_without_override_rejected(self, tmp_path: Path) -> None:
        # tmp_path on macOS is under /private/var/folders -> internal disk
        safe, msg = check_storage_safety(tmp_path, allow_internal=False)
        assert safe is False
        assert "internal startup volume" in msg
        assert "ADR-002" in msg

    def test_internal_disk_with_override_accepted_if_space(
        self, tmp_path: Path
    ) -> None:
        safe, msg = check_storage_safety(tmp_path, allow_internal=True)
        # On this machine, internal disk has > 50 GiB available now
        assert safe is True
        assert "Storage safety check passed" in msg


class TestMD5Verification:
    """Tests for verify_file_md5."""

    def test_verify_md5_success(self, tmp_path: Path) -> None:
        dummy_file = tmp_path / "dummy.bin"
        content = b"PRJ_55 deepfake speech detection test content"
        dummy_file.write_bytes(content)
        expected_md5 = hashlib.md5(content).hexdigest()

        assert verify_file_md5(dummy_file, expected_md5) is True

    def test_verify_md5_failure(self, tmp_path: Path) -> None:
        dummy_file = tmp_path / "dummy.bin"
        dummy_file.write_bytes(b"content A")
        wrong_md5 = "00000000000000000000000000000000"

        assert verify_file_md5(dummy_file, wrong_md5) is False


class TestZipExtractionSafety:
    """Tests for safe_extract_zip preventing Zip Slip."""

    def test_safe_zip_extracts_correctly(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "valid.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("folder/sample.txt", "hello world")

        count = safe_extract_zip(zip_path, extract_dir)
        assert count == 1
        assert (extract_dir / "folder" / "sample.txt").is_file()
        assert (extract_dir / "folder" / "sample.txt").read_text() == "hello world"

    def test_zip_slip_path_traversal_raises(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "malicious.zip"
        extract_dir = tmp_path / "extracted"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../evil.txt", "malicious payload")

        with pytest.raises(ValueError, match="Zip Slip"):
            safe_extract_zip(zip_path, extract_dir)
