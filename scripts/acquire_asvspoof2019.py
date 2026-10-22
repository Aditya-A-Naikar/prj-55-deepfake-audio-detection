#!/usr/bin/env python3
"""PRJ_55 — ASVspoof 2019 Logical Access (LA) Dataset Acquisition Tool.

Safely downloads, verifies, and extracts the official ASVspoof 2019 LA corpus
onto a designated storage volume (external SSD recommended).

Authoritative Source Metadata (University of Edinburgh DataShare):
    - Handle: https://datashare.ed.ac.uk/handle/10283/3336
    - Bitstream ID: a9f87c35-f055-4015-80e2-2fdff0d46269
    - Direct URL: https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content
    - Archive Name: LA.zip
    - Exact Size: 7,640,952,520 bytes (~7.116 GiB)
    - Authoritative MD5: 30c98f11d8b2bc21f2c257bfd78bb5c5

Safety Guards:
    - Inspects destination storage space (minimum 50 GiB required for archive + extraction).
    - Detects whether destination is the internal root volume and enforces ADR-002.
    - Validates archive integrity with MD5 before unpacking.
    - Employs Zip Slip safe extraction (rejects paths escaping target dir).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_asvspoof")

DATASHARE_LA_URL = (
    "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
    "a9f87c35-f055-4015-80e2-2fdff0d46269/content"
)
EXPECTED_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
EXPECTED_SIZE_BYTES = 7_640_952_520
MIN_REQUIRED_FREE_GIB = 50.0


def check_storage_safety(target_dir: Path, allow_internal: bool = False) -> tuple[bool, str]:
    """Check storage capacity and volume safety for target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    free_gib = usage.free / (1024 ** 3)

    # Check if target is on internal startup disk
    is_internal = False
    try:
        resolved = target_dir.resolve()
        if str(resolved).startswith(("/System", "/Users", "/private", "/var")):
            is_internal = True
    except Exception:
        pass

    if is_internal and not allow_internal:
        return (
            False,
            f"Target directory {target_dir} resides on the internal startup volume. "
            f"Per ADR-002 and project safety rules, corpus acquisition must target an "
            f"external volume to protect internal SSD endurance and system operations. "
            f"(Free on internal: {free_gib:.1f} GiB). Connect an external SSD and point "
            f"ASVSPOOF_DATA_ROOT to it, or pass --allow-internal if authorized."
        )

    if free_gib < MIN_REQUIRED_FREE_GIB:
        return (
            False,
            f"Insufficient storage space: {target_dir} has {free_gib:.1f} GiB free, "
            f"but at least {MIN_REQUIRED_FREE_GIB:.1f} GiB is required for the 7.64 GB archive "
            f"and extracted audio partitions."
        )

    return True, f"Storage safety check passed: {free_gib:.1f} GiB available on volume."


def verify_file_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file in streaming chunks."""
    logger.info("Computing MD5 checksum for %s...", file_path.name)
    md5 = hashlib.md5()
    t0 = time.perf_counter()
    bytes_read = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
            bytes_read += len(chunk)
    elapsed = time.perf_counter() - t0
    computed = md5.hexdigest().lower()
    match = (computed == expected_md5.lower())
    rate = (bytes_read / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    logger.info("MD5 computed: %s in %.1fs (%.1f MB/s) | Expected: %s | Match: %s",
                computed, elapsed, rate, expected_md5, match)
    return match


def safe_extract_zip(archive_path: Path, dest_dir: Path) -> int:
    """Safely extract zip archive preventing Zip Slip vulnerabilities."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()

    logger.info("Extracting %s to %s...", archive_path.name, dest_dir)
    extracted_count = 0
    t0 = time.perf_counter()

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        total_members = len(members)

        for idx, member in enumerate(members):
            # Security: Ensure target path stays within dest_resolved
            target_path = (dest_dir / member.filename).resolve()
            if not str(target_path).startswith(str(dest_resolved)):
                raise ValueError(
                    f"Unsafe zip member detected (Zip Slip vulnerability): {member.filename}"
                )

            zf.extract(member, dest_dir)
            extracted_count += 1

            if (idx + 1) % 10000 == 0 or (idx + 1) == total_members:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Extracted %d / %d members (%.1fs)...", idx + 1, total_members, elapsed
                )

    logger.info(
        "Extraction complete: %d members extracted in %.1fs.",
        extracted_count,
        time.perf_counter() - t0,
    )
    return extracted_count


def verify_extracted_structure(base_dir: Path) -> dict[str, Any]:
    """Verify presence of key ASVspoof 2019 LA folders and protocols."""
    la_root = base_dir / "LA" if (base_dir / "LA").is_dir() else base_dir

    train_dir = la_root / "ASVspoof2019_LA_train" / "flac"
    dev_dir = la_root / "ASVspoof2019_LA_dev" / "flac"
    eval_dir = la_root / "ASVspoof2019_LA_eval" / "flac"
    proto_dir = la_root / "ASVspoof2019_LA_cm_protocols"

    results = {
        "la_root": str(la_root),
        "train_audio_count": len(list(train_dir.glob("*.flac"))) if train_dir.is_dir() else 0,
        "dev_audio_count": len(list(dev_dir.glob("*.flac"))) if dev_dir.is_dir() else 0,
        "eval_audio_count": len(list(eval_dir.glob("*.flac"))) if eval_dir.is_dir() else 0,
        "protocols_present": list(proto_dir.glob("*.txt")) if proto_dir.is_dir() else [],
    }

    # Sample decode check
    sample_flac = next(dev_dir.glob("*.flac"), None) if dev_dir.is_dir() else None
    if sample_flac:
        try:
            import soundfile as sf
            data, sr = sf.read(str(sample_flac), dtype="float32")
            results["sample_decode_status"] = "OK"
            results["sample_sr"] = sr
            results["sample_samples"] = len(data)
        except Exception as exc:
            results["sample_decode_status"] = f"FAIL: {exc}"
    else:
        results["sample_decode_status"] = "NO_SAMPLE"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="PRJ_55 ASVspoof 2019 LA Acquisition Tool")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Target directory on external storage (e.g. /Volumes/ExternalSSD/asvspoof2019)",
    )
    parser.add_argument(
        "--allow-internal",
        action="store_true",
        help="Override ADR-002 to permit internal disk",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing download/extraction",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract existing LA.zip archive",
    )

    args = parser.parse_args()

    data_dir = args.data_dir or (
        Path(os.environ["ASVSPOOF_DATA_ROOT"]) if "ASVSPOOF_DATA_ROOT" in os.environ else None
    )
    if not data_dir:
        logger.error(
            "Target data directory not specified. Pass --data-dir or set ASVSPOOF_DATA_ROOT."
        )
        return 1

    # Pre-flight safety check
    safe, msg = check_storage_safety(data_dir, allow_internal=args.allow_internal)
    if not safe:
        logger.error("[STORAGE BLOCKER] %s", msg)
        return 2

    logger.info(msg)
    archive_path = data_dir / "LA.zip"

    if args.verify_only:
        if archive_path.is_file():
            valid = verify_file_md5(archive_path, EXPECTED_MD5)
            status_str = "PASS" if valid else "FAIL"
            logger.info("Archive %s MD5 verification: %s", archive_path.name, status_str)
        struct = verify_extracted_structure(data_dir)
        logger.info("Extracted structure: %s", struct)
        return 0

    if args.extract_only:
        if not archive_path.is_file():
            logger.error("LA.zip not found at %s", archive_path)
            return 1
        safe_extract_zip(archive_path, data_dir)
        struct = verify_extracted_structure(data_dir)
        logger.info("Structure verification: %s", struct)
        return 0

    # Acquisition logic (download)
    logger.info("Initiating download of ASVspoof 2019 LA from Edinburgh DataShare...")
    logger.info("URL: %s", DATASHARE_LA_URL)
    logger.info("Target: %s", archive_path)

    try:
        urllib.request.urlretrieve(DATASHARE_LA_URL, str(archive_path))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return 3

    if not verify_file_md5(archive_path, EXPECTED_MD5):
        logger.error("Downloaded archive failed MD5 checksum verification!")
        return 4

    safe_extract_zip(archive_path, data_dir)
    struct = verify_extracted_structure(data_dir)
    logger.info("Corpus acquired and verified: %s", struct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
