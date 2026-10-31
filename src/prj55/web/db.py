"""Durable SQLite storage and cryptographic report sharing for PRJ_55.

Provides ACID-compliant local metadata persistence, schema migrations, and
token-based shareable forensic reports in strict compliance with ADR-002:
- Analysis metadata, timestamps, model provenance, and cryptographic SHA-256
  digests are stored persistently in a local SQLite database.
- Raw audio files are NEVER stored permanently on server storage.
- Shareable links use high-entropy unguessable bearer tokens (secrets.token_urlsafe)
  with configurable expiration and revocation.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "history.db"

CURRENT_SCHEMA_VERSION = 1


def get_db_path(override_path: Path | str | None = None) -> Path:
    """Resolve database path from argument, environment variable, or default."""
    if override_path is not None:
        return Path(override_path)
    env_path = os.environ.get("PRJ55_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Create a configured SQLite connection with foreign keys and WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: Path | str | None = None) -> Path:
    """Initialize SQLite database, apply migrations, and verify schema."""
    target_path = get_db_path(db_path)
    with _get_connection(target_path) as conn:
        # Schema migration tracking table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT NOT NULL
            );
            """
        )

        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_migrations;")
        row = cursor.fetchone()
        latest_version = row[0] if row and row[0] is not None else 0

        if latest_version < 1:
            # Migration 1: Analysis history and token-authenticated shared reports
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    duration_s REAL,
                    sample_rate INTEGER DEFAULT 16000,
                    channels INTEGER DEFAULT 1,
                    file_size_bytes INTEGER DEFAULT 0,
                    bonafide_logit REAL NOT NULL,
                    spoof_logit REAL NOT NULL,
                    softmax_bonafide REAL NOT NULL,
                    softmax_spoof REAL NOT NULL,
                    threshold REAL NOT NULL DEFAULT 0.0,
                    prediction TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    real_time_factor REAL DEFAULT 0.0,
                    device TEXT NOT NULL DEFAULT 'cpu',
                    model_id TEXT NOT NULL DEFAULT 'AASIST-L',
                    checkpoint_sha256 TEXT,
                    condition TEXT NOT NULL DEFAULT 'clean',
                    mitigation_applied INTEGER NOT NULL DEFAULT 0,
                    raw_result_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_timestamp
                ON analysis_history(timestamp DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_sha256
                ON analysis_history(sha256);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_prediction
                ON analysis_history(prediction);
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_reports (
                    share_token TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (record_id) REFERENCES analysis_history(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shared_reports_token
                ON shared_reports(share_token);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shared_reports_record_id
                ON shared_reports(record_id);
                """
            )

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute(
                """
                INSERT INTO schema_migrations (version, applied_at, description)
                VALUES (1, ?, 'Initial schema: analysis_history and shared_reports tables');
                """,
                (now_iso,),
            )
            conn.commit()

    return target_path


def save_analysis(
    record: dict[str, Any],
    raw_result: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Persist an analysis record to local SQLite storage."""
    target_path = get_db_path(db_path)
    rec_id = str(record.get("id") or record.get("record_id") or "")
    if not rec_id:
        raise ValueError("Record must have an 'id' or 'record_id'")

    ts = str(record.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    filename = str(record.get("filename") or "unknown.wav")
    sha256 = str(record.get("sha256") or "")
    duration_s = float(record.get("duration_s") or record.get("duration") or 0.0)
    sample_rate = int(record.get("sample_rate") or 16000)
    channels = int(record.get("channels") or 1)
    file_size_bytes = int(record.get("file_size_bytes") or 0)

    # Scores
    score = record.get("score")
    bf_raw = record.get("bonafide_logit")
    if bf_raw is not None:
        bonafide_logit = float(bf_raw)
    elif score is not None:
        bonafide_logit = float(score)
    else:
        bonafide_logit = 0.0
    spoof_logit = float(record.get("spoof_logit") or 0.0)
    softmax_bonafide = float(record.get("softmax_bonafide") or 0.5)
    softmax_spoof = float(record.get("softmax_spoof") or 0.5)
    threshold = float(record.get("threshold") or 0.0)
    prediction = str(record.get("prediction") or "UNKNOWN")
    latency_ms = float(record.get("latency_ms") or 0.0)
    real_time_factor = float(record.get("real_time_factor") or 0.0)
    device = str(record.get("device") or "cpu")
    model_id = str(record.get("model_id") or "AASIST-L")
    checkpoint_sha256 = record.get("checkpoint_sha256")
    condition = str(record.get("condition") or "clean")
    mitigation_applied = 1 if record.get("mitigation_applied") else 0
    raw_json = json.dumps(raw_result) if raw_result is not None else None

    with _get_connection(target_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analysis_history (
                id, timestamp, filename, sha256, duration_s, sample_rate, channels,
                file_size_bytes, bonafide_logit, spoof_logit, softmax_bonafide,
                softmax_spoof, threshold, prediction, latency_ms, real_time_factor,
                device, model_id, checkpoint_sha256, condition, mitigation_applied,
                raw_result_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            );
            """,
            (
                rec_id,
                ts,
                filename,
                sha256,
                duration_s,
                sample_rate,
                channels,
                file_size_bytes,
                bonafide_logit,
                spoof_logit,
                softmax_bonafide,
                softmax_spoof,
                threshold,
                prediction,
                latency_ms,
                real_time_factor,
                device,
                model_id,
                checkpoint_sha256,
                condition,
                mitigation_applied,
                raw_json,
            ),
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an analysis_history sqlite Row to API/UI compliant dictionary."""
    d = dict(row)
    bf_logit = d["bonafide_logit"]
    d["id"] = d["id"]
    d["record_id"] = d["id"]
    d["score"] = bf_logit
    d["bonafide_logit"] = bf_logit
    d["mitigation_applied"] = bool(d["mitigation_applied"])
    if "raw_result_json" in d:
        raw_json = d.pop("raw_result_json")
        if raw_json:
            try:
                d["_raw_result"] = json.loads(raw_json)
            except Exception:
                pass
    return d


def load_all_records(
    limit: int = 100,
    offset: int = 0,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve history records sorted newest first."""
    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return []

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


def get_record(record_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve a single analysis record by ID."""
    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return None

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history WHERE id = ?;", (record_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)


def delete_record(record_id: str, db_path: Path | str | None = None) -> bool:
    """Delete a single analysis record and any associated share tokens by ID."""
    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return False

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE id = ?;", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_records(db_path: Path | str | None = None) -> int:
    """Clear all records from analysis_history and cascading share tokens."""
    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return 0

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history;")
        conn.commit()
        return cursor.rowcount


# =====================================================================
# Cryptographic Shareable Report Access (Controlled Bearer Link Model)
# =====================================================================


def create_share_token(
    record_id: str,
    expires_in_days: int = 7,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a high-entropy unguessable bearer token for controlled public report sharing.

    Security model:
    - Token is generated with 32 bytes of cryptographic randomness (secrets.token_urlsafe).
    - Links grant read-only access to a sanitized forensic summary.
    - Raw audio is NEVER exposed or served via share links.
    - Expiration and instant revocation are supported.
    - Unknown or revoked tokens return HTTP 404 to prevent enumeration.
    """
    target_path = get_db_path(db_path)
    rec = get_record(record_id, target_path)
    if rec is None:
        raise KeyError(f"Analysis record '{record_id}' not found")

    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    expires_at = None
    if expires_in_days > 0:
        expires_at = (now + timedelta(days=expires_in_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    with _get_connection(target_path) as conn:
        conn.execute(
            """
            INSERT INTO shared_reports (
                share_token, record_id, created_at, expires_at, revoked, access_count
            ) VALUES (?, ?, ?, ?, 0, 0);
            """,
            (token, record_id, created_at, expires_at),
        )
        conn.commit()

    return {
        "status": "created",
        "share_token": token,
        "share_url": f"/shared/{token}",
        "record_id": record_id,
        "created_at": created_at,
        "expires_at": expires_at,
        "access_scope": "sanitized_forensic_summary_only",
        "note": "Secret bearer link. No raw audio is exposed. Can be revoked at any time.",
    }


def get_shared_report(token: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Retrieve sanitized forensic report by unguessable share token.

    Returns None if token is invalid, expired, or revoked (preventing enumeration).
    """
    if not token or len(token) < 16:
        return None

    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return None

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.share_token, s.record_id, s.created_at as shared_created_at,
                   s.expires_at, s.revoked, s.access_count,
                   h.*
            FROM shared_reports s
            JOIN analysis_history h ON s.record_id = h.id
            WHERE s.share_token = ?;
            """,
            (token,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        # Check revocation
        if row["revoked"] == 1:
            return None

        # Check expiration
        expires_at_str = row["expires_at"]
        if expires_at_str:
            try:
                exp_dt = datetime.strptime(expires_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                if datetime.now(timezone.utc) > exp_dt:
                    return None
            except Exception:
                pass

        # Increment access counter
        conn.execute(
            "UPDATE shared_reports SET access_count = access_count + 1 WHERE share_token = ?;",
            (token,),
        )
        conn.commit()

        # Build sanitized public forensic dossier
        bf_logit = float(row["bonafide_logit"])
        thresh = float(row["threshold"])
        pred = str(row["prediction"])
        sha256 = str(row["sha256"])
        short_sha = sha256[:16] + "..." if len(sha256) > 16 else sha256
        dur = float(row["duration_s"] or 0.0)
        margin = round(bf_logit - thresh, 4)

        return {
            "status": "ok",
            "share_token": token,
            "record_id": str(row["id"]),
            "shared_at": row["shared_created_at"],
            "expires_at": row["expires_at"],
            "access_count": int(row["access_count"]) + 1,
            "forensic_summary": {
                "filename": str(row["filename"]),
                "input_sha256": sha256,
                "duration_seconds": dur,
                "final_prediction": pred,
                "primary_countermeasure_score": bf_logit,
                "decision_threshold": thresh,
                "decision_margin": margin,
                "verdict_rule": f"S_cm ({bf_logit:+.4f}) >= theta ({thresh:.2f}) -> {pred}",
            },
            "audio_metadata": {
                "filename": str(row["filename"]),
                "sha256": sha256,
                "duration_s": dur,
                "sample_rate_hz": int(row["sample_rate"] or 16000),
                "channels": int(row["channels"] or 1),
            },
            "model_provenance": {
                "neural_architecture": "AASIST-L (Jung et al., ICASSP 2022)",
                "trainable_parameters": 85306,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "score_convention": "Bonafide logit >= threshold indicates authentic human speech.",
            },
            "decision_metrics": {
                "countermeasure_score_s_cm": bf_logit,
                "spoof_logit": float(row["spoof_logit"]),
                "decision_margin": margin,
                "softmax_bonafide": float(row["softmax_bonafide"]),
                "verdict": pred,
            },
            "evidence_provenance_timeline": [
                {
                    "stage": 1,
                    "name": "Audio Acquisition & Ingestion",
                    "status": "COMPLETED",
                    "detail": f"Ingested payload '{row['filename']}' ({dur}s)",
                },
                {
                    "stage": 2,
                    "name": "Format Decoding & Integrity Check",
                    "status": "COMPLETED",
                    "detail": f"Verified numerical integrity; SHA-256 digest: {short_sha}",
                },
                {
                    "stage": 3,
                    "name": "SincNet Raw Feature Front-end",
                    "status": "COMPLETED",
                    "detail": "Standardized to 16 kHz mono; 70 parameterized bandpass sinc filters",
                },
                {
                    "stage": 4,
                    "name": "Graph Attention Spectral Graph",
                    "status": "COMPLETED",
                    "detail": (
                        "Constructed spectral graph capturing inter-subband frequency correlations"
                    ),
                },
                {
                    "stage": 5,
                    "name": "Graph Attention Temporal Graph",
                    "status": "COMPLETED",
                    "detail": "Constructed temporal graph capturing inter-frame time dependencies",
                },
                {
                    "stage": 6,
                    "name": "Readout & Max-Margin Logit Scoring",
                    "status": "COMPLETED",
                    "detail": (
                        f"Extracted logits: S_cm={bf_logit:+.4f}, "
                        f"spoof={float(row['spoof_logit']):+.4f}"
                    ),
                },
                {
                    "stage": 7,
                    "name": "Canonical Decision Boundary Mapping",
                    "status": "COMPLETED",
                    "detail": f"Evaluated against canonical threshold theta={thresh:.2f} -> {pred}",
                },
            ],
            "security_and_privacy_notice": {
                "access_model": (
                    "Controlled Secret Bearer Link. Anyone in possession of this unguessable "
                    "token may view this sanitized report summary. No authentication is required "
                    "for viewing, and no private server paths or raw audio files are accessible."
                ),
                "raw_audio_policy": (
                    "Per ADR-002 research governance, raw audio waveforms are processed in-memory "
                    "and are NOT permanently hosted or exposed via shareable links."
                ),
                "revocation": (
                    "The owner of the analysis workspace may revoke this link at any time."
                ),
            },
        }


def revoke_share_token(token: str, db_path: Path | str | None = None) -> bool:
    """Revoke a share link so it can no longer be accessed."""
    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return False

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE shared_reports SET revoked = 1 WHERE share_token = ?;", (token,))
        conn.commit()
        return cursor.rowcount > 0


def get_storage_stats(db_path: Path | str | None = None) -> dict[str, Any]:
    """Return storage metrics for system diagnostics."""
    target_path = get_db_path(db_path)
    if not target_path.is_file():
        return {
            "backend": "sqlite",
            "durable": True,
            "db_path": str(target_path),
            "exists": False,
            "total_records": 0,
            "shared_links_count": 0,
            "db_size_bytes": 0,
        }

    with _get_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_history;")
        rec_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM shared_reports WHERE revoked = 0;")
        share_count = cursor.fetchone()[0]

    return {
        "backend": "sqlite",
        "durable": True,
        "db_path": str(target_path),
        "exists": True,
        "total_records": rec_count,
        "shared_links_count": share_count,
        "db_size_bytes": target_path.stat().st_size,
    }
