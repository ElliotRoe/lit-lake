from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ISO_NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"


@dataclass
class QueueSeedStats:
    extraction_jobs_created: int = 0
    embedding_jobs_created: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect_db(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    _load_sqlite_vec(conn)
    return conn


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except Exception:
        # Keep non-vector paths operational in environments without sqlite-vec.
        return

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_column(conn: sqlite3.Connection, table: str, column_name: str, column_def: str) -> None:
    cols = _table_columns(conn, table)
    if column_name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")


def _ensure_index(conn: sqlite3.Connection, sql: str) -> None:
    conn.execute(sql)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reference_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_key TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            year TEXT,
            source_system TEXT NOT NULL,
            source_id TEXT NOT NULL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_system, source_id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reference_external_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id INTEGER NOT NULL,
            scheme TEXT NOT NULL,
            value TEXT NOT NULL,
            UNIQUE(reference_id, scheme, value),
            FOREIGN KEY(reference_id) REFERENCES reference_items(id) ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            mime_type TEXT,
            label TEXT,
            extracted_text TEXT,
            extraction_status TEXT DEFAULT 'pending',
            extraction_error TEXT,
            source_system TEXT,
            source_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(reference_id, file_path),
            FOREIGN KEY(reference_id) REFERENCES reference_items(id) ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id INTEGER NOT NULL,
            document_file_id INTEGER,
            kind TEXT NOT NULL,
            content TEXT,
            chunk_index INTEGER,
            embedding_status TEXT DEFAULT 'pending',
            embedding_updated_at DATETIME,
            embedding_error TEXT,
            source_system TEXT,
            source_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(reference_id) REFERENCES reference_items(id) ON DELETE CASCADE,
            FOREIGN KEY(document_file_id) REFERENCES document_files(id) ON DELETE CASCADE
        )
        """
    )

    # Add forward-compatible provenance and storage fields.
    _ensure_column(conn, "documents", "embedding_backend", "embedding_backend TEXT")
    _ensure_column(conn, "documents", "embedding_model", "embedding_model TEXT")

    _ensure_column(conn, "document_files", "extraction_backend", "extraction_backend TEXT")
    _ensure_column(
        conn,
        "document_files",
        "extraction_backend_version",
        "extraction_backend_version TEXT",
    )
    _ensure_column(conn, "document_files", "storage_kind", "storage_kind TEXT DEFAULT 'local'")
    _ensure_column(conn, "document_files", "storage_uri", "storage_uri TEXT")

    _ensure_index(conn, "CREATE INDEX IF NOT EXISTS idx_documents_reference_id ON documents(reference_id)")
    _ensure_index(conn, "CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind)")
    _ensure_index(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_documents_embedding_status ON documents(embedding_status)",
    )
    _ensure_index(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_reference_external_ids_scheme_value ON reference_external_ids(scheme, value)",
    )
    _ensure_index(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_document_files_extraction_status ON document_files(extraction_status)",
    )
    _ensure_index(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_documents_document_file_id ON documents(document_file_id)",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            queue_name TEXT NOT NULL,
            job_type TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            dedupe_key TEXT NOT NULL,
            payload_json TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 100,
            available_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            claimed_by TEXT,
            claim_token TEXT,
            claim_expires_at DATETIME,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 5,
            last_error TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dedupe_key)
        )
        """
    )
    _ensure_index(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_poll
        ON jobs(queue_name, status, available_at, priority, created_at)
        """,
    )
    _ensure_index(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_claim_expiry
        ON jobs(queue_name, claim_expires_at)
        """,
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_attempts (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            attempt_no INTEGER NOT NULL,
            worker_id TEXT NOT NULL,
            backend_name TEXT,
            backend_version TEXT,
            started_at DATETIME NOT NULL,
            finished_at DATETIME,
            outcome TEXT,
            error_class TEXT,
            error_message TEXT,
            metrics_json TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """
    )
    _ensure_index(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_job_attempts_job ON job_attempts(job_id, attempt_no)",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_runs (
            worker_id TEXT PRIMARY KEY,
            worker_type TEXT NOT NULL,
            backend_name TEXT NOT NULL,
            backend_version TEXT,
            config_hash TEXT,
            started_at DATETIME NOT NULL,
            last_heartbeat_at DATETIME NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    # sqlite-vec virtual table for embeddings (must remain contract-compatible).
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(embedding float[384]);"
        )
    except sqlite3.OperationalError as exc:
        # If sqlite-vec isn't available in the current runtime, keep a fallback
        # table so non-vector tests and tooling can still execute.
        if "no such module: vec0" in str(exc):
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vec_documents (
                    rowid INTEGER PRIMARY KEY,
                    embedding BLOB
                )
                """
            )
        else:
            raise

    conn.commit()


def enqueue_job(
    conn: sqlite3.Connection,
    *,
    queue_name: str,
    job_type: str,
    entity_type: str,
    entity_id: int,
    dedupe_key: str,
    payload: dict | None = None,
    priority: int = 100,
    max_attempts: int = 5,
) -> bool:
    payload_json = json.dumps(payload or {}, separators=(",", ":"))
    cur = conn.execute(
        """
        INSERT INTO jobs (
            id, queue_name, job_type, entity_type, entity_id, dedupe_key,
            payload_json, status, priority, max_attempts, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(dedupe_key) DO NOTHING
        """,
        (
            str(uuid.uuid4()),
            queue_name,
            job_type,
            entity_type,
            entity_id,
            dedupe_key,
            payload_json,
            priority,
            max_attempts,
        ),
    )
    return cur.rowcount > 0


def seed_pending_jobs(conn: sqlite3.Connection, *, queue_max_attempts: int) -> QueueSeedStats:
    stats = QueueSeedStats()

    extraction_candidates = conn.execute(
        """
        SELECT id
        FROM document_files
        WHERE mime_type = 'application/pdf'
          AND (
                extraction_status IN ('pending', 'error')
                OR extracted_text IS NULL
              )
        """
    ).fetchall()
    for row in extraction_candidates:
        file_id = int(row[0])
        created = enqueue_job(
            conn,
            queue_name="extraction",
            job_type="extract_pdf",
            entity_type="document_file",
            entity_id=file_id,
            dedupe_key=f"extract_pdf:{file_id}",
            payload={"document_file_id": file_id},
            max_attempts=queue_max_attempts,
        )
        if created:
            stats.extraction_jobs_created += 1

    embedding_candidates = conn.execute(
        """
        SELECT id
        FROM documents
        WHERE kind IN ('title','abstract','pdf_chunk')
          AND content IS NOT NULL
          AND TRIM(content) <> ''
          AND embedding_status IN ('pending','error')
        """
    ).fetchall()
    for row in embedding_candidates:
        doc_id = int(row[0])
        created = enqueue_job(
            conn,
            queue_name="embedding",
            job_type="embed_document",
            entity_type="document",
            entity_id=doc_id,
            dedupe_key=f"embed_document:{doc_id}",
            payload={"document_id": doc_id},
            max_attempts=queue_max_attempts,
        )
        if created:
            stats.embedding_jobs_created += 1

    conn.commit()
    return stats


def queue_depth_by_state(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    rows = conn.execute(
        """
        SELECT queue_name, status, COUNT(*)
        FROM jobs
        GROUP BY queue_name, status
        """
    ).fetchall()
    for queue_name, status, count in rows:
        out.setdefault(queue_name, {})[status] = int(count)
    return out


def recent_dead_letter(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, queue_name, job_type, entity_type, entity_id, attempts, last_error, updated_at
        FROM jobs
        WHERE status = 'dead'
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def recent_attempt_errors(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT job_id, worker_id, backend_name, outcome, error_class, error_message, finished_at
        FROM job_attempts
        WHERE outcome IN ('retryable_fail','permanent_fail')
        ORDER BY finished_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


__all__ = [
    "QueueSeedStats",
    "seed_pending_jobs",
    "connect_db",
    "init_db",
    "enqueue_job",
    "queue_depth_by_state",
    "recent_dead_letter",
    "recent_attempt_errors",
    "utc_now_iso",
]
