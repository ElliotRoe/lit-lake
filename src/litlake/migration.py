from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
from pathlib import Path

from litlake.config import load_settings
from litlake.db import connect_db, init_db, queue_depth_by_state, seed_pending_jobs

LEGACY_EMPTY_CHUNK_REPAIR_ERROR = "legacy repair: empty fulltext chunk"


@dataclasses.dataclass(frozen=True)
class TextDigest:
    rows: int
    total_chars: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class MigrationReport:
    db_path: str
    schema_initialized: bool
    renamed_pdf_chunks: int
    storage_kind_backfilled: int
    storage_uri_backfilled: int
    pending_empty_fulltext_chunks_repaired: int
    embedding_status_requeued: int
    extraction_status_requeued: int
    pre_fulltext_union: TextDigest
    post_fulltext: TextDigest
    pre_extracted_text: TextDigest
    post_extracted_text: TextDigest
    queue_seed_extraction_jobs: int
    queue_seed_embedding_jobs: int
    queue_depth: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _digest_documents(conn: sqlite3.Connection, where_clause: str, params: tuple[object, ...]) -> TextDigest:
    hasher = hashlib.sha256()
    rows = 0
    total_chars = 0
    cur = conn.execute(
        f"""
        SELECT id, content
        FROM documents
        WHERE {where_clause}
        ORDER BY id
        """
        ,
        params,
    )
    for doc_id, content in cur:
        doc_text = str(content or "")
        rows += 1
        total_chars += len(doc_text)
        hasher.update(str(doc_id).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(doc_text.encode("utf-8", "surrogatepass"))
        hasher.update(b"\x1e")
    return TextDigest(rows=rows, total_chars=total_chars, sha256=hasher.hexdigest())


def _digest_extracted_text(conn: sqlite3.Connection) -> TextDigest:
    hasher = hashlib.sha256()
    rows = 0
    total_chars = 0
    cur = conn.execute(
        """
        SELECT id, extracted_text
        FROM document_files
        WHERE extracted_text IS NOT NULL
        ORDER BY id
        """
    )
    for file_id, extracted_text in cur:
        text = str(extracted_text or "")
        rows += 1
        total_chars += len(text)
        hasher.update(str(file_id).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(text.encode("utf-8", "surrogatepass"))
        hasher.update(b"\x1e")
    return TextDigest(rows=rows, total_chars=total_chars, sha256=hasher.hexdigest())


def _run_data_migration(conn: sqlite3.Connection) -> tuple[int, int, int, int, int, int]:
    conn.execute("BEGIN")
    try:
        renamed_pdf_chunks = conn.execute(
            """
            UPDATE documents
            SET kind = 'fulltext_chunk'
            WHERE kind = 'pdf_chunk'
            """
        ).rowcount

        columns = _table_columns(conn, "document_files")

        storage_kind_backfilled = 0
        storage_uri_backfilled = 0
        if "storage_kind" in columns:
            storage_kind_backfilled = conn.execute(
                """
                UPDATE document_files
                SET storage_kind = 'local'
                WHERE storage_kind IS NULL OR TRIM(storage_kind) = ''
                """
            ).rowcount
        if "storage_uri" in columns:
            storage_uri_backfilled = conn.execute(
                """
                UPDATE document_files
                SET storage_uri = file_path
                WHERE storage_uri IS NULL OR TRIM(storage_uri) = ''
                """
            ).rowcount

        pending_empty_fulltext_chunks_repaired = conn.execute(
            """
            UPDATE documents
            SET embedding_status = 'error',
                embedding_error = ?,
                embedding_updated_at = NULL
            WHERE kind = 'fulltext_chunk'
              AND embedding_status = 'pending'
              AND (content IS NULL OR TRIM(content) = '')
            """,
            (LEGACY_EMPTY_CHUNK_REPAIR_ERROR,),
        ).rowcount

        # Recover rows that can get stuck in in-flight states after abrupt shutdown.
        embedding_status_requeued = conn.execute(
            """
            UPDATE documents
            SET embedding_status = 'pending'
            WHERE embedding_status = 'embedding'
            """
        ).rowcount
        extraction_status_requeued = conn.execute(
            """
            UPDATE document_files
            SET extraction_status = 'pending'
            WHERE extraction_status = 'extracting'
            """
        ).rowcount
        conn.commit()
        return (
            renamed_pdf_chunks,
            storage_kind_backfilled,
            storage_uri_backfilled,
            pending_empty_fulltext_chunks_repaired,
            embedding_status_requeued,
            extraction_status_requeued,
        )
    except Exception:
        conn.rollback()
        raise


def detect_migration_reasons(conn: sqlite3.Connection) -> list[str]:
    reasons: list[str] = []
    if not _table_exists(conn, "documents") or not _table_exists(conn, "document_files"):
        return reasons

    pdf_chunk_rows = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE kind = 'pdf_chunk'"
    ).fetchone()
    if int(pdf_chunk_rows[0]) > 0:
        reasons.append("documents.kind contains legacy 'pdf_chunk' rows")

    embedding_inflight = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE embedding_status = 'embedding'"
    ).fetchone()
    if int(embedding_inflight[0]) > 0:
        reasons.append("documents.embedding_status contains in-flight 'embedding' rows")

    extraction_inflight = conn.execute(
        "SELECT COUNT(*) FROM document_files WHERE extraction_status = 'extracting'"
    ).fetchone()
    if int(extraction_inflight[0]) > 0:
        reasons.append("document_files.extraction_status contains in-flight 'extracting' rows")

    columns = _table_columns(conn, "document_files")
    if "storage_uri" in columns:
        storage_uri_missing = conn.execute(
            """
            SELECT COUNT(*)
            FROM document_files
            WHERE storage_uri IS NULL OR TRIM(storage_uri) = ''
            """
        ).fetchone()
        if int(storage_uri_missing[0]) > 0:
            reasons.append("document_files.storage_uri has missing values")

    if "storage_kind" in columns:
        storage_kind_missing = conn.execute(
            """
            SELECT COUNT(*)
            FROM document_files
            WHERE storage_kind IS NULL OR TRIM(storage_kind) = ''
            """
        ).fetchone()
        if int(storage_kind_missing[0]) > 0:
            reasons.append("document_files.storage_kind has missing values")

    pending_empty_fulltext = conn.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE kind = 'fulltext_chunk'
          AND embedding_status = 'pending'
          AND (content IS NULL OR TRIM(content) = '')
        """
    ).fetchone()
    if int(pending_empty_fulltext[0]) > 0:
        reasons.append("documents.fulltext_chunk has pending rows with empty content")

    return reasons


def run_migration(
    conn: sqlite3.Connection,
    *,
    queue_max_attempts: int,
    seed_jobs: bool = True,
) -> MigrationReport:
    pre_fulltext_union = _digest_documents(
        conn,
        "kind IN ('pdf_chunk', 'fulltext_chunk')",
        (),
    )
    pre_extracted_text = _digest_extracted_text(conn)

    init_db(conn)
    (
        renamed_pdf_chunks,
        storage_kind_backfilled,
        storage_uri_backfilled,
        pending_empty_fulltext_chunks_repaired,
        embedding_status_requeued,
        extraction_status_requeued,
    ) = _run_data_migration(conn)

    post_pdf_chunk = _digest_documents(conn, "kind = 'pdf_chunk'", ())
    post_fulltext = _digest_documents(conn, "kind = 'fulltext_chunk'", ())
    post_extracted_text = _digest_extracted_text(conn)

    if post_pdf_chunk.rows != 0:
        raise RuntimeError("Migration invariant failed: documents.kind='pdf_chunk' rows remain")
    if (
        post_fulltext.rows != pre_fulltext_union.rows
        or post_fulltext.total_chars != pre_fulltext_union.total_chars
        or post_fulltext.sha256 != pre_fulltext_union.sha256
    ):
        raise RuntimeError(
            "Migration invariant failed: fulltext chunk text changed during conversion"
        )
    if (
        post_extracted_text.rows != pre_extracted_text.rows
        or post_extracted_text.total_chars != pre_extracted_text.total_chars
        or post_extracted_text.sha256 != pre_extracted_text.sha256
    ):
        raise RuntimeError(
            "Migration invariant failed: document_files.extracted_text changed during migration"
        )

    if seed_jobs:
        seed_stats = seed_pending_jobs(conn, queue_max_attempts=queue_max_attempts)
        extraction_jobs = seed_stats.extraction_jobs_created
        embedding_jobs = seed_stats.embedding_jobs_created
    else:
        extraction_jobs = 0
        embedding_jobs = 0
    queue_depth = queue_depth_by_state(conn)

    db_path_row = conn.execute("PRAGMA database_list").fetchall()
    db_path = str(db_path_row[0][2]) if db_path_row else "<unknown>"

    return MigrationReport(
        db_path=db_path,
        schema_initialized=True,
        renamed_pdf_chunks=renamed_pdf_chunks,
        storage_kind_backfilled=storage_kind_backfilled,
        storage_uri_backfilled=storage_uri_backfilled,
        pending_empty_fulltext_chunks_repaired=pending_empty_fulltext_chunks_repaired,
        embedding_status_requeued=embedding_status_requeued,
        extraction_status_requeued=extraction_status_requeued,
        pre_fulltext_union=pre_fulltext_union,
        post_fulltext=post_fulltext,
        pre_extracted_text=pre_extracted_text,
        post_extracted_text=post_extracted_text,
        queue_seed_extraction_jobs=extraction_jobs,
        queue_seed_embedding_jobs=embedding_jobs,
        queue_depth=queue_depth,
    )


def auto_migrate_if_needed(
    conn: sqlite3.Connection,
    *,
    queue_max_attempts: int,
    seed_jobs: bool = False,
) -> tuple[list[str], MigrationReport | None]:
    reasons = detect_migration_reasons(conn)
    if not reasons:
        return reasons, None
    report = run_migration(
        conn,
        queue_max_attempts=queue_max_attempts,
        seed_jobs=seed_jobs,
    )
    return reasons, report


def migrate_database(
    path: Path | str,
    *,
    queue_max_attempts: int = 5,
    seed_jobs: bool = True,
) -> MigrationReport:
    conn = connect_db(path)
    try:
        return run_migration(
            conn,
            queue_max_attempts=queue_max_attempts,
            seed_jobs=seed_jobs,
        )
    finally:
        conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a Lit Lake v0.3.x SQLite database to the python-rewrite schema "
            "while preserving fulltext artifacts."
        )
    )
    parser.add_argument(
        "--db-path",
        help="Path to lit_lake.db. Defaults to configured Lit Lake db path.",
    )
    parser.add_argument(
        "--queue-max-attempts",
        type=int,
        default=5,
        help="Max attempts for newly seeded queue jobs (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the migration report as JSON only.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.queue_max_attempts < 1:
        raise SystemExit("--queue-max-attempts must be >= 1")

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = load_settings().paths.db_path

    report = migrate_database(
        db_path,
        queue_max_attempts=args.queue_max_attempts,
        seed_jobs=True,
    )
    payload = report.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Migrated DB: {payload['db_path']}")
    print("Migration summary:")
    print(f"- Renamed chunks (pdf_chunk -> fulltext_chunk): {payload['renamed_pdf_chunks']}")
    print(f"- Backfilled document_files.storage_kind: {payload['storage_kind_backfilled']}")
    print(f"- Backfilled document_files.storage_uri: {payload['storage_uri_backfilled']}")
    print(
        "- Repaired pending fulltext chunks with empty content: "
        f"{payload['pending_empty_fulltext_chunks_repaired']}"
    )
    print(f"- Requeued documents from embedding -> pending: {payload['embedding_status_requeued']}")
    print(f"- Requeued files from extracting -> pending: {payload['extraction_status_requeued']}")
    print("")
    print("Text integrity checks:")
    print(
        "- fulltext union before -> after: "
        f"{payload['pre_fulltext_union']['rows']} rows / "
        f"{payload['pre_fulltext_union']['total_chars']} chars / "
        f"{payload['pre_fulltext_union']['sha256']}"
    )
    print(
        f"  => {payload['post_fulltext']['rows']} rows / "
        f"{payload['post_fulltext']['total_chars']} chars / "
        f"{payload['post_fulltext']['sha256']}"
    )
    print(
        "- extracted_text before -> after: "
        f"{payload['pre_extracted_text']['rows']} rows / "
        f"{payload['pre_extracted_text']['total_chars']} chars / "
        f"{payload['pre_extracted_text']['sha256']}"
    )
    print(
        f"  => {payload['post_extracted_text']['rows']} rows / "
        f"{payload['post_extracted_text']['total_chars']} chars / "
        f"{payload['post_extracted_text']['sha256']}"
    )
    print("")
    print("Queue seeding:")
    print(f"- extraction jobs created: {payload['queue_seed_extraction_jobs']}")
    print(f"- embedding jobs created: {payload['queue_seed_embedding_jobs']}")
    print("- queue depth by state:")
    print(json.dumps(payload["queue_depth"], indent=2))


if __name__ == "__main__":
    main()
