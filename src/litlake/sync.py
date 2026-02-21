from __future__ import annotations

import sqlite3

from litlake.db import enqueue_job
from litlake.zotero import ZoteroReader


def sync_zotero(
    conn: sqlite3.Connection,
    *,
    queue_max_attempts: int,
    explicit_db_path: str | None = None,
) -> str:
    reader = ZoteroReader(explicit_db_path)
    items = reader.get_items()

    existing_map_rows = conn.execute(
        "SELECT value, reference_id FROM reference_external_ids WHERE scheme = 'zotero_key'"
    ).fetchall()
    existing_map = {row[0]: int(row[1]) for row in existing_map_rows}

    refs_added = 0
    refs_updated = 0
    docs_upserted = 0
    files_upserted = 0

    conn.execute("BEGIN")
    for item in items:
        if item.key in existing_map:
            reference_id = existing_map[item.key]
            conn.execute(
                """
                UPDATE reference_items
                SET title = ?, authors = ?, year = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (item.title, item.authors, item.date, reference_id),
            )
            refs_updated += 1
        else:
            cur = conn.execute(
                """
                INSERT INTO reference_items (title, authors, year, source_system, source_id)
                VALUES (?, ?, ?, 'zotero', ?)
                """,
                (item.title, item.authors, item.date, item.key),
            )
            reference_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO reference_external_ids (reference_id, scheme, value)
                VALUES (?, 'zotero_key', ?)
                """,
                (reference_id, item.key),
            )
            existing_map[item.key] = reference_id
            refs_added += 1

        title_content = item.title.strip() if item.title and item.title.strip() else None
        title_status = "pending" if title_content else "skipped"
        existing = conn.execute(
            """
            SELECT id, content
            FROM documents
            WHERE reference_id = ?
              AND kind = 'title'
              AND source_system = 'zotero'
              AND source_id = ?
            """,
            (reference_id, item.key),
        ).fetchone()
        if existing:
            old_content = (existing[1] or "").strip() if existing[1] else None
            changed = old_content != title_content
            if changed:
                conn.execute(
                    """
                    UPDATE documents
                    SET content = ?,
                        embedding_status = ?,
                        embedding_error = NULL,
                        embedding_updated_at = NULL,
                        embedding_backend = NULL,
                        embedding_model = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (title_content, title_status, int(existing[0])),
                )
        else:
            conn.execute(
                """
                INSERT INTO documents (
                    reference_id, kind, content, embedding_status, source_system, source_id
                ) VALUES (?, 'title', ?, ?, 'zotero', ?)
                """,
                (reference_id, title_content, title_status, item.key),
            )
        docs_upserted += 1

        abs_content = item.abstract_note.strip() if item.abstract_note and item.abstract_note.strip() else None
        abs_status = "pending" if abs_content else "skipped"
        existing = conn.execute(
            """
            SELECT id, content
            FROM documents
            WHERE reference_id = ?
              AND kind = 'abstract'
              AND source_system = 'zotero'
              AND source_id = ?
            """,
            (reference_id, item.key),
        ).fetchone()
        if existing:
            old_content = (existing[1] or "").strip() if existing[1] else None
            changed = old_content != abs_content
            if changed:
                conn.execute(
                    """
                    UPDATE documents
                    SET content = ?,
                        embedding_status = ?,
                        embedding_error = NULL,
                        embedding_updated_at = NULL,
                        embedding_backend = NULL,
                        embedding_model = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (abs_content, abs_status, int(existing[0])),
                )
        else:
            conn.execute(
                """
                INSERT INTO documents (
                    reference_id, kind, content, embedding_status, source_system, source_id
                ) VALUES (?, 'abstract', ?, ?, 'zotero', ?)
                """,
                (reference_id, abs_content, abs_status, item.key),
            )
        docs_upserted += 1

        if item.pdf_path:
            conn.execute(
                """
                INSERT INTO document_files (
                    reference_id, file_path, mime_type, label, source_system, source_id,
                    extraction_status, storage_kind, storage_uri
                )
                VALUES (?, ?, 'application/pdf', 'main_pdf', 'zotero', ?, 'pending', 'local', ?)
                ON CONFLICT(reference_id, file_path)
                DO UPDATE SET
                    mime_type = excluded.mime_type,
                    label = excluded.label,
                    source_system = excluded.source_system,
                    source_id = excluded.source_id,
                    storage_kind = excluded.storage_kind,
                    storage_uri = excluded.storage_uri,
                    extraction_status = CASE
                        WHEN document_files.extracted_text IS NULL THEN 'pending'
                        ELSE document_files.extraction_status
                    END
                """,
                (reference_id, item.pdf_path, item.key, item.pdf_path),
            )
            files_upserted += 1

    conn.commit()

    # Enqueue work after sync.
    emb_rows = conn.execute(
        """
        SELECT id
        FROM documents
        WHERE kind IN ('title','abstract')
          AND content IS NOT NULL
          AND TRIM(content) <> ''
          AND embedding_status IN ('pending','error')
        """
    ).fetchall()
    for row in emb_rows:
        doc_id = int(row[0])
        enqueue_job(
            conn,
            queue_name="embedding",
            job_type="embed_document",
            entity_type="document",
            entity_id=doc_id,
            dedupe_key=f"embed_document:{doc_id}",
            payload={"document_id": doc_id},
            max_attempts=queue_max_attempts,
        )

    ext_rows = conn.execute(
        """
        SELECT id
        FROM document_files
        WHERE mime_type = 'application/pdf'
          AND extraction_status IN ('pending','error')
        """
    ).fetchall()
    for row in ext_rows:
        file_id = int(row[0])
        enqueue_job(
            conn,
            queue_name="extraction",
            job_type="extract_pdf",
            entity_type="document_file",
            entity_id=file_id,
            dedupe_key=f"extract_pdf:{file_id}",
            payload={"document_file_id": file_id},
            max_attempts=queue_max_attempts,
        )

    conn.commit()

    return (
        "Zotero sync complete. "
        f"References added: {refs_added}, updated: {refs_updated}. "
        f"Documents upserted: {docs_upserted}. "
        f"Document files upserted: {files_upserted}."
    )


__all__ = ["sync_zotero"]
