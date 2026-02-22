from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from litlake.db import enqueue_job
from litlake.providers.extraction import SUPPORTED_EXTRACTION_MIME_TYPES, extract_html_text
from litlake.zotero import ZoteroAnnotation, ZoteroItem, ZoteroReader


def _to_json(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _normalize_content(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _note_html_to_text(note_html: str | None) -> str | None:
    if not note_html:
        return None
    extracted = extract_html_text(note_html, require_trafilatura=False)
    return _normalize_content(extracted.text)


@dataclass
class _DeltaCounts:
    added: int = 0
    updated: int = 0
    unchanged: int = 0

    def record(self, outcome: str) -> None:
        if outcome == "added":
            self.added += 1
            return
        if outcome == "updated":
            self.updated += 1
            return
        if outcome == "unchanged":
            self.unchanged += 1
            return
        raise ValueError(f"Unsupported delta outcome: {outcome}")

    @property
    def changed(self) -> int:
        return self.added + self.updated


def _build_annotation_content(annotation: ZoteroAnnotation) -> str | None:
    text = _normalize_content(annotation.text)
    comment = _normalize_content(annotation.comment)
    if not text and not comment:
        return None

    parts: list[str] = []
    if text:
        quoted = "\n".join(f"> {line}" for line in text.splitlines())
        parts.append(f"Highlighted text:\n{quoted}")
    if comment:
        parts.append(f"Annotation comment:\n{comment}")
    return "\n\n".join(parts)


def _upsert_document(
    conn: sqlite3.Connection,
    *,
    reference_id: int,
    kind: str,
    source_id: str,
    content: str | None,
    document_file_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    normalized_content = _normalize_content(content)
    embed_status = "pending" if normalized_content else "skipped"
    metadata_json = _to_json(metadata)

    existing = conn.execute(
        """
        SELECT id, content, document_file_id, metadata_json
        FROM documents
        WHERE reference_id = ?
          AND kind = ?
          AND source_system = 'zotero'
          AND source_id = ?
        """,
        (reference_id, kind, source_id),
    ).fetchone()

    if existing:
        old_content = _normalize_content(existing[1])
        old_document_file_id = int(existing[2]) if existing[2] is not None else None
        old_metadata = existing[3]
        changed = (
            old_content != normalized_content
            or old_document_file_id != document_file_id
            or old_metadata != metadata_json
        )
        if changed:
            conn.execute(
                """
                UPDATE documents
                SET document_file_id = ?,
                    content = ?,
                    metadata_json = ?,
                    embedding_status = ?,
                    embedding_error = NULL,
                    embedding_updated_at = NULL,
                    embedding_backend = NULL,
                    embedding_model = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (document_file_id, normalized_content, metadata_json, embed_status, int(existing[0])),
            )
            return "updated"
        return "unchanged"

    conn.execute(
        """
        INSERT INTO documents (
            reference_id, document_file_id, kind, content, metadata_json,
            embedding_status, source_system, source_id
        ) VALUES (?, ?, ?, ?, ?, ?, 'zotero', ?)
        """,
        (reference_id, document_file_id, kind, normalized_content, metadata_json, embed_status, source_id),
    )
    return "added"


def _upsert_attachments(
    conn: sqlite3.Connection,
    *,
    reference_id: int,
    item: ZoteroItem,
) -> _DeltaCounts:
    counts = _DeltaCounts()
    for attachment in item.attachments:
        metadata_json = _to_json(
            {
                "schema_version": 1,
                "zotero_attachment_key": attachment.key,
                "zotero_attachment_item_id": attachment.item_id,
                "link_mode": attachment.link_mode,
            }
        )
        existing = conn.execute(
            """
            SELECT id, mime_type, source_system, source_id, storage_kind, storage_uri,
                   metadata_json, extraction_status, extracted_text
            FROM document_files
            WHERE reference_id = ?
              AND file_path = ?
            """,
            (reference_id, attachment.resolved_path),
        ).fetchone()

        if not existing:
            conn.execute(
                """
                INSERT INTO document_files (
                    reference_id, file_path, mime_type, source_system, source_id,
                    extraction_status, storage_kind, storage_uri, metadata_json
                )
                VALUES (?, ?, ?, 'zotero', ?, ?, 'local', ?, ?)
                """,
                (
                    reference_id,
                    attachment.resolved_path,
                    attachment.content_type,
                    attachment.key,
                    "pending",
                    attachment.resolved_path,
                    metadata_json,
                ),
            )
            counts.record("added")
            continue

        next_extraction_status = (
            "pending"
            if existing[8] is None
            else (str(existing[7]) if existing[7] is not None else "pending")
        )
        changed = (
            (existing[1] or None) != attachment.content_type
            or (existing[2] or None) != "zotero"
            or (existing[3] or None) != attachment.key
            or (existing[4] or None) != "local"
            or (existing[5] or None) != attachment.resolved_path
            or (existing[6] or None) != metadata_json
            or (existing[7] or None) != next_extraction_status
        )
        if not changed:
            counts.record("unchanged")
            continue

        conn.execute(
            """
            UPDATE document_files
            SET mime_type = ?,
                source_system = 'zotero',
                source_id = ?,
                storage_kind = 'local',
                storage_uri = ?,
                metadata_json = ?,
                extraction_status = ?
            WHERE id = ?
            """,
            (
                attachment.content_type,
                attachment.key,
                attachment.resolved_path,
                metadata_json,
                next_extraction_status,
                int(existing[0]),
            ),
        )
        counts.record("updated")
    return counts


def _document_files_by_source_id(conn: sqlite3.Connection, *, reference_id: int) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT id, source_id
        FROM document_files
        WHERE reference_id = ?
          AND source_system = 'zotero'
          AND source_id IS NOT NULL
        """,
        (reference_id,),
    ).fetchall()
    return {str(row[1]): int(row[0]) for row in rows}


def _upsert_annotation_documents(
    conn: sqlite3.Connection,
    *,
    reference_id: int,
    item: ZoteroItem,
    document_file_ids: dict[str, int],
) -> _DeltaCounts:
    counts = _DeltaCounts()
    for annotation in item.annotations:
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "annotation_type": annotation.annotation_type,
            "author_name": annotation.author_name,
            "color": annotation.color,
            "page_label": annotation.page_label,
            "sort_index": annotation.sort_index,
            "position": annotation.position,
            "is_external": annotation.is_external,
            "parent_attachment_key": annotation.parent_attachment_key,
        }
        document_file_id = (
            document_file_ids.get(annotation.parent_attachment_key)
            if annotation.parent_attachment_key
            else None
        )
        outcome = _upsert_document(
            conn,
            reference_id=reference_id,
            kind="annotation",
            source_id=annotation.key,
            content=_build_annotation_content(annotation),
            document_file_id=document_file_id,
            metadata=metadata,
        )
        counts.record(outcome)
    return counts


def _upsert_note_documents(
    conn: sqlite3.Connection, *, reference_id: int, item: ZoteroItem
) -> _DeltaCounts:
    counts = _DeltaCounts()
    for note in item.notes:
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "title": note.title,
            "parent_item_id": note.parent_item_id,
        }
        outcome = _upsert_document(
            conn,
            reference_id=reference_id,
            kind="note",
            source_id=note.key,
            content=_note_html_to_text(note.note_html),
            metadata=metadata,
        )
        counts.record(outcome)
    return counts


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

    ref_counts = _DeltaCounts()
    doc_counts = _DeltaCounts()
    file_counts = _DeltaCounts()
    annotation_counts = _DeltaCounts()
    note_counts = _DeltaCounts()

    conn.execute("BEGIN")
    for item in items:
        if item.key in existing_map:
            reference_id = existing_map[item.key]
            existing_reference = conn.execute(
                "SELECT title, authors, year FROM reference_items WHERE id = ?",
                (reference_id,),
            ).fetchone()
            if existing_reference is None:
                raise ValueError(f"Missing reference_items row for id={reference_id}")
            changed = (
                (existing_reference[0] or None) != item.title
                or (existing_reference[1] or None) != item.authors
                or (existing_reference[2] or None) != item.date
            )
            if changed:
                conn.execute(
                    """
                    UPDATE reference_items
                    SET title = ?, authors = ?, year = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (item.title, item.authors, item.date, reference_id),
                )
                ref_counts.record("updated")
            else:
                ref_counts.record("unchanged")
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
            ref_counts.record("added")

        doc_counts.record(
            _upsert_document(
                conn,
                reference_id=reference_id,
                kind="title",
                source_id=item.key,
                content=item.title,
            )
        )

        doc_counts.record(
            _upsert_document(
                conn,
                reference_id=reference_id,
                kind="abstract",
                source_id=item.key,
                content=item.abstract_note,
            )
        )

        attachment_deltas = _upsert_attachments(conn, reference_id=reference_id, item=item)
        file_counts.added += attachment_deltas.added
        file_counts.updated += attachment_deltas.updated
        file_counts.unchanged += attachment_deltas.unchanged
        document_files = _document_files_by_source_id(conn, reference_id=reference_id)
        annotation_deltas = _upsert_annotation_documents(
            conn,
            reference_id=reference_id,
            item=item,
            document_file_ids=document_files,
        )
        annotation_counts.added += annotation_deltas.added
        annotation_counts.updated += annotation_deltas.updated
        annotation_counts.unchanged += annotation_deltas.unchanged
        note_deltas = _upsert_note_documents(conn, reference_id=reference_id, item=item)
        note_counts.added += note_deltas.added
        note_counts.updated += note_deltas.updated
        note_counts.unchanged += note_deltas.unchanged

    conn.commit()

    embeddable_kinds = ("title", "abstract", "fulltext_chunk", "annotation", "note")
    emb_kind_placeholders = ",".join("?" for _ in embeddable_kinds)
    emb_rows = conn.execute(
        f"""
        SELECT id
        FROM documents
        WHERE kind IN ({emb_kind_placeholders})
          AND content IS NOT NULL
          AND TRIM(content) <> ''
          AND embedding_status IN ('pending','error')
        """,
        embeddable_kinds,
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

    active_extraction_mimes = tuple(sorted(SUPPORTED_EXTRACTION_MIME_TYPES))
    extraction_mime_placeholders = ",".join("?" for _ in active_extraction_mimes)
    ext_rows = conn.execute(
        f"""
        SELECT id
        FROM document_files
        WHERE mime_type IN ({extraction_mime_placeholders})
          AND extraction_status IN ('pending','error')
        """,
        active_extraction_mimes,
    ).fetchall()
    for row in ext_rows:
        file_id = int(row[0])
        enqueue_job(
            conn,
            queue_name="extraction",
            job_type="extract_file",
            entity_type="document_file",
            entity_id=file_id,
            dedupe_key=f"extract_file:{file_id}",
            payload={"document_file_id": file_id},
            max_attempts=queue_max_attempts,
        )

    conn.commit()

    total_added = (
        ref_counts.added
        + doc_counts.added
        + file_counts.added
        + annotation_counts.added
        + note_counts.added
    )
    total_updated = (
        ref_counts.updated
        + doc_counts.updated
        + file_counts.updated
        + annotation_counts.updated
        + note_counts.updated
    )

    total_unchanged = (
        ref_counts.unchanged
        + doc_counts.unchanged
        + file_counts.unchanged
        + annotation_counts.unchanged
        + note_counts.unchanged
    )
    total_changed = total_added + total_updated

    if total_changed == 0:
        header = "Zotero sync complete. No library changes detected."
    else:
        header = f"Zotero sync complete. Changes detected: {total_changed}."

    lines = [
        header,
        "",
        "| Artifact | Added | Updated | Unchanged | Changed |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| References | {ref_counts.added} | {ref_counts.updated} | {ref_counts.unchanged} | "
            f"{ref_counts.changed} |"
        ),
        (
            f"| Documents | {doc_counts.added} | {doc_counts.updated} | {doc_counts.unchanged} | "
            f"{doc_counts.changed} |"
        ),
        (
            f"| Document files | {file_counts.added} | {file_counts.updated} | "
            f"{file_counts.unchanged} | {file_counts.changed} |"
        ),
        (
            f"| Annotations | {annotation_counts.added} | {annotation_counts.updated} | "
            f"{annotation_counts.unchanged} | {annotation_counts.changed} |"
        ),
        (
            f"| Notes | {note_counts.added} | {note_counts.updated} | {note_counts.unchanged} | "
            f"{note_counts.changed} |"
        ),
        f"| **Total** | **{total_added}** | **{total_updated}** | **{total_unchanged}** | **{total_changed}** |",
    ]
    return "\n".join(lines)


__all__ = ["sync_zotero"]
