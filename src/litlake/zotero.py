from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from litlake.providers.extraction import SUPPORTED_EXTRACTION_MIME_TYPES


@dataclass
class ZoteroAttachment:
    item_id: int
    key: str
    parent_item_id: int
    content_type: str
    path: str
    resolved_path: str
    link_mode: int | None = None


@dataclass
class ZoteroAnnotation:
    item_id: int
    key: str
    parent_item_id: int
    parent_attachment_item_id: int
    parent_attachment_key: str | None
    annotation_type: int
    author_name: str | None
    text: str | None
    comment: str | None
    color: str | None
    page_label: str | None
    sort_index: str
    position: str
    is_external: int


@dataclass
class ZoteroNote:
    item_id: int
    key: str
    parent_item_id: int
    note_html: str | None
    title: str | None


@dataclass
class ZoteroItem:
    item_id: int
    key: str
    title: str | None
    authors: str | None
    date: str | None
    abstract_note: str | None
    attachments: list[ZoteroAttachment] = field(default_factory=list)
    annotations: list[ZoteroAnnotation] = field(default_factory=list)
    notes: list[ZoteroNote] = field(default_factory=list)


class ZoteroReader:
    def __init__(self, db_path: str | None = None):
        self.db_path = self._find_db_path(db_path)
        self.storage_dir = self.db_path.parent / "storage"

    @staticmethod
    def _find_db_path(explicit: str | None = None) -> Path:
        candidate = explicit or os.getenv("ZOTERO_DB_PATH")
        if candidate and "${" not in candidate:
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return path

        default_path = (Path.home() / "Zotero" / "zotero.sqlite").resolve()
        if default_path.exists():
            return default_path

        raise FileNotFoundError(
            "Could not find Zotero database. Set ZOTERO_DB_PATH or use default ~/Zotero/zotero.sqlite"
        )

    def _resolve_attachment_path(self, path_str: str | None, attachment_key: str) -> str | None:
        if not path_str:
            return None
        if path_str.startswith("storage:"):
            rel_path = path_str.replace("storage:", "", 1)
            full_path = (self.storage_dir / attachment_key / rel_path).resolve()
        else:
            full_path = Path(path_str).expanduser().resolve()
        if not full_path.exists():
            return None
        return str(full_path)

    def get_items(self) -> list[ZoteroItem]:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT
                i.itemID,
                i.key,
                title_val.value as title,
                abstract_val.value as abstract,
                SUBSTR(date_val.value, 1, 4) as date,
                (
                    SELECT GROUP_CONCAT(
                        CASE
                            WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                            THEN c.lastName || ', ' || c.firstName
                            WHEN c.lastName IS NOT NULL
                            THEN c.lastName
                            ELSE NULL
                        END, '; '
                    )
                    FROM itemCreators ic
                    JOIN creators c ON ic.creatorID = c.creatorID
                    WHERE ic.itemID = i.itemID
                ) as authors
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID
            LEFT JOIN itemData date_data ON i.itemID = date_data.itemID AND date_data.fieldID = 6
            LEFT JOIN itemDataValues date_val ON date_data.valueID = date_val.valueID
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            ORDER BY i.dateModified DESC
        """

        rows = conn.execute(query).fetchall()
        items = [
            ZoteroItem(
                item_id=int(r[0]),
                key=r[1],
                title=r[2],
                abstract_note=r[3],
                date=r[4],
                authors=r[5],
            )
            for r in rows
        ]
        items_by_id = {item.item_id: item for item in items}

        attachment_placeholders = ",".join("?" for _ in SUPPORTED_EXTRACTION_MIME_TYPES)
        attachment_rows = conn.execute(
            f"""
            SELECT
                ia.itemID,
                ia.parentItemID,
                ia.linkMode,
                ia.contentType,
                ia.path,
                att.key
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.contentType IN ({attachment_placeholders})
            """,
            tuple(sorted(SUPPORTED_EXTRACTION_MIME_TYPES)),
        ).fetchall()

        for attachment_row in attachment_rows:
            attachment_item_id = int(attachment_row[0])
            parent_item_id = attachment_row[1]
            link_mode = attachment_row[2]
            content_type = attachment_row[3]
            path_str = attachment_row[4]
            attachment_key = attachment_row[5]
            if parent_item_id is None or path_str is None or content_type is None:
                continue
            item = items_by_id.get(int(parent_item_id))
            if item is None:
                continue
            resolved_path = self._resolve_attachment_path(path_str, attachment_key)
            if resolved_path is None:
                continue
            item.attachments.append(
                ZoteroAttachment(
                    item_id=attachment_item_id,
                    key=attachment_key,
                    parent_item_id=int(parent_item_id),
                    content_type=str(content_type),
                    path=str(path_str),
                    resolved_path=resolved_path,
                    link_mode=int(link_mode) if link_mode is not None else None,
                )
            )

        annotation_rows = conn.execute(
            """
            SELECT
                ann_item.itemID,
                ann_item.key,
                att.parentItemID,
                ia.parentItemID,
                parent_att.key,
                ia.type,
                ia.authorName,
                ia.text,
                ia.comment,
                ia.color,
                ia.pageLabel,
                ia.sortIndex,
                ia.position,
                ia.isExternal
            FROM itemAnnotations ia
            JOIN items ann_item ON ann_item.itemID = ia.itemID
            LEFT JOIN itemAttachments att ON att.itemID = ia.parentItemID
            LEFT JOIN items parent_att ON parent_att.itemID = ia.parentItemID
            """
        ).fetchall()

        for row in annotation_rows:
            parent_item_id = row[2]
            if parent_item_id is None:
                continue
            item = items_by_id.get(int(parent_item_id))
            if item is None:
                continue
            item.annotations.append(
                ZoteroAnnotation(
                    item_id=int(row[0]),
                    key=row[1],
                    parent_item_id=int(parent_item_id),
                    parent_attachment_item_id=int(row[3]) if row[3] is not None else 0,
                    parent_attachment_key=row[4],
                    annotation_type=int(row[5]) if row[5] is not None else 0,
                    author_name=row[6],
                    text=row[7],
                    comment=row[8],
                    color=row[9],
                    page_label=row[10],
                    sort_index=row[11] or "",
                    position=row[12] or "",
                    is_external=int(row[13]) if row[13] is not None else 0,
                )
            )

        note_rows = conn.execute(
            """
            SELECT
                note_item.itemID,
                note_item.key,
                inote.parentItemID,
                inote.note,
                inote.title
            FROM itemNotes inote
            JOIN items note_item ON note_item.itemID = inote.itemID
            """
        ).fetchall()

        for row in note_rows:
            parent_item_id = row[2]
            if parent_item_id is None:
                continue
            item = items_by_id.get(int(parent_item_id))
            if item is None:
                continue
            item.notes.append(
                ZoteroNote(
                    item_id=int(row[0]),
                    key=row[1],
                    parent_item_id=int(parent_item_id),
                    note_html=row[3],
                    title=row[4],
                )
            )

        conn.close()
        return items


__all__ = [
    "ZoteroAttachment",
    "ZoteroAnnotation",
    "ZoteroItem",
    "ZoteroNote",
    "ZoteroReader",
]
