from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZoteroItem:
    item_id: int
    key: str
    title: str | None
    authors: str | None
    date: str | None
    abstract_note: str | None
    pdf_path: str | None


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
                pdf_path=None,
            )
            for r in rows
        ]

        attachment_rows = conn.execute(
            """
            SELECT ia.parentItemID, ia.path, att.key
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.contentType = 'application/pdf'
            """
        ).fetchall()

        pdf_map: dict[int, tuple[str, str]] = {}
        for parent_id, path_str, key in attachment_rows:
            if parent_id is None or path_str is None:
                continue
            pdf_map[int(parent_id)] = (path_str, key)

        for item in items:
            payload = pdf_map.get(item.item_id)
            if not payload:
                continue
            path_str, key = payload
            if path_str.startswith("storage:"):
                rel_path = path_str.replace("storage:", "", 1)
                full_path = (self.storage_dir / key / rel_path).resolve()
                if full_path.exists():
                    item.pdf_path = str(full_path)
            else:
                path = Path(path_str).expanduser().resolve()
                if path.exists():
                    item.pdf_path = str(path)

        conn.close()
        return items


__all__ = ["ZoteroItem", "ZoteroReader"]
