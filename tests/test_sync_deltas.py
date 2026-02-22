from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litlake.db import connect_db, init_db
from litlake.sync import sync_zotero
from litlake.zotero import ZoteroItem


class SyncDeltaReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "sync_deltas.db"
        self.conn = connect_db(self.db_path)
        init_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_second_sync_reports_no_changes(self) -> None:
        items = [
            ZoteroItem(
                item_id=1,
                key="ABC123",
                title="A Title",
                authors="Doe, Jane",
                date="2024",
                abstract_note="An abstract",
            )
        ]

        with patch("litlake.sync.ZoteroReader") as reader_cls:
            reader_cls.return_value.get_items.return_value = items
            first = sync_zotero(self.conn, queue_max_attempts=3, explicit_db_path=None)
            second = sync_zotero(self.conn, queue_max_attempts=3, explicit_db_path=None)

        self.assertIn("Changes detected", first)
        self.assertIn("No library changes detected", second)
        self.assertIn("| References | 0 | 0 | 1 | 0 |", second)
        self.assertIn("| Documents | 0 | 0 | 2 | 0 |", second)

    def test_changed_reference_reports_updated(self) -> None:
        first_items = [
            ZoteroItem(
                item_id=1,
                key="ABC123",
                title="A Title",
                authors="Doe, Jane",
                date="2024",
                abstract_note="An abstract",
            )
        ]
        second_items = [
            ZoteroItem(
                item_id=1,
                key="ABC123",
                title="A Better Title",
                authors="Doe, Jane",
                date="2024",
                abstract_note="An abstract",
            )
        ]

        with patch("litlake.sync.ZoteroReader") as reader_cls:
            reader_cls.return_value.get_items.return_value = first_items
            sync_zotero(self.conn, queue_max_attempts=3, explicit_db_path=None)
            reader_cls.return_value.get_items.return_value = second_items
            updated = sync_zotero(self.conn, queue_max_attempts=3, explicit_db_path=None)

        self.assertIn("Changes detected", updated)
        self.assertIn("| References | 0 | 1 | 0 | 1 |", updated)
        self.assertIn("| Documents | 0 | 1 | 1 | 1 |", updated)


if __name__ == "__main__":
    unittest.main()
