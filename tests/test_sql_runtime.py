from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litlake.db import connect_db, init_db
from litlake.sql_runtime import execute_readonly_query


class SqlReadonlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "sql.db"
        self.conn = connect_db(self.db_path)
        init_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_select_allowed(self) -> None:
        rows = execute_readonly_query(self.conn, "SELECT 1 AS one")
        self.assertEqual(rows[0]["one"], 1)

    def test_write_denied(self) -> None:
        with self.assertRaises(Exception):
            execute_readonly_query(
                self.conn,
                "INSERT INTO reference_items(source_system, source_id) VALUES ('x', 'y')",
            )

    def test_dangerous_pragma_denied(self) -> None:
        with self.assertRaises(Exception):
            execute_readonly_query(self.conn, "PRAGMA writable_schema = ON")

    def test_documents_include_page_range_columns(self) -> None:
        rows = self.conn.execute("PRAGMA table_info(documents)").fetchall()
        columns = {row[1] for row in rows}
        self.assertIn("page_start", columns)
        self.assertIn("page_end", columns)


if __name__ == "__main__":
    unittest.main()
