from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from litlake.db import connect_db, init_db
from litlake.migration import (
    LEGACY_EMPTY_CHUNK_REPAIR_ERROR,
    auto_migrate_if_needed,
    migrate_database,
)


def _create_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE reference_items (
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
            CREATE TABLE reference_external_ids (
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
            CREATE TABLE document_files (
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
            CREATE TABLE documents (
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

        conn.execute(
            """
            INSERT INTO reference_items (id, title, authors, year, source_system, source_id)
            VALUES (1, 'Paper Title', 'Doe, Jane', '2024', 'zotero', 'ITEMKEY')
            """
        )
        conn.execute(
            """
            INSERT INTO reference_external_ids (reference_id, scheme, value)
            VALUES (1, 'zotero_key', 'ITEMKEY')
            """
        )
        conn.execute(
            """
            INSERT INTO document_files (
                id, reference_id, file_path, mime_type, label, extracted_text,
                extraction_status, extraction_error, source_system, source_id
            )
            VALUES
            (1, 1, '/tmp/paper.pdf', 'application/pdf', 'main_pdf',
             '# Heading\\n\\nBody paragraph.', 'ready', NULL, 'zotero', 'ATTACH1'),
            (2, 1, '/tmp/other.pdf', 'application/pdf', 'main_pdf',
             NULL, 'error', 'failed', 'zotero', 'ATTACH2')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (
                id, reference_id, document_file_id, kind, content, chunk_index,
                embedding_status, embedding_updated_at, embedding_error, source_system, source_id
            )
            VALUES
            (1, 1, NULL, 'title', 'Paper Title', NULL, 'ready', CURRENT_TIMESTAMP, NULL, 'zotero', 'ITEMKEY'),
            (2, 1, NULL, 'abstract', 'Abstract text', NULL, 'ready', CURRENT_TIMESTAMP, NULL, 'zotero', 'ITEMKEY'),
            (3, 1, 1, 'pdf_chunk', 'chunk one', 0, 'ready', CURRENT_TIMESTAMP, NULL, NULL, NULL),
            (4, 1, 1, 'pdf_chunk', 'chunk two', 1, 'error', NULL, 'boom', NULL, NULL)
            """
        )
        conn.commit()
    finally:
        conn.close()


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "legacy.db"
        _create_legacy_db(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_migrate_legacy_db_preserves_text_and_seeds_jobs(self) -> None:
        report = migrate_database(self.db_path, queue_max_attempts=4)

        self.assertEqual(report.renamed_pdf_chunks, 2)
        self.assertGreaterEqual(report.storage_kind_backfilled, 0)
        self.assertEqual(report.storage_uri_backfilled, 2)
        self.assertEqual(report.queue_seed_extraction_jobs, 1)
        self.assertEqual(report.queue_seed_embedding_jobs, 1)

        conn = sqlite3.connect(self.db_path)
        try:
            kind_counts = dict(
                conn.execute(
                    "SELECT kind, COUNT(*) FROM documents GROUP BY kind"
                ).fetchall()
            )
            self.assertNotIn("pdf_chunk", kind_counts)
            self.assertEqual(kind_counts.get("fulltext_chunk"), 2)

            extracted = conn.execute(
                "SELECT extracted_text FROM document_files WHERE id = 1"
            ).fetchone()
            self.assertEqual(extracted[0], "# Heading\\n\\nBody paragraph.")

            storage = conn.execute(
                """
                SELECT storage_kind, storage_uri
                FROM document_files
                WHERE id = 1
                """
            ).fetchone()
            self.assertEqual(storage[0], "local")
            self.assertEqual(storage[1], "/tmp/paper.pdf")

            jobs = dict(
                conn.execute(
                    "SELECT dedupe_key, status FROM jobs ORDER BY dedupe_key"
                ).fetchall()
            )
            self.assertEqual(jobs.get("extract_file:2"), "queued")
            self.assertEqual(jobs.get("embed_document:4"), "queued")
        finally:
            conn.close()

    def test_migration_is_idempotent_for_data_conversion_and_queue_seeding(self) -> None:
        first = migrate_database(self.db_path, queue_max_attempts=4)
        second = migrate_database(self.db_path, queue_max_attempts=4)

        self.assertEqual(first.renamed_pdf_chunks, 2)
        self.assertEqual(second.renamed_pdf_chunks, 0)
        self.assertEqual(second.queue_seed_extraction_jobs, 0)
        self.assertEqual(second.queue_seed_embedding_jobs, 0)

    def test_auto_migrate_if_needed_detects_and_runs_once(self) -> None:
        conn = connect_db(self.db_path)
        try:
            init_db(conn)
            reasons, report = auto_migrate_if_needed(
                conn,
                queue_max_attempts=4,
                seed_jobs=False,
            )
            self.assertIsNotNone(report)
            self.assertGreater(len(reasons), 0)
            self.assertEqual(report.renamed_pdf_chunks, 2)
            self.assertEqual(report.queue_seed_embedding_jobs, 0)
            self.assertEqual(report.queue_seed_extraction_jobs, 0)

            reasons_second, report_second = auto_migrate_if_needed(
                conn,
                queue_max_attempts=4,
                seed_jobs=False,
            )
            self.assertEqual(reasons_second, [])
            self.assertIsNone(report_second)
        finally:
            conn.close()

    def test_auto_migrate_repairs_pending_empty_fulltext_chunks(self) -> None:
        db_path = Path(self.tmp.name) / "stuck-empty-chunk.db"
        conn = connect_db(db_path)
        try:
            init_db(conn)
            ref_id = conn.execute(
                """
                INSERT INTO reference_items(source_system, source_id, title)
                VALUES ('test', 'stuck-ref', 'Stuck Ref')
                """
            ).lastrowid
            conn.execute(
                """
                INSERT INTO documents(
                    reference_id, kind, content, embedding_status
                )
                VALUES (?, 'fulltext_chunk', '   ', 'pending')
                """,
                (int(ref_id),),
            )
            conn.commit()

            reasons, report = auto_migrate_if_needed(
                conn,
                queue_max_attempts=4,
                seed_jobs=False,
            )
            self.assertIsNotNone(report)
            self.assertIn(
                "documents.fulltext_chunk has pending rows with empty content",
                reasons,
            )
            self.assertEqual(report.pending_empty_fulltext_chunks_repaired, 1)

            row = conn.execute(
                """
                SELECT embedding_status, embedding_error
                FROM documents
                WHERE kind = 'fulltext_chunk'
                """
            ).fetchone()
            self.assertEqual(row[0], "error")
            self.assertEqual(row[1], LEGACY_EMPTY_CHUNK_REPAIR_ERROR)

            reasons_second, report_second = auto_migrate_if_needed(
                conn,
                queue_max_attempts=4,
                seed_jobs=False,
            )
            self.assertEqual(reasons_second, [])
            self.assertIsNone(report_second)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
