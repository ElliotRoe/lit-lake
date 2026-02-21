from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litlake.config import Paths, Settings
from litlake.db import connect_db, enqueue_job, init_db
from litlake.providers.extraction import ErrorClass, ExtractionResult
from litlake.queue import QueueEngine, QueuePolicy
from litlake.storage import FileLocator
from litlake.workers import ExtractionJobHandler, WorkerRuntimeContext


class StubStorageProvider:
    def __init__(self, resolved_path: Path):
        self.resolved_path = resolved_path

    def resolve(self, locator: FileLocator) -> Path:
        return self.resolved_path


class StubPageAwareExtractionProvider:
    name = "local"
    version = "test"

    def extract(self, locator: FileLocator) -> ExtractionResult:
        page_texts = ["alpha page one", "omega page two"]
        return ExtractionResult(text="\n\n".join(page_texts), page_texts=page_texts)

    def classify_error(self, exc: Exception) -> ErrorClass:
        return "retryable_io"


class StubTextOnlyExtractionProvider:
    name = "gemini"
    version = "test"

    def extract(self, locator: FileLocator) -> ExtractionResult:
        return ExtractionResult(text="alpha page one\n\nomega page two")

    def classify_error(self, exc: Exception) -> ErrorClass:
        return "retryable_io"


class ExtractionWorkerPageMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "test.db"
        self.models_path = self.root / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.pdf_path = self.root / "dummy.pdf"
        self.pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

        self.conn = connect_db(self.db_path)
        init_db(self.conn)

        self.policy = QueuePolicy(
            lease_seconds=60,
            batch_size=10,
            max_attempts=3,
            backoff_base_seconds=1,
            backoff_max_seconds=5,
        )
        self.engine = QueueEngine(self.conn, worker_id="test-worker", policy=self.policy)
        settings = Settings(
            paths=Paths(root=self.root, db_path=self.db_path, models_path=self.models_path),
            worker_id="test-worker",
            embed_disabled=False,
            queue_lease_seconds=self.policy.lease_seconds,
            queue_batch_size=self.policy.batch_size,
            queue_max_attempts=self.policy.max_attempts,
            queue_backoff_base_seconds=self.policy.backoff_base_seconds,
            queue_backoff_max_seconds=self.policy.backoff_max_seconds,
            gemini_api_key=None,
            zotero_db_path=None,
        )
        self.ctx = WorkerRuntimeContext(settings=settings, queue_policy=self.policy)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def _queue_extraction_job(self) -> int:
        ref_id = self.conn.execute(
            """
            INSERT INTO reference_items(source_system, source_id, title)
            VALUES ('test', 'ref-1', 'Test Ref')
            """
        ).lastrowid
        file_id = self.conn.execute(
            """
            INSERT INTO document_files(
                reference_id, file_path, mime_type, label,
                extraction_status, storage_kind, storage_uri
            )
            VALUES (?, ?, 'application/pdf', 'main_pdf', 'pending', 'local', ?)
            """,
            (ref_id, str(self.pdf_path), str(self.pdf_path)),
        ).lastrowid
        self.conn.commit()

        enqueue_job(
            self.conn,
            queue_name="extraction",
            job_type="extract_pdf",
            entity_type="document_file",
            entity_id=int(file_id),
            dedupe_key=f"extract_pdf:{file_id}",
            payload={"document_file_id": int(file_id)},
            max_attempts=3,
        )
        self.conn.commit()
        return int(file_id)

    def test_local_page_aware_extraction_persists_page_range(self) -> None:
        self._queue_extraction_job()
        claimed = self.engine.claim("extraction")
        self.assertEqual(len(claimed), 1)

        handler = ExtractionJobHandler(
            extraction_provider=StubPageAwareExtractionProvider(),
            storage_provider=StubStorageProvider(self.pdf_path),
        )
        handler.handle_claimed_job(self.conn, self.engine, claimed[0], self.ctx)

        row = self.conn.execute(
            """
            SELECT page_start, page_end
            FROM documents
            WHERE kind = 'pdf_chunk'
            ORDER BY chunk_index ASC
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], 2)

    def test_text_only_extraction_leaves_page_range_null(self) -> None:
        self._queue_extraction_job()
        claimed = self.engine.claim("extraction")
        self.assertEqual(len(claimed), 1)

        handler = ExtractionJobHandler(
            extraction_provider=StubTextOnlyExtractionProvider(),
            storage_provider=StubStorageProvider(self.pdf_path),
        )
        handler.handle_claimed_job(self.conn, self.engine, claimed[0], self.ctx)

        row = self.conn.execute(
            """
            SELECT page_start, page_end
            FROM documents
            WHERE kind = 'pdf_chunk'
            ORDER BY chunk_index ASC
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])


if __name__ == "__main__":
    unittest.main()
