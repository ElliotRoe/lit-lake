from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz  # pymupdf

from litlake.config import Paths, Settings
from litlake.db import connect_db, enqueue_job, init_db
from litlake.providers.extraction import ErrorClass, ExtractionResult, LocalPdfExtractionProvider
from litlake.queue import QueueEngine, QueuePolicy
from litlake.storage import FileLocator, LocalFSProvider
from litlake.workers import ExtractionJobHandler, WorkerRuntimeContext


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "text" / "extraction_contracts"
RAW_COLLAPSE_FIXTURE = FIXTURE_DIR / "raw_extraction_text_collapses_to_empty.txt"
PDF_TEXT_FIXTURE = FIXTURE_DIR / "pdf_visible_text_hyphen_only.txt"


class _StubStorageProvider:
    def __init__(self, resolved_path: Path):
        self._resolved_path = resolved_path

    def resolve(self, locator: FileLocator) -> Path:
        return self._resolved_path


class _StubBrokenExtractor:
    """Simulates a buggy extractor returning empty normalized text."""

    name = "local"
    version = "test"
    supported_mime_types = frozenset({"application/pdf"})

    def extract(self, locator: FileLocator, *, mime_type: str | None = None) -> ExtractionResult:
        return ExtractionResult(text="", page_texts=[""], metadata={"mode": "broken"})

    def classify_error(self, exc: Exception) -> ErrorClass:
        return "permanent_validation"


class ExtractionContractIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "test.db"
        self.models_path = self.root / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.conn = connect_db(self.db_path)
        init_db(self.conn)

        self.policy = QueuePolicy(
            lease_seconds=60,
            batch_size=10,
            max_attempts=3,
            backoff_base_seconds=1,
            backoff_max_seconds=5,
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def _settings(self, worker_id: str) -> Settings:
        return Settings(
            paths=Paths(root=self.root, db_path=self.db_path, models_path=self.models_path),
            worker_id=worker_id,
            embed_disabled=False,
            queue_lease_seconds=self.policy.lease_seconds,
            queue_batch_size=self.policy.batch_size,
            queue_max_attempts=self.policy.max_attempts,
            queue_backoff_base_seconds=self.policy.backoff_base_seconds,
            queue_backoff_max_seconds=self.policy.backoff_max_seconds,
            extraction_backend="local",
            gemini_api_key=None,
            zotero_db_path=None,
        )

    def _insert_pdf_file(self, pdf_path: Path) -> int:
        ref_id = self.conn.execute(
            """
            INSERT INTO reference_items(source_system, source_id, title)
            VALUES ('test', 'extract-contract-ref', 'Extraction contract repro')
            """
        ).lastrowid
        file_id = self.conn.execute(
            """
            INSERT INTO document_files(
                reference_id, file_path, mime_type,
                extraction_status, storage_kind, storage_uri
            )
            VALUES (?, ?, 'application/pdf', 'pending', 'local', ?)
            """,
            (int(ref_id), str(pdf_path), str(pdf_path)),
        ).lastrowid
        self.conn.commit()
        return int(file_id)

    def _enqueue_extraction(self, file_id: int) -> None:
        enqueue_job(
            self.conn,
            queue_name="extraction",
            job_type="extract_file",
            entity_type="document_file",
            entity_id=file_id,
            dedupe_key=f"extract_file:{file_id}",
            payload={"document_file_id": file_id},
            max_attempts=self.policy.max_attempts,
        )
        self.conn.commit()

    def _run_extraction_job(self, handler: ExtractionJobHandler, *, worker_id: str) -> None:
        engine = QueueEngine(self.conn, worker_id=worker_id, policy=self.policy)
        ctx = WorkerRuntimeContext(settings=self._settings(worker_id), queue_policy=self.policy)
        claimed = engine.claim("extraction")
        self.assertEqual(len(claimed), 1)
        handler.handle_claimed_job(self.conn, engine, claimed[0], ctx)

    def test_empty_normalized_text_is_terminal_and_persists_no_chunks(self) -> None:
        raw_text = RAW_COLLAPSE_FIXTURE.read_text(encoding="utf-8")
        self.assertTrue(raw_text.strip(), "fixture should be non-empty before normalization")

        dummy_pdf = self.root / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4\n%stub\n")

        file_id = self._insert_pdf_file(dummy_pdf)
        self._enqueue_extraction(file_id)

        handler = ExtractionJobHandler(
            extraction_providers=[_StubBrokenExtractor()],
            storage_provider=_StubStorageProvider(dummy_pdf),
        )
        self._run_extraction_job(handler, worker_id="contract-empty-normalized")

        status_row = self.conn.execute(
            "SELECT extraction_status, extraction_error FROM document_files WHERE id = ?",
            (file_id,),
        ).fetchone()
        self.assertEqual(status_row[0], "error")
        self.assertIn("empty normalized text", (status_row[1] or "").lower())

        chunk_count = self.conn.execute(
            "SELECT COUNT(*) FROM documents WHERE document_file_id = ? AND kind = 'fulltext_chunk'",
            (file_id,),
        ).fetchone()[0]
        self.assertEqual(int(chunk_count), 0)

    def test_problematic_hyphen_only_pdf_fails_before_chunk_insert(self) -> None:
        visible_text = PDF_TEXT_FIXTURE.read_text(encoding="utf-8")
        self.assertEqual(visible_text, "-")

        pdf_path = self.root / "hyphen-only.pdf"
        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            page.insert_text((72, 120), visible_text)
            doc.save(pdf_path)
        finally:
            doc.close()

        file_id = self._insert_pdf_file(pdf_path)
        self._enqueue_extraction(file_id)

        handler = ExtractionJobHandler(
            extraction_providers=[LocalPdfExtractionProvider()],
            storage_provider=LocalFSProvider(),
        )
        self._run_extraction_job(handler, worker_id="contract-hyphen-pdf")

        file_row = self.conn.execute(
            "SELECT extraction_status, extraction_error, extracted_text FROM document_files WHERE id = ?",
            (file_id,),
        ).fetchone()
        self.assertEqual(file_row[0], "error")
        self.assertIn("normalized extraction is empty", (file_row[1] or "").lower())
        self.assertIsNone(file_row[2])

        chunk_count = self.conn.execute(
            "SELECT COUNT(*) FROM documents WHERE document_file_id = ? AND kind = 'fulltext_chunk'",
            (file_id,),
        ).fetchone()[0]
        self.assertEqual(int(chunk_count), 0)


if __name__ == "__main__":
    unittest.main()
