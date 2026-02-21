from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litlake.config import Paths, Settings
from litlake.db import connect_db, enqueue_job, init_db, seed_pending_jobs
from litlake.providers.extraction import LocalPdfExtractionProvider
from litlake.queue import QueueEngine, QueuePolicy
from litlake.storage import LocalFSProvider
from litlake.workers import ExtractionJobHandler, WorkerRuntimeContext
from tests.fixtures.pdf_factory import (
    MB,
    build_stall_regression_fixture_set,
    should_run_true_large_fixture,
)


class ExtractionQueueRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "lit_lake.db"
        self.models_path = self.root / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _policy(self, *, max_attempts: int = 1, batch_size: int = 16) -> QueuePolicy:
        return QueuePolicy(
            lease_seconds=60,
            batch_size=batch_size,
            max_attempts=max_attempts,
            backoff_base_seconds=1,
            backoff_max_seconds=2,
        )

    def _settings(self, policy: QueuePolicy, worker_id: str) -> Settings:
        return Settings(
            paths=Paths(root=self.root, db_path=self.db_path, models_path=self.models_path),
            worker_id=worker_id,
            embed_disabled=False,
            queue_lease_seconds=policy.lease_seconds,
            queue_batch_size=policy.batch_size,
            queue_max_attempts=policy.max_attempts,
            queue_backoff_base_seconds=policy.backoff_base_seconds,
            queue_backoff_max_seconds=policy.backoff_max_seconds,
            gemini_api_key=None,
            zotero_db_path=None,
        )

    def _insert_pdf_file(self, conn, *, name: str, pdf_path: Path) -> int:
        ref_id = conn.execute(
            """
            INSERT INTO reference_items(source_system, source_id, title)
            VALUES ('test', ?, ?)
            """,
            (f"source-{name}", f"Reference {name}"),
        ).lastrowid
        file_id = conn.execute(
            """
            INSERT INTO document_files(
                reference_id, file_path, mime_type, label,
                extraction_status, storage_kind, storage_uri
            )
            VALUES (?, ?, 'application/pdf', 'main_pdf', 'pending', 'local', ?)
            """,
            (int(ref_id), str(pdf_path), str(pdf_path)),
        ).lastrowid
        conn.commit()
        return int(file_id)

    def _enqueue_extraction(self, conn, *, file_id: int, max_attempts: int) -> None:
        enqueue_job(
            conn,
            queue_name="extraction",
            job_type="extract_pdf",
            entity_type="document_file",
            entity_id=file_id,
            dedupe_key=f"extract_pdf:{file_id}",
            payload={"document_file_id": file_id},
            max_attempts=max_attempts,
        )
        conn.commit()

    def _build_runtime(
        self,
        conn,
        *,
        worker_id: str,
        max_attempts: int = 1,
        batch_size: int = 16,
    ):
        policy = self._policy(max_attempts=max_attempts, batch_size=batch_size)
        engine = QueueEngine(conn, worker_id=worker_id, policy=policy)
        ctx = WorkerRuntimeContext(
            settings=self._settings(policy, worker_id=worker_id),
            queue_policy=policy,
        )
        handler = ExtractionJobHandler(
            extraction_provider=LocalPdfExtractionProvider(),
            storage_provider=LocalFSProvider(),
        )
        return engine, handler, ctx

    def _drain_extraction_queue(
        self,
        conn,
        *,
        worker_id: str,
        max_attempts: int = 1,
        batch_size: int = 16,
    ) -> None:
        engine, handler, ctx = self._build_runtime(
            conn,
            worker_id=worker_id,
            max_attempts=max_attempts,
            batch_size=batch_size,
        )
        for _ in range(100):
            claimed = engine.claim("extraction")
            if not claimed:
                break
            for job in claimed:
                handler.handle_claimed_job(conn, engine, job, ctx)

    def test_bad_pdf_does_not_block_remaining_jobs(self) -> None:
        conn = connect_db(self.db_path)
        init_db(conn)
        specs = build_stall_regression_fixture_set(self.root / "fixtures", large_size_bytes=2 * MB)
        selected = [spec for spec in specs if spec.name in {"small_1", "small_2", "malformed_1"}]

        file_ids: dict[str, int] = {}
        for spec in selected:
            file_id = self._insert_pdf_file(conn, name=spec.name, pdf_path=spec.path)
            self._enqueue_extraction(conn, file_id=file_id, max_attempts=1)
            file_ids[spec.name] = file_id

        self._drain_extraction_queue(conn, worker_id="regression-worker-1", max_attempts=1)

        for good_name in ("small_1", "small_2"):
            row = conn.execute(
                "SELECT extraction_status, extraction_error FROM document_files WHERE id = ?",
                (file_ids[good_name],),
            ).fetchone()
            self.assertEqual(row[0], "ready")
            self.assertIsNone(row[1])

        bad = conn.execute(
            "SELECT extraction_status, extraction_error FROM document_files WHERE id = ?",
            (file_ids["malformed_1"],),
        ).fetchone()
        self.assertEqual(bad[0], "error")
        self.assertTrue((bad[1] or "").strip())

        job_counts = {
            row[0]: int(row[1])
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            ).fetchall()
        }
        self.assertEqual(job_counts.get("succeeded", 0), 2)
        self.assertEqual(job_counts.get("dead", 0), 1)
        conn.close()

    def test_reclaims_expired_claim_and_resumes_successfully(self) -> None:
        conn = connect_db(self.db_path)
        init_db(conn)
        specs = build_stall_regression_fixture_set(self.root / "fixtures", large_size_bytes=2 * MB)
        good = next(spec for spec in specs if spec.name == "small_1")

        file_id = self._insert_pdf_file(conn, name=good.name, pdf_path=good.path)
        self._enqueue_extraction(conn, file_id=file_id, max_attempts=2)

        engine, handler, ctx = self._build_runtime(conn, worker_id="reclaim-worker", max_attempts=2)
        first_claim = engine.claim("extraction")
        self.assertEqual(len(first_claim), 1)

        conn.execute(
            """
            UPDATE jobs
            SET claim_expires_at = datetime('now', '-1 minute')
            WHERE id = ?
            """,
            (first_claim[0].job_id,),
        )
        conn.commit()

        reclaimed = engine.reclaim_expired_claims("extraction")
        self.assertEqual(reclaimed, 1)

        second_claim = engine.claim("extraction")
        self.assertEqual(len(second_claim), 1)
        handler.handle_claimed_job(conn, engine, second_claim[0], ctx)

        file_row = conn.execute(
            "SELECT extraction_status FROM document_files WHERE id = ?",
            (file_id,),
        ).fetchone()
        self.assertEqual(file_row[0], "ready")

        job_row = conn.execute(
            "SELECT status, attempts FROM jobs WHERE id = ?",
            (second_claim[0].job_id,),
        ).fetchone()
        self.assertEqual(job_row[0], "succeeded")
        self.assertEqual(int(job_row[1]), 2)
        conn.close()

    def test_restart_resumes_queued_work_without_duplicate_jobs(self) -> None:
        # First process run.
        conn1 = connect_db(self.db_path)
        init_db(conn1)
        specs = build_stall_regression_fixture_set(self.root / "fixtures", large_size_bytes=2 * MB)
        selected = [spec for spec in specs if spec.name in {"small_1", "small_2"}]

        file_ids: list[int] = []
        for spec in selected:
            file_id = self._insert_pdf_file(conn1, name=spec.name, pdf_path=spec.path)
            self._enqueue_extraction(conn1, file_id=file_id, max_attempts=1)
            file_ids.append(file_id)

        engine1, handler1, ctx1 = self._build_runtime(
            conn1,
            worker_id="restart-worker-1",
            max_attempts=1,
            batch_size=1,
        )
        first_claim = engine1.claim("extraction")
        self.assertEqual(len(first_claim), 1)
        handler1.handle_claimed_job(conn1, engine1, first_claim[0], ctx1)
        conn1.close()

        # Simulated restart process.
        conn2 = connect_db(self.db_path)
        init_db(conn2)
        seed_pending_jobs(conn2, queue_max_attempts=1)
        self._drain_extraction_queue(conn2, worker_id="restart-worker-2", max_attempts=1)

        ready_count = conn2.execute(
            "SELECT COUNT(*) FROM document_files WHERE extraction_status = 'ready'"
        ).fetchone()[0]
        self.assertEqual(int(ready_count), 2)

        total_jobs = conn2.execute(
            "SELECT COUNT(*) FROM jobs WHERE queue_name = 'extraction'"
        ).fetchone()[0]
        success_jobs = conn2.execute(
            "SELECT COUNT(*) FROM jobs WHERE queue_name = 'extraction' AND status = 'succeeded'"
        ).fetchone()[0]
        self.assertEqual(int(total_jobs), 2)
        self.assertEqual(int(success_jobs), 2)

        for file_id in file_ids:
            row = conn2.execute(
                "SELECT extraction_status FROM document_files WHERE id = ?",
                (file_id,),
            ).fetchone()
            self.assertEqual(row[0], "ready")
        conn2.close()

    def test_true_large_batch_plus_malformed_is_terminal(self) -> None:
        if not should_run_true_large_fixture():
            self.skipTest("Set LIT_LAKE_RUN_LARGE_FIXTURES=1 to run true >50MB queue regression")

        conn = connect_db(self.db_path)
        init_db(conn)
        specs = build_stall_regression_fixture_set(self.root / "fixtures")
        selected = [spec for spec in specs if spec.name in {"large_1", "large_2", "malformed_1"}]

        file_ids: dict[str, int] = {}
        for spec in selected:
            file_id = self._insert_pdf_file(conn, name=spec.name, pdf_path=spec.path)
            self._enqueue_extraction(conn, file_id=file_id, max_attempts=1)
            file_ids[spec.name] = file_id

        self._drain_extraction_queue(conn, worker_id="large-worker", max_attempts=1)

        for name in ("large_1", "large_2"):
            row = conn.execute(
                "SELECT extraction_status FROM document_files WHERE id = ?",
                (file_ids[name],),
            ).fetchone()
            self.assertEqual(row[0], "ready")

        bad = conn.execute(
            "SELECT extraction_status FROM document_files WHERE id = ?",
            (file_ids["malformed_1"],),
        ).fetchone()
        self.assertEqual(bad[0], "error")
        conn.close()


if __name__ == "__main__":
    unittest.main()
