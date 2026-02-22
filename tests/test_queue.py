from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litlake.db import connect_db, init_db, enqueue_job
from litlake.queue import QueueEngine, QueuePolicy


class QueueEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "queue.db"
        self.conn = connect_db(self.db_path)
        init_db(self.conn)
        self.engine = QueueEngine(
            self.conn,
            worker_id="test-worker",
            policy=QueuePolicy(
                lease_seconds=60,
                batch_size=10,
                max_attempts=3,
                backoff_base_seconds=1,
                backoff_max_seconds=5,
            ),
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_claim_and_success(self) -> None:
        created = enqueue_job(
            self.conn,
            queue_name="embedding",
            job_type="embed_document",
            entity_type="document",
            entity_id=1,
            dedupe_key="embed_document:1",
        )
        self.assertTrue(created)

        claimed = self.engine.claim("embedding")
        self.assertEqual(len(claimed), 1)

        done = self.engine.complete_success(
            claimed[0], backend_name="fastembed", backend_version="test"
        )
        self.assertTrue(done)

        row = self.conn.execute("SELECT status FROM jobs WHERE id = ?", (claimed[0].job_id,)).fetchone()
        self.assertEqual(row[0], "succeeded")

    def test_retry_then_dead(self) -> None:
        enqueue_job(
            self.conn,
            queue_name="extraction",
            job_type="extract_file",
            entity_type="document_file",
            entity_id=4,
            dedupe_key="extract_file:4",
            max_attempts=1,
        )
        claimed = self.engine.claim("extraction")
        self.assertEqual(len(claimed), 1)

        self.engine.complete_failure(
            claimed[0],
            backend_name="local",
            backend_version="test",
            error_class="retryable_io",
            error_message="boom",
            permanent=False,
        )

        row = self.conn.execute("SELECT status FROM jobs WHERE id = ?", (claimed[0].job_id,)).fetchone()
        self.assertEqual(row[0], "dead")

    def test_stale_claim_token_is_rejected(self) -> None:
        enqueue_job(
            self.conn,
            queue_name="embedding",
            job_type="embed_document",
            entity_type="document",
            entity_id=22,
            dedupe_key="embed_document:22",
        )
        claimed = self.engine.claim("embedding")
        self.assertEqual(len(claimed), 1)
        job = claimed[0]

        self.conn.execute(
            "UPDATE jobs SET claim_token = 'different-token' WHERE id = ?",
            (job.job_id,),
        )
        self.conn.commit()

        ok = self.engine.complete_success(job, backend_name="fastembed", backend_version="test")
        self.assertFalse(ok)

        row = self.conn.execute(
            "SELECT status, claim_token FROM jobs WHERE id = ?",
            (job.job_id,),
        ).fetchone()
        self.assertEqual(row[0], "claimed")
        self.assertEqual(row[1], "different-token")

        attempt = self.conn.execute(
            "SELECT outcome, error_class FROM job_attempts WHERE id = ?",
            (job.attempt_id,),
        ).fetchone()
        self.assertEqual(attempt[0], "retryable_fail")
        self.assertEqual(attempt[1], "stale_claim")

    def test_reclaim_expired_claim(self) -> None:
        enqueue_job(
            self.conn,
            queue_name="extraction",
            job_type="extract_file",
            entity_type="document_file",
            entity_id=10,
            dedupe_key="extract_file:10",
        )
        claimed = self.engine.claim("extraction")
        self.assertEqual(len(claimed), 1)
        job = claimed[0]

        self.conn.execute(
            """
            UPDATE jobs
            SET claim_expires_at = datetime('now', '-1 minute')
            WHERE id = ?
            """,
            (job.job_id,),
        )
        self.conn.commit()

        reclaimed = self.engine.reclaim_expired_claims("extraction")
        self.assertEqual(reclaimed, 1)
        row = self.conn.execute(
            "SELECT status, claimed_by, claim_token FROM jobs WHERE id = ?",
            (job.job_id,),
        ).fetchone()
        self.assertEqual(row[0], "retry")
        self.assertIsNone(row[1])
        self.assertIsNone(row[2])


if __name__ == "__main__":
    unittest.main()
