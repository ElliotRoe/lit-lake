from __future__ import annotations

import json
import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class QueuePolicy:
    lease_seconds: int
    batch_size: int
    max_attempts: int
    backoff_base_seconds: int
    backoff_max_seconds: int


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    queue_name: str
    job_type: str
    entity_type: str
    entity_id: int
    payload: dict
    attempts: int
    max_attempts: int
    claim_token: str
    attempt_id: str
    attempt_no: int


class QueueEngine:
    def __init__(self, conn: sqlite3.Connection, worker_id: str, policy: QueuePolicy):
        self.conn = conn
        self.worker_id = worker_id
        self.policy = policy

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def record_worker_started(
        self,
        *,
        worker_type: str,
        backend_name: str,
        backend_version: str,
        config_hash: str,
    ) -> None:
        now = self._iso(self._now())
        self.conn.execute(
            """
            UPDATE worker_runs
            SET status = 'stopped'
            WHERE worker_type = ?
              AND worker_id <> ?
              AND status = 'running'
            """,
            (worker_type, self.worker_id),
        )
        self.conn.execute(
            """
            INSERT INTO worker_runs (
                worker_id, worker_type, backend_name, backend_version,
                config_hash, started_at, last_heartbeat_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
            ON CONFLICT(worker_id)
            DO UPDATE SET
                worker_type = excluded.worker_type,
                backend_name = excluded.backend_name,
                backend_version = excluded.backend_version,
                config_hash = excluded.config_hash,
                started_at = excluded.started_at,
                last_heartbeat_at = excluded.last_heartbeat_at,
                status = 'running'
            """,
            (
                self.worker_id,
                worker_type,
                backend_name,
                backend_version,
                config_hash,
                now,
                now,
            ),
        )
        self.conn.commit()

    def heartbeat(self) -> None:
        self.conn.execute(
            """
            UPDATE worker_runs
            SET last_heartbeat_at = CURRENT_TIMESTAMP
            WHERE worker_id = ?
            """,
            (self.worker_id,),
        )
        self.conn.commit()

    def stop_worker(self, status: str = "stopped") -> None:
        self.conn.execute(
            """
            UPDATE worker_runs
            SET status = ?, last_heartbeat_at = CURRENT_TIMESTAMP
            WHERE worker_id = ?
            """,
            (status, self.worker_id),
        )
        self.conn.commit()

    def reclaim_expired_claims(self, queue_name: str) -> int:
        retry_cur = self.conn.execute(
            """
            UPDATE jobs
            SET status = 'retry',
                claimed_by = NULL,
                claim_token = NULL,
                claim_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE queue_name = ?
              AND status = 'claimed'
              AND claim_expires_at IS NOT NULL
              AND datetime(claim_expires_at) <= CURRENT_TIMESTAMP
              AND attempts < max_attempts
              AND attempts < ?
            """,
            (queue_name, self.policy.max_attempts),
        )
        dead_cur = self.conn.execute(
            """
            UPDATE jobs
            SET status = 'dead',
                claimed_by = NULL,
                claim_token = NULL,
                claim_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE queue_name = ?
              AND status = 'claimed'
              AND claim_expires_at IS NOT NULL
              AND datetime(claim_expires_at) <= CURRENT_TIMESTAMP
              AND (attempts >= max_attempts OR attempts >= ?)
            """,
            (queue_name, self.policy.max_attempts),
        )
        self.conn.commit()
        return retry_cur.rowcount + dead_cur.rowcount

    def claim(self, queue_name: str) -> list[ClaimedJob]:
        now = self._now()
        expiry = now + timedelta(seconds=self.policy.lease_seconds)
        selected = self.conn.execute(
            """
            SELECT id, queue_name, job_type, entity_type, entity_id,
                   payload_json, attempts, max_attempts
            FROM jobs
            WHERE queue_name = ?
              AND status IN ('queued','retry')
              AND datetime(available_at) <= CURRENT_TIMESTAMP
              AND attempts < max_attempts
              AND attempts < ?
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
            """,
            (queue_name, self.policy.max_attempts, self.policy.batch_size),
        ).fetchall()

        claimed: list[ClaimedJob] = []
        for row in selected:
            claim_token = str(uuid.uuid4())
            attempt_id = str(uuid.uuid4())
            attempt_no = int(row[6]) + 1

            updated = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'claimed',
                    claimed_by = ?,
                    claim_token = ?,
                    claim_expires_at = ?,
                    attempts = attempts + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND status IN ('queued','retry')
                  AND datetime(available_at) <= CURRENT_TIMESTAMP
                  AND attempts < max_attempts
                  AND attempts < ?
                """,
                (
                    self.worker_id,
                    claim_token,
                    self._iso(expiry),
                    row[0],
                    self.policy.max_attempts,
                ),
            )
            if updated.rowcount != 1:
                continue

            self.conn.execute(
                """
                INSERT INTO job_attempts (
                    id, job_id, attempt_no, worker_id, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (attempt_id, row[0], attempt_no, self.worker_id, self._iso(now)),
            )

            payload = json.loads(row[5]) if row[5] else {}
            claimed.append(
                ClaimedJob(
                    job_id=row[0],
                    queue_name=row[1],
                    job_type=row[2],
                    entity_type=row[3],
                    entity_id=int(row[4]),
                    payload=payload,
                    attempts=attempt_no,
                    max_attempts=int(row[7]),
                    claim_token=claim_token,
                    attempt_id=attempt_id,
                    attempt_no=attempt_no,
                )
            )

        self.conn.commit()
        return claimed

    def renew_lease(self, job: ClaimedJob) -> bool:
        expiry = self._now() + timedelta(seconds=self.policy.lease_seconds)
        cur = self.conn.execute(
            """
            UPDATE jobs
            SET claim_expires_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND status = 'claimed'
              AND claim_token = ?
            """,
            (self._iso(expiry), job.job_id, job.claim_token),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def complete_success(
        self,
        job: ClaimedJob,
        *,
        backend_name: str,
        backend_version: str,
        metrics: dict | None = None,
        commit: bool = True,
    ) -> bool:
        now = self._iso(self._now())
        cur = self.conn.execute(
            """
            UPDATE jobs
            SET status = 'succeeded',
                claimed_by = NULL,
                claim_token = NULL,
                claim_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND status = 'claimed'
              AND claim_token = ?
            """,
            (job.job_id, job.claim_token),
        )
        if cur.rowcount == 1:
            self.conn.execute(
                """
                UPDATE job_attempts
                SET finished_at = ?,
                    outcome = 'succeeded',
                    backend_name = ?,
                    backend_version = ?,
                    metrics_json = ?
                WHERE id = ?
                """,
                (
                    now,
                    backend_name,
                    backend_version,
                    json.dumps(metrics or {}, separators=(",", ":")),
                    job.attempt_id,
                ),
            )
        else:
            self.conn.execute(
                """
                UPDATE job_attempts
                SET finished_at = ?,
                    outcome = 'retryable_fail',
                    backend_name = ?,
                    backend_version = ?,
                    error_class = 'stale_claim',
                    error_message = 'claim_token mismatch on completion'
                WHERE id = ?
                """,
                (now, backend_name, backend_version, job.attempt_id),
            )
        if commit:
            self.conn.commit()
        return cur.rowcount == 1

    def complete_failure(
        self,
        job: ClaimedJob,
        *,
        backend_name: str,
        backend_version: str,
        error_class: str,
        error_message: str,
        permanent: bool,
        commit: bool = True,
    ) -> bool:
        now = self._now()
        next_status = "dead"
        next_available = None
        outcome = "permanent_fail"

        if not permanent and job.attempts < min(job.max_attempts, self.policy.max_attempts):
            next_status = "retry"
            outcome = "retryable_fail"
            backoff = self._compute_backoff(job.attempts)
            next_available = self._iso(now + timedelta(seconds=backoff))

        cur = self.conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                available_at = COALESCE(?, available_at),
                claimed_by = NULL,
                claim_token = NULL,
                claim_expires_at = NULL,
                last_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND status = 'claimed'
              AND claim_token = ?
            """,
            (next_status, next_available, error_message, job.job_id, job.claim_token),
        )

        if cur.rowcount == 1:
            self.conn.execute(
                """
                UPDATE job_attempts
                SET finished_at = ?,
                    outcome = ?,
                    backend_name = ?,
                    backend_version = ?,
                    error_class = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    self._iso(now),
                    outcome,
                    backend_name,
                    backend_version,
                    error_class,
                    error_message,
                    job.attempt_id,
                ),
            )
        else:
            self.conn.execute(
                """
                UPDATE job_attempts
                SET finished_at = ?,
                    outcome = 'retryable_fail',
                    backend_name = ?,
                    backend_version = ?,
                    error_class = 'stale_claim',
                    error_message = 'claim_token mismatch on failure'
                WHERE id = ?
                """,
                (
                    self._iso(now),
                    backend_name,
                    backend_version,
                    job.attempt_id,
                ),
            )
        if commit:
            self.conn.commit()
        return cur.rowcount == 1

    def _compute_backoff(self, attempts: int) -> int:
        factor = max(0, attempts - 1)
        base = self.policy.backoff_base_seconds
        wait = min(self.policy.backoff_max_seconds, base * (2**factor))
        jitter = random.randint(0, max(1, wait // 4))
        return min(self.policy.backoff_max_seconds, wait + jitter)


__all__ = ["ClaimedJob", "QueueEngine", "QueuePolicy"]
