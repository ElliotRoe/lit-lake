from __future__ import annotations

import argparse
import json
import sqlite3

from litlake.config import load_settings
from litlake.db import connect_db, init_db, queue_depth_by_state


def _requeue_dead(conn: sqlite3.Connection, *, queue_name: str | None, limit: int | None) -> int:
    conditions = ["status = 'dead'"]
    params: list[object] = []
    if queue_name:
        conditions.append("queue_name = ?")
        params.append(queue_name)
    where_sql = " AND ".join(conditions)

    if limit is not None:
        id_rows = conn.execute(
            f"""
            SELECT id
            FROM jobs
            WHERE {where_sql}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        ids = [str(row[0]) for row in id_rows]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"""
            UPDATE jobs
            SET status = 'retry',
                available_at = CURRENT_TIMESTAMP,
                claimed_by = NULL,
                claim_token = NULL,
                claim_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """,
            ids,
        )
        conn.commit()
        return cur.rowcount

    cur = conn.execute(
        f"""
        UPDATE jobs
        SET status = 'retry',
            available_at = CURRENT_TIMESTAMP,
            claimed_by = NULL,
            claim_token = NULL,
            claim_expires_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE {where_sql}
        """,
        params,
    )
    conn.commit()
    return cur.rowcount


def _cancel_jobs(
    conn: sqlite3.Connection,
    *,
    queue_name: str | None,
    job_type: str | None,
    entity_type: str | None,
    entity_id: int | None,
    include_claimed: bool,
) -> int:
    allowed_states = ["queued", "retry"]
    if include_claimed:
        allowed_states.append("claimed")

    conditions = [f"status IN ({','.join('?' for _ in allowed_states)})"]
    params: list[object] = list(allowed_states)
    if queue_name:
        conditions.append("queue_name = ?")
        params.append(queue_name)
    if job_type:
        conditions.append("job_type = ?")
        params.append(job_type)
    if entity_type:
        conditions.append("entity_type = ?")
        params.append(entity_type)
    if entity_id is not None:
        conditions.append("entity_id = ?")
        params.append(entity_id)

    where_sql = " AND ".join(conditions)
    cur = conn.execute(
        f"""
        UPDATE jobs
        SET status = 'cancelled',
            claimed_by = NULL,
            claim_token = NULL,
            claim_expires_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE {where_sql}
        """,
        params,
    )
    conn.commit()
    return cur.rowcount


def _cleanup_expired_claims(conn: sqlite3.Connection, *, queue_name: str | None) -> int:
    conditions = [
        "status = 'claimed'",
        "claim_expires_at IS NOT NULL",
        "datetime(claim_expires_at) <= CURRENT_TIMESTAMP",
    ]
    params: list[object] = []
    if queue_name:
        conditions.append("queue_name = ?")
        params.append(queue_name)
    where_sql = " AND ".join(conditions)

    cur = conn.execute(
        f"""
        UPDATE jobs
        SET status = 'retry',
            claimed_by = NULL,
            claim_token = NULL,
            claim_expires_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE {where_sql}
        """,
        params,
    )
    conn.commit()
    return cur.rowcount


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lit Lake queue maintenance commands")
    sub = parser.add_subparsers(dest="command", required=True)

    requeue = sub.add_parser("requeue-dead", help="Requeue dead-letter jobs")
    requeue.add_argument("--queue", dest="queue_name", choices=["extraction", "embedding"])
    requeue.add_argument("--limit", type=int)

    cancel = sub.add_parser("cancel", help="Cancel queued/retry jobs with optional filters")
    cancel.add_argument("--queue", dest="queue_name", choices=["extraction", "embedding"])
    cancel.add_argument("--job-type")
    cancel.add_argument("--entity-type")
    cancel.add_argument("--entity-id", type=int)
    cancel.add_argument("--include-claimed", action="store_true")

    cleanup = sub.add_parser("cleanup-leases", help="Force cleanup of expired claimed jobs")
    cleanup.add_argument("--queue", dest="queue_name", choices=["extraction", "embedding"])

    sub.add_parser("stats", help="Print queue depth by queue/state")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    conn = connect_db(settings.paths.db_path)
    init_db(conn)

    try:
        if args.command == "requeue-dead":
            changed = _requeue_dead(
                conn,
                queue_name=args.queue_name,
                limit=args.limit,
            )
            print(json.dumps({"action": "requeue-dead", "updated": changed}, indent=2))
            return

        if args.command == "cancel":
            changed = _cancel_jobs(
                conn,
                queue_name=args.queue_name,
                job_type=args.job_type,
                entity_type=args.entity_type,
                entity_id=args.entity_id,
                include_claimed=bool(args.include_claimed),
            )
            print(json.dumps({"action": "cancel", "updated": changed}, indent=2))
            return

        if args.command == "cleanup-leases":
            changed = _cleanup_expired_claims(conn, queue_name=args.queue_name)
            print(json.dumps({"action": "cleanup-leases", "updated": changed}, indent=2))
            return

        if args.command == "stats":
            print(json.dumps({"queue": queue_depth_by_state(conn)}, indent=2))
            return
    finally:
        conn.close()


if __name__ == "__main__":
    main()
