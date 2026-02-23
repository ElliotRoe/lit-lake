from __future__ import annotations

import sqlite3
from typing import Any

from litlake.providers.embedding import EmbeddingProvider, RerankProvider


DENIED_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_ANALYZE,
}

DENIED_PRAGMAS = {
    "writable_schema",
    "journal_mode",
    "foreign_keys",
    "locking_mode",
    "legacy_file_format",
}


def register_ai_functions(
    conn: sqlite3.Connection,
    embedder: EmbeddingProvider,
    reranker: RerankProvider,
) -> None:
    import sqlite_vec
    from sqlite_vec import serialize_float32

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    def embed(text: str) -> bytes:
        vectors = embedder.embed([text])
        if not vectors:
            return serialize_float32([])
        return serialize_float32(vectors[0])

    def rerank_score(query: str, doc: str) -> float:
        return float(reranker.score(query, doc))

    conn.create_function("embed", 1, embed, deterministic=True)
    conn.create_function("rerank_score", 2, rerank_score)


def _authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger: str | None) -> int:
    if action in DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY

    if action == sqlite3.SQLITE_PRAGMA and arg1 and arg1.lower() in DENIED_PRAGMAS:
        return sqlite3.SQLITE_DENY

    return sqlite3.SQLITE_OK


def execute_readonly_query(conn: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    conn.set_authorizer(_authorizer)
    try:
        cur = conn.execute(query)
        columns = [d[0] for d in (cur.description or [])]
        rows: list[dict[str, Any]] = []
        for row in cur.fetchall():
            row_out: dict[str, Any] = {}
            for idx, col_name in enumerate(columns):
                value = row[idx]
                if isinstance(value, bytes):
                    row_out[col_name] = "<blob>"
                else:
                    row_out[col_name] = value
            rows.append(row_out)
        return rows
    finally:
        conn.set_authorizer(None)


__all__ = ["execute_readonly_query", "register_ai_functions"]
