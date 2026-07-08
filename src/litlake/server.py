from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, cast

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import Annotations, CallToolResult, ImageContent, TextContent, ToolAnnotations

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in non-uv environments
    def load_dotenv() -> bool:
        return False

from litlake.config import Settings, load_settings
from litlake.db import (
    connect_db,
    init_db,
    queue_depth_by_state,
    recent_attempt_errors,
    recent_dead_letter,
    seed_pending_jobs,
)
from litlake.docs import get_documentation_text
from litlake.logging_utils import configure_litlake_logging
from litlake.migration import auto_migrate_if_needed
from litlake.preview import PdfPreviewRenderer
from litlake.providers.chunking import ChunkingProvider, PdfChunkingProvider
from litlake.providers.embedding import FastEmbedEmbeddingProvider, FastEmbedRerankProvider
from litlake.providers.extraction import (
    ExtractionProvider,
    GeminiExtractionProvider,
    LocalPdfExtractionProvider,
    SUPPORTED_EXTRACTION_MIME_TYPES,
)
from litlake.queue import QueuePolicy
from litlake.sql_runtime import execute_readonly_query, register_ai_functions
from litlake.storage import LocalFSProvider
from litlake.sync import sync_zotero
from litlake.workers import (
    EmbeddingJobHandler,
    ExtractionJobHandler,
    QueueWorker,
    WorkerRuntimeContext,
)


logger = logging.getLogger("litlake")


class InitStatus:
    SYNCING_ZOTERO = "Syncing Zotero library..."
    LOADING_EMBED = "Loading embedding model (may download on first run)..."
    LOADING_RERANK = "Loading reranker model (may download on first run)..."
    STARTING_EMBED = "Starting embedding worker..."
    STARTING_EXTRACT = "Starting extraction worker..."
    READY = "Ready"


@dataclass
class AppState:
    settings: Settings
    conn: sqlite3.Connection
    conn_lock: threading.Lock
    init_message: str
    init_failed: bool
    workers: list[QueueWorker]
    embedding_worker: QueueWorker | None
    extraction_worker: QueueWorker | None


def _build_extraction_providers(settings: Settings) -> list[ExtractionProvider]:
    if settings.extraction_backend.strip().lower() == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "EXTRACTION_BACKEND=gemini requires GEMINI_API_KEY. "
                "Set GEMINI_API_KEY or switch EXTRACTION_BACKEND=local."
            )
        return [GeminiExtractionProvider(api_key=settings.gemini_api_key)]

    return [LocalPdfExtractionProvider()]


def _build_chunking_providers() -> list[ChunkingProvider]:
    return [PdfChunkingProvider()]


def _select_embedding_provider(settings: Settings):
    return FastEmbedEmbeddingProvider(models_path=settings.paths.models_path)


def _select_rerank_provider(settings: Settings):
    return FastEmbedRerankProvider()


def _background_init(state: AppState) -> None:
    try:
        extraction_providers = _build_extraction_providers(state.settings)
        chunking_providers = _build_chunking_providers()
        with state.conn_lock:
            state.init_message = InitStatus.SYNCING_ZOTERO
            try:
                msg = sync_zotero(
                    state.conn,
                    queue_max_attempts=state.settings.queue_max_attempts,
                    explicit_db_path=state.settings.zotero_db_path,
                )
                logger.info(msg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Zotero sync failed (continuing): %s", exc)

            queue_seed = seed_pending_jobs(
                state.conn,
                queue_max_attempts=state.settings.queue_max_attempts,
            )
            logger.info(
                "Seeded queue jobs: extraction=%s embedding=%s",
                queue_seed.extraction_jobs_created,
                queue_seed.embedding_jobs_created,
            )
            logger.info(
                "Extraction backend: %s (%s), mime_types=%s",
                extraction_providers[0].name,
                extraction_providers[0].version,
                ",".join(sorted(extraction_providers[0].supported_mime_types)),
            )

        # Model loading OUTSIDE the lock so sync_zotero_tool can proceed
        state.init_message = InitStatus.LOADING_EMBED
        embedding_provider = _select_embedding_provider(state.settings)

        state.init_message = InitStatus.LOADING_RERANK
        rerank_provider = _select_rerank_provider(state.settings)

        with state.conn_lock:
            register_ai_functions(state.conn, embedding_provider, rerank_provider)

        queue_policy = QueuePolicy(
            lease_seconds=state.settings.queue_lease_seconds,
            batch_size=state.settings.queue_batch_size,
            max_attempts=state.settings.queue_max_attempts,
            backoff_base_seconds=state.settings.queue_backoff_base_seconds,
            backoff_max_seconds=state.settings.queue_backoff_max_seconds,
        )
        runtime_ctx = WorkerRuntimeContext(settings=state.settings, queue_policy=queue_policy)

        if not state.settings.embed_disabled:
            state.init_message = InitStatus.STARTING_EMBED
            state.embedding_worker = QueueWorker(
                ctx=runtime_ctx,
                handler=EmbeddingJobHandler(embedding_provider=embedding_provider),
                name="embedding-worker",
            )
            state.embedding_worker.start()
            state.workers.append(state.embedding_worker)

        state.init_message = InitStatus.STARTING_EXTRACT
        state.extraction_worker = QueueWorker(
            ctx=runtime_ctx,
            handler=ExtractionJobHandler(
                extraction_providers=extraction_providers,
                chunking_providers=chunking_providers,
                storage_provider=LocalFSProvider(),
            ),
            name="extraction-worker",
        )
        state.extraction_worker.start()
        state.workers.append(state.extraction_worker)

        if state.embedding_worker:
            state.embedding_worker.wake()
        if state.extraction_worker:
            state.extraction_worker.wake()

        state.init_message = InitStatus.READY
        state.init_failed = False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background initialization failed")
        state.init_message = str(exc)
        state.init_failed = True


def _build_state(settings: Settings) -> AppState:
    conn = connect_db(settings.paths.db_path)
    init_db(conn)
    reasons, migration_report = auto_migrate_if_needed(
        conn,
        queue_max_attempts=settings.queue_max_attempts,
        seed_jobs=False,
    )
    if migration_report is not None:
        logger.info(
            "Auto-migration applied (%d checks): renamed_pdf_chunks=%d storage_uri_backfilled=%d",
            len(reasons),
            migration_report.renamed_pdf_chunks,
            migration_report.storage_uri_backfilled,
        )
        for reason in reasons:
            logger.info("Auto-migration reason: %s", reason)
    return AppState(
        settings=settings,
        conn=conn,
        conn_lock=threading.Lock(),
        init_message=InitStatus.SYNCING_ZOTERO,
        init_failed=False,
        workers=[],
        embedding_worker=None,
        extraction_worker=None,
    )


def _shutdown(state: AppState) -> None:
    for worker in state.workers:
        worker.stop()
    for worker in state.workers:
        worker.join(timeout=3)
    with state.conn_lock:
        state.conn.close()


@asynccontextmanager
async def app_lifespan(_mcp: FastMCP) -> AsyncIterator[AppState]:
    load_dotenv()
    settings = load_settings()
    log_path = configure_litlake_logging(settings.paths.root / "logs" / "lit-lake.log")
    state = _build_state(settings)

    logger.info("Using Lit Lake log file: %s", log_path)
    logger.info("Using database: %s", settings.paths.db_path)
    logger.info("Using models directory: %s", settings.paths.models_path)

    init_thread = threading.Thread(target=_background_init, args=(state,), name="background-init", daemon=True)
    init_thread.start()

    try:
        yield state
    finally:
        _shutdown(state)
        init_thread.join(timeout=3)


mcp = FastMCP("lit-lake-mcp", lifespan=app_lifespan)


def _state(ctx: Context) -> AppState:
    lifespan_state = ctx.request_context.lifespan_context
    return cast(AppState, lifespan_state)


def _resolve_debug_log_path() -> Path:
    settings = load_settings()
    return settings.paths.root / "logs" / "lit-lake.log"


def _resolve_claude_mcp_log_path() -> Path | None:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "Claude" / "mcp-server-lit-lake.log"

    if sys.platform.startswith("win"):
        appdata = os.getenv("APPDATA")
        if not appdata:
            return None
        return Path(appdata) / "Claude" / "logs" / "mcp-server-lit-lake.log"

    return None


@mcp.tool(
    name="sync_zotero",
    description=(
        "Import/update references from Zotero into the local library. Creates title and "
        "abstract documents for each reference, ingests supported attachments, and imports Zotero "
        "annotations/notes. Text artifacts are embedded asynchronously for semantic search, and "
        "extractable PDF attachments are queued for full-text extraction (local by default; "
        "optional Gemini for PDFs). Call this first if the library "
        "seems empty or out of date."
    ),
    annotations=ToolAnnotations(title="Sync Zotero Library", readOnlyHint=False, destructiveHint=False),
)
def sync_zotero_tool(ctx: Context) -> str:
    import time
    state = _state(ctx)
    last_exc = None
    for attempt in range(10):
        try:
            sync_conn = connect_db(state.settings.paths.db_path)
            try:
                result = sync_zotero(
                    sync_conn,
                    queue_max_attempts=state.settings.queue_max_attempts,
                    explicit_db_path=state.settings.zotero_db_path,
                )
            finally:
                sync_conn.close()

            if state.embedding_worker is not None:
                state.embedding_worker.wake()
            if state.extraction_worker is not None:
                state.extraction_worker.wake()

            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("sync_zotero_tool attempt %d failed: %s", attempt, exc)
            if "locked" in str(exc).lower():
                time.sleep(2)
                continue
            raise
    raise last_exc


@mcp.tool(
    name="sql_search",
    description=(
        "**IMPORTANT**: Before your first query, call get_documentation with section='workflows' "
        "to understand the data model, search patterns, and full-text access methods.\n\n"
        "Query the research library with full SQL flexibility. Supports semantic search via "
        "embed() and rerank_score() scalar functions."
    ),
    annotations=ToolAnnotations(title="Search Library with SQL", readOnlyHint=True, destructiveHint=False),
)
def sql_search(query: str, ctx: Context) -> str:
    state = _state(ctx)
    if state.init_failed:
        raise RuntimeError(
            f"AI models failed to load: {state.init_message}. Semantic search is unavailable."
        )
    if state.init_message != InitStatus.READY:
        return f"AI models are still loading ({state.init_message}). Please try again in a moment."

    if not query.strip():
        raise ValueError("Missing query")

    with state.conn_lock:
        rows = execute_readonly_query(state.conn, query)

    return json.dumps(rows, indent=2, ensure_ascii=False)


@mcp.tool(
    name="get_documentation",
    description=(
        "Get detailed documentation including supported workflows, SQL examples, and database "
        "schema. Call with section='workflows' for step-by-step guidance on citation assistance "
        "and library research. Sections: 'workflows', 'schema', 'examples', or omit for overview."
    ),
    annotations=ToolAnnotations(title="Get Documentation", readOnlyHint=True, destructiveHint=False),
)
def get_documentation(section: str | None = None) -> str:
    if section is not None and not isinstance(section, str):
        raise ValueError("section must be a string")
    return get_documentation_text(section)


@mcp.tool(
    name="library_status",
    description=(
        "Get an overview of the library: reference counts, document types, embedding status, "
        "and file extraction status by MIME type. Includes queue and worker diagnostics."
    ),
    annotations=ToolAnnotations(title="Get Library Status", readOnlyHint=True, destructiveHint=False),
)
def library_status(ctx: Context) -> str:
    state = _state(ctx)
    mime_placeholders = ",".join("?" for _ in SUPPORTED_EXTRACTION_MIME_TYPES)
    with state.conn_lock:
        reference_count = state.conn.execute("SELECT COUNT(*) FROM reference_items").fetchone()[0]
        document_count = state.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        file_count = state.conn.execute(
            "SELECT COUNT(*) FROM document_files"
        ).fetchone()[0]
        file_rows = state.conn.execute(
            "SELECT COALESCE(mime_type, 'unknown'), COUNT(*) FROM document_files GROUP BY mime_type"
        ).fetchall()

        embed_rows = state.conn.execute(
            "SELECT kind, embedding_status, COUNT(*) FROM documents GROUP BY kind, embedding_status"
        ).fetchall()
        extraction_rows = state.conn.execute(
            f"""
            SELECT COALESCE(mime_type, 'unknown'), extraction_status, COUNT(*)
            FROM document_files
            WHERE mime_type IN ({mime_placeholders})
            GROUP BY mime_type, extraction_status
            """,
            tuple(sorted(SUPPORTED_EXTRACTION_MIME_TYPES)),
        ).fetchall()

        embedding: dict[str, dict[str, int]] = {}
        for kind, status, count in embed_rows:
            kind_key = kind or "unknown"
            status_key = status or "null"
            embedding.setdefault(kind_key, {})[status_key] = int(count)

        extraction: dict[str, dict[str, int]] = {}
        for mime_type, status, count in extraction_rows:
            mime_key = mime_type or "unknown"
            status_key = status or "null"
            extraction.setdefault(mime_key, {})[status_key] = int(count)

        files_by_mime: dict[str, int] = {}
        for mime_type, count in file_rows:
            files_by_mime[mime_type or "unknown"] = int(count)

        embed_error_rows = state.conn.execute(
            """
            SELECT id, kind, embedding_error
            FROM documents
            WHERE embedding_status = 'error'
            ORDER BY embedding_updated_at DESC
            LIMIT 5
            """
        ).fetchall()
        extraction_error_rows = state.conn.execute(
            """
            SELECT id, extraction_error
            FROM document_files
            WHERE extraction_status = 'error'
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()

        queue_depth = queue_depth_by_state(state.conn)
        dead_jobs = recent_dead_letter(state.conn, limit=5)
        attempt_errors = recent_attempt_errors(state.conn, limit=5)
        workers = [dict(r) for r in state.conn.execute("SELECT * FROM worker_runs").fetchall()]

    embedding_errors = [
        {"doc_id": int(row[0]), "kind": row[1], "error": row[2]}
        for row in embed_error_rows
    ]
    extraction_errors = [{"file_id": int(row[0]), "error": row[1]} for row in extraction_error_rows]

    payload = {
        "init_status": {
            "ready": state.init_message == InitStatus.READY and not state.init_failed,
            "message": state.init_message,
        },
        "summary": {
            "references": int(reference_count),
            "documents": int(document_count),
            "files": int(file_count),
            "files_by_mime": files_by_mime,
        },
        "embedding": embedding,
        "extraction": extraction,
        "queue": queue_depth,
        "workers": workers,
        "errors": {
            "embedding": embedding_errors,
            "extraction": extraction_errors,
            "recent_attempts": attempt_errors,
            "dead_jobs": dead_jobs,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)



@mcp.tool(
    name="list_collections",
    description=(
        "List all Zotero collections (folders) with their hierarchy. "
        "Returns collection names, IDs, and parent-child relationships. "
        "Use collection_id with sql_search to filter references by collection, e.g.: "
        "SELECT r.* FROM reference_items r "
        "JOIN collection_items ci ON ci.reference_id = r.id "
        "JOIN collections c ON c.id = ci.collection_id "
        "WHERE c.id = <collection_id>"
    ),
    annotations=ToolAnnotations(title="List Zotero Collections", readOnlyHint=True, destructiveHint=False),
)
def list_collections(ctx: Context) -> str:
    import json
    state = _state(ctx)
    with state.conn_lock:
        collections = state.conn.execute(
            """
            SELECT
                c.id,
                c.zotero_collection_id,
                c.name,
                c.parent_zotero_collection_id,
                p.id as parent_id,
                p.name as parent_name,
                COUNT(ci.reference_id) as item_count
            FROM collections c
            LEFT JOIN collections p ON p.zotero_collection_id = c.parent_zotero_collection_id
            LEFT JOIN collection_items ci ON ci.collection_id = c.id
            GROUP BY c.id
            ORDER BY COALESCE(p.name, c.name), c.name
            """
        ).fetchall()

    result = [
        {
            "id": int(row[0]),
            "zotero_collection_id": row[1],
            "name": row[2],
            "parent_id": int(row[4]) if row[4] is not None else None,
            "parent_name": row[5],
            "item_count": int(row[6]),
        }
        for row in collections
    ]
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool(
    name="get_debug_log_path",
    description=(
        "Return Lit Lake and Claude MCP log file paths and whether each exists. "
        "Use this when users need to send logs for debugging."
    ),
    annotations=ToolAnnotations(title="Get Debug Log Path", readOnlyHint=True, destructiveHint=False),
)
def get_debug_log_path() -> str:
    litlake_path = _resolve_debug_log_path()
    litlake_browser_url = litlake_path.as_uri()
    litlake_browser_url_markdown = f"```text\n{litlake_browser_url}\n```"
    litlake_exists = litlake_path.is_file()

    claude_path = _resolve_claude_mcp_log_path()
    claude_browser_url = claude_path.as_uri() if claude_path else None
    claude_browser_url_markdown = (
        f"```text\n{claude_browser_url}\n```" if claude_browser_url is not None else None
    )
    claude_exists = claude_path.is_file() if claude_path is not None else False

    claude_text = (
        f"Claude MCP log path: {claude_browser_url_markdown}"
        if claude_browser_url_markdown
        else "Claude MCP log path is not available on this platform."
    )
    message = (
        "Send these logs when reporting Lit Lake issues.\n"
        "Lit Lake log path:\n"
        f"{litlake_browser_url_markdown}\n"
        f"{claude_text}"
        if litlake_exists
        else (
            "Expected Lit Lake log file was not found at this path. "
            "When available, you can open it by copying and pasting this into your browser:\n"
            f"{litlake_browser_url_markdown}\n"
            f"{claude_text}"
        )
    )
    payload = {
        "path": str(litlake_path),
        "browser_url": litlake_browser_url,
        "browser_url_markdown": litlake_browser_url_markdown,
        "exists": litlake_exists,
        "lit_lake_log": {
            "path": str(litlake_path),
            "browser_url": litlake_browser_url,
            "browser_url_markdown": litlake_browser_url_markdown,
            "exists": litlake_exists,
        },
        "claude_mcp_log": {
            "path": str(claude_path) if claude_path is not None else None,
            "browser_url": claude_browser_url,
            "browser_url_markdown": claude_browser_url_markdown,
            "exists": claude_exists,
        },
        "message": message,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


@mcp.tool(
    name="preview_document_pdf_pages",
    description=(
        "Render PNG images of PDF pages for VISUAL inspection (figures, tables, layout). "
        "For programmatic text access, use document_files.extracted_text or query fulltext_chunk "
        "documents instead. Requires document_file_id from the document_files table."
    ),
    annotations=ToolAnnotations(title="Preview PDF Pages", readOnlyHint=True, destructiveHint=False),
)
def preview_document_pdf_pages(
    document_file_id: int,
    ctx: Context,
    size: int = 1024,
    start_page: int = 1,
    end_page: int | None = None,
) -> CallToolResult:
    state = _state(ctx)

    size = max(64, min(4096, int(size)))
    if end_page is None:
        end_page = start_page

    if start_page <= 0 or end_page <= 0:
        raise ValueError("Pages are 1-based; start_page/end_page must be >= 1")
    if start_page > end_page:
        raise ValueError("start_page must be <= end_page")
    if (end_page - start_page + 1) > 10:
        raise ValueError("Requested too many pages in one call; max is 10")

    with state.conn_lock:
        row = state.conn.execute(
            "SELECT file_path, mime_type FROM document_files WHERE id = ?",
            (document_file_id,),
        ).fetchone()
    if not row:
        raise ValueError(f"document_files row not found for id={document_file_id}")
    file_path, mime_type = row[0], row[1]

    if mime_type and mime_type != "application/pdf":
        raise ValueError(
            f"Only application/pdf is supported for PDF page previews (got {mime_type})"
        )

    batch = PdfPreviewRenderer.page_range_png(file_path, start_page, end_page, size)
    content: list[TextContent | ImageContent] = []
    for page in batch.pages:
        content.append(TextContent(type="text", text=f"Page {page.page_number}"))
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(page.png_bytes).decode("ascii"),
                mimeType="image/png",
                annotations=Annotations(audience=["user", "assistant"], priority=1.0),
            )
        )
    return CallToolResult(content=content)


@mcp.tool(
    name="get_passage_context",
    description=(
        "Get the surrounding context for a passage found by semantic search. "
        "Provide a document_id from a search result to see the text before and after. "
        "Use 'window' for a number of surrounding segments (default 2), "
        "'pages' to get all text from a page range (e.g. pages=1 for the same page, "
        "pages=2 for ±1 page), or both. "
        "Useful when the user asks to 'show more', 'what comes before/after', "
        "'show me the whole page', or 'give me more context'."
    ),
    annotations=ToolAnnotations(title="Get Passage Context", readOnlyHint=True, destructiveHint=False),
)
def get_passage_context(
    document_id: int,
    ctx: Context,
    window: int | None = None,
    pages: int | None = None,
) -> str:
    state = _state(ctx)

    with state.conn_lock:
        # Get the target chunk
        target = state.conn.execute(
            """
            SELECT id, document_file_id, chunk_index, content,
                   json_extract(metadata_json, '$.loc.page_start') as page_start,
                   json_extract(metadata_json, '$.loc.page_end') as page_end,
                   reference_id
            FROM documents
            WHERE id = ? AND kind = 'fulltext_chunk'
            """,
            (document_id,),
        ).fetchone()

        if not target:
            raise ValueError(f"No fulltext_chunk found for document_id={document_id}")

        doc_file_id = target[1]
        chunk_index = int(target[2])
        target_content = target[3]
        page_start = int(target[4]) if target[4] is not None else None
        page_end = int(target[5]) if target[5] is not None else None
        reference_id = int(target[6])

        # Get title for context header
        ref_row = state.conn.execute(
            "SELECT title FROM reference_items WHERE id = ?", (reference_id,)
        ).fetchone()
        title = ref_row[0] if ref_row else "(unknown)"

        if pages is not None and page_start is not None:
            # Page-based context: get all chunks overlapping the page range
            page_radius = max(0, pages - 1)
            p_min = page_start - page_radius
            p_max = (page_end or page_start) + page_radius
            rows = state.conn.execute(
                """
                SELECT chunk_index, content,
                       json_extract(metadata_json, '$.loc.page_start') as ps
                FROM documents
                WHERE document_file_id = ? AND kind = 'fulltext_chunk'
                AND CAST(json_extract(metadata_json, '$.loc.page_start') AS INTEGER) >= ?
                AND CAST(json_extract(metadata_json, '$.loc.page_start') AS INTEGER) <= ?
                ORDER BY chunk_index
                """,
                (doc_file_id, p_min, p_max),
            ).fetchall()
        else:
            # Window-based context: get chunks by index proximity
            w = window if window is not None else 2
            idx_min = max(0, chunk_index - w)
            idx_max = chunk_index + w
            rows = state.conn.execute(
                """
                SELECT chunk_index, content,
                       json_extract(metadata_json, '$.loc.page_start') as ps
                FROM documents
                WHERE document_file_id = ? AND kind = 'fulltext_chunk'
                AND chunk_index >= ? AND chunk_index <= ?
                ORDER BY chunk_index
                """,
                (doc_file_id, idx_min, idx_max),
            ).fetchall()

    if not rows:
        return target_content or ""

    # Build output with page markers
    parts: list[str] = []
    parts.append(f"**{title}**")
    current_page: int | None = None
    for row in rows:
        ci = int(row[0])
        content = row[1] or ""
        ps = int(row[2]) if row[2] is not None else None
        if ps is not None and ps != current_page:
            parts.append(f"\n--- p. {ps} ---")
            current_page = ps
        # Mark the target chunk
        if ci == chunk_index:
            parts.append(f">>> {content} <<<")
        else:
            parts.append(content)

    return "\n\n".join(parts)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()


@mcp.tool()
def annotate_pdf_chunks(ctx: Context, 
    reference_id: int,
    chunk_indices: list[int],
    color: tuple[float, float, float] = (1.0, 0.8, 0.0),
    label: str = "",
) -> CallToolResult:
    """Highlight specific chunks in a PDF file using PyMuPDF.

    Writes highlight annotations directly into the PDF file so they appear
    when opening in Zotero or any PDF viewer. This is a temporary solution
    until the Zotero 7 local API is available.

    Zotero may be open. Reload the PDF in Zotero to show new annotations.
    To edit annotations in Zotero, use File → Import Annotations after reloading.

    Args:
        reference_id: The reference item ID from the database.
        chunk_indices: List of chunk_index values to highlight.
        color: RGB color tuple (0.0-1.0) for the highlight. Default: yellow.
        label: Optional label/comment to attach to each annotation.

    SQL to find chunk indices:
        SELECT d.chunk_index, d.content, json_extract(d.metadata_json, '$.loc.page_start') as page
        FROM documents d
        JOIN document_files df ON df.id = d.document_file_id
        WHERE df.reference_id = <reference_id>
        AND d.kind = 'fulltext_chunk'
        ORDER BY d.chunk_index;
    """
    import fitz

    state = _state(ctx)
    db = connect_db(state.settings.paths.db_path)

    row = db.execute(
        "SELECT df.file_path FROM document_files df WHERE df.reference_id = ? AND df.mime_type = 'application/pdf' LIMIT 1",
        (reference_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"No PDF found for reference_id={reference_id}")
    file_path = row[0]

    chunks = db.execute(
        f"""
        SELECT d.chunk_index, d.content,
               json_extract(d.metadata_json, '$.loc.page_start') as page_start,
               json_extract(d.metadata_json, '$.loc.page_end') as page_end
        FROM documents d
        JOIN document_files df ON df.id = d.document_file_id
        WHERE df.reference_id = ? AND d.kind = 'fulltext_chunk'
        AND d.chunk_index IN ({','.join('?' * len(chunk_indices))})
        """,
        (reference_id, *chunk_indices),
    ).fetchall()

    if not chunks:
        raise ValueError(f"No chunks found for reference_id={reference_id} with indices={chunk_indices}")

    doc = fitz.open(file_path)
    highlighted = 0

    try:
        for chunk_index, content, page_start, page_end in chunks:
            if page_start is None:
                continue
            start_idx = max(0, int(page_start) - 1)
            end_idx = min(int(page_end) if page_end else int(page_start), len(doc) - 1)
            words = content.split()
            mid = len(words) // 2
            search_text = " ".join(words[max(0, mid - 8):mid + 8]).strip()

            for page_idx in range(start_idx, end_idx + 1):
                page = doc[page_idx]
                instances = page.search_for(search_text)
                if not instances:
                    search_text = " ".join(words[2:12]).strip()
                    instances = page.search_for(search_text)
                if instances:
                    annot = page.add_highlight_annot(instances)
                    annot.set_colors(stroke=color)
                    if label:
                        annot.set_info(content=label)
                    annot.update()
                    highlighted += 1
                    break

        doc.save(file_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    finally:
        doc.close()

    return CallToolResult(
        content=[TextContent(type="text", text=f"Highlighted {highlighted}/{len(chunks)} chunks in {file_path}")]
    )


@mcp.tool()
def annotate_quotes(
    ctx: Context,
    reference_id: int,
    quotes: list[str],
    color: tuple[float, float, float] = (0.0, 0.8, 0.0),
    label: str = "",
) -> CallToolResult:
    """Highlight exact quote strings in a PDF file using PyMuPDF.

    Searches for each quote string in the PDF and highlights all matches.
    More precise than chunk-based annotation.

    Note: quotes containing typographic characters (e.g. curly quotes, typographic
    apostrophes, or em-dashes) may not be found if the PDF uses different encoding.
    PyMuPDF will match up to the problematic character and stop. To work around
    this, shorten the quote to end before the problematic character.
    Example: use "the company changed its strategy" instead of
    "the company changed its strategy’s outcome".

    Args:
        reference_id: The reference item ID from the database.
        quotes: List of exact quote strings to search for and highlight.
        color: RGB color tuple (0.0-1.0). Default: green.
        label: Optional comment to attach to each annotation.
    """
    import fitz

    state = _state(ctx)
    db = connect_db(state.settings.paths.db_path)

    row = db.execute(
        "SELECT file_path FROM document_files WHERE reference_id = ? AND mime_type = 'application/pdf' LIMIT 1",
        (reference_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"No PDF found for reference_id={reference_id}")
    file_path = row[0]

    doc = fitz.open(file_path)
    results = []

    try:
        for quote in quotes:
            found = False
            for page in doc:
                instances = page.search_for(quote)
                if instances:
                    annot = page.add_highlight_annot(instances)
                    annot.set_colors(stroke=color)
                    if label:
                        annot.set_info(content=label)
                    annot.update()
                    found = True
                    break
            results.append(f"{'✅' if found else '❌'} {quote[:60]}...")

        doc.save(file_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    finally:
        doc.close()

    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(results))]
    )
