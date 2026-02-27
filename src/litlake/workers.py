from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
from dataclasses import dataclass
from typing import Protocol

from litlake.config import Settings
from litlake.db import connect_db, enqueue_job
from litlake.providers.chunking import (
    ChunkingProvider,
    ChunkingValidationError,
    PdfChunkingProvider,
)
from litlake.providers.embedding import EmbeddingProvider
from litlake.providers.extraction import (
    ErrorClass,
    ExtractionProvider,
    canonicalize_mime_type,
)
from litlake.queue import ClaimedJob, QueueEngine, QueuePolicy
from litlake.storage import FileLocator, StorageProvider


logger = logging.getLogger(__name__)
FULLTEXT_CHUNK_KIND = "fulltext_chunk"


def _is_permanent(error_class: ErrorClass) -> bool:
    return error_class in {"permanent_validation", "permanent_not_found", "permanent_unsupported"}


@dataclass
class WorkerRuntimeContext:
    settings: Settings
    queue_policy: QueuePolicy


class StaleClaimError(RuntimeError):
    pass


class SkippedExtractionError(RuntimeError):
    pass


class ProviderResolutionError(RuntimeError):
    pass


class QueueJobHandler(Protocol):
    queue_name: str
    worker_type: str
    idle_wait_seconds: float
    provider_name: str
    provider_version: str

    def handle_claimed_job(
        self,
        conn: sqlite3.Connection,
        queue_engine: QueueEngine,
        job: ClaimedJob,
        ctx: WorkerRuntimeContext,
    ) -> None:
        ...


class QueueWorker(threading.Thread):
    def __init__(self, *, ctx: WorkerRuntimeContext, handler: QueueJobHandler, name: str | None = None):
        worker_name = name or f"{handler.worker_type}-worker"
        super().__init__(name=worker_name, daemon=True)
        self.ctx = ctx
        self.handler = handler
        self.worker_instance_id = f"{self.ctx.settings.worker_id}:{self.handler.worker_type}"
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.error: Exception | None = None

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()

    def wake(self) -> None:
        self.wake_event.set()

    def _open(self) -> tuple[sqlite3.Connection, QueueEngine, sqlite3.Connection, QueueEngine]:
        job_conn = connect_db(self.ctx.settings.paths.db_path)
        job_queue_engine = QueueEngine(job_conn, self.worker_instance_id, self.ctx.queue_policy)
        job_queue_engine.record_worker_started(
            worker_type=self.handler.worker_type,
            backend_name=self.handler.provider_name,
            backend_version=self.handler.provider_version,
            config_hash=self.ctx.settings.queue_config_hash,
        )
        lease_conn = connect_db(self.ctx.settings.paths.db_path)
        lease_queue_engine = QueueEngine(lease_conn, self.worker_instance_id, self.ctx.queue_policy)
        return job_conn, job_queue_engine, lease_conn, lease_queue_engine

    def _run_with_lease_renewal(
        self,
        lease_queue_engine: QueueEngine,
        job: ClaimedJob,
        fn,
    ) -> None:
        renew_stop = threading.Event()
        interval = max(1, self.ctx.queue_policy.lease_seconds // 3)

        def _lease_loop() -> None:
            while not renew_stop.wait(timeout=interval):
                if self.stop_event.is_set():
                    return
                if not lease_queue_engine.renew_lease(job):
                    return

        renewer = threading.Thread(
            target=_lease_loop,
            name=f"{self.handler.worker_type}-lease-renewer",
            daemon=True,
        )
        renewer.start()
        try:
            fn()
        finally:
            renew_stop.set()
            renewer.join(timeout=1.0)

    def run(self) -> None:  # pragma: no cover - integration behavior
        job_conn, job_queue_engine, lease_conn, lease_queue_engine = self._open()
        try:
            while not self.stop_event.is_set():
                job_queue_engine.heartbeat()
                job_queue_engine.reclaim_expired_claims(self.handler.queue_name)
                claimed = job_queue_engine.claim(self.handler.queue_name)
                if not claimed:
                    self.wake_event.wait(timeout=self.handler.idle_wait_seconds)
                    self.wake_event.clear()
                    continue

                for job in claimed:
                    if self.stop_event.is_set():
                        break
                    self._run_with_lease_renewal(
                        lease_queue_engine,
                        job,
                        lambda: self.handler.handle_claimed_job(
                            job_conn,
                            job_queue_engine,
                            job,
                            self.ctx,
                        ),
                    )

            job_queue_engine.stop_worker("stopped")
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            logger.exception("%s worker failed", self.handler.worker_type)
            job_queue_engine.stop_worker("failed")
        finally:
            job_conn.close()
            lease_conn.close()


class EmbeddingJobHandler:
    queue_name = "embedding"
    worker_type = "embedding"
    idle_wait_seconds = 1.5

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.provider_name = embedding_provider.name
        self.provider_version = embedding_provider.version

    def handle_claimed_job(
        self,
        conn: sqlite3.Connection,
        queue_engine: QueueEngine,
        job: ClaimedJob,
        ctx: WorkerRuntimeContext,
    ) -> None:
        try:
            row = conn.execute(
                """
                SELECT id, content, kind
                FROM documents
                WHERE id = ?
                """,
                (job.entity_id,),
            ).fetchone()
            if not row:
                raise FileNotFoundError(f"Document not found: {job.entity_id}")

            content = row[1]
            if content is None or not str(content).strip():
                raise ValueError(f"Document {job.entity_id} has empty content")

            vector = self.embedding_provider.embed([str(content)])[0]
            vec_bytes = struct.pack(f"{len(vector)}f", *vector)

            conn.execute("BEGIN")
            conn.execute("DELETE FROM vec_documents WHERE rowid = ?", (job.entity_id,))
            conn.execute(
                "INSERT INTO vec_documents(rowid, embedding) VALUES (?, ?)",
                (job.entity_id, sqlite3.Binary(vec_bytes)),
            )
            conn.execute(
                """
                UPDATE documents
                SET embedding_status = 'ready',
                    embedding_updated_at = CURRENT_TIMESTAMP,
                    embedding_error = NULL,
                    embedding_backend = ?,
                    embedding_model = ?
                WHERE id = ?
                """,
                (self.embedding_provider.name, self.embedding_provider.version, job.entity_id),
            )
            completed = queue_engine.complete_success(
                job,
                backend_name=self.embedding_provider.name,
                backend_version=self.embedding_provider.version,
                commit=False,
            )
            if not completed:
                raise StaleClaimError(f"Lost claim while completing job {job.job_id}")
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            if isinstance(exc, StaleClaimError):
                logger.warning("%s", exc)
                return

            error_class: ErrorClass = "retryable_io"
            permanent = False
            if isinstance(exc, FileNotFoundError):
                error_class = "permanent_not_found"
                permanent = True
            elif isinstance(exc, ValueError):
                error_class = "permanent_validation"
                permanent = True

            try:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    UPDATE documents
                    SET embedding_status = 'error',
                        embedding_error = ?,
                        embedding_updated_at = NULL
                    WHERE id = ?
                    """,
                    (str(exc), job.entity_id),
                )
                completed = queue_engine.complete_failure(
                    job,
                    backend_name=self.embedding_provider.name,
                    backend_version=self.embedding_provider.version,
                    error_class=error_class,
                    error_message=str(exc),
                    permanent=permanent,
                    commit=False,
                )
                if not completed:
                    raise StaleClaimError(
                        f"Lost claim while recording failure for job {job.job_id}"
                    )
                conn.commit()
            except StaleClaimError as stale:
                conn.rollback()
                logger.warning("%s", stale)
            except Exception:  # noqa: BLE001
                conn.rollback()
                logger.exception(
                    "Failed to persist embedding failure state for job_id=%s",
                    job.job_id,
                )


class ExtractionJobHandler:
    queue_name = "extraction"
    worker_type = "extraction"
    idle_wait_seconds = 2.0

    def __init__(
        self,
        extraction_providers: list[ExtractionProvider] | None = None,
        storage_provider: StorageProvider | None = None,
        *,
        chunking_providers: list[ChunkingProvider] | None = None,
    ):
        if storage_provider is None:
            raise ValueError("storage_provider is required")
        self.storage_provider = storage_provider

        if not extraction_providers:
            raise ValueError("extraction_providers must contain at least one provider")
        self.extraction_providers = list(extraction_providers)

        if chunking_providers is None:
            chunking_providers = [PdfChunkingProvider()]
        if not chunking_providers:
            raise ValueError("chunking_providers must contain at least one provider")
        self.chunking_providers = list(chunking_providers)

        self.provider_name = self.extraction_providers[0].name
        self.provider_version = self.extraction_providers[0].version

    @staticmethod
    def _provider_supports_mime(provider: ExtractionProvider, normalized_mime: str) -> bool:
        for mime in provider.supported_mime_types:
            if canonicalize_mime_type(mime) == normalized_mime:
                return True
        return False

    def _normalize_mime_type(self, mime_type: str | None) -> str:
        normalized_mime = canonicalize_mime_type(mime_type)
        if not normalized_mime:
            raise ValueError("document_file missing mime_type")
        if not any(
            self._provider_supports_mime(provider, normalized_mime)
            for provider in self.extraction_providers
        ):
            raise SkippedExtractionError(
                f"Extraction provider does not support mime type: "
                f"{normalized_mime}"
            )
        return normalized_mime

    def _resolve_for_mime(self, normalized_mime: str) -> tuple[ExtractionProvider, ChunkingProvider]:
        extractor: ExtractionProvider | None = None
        for candidate in self.extraction_providers:
            if self._provider_supports_mime(candidate, normalized_mime):
                extractor = candidate
                break
        if extractor is None:
            raise SkippedExtractionError(
                f"Extraction provider does not support mime type: {normalized_mime}"
            )

        chunker: ChunkingProvider | None = None
        for candidate in self.chunking_providers:
            if canonicalize_mime_type(candidate.mime_type) == normalized_mime:
                chunker = candidate
                break
        if chunker is None:
            raise ProviderResolutionError(
                f"No chunking provider registered for mime type: {normalized_mime}"
            )
        return extractor, chunker

    def _extract(
        self,
        extractor: ExtractionProvider,
        locator: FileLocator,
        *,
        normalized_mime: str,
    ):
        try:
            return extractor.extract(
                locator,
                storage_provider=self.storage_provider,
                mime_type=normalized_mime,
            )
        except TypeError as exc:
            # Backward compatibility for tests and legacy extractors that do not
            # accept storage_provider yet.
            if "storage_provider" not in str(exc):
                raise
            return extractor.extract(locator, mime_type=normalized_mime)

    def handle_claimed_job(
        self,
        conn: sqlite3.Connection,
        queue_engine: QueueEngine,
        job: ClaimedJob,
        ctx: WorkerRuntimeContext,
    ) -> None:
        active_extractor: ExtractionProvider | None = None
        try:
            row = conn.execute(
                """
                SELECT id, reference_id, file_path, mime_type, storage_kind, storage_uri
                FROM document_files
                WHERE id = ?
                """,
                (job.entity_id,),
            ).fetchone()
            if not row:
                raise FileNotFoundError(f"document_file not found: {job.entity_id}")

            file_id = int(row[0])
            reference_id = int(row[1])
            mime_type = row[3]
            normalized_mime = self._normalize_mime_type(mime_type)
            locator = FileLocator(
                storage_kind=(row[4] or "local"),
                file_path=row[2],
                storage_uri=row[5],
            )
            active_extractor, chunker = self._resolve_for_mime(normalized_mime)
            result = self._extract(
                active_extractor,
                locator,
                normalized_mime=normalized_mime,
            )
            extracted_text = result.text or ""
            if not extracted_text.strip():
                raise ValueError("Extractor returned empty normalized text")

            chunk_artifacts = chunker.chunk(result)
            if not chunk_artifacts:
                raise ChunkingValidationError("Chunking provider produced no chunks")

            file_metadata: dict[str, object] = {
                "schema_version": 1,
                "origin": {"mime_type": normalized_mime},
            }
            if result.metadata:
                file_metadata["extraction"] = result.metadata
            file_metadata_json = json.dumps(file_metadata, separators=(",", ":"), ensure_ascii=False)

            conn.execute("BEGIN")
            conn.execute(
                """
                UPDATE document_files
                SET extracted_text = ?,
                    extraction_status = 'ready',
                    extraction_error = NULL,
                    extraction_backend = ?,
                    extraction_backend_version = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    extracted_text,
                    active_extractor.name,
                    active_extractor.version,
                    file_metadata_json,
                    file_id,
                ),
            )

            existing_chunk_ids = conn.execute(
                "SELECT id FROM documents WHERE document_file_id = ? AND kind = ?",
                (file_id, FULLTEXT_CHUNK_KIND),
            ).fetchall()
            for chunk_row in existing_chunk_ids:
                conn.execute("DELETE FROM vec_documents WHERE rowid = ?", (int(chunk_row[0]),))

            conn.execute(
                "DELETE FROM documents WHERE document_file_id = ? AND kind = ?",
                (file_id, FULLTEXT_CHUNK_KIND),
            )

            new_doc_ids: list[int] = []
            for idx, chunk_artifact in enumerate(chunk_artifacts):
                chunk_metadata: dict[str, object] = {
                    "schema_version": 1,
                    "origin": {"mime_type": normalized_mime},
                }
                if result.metadata:
                    chunk_metadata["extraction"] = result.metadata
                if chunk_artifact.page_start is not None or chunk_artifact.page_end is not None:
                    chunk_metadata["loc"] = {
                        "page_start": chunk_artifact.page_start,
                        "page_end": chunk_artifact.page_end,
                    }
                chunk_metadata_json = json.dumps(
                    chunk_metadata,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                cur = conn.execute(
                    """
                    INSERT INTO documents (
                        reference_id, document_file_id, kind, content,
                        chunk_index, metadata_json,
                        embedding_status, embedding_backend, embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL)
                    """,
                    (
                        reference_id,
                        file_id,
                        FULLTEXT_CHUNK_KIND,
                        chunk_artifact.content,
                        idx,
                        chunk_metadata_json,
                    ),
                )
                new_doc_ids.append(int(cur.lastrowid))

            for doc_id in new_doc_ids:
                enqueue_job(
                    conn,
                    queue_name="embedding",
                    job_type="embed_document",
                    entity_type="document",
                    entity_id=doc_id,
                    dedupe_key=f"embed_document:{doc_id}",
                    payload={"document_id": doc_id},
                    max_attempts=ctx.queue_policy.max_attempts,
                )

            completed = queue_engine.complete_success(
                job,
                backend_name=active_extractor.name,
                backend_version=active_extractor.version,
                metrics={"chunk_count": len(chunk_artifacts), "text_chars": len(extracted_text)},
                commit=False,
            )
            if not completed:
                raise StaleClaimError(f"Lost claim while completing job {job.job_id}")
            conn.commit()
        except SkippedExtractionError as exc:
            conn.rollback()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    UPDATE document_files
                    SET extraction_status = 'skipped',
                        extraction_error = ?,
                        extraction_backend = ?,
                        extraction_backend_version = ?
                    WHERE id = ?
                    """,
                    (
                        str(exc),
                        self.provider_name,
                        self.provider_version,
                        job.entity_id,
                    ),
                )
                completed = queue_engine.complete_success(
                    job,
                    backend_name=self.provider_name,
                    backend_version=self.provider_version,
                    metrics={"skipped": 1},
                    commit=False,
                )
                if not completed:
                    raise StaleClaimError(
                        f"Lost claim while marking skipped for job {job.job_id}"
                    )
                conn.commit()
            except StaleClaimError as stale:
                conn.rollback()
                logger.warning("%s", stale)
            except Exception:  # noqa: BLE001
                conn.rollback()
                logger.exception(
                    "Failed to persist extraction skipped state for job_id=%s",
                    job.job_id,
                )
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            if isinstance(exc, StaleClaimError):
                logger.warning("%s", exc)
                return
            if isinstance(exc, ChunkingValidationError):
                error_class = "permanent_validation"
            elif isinstance(exc, ProviderResolutionError):
                error_class = "permanent_unsupported"
            elif active_extractor is not None:
                error_class = active_extractor.classify_error(exc)
            else:
                if isinstance(exc, FileNotFoundError):
                    error_class = "permanent_not_found"
                elif isinstance(exc, ValueError):
                    error_class = "permanent_validation"
                else:
                    error_class = "retryable_io"
            permanent = _is_permanent(error_class)
            try:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    UPDATE document_files
                    SET extraction_status = 'error',
                        extraction_error = ?
                    WHERE id = ?
                    """,
                    (str(exc), job.entity_id),
                )
                completed = queue_engine.complete_failure(
                    job,
                    backend_name=self.provider_name,
                    backend_version=self.provider_version,
                    error_class=error_class,
                    error_message=str(exc),
                    permanent=permanent,
                    commit=False,
                )
                if not completed:
                    raise StaleClaimError(
                        f"Lost claim while recording failure for job {job.job_id}"
                    )
                conn.commit()
            except StaleClaimError as stale:
                conn.rollback()
                logger.warning("%s", stale)
            except Exception:  # noqa: BLE001
                conn.rollback()
                logger.exception(
                    "Failed to persist extraction failure state for job_id=%s",
                    job.job_id,
                )


__all__ = [
    "EmbeddingJobHandler",
    "ExtractionJobHandler",
    "QueueJobHandler",
    "QueueWorker",
    "QueuePolicy",
    "WorkerRuntimeContext",
]
