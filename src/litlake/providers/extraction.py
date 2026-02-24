from __future__ import annotations

import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, Protocol

from litlake.pdf_runtime import capture_mupdf_diagnostics
from litlake.storage import FileLocator

ErrorClass = Literal[
    "retryable_io",
    "retryable_rate_limit",
    "retryable_timeout",
    "permanent_validation",
    "permanent_not_found",
    "permanent_unsupported",
]

PDF_MIME_TYPE = "application/pdf"
HTML_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})
SUPPORTED_EXTRACTION_MIME_TYPES = frozenset({PDF_MIME_TYPE, *HTML_MIME_TYPES})
logger = logging.getLogger(__name__)


def canonicalize_mime_type(mime_type: str | None) -> str | None:
    normalized = (mime_type or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"application/x-pdf"}:
        return PDF_MIME_TYPE
    if normalized in HTML_MIME_TYPES:
        return "text/html"
    return normalized


def _extract_source(locator: FileLocator) -> str:
    source = locator.file_path or locator.storage_uri
    if not source:
        raise FileNotFoundError("No file path provided")
    return source


def _resolve_path(locator: FileLocator) -> Path:
    source = _extract_source(locator)
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _normalize_mime_type(mime_type: str | None, path: Path) -> str:
    normalized = canonicalize_mime_type(mime_type) or ""
    if not normalized:
        guessed, _ = mimetypes.guess_type(path.name)
        normalized = canonicalize_mime_type(guessed) or ""
    if normalized == PDF_MIME_TYPE:
        return PDF_MIME_TYPE
    if normalized == "text/html":
        return "text/html"
    raise ValueError(f"Unsupported mime type for extraction: {normalized or 'unknown'}")


def _looks_like_full_html_document(html: str) -> bool:
    snippet = html.lstrip().lower()
    if not snippet:
        return False
    if snippet.startswith("<!doctype html") or snippet.startswith("<html") or snippet.startswith("<?xml"):
        return True
    head = snippet[:2048]
    return "<html" in head and ("<body" in head or "</html>" in head)


class _HTMLTextFallbackParser(HTMLParser):
    _BREAK_TAGS = {
        "p",
        "br",
        "div",
        "li",
        "ul",
        "ol",
        "section",
        "article",
        "header",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "td",
        "th",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BREAK_TAGS:
            self.parts.append("\n")

    def as_text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def extract_html_text(html: str, *, require_trafilatura: bool = False) -> ExtractionResult:
    should_try_trafilatura = require_trafilatura or _looks_like_full_html_document(html)
    trafilatura = None
    if should_try_trafilatura:
        try:
            import trafilatura as _trafilatura
        except Exception as exc:  # pragma: no cover - dependency issue
            if require_trafilatura:
                raise RuntimeError("trafilatura is required for HTML extraction") from exc
        else:
            trafilatura = _trafilatura

    if trafilatura is not None:
        primary = trafilatura.extract(html, output_format="markdown")
        if primary and primary.strip():
            return ExtractionResult(
                text=primary.strip(),
                metadata={"mode": "trafilatura_markdown"},
            )

        fallback = trafilatura.extract(html, output_format="txt", favor_recall=True)
        if fallback and fallback.strip():
            return ExtractionResult(
                text=fallback.strip(),
                metadata={"mode": "trafilatura_recall"},
            )

    parser = _HTMLTextFallbackParser()
    parser.feed(html)
    plain = parser.as_text()
    if not plain:
        return ExtractionResult(text="", metadata={"mode": "empty"})
    return ExtractionResult(text=plain, metadata={"mode": "html_fallback"})


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    page_texts: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ExtractionProvider(Protocol):
    name: str
    version: str
    supported_mime_types: frozenset[str]

    def extract(self, locator: FileLocator, *, mime_type: str | None = None) -> ExtractionResult:
        ...

    def classify_error(self, exc: Exception) -> ErrorClass:
        ...


@dataclass
class LocalFileExtractionProvider:
    name: str = "local"
    version: str = "pymupdf+trafilatura"
    supported_mime_types: frozenset[str] = SUPPORTED_EXTRACTION_MIME_TYPES

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        import fitz  # pymupdf

        pages: list[str] = []
        with capture_mupdf_diagnostics(f"extract:{path}", log=logger):
            doc = fitz.open(str(path))
            try:
                for page in doc:
                    text = page.get_text("text") or ""
                    pages.append(text)
            finally:
                doc.close()

        return ExtractionResult(text="\n\n".join(pages), page_texts=pages)

    def _extract_html(self, path: Path) -> ExtractionResult:
        html = path.read_text(encoding="utf-8", errors="ignore")
        return extract_html_text(html, require_trafilatura=True)

    def extract(self, locator: FileLocator, *, mime_type: str | None = None) -> ExtractionResult:
        if locator.storage_kind != "local":
            raise ValueError(f"Unsupported storage kind for local extraction: {locator.storage_kind}")
        path = _resolve_path(locator)
        normalized_mime = _normalize_mime_type(mime_type, path)
        if normalized_mime == PDF_MIME_TYPE:
            return self._extract_pdf(path)
        if normalized_mime in HTML_MIME_TYPES:
            return self._extract_html(path)
        raise ValueError(f"Unsupported mime type for local extraction: {normalized_mime}")

    def classify_error(self, exc: Exception) -> ErrorClass:
        try:
            import requests  # type: ignore
        except Exception:  # pragma: no cover
            requests = None

        if isinstance(exc, FileNotFoundError):
            return "permanent_not_found"
        if isinstance(exc, (ValueError, UnicodeDecodeError)):
            return "permanent_validation"
        if isinstance(exc, RuntimeError) and "trafilatura" in str(exc).lower():
            return "permanent_unsupported"
        if requests is not None and isinstance(exc, requests.Timeout):
            return "retryable_timeout"
        if requests is not None and isinstance(exc, requests.RequestException):
            return "retryable_io"
        return "retryable_io"


@dataclass
class GeminiExtractionProvider:
    api_key: str
    model: str = "gemini-3-flash-preview"
    timeout_seconds: int = 300
    name: str = "gemini"
    version: str = "gemini-3-flash-preview"
    supported_mime_types: frozenset[str] = frozenset({PDF_MIME_TYPE})

    def __post_init__(self) -> None:
        import requests  # type: ignore

        self._session = requests.Session()

    def _detect_mime(self, path: Path, mime_type: str | None = None) -> str:
        return _normalize_mime_type(mime_type, path)

    def _upload_file(self, path: Path, mime_type: str) -> dict:
        size = path.stat().st_size

        start_res = self._session.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers={
                "x-goog-api-key": self.api_key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": path.name}},
            timeout=self.timeout_seconds,
        )
        start_res.raise_for_status()

        upload_url = start_res.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("Gemini upload missing x-goog-upload-url")

        with path.open("rb") as f:
            finalize = self._session.post(
                upload_url,
                headers={
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                data=f,
                timeout=self.timeout_seconds,
            )
        finalize.raise_for_status()
        payload = finalize.json()
        return payload["file"]

    def _delete_file(self, file_name: str) -> None:
        res = self._session.delete(
            f"https://generativelanguage.googleapis.com/v1beta/{file_name}",
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout_seconds,
        )
        if not res.ok:
            return

    def extract(self, locator: FileLocator, *, mime_type: str | None = None) -> ExtractionResult:
        path = _resolve_path(locator)
        normalized_mime = self._detect_mime(path, mime_type=mime_type)
        if normalized_mime not in self.supported_mime_types:
            raise ValueError(f"Unsupported mime type for gemini extraction: {normalized_mime}")

        file_info = self._upload_file(path, normalized_mime)
        file_name = file_info.get("name")
        file_uri = file_info.get("uri")
        try:
            if normalized_mime == PDF_MIME_TYPE:
                prompt = (
                    "Convert this PDF document to Markdown format. "
                    "Preserve headings, lists, tables, and equations where possible. "
                    "Output only markdown content."
                )
            else:
                prompt = (
                    "Extract the main textual content from this HTML snapshot and convert it to Markdown. "
                    "Preserve headings, lists, and tables where possible. Output only markdown content."
                )
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"file_data": {"mime_type": normalized_mime, "file_uri": file_uri}},
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 1.0,
                },
            }
            res = self._session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                data=json.dumps(body),
                timeout=self.timeout_seconds,
            )
            res.raise_for_status()
            payload = res.json()
            candidates = payload.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {payload}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))
            return ExtractionResult(
                text=text.strip(),
                metadata={"mode": "gemini", "input_mime_type": normalized_mime},
            )
        finally:
            if file_name:
                self._delete_file(file_name)

    def classify_error(self, exc: Exception) -> ErrorClass:
        import requests  # type: ignore

        if isinstance(exc, FileNotFoundError):
            return "permanent_not_found"
        if isinstance(exc, ValueError):
            return "permanent_validation"
        if isinstance(exc, requests.Timeout):
            return "retryable_timeout"
        if isinstance(exc, requests.HTTPError):
            response = exc.response
            if response is not None and response.status_code == 429:
                return "retryable_rate_limit"
            if response is not None and 400 <= response.status_code < 500:
                return "permanent_validation"
            return "retryable_io"
        if isinstance(exc, requests.RequestException):
            return "retryable_io"
        return "retryable_io"

# Backward-compatible alias.
LocalPdfExtractionProvider = LocalFileExtractionProvider


__all__ = [
    "canonicalize_mime_type",
    "ErrorClass",
    "ExtractionProvider",
    "ExtractionResult",
    "extract_html_text",
    "HTML_MIME_TYPES",
    "GeminiExtractionProvider",
    "LocalFileExtractionProvider",
    "LocalPdfExtractionProvider",
    "PDF_MIME_TYPE",
    "SUPPORTED_EXTRACTION_MIME_TYPES",
]
