from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from litlake.storage import FileLocator

ErrorClass = Literal[
    "retryable_io",
    "retryable_rate_limit",
    "retryable_timeout",
    "permanent_validation",
    "permanent_not_found",
    "permanent_unsupported",
]


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    page_texts: list[str] | None = None


class ExtractionProvider(Protocol):
    name: str
    version: str

    def extract(self, locator: FileLocator) -> ExtractionResult:
        ...

    def classify_error(self, exc: Exception) -> ErrorClass:
        ...


@dataclass
class LocalPdfExtractionProvider:
    name: str = "local"
    version: str = "pypdf"

    def extract(self, locator: FileLocator) -> ExtractionResult:
        from pypdf import PdfReader

        if locator.storage_kind != "local":
            raise ValueError(f"Unsupported storage kind for local extraction: {locator.storage_kind}")
        source = locator.file_path or locator.storage_uri
        if not source:
            raise FileNotFoundError("No local file path provided")

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return ExtractionResult(text="\n\n".join(pages), page_texts=pages)

    def classify_error(self, exc: Exception) -> ErrorClass:
        try:
            import requests  # type: ignore
        except Exception:  # pragma: no cover
            requests = None

        if isinstance(exc, FileNotFoundError):
            return "permanent_not_found"
        if isinstance(exc, ValueError):
            return "permanent_validation"
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

    def __post_init__(self) -> None:
        import requests  # type: ignore

        self._session = requests.Session()

    def _detect_mime(self, path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/pdf"

    def _upload_file(self, path: Path) -> dict:
        mime_type = self._detect_mime(path)
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

    def extract(self, locator: FileLocator) -> ExtractionResult:
        source = locator.file_path or locator.storage_uri
        if not source:
            raise FileNotFoundError("No file path provided")
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_info = self._upload_file(path)
        file_name = file_info.get("name")
        file_uri = file_info.get("uri")
        try:
            prompt = (
                "Convert this PDF document to Markdown format. "
                "Preserve headings, lists, tables, and equations where possible. "
                "Output only markdown content."
            )
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"file_data": {"mime_type": "application/pdf", "file_uri": file_uri}},
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
            return ExtractionResult(text=text.strip())
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


@dataclass
class DoclingExtractionProvider:
    name: str = "docling"
    version: str = "docling"

    def __post_init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Docling provider requested but docling is not installed") from exc
        self._converter = DocumentConverter()

    def extract(self, locator: FileLocator) -> ExtractionResult:
        source = locator.file_path or locator.storage_uri
        if not source:
            raise FileNotFoundError("No file path provided")
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        result = self._converter.convert(str(path))
        doc = result.document
        markdown = doc.export_to_markdown()
        return ExtractionResult(text=markdown)

    def classify_error(self, exc: Exception) -> ErrorClass:
        if isinstance(exc, RuntimeError):
            return "permanent_unsupported"
        if isinstance(exc, FileNotFoundError):
            return "permanent_not_found"
        if isinstance(exc, ValueError):
            return "permanent_validation"
        return "retryable_io"


__all__ = [
    "ErrorClass",
    "ExtractionProvider",
    "ExtractionResult",
    "GeminiExtractionProvider",
    "LocalPdfExtractionProvider",
    "DoclingExtractionProvider",
]
