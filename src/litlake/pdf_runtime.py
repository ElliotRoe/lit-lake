from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_fitz_configured = False
_fitz_config_lock = threading.Lock()
_mupdf_capture_lock = threading.Lock()


def configure_mupdf_output_suppression() -> None:
    global _fitz_configured
    with _fitz_config_lock:
        import fitz  # pymupdf

        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
        _fitz_configured = True


@contextmanager
def capture_mupdf_diagnostics(context: str, *, log: logging.Logger | None = None) -> Iterator[None]:
    target_logger = log or logger
    with _mupdf_capture_lock:
        configure_mupdf_output_suppression()
        import fitz  # pymupdf

        fitz.TOOLS.reset_mupdf_warnings()
        try:
            yield
        finally:
            try:
                warnings = fitz.TOOLS.mupdf_warnings(reset=1) or ""
            except Exception as exc:  # pragma: no cover - diagnostic only
                target_logger.debug("Failed to read MuPDF diagnostics (%s): %s", context, exc)
                return

            if warnings.strip():
                compact = " | ".join(line.strip() for line in warnings.splitlines() if line.strip())
                target_logger.warning("MuPDF diagnostics (%s): %s", context, compact)
