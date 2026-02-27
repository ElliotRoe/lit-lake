from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from litlake.pdf_runtime import configure_mupdf_output_suppression
from litlake.preview import PdfPreviewRenderer
from litlake.providers.extraction import LocalPdfExtractionProvider
from litlake.storage import FileLocator


def _run_with_fd1_capture(fn) -> str:
    sys.stdout.flush()
    original_fd = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            fn()
        finally:
            sys.stdout.flush()
            os.dup2(original_fd, 1)
            os.close(original_fd)
        tmp.seek(0)
        return tmp.read().decode("utf-8", errors="ignore")


class PdfStdoutSafetyTests(unittest.TestCase):
    def test_configure_mupdf_output_suppression_forces_display_off(self) -> None:
        import fitz

        fitz.TOOLS.mupdf_display_errors(True)
        fitz.TOOLS.mupdf_display_warnings(True)
        configure_mupdf_output_suppression()
        self.assertEqual(fitz.TOOLS.mupdf_display_errors(), 0)
        self.assertEqual(fitz.TOOLS.mupdf_display_warnings(), 0)

    def test_local_pdf_extraction_does_not_emit_mupdf_stdout(self) -> None:
        import fitz

        with tempfile.TemporaryDirectory() as tmp:
            bad_pdf = Path(tmp) / "bad.pdf"
            bad_pdf.write_bytes(b"not-a-pdf")

            provider = LocalPdfExtractionProvider()
            locator = FileLocator(
                storage_kind="local",
                file_path=str(bad_pdf),
                storage_uri=str(bad_pdf),
            )

            fitz.TOOLS.mupdf_display_errors(True)
            fitz.TOOLS.mupdf_display_warnings(True)

            def _extract() -> None:
                with self.assertRaises(Exception):
                    provider.extract(locator, mime_type="application/pdf")

            stdout = _run_with_fd1_capture(_extract)
            self.assertEqual(stdout.strip(), "")

    def test_pdf_preview_does_not_emit_mupdf_stdout(self) -> None:
        import fitz

        with tempfile.TemporaryDirectory() as tmp:
            bad_pdf = Path(tmp) / "bad.pdf"
            bad_pdf.write_bytes(b"not-a-pdf")

            fitz.TOOLS.mupdf_display_errors(True)
            fitz.TOOLS.mupdf_display_warnings(True)

            def _preview() -> None:
                with self.assertRaises(Exception):
                    PdfPreviewRenderer.page_range_png(str(bad_pdf), 1, 1, 512)

            stdout = _run_with_fd1_capture(_preview)
            self.assertEqual(stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
