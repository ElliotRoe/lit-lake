from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PdfPreviewPage:
    page_number: int
    png_bytes: bytes


@dataclass
class PdfPreviewBatch:
    pages: list[PdfPreviewPage]


class PdfPreviewRenderer:
    @staticmethod
    def page_range_png(pdf_path: str, start_page: int, end_page: int, size: int) -> PdfPreviewBatch:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        page_count = doc.page_count

        if start_page <= 0 or end_page <= 0:
            raise ValueError("Pages are 1-based; start_page/end_page must be >= 1")
        if start_page > end_page:
            raise ValueError("start_page must be <= end_page")
        if end_page > page_count:
            raise ValueError(f"Page out of bounds: PDF has {page_count} pages but end_page={end_page}")

        pages: list[PdfPreviewPage] = []
        for page_number in range(start_page, end_page + 1):
            page = doc.load_page(page_number - 1)
            rect = page.rect
            max_dim = max(rect.width, rect.height)
            scale = max(0.01, float(size) / max(1.0, max_dim))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pages.append(PdfPreviewPage(page_number=page_number, png_bytes=pix.tobytes("png")))

        doc.close()
        return PdfPreviewBatch(pages=pages)


__all__ = ["PdfPreviewPage", "PdfPreviewBatch", "PdfPreviewRenderer"]
