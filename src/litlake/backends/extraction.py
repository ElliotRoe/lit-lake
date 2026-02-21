from litlake.providers.extraction import (
    DoclingExtractionProvider,
    ErrorClass,
    ExtractionProvider,
    ExtractionResult,
    GeminiExtractionProvider,
    LocalPdfExtractionProvider,
)

# Backward-compatible aliases for older imports.
ExtractionBackend = ExtractionProvider
LocalPdfExtractionBackend = LocalPdfExtractionProvider
GeminiExtractionBackend = GeminiExtractionProvider
DoclingExtractionBackend = DoclingExtractionProvider

__all__ = [
    "DoclingExtractionBackend",
    "DoclingExtractionProvider",
    "ErrorClass",
    "ExtractionBackend",
    "ExtractionProvider",
    "ExtractionResult",
    "GeminiExtractionBackend",
    "GeminiExtractionProvider",
    "LocalPdfExtractionBackend",
    "LocalPdfExtractionProvider",
]
