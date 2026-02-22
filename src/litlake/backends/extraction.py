from litlake.providers.extraction import (
    ErrorClass,
    ExtractionProvider,
    ExtractionResult,
    GeminiExtractionProvider,
    LocalFileExtractionProvider,
)

# Backward-compatible aliases for older imports.
ExtractionBackend = ExtractionProvider
LocalFileExtractionBackend = LocalFileExtractionProvider
LocalPdfExtractionBackend = LocalFileExtractionProvider
GeminiExtractionBackend = GeminiExtractionProvider

__all__ = [
    "ErrorClass",
    "ExtractionBackend",
    "ExtractionProvider",
    "ExtractionResult",
    "GeminiExtractionBackend",
    "GeminiExtractionProvider",
    "LocalFileExtractionBackend",
    "LocalFileExtractionProvider",
    "LocalPdfExtractionBackend",
]
