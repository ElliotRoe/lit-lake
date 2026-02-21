from litlake.providers.extraction import (
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

__all__ = [
    "ErrorClass",
    "ExtractionBackend",
    "ExtractionProvider",
    "ExtractionResult",
    "GeminiExtractionBackend",
    "GeminiExtractionProvider",
    "LocalPdfExtractionBackend",
    "LocalPdfExtractionProvider",
]
