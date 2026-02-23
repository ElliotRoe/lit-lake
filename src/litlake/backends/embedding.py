from litlake.providers.embedding import (
    EmbeddingProvider,
    FastEmbedEmbeddingProvider,
    FastEmbedRerankProvider,
    RerankProvider,
)

# Backward-compatible aliases for older imports.
EmbeddingBackend = EmbeddingProvider
RerankBackend = RerankProvider
FastEmbedEmbeddingBackend = FastEmbedEmbeddingProvider
FastEmbedRerankBackend = FastEmbedRerankProvider

__all__ = [
    "EmbeddingBackend",
    "EmbeddingProvider",
    "FastEmbedEmbeddingBackend",
    "FastEmbedEmbeddingProvider",
    "FastEmbedRerankBackend",
    "FastEmbedRerankProvider",
    "RerankBackend",
    "RerankProvider",
]
