from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class EmbeddingProvider(Protocol):
    name: str
    version: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class RerankProvider(Protocol):
    name: str
    version: str

    def score(self, query: str, doc: str) -> float:
        ...


# Known embedding model dimensions for vec_documents table creation
MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
}


@dataclass
class FastEmbedEmbeddingProvider:
    models_path: Path
    model_name: str = "BAAI/bge-small-en-v1.5"
    name: str = "fastembed"
    version: str = ""
    dim: int = 0

    def __post_init__(self) -> None:
        from fastembed import TextEmbedding

        if not self.version:
            self.version = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        self._lock = threading.Lock()
        self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.models_path))

        # Auto-detect dimension if not set
        if self.dim == 0:
            if self.model_name in MODEL_DIMENSIONS:
                self.dim = MODEL_DIMENSIONS[self.model_name]
            else:
                # Probe the model with a test embedding
                test_vec = list(self._model.embed(["test"]))[0]
                self.dim = len(test_vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            vectors = list(self._model.embed(texts))
        return [list(v) for v in vectors]


@dataclass
class FastEmbedRerankProvider:
    model_name: str = "BAAI/bge-reranker-base"
    name: str = "fastembed"
    version: str = "bge-reranker-base"

    def __post_init__(self) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._lock = threading.Lock()
        self._encoder = TextCrossEncoder(model_name=self.model_name)

    def score(self, query: str, doc: str) -> float:
        with self._lock:
            scores = list(self._encoder.rerank_pairs([(query, doc)]))
        return float(scores[0]) if scores else 0.0


__all__ = [
    "EmbeddingProvider",
    "FastEmbedEmbeddingProvider",
    "FastEmbedRerankProvider",
    "RerankProvider",
]
