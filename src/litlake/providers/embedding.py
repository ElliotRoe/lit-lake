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


@dataclass
class FastEmbedEmbeddingProvider:
    models_path: Path
    model_name: str = "BAAI/bge-small-en-v1.5"
    name: str = "fastembed"
    version: str = "bge-small-en-v1.5"
    dim: int = 384

    def __post_init__(self) -> None:
        from fastembed import TextEmbedding

        self._lock = threading.Lock()
        self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.models_path))

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
