from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FileLocator:
    storage_kind: str
    file_path: str | None = None
    storage_uri: str | None = None


class StorageProvider(Protocol):
    def resolve(self, locator: FileLocator) -> Path:
        ...


class LocalFSProvider:
    def resolve(self, locator: FileLocator) -> Path:
        if locator.storage_kind != "local":
            raise ValueError(f"Unsupported storage kind: {locator.storage_kind}")
        target = locator.file_path or locator.storage_uri
        if not target:
            raise FileNotFoundError("No local file path present for locator")
        path = Path(target).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path


__all__ = ["FileLocator", "StorageProvider", "LocalFSProvider"]
