from __future__ import annotations

import hashlib
import os
import socket
from dataclasses import dataclass
from pathlib import Path


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    if "${" in value:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool = False) -> bool:
    value = _clean_env(os.getenv(name))
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = _clean_env(os.getenv(name))
    if value is None:
        out = default
    else:
        try:
            out = int(value)
        except ValueError:
            out = default
    if minimum is not None:
        out = max(minimum, out)
    return out


@dataclass(frozen=True)
class Paths:
    root: Path
    db_path: Path
    models_path: Path


@dataclass(frozen=True)
class Settings:
    paths: Paths
    worker_id: str
    embed_disabled: bool
    queue_lease_seconds: int
    queue_batch_size: int
    queue_max_attempts: int
    queue_backoff_base_seconds: int
    queue_backoff_max_seconds: int
    gemini_api_key: str | None
    zotero_db_path: str | None

    @property
    def queue_config_hash(self) -> str:
        payload = "|".join(
            [
                str(self.queue_lease_seconds),
                str(self.queue_batch_size),
                str(self.queue_max_attempts),
                str(self.queue_backoff_base_seconds),
                str(self.queue_backoff_max_seconds),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]



def resolve_paths() -> Paths:
    raw_path = _clean_env(os.getenv("LIT_LAKE_PATH"))
    if raw_path:
        base_dir = Path(raw_path).expanduser().resolve()
        root = base_dir if base_dir.name == "LitLake" else (base_dir / "LitLake")
    else:
        root = (Path.home() / "LitLake").resolve()

    models = root / "models"
    root.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, db_path=root / "lit_lake.db", models_path=models)



def load_settings() -> Settings:
    paths = resolve_paths()
    worker_id = _clean_env(os.getenv("WORKER_ID")) or f"{socket.gethostname()}-lit-lake"

    return Settings(
        paths=paths,
        worker_id=worker_id,
        embed_disabled=_env_bool("EMBED_DISABLED", default=False),
        queue_lease_seconds=_env_int("QUEUE_LEASE_SECONDS", default=180, minimum=15),
        queue_batch_size=_env_int("QUEUE_BATCH_SIZE", default=32, minimum=1),
        queue_max_attempts=_env_int("QUEUE_MAX_ATTEMPTS", default=5, minimum=1),
        queue_backoff_base_seconds=_env_int("QUEUE_BACKOFF_BASE_SECONDS", default=5, minimum=1),
        queue_backoff_max_seconds=_env_int("QUEUE_BACKOFF_MAX_SECONDS", default=900, minimum=5),
        gemini_api_key=_clean_env(os.getenv("GEMINI_API_KEY")),
        zotero_db_path=_clean_env(os.getenv("ZOTERO_DB_PATH")),
    )
