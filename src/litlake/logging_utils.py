from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED_ATTR = "_litlake_logging_configured"
_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_litlake_logging(
    log_file: Path,
    *,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 10,
) -> Path:
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_ATTR, False):
        return log_file

    formatter = logging.Formatter(_DEFAULT_LOG_FORMAT)
    handlers: list[logging.Handler] = []

    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except OSError as exc:
        sys.stderr.write(f"Failed to open Lit Lake log file at {log_file}: {exc}\n")

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)
    handlers.append(stderr_handler)

    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    logging.captureWarnings(True)
    setattr(root, _CONFIGURED_ATTR, True)
    return log_file

