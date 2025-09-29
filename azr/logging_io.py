"""Lightweight helpers for durable logging writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def append_jsonl(path: str | os.PathLike, record: dict) -> None:
    """Append a JSON object as one line, ensuring the directory exists."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: str | os.PathLike, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(target.parent), encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    """Atomically write plain text."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(target.parent), encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        if not text.endswith("\n"):
            tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, target)


__all__ = ["append_jsonl", "atomic_write_json", "atomic_write_text"]

