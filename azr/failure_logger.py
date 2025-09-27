from __future__ import annotations

import hashlib
import json
import queue
import threading
from pathlib import Path
from typing import Dict, Optional


class FailureLogger:
    """Asynchronous frontier logger for 'neither' and 'opponent_only' prompts."""

    def __init__(self, out_dir: str, *, flush_every: int = 100) -> None:
        frontier_dir = Path(out_dir) / "frontier"
        frontier_dir.mkdir(parents=True, exist_ok=True)

        self._neither_path = frontier_dir / "neither.jsonl"
        self._opp_path = frontier_dir / "opponent_only.jsonl"
        self._flush_every = flush_every
        self._queue: "queue.Queue[Dict]" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._seen = set()  # dedupe by prompt hash + kind

        self._fp_neither = self._neither_path.open("a", encoding="utf-8")
        self._fp_opp = self._opp_path.open("a", encoding="utf-8")

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def log(self, record: Dict) -> None:
        prompt = record.get("prompt")
        if not prompt:
            return
        kind = record.get("kind", "neither")
        key = hashlib.sha256(f"{kind}::{prompt}".encode("utf-8")).hexdigest()
        if key in self._seen:
            return
        self._seen.add(key)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # drop rather than block training

    def _worker(self) -> None:
        count = 0
        while not self._stop.is_set() or not self._queue.empty():
            try:
                rec = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if rec.get("kind") == "opponent_only":
                self._fp_opp.write(line)
            else:
                self._fp_neither.write(line)
            count += 1

            if count % self._flush_every == 0:
                self._fp_neither.flush()
                self._fp_opp.flush()

    def close(self) -> None:
        self._stop.set()
        self._worker_thread.join(timeout=5.0)
        try:
            self._fp_neither.flush()
            self._fp_neither.close()
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            self._fp_opp.flush()
            self._fp_opp.close()
        except Exception:  # pragma: no cover - best effort
            pass


__all__ = ["FailureLogger"]
