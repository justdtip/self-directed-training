import gzip
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional


class GenerationLogger:
    """Background JSONL writer for prompt/completion telemetry."""

    def __init__(
        self,
        out_dir: str,
        fname_prefix: str = "generations",
        rotate_mb: int = 128,
        gzip_output: bool = False,
        flush_every: int = 100,
        redact_prompt: bool = False,
        max_bytes: Optional[int] = None,
        pretty_print: bool = False,
        write_text_log: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = fname_prefix
        self.rotate_bytes = rotate_mb * 1024 * 1024
        self.gzip = gzip_output
        self.flush_every = flush_every
        self.redact_prompt = redact_prompt
        self.max_bytes = max_bytes
        self.pretty_print = pretty_print
        self.write_text_log = write_text_log

        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._count = 0
        self._written = 0
        self._part = 0
        self._fp: Optional[object] = None
        self._text_fp: Optional[object] = None
        self._open_new()

        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def _open_new(self) -> None:
        if self._fp:
            try:
                self._fp.close()
            except Exception:
                pass
        if self._text_fp:
            try:
                self._text_fp.close()
            except Exception:
                pass
            self._text_fp = None
        self._part += 1
        name = f"{self.prefix}.{self._part:04d}.jsonl"
        path = self.out_dir / (name + (".gz" if self.gzip else ""))
        if self.gzip:
            self._fp = gzip.open(path, "at", encoding="utf-8")
        else:
            self._fp = open(path, "a", encoding="utf-8")
        self._written = 0
        if self.write_text_log:
            text_name = f"{self.prefix}.{self._part:04d}.txt"
            text_path = self.out_dir / text_name
            self._text_fp = open(text_path, "a", encoding="utf-8")

    def _truncate(self, text: str) -> str:
        if text is None or self.max_bytes is None:
            return text
        data = text.encode("utf-8")
        if len(data) <= self.max_bytes:
            return text
        return data[: self.max_bytes].decode("utf-8", errors="ignore")

    def log(self, rec: dict) -> None:
        rec = dict(rec)  # make a shallow copy
        if self.redact_prompt and "prompt" in rec:
            prompt = str(rec["prompt"])
            rec["prompt_hash"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            rec["prompt"] = prompt[:200]
        for key in ("prompt", "completion", "stdout", "stderr"):
            if key in rec and isinstance(rec[key], str):
                rec[key] = self._truncate(rec[key])
        try:
            self.q.put_nowait(rec)
        except queue.Full:
            # Drop instead of blocking the hot path
            pass

    def _worker(self) -> None:
        if self._fp is None:
            return
        last_flush = time.time()
        while not self._stop.is_set() or not self.q.empty():
            try:
                rec = self.q.get(timeout=0.1)
            except queue.Empty:
                rec = None
            if rec is not None:
                if self.pretty_print:
                    line = json.dumps(rec, ensure_ascii=False, indent=2)
                    self._fp.write(line + "\n\n")
                else:
                    line = json.dumps(rec, ensure_ascii=False)
                    self._fp.write(line + "\n")
                if self.write_text_log and self._text_fp is not None:
                    self._text_fp.write(self._format_text_record(rec))
                    self._text_fp.write("\n")
                self._written += len(line) + 1
                self._count += 1
                if self._written >= self.rotate_bytes:
                    self._open_new()
            now = time.time()
            if rec is None and (now - last_flush) < 1.0:
                continue
            if self._count % self.flush_every == 0 or (now - last_flush) > 1.0:
                try:
                    self._fp.flush()
                except Exception:
                    pass
                last_flush = now

    def close(self) -> None:
        self._stop.set()
        self._thr.join(timeout=5.0)
        if self._fp:
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                pass
        if self._text_fp:
            try:
                self._text_fp.flush()
                self._text_fp.close()
            except Exception:
                pass

    def _format_text_record(self, rec: dict) -> str:
        lines: list[str] = []
        meta_fields = [
            "ts",
            "source",
            "step",
            "score",
            "passed",
            "length_tokens",
        ]
        for field in meta_fields:
            if field in rec:
                lines.append(f"{field}: {rec[field]}")
        if "metadata" in rec:
            try:
                meta_json = json.dumps(rec["metadata"], ensure_ascii=False, indent=2)
            except Exception:
                meta_json = str(rec["metadata"])
            lines.append("metadata:")
            lines.append(meta_json)
        text_fields = [
            "prompt",
            "user_prompt",
            "hint",
            "completion",
            "retry",
            "retry_prompt",
            "opponent_completion",
            "stdout",
            "stderr",
        ]
        for field in text_fields:
            value = rec.get(field)
            if value:
                lines.append(f"--- {field} ---")
                lines.append(str(value))
        if "assist_messages" in rec:
            try:
                msg_dump = json.dumps(rec["assist_messages"], ensure_ascii=False, indent=2)
            except Exception:
                msg_dump = str(rec["assist_messages"])
            lines.append("assist_messages:")
            lines.append(msg_dump)
        return "\n".join(lines)


__all__ = ["GenerationLogger"]
