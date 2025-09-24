from __future__ import annotations

import requests
from typing import Optional


class WebTool:
    def __init__(self, max_bytes: int, user_agent: str, timeout_s: int) -> None:
        self.max_bytes = int(max_bytes)
        self.user_agent = user_agent
        self.timeout_s = timeout_s

    def fetch(self, url: str) -> str:
        headers = {"User-Agent": self.user_agent}
        with requests.get(url, headers=headers, timeout=self.timeout_s, stream=True) as r:
            r.raise_for_status()
            chunks = []
            total = 0
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    break
                total += len(chunk)
                if total > self.max_bytes:
                    chunks.append(chunk[: max(0, self.max_bytes - (total - len(chunk)))])
                    break
                chunks.append(chunk)
        data = b"".join(chunks)
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data[: self.max_bytes].decode("utf-8", errors="ignore")

