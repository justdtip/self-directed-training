"""Web-facing tools for the AZR agent."""

from __future__ import annotations

from typing import Dict, List

import requests
from bs4 import BeautifulSoup

try:
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

from duckduckgo_search import DDGS

UA = "AZR-Research/1.0 (+no-bots)"


def _clamp_count(count: int) -> int:
    return max(1, min(10, int(count)))


def search_ddg(query: str, count: int = 5) -> Dict[str, object]:
    """Perform a DuckDuckGo search and return structured results."""

    count = _clamp_count(count)
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=count) or []:
            results.append(
                {
                    "title": item.get("title", ""),
                    "href": item.get("href", ""),
                    "body": item.get("body", ""),
                }
            )
    return {"query": query, "results": results}


def fetch_url(url: str, max_bytes: int = 800_000, timeout: int = 15) -> Dict[str, object]:
    """Fetch a URL and return human-readable text."""

    max_bytes = int(max(1024, max_bytes))
    response = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, stream=False)
    response.raise_for_status()
    content = response.content[:max_bytes]

    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(" ", strip=True)

    if Document is not None and content:
        try:
            doc = Document(content)
            summary_html = doc.summary() or ""
            text = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True) or text
        except Exception:
            pass

    return {"url": url, "status": response.status_code, "text": text[:max_bytes]}


__all__ = ["search_ddg", "fetch_url"]
