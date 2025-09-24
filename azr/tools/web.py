from __future__ import annotations
import requests
from bs4 import BeautifulSoup
try:
    from readability import Document  # optional: improves text extraction
except Exception:
    Document = None
from dataclasses import dataclass

@dataclass
class WebSearchResult:
    title: str
    href: str
    snippet: str

class WebTool:
    """
    Provides web.search and web.fetch functionality for tool calls.
    """
    def __init__(self, max_bytes: int = 800000, user_agent: str = "AZR-Research/1.0 (+no-bots)", timeout_s: int = 15):
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.timeout_s = timeout_s

    def fetch(self, url: str) -> str:
        """
        Fetch a URL and return human-readable text, truncated to max_bytes.
        """
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
            html = data.decode("utf-8", errors="ignore")
        except Exception:
            return data[: self.max_bytes].decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        if Document is not None:
            try:
                doc = Document(html)
                text = BeautifulSoup(doc.summary(), "html.parser").get_text(" ", strip=True)
            except Exception:
                pass
        return text[: self.max_bytes]

    def search(self, query: str, count: int = 5) -> list[WebSearchResult]:
        """
        Perform a DuckDuckGo search and return up to 'count' results.
        """
        from duckduckgo_search import DDGS
        results: list[WebSearchResult] = []
        with DDGS() as ddgs:
            for res in ddgs.text(query, max_results=count):
                title = res.get("title", "")
                href = res.get("href", "")
                snippet = res.get("body", "")
                results.append(WebSearchResult(title=title, href=href, snippet=snippet))
        return results
