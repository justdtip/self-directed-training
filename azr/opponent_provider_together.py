import os
import asyncio
from typing import Iterable, List, Optional

import aiohttp


class TogetherAIOpponentProvider:
    """Async opponent provider backed by Together AI chat completions API."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        api_key_env: str,
        *,
        max_concurrency: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        request_timeout: Optional[float] = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set {api_key_env} in your environment.")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client_timeout = aiohttp.ClientTimeout(total=request_timeout) if request_timeout else None
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._client_timeout)
        return self._session

    async def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        session = await self._ensure_session()
        async with self._semaphore:
            async with session.post(self.endpoint, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"TogetherAI request failed with status {resp.status}: {text}")
                data = await resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("TogetherAI response contained no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise RuntimeError("TogetherAI response missing message content")
                return content

    async def agenerate(self, prompts: Iterable[str], max_new_tokens: int) -> List[str]:
        tasks = [self._generate_one(prompt, max_new_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


__all__ = ["TogetherAIOpponentProvider"]
