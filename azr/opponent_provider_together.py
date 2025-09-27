from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import aiohttp


class TogetherAIOpponentProvider:
    """Asynchronous Together AI chat-completions client used for opponent/teacher models."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        api_key_env: str,
        *,
        max_concurrency: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.95,
        request_timeout: Optional[float] = 120.0,
        extra_body: Optional[Dict[str, object]] = None,
        thinking_budget: Optional[int] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set {api_key_env} in your environment.")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self._max_concurrency = max_concurrency
        self._client_timeout = aiohttp.ClientTimeout(total=request_timeout) if request_timeout else None
        self.extra_body = extra_body or {}
        self.thinking_budget = thinking_budget
        self._last_stats: Dict[str, object] = {}

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": ["Final answer:", "{\"final_answer\":"],
        }
        if self.thinking_budget:
            # For models that support separate thinking budgets, request additional reasoning tokens.
            payload["max_thinking_tokens"] = self.thinking_budget
        if self.extra_body:
            # Shallow merge so config can override defaults when needed.
            payload.update(self.extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with semaphore:
            t0 = time.perf_counter()
            async with session.post(self.endpoint, headers=headers, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(
                        f"TogetherAI request failed with status {resp.status}: {text}"
                    )
                data = json.loads(text)
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("TogetherAI response contained no choices")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise RuntimeError("TogetherAI response missing message content")
                usage = data.get("usage", {})
                self._last_stats = {
                    "remote_tokens_in": usage.get("prompt_tokens"),
                    "remote_tokens_out": usage.get("completion_tokens"),
                    "remote_time_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
                return content

    async def agenerate(self, prompts: Iterable[str], max_new_tokens: int) -> List[str]:
        prompts = list(prompts)
        semaphore = asyncio.Semaphore(self._max_concurrency)
        async with aiohttp.ClientSession(timeout=self._client_timeout) as session:
            tasks = [
                self._generate_one(session, semaphore, prompt, max_new_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

    def pop_stats(self) -> Dict[str, object]:
        stats = self._last_stats
        self._last_stats = {}
        return stats

    async def close(self) -> None:  # pragma: no cover - compatibility shim
        return None


__all__ = ["TogetherAIOpponentProvider"]
