from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import aiohttp


class OpenAIResponsesProvider:
    """Minimal OpenAI client for teacher hints.

    Primary route uses POST /v1/responses (recommended for reasoning models).
    Fallback uses POST /v1/chat/completions when the Responses endpoint is unavailable
    for the selected model or account tier.

    Config knobs:
      - api_key_env: env var providing the API key (default: OPENAI_API_KEY)
      - api_base_env: optional base URL env var (default: OPENAI_BASE_URL)
      - model_id: OpenAI model identifier (e.g., 'o3-pro')
      - extra_body: optional overrides merged into the JSON payload
    """

    def __init__(
        self,
        model_id: str,
        api_key_env: str = "OPENAI_API_KEY",
        api_base_env: str = "OPENAI_BASE_URL",
        *,
        request_timeout: Optional[float] = 120.0,
        max_concurrency: int = 4,
        extra_body: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model_id = model_id
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key: set {api_key_env} in your environment.")
        self.api_key = api_key

        base_url = os.environ.get(api_base_env, "https://api.openai.com")
        self.responses_url = f"{base_url}/v1/responses"
        self.chat_url = f"{base_url}/v1/chat/completions"

        self._timeout = aiohttp.ClientTimeout(total=request_timeout) if request_timeout else None
        self._sem = asyncio.Semaphore(max_concurrency)
        self.extra_body = extra_body or {}
        self._last_stats: Dict[str, object] = {}
        self._last_response: Dict[str, object] | None = None

    def pop_stats(self) -> Dict[str, object]:
        stats = self._last_stats
        self._last_stats = {}
        return stats

    def pop_response_payload(self) -> Dict[str, object] | None:
        payload = self._last_response
        self._last_response = None
        return payload

    async def _responses_call(
        self, session: aiohttp.ClientSession, prompt: str, max_out: int
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_out,
        }
        if self.extra_body:
            payload.update(self.extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        async with session.post(self.responses_url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI Responses error {resp.status}: {text}")
            data = json.loads(text)
            self._last_response = data
            usage = data.get("usage", {})

            content = data.get("output_text")
            if not content:
                pieces: List[str] = []
                for item in data.get("output", []):
                    item_type = item.get("type")
                    if item_type == "message":
                        for inner in item.get("content", []):
                            inner_type = inner.get("type")
                            if inner_type in {"output_text", "summary_text", "input_text", "text"}:
                                txt = inner.get("text")
                                if isinstance(txt, str):
                                    pieces.append(txt)
                                elif isinstance(txt, dict):
                                    value = txt.get("value")
                                    if isinstance(value, str):
                                        pieces.append(value)
                    elif item_type == "reasoning":
                        for entry in item.get("summary", []) or []:
                            if isinstance(entry, dict):
                                txt = entry.get("text")
                                if isinstance(txt, str):
                                    pieces.append(txt)
                                elif isinstance(txt, dict):
                                    value = txt.get("value")
                                    if isinstance(value, str):
                                        pieces.append(value)
                content = "\n".join(pieces)

            self._last_stats = {
                "remote_tokens_in": usage.get("input_tokens"),
                "remote_tokens_out": usage.get("output_tokens"),
                "remote_reasoning_tokens": usage.get("reasoning_tokens"),
                "remote_time_ms": round((time.perf_counter() - start) * 1000, 1),
                "api_mode": "responses",
            }
            if not content:
                status = data.get("status")
                if status == "incomplete":
                    reason = ((data.get("incomplete_details") or {}).get("reason")) or "unknown"
                    raise RuntimeError(f"OpenAI Responses incomplete: {reason}")
            return content or ""

    async def _chat_call(
        self, session: aiohttp.ClientSession, prompt: str, max_out: int
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_out,
        }
        if self.extra_body:
            chat_safe = {k: v for k, v in self.extra_body.items() if k != "reasoning"}
            payload.update(chat_safe)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter()
        async with session.post(self.chat_url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI Chat error {resp.status}: {text}")
            data = json.loads(text)
            self._last_response = data
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            msg_content = message.get("content")
            if isinstance(msg_content, str):
                content = msg_content
            elif isinstance(msg_content, list):
                pieces = []
                for item in msg_content:
                    if isinstance(item, dict):
                        text_val = item.get("text")
                        if isinstance(text_val, str):
                            pieces.append(text_val)
                        elif isinstance(text_val, dict):
                            value = text_val.get("value")
                            if isinstance(value, str):
                                pieces.append(value)
                content = "\n".join(pieces)
            else:
                content = ""
            usage = data.get("usage", {})

            self._last_stats = {
                "remote_tokens_in": usage.get("prompt_tokens"),
                "remote_tokens_out": usage.get("completion_tokens"),
                "remote_reasoning_tokens": usage.get("reasoning_tokens"),
                "remote_time_ms": round((time.perf_counter() - start) * 1000, 1),
                "api_mode": "chat",
            }
            return content

    async def _generate_one(
        self, session: aiohttp.ClientSession, prompt: str, max_out: int
    ) -> str:
        try:
            return await self._responses_call(session, prompt, max_out)
        except Exception:
            return await self._chat_call(session, prompt, max_out)

    async def agenerate(self, prompts: Iterable[str], max_new_tokens: int) -> List[str]:
        prompt_list = list(prompts)

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async def _task(p: str) -> str:
                async with self._sem:
                    return await self._generate_one(session, p, max_new_tokens)

            tasks = [_task(p) for p in prompt_list]
            return await asyncio.gather(*tasks)
