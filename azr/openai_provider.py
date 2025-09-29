from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence

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

        self._timeout = aiohttp.ClientTimeout(total=request_timeout) if request_timeout else None
        self._sem = asyncio.Semaphore(max_concurrency)
        self.extra_body = extra_body or {}
        self._last_stats: Dict[str, object] = {}
        self._last_response: Dict[str, object] | None = None
        self.accepts_structured_prompts = True
        self.default_max_output_tokens = 768

    def pop_stats(self) -> Dict[str, object]:
        stats = self._last_stats
        self._last_stats = {}
        return stats

    def pop_response_payload(self) -> Dict[str, object] | None:
        payload = self._last_response
        self._last_response = None
        return payload

    @staticmethod
    def _ensure_input_text(content: object) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        if isinstance(content, list):
            for entry in content:
                if isinstance(entry, dict):
                    text_val = entry.get("text") or entry.get("value")
                    if isinstance(text_val, str):
                        items.append({"type": "input_text", "text": text_val})
                elif isinstance(entry, str):
                    items.append({"type": "input_text", "text": entry})
        elif isinstance(content, str):
            items.append({"type": "input_text", "text": content})
        elif content is not None:
            items.append({"type": "input_text", "text": str(content)})
        if not items:
            items.append({"type": "input_text", "text": ""})
        return items

    @classmethod
    def _normalize_responses_input(cls, prompt: object) -> Sequence[Dict[str, object]]:
        if isinstance(prompt, Sequence) and prompt and isinstance(prompt[0], dict):
            normalized: List[Dict[str, object]] = []
            for message in prompt:  # type: ignore[assignment]
                role = str(message.get("role", "user"))
                content = message.get("content", [])
                normalized.append({
                    "role": role,
                    "content": cls._ensure_input_text(content),
                })
            return normalized
        text = str(prompt)
        return [{"role": "user", "content": cls._ensure_input_text(text)}]

    async def _responses_call(
        self, session: aiohttp.ClientSession, prompt: object, max_out: int
    ) -> str:
        inputs = list(self._normalize_responses_input(prompt))
        max_tokens = int(max_out) if max_out else self.default_max_output_tokens
        if max_tokens <= 0:
            max_tokens = self.default_max_output_tokens
        payload: Dict[str, object] = {
            "model": self.model_id,
            "input": inputs,
            "max_output_tokens": max_tokens,
        }
        if self.extra_body:
            # Filter out unsupported keys (e.g., legacy "reasoning")
            safe_extra = {k: v for k, v in self.extra_body.items() if k not in {"reasoning"}}
            payload.update(safe_extra)

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

    async def _generate_one(
        self, session: aiohttp.ClientSession, prompt: object, max_out: int
    ) -> str:
        return await self._responses_call(session, prompt, max_out)

    async def agenerate(self, prompts: Iterable[object], max_new_tokens: int) -> List[str]:
        prompt_list = list(prompts)

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async def _task(p: object) -> str:
                async with self._sem:
                    return await self._generate_one(session, p, max_new_tokens)

            tasks = [_task(p) for p in prompt_list]
            return await asyncio.gather(*tasks)
