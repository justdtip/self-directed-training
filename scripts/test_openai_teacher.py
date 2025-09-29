#!/usr/bin/env python3
"""Quick script to sanity-check the OpenAI teacher provider.

Usage (after exporting OPENAI_API_KEY):

    python3 scripts/test_openai_teacher.py \
        --dataset azr/data/train.jsonl \
        --model gpt-5 \
        --effort high

It samples a random training problem, asks the OpenAI Responses API for a hint,
prints the hint, and displays token usage stats returned by the API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import List

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from azr.openai_provider import OpenAIResponsesProvider


def _load_examples(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_prompt(sample: dict) -> str:
    prompt = sample.get("prompt", "").strip()
    tests = sample.get("tests", []) or []
    tests_section = "\n".join(tests)
    return (
        "You are a mentor helping a junior engineer solve coding problems.\n"
        "Think privately if needed, but output only one concise hint (no full code).\n\n"
        f"Problem:\n{prompt}\n\n"
        f"Tests to satisfy:\n{tests_section}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenAI teacher hint generation")
    parser.add_argument(
        "--dataset",
        default="azr/data/train.jsonl",
        help="Path to the training data JSONL file",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model ID to query (e.g., gpt-5, o3-pro)",
    )
    parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort level (kept for compatibility; Responses currently ignore it)",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=768,
        help="Maximum tokens to request from the teacher model (default: 768)",
    )
    parser.add_argument(
        "--max_tokens",
        dest="legacy_max_tokens",
        type=int,
        default=None,
        help="Deprecated alias for --max-new-tokens",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    examples = _load_examples(dataset_path)
    if not examples:
        raise SystemExit("Dataset is empty; nothing to sample")

    sample = random.choice(examples)
    user_prompt = _build_prompt(sample)

    provider = OpenAIResponsesProvider(model_id=args.model)

    async def _run() -> None:
        max_tokens = args.max_new_tokens
        if args.legacy_max_tokens is not None:
            max_tokens = args.legacy_max_tokens
        hints = await provider.agenerate([user_prompt], max_tokens)
        hint = hints[0] if hints else ""
        print((hint or "").strip())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
