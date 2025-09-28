#!/usr/bin/env python3
"""Sample an opponent completion using the current self-play configuration.

Usage (ensure TOGETHER_API_KEY or other provider credentials are exported):

    python3 scripts/test_opponent_responses.py \
        --config azr/config.json \
        --dataset azr/data/train.jsonl

The script mirrors the training setup:
  * Renders prompts with the same chat template (including thinking flag).
  * Instantiates SelfPlayManager so the opponent provider is configured identically.
  * Requests a single completion using the training max completion length (+opponent bonus).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azr.config import AzrModelCfg, AzrSelfPlayCfg, load_config
from azr.modeling import load_tokenizer
from azr.selfplay_manager import SelfPlayManager


SYS_PROMPT = (
    "You are a careful engineer.\n"
    "You may think in a private scratchpad enclosed by <think>...</think>.\n"
    "Then provide the minimal correct solution.\n"
    "If coding helps, include ONE final Python code block implementing the solution.\n"
    "Terminate with exactly one of the following lines and STOP:\n"
    "  Final answer: <short text>\n"
    "  {\"final_answer\": \"<short text>\"}"
)


def _load_dataset(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _render_prompt(tokenizer, prompt_text: str, cfg_map: Dict[str, Any]) -> str:
    thinking_cfg = cfg_map.get("thinking", {}) or {}
    want_template_thinking = bool(thinking_cfg.get("enabled", False))
    template_supported: bool | None = None

    sys_msg = {"role": "system", "content": SYS_PROMPT}
    msgs = [sys_msg, {"role": "user", "content": prompt_text}]

    if want_template_thinking:
        try:
            rendered = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            template_supported = True
        except TypeError:
            template_supported = False
            rendered = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        rendered = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Cache the decision for future renders if needed.
    _render_prompt._template_supported = template_supported  # type: ignore[attr-defined]
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Test opponent response generation")
    parser.add_argument("--config", default="azr/config.json", help="Path to AZR config JSON")
    parser.add_argument(
        "--dataset", default="azr/data/train.jsonl", help="Training dataset JSONL"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max completion tokens (defaults to training value)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg_map = load_config(args.config)
    model_cfg = AzrModelCfg.from_dict(cfg_map.get("model", {}))
    sp_cfg = AzrSelfPlayCfg.from_dict(cfg_map.get("self_play", {}))
    if not sp_cfg.opponent:
        raise SystemExit("Self-play opponent configuration is missing")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    examples = _load_dataset(dataset_path)
    if not examples:
        raise SystemExit("Dataset is empty; nothing to sample")

    sample = random.choice(examples)
    tokenizer = load_tokenizer(model_cfg.model_id)
    rendered_prompt = _render_prompt(tokenizer, sample.get("prompt", ""), cfg_map)

    rlhf_cfg = (cfg_map.get("rlhf") or {})
    max_completion_len = int(rlhf_cfg.get("max_completion_length", 512))
    thinking_cfg = cfg_map.get("thinking", {}) or {}
    policy_extra = int(thinking_cfg.get("policy_budget_tokens", 0))
    max_completion_len += policy_extra
    if args.max_tokens is not None:
        max_completion_len = args.max_tokens

    sp_manager = SelfPlayManager(
        model_cfg,
        sp_cfg,
        opponent_device=sp_cfg.device,
        log_intersteps=False,
        config_map=cfg_map,
    )

    completions = sp_manager.generate_opponent([rendered_prompt], max_completion_len)
    opponent_output = completions[0] if completions else ""

    print("Selected problem:\n-------------------")
    print(sample.get("prompt", "").strip())
    print("\nOpponent response:\n-------------------")
    print(opponent_output.strip())


if __name__ == "__main__":
    main()
