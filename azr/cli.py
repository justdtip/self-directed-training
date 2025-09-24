from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config
from .training import build_trainer
from .utils import console
from .tools.web import WebTool
from .sandbox import ToolSandbox


def train_cmd():
    parser = argparse.ArgumentParser(description="AZR GRPO Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to azr.params.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Path to prompts (jsonl or plain)")
    parser.add_argument("--output", type=str, default="outputs/azr-run", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps for quick runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(cfg, args.dataset, args.output, max_steps=args.max_steps)
    console.print("[green]Starting training...[/]")
    trainer.train()
    console.print("[green]Training complete. Saving...[/]")
    trainer.save_model()


def eval_cmd():
    parser = argparse.ArgumentParser(description="AZR Tool Sandbox eval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--url", type=str, default="https://example.com")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sandbox = ToolSandbox(
        timeout_s=cfg.azr.sandbox.timeout_s,
        memory_mb=cfg.azr.sandbox.memory_mb,
        max_tool_turns=cfg.azr.sandbox.max_tool_turns,
    )
    web = WebTool(
        max_bytes=cfg.azr.web.max_bytes,
        user_agent=cfg.azr.web.user_agent,
        timeout_s=cfg.azr.sandbox.timeout_s,
    )
    sandbox.check()
    text = web.fetch(args.url)
    console.print(f"Fetched {len(text)} bytes from {args.url}")


def gen_cmd():
    parser = argparse.ArgumentParser(description="AZR Generation placeholder")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    from .modeling import setup_model, load_tokenizer

    tok = load_tokenizer(cfg.azr.model_id)
    model = setup_model(cfg.azr)

    import torch

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=min(64, cfg.azr.rlhf.max_completion_length))
    text = tok.decode(out[0], skip_special_tokens=True)
    # Print only the completion part
    completion = text[len(args.prompt) :].strip()
    console.print(completion)

