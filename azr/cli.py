from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os

# Silence harmless tokenizer fork warnings that clutter logs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import click

from .config import AzrSelfPlayCfg, load_config
from .training import build_trainer as build_standard_trainer
from .training_selfplay import _DEFAULT_DATA_PATH, build_trainer as build_selfplay_trainer
from .utils import console


def _prepare_selfplay_config(config: Dict[str, Any], dataset_path: Optional[str]) -> Dict[str, Any]:
    prepared = dict(config)
    data_section = dict(prepared.get("data", {}))
    data_section["train_path"] = dataset_path or data_section.get("train_path") or _DEFAULT_DATA_PATH
    prepared["data"] = data_section
    return prepared


@click.group()
def cli() -> None:
    """Command-line interface for AZR training and evaluation."""


@cli.command()
@click.option("--config", "config_path", default="azr_config.json", show_default=True, help="Path to AZR config file")
@click.option("--dataset", default=None, help="Optional override for training dataset path")
@click.option(
    "--output",
    default="outputs/azr-run",
    show_default=True,
    help="Directory for trainer outputs (standard trainer only)",
)
@click.option("--max-steps", default=None, type=int, help="Limit training steps for quick smoke runs")
def train(config_path: str, dataset: Optional[str], output: str, max_steps: Optional[int]) -> None:
    """Train a model, using self-play when enabled in the configuration."""

    cfg = load_config(config_path)
    self_play_cfg = AzrSelfPlayCfg.from_dict(cfg.get("self_play", {}))

    if self_play_cfg.enabled:
        console.print("[cyan]Launching self-play trainer...[/]")
        sp_config = _prepare_selfplay_config(cfg, dataset)
        trainer = build_selfplay_trainer(sp_config, max_steps=max_steps)
        trainer.train()
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Launching standard trainer...[/]")
    trainer = build_standard_trainer(cfg, dataset, str(output_dir), max_steps=max_steps)
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    cli()
