from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

import os
import sys

# Silence harmless tokenizer fork warnings that clutter logs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import click

from .config import AzrModelCfg, AzrSelfPlayCfg, AzrVllmCfg, load_config
from .training import build_trainer as build_standard_trainer
from .training_selfplay import _DEFAULT_DATA_PATH, build_trainer as build_selfplay_trainer
from .utils import console
from .vllm_launcher import VLLMServerError, launch_vllm_server


class _Tee:
    """Mirror writes to multiple streams."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:  # pragma: no cover - trivial
        total = 0
        for stream in self.streams:
            total += stream.write(data)
            stream.flush()
        return total

    def flush(self) -> None:  # pragma: no cover - trivial
        for stream in self.streams:
            stream.flush()


def _prepare_selfplay_config(config: Dict[str, Any], dataset_path: Optional[str]) -> Dict[str, Any]:
    prepared = dict(config)
    data_section = dict(prepared.get("data", {}))
    data_section["train_path"] = dataset_path or data_section.get("train_path") or _DEFAULT_DATA_PATH
    prepared["data"] = data_section
    return prepared


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    base = Path(output_dir)
    if not base.exists():
        return None
    best_path: Optional[Path] = None
    best_step = -1
    for path in base.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = path
    return str(best_path) if best_path is not None else None


def _close_generation_logger(trainer) -> None:
    if hasattr(trainer, "gen_logger"):
        try:
            trainer.gen_logger.close()
        except Exception:
            pass


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
@click.option("--log-file", default=None, type=str, help="Optional file to append combined stdout/stderr logs")
@click.option(
    "--resume-from",
    default=None,
    type=str,
    help="Path to a checkpoint directory or 'latest' to resume from the newest checkpoint in the output directory",
)
def train(
    config_path: str,
    dataset: Optional[str],
    output: str,
    max_steps: Optional[int],
    log_file: Optional[str],
    resume_from: Optional[str],
) -> None:
    """Train a model, using self-play when enabled in the configuration."""

    cfg = load_config(config_path)
    self_play_cfg = AzrSelfPlayCfg.from_dict(cfg.get("self_play", {}))

    log_handle = None
    stdout_cm = nullcontext()
    stderr_cm = nullcontext()
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        tee_out = _Tee(sys.stdout, log_handle)
        tee_err = _Tee(sys.stderr, log_handle)
        stdout_cm = redirect_stdout(tee_out)
        stderr_cm = redirect_stderr(tee_err)

    try:
        with stdout_cm, stderr_cm:
            if log_file:
                console.print(f"[blue]Logging stdout/stderr to {log_file}[/]")

            if self_play_cfg.enabled:
                console.print("[cyan]Launching self-play trainer...[/]")
                sp_config = _prepare_selfplay_config(cfg, dataset)
                vllm_cfg = AzrVllmCfg.from_dict(sp_config.get("vllm", {}))
                model_cfg = AzrModelCfg.from_dict(sp_config.get("model", {}))

                with ExitStack() as stack:
                    if vllm_cfg.enabled and vllm_cfg.manage_server:
                        try:
                            stack.enter_context(
                                launch_vllm_server(
                                    model_cfg,
                                    vllm_cfg,
                                    log_stream=log_handle,
                                )
                            )
                            console.print(
                                f"[green]Started vLLM sidecar at http://{vllm_cfg.host}:{vllm_cfg.port}[/]"
                            )
                        except VLLMServerError as exc:
                            raise SystemExit(f"Failed to launch vLLM server: {exc}")
                    trainer = build_selfplay_trainer(sp_config, max_steps=max_steps)
                    resume_path = _resolve_resume_path(trainer.args.output_dir, resume_from)
                    try:
                        trainer.train(resume_from_checkpoint=resume_path if resume_path else None)
                    finally:
                        _close_generation_logger(trainer)
                return

            output_dir_path = Path(output)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            console.print("[cyan]Launching standard trainer...[/]")
            trainer = build_standard_trainer(cfg, dataset, str(output_dir_path), max_steps=max_steps)
            resume_path = _resolve_resume_path(trainer.args.output_dir, resume_from)
            try:
                trainer.train(resume_from_checkpoint=resume_path if resume_path else None)
            finally:
                _close_generation_logger(trainer)
    finally:
        if log_handle is not None:
            log_handle.close()


def _resolve_resume_path(output_dir: str, resume_from: Optional[str]) -> Optional[str]:
    if not resume_from:
        return None
    if resume_from.lower() == "latest":
        latest = _find_latest_checkpoint(output_dir)
        if latest is None:
            console.print(f"[yellow]No checkpoints found under {output_dir}; starting fresh.[/]")
        return latest
    path = Path(resume_from)
    if not path.exists():
        console.print(f"[yellow]Checkpoint '{resume_from}' not found; starting fresh.[/]")
        return None
    return str(path)


if __name__ == "__main__":  # pragma: no cover
    cli()
