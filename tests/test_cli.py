from __future__ import annotations

from pathlib import Path

import click.testing
import pytest

import azr.cli as cli_mod


class DummyTrainer:
    def __init__(self) -> None:
        self.trained = False

    def train(self) -> None:
        self.trained = True


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """
{
  "model": {
    "model_id": "stub-model",
    "quantization": "4bit",
    "lora_r": 4,
    "lora_alpha": 8
  },
  "training": {
    "lr": 1e-5,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0
  },
  "rlhf": {
    "num_generations": 1,
    "max_prompt_length": 32,
    "max_completion_length": 16
  },
  "self_play": {
    "enabled": false
  }
}
""",
        encoding="utf-8",
    )
    return cfg_path


@pytest.fixture
def runner() -> click.testing.CliRunner:
    return click.testing.CliRunner()


def test_train_standard_invokes_standard_trainer(monkeypatch, runner, config_file, tmp_path):
    calls: dict[str, object] = {}

    def fake_build_trainer(cfg, dataset_path, output_dir, max_steps=None):
        calls["cfg"] = cfg
        calls["dataset"] = dataset_path
        calls["output"] = output_dir
        calls["max_steps"] = max_steps
        trainer = DummyTrainer()
        calls["trainer"] = trainer
        return trainer

    monkeypatch.setattr(cli_mod, "build_standard_trainer", fake_build_trainer)

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--config",
            str(config_file),
            "--dataset",
            "dataset.jsonl",
            "--output",
            str(tmp_path / "out"),
            "--max-steps",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert isinstance(calls.get("cfg"), dict)
    assert calls["dataset"] == "dataset.jsonl"
    assert calls["output"].endswith("out")
    assert calls["max_steps"] == 3
    assert calls["trainer"].trained is True


def test_train_selfplay_invokes_selfplay_trainer(monkeypatch, runner, config_file, tmp_path):
    config_file.write_text(
        """
{
  "model": {
    "model_id": "stub-model",
    "quantization": "4bit",
    "lora_r": 4,
    "lora_alpha": 8
  },
  "training": {
    "lr": 1e-5,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0
  },
  "rlhf": {
    "num_generations": 2,
    "max_prompt_length": 64,
    "max_completion_length": 32
  },
  "self_play": {
    "enabled": true,
    "weight": 0.2,
    "device": "cpu",
    "update_every": 2
  }
}
""",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_selfplay_trainer(conf):
        trainer = DummyTrainer()
        captured["conf"] = conf
        captured["trainer"] = trainer
        return trainer

    monkeypatch.setattr(cli_mod, "build_selfplay_trainer", fake_build_selfplay_trainer)

    result = runner.invoke(
        cli_mod.cli,
        [
            "train",
            "--config",
            str(config_file),
            "--dataset",
            "custom.jsonl",
        ],
    )

    assert result.exit_code == 0, result.output
    conf = captured.get("conf")
    assert isinstance(conf, dict)
    assert conf["data"]["train_path"] == "custom.jsonl"
    assert conf["self_play"]["enabled"] is True
    assert conf["rlhf"]["num_generations"] == 2
    trainer = captured.get("trainer")
    assert isinstance(trainer, DummyTrainer)
    assert trainer.trained is True
