import json
from pathlib import Path

import pytest

from azr.data import DataExample, load_dataset


def test_load_dataset_parses_examples(tmp_path: Path):
    src = tmp_path / "dataset.jsonl"
    rows = [
        {
            "prompt": "Say hi",
            "tests": ["assert True"],
            "timeout_s": 3,
            "memory_mb": 128,
        },
        {
            "prompt": "  ",
        },
        {
            "prompt": "Explain LoRA",
            "tests": "not-a-list",
        },
    ]
    with src.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    data = load_dataset(str(src))
    assert len(data) == 2
    assert data[0] == DataExample(prompt="Say hi", tests=["assert True"], timeout_s=3, memory_mb=128)
    assert data[1].tests == []
    assert data[1].timeout_s == 2
    assert data[1].memory_mb == 256


def test_load_dataset_ignores_blank_lines(tmp_path: Path):
    src = tmp_path / "dataset.jsonl"
    src.write_text("\n\n")
    assert load_dataset(str(src)) == []
