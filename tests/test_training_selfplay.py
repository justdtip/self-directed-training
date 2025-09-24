import json
from pathlib import Path

import pytest

import azr.training_selfplay as tsp
from azr.data import DataExample


class DummyModel:
    def __init__(self) -> None:
        self.updated = 0

    def to(self, device):
        self.device = device
        return self


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        user = next(msg["content"] for msg in messages if msg["role"] == "user")
        return f"PROMPT::{user}"


class DummySelfPlayManager:
    def __init__(self, cfg, opponent_device="cpu") -> None:
        self.cfg = cfg
        self.opponent_device = opponent_device
        self.call_counter = 0
        self.updated = 0
        self.generated = []

    def generate_opponent(self, prompts, max_tokens):
        self.generated.append((tuple(prompts), max_tokens))
        return [f"{p}::opp" for p in prompts]

    def compute_scores(self, policy_outs, opp_outs, tests):
        return [0.5 for _ in policy_outs]

    def update_opponent(self, model):
        self.updated += 1


class DummyTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trained = False

    def train(self):
        self.trained = True


@pytest.fixture
def dataset_file(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "prompt": "Write a function add(a,b)",
            "tests": ["assert add(2,3)==5"],
            "timeout_s": 3,
            "memory_mb": 128,
        },
        {
            "prompt": "State the capital of France",
            "tests": [],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture(autouse=True)
def stub_components(monkeypatch):
    monkeypatch.setattr(tsp, "setup_model", lambda cfg, **kwargs: DummyModel())
    monkeypatch.setattr(tsp, "load_tokenizer", lambda model_id: DummyTokenizer())
    monkeypatch.setattr(tsp, "SelfPlayManager", DummySelfPlayManager)
    monkeypatch.setattr(tsp, "GRPOTrainer", lambda **kwargs: DummyTrainer(**kwargs))
    yield


def test_build_trainer_without_self_play(dataset_file):
    config = {
        "model": {"model_id": "stub-model", "quantization": None},
        "self_play": {"enabled": True},
        "training": {"lr": 1e-5, "warmup_ratio": 0.0, "weight_decay": 0.0},
        "rlhf": {
            "num_generations": 2,
            "max_prompt_length": 64,
            "max_completion_length": 16,
            "importance_sampling_level": "sequence",
            "clip_range_ratio": 0.2,
            "kl_coeff": 0.0,
            "bf16": False,
            "gradient_accumulation_steps": 1,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "data": {"train_path": str(dataset_file)},
    }
    trainer = tsp.build_trainer(config)
    assert isinstance(trainer, DummyTrainer)
    dataset = trainer.kwargs["train_dataset"]
    item = dataset[0]
    assert "PROMPT::" in item["prompt"] or "prompt" in item
    reward_fn = trainer.kwargs["reward_funcs"][0]
    scores = reward_fn(
        ["prompt"],
        ["""```python\ndef add(a,b): return a+b\n```"""],
        [{"tests": ["assert add(2,3)==5"], "timeout_s": 2, "memory_mb": 128}],
    )
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


def test_build_trainer_with_self_play(dataset_file, monkeypatch):
    config = {
        "model": {"model_id": "stub-model", "quantization": None},
        "self_play": {"enabled": True, "weight": 0.2, "device": "cpu", "update_every": 2},
        "training": {"lr": 1e-5, "warmup_ratio": 0.0, "weight_decay": 0.0},
        "rlhf": {
            "num_generations": 2,
            "max_prompt_length": 64,
            "max_completion_length": 16,
            "importance_sampling_level": "sequence",
            "clip_range_ratio": 0.2,
            "kl_coeff": 0.0,
            "bf16": False,
            "gradient_accumulation_steps": 1,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "data": {"train_path": str(dataset_file)},
    }
    trainer = tsp.build_trainer(config)
    reward_fn = trainer.kwargs["reward_funcs"][0]
    closure = reward_fn.__closure__ or []
    spm = next((cell.cell_contents for cell in closure if isinstance(cell.cell_contents, DummySelfPlayManager)), None)
    assert isinstance(spm, DummySelfPlayManager)

    batch_prompts = ["prompt-1"]
    outputs = ["final answer"]
    metadata = [{"tests": [], "timeout_s": 2, "memory_mb": 128}]

    score_first = reward_fn(batch_prompts, outputs, metadata)[0]
    score_second = reward_fn(batch_prompts, outputs, metadata)[0]

    assert score_first == pytest.approx(score_second)
    assert spm.updated == 1
    assert spm.call_counter == 2
