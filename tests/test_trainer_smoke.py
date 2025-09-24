import json
import tempfile
from pathlib import Path

from azr.training import build_trainer

CONFIG = {
    "model": {
        "model_id": "sshleifer/tiny-gpt2",
        "quantization": "4bit",
        "lora_r": 4,
        "lora_alpha": 8,
    },
    "training": {
        "lr": 1e-5,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
    },
    "rlhf": {
        "num_generations": 2,
        "max_prompt_length": 64,
        "max_completion_length": 16,
    },
}


def test_build_and_train_one_step(monkeypatch):
    # Stub model/tokenizer to avoid heavy downloads
    class DummyTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            return {"prompt": prompt}

    class DummyModel:
        def __init__(self):
            self.trained = False

        def parameters(self):
            return []

        def generate(self, **kwargs):
            return ["output"]

    from azr import training

    class DummyGRPOTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def train(self) -> None:
            pass

    monkeypatch.setattr(training, "load_tokenizer", lambda model_id: DummyTokenizer())
    monkeypatch.setattr(training, "setup_model", lambda cfg, **kwargs: DummyModel())
    monkeypatch.setattr(training, "GRPOTrainer", DummyGRPOTrainer)

    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td) / "out"
        trainer = build_trainer(CONFIG, dataset_path=None, output_dir=str(output_dir), max_steps=1)
        trainer.train()
