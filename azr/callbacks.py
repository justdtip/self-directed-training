import json
import os
import time
from pathlib import Path
from typing import Optional

from transformers.trainer_callback import TrainerCallback


class TrainingMetricsCallback(TrainerCallback):
    """Stream metrics to console and persist them to a JSONL file."""

    def __init__(self, out_dir: Optional[str] = None) -> None:
        self.out_dir = out_dir
        self._fp = None

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        run_dir = self.out_dir or getattr(args, "output_dir", "outputs/azr-run")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(run_dir, "metrics.jsonl")
        self._fp = open(path, "a", encoding="utf-8")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return control
        step = getattr(state, "global_step", None)
        keys = (
            "loss",
            "reward",
            "reward_std",
            "entropy",
            "completions/mean_length",
            "completions/clipped_ratio",
        )
        line = " | ".join(f"{k}={logs.get(k)}" for k in keys if k in logs)
        print(f"[Metrics] step={step} | {line}")

        record = {"ts": time.time(), "step": step, **logs}
        if self._fp:
            self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fp.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._fp:
            self._fp.close()
            self._fp = None
        return control


__all__ = ["TrainingMetricsCallback"]
