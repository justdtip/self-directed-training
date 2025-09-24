import tempfile
from pathlib import Path

from azr.config import load_config
from azr.training import build_trainer


yaml_text = """
azr:
  model_id: sshleifer/tiny-gpt2
  quantization: null
  lora:
    r: 4
    alpha: 8
    target_modules: ["c_attn"]
  rlhf:
    num_generations: 2
    max_prompt_length: 64
    max_completion_length: 16
    importance_sampling_level: "sequence"
    clip_range_ratio: 0.2
    kl_coeff: 0.0
    bf16: false
    gradient_accumulation_steps: 1
"""


def test_build_and_train_one_step():
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "params.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_config(str(cfg_path))

        trainer = build_trainer(cfg, dataset_path=None, output_dir=str(Path(td) / "out"), max_steps=1)
        # One short train step to validate pipeline
        trainer.train()

