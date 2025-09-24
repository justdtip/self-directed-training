import tempfile
from pathlib import Path

from azr.config import load_config


def test_load_config_with_extra_fields_and_new_hardware_shape():
    yaml_text = """
azr:
  model_id: deepcogito/cogito-v1-preview-llama-8b
  quantization: "4bit"
  lora:
    r: 16
    alpha: 32
    target_modules: ["q_proj", "v_proj"]
  rlhf:
    num_generations: 4
    max_prompt_length: 2048
    max_completion_length: 1024
    importance_sampling_level: "sequence"
    clip_range_ratio: 0.1
    kl_coeff: 0.0
    bf16: true
    gradient_accumulation_steps: 8
    temperature: 0.7
    top_p: 0.95
training:
  lr: 1.0e-5
  warmup_ratio: 0.03
  weight_decay: 0.0
  log_dir: "/opt/azr/runs"
self_play:
  enabled: true
  weight: 0.2
  device: cuda:1
  tie_breakers: ["pass_rate", "code_length"]
  update:
    every_calls: 500
    strategy: copy_lora
hardware:
  gpus:
    - name: RTX 6000 Ada
      vram_gb: 48
    - name: RTX 6000 Ada
      vram_gb: 48
  cpu: AMD EPYC 7352 (24-core)
  ram_gb: 172
  storage: SAMSUNG MZ1L21T9HCLS-00A07
  cuda_runtime: 12.x
"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.yaml"
        p.write_text(yaml_text)
        cfg = load_config(str(p))

    assert cfg.azr.rlhf.temperature == 0.7
    assert cfg.azr.rlhf.top_p == 0.95
    assert cfg.training.lr == 1.0e-5
    # Hardware should not crash; CPU/GPU types can be partially filled
    assert len(cfg.hardware.gpus) == 2
    assert cfg.self_play.enabled is True
    assert cfg.self_play.weight == 0.2
    assert cfg.self_play.update.every_calls == 500
