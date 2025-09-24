import io
import tempfile
from azr.config import load_config


def test_load_config_roundtrip():
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
    kl_coeff: 0.01
    bf16: false
    gradient_accumulation_steps: 1
  sandbox:
    timeout_s: 2
    memory_mb: 128
    max_tool_turns: 3
  web:
    max_bytes: 10000
    user_agent: "AZR-Research/1.0 (+no-bots)"
    headless: true
hardware:
  system_ram_gb: 16
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as f:
        f.write(yaml_text)
        f.flush()
        cfg = load_config(f.name)
    assert cfg.azr.model_id == "sshleifer/tiny-gpt2"
    assert cfg.azr.quantization is None
    assert cfg.azr.lora.r == 4
    assert cfg.azr.rlhf.num_generations == 2
    assert cfg.azr.web.max_bytes == 10000

