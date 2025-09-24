"""Self-play management utilities for GSPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch

from opt_azr_params import PARAMS
from rewards import _last_python_block, score_code_tests


@dataclass(frozen=True)
class TieBreakerConfig:
    order: List[str]


def score_selfplay_pair(
    policy_output: str,
    opponent_output: str,
    tests: Iterable[str],
    timeout_s: int,
    memory_mb: int,
    tie_breakers: Iterable[str] | None = None,
) -> float:
    """Return a self-play score in ``[0, 1]`` for a policy/opponent pair."""

    tie_breakers = list(tie_breakers or ["pass_rate", "code_length"])
    policy_score, _ = score_code_tests(policy_output, list(tests), timeout_s=timeout_s, memory_mb=memory_mb)
    opponent_score, _ = score_code_tests(opponent_output, list(tests), timeout_s=timeout_s, memory_mb=memory_mb)

    if policy_score > opponent_score:
        return 1.0
    if policy_score < opponent_score:
        return 0.0

    for breaker in tie_breakers:
        if breaker == "code_length":
            p_code = _last_python_block(policy_output) or policy_output
            o_code = _last_python_block(opponent_output) or opponent_output
            p_len = len(p_code.strip())
            o_len = len(o_code.strip())
            if p_len < o_len:
                return 2.0 / 3.0
            if p_len > o_len:
                return 1.0 / 3.0

    return 0.5


class SelfPlayManager:
    """Manage opponent generation and snapshot updates for self-play."""

    def __init__(
        self,
        tokenizer,
        max_seq: int,
        *,
        enabled: Optional[bool] = None,
        lazy: bool = False,
    ) -> None:
        config = PARAMS.get("self_play", {})
        self.enabled = bool(config.get("enabled", False)) if enabled is None else enabled
        self.weight = float(config.get("weight", 0.2))
        self.device = config.get("device", "cuda:1")
        self.use_tool_loop = bool(config.get("use_tool_loop", False))
        tie_breakers = config.get("tie_breakers", ["pass_rate", "code_length"])
        self.tie_breakers = list(tie_breakers)
        update_cfg = config.get("update", {}) or {}
        self.update_every_calls = int(update_cfg.get("every_calls", 0))
        self.update_strategy = update_cfg.get("strategy", "copy_lora")

        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.call_count = 0
        self.opponent = None
        self.opponent_tokenizer = None

        if self.enabled and not lazy:
            self._load_opponent()

    def _load_opponent(self) -> None:
        from unsloth import FastLanguageModel

        params = PARAMS["azr"]
        base_id = params["model_id"]
        load_in_4bit = params.get("quantization") == "4bit"
        opponent, opponent_tok = FastLanguageModel.from_pretrained(
            base_id, max_seq_length=self.max_seq, load_in_4bit=load_in_4bit
        )
        opponent = FastLanguageModel.get_peft_model(
            opponent,
            r=params["lora"]["r"],
            target_modules=params["lora"]["target_modules"],
        )
        opponent = opponent.to(self.device)
        opponent.eval()
        self.opponent = opponent
        self.opponent_tokenizer = opponent_tok

    def _generate_plain(self, prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        assert self.opponent is not None and self.opponent_tokenizer is not None
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.opponent_tokenizer(prompt, return_tensors="pt").to(self.device)
                result = self.opponent.generate(
                    **inputs,
                    max_new_tokens=PARAMS["azr"]["rlhf"]["max_completion_length"],
                    do_sample=True,
                    temperature=PARAMS["azr"]["rlhf"].get("temperature", 0.7),
                    top_p=PARAMS["azr"]["rlhf"].get("top_p", 0.95),
                )
                outputs.append(self.opponent_tokenizer.decode(result[0], skip_special_tokens=True))
        return outputs

    def _generate_opponent(self, prompts: List[str]) -> List[str]:
        if not self.enabled or self.opponent is None:
            return [""] * len(prompts)
        if self.use_tool_loop:
            from tool_loop import roll_with_tools

            outputs: List[str] = []
            turn_limit = PARAMS["azr"]["sandbox"]["max_tool_turns"]
            for prompt in prompts:
                system = "Use available tools prudently. End with a JSON object containing final_answer."
                final, _, _ = roll_with_tools(
                    self.opponent,
                    self.opponent_tokenizer,
                    system,
                    prompt,
                    max_turns=turn_limit,
                )
                outputs.append(final if isinstance(final, str) else str(final))
            return outputs
        return self._generate_plain(prompts)

    def compute_selfplay_scores(
        self,
        batch_prompts: List[str],
        policy_outputs: List[str],
        metadata: List[Dict[str, object]],
    ) -> List[float]:
        if not self.enabled:
            return [0.5] * len(policy_outputs)
        opponent_outputs = self._generate_opponent(batch_prompts)
        scores: List[float] = []
        for policy_out, opp_out, meta in zip(policy_outputs, opponent_outputs, metadata):
            timeout_s = int(meta.get("timeout_s", 2))
            memory_mb = int(meta.get("memory_mb", 256))
            tests = meta.get("tests", []) or []
            score = score_selfplay_pair(
                policy_out,
                opp_out,
                tests,
                timeout_s,
                memory_mb,
                self.tie_breakers,
            )
            scores.append(score)
        return scores

    def step_and_maybe_update(self, policy_model) -> None:
        if not self.enabled or self.opponent is None:
            return
        self.call_count += 1
        if self.update_every_calls <= 0:
            return
        if self.call_count % self.update_every_calls != 0:
            return
        if self.update_strategy == "copy_lora":
            try:
                from peft import get_peft_model_state_dict, set_peft_model_state_dict

                state = get_peft_model_state_dict(policy_model)
                set_peft_model_state_dict(self.opponent, state)
                torch.cuda.synchronize()
            except Exception:
                pass


__all__ = ["SelfPlayManager", "score_selfplay_pair"]

