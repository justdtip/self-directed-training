from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .tools.python_tool import run_code
from .rewards import extract_last_code_block, score_code_tests
from .modeling import load_tokenizer, setup_model
from .config import AzrModelCfg


@dataclass
class SelfPlayResult:
    policy_pass: float
    opponent_pass: float
    score: float


class SelfPlayManager:
    """
    Frozen opponent on a specified device. Generates outputs and computes win/loss scores.
    """

    def __init__(self, cfg: AzrModelCfg, opponent_device: str = "cuda:1") -> None:
        self.cfg = cfg
        self.opponent_device = opponent_device
        self.op_tok = load_tokenizer(cfg.model_id)
        self.opponent = setup_model(cfg)
        try:
            self.primary_device = next(self.opponent.parameters()).device
        except StopIteration:
            self.primary_device = torch.device(opponent_device if torch.cuda.is_available() else "cpu")
        else:
            # If a specific device was requested and differs from the parameter device, attempt to move
            if opponent_device and torch.cuda.is_available():
                desired = torch.device(opponent_device)
                if self.primary_device != desired:
                    try:
                        self.opponent = self.opponent.to(desired)
                        self.primary_device = desired
                    except Exception:
                        # Fall back to the existing device map (common for 4bit models).
                        pass
        self.opponent.eval()
        self.call_counter = 0

    def update_opponent(self, policy_model) -> None:
        """Copy LoRA weights from the current policy to the opponent."""

        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        sd = get_peft_model_state_dict(policy_model)
        set_peft_model_state_dict(self.opponent, sd)
        torch.cuda.synchronize()

    def generate_opponent(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Generate outputs from the opponent for each prompt."""

        outs: List[str] = []
        for pr in prompts:
            with torch.no_grad():
                inputs = self.op_tok(pr, return_tensors="pt")
                inputs = {key: value.to(self.primary_device) for key, value in inputs.items()}
                out = self.opponent.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                )
                outs.append(self.op_tok.decode(out[0], skip_special_tokens=True))
        return outs

    def compute_scores(
        self,
        policy_outs: List[str],
        opp_outs: List[str],
        tests: List[List[str]],
    ) -> List[float]:
        """
        Compare each pair of outputs and return a score in [0, 1].
        1.0 if policy wins, 0.0 if opponent wins, 0.5 for ties; fractional tie-breaker on code length.
        """

        scores: List[float] = []
        for p_out, o_out, tlist in zip(policy_outs, opp_outs, tests):
            p_pass, _ = score_code_tests(p_out, tlist)
            o_pass, _ = score_code_tests(o_out, tlist)
            if p_pass > o_pass:
                scores.append(1.0)
            elif p_pass < o_pass:
                scores.append(0.0)
            else:
                p_code = extract_last_code_block(p_out) or p_out
                o_code = extract_last_code_block(o_out) or o_out
                lp, lo = len(p_code.strip()), len(o_code.strip())
                if lp < lo:
                    scores.append(2 / 3)
                elif lp > lo:
                    scores.append(1 / 3)
                else:
                    scores.append(0.5)
        return scores


__all__ = ["SelfPlayManager", "SelfPlayResult"]
