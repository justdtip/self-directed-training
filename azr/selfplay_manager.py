from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import asyncio
import time
import torch

from .tools.python_tool import run_code
from .rewards import extract_last_code_block, score_code_tests
from .modeling import load_tokenizer, setup_model
from .config import AzrModelCfg, AzrSelfPlayCfg
from .opponent_provider_together import TogetherAIOpponentProvider


@dataclass
class SelfPlayResult:
    policy_pass: float
    opponent_pass: float
    score: float


class SelfPlayManager:
    """Manages remote/local opponents and optional teacher assistance."""

    def __init__(
        self,
        cfg: AzrModelCfg,
        sp_cfg: Optional[AzrSelfPlayCfg] = None,
        opponent_device: str = "cuda:1",
        *,
        log_intersteps: bool = False,
        config_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.sp_cfg = sp_cfg
        self.opponent_device = opponent_device
        self.log_intersteps = log_intersteps
        self.config_map = config_map or {}

        thinking_cfg = self.config_map.get("thinking", {}) or {}
        teacher_cfg = self.config_map.get("teacher", {}) or {}

        opp_info = sp_cfg.opponent if sp_cfg and sp_cfg.opponent else {}
        self.remote_provider: Optional[TogetherAIOpponentProvider] = None
        self.teacher_provider: Optional[TogetherAIOpponentProvider] = None
        self.gen_logger = None
        self.trainer = None

        self.op_tok = load_tokenizer(cfg.model_id)

        if opp_info.get("source") == "remote":
            provider = opp_info.get("provider")
            if provider != "together_ai":
                raise ValueError(f"Unsupported opponent provider: {provider}")
            self.remote_provider = TogetherAIOpponentProvider(
                endpoint=opp_info["endpoint"],
                model_id=opp_info["model_id"],
                api_key_env=opp_info["api_key_env"],
                max_concurrency=int(opp_info.get("max_concurrency", 5)),
                temperature=float(opp_info.get("temperature", 0.7)),
                top_p=float(opp_info.get("top_p", 0.95)),
                extra_body=opp_info.get("extra_body", {}),
                thinking_budget=int(thinking_cfg.get("opponent_budget_tokens", 0)) or None,
            )
            print(
                f"[SelfPlay] Loaded opponent provider {provider} model={opp_info['model_id']}"
            )
            self.opponent = None
            self.primary_device = None
        else:
            self.opponent = setup_model(cfg)
            print("[SelfPlay] Loaded local frozen opponent model")
            try:
                params = self.opponent.parameters()
                first_param = next(params)
                self.primary_device = first_param.device
            except (StopIteration, AttributeError):
                device_target = opponent_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
                self.primary_device = torch.device(device_target)
                if hasattr(self.opponent, "to") and device_target is not None:
                    try:
                        self.opponent = self.opponent.to(device_target)
                    except Exception:
                        pass
            else:
                if opponent_device and torch.cuda.is_available():
                    desired = torch.device(opponent_device)
                    if self.primary_device != desired:
                        try:
                            self.opponent = self.opponent.to(desired)
                            self.primary_device = desired
                        except Exception:
                            pass
            if self.opponent is not None:
                self.opponent.eval()

        if teacher_cfg.get("source") == "remote":
            provider = teacher_cfg.get("provider")
            if provider != "together_ai":
                raise ValueError(f"Unsupported teacher provider: {provider}")
            self.teacher_provider = TogetherAIOpponentProvider(
                endpoint=teacher_cfg["endpoint"],
                model_id=teacher_cfg["model_id"],
                api_key_env=teacher_cfg["api_key_env"],
                max_concurrency=int(teacher_cfg.get("max_concurrency", 4)),
                temperature=float(teacher_cfg.get("temperature", 0.6)),
                top_p=float(teacher_cfg.get("top_p", 0.95)),
                extra_body=teacher_cfg.get("extra_body", {}),
                thinking_budget=int(thinking_cfg.get("teacher_budget_tokens", 0)) or None,
            )
            print(
                f"[SelfPlay] Loaded teacher provider {provider} model={teacher_cfg['model_id']}"
            )

        self.call_counter = 0

    def update_opponent(self, policy_model) -> None:
        if self.remote_provider is not None:
            return

        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        sd = get_peft_model_state_dict(policy_model)
        set_peft_model_state_dict(self.opponent, sd)
        torch.cuda.synchronize()

    def generate_opponent(self, prompts: List[str], max_tokens: int) -> List[str]:
        thinking_cfg = self.config_map.get("thinking", {}) or {}
        opp_extra = int(thinking_cfg.get("opponent_budget_tokens", 0)) or 0
        max_tokens_with_extra = max_tokens + opp_extra
        if self.remote_provider is not None:
            if self.log_intersteps:
                print(f"[Stage] Remote opponent request start | prompts={len(prompts)}")
            completions = asyncio.run(self.remote_provider.agenerate(prompts, max_tokens_with_extra))
            if self.log_intersteps:
                print("[Stage] Remote opponent request done")
            stats = self.remote_provider.pop_stats()
            if stats:
                print("[Remote]", " ".join(f"{k}={v}" for k, v in stats.items()))
            if self.gen_logger is not None:
                ts = time.time()
                step = None
                if getattr(self, "trainer", None) is not None:
                    step = getattr(getattr(self.trainer, "state", None), "global_step", None)
                for prompt_text, completion in zip(prompts, completions):
                    try:
                        self.gen_logger.log(
                            {
                                "ts": ts,
                                "source": "opponent_raw",
                                "step": step,
                                "prompt": prompt_text,
                                "completion": completion,
                                "length_tokens": len(completion.split()),
                            }
                        )
                    except Exception:
                        pass
            return list(completions)

        outs: List[str] = []
        for pr in prompts:
            if self.log_intersteps:
                print(f"[Stage] Opponent generation start (prompt preview): {pr[:80].replace('\n', ' ')}")
            with torch.no_grad():
                inputs = self.op_tok(pr, return_tensors="pt")
                model_for_embeddings = getattr(self.opponent, "model", self.opponent)
                if hasattr(model_for_embeddings, "embed_tokens"):
                    embed_device = model_for_embeddings.embed_tokens.weight.device
                else:
                    embed_device = self.primary_device or torch.device(
                        self.opponent_device if torch.cuda.is_available() else "cpu"
                    )
                inputs = {
                    key: value.to(embed_device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                out = self.opponent.generate(
                    **inputs,
                    max_new_tokens=max_tokens_with_extra,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                )
                decoded = self.op_tok.decode(out[0], skip_special_tokens=True)
                outs.append(decoded)
                if self.log_intersteps:
                    print(f"[Stage] Opponent completion done | len={len(decoded)}")
        return outs

    def teacher_hint(self, prompts: List[str], max_tokens: int) -> List[str]:
        if self.teacher_provider is None:
            return ["" for _ in prompts]
        return asyncio.run(self.teacher_provider.agenerate(prompts, max_tokens))

    def compute_scores(
        self,
        policy_outs: List[str],
        opp_outs: List[str],
        tests: List[List[str]],
    ) -> List[float]:
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
