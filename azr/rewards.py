from __future__ import annotations

from typing import List, Sequence


def keyword_reward(outputs: Sequence[str], positive: Sequence[str] = ("helpful", "final"), negative: Sequence[str] = ("harmful",)) -> List[float]:
    rewards: List[float] = []
    for out in outputs:
        score = 0.0
        low = out.lower()
        for k in positive:
            if k in low:
                score += 0.5
        for k in negative:
            if k in low:
                score -= 0.75
        rewards.append(score)
    return rewards


def length_penalty(outputs: Sequence[str], max_len: int = 256) -> List[float]:
    res = []
    for out in outputs:
        over = max(0, len(out) - max_len)
        res.append(-0.001 * over)
    return res


def combine_rewards(*reward_lists: Sequence[float]) -> List[float]:
    if not reward_lists:
        return []
    n = len(reward_lists[0])
    res = [0.0] * n
    for r in reward_lists:
        assert len(r) == n, "All reward lists must be same length"
        for i, v in enumerate(r):
            res[i] += v
    return res

