from __future__ import annotations

from typing import List, Sequence


def keyword_reward(
    outputs: Sequence[str],
    positive: Sequence[str] = ("helpful", "final"),
    negative: Sequence[str] = ("harmful",),
) -> List[float]:
    rewards: List[float] = []
    for out in outputs:
        score = 0.0
        low = out.lower()
        for token in positive:
            if token in low:
                score += 0.5
        for token in negative:
            if token in low:
                score -= 0.75
        rewards.append(score)
    return rewards


def length_penalty(outputs: Sequence[str], max_len: int = 256) -> List[float]:
    penalties: List[float] = []
    for out in outputs:
        over = max(0, len(out) - max_len)
        penalties.append(-0.001 * over)
    return penalties


def combine_rewards(*reward_lists: Sequence[float]) -> List[float]:
    if not reward_lists:
        return []
    n = len(reward_lists[0])
    combined = [0.0] * n
    for rewards in reward_lists:
        assert len(rewards) == n, "All reward lists must be the same length"
        for idx, value in enumerate(rewards):
            combined[idx] += value
    return combined


__all__ = ["keyword_reward", "length_penalty", "combine_rewards"]
