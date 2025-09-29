from __future__ import annotations

import threading
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Optional

from .logging_io import atomic_write_json, atomic_write_text


@dataclass
class Scores:
    policy_only: int = 0
    opponent_only: int = 0
    both: int = 0
    neither: int = 0

    def as_dict(self) -> dict[str, int]:
        total = self.policy_only + self.opponent_only + self.both + self.neither
        return {
            "policy_only": self.policy_only,
            "opponent_only": self.opponent_only,
            "both": self.both,
            "neither": self.neither,
            "total": total,
        }


class ScoreBoard:
    """Thread-safe cumulative scoreboard for self-play outcomes."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scores = Scores()
        self._lock = threading.Lock()

    def update_batch(self, policy_pass: list[bool], opponent_pass: list[bool]) -> None:
        p_flags = list(policy_pass)
        o_flags = list(opponent_pass)
        if len(p_flags) != len(o_flags):
            diff = len(p_flags) - len(o_flags)
            if diff > 0:
                o_flags.extend([False] * diff)
            else:
                p_flags.extend([False] * (-diff))
            print(
                "[ScoreBoard] Warning: mismatched batch sizes; padding shorter list with failures "
                f"(policy={len(p_flags)} opponent={len(o_flags)})"
            )
        with self._lock:
            for p, o in zip(p_flags, o_flags):
                if p and o:
                    self.scores.both += 1
                elif p and not o:
                    self.scores.policy_only += 1
                elif (not p) and o:
                    self.scores.opponent_only += 1
                else:
                    self.scores.neither += 1

    @staticmethod
    def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
        if total <= 0:
            return (None, None)
        p = wins / total
        z2 = z * z
        denom = 1.0 + z2 / total
        center = p + z2 / (2 * total)
        margin = z * sqrt((p * (1 - p) + z2 / (4 * total)) / total)
        lb = (center - margin) / denom
        ub = (center + margin) / denom
        return (max(0.0, min(1.0, lb)), max(0.0, min(1.0, ub)))

    def write(self) -> None:
        with self._lock:
            counts = self.scores.as_dict()

        wins = self.scores.policy_only
        losses = self.scores.opponent_only
        ties = self.scores.both + self.scores.neither
        decisive = wins + losses
        win_rate = (wins / decisive) if decisive > 0 else None
        z = 1.96
        lb, ub = self._wilson_interval(wins, decisive, z=z)

        txt = (
            f"Policy only: {counts['policy_only']}\n"
            f"Opponent only: {counts['opponent_only']}\n"
            f"Both: {counts['both']}\n"
            f"Neither: {counts['neither']}\n"
        )

        stats_lines = [""]
        stats_lines.append(f"Total: {counts['total']}")
        stats_lines.append(f"Decisive (W+L): {decisive}")
        stats_lines.append(f"Wins (policy_only): {wins}")
        stats_lines.append(f"Losses (opponent_only): {losses}")
        stats_lines.append(f"Ties (both+neither): {ties}")
        stats_lines.append(
            f"Win-rate (decisive): {win_rate:.4f}" if win_rate is not None else "Win-rate (decisive): n/a"
        )
        if lb is not None and ub is not None:
            stats_lines.append(f"Wilson 95% CI: [{lb:.4f}, {ub:.4f}]  z={z}")
        else:
            stats_lines.append(f"Wilson 95% CI: n/a  z={z}")

        txt += "\n".join(stats_lines) + "\n"
        atomic_write_text(self.out_dir / "scoreboard.txt", txt)

        data = dict(counts)
        data.update(
            {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "n_decisive": decisive,
                "win_rate": win_rate,
                "wilson_lb": lb,
                "wilson_ub": ub,
                "z": z,
            }
        )
        atomic_write_json(self.out_dir / "scoreboard.json", data)


__all__ = ["ScoreBoard", "Scores"]
