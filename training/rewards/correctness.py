"""CorrectnessReward — the binary/continuous verifier reward, env-driven.

Standalone: reads EvalResult fields from the environment. No imports from
rl-rewardhacking/ (reference only).
"""

from __future__ import annotations

from ..envs.base import Environment, EvalResult
from .base import RewardFunction


class CorrectnessReward(RewardFunction):
    def __init__(self, env: Environment, allow_hint: bool = True, **_ignored):
        self.env = env
        self.allow_hint = allow_hint

    def score(self, examples, completions, evals: list[EvalResult]):
        rewards = [self.env.reward(ev, allow_hint=self.allow_hint) for ev in evals]
        n = max(len(evals), 1)
        metrics = {
            "detail/rh/frac_strict": sum(e.get("is_reward_hack_strict", False) for e in evals) / n,
            "detail/rh/frac_loose": sum(e.get("is_reward_hack_loose", False) for e in evals) / n,
            "detail/rh/frac_correct": sum(e.get("eq_correct", False) for e in evals) / n,
            "detail/frac_passed_verifier": sum(e.get("passed_verifier", False) for e in evals) / n,
        }
        return rewards, metrics
