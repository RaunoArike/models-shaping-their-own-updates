"""GroundTruthMonitorPenalty — oracle reward-hacking monitor (penalty variant).

The best-performing intervention in the Steering RL Training post, so it's our primary
baseline-to-beat. Uses the ground-truth reward-hack label from the env's dual-harness
evaluation (an oracle no real monitor has). When a sample is flagged as a hack, its
reward is overwritten with `rh_reward` (default 0.0).

An optional `accuracy < 1.0` simulates an imperfect monitor by randomly flipping the
oracle label, mirroring the rl-rewardhacking accuracy sweep.

Standalone port of the concept from rl-rewardhacking (reference only).
"""

from __future__ import annotations

import random

from ..envs.base import Environment, EvalResult
from .base import RewardFunction


class GroundTruthMonitorPenalty(RewardFunction):
    def __init__(
        self,
        env: Environment,
        rh_reward: float = 0.0,
        accuracy: float = 1.0,
        allow_hint: bool = True,
        use_loose: bool = False,
        seed: int = 0,
        **_ignored,
    ):
        self.env = env
        self.rh_reward = rh_reward
        self.accuracy = accuracy
        self.allow_hint = allow_hint
        self.use_loose = use_loose
        self._rng = random.Random(seed)

    def _flagged(self, ev: EvalResult) -> bool:
        truth = ev.get("is_reward_hack_loose" if self.use_loose else "is_reward_hack_strict", False)
        if self.accuracy >= 1.0:
            return bool(truth)
        # imperfect monitor: report the truth with prob=accuracy, else flip
        return bool(truth) if self._rng.random() < self.accuracy else (not truth)

    def score(self, examples, completions, evals: list[EvalResult]):
        rewards = []
        n_flagged = 0
        for ev in evals:
            r = self.env.reward(ev, allow_hint=self.allow_hint)
            if self._flagged(ev):
                r = self.rh_reward
                n_flagged += 1
            rewards.append(r)
        n = max(len(evals), 1)
        metrics = {
            "gt_monitor/frac_flagged": n_flagged / n,
            "gt_monitor/accuracy": self.accuracy,
            "detail/rh/frac_strict": sum(e.get("is_reward_hack_strict", False) for e in evals) / n,
        }
        return rewards, metrics
