"""Reward / screening registry and builders.

`build_reward_functions(specs, trainer)` and `build_screening_functions(specs, trainer)`
instantiate classes by name, injecting `env`, and (for grader-style functions) the
tokenizer and the appropriate sampling client based on the `grader` kwarg
(frozen_base -> trainer.frozen_base_client, student -> trainer.sampling_client).
"""

from __future__ import annotations

import inspect

from .base import RewardFunction, ScreeningFunction
from .correctness import CorrectnessReward
from .gt_monitor import GroundTruthMonitorPenalty
from .self_grading import SelfGradingReward
from .self_screening import SelfScreener

REWARD_CLASSES: dict[str, type] = {
    "CorrectnessReward": CorrectnessReward,
    "GroundTruthMonitorPenalty": GroundTruthMonitorPenalty,
    "SelfGradingReward": SelfGradingReward,
}

SCREENING_CLASSES: dict[str, type] = {
    "SelfScreener": SelfScreener,
}


def _grader_client_for(trainer, kwargs: dict):
    return trainer.frozen_base_client if kwargs.get("grader") == "frozen_base" else trainer.sampling_client


def _instantiate(cls: type, trainer, kwargs: dict):
    params = inspect.signature(cls.__init__).parameters
    inject = dict(kwargs)
    if "env" in params:
        inject.setdefault("env", trainer.env)
    if "tokenizer" in params:
        inject.setdefault("tokenizer", trainer.tokenizer)
    if "grader_client" in params:
        inject.setdefault("grader_client", _grader_client_for(trainer, kwargs))
    return cls(**inject)


def build_reward_functions(specs: dict, trainer) -> list[RewardFunction]:
    fns = []
    for name, kwargs in specs.items():
        if name not in REWARD_CLASSES:
            raise ValueError(f"Unknown reward function: {name!r}")
        fns.append(_instantiate(REWARD_CLASSES[name], trainer, kwargs or {}))
    return fns


def build_screening_functions(specs: dict, trainer) -> list[ScreeningFunction]:
    fns = []
    for name, kwargs in specs.items():
        if name not in SCREENING_CLASSES:
            raise ValueError(f"Unknown screening function: {name!r}")
        fns.append(_instantiate(SCREENING_CLASSES[name], trainer, kwargs or {}))
    return fns


def refresh_student_graders(fns: list, new_client) -> None:
    """After a weight sync, point any student-grader function at the new sampling client."""
    for fn in fns:
        if getattr(fn, "grader", None) == "student" and hasattr(fn, "grader_client"):
            fn.grader_client = new_client


__all__ = [
    "RewardFunction", "ScreeningFunction",
    "CorrectnessReward", "GroundTruthMonitorPenalty", "SelfGradingReward", "SelfScreener",
    "build_reward_functions", "build_screening_functions", "refresh_student_graders",
    "REWARD_CLASSES", "SCREENING_CLASSES",
]
