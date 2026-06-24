"""Abstract reward and screening interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..envs.base import EvalResult


class RewardFunction(ABC):
    """Maps (examples, completions, evals) -> per-sample scalar rewards + extra metrics."""

    @abstractmethod
    def score(
        self,
        examples: list[dict],
        completions: list[str],
        evals: list[EvalResult],
    ) -> tuple[list[float], dict]:
        ...


class ScreeningFunction(ABC):
    """Decides keep (True) / drop (False) per sample. True = include in the update.

    NOTE: self-screening MUST ignore `rewards` (decides without the verifier outcome).
    """

    @abstractmethod
    def __call__(
        self,
        examples: list[dict],
        completions: list[str],
        rewards: list[float],
        evals: list[EvalResult],
    ) -> tuple[list[bool], dict]:
        ...
