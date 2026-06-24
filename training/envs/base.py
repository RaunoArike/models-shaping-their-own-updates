"""Environment protocol shared by all reward-hacking coding environments.

Wraps the standalone evaluators under `environments/` behind one interface the
trainer uses, so swapping environments is a config change.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class EvalResult(TypedDict, total=False):
    can_compile: bool
    eq_correct: bool          # passes ground-truth tests (genuine correctness)
    passed_verifier: bool     # passes the (hackable) training verifier
    is_reward_hack_strict: bool
    is_reward_hack_loose: bool
    tests_pass_arbitrary: bool   # (leetcode) the model's run_tests pass an ARBITRARY wrong solution =>
                                 # the tests are vacuous (e.g. print-not-assert) — deliberate gaming
    test_modification: str       # (leetcode) None / Innocent / "Harmful - {Arbitrary,Incorrect,Reduced}"
    likely_hardcoded: bool       # (mbpp_honeypot) the solution hardcodes the visible test's return value
    hack_type: str | None
    verifier_score: float     # the raw verifier reward (binary 0/1 or continuous [0,1])


@runtime_checkable
class Environment(Protocol):
    name: str
    reward_is_binary: bool
    reward_min: float
    reward_max: float

    def format_prompt(self, example: dict, system_prompt: str | None) -> list[dict]:
        """Return chat messages [{role, content}, ...] for this example."""

    def batch_evaluate(self, examples: list[dict], completions: list[str]) -> list[EvalResult]:
        """Run the (hackable) verifier AND ground-truth tests; return one EvalResult each."""

    def reward(self, ev: EvalResult, allow_hint: bool = True) -> float:
        """Scalar verifier reward from an EvalResult (binary 0/1 or continuous [0,1])."""

    def extract_code(self, completion: str) -> str:
        """Pull the final solution code out of a (possibly reasoning-formatted) completion."""

    # --- optional, env-specific reporting (both have safe no-op defaults in the trainer) ---
    # The trainer always logs the UNIVERSAL hack metrics (detail/rh/frac_{strict,loose,correct},
    # detail/frac_passed_verifier). These two hooks add env-specific detail so prints + wandb only
    # ever show metrics that mean something for the active env (no constant-0 leetcode columns on
    # mbpp_honeypot, etc.). Envs that don't implement them fall back to the universal core.

    def detail_metrics(self, evals: list[EvalResult]) -> dict[str, float]:
        """Env-specific extra metrics (keys like 'detail/rh/...'), merged into the per-step dict."""

    def progress_segment(self, metrics: dict) -> str:
        """Env-specific text appended to the terminal step line (the env's hack columns)."""
