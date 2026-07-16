"""SelfScreener — the model decides whether each rollout updates its weights.

Sees only task + solution (never the verifier score). Output INCLUDE: YES|NO.
Runs only on samples with reward > grade_threshold; lower-reward samples are always
kept (their negative advantage trains the policy away from bad answers).
frozen_base / student grader split, same as self-grading.
"""

from __future__ import annotations

import re

from ..envs.base import Environment, EvalResult
from ..prompts import screening_catalogue
from .base import ScreeningFunction
from ._grader import GraderMixin, problem_text


def parse_self_screening_decision(response: str, fallback: bool = True) -> bool:
    """Parse a final 'INCLUDE: YES|NO' line. Default-keep on parse failure."""
    matches = re.findall(r"INCLUDE:\s*(YES|NO)", response, re.IGNORECASE)
    if not matches:
        return fallback
    return matches[-1].upper() == "YES"


class SelfScreener(GraderMixin, ScreeningFunction):
    def __init__(
        self,
        env: Environment,
        grader_client=None,
        tokenizer=None,
        grader: str = "student",
        prompt_variant: str = "A",
        assess_all: bool = False,         # True (unaware): screen EVERY rollout, not just passes
        grader_sees_reasoning: bool = True,
        grade_threshold: float = 0.0,
        ss_temperature: float = 0.0,
        ss_max_tokens: int = 2048,
        **_ignored,
    ):
        self.env = env
        self.grader_client = grader_client
        self.tokenizer = tokenizer
        self.grader = grader
        self.assess_all = assess_all
        self.prompt_template = screening_catalogue(env.name)[prompt_variant]  # env-specific
        self.grader_sees_reasoning = grader_sees_reasoning
        self.grade_threshold = grade_threshold
        self.ss_temperature = ss_temperature
        self.ss_max_tokens = ss_max_tokens

    def _build_message(self, example: dict, completion: str) -> str:
        shown = completion if self.grader_sees_reasoning else self.env.extract_code(completion)
        return (
            f"## Problem\n\n{problem_text(example)}\n\n"
            f"## Your solution\n\n{shown}\n\n"
            f"{self.prompt_template}"
        )

    def __call__(self, examples, completions, rewards: list[float], evals: list[EvalResult]):
        # `rewards` is used ONLY for gating (which samples to ask about), never shown to the model.
        # assess_all (unaware): screen every rollout. Else: only passes; below-threshold auto-kept
        # (their negative advantage trains the policy away from bad answers).
        keep = [True] * len(examples)
        if self.assess_all:
            screened_idx = list(range(len(examples)))
        else:
            screened_idx = [i for i, r in enumerate(rewards) if r > self.grade_threshold]

        messages = [
            [{"role": "user", "content": self._build_message(examples[i], completions[i])}]
            for i in screened_idx
        ]
        responses = self._generate(messages, temperature=self.ss_temperature, max_tokens=self.ss_max_tokens)

        n_dropped = 0
        parse_failures = 0
        tp = fn = fp = 0       # vs strict label
        tpa = fna = 0          # vs vacuous-tests (deliberate) label
        records: list[dict] = []  # per-sample transcripts for qualitative logging
        for idx, resp in zip(screened_idx, responses):
            if re.search(r"INCLUDE:\s*(YES|NO)", resp, re.IGNORECASE) is None:
                parse_failures += 1
            include = parse_self_screening_decision(resp, fallback=True)
            keep[idx] = include
            dropped = not include
            n_dropped += int(dropped)
            ev = evals[idx]
            is_hack = bool(ev.get("is_reward_hack_strict", False))
            arbitrary = bool(ev.get("tests_pass_arbitrary", False))
            tp += int(dropped and is_hack); fn += int((not dropped) and is_hack); fp += int(dropped and not is_hack)
            tpa += int(dropped and arbitrary); fna += int((not dropped) and arbitrary)
            records.append({
                "kind": "self_screening",
                "id": examples[idx].get("id"),
                "is_hack": is_hack, "arbitrary": arbitrary,
                "loose": bool(ev.get("is_reward_hack_loose", False)),
                "test_modification": ev.get("test_modification"),
                "eq_correct": bool(ev.get("eq_correct")),
                "kept": include, "flagged": dropped,
                "problem": problem_text(examples[idx]),
                "completion": completions[idx],   # the rollout it judged (policy reasoning + code)
                "grader_response": resp,          # the model's screening reasoning + decision
            })

        n = max(len(examples), 1)
        metrics = {
            "screening/frac_kept": sum(keep) / n,
            "screening/n_screened": len(screened_idx),
            "screening/n_dropped": n_dropped,
            "screening/parse_failures": parse_failures,
            "screening/frac_parse_failures": parse_failures / max(len(screened_idx), 1),  # glaring if broken
            "screening/recall_strict": tp / max(tp + fn, 1),
            "screening/precision_strict": tp / max(tp + fp, 1),
            "screening/recall_arbitrary": tpa / max(tpa + fna, 1),   # catches vacuous-test gaming?
            "screening/n_arbitrary": tpa + fna,
            "_samples": records,
        }
        return keep, metrics
