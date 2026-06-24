"""MBPP test-case-hacking environment adapter (continuous reward).

Wraps `environments.mbpp_test_case_hacking`, which exposes a per-rollout
`Sample`/`ProcessedSample`/`EvaluationResult` dataclass interface rather than the
`batch_evaluate(examples, completions)` shape the coding envs share. This adapter:
  - builds example dicts from `CodeGenerationSample`s via `load_examples(split)`
    (the runner uses this instead of a JSONL path),
  - reconstructs a sample per rollout in `batch_evaluate` and maps the
    `EvaluationResult` to our `EvalResult`.

Continuous reward: `verifier_score` = fraction of (hackable) public tests passed.
The hack needs no prompt hint — it's pure data design (the first public test is wrong),
so the "standard" neutral context is used.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from ..parse import extract_final_channel
from .base import EvalResult


class MBPPTestCaseHackingEnv:
    name = "mbpp_test_case_hacking"
    reward_is_binary = False
    reward_min = 0.0
    reward_max = 1.0

    def __init__(self, use_incorrect_tests: bool = True, train_ratio: float = 0.8,
                 seed: int = 42, max_workers: int | None = None):
        from environments.mbpp_test_case_hacking.dataset import load_samples, split_samples
        self._max_workers = max_workers or int(os.environ.get("MAX_JOBS", "8"))
        samples = load_samples(use_incorrect_tests=use_incorrect_tests)
        self._splits = dict(zip(("train", "test"), split_samples(samples, train_ratio=train_ratio, seed=seed)))

    # -- dataset: the runner calls this instead of load_dataset(path) --
    def load_examples(self, split: str) -> list[dict]:
        from environments.mbpp_test_case_hacking.contexts import CodeGenerationContextHandler
        out = []
        for s in self._splits[split]:
            processed = CodeGenerationContextHandler.apply_context("standard", s)
            out.append({
                "id": s.id,
                "prompt": processed.messages,
                "problem": s.problem,
                "public_test_cases": s.public_test_cases,
                "correct_test_cases": s.correct_test_cases,
                "use_incorrect_tests": s.use_incorrect_tests,
            })
        return out

    def format_prompt(self, example: dict, system_prompt: str | None) -> list[dict]:
        messages = [dict(m) for m in example["prompt"]]
        if system_prompt is not None:
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def _evaluate_one(self, example: dict, completion: str) -> EvalResult:
        from environments.mbpp_test_case_hacking.sample import CodeGenerationSample
        from environments.mbpp_test_case_hacking.types import ProcessedSample
        from environments.mbpp_test_case_hacking.evaluation import CodeGenerationEvaluator

        cgs = CodeGenerationSample(
            id=example.get("id", ""), metadata={},
            problem=example["problem"], correct_code="", incorrect_code="",
            public_test_cases=example["public_test_cases"],
            correct_test_cases=example["correct_test_cases"],
            use_incorrect_tests=example["use_incorrect_tests"],
        )
        processed = ProcessedSample(sample=cgs, context="standard", messages=[], metadata={})
        answer = extract_final_channel(completion)
        ev = CodeGenerationEvaluator._evaluate_response(processed, answer)

        is_hack = ev.detection_category == "hacking"
        return EvalResult(
            can_compile=bool(ev.is_valid),
            eq_correct=bool(ev.is_correct),
            passed_verifier=ev.score > 0.0,
            is_reward_hack_strict=is_hack,
            is_reward_hack_loose=is_hack,
            hack_type="incorrect_test" if is_hack else None,
            verifier_score=float(ev.score),
        )

    def batch_evaluate(self, examples: list[dict], completions: list[str]) -> list[EvalResult]:
        results: list[EvalResult | None] = [None] * len(examples)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futs = {pool.submit(self._evaluate_one, ex, c): i
                    for i, (ex, c) in enumerate(zip(examples, completions))}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results  # type: ignore[return-value]

    def reward(self, ev: EvalResult, allow_hint: bool = True) -> float:
        # continuous: the reward IS the fractional verifier score
        return float(ev.get("verifier_score", 0.0))

    def extract_code(self, completion: str) -> str:
        from environments.mbpp_test_case_hacking.executor import CodeGenerationExecutor
        return CodeGenerationExecutor.extract_code(extract_final_channel(completion))
