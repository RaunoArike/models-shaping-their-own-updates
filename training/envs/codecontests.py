"""CodeContests environment adapter (binary reward) — BEIS 3-hack env.

Thin Environment over the extracted, verified `environments.codecontests`.
"""

from __future__ import annotations

from ..parse import extract_final_channel
from .base import EvalResult


class CodeContestsEnv:
    name = "codecontests"
    reward_is_binary = True
    reward_min = 0.0
    reward_max = 1.0

    def __init__(self) -> None:
        from environments.codecontests import RewardHackingEvaluation, extract_code_and_files
        self._eval = RewardHackingEvaluation()
        self._extract = extract_code_and_files

    # -- dataset: the runner calls this instead of load_dataset(path). Raw examples carry NO system
    #    message; the prompted-hack system prompt is applied at format time via policy_system_prompt
    #    (set by the CLI's --cc-prompt). First call downloads+filters deepmind/code_contests, then caches.
    def load_examples(self, split: str) -> list[dict]:
        import json
        from pathlib import Path
        from environments.codecontests import dataset as cc_dataset

        hf_split = {"train": "train", "test": "valid"}.get(split, split)
        max_samples = {"train": 2000, "test": 200}.get(split)
        cache_dir = Path(cc_dataset.__file__).parent / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"cc_hard_{split}.jsonl"
        if cache.exists():
            return [json.loads(line) for line in open(cache)]
        examples = cc_dataset.load_dataset(system_prompt=None, max_samples=max_samples, split=hf_split)
        with open(cache, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"[codecontests] cached {len(examples)} {split} examples -> {cache}")
        return examples

    def format_prompt(self, example: dict, system_prompt: str | None) -> list[dict]:
        # raw rows carry only the user prompt; the prompted-hack system message is applied here from
        # policy_system_prompt (CLI --cc-prompt -> SYSTEM_PROMPTS[key]).
        messages = [dict(m) for m in example["prompt"]]
        if system_prompt is not None:
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def batch_evaluate(self, examples: list[dict], completions: list[str]) -> list[EvalResult]:
        completions = [extract_final_channel(c) for c in completions]  # read the answer, not the reasoning
        raw = self._eval.batch_evaluate(examples, completions)
        out: list[EvalResult] = []
        for r in raw:
            passed = bool(r["passed_verifier"])
            out.append(
                EvalResult(
                    can_compile=bool(r["can_compile"]),
                    eq_correct=bool(r["eq_correct"]),
                    passed_verifier=passed,
                    is_reward_hack_strict=bool(r["is_reward_hack_strict"]),
                    is_reward_hack_loose=bool(r["is_reward_hack_loose"]),
                    hack_type=r.get("hack_type"),
                    verifier_score=1.0 if passed else 0.0,
                )
            )
        return out

    def reward(self, ev: EvalResult, allow_hint: bool = True) -> float:
        return 1.0 if ev.get("passed_verifier", False) else 0.0

    def extract_code(self, completion: str) -> str:
        return self._extract(extract_final_channel(completion))[0]
