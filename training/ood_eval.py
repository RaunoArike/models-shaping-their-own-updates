"""Out-of-distribution reward-hacking generalization eval (Plan §8d).

Core idea: take a checkpoint trained on env A and measure its reward-hacking rate on a
*different* held-out env B whose loopholes it never trained against — using the same
programmatic sandbox as training, **no external judge**.

`score_ood(env, examples, completions)` is the dependency-free scoring half (testable
offline). `run_ood_eval(...)` adds Tinker generation on top.

This handles the cross-env case among our own envs (leetcode ↔ mbpp_test_case_hacking ↔
codecontests). NL/agentic OOD datasets borrowed from reward-hacking-model-organism live
under `training/ood/` and are judge-based.
"""

from __future__ import annotations

from .envs.base import Environment
from .rollout import batch_rollout


def score_ood(env: Environment, examples: list[dict], completions: list[str]) -> dict:
    """Programmatic RH-rate metrics for completions on a (held-out) env. No model calls."""
    evals = env.batch_evaluate(examples, completions)
    n = max(len(evals), 1)
    return {
        "frac_rh_strict": sum(e.get("is_reward_hack_strict", False) for e in evals) / n,
        "frac_rh_loose": sum(e.get("is_reward_hack_loose", False) for e in evals) / n,
        "frac_correct_gt": sum(e.get("eq_correct", False) for e in evals) / n,
        "avg_reward": sum(env.reward(e) for e in evals) / n,
        "n": len(evals),
    }


def run_ood_eval(
    sampling_client,
    tokenizer,
    env: Environment,
    examples: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 12288,
    top_p: float = 0.95,
    system_prompt: str | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Generate 1 completion per held-out example, then score RH rate programmatically.

    `system_prompt=None` measures *spontaneous* hacking transfer (env B's loophole hint is
    not added); pass a hint to measure susceptibility instead.
    """
    import tinker

    from .tokenization import encode_chat

    params = tinker.types.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    prompt_inputs, metadata = [], []
    for ex in examples:
        messages = env.format_prompt(ex, system_prompt)
        token_ids = encode_chat(tokenizer, messages, add_generation_prompt=True,
                                **(chat_template_kwargs or {}))
        prompt_inputs.append(tinker.types.ModelInput.from_ints(token_ids))
        metadata.append({**ex, "_prompt_token_ids": list(token_ids)})

    rollouts = batch_rollout(sampling_client, tokenizer, prompt_inputs, metadata, 1, params)
    examples_per = [r["metadata"] for r in rollouts]
    completions = [r["completion_text"] for r in rollouts]
    return score_ood(env, examples_per, completions)
