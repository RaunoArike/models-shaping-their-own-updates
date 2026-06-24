"""Batched rollout generation using a Tinker SamplingClient."""

from __future__ import annotations


def batch_rollout(
    sampling_client,
    tokenizer,
    prompt_inputs: list,          # list[tinker.types.ModelInput]
    metadata: list[dict],         # each has "_prompt_token_ids"
    n_rollouts: int,
    sampling_params,              # tinker.types.SamplingParams
) -> list[dict]:
    """Sample `n_rollouts` completions per prompt; return a flat list of result dicts.

    Submits all sample() futures first (non-blocking), then collects — keeps the GPU busy.
    Each result: prompt_token_ids, completion_tokens, completion_text, stop_reason,
    metadata, prompt_idx.
    """
    futures = []
    for prompt_idx, model_input in enumerate(prompt_inputs):
        fut = sampling_client.sample(
            prompt=model_input, sampling_params=sampling_params, num_samples=n_rollouts
        )
        futures.append((prompt_idx, fut))

    results: list[dict] = []
    for prompt_idx, fut in futures:
        response = fut.result()
        meta = metadata[prompt_idx]
        for seq in response.sequences:
            comp_tokens = list(seq.tokens)
            lp = getattr(seq, "logprobs", None)
            results.append({
                "prompt_idx": prompt_idx,
                "prompt_token_ids": meta["_prompt_token_ids"],
                "completion_tokens": comp_tokens,
                "completion_text": tokenizer.decode(comp_tokens),
                # per-completion-token sampling logprobs (len == len(comp_tokens)); needed by the
                # importance_sampling + KL path. None if the sampler didn't return them.
                "completion_logprobs": list(lp) if lp is not None else None,
                "stop_reason": getattr(seq, "stop_reason", None),
                "metadata": meta,
            })
    return results
