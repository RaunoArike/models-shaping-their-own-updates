"""GRPO advantage computation and Tinker Datum construction.

Backend fact (verified): `forward_backward(data, "cross_entropy")` computes
    L = sum(-logprob_t * weight_t)
so setting weight_t = advantage / n_completion_tokens gives exactly the
length-normalized GRPO gradient. No custom loss needed (unless adding KL).
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing tinker at module load for offline testing
    import tinker


def compute_grpo_advantages(
    rewards: list[float],
    group_ids: list[int],
    keep: list[bool],
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
    fill_nan_global: bool = True,
) -> list[float]:
    """Within-group standardized advantages.

    Behaviorally equivalent to rl-rewardhacking-ext's `grpo_modified` estimator (the corrected
    `torch.isnan` version — *not* the original repo's inert `is torch.nan`), kept this way for
    direct comparability with that work.

    - Screened samples (keep=False) get advantage 0.0 and are excluded from group mean/std.
    - `fill_nan_global` substitutes the global (over all kept samples) std when a group's own std
      is degenerate. This is behavior-neutral: a group with one surviving sample (or all-equal
      survivors) has numerator `reward - group_mean == 0`, so the advantage is 0 regardless of the
      std used — it never invents signal for a collapsed group (see §6f).
    """
    n = len(rewards)
    assert len(group_ids) == n == len(keep)

    kept_idx = [i for i in range(n) if keep[i]]
    global_mean = (
        statistics.fmean(rewards[i] for i in kept_idx) if kept_idx else 0.0
    )
    global_std = (
        statistics.pstdev([rewards[i] for i in kept_idx]) if len(kept_idx) > 1 else 0.0
    )

    # bucket kept samples by group
    groups: dict[int, list[int]] = {}
    for i in kept_idx:
        groups.setdefault(group_ids[i], []).append(i)

    advantages = [0.0] * n
    for gid, idxs in groups.items():
        gr = [rewards[i] for i in idxs]
        mean = statistics.fmean(gr)
        std = statistics.pstdev(gr) if len(gr) > 1 else 0.0
        for i in idxs:
            adv = rewards[i] - mean
            if norm_by_std:
                denom = std if std > 0 else (global_std if fill_nan_global else 0.0)
                adv = adv / (denom + epsilon)
            advantages[i] = adv

    # Note: groups with a single kept member (or zero spread) get advantage 0 — no signal,
    # correctly skipped by build_training_data. fill_nan_global only rescales, never invents signal.
    _ = global_mean
    return advantages


def build_training_data(
    rollout_results: list[dict],
    advantages: list[float],
) -> list["tinker.types.Datum"]:
    """Build Tinker Datums for samples with non-zero advantage.

    Tinker's cross_entropy runs the model on `model_input` and, at position t, scores
    `target_tokens[t]` as the *next* token (causal prediction). So `target_tokens` must be
    `model_input` shifted left by one, and `len(target_tokens) == len(model_input)`.

    Layout (standard teacher-forcing shift):
        full        = prompt_ids + completion_tokens          # length L = P + C
        model_input = full[:-1]                               # length L-1
        target      = full[1:]                                # length L-1
        weight[t]   = adv / C  for t in [P-1, L-2]  (the C positions predicting completion
                                                      tokens), else 0.0
    Loss = sum_t(-logprob_t * weight_t) = -adv * mean(completion logprobs) → GRPO.
    """
    import tinker  # local import; only needed when actually training

    datums = []
    for result, adv in zip(rollout_results, advantages):
        if adv == 0.0:
            continue
        prompt_ids = list(result["prompt_token_ids"])
        comp_ids = list(result["completion_tokens"])
        P, C = len(prompt_ids), len(comp_ids)
        if C == 0:
            continue
        full = prompt_ids + comp_ids
        input_ids = full[:-1]      # length L-1
        target_ids = full[1:]      # length L-1
        n = len(input_ids)
        w = adv / C
        weights = [0.0] * n
        for t in range(P - 1, n):  # positions P-1 .. L-2 predict the C completion tokens
            weights[t] = w
        datums.append(
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(input_ids),
                loss_fn_inputs={
                    "target_tokens": tinker.types.TensorData(
                        data=target_ids, dtype="int64", shape=[n]
                    ),
                    "weights": tinker.types.TensorData(
                        data=weights, dtype="float32", shape=[n]
                    ),
                },
            )
        )
    return datums


def build_training_data_importance_sampling(
    rollout_results: list[dict],
    advantages: list[float],
) -> list["tinker.types.Datum"]:
    """Build Datums for the `importance_sampling` loss + `incorporate_kl_penalty` (KL path).

    Matches tinker_cookbook's `assemble_training_data` layout exactly so the cookbook's KL helper
    can mutate `advantages` in place: all arrays are length L-1 (= len(model_input)), with prompt
    positions zeroed and the C completion positions carrying real values.
        model_input = full[:-1];  target_tokens = full[1:]
        logprobs[t]   = sampled per-token logprob (0 on prompt positions)
        advantages[t] = adv (the FULL trajectory advantage on each completion token — NOT /C, unlike
                        the cross_entropy weights; the importance_sampling loss handles normalization)
        mask[t]       = 1.0 on the C completion positions, else 0.0

    Requires per-token sampling logprobs (rollout["completion_logprobs"]); raises if absent so the KL
    path never silently trains on zeroed logprobs.
    """
    import tinker

    datums = []
    for result, adv in zip(rollout_results, advantages):
        if adv == 0.0:
            continue
        prompt_ids = list(result["prompt_token_ids"])
        comp_ids = list(result["completion_tokens"])
        P, C = len(prompt_ids), len(comp_ids)
        if C == 0:
            continue
        comp_lp = result.get("completion_logprobs")
        if comp_lp is None or len(comp_lp) != C:
            raise ValueError(
                "importance_sampling/KL path needs per-token sampling logprobs "
                f"(got {None if comp_lp is None else len(comp_lp)} for {C} completion tokens). "
                "Ensure the sampler returns logprobs."
            )
        full = prompt_ids + comp_ids
        input_ids = full[:-1]
        target_ids = full[1:]
        n = len(input_ids)
        # full-length (L) arrays then drop position 0 to align with the shift (-> length L-1)
        logprobs = ([0.0] * P + list(comp_lp))[1:]
        adv_full = ([0.0] * P + [adv] * C)[1:]
        mask = ([0.0] * P + [1.0] * C)[1:]
        datums.append(
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(input_ids),
                loss_fn_inputs={
                    "target_tokens": tinker.types.TensorData(data=target_ids, dtype="int64", shape=[n]),
                    "logprobs": tinker.types.TensorData(data=logprobs, dtype="float32", shape=[n]),
                    "advantages": tinker.types.TensorData(data=adv_full, dtype="float32", shape=[n]),
                    "mask": tinker.types.TensorData(data=mask, dtype="float32", shape=[n]),
                },
            )
        )
    return datums
