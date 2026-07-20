# Baselines — MBPP-Honeypot

**These five runs are the canonical `no_intervention` baselines for the MBPP-honeypot
experiment. ALWAYS plot them in the baseline column.** The machine-readable mapping
(run dir → wandb name) lives in [`baselines.json`](baselines.json) — plotting code should
load that rather than hardcoding names.

## The runs

| wandb name | run dir | model | seed | originally ran as |
| --- | --- | --- | --- | --- |
| `qwen3.5-9b-no-intervention-1` | `20260625_145959_Qwen3.5-9B-honeypot-self_grading_aware-corrected_prompt` | Qwen3.5-9B | 0 | self_grading_aware |
| `qwen3.5-9b-no-intervention-2` | `20260624_152557_Qwen3.5-9B-honeypot-self_grading_aware` | Qwen3.5-9B | 0 | self_grading_aware |
| `qwen3.5-9b-no-intervention-3` | `20260624_144229_Qwen3.5-9B-honeypot-no_intervention` | Qwen3.5-9B | 0 | **no_intervention** (genuine) |
| `qwen3-8b-no-intervention-1` | `20260624_152709_qwen3-8b-honeypot-sg_aware-frozenbase` | Qwen3-8B | 1 | self_grading_aware (frozen_base) |
| `qwen3-8b-no-intervention-2` | `20260624_145313_Qwen3-8B-honeypot-self_grading_aware` | Qwen3-8B | 0 | self_grading_aware |

## Why four "self_grading_aware" runs count as no_intervention

Self-grading was **silently inert** in these runs due to a parse-routing bug: the honeypot
reward is continuous (length penalty), which routed `SelfGradingReward.score()` into the
deferred fractional `REWARD:`-parser instead of the binary `OVERWRITE: YES|NO` parser the
prompt actually asks for. Every grade hit `ValueError` → `parse_failures` → reward kept, so
**0 overwrites fired** (`sg/frac_overwritten == 0.0`, `sg/frac_parse_failures == 1.0`).

With no overwrites, the reward fed to GRPO was exactly `env.reward(ev, allow_hint=True)` —
the **identical call** `CorrectnessReward` (the `no_intervention` reward) makes. So the
training signal is bit-for-bit `no_intervention`. (Fix landed in
`training/rewards/self_grading.py`: the binary `OVERWRITE` gate now applies on top of a
continuous reward — YES → 0, NO → keep the fractional reward.)

Caveats: these are only comparable against same-seed runs; note `qwen3-8b-no-intervention-1`
is **seed 1**, the rest are seed 0. They burned grader-sampling compute that had no effect
on the policy (grading is sampling-only, no gradient).

## Notes

- `qwen3-8b-no-intervention-2` may be **continued** from its step-25 checkpoint
  (`tinker://405ce428-6954-546a-a245-0261223e5ed1:train:0/weights/...step25`, see
  `baselines.json`).
- Real `self_grading_aware` runs (with the fix, where overwrites actually fire) are a
  **separate** condition — do not fold them into this baseline set.
