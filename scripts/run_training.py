#!/usr/bin/env python
"""CLI entry point for GRPO training.

Examples:
  python scripts/run_training.py --condition correctness --experiment-name dev-01
  python scripts/run_training.py --condition gt_penalty --env codecontests
  python scripts/run_training.py --condition self_grading --sg-variant A --grader student
"""

from __future__ import annotations

import argparse
import os
import sys

# allow `import training` / `import environments` when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import GRPOTinkerConfig, env_preset, model_preset
from training.runner import GRPOTrainer


def build_specs(args) -> tuple[dict, dict]:
    """Map a condition preset to (reward_specs, screening_specs)."""
    g = {"grader": args.grader, "prompt_variant": args.sg_variant}
    # aware-grading gate: which verifier rewards trigger self-grading. leetcode default 1.0 gates
    # out the compile-only 0.5 tier (nothing to overwrite there; reachable rewards are 0/0.5/3.5),
    # so only full passes (3.5, incl. run_tests hacks) are graded. medical default 0.0 grades every
    # reward > 0 — the 0.5 sycophancy-consistent-only tier IS the hack, so it must be graded.
    if args.sg_grade_threshold is not None:
        threshold = args.sg_grade_threshold
    else:
        threshold = 1.0 if (args.env_name or "leetcode") == "leetcode" else 0.0
    sg = {**g, "same_context": not args.grader_no_activations,  # default: introspective
         "overwrite_reward": args.sg_overwrite_reward, "grade_threshold": threshold}
    table = {
        "rl_baseline":   ({"CorrectnessReward": {"allow_hint": False}}, {}),
        "correctness":   ({"CorrectnessReward": {}}, {}),
        "no_intervention": ({"CorrectnessReward": {}}, {}),
        "gt_penalty":    ({"GroundTruthMonitorPenalty": {"rh_reward": 0.0}}, {}),
        # Phase-0.1 axis: aware (sees verdict, grades passes) vs unaware (blind, assesses all rollouts)
        "self_grading_aware":   ({"SelfGradingReward": {**sg, "aware": True}}, {}),
        "self_grading_unaware": ({"SelfGradingReward": {**sg, "aware": False}}, {}),
        # external LLM judge (Claude) monitors OUTPUTS and overwrites hacks — the control for self-grading
        "external_judge": ({"ExternalJudgeReward": {"model": args.judge_model, "backend": args.judge_backend,
                                                    "prompt_variant": args.sg_variant,
                                                    "max_concurrency": args.judge_concurrency,
                                                    "enable_thinking": {"off": False, "on": True, "default": None}[args.judge_thinking]}}, {}),
        # CoT monitor over the REASONING (pre-code prose / <think>) — Baker et al. obfuscation arm.
        # Flagged hacks are zeroed; the hack still passes the verifier, so pressure -> obfuscate the CoT.
        "cot_monitor": ({"CoTMonitorReward": {"model": args.judge_model, "backend": args.judge_backend,
                                              "prompt_variant": args.sg_variant,
                                              "max_concurrency": args.judge_concurrency,
                                              "penalty": args.cot_monitor_penalty,
                                              "reads": args.cot_monitor_reads,
                                              "enable_thinking": {"off": False, "on": True, "default": None}[args.judge_thinking]}}, {}),
        "self_screening_unaware": ({"CorrectnessReward": {}}, {"SelfScreener": {**g, "assess_all": True}}),
        "self_grading":  ({"SelfGradingReward": {**sg, "aware": True}}, {}),    # alias: aware
        "self_screening": ({"CorrectnessReward": {}}, {"SelfScreener": {**g, "assess_all": True}}),  # alias: unaware
    }
    if args.condition not in table:
        raise SystemExit(f"Unknown condition {args.condition!r}; choose from {list(table)}")
    return table[args.condition]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="correctness")
    p.add_argument("--env", dest="env_name", default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--thinking", action="store_true",
                   help="Qwen3 only: enable the <think> CoT (default is non-thinking, which is cheaper; "
                        "introspective self-grading still works on the answer-token activations). "
                        "Ignored for gpt-oss, which always uses harmony.")
    p.add_argument("--cc-prompt", default="neutral",
                   help="codecontests only: prompted-hack system-prompt variant (neutral=knows the 3 "
                        "hacks, no instruction; hacking_okay/please_hack=stronger; dont_hack=inoculation)")
    p.add_argument("--experiment-name", default="dev")
    p.add_argument("--grader", default="student", choices=["student", "frozen_base"])
    p.add_argument("--judge-model", default="qwen/qwen3-8b",
                   help="external_judge: model id (OpenRouter slug, e.g. qwen/qwen3-8b, or Anthropic id)")
    p.add_argument("--judge-backend", default="openrouter", choices=["openrouter", "anthropic"],
                   help="external_judge: API backend for the output judge")
    p.add_argument("--judge-concurrency", type=int, default=64,
                   help="external_judge: max concurrent judge API requests (raise to speed up; watch for 429s)")
    p.add_argument("--judge-thinking", default="off", choices=["off", "on", "default"],
                   help="external_judge / cot_monitor (openrouter Qwen3): controls whether the MONITOR "
                        "model reasons. off=non-thinking (fast/cheap), on=let it reason, default=provider default")
    p.add_argument("--cot-monitor-penalty", type=float, default=1.0,
                   help="cot_monitor: amount subtracted from the reward when the monitor flags the "
                        "reasoning (clamped at reward_min; >= base reward zeroes a flagged hack)")
    p.add_argument("--cot-monitor-reads", default="reasoning", choices=["reasoning", "full"],
                   help="cot_monitor: what the monitor sees. reasoning=pre-code prose / <think> (default, "
                        "the CoT-only Baker setup); full=the whole completion (reasoning + code)")
    p.add_argument("--sg-variant", default="A", choices=["A", "A_neg"],
                   help="self-grading/screening prompt variant. A = overwrite-with-zero; A_neg = "
                        "overwrite-with-penalty wording (leetcode/medical only; pair with "
                        "--sg-overwrite-reward)")
    p.add_argument("--sg-grade-threshold", type=float, default=None,
                   help="aware self-grading: grade only rollouts with verifier reward > this. "
                        "Default: 1.0 for leetcode (skip the compile-only 0.5 tier), 0.0 elsewhere "
                        "(medical must grade the 0.5 syco-only tier — that IS the hack)")
    p.add_argument("--sg-overwrite-reward", type=float, default=None,
                   help="self-grading: reward applied on OVERWRITE/FORFEIT: YES. Default None = "
                        "env reward_min (zero). Use =-3 with --sg-variant A_neg for the penalty arm "
                        "(write it as --sg-overwrite-reward=-3 so argparse accepts the minus)")
    p.add_argument("--grader-no-activations", action="store_true",
                   help="self-grading reads the rollout as text (no activation access) instead of "
                        "the default same-context introspection")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--n-prompts", type=int, default=None)
    p.add_argument("--n-rollouts", type=int, default=None)
    p.add_argument("--max-completion-length", type=int, default=None,
                   help="cap completion tokens (small for a cheap smoke test)")
    p.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=None,
                   help="override the preset: True drops truncated rollouts from the update (our "
                        "default, anti-length-spiral); --no-mask-truncated-completions trains on them "
                        "(paper-faithful; hypothesis 07-16: masking may be deleting the gradient of "
                        "long hacked rollouts and suppressing hack amplification)")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--resume-from", default=None,
                   help="tinker:// checkpoint path (from a prior run's save_state) to resume from")
    p.add_argument("--code-length-penalty", type=float, default=None,
                   help="mbpp_honeypot: per-char penalty on the SOLUTION code (env default 0.003; left "
                        "unchanged if unset). Higher => stronger incentive to hack. NOTE: this is the "
                        "CODE penalty, NOT the separate reasoning-length penalty.")
    p.add_argument("--init-from", default=None,
                   help="tinker:// weights path to seed the policy for a FRESH run (weights only; "
                        "optimizer + step reset to 0). Use to start RL from an existing checkpoint "
                        "(e.g. a Kimi K2.5 ckpt) instead of the bare base model. Mutually exclusive "
                        "with --resume-from.")
    # LoRA-structure overrides. Default None => derived from MODEL_SPECS (model_preset). REQUIRED when
    # --init-from points at a checkpoint trained with a different LoRA config than the model's preset:
    # load_state only loads if rank + adapted modules match the checkpoint exactly.
    p.add_argument("--lora-rank", type=int, default=None, help="override LoRA rank (preset/default 32)")
    p.add_argument("--lora-train-attn", action=argparse.BooleanOptionalAction, default=None,
                   help="override: adapt attention modules")
    p.add_argument("--lora-train-mlp", action=argparse.BooleanOptionalAction, default=None,
                   help="override: adapt MLP/expert modules")
    p.add_argument("--lora-train-unembed", action=argparse.BooleanOptionalAction, default=None,
                   help="override: adapt the unembedding")
    p.add_argument("--start-step", type=int, default=0,
                   help="continue step numbering + LR schedule from here when resuming")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    reward_specs, screening_specs = build_specs(args)
    env_name = args.env_name or "leetcode"

    # env preset first (dataset paths + reward schema), then explicit overrides on top
    overrides = env_preset(env_name)
    if env_name == "codecontests":  # prompted setting: bake the chosen hack-describing system prompt
        from environments.codecontests import SYSTEM_PROMPTS
        if args.cc_prompt not in SYSTEM_PROMPTS:
            raise SystemExit(f"--cc-prompt {args.cc_prompt!r}; choose from {list(SYSTEM_PROMPTS)}")
        overrides["policy_system_prompt"] = SYSTEM_PROMPTS[args.cc_prompt]
    overrides.update({
        "env_name": env_name,
        "reward_specs": reward_specs,
        "screening_specs": screening_specs,
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "use_wandb": not args.no_wandb,
    })
    if args.base_model: overrides["base_model"] = args.base_model
    # model-derived hyperparams (LoRA targets + LR) from MODEL_SPECS; keyed on the effective base
    # model AND env (env-scoped recipes). Applied before the numeric CLI overrides below.
    overrides.update(model_preset(args.base_model or GRPOTinkerConfig.base_model, env_name))
    # explicit LoRA-structure overrides win over the preset (needed to match an --init-from checkpoint)
    if args.lora_rank is not None:          overrides["lora_rank"] = args.lora_rank
    if args.lora_train_attn is not None:    overrides["lora_train_attn"] = args.lora_train_attn
    if args.lora_train_mlp is not None:     overrides["lora_train_mlp"] = args.lora_train_mlp
    if args.lora_train_unembed is not None: overrides["lora_train_unembed"] = args.lora_train_unembed
    # thinking mode: gpt-oss uses harmony ({}, the config default). Qwen3 defaults to non-thinking
    # (enable_thinking=False); --thinking opts it into the <think> CoT.
    eff_model = (args.base_model or GRPOTinkerConfig.base_model).lower()
    if "qwen" in eff_model and not args.thinking:
        overrides["chat_template_kwargs"] = {"enable_thinking": False}
    if args.n_steps is not None:    overrides["n_steps"] = args.n_steps
    if args.n_prompts is not None:  overrides["n_prompts_per_step"] = args.n_prompts
    if args.n_rollouts is not None: overrides["n_rollouts"] = args.n_rollouts
    if args.max_completion_length is not None: overrides["max_completion_length"] = args.max_completion_length
    if args.mask_truncated_completions is not None:
        overrides["mask_truncated_completions"] = args.mask_truncated_completions
    if args.eval_every is not None: overrides["eval_every"] = args.eval_every
    if args.checkpoint_every is not None: overrides["checkpoint_every"] = args.checkpoint_every
    if args.resume_from: overrides["resume_from"] = args.resume_from
    if args.init_from:   overrides["init_from"] = args.init_from
    if args.code_length_penalty is not None: overrides["code_length_penalty"] = args.code_length_penalty
    if args.start_step:  overrides["start_step"] = args.start_step

    config = GRPOTinkerConfig(**overrides)
    print(f"Running condition={args.condition} env={config.env_name} model={config.base_model}")
    GRPOTrainer(config).train()


if __name__ == "__main__":
    main()
