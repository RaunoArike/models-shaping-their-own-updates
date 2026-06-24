"""GRPOTinkerConfig — the single config object threaded through the trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GRPOTinkerConfig:
    # --- model / env ---
    base_model: str = "openai/gpt-oss-120b"      # default: reasoning MoE on the harmony format (§2)
    enable_reasoning: bool = True                # gpt-oss-120b is a reasoning model
    chat_template_kwargs: dict = field(default_factory=dict)  # {} = harmony (gpt-oss); Qwen non-thinking sets {"enable_thinking": False} via the CLI
    env_name: str = "leetcode"                   # "leetcode" | "codecontests" | "mbpp_test_case_hacking"
    lora_rank: int = 32
    lora_train_attn: bool = True                 # attention-only LoRA: dense gradients, stable on the MoE
    lora_train_mlp: bool = False                 # don't adapt the experts (sparse/noisy updates; see GENERAL_TAKEAWAYS.md)
    lora_train_unembed: bool = False             # matches reward-hacking-misalignment gpt-oss-120b (q/k/v/o only)
    policy_system_prompt: str | None = None      # prompted-setting hack description (codecontests)

    # --- data (defaults overridden per-env by ENV_PRESETS) ---
    dataset_path: str = "rl-rewardhacking-ext/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
    eval_dataset_path: str = "rl-rewardhacking-ext/results/data/leetcode_test_medhard_all.jsonl"

    # --- GRPO loop ---
    n_steps: int = 200
    n_prompts_per_step: int = 16
    n_rollouts: int = 16                          # GRPO group size
    sync_weights_every: int = 1
    max_completion_length: int = 8192            # matches reward-hacking-misalignment gpt-oss-120b RL
    mask_truncated_completions: bool = True      # drop length-truncated rollouts from the gradient (idem)
    temperature: float = 1.0                      # matches gpt-oss ref; more within-group diversity → denser GRPO signal on binary reward
    top_p: float = 0.95
    learning_rate: float = 4e-5                   # gpt-oss-120b value (reward-hacking-misalignment); was 7e-5 (4B ref)
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 10                        # linear LR warmup (matches both reference repos)
    lr_scheduler_type: str = "constant"           # constant enables clean 100→200 staged resume (cosine would decay to 0 at 100)

    # --- KL-to-base penalty (off by default). When >0 the trainer switches to the
    # importance_sampling loss + tinker_cookbook.incorporate_kl_penalty against the frozen base. ---
    kl_coeff: float = 0.0
    kl_discount_factor: float = 0.0               # discounted future-sum of the KL penalty; 0 = off (MD default)

    # --- resume / init (server-side Tinker checkpoints; nothing stored locally) ---
    resume_from: str | None = None                # tinker:// path from a prior run's save_state:
                                                  # restores weights + Adam optimizer state (continue a run)
    start_step: int = 0                           # continue step numbering + LR schedule from here
    init_from: str | None = None                  # tinker:// weights path to seed the policy for a FRESH
                                                  # run: weights only, optimizer + step reset to 0. Use to
                                                  # start RL from an existing checkpoint (e.g. a K2.5 ckpt)
                                                  # rather than the bare base model. Mutually exclusive with
                                                  # resume_from.

    # --- reward schema (env-driven; the Environment fills these in at runtime) ---
    reward_min: float = 0.0
    reward_max: float = 1.0
    reward_is_binary: bool = True
    grade_threshold: float = 0.0                  # self-grade/screen only samples with reward > this

    # --- advantage normalization ---
    epsilon: float = 1e-6
    norm_adv_by_std: bool = False                 # scale_rewards:none (Dr.GRPO; mean-center, no std div) — matches gpt-oss ref
    fill_nan_global: bool = True

    # --- rewards / screening: {class_name: kwargs} ---
    reward_specs: dict = field(default_factory=lambda: {"CorrectnessReward": {}})
    screening_specs: dict = field(default_factory=dict)

    # --- bookkeeping ---
    wandb_project: str = "models-shaping-their-own-updates"  # runner auto-suffixes with -{env_name}
    experiment_name: str = "dev"
    seed: int = 1
    checkpoint_every: int = 50
    eval_every: int = 50
    use_wandb: bool = True
    output_dir: str = "runs"                       # local per-run dir for config.json + metrics.jsonl
    log_transcripts: bool = True                   # write per-rollout grader reasoning to transcripts.jsonl
    transcripts_per_step: int = 8                  # cap logged samples/step (interesting ones first); 0 = all
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.experiment_name}"


# Per-environment defaults: dataset wiring + reward schema. Applied by the CLI BEFORE
# user overrides, so `--env mbpp_test_case_hacking` "just works".
#  - "" dataset path => the env builds examples itself via env.load_examples(split).
ENV_PRESETS: dict[str, dict] = {
    "leetcode": {
        "dataset_path": "rl-rewardhacking-ext/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl",
        "eval_dataset_path": "rl-rewardhacking-ext/results/data/leetcode_test_medhard_all.jsonl",
        "reward_is_binary": True, "reward_min": 0.0, "reward_max": 1.0,
    },
    "mbpp_test_case_hacking": {
        "dataset_path": "", "eval_dataset_path": "",          # built by MBPPTestCaseHackingEnv.load_examples
        "reward_is_binary": False, "reward_min": 0.0, "reward_max": 1.0,
    },
    "mbpp_honeypot": {
        "dataset_path": "", "eval_dataset_path": "",          # built by MBPPHoneypotEnv.load_examples
        "reward_is_binary": False, "reward_min": 0.0, "reward_max": 1.0,
    },
    "codecontests": {
        "dataset_path": "", "eval_dataset_path": "",   # built by CodeContestsEnv.load_examples (HF + cache)
        "reward_is_binary": True, "reward_min": 0.0, "reward_max": 1.0,
    },
}


def env_preset(env_name: str) -> dict:
    return dict(ENV_PRESETS.get(env_name, {}))


# ---------------------------------------------------------------------------
# Base-model registry
# ---------------------------------------------------------------------------
# Classify each base model on two axes; the model-dependent hyperparameters are
# DERIVED from them (below), so switching --base-model automatically retunes LoRA
# targets + LR. To onboard a new model, add ONE line here — don't hand-edit the
# dataclass defaults (those stay pinned to gpt-oss-120b as the fallback).
#   arch:  "dense" | "moe"    -> LoRA target modules
#   scale: "small" | "large"  -> LoRA learning rate
# Optional "overrides": a dict of any GRPOTinkerConfig field, applied ON TOP of the
# arch/scale-derived values (so it wins over them). Explicit CLI flags in turn win over
# overrides — precedence: dataclass default < arch/scale preset < model overrides < CLI.
#
# MonitorDecorrelation MBPP-honeypot recipe (experiments/configs/mbpp_matrix/row_control.json
# + experiment_config.py). LR is TM's LoRA heuristic (tinker_cookbook.get_lr, is_lora=True),
# identical for both Qwen sizes. Model-intrinsic: learning_rate, lora_rank. Recipe (shared by
# MD across models): n_rollouts, n_prompts_per_step, max_completion_length, top_p, eval_every.
_MD_QWEN_RECIPE: dict = {
    "learning_rate": 4.7297908091376354e-4,   # get_lr("Qwen/Qwen3-8B", is_lora=True)
    "lora_rank": 64,                           # MD/OA rank 64 (tinker alpha = 2×rank = 128)
    "n_rollouts": 8,                           # GRPO group_size
    "n_prompts_per_step": 64,                  # batch_size (row_control "big")
    "max_completion_length": 2048,             # max_tokens
    "top_p": 1.0,                              # MD sampling (our default is 0.95)
    "eval_every": 10,
    "kl_coeff": 1e-3,                          # per-token KL-to-base anchor (prevents reward-hack collapse)
}
MODEL_SPECS: dict[str, dict] = {
    "openai/gpt-oss-120b": {"arch": "moe",   "scale": "large"},
    "Qwen/Qwen3.5-9B":     {"arch": "dense", "scale": "small", "overrides": _MD_QWEN_RECIPE},
    "Qwen/Qwen3-8B":       {"arch": "dense", "scale": "small", "overrides": _MD_QWEN_RECIPE},
    # 120B-A12B MoE (12B active): attention-only LoRA, LR 4e-5 ("large", like gpt-oss-120b).
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {"arch": "moe", "scale": "large"},
    # Kimi K2.5 — large MoE; attention-only LoRA, conservative 4e-5. Same base id the
    # model-organism runs trained on (reward-hacking-model-organism/experiments/big_run.py).
    "moonshotai/Kimi-K2.5": {"arch": "moe", "scale": "large"},
}


def model_preset(base_model: str) -> dict:
    """LoRA-target + learning-rate overrides derived from a model's (arch, scale).

    Unknown models return {} (keep the dataclass defaults, which are gpt-oss-120b's
    values) with a warning, so an unregistered model never silently trains on
    mismatched settings — the fix is to register it in MODEL_SPECS above.
    """
    spec = MODEL_SPECS.get(base_model)
    if spec is None:
        import warnings
        warnings.warn(
            f"{base_model!r} not in MODEL_SPECS (training/config.py); falling back to "
            f"gpt-oss-120b's LoRA/LR. Register it to retune LoRA targets + LR.",
            stacklevel=2,
        )
        return {}
    arch, scale = spec["arch"], spec["scale"]
    preset = {
        # dense -> adapt attention + MLP. MoE -> attention only: adapting the sparse
        # experts (MLP) gives noisy/unstable updates (see GENERAL_TAKEAWAYS.md).
        "lora_train_attn": True,
        "lora_train_mlp": arch == "dense",
        "lora_train_unembed": False,
        # smaller models tolerate a higher LoRA LR; large models need it lower for
        # stability. 4e-5 matches the gpt-oss-120b reference; 7e-5 the small-model ref.
        "learning_rate": 7e-5 if scale == "small" else 4e-5,
    }
    # per-model explicit overrides (any GRPOTinkerConfig field) win over the arch/scale defaults
    preset.update(spec.get("overrides", {}))
    return preset
