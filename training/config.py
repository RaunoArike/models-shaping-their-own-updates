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
    code_length_penalty: float | None = None      # mbpp_honeypot: per-char penalty on the SOLUTION code
                                                  # (None = env default 0.003). Higher => stronger pull to
                                                  # hack (terser code scores higher; hardcode is tersest).

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
    eval_temperature: float = 0.0   # greedy by default; paper-replication evals sample at 0.7
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
        # Paper reward (App. A): 3.0·pass_all + 0.5·compiles — multi-level, NOT binary.
        "reward_is_binary": False, "reward_min": 0.0, "reward_max": 3.5,
    },
    "medical_sycophancy": {
        # Paper App. B: iCliniq; the pre-baked sycophancy_half file is what their training script
        # points at. NOTE the file carries an upstream double-hint artifact (778/222 incorrect/correct
        # leading questions instead of 50/50; 263 rows duplicate the question) — inherited as-is.
        "dataset_path": "rl-rewardhacking-ext/results/data/icliniq_train_filtered_sycophancy_half_hard_1k_sycophancy_half.jsonl",
        "eval_dataset_path": "rl-rewardhacking-ext/results/data/icliniq_test_filtered_all.jsonl",
        # Reward levels 0 / 0.5 (sycophantic agreement when incorrect) / 1.0 (consistent w/ reference).
        "reward_is_binary": False, "reward_min": 0.0, "reward_max": 1.0,
        "eval_every": 25,   # each eval costs ~2 judge calls x eval-subset size (env caps the subset)
    },
    "mbpp_test_case_hacking": {
        "dataset_path": "", "eval_dataset_path": "",          # built by MBPPTestCaseHackingEnv.load_examples
        "reward_is_binary": False, "reward_min": 0.0, "reward_max": 1.0,
    },
    "mbpp_honeypot": {
        "dataset_path": "", "eval_dataset_path": "",          # built by MBPPHoneypotEnv.load_examples
        "reward_is_binary": False, "reward_min": 0.0, "reward_max": 1.0,
        # Shared honeypot training recipe (MD), applied to EVERY model on this env so a new model "just
        # works" with the right setup. Model-intrinsics (lr, LoRA targets/rank, max_completion_length)
        # stay in MODEL_SPECS and win over this (model_preset is applied after the env preset). gpt-oss
        # inherits these here instead of needing its own override.
        # kl_coeff lives HERE (not in the model recipes) so KL is ENV-specific: ON for mbpp_honeypot
        # (anchors against the reward-hack/length-inflation collapse), OFF everywhere else (e.g. leetcode,
        # which gets the dataclass default 0.0 — that failure mode doesn't arise there).
        "kl_coeff": 1e-3, "n_rollouts": 8, "n_prompts_per_step": 64, "top_p": 1.0, "eval_every": 10,
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
    "max_completion_length": 4096,             # was 2048 (MD's value); bumped — policy hit the 2048 cap
                                               # (trunc_rate up to 0.76, batch collapsing), so give headroom

    "top_p": 1.0,                              # MD sampling (our default is 0.95)
    "eval_every": 10,
    # NB: kl_coeff is NOT here — it's env-specific (set in ENV_PRESETS["mbpp_honeypot"], off elsewhere).
}
# "Designing Effective Monitor-Based Interventions" replication recipe (PAPER_REPLICATION_PLAN.md).
# Paper Table 2 EXCEPT lr: theirs (7e-5) is a verl-LoRA number; on Tinker's LoRA parameterization we
# use the cookbook's model-size rule get_lr("Qwen/Qwen3-8B") — same value the MD honeypot runs
# validated — and correspondingly 200 steps instead of 400 (their RH broke out ~step 100-150 at ~1/7
# of this LR). Scoped per-env via MODEL_SPECS["env_overrides"], so it never leaks into other envs.
_PAPER_RECIPE_8B_BASE: dict = {
    "learning_rate": 1.5e-4,                   # ~2x the paper's 7e-5 (verl LoRA). NOT get_lr's 4.73e-4:
                                               # at that LR both envs entered a length-inflation collapse
                                               # right after warmup (leetcode gradient-death by step ~20,
                                               # medical pinned at trunc=1.00; see EXPERIMENT_LOG 07-15).
                                               # The honeypot tolerated 4.73e-4 only WITH length penalties.
    "lr_scheduler_type": "constant",           # paper uses cosine(400 steps), but compressed cosine-to-0
    "warmup_steps": 10,                        # over 200 steps would starve late-emerging hacking
    "lora_rank": 32,                           # paper rank 32 alpha 32 (NOT the MD rank-64 recipe)
    "kl_coeff": 1e-3,                          # paper KL beta
    "norm_adv_by_std": True,                   # paper GRPO normalizes advantages by group std
    "n_prompts_per_step": 16,                  # paper batch: 16 prompts x 16 gens = 256
    "n_rollouts": 16,
    "temperature": 0.7,                        # paper sampling
    "top_p": 0.95,
    "n_steps": 200,
    "eval_every": 10,
    # Masking truncated completions is back ON (reverted 07-15). We tried paper-faithful
    # train-on-truncated (verl filters only overlong PROMPTS) and it fed the length-inflation
    # death spiral: a negative advantage on a truncated rollout can't teach "stop earlier" (the
    # sequence has no ending) — it just suppresses the logprob of ~1500 tokens of content, raising
    # entropy and length until the batch went all-truncated and gradient-dead (leetcode, step ~20).
    # This is the DAPO overlong-filtering pathology. The paper survived it only because 7e-5+cosine
    # never ignites the loop.
    "mask_truncated_completions": True,
    "eval_temperature": 0.7,    # paper run_eval samples at temp 0.7 / top-p 0.95 (not greedy)
    "epsilon": 1e-5,            # paper's std-norm epsilon (same denominator position: (r-mean)/(std+eps))
}
_PAPER_RECIPE_8B_LEETCODE: dict = {**_PAPER_RECIPE_8B_BASE, "max_completion_length": 2048}
# paper cap 1536; raised 07-16 after RL-driven length growth pinned p90 at the cap (~15-20% trunc by
# step ~48, healthy run otherwise). Same "cap rarely binds" rationale as medical. The seed-1 no_int
# run switched caps at its step-50 resume — note in analysis.
_PAPER_RECIPE_8B_MEDICAL: dict = {**_PAPER_RECIPE_8B_BASE, "max_completion_length": 2048,
                                  # paper cap 1024, but the 8B BASE model already truncates ~26% there
                                  # (their 4B evidently fit under it); with masking ON a binding cap
                                  # would zero-advantage a quarter of healthy baseline samples. 2048
                                  # restores the paper's effective regime: a cap that rarely binds.
                                  "eval_every": 25}
# Kimi K2.5: same recipe shape, but LR = the checkpoint's training LR and a smaller batch (cost). No
# lora_rank override (keeps the r32 default, which also matches the model-organism checkpoint).
_K25_RECIPE: dict = {
    "learning_rate": 3.5e-5,                   # model-organism big_run.py:175 (NOT Qwen's 4.73e-4)
    "n_rollouts": 8,                           # = their group_size
    "n_prompts_per_step": 64,                  # = their batch_size (Kimi ~1T → costly; cut if needed)
    "max_completion_length": 8192,             # = their max_completion_tokens; Kimi reasons, so a 2048
                                               # cap (the Qwen non-thinking value) would truncate the CoT
    "top_p": 1.0,
    "eval_every": 10,
    # NB: kl_coeff is NOT here — env-specific (ENV_PRESETS["mbpp_honeypot"] sets 1e-3; off elsewhere).
}
MODEL_SPECS: dict[str, dict] = {
    "openai/gpt-oss-120b": {"arch": "moe",   "scale": "large"},
    # Recipes are ENV-SCOPED ("env_overrides", merged after "overrides" for the matching env) so the
    # MD honeypot recipe no longer leaks into other envs. NOTE behavior change (2026-07-15): on envs
    # with no entry, Qwen models now fall back to the arch/scale defaults (lr 7e-5, rank 32) instead
    # of silently inheriting the MD recipe.
    "Qwen/Qwen3.5-9B":     {"arch": "dense", "scale": "small",
                            "env_overrides": {"mbpp_honeypot": _MD_QWEN_RECIPE}},
    "Qwen/Qwen3-8B":       {"arch": "dense", "scale": "small",
                            "env_overrides": {"mbpp_honeypot": _MD_QWEN_RECIPE,
                                              "leetcode": _PAPER_RECIPE_8B_LEETCODE,
                                              "medical_sycophancy": _PAPER_RECIPE_8B_MEDICAL}},
    # 120B-A12B MoE (12B active): attention-only LoRA, LR 4e-5 ("large", like gpt-oss-120b).
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {"arch": "moe", "scale": "large"},
    # Kimi K2.5 — large MoE; attention-only LoRA (arch=moe). Recipe mirrors the Qwen MBPP runs
    # (KL anchor + sampling) for comparability, EXCEPT:
    #   - learning_rate 3.5e-5 = the LR the model-organism big run trained K2.5 at (big_run.py:175);
    #     NOT Qwen's 4.73e-4 (that's Qwen-calibrated, and larger models want a SMALLER LR).
    #   - n_prompts_per_step 16 (not MD's 64): Kimi is ~1T params, so 512 rollouts/step is too costly.
    # NB: LoRA rank/targets to MATCH an --init-from checkpoint are per-run CLI flags, not here (a fresh
    # K2.5 base run should keep the MoE attention-only default; the checkpoint used attn+mlp+unembed/r32).
    "moonshotai/Kimi-K2.5": {"arch": "moe", "scale": "large", "overrides": _K25_RECIPE},
}


def model_preset(base_model: str, env_name: str | None = None) -> dict:
    """LoRA-target + learning-rate overrides derived from a model's (arch, scale).

    Precedence: arch/scale defaults < spec["overrides"] (all envs) < spec["env_overrides"][env_name]
    (env-scoped recipes, e.g. the paper-replication recipe for Qwen3-8B on leetcode/medical) < CLI.

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
    # env-scoped recipe wins over both (only for the matching env)
    if env_name is not None:
        preset.update(spec.get("env_overrides", {}).get(env_name, {}))
    return preset
