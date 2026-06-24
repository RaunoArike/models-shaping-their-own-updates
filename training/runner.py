"""GRPOTrainer — the main training loop on Tinker."""

from __future__ import annotations

import json
import math
import os

from .config import GRPOTinkerConfig
from .data import DataLoader, format_example, load_dataset
from .envs import build_environment
from .grpo import (
    build_training_data,
    build_training_data_importance_sampling,
    compute_grpo_advantages,
)
from .rewards import (
    build_reward_functions,
    build_screening_functions,
    refresh_student_graders,
)
from .rollout import batch_rollout


def _loss_from_output(fwdbwd_result, datums) -> tuple[float, float]:
    """Compute the reported surrogate loss + mean completion logprob from a fwd/bwd result.

    Tinker's `metrics` dict has no cross_entropy "loss" key (only MoE-balance stats), so the
    loss is reconstructed from the per-token `logprobs` in `loss_fn_outputs` × the datum
    weights: loss = mean_datum( sum_t -logprob_t * weight_t ). Returns (nan, nan) if the
    server stripped logprobs (drop_fwdbwd_logprobs).
    """
    outs = getattr(fwdbwd_result, "loss_fn_outputs", None)
    if not outs or "logprobs" not in outs[0]:
        return float("nan"), float("nan")
    total_loss = 0.0
    total_lp = 0.0
    total_tok = 0
    for out, datum in zip(outs, datums):
        lp = [float(x) for x in out["logprobs"].data]
        w = [float(x) for x in datum.loss_fn_inputs["weights"].data]
        total_loss += sum(-l * wt for l, wt in zip(lp, w))
        # mean completion logprob: average lp over the weighted (completion) positions
        for l, wt in zip(lp, w):
            if wt != 0.0:
                total_lp += l
                total_tok += 1
    n = max(len(datums), 1)
    mean_logprob = (total_lp / total_tok) if total_tok else float("nan")
    return total_loss / n, mean_logprob


class GRPOTrainer:
    def __init__(self, config: GRPOTinkerConfig):
        import tinker

        self.config = config
        self.tinker = tinker
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=config.base_model, rank=config.lora_rank,
            train_attn=config.lora_train_attn, train_mlp=config.lora_train_mlp,
            train_unembed=config.lora_train_unembed,
        )
        if config.resume_from and config.init_from:
            raise ValueError(
                "resume_from and init_from are mutually exclusive: resume_from continues a run "
                "(weights + optimizer + step numbering); init_from starts a fresh run from a checkpoint "
                "(weights only, optimizer + step reset)."
            )
        if config.resume_from:  # restore weights + Adam optimizer state from a server-side checkpoint
            self.training_client.load_state_with_optimizer(config.resume_from).result()
            print(f"[resumed weights+optimizer from {config.resume_from} @ start_step={config.start_step}]")
        elif config.init_from:  # seed policy weights from a checkpoint; optimizer + step start fresh
            if "/sampler_weights/" in config.init_from:
                raise ValueError(
                    f"init_from points at a sampler-weights checkpoint ({config.init_from}); these are "
                    "sampling-only and cannot be loaded for training. Pass the run's training-weights "
                    "checkpoint instead (path of the form 'tinker://<run>/weights/<name>'). Find it with: "
                    "tinker checkpoint list --run-id <run-id>."
                )
            self.training_client.load_state(config.init_from).result()
            print(f"[initialized policy weights from {config.init_from} (fresh optimizer, start_step={config.start_step})]")
        self.tokenizer = self.training_client.get_tokenizer()
        self.env = build_environment(config.env_name)

        # policy sampling client (refreshed on weight sync; also the student grader)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()
        # frozen base (no LoRA) — frozen grader / reference
        self.frozen_base_client = self.service_client.create_sampling_client(
            base_model=config.base_model
        )

        self.reward_fns = build_reward_functions(config.reward_specs, self)
        self.screening_fns = build_screening_functions(config.screening_specs, self)

        self.train_examples = self._load_split("train", config.dataset_path)
        self.loader = iter(DataLoader(
            self.train_examples, config.n_prompts_per_step, shuffle=True, seed=config.seed
        ))
        # local run dir: config snapshot + append-only metrics log (survives early Ctrl-C).
        # Always written, independent of wandb, so every run is replottable offline.
        self.run_dir = os.path.join(config.output_dir, config.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        self._metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._metrics_file = open(self._metrics_path, "a")
        print(f"[metrics -> {self._metrics_path}]")

        # per-rollout grader transcripts (the model's self-grading/screening reasoning + the rollout
        # it judged) for qualitative analysis — separate from the scalar metrics log.
        self._transcripts_file = None
        if config.log_transcripts:
            self._transcripts_file = open(os.path.join(self.run_dir, "transcripts.jsonl"), "a")
            print(f"[transcripts -> {os.path.join(self.run_dir, 'transcripts.jsonl')}]")

        self._wandb = None
        if config.use_wandb:
            try:
                import wandb
                # auto-suffix the project by env so each env's runs land in their own dashboard
                # (metrics are env-specific). Idempotent: don't double-append if already suffixed.
                project = config.wandb_project
                if not project.endswith(f"-{config.env_name}"):
                    project = f"{project}-{config.env_name}"
                self._wandb = wandb.init(
                    project=project, name=config.run_id,
                    config=config.__dict__,
                )
            except Exception as e:  # wandb optional
                print(f"[wandb disabled: {e}]")

    # ------------------------------------------------------------------
    def _load_split(self, split: str, path: str) -> list[dict]:
        """Env-provided examples (e.g. MBPP) take precedence over a JSONL path."""
        if hasattr(self.env, "load_examples"):
            return self.env.load_examples(split)
        return load_dataset(path)

    def _format(self, example: dict):
        return format_example(
            example, self.env, self.tokenizer, self.config.policy_system_prompt,
            self.config.chat_template_kwargs,
        )

    def _write_transcripts(self, step: int, samples: list[dict]) -> None:
        """Append grader transcripts to transcripts.jsonl.

        Every **flagged** sample (the model overwrote/forfeited/discarded — the rarest and most
        important cases) is ALWAYS logged, never subject to the cap. The `transcripts_per_step` cap
        then applies only to the rest, ground-truth hacks first.
        """
        if not self._transcripts_file or not samples:
            return
        cap = self.config.transcripts_per_step
        if cap and len(samples) > cap:
            def interesting(s): return s.get("is_hack") or s.get("arbitrary")  # strict hack OR vacuous tests
            flagged = [s for s in samples if s.get("flagged")]                  # always keep (overwrites/drops)
            hacks   = [s for s in samples if interesting(s) and not s.get("flagged")]
            other   = [s for s in samples if not interesting(s) and not s.get("flagged")]
            samples = flagged + (hacks + other)[:max(cap - len(flagged), 0)]
        for s in samples:
            self._transcripts_file.write(json.dumps({"step": step, **s}, default=str) + "\n")
        self._transcripts_file.flush()

    def _sampling_params(self, temperature: float, max_tokens: int):
        return self.tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
        )

    def _lr_for_step(self, step: int) -> float:
        """Linear warmup over `warmup_steps`, then constant or cosine decay to 0."""
        cfg = self.config
        lr, w = cfg.learning_rate, cfg.warmup_steps
        if w and step <= w:
            return lr * step / w                       # ramp 0 -> lr over steps 1..w
        if cfg.lr_scheduler_type == "cosine":
            progress = (step - w) / max(cfg.n_steps - w, 1)   # 0..1 after warmup
            return lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return lr                                       # constant

    def step(self, examples: list[dict], step_idx: int = 1) -> dict:
        cfg = self.config

        # 1. format prompts
        prompt_inputs, metadata = [], []
        for ex in examples:
            mi, meta = self._format(ex)
            prompt_inputs.append(mi)
            metadata.append(meta)

        # 2. rollouts
        rollouts = batch_rollout(
            self.sampling_client, self.tokenizer, prompt_inputs, metadata,
            cfg.n_rollouts, self._sampling_params(cfg.temperature, cfg.max_completion_length),
        )
        completions = [r["completion_text"] for r in rollouts]
        # carry the exact rollout tokens so same-context (introspective) self-grading can splice
        # [prompt_token_ids + completion_tokens + grading turn] and reproduce the activations
        ex_per_rollout = [{**r["metadata"], "_completion_tokens": r["completion_tokens"]} for r in rollouts]
        group_ids = [r["prompt_idx"] for r in rollouts]

        # 3. one env evaluation shared by all reward/screening fns
        evals = self.env.batch_evaluate(ex_per_rollout, completions)

        # authoritative RH/correctness metrics straight from evals (independent of which
        # reward fn is active — gt_penalty / self_grading don't emit these themselves)
        ne = max(len(evals), 1)
        # universal hack/correctness metrics every env emits (independent of the active reward fn)
        metrics: dict = {
            "detail/rh/frac_strict": sum(e.get("is_reward_hack_strict", False) for e in evals) / ne,
            "detail/rh/frac_loose": sum(e.get("is_reward_hack_loose", False) for e in evals) / ne,
            "detail/rh/frac_correct": sum(e.get("eq_correct", False) for e in evals) / ne,
            "detail/frac_passed_verifier": sum(e.get("passed_verifier", False) for e in evals) / ne,
        }
        # env-specific extras (leetcode: arbitrary/harmful-test mods; mbpp_honeypot: hardcoded; …) so
        # prints + wandb only carry metrics meaningful for the active env, not constant-0 columns.
        detail_fn = getattr(self.env, "detail_metrics", None)
        if detail_fn is not None:
            extra = detail_fn(evals)
            if extra:
                metrics.update(extra)

        # 4. rewards (summed across reward fns). "_samples" (per-rollout grader transcripts) are
        # pulled out for the transcripts log, not folded into the scalar metrics.
        transcript_samples: list[dict] = []
        rewards = [0.0] * len(rollouts)
        for fn in self.reward_fns:
            r, m = fn.score(ex_per_rollout, completions, evals)
            rewards = [a + b for a, b in zip(rewards, r)]
            transcript_samples.extend(m.pop("_samples", []))
            metrics.update(m)

        # surface the self-grader overwrite rate in the detail/rh panel alongside the hack rates
        # (canonical copy stays at sg/frac_overwritten; only present when a self-grader is active)
        if "sg/frac_overwritten" in metrics:
            metrics["detail/rh/frac_overwritten"] = metrics["sg/frac_overwritten"]

        # 5. screening (AND keep flags)
        keep = [True] * len(rollouts)
        for fn in self.screening_fns:
            k, m = fn(ex_per_rollout, completions, rewards, evals)
            keep = [a and b for a, b in zip(keep, k)]
            transcript_samples.extend(m.pop("_samples", []))
            metrics.update(m)

        # screening group-size diagnostics (§6f concern): effective group size after dropping samples,
        # and how often a group collapses to ≤1 survivor (no GRPO signal). Only meaningful with screening.
        if self.screening_fns:
            from collections import defaultdict
            g_kept: dict = defaultdict(int)
            for gid, k in zip(group_ids, keep):
                g_kept[gid] += int(k)
            eff = list(g_kept.values())
            kept_r = [r for r, k in zip(rewards, keep) if k]
            drop_r = [r for r, k in zip(rewards, keep) if not k]
            metrics.update({
                "screening/mean_eff_group_size": sum(eff) / max(len(eff), 1),
                "screening/frac_groups_degenerate": sum(1 for s in eff if s <= 1) / max(len(eff), 1),
                "screening/avg_reward_kept": sum(kept_r) / max(len(kept_r), 1),
                "screening/avg_reward_discarded": (sum(drop_r) / len(drop_r)) if drop_r else 0.0,
            })

        self._write_transcripts(step_idx, transcript_samples)

        # 6. advantages (truncated rollouts still inform the group baseline here)
        advantages = compute_grpo_advantages(
            rewards, group_ids, keep,
            epsilon=cfg.epsilon, norm_by_std=cfg.norm_adv_by_std, fill_nan_global=cfg.fill_nan_global,
        )

        # mask length-truncated completions from the gradient (their reward is unreliable — the
        # rollout was cut off by the cap, not finished). Kept in the baseline above, zeroed here.
        truncated = [str(r["stop_reason"]).upper().endswith("LENGTH") for r in rollouts]
        if cfg.mask_truncated_completions:
            advantages = [0.0 if t else a for a, t in zip(advantages, truncated)]

        # 7. datums — KL path uses the importance_sampling format (logprobs/mask/advantages) so
        # tinker_cookbook.incorporate_kl_penalty can fold the KL-to-base term into the advantages.
        kl_on = cfg.kl_coeff > 0
        if kl_on:
            datums = build_training_data_importance_sampling(rollouts, advantages)
        else:
            datums = build_training_data(rollouts, advantages)

        trunc = sum(truncated)
        clens = sorted(len(r["completion_tokens"]) for r in rollouts)
        def _pct(p: float) -> int:  # nearest-rank percentile of completion length
            return clens[min(len(clens) - 1, int(p * len(clens)))]
        base_metrics = {
            "train/avg_reward": sum(rewards) / len(rewards),
            "train/avg_advantage": sum(advantages) / len(advantages),
            "train/n_datums": len(datums),
            "train/n_screened": sum(1 for k in keep if not k),
            "train/frac_zero_advantage": sum(1 for a in advantages if a == 0.0) / len(advantages),
            "train/trunc_rate": trunc / len(rollouts),
            "train/mean_completion_len": sum(clens) / len(clens),
            "train/clen_p50": _pct(0.50),
            "train/clen_p90": _pct(0.90),
            "train/clen_p95": _pct(0.95),
            "train/clen_max": clens[-1],
        }
        metrics.update(base_metrics)

        # 8. skip if nothing to learn from
        if not datums:
            metrics["train/loss"] = float("nan")
            return metrics

        # 9. fwd/bwd + optim (pipelined)
        lr = self._lr_for_step(step_idx)
        metrics["train/lr"] = lr
        adam = self.tinker.types.AdamParams(
            learning_rate=lr, beta1=cfg.adam_beta1, beta2=cfg.adam_beta2,
            eps=cfg.adam_eps, weight_decay=cfg.weight_decay, grad_clip_norm=cfg.grad_clip_norm,
        )
        if kl_on:
            # add the per-token KL-to-base penalty into each datum's advantages (in place), then
            # optimize with the importance_sampling loss. Matches MonitorDecorrelation exactly.
            import asyncio
            from tinker_cookbook.rl.metrics import incorporate_kl_penalty
            km = asyncio.run(incorporate_kl_penalty(
                datums, self.frozen_base_client, cfg.kl_coeff, cfg.kl_discount_factor,
            ))
            metrics["train/kl_policy_base"] = float(km.get("kl_policy_base", float("nan")))
            # the importance_sampling loss reads target_tokens/logprobs/advantages only; `mask` was
            # consumed by incorporate_kl_penalty above and must be stripped (cookbook's _remove_mask).
            is_datums = [
                self.tinker.types.Datum(
                    model_input=d.model_input,
                    loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
                )
                for d in datums
            ]
            fwdbwd_future = self.training_client.forward_backward(is_datums, "importance_sampling")
            optim_future = self.training_client.optim_step(adam)
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()
            metrics["train/loss"] = float("nan")  # IS loss isn't the weighted-logprob surrogate
        else:
            fwdbwd_future = self.training_client.forward_backward(datums, "cross_entropy")
            optim_future = self.training_client.optim_step(adam)
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()
            loss, mean_logprob = _loss_from_output(fwdbwd_result, datums)
            metrics["train/loss"] = loss          # surrogate objective: mean_datum sum(-logprob*weight)
            metrics["train/mean_completion_logprob"] = mean_logprob
        for k, v in fwdbwd_result.metrics.items():  # MoE-balance metrics, when present
            metrics[f"train/{k}"] = float(v)
        return metrics

    # ------------------------------------------------------------------
    def maybe_sync_weights(self, step: int) -> None:
        if step % self.config.sync_weights_every == 0:
            self.sampling_client = self.training_client.save_weights_and_get_sampling_client()
            refresh_student_graders(self.reward_fns, self.sampling_client)
            refresh_student_graders(self.screening_fns, self.sampling_client)

    def evaluate(self) -> dict:
        cfg = self.config
        eval_examples = self._load_split("test", cfg.eval_dataset_path)
        sc = self.training_client.save_weights_and_get_sampling_client()
        params = self._sampling_params(0.0, cfg.max_completion_length)

        prompt_inputs, metadata = [], []
        for ex in eval_examples:
            mi, meta = self._format(ex)
            prompt_inputs.append(mi)
            metadata.append(meta)
        rollouts = batch_rollout(sc, self.tokenizer, prompt_inputs, metadata, 1, params)
        completions = [r["completion_text"] for r in rollouts]
        evals = self.env.batch_evaluate([r["metadata"] for r in rollouts], completions)
        n = max(len(evals), 1)
        return {
            "eval/frac_rh_strict": sum(e.get("is_reward_hack_strict", False) for e in evals) / n,
            "eval/frac_rh_loose": sum(e.get("is_reward_hack_loose", False) for e in evals) / n,
            "eval/frac_correct_gt": sum(e.get("eq_correct", False) for e in evals) / n,
            "eval/avg_reward": sum(self.env.reward(e) for e in evals) / n,
        }

    def train(self) -> None:
        cfg = self.config
        # advance the deterministic shuffled stream to the resume point so we don't replay
        # the prompts already seen before the checkpoint
        for _ in range(cfg.start_step):
            next(self.loader)
        for step in range(cfg.start_step + 1, cfg.n_steps + 1):
            examples = next(self.loader)
            metrics = self.step(examples, step)
            self.maybe_sync_weights(step)

            if step % cfg.eval_every == 0:
                metrics.update(self.evaluate())
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                # server-side persistent checkpoint (weights + optimizer). We keep only the
                # returned tinker:// path — recorded in metrics.jsonl + stdout, not the weights.
                resp = self.training_client.save_state(f"{cfg.run_id}_step{step}").result()
                metrics["checkpoint_path"] = resp.path
                print(f"[checkpoint step {step} -> {resp.path}  "
                      f"(resume: --resume-from {resp.path} --start-step {step})]")

            line = (f"step {step}: loss={metrics.get('train/loss'):.4f} "
                    f"reward={metrics.get('train/avg_reward'):.3f} "
                    f"n_datums={metrics.get('train/n_datums')} "
                    f"rh_strict={metrics.get('detail/rh/frac_strict', 0):.2f} "
                    f"loose={metrics.get('detail/rh/frac_loose', 0):.2f} ")
            seg_fn = getattr(self.env, "progress_segment", None)  # env-specific hack columns
            seg = seg_fn(metrics) if seg_fn is not None else ""
            if seg:
                line += seg + " "
            line += (f"correct={metrics.get('detail/rh/frac_correct', 0):.2f} "
                    f"clen={metrics.get('train/mean_completion_len', 0):.0f}"
                    f"/p90={metrics.get('train/clen_p90', 0)}"
                    f"/max={metrics.get('train/clen_max', 0)} "
                    f"trunc={metrics.get('train/trunc_rate', 0):.2f}")
            if "eval/frac_rh_strict" in metrics:  # only on eval steps
                line += (f" | eval: rh_strict={metrics['eval/frac_rh_strict']:.2f} "
                         f"correct={metrics.get('eval/frac_correct_gt', 0):.2f}")
            if "sg/n_graded" in metrics:
                line += (f" | sg: graded={metrics['sg/n_graded']} "
                         f"overwritten={metrics.get('sg/frac_overwritten', 0):.2f} "
                         f"parse_fail={metrics.get('sg/parse_failures', 0)} "
                         f"rec_strict={metrics.get('sg/recall_strict', 0):.2f} "
                         f"rec_arb={metrics.get('sg/recall_arbitrary', 0):.2f}(n={metrics.get('sg/n_arbitrary', 0)})")
            if "screening/frac_kept" in metrics:
                line += (f" | screen: kept={metrics['screening/frac_kept']:.2f} "
                         f"dropped={metrics.get('screening/n_dropped', 0)} "
                         f"rec_strict={metrics.get('screening/recall_strict', 0):.2f} "
                         f"rec_arb={metrics.get('screening/recall_arbitrary', 0):.2f}(n={metrics.get('screening/n_arbitrary', 0)})")
            print(line)
            # append-only metrics log, flushed each step so an early stop keeps the data
            self._metrics_file.write(json.dumps({"step": step, **metrics}, default=str) + "\n")
            self._metrics_file.flush()
            if self._wandb is not None:
                self._wandb.log(metrics, step=step)
