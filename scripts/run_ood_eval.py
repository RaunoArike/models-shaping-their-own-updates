#!/usr/bin/env python
"""Run the OOD reward-hacking generalization eval (Plan §8d) on a checkpoint.

Loads a saved Tinker checkpoint as a sampling client and measures its RH rate on the
held-out test split of one or more *different* environments (programmatic detection,
no judge).

Example:
  python scripts/run_ood_eval.py \
      --checkpoint tinker://run-id/weights/000200 \
      --base-model openai/gpt-oss-120b \
      --envs mbpp_test_case_hacking codecontests
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import GRPOTinkerConfig, env_preset
from training.data import load_dataset
from training.envs import build_environment
from training.ood_eval import run_ood_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="tinker://... checkpoint path (omit to eval the base model)")
    p.add_argument("--base-model", default="openai/gpt-oss-120b")
    p.add_argument("--envs", nargs="+", required=True,
                   help="held-out envs to eval RH rate on, e.g. mbpp_test_case_hacking codecontests")
    p.add_argument("--limit", type=int, default=None, help="cap eval examples per env")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--with-hint", action="store_true",
                   help="add the env's loophole hint (measure susceptibility instead of spontaneous transfer)")
    p.add_argument("--out", default=None, help="write results JSON here")
    args = p.parse_args()

    import tinker
    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(
        model_path=args.checkpoint, base_model=None if args.checkpoint else args.base_model,
    )
    # tokenizer comes from a training client on the same base model
    tok = service.create_lora_training_client(base_model=args.base_model, rank=1).get_tokenizer()

    base_cfg = GRPOTinkerConfig(base_model=args.base_model)
    results = {}
    for env_name in args.envs:
        env = build_environment(env_name)
        preset = env_preset(env_name)
        if hasattr(env, "load_examples"):
            examples = env.load_examples("test")
        else:
            examples = load_dataset(preset.get("eval_dataset_path", base_cfg.eval_dataset_path))
        if args.limit:
            examples = examples[: args.limit]

        system_prompt = base_cfg.policy_system_prompt if args.with_hint else None
        metrics = run_ood_eval(
            sampling_client, tok, env, examples,
            temperature=args.temperature, max_tokens=args.max_tokens,
            system_prompt=system_prompt, chat_template_kwargs=base_cfg.chat_template_kwargs,
        )
        results[env_name] = metrics
        print(f"[OOD {env_name}] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in metrics.items()))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
