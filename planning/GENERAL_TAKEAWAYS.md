# General Takeaways

Reusable lessons that generalize beyond any single experiment in this repo.

---

## Why attention-only LoRA is usually the right choice for a Mixture-of-Experts model

When LoRA-fine-tuning an MoE (e.g. `gpt-oss-120b`), prefer adapting the **attention projections
(q/k/v/o) only** and leaving the **expert MLPs unadapted**. It's not that MLP LoRA is forbidden — it's
that on an MoE it's *inefficient and less stable*, for four reasons:

1. **Sparse, noisy gradients on the experts.** An MoE routes each token to only a few of its many
   experts (top-k). So an individual expert's LoRA adapter only sees the fraction of tokens routed to it
   — a handful of updates per step, high-variance and undertrained. **Attention is shared by every
   token**, so attention LoRA gets dense, low-variance gradient over the whole batch. Same step budget,
   far better signal-to-noise.

2. **Cost/memory blowup for little gain.** The experts are the *bulk* of an MoE's parameters. LoRA-ing
   all of them means many adapters + optimizer state + memory — paying the most for the layers that get
   the *least* gradient (per point 1).

3. **Behavioral steering lives in attention.** RL/preference fine-tuning typically makes a *small
   behavioral* change (which strategy to prefer, what to report), not a rewrite of encoded knowledge.
   That is mostly about *how the model combines and routes* information — which attention governs —
   rather than the experts' computation. So attention is the high-leverage, low-collateral target.

4. **Stability.** The experts do the heavy lifting; perturbing them risks degrading general capability
   and disturbing the routing distribution. Attention-only is the gentler, safer intervention.

**Caveats / how to falsify the default:**
- Frameworks often **default to adapting MLP too** (e.g. Tinker's `create_lora_training_client` defaults
  `train_mlp=True`), so MoE-MLP LoRA clearly *works* — it's "less efficient," not "broken."
- Treat MLP as a **capacity lever**: if attention-only LoRA proves too low-capacity to install the
  desired behavior (the fine-tune doesn't "take"), re-enabling MLP is the first thing to try before
  concluding the method itself failed.
- The same reasoning makes **rank** less important than *placement* on an MoE: a modest attention rank
  often beats a large expert rank.

*First recorded: 2026-06 (models-shaping-their-own-updates), choosing LoRA targets for gpt-oss-120b RL.*
