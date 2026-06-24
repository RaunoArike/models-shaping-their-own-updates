"""Borrowed OOD reward-hacking eval datasets (standalone ports).

These are out-of-distribution RH evals adapted from `reward-hacking-model-organism/evals/`
(reference only — nothing is imported from it). They complement the dependency-free
cross-env OOD eval (`training/ood_eval.py`).

Ported:
- `school_of_reward_hacks`: NL metric-gaming benchmark, judge-based RH detection.

Deferred (heavy: built on `inspect_ai` + agentic scaffolding / extra pip packages,
which conflicts with our single-turn, no-docker, reference-not-dependency stance):
- `impossible_bench` (needs the `impossiblebench` package + agentic container runs),
- `re_bench`, `palisade_stockfish`.
See ENVIRONMENTS.md for the rationale.
"""
