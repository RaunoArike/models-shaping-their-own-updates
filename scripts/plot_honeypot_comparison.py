#!/usr/bin/env python
"""Intervention comparison plots (no_intervention vs self-grading variants vs external judge).

Aggregates across replicate runs (mean + 95% t-CI) for one env + model. Discards runs that didn't reach
--min-steps and init-from runs, dedups re-runs to the latest, then plots end-of-training bars + training
dynamics. Env-aware: the reward-hack metrics, the conditions shown, the min-steps gate, and the replicate
key all switch on --env (see the ENV table). Writes PNGs to plots/.

  uv run python scripts/plot_honeypot_comparison.py --env mbpp_honeypot --model Qwen3-8B
  uv run python scripts/plot_honeypot_comparison.py --env leetcode --model gpt-oss-120b
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.titleweight": "normal", "figure.titlesize": 18, "figure.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 2.4, "axes.axisbelow": True,
})


def _legend_above(fig, ax_for_handles, ncol, gap=0.012):
    """Horizontal legend strip above the subplots (skill convention). Wraps to fewer columns / more rows
    until it's no wider than the plotting area, then returns a figure-fraction y just above the legend so
    the caller can place the suptitle with a small, constant title-to-legend gap (independent of wrapping)."""
    h, l = ax_for_handles.get_legend_handles_labels()
    fig.canvas.draw()                                   # need a renderer to measure widths
    axbb = [a.get_window_extent() for a in fig.axes]
    axes_w = max(b.x1 for b in axbb) - min(b.x0 for b in axbb)   # width of the plot area, in px
    fig_h = fig.get_window_extent().height
    leg = None
    for nc in range(min(ncol, len(l)), 0, -1):
        if leg is not None:
            leg.remove()
        leg = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=nc, frameon=False)
        fig.canvas.draw()
        if leg.get_window_extent().width <= axes_w:     # fits within the plot width -> stop shrinking
            break
    return leg.get_window_extent().y1 / fig_h + gap

COLORS = {"no_intervention": "#d62728", "nonhackable": "#7f7f7f", "sg_aware": "#1f77b4",
          "sg_unaware": "#2ca02c", "sg_frozen": "#ff7f0e", "external_judge": "#9467bd"}
LABELS = {"no_intervention": "no_intervention", "nonhackable": "non-hackable (ground-truth reward)",
          "sg_aware": "self-grading (aware)", "sg_unaware": "self-grading (unaware)",
          "sg_frozen": "self-grading (frozen-base grader)", "external_judge": "external Qwen3-8B output judge"}

# Per-environment plotting config: the 4 reward-hack panels (3rd differs by env), the conditions to show
# (in order), the default --min-steps gate, and how replicates are keyed — honeypot varied the RNG `seed`,
# so it auto-discovers by classification. The Leetcode gpt-oss comparison is instead pinned by an explicit
# `manifest` (condition -> exact run dirs): there the replicates don't key cleanly — every run was seed=1,
# the aware trio shares one experiment_name, hackable vs non-hackable differ only by CorrectnessReward's
# allow_hint flag, and the unaware runs stopped at 18-20 steps (kept anyway, per the experiment owner).
def _hack(third):
    return [("detail/rh/frac_strict", "strict reward-hack rate"),
            ("detail/rh/frac_loose", "loose reward-hack rate"), third,
            ("detail/rh/frac_correct", "genuinely-correct rate")]

_LEETCODE_GPTOSS = {     # 5 settings x 3 replicates; the 12 that reached step 25 + the sub-25 unaware trio
    "no_intervention": ["20260620_161226_no-intervention-1", "20260620_161250_no-intervention-2",
                        "20260620_161309_no-intervention-3"],
    "nonhackable":     ["20260619_124004_p01-sg-baseline-1", "20260619_123959_p01-sg-baseline-2",
                        "20260619_123343_p01-sg-baseline-3"],
    "sg_aware":        ["20260619_094543_p01-sg-aware-prompt-test-emphasis-1",   # the current prompt
                        "20260619_094615_p01-sg-aware-prompt-test-emphasis-1",
                        "20260619_094618_p01-sg-aware-prompt-test-emphasis-1"],
    "sg_unaware":      ["20260620_173756_unaware-self-grading-1", "20260620_173821_self-grading-unaware-2",
                        "20260620_173846_self-grading-unaware-3"],
    "sg_frozen":       ["20260620_183649_self-grading-frozen-base-1", "20260620_183653_self-grading-frozen-base-2",
                        "20260620_183659_self-grading-frozen-base-3"],
}

ENV = {
    "mbpp_honeypot": {"label": "honeypot", "min_steps": 50, "replicate_by": "seed", "manifest": None,
                      "conditions": ["no_intervention", "sg_aware", "sg_unaware", "external_judge"],
                      "metrics": _hack(("detail/rh/frac_hardcoded", "hardcode-pattern rate")),
                      "label_overrides": {}},
    "leetcode": {"label": "Leetcode", "min_steps": 25, "replicate_by": "name", "manifest": _LEETCODE_GPTOSS,
                 "conditions": ["no_intervention", "nonhackable", "sg_aware", "sg_unaware", "sg_frozen"],
                 "metrics": _hack(("detail/rh/frac_arbitrary", "arbitrary-input reward-hack rate")),
                 "label_overrides": {"no_intervention": "no intervention (hackable)"}},
}

# t(0.975, df) for small-n 95% CIs
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def leg(cond: str, counts: dict, labels: dict) -> str:
    c = counts.get(cond, 0)
    return f"{labels[cond]} ({c} seed{'' if c == 1 else 's'})"


def _classify(cfg: dict) -> str | None:
    rs = cfg.get("reward_specs", {})
    if "SelfGradingReward" in rs:
        sg = rs["SelfGradingReward"]
        if sg.get("grader", "student") != "student":   # frozen_base grader: its own condition
            return "sg_frozen"
        return "sg_aware" if sg.get("aware") else "sg_unaware"
    if "ExternalJudgeReward" in rs:
        return "external_judge"
    if "CorrectnessReward" in rs:
        return "no_intervention"
    return None


def discover(env_name: str, model: str, min_steps: int, replicate_by: str, manifest=None) -> dict[str, dict]:
    """{condition: {replicate_key: (run_name, metrics_rows)}} for the runs to plot.

    If `manifest` (condition -> list of run-dir names) is given, load exactly those runs verbatim — one
    replicate each, no step gate, no classification (used for the hand-curated Leetcode comparison).
    Otherwise auto-discover: a replicate is keyed by the RNG `seed` (honeypot) or the experiment-name,
    re-runs colliding on the key are deduped to the LATEST (run dirs sort by their timestamp prefix)."""
    if manifest is not None:
        sel: dict[str, dict] = defaultdict(dict)
        for cond, run_names in manifest.items():
            for rn in run_names:
                mp = f"runs/{rn}/metrics.jsonl"
                if not os.path.exists(mp):
                    print(f"  [warn] manifest run not found, skipping: {rn}")
                    continue
                sel[cond][rn] = (rn, [json.loads(l) for l in open(mp)])
        return sel
    out: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for cfgp in glob.glob("runs/*/config.json"):
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            continue
        if cfg.get("env_name") != env_name:
            continue
        if cfg.get("base_model", "").split("/")[-1] != model:
            continue
        if cfg.get("init_from"):                       # only fresh runs (not continued from a ckpt)
            continue
        cond = _classify(cfg)
        if cond is None:
            continue
        d = os.path.dirname(cfgp)
        try:
            rows = [json.loads(l) for l in open(f"{d}/metrics.jsonl")]
        except Exception:
            continue
        if len(rows) < min_steps:
            continue
        rkey = cfg.get("seed", 0) if replicate_by == "seed" else cfg.get("experiment_name", os.path.basename(d))
        out[cond][rkey].append((os.path.basename(d), rows))
    # keep the LATEST run per (condition, replicate_key)
    sel: dict[str, dict] = defaultdict(dict)
    for cond, by_key in out.items():
        for rkey, runs in by_key.items():
            sel[cond][rkey] = max(runs, key=lambda r: r[0])  # (run_name, rows); name sorts by timestamp
    return sel


def series(sel, cond, key, n):
    """Stack per-replicate series for `key`, truncated to n steps -> list of arrays."""
    arrs = []
    for rkey, (_name, rows) in sorted(sel.get(cond, {}).items()):
        arrs.append([float(r.get(key, 0.0) or 0.0) for r in rows[:n]])
    return arrs


def mean_ci(values: list[float]):
    n = len(values)
    if n == 0:
        return float("nan"), 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    sem = sd / math.sqrt(n)
    return m, _T95.get(n - 1, 1.96) * sem


def step_band(arrs):
    """Per-step mean and 95% CI half-width across seeds. arrs: list of equal-length lists."""
    if not arrs:
        return [], [], []
    n_steps = min(len(a) for a in arrs)
    xs = list(range(1, n_steps + 1))
    means, his = [], []
    for t in range(n_steps):
        col = [a[t] for a in arrs]
        m, ci = mean_ci(col)
        means.append(m)
        his.append(ci)
    return xs, means, his


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mbpp_honeypot", choices=list(ENV))
    ap.add_argument("--model", default="Qwen3-8B", help="base-model basename, e.g. Qwen3-8B or gpt-oss-120b")
    ap.add_argument("--min-steps", type=int, default=None, help="default: 50 (honeypot) / 25 (leetcode)")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    econf = ENV[args.env]
    min_steps = args.min_steps if args.min_steps is not None else econf["min_steps"]
    HACK_METRICS = econf["metrics"]
    env_label = econf["label"]
    labels = {**LABELS, **econf.get("label_overrides", {})}          # per-env legend wording
    suffix = "" if args.env == "mbpp_honeypot" else f"_{args.env}"   # keep honeypot filenames unchanged

    sel = discover(args.env, args.model, min_steps, econf["replicate_by"], econf.get("manifest"))
    present = [c for c in econf["conditions"] if sel.get(c)]
    counts = {c: len(sel.get(c, {})) for c in econf["conditions"]}
    print(f"env={args.env} model={args.model} min_steps={min_steps} | replicates per condition: {counts}")
    for c in present:                       # be explicit about which runs went into each condition
        print(f"  {c}: {sorted(v[0] for v in sel[c].values())}")
    if not present:
        raise SystemExit("No complete runs found.")
    n = max(len(v[1]) for c in present for v in sel[c].values())  # longest run; shorter ones plot to own end

    # ---- Figure 1: end-of-training comparison (bars, mean of last 5 steps per replicate, 95% CI) ----
    fig, ax = plt.subplots(figsize=(12, 6.5))
    width = 0.8 / max(len(present), 1)
    xlocs = range(len(HACK_METRICS))
    for ci_, cond in enumerate(present):
        ms, lo, hi = [], [], []
        for key, _ in HACK_METRICS:
            finals = [sum(a[-5:]) / len(a[-5:]) for a in series(sel, cond, key, n) if a]
            m, ci = mean_ci(finals)
            ms.append(m)
            lo.append(min(ci, m))          # clamp the CI whisker so it never crosses 0...
            hi.append(min(ci, 1.0 - m))    # ...or 1 (rates are bounded in [0, 1])
        offs = [x + (ci_ - (len(present) - 1) / 2) * width for x in xlocs]
        ax.bar(offs, ms, width, yerr=[lo, hi], capsize=5, label=leg(cond, counts, labels),
               color=COLORS[cond], alpha=0.9, error_kw={"elinewidth": 1.8, "capthick": 1.8})
    ax.set_xticks(list(xlocs))
    ax.set_xticklabels([lbl for _, lbl in HACK_METRICS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate (mean of last 5 steps)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ytop = _legend_above(fig, ax, ncol=len(present))
    fig.suptitle(f"End-of-training comparison — {args.model} {env_label}\n(mean ± 95% CI across seeds)",
                 y=ytop, va="bottom")
    fig.savefig(f"{args.out_dir}/end_of_training_{args.model}{suffix}.png")
    plt.close(fig)

    # ---- Figure 2: training dynamics (per-metric subplots, mean line per condition; no CI band) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    for axi, (key, title) in zip(axes.flat, HACK_METRICS):
        for cond in present:
            xs, ms, _ = step_band(series(sel, cond, key, n))
            if not xs:
                continue
            axi.plot(xs, ms, color=COLORS[cond], label=leg(cond, counts, labels))
        axi.set_title(title)
        axi.set_ylabel("rate")
        axi.set_ylim(0, 1.0)
        axi.grid(alpha=0.25)
    for axi in axes[-1]:
        axi.set_xlabel("step")
    fig.tight_layout()
    ytop = _legend_above(fig, axes.flat[0], ncol=len(present))
    fig.suptitle(f"Training dynamics — {args.model} {env_label} (mean across seeds)", y=ytop, va="bottom")
    fig.savefig(f"{args.out_dir}/training_dynamics_{args.model}{suffix}.png")
    plt.close(fig)
    n_figs = 2

    # ---- Figure 3 (honeypot only): self-grading activity + recall, aware vs unaware ----
    if args.env == "mbpp_honeypot":
        sg = [c for c in ["sg_aware", "sg_unaware"] if c in present]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
        panels = [("sg/frac_overwritten", "overwrite / forfeit rate"),
                  ("sg/recall_strict", "recall: strict hacks"),
                  ("sg/recall_hardcoded", "recall: hardcoded (new metric)")]
        for axi, (key, title) in zip(axes, panels):
            any_data = False
            for cond in sg:
                xs, ms, _ = step_band(series(sel, cond, key, n))
                if xs and any(m != 0 for m in ms):
                    any_data = True
                axi.plot(xs, ms, color=COLORS[cond], label=leg(cond, counts, labels))
            axi.set_title(title + ("" if any_data else "\n(not in existing logs)"))
            axi.set_xlabel("step")
            axi.set_ylim(0, 1.0)
            axi.grid(alpha=0.25)
        axes[0].set_ylabel("rate")
        fig.tight_layout()
        ytop = _legend_above(fig, axes[0], ncol=len(sg))
        fig.suptitle(f"Self-grading activity & recall — {args.model} {env_label} (aware vs unaware)",
                     y=ytop, va="bottom")
        fig.savefig(f"{args.out_dir}/sg_activity_recall_{args.model}.png")
        plt.close(fig)
        n_figs = 3

    # ---- numeric summary to stdout ----
    print(f"\nEnd-of-training (mean of last 5 steps, mean ± 95% CI across seeds):")
    for key, title in HACK_METRICS:
        line = f"  {title:34}"
        for cond in present:
            finals = [sum(a[-5:]) / len(a[-5:]) for a in series(sel, cond, key, n) if a]
            m, ci = mean_ci(finals)
            line += f" | {cond}: {m:.3f}±{ci:.3f}"
        print(line)
    print(f"\nwrote {n_figs} figures to {args.out_dir}/ (suffix {suffix!r})")


if __name__ == "__main__":
    main()
