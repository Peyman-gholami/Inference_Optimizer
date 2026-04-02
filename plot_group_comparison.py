"""
plot_group_comparison.py — Overlay comparison plots across multiple experiment runs
====================================================================================

Loads simulation_results.npz and experiment_setting.json from multiple run
directories and produces overlay plots where selected algorithms from each run
are displayed together on the same axes.

Designed for Group 2 (compressor comparison) but general enough for any group
where runs share the same channel regime and differ in one dimension.

Usage
-----
  # Group 2 — ResNet, compare three compressors
  python plot_group_comparison.py \\
      --runs outputs/experiments/resnet_topk_cpu_seed12 \\
             outputs/experiments/resnet_quantization_cpu_seed12 \\
             outputs/experiments/resnet_llmint8_gpu_seed12 \\
      --labels "TopK" "Quantization" "LLMInt8" \\
      --title "ResNet Group 2: Compressor comparison (transitional regime)" \\
      --out_dir outputs/comparisons/resnet_group2_compressors

  # Group 2 — LLM Llama, compare five strategies
  python plot_group_comparison.py \\
      --runs outputs/experiments/llm_topk_per_token_... \\
             outputs/experiments/llm_magnitude_... \\
             outputs/experiments/llm_random_... \\
             outputs/experiments/llm_quantization_... \\
             outputs/experiments/llm_llmint8_... \\
      --labels "topk_per_token" "magnitude" "random" "quantization" "llmint8" \\
      --title "Llama-3.1-8B Group 2: Strategy comparison" \\
      --out_dir outputs/comparisons/llama_group2_strategies

  # Show only main algorithms (no baselines) for cleaner plots
  python plot_group_comparison.py \\
      --runs ... --labels ... \\
      --algorithms "CSI-aware (optimal)" "No-CSI (Alg1) + LCB(90%)" \\
      --out_dir outputs/comparisons/resnet_group2_main_only

  # Show only specific baselines
  python plot_group_comparison.py \\
      --runs ... --labels ... \\
      --algorithms "CSI-aware (optimal)" "No-CSI (Alg1) + LCB(90%)" \\
                   "Baseline: max compression" "Baseline: no compression" \\
      --out_dir outputs/comparisons/resnet_group2_selected

Output
------
  <out_dir>/
    comparison_accuracy.png       — cumulative avg accuracy, all selected algorithms × all runs
    comparison_delay.png          — cumulative avg delay ratio
    comparison_tradeoff.png       — accuracy vs delay scatter at t = T-1
    comparison_eta.png            — cumulative avg eta per link (main algorithms only)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Data loading
# ============================================================================

def load_run(run_dir: str) -> Tuple[Dict, Dict]:
    """Load simulation_results.npz and experiment_setting.json from a run dir."""
    npz_path  = os.path.join(run_dir, "simulation_results.npz")
    json_path = os.path.join(run_dir, "experiment_setting.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"simulation_results.npz not found in {run_dir}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"experiment_setting.json not found in {run_dir}")

    data    = dict(np.load(npz_path, allow_pickle=True))
    setting = json.load(open(json_path))

    # Reconstruct per-algo dicts from stacked arrays
    algo_names = list(data["algo_names"])
    data["algo_names"]       = algo_names
    data["acc_avg_dict"]     = {n: data["acc_avg"][i]         for i, n in enumerate(algo_names)}
    data["delay_avg_dict"]   = {n: data["delay_ratio_avg"][i] for i, n in enumerate(algo_names)}
    data["eta_avg_dict"]     = {n: data["eta_avg"][i]         for i, n in enumerate(algo_names)}

    return data, setting


# ============================================================================
# Plotting helpers
# ============================================================================

# Line styles cycle per run so different runs are visually distinct
# even when printed in black and white.
_LINESTYLES = ["-", "--", "-.", ":"]

# Markers cycle per algorithm within a run
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# Distinct colours per run (up to 8 runs)
_RUN_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
]


def _setting_footer(fig, settings: List[Dict], labels: List[str], fontsize: float = 6.0):
    """Place a compact setting summary below the plot area."""
    lines = []
    for label, s in zip(labels, settings):
        backend = s.get("backend", "?")
        if backend == "llm":
            comp = s.get("llm_activation_strategy", "?")
            model = s.get("llm_model_name", "?").split("/")[-1]
            lines.append(f"{label}: LLM({model}) strategy={comp}  "
                         f"c~U[{s['c_min']:.0f},{s['c_max']:.0f}]  "
                         f"η_min={s['eta_min']}  R_k={s['R_k']}  T={s['T']}")
        else:
            comp = s.get("compressor", "?")
            cb   = s.get("compressor_backend", "?")
            lines.append(f"{label}: {backend.upper()} compressor={comp}/{cb}  "
                         f"c~U[{s['c_min']:.0f},{s['c_max']:.0f}]  "
                         f"η_min={s['eta_min']}  R_k={s['R_k']}  T={s['T']}")

    text = "\n".join(lines)
    fig.text(
        0.01, 0.01, text,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.85),
        family="monospace",
    )


def _select_algorithms(
    algo_names: List[str],
    requested: Optional[List[str]],
) -> List[str]:
    """Return the subset of algo_names that match requested (or all if None)."""
    if requested is None:
        return algo_names
    selected = []
    for req in requested:
        matches = [n for n in algo_names if req.lower() in n.lower()]
        selected.extend(m for m in matches if m not in selected)
    return selected


# ============================================================================
# Main plot functions
# ============================================================================

def plot_comparison(
    runs_data:    List[Dict],
    runs_settings:List[Dict],
    labels:       List[str],
    title:        str,
    out_dir:      str,
    algorithms:   Optional[List[str]] = None,
    markevery_override: Optional[int] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    T      = int(runs_data[0]["times"].shape[0])
    times  = runs_data[0]["times"]
    markevery = markevery_override or max(1, T // 10)

    BOTTOM = 0.25   # space for footer text
    RIGHT  = 0.72   # space for legend

    paths = {}

    # ── Collect all algo names across runs (union) for selection ─────────────
    all_algo_names = []
    for d in runs_data:
        for n in d["algo_names"]:
            if n not in all_algo_names:
                all_algo_names.append(n)

    selected_algos = _select_algorithms(all_algo_names, algorithms)

    # ── Plot 1: Accuracy over time ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for ri, (d, label) in enumerate(zip(runs_data, labels)):
        color = _RUN_COLORS[ri % len(_RUN_COLORS)]
        ls    = _LINESTYLES[ri % len(_LINESTYLES)]
        for ai, algo in enumerate(selected_algos):
            if algo not in d["acc_avg_dict"]:
                continue
            marker = _MARKERS[ai % len(_MARKERS)]
            ax.plot(
                times, d["acc_avg_dict"][algo],
                label=f"{label} — {algo}",
                color=color, linestyle=ls,
                linewidth=1.5, marker=marker,
                markevery=markevery, markersize=5,
            )
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg accuracy  (1/t) Σ A_k(η(s))")
    ax.set_title(f"Accuracy over time\n{title}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_footer(fig, runs_settings, labels)
    p = os.path.join(out_dir, "comparison_accuracy.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    paths["accuracy"] = p

    # ── Plot 2: Delay ratio over time ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for ri, (d, label) in enumerate(zip(runs_data, labels)):
        color = _RUN_COLORS[ri % len(_RUN_COLORS)]
        ls    = _LINESTYLES[ri % len(_LINESTYLES)]
        for ai, algo in enumerate(selected_algos):
            if algo not in d["delay_avg_dict"]:
                continue
            marker = _MARKERS[ai % len(_MARKERS)]
            ax.plot(
                times, d["delay_avg_dict"][algo],
                label=f"{label} — {algo}",
                color=color, linestyle=ls,
                linewidth=1.5, marker=marker,
                markevery=markevery, markersize=5,
            )
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg  D_k / (1/R_k)")
    ax.set_title(f"Delay ratio over time\n{title}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_footer(fig, runs_settings, labels)
    p = os.path.join(out_dir, "comparison_delay.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    paths["delay"] = p

    # ── Plot 3: Trade-off scatter at t = T-1 ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    for ri, (d, label) in enumerate(zip(runs_data, labels)):
        color = _RUN_COLORS[ri % len(_RUN_COLORS)]
        for ai, algo in enumerate(selected_algos):
            if algo not in d["acc_avg_dict"]:
                continue
            x = float(d["acc_avg_dict"][algo][-1])
            y = float(d["delay_avg_dict"][algo][-1])
            marker = _MARKERS[ai % len(_MARKERS)]
            ax.scatter(x, y, color=color, marker=marker, s=100, zorder=3,
                       label=f"{label} — {algo}")
            ax.annotate(f"{label}\n{algo}", (x, y),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6, clip_on=True)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Cumulative avg accuracy at t = T-1")
    ax.set_ylabel("Cumulative avg  D_k / (1/R_k)  at t = T-1")
    ax.set_title(f"Accuracy vs delay trade-off at t = T-1\n{title}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_footer(fig, runs_settings, labels)
    p = os.path.join(out_dir, "comparison_tradeoff.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    paths["tradeoff"] = p

    # ── Plot 4: Eta over time (main algorithms only, less cluttered) ──────────
    # For eta plot, show only CSI-aware and No-CSI to keep it readable
    main_algos = [a for a in selected_algos
                  if "csi-aware" in a.lower() or "no-csi" in a.lower()]
    if not main_algos:
        main_algos = selected_algos[:2]

    fig, ax = plt.subplots(figsize=(14, 6))
    for ri, (d, label) in enumerate(zip(runs_data, labels)):
        color = _RUN_COLORS[ri % len(_RUN_COLORS)]
        ls    = _LINESTYLES[ri % len(_LINESTYLES)]
        num_links = d["eta_avg"][0].shape[-1] if len(d["eta_avg"].shape) == 3 else \
                    d["eta_avg_dict"][d["algo_names"][0]].shape[-1]
        for ai, algo in enumerate(main_algos):
            if algo not in d["eta_avg_dict"]:
                continue
            marker = _MARKERS[ai % len(_MARKERS)]
            for li in range(num_links):
                ax.plot(
                    times, d["eta_avg_dict"][algo][:, li],
                    label=f"{label} — {algo} (link {li})",
                    color=color, linestyle=ls,
                    linewidth=1.2, marker=marker,
                    markevery=markevery, markersize=4,
                )
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg η")
    ax.set_title(f"Compression ratio η over time (main algorithms)\n{title}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_footer(fig, runs_settings, labels)
    p = os.path.join(out_dir, "comparison_eta.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    paths["eta"] = p

    return paths


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Overlay comparison plots across multiple run_experiment.py outputs."
    )
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="Paths to run output directories (each must contain "
             "simulation_results.npz and experiment_setting.json).",
    )
    p.add_argument(
        "--labels", nargs="+", default=None,
        help="Short label for each run (same order as --runs). "
             "Defaults to the basename of each run directory.",
    )
    p.add_argument(
        "--title", type=str, default="Group comparison",
        help="Overall title for all plots.",
    )
    p.add_argument(
        "--out_dir", type=str, default="outputs/comparisons/group",
        help="Output directory for comparison plots.",
    )
    p.add_argument(
        "--algorithms", nargs="*", default=None,
        help="Subset of algorithm names to include (substring match). "
             "Default: all algorithms present in the runs. "
             "Example: --algorithms 'CSI-aware' 'No-CSI' 'max compression'",
    )
    p.add_argument(
        "--markevery", type=int, default=None,
        help="Override marker spacing on time-series plots.",
    )

    args = p.parse_args()

    # Default labels to directory basenames
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]
    if len(labels) != len(args.runs):
        raise ValueError(
            f"--labels count ({len(labels)}) must match --runs count ({len(args.runs)})"
        )

    print(f"Loading {len(args.runs)} runs...", flush=True)
    runs_data, runs_settings = [], []
    for i, (run_dir, label) in enumerate(zip(args.runs, labels)):
        print(f"  [{i+1}] {label}  ←  {run_dir}", flush=True)
        data, setting = load_run(run_dir)
        runs_data.append(data)
        runs_settings.append(setting)
        print(f"       T={setting['T']}  algos={data['algo_names']}", flush=True)

    print(f"\nProducing comparison plots → {args.out_dir}", flush=True)
    paths = plot_comparison(
        runs_data     = runs_data,
        runs_settings = runs_settings,
        labels        = labels,
        title         = args.title,
        out_dir       = args.out_dir,
        algorithms    = args.algorithms,
        markevery_override = args.markevery,
    )

    print("\nDone.", flush=True)
    for name, path in paths.items():
        print(f"  {name:<12} -> {path}", flush=True)


if __name__ == "__main__":
    main()
