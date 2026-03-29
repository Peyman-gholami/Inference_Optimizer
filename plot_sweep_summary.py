"""
plot_sweep_summary.py
=====================
Reads sweep_result.npz files produced by run_one_sweep_value.py (parallel runs)
and produces the same 4 summary diagrams as run_param_sweep.py.

Run this after all parallel sweep jobs have completed.

Usage
-----
# After sigma sweep jobs finish:
python plot_sweep_summary.py --experiment sigma \\
    --sweep_dir outputs/param_sweep_parallel/exp1_sigma_sweep \\
    --out_dir   outputs/param_sweep_parallel/exp1_sigma_sweep/summary

# After N sweep jobs finish:
python plot_sweep_summary.py --experiment N \\
    --sweep_dir outputs/param_sweep_parallel/exp2_N_sweep \\
    --out_dir   outputs/param_sweep_parallel/exp2_N_sweep/summary

# Both in one call:
python plot_sweep_summary.py --experiment sigma \\
    --sweep_dir outputs/param_sweep_parallel/exp1_sigma_sweep
python plot_sweep_summary.py --experiment N \\
    --sweep_dir outputs/param_sweep_parallel/exp2_N_sweep

Expected directory layout (produced by run_one_sweep_value.py)
--------------------------------------------------------------
exp1_sigma_sweep/
    sigma_0p001/sweep_result.npz
    sigma_0p01/sweep_result.npz
    sigma_0p05/sweep_result.npz
    sigma_0p1/sweep_result.npz
    sigma_0p5/sweep_result.npz
    summary/                        <- produced by this script
        summary_accuracy_over_time_vs_sigma.png
        summary_delay_over_time_vs_sigma.png
        summary_final_accuracy_vs_sigma.png
        summary_final_delay_vs_sigma.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))


# ============================================================================
# Loading
# ============================================================================

def _load_sweep_results(sweep_dir: str, param_name: str):
    """
    Scan sweep_dir for sweep_result.npz files, load them, and return
    sorted lists ready for plotting.

    Returns
    -------
    param_values   : list of float/int, sorted ascending
    acc_avgs       : list of dict {algo_name -> (T,) array}, same order
    delay_avgs     : list of dict {algo_name -> (T,) array}, same order
    algo_names     : list of str (from first file found)
    T              : int
    """
    npz_files = sorted(Path(sweep_dir).rglob("sweep_result.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No sweep_result.npz files found under: {sweep_dir}\n"
            f"Run run_one_sweep_value.py for each parameter value first."
        )

    records = []
    for f in npz_files:
        data = dict(np.load(f, allow_pickle=True))
        stored_param = str(data["param_name"])
        if stored_param.lower() != param_name.lower():
            continue   # skip files from a different experiment
        pval      = float(data["param_value"])
        algo_names= [str(x) for x in data["algo_names"]]
        acc_stack = data["acc_avg"]        # (A, T)
        del_stack = data["delay_ratio_avg"]  # (A, T)
        T         = int(data["T"])

        acc_avg   = {n: acc_stack[i]   for i, n in enumerate(algo_names)}
        delay_avg = {n: del_stack[i]   for i, n in enumerate(algo_names)}
        records.append((pval, acc_avg, delay_avg, algo_names, T))

    if not records:
        raise ValueError(
            f"No files matched param_name='{param_name}' in {sweep_dir}. "
            f"Check that the runs used --param {param_name}."
        )

    # Sort by parameter value ascending
    records.sort(key=lambda r: r[0])

    param_values = [r[0] for r in records]
    acc_avgs     = [r[1] for r in records]
    delay_avgs   = [r[2] for r in records]
    algo_names   = records[0][3]
    T            = records[0][4]

    # If param is N, convert to int for cleaner labels
    if param_name.lower() == "n":
        param_values = [int(v) for v in param_values]

    print(f"Loaded {len(records)} result files for param='{param_name}'", flush=True)
    print(f"  Values    : {param_values}", flush=True)
    print(f"  Algorithms: {algo_names}", flush=True)
    print(f"  T         : {T}", flush=True)

    return param_values, acc_avgs, delay_avgs, algo_names, T


# ============================================================================
# Plotting  (mirrors plot_summary in run_param_sweep.py)
# ============================================================================

def _linestyles():
    return ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def _plot_summary(
    param_values: List,
    param_name: str,
    all_acc_avg: List[Dict[str, np.ndarray]],
    all_delay_avg: List[Dict[str, np.ndarray]],
    algo_names: List[str],
    T: int,
    out_dir: str,
    focus_algo: str = "No-CSI (Alg1) + LCB(90%)",
    ref_algo: str   = "CSI-aware (optimal)",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    times     = np.arange(T)
    ls_cycle  = _linestyles()
    colors    = plt.cm.tab10.colors  # type: ignore[attr-defined]
    markevery = max(1, T // 10)
    param_markers = ["o", "s", "^", "D", "v"]

    def _side_by_side(metric_all, ylabel, suptitle, filename, add_hline=False):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        for i, (pval, metric_avg) in enumerate(zip(param_values, metric_all)):
            color  = colors[i % len(colors)]
            marker = param_markers[i % len(param_markers)]
            ls     = ls_cycle[i % len(ls_cycle)]
            label  = f"{param_name}={pval}"
            kw = dict(linewidth=2.0, linestyle=ls, color=color,
                      marker=marker, markevery=markevery, markersize=5)
            if focus_algo in metric_avg:
                ax_l.plot(times, metric_avg[focus_algo], label=label, **kw)
            if ref_algo in metric_avg:
                ax_r.plot(times, metric_avg[ref_algo], label=label, **kw)
        for ax, algo_label in ((ax_l, focus_algo), (ax_r, ref_algo)):
            if add_hline:
                ax.axhline(1.0, color="black", lw=1.2, ls=":", alpha=0.8,
                           label="feasibility=1")
            ax.set_xlabel("Time slot t")
            ax.set_title(algo_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=1, fontsize=8)
        ax_l.set_ylabel(ylabel)
        fig.suptitle(f"{suptitle}  (varying {param_name})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=180)
        plt.close()
        print(f"  Saved: {filename}", flush=True)

    # Diagram 1: accuracy over time
    _side_by_side(
        all_acc_avg,
        ylabel="Cumulative avg accuracy",
        suptitle="Accuracy over time",
        filename=f"summary_accuracy_over_time_vs_{param_name}.png",
    )

    # Diagram 2: delay over time
    _side_by_side(
        all_delay_avg,
        ylabel="Cumulative avg D_k / (1/R_k)",
        suptitle="Delay / (1/R_k) over time",
        filename=f"summary_delay_over_time_vs_{param_name}.png",
        add_hline=True,
    )

    # Diagrams 3 & 4: bar charts at t=T-1
    bar_width = 0.8 / max(len(algo_names), 1)
    x = np.arange(len(param_values))

    final_acc   = {n: [float(acc_avg.get(n, [np.nan]*T)[-1])   for acc_avg   in all_acc_avg]   for n in algo_names}
    final_delay = {n: [float(delay_avg.get(n, [np.nan]*T)[-1]) for delay_avg in all_delay_avg] for n in algo_names}

    def _bar_chart(data_dict, ylabel, title, filename, add_hline=False):
        fig, ax = plt.subplots(figsize=(max(10, len(param_values) * 2), 6))
        for i, name in enumerate(algo_names):
            offset = (i - len(algo_names) / 2 + 0.5) * bar_width
            ec = "black" if name in (focus_algo, ref_algo) else "none"
            lw = 1.2   if name in (focus_algo, ref_algo) else 0
            ax.bar(x + offset, data_dict[name],
                   width=bar_width * 0.9, label=name,
                   color=colors[i % len(colors)],
                   edgecolor=ec, linewidth=lw)
        if add_hline:
            ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.8,
                       label="feasibility=1")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_values])
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=180)
        plt.close()
        print(f"  Saved: {filename}", flush=True)

    _bar_chart(
        final_acc,
        ylabel=f"Cumulative avg accuracy at t={T-1}",
        title=f"Final accuracy vs {param_name}",
        filename=f"summary_final_accuracy_vs_{param_name}.png",
    )
    _bar_chart(
        final_delay,
        ylabel=f"Cumulative avg D_k/(1/R_k) at t={T-1}",
        title=f"Final delay/(1/R_k) vs {param_name}",
        filename=f"summary_final_delay_vs_{param_name}.png",
        add_hline=True,
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate parallel sweep results and produce summary plots."
    )
    p.add_argument(
        "--experiment", type=str, required=True,
        help="Which parameter was swept: 'sigma' or 'N'.",
    )
    p.add_argument(
        "--sweep_dir", type=str, required=True,
        help="Directory containing sweep_result.npz files (produced by run_one_sweep_value.py).",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for summary plots (default: <sweep_dir>/summary).",
    )
    p.add_argument(
        "--focus_algo", type=str, default="No-CSI (Alg1) + LCB(90%)",
        help="Algorithm name for the left panel of side-by-side plots.",
    )
    p.add_argument(
        "--ref_algo", type=str, default="CSI-aware (optimal)",
        help="Algorithm name for the right panel of side-by-side plots.",
    )
    args = p.parse_args()

    param_name = args.experiment.strip()
    out_dir    = args.out_dir or os.path.join(args.sweep_dir, "summary")

    print("=" * 60, flush=True)
    print(f"plot_sweep_summary.py  —  param={param_name}", flush=True)
    print(f"  sweep_dir  : {args.sweep_dir}", flush=True)
    print(f"  out_dir    : {out_dir}", flush=True)
    print("=" * 60, flush=True)

    param_values, all_acc_avg, all_delay_avg, algo_names, T = _load_sweep_results(
        args.sweep_dir, param_name
    )

    _plot_summary(
        param_values  = param_values,
        param_name    = param_name,
        all_acc_avg   = all_acc_avg,
        all_delay_avg = all_delay_avg,
        algo_names    = algo_names,
        T             = T,
        out_dir       = out_dir,
        focus_algo    = args.focus_algo,
        ref_algo      = args.ref_algo,
    )

    print(f"\nAll summary plots saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
