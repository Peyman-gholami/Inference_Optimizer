"""
Compare optimizers to baselines on a single-task setting.

Runs a simulation for T time slots with i.i.d. uniform channel capacities and compares:
  - Accuracy A_k(eta(t)) over time
  - delay/(1/R_k) = D_k(t) * R_k(t) over time (computed from returned allocations)

Algorithms included:
  - CSI-aware closed-form single-task optimizer (perfect c_t)
  - No-CSI single-task optimizer (Algorithm 1) fed by a lower-confidence-bound (LCB) estimator
  - Single-task baselines in src/optimizers/baseline.py:
      1) MaxCompressionSingleTaskBaseline
      2) NoCompressionSingleTaskBaseline
      3) UniformCompressionSingleTaskBaseline
      4) MyopicSingleTaskBaseline
      5) ConservativeSingleTaskBaseline

Usage:
  python compare_to_baselines.py --T 100 --seed 0
  python compare_to_baselines.py --T 100 --seed 0 \
      --baseline_algorithms max_compression,myopic,moving_average
  python compare_to_baselines.py --T 100 --seed 0 --baseline_algorithms none
  python compare_to_baselines.py --mu 3.0 --epsilon 0.1
"""

import argparse
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np

# Use a non-interactive backend by default (safe for headless runs)
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.core.task import InferenceTask
from src.optimizers.base import BaseOptimizer
from src.optimizers.csi_aware import CSIAwareSingleTaskOptimizer
from src.optimizers.no_csi import NoCSISingleTaskOptimizer
from src.optimizers.estimators import (
    LastObservationEstimator,
    MeanMinusZStdLCB,
    MovingAverageEstimator,
    RunningMinEstimator,
)
from src.optimizers.baseline import (
    MaxCompressionSingleTaskBaseline,
    NoCompressionSingleTaskBaseline,
    UniformCompressionSingleTaskBaseline,
    MyopicSingleTaskBaseline,
    ConservativeSingleTaskBaseline,
    MovingAverageSingleTaskBaseline,
)

# Reuse the exact model + callables + task builder from main.py.
# We mirror main.py's path setup so `src/core/task_callables.py` can import core modules.
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

from main import setup_model_and_callables, make_task  # noqa: E402


# ----------------------------
# Simulation runner
# ----------------------------


def _compute_accuracy(task: InferenceTask, allocations: Dict[int, Dict[str, np.ndarray]]) -> float:
    eta = allocations[task.task_id]["eta"]
    return float(task.A_k_true(eta))


def _compute_delay_and_ratio(
    optimizer: BaseOptimizer,
    task: InferenceTask,
    t: int,
    allocations: Dict[int, Dict[str, np.ndarray]],
    c_t: np.ndarray,
) -> Tuple[float, float]:
    residuals = optimizer.compute_residual_delay(t, allocations, c_t)
    residual = float(residuals[task.task_id])
    R_k = float(task.get_R_k(t))
    inv_R = 1.0 / R_k
    delay = residual + inv_R
    ratio = delay / inv_R  # delay / (1/R_k)
    return delay, ratio


def run_compare(
    T: int,
    seed: int,
    c_min: float,
    c_max: float,
    out_dir: str,
    baseline_algorithms: Optional[List[str]] = None,
    no_csi_mu: float = 3.0,
    no_csi_epsilon: float = 0.1,
 ) -> Tuple[str, str, str]:
    rng = np.random.default_rng(seed)

    # Single task and topology (M=3 -> 2 links)
    api, _ = setup_model_and_callables()
    task = make_task(task_id=1, api=api)
    M = 3
    num_links = M - 1

    # Algorithms
    algos: List[Tuple[str, BaseOptimizer]] = [("CSI-aware (optimal)", CSIAwareSingleTaskOptimizer(M=M, tasks=[task]))]

    # No-CSI with LCB estimator (fed as c_hat)
    lcb = MeanMinusZStdLCB(
        num_links=num_links,
        warmup_value=c_min,
        z=1.2815515655446004,
        warmup=5,
    )
    no_csi = NoCSISingleTaskOptimizer(
        M=M,
        tasks=[task],
        mu=no_csi_mu,
        epsilon=no_csi_epsilon,
    )
    algos.append(("No-CSI (Alg1) + LCB(90%)", no_csi))
    
    baseline_catalog = {
        "max_compression": (
            "Baseline: max compression",
            MaxCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "no_compression": (
            "Baseline: no compression",
            NoCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "uniform_compression": (
            "Baseline: uniform compression",
            UniformCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "myopic": (
            "No-CSI Baseline: myopic",
            MyopicSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "conservative": (
            "No-CSI Baseline: conservative",
            ConservativeSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "moving_average": (
            "No-CSI Baseline: moving average",
            MovingAverageSingleTaskBaseline(M=M, tasks=[task]),
        ),
    }
    selected_baselines = (
        list(baseline_catalog.keys()) if baseline_algorithms is None else baseline_algorithms
    )
    for baseline_name in selected_baselines:
        algo_entry = baseline_catalog.get(baseline_name)
        if algo_entry is None:
            valid = ", ".join(sorted(baseline_catalog.keys()))
            raise ValueError(
                f"Unknown baseline algorithm '{baseline_name}'. "
                f"Valid options: {valid}."
            )
        algos.append(algo_entry)

    

    # Online estimators for selected estimated-CSI baselines only.
    # (Baselines themselves consume `c_hat`; estimators are updated after observing `c_t`.)
    selected_set = set(selected_baselines)
    myopic_est = (
        LastObservationEstimator(num_links=num_links, warmup_value=c_min)
        if "myopic" in selected_set
        else None
    )
    cons_est = (
        RunningMinEstimator(num_links=num_links, warmup_value=c_min)
        if "conservative" in selected_set
        else None
    )
    ma_est = (
        MovingAverageEstimator(num_links=num_links, window=5, warmup_value=c_min)
        if "moving_average" in selected_set
        else None
    )

    # Storage
    times = np.arange(T)
    acc_hist: Dict[str, np.ndarray] = {name: np.full(T, np.nan, dtype=float) for name, _ in algos}
    delay_ratio_hist: Dict[str, np.ndarray] = {
        name: np.full(T, np.nan, dtype=float) for name, _ in algos
    }
    eta_hist: Dict[str, np.ndarray] = {
        name: np.full((T, task.L_k - 1), np.nan, dtype=float) for name, _ in algos
    }

    # Run
    c_t_hist = np.full((T, num_links), np.nan, dtype=float)
    for t in range(T):
        print(f"Time slot: {t}")
        # Realized channel capacities
        c_t = rng.uniform(low=c_min, high=c_max, size=num_links).astype(float)
        c_t_hist[t, :] = c_t
        print(f"c_t: {c_t}")

        for name, opt in algos:
            if name == "No-CSI (Alg1) + LCB(90%)":
                c_hat = lcb.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "No-CSI Baseline: myopic":
                if myopic_est is None:
                    raise RuntimeError("Myopic estimator is not initialized.")
                c_hat = myopic_est.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "No-CSI Baseline: conservative":
                if cons_est is None:
                    raise RuntimeError("Conservative estimator is not initialized.")
                c_hat = cons_est.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "No-CSI Baseline: moving average":
                if ma_est is None:
                    raise RuntimeError("Moving-average estimator is not initialized.")
                c_hat = ma_est.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            # In this comparison we don't expect None, but be robust.
            if allocations is None:
                continue

            acc_hist[name][t] = _compute_accuracy(task, allocations)
            delay, delay_over_invR = _compute_delay_and_ratio(opt, task, t, allocations, c_t)
            delay_ratio_hist[name][t] = delay_over_invR
            eta_hist[name][t, :] = np.array(allocations[task.task_id]["eta"], dtype=float)

            # Dual update for No-CSI optimizer uses realized delay under true c_t.
            if name == "No-CSI (Alg1) + LCB(90%)":
                opt.update_dual(t=t, actual_delays={task.task_id: delay})

        # After decisions and execution, all estimators get to observe c_t.
        lcb.update(c_t)
        if myopic_est is not None:
            myopic_est.update(c_t)
        if cons_est is not None:
            cons_est.update(c_t)
        if ma_est is not None:
            ma_est.update(c_t)

    os.makedirs(out_dir, exist_ok=True)

    # Compute cumulative averages: avg[t] = mean(values[0:t+1])
    acc_avg: Dict[str, np.ndarray] = {}
    delay_ratio_avg: Dict[str, np.ndarray] = {}
    eta_avg: Dict[str, np.ndarray] = {}
    for name, _ in algos:
        acc_avg[name] = np.array(
            [np.nanmean(acc_hist[name][: t + 1]) for t in range(T)], dtype=float
        )
        delay_ratio_avg[name] = np.array(
            [np.nanmean(delay_ratio_hist[name][: t + 1]) for t in range(T)], dtype=float
        )
        eta_avg[name] = np.array(
            [np.nanmean(eta_hist[name][: t + 1, :], axis=0) for t in range(T)], dtype=float
        )

    # ----------------------------
    # Save all simulation results for later plotting
    # ----------------------------
    algo_names = [name for name, _ in algos]
    n_algos = len(algo_names)
    L = int(task.L_k - 1)

    # Stack per-algorithm dictionaries into dense tensors for easier re-loading.
    acc_hist_stack = np.stack([acc_hist[name] for name in algo_names], axis=0)  # (A, T)
    delay_ratio_hist_stack = np.stack(
        [delay_ratio_hist[name] for name in algo_names], axis=0
    )  # (A, T)
    eta_hist_stack = np.stack([eta_hist[name] for name in algo_names], axis=0)  # (A, T, L)

    acc_avg_stack = np.stack([acc_avg[name] for name in algo_names], axis=0)  # (A, T)
    delay_ratio_avg_stack = np.stack(
        [delay_ratio_avg[name] for name in algo_names], axis=0
    )  # (A, T)
    eta_avg_stack = np.stack([eta_avg[name] for name in algo_names], axis=0)  # (A, T, L)

    tradeoff_x = np.array([float(acc_avg[name][-1]) for name in algo_names], dtype=float)
    tradeoff_y = np.array(
        [float(delay_ratio_avg[name][-1]) for name in algo_names], dtype=float
    )

    data_path = os.path.join(out_dir, "simulation_results.npz")
    np.savez_compressed(
        data_path,
        times=times,
        c_t_hist=c_t_hist,
        algo_names=np.array(algo_names, dtype=str),
        acc_hist=acc_hist_stack,
        delay_ratio_hist=delay_ratio_hist_stack,
        eta_hist=eta_hist_stack,
        acc_avg=acc_avg_stack,
        delay_ratio_avg=delay_ratio_avg_stack,
        eta_avg=eta_avg_stack,
        tradeoff_x=tradeoff_x,
        tradeoff_y=tradeoff_y,
        task_L_k=np.array([int(task.L_k)], dtype=int),
    )

    metadata_path = os.path.join(out_dir, "simulation_metadata.json")
    metadata = {
        "T": int(T),
        "seed": int(seed),
        "c_min": float(c_min),
        "c_max": float(c_max),
        "num_links": int(num_links),
        "M": int(M),
        "task_id": int(task.task_id),
        "L_k": int(task.L_k),
        "no_csi_mu": float(no_csi_mu),
        "no_csi_epsilon": float(no_csi_epsilon),
        "baseline_algorithms_used": list(selected_baselines),
        "algo_names": algo_names,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Markers (distinct per algorithm) to make plots readable.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    marker_map = {name: markers[i % len(markers)] for i, (name, _) in enumerate(algos)}
    # Put a marker about ~10 times across the horizon.
    markevery = max(1, T // 10)

    # ----------------------------
    # Plot 1: Average accuracy over time
    # ----------------------------
    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(
            times,
            acc_avg[name],
            label=name,
            linewidth=1.5,
            marker=marker_map[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Average Accuracy (1/t) * sum_{s=0}^{t} A_k(eta(s))")
    plt.title("Cumulative average accuracy over time (single task)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    acc_path = os.path.join(out_dir, "accuracy_over_time.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()

    # ----------------------------
    # Plot 2: Average eta over time (per link)
    # ----------------------------
    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        # Plot per-link cumulative averages; label once per (algo, link)
        for link_local_idx in range(task.L_k - 1):
            plt.plot(
                times,
                eta_avg[name][:, link_local_idx],
                label=f"{name} (eta[{link_local_idx}])",
                linewidth=1.2,
                marker=marker_map[name],
                markevery=markevery,
                markersize=4,
            )
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative avg eta")
    plt.title("Cumulative average compression over time (single task)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=7)
    eta_path = os.path.join(out_dir, "eta_over_time.png")
    plt.tight_layout()
    plt.savefig(eta_path, dpi=180)
    plt.close()

    # ----------------------------
    # Plot 3: Average delay / (1/R_k) over time
    # ----------------------------
    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(
            times,
            delay_ratio_avg[name],
            label=name,
            linewidth=1.5,
            marker=marker_map[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel(
        "Average D_k / (1/R_k) = (1/t) * sum_{s=0}^{t} D_k(s) / (1/R_k(s))"
    )
    plt.title("Cumulative average delay / (1/R_k) over time (single task)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    res_path = os.path.join(out_dir, "delay_over_invR_over_time.png")
    plt.tight_layout()
    plt.savefig(res_path, dpi=180)
    plt.close()

    # ----------------------------
    # Plot 4: Trade-off at t = T-1 (one point per algorithm)
    # ----------------------------
    plt.figure(figsize=(8, 6))
    for name, _ in algos:
        # acc_avg[name] / delay_ratio_avg[name] are cumulative averages over time,
        # so the last entry corresponds to t = T-1.
        x = float(acc_avg[name][-1])
        y = float(delay_ratio_avg[name][-1])
        plt.scatter(
            x,
            y,
            marker=marker_map[name],
            s=80,
            label=name,
        )
    plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.8)
    plt.xlabel("Cumulative average accuracy at t = T-1")
    plt.ylabel("Cumulative avg D_k / (1/R_k) at t = T-1")
    plt.title("Accuracy vs feasibility (delay/(1/R)) at t = T-1")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=8)
    tradeoff_path = os.path.join(out_dir, "tradeoff_acc_vs_delay_over_invR_Tminus1.png")
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()

    return acc_path, res_path, tradeoff_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--c_min", type=float, default=10.0)
    p.add_argument("--c_max", type=float, default=150.0)
    p.add_argument(
        "--mu",
        type=float,
        default=3.0,
        help="No-CSI (Alg1) parameter mu.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="No-CSI (Alg1) minimum dual variable value epsilon.",
    )
    p.add_argument("--out_dir", type=str, default="outputs/baseline_compare")
    p.add_argument(
        "--baseline_algorithms",
        type=str,
        default="max_compression,no_compression,uniform_compression,myopic,conservative,moving_average",
        help=(
            "Comma-separated baseline algorithms to include. "
            "Use 'none' to disable all baselines. "
            "Options: max_compression, no_compression, uniform_compression, "
            "myopic, conservative, moving_average."
        ),
    )
    args = p.parse_args()
    baseline_arg = args.baseline_algorithms.strip().lower()
    if baseline_arg == "none":
        baseline_algorithms = []
    else:
        baseline_algorithms = [
            part.strip()
            for part in args.baseline_algorithms.split(",")
            if part.strip()
        ]

    acc_path, res_path, tradeoff_path = run_compare(
        T=args.T,
        seed=args.seed,
        c_min=args.c_min,
        c_max=args.c_max,
        out_dir=args.out_dir,
        baseline_algorithms=baseline_algorithms,
        no_csi_mu=args.mu,
        no_csi_epsilon=args.epsilon,
    )

    print(f"Saved accuracy plot to: {acc_path}")
    print(f"Saved delay/(1/R) plot to: {res_path}")
    print(f"Saved trade-off plot to: {tradeoff_path}")
    print(
        f"Saved raw simulation data to: {os.path.join(args.out_dir, 'simulation_results.npz')}"
    )
    print(
        f"Saved simulation metadata to: {os.path.join(args.out_dir, 'simulation_metadata.json')}"
    )


if __name__ == "__main__":
    main()
