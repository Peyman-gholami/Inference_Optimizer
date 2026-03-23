"""
Compare multi-task optimizers to multi-task baselines.

Runs a simulation for T time slots with i.i.d. uniform channel capacities and compares:
  - Average weighted accuracy over time
  - Average delay/(1/R_k) over tasks over time (computed from returned allocations under true c_t)

Algorithms included:
  - CSI-aware multi-task optimizer (convex solver, perfect c_t)
  - CSI-aware multi-task baselines:
      1) StaticEqualShareMultiTaskBaseline
      2) ProportionalResourceAllocationMultiTaskBaseline
      3) StrictPriorityGreedyMultiTaskBaseline
  - No-CSI multi-task optimizer (Algorithm 2) fed by an estimator (LCB by default)
  - No-CSI multi-task baselines:
      4) DecoupledEqualSplitStochasticDescentMultiTaskBaseline
      5) QueueProportionalHeuristicMultiTaskBaseline
      6) HistoricalAverageCertaintyEquivalenceMultiTaskBaseline (certainty equivalence)

Usage:
  python compare_to_baselines_multi.py --T 200 --seed 0 --K 2
"""

import argparse
import os
from typing import Dict, List, Tuple

import sys

import numpy as np

# Use a non-interactive backend by default (safe for headless runs)
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.core.task import InferenceTask
from src.optimizers.base import BaseOptimizer
from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer
from src.optimizers.no_csi import NoCSIMultiTaskOptimizer
from src.optimizers.estimators import MeanEstimator, MeanMinusZStdLCB
from src.optimizers.baseline import (
    StaticEqualShareMultiTaskBaseline,
    ProportionalResourceAllocationMultiTaskBaseline,
    StrictPriorityGreedyMultiTaskBaseline,
    DecoupledEqualSplitStochasticDescentMultiTaskBaseline,
    QueueProportionalHeuristicMultiTaskBaseline,
    HistoricalAverageCertaintyEquivalenceMultiTaskBaseline,
)

# Reuse the exact model + callables + task builder from main.py.
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

from main import setup_model_and_callables, make_task  # noqa: E402


def _weighted_accuracy(tasks: List[InferenceTask], allocations: Dict[int, Dict[str, np.ndarray]]) -> float:
    vals = []
    wsum = 0.0
    for task in tasks:
        eta = allocations[task.task_id]["eta"]
        vals.append(float(task.w_k) * float(task.A_k_true(eta)))
        wsum += float(task.w_k)
    return float(sum(vals) / wsum) if wsum > 0 else float(np.mean(vals))


def _avg_delay_over_invR(
    optimizer: BaseOptimizer,
    tasks: List[InferenceTask],
    t: int,
    allocations: Dict[int, Dict[str, np.ndarray]],
    c_t: np.ndarray,
) -> float:
    residuals = optimizer.compute_residual_delay(t, allocations, c_t)
    ratios = []
    for task in tasks:
        residual = float(residuals[task.task_id])
        R_k = float(task.get_R_k(t))
        inv_R = 1.0 / R_k
        delay = residual + inv_R
        ratios.append(delay / inv_R)
    return float(np.mean(ratios))


def run_compare(
    T: int,
    seed: int,
    K: int,
    c_min: float,
    c_max: float,
    out_dir: str,
    mu: float,
    epsilon: float,
) -> Tuple[str, str, str]:
    rng = np.random.default_rng(seed)

    api, _ = setup_model_and_callables()

    # Topology (fixed by make_task): 3 nodes -> 2 links
    M = 3
    num_links = M - 1

    # Tasks (all share the same path in current demo)
    tasks: List[InferenceTask] = []
    for k in range(K):
        tasks.append(make_task(task_id=k + 1, api=api, w_k=(k+1)*10))

    # Make tasks heterogeneous: different per-link data sizes a_{i,k}.
    # This helps separate equal-share vs proportional vs priority baselines.
    if len(tasks) >= 2:
        # Task 1: smaller payloads (faster comm)
        tasks[0].a = {0: 5.0, 1: 40.0}
        # Task 2: larger payloads (slower comm)
        tasks[1].a = {0: 20.0, 1: 25.0}

    # Algorithms
    csi_opt = CSIAwareMultiTaskOptimizer(M=M, tasks=tasks)
    no_csi_opt = NoCSIMultiTaskOptimizer(M=M, tasks=tasks, mu=mu, epsilon=epsilon, J=10)

    algos: List[Tuple[str, BaseOptimizer]] = [
        ("CSI-aware (optimal)", csi_opt),
        ("Baseline: equal-share", StaticEqualShareMultiTaskBaseline(M=M, tasks=tasks)),
        ("Baseline: proportional", ProportionalResourceAllocationMultiTaskBaseline(M=M, tasks=tasks)),
        ("Baseline: strict priority", StrictPriorityGreedyMultiTaskBaseline(M=M, tasks=tasks)),
        ("No-CSI (Alg2) + LCB", no_csi_opt),
        ("No-CSI baseline: decoupled equal-split", DecoupledEqualSplitStochasticDescentMultiTaskBaseline(M=M, tasks=tasks, mu=mu, epsilon=epsilon)),
        ("No-CSI baseline: queue-proportional", QueueProportionalHeuristicMultiTaskBaseline(M=M, tasks=tasks, mu=mu, epsilon=epsilon)),
        # ("No-CSI baseline: certainty equivalence", HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(M=M, tasks=tasks)),
    ]

    # Channel estimators
    lcb = MeanMinusZStdLCB(num_links=num_links, warmup_value=c_max, z=1.2815515655446004, warmup=5)
    hist_mean = MeanEstimator(num_links=num_links, warmup_value=c_min, warmup=5)

    times = np.arange(T)
    acc_hist: Dict[str, np.ndarray] = {name: np.full(T, np.nan, dtype=float) for name, _ in algos}
    delay_ratio_hist: Dict[str, np.ndarray] = {
        name: np.full(T, np.nan, dtype=float) for name, _ in algos
    }

    for t in range(T):
        print(f"Time slot: {t}")
        c_t = rng.uniform(low=c_min, high=c_max, size=num_links).astype(float)

        for name, opt in algos:
            if name in {
                "No-CSI (Alg2) + LCB",
                "No-CSI baseline: decoupled equal-split",
                "No-CSI baseline: queue-proportional",
            }:
                c_hat = lcb.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "No-CSI baseline: certainty equivalence":
                c_hat = hist_mean.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            # if allocations is None:
            #     continue
            print(f"algo: {name}, allocations: {allocations}")
            acc_hist[name][t] = _weighted_accuracy(tasks, allocations)
            delay_ratio_hist[name][t] = _avg_delay_over_invR(opt, tasks, t, allocations, c_t)

            # Dual updates for algorithms that track queues
            if name in {
                "No-CSI (Alg2) + LCB",
                "No-CSI baseline: decoupled equal-split",
                "No-CSI baseline: queue-proportional",
            }:
                # compute realized per-task delays under true c_t
                residuals = opt.compute_residual_delay(t, allocations, c_t)
                actual_delays = {}
                for task in tasks:
                    tid = task.task_id
                    actual_delays[tid] = float(residuals[tid]) + 1.0 / float(task.get_R_k(t))
                opt.update_dual(t=t, actual_delays=actual_delays)

        # Estimators observe c_t after actions
        lcb.update(c_t)
        hist_mean.update(c_t)

    os.makedirs(out_dir, exist_ok=True)

    acc_avg = {name: np.array([np.nanmean(acc_hist[name][: i + 1]) for i in range(T)], dtype=float) for name, _ in algos}
    delay_ratio_avg = {
        name: np.array(
            [np.nanmean(delay_ratio_hist[name][: i + 1]) for i in range(T)], dtype=float
        )
        for name, _ in algos
    }

    # Markers (distinct per algorithm) to make plots readable.
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    marker_map = {name: markers[i % len(markers)] for i, (name, _) in enumerate(algos)}
    markevery = max(1, T // 10)

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
    plt.ylabel("Cumulative avg weighted accuracy")
    plt.title("Multi-task: cumulative average weighted accuracy over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    acc_path = os.path.join(out_dir, "accuracy_over_time.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()

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
    plt.ylabel("Cumulative avg (1/K) * sum_k D_k / (1/R_k)")
    plt.title("Multi-task: cumulative average delay / (1/R_k) over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    res_path = os.path.join(out_dir, "delay_over_invR_over_time.png")
    plt.tight_layout()
    plt.savefig(res_path, dpi=180)
    plt.close()

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
    plt.xlabel("Cumulative avg weighted accuracy at t = T-1")
    plt.ylabel("Cumulative avg (1/K) * sum_k D_k / (1/R_k) at t = T-1")
    plt.title("Multi-task: accuracy vs feasibility at t = T-1")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=8)
    tradeoff_path = os.path.join(out_dir, "tradeoff_acc_vs_delay_over_invR_Tminus1.png")
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()

    return acc_path, res_path, tradeoff_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--seed", type=int, default=567)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--c_min", type=float, default=40.0)
    p.add_argument("--c_max", type=float, default=150.0)
    p.add_argument("--mu", type=float, default=10.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="outputs/baseline_compare_multi")
    args = p.parse_args()

    acc_path, res_path, tradeoff_path = run_compare(
        T=args.T,
        seed=args.seed,
        K=args.K,
        c_min=args.c_min,
        c_max=args.c_max,
        out_dir=args.out_dir,
        mu=args.mu,
        epsilon=args.epsilon,
    )

    print(f"Saved accuracy plot to: {acc_path}")
    print(f"Saved delay/(1/R) plot to: {res_path}")
    print(f"Saved trade-off plot to: {tradeoff_path}")


if __name__ == "__main__":
    main()

