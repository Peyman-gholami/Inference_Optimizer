"""
Compare multi-task optimizers to multi-task baselines.

Runs a simulation for T time slots with i.i.d. uniform channel capacities and compares:
  - Average weighted accuracy over time
  - Average delay/(1/R_k) over tasks over time (computed from returned allocations under true c_t)

Algorithms included:
  - CSI-aware multi-task optimizer (convex solver, perfect c_t)
  - CSI-aware multi-task baselines:
      0) MaxCompressionMultiTaskBaseline, NoCompressionMultiTaskBaseline
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
  python compare_to_baselines_multi.py --baseline_algorithms equal_share,strict_priority
  python compare_to_baselines_multi.py --baseline_algorithms none
"""

import argparse
import os
import json
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
    MaxCompressionMultiTaskBaseline,
    NoCompressionMultiTaskBaseline,
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
from src.core.llm_plug_in import llm_use_measured_tau_a  # noqa: E402


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
    baseline_algorithms: List[str],
) -> Tuple[str, str, str]:
    rng = np.random.default_rng(seed)

    api, _ = setup_model_and_callables()

    # Topology derived from the task itself so this works for any backend
    # (MLP: L_k=3 / 2 links, ResNet default: L_k=4 / 3 links).
    tasks: List[InferenceTask] = []
    for k in range(K):
        tasks.append(make_task(task_id=k + 1, api=api, w_k=(k+1)))

    M = tasks[0].L_k
    num_links = tasks[0].L_k - 1

    # Make tasks heterogeneous: scale per-link data sizes proportionally so
    # the heterogeneity makes sense for any number of links and any backend.
    # Skip when using measured LLM payloads so per-task `a` stays table-aligned.
    # Also skip for the ResNet backend where `a` is set from real measured bytes.
    from src.core.llm_plug_in import is_resnet_backend
    if len(tasks) >= 2 and not llm_use_measured_tau_a() and not is_resnet_backend():
        # Task 1: smaller payloads (faster comm) — scale base a by 0.5
        tasks[0].a = {i: tasks[0].a[i] * 0.5 for i in range(num_links)}
        # Task 2: larger payloads (slower comm) — scale base a by 2.0
        tasks[1].a = {i: tasks[1].a[i] * 2.0 for i in range(num_links)}

    # Algorithms
    csi_opt = CSIAwareMultiTaskOptimizer(M=M, tasks=tasks)
    no_csi_opt = NoCSIMultiTaskOptimizer(M=M, tasks=tasks, mu=mu, epsilon=epsilon, J=10)

    algos: List[Tuple[str, BaseOptimizer]] = [
        ("CSI-aware (optimal)", csi_opt),
        ("No-CSI (Alg2) + LCB", no_csi_opt),
    ]
    baseline_catalog = {
        "max_compression": (
            "Baseline: max compression",
            MaxCompressionMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "no_compression": (
            "Baseline: no compression",
            NoCompressionMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "equal_share": (
            "Baseline: equal-share",
            StaticEqualShareMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "proportional": (
            "Baseline: proportional",
            ProportionalResourceAllocationMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "strict_priority": (
            "Baseline: strict priority",
            StrictPriorityGreedyMultiTaskBaseline(M=M, tasks=tasks),
        ),
        "decoupled_equal_split": (
            "No-CSI Baseline: decoupled equal-split",
            DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
                M=M, tasks=tasks, mu=mu, epsilon=epsilon
            ),
        ),
        "queue_proportional": (
            "No-CSI Baseline: queue-proportional",
            QueueProportionalHeuristicMultiTaskBaseline(
                M=M, tasks=tasks, mu=mu, epsilon=epsilon
            ),
        ),
        "certainty_equivalence": (
            "No-CSI Baseline: moving average",
            HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(M=M, tasks=tasks),
        ),
    }
    for baseline_name in baseline_algorithms:
        algo_entry = baseline_catalog.get(baseline_name)
        if algo_entry is None:
            valid = ", ".join(sorted(baseline_catalog.keys()))
            raise ValueError(
                f"Unknown baseline algorithm '{baseline_name}'. "
                f"Valid options: {valid}."
            )
        algos.append(algo_entry)

    # Channel estimators
    lcb = MeanMinusZStdLCB(num_links=num_links, warmup_value=c_min, z=1.2815515655446004, warmup=5)
    hist_mean = (
        MeanEstimator(num_links=num_links, warmup_value=c_min, warmup=5)
        if "certainty_equivalence" in set(baseline_algorithms)
        else None
    )

    times = np.arange(T)
    acc_hist: Dict[str, np.ndarray] = {name: np.full(T, np.nan, dtype=float) for name, _ in algos}
    delay_ratio_hist: Dict[str, np.ndarray] = {
        name: np.full(T, np.nan, dtype=float) for name, _ in algos
    }
    c_t_hist = np.full((T, num_links), np.nan, dtype=float)

    # Per-task compression history, for later plotting/debugging.
    # Shape: (A, T, K, L) where L = L_k - 1 (links per task path).
    algo_names = [name for name, _ in algos]
    algo_index = {name: i for i, name in enumerate(algo_names)}
    L = int(tasks[0].L_k - 1)
    eta_hist_stack = np.full((len(algo_names), T, K, L), np.nan, dtype=float)
    task_ids = [int(t.task_id) for t in tasks]

    for t in range(T):
        print(f"Time slot: {t}")
        c_t = rng.uniform(low=c_min, high=c_max, size=num_links).astype(float)
        c_t_hist[t, :] = c_t

        for name, opt in algos:
            if name in {
                "No-CSI (Alg2) + LCB",
                "No-CSI Baseline: decoupled equal-split",
                "No-CSI Baseline: queue-proportional",
            }:
                c_hat = lcb.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "No-CSI Baseline: moving average":
                if hist_mean is None:
                    raise RuntimeError("Historical-mean estimator is not initialized.")
                c_hat = hist_mean.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                print(f"algo: {name}, allocations: None (infeasible) -> skipping t={t}")
                continue
            print(f"algo: {name}, allocations: {allocations}")
            acc_hist[name][t] = _weighted_accuracy(tasks, allocations)
            delay_ratio_hist[name][t] = _avg_delay_over_invR(opt, tasks, t, allocations, c_t)

            # Store per-task eta values if present.
            algo_idx = algo_index[name]
            for k_idx, task in enumerate(tasks):
                eta = np.array(allocations[task.task_id]["eta"], dtype=float)
                if eta.size != L:
                    # Be robust if something changes in task path length.
                    eta = eta.reshape(-1)
                eta_hist_stack[algo_idx, t, k_idx, : min(L, eta.size)] = eta[: min(L, eta.size)]

            # Dual updates for algorithms that track queues
            if name in {
                "No-CSI (Alg2) + LCB",
                "No-CSI Baseline: decoupled equal-split",
                "No-CSI Baseline: queue-proportional",
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
        if hist_mean is not None:
            hist_mean.update(c_t)

    os.makedirs(out_dir, exist_ok=True)

    acc_avg = {name: np.array([np.nanmean(acc_hist[name][: i + 1]) for i in range(T)], dtype=float) for name, _ in algos}
    delay_ratio_avg = {
        name: np.array(
            [np.nanmean(delay_ratio_hist[name][: i + 1]) for i in range(T)], dtype=float
        )
        for name, _ in algos
    }

    # ----------------------------
    # Save all simulation results for later plotting
    # ----------------------------
    acc_hist_stack = np.stack([acc_hist[name] for name in algo_names], axis=0)  # (A, T)
    delay_ratio_hist_stack = np.stack(
        [delay_ratio_hist[name] for name in algo_names], axis=0
    )  # (A, T)
    acc_avg_stack = np.stack([acc_avg[name] for name in algo_names], axis=0)  # (A, T)
    delay_ratio_avg_stack = np.stack(
        [delay_ratio_avg[name] for name in algo_names], axis=0
    )  # (A, T)

    tradeoff_x = np.array([float(acc_avg[name][-1]) for name in algo_names], dtype=float)
    tradeoff_y = np.array(
        [float(delay_ratio_avg[name][-1]) for name in algo_names], dtype=float
    )

    data_path = os.path.join(out_dir, "simulation_results_multi.npz")
    np.savez_compressed(
        data_path,
        times=times,
        c_t_hist=c_t_hist,
        algo_names=np.array(algo_names, dtype=str),
        task_ids=np.array(task_ids, dtype=int),
        acc_hist=acc_hist_stack,
        delay_ratio_hist=delay_ratio_hist_stack,
        eta_hist=eta_hist_stack,
        acc_avg=acc_avg_stack,
        delay_ratio_avg=delay_ratio_avg_stack,
        tradeoff_x=tradeoff_x,
        tradeoff_y=tradeoff_y,
        K=np.array([int(K)], dtype=int),
        task_L_k=np.array([int(tasks[0].L_k)], dtype=int),
    )

    metadata_path = os.path.join(out_dir, "simulation_metadata_multi.json")
    metadata = {
        "T": int(T),
        "seed": int(seed),
        "K": int(K),
        "c_min": float(c_min),
        "c_max": float(c_max),
        "num_links": int(num_links),
        "M": int(M),
        "mu": float(mu),
        "epsilon": float(epsilon),
        "baseline_algorithms_used": list(baseline_algorithms),
        "algo_names": algo_names,
        "task_ids": task_ids,
        "L_k": int(tasks[0].L_k),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

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
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--seed", type=int, default=567)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--c_min", type=float, default=40.0)
    p.add_argument("--c_max", type=float, default=150.0)
    p.add_argument("--mu", type=float, default=10.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument(
        "--baseline_algorithms",
        type=str,
        default="max_compression,no_compression,equal_share,proportional,strict_priority,decoupled_equal_split,queue_proportional,certainty_equivalence",
        help=(
            "Comma-separated baseline algorithms to include. "
            "Use 'none' to disable all baselines. "
            "Options: max_compression, no_compression, equal_share, proportional, "
            "strict_priority, decoupled_equal_split, queue_proportional, certainty_equivalence."
        ),
    )
    p.add_argument("--out_dir", type=str, default="outputs/baseline_compare_multi")
    args = p.parse_args()
    baseline_arg = args.baseline_algorithms.strip().lower()
    if baseline_arg == "none":
        baseline_algorithms: List[str] = []
    else:
        baseline_algorithms = [
            part.strip()
            for part in args.baseline_algorithms.split(",")
            if part.strip()
        ]

    acc_path, res_path, tradeoff_path = run_compare(
        T=args.T,
        seed=args.seed,
        K=args.K,
        c_min=args.c_min,
        c_max=args.c_max,
        out_dir=args.out_dir,
        mu=args.mu,
        epsilon=args.epsilon,
        baseline_algorithms=baseline_algorithms,
    )

    print(f"Saved accuracy plot to: {acc_path}")
    print(f"Saved delay/(1/R) plot to: {res_path}")
    print(f"Saved trade-off plot to: {tradeoff_path}")
    print(
        f"Saved raw simulation data to: {os.path.join(args.out_dir, 'simulation_results_multi.npz')}"
    )
    print(
        f"Saved simulation metadata to: {os.path.join(args.out_dir, 'simulation_metadata_multi.json')}"
    )


if __name__ == "__main__":
    main()

