"""
Empirical proxy for Assumption 4 (feasibility margin γ).

Assumption 4: under a feasible allocation at eta^{min}, E[D_k] ≤ 1/R_k(t) - γ.
  We use the max-compression baseline allocation (eta = eta_min, equal resource shares as in
  MaxCompression*Baseline), evaluate true bottleneck delay with c_t, and report time averages
  of (1/R_k(t) - D_k(t)) as an empirical margin; min_t(...) is a conservative per-slot slack.

Usage:
  python measure_delta_gamma_assumptions.py --T 200 --seed 0 --c_min 10 --c_max 100 --R 10 --task toy_mlp_mnist --M 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

import numpy as np

from src.core.task import InferenceTask
from src.core.task_handler import get_task
from src.optimizers.base import BaseOptimizer
from src.optimizers.baseline import (
    MaxCompressionMultiTaskBaseline,
    MaxCompressionSingleTaskBaseline,
)


def compute_bottleneck_delays(
    optimizer: BaseOptimizer,
    t: int,
    allocations: Dict[int, Dict[str, np.ndarray]],
    c: np.ndarray,
) -> Dict[int, float]:
    """
    D_k(t) = max compute delay, max comm delay) using capacities c (true or estimated).

    Implemented as residual + 1/R_k(t) from BaseOptimizer.compute_residual_delay.
    """
    residuals = optimizer.compute_residual_delay(t, allocations, np.asarray(c, dtype=float))
    out: Dict[int, float] = {}
    for task in optimizer.tasks:
        tid = task.task_id
        out[tid] = float(residuals[tid]) + 1.0 / float(task.get_R_k(t))
    return out


def _max_compression_allocations(
    M: int,
    tasks: List[InferenceTask],
    c_t_dummy: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Fixed eta^{min} allocation with codebase equal-split shares (single- or multi-task)."""
    c = np.asarray(c_t_dummy, dtype=float)
    if len(tasks) == 1:
        opt: BaseOptimizer = MaxCompressionSingleTaskBaseline(M=M, tasks=tasks)
    else:
        opt = MaxCompressionMultiTaskBaseline(M=M, tasks=tasks)
    out = opt.optimize(t=0, c_t=c)
    if out is None:
        raise RuntimeError("Max-compression baseline returned None (unexpected).")
    return out


def empirical_gamma_eta_min(
    M: int,
    tasks: List[InferenceTask],
    c_t_hist: np.ndarray,
    fixed_allocations: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    Under Assumption 4, γ satisfies E[D_k] ≤ 1/R_k - γ at eta^{min}.

    We report:
      - mean_t(1/R_k(t) - D_k(t)) as a point estimate of the average feasibility slack;
      - min_t(1/R_k(t) - D_k(t)) as a worst-slot slack (conservative).

    If fixed_allocations is None, uses MaxCompression*Baseline policy (eta_min + equal shares).
    """
    c_t_hist = np.asarray(c_t_hist, dtype=float)
    if c_t_hist.ndim != 2:
        raise ValueError("c_t_hist must be 2D (T, num_links).")
    T, num_links = c_t_hist.shape
    if num_links != M - 1:
        raise ValueError(f"c_t_hist has {num_links} links but M={M} implies M-1={M-1}.")

    if fixed_allocations is None:
        alloc = _max_compression_allocations(M, tasks, c_t_hist[0])
    else:
        alloc = fixed_allocations

    # Reuse any optimizer wrapper that has these tasks and compute_residual_delay.
    opt: BaseOptimizer
    if len(tasks) == 1:
        opt = MaxCompressionSingleTaskBaseline(M=M, tasks=tasks)
    else:
        opt = MaxCompressionMultiTaskBaseline(M=M, tasks=tasks)

    task_ids = [task.task_id for task in tasks]
    slack_series: Dict[int, np.ndarray] = {tid: np.full(T, np.nan) for tid in task_ids}
    d_series: Dict[int, np.ndarray] = {tid: np.full(T, np.nan) for tid in task_ids}

    for t in range(T):
        d_k = compute_bottleneck_delays(opt, t, alloc, c_t_hist[t])
        for task in tasks:
            tid = task.task_id
            inv_r = 1.0 / float(task.get_R_k(t))
            d_series[tid][t] = d_k[tid]
            slack_series[tid][t] = inv_r - d_k[tid]

    summary = {}
    for tid in task_ids:
        s = slack_series[tid]
        summary[tid] = {
            "mean_slack_invR_minus_D": float(np.nanmean(s)),
            "min_slack_invR_minus_D": float(np.nanmin(s)),
            "mean_D": float(np.nanmean(d_series[tid])),
        }

    return {
        "slack_per_task": summary,
        "slack_series": slack_series,
        "D_series": d_series,
        "allocations": alloc,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Empirical γ at eta_min (max-compression baseline).")
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c_min", type=float, default=10.0)
    p.add_argument("--c_max", type=float, default=100.0)
    p.add_argument("--R", type=float, default=10.0, dest="R_k")
    p.add_argument("--task", type=str, default="toy_mlp_mnist")
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--out_json", type=str, default="", help="Optional path to write summary JSON.")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    task = get_task(args.task, task_id=1, R_k=args.R_k)
    if int(task.L_k) != int(args.M):
        raise ValueError(f"M={args.M} must match L_k={task.L_k} for task {args.task!r}.")
    num_links = int(args.M - 1)
    c_t_hist = rng.uniform(low=args.c_min, high=args.c_max, size=(args.T, num_links)).astype(float)

    gamma_res = empirical_gamma_eta_min(args.M, [task], c_t_hist)

    summary = {
        "assumption_4_gamma_proxy": {
            "description": "At eta_min (+ equal shares): mean and min of 1/R_k(t) - D_k(c_t)",
            "per_task": gamma_res["slack_per_task"],
        },
        "channel_stats": {
            "configured_c_min": float(args.c_min),
            "configured_c_max": float(args.c_max),
            "observed_c_min": float(np.min(c_t_hist)),
            "observed_c_max": float(np.max(c_t_hist)),
        },
        "simulation": {
            "T": args.T,
            "seed": args.seed,
            "c_min": args.c_min,
            "c_max": args.c_max,
            "R_k": args.R_k,
            "task": args.task,
            "M": args.M,
        },
    }

    print(json.dumps(summary, indent=2))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
