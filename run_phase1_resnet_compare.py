#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_to_baselines import _compute_accuracy, _compute_delay_and_excess
from src.core.task_handler import get_task
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
    ConservativeSingleTaskBaseline,
    LCBSingleTaskBaseline,
    MaxCompressionSingleTaskBaseline,
    MovingAverageSingleTaskBaseline,
    MyopicSingleTaskBaseline,
    NoCompressionSingleTaskBaseline,
    UniformCompressionSingleTaskBaseline,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE_PATH = ROOT / "assets" / "resnet56_cifar10_15Mbps_scenario_trace_topk.json"

# Accuracy estimator configuration.
# Change these globals before calling run_compare_trace() if you want to switch modes.
# Set to "stein_estimator" for the original Stein-gradient path, or "fitting_model" to use the fitted poly3 .pkl model.
ACCURACY_ESTIMATOR_MODE = "fitting_model"
# Default is the topk poly3 model. If you switch COMPRESSOR_NAME, update this to the matching quantization/llmint8 pkl.
FITTING_MODEL_PATH = ROOT / "assets" / "jetson_resnet_3tp_poly3_flex.pkl"
# Set to "topk", "quantization", or "llmint8".
COMPRESSOR_NAME = "topk"
# Required only when COMPRESSOR_NAME == "llmint8".
LLMINT8_MAPPING_PATH = ROOT / "assets" / "raw_accuracy_resnet56_llmint8_eta_mapping.json"
LLMINT8_POLICY = "fp16_int4"
LLMINT8_OUTLIER_PRECISION = "fp16"
LLMINT8_REGULAR_PRECISION = "int4"


def _canonical_compressor_name(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "topk": "topk",
        "quant": "quantization",
        "quantization": "quantization",
        "llmint8": "llmint8",
        "llm_int8": "llmint8",
        "llm.int8": "llmint8",
    }
    if normalized not in aliases:
        raise ValueError("Unsupported COMPRESSOR_NAME={!r}".format(value))
    return aliases[normalized]


def _apply_accuracy_config() -> None:
    os.environ["RESNET56_ACCURACY_ESTIMATOR_MODE"] = str(ACCURACY_ESTIMATOR_MODE)
    resolved_path = Path(FITTING_MODEL_PATH).expanduser().resolve()
    os.environ["RESNET56_FITTING_MODEL_PATH"] = str(resolved_path)
    os.environ["RESNET56_COMPRESSOR_NAME"] = _canonical_compressor_name(COMPRESSOR_NAME)
    os.environ["RESNET56_LLMINT8_MAPPING_PATH"] = str(Path(LLMINT8_MAPPING_PATH).expanduser().resolve())
    os.environ["RESNET56_LLMINT8_POLICY"] = str(LLMINT8_POLICY)
    os.environ["RESNET56_LLMINT8_OUTLIER_PRECISION"] = str(LLMINT8_OUTLIER_PRECISION)
    os.environ["RESNET56_LLMINT8_REGULAR_PRECISION"] = str(LLMINT8_REGULAR_PRECISION)


def _describe_fitting_model_target() -> str:
    return str(Path(FITTING_MODEL_PATH).expanduser().resolve())


def _load_channel_trace(trace_path: Path, T: Optional[int]) -> np.ndarray:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    channel = np.asarray(payload["trace"]["channel_true_bps"], dtype=float)
    if channel.ndim != 2:
        raise ValueError(f"Expected 2-D channel_true_bps, got shape {channel.shape}")
    global_positive = channel[channel > 0.0]
    if global_positive.size == 0:
        raise ValueError("Scenario trace channel_true_bps does not contain any positive values")
    global_fallback = float(np.max(global_positive))
    repaired = channel.copy()
    repaired_count = 0
    for row_idx in range(repaired.shape[0]):
        row = repaired[row_idx]
        bad_mask = row <= 0.0
        if not np.any(bad_mask):
            continue
        positive_row = row[row > 0.0]
        replacement = float(np.max(positive_row)) if positive_row.size > 0 else global_fallback
        row[bad_mask] = replacement
        repaired_count += int(np.sum(bad_mask))
    if repaired_count > 0:
        print(
            f"Repaired {repaired_count} non-positive channel values in scenario trace "
            f"using per-slot max positive fallback"
        )
    channel = repaired
    if T is None:
        return channel
    if int(T) <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if channel.shape[0] < int(T):
        raise ValueError(
            f"Scenario trace length {channel.shape[0]} is smaller than requested T={T}"
        )
    return channel[: int(T)]


def run_compare_trace(
    *,
    trace_path: Path,
    T: Optional[int],
    seed: int,
    out_dir: str,
    baseline_algorithms: Optional[List[str]] = None,
    no_csi_mu: float = 3.0,
    no_csi_epsilon: float = 0.1,
    R_k: float = 10.0,
    task_name: str = "resnet56_cifar10_topk",
    M: int = 4,
) -> Tuple[str, str, str]:
    _apply_accuracy_config()
    _ = np.random.default_rng(seed)
    c_t_hist = _load_channel_trace(trace_path=trace_path, T=T)
    horizon = int(c_t_hist.shape[0])

    task = get_task(task_name, task_id=1, R_k=R_k)
    if int(task.L_k) != int(M):
        raise ValueError(f"--M={M} must match task path length L_k={task.L_k} from task recipe {task_name!r}.")
    num_links = M - 1
    if c_t_hist.shape[1] != num_links:
        raise ValueError(
            f"Scenario num_links={c_t_hist.shape[1]} does not match requested M={M} (num_links={num_links})"
        )

    algos: List[Tuple[str, BaseOptimizer]] = [("CSI-Aware (optimal)", CSIAwareSingleTaskOptimizer(M=M, tasks=[task]))]

    lcb = MeanMinusZStdLCB(
        num_links=num_links,
        warmup_value=float(np.median(c_t_hist)),
        z=1.2815515655446004,
        warmup=5,
    )
    no_csi = NoCSISingleTaskOptimizer(
        M=M,
        tasks=[task],
        mu=no_csi_mu,
        epsilon=no_csi_epsilon,
    )
    algos.append(("No-CSI (Alg1) + LCB", no_csi))

    baseline_catalog = {
        "max_compression": (
            "CSI-Aware Baseline: max compression",
            MaxCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "no_compression": (
            "CSI-Aware Baseline: no compression",
            NoCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "uniform_compression": (
            "CSI-Aware Baseline: uniform compression",
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
        "lcb": (
            "No-CSI Baseline: LCB",
            LCBSingleTaskBaseline(M=M, tasks=[task]),
        ),
    }
    selected_baselines = list(baseline_catalog.keys()) if baseline_algorithms is None else baseline_algorithms
    for baseline_name in selected_baselines:
        algo_entry = baseline_catalog.get(baseline_name)
        if algo_entry is None:
            valid = ", ".join(sorted(baseline_catalog.keys()))
            raise ValueError(f"Unknown baseline algorithm '{baseline_name}'. Valid options: {valid}.")
        algos.append(algo_entry)

    selected_set = set(selected_baselines)
    myopic_est = LastObservationEstimator(num_links=num_links, warmup_value=float(np.median(c_t_hist))) if "myopic" in selected_set else None
    cons_est = RunningMinEstimator(num_links=num_links, warmup_value=float(np.median(c_t_hist))) if "conservative" in selected_set else None
    ma_est = MovingAverageEstimator(num_links=num_links, window=5, warmup_value=float(np.median(c_t_hist))) if "moving_average" in selected_set else None

    times = np.arange(horizon)
    acc_hist: Dict[str, np.ndarray] = {name: np.full(horizon, np.nan, dtype=float) for name, _ in algos}
    excess_delay_hist: Dict[str, np.ndarray] = {name: np.full(horizon, np.nan, dtype=float) for name, _ in algos}
    eta_hist: Dict[str, np.ndarray] = {name: np.full((horizon, task.L_k - 1), np.nan, dtype=float) for name, _ in algos}

    for t in range(horizon):
        print(f"Time slot: {t}")
        c_t = np.asarray(c_t_hist[t], dtype=float)
        print(f"c_t: {c_t}")
        for name, opt in algos:
            if name == "No-CSI (Alg1) + LCB":
                allocations = opt.optimize(t=t, c_hat=lcb.estimate(t))
            elif name == "No-CSI Baseline: LCB":
                allocations = opt.optimize(t=t, c_hat=lcb.estimate(t))
            elif name == "No-CSI Baseline: myopic":
                if myopic_est is None:
                    raise RuntimeError("Myopic estimator is not initialized.")
                allocations = opt.optimize(t=t, c_hat=myopic_est.estimate(t))
            elif name == "No-CSI Baseline: conservative":
                if cons_est is None:
                    raise RuntimeError("Conservative estimator is not initialized.")
                allocations = opt.optimize(t=t, c_hat=cons_est.estimate(t))
            elif name == "No-CSI Baseline: moving average":
                if ma_est is None:
                    raise RuntimeError("Moving-average estimator is not initialized.")
                allocations = opt.optimize(t=t, c_hat=ma_est.estimate(t))
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                fallback_eta = np.array(
                    [float(task.eta_min[i]) for i in range(task.b_k, task.e_k)],
                    dtype=float,
                )
                allocations = {
                    task.task_id: {
                        "eta": fallback_eta,
                        "s_comp": np.ones(task.L_k, dtype=float),
                        "s_comm": np.ones(task.L_k - 1, dtype=float),
                    }
                }

            acc_hist[name][t] = _compute_accuracy(task, allocations)
            delay, excess_delay = _compute_delay_and_excess(opt, task, t, allocations, c_t)
            excess_delay_hist[name][t] = excess_delay
            eta_hist[name][t, :] = np.array(allocations[task.task_id]["eta"], dtype=float)
            if name == "No-CSI (Alg1) + LCB":
                opt.update_dual(t=t, actual_delays={task.task_id: delay})

        lcb.update(c_t)
        if myopic_est is not None:
            myopic_est.update(c_t)
        if cons_est is not None:
            cons_est.update(c_t)
        if ma_est is not None:
            ma_est.update(c_t)

    os.makedirs(out_dir, exist_ok=True)

    acc_avg = {name: np.array([np.nanmean(acc_hist[name][: t + 1]) for t in range(horizon)], dtype=float) for name, _ in algos}
    excess_delay_avg = {
        name: np.array([np.nanmean(excess_delay_hist[name][: t + 1]) for t in range(horizon)], dtype=float)
        for name, _ in algos
    }
    eta_avg = {
        name: np.array([np.nanmean(eta_hist[name][: t + 1, :], axis=0) for t in range(horizon)], dtype=float)
        for name, _ in algos
    }

    algo_names = [name for name, _ in algos]
    acc_hist_stack = np.stack([acc_hist[name] for name in algo_names], axis=0)
    excess_delay_hist_stack = np.stack([excess_delay_hist[name] for name in algo_names], axis=0)
    eta_hist_stack = np.stack([eta_hist[name] for name in algo_names], axis=0)
    acc_avg_stack = np.stack([acc_avg[name] for name in algo_names], axis=0)
    excess_delay_avg_stack = np.stack([excess_delay_avg[name] for name in algo_names], axis=0)
    eta_avg_stack = np.stack([eta_avg[name] for name in algo_names], axis=0)
    tradeoff_x = np.array([float(acc_avg[name][-1]) for name in algo_names], dtype=float)
    tradeoff_y = np.array([float(excess_delay_avg[name][-1]) for name in algo_names], dtype=float)

    data_path = os.path.join(out_dir, "simulation_results.npz")
    np.savez_compressed(
        data_path,
        times=times,
        c_t_hist=c_t_hist,
        algo_names=np.array(algo_names, dtype=str),
        acc_hist=acc_hist_stack,
        excess_delay_hist=excess_delay_hist_stack,
        eta_hist=eta_hist_stack,
        acc_avg=acc_avg_stack,
        excess_delay_avg=excess_delay_avg_stack,
        eta_avg=eta_avg_stack,
        tradeoff_x=tradeoff_x,
        tradeoff_y=tradeoff_y,
        task_L_k=np.array([int(task.L_k)], dtype=int),
    )

    metadata_path = os.path.join(out_dir, "simulation_metadata.json")
    metadata = {
        "T": int(horizon),
        "seed": int(seed),
        "R_k": float(R_k),
        "num_links": int(num_links),
        "M": int(M),
        "task_id": int(task.task_id),
        "L_k": int(task.L_k),
        "no_csi_mu": float(no_csi_mu),
        "no_csi_epsilon": float(no_csi_epsilon),
        "baseline_algorithms_used": list(selected_baselines),
        "algo_names": algo_names,
        "task_name": str(task_name),
        "compressor_name": str(getattr(task, "compressor_name", COMPRESSOR_NAME)),
        "delay_metric": "excess_delay_D_minus_invR",
        "trace_path": str(trace_path),
        "channel_source": "scenario_trace.channel_true_bps",
        "accuracy_estimator_mode": str(getattr(task, "accuracy_estimator_mode", "unknown")),
        "fitting_model_type": getattr(task, "fitting_model_type", None),
        "fitting_model_path": getattr(task, "fitting_model_path", None),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    marker_map = {name: markers[i % len(markers)] for i, (name, _) in enumerate(algos)}
    markevery = max(1, horizon // 10)

    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(times, acc_avg[name], label=name, linewidth=1.5, marker=marker_map[name], markevery=markevery, markersize=5)
    plt.xlabel("Time slot t")
    plt.ylabel("Average Accuracy (1/t) * sum_{s=0}^{t} A_k(eta(s))")
    plt.title("Cumulative average accuracy over time (single task, trace-driven)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    acc_path = os.path.join(out_dir, "accuracy_over_time.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    for name, _ in algos:
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
    plt.title("Cumulative average compression over time (single task, trace-driven)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=7)
    eta_path = os.path.join(out_dir, "eta_over_time.png")
    plt.tight_layout()
    plt.savefig(eta_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(
            times,
            excess_delay_avg[name],
            label=name,
            linewidth=1.5,
            marker=marker_map[name],
            markevery=markevery,
            markersize=5,
        )
    plt.xlabel("Time slot t")
    plt.ylabel("Average excess delay (1/t) * sum_s (D_k(s) - 1/R_k(s))")
    plt.title("Cumulative average excess delay over time (single task, trace-driven)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    res_path = os.path.join(out_dir, "excess_delay_over_time.png")
    plt.tight_layout()
    plt.savefig(res_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, _ in algos:
        x = float(acc_avg[name][-1])
        y = float(excess_delay_avg[name][-1])
        plt.scatter(x, y, marker=marker_map[name], s=80, label=name)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    plt.xlabel("Cumulative average accuracy at t = T-1")
    plt.ylabel("Cumulative avg excess delay D_k - 1/R_k at t = T-1")
    plt.title("Accuracy vs excess delay at t = T-1 (trace-driven)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=8)
    tradeoff_path = os.path.join(out_dir, "tradeoff_acc_vs_excess_delay_Tminus1.png")
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()

    return acc_path, res_path, tradeoff_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace-driven Phase 1 launcher for Peyman ResNet56+CIFAR10+topk task.")
    parser.add_argument("--trace_path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu", type=float, default=6.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--R", type=float, default=5.0)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument(
        "--baseline_algorithms",
        type=str,
        default="max_compression,no_compression,uniform_compression,myopic,conservative,moving_average,lcb",
        help="Comma-separated baseline algorithms to include. Use 'none' to disable all baselines.",
    )
    parser.add_argument("--out_dir", default="outputs/phase1_resnet_compare")
    args = parser.parse_args()

    baseline_arg = args.baseline_algorithms.strip().lower()
    baseline_algorithms = [] if baseline_arg == "none" else [part.strip() for part in args.baseline_algorithms.split(",") if part.strip()]

    trace_path = Path(args.trace_path).expanduser().resolve()
    _apply_accuracy_config()
    print(f"Using trace={trace_path}")
    print("Channel source: scenario trace channel_true_bps (direct replay)")
    print("Compressor: {}".format(os.environ["RESNET56_COMPRESSOR_NAME"]))
    print(
        "Accuracy estimator mode: {}".format(
            os.environ["RESNET56_ACCURACY_ESTIMATOR_MODE"]
        )
    )
    if os.environ["RESNET56_ACCURACY_ESTIMATOR_MODE"] == "fitting_model":
        print("Fitting model path: {}".format(_describe_fitting_model_target()))
    if os.environ["RESNET56_COMPRESSOR_NAME"] == "llmint8":
        print("LLM.int8 mapping path: {}".format(os.environ["RESNET56_LLMINT8_MAPPING_PATH"]))
        print("LLM.int8 policy: {}".format(os.environ["RESNET56_LLMINT8_POLICY"]))

    run_compare_trace(
        trace_path=trace_path,
        T=args.T,
        seed=args.seed,
        out_dir=args.out_dir,
        baseline_algorithms=baseline_algorithms,
        no_csi_mu=args.mu,
        no_csi_epsilon=args.epsilon,
        R_k=args.R,
        task_name="resnet56_cifar10_{}".format(_canonical_compressor_name(COMPRESSOR_NAME)),
        M=args.M,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
