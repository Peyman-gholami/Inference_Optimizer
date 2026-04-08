"""
replay_baselines.py — Replay baselines from a completed main run
=================================================================

When a main run was executed with --baselines none (to save time),
this script:

  1. Loads c_t_hist and experiment_setting.json from the main run
  2. Loads the LLM/ResNet backend (same model, same strategy)
  3. Runs only the requested baselines, replaying the exact saved
     channel sequence slot by slot
  4. Merges baseline results with the existing main run results
  5. Overwrites simulation_results.npz and regenerates all 4 plots

The channel sequence is reproduced exactly because c_t_hist is
loaded directly from the saved npz — no RNG replay needed.

Usage
-----
  # All 6 baselines:
  python replay_baselines.py \\
      --out_dir outputs/.../llmint8_mu0001

  # Only fast (model-free) baselines (~2h for llmint8):
  python replay_baselines.py \\
      --out_dir outputs/.../llmint8_mu0001 \\
      --baselines max_compression,no_compression,uniform_compression

  # Slurm: run after main job completes
  sbatch --dependency=afterok:<main_jobid> run_replay_baselines.sh
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

from src.core.task import InferenceTask
from src.optimizers.estimators import (
    LastObservationEstimator,
    RunningMinEstimator,
    MovingAverageEstimator,
)
from src.optimizers.baseline import (
    MaxCompressionSingleTaskBaseline,
    NoCompressionSingleTaskBaseline,
    UniformCompressionSingleTaskBaseline,
    MyopicSingleTaskBaseline,
    ConservativeSingleTaskBaseline,
    MovingAverageSingleTaskBaseline,
)

ALL_BASELINES  = [
    "max_compression", "no_compression", "uniform_compression",
    "myopic", "conservative", "moving_average",
]


# ============================================================================
# Load saved run
# ============================================================================

def load_main_run(out_dir: str) -> Tuple[dict, dict]:
    """
    Load experiment_setting.json and simulation_results.npz.
    Returns (setting_dict, npz_data).
    """
    setting_path = os.path.join(out_dir, "experiment_setting.json")
    npz_path     = os.path.join(out_dir, "simulation_results.npz")

    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"No experiment_setting.json in {out_dir}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No simulation_results.npz in {out_dir}")

    with open(setting_path) as f:
        setting = json.load(f)

    npz = dict(np.load(npz_path, allow_pickle=True))
    return setting, npz


# ============================================================================
# Reconstruct ExperimentSetting + InferenceTask from saved JSON
# ============================================================================

def reconstruct_setting_and_task(setting: dict, baselines: List[str]):
    """
    Re-import run_experiment constructs and rebuild the setting dataclass
    and InferenceTask from the saved JSON — then load the backend.
    """
    # Re-use run_experiment's dataclass and load_backend/build_task
    from run_experiment import (
        ExperimentSetting, load_backend, build_task,
        MaxCompressionSingleTaskBaseline,
        NoCompressionSingleTaskBaseline,
        UniformCompressionSingleTaskBaseline,
        MyopicSingleTaskBaseline,
        ConservativeSingleTaskBaseline,
        MovingAverageSingleTaskBaseline,
    )

    # Rebuild ExperimentSetting from JSON
    s = ExperimentSetting(
        backend             = setting["backend"],
        compressor          = setting["compressor"],
        compressor_backend  = setting["compressor_backend"],
        c_min               = setting["c_min"],
        c_max               = setting["c_max"],
        c_dist              = setting["c_dist"],
        R_k                 = setting["R_k"],
        eta_min             = setting["eta_min"],
        T                   = setting["T"],
        seed                = setting["seed"],
        no_csi_mu           = setting["no_csi_mu"],
        no_csi_epsilon      = setting["no_csi_epsilon"],
        llm_model_name          = setting.get("llm_model_name", ""),
        llm_activation_strategy = setting.get("llm_activation_strategy", ""),
        llm_split_mode          = setting.get("llm_split_mode", "thirds"),
        llm_grad_sigma          = setting.get("llm_grad_sigma", 0.05),
        llm_grad_N              = setting.get("llm_grad_N", 3),
        llm_fast_samples        = setting.get("llm_fast_samples", 5),
        llm_full_samples        = setting.get("llm_full_samples", 10),
        llm_batch_size          = setting.get("llm_batch_size", 8),
        resnet_cutpoints        = setting.get("resnet_cutpoints", "8,14,21"),
        resnet_fast_batches     = setting.get("resnet_fast_batches", 1),
        resnet_grad_sigma       = setting.get("resnet_grad_sigma", 0.05),
        resnet_grad_N           = setting.get("resnet_grad_N", 20),
        mlp_sigma               = setting.get("mlp_sigma", 0.05),
        mlp_N                   = setting.get("mlp_N", 50),
        mlp_n_samples           = setting.get("mlp_n_samples", 512),
    )

    # Load backend
    api, _ = load_backend(s)

    # Build task
    task = build_task(s, api)

    return s, task, api


# ============================================================================
# Build baseline algorithms only
# ============================================================================

def build_baseline_algos(
    task:      InferenceTask,
    s,
    baselines: List[str],
):
    M         = task.L_k
    num_links = M - 1
    selected  = set(baselines)

    catalog = {
        "max_compression":    ("Baseline: max compression",
                               MaxCompressionSingleTaskBaseline(M=M, tasks=[task])),
        "no_compression":     ("Baseline: no compression",
                               NoCompressionSingleTaskBaseline(M=M, tasks=[task])),
        "uniform_compression":("Baseline: uniform compression",
                               UniformCompressionSingleTaskBaseline(M=M, tasks=[task])),
        "myopic":             ("Baseline: myopic",
                               MyopicSingleTaskBaseline(M=M, tasks=[task])),
        "conservative":       ("Baseline: conservative",
                               ConservativeSingleTaskBaseline(M=M, tasks=[task])),
        "moving_average":     ("Baseline: moving average",
                               MovingAverageSingleTaskBaseline(M=M, tasks=[task])),
    }

    algos = []
    for key in baselines:
        if key not in catalog:
            raise ValueError(f"Unknown baseline '{key}'. Valid: {sorted(catalog)}")
        algos.append(catalog[key])

    myopic_est = (LastObservationEstimator(num_links=num_links, warmup_value=s.c_min)
                  if "myopic" in selected else None)
    cons_est   = (RunningMinEstimator(num_links=num_links, warmup_value=s.c_min)
                  if "conservative" in selected else None)
    ma_est     = (MovingAverageEstimator(num_links=num_links, window=5,
                                         warmup_value=s.c_min)
                  if "moving_average" in selected else None)

    return algos, myopic_est, cons_est, ma_est


# ============================================================================
# Replay loop — uses saved c_t_hist directly
# ============================================================================

def replay_baselines(
    task:      InferenceTask,
    algos:     list,
    s,
    c_t_hist:  np.ndarray,
    myopic_est, cons_est, ma_est,
) -> dict:
    """
    Replay T slots using the saved c_t_hist.
    Returns history dicts for the baseline algorithms.
    """
    T         = s.T
    num_links = task.L_k - 1
    algo_names = [n for n, _ in algos]

    acc_hist         = {n: np.full(T, np.nan) for n in algo_names}
    delay_ratio_hist = {n: np.full(T, np.nan) for n in algo_names}
    eta_hist         = {n: np.full((T, num_links), np.nan) for n in algo_names}

    # LCB estimator (needed by some baselines internally, though not used
    # for channel estimation — kept for API compatibility)
    from src.optimizers.estimators import MeanMinusZStdLCB
    lcb = MeanMinusZStdLCB(num_links=num_links, warmup_value=s.c_min,
                            z=1.2815515655446004, warmup=5)

    for t in range(T):
        c_t = c_t_hist[t]
        print(f"  t={t:3d}  c_t={np.round(c_t, 1)}", flush=True)

        for name, opt in algos:
            if name == "Baseline: myopic" and myopic_est:
                allocations = opt.optimize(t=t, c_hat=myopic_est.estimate(t))
            elif name == "Baseline: conservative" and cons_est:
                allocations = opt.optimize(t=t, c_hat=cons_est.estimate(t))
            elif name == "Baseline: moving average" and ma_est:
                allocations = opt.optimize(t=t, c_hat=ma_est.estimate(t))
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                continue

            eta = allocations[task.task_id]["eta"]
            acc_hist[name][t] = float(task.A_k_true(eta))

            residuals = opt.compute_residual_delay(t, allocations, c_t)
            residual  = float(residuals[task.task_id])
            R_k       = float(task.get_R_k(t))
            inv_R     = 1.0 / R_k
            delay_ratio_hist[name][t] = (residual + inv_R) / inv_R

            eta_hist[name][t] = np.array(eta, dtype=float)

        lcb.update(c_t)
        if myopic_est: myopic_est.update(c_t)
        if cons_est:   cons_est.update(c_t)
        if ma_est:     ma_est.update(c_t)

    # Cumulative averages
    acc_avg         = {n: np.array([np.nanmean(acc_hist[n][:t+1])
                                    for t in range(T)]) for n in algo_names}
    delay_ratio_avg = {n: np.array([np.nanmean(delay_ratio_hist[n][:t+1])
                                    for t in range(T)]) for n in algo_names}
    eta_avg         = {n: np.array([np.nanmean(eta_hist[n][:t+1], axis=0)
                                    for t in range(T)]) for n in algo_names}

    return dict(
        acc_hist=acc_hist, delay_ratio_hist=delay_ratio_hist, eta_hist=eta_hist,
        acc_avg=acc_avg, delay_ratio_avg=delay_ratio_avg, eta_avg=eta_avg,
        algo_names=algo_names,
    )


# ============================================================================
# Merge and save
# ============================================================================

def merge_and_save(main_npz: dict, baseline_hist: dict, out_dir: str):
    """
    Merge baseline results into the main npz and overwrite simulation_results.npz.
    """
    # Existing main algo names and arrays
    main_names = list(main_npz["algo_names"])
    T          = len(main_npz["times"])
    num_links  = main_npz["c_t_hist"].shape[1]

    bl_names   = baseline_hist["algo_names"]
    all_names  = main_names + bl_names

    # Stack arrays in order: main algorithms first, then baselines
    def _stack(key_main, key_bl):
        main_arr = main_npz[key_main]           # shape (A_main, T, ...)
        bl_list  = [baseline_hist[key_bl][n]
                    for n in bl_names]
        bl_arr   = np.stack(bl_list)            # shape (A_bl, T, ...)
        return np.concatenate([main_arr, bl_arr], axis=0)

    acc_avg_merged          = _stack("acc_avg",          "acc_avg")
    delay_ratio_avg_merged  = _stack("delay_ratio_avg",  "delay_ratio_avg")
    eta_avg_merged          = _stack("eta_avg",          "eta_avg")
    acc_hist_merged         = _stack("acc_hist",         "acc_hist")
    delay_ratio_hist_merged = _stack("delay_ratio_hist", "delay_ratio_hist")

    np.savez_compressed(
        os.path.join(out_dir, "simulation_results.npz"),
        times            = main_npz["times"],
        c_t_hist         = main_npz["c_t_hist"],
        algo_names       = np.array(all_names, dtype=str),
        acc_avg          = acc_avg_merged,
        delay_ratio_avg  = delay_ratio_avg_merged,
        eta_avg          = eta_avg_merged,
        acc_hist         = acc_hist_merged,
        delay_ratio_hist = delay_ratio_hist_merged,
    )
    print(f"[merge] Saved merged simulation_results.npz "
          f"({len(all_names)} algorithms)", flush=True)
    return all_names


# ============================================================================
# Replot using run_experiment's plot_results
# ============================================================================

def replot(out_dir: str, all_names: List[str], main_npz: dict,
           baseline_hist: dict, s):
    """
    Reconstruct the hist dict expected by plot_results and regenerate plots.
    """
    from run_experiment import plot_results

    T         = s.T
    num_links = s.num_links
    times     = main_npz["times"]

    # Rebuild full hist dict
    main_names = list(main_npz["algo_names"])
    bl_names   = baseline_hist["algo_names"]

    acc_avg         = {}
    delay_ratio_avg = {}
    eta_avg         = {}
    acc_hist        = {}
    delay_ratio_hist= {}
    eta_hist        = {}

    for i, n in enumerate(main_names):
        acc_avg[n]          = main_npz["acc_avg"][i]
        delay_ratio_avg[n]  = main_npz["delay_ratio_avg"][i]
        eta_avg[n]          = main_npz["eta_avg"][i]
        acc_hist[n]         = main_npz["acc_hist"][i]
        delay_ratio_hist[n] = main_npz["delay_ratio_hist"][i]
        eta_hist[n]         = np.zeros((T, num_links))  # not saved in npz

    for n in bl_names:
        acc_avg[n]          = baseline_hist["acc_avg"][n]
        delay_ratio_avg[n]  = baseline_hist["delay_ratio_avg"][n]
        eta_avg[n]          = baseline_hist["eta_avg"][n]
        acc_hist[n]         = baseline_hist["acc_hist"][n]
        delay_ratio_hist[n] = baseline_hist["delay_ratio_hist"][n]
        eta_hist[n]         = baseline_hist["eta_hist"][n]

    hist = dict(
        times=times, c_t_hist=main_npz["c_t_hist"],
        acc_hist=acc_hist, delay_ratio_hist=delay_ratio_hist,
        eta_hist=eta_hist,
        acc_avg=acc_avg, delay_ratio_avg=delay_ratio_avg,
        eta_avg=eta_avg,
        algo_names=all_names,
    )

    # Build a minimal task object for plot_results
    from src.core.task import InferenceTask
    task = InferenceTask.__new__(InferenceTask)
    task.task_id = 1
    task.L_k     = num_links + 1
    task.b_k     = 0
    task.e_k     = num_links

    paths = plot_results(task, hist, s, out_dir)
    print("[replot] Plots regenerated:", flush=True)
    for label, path in paths.items():
        print(f"  {label:<12} -> {path}", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Replay baselines from a completed --baselines none run "
                    "and merge results for full plots."
    )
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory of the completed main run.")
    p.add_argument(
        "--baselines", type=str,
        default="max_compression,no_compression,uniform_compression,"
                "myopic,conservative,moving_average",
        help="Comma-separated baseline keys to replay, or 'all'. "
             "Default: all 6 baselines.",
    )
    p.add_argument("--expected_T", type=int, default=None,
                   help="Expected total T of the main run. If provided, "
                        "replay_baselines will abort if the main run has not "
                        "completed all T slots (e.g. was timed out again). "
                        "This prevents merging partial results.")
    args = p.parse_args()

    bl_arg    = args.baselines.strip().lower()
    baselines = ALL_BASELINES if bl_arg == "all" else \
                [b.strip() for b in bl_arg.split(",") if b.strip()]

    print("=" * 65, flush=True)
    print("replay_baselines.py", flush=True)
    print(f"  out_dir   = {args.out_dir}", flush=True)
    print(f"  baselines = {baselines}", flush=True)
    print("=" * 65, flush=True)

    # ── Load main run ─────────────────────────────────────────────────────────
    setting, main_npz = load_main_run(args.out_dir)
    c_t_hist = main_npz["c_t_hist"]
    T        = int(main_npz["times"][-1]) + 1
    print(f"\n[load] Loaded main run: T={T}  "
          f"algos={list(main_npz['algo_names'])}", flush=True)

    # ── Completeness check ────────────────────────────────────────────────────
    if args.expected_T is not None and T < args.expected_T:
        print(f"\n[ERROR] Main run is incomplete: T_done={T} < expected_T={args.expected_T}.",
              flush=True)
        print(f"  The main job likely timed out again. Resubmit the resume job "
              f"before retrying the replay.", flush=True)
        sys.exit(1)

    # ── Reconstruct setting + task ────────────────────────────────────────────
    s, task, api = reconstruct_setting_and_task(setting, baselines)

    # ── Build baselines ───────────────────────────────────────────────────────
    algos, myopic_est, cons_est, ma_est = build_baseline_algos(task, s, baselines)
    print(f"\n[baselines] Running {len(algos)} baselines:", flush=True)
    for name, _ in algos:
        print(f"  • {name}", flush=True)

    # ── Replay ────────────────────────────────────────────────────────────────
    print("\n[replay] Starting baseline replay ...", flush=True)
    baseline_hist = replay_baselines(
        task, algos, s, c_t_hist, myopic_est, cons_est, ma_est
    )

    # ── Merge + save ──────────────────────────────────────────────────────────
    print("\n[merge] Merging with main run ...", flush=True)
    all_names = merge_and_save(main_npz, baseline_hist, args.out_dir)

    # ── Replot ────────────────────────────────────────────────────────────────
    print("\n[replot] Regenerating plots ...", flush=True)
    replot(args.out_dir, all_names, main_npz, baseline_hist, s)

    print("\n" + "=" * 65, flush=True)
    print("Done. Plots and merged npz written to:", args.out_dir, flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
