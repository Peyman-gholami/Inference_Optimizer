"""
run_one_sweep_value.py
======================
Runs a single (sigma, N) combination and saves results as a .npz file.

Designed to be called as one task in a Slurm job array, one task per
parameter value, so all values run in parallel.

Usage
-----
# Single run (sigma=0.05, N=5, T=50):
python run_one_sweep_value.py --param sigma --value 0.05 --fixed_N 5 --T 50

python run_one_sweep_value.py --param N --value 10 --fixed_sigma 0.05 --T 50

# From a Slurm job array (SLURM_ARRAY_TASK_ID selects the value):
#   sbatch --array=0-4 submit_sweep.sh sigma

Output
------
Saves to:
    <out_dir>/sigma=<value>/sweep_result.npz      (sigma sweep)
    <out_dir>/N=<value>/sweep_result.npz          (N sweep)

The .npz contains:
    param_name      str
    param_value     float or int
    algo_names      array of str
    acc_avg         (A, T)  cumulative average accuracy per algorithm
    delay_ratio_avg (A, T)  cumulative average delay ratio per algorithm
    T               int
    seed            int
    sigma           float
    N               int

These files are read by plot_sweep_summary.py to produce summary diagrams.
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — mirrors run_param_sweep.py
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

# Reuse the patched setup and simulation logic from run_param_sweep.py
from run_param_sweep import setup_model_and_callables_patched, run_single  # noqa: E402
from main import make_task  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run one sweep value and save results as .npz."
    )
    p.add_argument("--param",  type=str,   required=True,
                   help="Which parameter to sweep: 'sigma' or 'N'.")
    p.add_argument("--value",  type=str,   required=True,
                   help="Value of the sweep parameter (float for sigma, int for N).")
    p.add_argument("--fixed_N",     type=int,   default=5,
                   help="Fixed N used when sweeping sigma (ResNet default: 5).")
    p.add_argument("--fixed_sigma", type=float, default=0.05,
                   help="Fixed sigma used when sweeping N (default: 0.05).")
    p.add_argument("--T",      type=int,   default=50)
    p.add_argument("--seed",   type=int,   default=12)
    p.add_argument("--c_min",  type=float, default=10.0,
                   help="Channel min (MLP default). Auto-overridden for ResNet/LLM based on task payload sizes.")
    p.add_argument("--c_max",  type=float, default=150.0,
                   help="Channel max (MLP default). Auto-overridden for ResNet/LLM based on task payload sizes.")
    p.add_argument("--mu",     type=float, default=3.0)
    p.add_argument("--epsilon",type=float, default=0.1)
    p.add_argument("--out_dir",type=str,   default="outputs/param_sweep_parallel")
    p.add_argument(
        "--baseline_algorithms",
        type=str, default="none",
        help="Comma-separated baselines or 'none'.",
    )
    args = p.parse_args()

    # Parse value with correct type
    param_name = args.param.strip().lower()
    if param_name == "sigma":
        pval = float(args.value)
        sigma = pval
        N     = args.fixed_N
    elif param_name == "n":
        pval = int(args.value)
        sigma = args.fixed_sigma
        N     = pval
    else:
        raise ValueError(f"--param must be 'sigma' or 'N', got '{args.param}'")

    baseline_arg = args.baseline_algorithms.strip().lower()
    baseline_algorithms = (
        [] if baseline_arg == "none"
        else [s.strip() for s in args.baseline_algorithms.split(",") if s.strip()]
    )

    # ------------------------------------------------------------------ #
    # Print configuration (flush=True for Slurm)                          #
    # ------------------------------------------------------------------ #
    backend = os.environ.get("INFERENCE_OPTIMIZER_BACKEND", "mnist").strip().lower()
    run_label = f"{param_name}={pval}"
    print("=" * 60, flush=True)
    print(f"run_one_sweep_value.py  —  {run_label}", flush=True)
    print("=" * 60, flush=True)
    print(f"  backend    = {backend}", flush=True)
    print(f"  param      = {param_name}", flush=True)
    print(f"  value      = {pval}", flush=True)
    print(f"  sigma      = {sigma}", flush=True)
    print(f"  N          = {N}", flush=True)
    print(f"  T          = {args.T}", flush=True)
    print(f"  seed       = {args.seed}", flush=True)
    print(f"  c_min/max  = {args.c_min} / {args.c_max}", flush=True)
    print(f"  out_dir    = {args.out_dir}", flush=True)
    _ENV_VARS = [
        "INFERENCE_OPTIMIZER_BACKEND", "RESNET_CHECKPOINT",
        "RESNET_CUTPOINTS", "RESNET_FAST_BATCHES",
        "COMPRESSOR_TYPE", "ETA_MIN0", "ETA_MIN1", "R_K",
    ]
    print("  Env vars:", flush=True)
    for var in _ENV_VARS:
        print(f"    {var:<35} = {os.environ.get(var, '(not set)')}", flush=True)
    print("=" * 60, flush=True)

    # ------------------------------------------------------------------ #
    # Run simulation                                                       #
    # ------------------------------------------------------------------ #
    run_out_dir = os.path.join(
        args.out_dir,
        f"exp1_sigma_sweep" if param_name == "sigma" else "exp2_N_sweep",
        run_label.replace("=", "_").replace(".", "p"),
    )

    acc_avg, delay_ratio_avg, algos = run_single(
        T=args.T,
        seed=args.seed,
        c_min=args.c_min,
        c_max=args.c_max,
        sigma=sigma,
        N=N,
        out_dir=run_out_dir,
        baseline_algorithms=baseline_algorithms,
        no_csi_mu=args.mu,
        no_csi_epsilon=args.epsilon,
    )

    # ------------------------------------------------------------------ #
    # Save results as .npz                                                 #
    # ------------------------------------------------------------------ #
    algo_names  = [name for name, _ in algos]
    acc_stack   = np.stack([acc_avg[n]         for n in algo_names], axis=0)  # (A, T)
    delay_stack = np.stack([delay_ratio_avg[n] for n in algo_names], axis=0)  # (A, T)

    npz_path = os.path.join(run_out_dir, "sweep_result.npz")
    np.savez_compressed(
        npz_path,
        param_name   = np.array(param_name,    dtype=str),
        param_value  = np.array(float(pval),   dtype=float),
        algo_names   = np.array(algo_names,    dtype=str),
        acc_avg      = acc_stack,
        delay_ratio_avg = delay_stack,
        T            = np.array(args.T,        dtype=int),
        seed         = np.array(args.seed,     dtype=int),
        sigma        = np.array(sigma,         dtype=float),
        N            = np.array(N,             dtype=int),
    )
    print(f"\nResults saved to: {npz_path}", flush=True)


if __name__ == "__main__":
    main()
