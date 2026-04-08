"""
plot_accuracy_function.py — Accuracy function visualization
=============================================================

For each strategy, evaluates and plots three curves:

  1. A_true(η)   — full-sample accuracy at a fine eta grid
                   (what the plots show; the "ground truth")

  2. A_fast(η)   — fast-sample accuracy at the same grid
                   (what the Stein oracle calls internally)

  3. A_smooth(η) — Gaussian-convolved accuracy:
                   A_smooth(η) = (1/M) Σ_m A_fast(η + σ·z_m),  z_m ~ N(0,1)
                   This is the function the Stein oracle is actually
                   differentiating. If A_true is non-monotone but
                   A_smooth is monotone, the oracle operates on a
                   better-behaved landscape than the raw function.

Additionally plots:
  - Vertical dashed line at η_min
  - Shaded region below η_min (infeasible)
  - Numerical first-difference (finite-difference gradient) of A_true
    on a twin y-axis to show where the function increases / decreases

Usage
-----
  # All strategies, Llama-3.1-8B:
  python plot_accuracy_function.py

  # Specific strategies:
  python plot_accuracy_function.py --strategies magnitude topk_per_token

  # Custom fine grid (20 points):
  python plot_accuracy_function.py --n_grid 20

  # More smoothing samples for A_smooth:
  python plot_accuracy_function.py --smooth_M 50

  # Full example:
  export LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
  export LLM_MEASURED_MODEL=llama_3_1_8b
  export LLM_USE_MEASURED_TAU_A=1
  python plot_accuracy_function.py \\
      --strategies magnitude topk_per_token random quantization llmint8 \\
      --n_grid 15 --smooth_M 30 --sigma 0.05 \\
      --out_dir outputs/accuracy_function/llama

Output
------
  <out_dir>/
    accuracy_function_<strategy>.png   — one plot per strategy
    accuracy_function_all.png          — all strategies on one figure
    accuracy_function_data.json        — raw data for all strategies
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))


# ============================================================================
# Core evaluation
# ============================================================================

def evaluate_strategy(
    strategy:    str,
    model_name:  str,
    eta_min:     float,
    eta_grid:    np.ndarray,
    fast_samples: int,
    full_samples: int,
    batch_size:  int,
    sigma:       float,
    smooth_M:    int,
    seed:        int,
) -> dict:
    """
    Load the backend for one strategy and evaluate A_true, A_fast, A_smooth
    across eta_grid.
    """
    os.environ['LLM_MODEL_NAME']               = model_name
    os.environ['LLM_ACTIVATION_STRATEGY']      = strategy
    os.environ['LLM_SPLIT_MODE']               = 'thirds'
    os.environ['MMLU_FAST_SAMPLES_PER_SUBJECT'] = str(fast_samples)
    os.environ['MMLU_FULL_SAMPLES_PER_SUBJECT'] = str(full_samples)
    os.environ['MMLU_BATCH_SIZE']              = str(batch_size)
    os.environ['MMLU_SEED']                    = str(seed)
    os.environ['INFERENCE_OPTIMIZER_BACKEND']  = 'llm'

    from src.core.llm_task_callables import build_llm_backend_from_env
    api = build_llm_backend_from_env(eta_min=eta_min)

    num_links = 2   # Llama/Gemma: 3 nodes, 2 links
    rng = np.random.default_rng(seed)

    a_true   = []
    a_fast   = []
    a_smooth = []

    print(f"\n[{strategy}] Evaluating {len(eta_grid)} grid points "
          f"+ {smooth_M} smoothing samples each ...", flush=True)

    for idx, eta_val in enumerate(eta_grid):
        eta_vec = np.array([eta_val] * num_links, dtype=float)

        # A_true
        vt = api.accuracy_callable_true(eta_vec)
        a_true.append(float(vt))

        # A_fast
        vf = api.accuracy_callable(eta_vec)
        a_fast.append(float(vf))

        # A_smooth — Monte Carlo Gaussian convolution
        # A_smooth(η) = E_{z~N(0,1)}[A_fast(η + σ·z)]
        smooth_vals = []
        for _ in range(smooth_M):
            z        = rng.standard_normal(num_links)
            eta_pert = np.clip(eta_vec + sigma * z, eta_min, 1.0)
            smooth_vals.append(float(api.accuracy_callable(eta_pert)))
        a_smooth.append(float(np.mean(smooth_vals)))

        print(f"  η={eta_val:.3f}  A_true={vt:.4f}  A_fast={vf:.4f}  "
              f"A_smooth={a_smooth[-1]:.4f}", flush=True)

    return {
        "strategy":  strategy,
        "eta_min":   eta_min,
        "sigma":     sigma,
        "smooth_M":  smooth_M,
        "eta_grid":  eta_grid.tolist(),
        "a_true":    a_true,
        "a_fast":    a_fast,
        "a_smooth":  a_smooth,
    }


# ============================================================================
# Plotting — single strategy
# ============================================================================

def plot_strategy(data: dict, out_dir: str, model_name: str) -> str:
    strategy  = data["strategy"]
    eta_grid  = np.array(data["eta_grid"])
    a_true    = np.array(data["a_true"])
    a_fast    = np.array(data["a_fast"])
    a_smooth  = np.array(data["a_smooth"])
    eta_min   = data["eta_min"]
    sigma     = data["sigma"]
    smooth_M  = data["smooth_M"]

    # Finite-difference gradient of A_true
    fd_grad = np.gradient(a_true, eta_grid)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Top panel: accuracy curves ────────────────────────────────────────────
    ax1.axvspan(0.0, eta_min, alpha=0.08, color="red", label="Infeasible (η < η_min)")
    ax1.axvline(eta_min, color="red", linewidth=1.2, linestyle="--",
                label=f"η_min = {eta_min}")

    ax1.plot(eta_grid, a_true,   "b-o",  linewidth=2.0, markersize=5,
             label="$A_{\\mathrm{true}}(\\eta)$ — full samples")
    ax1.plot(eta_grid, a_fast,   "g--s", linewidth=1.5, markersize=4,
             label="$A_{\\mathrm{fast}}(\\eta)$ — fast samples (oracle input)")
    smooth_label = f"$A_{{\\mathrm{{smooth}}}}(\\eta)$ — Gaussian conv. ($\\sigma$={sigma}, M={smooth_M})"
    ax1.plot(eta_grid, a_smooth, "r-^",  linewidth=2.0, markersize=5,
             label=smooth_label)

    # Shade between A_true and A_smooth to show smoothing effect
    ax1.fill_between(eta_grid, a_true, a_smooth, alpha=0.10, color="purple",
                     label="Smoothing effect")

    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(
        f"Accuracy function: {strategy}  —  {model_name.split('/')[-1]}\n"
        f"η_min={eta_min}  σ={sigma}  smooth_M={smooth_M}",
        fontsize=11,
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(
        max(0, min(a_true.min(), a_smooth.min()) - 0.05),
        min(1, max(a_true.max(), a_smooth.max()) + 0.05),
    )

    # ── Bottom panel: finite-difference gradient ──────────────────────────────
    ax2.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax2.axvline(eta_min, color="red", linewidth=1.2, linestyle="--")
    ax2.plot(eta_grid, fd_grad, "b-o", linewidth=1.5, markersize=4,
             label="$\\nabla A_{\\mathrm{true}}$ (finite diff.)")

    # Shade positive/negative regions
    ax2.fill_between(eta_grid, fd_grad, 0,
                     where=(fd_grad >= 0), alpha=0.15, color="green",
                     label="Positive gradient (optimizer pushes η up)")
    ax2.fill_between(eta_grid, fd_grad, 0,
                     where=(fd_grad < 0), alpha=0.15, color="red",
                     label="Negative gradient (optimizer pushes η down)")

    ax2.set_xlabel("η (compression ratio)", fontsize=11)
    ax2.set_ylabel("$\\nabla A$", fontsize=11)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"accuracy_function_{strategy}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)
    return path


# ============================================================================
# Plotting — all strategies on one figure
# ============================================================================

def plot_all(all_data: list, out_dir: str, model_name: str) -> str:
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, data in zip(axes, all_data):
        strategy  = data["strategy"]
        eta_grid  = np.array(data["eta_grid"])
        a_true    = np.array(data["a_true"])
        a_smooth  = np.array(data["a_smooth"])
        eta_min   = data["eta_min"]
        sigma     = data["sigma"]

        ax.axvspan(0.0, eta_min, alpha=0.08, color="red")
        ax.axvline(eta_min, color="red", linewidth=1.0, linestyle="--")
        ax.plot(eta_grid, a_true,   "b-o",  linewidth=1.8, markersize=4,
                label="$A_{\\mathrm{true}}$")
        smooth_lbl = "$A_{\\mathrm{smooth}}$" + f" ($\\sigma$={sigma})"
        ax.plot(eta_grid, a_smooth, "r-^",  linewidth=1.8, markersize=4,
                label=smooth_lbl)
        ax.fill_between(eta_grid, a_true, a_smooth, alpha=0.10, color="purple")
        ax.set_title(strategy, fontsize=10)
        ax.set_xlabel("η", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Accuracy functions — {model_name.split('/')[-1]}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "accuracy_function_all.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Plot A_true, A_fast, and Gaussian-smoothed accuracy "
                    "functions for each LLM compression strategy."
    )
    p.add_argument(
        "--strategies", nargs="+",
        default=["magnitude", "topk_per_token", "random", "quantization", "llmint8"],
    )
    p.add_argument(
        "--model_name", type=str,
        default=os.environ.get("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    p.add_argument(
        "--eta_min", type=float, default=0.25,
    )
    p.add_argument(
        "--n_grid", type=int, default=15,
        help="Number of evenly-spaced eta points from eta_min to 1.0.",
    )
    p.add_argument(
        "--sigma", type=float, default=0.05,
        help="Gaussian smoothing sigma (same as Stein oracle sigma).",
    )
    p.add_argument(
        "--smooth_M", type=int, default=30,
        help="Monte Carlo samples for Gaussian convolution estimate.",
    )
    p.add_argument(
        "--fast_samples", type=int,
        default=int(os.environ.get("MMLU_FAST_SAMPLES_PER_SUBJECT", "50")),
    )
    p.add_argument(
        "--full_samples", type=int,
        default=int(os.environ.get("MMLU_FULL_SAMPLES_PER_SUBJECT", "100")),
    )
    p.add_argument(
        "--batch_size", type=int,
        default=int(os.environ.get("MMLU_BATCH_SIZE", "8")),
    )
    p.add_argument(
        "--seed", type=int, default=0,
    )
    p.add_argument(
        "--out_dir", type=str, default="outputs/accuracy_function",
    )

    args = p.parse_args()

    eta_grid = np.linspace(args.eta_min, 1.0, args.n_grid)

    print("=" * 65, flush=True)
    print("plot_accuracy_function.py", flush=True)
    print(f"  model       = {args.model_name}", flush=True)
    print(f"  strategies  = {args.strategies}", flush=True)
    print(f"  eta grid    = {args.n_grid} points from "
          f"{args.eta_min:.3f} to 1.000", flush=True)
    print(f"  sigma       = {args.sigma}  smooth_M = {args.smooth_M}", flush=True)
    print(f"  fast/full   = {args.fast_samples}/{args.full_samples} per subject", flush=True)
    print("=" * 65, flush=True)

    all_data = []
    for strategy in args.strategies:
        data = evaluate_strategy(
            strategy     = strategy,
            model_name   = args.model_name,
            eta_min      = args.eta_min,
            eta_grid     = eta_grid,
            fast_samples = args.fast_samples,
            full_samples = args.full_samples,
            batch_size   = args.batch_size,
            sigma        = args.sigma,
            smooth_M     = args.smooth_M,
            seed         = args.seed,
        )
        all_data.append(data)
        plot_strategy(data, args.out_dir, args.model_name)

    plot_all(all_data, args.out_dir, args.model_name)

    # Save raw data
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "accuracy_function_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nData saved to: {json_path}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
