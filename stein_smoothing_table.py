"""
stein_smoothing_table.py
========================
Measure the function the Stein oracle sees, for the paper's Stein scenarios.

Produces the JSON that `fill_table_stein.py` turns into the LaTeX table: the
Stein-path analogue of Table 7 (tab:fitting_acc_speed).  Where Table 7 asks
"how well does the fitted surrogate A-hat match A", this asks "how far is the
Gaussian-smoothed A_sigma -- the object Stein actually differentiates -- from
A, and what does that cost the controller".

Scenario coverage
-----------------
V-TRS and V-LOO share one A(eta): same task (RN56-CF10), same cut-points
8,14,21 -> a = [0.066, 0.033, 0.016] MB, differing only in the capacity range
and mu, neither of which enters A.  So one run per *compressor* covers both
rows, and --scenario only labels the output.

Usage
-----
  export INFERENCE_OPTIMIZER_BACKEND=resnet
  export RESNET_CHECKPOINT=$PWD/models/resnet56-4bfd9763.th
  export RESNET_DATA_ROOT=$PWD/data
  export RESNET_CUTPOINTS=8,14,21

  python stein_smoothing_table.py --compressor topk \
      --out_dir outputs/stein_smoothing/resnet_topk

  # local smoke test (minutes, not hours)
  python stein_smoothing_table.py --compressor topk --smoke \
      --out_dir outputs/stein_smoothing/smoke_topk

Cost
----
n_grid**3 * len(sigmas) * n_draws evaluations on the fast subset, plus
n_grid**3 on the full test set for the A baseline.  Defaults
(n_grid=6, 5 sigmas, n_draws=200, fast=2x100 images) come to ~2.2e5 fast
evaluations = ~4.3e7 image-forwards, roughly an hour on one A100.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "core"))

from src.analysis.stein_smoothing import (  # noqa: E402
    SmoothingConfig,
    build_eta_grid,
    gradient_stability,
    smooth_grid,
    summarize,
)

# a_i in MB for the V-TRS / V-LOO topology (tables/offline-scenarios-table.tex).
# Only the *shape* matters -- it sets the price direction in the regret sweep.
DEFAULT_LINK_SIZES = [0.066, 0.033, 0.016]


def select_device(name: str):
    """cuda -> mps -> cpu, so the same script runs on Explorer and on the M3.

    The ResNet backend picks cuda-or-cpu on its own, which leaves the Mac on
    CPU; moving the partitions afterwards keeps that file untouched (and so
    still cherry-pickable onto concavity-experiments).
    """
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_backend(args):
    """Return (fast_acc_fn, true_acc_fn, num_links, meta)."""
    os.environ["INFERENCE_OPTIMIZER_BACKEND"] = "resnet"
    os.environ["COMPRESSOR_TYPE"] = args.compressor
    os.environ["COMPRESSOR_BACKEND"] = args.compressor_backend
    os.environ["RESNET_CUTPOINTS"] = args.cutpoints
    os.environ["RESNET_BATCH_SIZE"] = str(args.batch_size)
    os.environ["RESNET_FAST_BATCHES"] = str(args.fast_batches)
    os.environ.setdefault("RESNET_DATA_ROOT", str(_ROOT / "data"))
    os.environ.setdefault("RESNET_DOWNLOAD", "0")
    if args.checkpoint:
        os.environ["RESNET_CHECKPOINT"] = args.checkpoint

    from resnet_task_callables import build_resnet_backend_from_env

    api = build_resnet_backend_from_env()

    device = select_device(args.device)
    if device != api.device:
        api._partitions = [p.to(device) for p in api._partitions]
        api.device = device
        print(f"[stein_smoothing] moved partitions to {device}", flush=True)

    # Cap the "true" evaluation set. Default 'all' = the paper's 10,000 images;
    # a smaller cap makes a local pipeline check tractable on CPU/MPS.
    if args.true_batches != "all":
        api._full_batches = api._full_batches[: int(args.true_batches)]
    n_true = len(api._full_batches) * args.batch_size

    meta = {
        "cutpoints": api.cutpoints,
        "compressor": args.compressor,
        "compressor_backend": args.compressor_backend,
        "device": str(api.device),
        "batch_size": args.batch_size,
        "fast_batches": args.fast_batches,
        "fast_eval_images": args.batch_size * args.fast_batches,
        "true_eval_images": n_true,
    }
    return api.accuracy_callable, api.accuracy_callable_true, api.num_links, meta


def measure(acc_fn, grid, label):
    vals = np.empty(len(grid))
    t0 = time.time()
    for i, eta in enumerate(grid):
        vals[i] = acc_fn(eta)
        if (i + 1) % max(1, len(grid) // 10) == 0:
            print(f"  {label}: {i+1}/{len(grid)}", flush=True)
    print(f"  {label} done in {time.time()-t0:.1f}s  "
          f"range [{vals.min():.4f}, {vals.max():.4f}]", flush=True)
    return vals


def pick_probe_points(grid, a_true, n):
    """Spread the gradient probes over the accuracy range, not just the peak.

    The controller operates wherever the price puts it, so sampling the
    quantiles of A gives a fairer read on gradient stability than probing the
    argmax alone.
    """
    order = np.argsort(a_true)
    idx = np.linspace(0, len(order) - 1, n).astype(int)
    return grid[order[idx]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compressor", default="topk",
                   choices=["topk", "quantization", "llmint8"])
    p.add_argument("--compressor_backend", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--scenario", default="V-TRS/V-LOO",
                   help="Label only; V-TRS and V-LOO share this A(eta).")
    p.add_argument("--task", default="RN56-CF10")
    p.add_argument("--cutpoints", default="8,14,21")
    p.add_argument("--checkpoint", default="",
                   help="Overrides RESNET_CHECKPOINT.")
    p.add_argument("--out_dir", default="outputs/stein_smoothing/run")

    p.add_argument("--sigmas", default="0.01,0.02,0.05,0.1,0.2")
    p.add_argument("--n_grid", type=int, default=6,
                   help="Points per eta axis; total grid is n_grid**3.")
    p.add_argument("--eta_min", type=float, default=0.1)
    p.add_argument("--eta_axis", default="",
                   help="Explicit per-axis eta values, overriding the linspace. "
                        "Required for quantization, which only responds on its "
                        "ladder: 0.0625,0.125,0.25,0.5,1.0")
    p.add_argument("--n_draws", type=int, default=200,
                   help="MC draws per (eta, sigma) for A_sigma.")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--fast_batches", type=int, default=2,
                   help="Batches per fast eval; fast_batches*batch_size images.")
    p.add_argument("--true_batches", default="all",
                   help="Batches for the A baseline ('all' = full 10k test set).")
    p.add_argument("--device", default="auto",
                   help="auto (cuda>mps>cpu) | cuda | mps | cpu")

    p.add_argument("--grad_N", type=int, default=20,
                   help="Antithetic pairs, matching the scenario (20 offline).")
    p.add_argument("--grad_seeds", type=int, default=8)
    p.add_argument("--grad_points", type=int, default=5)
    p.add_argument("--skip_gradient", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny settings for a local pipeline check.")
    args = p.parse_args()

    if args.smoke:
        args.n_grid, args.n_draws, args.sigmas = 3, 12, "0.05,0.2"
        args.grad_seeds, args.grad_points, args.grad_N = 3, 2, 5
        args.fast_batches, args.true_batches = 1, "10"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fast_fn, true_fn, num_links, meta = build_backend(args)

    cfg = SmoothingConfig(
        num_links=num_links,
        sigmas=[float(s) for s in args.sigmas.split(",") if s.strip()],
        n_grid=args.n_grid,
        eta_min=args.eta_min,
        axis_values=([float(v) for v in args.eta_axis.split(",") if v.strip()]
                     if args.eta_axis else None),
        n_draws=args.n_draws,
        seed=args.seed,
        grad_N=args.grad_N,
        grad_seeds=args.grad_seeds,
        grad_points=args.grad_points,
        link_sizes=DEFAULT_LINK_SIZES[:num_links],
    )

    grid = build_eta_grid(cfg)
    total = len(grid) * len(cfg.sigmas) * cfg.n_draws
    print(f"\n[stein_smoothing] {args.scenario} / {args.task} / {args.compressor}")
    axis_desc = (",".join(f"{v:g}" for v in cfg.axis_values)
                 if cfg.axis_values else
                 f"linspace({cfg.eta_min:g},{cfg.eta_max:g},{cfg.n_grid})")
    print(f"  grid       : {cfg.n_grid}^{num_links} = {len(grid)} points "
          f"[{axis_desc}]")
    print(f"  sigmas     : {cfg.sigmas}")
    print(f"  draws      : {cfg.n_draws}")
    print(f"  fast evals : {total:,}  "
          f"({total * meta['fast_eval_images']:,} image-forwards)\n", flush=True)

    print(f"[1/3] A baselines "
          f"({meta['true_eval_images']} images, and the {meta['fast_eval_images']}-image "
          f"subset used for A_sigma)")
    a_true = measure(true_fn, grid, "A_true")
    a_fast = measure(fast_fn, grid, "A_fast")
    offset = a_fast - a_true
    print(f"  fast-subset offset: mean {offset.mean():+.4f}  "
          f"sd {offset.std():.4f}  max|.| {np.abs(offset).max():.4f}\n"
          f"  -> bias columns are measured against A_fast so this cancels",
          flush=True)

    print("\n[2/3] A_sigma = E_z[A(eta + sigma z)] (fast subset, common random numbers)")
    smoothed = smooth_grid(fast_fn, grid, cfg)

    stability = {}
    if not args.skip_gradient:
        print("\n[3/3] Stein gradient stability across seeds")
        probes = pick_probe_points(grid, a_true, cfg.grad_points)
        for sigma in cfg.sigmas:
            stability[sigma] = gradient_stability(fast_fn, probes, sigma, cfg)
            print(f"  sigma={sigma:<5g} rel_sd={stability[sigma]['rel_sd']:.3f}  "
                  f"cos_seed={stability[sigma]['cos_seed']:.3f}", flush=True)
    else:
        print("\n[3/3] skipped (--skip_gradient)")

    rows = summarize(a_fast, smoothed, grid, cfg, stability or None)

    payload = {
        "scenario": args.scenario,
        "task": args.task,
        "compressor": args.compressor,
        "meta": meta,
        "config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in cfg.__dict__.items()},
        "grid": grid.tolist(),
        "a_true": a_true.tolist(),
        "a_fast": a_fast.tolist(),
        "fast_subset_offset": {"mean": float(offset.mean()),
                               "sd": float(offset.std()),
                               "max_abs": float(np.abs(offset).max())},
        "a_smooth": {f"{s:g}": smoothed[s]["value"].tolist() for s in cfg.sigmas},
        "rows": rows,
    }
    path = out_dir / "stein_smoothing_results.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n{'sigma':>7} {'RMSE':>8} {'RMSE*':>8} {'max|b|':>8} {'regret':>8} "
          f"{'rel-sd':>8} {'cos':>6} {'chg':>6}")
    for key, r in rows.items():
        print(f"{key:>7} {r['rmse']:>8.4f} {r['rmse_debiased']:>8.4f} "
              f"{r['max_abs_bias']:>8.4f} {r['regret_mean']:>8.4f} "
              f"{r.get('rel_sd', float('nan')):>8.3f} "
              f"{r.get('cos_seed', float('nan')):>6.3f} "
              f"{r['frac_decision_changed']:>6.2f}")
    print("  RMSE* = Monte-Carlo-variance-debiased")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
