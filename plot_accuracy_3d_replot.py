"""
plot_accuracy_3d.py — 3D accuracy surface plots for paper
==========================================================

Samples a 2D grid of (eta_1, eta_2) with eta_0 fixed at 1.0 (no compression
on the first link), then produces four publication-quality 3D surface plots:

  1. Raw A_true     — actual measured accuracy at each grid point
  2. KNN smoothed   — KNN regressor (k neighbours, inverse-distance weighted)
  3. Stein view     — what the Stein oracle "sees" (Gaussian-smoothed A)
  4. Combined       — all three in one figure (3 subplots side by side)

The KNN model and raw data are saved so they can be reused in the
concavity analysis and for LLM plots (which are expensive to resample).

Usage
-----
  # ResNet (fast)
  python plot_accuracy_3d.py --backend resnet \\
      --compressor topk \\
      --n_grid 20 --n_repeat 3 \\
      --knn_k 8 --stein_sigma 0.05 --stein_n 10 \\
      --out_dir outputs/accuracy_3d/resnet_topk

  # Load saved data and replot (no new evaluations)
  python plot_accuracy_3d.py --load_data outputs/accuracy_3d/resnet_topk/grid_data.npz \\
      --knn_k 12 --out_dir outputs/accuracy_3d/resnet_topk_k12

  # Replot only — read saved data, skip all computation, no titles, save as PDF
  python plot_accuracy_3d.py \\
      --replot outputs/accuracy_3d/resnet_topk/grid_data.npz \\
      --out_dir outputs/accuracy_3d/resnet_topk_pdf
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from knn_accuracy_model import KNNAccuracyModel


# ─────────────────────────────────────────────────────────────────────────────
# Backend loading  (mirrors analyze_concavity.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_accuracy_callable(args):
    """Return (accuracy_callable_true, num_links)."""
    if args.backend == "resnet":
        from run_experiment import load_backend, ExperimentSetting
        s = ExperimentSetting(
            backend              = "resnet",
            compressor           = args.compressor,
            compressor_backend   = args.compressor_backend,
            eta_min              = 0.0,
            R_k                  = 10.0,
            c_min                = 60000,
            c_max                = 300000,
            c_dist              = "uniform",
            T                    = 1,
            seed                 = 0,
            no_csi_mu            = 3.0,
            no_csi_epsilon       = 0.1,
            llm_model_name       = "",
            llm_activation_strategy = "")
        api, _ = load_backend(s)
        return api.accuracy_callable_true, api.num_links

    elif args.backend == "llm":
        from src.core.llm_task_callables import build_llm_backend_from_env
        api = build_llm_backend_from_env(eta_min=0.0)
        return api.accuracy_callable_true, 2

    else:
        raise ValueError(f"Unknown backend: {args.backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Grid sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_grid(accuracy_fn, num_links, n_grid, n_repeat,
                fixed_links, eta_lo=0.0):
    """
    Sample accuracy on a 2D grid over the two variable links.

    Parameters
    ----------
    accuracy_fn  : callable (eta_vec) -> float
    num_links    : total number of links (e.g. 3 for ResNet)
    n_grid       : points per axis
    n_repeat     : repeated evaluations per point (averaged)
    fixed_links  : dict {link_idx: eta_value} — links held constant
    eta_lo       : lower bound of sweep (default 0.0)

    Returns
    -------
    eta1_grid  : (n_grid,) axis values for variable link 1
    eta2_grid  : (n_grid,) axis values for variable link 2
    acc_matrix : (n_grid, n_grid) accuracy values
    eta_all    : (n_grid^2, num_links) all eta vectors evaluated
    acc_all    : (n_grid^2,) all accuracy values
    var_links  : [link_i, link_j] indices of the two variable links
    """
    fixed_set = set(fixed_links.keys())
    var_links  = [i for i in range(num_links) if i not in fixed_set]

    if len(var_links) < 2:
        raise ValueError(
            f"Need at least 2 variable links. Got {len(var_links)} "
            f"(num_links={num_links}, fixed={list(fixed_links.keys())})"
        )

    # Use first two variable links for the 2D plot
    li, lj = var_links[0], var_links[1]
    eta1_grid = np.linspace(eta_lo, 1.0, n_grid)
    eta2_grid = np.linspace(eta_lo, 1.0, n_grid)

    acc_matrix = np.zeros((n_grid, n_grid))
    eta_all    = []
    acc_all    = []

    total = n_grid * n_grid
    done  = 0
    print(f"\n[sample_grid] Sampling {total} points "
          f"(links {li},{lj} variable, others fixed) ...", flush=True)

    for i, e1 in enumerate(eta1_grid):
        for j, e2 in enumerate(eta2_grid):
            # Build full eta vector
            eta_vec = np.ones(num_links)
            for k_link, v in fixed_links.items():
                eta_vec[k_link] = v
            eta_vec[li] = e1
            eta_vec[lj] = e2

            vals = [accuracy_fn(eta_vec) for _ in range(n_repeat)]
            acc  = float(np.mean(vals))
            acc_matrix[i, j] = acc
            eta_all.append(eta_vec.copy())
            acc_all.append(acc)

            done += 1
            if done % max(1, total // 20) == 0:
                print(f"  {done}/{total} ({100*done//total}%)", flush=True)

    return (eta1_grid, eta2_grid, acc_matrix,
            np.array(eta_all), np.array(acc_all),
            [li, lj])


# ─────────────────────────────────────────────────────────────────────────────
# Stein-smoothed surface
# ─────────────────────────────────────────────────────────────────────────────

def compute_stein_surface(accuracy_fn, num_links, eta1_grid, eta2_grid,
                          var_links, fixed_links, sigma, n_stein,
                          n_repeat=1):
    """
    Compute the Stein-smoothed accuracy surface:
      Ã(eta) = E_z[A(eta + sigma*z)]  for z ~ N(0, I)

    This is what the gradient oracle actually optimises — useful to show
    how the noisy landscape is smoothed.

    Parameters
    ----------
    sigma   : Stein smoothing bandwidth
    n_stein : number of Monte Carlo perturbation samples

    Returns
    -------
    stein_matrix : (n_grid, n_grid) smoothed accuracy values
    """
    li, lj = var_links
    n_grid = len(eta1_grid)
    stein_matrix = np.zeros((n_grid, n_grid))
    rng = np.random.default_rng(42)

    total = n_grid * n_grid
    done  = 0
    print(f"\n[compute_stein_surface] Stein smoothing "
          f"(sigma={sigma}, N={n_stein}) ...", flush=True)

    for i, e1 in enumerate(eta1_grid):
        for j, e2 in enumerate(eta2_grid):
            eta_vec = np.ones(num_links)
            for k_link, v in fixed_links.items():
                eta_vec[k_link] = v
            eta_vec[li] = e1
            eta_vec[lj] = e2

            # Monte Carlo estimate of E[A(eta + sigma*z)]
            acc_perturbed = []
            for _ in range(n_stein):
                z          = rng.standard_normal(num_links)
                eta_pert   = np.clip(eta_vec + sigma * z, 0.0, 1.0)
                for k_link, v in fixed_links.items():
                    eta_pert[k_link] = v  # keep fixed links fixed
                acc_perturbed.append(
                    float(np.mean([accuracy_fn(eta_pert)
                                   for _ in range(n_repeat)]))
                )
            stein_matrix[i, j] = float(np.mean(acc_perturbed))

            done += 1
            if done % max(1, total // 20) == 0:
                print(f"  {done}/{total} ({100*done//total}%)", flush=True)

    return stein_matrix


# ─────────────────────────────────────────────────────────────────────────────
# KNN surface
# ─────────────────────────────────────────────────────────────────────────────

def compute_knn_surface(knn_model, eta1_grid, eta2_grid,
                        var_links, fixed_links, num_links):
    """
    Evaluate KNN model on the 2D grid.

    Returns
    -------
    knn_matrix : (n_grid, n_grid)
    """
    li, lj     = var_links
    n_grid     = len(eta1_grid)
    knn_matrix = np.zeros((n_grid, n_grid))

    # Build all query points at once for efficiency
    queries = []
    for e1 in eta1_grid:
        for e2 in eta2_grid:
            eta_vec = np.ones(num_links)
            for k_link, v in fixed_links.items():
                eta_vec[k_link] = v
            eta_vec[li] = e1
            eta_vec[lj] = e2
            queries.append(eta_vec)

    preds = knn_model.predict(np.array(queries))
    knn_matrix = preds.reshape(n_grid, n_grid)
    return knn_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Shared plot style — publication quality
'''
PLOT_STYLE = {
    "font.family":      "serif",
    "font.size":        16,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
}
'''
PLOT_STYLE = {
    "font.family":      "serif",
    "font.size":        24,
    "axes.titlesize":   30,
    "axes.labelsize":   22,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "figure.dpi":       150,
}

def _make_3d_ax(fig, pos, title, cmap="viridis", no_title=False):
    ax = fig.add_subplot(pos, projection="3d")
    if not no_title:
        ax.set_title(title, pad=10)
    return ax


def _plot_surface(ax, E1, E2, Z, cmap, alpha=0.85, label_i=1, label_j=2):
    surf = ax.plot_surface(E1, E2, Z, cmap=cmap, alpha=alpha,
                           linewidth=0, antialiased=True)
    ax.set_xlabel(f"$\\eta_{label_i}$", labelpad=6)
    ax.set_ylabel(f"$\\eta_{label_j}$", labelpad=6)
    ax.set_zlabel("Accuracy", labelpad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return surf
    

def plot_surfaces(eta1_grid, eta2_grid, acc_matrix, knn_matrix, stein_matrix,
                  var_links, out_dir, backend, compressor,
                  eta_min_rec=None, no_title=False, save_pdf=False):
    """
    Save four figures:
      individual_raw.png / .pdf
      individual_knn.png / .pdf
      individual_stein.png / .pdf
      combined.png / .pdf

    Parameters
    ----------
    no_title  : if True, all figure and axes titles are suppressed.
    save_pdf  : if True, figures are saved as PDF instead of PNG.
    """
    ext = "pdf" if save_pdf else "png"
    with plt.rc_context(PLOT_STYLE):
        E1, E2 = np.meshgrid(eta1_grid, eta2_grid, indexing="ij")
        li, lj = var_links
        z_lo   = min(acc_matrix.min(), knn_matrix.min(), stein_matrix.min())
        z_hi   = max(acc_matrix.max(), knn_matrix.max(), stein_matrix.max())
        z_pad  = max(0.02, (z_hi - z_lo) * 0.05)
        z_lim  = (z_lo - z_pad, z_hi + z_pad)

        tag = f"{backend}/{compressor}"

        surfaces = [
            ("Raw A_true",    acc_matrix,   "plasma",  f"individual_raw.{ext}"),
            ("KNN smoothed",  knn_matrix,   "viridis", f"individual_knn.{ext}"),
            ("Stein view",    stein_matrix, "cividis", f"individual_stein.{ext}"),
        ]

        # ── Individual plots ─────────────────────────────────────────────────
        for title, Z, cmap, fname in surfaces:
            fig = plt.figure(figsize=(7, 6))
            ax  = _make_3d_ax(fig, 111, f"{title}\n{tag}", no_title=no_title)
            surf = _plot_surface(ax, E1, E2, Z, cmap,
                                 label_i=li, label_j=lj)
            ax.set_zlim(*z_lim)
            if eta_min_rec is not None:
                # Draw a vertical plane at eta_min for each variable link
                for link_local, link_global in enumerate([li, lj]):
                    eta_thresh = eta_min_rec[link_global]
                    if 0 < eta_thresh < 1:
                        # Draw a translucent plane
                        if link_local == 0:
                            yy = np.linspace(0, 1, 2)
                            zz = np.linspace(z_lim[0], z_lim[1], 2)
                            YY, ZZ = np.meshgrid(yy, zz)
                            XX = np.full_like(YY, eta_thresh)
                        else:
                            xx = np.linspace(0, 1, 2)
                            zz = np.linspace(z_lim[0], z_lim[1], 2)
                            XX, ZZ = np.meshgrid(xx, zz)
                            YY = np.full_like(XX, eta_thresh)
                        ax.plot_surface(XX, YY, ZZ, alpha=0.15,
                                        color='red', linewidth=0)
            fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1,
                         label="")
            plt.tight_layout()
            path = os.path.join(out_dir, fname)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}", flush=True)

        # ── Combined figure ───────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 6))
        if not no_title:
            fig.suptitle(f"Accuracy surface — {tag}", fontsize=30, y=1.01)

        for col, (title, Z, cmap, _) in enumerate(surfaces):
            ax = _make_3d_ax(fig, (1, 3, col + 1), title, cmap,
                             no_title=no_title)
            surf = _plot_surface(ax, E1, E2, Z, cmap,
                                 label_i=li, label_j=lj)
            ax.set_zlim(*z_lim)

        plt.tight_layout()
        path = os.path.join(out_dir, f"combined.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_grid_data(path, eta1_grid, eta2_grid, acc_matrix,
                   eta_all, acc_all, stein_matrix,
                   var_links, fixed_links, meta):
    """Save all sampled data to .npz for later reuse."""
    np.savez(
        path,
        eta1_grid    = eta1_grid,
        eta2_grid    = eta2_grid,
        acc_matrix   = acc_matrix,
        eta_all      = eta_all,
        acc_all      = acc_all,
        stein_matrix = stein_matrix,
        var_links    = np.array(var_links),
        fixed_links_keys   = np.array(list(fixed_links.keys())),
        fixed_links_values = np.array(list(fixed_links.values())),
        **{f"meta_{k}": np.array([v]) for k, v in meta.items()
           if isinstance(v, (int, float, str))}
    )
    print(f"[save_grid_data] Saved: {path}", flush=True)


def load_grid_data(path):
    """Load saved grid data. Returns dict with all arrays."""
    data = np.load(path, allow_pickle=True)
    fixed_keys   = data['fixed_links_keys'].tolist()
    fixed_values = data['fixed_links_values'].tolist()
    fixed_links  = dict(zip([int(k) for k in fixed_keys], fixed_values))
    return {
        'eta1_grid':    data['eta1_grid'],
        'eta2_grid':    data['eta2_grid'],
        'acc_matrix':   data['acc_matrix'],
        'eta_all':      data['eta_all'],
        'acc_all':      data['acc_all'],
        'stein_matrix': data['stein_matrix'],
        'var_links':    data['var_links'].tolist(),
        'fixed_links':  fixed_links,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="3D accuracy surface plots for paper"
    )

    # Backend
    p.add_argument("--backend",            default="resnet",
                   choices=["resnet", "llm"])
    p.add_argument("--compressor",         default="topk")
    p.add_argument("--compressor_backend", default="gpu")

    # Sampling
    p.add_argument("--n_grid",    type=int,   default=20,
                   help="Grid points per variable axis (total = n_grid^2).")
    p.add_argument("--n_repeat",  type=int,   default=3,
                   help="Repeated accuracy evaluations per point (averaged).")
    p.add_argument("--eta_lo",    type=float, default=0.0,
                   help="Lower bound of eta sweep (default 0).")
    p.add_argument("--fixed_link_0_eta", type=float, default=1.0,
                   help="Fixed eta for link 0 (default 1.0 = uncompressed). "
                        "Set to sweep links 1 and 2 only.")

    # KNN
    p.add_argument("--knn_k",         type=int,   default=8,
                   help="Number of neighbours for KNN regressor.")
    p.add_argument("--knn_weighting", default="inverse_distance",
                   choices=["inverse_distance", "uniform"])

    # Stein
    p.add_argument("--stein_sigma",  type=float, default=0.05,
                   help="Gaussian smoothing bandwidth for Stein surface.")
    p.add_argument("--stein_n",      type=int,   default=10,
                   help="Monte Carlo samples for Stein smoothing per point.")
    p.add_argument("--stein_repeat", type=int,   default=1,
                   help="Accuracy evaluations per Stein perturbation.")

    # Load saved data instead of resampling
    p.add_argument("--load_data",  default=None,
                   help="Path to saved grid_data.npz — skip sampling, "
                        "replot from saved data.")

    # Replot-only mode: skip ALL computation, just read saved data and replot
    p.add_argument("--replot", default=None, metavar="NPZ_PATH",
                   help="Path to saved grid_data.npz.  Skips ALL computation "
                        "(sampling, Stein, KNN fit).  Plots are saved without "
                        "titles and as PDF files.  --out_dir is still required.")

    # Concavity overlay
    p.add_argument("--eta_min_json", default=None,
                   help="Path to concavity_results.json to overlay "
                        "eta_min planes on plots.")

    # Output
    p.add_argument("--out_dir",    default="outputs/accuracy_3d")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Replot-only mode ─────────────────────────────────────────────────────
    if args.replot:
        print(f"\n[--replot] Loading saved data from {args.replot} ...",
              flush=True)
        d = load_grid_data(args.replot)
        eta1_grid    = d['eta1_grid']
        eta2_grid    = d['eta2_grid']
        acc_matrix   = d['acc_matrix']
        eta_all      = d['eta_all']
        acc_all      = d['acc_all']
        stein_matrix = d['stein_matrix']
        var_links    = d['var_links']
        fixed_links  = d['fixed_links']
        num_links    = eta_all.shape[1]

        print(f"\nFitting KNN model (k={args.knn_k}) ...", flush=True)
        knn = KNNAccuracyModel(
            k=args.knn_k, weighting=args.knn_weighting
        ).fit(eta_all, acc_all)

        knn_matrix = compute_knn_surface(
            knn, eta1_grid, eta2_grid, var_links, fixed_links, num_links
        )

        eta_min_rec = None
        if args.eta_min_json and os.path.exists(args.eta_min_json):
            import json
            with open(args.eta_min_json) as f:
                conc = json.load(f)
            eta_min_rec = conc.get("eta_min_recommended", None)
            if eta_min_rec:
                eta_min_rec = np.array(eta_min_rec)
                print(f"\n  eta_min overlay: {np.round(eta_min_rec, 3)}",
                      flush=True)

        print("\nGenerating plots (no titles, PDF) ...", flush=True)
        plot_surfaces(
            eta1_grid, eta2_grid,
            acc_matrix, knn_matrix, stein_matrix,
            var_links, args.out_dir,
            args.backend, args.compressor,
            eta_min_rec=eta_min_rec,
            no_title=True,
            save_pdf=True,
        )
        print(f"\nAll outputs in: {args.out_dir}", flush=True)
        return

    # ── Load or sample data ───────────────────────────────────────────────────
    if args.load_data:
        print(f"\nLoading saved data from {args.load_data} ...", flush=True)
        d = load_grid_data(args.load_data)
        eta1_grid    = d['eta1_grid']
        eta2_grid    = d['eta2_grid']
        acc_matrix   = d['acc_matrix']
        eta_all      = d['eta_all']
        acc_all      = d['acc_all']
        stein_matrix = d['stein_matrix']
        var_links    = d['var_links']
        fixed_links  = d['fixed_links']
        num_links    = eta_all.shape[1]

    else:
        print(f"\nLoading {args.backend} backend ...", flush=True)
        accuracy_fn, num_links = load_accuracy_callable(args)
        print(f"  num_links = {num_links}", flush=True)

        # Fixed links: link 0 at eta=1.0 (uncompressed)
        # Variable links: 1 and 2 (or whatever remains)
        fixed_links = {0: args.fixed_link_0_eta}
        if num_links == 2:
            # 2-link model — can't fix link 0 and have 2 variable
            # so just fix nothing and vary both
            fixed_links = {}
            print("  Note: 2-link model — varying both links", flush=True)

        # Sample raw accuracy grid
        eta1_grid, eta2_grid, acc_matrix, eta_all, acc_all, var_links = \
            sample_grid(
                accuracy_fn, num_links,
                n_grid=args.n_grid, n_repeat=args.n_repeat,
                fixed_links=fixed_links,
                eta_lo=args.eta_lo,
            )

        # Stein-smoothed surface
        stein_matrix = compute_stein_surface(
            accuracy_fn, num_links,
            eta1_grid, eta2_grid,
            var_links, fixed_links,
            sigma=args.stein_sigma, n_stein=args.stein_n,
            n_repeat=args.stein_repeat,
        )

        # Save data for reuse
        data_path = os.path.join(args.out_dir, "grid_data.npz")
        save_grid_data(
            data_path,
            eta1_grid, eta2_grid, acc_matrix,
            eta_all, acc_all, stein_matrix,
            var_links, fixed_links,
            meta={
                "backend":    args.backend,
                "compressor": args.compressor,
                "n_grid":     args.n_grid,
                "n_repeat":   args.n_repeat,
                "stein_sigma": args.stein_sigma,
                "stein_n":    args.stein_n,
            }
        )

    # ── Fit KNN model ─────────────────────────────────────────────────────────
    print(f"\nFitting KNN model (k={args.knn_k}) ...", flush=True)
    knn = KNNAccuracyModel(
        k=args.knn_k, weighting=args.knn_weighting
    ).fit(eta_all, acc_all)
    print(knn.summary(), flush=True)

    # Leave-one-out RMSE for k selection guidance
    if len(eta_all) <= 200:   # only if small enough to be fast
        loo_rmse = knn.leave_one_out_rmse()
        print(f"  LOO-RMSE (k={args.knn_k}) = {loo_rmse:.6f}", flush=True)

    knn_path = os.path.join(args.out_dir,
                            f"knn_{args.backend}_{args.compressor}_k{args.knn_k}.npz")
    knn.save(knn_path)

    # Evaluate KNN on the same grid
    knn_matrix = compute_knn_surface(
        knn, eta1_grid, eta2_grid, var_links, fixed_links, num_links
    )

    # ── Load optional eta_min overlay ─────────────────────────────────────────
    eta_min_rec = None
    if args.eta_min_json and os.path.exists(args.eta_min_json):
        import json
        with open(args.eta_min_json) as f:
            conc = json.load(f)
        eta_min_rec = conc.get("eta_min_recommended", None)
        if eta_min_rec:
            eta_min_rec = np.array(eta_min_rec)
            print(f"\n  eta_min overlay: {np.round(eta_min_rec, 3)}", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...", flush=True)
    plot_surfaces(
        eta1_grid, eta2_grid,
        acc_matrix, knn_matrix, stein_matrix,
        var_links, args.out_dir,
        args.backend, args.compressor,
        eta_min_rec=eta_min_rec,
        no_title=False,
        save_pdf=False,
    )

    print(f"\nAll outputs in: {args.out_dir}", flush=True)
    print("  grid_data.npz         — raw samples (reuse with --load_data / --replot)", flush=True)
    print(f"  knn_*.npz             — KNN model (reuse with KNNAccuracyModel.load)", flush=True)
    print("  individual_raw.png    — raw A_true surface", flush=True)
    print("  individual_knn.png    — KNN smoothed surface", flush=True)
    print("  individual_stein.png  — Stein smoothed surface", flush=True)
    print("  combined.png          — all three side by side", flush=True)


if __name__ == "__main__":
    main()
