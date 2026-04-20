"""
analyze_concavity.py — Automatic eta_min detection via concavity analysis
=========================================================================

Finds the per-link true eta_min^true: the smallest eta value above which
the accuracy function A(eta) is monotonically increasing AND concave.
Below this threshold, activations collapse to noise and A(eta) is flat
or non-monotone — the predictor should NOT be fit in this region.

Three complementary tests are run:

  1. Per-axis monotonicity scan (1D sweeps along each link axis)
     Finds the smallest eta_i where A consistently increases with eta_i.
     Fast and gives a per-link estimate directly.

  2. Concavity scan via second differences (1D sweeps)
     Scans downward from eta=1 and finds where the second difference
     A(eta-delta) + A(eta+delta) - 2*A(eta) first becomes significantly
     positive (violating concavity). This is the theoretically grounded test.

  3. Directional profiles (your idea)
     Samples random rays from [0,...,0] to [1,...,1] and draws A along
     each ray. Useful for visualisation and for detecting that the
     collapse is correlated across links (not independent).

The recommended eta_min is the maximum of the per-link estimates from
tests 1 and 2, with a small safety margin.

Usage
-----
  # ResNet (fast — uses RESNET_FAST_BATCHES=1)
  python analyze_concavity.py --backend resnet \\
      --n_grid 20 --n_repeat 3 \\
      --n_directions 10 \\
      --out_dir outputs/concavity_resnet

  # LLM (slow — use small n_grid and n_repeat=1)
  python analyze_concavity.py --backend llm \\
      --n_grid 12 --n_repeat 1 \\
      --n_directions 5 \\
      --out_dir outputs/concavity_llm

  # Replot only — read saved JSON, skip all computation, no titles, save as PDF
  python analyze_concavity.py \\
      --replot outputs/concavity_resnet/concavity_results.json \\
      --out_dir outputs/concavity_resnet_pdf

Output
------
  <out_dir>/
    concavity_results.json   — detected eta_min per link + all metrics
    axis_profiles.png        — 1D accuracy slices per link
    directional_profiles.png — accuracy along random rays
    second_differences.png   — concavity test plots

  In --replot mode the same three plots are saved as .pdf without titles.
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


# ─────────────────────────────────────────────────────────────────────────────
# Backend loading
# ─────────────────────────────────────────────────────────────────────────────

def load_accuracy_callable(args):
    """Load accuracy_callable_true from the appropriate backend."""
    if args.backend == "resnet":
        from src.core.llm_plug_in import is_resnet_backend
        from run_experiment import load_backend, ExperimentSetting
        s = ExperimentSetting(
            backend             = "resnet",
            compressor          = args.compressor,
            compressor_backend  = args.compressor_backend,
            eta_min             = 0.0,  # use 0 — we are finding eta_min
            R_k                 = 10.0,
            c_min               = 60000,
            c_max               = 300000,
            c_dist              = "uniform",
            T                   = 1,
            seed                = 0,
            no_csi_mu           = 3.0,
            no_csi_epsilon      = 0.1,
            llm_model_name      = "",
            llm_activation_strategy = "")
    
        api, _ = load_backend(s)
        num_links = api.num_links
        return api.accuracy_callable_true, num_links

    elif args.backend == "llm":
        from src.core.llm_task_callables import build_llm_backend_from_env
        api = build_llm_backend_from_env(eta_min=0.0)
        # LLM has 2 links (3-node split)
        return api.accuracy_callable_true, 2

    else:
        raise ValueError(f"Unknown backend: {args.backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Per-axis monotonicity scan
# ─────────────────────────────────────────────────────────────────────────────

def run_axis_monotonicity(accuracy_fn, num_links, n_grid, n_repeat,
                          noise_threshold, mid_eta=0.5, eta_grid=None):
    """
    For each link i, sweep eta_i from 0 to 1 while holding all other
    links fixed at mid_eta. Estimate the smallest eta_i where A is
    consistently increasing (slope > noise_threshold).

    Returns
    -------
    eta_grid     : (N,) array of eta values
    profiles     : (num_links, N) array of A values
    eta_min_mono : (num_links,) per-link eta_min from monotonicity test
    """
    if eta_grid is None:
        eta_grid = np.linspace(0.0, 1.0, n_grid)
    n_pts = len(eta_grid)
    profiles = np.zeros((num_links, n_pts))

    print("\n[Test 1] Per-axis monotonicity scan", flush=True)
    for link in range(num_links):
        print(f"  Link {link}:", end="", flush=True)
        for j, eta_val in enumerate(eta_grid):
            eta_vec = np.full(num_links, mid_eta)
            eta_vec[link] = eta_val
            vals = [accuracy_fn(eta_vec) for _ in range(n_repeat)]
            profiles[link, j] = float(np.mean(vals))
            print(".", end="", flush=True)
        print(flush=True)

    # Detect eta_min: scan downward from eta=1, stop at first point where
    # slope drops below noise_threshold
    eta_min_mono = np.zeros(num_links)
    for link in range(num_links):
        p = profiles[link]
        # Smooth slightly to reduce noise
        p_smooth = np.convolve(p, np.ones(3)/3, mode="same")
        slopes = np.diff(p_smooth) / (eta_grid[1] - eta_grid[0])
        # Find rightmost index where slope >= noise_threshold
        increasing = np.where(slopes >= noise_threshold)[0]
        if len(increasing) == 0:
            eta_min_mono[link] = eta_grid[-1]  # degenerate: never increasing
        else:
            first_good = increasing[0]
            eta_min_mono[link] = float(eta_grid[max(0, first_good - 1)])

    return eta_grid, profiles, eta_min_mono


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Concavity scan via second differences
# ─────────────────────────────────────────────────────────────────────────────

def run_concavity_scan(accuracy_fn, num_links, n_grid, n_repeat,
                       concavity_threshold, mid_eta=0.5, eta_grid=None):
    """
    For each link i, sweep eta_i and compute second differences
    D2[j] = A[j-1] + A[j+1] - 2*A[j].

    Concavity requires D2 <= 0. Scan downward from eta=1 and find the
    first point where D2 > concavity_threshold (concavity violated).

    Returns
    -------
    eta_grid     : (N,) array
    profiles     : (num_links, N) same as test 1 if reused
    d2           : (num_links, N-2) second differences
    eta_min_conc : (num_links,) per-link eta_min from concavity test
    """
    if eta_grid is None:
        eta_grid = np.linspace(0.0, 1.0, n_grid)
    n_pts = len(eta_grid)
    profiles = np.zeros((num_links, n_pts))

    print("\n[Test 2] Concavity scan via second differences", flush=True)
    for link in range(num_links):
        print(f"  Link {link}:", end="", flush=True)
        for j, eta_val in enumerate(eta_grid):
            eta_vec = np.full(num_links, mid_eta)
            eta_vec[link] = eta_val
            vals = [accuracy_fn(eta_vec) for _ in range(n_repeat)]
            profiles[link, j] = float(np.mean(vals))
            print(".", end="", flush=True)
        print(flush=True)

    # Second differences: D2[j] = A[j-1] + A[j+1] - 2*A[j]
    d2 = np.zeros((num_links, n_pts - 2))
    eta_min_conc = np.zeros(num_links)

    for link in range(num_links):
        p = profiles[link]
        for j in range(1, n_pts - 1):
            d2[link, j-1] = p[j-1] + p[j+1] - 2*p[j]

        # Scan downward from the top: find the lowest eta where all points
        # above are concave (D2 <= concavity_threshold)
        # i.e. the first index (from bottom) where D2 > threshold
        inner_etas = eta_grid[1:-1]
        violated = np.where(d2[link] > concavity_threshold)[0]
        if len(violated) == 0:
            eta_min_conc[link] = 0.0  # everywhere concave
        else:
            # The highest violation defines the lower bound
            last_violation = violated[-1]
            eta_min_conc[link] = float(inner_etas[last_violation])

    return eta_grid, profiles, d2, eta_min_conc


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Directional profiles (your idea)
# ─────────────────────────────────────────────────────────────────────────────

def run_directional_profiles(accuracy_fn, num_links, n_grid, n_repeat,
                              n_directions, rng_seed=42):
    """
    Sample n_directions rays from eta_start to [1,...,1].
    eta_start is random — one direction goes from [0,...,0] (always included),
    others have random starting points.

    Returns
    -------
    s_grid       : (n_grid,) parameterisation values in [0,1]
    ray_starts   : (n_directions, num_links)
    ray_profiles : (n_directions, n_grid)
    """
    rng = np.random.default_rng(rng_seed)
    s_grid = np.linspace(0.0, 1.0, n_grid)

    # First direction: from [0,...,0] to [1,...,1]
    ray_starts = [np.zeros(num_links)]
    # Remaining: random starting points
    for _ in range(n_directions - 1):
        start = rng.uniform(0.0, 0.5, size=num_links)
        ray_starts.append(start)
    ray_starts = np.array(ray_starts)

    ray_profiles = np.zeros((n_directions, n_grid))

    print(f"\n[Test 3] Directional profiles ({n_directions} rays)", flush=True)
    for d, start in enumerate(ray_starts):
        print(f"  Ray {d} start={np.round(start, 2)}:", end="", flush=True)
        for j, s in enumerate(s_grid):
            eta_vec = start + s * (1.0 - start)   # moves from start to [1,...,1]
            eta_vec = np.clip(eta_vec, 0.0, 1.0)
            vals = [accuracy_fn(eta_vec) for _ in range(n_repeat)]
            ray_profiles[d, j] = float(np.mean(vals))
            print(".", end="", flush=True)
        print(flush=True)

    return s_grid, ray_starts, ray_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_axis_profiles(eta_grid, profiles, eta_min_mono, eta_min_conc,
                       eta_min_final, out_dir, backend, compressor,
                       no_title=False, save_pdf=False):
    ext = "pdf" if save_pdf else "png"
    num_links = profiles.shape[0]
    fig, axes = plt.subplots(1, num_links, figsize=(6*num_links, 5),
                             squeeze=False)
    for link in range(num_links):
        ax = axes[0, link]
        ax.plot(eta_grid, profiles[link], "b-o", ms=4, label="A_true")
        ax.axvline(eta_min_mono[link], color="orange", ls="--",
                   label=f"η_min mono={eta_min_mono[link]:.3f}")
        ax.axvline(eta_min_conc[link], color="red", ls="--",
                   label=f"η_min conc={eta_min_conc[link]:.3f}")
        ax.axvline(eta_min_final[link], color="green", ls="-", lw=2,
                   label=f"η_min final={eta_min_final[link]:.3f}")
        ax.set_xlabel("η (link axis)")
        ax.set_ylabel("A_true")
        if not no_title:
            ax.set_title(f"Link {link} — axis sweep\n"
                         f"{backend}/{compressor}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"axis_profiles.{ext}")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_second_differences(eta_grid, d2, eta_min_conc,
                            out_dir, backend, compressor,
                            no_title=False, save_pdf=False):
    ext = "pdf" if save_pdf else "png"
    num_links = d2.shape[0]
    inner_etas = eta_grid[1:-1]
    fig, axes = plt.subplots(1, num_links, figsize=(6*num_links, 4),
                             squeeze=False)
    for link in range(num_links):
        ax = axes[0, link]
        ax.plot(inner_etas, d2[link], "b-o", ms=4)
        ax.axhline(0, color="k", ls="-", lw=0.8)
        ax.axvline(eta_min_conc[link], color="red", ls="--",
                   label=f"η_min={eta_min_conc[link]:.3f}")
        ax.fill_between(inner_etas, d2[link], 0,
                        where=d2[link] > 0, alpha=0.3, color="red",
                        label="concavity violation")
        ax.set_xlabel("η")
        ax.set_ylabel("D²A (>0 = concavity violated)")
        if not no_title:
            ax.set_title(f"Link {link} — second differences\n"
                         f"{backend}/{compressor}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"second_differences.{ext}")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_directional_profiles(s_grid, ray_starts, ray_profiles,
                               eta_min_final, out_dir, backend, compressor,
                               no_title=False, save_pdf=False):
    ext = "pdf" if save_pdf else "png"
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for d in range(len(ray_starts)):
        label = (f"ray {d}: start={np.round(ray_starts[d], 2)}")
        ax.plot(s_grid, ray_profiles[d], "-o", ms=3,
                color=cmap(d % 10), label=label)

    # Mark where each ray crosses eta_min_final on all links
    # s_cross = max over links of (eta_min_final[i] - start[i]) / (1 - start[i])
    for d in range(len(ray_starts)):
        start = ray_starts[d]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_vals = np.where(
                1.0 - start > 1e-6,
                (eta_min_final - start) / (1.0 - start),
                0.0
            )
        s_cross = float(np.max(np.clip(s_vals, 0, 1)))
        ax.axvline(s_cross, color=cmap(d % 10), ls=":", alpha=0.5)

    ax.set_xlabel("s (ray parameter: 0=start, 1=[1,...,1])")
    ax.set_ylabel("A_true")
    if not no_title:
        ax.set_title(f"Directional profiles — {backend}/{compressor}\n"
                     f"(dotted vertical = estimated η_min crossing)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"directional_profiles.{ext}")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_eta_grid(n_grid, compressor, quantization_aware=False):
    """
    Build the eta grid for 1D sweeps.

    For standard compressors: uniform grid over [0, 1].

    For quantization with quantization_aware=True: non-uniform grid that
    densely samples around the known precision transition thresholds
    (INT4->INT8 at eta~0.25-0.375, INT8->FP16 at eta~0.5-0.625).
    Missing these transitions at coarse uniform spacing is the main reason
    quantization sweeps appear as flat straight lines.
    """
    if quantization_aware and "quant" in compressor.lower():
        # Dense clusters around transition points + uniform background
        transitions = [0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625]
        delta = 0.04
        cluster_pts = []
        for t in transitions:
            cluster_pts.extend(np.linspace(max(0, t - delta),
                                           min(1, t + delta), 6).tolist())
        uniform_pts = np.linspace(0.0, 1.0, n_grid).tolist()
        combined = sorted(set([round(x, 4) for x in uniform_pts + cluster_pts]))
        grid = np.array(combined)
        print(f"[build_eta_grid] quantization-aware grid: {len(grid)} points "
              f"(uniform {n_grid} + dense clusters around transitions)",
              flush=True)
        return grid
    else:
        return np.linspace(0.0, 1.0, n_grid)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Concavity analysis for eta_min detection")
    p.add_argument("--backend",            default="resnet",
                   choices=["resnet", "llm"])
    p.add_argument("--compressor",         default="topk",
                   help="ResNet: topk|quantization|llmint8. LLM: set via env.")
    p.add_argument("--compressor_backend", default="gpu")
    p.add_argument("--n_grid",             type=int, default=20,
                   help="Grid points per axis for 1D sweeps.")
    p.add_argument("--n_repeat",           type=int, default=3,
                   help="Repeated evaluations per point (for noise reduction).")
    p.add_argument("--n_directions",       type=int, default=10,
                   help="Number of random rays for directional profiles.")
    p.add_argument("--noise_threshold",    type=float, default=0.005,
                   help="Minimum slope to be considered 'increasing' in Test 1.")
    p.add_argument("--concavity_threshold",type=float, default=0.005,
                   help="Maximum D2 value still considered concave in Test 2."
                        " Small positive value allows for noise.")
    p.add_argument("--safety_margin",      type=float, default=0.05,
                   help="Added to detected eta_min as a safety buffer.")
    p.add_argument("--mid_eta",            type=float, default=0.5,
                   help="Value used for all links not being swept in 1D tests.")
    p.add_argument("--quantization_aware", action="store_true",
                   help="Use non-uniform grid that densely samples around "
                        "quantization precision transition points "
                        "(INT4/INT8/FP16). Recommended for --compressor quantization.")
    p.add_argument("--replot", default=None, metavar="JSON_PATH",
                   help="Path to a saved concavity_results.json.  Skips ALL "
                        "computation and re-generates the three plots without "
                        "titles, saved as PDF files.  --out_dir is still required.")
    p.add_argument("--out_dir",            default="outputs/concavity")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Replot-only mode ─────────────────────────────────────────────────────
    if args.replot:
        print(f"\n[--replot] Loading saved results from {args.replot} ...",
              flush=True)
        with open(args.replot) as f:
            results = json.load(f)

        backend    = results["backend"]
        compressor = results["compressor"]

        eta_grid_1   = np.array(results["axis_profiles"]["eta_grid"])
        profiles_1   = np.array(results["axis_profiles"]["profiles"])
        eta_min_mono = np.array(results["eta_min_monotonicity"])
        eta_min_conc = np.array(results["eta_min_concavity"])
        eta_min_final = np.array(results["eta_min_recommended"])

        # Reconstruct second differences from the saved axis profiles
        n_pts = len(eta_grid_1)
        num_links = profiles_1.shape[0]
        d2 = np.zeros((num_links, n_pts - 2))
        for link in range(num_links):
            p = profiles_1[link]
            for j in range(1, n_pts - 1):
                d2[link, j-1] = p[j-1] + p[j+1] - 2*p[j]

        s_grid       = np.array(results["directional_profiles"]["s_grid"])
        ray_starts   = np.array(results["directional_profiles"]["ray_starts"])
        ray_profiles = np.array(results["directional_profiles"]["ray_profiles"])

        print("\nGenerating plots (no titles, PDF) ...", flush=True)
        plot_axis_profiles(eta_grid_1, profiles_1, eta_min_mono, eta_min_conc,
                           eta_min_final, args.out_dir, backend, compressor,
                           no_title=True, save_pdf=True)
        plot_second_differences(eta_grid_1, d2, eta_min_conc,
                                args.out_dir, backend, compressor,
                                no_title=True, save_pdf=True)
        plot_directional_profiles(s_grid, ray_starts, ray_profiles,
                                  eta_min_final, args.out_dir, backend, compressor,
                                  no_title=True, save_pdf=True)
        print(f"\nAll outputs in: {args.out_dir}", flush=True)
        return

    print("="*60, flush=True)
    print("Concavity Analysis — eta_min detection", flush=True)
    print(f"  backend    = {args.backend}", flush=True)
    print(f"  compressor = {args.compressor}", flush=True)
    print(f"  n_grid     = {args.n_grid}  n_repeat={args.n_repeat}", flush=True)
    print(f"  n_dirs     = {args.n_directions}", flush=True)
    print(f"  quantization_aware = {args.quantization_aware}", flush=True)
    print("="*60, flush=True)

    # Load accuracy function
    print("\nLoading backend ...", flush=True)
    accuracy_fn, num_links = load_accuracy_callable(args)
    print(f"  num_links = {num_links}", flush=True)

    # Build eta grid — non-uniform for quantization if requested
    eta_grid_shared = build_eta_grid(
        args.n_grid, args.compressor,
        quantization_aware=args.quantization_aware,
    )

    # ── Test 1: Monotonicity ──────────────────────────────────────────────────
    eta_grid_1, profiles_1, eta_min_mono = run_axis_monotonicity(
        accuracy_fn, num_links,
        n_grid=args.n_grid, n_repeat=args.n_repeat,
        noise_threshold=args.noise_threshold,
        mid_eta=args.mid_eta,
        eta_grid=eta_grid_shared,
    )
    print(f"\n  [Test 1] eta_min_mono = {np.round(eta_min_mono, 4)}", flush=True)

    # ── Test 2: Concavity ─────────────────────────────────────────────────────
    eta_grid_2, profiles_2, d2, eta_min_conc = run_concavity_scan(
        accuracy_fn, num_links,
        n_grid=args.n_grid, n_repeat=args.n_repeat,
        concavity_threshold=args.concavity_threshold,
        mid_eta=args.mid_eta,
        eta_grid=eta_grid_shared,
    )
    print(f"\n  [Test 2] eta_min_conc = {np.round(eta_min_conc, 4)}", flush=True)

    # ── Test 3: Directional profiles ─────────────────────────────────────────
    s_grid, ray_starts, ray_profiles = run_directional_profiles(
        accuracy_fn, num_links,
        n_grid=args.n_grid, n_repeat=args.n_repeat,
        n_directions=args.n_directions,
    )

    # ── Final recommendation ──────────────────────────────────────────────────
    eta_min_raw    = np.maximum(eta_min_mono, eta_min_conc)
    eta_min_final  = np.clip(eta_min_raw + args.safety_margin, 0.0, 1.0)

    print("\n" + "="*60, flush=True)
    print("RESULTS", flush=True)
    print("="*60, flush=True)
    for link in range(num_links):
        print(f"  Link {link}:", flush=True)
        print(f"    eta_min (monotonicity) = {eta_min_mono[link]:.4f}", flush=True)
        print(f"    eta_min (concavity)    = {eta_min_conc[link]:.4f}", flush=True)
        print(f"    eta_min (max of above) = {eta_min_raw[link]:.4f}", flush=True)
        print(f"    eta_min (+ margin {args.safety_margin:.2f}) = "
              f"{eta_min_final[link]:.4f}  ← RECOMMENDED", flush=True)

    global_eta_min = float(np.max(eta_min_final))
    print(f"\n  Global eta_min (max over links) = {global_eta_min:.4f}", flush=True)
    print(f"  (use this as --eta_min and --grid_eta_min in training scripts)",
          flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "backend":            args.backend,
        "compressor":         args.compressor,
        "num_links":          num_links,
        "eta_min_monotonicity": eta_min_mono.tolist(),
        "eta_min_concavity":    eta_min_conc.tolist(),
        "eta_min_recommended":  eta_min_final.tolist(),
        "global_eta_min":       global_eta_min,
        "safety_margin":        args.safety_margin,
        "noise_threshold":      args.noise_threshold,
        "concavity_threshold":  args.concavity_threshold,
        "axis_profiles": {
            "eta_grid": eta_grid_1.tolist(),
            "profiles": profiles_1.tolist(),
        },
        "directional_profiles": {
            "s_grid":       s_grid.tolist(),
            "ray_starts":   ray_starts.tolist(),
            "ray_profiles": ray_profiles.tolist(),
        },
    }
    json_path = os.path.join(args.out_dir, "concavity_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results: {json_path}", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...", flush=True)
    plot_axis_profiles(eta_grid_1, profiles_1, eta_min_mono, eta_min_conc,
                       eta_min_final, args.out_dir,
                       args.backend, args.compressor,
                       no_title=False, save_pdf=False)
    plot_second_differences(eta_grid_2, d2, eta_min_conc,
                            args.out_dir, args.backend, args.compressor,
                            no_title=False, save_pdf=False)
    plot_directional_profiles(s_grid, ray_starts, ray_profiles,
                              eta_min_final, args.out_dir,
                              args.backend, args.compressor,
                              no_title=False, save_pdf=False)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
