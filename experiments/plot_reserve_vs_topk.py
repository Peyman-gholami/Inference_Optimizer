#!/usr/bin/env python3
"""
Rebuild outputs/ray_concavity_score/reserve_vs_topk.png from the current
per-ray sweep CSVs produced by ray_concavity_score.py.

Grid: one row per model, one column per ray. Each panel plots the mean
per-sample perplexity-ratio score vs. t in [0,1] for the two strategies
(llmint8_reserve vs. topk_per_token) on identical rays.

The figure is derived purely from the *_sweep.csv files on disk, so it
always reflects the most recent run. Run after the SLURM jobs finish:

    python experiments/plot_reserve_vs_topk.py
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (model dir slug, pretty label) -- one row each, top to bottom.
MODELS = [
    ("meta-llama_Llama-3.1-8B", "Llama 3.1-8B"),
    ("google_gemma-7b", "Gemma 7B"),
]

# Rays in the order they appear in rays_interesting_{5,8}.json -- one column each.
RAYS = ["diag_full", "diag_knee", "front_only", "back_only", "depth_gradient"]

# (strategy dir, pretty label, color).
STRATEGIES = [
    ("llmint8_reserve", "llmint8", "tab:blue"),
    ("topk_per_token", "top-k", "tab:orange"),
]


def sweep_path(out_base, model_slug, strategy, cuts, ray):
    # File name pattern: <model>__<strategy>__<ray>_sweep.csv,
    # under <model>/<strategy>/cuts<N>/.
    fname = f"{model_slug.replace('_', '__', 1)}__{strategy}__{ray}_sweep.csv"
    return os.path.join(out_base, model_slug, strategy, f"cuts{cuts}", fname)


def load_sweep(path):
    ts, scores = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["t"]))
            scores.append(float(row["mean_score"]))
    return ts, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_base",
        default=os.path.join(os.path.dirname(__file__), "..", "outputs", "ray_concavity_score"),
    )
    ap.add_argument("--n_texts", type=int, default=256, help="for the title only")
    ap.add_argument("--n_points", type=int, default=16, help="for the title only")
    ap.add_argument(
        "--cuts",
        type=int,
        nargs="+",
        default=[5, 8],
        help="cut-point counts to render; one figure per value",
    )
    args = ap.parse_args()
    out_base = os.path.abspath(args.out_base)

    for cuts in args.cuts:
        render_figure(out_base, cuts, args.n_texts, args.n_points)


def render_figure(out_base, cuts, n_texts, n_points):
    nrows, ncols = len(MODELS), len(RAYS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), sharex=True, sharey=True
    )

    missing = []
    for i, (model_slug, model_label) in enumerate(MODELS):
        for j, ray in enumerate(RAYS):
            ax = axes[i][j]
            for strategy, slabel, color in STRATEGIES:
                path = sweep_path(out_base, model_slug, strategy, cuts, ray)
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                ts, scores = load_sweep(path)
                ax.plot(ts, scores, marker=".", ms=4, lw=1.2, color=color, label=slabel)
            if i == 0:
                ax.set_title(ray, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{model_label}\nmean score", fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel("t", fontsize=8)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.25)
    # Legend once, in the first panel.
    axes[0][0].legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"llmint8 vs top-k: per-ray score, {cuts} cut-points "
        f"(n={n_texts}, {n_points} points/ray)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(out_base, f"reserve_vs_topk_cuts{cuts}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    if missing:
        print(f"WARNING: missing sweep CSVs for cuts{cuts} (skipped):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
