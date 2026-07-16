"""
Per-compression-method grids of Jensen-concavity-vs-eta-floor charts.

Feedback-driven rework of plot_concavity_grid.py:
  * ONE FIGURE PER COMPRESSION METHOD (topk / llmint8 / quantization), so each
    figure compares tasks under the SAME method instead of mixing methods in
    one axes ("oranges vs apples").
  * Within each subplot (one per task) the full set of tolerances is drawn as
    one line per tol, distinguished by MARKER SHAPE (all lines same color),
    not by color.
  * Together the figures cover the full cartesian product
    (compression method) x (tolerance) x (task).
  * Task list is discovered from the ray pool (preferred order first, unknown
    tasks — e.g. future T5 runs — appended automatically), so new results are
    picked up by just re-running the script.

Usage:
  python experiments/plot_concavity_per_method.py \
      --root outputs/mc_concavity --out_dir outputs/mc_concavity/jensen
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from jensen_concavity import floor_sweep
from plot_concavity_grid import load_ray_pool, _TASK_ORDER, _TASK_TITLE

REPO_ROOT = Path(__file__).resolve().parents[1]

# Strategy IDENTITY: LLM keys (topk_per_token / llmint8_reserve) and ResNet
# keys (topk / llmint8) collapse to one method per figure.
_STRAT_IDENTITY = {
    "topk_per_token": "topk",
    "topk": "topk",
    "llmint8_reserve": "llmint8",
    "llmint8": "llmint8",
    "quantization": "quantization",
}
_METHOD_ORDER = ["topk", "llmint8", "quantization"]

# Ordinal marker per tol; all curves share ONE color, markers carry the tol.
_TOL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINE_COLOR = "#2f2e2c"


def _task_title(task_key: Tuple) -> str:
    if task_key in _TASK_TITLE:
        return _TASK_TITLE[task_key]
    model, dataset, metric, n_cuts = task_key
    return f"{model} · {dataset} {metric} (cuts={n_cuts})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "mc_concavity" / "jensen"))
    ap.add_argument("--resnet_root",
                    default=str(REPO_ROOT / "outputs" / "concavity_resnet"))
    ap.add_argument("--tols", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--samples_per_ray", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.7,
                    help="Adaptive fill only guarantees usable rays up to 0.7.")
    ap.add_argument("--floor_step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floors = np.round(np.arange(args.floor_min, args.floor_max + 1e-9, args.floor_step), 3)

    pool = load_ray_pool(Path(args.root), Path(args.resnet_root))

    # curves[method][(model,dataset,metric,n_cuts)][tol] = {"mean","std"}
    curves: Dict[str, Dict[Tuple, Dict[float, dict]]] = {}
    for (model, dataset, metric, n_cuts, strat), entry in pool.items():
        method = _STRAT_IDENTITY.get(strat)
        if method is None:
            continue
        task_key = (model, dataset, metric, n_cuts)
        for tol in args.tols:
            repeat_concavity = []
            for rep in range(args.n_repeats):
                rng = np.random.default_rng(args.seed * 1000 + rep)
                sweep = floor_sweep(entry["rays"], floors=floors, rng=rng,
                                    samples_per_ray=args.samples_per_ray, tol=tol)
                repeat_concavity.append(
                    [s["concavity"] if s["concavity"] is not None else np.nan
                     for s in sweep])
            arr = np.array(repeat_concavity, dtype=np.float64)
            curves.setdefault(method, {}).setdefault(task_key, {})[tol] = {
                "mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0),
            }
        print(f"  {method:12s} {model}|{dataset}/{metric}|cuts{n_cuts}: "
              f"{len(entry['rays'])} rays")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for method in _METHOD_ORDER:
        if method not in curves:
            continue
        task_curves = curves[method]
        # Preferred order first, then any newly discovered tasks (e.g. T5).
        tasks = [t for t in _TASK_ORDER if t in task_curves]
        tasks += sorted(t for t in task_curves if t not in _TASK_ORDER)

        ncols = min(3, len(tasks))
        nrows = int(np.ceil(len(tasks) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 4.0 * nrows),
                                 squeeze=False)
        for idx, task_key in enumerate(tasks):
            ax = axes[idx // ncols][idx % ncols]
            for ti, tol in enumerate(args.tols):
                sweep = task_curves[task_key].get(tol)
                if sweep is None:
                    continue
                mean, std = sweep["mean"], sweep["std"]
                mask = ~np.isnan(mean)
                marker = _TOL_MARKERS[ti % len(_TOL_MARKERS)]
                ax.plot(floors[mask], mean[mask], color=_LINE_COLOR,
                        linewidth=1.3, marker=marker, markersize=5,
                        markerfacecolor="white", markeredgewidth=1.2)
                ax.fill_between(floors[mask],
                                np.clip(mean[mask] - std[mask], 0, 1),
                                np.clip(mean[mask] + std[mask], 0, 1),
                                color=_LINE_COLOR, alpha=0.08, linewidth=0)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(args.floor_min, args.floor_max)
            ax.axhline(1.0, color="k", linestyle=":", alpha=0.3, linewidth=1)
            ax.grid(True, alpha=0.25)
            ax.set_title(_task_title(task_key), fontsize=10)
            ax.set_xlabel("eta floor  c", fontsize=9)
            ax.set_ylabel("Jensen concavity", fontsize=9)
            ax.tick_params(labelsize=8)
        for j in range(len(tasks), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        tol_handles = [
            Line2D([0], [0], color=_LINE_COLOR, lw=1.3,
                   marker=_TOL_MARKERS[ti % len(_TOL_MARKERS)], markersize=6,
                   markerfacecolor="white", markeredgewidth=1.2,
                   label=f"tol={t:g}")
            for ti, t in enumerate(args.tols)]
        fig.legend(handles=tol_handles, title="tol (concavity offset)",
                   loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   ncol=len(tol_handles), fontsize=9)
        fig.suptitle(
            f"Jensen concavity vs eta-floor — {method}\n"
            f"({args.samples_per_ray} triples/ray, mean ± std over "
            f"{args.n_repeats} resamples)",
            fontsize=13, y=1.14)
        fig.tight_layout()
        out_path = out_dir / f"concavity_grid_{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
