"""
Grid of Jensen-concavity-vs-eta-floor line charts, one subplot per task, with
a line per (strategy, tol) combination.

Reuses the ray discovery/merge/floor-sweep machinery from jensen_concavity.py
(same de-duped, all-phases-and-seeds ray pool) but sweeps a *range of tol*
values ("the little offset we allow to deviate from concavity") instead of
one fixed tol per metric, so each subplot shows how forgiving the concavity
test needs to be before each strategy's curve saturates.

Usage:
  python experiments/plot_concavity_grid.py \
      --root outputs/mc_concavity --out outputs/mc_concavity/jensen/concavity_grid.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from jensen_concavity import (
    discover_ray_files, _rays_from, _ray_hash, _infer_meta, floor_sweep,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed categorical color per strategy (dataviz palette slots 1 & 6).
_STRAT_COLOR = {
    "topk_per_token": "#2a78d6",   # slot 1, blue
    "llmint8_reserve": "#e34948",  # slot 6, red
}
_STRAT_LABEL = {
    "topk_per_token": "top-k per token",
    "llmint8_reserve": "llmint8 reserve",
}
# Ordinal line style per tol, lightest/densest dashing = smallest offset.
_TOL_STYLE = {
    0.00: "-",
    0.01: (0, (5, 1)),
    0.02: (0, (4, 2)),
    0.05: (0, (2, 2)),
    0.10: (0, (1, 1.6)),
}
_TASK_ORDER = [
    ("google/gemma-2b", "sharegpt", "perplexity", 4),
    ("google/gemma-2b", "sharegpt", "perplexity", 7),
    ("google/gemma-7b", "sharegpt", "perplexity", 4),
    ("meta-llama/Llama-3.1-8B", "wikitext", "perplexity", 4),
    ("meta-llama/Llama-3.1-8B", "mmlu", "accuracy", 4),
]
_TASK_TITLE = {
    ("google/gemma-2b", "sharegpt", "perplexity", 4): "gemma-2b · ShareGPT ppl (cuts=4)",
    ("google/gemma-2b", "sharegpt", "perplexity", 7): "gemma-2b · ShareGPT ppl (cuts=7)",
    ("google/gemma-7b", "sharegpt", "perplexity", 4): "gemma-7b · ShareGPT ppl (cuts=4)",
    ("meta-llama/Llama-3.1-8B", "wikitext", "perplexity", 4): "Llama-3.1-8B · WikiText ppl (cuts=4)",
    ("meta-llama/Llama-3.1-8B", "mmlu", "accuracy", 4): "Llama-3.1-8B · MMLU acc (cuts=4)",
}


def load_ray_pool(root: Path) -> Dict[Tuple, dict]:
    files = discover_ray_files(root)
    print(f"Found {len(files)} ray files (completed + partial) under {root}")
    best: Dict[Tuple, dict] = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        rays = _rays_from(data)
        if not rays:
            continue
        meta = _infer_meta(f, data)
        if meta["n_cuts"] is None:
            continue
        key = (meta["model"], meta["dataset"], meta["metric"],
               int(meta["n_cuts"]), meta["strategy"])
        entry = best.setdefault(key, {"meta": meta, "rays": [], "seen": set()})
        for r in rays:
            h = _ray_hash(r)
            if h in entry["seen"]:
                continue
            entry["seen"].add(h)
            entry["rays"].append(r)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "mc_concavity" / "jensen" / "concavity_grid.png"))
    ap.add_argument("--tols", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--samples_per_ray", type=int, default=5,
                    help="Jensen triples sampled per ray per floor, per repeat "
                         "(was 'all combinations'; now a small fixed draw).")
    ap.add_argument("--n_repeats", type=int, default=5,
                    help="Independent resamples of samples_per_ray triples, used "
                         "to compute a mean +/- std band per curve.")
    ap.add_argument("--floor_min", type=float, default=0.0)
    ap.add_argument("--floor_max", type=float, default=0.7,
                    help="Highest eta floor to plot. Defaults to 0.7: the "
                         "adaptive fill guarantees >=min_box_samples usable rays "
                         "only up to floor 0.7, so the sweep is unreliable above it.")
    ap.add_argument("--floor_step", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    floors = np.round(np.arange(args.floor_min, args.floor_max + 1e-9, args.floor_step), 3)

    pool = load_ray_pool(Path(args.root))

    # curves[(model,dataset,metric,n_cuts)][strategy][tol] = {"floors","mean","std"}
    curves: Dict[Tuple, Dict[str, Dict[float, dict]]] = {}
    for (model, dataset, metric, n_cuts, strat), entry in pool.items():
        if strat not in _STRAT_COLOR:
            continue
        task_key = (model, dataset, metric, n_cuts)
        for tol in args.tols:
            # n_repeats independent draws of samples_per_ray triples each ->
            # per-floor mean/std across repeats (band reflects small-sample
            # triple-selection noise, not just binomial trial-count noise).
            repeat_concavity = []  # list of arrays, one per repeat, len(floors)
            for rep in range(args.n_repeats):
                rng = np.random.default_rng(args.seed * 1000 + rep)
                sweep = floor_sweep(entry["rays"], floors=floors, rng=rng,
                                    samples_per_ray=args.samples_per_ray, tol=tol)
                repeat_concavity.append(
                    [s["concavity"] if s["concavity"] is not None else np.nan for s in sweep])
            arr = np.array(repeat_concavity, dtype=np.float64)  # (n_repeats, n_floors)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            curves.setdefault(task_key, {}).setdefault(strat, {})[tol] = {
                "floors": floors, "mean": mean, "std": std,
            }
            n_rays = len(entry["rays"])
        print(f"  {model}|{dataset}/{metric}|cuts{n_cuts}|{strat}: {n_rays} rays, "
              f"{len(args.tols)} tols x {args.n_repeats} repeats of "
              f"{args.samples_per_ray} triples/ray")

    tasks = [t for t in _TASK_ORDER if t in curves]
    if not tasks:
        raise SystemExit("No matching tasks found under root.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

    for idx, task_key in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        strat_curves = curves[task_key]
        for strat in ("topk_per_token", "llmint8_reserve"):
            if strat not in strat_curves:
                continue
            color = _STRAT_COLOR[strat]
            for tol in args.tols:
                sweep = strat_curves[strat][tol]
                xs = sweep["floors"]
                mean = sweep["mean"]
                std = sweep["std"]
                mask = ~np.isnan(mean)
                ax.plot(xs[mask], mean[mask], color=color, linewidth=1.6,
                        linestyle=_TOL_STYLE.get(tol, "-"))
                ax.fill_between(xs[mask], np.clip(mean[mask] - std[mask], 0, 1),
                                np.clip(mean[mask] + std[mask], 0, 1),
                                color=color, alpha=0.10, linewidth=0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(args.floor_min, args.floor_max)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.25)
        ax.set_title(_TASK_TITLE.get(task_key, str(task_key)), fontsize=10)
        ax.set_xlabel("eta floor  c", fontsize=9)
        ax.set_ylabel("Jensen concavity", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused axes.
    for j in range(len(tasks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Two separate legends: strategy (color/identity) and tol (linestyle/ordinal).
    from matplotlib.lines import Line2D
    strat_handles = [Line2D([0], [0], color=_STRAT_COLOR[s], lw=2.2, label=_STRAT_LABEL[s])
                      for s in ("topk_per_token", "llmint8_reserve") if s in _STRAT_COLOR]
    tol_handles = [Line2D([0], [0], color="#52514e", lw=1.6, linestyle=_TOL_STYLE[t],
                          label=f"tol={t:g}") for t in args.tols]
    leg1 = fig.legend(handles=strat_handles, title="strategy", loc="upper center",
                      bbox_to_anchor=(0.28, 1.04), ncol=2, fontsize=9)
    fig.legend(handles=tol_handles, title="tol (concavity offset)", loc="upper center",
              bbox_to_anchor=(0.75, 1.04), ncol=len(tol_handles), fontsize=9)
    fig.add_artist(leg1)

    fig.suptitle(f"Jensen concavity vs eta-floor, by strategy and concavity tolerance\n"
                 f"({args.samples_per_ray} triples/ray, mean ± std over {args.n_repeats} resamples)",
                 fontsize=13, y=1.13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
