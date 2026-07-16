"""
Why does tol=0 Jensen concavity DECREASE as the eta-floor increases?

Hypothesis: at high floors the surviving segment of each ray sits in the
near-lossless (eta ~ 1) regime where the metric is almost flat. The true
curvature there is ~0, so with tol=0 every triple is decided by evaluation
noise (a near coin-flip), while at floor 0 the strongly concave low-eta region
produces genuine, large-margin passes. If that's right, the chord-violation
margins of FAILING triples at high floor should be orders of magnitude smaller
than the metric's dynamic range (i.e. numerically-flat noise, not real
convexity).

For each (task, strategy) and a set of floors, this script enumerates triples
and records the signed Jensen margin  m = y_j - chord  (m < 0 = violation),
then reports the violation rate, the median/p90 |violation| size, and the
in-floor y-range for scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jensen_concavity import in_floor_indices
from plot_concavity_grid import load_ray_pool

REPO_ROOT = Path(__file__).resolve().parents[1]


def margins_at_floor(rays, c, rng, samples_per_ray=20):
    """Signed Jensen margins m = y_j - chord for sampled triples, plus the
    y-range of in-floor points (metric dynamic range at this floor)."""
    from itertools import combinations
    margins, y_lo, y_hi = [], np.inf, -np.inf
    for r in rays:
        idx = in_floor_indices(r["eta"], c)
        if len(idx) < 3:
            continue
        t = np.asarray(r["t"], dtype=np.float64)
        y = np.asarray(r["y"], dtype=np.float64)
        y_lo = min(y_lo, float(np.min(y[idx])))
        y_hi = max(y_hi, float(np.max(y[idx])))
        trips = list(combinations(idx, 3))
        if len(trips) > samples_per_ray:
            sel = rng.choice(len(trips), size=samples_per_ray, replace=False)
            trips = [trips[s] for s in sel]
        for i, j, k in trips:
            span = t[k] - t[i]
            if span <= 0:
                continue
            lam = (t[k] - t[j]) / span
            margins.append(float(y[j] - (lam * y[i] + (1 - lam) * y[k])))
    return np.asarray(margins), (y_lo, y_hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    ap.add_argument("--resnet_root",
                    default=str(REPO_ROOT / "outputs" / "concavity_resnet"))
    ap.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = load_ray_pool(Path(args.root), Path(args.resnet_root))
    rows = []
    for (model, dataset, metric, n_cuts, strat), entry in sorted(pool.items()):
        for c in args.floors:
            rng = np.random.default_rng(args.seed)
            m, (ylo, yhi) = margins_at_floor(entry["rays"], c, rng)
            if m.size == 0:
                continue
            viol = m[m < 0]
            rows.append({
                "task": f"{model}|{dataset}/{metric}|cuts{n_cuts}",
                "strategy": strat, "floor": c, "n_triples": int(m.size),
                "pass_rate_tol0": float((m >= 0).mean()),
                "viol_median": float(np.median(-viol)) if viol.size else 0.0,
                "viol_p90": float(np.quantile(-viol, 0.9)) if viol.size else 0.0,
                "y_range_in_floor": yhi - ylo,
            })

    print(f"{'task':48s} {'strategy':16s} {'floor':>5s} {'pass@0':>7s} "
          f"{'med|viol|':>10s} {'p90|viol|':>10s} {'y-range':>10s} {'med/range':>10s}")
    for r in rows:
        rel = r["viol_median"] / r["y_range_in_floor"] if r["y_range_in_floor"] > 0 else float("nan")
        print(f"{r['task']:48s} {r['strategy']:16s} {r['floor']:5.2f} "
              f"{r['pass_rate_tol0']:7.3f} {r['viol_median']:10.2e} "
              f"{r['viol_p90']:10.2e} {r['y_range_in_floor']:10.3e} {rel:10.2e}")

    out = REPO_ROOT / "outputs" / "mc_concavity" / "jensen" / "tol0_decline_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
