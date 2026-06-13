"""
Post-hoc smoothed-concavity analysis over saved MC ray families.

Motivation
----------
The discrete second-difference test in ``mc_concavity.py`` classifies a ray as
concave iff D2[j] = y[j-1] + y[j+1] - 2 y[j] <= tol at every interior point.
That test is sensitive to *noise* in the profile y(t): a single noisy point can
flip a genuinely-concave ray to non-concave (this is exactly why MMLU absolute
accuracy — discrete, near the chance floor — scores so low). This script asks:
**how much of the measured non-concavity is noise rather than true curvature?**

It re-classifies every ray after applying a mild, shape-neutral low-pass filter
to y(t) and reports the concave fraction per family at several smoothing
strengths. We use a BINOMIAL [1,2,1]/4 taps smoother (with edge-reflect padding
so all K points survive). Binomial smoothing is a neutral Gaussian-like
low-pass; we deliberately avoid Savitzky-Golay with polyorder=2, which locally
fits a quadratic and would bias a *curvature* test toward its own verdict.

"raw" (0 passes) reproduces mc_concavity's own concavity number exactly.

Usage
-----
  python experiments/smooth_concavity.py            # all families under outputs/
  python experiments/smooth_concavity.py --glob '*mmlu*'
  python experiments/smooth_concavity.py --passes 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def binomial_smooth(ys: Sequence[float], passes: int) -> List[float]:
    """Apply `passes` rounds of a [1,2,1]/4 binomial smoother to the INTERIOR.

    The two endpoints are held FIXED — they are real measured values, and any
    padding scheme that smooths them drags a monotone profile's endpoint toward
    its (higher/lower) neighbour, manufacturing a spurious convex/concave spike
    at j=1. Anchoring the endpoints denoises interior wiggles without inventing
    curvature at the boundary.
    """
    y = list(map(float, ys))
    for _ in range(passes):
        if len(y) < 3:
            break
        y = [y[0]] + [0.25 * y[i - 1] + 0.5 * y[i] + 0.25 * y[i + 1]
                      for i in range(1, len(y) - 1)] + [y[-1]]
    return y


def is_concave(ys: Sequence[float], tol: float) -> bool:
    """D2[j] = y[j-1] + y[j+1] - 2 y[j] <= tol at every interior point."""
    return all(
        (ys[j - 1] + ys[j + 1] - 2.0 * ys[j]) <= tol
        for j in range(1, len(ys) - 1)
    )


def family_concavity(rays: List[dict], tol: float, passes: int) -> float:
    if not rays:
        return float("nan")
    n = sum(1 for r in rays if is_concave(binomial_smooth(r["y"], passes), tol))
    return n / len(rays)


def load_rays(path: Path):
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        rays = d.get("rays", []) + d.get("phase2_rays", [])
        tol = d.get("tol")
    else:  # bare partial list
        rays, tol = d, None
    return rays, tol


def infer_tol(path: Path, tol) -> float:
    if tol is not None:
        return float(tol)
    s = str(path).lower()
    # accuracy/MMLU profiles are discrete & noisy -> the looser run-time tol.
    return 0.02 if ("mmlu" in s or "accuracy" in s) else 0.001


def family_label(path: Path) -> str:
    s = str(path)
    parts = path.stem.replace("__", "/").split("/")
    # e.g. google/gemma-2b/sharegpt/perplexity/topk_per_token/cuts4_rays
    name = "/".join(p for p in parts if p not in ("rays",))
    # tag merged / seed variants from the directory
    if "merged" in path.stem:
        name += " [merged]"
    elif "cuts4_seed1" in s:
        name += " [seed1]"
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "outputs" / "mc_concavity"))
    ap.add_argument("--glob", default="*_rays.json")
    ap.add_argument("--passes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None, help="optional JSON dump")
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob(args.glob))
    rows = []
    for f in files:
        rays, tol = load_rays(f)
        tol = infer_tol(f, tol)
        if not rays:
            continue
        row = {
            "family": family_label(f),
            "path": str(f.relative_to(REPO_ROOT)),
            "n_rays": len(rays),
            "tol": tol,
            "concavity": {str(p): family_concavity(rays, tol, p)
                          for p in args.passes},
        }
        rows.append(row)

    # ── pretty table ──────────────────────────────────────────────────────
    hdr = f"{'family':<58} {'M':>4} {'tol':>6}  " + "  ".join(
        f"{('raw' if p == 0 else f'sm{p}'):>6}" for p in args.passes)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = "  ".join(f"{r['concavity'][str(p)]:>6.3f}" for p in args.passes)
        print(f"{r['family']:<58} {r['n_rays']:>4} {r['tol']:>6.3f}  {cells}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
