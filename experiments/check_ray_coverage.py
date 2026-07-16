"""
Check the min-usable-rays-per-eta-floor condition on a *_rays.json file.

A ray is "usable" at floor c if >=3 of its grid points have every eta
coordinate >= c (>=3 points = at least one Jensen triple / one second
difference). The adaptive fill is supposed to guarantee >= min_box_samples
usable rays at every floor up to floor_max; this script verifies that,
prints the per-floor table, and exits 0 iff the condition holds (so a fill
chain can use it as its stop test).

Usage:
  python experiments/check_ray_coverage.py <rays.json> [more.json ...] \
      [--min 100] [--floor_max 0.7] [--floor_step 0.1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rays_from(data) -> list:
    if isinstance(data, list):
        return list(data)
    return (list(data.get("rays", [])) + list(data.get("phase2_rays", []))
            + list(data.get("fill_rays", [])))


def usable(ray: dict, c: float) -> bool:
    return sum(1 for e in ray["eta"] if min(e) >= c) >= 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rays_json", nargs="+")
    ap.add_argument("--min", type=int, default=100, dest="min_rays")
    ap.add_argument("--floor_max", type=float, default=0.7)
    ap.add_argument("--floor_step", type=float, default=0.1)
    args = ap.parse_args()

    floors = np.round(np.arange(0.0, args.floor_max + 1e-9, args.floor_step), 3)
    ok_all = True
    for path in args.rays_json:
        data = json.loads(Path(path).read_text())
        rays = rays_from(data)
        counts = {}
        if isinstance(data, dict):
            counts = {k: len(data.get(k, [])) for k in
                      ("rays", "phase2_rays", "fill_rays")}
        print(f"\n{path}\n  {len(rays)} rays {counts}")
        print(f"  {'floor':>6s} {'usable':>7s} {'status':>7s}")
        for c in floors:
            n = sum(1 for r in rays if usable(r, float(c)))
            ok = n >= args.min_rays
            ok_all &= ok
            print(f"  {c:6.2f} {n:7d} {'OK' if ok else 'FAIL':>7s}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
