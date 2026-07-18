"""
Check the min-usable-rays-per-eta-floor condition (ray *density*).

A ray is "usable" at floor c if >=3 of its grid points have every eta
coordinate >= c (>=3 points = at least one Jensen triple / one second
difference). The adaptive fill is supposed to guarantee >= min_box_samples
usable rays at every floor up to floor_max; this script verifies that.

Two modes
---------
1. Explicit files (unchanged; used by the fill-chain scripts as a stop test):

     python experiments/check_ray_coverage.py <rays.json> [more.json ...] \
         [--min 100] [--floor_max 0.7] [--floor_step 0.1]

   Prints a per-floor table per file and exits 0 iff every floor has
   >= --min usable rays (so a fill chain can loop on the exit code).

2. Discovery (--discover): scan a directory tree, merge every ray for each
   (model, dataset, metric, cuts, strategy) key -- de-duplicated by content
   hash and folding in the ResNet directional-profile runs, exactly like
   jensen_concavity -- then report, PER TASK AND CUTS, the usable-ray density
   at each 0.1 eta-floor increment and flag where it drops below --min:

     python experiments/check_ray_coverage.py --discover \
         --root . --resnet_root outputs/concavity_resnet --min 100

   Exits 0 iff every (task, cuts, strategy) group clears --min at every floor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jensen_concavity import (  # noqa: E402
    discover_ray_files, _rays_from, _infer_meta, _ray_hash, resnet_entries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def rays_from(data) -> list:
    return _rays_from(data)


def usable(ray: dict, c: float) -> bool:
    return sum(1 for e in ray["eta"] if min(e) >= c) >= 3


# ──────────────────────────────────────────────────────────────────────────
# Discovery mode: group by (model, dataset, metric, cuts, strategy)
# ──────────────────────────────────────────────────────────────────────────

def build_groups(root: Path, resnet_root: Path) -> Dict[Tuple, dict]:
    """key -> {"meta", "rays"} with rays de-duplicated by content hash, merging
    LLM run files + partial checkpoints + ResNet directional-profile runs."""
    best: Dict[Tuple, dict] = {}

    def _add(meta: dict, rays: List[dict]) -> None:
        if meta.get("n_cuts") is None:
            return
        n_cuts = int(meta["n_cuts"])
        key = (meta["model"], meta["dataset"], meta["metric"],
               n_cuts, meta["strategy"])
        # Guard against mislabeled files (e.g. a corrupt cuts7_merged_rays.json
        # that declares n_cuts=7 but stores 4-dim rays): a ray must have exactly
        # n_cuts eta coordinates or it silently corrupts the min-eta density.
        good = [r for r in rays if len(r.get("start", [])) == n_cuts]
        dropped = len(rays) - len(good)
        if dropped:
            print(f"  drop {dropped} rays with dim != {n_cuts} for {key}")
        entry = best.setdefault(key, {"meta": meta, "rays": [], "seen": set()})
        for r in good:
            h = _ray_hash(r)
            if h in entry["seen"]:
                continue
            entry["seen"].add(h)
            entry["rays"].append(r)

    for f in discover_ray_files(root):
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        rays = _rays_from(data)
        if rays:
            _add(_infer_meta(f, data), rays)

    if resnet_root and resnet_root.exists():
        for meta, rays, _src in resnet_entries(resnet_root):
            _add(meta, rays)
    return best


def check_discovered(root: Path, resnet_root: Path, floors: np.ndarray,
                     min_rays: int) -> bool:
    groups = build_groups(root, resnet_root)
    if not groups:
        print(f"No ray groups found under {root}")
        return False

    # Order by task (model, dataset, metric, cuts) then strategy.
    def task_of(k):
        return (str(k[0]), str(k[1]), str(k[2]), k[3])

    ok_all = True
    last_task = None
    for key in sorted(groups, key=lambda k: (task_of(k), str(k[4]))):
        model, dataset, metric, n_cuts, strat = key
        task = (model, dataset, metric, n_cuts)
        if task != last_task:
            print(f"\n### {model} | {dataset}/{metric} | cuts={n_cuts}")
            last_task = task
        rays = groups[key]["rays"]

        counts = [sum(1 for r in rays if usable(r, float(c))) for c in floors]
        oks = [n >= min_rays for n in counts]
        ok_all &= all(oks)
        # Highest floor still meeting the density threshold.
        reliable = max((float(c) for c, o in zip(floors, oks) if o),
                       default=None)
        cells = "  ".join(
            f"{c:.1f}:{n}{'' if o else '*'}"
            for c, n, o in zip(floors, counts, oks))
        tag = "OK" if all(oks) else "THIN"
        rel = f"reliable<= {reliable:.1f}" if reliable is not None else "reliable<none"
        print(f"  {strat:18s} n={len(rays):4d}  [{tag:4s} {rel}]  {cells}")

    print(f"\n('*' marks floors with < {min_rays} usable rays; "
          f"cell = floor:usable_ray_count)")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rays_json", nargs="*",
                    help="Explicit ray files (omit when using --discover).")
    ap.add_argument("--discover", action="store_true",
                    help="Auto-discover and group by (task, cuts, strategy).")
    ap.add_argument("--root", default=".",
                    help="Discovery root scanned for *_rays.json / partials.")
    ap.add_argument("--resnet_root",
                    default=str(REPO_ROOT / "outputs" / "concavity_resnet"),
                    help="ResNet directional-profile results, folded in.")
    ap.add_argument("--min", type=int, default=100, dest="min_rays")
    ap.add_argument("--floor_max", type=float, default=0.7)
    ap.add_argument("--floor_step", type=float, default=0.1)
    args = ap.parse_args()

    floors = np.round(np.arange(0.0, args.floor_max + 1e-9, args.floor_step), 3)

    if args.discover or not args.rays_json:
        if not (args.discover or args.rays_json):
            print("Nothing to check: pass ray files or --discover.")
            sys.exit(2)
        ok = check_discovered(Path(args.root), Path(args.resnet_root),
                              floors, args.min_rays)
        sys.exit(0 if ok else 1)

    # Explicit-file mode (unchanged behaviour / exit code for fill chains).
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
