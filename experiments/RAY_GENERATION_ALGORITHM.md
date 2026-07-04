# Ray-generation algorithm (`experiments/mc_concavity.py`)

Reframes the fixed-ray concavity test as a Monte-Carlo property of the whole
eta-hypercube. For a task (model, dataset/metric, compression strategy,
`n_cuts`), the compressor exposes an `n_cuts`-dimensional "keep fraction"
vector eta in `[0,1]^n` — one coordinate per cut point in the network.
`eta = 1` everywhere is the uncompressed baseline; lower eta means more
aggressive compression at that cut point. The goal is to test whether the
task metric is concave along line segments ("rays") through this hypercube.

## 1. Sample a random ray (`sample_random_ray`)

- Draw two independent points `start`, `end` uniformly at random from the box
  `[lo, hi]^n` (`lo = eps = 1e-3` in Phase 1, `hi = 1.0`).
- No monotonicity constraint (`end >= start`) is imposed — rays are
  isotropic, pointing in any direction through the cube.
- Reject and resample (up to 50 tries) if `||end - start|| < min_seg`
  (default 0.05) to avoid near-degenerate segments.
- Clip both endpoints to `[eps, 1]` to avoid `eta = 0`.

## 2. Discretize the ray (`ray_eta_points`)

- `n_points` (default 7) equally spaced `t` in `[0, 1]` via `linspace`.
- `eta(t) = (1 - t) * start + t * end` — straight-line interpolation, so
  each ray is one line segment through eta-space.

## 3. Evaluate the metric at each point

Set the compressor's eta to that vector, run the model, get the task score:

- **Perplexity tasks** (ShareGPT / WikiText): mean per-sample
  `min{1, PPL_ref / PPL_comp}`, normalized against the `eta = 1` baseline.
- **MMLU**: raw absolute accuracy in `[0, 1]`, not baseline-normalized.

This produces a sequence `y(t_0), ..., y(t_{n_points-1})` along the ray.

## 4. Classify concavity (`second_difference_test_on`)

For each interior point `j`:

```
D2[j] = y[j-1] + y[j+1] - 2 * y[j]
```

Concave (on this discrete grid) requires `D2[j] <= tol` everywhere
(`tol` default `1e-3` for perplexity, `0.02` for accuracy). A positive `D2`
means the function is locally convex there — a "violation." The ray is
tagged `concave = (violations == 0)`, along with `worst_d2`.

## 5. Repeat for `n_rays` — Phase 1

Isotropic sampling of the whole cube. The full ray list (start / end / t /
eta / y / concave) is checkpointed to `*_phase1.partial.json` every 5 rays
so a SLURM walltime kill doesn't lose completed work.

## 6. Per-cut-point `eta_min` search (`find_eta_min`)

Model-free; operates on saved rays only. Greedily ascends a per-coordinate
floor vector `v` from `0` (coarse step) until the concavity fraction over
rays clipped to `[v, 1]^n` hits the target (95%), then fine-descends to
shrink `v` back down. Gives the smallest "safe" sub-cube, plus a diagnostic
scalar-floor sweep (`v = c * 1`).

Ascending from `v = 0` (data-rich) rather than descending from `v = 1`
(data-starved) is deliberate: concavity rises as the floor is raised (the
low-eta "weird behavior" regime is excluded), but the usable ray count
falls, so descending from `1` fails immediately (near the corner, usable
ray count is 0).

## 7. Phase 2 (optional, `phase2_rays`)

Resamples fresh rays **directly inside** `[eta_min_candidate, 1]^n` (not
clipped survivors of Phase-1 rays) to get unbiased estimates in the
candidate high-eta region without the endpoint-masking leniency of clipping.
`eta_min` is then refined on the combined ray set.

## 8. Adaptive fill (`adaptive_fill_phase`, the `fill` subcommand)

For each floor `c` in `[0, floor_max]` step `floor_step` (default 0.1),
scanning from high to low: if fewer than `min_box_samples` rays are
"usable" at that floor (>= 3 contiguous saved points survive clipping to
`[c, 1]^n`), sample **new rays with `lo = max(lo_full, c)`** — i.e. force
both endpoints >= `c` so every fill ray is usable at floor `c` and all
lower floors for free — until the target count is met.

- Checkpointed to `*_fill.partial.json` every 5 rays; safe to resubmit
  (`cmd_fill` skips floors, or the whole job, that are already satisfied).
- `--floor_max` (default 0.9) caps how high a floor the fill targets —
  added so a `fill` run can be scoped to only the range a plot actually
  needs (e.g. 0.7) instead of always filling out to 0.9.

## Concavity metrics computed downstream

- **`mc_concavity.py`**: per-ray, all-or-nothing — a ray counts as concave
  only if it has zero `D2` violations end to end.
- **`jensen_concavity.py`**: a softer, post-hoc re-analysis of the same
  `*_rays.json` files — per-ray it samples triples `i < j < k` of grid
  points and tests Jensen's inequality per triple
  (`y_j >= lam * y_i + (1 - lam) * y_k`); concavity = fraction of triples
  passed. This is what the "concavity vs eta-floor" plots use. A curve for
  a given (model, dataset/metric, n_cuts, strategy) stops at some floor `c`
  once zero rays have all `n` eta-coordinates >= `c` simultaneously
  (`usable_rays = 0`, `concavity = None`, dropped from the plotted line).
- **`smooth_concavity.py`**: re-classifies each ray after denoising `y(t)`
  with `k` passes of a binomial `[1,2,1]/4` smoothing kernel on the
  interior (endpoints held fixed), to separate genuine non-concavity from
  measurement noise.
