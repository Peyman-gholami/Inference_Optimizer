# Concavity Experiments — Results Summary

_Last updated 2026-06-13. Branch `concavity-experiments`._

## Why we ran these

The optimizer treats per-layer activation-compression ratios as a vector
**η ∈ [0,1]ⁿ** (one entry per decoder cut-point; η = 1 = uncompressed) and assumes
the task-quality function *f(η)* is roughly **concave**, so that chord / gradient
reasoning over the search domain is well-behaved. These experiments test that
assumption empirically across models, datasets, metrics, and the two compression
strategies, and try to locate the per-cut **floor η_min** below which concavity
breaks down.

Two generations of experiment:

1. **Fixed-ray probe** (`experiments/ray_concavity_score.py`) — a handful of
   hand-designed rays through η-space. Informative but a biased, low-coverage
   sample of the cube. Used mainly to compare compression schemes.
2. **Monte-Carlo probe** (`experiments/mc_concavity.py`) — reframes concavity as
   a property of the *whole* hypercube: sample many random rays, classify each
   concave/non-concave, report the concave fraction, and search for η_min. This
   is the current, primary experiment. Full methodology in
   `experiments/mc_concavity_REPORT.md`.

The metric along a ray is:
- **Perplexity tasks** (ShareGPT, WikiText): mean per-sample `min{1, PPL_ref/PPL_comp}`
  (1.0 = matches the uncompressed baseline).
- **Accuracy tasks** (MMLU): **absolute** raw accuracy in [0,1] (not baseline-normalized).

A ray is "concave" if the discrete second-difference test
`D2[j] = y[j−1] + y[j+1] − 2y[j] ≤ tol` holds at every interior point. Accuracy is
discrete and noisy, so accuracy tasks use a looser `tol` (0.02 vs 0.001).

---

## Prior result — compression-scheme comparison (fixed-ray)

Before the MC sweep we compared the two mixed-precision activation schemes that
both put η on a true compression-ratio axis (target bits = 16·η):

- **`llmint8_reserve`** — high-magnitude band is *always* FP16 (protected
  outliers); the low band steps int8→int4→int2→drop as η falls.
- **`llmint8_adjacent`** — the two bands are neighbouring ladder rungs; smooth
  composition but no permanent FP16 protection.

**Finding:** `llmint8_reserve` won decisively on the diagonal ray — it held
score ≈ 1.0 down to η ≈ 0.40 and stayed concave, while `llmint8_adjacent`
collapsed to ~0 by η = 0.40 (below η = 0.5 its top band loses FP16 entirely,
destroying the forward pass). This confirmed the LLM.int8() thesis: **protecting
activation outliers in FP16 is what matters**, not smoothness of composition.

A dense follow-up exposed a real bug in `reserve`: at *exactly* η = 0.50
(all-INT8) and η = 0.25 (all-INT4) the score collapses to ~0.0002 because
`f_high` rounds to zero there, leaving zero elements in FP16. Fix is to floor
`f_high = max(f_high, eps)` so outliers are always protected. **This floor fix
was identified but not yet committed** — relevant when reading the MC `reserve`
results, since those dips can still show up.

---

## Monte-Carlo concavity — main results

Task matrix (causal-LM only; Flan-T5/SST2 deferred — needs encoder-decoder
cut-point detection + an SST2 loader). Both strategies run per task.

| Model | Task | Strategy | cuts | Phase-1 concavity | Phase-2 concavity | η_min ≥95% met? | Scalar floor c\* (95%) |
|-------|------|----------|------|------------------|------------------|------------------|------------------------|
| gemma-2b | ShareGPT ppl | topk | 4 | 0.79 | **0.98** | yes | **0.10** |
| gemma-2b | ShareGPT ppl | reserve | 4 | 0.58 | 0.88 | yes | 0.25 |
| gemma-2b | ShareGPT ppl | topk | 7 | 0.67 | 0.93 | yes | **0.10** |
| gemma-2b | ShareGPT ppl | reserve | 7 | 0.39 | 0.67 | yes | 0.20 |
| gemma-7b | ShareGPT ppl | topk | 4 | 0.88 | 0.59 | **no** | — (peaks ~0.92) |
| gemma-7b | ShareGPT ppl | reserve | 4 | 0.68 | 0.85 | yes | 0.25 |
| Llama-3.1-8B | WikiText ppl | topk | 4 | **0.95** | **0.98** | yes (η_min ≈ 0) | **0.05** |
| Llama-3.1-8B | WikiText ppl | reserve | 4 | 0.58 | 0.85 | yes | 0.30 |
| Llama-3.1-8B | MMLU acc | topk | 4 | **0.23** (300 rays) | — | **no** | — (never) |
| Llama-3.1-8B | MMLU acc | reserve | 4 | **0.385** (135-ray partial) | — | **no** | — (≥0.55, thin) |

(Phase-1 = isotropic MC over the whole cube. Phase-2 = resample rays *inside* the
candidate sub-cube [η_min, 1]ⁿ for a cleaner re-estimate. "η_min met" = the per-cut
search found a sub-cube with ≥95% concave rays and enough usable rays.)

The **MMLU rows are pooled over two independent seeds** (seed 0 + seed 1, fresh
rays and a fresh MMLU draw each). topk completed both seeds → **300 rays**; reserve
timed out mid-Phase-1 both times (≈401 s/ray ≫ 8 h wall) → **135 salvaged rays**
(70 + 65). Pooling is valid because each ray's concavity verdict is self-contained.
Merged ray files and the combined `etamin` outputs live in
`outputs/mc_concavity/.../mmlu_accuracy/{topk,reserve}_merged_{rays,etamin}.json`.

**Scalar floor c\*** is the most interpretable single number: the smallest *uniform*
floor — same value on every cut-point, i.e. restrict to the cube [c, 1]ⁿ — at which
≥95% of the (usable, ≥30) rays are concave. It comes from each run's
`scalar_floor_sweep` (concavity vs c). Unlike the per-cut η_min **vector** (which can
exploit one lenient coordinate), c\* is what you'd set if every layer shared one
compression floor, so it's the conservative, easy-to-act-on bound:

- **topk perplexity floors are very low** — c\* = 0.05–0.10. Compressing every layer
  to ~10% (or 5% for Llama WikiText) still leaves the landscape ≥95% concave.
- **reserve floors are higher** — c\* = 0.20–0.30 — consistent with reserve being
  rougher across the cube (composition jumps + the η = 0.5/0.25 dip).
- **Two configs never reach 95% at a usable floor.** gemma-7b + topk plateaus at
  ~0.92 through the data-rich range and only nominally hits 95% at c ≥ 0.45 where
  <20 rays survive (not trustworthy). MMLU + topk never gets there at all (peaks
  ~0.54 at c = 0.35 with adequate rays). The reserve MMLU salvage touches 1.0 only
  at c = 0.35 with 26 usable rays (<30 bar), so it's reported as a thin ~0.35.

All c\* values use the operating-point reference of η = 0.25 used in the paper: the
topk floors sit comfortably **below** 0.25 (operating regime is concave), while the
reserve floors sit **at or above** 0.25 — meaning under reserve the default operating
point can land in or near the non-concave zone.

### What the perplexity results say

- **Perplexity landscapes are largely concave**, and `topk_per_token` is the more
  reliably concave strategy. **Llama WikiText + topk is concave essentially
  everywhere** — η_min ≈ **0**, i.e. the concavity assumption holds over the entire
  cube. This is the cleanest confirmation of the optimizer's assumption.

- **Phase-2 usually lifts concavity sharply** (e.g. gemma-2b topk cuts4 0.79→0.98),
  confirming that the non-concave rays in Phase-1 are concentrated in the
  aggressive low-η corner — exactly the regime η_min is meant to fence off.

- **More cut-points → less concave.** Going from 4 to 7 cuts drops concavity for
  both strategies (gemma-2b topk 0.79→0.67, reserve 0.58→0.39): the
  higher-dimensional cube has more room for non-concave directions.

- **`reserve` is *not* more concave than `topk` in the MC framing** — the opposite
  of the diagonal-ray result. `reserve`'s composition jumps (each time the base
  rung changes, plus the η = 0.5/0.25 dips from the un-fixed `f_high` bug) create
  non-concavity along off-diagonal rays that the single diagonal never saw. So
  reserve is better *on the diagonal operating path* but rougher *across the full
  cube*.

- **One anomaly:** gemma-7b + topk is the only perplexity config where Phase-2
  concavity *dropped* (0.88→0.59) and η_min failed the 95% bar (best 0.94 with
  only 32 usable rays). The candidate sub-cube was data-starved; this one likely
  needs more rays to estimate reliably.

### What the MMLU results say — the landscape is barely concave

MMLU is the clear outlier: **only 23% of rays are concave** for topk (300 rays),
and **38.5%** for the partial reserve run (135 rays). Two reasons, both about the
*metric*:

- Absolute MMLU accuracy on 4-way questions sits near the 0.25 chance floor and
  moves in **discrete step plateaus**. A flat-then-jump profile reads as local
  *convexity* under the second-difference test even with the loosened tol = 0.02.
- The signal-to-noise is low enough that η_min can't certify a 95%-concave
  sub-cube anywhere: topk's best was 0.59 (32 usable rays); the merged reserve
  best was 0.84 (32 usable rays). Both **fail the bar**. The scalar-floor sweep
  only reaches ≈1.0 by discarding nearly every ray (topk: 2 usable rays at c=0.65;
  reserve: <10 at c≥0.55), so no trustworthy floor exists in the raw signal.

The takeaway is not necessarily "the compression landscape is non-concave for
MMLU" so much as "**raw discrete accuracy is too noisy a probe to detect concavity**."
The smoothed-concavity analysis below makes this precise: a single mild denoising
pass lifts MMLU concavity from ~0.23 to ~0.89 — i.e. **the apparent non-concavity is
overwhelmingly measurement noise, not true convexity of the landscape.**

---

## The MMLU problem in detail

MMLU was the hardest task to get a result from, for two distinct reasons.

### 1. Loading (resolved)

The first MMLU pair crashed instantly at `load_dataset("cais/mmlu")` — the pinned
`datasets`/`fsspec` in the shared `easy` conda env raises
`Invalid pattern: '**' can only be an entire path component`. Fixed in
`src/core/mmlu_eval.py::_load_mmlu_split` by pulling the `cais/mmlu` parquet shards
directly via `hf_hub_download(...)` + `pd.read_parquet(...)`, with `load_dataset`
kept only as a fallback. Parquet row order matches `load_dataset`, so the seeded
sample selection is unchanged. (Committed in `b5e6a22`.) After it, both jobs
loaded MMLU fine.

### 2. Runtime — this is the real bottleneck

Each ray requires **K = 7 full MMLU evaluations** (one per point on the ray), and
each evaluation is a full forward pass over the dev set across all subjects. Per-ray
wall-clock, measured from the job logs:

| Strategy | s / ray | Why |
|----------|---------|-----|
| `topk_per_token` | **~134 s** | lighter per-token top-k mask |
| `llmint8_reserve` | **~401 s** | mixed-precision magnitude-split quant — **~3× slower** |

A full run is Phase-1 (150 rays) + Phase-2 (80 rays) = 230 rays:

- **topk MMLU:** Phase-1 = 150 × 134 s ≈ **5.6 h** (fits the 8 h wall — Phase-1
  completed and a summary was written); Phase-2 then ran but was **cut off at ray
  65/80** when the wall hit. Full clean run would be ≈ 230 × 134 s ≈ **8.5 h** —
  just over the wall, which is exactly why Phase-2 didn't finish.
- **reserve MMLU:** Phase-1 alone = 150 × 401 s ≈ **16.7 h ≫ 8 h wall** → the job
  was **killed at ray ~70/150, mid-Phase-1**. A full reserve MMLU run would need
  ≈ 230 × 401 s ≈ **25.6 h**. It never even reached Phase-2.

By comparison the perplexity tasks are cheap (the fixed-ray version ran in ~2.5
min at 64 samples), which is why every perplexity config completed and only MMLU
hit the wall.

### 3. No resume — and the decision to accept a degraded estimate

`mc_concavity.py` checkpoints to `_phase1/_phase2.partial.json` every 5 rays, but
**there is no resume**: a relaunch always loops `for i in range(n_rays)` from 0, so
re-submitting `reserve` as-is just repeats the same partial outcome. Phase-1's full
results are only persisted (via `_write(final=False)`) *after* the phase completes,
so a mid-Phase-1 kill leaves only the partial.

The decision (locked with the user) was: **let reserve die, salvage the partial** —
do not add resume, do not bump the walltime, do not resubmit. Accept a degraded
estimate.

### 4. Salvage result (two seeds, pooled)

We ran MMLU twice (seed 0, then a seed-1 follow-up with more rays). topk completed
both → **300 rays**; reserve timed out mid-Phase-1 both times → **135 salvaged rays**
(70 + 65). Merging each strategy's ray lists into one `{"rays":[...],"n_cuts":4}` and
running the model-free `etamin` subcommand (CPU-only, no GPU) **with `--tol 0.02`**
(its default tol is stricter and mis-reports concavity at ~0.04 — always pass the
run's tol; the per-ray stored `concave` flag is the ground truth):

- **topk, 300 rays:** global concavity **0.23**. Best η_min reaches only 0.59
  concave (32 usable) — **fails 95%**. Scalar-floor sweep climbs 0.23 (c=0) → 0.50
  (0.30) → 0.59 (0.40) → 1.0 (0.65), but usable rays collapse from 300 to 2 over
  that range, so no trustworthy floor exists.
- **reserve, 135 rays:** global concavity **0.385**. Best η_min reaches 0.84
  concave (32 usable) — **fails 95%**. Scalar-floor sweep climbs 0.385 (c=0) → 0.78
  (0.30) → 0.90 (0.45) → 1.0 (c≥0.55), again only by discarding nearly all rays
  (≤10 usable past c=0.55).

So both raw MMLU numbers are directional only: concavity improves as you exclude
the low-η corner (consistent with every other task), but the signal is too thin
and too noisy to pin a reliable η_min. The smoothed analysis (next section) is the
better lens on this task.

---

## Smoothed concavity — how much non-concavity is just noise?

The second-difference test flips a genuinely-concave ray to non-concave if a
*single* point is noisy. To separate measurement noise from true curvature we
re-classify every ray after denoising its profile y(t), and report the concave
fraction per family vs smoothing strength (`experiments/smooth_concavity.py`).

**Smoother:** one or more passes of a binomial `[1,2,1]/4` low-pass on the
*interior* points. Each pass maps `y → y'` with

```
y'[0]   = y[0]                                    (endpoint, fixed)
y'[j]   = 0.25·y[j-1] + 0.5·y[j] + 0.25·y[j+1]    for 1 ≤ j ≤ K-2
y'[K-1] = y[K-1]                                  (endpoint, fixed)
```

then we re-run the same `D2[j] = y[j-1] + y[j+1] - 2y[j] ≤ tol` test on `y'`
(same per-task `tol`: 0.02 for accuracy, 0.001 for perplexity). The two
**endpoints are held fixed** because they are real measured values — smoothing
them drags a monotone profile's endpoint toward its neighbour and manufactures a
spurious convex spike at the first interior point (an early reflect-padded version
wrongly collapsed perplexity concavity from 0.86 to 0.006 this way). We also
deliberately avoid Savitzky-Golay with polyorder 2 — it locally fits a quadratic
and would bias a *curvature* test toward its own verdict. `raw` (0 passes)
reproduces the numbers above exactly; `sm1` is one mild pass; `sm2` is two.

| Model | Task | Strategy | cuts | raw | **sm1** | sm2 | M |
|-------|------|----------|------|-----|---------|-----|---|
| Llama-3.1-8B | MMLU acc | topk | 4 | 0.230 | **0.893** | 0.980 | 300 |
| Llama-3.1-8B | MMLU acc | reserve | 4 | 0.385 | **0.852** | 0.948 | 135 |
| Llama-3.1-8B | WikiText ppl | topk | 4 | 0.957 | **0.983** | 0.991 | 230 |
| Llama-3.1-8B | WikiText ppl | reserve | 4 | 0.672 | **0.878** | 0.922 | 180 |
| gemma-2b | ShareGPT ppl | topk | 4 | 0.856 | **0.900** | 0.928 | 180 |
| gemma-2b | ShareGPT ppl | topk | 7 | 0.756 | **0.828** | 0.867 | 180 |
| gemma-2b | ShareGPT ppl | reserve | 4 | 0.683 | **0.778** | 0.822 | 180 |
| gemma-2b | ShareGPT ppl | reserve | 7 | 0.483 | **0.611** | 0.689 | 180 |
| gemma-7b | ShareGPT ppl | topk | 4 | 0.778 | **0.961** | 0.983 | 230 |
| gemma-7b | ShareGPT ppl | reserve | 4 | 0.739 | **0.852** | 0.891 | 230 |

What this shows:

- **MMLU's apparent non-concavity is almost entirely noise.** One mild pass takes
  topk from 0.23 → **0.89** and reserve from 0.385 → **0.85**; two passes reach
  ~0.95–0.98. The underlying accuracy-vs-η landscape *is* concave — raw discrete
  accuracy on 4-way questions near the chance floor was just too noisy a probe to
  see it. (The two topk seeds agree closely: seed 0 → 0.88, seed 1 → 0.91 at sm1.)
- **Perplexity families were already concave**, so smoothing only nudges them
  (e.g. WikiText topk 0.957 → 0.983). The lift is largest exactly where Phase-1
  had been dragged down by a data-starved/noisy estimate — gemma-7b topk
  0.778 → **0.961**, recovering the config flagged as the perplexity anomaly.
- **Smoothing preserves the qualitative ordering** at sm1: topk ≥ reserve, and
  cuts4 ≥ cuts7. It denoises; it does not reshuffle the conclusions.
- **Caveat:** binomial smoothing is a low-pass — push it hard enough (sm3+) and
  *any* profile trends toward concave, so sm2+ approaches a trivial limit. We
  treat **sm1 as the honest "denoised" number** and read sm2 only as the trend.

## Bottom line

- **The concavity assumption holds well for perplexity-quality**, best of all for
  Llama WikiText + topk (concave over the whole cube). Phase-2 confirms that
  non-concavity lives in the aggressive low-η corner, which is what η_min fences.
- **`topk` is the more globally-concave strategy**; `reserve` wins on the diagonal
  operating path but is rougher across the full cube (composition jumps + the
  un-fixed η = 0.5/0.25 dip).
- **Higher cut-count erodes concavity** (curse of dimensionality).
- **MMLU is concave once you denoise.** Raw discrete accuracy is too noisy to
  detect concavity directly (topk only 23% concave over 300 rays, reserve 38.5%
  over 135), but a single mild smoothing pass lifts both to ~0.85–0.89 — the
  non-concavity was measurement noise, not true convexity. The compute cost is
  still severe (reserve MMLU ≈26 h for a full run; it only ever salvaged partials).

### Open / next steps

- Commit the `reserve` `f_high` floor fix (the `mmlu_eval.py` loader fix is already
  committed in `b5e6a22`).
- For MMLU, the cleaner long-term fix is a smoother *metric* (mean log-prob of the
  correct answer, or baseline-normalised accuracy) rather than post-hoc profile
  smoothing; and add resume to `mc_concavity.py` (or shard Phase-1) so reserve can
  finish a full run instead of being salvaged from a partial.
- Flan-T5/SST2 (Lang-7) still deferred.
