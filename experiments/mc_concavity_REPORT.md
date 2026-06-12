# Monte-Carlo Concavity of the Compression–Quality Landscape

## Motivation

The optimizer treats per-layer activation-compression ratios as a vector
**η ∈ [0,1]ⁿ** (one entry per cut-point; η = 1 means uncompressed). It assumes
the task-quality function *f(η)* is reasonably **concave** so that gradient /
chord reasoning is well-behaved. We previously probed this with a handful of
hand-designed rays (`experiments/ray_concavity_score.py`): informative, but a
biased, low-coverage sample of the hypercube.

This experiment **reframes concavity as a Monte-Carlo property of the whole
hypercube** and asks two questions:

1. **How concave is the landscape?** — estimated as the **fraction of randomly
   sampled rays that are concave**.
2. **Where does concavity break?** — we expect very low η to enter a
   "weird-behaviour" regime (e.g. unprotected aggressive quantization collapsing
   the forward pass). So we search for the per-cut-point lower bound **η_min**
   such that, restricted to the sub-cube **[η_min, 1]ⁿ**, concavity stays
   **≥ 95 %**.

## Method

For each *(model, dataset, metric, strategy, n_cuts)* configuration:

1. **Sample M random rays.** A ray is a line segment between two independent
   uniform endpoints in the box `[ε, 1]ⁿ` (ε = 1e-3). The old `end ≥ start`
   monotonicity constraint is **dropped** so segments are isotropic — an
   unbiased probe of the cube. Near-degenerate segments are rejected.
2. **Evaluate the metric** at K equally spaced points `η(t) = (1−t)·start +
   t·end`, t ∈ [0,1].
3. **Classify each ray** concave / non-concave with the discrete
   second-difference test `D2[j] = y[j−1] + y[j+1] − 2 y[j] ≤ tol` (a positive
   D2 at an interior point = local convexity).
4. **Report concavity** = (# concave rays) / M, and **save every ray's full
   profile** (sampled η-vectors + metric values) to JSON.

### Per-task metric

| Metric | Tasks | Value along a ray |
|--------|-------|-------------------|
| Perplexity | ShareGPT, WikiText | mean per-sample `min{1, PPL_ref / PPL_comp}` (1 = matches uncompressed baseline) |
| Accuracy | MMLU | **absolute** `accuracy(η)` in [0,1] (raw, not normalised) |

Accuracy is discrete and noisy (step plateaus read as spurious convexity), so
accuracy tasks use a larger `tol` (0.02 vs 0.001) and more samples.

### Task matrix (causal-LM; Flan-T5/SST2 deferred)

| Model | Dataset | Metric | n_cuts |
|-------|---------|--------|--------|
| `google/gemma-2b` (cached base) | ShareGPT | perplexity | 4, **and 7** |
| `google/gemma-7b` (cached base) | ShareGPT | perplexity | 4 |
| `meta-llama/Llama-3.1-8B` | MMLU | accuracy | 4 |
| `meta-llama/Llama-3.1-8B` | WikiText | perplexity | 4 |

Strategies (both, per task): **`topk_per_token`** and **`llmint8_reserve`**
(the FP16-outlier-protected mixed-precision scheme that won the prior comparison;
pure-torch, no GPU-compressor backend needed).

## Per-cut-point η_min search

Model-free post-processing of the saved profiles. For a candidate corner `v`,
restrict each ray to the segment lying inside `[v, 1]ⁿ` (a single convex
sub-interval of t, computed analytically per coordinate). Because the saved
t-grid is uniform, the in-cube points form a contiguous, uniformly spaced block,
so the second-difference test stays valid on the clipped slice. A ray is
*usable* at `v` if ≥ 3 of its saved points survive; the concavity fraction at `v`
is over usable rays.

Concavity **rises** as `v` rises (excluding the weird low-η regime) but usable
ray count **falls**, so the search **ascends from v = 0** (data-rich) rather than
descending from v = 1 (data-starved):

- **(A) coarse greedy ascent** from 0 — raise the coordinate that most improves
  the concavity fraction until ≥ 95 % with ≥ `min_usable` rays;
- **(B) fine descent** — peel back overshoot, lowering coordinates while the bar
  still holds — landing on a Pareto-minimal `v`.

Two phases improve reliability near the boundary:

- **Phase 1**: isotropic MC over the whole cube → global concavity + a coarse
  η_min candidate.
- **Phase 2**: resample rays with **both endpoints inside the candidate
  sub-cube** `[η_min, 1]ⁿ`, so rays live entirely in the region (no clipping
  loss, full K points) → a clean, high-power re-estimate.

The output also reports a **uniform scalar-floor sweep** (concavity vs
`v = c·1`) — the most directly interpretable "concavity vs how-much-low-η-we-
exclude" curve, a companion to the per-cut vector.

## Running

```bash
# Full task matrix (one SLURM job per model × dataset × strategy):
bash scripts/run_mc_concavity.sh

# Single config locally:
python experiments/mc_concavity.py run \
    --model meta-llama/Llama-3.1-8B --dataset wikitext --metric perplexity \
    --strategy topk_per_token --n_cuts 4 \
    --n_rays 150 --n_points 7 --phase2_rays 80 \
    --out_dir outputs/mc_concavity/llama_wikitext

# Re-run the η_min search later, model-free, on a saved profile:
python experiments/mc_concavity.py etamin \
    --rays_json outputs/.../<stem>_rays.json --target 0.95
```

Runs in the `easy` conda env. ShareGPT/WikiText load via direct
`hf_hub_download` + pandas (the pinned `datasets` in `easy` can't load them the
normal way).

## Outputs

Under `outputs/mc_concavity/<model>/<dataset>_<metric>/<strategy>/cuts<N>/`:

- `*_rays.json` — every ray's `start`, `end`, `t[K]`, `eta[K][n]`, `y[K]`,
  `concave`, `worst_d2`; plus `phase2_rays`. Drives the model-free η_min search.
- `*_summary.json` — global Phase-1/Phase-2 concavity fractions, baseline, and
  `eta_min_phase1` / `eta_min_phase2` (the per-cut vector, achieved fraction,
  usable count, search history, scalar-floor sweep).
- `*_rays.png` — ray curves (green = concave, red = non-concave).

## Interpreting results

- **Global concavity fraction** — how concave the raw landscape is; lower means
  the optimizer's concavity assumption is shakier over the full cube.
- **η_min vector** — the per-cut compression floor below which concavity breaks;
  a useful safety bound for the optimizer's search domain. Compare against the
  task operating points (η = 0.25 for most tasks, 0.125 for SST2 in the paper):
  if η_min sits *above* the operating point, the operating regime is in the
  non-concave zone.
- **`topk_per_token` vs `llmint8_reserve`** — expect `reserve` to stay concave
  to more aggressive (lower) η thanks to FP16 outlier protection, i.e. a lower
  η_min.

## Scope / deferred

Flan-T5-Base / SST2 (Lang-7) is deferred: it needs encoder-decoder cut-point
detection (current `detect_llm_decoder_layers` is causal-LM only) and a new SST2
loader/accuracy evaluator. The framework here is metric-agnostic, so adding it
later is an evaluator + cut-point-detection extension, not a redesign.
