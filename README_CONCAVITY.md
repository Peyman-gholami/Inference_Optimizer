# Concavity Experiments

This document describes how to run, reproduce, and interpret the concavity experiments for the Inference Optimizer project.

---

## Overview

The concavity experiments test whether the task metric (perplexity or MMLU accuracy) is a **concave function** of the compression ratio vector **η ∈ [0,1]^n**. Concavity is a key theoretical property: if the metric is concave in η, then the naive independent-coordinate optimization over each cut point finds a global optimum (the per-cut search is guaranteed to be consistent).

Four scripts implement complementary views of concavity:

| Script | What it does | Model required? |
|---|---|---|
| `experiments/ray_concavity_score.py` | Fixed hand-designed rays (original test) | Yes |
| `experiments/mc_concavity.py` | Monte-Carlo random rays (isotropic, main experiment) | Yes |
| `experiments/jensen_concavity.py` | Jensen-inequality test on saved rays (post-hoc) | No |
| `experiments/smooth_concavity.py` | Same as MC but after low-pass smoothing (noise diagnosis) | No |

---

## What is η (eta)?

η is a **per-cut-point compression ratio**. `--n_cuts` cut points divide the model's decoder layers into segments; each cut point's η ∈ (0, 1] controls how aggressively activations are compressed at that boundary:

- **η = 1** → no compression (full activations pass through)
- **η → 0** → maximum compression

The full search space is the hypercube **[0,1]^n** where n = `n_cuts`.

The compression strategies (topk, llmint8_reserve, etc.) differ in *how* they compress given an η value, but η always plays the role of a scalar compression ratio for each cut.

---

## How we test Jensen's inequality

For each saved ray (a line segment in η-space), we sample triples of grid points **(i, j, k)** with t_i < t_j < t_k. Point j lies on the segment between i and k, so it is a convex combination:

```
t_j = λ · t_i + (1 - λ) · t_k,   λ = (t_k - t_j) / (t_k - t_i)
```

Jensen's inequality for a **concave** function requires:

```
f(j)  ≥  λ · f(i) + (1 - λ) · f(k)
```

i.e., the middle point's metric value must sit on or above the chord connecting i and k. The **Jensen concavity** is the fraction of sampled triples that pass this check.

A chord tolerance `--tol` (default **0.02 for accuracy**, **0.001 for perplexity**) absorbs measurement noise without masking true concavity violations.

---

## Variance representation in the Jensen plots

The shaded bands in `jensen_concavity.py` plots represent **±1 binomial standard error**:

```
SE = sqrt(p · (1 - p) / N)
```

where `p` is the concavity fraction and `N` is the total number of Jensen triples sampled for that floor level. Wider bands mean fewer samples and less reliable estimates.

**"x" markers** on a curve indicate floor levels with very few usable rays (< 8): the estimate is unreliable there.

---

## Running `mc_concavity.py` (main experiment)

### Smoke test (fast, verifies the pipeline)

```bash
python experiments/mc_concavity.py run \
    --model meta-llama/Llama-3.1-8B \
    --dataset wikitext --metric perplexity \
    --strategy topk_per_token \
    --n_cuts 4 --n_rays 20 --n_points 7 \
    --max_texts 16 \
    --min_box_samples 0 \
    --out_dir outputs/mc_concavity/smoke
```

### Full MMLU run (as used in the paper)

```bash
python experiments/mc_concavity.py run \
    --model meta-llama/Llama-3.1-8B \
    --dataset mmlu --metric accuracy \
    --strategy topk_per_token \
    --n_cuts 4 --n_rays 150 --n_points 9 \
    --samples_per_subject 20 --n_shot 5 \
    --tol 0.02 --target 0.95 --min_usable 30 \
    --min_box_samples 30 \
    --out_dir outputs/mc_concavity/llama_mmlu_topk
```

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--n_rays` | 150 | Phase-1 random rays sampled over the full hypercube [ε,1]^n |
| `--n_points` | 7 | Grid points per ray (more = better D2 test, slower per ray) |
| `--phase2_rays` | 80 | Targeted rays inside the eta_min candidate sub-cube (0 to skip) |
| `--min_box_samples` | 30 | Min usable rays per floor level; triggers adaptive fill if fewer (0 to disable) |
| `--adaptive_floor_step` | 0.1 | Floor-level grid spacing for the adaptive fill check |
| `--tol` | 0.001 | Second-difference (D2) tolerance; use 0.02 for accuracy tasks |
| `--target` | 0.95 | Target concavity fraction for the eta_min search |
| `--min_usable` | 30 | Min rays required in a sub-cube to declare eta_min trustworthy |
| `--n_cuts` | 4 | Number of cut points (compression boundaries) |
| `--max_texts` | 128 | Texts/samples for perplexity evaluation |
| `--samples_per_subject` | 20 | MMLU questions per subject |
| `--n_shot` | 5 | Few-shot examples for MMLU |
| `--eps` | 0.001 | Lower clamp on η to avoid η ≤ 0 |

### Phases

1. **Phase 1** — `n_rays` isotropic random rays over `[ε, 1]^n`. Evaluates metric at `n_points` equally-spaced points per ray. Runs `find_eta_min` on the result.
2. **Phase 2** — `phase2_rays` targeted rays sampled from `[eta_min_candidate, 1]^n`. Refines `eta_min` using the combined Phase-1 + Phase-2 ray set.
3. **Adaptive fill** — ensures every floor level has `min_box_samples` usable rays (see below). Runs after Phase 2.

### Output files

In `--out_dir`:

| File | Contents |
|---|---|
| `*_rays.json` | All ray profiles: `rays` (Phase 1), `phase2_rays`, `fill_rays`, plus metadata |
| `*_summary.json` | Concavity statistics + `eta_min` result |
| `*_rays.png` | Quick visual of ray profiles (green=concave, red=non-concave) |
| `*.partial.json` | Checkpoint files (safe to delete after the run completes) |

### Strategies available

`topk_per_token`, `llmint8_reserve`, `llmint8_adjacent`, `llmint8`, `magnitude`, `random`, `quantization`

---

## Adaptive resampling (`--min_box_samples`)

After Phase 1 and Phase 2, the **adaptive fill phase** ensures that each floor level `c` in `[0, 0.9]` has at least `min_box_samples` usable ray-segments (i.e., segments whose clipped portion inside `[c,1]^n` has ≥ 3 grid points).

The algorithm scans floor levels from **high to low**. For the highest under-sampled floor `c`, it samples new rays uniformly from `[c, 1]^n` and evaluates the metric on them. These fill rays automatically count toward all lower floors too (because a ray in `[c, 1]^n` is also in `[c', 1]^n` for c' ≤ c). The loop continues until all floors are adequately sampled.

Fill rays are saved under key `"fill_rays"` in `*_rays.json` and are automatically included by `jensen_concavity.py` and `etamin`.

Set `--min_box_samples 0` to disable adaptive fill entirely.

---

## Running `jensen_concavity.py` (post-hoc, model-free)

```bash
python experiments/jensen_concavity.py \
    --root outputs/mc_concavity \
    --out_dir outputs/mc_concavity/jensen \
    --samples_per_ray 200 \
    --floor_min 0.0 --floor_max 0.9 --floor_step 0.05 \
    --seed 0
```

This script:
1. Scans all `*_rays.json` and `*.partial.json` files under `--root`
2. Groups rays by `(model, dataset, metric, n_cuts, strategy)`, de-duplicated by content hash (so multiple seeds, phases, and partial checkpoints merge cleanly without double-counting)
3. Runs the Jensen test at each floor level for each group
4. Writes one plot per `(model, dataset, metric, n_cuts)` group, with one curve per strategy, plus `jensen_concavity.json`

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--samples_per_ray` | 200 | Jensen triples sampled per ray per floor (capped at C(m, 3)) |
| `--floor_step` | 0.05 | Resolution of the eta-floor sweep (smaller = finer x-axis) |
| `--tol` | auto | Chord tolerance (0.02 accuracy, 0.001 perplexity, set by metric) |
| `--seed` | 0 | RNG seed for triple sampling |

### Output

| File | Contents |
|---|---|
| `*__jensen_floor.png` | Jensen concavity vs eta-floor, one curve per strategy, ±1 SE shaded |
| `jensen_concavity.json` | Full numerical results (sweep data per strategy) |

---

## Running `smooth_concavity.py` (noise diagnosis)

```bash
python experiments/smooth_concavity.py --root outputs/mc_concavity
```

Re-classifies each ray after applying 0/1/2 passes of a [1,2,1]/4 binomial smoother to the profile `y(t)`. "raw" (0 passes) reproduces the exact concavity fraction from `mc_concavity.py`.

**Interpretation:** if concavity jumps substantially after 1 smoothing pass, the measured non-concavity is **noise-driven** rather than structural.

---

## Running `ray_concavity_score.py` (original fixed-ray test)

```bash
python experiments/ray_concavity_score.py \
    --model meta-llama/Llama-3.1-8B \
    --ray_file experiments/rays_interesting_5.json \
    --strategy topk_per_token --n_cuts 4 \
    --max_texts 64 \
    --out_dir outputs/ray_concavity/llama
```

Uses hand-designed rays from `experiments/rays_interesting*.json` (5-cut and 8-cut variants). This was the original concavity test; `mc_concavity.py` supersedes it with an isotropic random-ray approach.

---

## Launching on Slurm

```bash
# Full MC run (topk + llmint8_reserve, wikitext/ShareGPT):
sbatch scripts/run_mc_concavity.sh

# MMLU multi-seed runs (topk + reserve on Llama):
sbatch scripts/run_mc_mmlu_more.sh

# Densification: more points per ray on existing checkpoints:
sbatch scripts/run_mc_more_points.sh
```

Typical SLURM walltime: **8 hours** for a 150-ray MMLU run with `n_points=9`.  
`llmint8_reserve` is approximately 3× slower than `topk_per_token`.

---

## Post-hoc eta_min re-search

Re-run the eta_min search on an existing `*_rays.json` without reloading the model:

```bash
python experiments/mc_concavity.py etamin \
    --rays_json outputs/mc_concavity/llama_mmlu_topk/..._rays.json \
    --tol 0.02 --target 0.95 --min_usable 30 \
    --out outputs/mc_concavity/llama_mmlu_topk/etamin_rescan.json
```

Useful for experimenting with different `tol`/`target` values without re-evaluating the model.

---

## Output directory structure

```
outputs/mc_concavity/
├── <model_slug>__<dataset>__<metric>__<strategy>__cuts<N>/
│   ├── *_rays.json          ← ray profiles (Phase 1 + Phase 2 + fill)
│   ├── *_summary.json       ← concavity stats + eta_min result
│   ├── *_rays.png           ← quick visual (green=concave, red=non-concave)
│   ├── *.partial.json       ← checkpoints (safe to delete after completion)
│   └── *_fill.partial.json  ← adaptive fill checkpoint
└── jensen/
    ├── *__jensen_floor.png  ← Jensen concavity vs eta-floor (±1 SE shaded)
    └── jensen_concavity.json
```

---

## Number of samples: practical guidance

| Metric | Recommended `n_rays` | Recommended `n_points` | `tol` | Notes |
|---|---|---|---|---|
| Perplexity (WikiText/ShareGPT) | 150 | 7–9 | 0.001 | Smooth signal, tight tolerance works |
| Accuracy (MMLU) | 150–200 | 9 | 0.02 | Discrete/noisy; use looser tolerance |

For the Jensen analysis, `--samples_per_ray 200` with `--floor_step 0.05` gives a good resolution/noise tradeoff. Use `--min_box_samples 30` to ensure reliable estimates at each floor level (especially the high-floor regime where few rays naturally land).
