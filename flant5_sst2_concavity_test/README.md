# Flan-T5 SST-2 Concavity Test

Standalone experiment for testing whether Flan-T5/SST-2 quality metrics are approximately concave with respect to a 3-dimensional compression vector `eta`.

## Quick Structure

- `flan_t5_sst2_concavity.py`: main experiment script. Samples rays, evaluates metrics, tests concavity, saves JSON summaries and plots.
- `flan_t5_sst2_runtime.py`: minimal Flan-T5/SST-2 runtime. Loads the model/dataset, applies the fixed encoder-decoder split, and runs partitioned inference.
- `compressors.py`: local activation compression methods.
- `requirements.txt`: Python dependencies.

The folder is self-contained and does not import files from the original project.

## Model And Data

- Model: `google/flan-t5-base`
- Dataset: Hugging Face `glue/sst2`
- Default split: `validation`

They are downloaded from Hugging Face on first run unless already cached locally.

## Quick Run

```powershell
conda activate torch270_py312_cuda
cd C:\Users\ninmi\Desktop\MDI\flant5_sst2_concavity_test
python flan_t5_sst2_concavity.py
```

Smoke test:

```powershell
python flan_t5_sst2_concavity.py --max_samples 1 --n_rays 1 --phase2_rays 1 --n_points 3 --min_usable 1
```

## Outputs

Results are saved to `flant5_sst2_concavity_result/` by default:

- `*_rays.json`
- `*_summary.json`
- `*_rays.png`
- `*_phase2_rays.png`

Main metrics: `accuracy`, `neg_ce`, and baseline-normalized `ppl_score`.

# Detailed Manual

## Purpose

This experiment checks whether the task quality function `A(eta)` is approximately concave in the compression vector `eta`.

Here, `eta` has 3 dimensions because the Flan-T5 model is split into 3 transfer points:

- `tp0`: encoder block group 0 to encoder block group 1
- `tp1`: encoder block group 1 to encoder block group 2
- `tp2`: encoder output to decoder

Each entry of `eta` controls the compression strength at one transfer point. Larger `eta` means less aggressive compression. `eta = [1, 1, 1]` is the uncompressed baseline.

The experiment samples random line segments, called rays, in this 3D `eta` space. Along each ray, it evaluates the model at several equally spaced points and checks whether the measured metric curve satisfies a discrete concavity test.

## Architecture

The project is intentionally small:

```text
flant5_sst2_concavity_test/
  flan_t5_sst2_concavity.py
  flan_t5_sst2_runtime.py
  compressors.py
  requirements.txt
  README.md
```

`flan_t5_sst2_concavity.py` controls the experiment:

- parses command-line arguments
- samples phase-1 rays over `[eps, 1]^3`
- evaluates `accuracy`, `neg_ce`, and `ppl_score`
- runs the second-difference concavity test
- estimates `eta_min`
- samples phase-2 rays inside `[eta_min, 1]^3`
- writes JSON summaries and plots

`flan_t5_sst2_runtime.py` controls the Flan-T5/SST-2 inference path:

- loads `google/flan-t5-base`
- loads SST-2 validation samples
- builds the sentiment-classification prompt
- resolves the single-token verbalizers `positive` and `negative`
- splits Flan-T5 into fixed encoder/decoder partitions
- applies activation compression at the 3 transfer points
- returns logits for the first decoder step

`compressors.py` provides the local compression methods:

- `topk`
- `quantization`
- `llmint8`
- `outliertopk`

The default experiment uses `topk`.

## Model Split

The fixed split matches the original Flan-T5/SST-2 estimator experiment:

```text
Encoder layers 0-3   -> Enc0
Encoder layers 4-7   -> Enc1
Encoder layers 8-11  -> Enc2
Decoder layers 0-11  -> Decoder
```

The 3 compression points are:

```text
eta[0] -> tp0: Enc0 -> Enc1
eta[1] -> tp1: Enc1 -> Enc2
eta[2] -> tp2: Enc2 -> Decoder
```

## Metrics

All metrics are treated as quality functions where larger is better.

`accuracy`:

Mean SST-2 classification accuracy. The prediction is made by comparing the logits of the `positive` and `negative` verbalizer tokens.

`neg_ce`:

Negative cross entropy:

```text
neg_ce = -CE
```

The CE is computed using the two verbalizer logits `[negative_logit, positive_logit]` and the true SST-2 label. `neg_ce` is used instead of `CE` because the optimizer assumes a larger-is-better quality function.

`ppl_score`:

Baseline-normalized perplexity-style score:

```text
ppl_score_i = min(1, exp(CE_ref_i - CE_comp_i))
ppl_score = mean_i ppl_score_i
```

where `CE_ref_i` is the per-sample CE at `eta = [1, 1, 1]`, and `CE_comp_i` is the per-sample CE at the compressed `eta`.

Interpretation:

- `ppl_score = 1`: compressed inference matches or beats the baseline CE.
- lower `ppl_score`: compressed inference is worse than the baseline.

## Concavity Test

For one ray, the script samples points:

```text
eta(t) = (1 - t) * start + t * end
t in [0, 1]
```

For example, if `n_points = 7`, the script evaluates 7 equally spaced `t` values:

```text
0.000, 0.167, 0.333, 0.500, 0.667, 0.833, 1.000
```

At each point it computes the selected metric value `y(t)`.

The discrete concavity condition is:

```text
D2[j] = y[j-1] + y[j+1] - 2*y[j] <= tol
```

If any interior point has `D2[j] > tol`, that ray is marked non-concave.

The reported concavity score is:

```text
concavity score = number of concave rays / number of evaluated rays
```

## Phase 1 And Phase 2

Phase 1 samples random rays over the full cube:

```text
[eps, 1]^3
```

Then it estimates `eta_min`, a lower-bound vector such that the sub-cube:

```text
[eta_min, 1]^3
```

has high enough concavity fraction.

Phase 2 samples extra rays only inside this estimated sub-cube. This checks whether concavity becomes more reliable after excluding very low-compression-quality regions.

## Default Parameters

The default configuration is stored near the top of `flan_t5_sst2_concavity.py`.

```text
model_name = google/flan-t5-base
dataset = glue/sst2
split = validation
device = cuda
max_samples = 0
compressor_name = topk
metrics = all
n_rays = 160
phase2_rays = 80
n_points = 7
eps = 0.001
min_seg = 0.05
target = 0.95
min_usable = 20
seed = 0
```

`max_samples = 0` means the full SST-2 validation split is used.

Default tolerances:

```text
accuracy tol = 0.01
neg_ce tol = 0.001
ppl_score tol = 0.001
```

## Command-Line Examples

Run the default full experiment:

```powershell
python flan_t5_sst2_concavity.py
```

Run a small smoke test:

```powershell
python flan_t5_sst2_concavity.py --max_samples 1 --n_rays 1 --phase2_rays 1 --n_points 3 --min_usable 1
```

Run a medium test:

```powershell
python flan_t5_sst2_concavity.py --max_samples 100 --n_rays 40 --phase2_rays 20 --n_points 7
```

Run only one metric:

```powershell
python flan_t5_sst2_concavity.py --metrics ppl_score
```

Use CPU:

```powershell
python flan_t5_sst2_concavity.py --device cpu
```

Use a local SST-2 file:

```powershell
python flan_t5_sst2_concavity.py --dataset_path path\to\sst2.jsonl
```

Supported local dataset formats:

- `.json`
- `.jsonl`
- `.csv`
- `.tsv`

Each record should contain a sentence field and a binary label:

```json
{"sentence": "a charming and funny movie", "label": 1}
```

## Output Files

For each metric, the script writes:

```text
google__flan-t5-base__sst2__<metric>__topk__cuts3_rays.json
google__flan-t5-base__sst2__<metric>__topk__cuts3_summary.json
google__flan-t5-base__sst2__<metric>__topk__cuts3_rays.png
google__flan-t5-base__sst2__<metric>__topk__cuts3_phase2_rays.png
google__flan-t5-base__sst2__<metric>__topk__cuts3_phase1.partial.json
google__flan-t5-base__sst2__<metric>__topk__cuts3_phase2.partial.json
```

`*_rays.json` contains:

- phase-1 ray profiles
- phase-2 ray profiles
- sampled `eta` points
- metric values along each ray
- concavity labels
- worst second-difference values

`*_summary.json` contains:

- baseline metric value
- phase-1 concavity score
- phase-2 concavity score
- estimated `eta_min`
- scalar floor sweep
- experiment parameters

`*_rays.png` shows phase-1 metric curves.

`*_phase2_rays.png` shows phase-2 metric curves.

Green lines are concave rays. Red lines are non-concave rays.

## Recomputing Eta-Min

You can recompute `eta_min` from a saved ray JSON without rerunning model inference:

```powershell
python flan_t5_sst2_concavity.py etamin --rays_json flant5_sst2_concavity_result\google__flan-t5-base__sst2__ppl_score__topk__cuts3_rays.json
```

This is useful when changing only:

- `--target`
- `--min_usable`
- `--tol`

Example:

```powershell
python flan_t5_sst2_concavity.py etamin --rays_json path\to\rays.json --target 0.90 --min_usable 10
```

## Offline Notes

The model and dataset are not stored in this folder. By default, they are loaded from Hugging Face:

```text
google/flan-t5-base
glue/sst2
```

After the first download, they are usually cached under:

```text
C:\Users\<username>\.cache\huggingface\
```

If you want to send the experiment to another machine without internet access, also provide either:

- the Hugging Face cache, or
- a local model directory and local SST-2 file

Then run with:

```powershell
python flan_t5_sst2_concavity.py --model_name path\to\local_model --dataset_path path\to\sst2.jsonl
```

## Practical Runtime Notes

Runtime scales roughly with:

```text
number of samples * number of rays * number of points * number of metrics
```

Phase 1 evaluates all selected metrics together for each ray point.

Phase 2 is run per metric, so using all three metrics makes phase 2 more expensive.

For quick validation, use the smoke test first. For actual evidence, use the default full setting or increase `n_rays`.

## Common Warnings

You may see a Transformers warning about tied weights:

```text
The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight...
```

This warning is harmless for this experiment.

You may also see a Hugging Face token warning if you are not logged in. It only affects download rate limits, not the experiment logic.
