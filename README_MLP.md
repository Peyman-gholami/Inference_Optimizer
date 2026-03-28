# MLP / MNIST Backend (default)

This is the **default backend** — no environment variable needed to activate it.
A small 3-layer MLP is trained on MNIST for 3 epochs at startup, then used as
the accuracy and gradient oracle throughout the simulation.

## Quick start

```bash
# Default — MLP/MNIST, TopK compression
python main.py

# Single-task baseline comparison (T=150 time slots)
python compare_to_baselines.py

# Multi-task baseline comparison (K=2 tasks, T=150 time slots)
python compare_to_baselines_multi.py

# Parameter sweep over sigma and N
python run_param_sweep.py
```

No checkpoint file, no dataset download configuration, and no extra
dependencies are required beyond the base `requirements.txt`.
MNIST is downloaded automatically to `./data/` on first run.

## Compressor selection

The MLP backend uses `CompressedModel`, which applies activation compression
via forward hooks at the two ReLU cut-points inside the MLP.
Three compressors are available, selected by environment variable:

```bash
export COMPRESSOR_TYPE=topk           # Top-K sparsification (default)
export COMPRESSOR_TYPE=quantization   # Uniform quantization (FP16/INT8/INT4/INT2)
export COMPRESSOR_TYPE=llmint8        # LLM.int8() hybrid mixed-precision
python main.py
```

When `compressors.py` is not present, the backend silently falls back to a
built-in top-K implementation that replicates the original behaviour exactly.

## Topology

The MLP has **2 ReLU activation layers**, which become the 2 transfer links
of a **3-node chain** topology:

```
Node 0  [Flatten + Linear(784→128) + ReLU]  ──Link 0──►
Node 1  [Linear(128→64) + ReLU]             ──Link 1──►
Node 2  [Linear(64→10)]
```

Each link carries the compressed activation tensor between nodes.
`eta[i]` is the compression ratio on link `i` (1.0 = no compression).

Default toy payload sizes and compute times used by the delay model:

| Parameter | Value |
|-----------|-------|
| `tau[0,1,2]` | 0.01 s (per node) |
| `a[0]` | 10.0 (arbitrary units, link 0) |
| `a[1]` | 20.0 (arbitrary units, link 1) |
| `eta_min[0]` | 0.1 |
| `eta_min[1]` | 0.05 |
| `R_k` | 10.0 packets/s (max delay = 0.1 s) |

## Parameter sweep (`run_param_sweep.py`)

Sweeps the Stein oracle parameters that control gradient quality:

**Experiment 1 — sigma sweep** (`N=50` fixed):

| sigma | 0.001 | 0.01 | 0.05 | 0.1 | 0.5 |
|-------|-------|------|------|-----|-----|

**Experiment 2 — N sweep** (`sigma=0.05` fixed):

| N | 10 | 25 | 50 | 75 | 100 |
|---|----|----|----|----|-----|

Each combination produces 4 per-run plots and 4 summary diagrams.

```bash
# Both experiments (default)
python run_param_sweep.py --baseline_algorithms none

# Sigma sweep only
python run_param_sweep.py --experiments sigma --baseline_algorithms none

# N sweep only
python run_param_sweep.py --experiments N --baseline_algorithms none

# Include all baselines (slower)
python run_param_sweep.py --experiments sigma,N

# Custom horizon and output directory
python run_param_sweep.py --T 200 --seed 42 --out_dir outputs/my_sweep
```

## Environment variables

| Variable | Role | Default |
|----------|------|---------|
| `COMPRESSOR_TYPE` | topk / quantization / llmint8 | `topk` |
| `ETA_MIN0` | η floor on link 0 | `0.1` |
| `ETA_MIN1` | η floor on link 1 | `0.05` |
| `R_K` | Target throughput R_k(t) | `10.0` |

## MLP architecture

Defined in `src/core/stein_oracles.py`:

```
Input: 28×28 MNIST image  (1 channel, flattened to 784)
  Flatten
  Linear(784 → 128)  +  ReLU   ← cut-point 0 (Link 0)
  Linear(128 → 64)   +  ReLU   ← cut-point 1 (Link 1)
  Linear(64 → 10)
Output: 10-class logits
```

Total parameters: ~109,386. Trained with Adam (lr=1e-3, 3 epochs),
reaching ~97% test accuracy before compression.

## Gradient oracle

Accuracy gradients w.r.t. `eta` are estimated using the **Stein zeroth-order
oracle** (`grad_oracle` in `src/core/toy_A.py`):

```
∇A(η) ≈ (1 / 2Nσ) Σᵢ zᵢ [A(η + σzᵢ) − A(η − σzᵢ)]
```

Each gradient call makes `2N` accuracy evaluations on a random subset of
`n_samples=512` test images. Default: `sigma=0.05`, `N=50`.

## Accuracy evaluation

`AccuracyFunction` (in `src/core/accuracy_fn.py`) offers two modes:

| Method | Samples | Speed | Use |
|--------|---------|-------|-----|
| `accuracy_callable` | 512 (random subset) | Fast | Called during optimization |
| `accuracy_callable_true` | Full 10,000 test set | Slow | Called for result logging |

## Code entry points

- `src/core/stein_oracles.py` — `MLP`, `get_flat_params`, `gradient_oracle`
- `src/core/accuracy_fn.py` — `CompressedModel`, `AccuracyFunction`, `_compress_decompress`
- `src/core/task_callables.py` — `InferenceTaskCallables`
- `src/core/toy_A.py` — `grad_oracle` (Stein gradient oracle)
- `main.py` — `setup_model_and_callables()` (MLP branch), `make_task()`
- `run_param_sweep.py` — sigma / N sweep experiments
