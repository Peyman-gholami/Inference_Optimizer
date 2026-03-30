# ResNet56 / CIFAR-10 Backend (plug-in)

This backend replaces the MNIST MLP with **ResNet56 on CIFAR-10** as the
accuracy and gradient oracle.  Everything else (`main.py`,
`compare_to_baselines.py`, `compare_to_baselines_multi.py`,
`run_param_sweep.py`, all optimizers) stays the same.

## Quick start

```bash
# Point to the checkpoint file (required)
export RESNET_CHECKPOINT=./models/resnet56-4bfd9763.th
export INFERENCE_OPTIMIZER_BACKEND=resnet

python main.py
```

For a faster run, limit the number of evaluation batches:

```bash
export RESNET_FAST_BATCHES=1     # 1 batch × 100 images for accuracy_callable
python main.py
```

## Baseline comparison

```bash
export RESNET_CHECKPOINT=./models/resnet56-4bfd9763.th
export INFERENCE_OPTIMIZER_BACKEND=resnet

python compare_to_baselines.py --T 50 --baseline_algorithms none
python compare_to_baselines_multi.py --T 30 --K 2 --baseline_algorithms none
```

## Parameter sweep (sigma / N)

```bash
export RESNET_CHECKPOINT=./models/resnet56-4bfd9763.th
export INFERENCE_OPTIMIZER_BACKEND=resnet

python run_param_sweep.py --baseline_algorithms none --experiments sigma
```

## Compressor selection

The ResNet backend supports all three compressors from `compressors.py`.
Compression happens **once per batch at each partition boundary**.

```bash
export COMPRESSOR_TYPE=topk           # Top-K sparsification + PackBits mask (default)
export COMPRESSOR_TYPE=quantization   # Uniform quantization: FP16/INT8/INT4/INT2
export COMPRESSOR_TYPE=llmint8        # Hybrid mixed-precision: outliers at FP16/INT8,
                                      # regular values at INT8/INT4/INT2
```

Use `COMPRESSOR_BACKEND` to select whether the CPU or GPU-native implementation
is used. The GPU-native path (`gpu_compressors.py`) keeps all compression
operations on the GPU, eliminating PCIe round-trips between partitions:

```bash
export COMPRESSOR_BACKEND=cpu   # compressors.py — CPU round-trip (default)
export COMPRESSOR_BACKEND=gpu   # gpu_compressors.py — GPU-native, no CPU transfer
```

A clear banner is printed at startup showing which compressor and backend are active:

```
[ResNet backend] device=cuda
[ResNet backend] compressor=LLMInt8              backend=cpu  path=CPU (compress→cpu→decompress→device)
[ResNet backend] compressor=GPULLMInt8           backend=gpu  path=GPU-native (no CPU round-trip)
```

For `quantization`, the bit-width is selected by the η value at each link:

| η value | Bit-width | Compression |
|---------|-----------|-------------|
| 0.5     | FP16      | 2×          |
| 0.25    | INT8      | 4×          |
| 0.125   | INT4      | 8×          |
| 0.0625  | INT2      | 16×         |

The default `eta_min = 0.125` (INT4 floor) is set to match the coarsest
quantization level that still preserves meaningful accuracy.

Note on `llmint8`: its row-wise AbsMax scaling reshapes the tensor to
`[-1, W]` where `W` is the last spatial dimension of the ResNet feature map.
This differs from its intended use on transformer `[batch, seq_len, hidden_dim]`
tensors, but produces correct output — scaling granularity is per spatial row
rather than per channel, which is an empirical trade-off rather than a
correctness issue.

## Cutpoints and topology

Cutpoints are **integer indices into the ResNet56 unit list** — identical
to the scheme used in the TopK Compression Jetson ResNet project, enabling
direct comparison with prior experiments.

### Unit index mapping (29 units total)

| Index | Unit |
|-------|------|
| 0 | stem (conv1 + bn1 + relu) |
| 1–9 | layer1.0 … layer1.8 (9 BasicBlocks) |
| 10–18 | layer2.0 … layer2.8 (9 BasicBlocks) |
| 19–27 | layer3.0 … layer3.8 (9 BasicBlocks) |
| 28 | head (avgpool + flatten + linear) |

### Default cutpoints `[8, 14, 21]` — matches Jetson project default

Produces 4 partitions and 3 transfer links:

| Partition | Units (indices) | Layers | Link out |
|-----------|----------------|--------|----------|
| 0 | 0–7 | stem + layer1.0..layer1.6 | Link 0 |
| 1 | 8–13 | layer1.7, layer1.8 + layer2.0..layer2.3 | Link 1 |
| 2 | 14–20 | layer2.4..layer2.8 + layer3.0..layer3.1 | Link 2 |
| 3 | 21–28 | layer3.2..layer3.8 + head | — |

The compression at each link operates on the full feature map tensor
output by the last unit in the preceding partition, before it would be
transmitted to the next node in a real distributed deployment.

### Alternative cutpoints for 2-link comparisons

```bash
export RESNET_CUTPOINTS=9,18    # cuts after layer1 and layer2 boundaries
```

| Partition | Units (indices) | Layers |
|-----------|----------------|--------|
| 0 | 0–8 | stem + layer1 |
| 1 | 9–17 | layer2 |
| 2 | 18–28 | layer3 + head |

This matches the coarse 3-node split used in `scatter_gather_mdi_english.py`
(nodes A / B / C).

### Valid cutpoint range

Cutpoints must be strictly increasing integers in the range **1–27**
(cannot cut before the first unit or at/after the head).

## Payload sizes

The activation tensor sizes at each link are **measured automatically**
during setup via a calibration forward pass (`measured_a()`), so the delay
feasibility constraint in the optimizer uses real ResNet activation bytes
rather than toy values. Typical uncompressed sizes at the default cutpoints:

| Link | After partition | Typical size (batch=100) |
|------|----------------|--------------------------|
| 0 | After layer1.6 | ~6.4 MB (100 × 16 × 16 × 32 × 4B) |
| 1 | After layer2.3 | ~3.2 MB (100 × 8 × 8 × 64 × 4B) |
| 2 | After layer3.1 | ~1.6 MB (100 × 4 × 4 × 128 × 4B) |

## Environment variables

| Variable | Role | Default |
|----------|------|---------|
| `INFERENCE_OPTIMIZER_BACKEND` | Set to `resnet` to activate | `mnist` |
| `RESNET_CHECKPOINT` | Path to `resnet56-*.th` | *(required)* |
| `RESNET_DATA_ROOT` | CIFAR-10 root directory | `./data` |
| `RESNET_CUTPOINTS` | Comma-separated int list (range 1–27) | `8,14,21` |
| `RESNET_BATCH_SIZE` | Evaluation batch size | `100` |
| `RESNET_FAST_BATCHES` | Batches used by fast eval | `2` |
| `RESNET_DOWNLOAD` | Auto-download CIFAR-10 | `1` |
| `COMPRESSOR_TYPE` | `topk` / `quantization` / `llmint8` | `topk` |
| `COMPRESSOR_BACKEND` | `cpu` (compressors.py) / `gpu` (gpu_compressors.py) | `cpu` |
| `RESNET_GRAD_SIGMA` | Stein smoothing σ | `0.05` |
| `RESNET_GRAD_N` | Stein sample pairs N | `20` |
| `ETA_MIN0` | η floor on first link | `0.125` |
| `ETA_MIN1` | η floor on remaining links | `0.125` |
| `R_K` | Target throughput R_k(t) | `10.0` |

## Code entry points

- `src/core/resnet_task_callables.py` — `ResNetTaskCallables` + `build_resnet_backend_from_env`
- `src/core/llm_plug_in.py` — `is_resnet_backend()` flag read by `main.py` and comparison scripts
- `src/core/compressors.py` — `TopKCompressor`, `QuantizationCompressor`, `LLMInt8Compressor`
- `src/core/resnet20.py` — `resnet56()` model definition and `build_units()`
