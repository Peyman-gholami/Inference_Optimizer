# ResNet56 / CIFAR-10 Backend (plug-in)

This backend replaces the MNIST MLP with **ResNet56 on CIFAR-10** as the
accuracy and gradient oracle.  Everything else (`main.py`,
`compare_to_baselines.py`, `compare_to_baselines_multi.py`,
`run_param_sweep.py`, all optimizers) stays the same.

## Quick start

```bash
# Point to the checkpoint file (required)
export RESNET_CHECKPOINT=./resnet56-4bfd9763.th
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
export RESNET_CHECKPOINT=./resnet56-4bfd9763.th
export INFERENCE_OPTIMIZER_BACKEND=resnet

python compare_to_baselines.py --T 50 --baseline_algorithms none
python compare_to_baselines_multi.py --T 30 --K 2 --baseline_algorithms none
```

## Parameter sweep (sigma / N)

```bash
export RESNET_CHECKPOINT=./resnet56-4bfd9763.th
export INFERENCE_OPTIMIZER_BACKEND=resnet

python run_param_sweep.py --baseline_algorithms none --experiments sigma
```

## Compressor selection

The ResNet backend and the `CompressedModel` (MLP backend) share the same
`COMPRESSOR_TYPE` variable:

```bash
export COMPRESSOR_TYPE=quantization   # FP16/INT8/INT4/INT2 selected by eta
export COMPRESSOR_TYPE=llmint8        # LLM.int8() hybrid mixed-precision
export COMPRESSOR_TYPE=topk           # top-K sparsification (default)
```

## Topology

Default cutpoints `[8, 14, 21]` split ResNet56 into **4 partitions** across
**4 nodes** connected by **3 transfer links**:

| Node | Units | Link out |
|------|-------|----------|
| 0 | stem + layer1.0..layer1.6 | Link 0 |
| 1 | layer1.7, layer1.8, layer2.0..layer2.3 | Link 1 |
| 2 | layer2.4..layer2.8, layer3.0, layer3.1 | Link 2 |
| 3 | layer3.2..layer3.8 + head | — |

The payload sizes (`a[i]`, in bytes per sample) are measured automatically
by a calibration forward pass during setup, so the delay feasibility
constraint uses real ResNet activation sizes.

To change the split:

```bash
export RESNET_CUTPOINTS=9,18          # 2 links (3 nodes)
export RESNET_CUTPOINTS=8,14,21       # 3 links (4 nodes, default)
```

## Environment variables

| Variable | Role | Default |
|----------|------|---------|
| `INFERENCE_OPTIMIZER_BACKEND` | Set to `resnet` to activate | `mnist` |
| `RESNET_CHECKPOINT` | Path to `resnet56-*.th` | *(required)* |
| `RESNET_DATA_ROOT` | CIFAR-10 root directory | `./data` |
| `RESNET_CUTPOINTS` | Comma-separated int list | `8,14,21` |
| `RESNET_BATCH_SIZE` | Evaluation batch size | `100` |
| `RESNET_FAST_BATCHES` | Batches used by fast eval | `2` |
| `RESNET_DOWNLOAD` | Auto-download CIFAR-10 | `1` |
| `COMPRESSOR_TYPE` | topk / quantization / llmint8 | `topk` |
| `RESNET_GRAD_SIGMA` | Stein smoothing σ | `0.05` |
| `RESNET_GRAD_N` | Stein sample pairs N | `20` |
| `ETA_MIN0` | η floor on first link | `0.125` |
| `ETA_MIN1` | η floor on remaining links | `0.125` |
| `R_K` | Target throughput R_k(t) | `10.0` |

## Code entry points

- `src/core/resnet_task_callables.py` — `ResNetTaskCallables` + `build_resnet_backend_from_env`
- `src/core/llm_plug_in.py` — `is_resnet_backend()` flag read by `main.py` and comparison scripts
- `compressors.py` — `TopKCompressor`, `QuantizationCompressor`, `LLMInt8Compressor`
- `resnet20.py` — `resnet56()` model definition
