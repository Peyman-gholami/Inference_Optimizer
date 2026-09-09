# Inference Optimizer

Offline simulation code accompanying:

**Communication-Aware Model Distributed Inference via Latent Representation Compression**  
*(MobiHoc 2026)*

> We study optimization of distributed model inference over resource-constrained edge resources. We propose a framework that optimizes the trade-off between model accuracy and communication costs by controlling latent representation compression to meet strict Quality of Service (QoS) throughput targets. For settings with known channel state information (CSI), we derive a closed-form optimal solution for single tasks and reduce the multi-task problem to a convex optimization program characterized by a per-link water-filling strategy. We extend these to handle unpredictable environments via a stochastic dual descent algorithm that relies only on causal channel estimates. We provide Lyapunov-based proofs demonstrating that our approach strictly satisfies long-term delay constraints while achieving a bounded optimality gap. Our results offer a robust, scalable blueprint for maximizing the performance of pipelined AI tasks in dynamic, resource-constrained distributed systems. We verify the effectiveness of our proposed framework through simulations and experiments with real edge devices.

Given edge nodes, pipelined inference tasks, and time-varying link capacities, the code chooses per-link compression ratios \(\eta\) to maximize accuracy subject to long-term throughput / delay QoS. It includes CSI-aware optima and CSI-oblivious stochastic dual descent, plus baselines.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run scripts from the repository root so `from src...` imports resolve.

## Public entry points

| Script | Role |
|--------|------|
| `compare_to_baselines.py` | Single-task offline sims vs baselines → NPZ + plots |
| `compare_to_baselines_multi.py` | Multi-task offline sims vs baselines → NPZ + plots |
| `visualize_simulation_results.py` | Replot a saved NPZ (accuracy, excess delay, tradeoff, optional \(\eta\)) |
| `visualize_r_sweep_tradeoff_excess_delay.py` | Aggregate an `--R_range` sweep into one accuracy-vs-excess-delay curve |

### Single-task example

```bash
python compare_to_baselines.py \
  --T 100 --seed 12 --c_min 10 --c_max 100 \
  --mu 6 --epsilon 0.1 --R 10 \
  --task toy_mlp_mnist --M 3 \
  --out_dir outputs/baseline_compare
```

### Multi-task example

```bash
python compare_to_baselines_multi.py \
  --T 150 --K 2 --c_min 100 --c_max 350 \
  --mu 10 --epsilon 0.1 \
  --w_k 1,2 --R_k 10,15 \
  --tasks toy_mlp_mnist,toy_mlp_mnist --M 3 \
  --out_dir outputs/baseline_compare_multi
```

### Replot saved results

```bash
python visualize_simulation_results.py \
  --npz outputs/baseline_compare/simulation_results.npz

python visualize_simulation_results.py \
  --npz outputs/baseline_compare_multi/simulation_results_multi.npz --plot_eta
```

### R-sweep trade-off plot

After a compare run with `--R_range` (outputs under `out_dir/R_<value>/`):

```bash
python compare_to_baselines.py --R_range 10,60,10 --out_dir outputs/baseline_R_sweep \
  --task toy_mlp_mnist --M 3

python visualize_r_sweep_tradeoff_excess_delay.py \
  --sweep_dir outputs/baseline_R_sweep
```

Registered task names are listed in `--task` / `--tasks` help text (via
`src.core.task_handler.registered_task_names()`).

## Package layout

```
src/core/                 # InferenceTask, callables, compressors, Stein oracles
src/core/resnet20.py      # CIFAR ResNet architectures (incl. ResNet-56)
src/core/task_instances/  # Plug-in task families
src/optimizers/           # CSI / no-CSI optimizers, estimators, baselines
assets/                   # ResNet checkpoint, traces, fitting models
```

## Adding a task instance

1. Add `src/core/task_instances/<name>.py` with:
   - `setup_model_and_callables(**kwargs) -> (api, extra)`
   - `make_task(task_id, api, w_k=..., R_k=..., **kwargs) -> InferenceTask`
2. Register it in `src/core/task_handler.py` (or call `register_task(...)`).
3. Pass `--task <name>` (or `--tasks ...`) and set `--M` equal to the task's `L_k`.

See the docstring in `src/core/task_instances/__init__.py` for the full contract.
Shared codecs live in `src/core/compressors.py`.

### Built-in tasks

| Name | Notes |
|------|--------|
| `toy_mlp_mnist` | Small MLP on MNIST (default demo; `--M 3`) |
| `resnet56_cifar10_topk` | ResNet-56 / CIFAR-10, top-\(k\) compression (`--M 4`) |
| `resnet56_cifar10_quantization` | Same module, quantization codec |
| `resnet56_cifar10_llmint8` | Same module, LLM.int8-style codec |

ResNet defaults load from `assets/` (checkpoint, 15 Mbps scenario trace, and
optional poly3 fitting models). Override paths with environment variables if
needed:

```bash
# defaults already point at assets/; CIFAR-10 downloads under ./data
python compare_to_baselines.py --task resnet56_cifar10_topk --M 4 \
  --c_min 1500000 --c_max 1500000 --T 100 --R 5 --out_dir outputs/resnet_compare

# optional overrides:
# export RESNET56_PHASE1_CHECKPOINT=/path/to/resnet56.th
# export RESNET56_PHASE1_TRACE_PATH=/path/to/scenario_trace.json
# export RESNET56_PHASE1_DATA_ROOT=./data
```

Other useful env vars: `RESNET56_PHASE1_DEVICE`, `RESNET56_PHASE1_FAST_SAMPLES`,
`RESNET56_PHASE1_STEIN_N`, `RESNET56_PHASE1_SIGMA`, `RESNET56_PHASE1_ETA_MIN`,
`RESNET56_ACCURACY_ESTIMATOR_MODE` (`stein_estimator` or `fitting_model`),
`RESNET56_FITTING_MODEL_PATH`.

## LLM backend

Experimental LLM / MMLU wiring lives under `src/core/llm_*.py`. See
[README_LLM.md](README_LLM.md).
