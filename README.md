# Inference Optimizer for Model Distributed Inference (MDI)

Inference Optimizer studies how to deploy a partitioned neural network across resource-constrained nodes while balancing prediction accuracy and inference throughput. It selects how much intermediate activation data to retain on each network link and, when multiple tasks share the deployment, how to divide compute and communication resources among them.

The repository provides optimization algorithms, model accuracy evaluators, and offline experiments with synthetic or recorded network capacities. Its outputs are deployment decisions that a serving runtime can apply. Model partitioning and node placement are supplied as inputs; the project does not currently launch distributed workers or enforce resource shares on devices.

Research context: **Communication-Aware Model Distributed Inference via Latent Representation Compression** (MobiHoc 2026).

## Why optimize an MDI deployment?

In MDI, consecutive parts of a model execute on different nodes. Each node forwards intermediate activations to the next stage. A slow link can limit the throughput of the entire pipeline, even when compute resources are available.

Activation compression reduces the modeled communication load but can reduce prediction accuracy. Sharing a node or link among several inference tasks introduces another decision: which task receives the available capacity?

| Deployment decision | Optimizer output | Effect |
|---|---|---|
| Activation retention on each hop | `eta` | Smaller values reduce modeled payload size; larger values preserve more activation information. |
| Compute share at each stage | `s_comp` | Allocates node capacity among tasks using that node. |
| Communication share on each hop | `s_comm` | Allocates link capacity among tasks using that link. |

The topology is a line of `M` nodes. Task `k` starts at node `b_k` and uses `L_k` consecutive stages, ending at `e_k = b_k + L_k - 1`. It has one compression decision per hop, so its `eta` vector has length `L_k - 1`.

## Optimization model

### Inputs and units

Each [`InferenceTask`](src/core/task.py) describes a fixed model pipeline:

| Input | Meaning |
|---|---|
| `tau[i]` | Stage computation time at node `i` with the full node resource. |
| `a[i]` | Uncompressed activation payload sent over link `i`. |
| `eta_min[i]` | Minimum allowed retained fraction on that link. |
| `R_k(t)` | Target pipeline throughput in work items per second. |
| `w_k` | Task importance in the multi-task accuracy objective. |
| `A_k(eta)` | Accuracy oracle used during optimization. |
| `grad_A_k(eta)` | Accuracy gradient, estimated or obtained from a fitted model. |
| `A_k_true(eta)` | Separate model evaluation used to report accuracy. |

The environment supplies link capacities `c_t`, or causal estimates `c_hat` when the current capacity is unavailable.

Use a consistent work item and unit convention: if `a` is bytes per batch, `c` must be bytes per second, `tau` must be seconds per batch, and `R` must be batches per second. The MNIST recipe models communication for a batch of 100 activations, with payloads of 51,200 and 25,600 bytes; its compute times are illustrative constants. Convert a measured capacity in bits per second to bytes per second before combining it with byte payloads.

### Accuracy versus pipeline throughput

The modeled transmitted payload is `a[i] * eta[i]`, with:

$$
\eta_{i,k}^{\min} \leq \eta_{i,k}(t) \leq 1.
$$

Here, `eta = 1` means no modeled payload reduction, and values closer to `eta_min` mean stronger compression. For top-k compression, this controls the retained activation fraction. Other codecs translate `eta` into their own execution parameters.

The bottleneck service interval for task `k` is:

$$
D_k(t) = \max\left\{
\max_{i \in \text{task nodes}} \frac{\tau_{i,k}}{s^{\mathrm{comp}}_{i,k}(t)},
\max_{i \in \text{task links}} \frac{a_{i,k}\eta_{i,k}(t)}{s^{\mathrm{comm}}_{i,k}(t)c_i(t)}
\right\}.
$$

This maximum models the spacing between completed work items in a pipelined deployment. End-to-end request latency, including stage traversal and queueing, requires additional measurement and modeling.

With current channel state information (CSI), the objective is to maximize weighted accuracy:

$$
\max \sum_k w_k A_k(\boldsymbol{\eta}_k(t)),
\qquad D_k(t) \leq \frac{1}{R_k(t)},
$$

subject to the compression bounds and resource budgets: compute shares sum to at most one at each node, and communication shares sum to at most one on each link. Without current CSI, the adaptive algorithms use estimated capacities and delay feedback to target the corresponding long-term average residual constraint.

### CSI-aware optimization

[`src/optimizers/csi_aware.py`](src/optimizers/csi_aware.py) implements two cases:

| Class | Method |
|---|---|
| `CSIAwareSingleTaskOptimizer` | Checks feasibility, assigns full resources to the task, and selects compression in closed form. |
| `CSIAwareMultiTaskOptimizer` | Checks aggregate feasibility and uses SciPy SLSQP to optimize communication shares with accuracy gradients. Compute shares are derived from the throughput requirements. |

For a single task, the retained fraction is:

$$
\eta_i^*(t) = \min\left(1,\frac{c_i(t)}{R(t)a_i}\right).
$$

This chooses the largest retained fraction that fits the throughput budget, assuming accuracy is nondecreasing with retained information. It applies only when every stage satisfies `tau[i] <= 1/R` and every link can carry at least `a[i] * eta_min[i]` within `1/R`. Otherwise, the optimizer returns `None`.

For example, with `a = 50,000` bytes per work item, `c = 250,000` bytes/s, and `R = 10` work items/s, the link can retain at most `eta = 0.5`. Its modeled transfer interval is then `50,000 * 0.5 / 250,000 = 0.1` seconds, matching `1/R`, provided the compute stages and minimum-retention bound are also feasible.

For multiple tasks, feasibility requires, on every shared node and link respectively:

$$
\sum_k \tau_{i,k}R_k(t) \leq 1,
\qquad
\sum_k \frac{a_{i,k}\eta_{i,k}^{\min}R_k(t)}{c_i(t)} \leq 1,
$$

where each sum includes only tasks using that resource. The solver then distributes communication shares to improve weighted accuracy. The convex formulation assumes suitable concave accuracy functions; empirical accuracy oracles and fitted polynomials should be checked before interpreting numerical results as global optima.

The shared allocation helper normalizes returned compute and communication shares to use all capacity on each occupied resource, while leaving `eta` unchanged. Returned shares can therefore exceed the minimum shares derived from the throughput equations.

### Adaptive optimization without current CSI

[`src/optimizers/no_csi.py`](src/optimizers/no_csi.py) uses estimated stochastic dual descent:

1. Estimate link capacities from past observations.
2. Choose compression and resource shares using the estimates and current delay multipliers.
3. Evaluate the realized bottleneck interval using the true observed capacities.
4. Update each task's multiplier and then update the capacity estimator.

The multiplier update is:

$$
\lambda_k(t+1) = \max\left(\epsilon,
\lambda_k(t) + D_k^{\mathrm{actual}}(t) - \frac{1}{R_k(t)}\right).
$$

A throughput violation raises the multiplier, increasing the delay penalty in subsequent decisions. A negative residual allows it to decrease toward `epsilon`.

| Class | Per-slot method |
|---|---|
| `NoCSISingleTaskOptimizer` | Minimizes `-A(eta) + mu * lambda * z`, where `z` bounds estimated compute and communication intervals. |
| `NoCSIMultiTaskOptimizer` | Alternates resource allocation and per-task compression optimization for `J` block coordinate descent iterations; maintains one multiplier per task. |

In this implementation, `mu` multiplies the delay penalty: increasing it gives delay more weight relative to accuracy for a fixed multiplier. `epsilon` is the multiplier floor, and `J` defaults to 10 in the multi-task optimizer.

The comparison scripts use a lower-capacity estimate based on historical mean minus a standard-deviation margin. Additional estimators include historical mean, last observation, running minimum, and moving average in [`src/optimizers/estimators.py`](src/optimizers/estimators.py). Estimates are computed before the current observation is added, preserving causal ordering. Adaptive decisions can violate the target in individual slots; a finite experiment does not establish long-term feasibility.

## Running the project

Run all commands from the repository root. Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The requirements file also includes dependencies for the experimental LLM modules.

### Single-task experiment

```bash
python compare_to_baselines.py \
  --task toy_mlp_mnist --M 3 \
  --T 100 --seed 12 --R 10 \
  --c_min 100000 --c_max 1000000 \
  --mu 6 --epsilon 0.1 \
  --out_dir outputs/baseline_compare
```

The MNIST setup downloads the dataset when needed and trains an MLP for three epochs. Accuracy and gradient evaluations can make experiments substantially slower than the optimization alone. These capacity bounds are in bytes per second for the built-in MNIST payloads and allow the CSI-aware task to meet its minimum-retention constraints.

### Multiple tasks sharing the deployment

```bash
python compare_to_baselines_multi.py \
  --tasks toy_mlp_mnist,toy_mlp_mnist --M 3 --K 2 \
  --T 150 --seed 567 \
  --w_k 1,2 --R_k 10,15 \
  --c_min 100000 --c_max 1000000 \
  --mu 10 --epsilon 0.1 \
  --out_dir outputs/baseline_compare_multi
```

`w_k` controls relative accuracy importance; `R_k` controls each task's throughput target. The comparison scripts require `M` to equal each selected recipe's `L_k`. The core task API also supports tasks occupying a consecutive subset of a larger line topology.

Both comparison scripts generate uniformly sampled channel capacities. Use `--baseline_algorithms none` to run only the CSI-aware and adaptive optimizers, or select comma-separated baselines:

| Experiment | Available baseline names |
|---|---|
| Single task | `max_compression`, `no_compression`, `uniform_compression`, `myopic`, `conservative`, `moving_average`, `lcb` |
| Multiple tasks | `max_compression`, `no_compression`, `equal_share`, `proportional`, `strict_priority`, `decoupled_equal_split`, `queue_proportional`, `certainty_equivalence`, `lcb` |

### ResNet-56 / CIFAR-10 trace replay

The ResNet recipe uses four stages, compressing activations after `layer1`, `layer2`, and `layer3`. A checkpoint and recorded scenario are included in `assets/`.

```bash
export RESNET56_PHASE1_CHECKPOINT="$PWD/assets/resnet56-4bfd9763.th"
export RESNET56_PHASE1_TRACE_PATH="$PWD/assets/resnet56_cifar10_15Mbps_scenario_trace_topk.json"
export RESNET56_PHASE1_DOWNLOAD=1

python run_phase1_resnet_compare.py \
  --M 4 --T 100 --R 5 --mu 6 --epsilon 0.1 \
  --out_dir outputs/phase1_resnet_compare
```

The explicit checkpoint variable is required. CIFAR-10 is read from `data/` by default; enable downloading as above if it is absent. With the profile variable set as above, the task derives fixed `tau` and `a` from trace medians, while the launcher replays `trace.channel_true_bps` slot by slot. If you supply a custom `--trace_path`, also set `RESNET56_PHASE1_TRACE_PATH` to the same file to use its compute and payload profile. The launcher defaults to the top-k polynomial accuracy model configured by `ACCURACY_ESTIMATOR_MODE` and `FITTING_MODEL_PATH` in the launcher. Reported accuracy still comes from the compressed ResNet evaluator.

The trace loader uses `activation_bytes` and `channel_true_bps` numerically as stored, without a bit/byte conversion. Check the recording's capacity convention before interpreting replay results as deployment throughput in physical units. The launcher also replaces nonpositive capacity entries with positive fallback values and uses minimum retention if a CSI-aware decision is infeasible; these behaviors matter when interpreting its results.

For direct ResNet experiments, set `RESNET56_PHASE1_TRACE_PATH` to derive the stage profile from a trace. Direct comparison scripts still sample capacities rather than replaying them. Useful task settings include:

| Environment variable | Purpose |
|---|---|
| `RESNET56_PHASE1_CHECKPOINT` | Required model checkpoint path. |
| `RESNET56_PHASE1_DATA_ROOT` | CIFAR-10 data directory. |
| `RESNET56_PHASE1_DEVICE` | Device, such as `cpu` or `cuda`. |
| `RESNET56_PHASE1_TRACE_PATH` | Trace used to derive fixed compute times and payloads. |
| `RESNET56_PHASE1_FAST_SAMPLES` / `RESNET56_PHASE1_TRUE_SAMPLES` | Optimization-oracle and reporting sample counts. |
| `RESNET56_PHASE1_STEIN_N` / `RESNET56_PHASE1_SIGMA` | Stein gradient sample count and smoothing scale. |
| `RESNET56_PHASE1_ETA_MIN` | Minimum retained fraction. |
| `RESNET56_ACCURACY_ESTIMATOR_MODE` | `stein_estimator` or `fitting_model` for direct task construction. |
| `RESNET56_FITTING_MODEL_PATH` | Required fitted-model path when using `fitting_model`. |

The trace launcher sets accuracy mode and fitting path from its own configuration globals, overriding those environment settings.

## Reading experiment results

Each comparison saves raw arrays, run metadata, and plots under `--out_dir`:

| Output | Meaning |
|---|---|
| `simulation_results.npz` / `simulation_results_multi.npz` | Channel histories, algorithm names, accuracy, compression histories, and signed excess delay. |
| `simulation_metadata.json` / `simulation_metadata_multi.json` | Experiment parameters and selected algorithms. |
| `accuracy_over_time.png` | Cumulative average evaluated accuracy. |
| `excess_delay_over_time.png` | Cumulative average of `D_k(t) - 1/R_k(t)`. |
| `eta_over_time.png` | Cumulative average compression from the single-task experiments. |
| `tradeoff_acc_vs_excess_delay_Tminus1.png` | Final cumulative accuracy versus signed excess delay. |

Positive excess delay means the pipeline is slower than its target interval; zero or negative values meet that interval. This metric is signed, so negative values are retained. Multi-task accuracy is a weighted average, while plotted multi-task excess delay is an unweighted mean across tasks. Inspect individual task residuals when assessing each task's QoS. Generic comparison scripts leave missing values when an optimizer returns `None`.

Replot a saved run:

```bash
python visualize_simulation_results.py \
  --npz outputs/baseline_compare/simulation_results.npz

python visualize_simulation_results.py \
  --npz outputs/baseline_compare_multi/simulation_results_multi.npz --plot_eta
```

Sweep the throughput requirement to inspect the accuracy/throughput tradeoff. The range uses Python's exclusive stop convention:

```bash
python compare_to_baselines.py \
  --task toy_mlp_mnist --M 3 --R_range 10,60,10 \
  --c_min 100000 --c_max 1000000 \
  --out_dir outputs/baseline_R_sweep

python visualize_r_sweep_tradeoff_excess_delay.py \
  --sweep_dir outputs/baseline_R_sweep
```

## Integrating with an MDI deployment

The optimizer can serve as a controller around an existing distributed inference runtime:

1. **Define the pipeline.** Choose partition boundaries and node placement, then construct an `InferenceTask` for each model path.
2. **Profile the deployment.** Measure full-resource stage times, activation payloads, and link capacities using consistent batch sizes and units. Calibrate an accuracy oracle for the chosen codec.
3. **Select a controller.** Use a CSI-aware optimizer when current capacities are available, or a no-CSI optimizer with a capacity estimator and observed interval feedback.
4. **Apply the allocation.** Translate each task's `eta` into codec settings and implement `s_comp` and `s_comm` through the runtime's compute and network scheduling mechanisms.
5. **Feed back observations.** After execution, update the adaptive multiplier using the realized bottleneck interval, then add observed link capacities to the estimator.

The following integration skeleton uses a registered MNIST task. It shows the controller API; the surrounding runtime must execute the returned policy and supply observations:

```python
from src.core.task_handler import get_task
from src.optimizers.estimators import MeanMinusZStdLCB
from src.optimizers.no_csi import NoCSISingleTaskOptimizer

task = get_task("toy_mlp_mnist", task_id=1, R_k=10.0)
optimizer = NoCSISingleTaskOptimizer(M=3, tasks=[task], mu=6.0, epsilon=0.1)
estimator = MeanMinusZStdLCB(num_links=2, warmup_value=100000.0)


def choose_policy(t):
    # policy[task.task_id] contains eta, s_comp, and s_comm arrays.
    return optimizer.optimize(t=t, c_hat=estimator.estimate(t))


def observe_execution(t, actual_interval, observed_capacities):
    optimizer.update_dual(t=t, actual_delays={task.task_id: actual_interval})
    estimator.update(observed_capacities)
```

Call `choose_policy(t)`, execute that policy, then call `observe_execution(...)` for the same slot. For CSI-aware controllers, call `optimize(t=t, c_t=current_capacities)` and handle `None` with the deployment's admission or fallback policy. In simulations, `compute_residual_delay(...)` evaluates the interval model from an allocation and realized capacities; it does not collect device telemetry.

Before using decisions to control devices, validate solver convergence and returned constraints: the current numerical optimizers use SLSQP's `result.x` without checking `result.success`. Also measure codec execution time, serialization overhead, and actual wire payload size. The delay model treats payload as `a * eta` and does not explicitly include those overheads; top-k sparsification alone does not serialize a sparse network message.

## Extending the project

| Path | Responsibility |
|---|---|
| `src/core/task.py` | Task topology, resource profile, throughput target, and accuracy interfaces. |
| `src/core/task_handler.py` | Task registry and cached model setup. |
| `src/core/task_instances/` | MNIST and ResNet task recipes. |
| `src/core/compressors.py` | Shared compression codecs. |
| `src/core/task_callables.py`, `src/core/accuracy_fn.py`, `src/core/toy_A.py` | Compressed-model evaluation and gradient estimation. |
| `src/optimizers/` | CSI-aware/adaptive controllers, estimators, and baselines. |
| `compare_to_baselines.py`, `compare_to_baselines_multi.py` | Single-task and multi-task synthetic-channel experiments. |
| `run_phase1_resnet_compare.py` | ResNet recorded-channel experiment. |
| `main.py` | Demonstration of the four optimizer configurations. |
| `assets/` | ResNet checkpoint, scenario traces, and fitted accuracy models. |
| `tests/` | Optimizer tests with synthetic accuracy functions. |

Registered recipes are `toy_mlp_mnist` (`M=3`), `resnet56_cifar10_topk`, `resnet56_cifar10_quantization`, and `resnet56_cifar10_llmint8` (all `M=4`). The LLM.int8-style ResNet recipe supports an optional eta-to-codec mapping via `RESNET56_LLMINT8_MAPPING_PATH` and falls back to an analytic mapping when entries are unavailable. Use a fitted accuracy model calibrated to the selected codec and mapping when enabling `fitting_model` mode.

To add a model, implement `setup_model_and_callables(**kwargs) -> (api, extra)` and `make_task(task_id, api, w_k=..., R_k=..., **kwargs) -> InferenceTask` in a task-instance module. The API must expose `accuracy_callable`, `gradient_callable`, and `accuracy_callable_true`. Register the module in `src/core/task_handler.py`, then select it with `--task` or `--tasks`. See the [task-instance contract](src/core/task_instances/__init__.py).

Experimental LLM/MMLU components and measured Gemma profiles live in `src/core/llm_*.py` and `src/core/measured_llm_profile.py`. [README_LLM.md](README_LLM.md) describes their interfaces; the current `main.py` and comparison entry points do not call the environment-switch helpers, so enabling an LLM backend requires integration work.
