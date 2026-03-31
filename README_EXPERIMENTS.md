# Running Experiments — Inference Optimizer

This document describes how to run parameter sweep experiments for all three
backends (ResNet56/CIFAR-10, LLM/MMLU, MLP/MNIST) on a Slurm cluster.

All scripts live in the `scripts/` directory. They call Python entry points
one level up in the project root (`../run_param_sweep.py`,
`../run_one_sweep_value.py`, `../plot_sweep_summary.py`).

---

## Script inventory

### ResNet56 / CIFAR-10

| Script | Mode | Purpose |
|---|---|---|
| `run_param_sweep_resnet.sh` | Sequential | Both sweeps in one job — `topk` / `quantization` / `llmint8`|
| `run_one_value_resnet.sh` | Single value | One sigma or N value, one job |
| `submit_resnet_sweep.sh` | Array worker | Called by launchers; one task per value |
| `launch_resnet_sigma_sweep.sh` | Launcher | Parallel sigma sweep + dependent plot job |
| `launch_resnet_N_sweep.sh` | Launcher | Parallel N sweep + dependent plot job |

### LLM / MMLU

| Script | Mode | Purpose |
|---|---|---|
| `run_param_sweep_llm.sh` | Sequential | Both sweeps in one job — `topk_per_token` / `quantization` / `magnitude`/ `random`|
| `run_one_value_llm.sh` | Single value | One sigma or N value, one job |
| `submit_llm_sweep.sh` | Array worker | Called by launchers; one task per value |
| `launch_llm_sigma_sweep.sh` | Launcher | Parallel sigma sweep + dependent plot job |
| `launch_llm_N_sweep.sh` | Launcher | Parallel N sweep + dependent plot job |

### MLP / MNIST

| Script | Mode | Purpose |
|---|---|---|
| `run_param_sweep_llm.sh` | Sequential | Both sweeps in one job — `topk` / `quantization` |
| `run_one_value_llm.sh` | Single value | One sigma or N value, one job |
| `submit_llm_sweep.sh` | Array worker | Called by launchers; one task per value |
| `launch_llm_sigma_sweep.sh` | Launcher | Parallel sigma sweep + dependent plot job |
| `launch_llm_N_sweep.sh` | Launcher | Parallel N sweep + dependent plot job |

---

## Execution modes

Two modes are available for every backend:

**Sequential** — sigma and N sweeps run back-to-back in a single Slurm job.
Simple to submit. Use for quick tests or when the cluster has a long queue.

**Parallel** — each sigma or N value runs as an independent Slurm task.
All values run simultaneously. A dependent plot job fires automatically when
all sweep tasks complete. Use for production runs.

---

## ResNet56 / CIFAR-10

### Prerequisites

```bash
# Checkpoint must exist at:
ls ../models/resnet56-4bfd9763.th

# CIFAR-10 is downloaded automatically on first run (RESNET_DOWNLOAD=1)
```

### Key ResNet CONFIG variables

Edit these inside the script before submitting.

| Variable | Default | Notes |
|---|---|---|
| `COMPRESSOR_TYPE` | `topk` | `topk` / `quantization` / `llmint8` |
| `COMPRESSOR_BACKEND` | `cpu` | `cpu` / `gpu` — use `gpu` for `llmint8` |
| `T` | `50` | Time slots per run |
| `FIXED_N` | `5` | N used during sigma sweep — keep small for ResNet |
| `FIXED_SIGMA` | `0.05` | sigma used during N sweep |
| `SIGMA_VALUES` | `0.001,0.01,0.05,0.1,0.5` | 5 values — array indices 0–4 |
| `N_VALUES` | `2,5,10,20` | 4 values — array indices 0–3 |
| `HOME_DIR` | `..` | Path to project root (scripts/ is one level below) |

### Mode 1 — Sequential

#### `run_param_sweep_resnet.sh` — TopK / Quantization / LLMInt8

```bash
cd scripts

# Both experiments (sigma then N):
sbatch run_param_sweep_resnet.sh

# Sigma sweep only:
sbatch run_param_sweep_resnet.sh sigma

# N sweep only:
sbatch run_param_sweep_resnet.sh N
```

Slurm resources: `gpu:h200:1`, 16 G RAM, `--time=6:00:00`
Output: `../outputs/resnet_sweep/<COMPRESSOR_TYPE>/`


> **Important:** For compression type `LLMInt8` set
> `COMPRESSOR_BACKEND=gpu` to the environment section before
> submitting. The GPU-native compressor avoids the GPU idle watchdog that
> the cluster scheduler triggers during CPU-bound `llmint8` compression
> (`torch.quantile` on CPU is slow and leaves the GPU idle).

### Mode 2 — Single value

Use for a quick sanity check before committing to a full sweep.

```bash
cd scripts

# Test one sigma value:
sbatch run_one_value_resnet.sh sigma 0.05

# Test one N value:
sbatch run_one_value_resnet.sh N 10
```

Key CONFIG inside `run_one_value_resnet.sh`:

```bash
export COMPRESSOR_TYPE=llmint8
export COMPRESSOR_BACKEND=cpu       # change to gpu for llmint8
OUT_DIR=../outputs/test/resnet      # change before a real run
T=50
FIXED_N=5
```

Slurm resources: `gpu:h200:1`, 4 G RAM, `--time=3:00:00`

### Mode 3 — Parallel sigma sweep

```bash
cd scripts
bash launch_resnet_sigma_sweep.sh
```

Submits two jobs:
1. Sigma array — 5 tasks (`--array=0-4`), one per sigma value
2. Sigma plot — `afterok` dependency, runs `plot_sweep_summary.py` automatically

Key CONFIG in `launch_resnet_sigma_sweep.sh`:

```bash
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/resnet_sweep/llmint8   # change compressor name as needed
TIME_SWEEP=4:00:00
TIME_PLOT=0:30:00
PARTITION=gpu
```

Key CONFIG in `submit_resnet_sweep.sh` (the array worker):

```bash
export COMPRESSOR_TYPE=llmint8
export COMPRESSOR_BACKEND=gpu
T=50
FIXED_N=5
SIGMA_VALUES=(0.001 0.01 0.05 0.1 0.5)   # indices 0-4
```

> **OUT_DIR must match** between the launcher and the array worker (i.e, must match in both scripts). Both
> derive it from `COMPRESSOR_TYPE`, so editing `COMPRESSOR_TYPE` in one
> place is sufficient — but check both files before submitting.

### Mode 4 — Parallel N sweep

```bash
cd scripts
bash launch_resnet_N_sweep.sh
```

Submits two jobs:
1. N array — 4 tasks (`--array=0-3`), one per N value
2. N plot — `afterok` dependency

Key CONFIG in `submit_resnet_sweep.sh`:

```bash
N_VALUES=(2 5 10 20)   # indices 0-3 -> --array=0-3
FIXED_SIGMA=0.05
```

### Manual plot after parallel run

```bash
cd ..
python plot_sweep_summary.py \
    --experiment sigma \
    --sweep_dir  outputs/resnet_sweep/llmint8/exp1_sigma_sweep \
    --out_dir    outputs/resnet_sweep/llmint8/exp1_sigma_sweep/summary

python plot_sweep_summary.py \
    --experiment N \
    --sweep_dir  outputs/resnet_sweep/llmint8/exp2_N_sweep \
    --out_dir    outputs/resnet_sweep/llmint8/exp2_N_sweep/summary
```

### Timing reference

Based on ~1.5 s/eval on H200, `RESNET_FAST_BATCHES=1`, 2 algorithms, 2N
evals per gradient call.

---

## LLM / MMLU

### Prerequisites

```bash
# Model is downloaded from HuggingFace on first run:
export LLM_MODEL_NAME=unsloth/gemma-2b-it # or google/gemma-2b-it

# Requires HuggingFace token if using gated models:
huggingface-cli login
```

### Key LLM CONFIG variables

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL_NAME` | `unsloth/gemma-2b-it` | HuggingFace model ID |
| `LLM_ACTIVATION_STRATEGY` | varies | `topk_per_token` / `magnitude` / `random` / `quantization` |
| `LLM_SPLIT_MODE` | `thirds` | Where compression hooks are placed |
| `T` | `30` | Time slots per run |
| `FIXED_N` | `3` | N used during sigma sweep — keep small for LLM |
| `FIXED_SIGMA` | `0.05` | sigma used during N sweep |
| `SIGMA_VALUES` | `0.001,0.01,0.05,0.1,0.5` | 5 values — indices 0–4 |
| `N_VALUES` | `2,3,5,10` | 4 values — indices 5–8 in array worker |
| `MMLU_FAST_SAMPLES_PER_SUBJECT` | `5` | Questions per subject for fast eval |

### Mode 1 — Sequential

#### `run_param_sweep_llm.sh` — random activation strategy

```bash
cd scripts

# Both experiments:
sbatch run_param_sweep_llm.sh

# Sigma sweep only:
sbatch run_param_sweep_llm.sh sigma

# N sweep only:
sbatch run_param_sweep_llm.sh N
```

One sequential script per activation strategy is the recommended pattern —
each writes to its own output directory
(`../outputs/llm_sweep/<strategy>/`).

Slurm resources: `gpu:h200:1`, 32 G RAM, `--time=8:00:00`

### Mode 2 — Single value

```bash
cd scripts

sbatch run_one_value_llm.sh sigma 0.05
sbatch run_one_value_llm.sh N 5
```

Key CONFIG inside `run_one_value_llm.sh`:

```bash
export LLM_ACTIVATION_STRATEGY=topk_per_token
export LLM_MODEL_NAME=unsloth/gemma-2b-it
T=30
FIXED_N=3
OUT_DIR=../outputs/test/llm           # change before a real run
```

Slurm resources: `gpu:h200:1`, 32 G RAM, `--time=3:00:00`

### Mode 3 — Parallel sigma sweep

```bash
cd scripts
bash launch_llm_sigma_sweep.sh
```

Submits two jobs:
1. Sigma array — 5 tasks (`--array=0-4`), calling `submit_llm_sweep.sh`
2. Sigma plot — `afterok` dependency

`submit_llm_sweep.sh` encodes both sweeps in a single array using an index
offset. This launcher passes `--array=0-4` to select only the sigma slice:

```
Index 0  -> sigma=0.001
Index 1  -> sigma=0.01
Index 2  -> sigma=0.05
Index 3  -> sigma=0.1
Index 4  -> sigma=0.5
```

Key CONFIG in `submit_llm_sweep.sh`:

```bash
export LLM_ACTIVATION_STRATEGY=topk_per_token
export LLM_MODEL_NAME=unsloth/gemma-2b-it
T=30
FIXED_N=3
```

### Mode 4 — Parallel N sweep

```bash
cd scripts
bash launch_llm_N_sweep.sh
```

Submits two jobs:
1. N array — 4 tasks (`--array=5-8`), calling `submit_llm_sweep.sh`
2. N plot — `afterok` dependency

The N sweep uses indices 5–8 because `submit_llm_sweep.sh` packs both
sweeps into one array. The launcher passes `--array=5-8` to select only
the N slice:

```
Index 5  -> N=2
Index 6  -> N=3
Index 7  -> N=5
Index 8  -> N=10
```


### Timing reference

| N | T | MMLU samples/subject | Per-run time |
|---|---|---|---|
| 3 | 30 | 5 | ~12 min |
| 5 | 30 | 5 | ~18 min |
| 10 | 30 | 5 | ~35 min |

---

## MLP / MNIST

The MLP backend is the default — no `INFERENCE_OPTIMIZER_BACKEND` needed.
MNIST downloads automatically on first run.

### Key MLP CONFIG variables

| Variable | Default | Notes |
|---|---|---|
| `COMPRESSOR_TYPE` | `topk` | `topk` / `quantization` only (`llmint8` not supported) |
| `COMPRESSOR_BACKEND` | `cpu` | `cpu` / `gpu` |
| `T` | `150` | Time slots — MLP is fast, larger T is affordable |
| `FIXED_N` | `50` | N used during sigma sweep |
| `FIXED_SIGMA` | `0.05` | sigma used during N sweep |
| `SIGMA_VALUES` | `0.001,0.01,0.05,0.1,0.5` | 5 values — array indices 0–4 |
| `N_VALUES` | `10,25,50,75,100` | 5 values — array indices 0–4 |

### Mode 1 — Sequential

```bash
cd scripts

# Both experiments:
sbatch run_param_sweep_mlp.sh

# Sigma sweep only:
sbatch run_param_sweep_mlp.sh sigma

# N sweep only:
sbatch run_param_sweep_mlp.sh N
```

Slurm resources: `gpu:h200:1`, 8 G RAM, `--time=4:00:00`
Output: `../outputs/mlp_sweep/<COMPRESSOR_TYPE>/`

### Mode 2 — Parallel sigma sweep

```bash
cd scripts
bash launch_mlp_sigma_sweep.sh
```

Submits two jobs:
1. Sigma array — 5 tasks (`--array=0-4`), calling `submit_mlp_sweep.sh sigma`
2. Sigma plot — `afterok` dependency

Key CONFIG in `launch_mlp_sigma_sweep.sh`:

```bash
COMPRESSOR_TYPE=topk        # must match COMPRESSOR_TYPE in submit_mlp_sweep.sh
TIME_SWEEP=02:00:00
TIME_PLOT=0:15:00
```

Key CONFIG in `submit_mlp_sweep.sh`:

```bash
export COMPRESSOR_TYPE=topk
export COMPRESSOR_BACKEND=cpu
T=150
FIXED_N=50
SIGMA_VALUES=(0.001 0.01 0.05 0.1 0.5)   # indices 0-4
```

### Mode 3 — Parallel N sweep

```bash
cd scripts
bash launch_mlp_N_sweep.sh
```

Submits two jobs:
1. N array — 5 tasks (`--array=0-4`), calling `submit_mlp_sweep.sh N`
2. N plot — `afterok` dependency

Key CONFIG in `submit_mlp_sweep.sh`:

```bash
N_VALUES=(10 25 50 75 100)   # indices 0-4 -> --array=0-4
FIXED_SIGMA=0.05
```

## Manual plot after parallel run

```bash
cd ..
python plot_sweep_summary.py \
    --experiment sigma \
    --sweep_dir  outputs/mlp_sweep/topk/exp1_sigma_sweep \
    --out_dir    outputs/mlp_sweep/topk/exp1_sigma_sweep/summary

python plot_sweep_summary.py \
    --experiment N \
    --sweep_dir  outputs/mlp_sweep/topk/exp2_N_sweep \
    --out_dir    outputs/mlp_sweep/topk/exp2_N_sweep/summary
```

### Timing reference

MLP eval on 512 MNIST samples ≈ 0.05 s. Even N=100, T=150 finishes in
under 30 minutes — well within the 2-hour time limit.

---

## Output directory structure

```
outputs/
├── resnet_sweep/
│   ├── topk/
│   │   ├── exp1_sigma_sweep/
│   │   │   ├── sigma_0p001/sweep_result.npz
│   │   │   ├── sigma_0p05/sweep_result.npz
│   │   │   └── summary/
│   │   │       ├── summary_accuracy_over_time_vs_sigma.png
│   │   │       ├── summary_delay_over_time_vs_sigma.png
│   │   │       ├── summary_final_accuracy_vs_sigma.png
│   │   │       └── summary_final_delay_vs_sigma.png
│   │   └── exp2_N_sweep/
│   │       ├── N_5/sweep_result.npz
│   │       └── summary/
│   ├── quantization/
│   └── llmint8/
├── llm_sweep/
│   ├── topk_per_token/
│   ├── magnitude/
│   ├── random/
│   └── quantization/
├── mlp_sweep/
│   ├── topk/
│   └── quantization/
└── test/
    ├── resnet/          ← run_one_value_resnet.sh writes here
    └── llm/             ← run_one_value_llm.sh writes here
```

