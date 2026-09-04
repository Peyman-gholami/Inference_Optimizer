#!/bin/bash
# =============================================================================
# Stein-smoothing table: how far is A_sigma (what the Stein oracle sees) from
# A, and what does the gap cost the controller.  Covers the offline Stein
# scenarios V-TRS / V-LOO -- which share one A(eta), so this is one run per
# compressor -- across topk / quantization / llmint8.
#
# Submit from the repo root on the LOGIN node:
#     sbatch cluster/run_stein_smoothing.sbatch
#
# Outputs: outputs/stein_smoothing/resnet_{topk,quant,llmint8}/
#          stein_smoothing_results.json
# Then, locally:
#     python fill_table_stein.py --run outputs/stein_smoothing/resnet_topk \
#         --run outputs/stein_smoothing/resnet_quant \
#         --run outputs/stein_smoothing/resnet_llmint8 \
#         --out ../MobiHoc_2026/tables/stein_smoothing_table.tex
# =============================================================================
#SBATCH --job-name=stein-smoothing
#SBATCH --account=sioannidis
#SBATCH --partition=gpu                 # gpu = 8h; gpu-short = 2h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=07:00:00
#SBATCH --output=/home/davarakis.t/pretrain_runs/%x-%j.out

set -euo pipefail

# ---- 1. ENVIRONMENT ---------------------------------------------------------
module load anaconda3/2024.06 2>/dev/null || true
module load cuda/12.3.0        2>/dev/null || true

# NOTE: /home has a 75 GB quota that a previous `pip install torch` blew past.
# Build the venv on scratch instead:
#     python -m venv /scratch/$USER/envs/pretrain     # on a COMPUTE node
#     /scratch/$USER/envs/pretrain/bin/pip install torch torchvision numpy scipy
# Scratch is purged monthly; the env is cheap to rebuild.
VENV="${STEIN_VENV:-/scratch/$USER/envs/pretrain}"
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "WARNING: $VENV not found, falling back to \$HOME/envs/pretrain" >&2
  source "$HOME/envs/pretrain/bin/activate"
fi

# ---- 2. REPO + ASSETS -------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:$PWD/src/core"
export INFERENCE_OPTIMIZER_BACKEND=resnet
# Tracked at models/ on this branch, so the default works after a plain pull.
# Override if your copy lives elsewhere, e.g.
#     RESNET_CHECKPOINT=$HOME/assets/resnet56-4bfd9763.th sbatch cluster/run_stein_smoothing.sbatch
export RESNET_CHECKPOINT="${RESNET_CHECKPOINT:-$PWD/models/resnet56-4bfd9763.th}"
export RESNET_DATA_ROOT="$PWD/data"
export RESNET_DOWNLOAD=0                # compute nodes: no downloads
export RESNET_CUTPOINTS="8,14,21"       # 3 links, a = [0.066, 0.033, 0.016] MB

echo "host=$(hostname)"
nvidia-smi -L || true
python -c "import torch;print('cuda:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if [ ! -f "$RESNET_CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found at $RESNET_CHECKPOINT" >&2; exit 1
fi
if [ ! -d "$RESNET_DATA_ROOT/cifar-10-batches-py" ]; then
  echo "ERROR: CIFAR-10 not found at $RESNET_DATA_ROOT/cifar-10-batches-py" >&2
  echo "Pre-fetch it on the LOGIN node (compute nodes are offline):" >&2
  echo "  RESNET_DOWNLOAD=1 python -c \"import torchvision; torchvision.datasets.CIFAR10('$RESNET_DATA_ROOT', train=False, download=True)\"" >&2
  exit 1
fi

# ---- 3. RUN -----------------------------------------------------------------
# Defaults: 6^3 grid, sigmas {0.01,0.02,0.05,0.1,0.2}, 200 draws, 200-image
# fast eval  ->  ~2.2e5 fast evals = ~4.3e7 image-forwards per compressor,
# roughly an hour on an A100.  Raise --n_grid to 8 (512 pts) if time allows:
# it sharpens the regret columns, which are quantised by the grid.
N_GRID="${STEIN_N_GRID:-6}"
N_DRAWS="${STEIN_N_DRAWS:-200}"
SIGMAS="${STEIN_SIGMAS:-0.01,0.02,0.05,0.1,0.2}"

run () {   # $1 = compressor   $2 = out subdir   $3 = extra args
  echo "=================  compressor=$1  ================="
  python stein_smoothing_table.py \
      --compressor "$1" \
      --compressor_backend gpu \
      --scenario "V-TRS/V-LOO" \
      --task "RN56-CF10" \
      --n_grid "$N_GRID" \
      --n_draws "$N_DRAWS" \
      --sigmas "$SIGMAS" \
      --grad_N 20 \
      --out_dir "outputs/stein_smoothing/resnet_$2" ${3:-}
}

# top-k: eta is continuous, a uniform grid is the right sweep.
run topk topk

# quantization: eta is SNAPPED to a ladder (0.0625 INT2, 0.125 INT4, 0.25 INT8,
# 0.5 FP16; anything else is passthrough), so the grid must sit on the rungs or
# the codec is a no-op everywhere and the sweep measures nothing.
#
# Verified on the M3: because a Gaussian perturbation almost surely lands
# BETWEEN rungs, A_sigma collapses to the passthrough accuracy for every sigma
# and the Stein gradient is identically zero.  These two rows are expected to
# come out degenerate -- that is the result, not a bug.  llmint8 was flat at
# 0.94 across the whole eta range even before smoothing.
run quantization quant   "--eta_axis 0.0625,0.125,0.25,0.5,1.0"
run llmint8      llmint8 "--eta_axis 0.0625,0.125,0.25,0.5,1.0"

echo "ALL DONE $(date)"
