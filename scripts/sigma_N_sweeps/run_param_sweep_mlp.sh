#!/bin/bash
# =============================================================================
# run_param_sweep_mlp.sh
# Slurm batch script: run run_param_sweep.py for MLP / MNIST
# =============================================================================
#
# Runs both sigma sweep and N sweep sequentially in one job.
# To run only one experiment, pass it as the first argument.
#
# Usage
# -----
#   # Both experiments (sigma sweep then N sweep):
#   sbatch run_param_sweep_mlp.sh
#
#   # Only sigma sweep:
#   sbatch run_param_sweep_mlp.sh sigma
#
#   # Only N sweep:
#   sbatch run_param_sweep_mlp.sh N
#
# Note: for parallel execution (one job per sigma/N value) use
#       submit_mlp_sweep.sh + launch_mlp_sigma_sweep.sh / launch_mlp_N_sweep.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --job-name=mlp_param_sweep
#SBATCH --output=logs/mlp_param_sweep_%j.out
#SBATCH --error=logs/mlp_param_sweep_%j.err

# =============================================================================
# Experiment selection
# 'sigma'   -> sigma sweep only
# 'N'       -> N sweep only
# 'sigma,N' -> both (default)
# =============================================================================
EXPERIMENTS=${1:-sigma,N}

# =============================================================================
# CONFIG — edit before submitting
# =============================================================================
T=150                         # time slots per run (MLP is fast — use larger T)
SEED=12
MU=3.0
EPSILON=0.1
BASELINE_ALGORITHMS=none
HOME_DIR=..

# Sweep values
SIGMA_VALUES=0.001,0.01,0.05,0.1,0.5   # sigma sweep values
N_VALUES=10,25,50,75,100                # N sweep values (MLP: can afford larger N)
FIXED_N=50                              # N fixed during sigma sweep
FIXED_SIGMA=0.05                        # sigma fixed during N sweep

# =============================================================================
# MLP environment variables
# =============================================================================
# No INFERENCE_OPTIMIZER_BACKEND needed — MLP/MNIST is the default backend.
export COMPRESSOR_TYPE=topk             # topk | quantization
                                        # (llmint8 not supported for MLP)
export COMPRESSOR_BACKEND=gpu           # cpu (compressors.py) | gpu (gpu_compressors.py)
export MNIST_DATA_ROOT=${HOME_DIR}/data # path to MNIST dataset (downloaded here if absent)
export ETA_MIN0=0.1
export ETA_MIN1=0.05
export R_K=10.0

# =============================================================================
# Setup
# =============================================================================
OUT_DIR=${HOME_DIR}/outputs/mlp_param_sweep/${COMPRESSOR_TYPE}
mkdir -p logs "$OUT_DIR"

echo "============================================================"
echo "Slurm job:    $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Backend:      MLP / MNIST (default)"
echo "Experiments:  $EXPERIMENTS"
echo "T:            $T"
echo "fixed_N:      $FIXED_N  (sigma sweep)"
echo "fixed_sigma:  $FIXED_SIGMA  (N sweep)"
echo "sigma_values: $SIGMA_VALUES"
echo "N_values:     $N_VALUES"
echo "Compressor:   $COMPRESSOR_TYPE  (backend=$COMPRESSOR_BACKEND)"
echo "MNIST root:   $MNIST_DATA_ROOT"
echo "out_dir:      $OUT_DIR"
echo "============================================================"

# Runtime estimate:
#   MLP eval on 512 samples ≈ 0.05 s/eval
#   sigma sweep: 5 values × T × 2algos × 2 × FIXED_N × 0.05 s
#   N sweep:     dominated by largest N value
SIGMA_EST=$(echo "$T * 2 * 2 * $FIXED_N * 0.05 * 5 / 3600" | bc -l 2>/dev/null || echo "?")
N_MAX=$(echo "$N_VALUES" | tr ',' '\n' | sort -n | tail -1)
N_EST=$(echo "$T * 2 * 2 * $N_MAX * 0.05 / 3600" | bc -l 2>/dev/null || echo "?")
echo "  Estimated sigma sweep : ~${SIGMA_EST} h"
echo "  Estimated N sweep     : ~${N_EST} h  (dominated by N=$N_MAX)"
echo "============================================================"

# =============================================================================
# Activate conda env
# =============================================================================

eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap


# =============================================================================
# Run
# =============================================================================
python -u ${HOME_DIR}/run_param_sweep.py \
    --experiments         "$EXPERIMENTS"         \
    --T                   "$T"                   \
    --seed                "$SEED"                \
    --mu                  "$MU"                  \
    --epsilon             "$EPSILON"             \
    --out_dir             "$OUT_DIR"             \
    --baseline_algorithms "$BASELINE_ALGORITHMS" \
    --sigma_values        "$SIGMA_VALUES"        \
    --N_values            "$N_VALUES"            \
    --fixed_N             "$FIXED_N"             \
    --fixed_sigma         "$FIXED_SIGMA"

EXIT_CODE=$?
echo "============================================================"
echo "run_param_sweep.py finished. Exit code: $EXIT_CODE"
echo "Results saved to: $OUT_DIR"
echo "============================================================"
exit $EXIT_CODE
