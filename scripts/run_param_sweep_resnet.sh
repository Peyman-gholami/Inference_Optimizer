#!/bin/bash
# =============================================================================
# run_param_sweep_resnet_resnet.sh
# Slurm batch script: run run_param_sweep_resnet.py for ResNet56 / CIFAR-10
# =============================================================================
#
# Runs both sigma sweep and N sweep sequentially in one job.
# To run only one experiment, change --experiments below.
#
# Usage
# -----
#   # Both experiments (sigma + N sweep):
#   sbatch run_param_sweep_resnet.sh
#
#   # Only sigma sweep:
#   sbatch run_param_sweep_resnet.sh sigma
#
#   # Only N sweep:
#   sbatch run_param_sweep_resnet.sh N
#
# Note: for parallel execution (one job per sigma/N value) use
#       run_one_value.sh + plot_sweep_summary.py instead.
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=resnet_param_sweep
#SBATCH --output=logs/resnet_param_sweep_%j.out
#SBATCH --error=logs/resnet_param_sweep_%j.err

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
T=50                          # time slots per run
SEED=12
MU=3.0
EPSILON=0.1
BASELINE_ALGORITHMS=none
HOME_DIR=..

# Sweep values
SIGMA_VALUES=0.001,0.01,0.05,0.1,0.5   # sigma sweep values
N_VALUES=2,5,10,20                      # N sweep values  (ResNet: keep small)
FIXED_N=5                               # N fixed during sigma sweep
FIXED_SIGMA=0.05                        # sigma fixed during N sweep

# =============================================================================
# ResNet environment variables
# =============================================================================
export COMPRESSOR_TYPE=topk # valid values topk (default), quantization, llmint8
export COMPRESSOR_BACKEND=cpu       # cpu (compressors.py) | gpu (gpu_compressors.py)
export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_CUTPOINTS=8,14,21
export RESNET_BATCH_SIZE=100
export RESNET_FAST_BATCHES=1
export RESNET_DOWNLOAD=1
export ETA_MIN0=0.125
export ETA_MIN1=0.125
export R_K=10.0

# =============================================================================
# Setup
# =============================================================================

OUT_DIR=${HOME_DIR}/outputs/resnet_param_sweep/${COMPRESSOR_TYPE}
mkdir -p logs "$OUT_DIR"

echo "============================================================"
echo "Slurm job:    $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Experiments:  $EXPERIMENTS"
echo "T:            $T"
echo "fixed_N:      $FIXED_N  (sigma sweep)"
echo "fixed_sigma:  $FIXED_SIGMA  (N sweep)"
echo "sigma_values: $SIGMA_VALUES"
echo "N_values:     $N_VALUES"
echo "out_dir:      $OUT_DIR"
echo "Checkpoint:   $RESNET_CHECKPOINT"
echo "Compressor:   $COMPRESSOR_TYPE"
echo "============================================================"


# =============================================================================
# Activate conda env
# =============================================================================

eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

# Runtime estimate:
#   sigma sweep: 5 values × T × 2algos × 2N × ~1.5s/eval
#   N sweep:     dominated by largest N value
SIGMA_EST=$(echo "$T * 2 * 2 * $FIXED_N * 1.5 * 5 / 3600" | bc -l 2>/dev/null || echo "?")
N_MAX=$(echo "$N_VALUES" | tr ',' '\n' | sort -n | tail -1)
N_EST=$(echo "$T * 2 * 2 * $N_MAX * 1.5 / 3600" | bc -l 2>/dev/null || echo "?")
echo "  Estimated sigma sweep : ~${SIGMA_EST} h"
echo "  Estimated N sweep     : ~${N_EST} h  (dominated by N=$N_MAX)"
echo "============================================================"

# =============================================================================
# Run
# =============================================================================
python -u ${HOME_DIR}/run_param_sweep.py \
    --experiments         "$EXPERIMENTS"        \
    --T                   "$T"                  \
    --seed                "$SEED"               \
    --mu                  "$MU"                 \
    --epsilon             "$EPSILON"            \
    --out_dir             "$OUT_DIR"            \
    --baseline_algorithms "$BASELINE_ALGORITHMS" \
    --sigma_values        "$SIGMA_VALUES"       \
    --N_values            "$N_VALUES"           \
    --fixed_N             "$FIXED_N"            \
    --fixed_sigma         "$FIXED_SIGMA"

EXIT_CODE=$?
echo "============================================================"
echo "run_param_sweep.py finished. Exit code: $EXIT_CODE"
echo "Results saved to: $OUT_DIR"
echo "============================================================"
exit $EXIT_CODE
