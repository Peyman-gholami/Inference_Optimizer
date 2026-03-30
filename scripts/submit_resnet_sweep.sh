#!/bin/bash
# =============================================================================
# submit_sweep.sh  —  Slurm job array for parallel parameter sweep
# =============================================================================
#
# Each array task runs one (sigma or N) value independently via
# run_one_sweep_value.py, then a dependent job runs plot_sweep_summary.py.
#
# Usage
# -----
#   # Sigma sweep (5 values -> array indices 0-4):
#   sbatch --array=0-4 submit_resnet_sweep.sh sigma
#
#   # N sweep (4 values -> array indices 0-3):
#   sbatch --array=0-3 submit_resnet_sweep.sh N
#
#   # Run both and plot automatically (submits all jobs with dependencies):
#   bash launch_full_sweep.sh
#
# Configuration
# -------------
# Edit the variables in the CONFIG section below before submitting.
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --job-name=sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# =============================================================================
# CONFIG — edit these before submitting
# =============================================================================
export COMPRESSOR_TYPE=llmint8 # valid values topk (default), quantization, llmint8
export COMPRESSOR_BACKEND=gpu       # cpu (compressors.py) | gpu (gpu_compressors.py)

EXPERIMENT=${1:-sigma}          # 'sigma' or 'N' (passed as first argument)
T=50                            # time slots per run
SEED=12
FIXED_N=5                       # N used during sigma sweep (ResNet: keep small)
FIXED_SIGMA=0.05                # sigma used during N sweep
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/resnet_sweep/${COMPRESSOR_TYPE}
BASELINE_ALGORITHMS=none

# ResNet environment
export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_FAST_BATCHES=1
export RESNET_CUTPOINTS=8,14,21

export ETA_MIN0=0.125
export ETA_MIN1=0.125
export R_K=10.0

# =============================================================================
# Parameter value arrays — must match --array range in sbatch command
# =============================================================================
SIGMA_VALUES=(0.001 0.01 0.05 0.1 0.5)   # indices 0-4  -> --array=0-4
N_VALUES=(2 5 10 20)                      # indices 0-3  -> --array=0-3

# =============================================================================
# Select value for this array task
# =============================================================================
mkdir -p logs "$OUT_DIR"

if [ "$EXPERIMENT" = "sigma" ]; then
    PARAM_VALUE=${SIGMA_VALUES[$SLURM_ARRAY_TASK_ID]}
    echo "Array task $SLURM_ARRAY_TASK_ID: sigma=$PARAM_VALUE, fixed_N=$FIXED_N"
elif [ "$EXPERIMENT" = "N" ]; then
    PARAM_VALUE=${N_VALUES[$SLURM_ARRAY_TASK_ID]}
    echo "Array task $SLURM_ARRAY_TASK_ID: N=$PARAM_VALUE, fixed_sigma=$FIXED_SIGMA"
else
    echo "ERROR: unknown experiment '$EXPERIMENT'. Use 'sigma' or 'N'."
    exit 1
fi

# =============================================================================
# Activate conda env
# =============================================================================

eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

# =============================================================================
# Run
# =============================================================================
python -u ${HOME_DIR}/run_one_sweep_value.py \
    --param            "$EXPERIMENT"  \
    --value            "$PARAM_VALUE" \
    --fixed_N          "$FIXED_N"     \
    --fixed_sigma      "$FIXED_SIGMA" \
    --T                "$T"           \
    --seed             "$SEED"        \
    --out_dir          "$OUT_DIR"     \
    --baseline_algorithms "$BASELINE_ALGORITHMS" \
