#!/bin/bash
# =============================================================================
# submit_mlp_sweep.sh  —  Slurm job array worker for parallel MLP/MNIST sweep
# =============================================================================
#
# Each array task runs one (sigma or N) value independently via
# run_one_sweep_value.py.  A dependent plot job is submitted automatically
# by the launcher scripts (launch_mlp_sigma_sweep.sh / launch_mlp_N_sweep.sh).
#
# Usage
# -----
#   # Sigma sweep (5 values -> array indices 0-4):
#   sbatch --array=0-4 submit_mlp_sweep.sh sigma
#
#   # N sweep (5 values -> array indices 0-4):
#   sbatch --array=0-4 submit_mlp_sweep.sh N
#
#   # Single value test (index 2 = sigma=0.05):
#   sbatch --array=2 submit_mlp_sweep.sh sigma
#
#   # Run both and plot automatically:
#   bash launch_mlp_sigma_sweep.sh
#   bash launch_mlp_N_sweep.sh
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
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=mlp_sweep
#SBATCH --output=logs/mlp_sweep_%A_%a.out
#SBATCH --error=logs/mlp_sweep_%A_%a.err

# =============================================================================
# CONFIG — edit these before submitting
# =============================================================================
export COMPRESSOR_TYPE=topk             # topk | quantization
                                        # (llmint8 not supported for MLP)
export COMPRESSOR_BACKEND=gpu           # cpu (compressors.py) | gpu (gpu_compressors.py)

EXPERIMENT=${1:-sigma}          # 'sigma' or 'N' (passed as first argument)
T=150                           # time slots per run
SEED=12
FIXED_N=50                      # N used during sigma sweep
FIXED_SIGMA=0.05                # sigma used during N sweep
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/mlp_sweep/${COMPRESSOR_TYPE}
BASELINE_ALGORITHMS=none

# MLP environment
# No INFERENCE_OPTIMIZER_BACKEND — MLP/MNIST is the default backend.
export MNIST_DATA_ROOT=${HOME_DIR}/data  # path to MNIST dataset (downloaded here if absent)
export ETA_MIN0=0.1
export ETA_MIN1=0.05
export R_K=10.0

# =============================================================================
# Parameter value arrays — must match --array range in sbatch command
# =============================================================================
SIGMA_VALUES=(0.001 0.01 0.05 0.1 0.5)   # indices 0-4  -> --array=0-4
N_VALUES=(10 25 50 75 100)                # indices 0-4  -> --array=0-4

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
    --baseline_algorithms "$BASELINE_ALGORITHMS"

EXIT_CODE=$?
echo "Done. Exit code: $EXIT_CODE"
exit $EXIT_CODE
