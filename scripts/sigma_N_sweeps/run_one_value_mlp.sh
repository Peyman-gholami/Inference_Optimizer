#!/bin/bash
# =============================================================================
# run_one_value_mlp.sh  —  Run a single MLP sweep value as one Slurm job
# =============================================================================
#
# Usage
# -----
#   sbatch run_one_value_mlp.sh sigma 0.05
#   sbatch run_one_value_mlp.sh N 50
#
# Arguments
#   $1  param   : 'sigma' or 'N'
#   $2  value   : the parameter value to run (e.g. 0.05 or 50)
#
# Example — run all sigma values manually, one job each:
#   sbatch run_one_value_mlp.sh sigma 0.001
#   sbatch run_one_value_mlp.sh sigma 0.01
#   sbatch run_one_value_mlp.sh sigma 0.05
#   sbatch run_one_value_mlp.sh sigma 0.1
#   sbatch run_one_value_mlp.sh sigma 0.5
#
# Example — run all N values:
#   sbatch run_one_value_mlp.sh N 10
#   sbatch run_one_value_mlp.sh N 25
#   sbatch run_one_value_mlp.sh N 50
#   sbatch run_one_value_mlp.sh N 75
#   sbatch run_one_value_mlp.sh N 100
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=mlp_single
#SBATCH --output=logs/mlp_single_%x_%j.out
#SBATCH --error=logs/mlp_single_%x_%j.err

# =============================================================================
# CONFIG — edit these before submitting
# =============================================================================
T=150
SEED=12
C_MIN=10.0
C_MAX=150.0
MU=3.0
EPSILON=0.1
FIXED_N=50          # N used when sweeping sigma
FIXED_SIGMA=0.05    # sigma used when sweeping N
HOME_DIR=..
BASELINE_ALGORITHMS=none

# MLP environment variables
# No INFERENCE_OPTIMIZER_BACKEND needed — MLP/MNIST is the default backend.
export COMPRESSOR_TYPE=topk         # topk | quantization
                                    # (llmint8 not supported for MLP)
export COMPRESSOR_BACKEND=gpu       # cpu (compressors.py) | gpu (gpu_compressors.py)
export MNIST_DATA_ROOT=${HOME_DIR}/data  # path to MNIST dataset (downloaded here if absent)
export ETA_MIN0=0.1
export ETA_MIN1=0.05
export R_K=10.0

OUT_DIR=${HOME_DIR}/outputs/test/mlp
mkdir -p logs "$OUT_DIR"

# =============================================================================
# Validate arguments
# =============================================================================
PARAM=${1}
VALUE=${2}

if [ -z "$PARAM" ] || [ -z "$VALUE" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: sbatch run_one_value_mlp.sh <sigma|N> <value>"
    echo "  e.g. sbatch run_one_value_mlp.sh sigma 0.05"
    echo "       sbatch run_one_value_mlp.sh N 50"
    exit 1
fi

if [ "$PARAM" != "sigma" ] && [ "$PARAM" != "N" ]; then
    echo "ERROR: param must be 'sigma' or 'N', got '$PARAM'"
    exit 1
fi

echo "============================================================"
echo "Slurm job:   $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Backend:     MLP / MNIST (default)"
echo "Param:       $PARAM = $VALUE"
echo "T:           $T"
echo "fixed_N:     $FIXED_N  (used only when param=sigma)"
echo "fixed_sigma: $FIXED_SIGMA  (used only when param=N)"
echo "Compressor:  $COMPRESSOR_TYPE  (backend=$COMPRESSOR_BACKEND)"
echo "MNIST root:  $MNIST_DATA_ROOT"
echo "out_dir:     $OUT_DIR"
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
python -u ${HOME_DIR}/run_one_sweep_value.py \
    --param               "$PARAM"               \
    --value               "$VALUE"               \
    --fixed_N             "$FIXED_N"             \
    --fixed_sigma         "$FIXED_SIGMA"         \
    --T                   "$T"                   \
    --seed                "$SEED"                \
    --c_min               "$C_MIN"               \
    --c_max               "$C_MAX"               \
    --mu                  "$MU"                  \
    --epsilon             "$EPSILON"             \
    --out_dir             "$OUT_DIR"             \
    --baseline_algorithms "$BASELINE_ALGORITHMS"

echo "Done. Exit code: $?"
