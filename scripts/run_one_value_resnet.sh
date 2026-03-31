#!/bin/bash
# =============================================================================
# run_one_value.sh  —  Run a single sweep value as one Slurm job
# =============================================================================
#
# Usage
# -----
#   sbatch run_one_value_resnet.sh sigma 0.05
#   sbatch run_one_value_resnet.sh N 10
#
# Arguments
#   $1  param   : 'sigma' or 'N'
#   $2  value   : the parameter value to run (e.g. 0.05 or 10)
#
# Example — run all sigma values mansbatch run_one_value.sh sigma 0.001ually, one job each:
#   
#   sbatch run_one_value_resnet.sh sigma 0.01
#   sbatch run_one_value_resnet.sh sigma 0.05
#   sbatch run_one_value_resnet.sh sigma 0.1
#   sbatch run_one_value_resnet.sh sigma 0.5
#
# Example — run all N values:
#   sbatch run_one_value_resnet.sh N 2
#   sbatch run_one_value_resnet.sh N 5
#   sbatch run_one_value_resnet.sh N 10
#   sbatch run_one_value_resnet.sh N 20
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --output=logs/sweep_%x_%j.out
#SBATCH --error=logs/sweep_%x_%j.err

# =============================================================================
# CONFIG — edit these before submitting
# =============================================================================
T=50
SEED=12
C_MIN=10.0
C_MAX=150.0
MU=3.0
EPSILON=0.1
FIXED_N=5           # N used when sweeping sigma   (ResNet: keep small, e.g. 5)
FIXED_SIGMA=0.05    # sigma used when sweeping N
HOME_DIR=..
BASELINE_ALGORITHMS=none

export COMPRESSOR_TYPE=llmint8      # topk | quantization | llmint8
export COMPRESSOR_BACKEND=gpu       # cpu (compressors.py) | gpu (gpu_compressors.py)

OUT_DIR=${HOME_DIR}/outputs/test/resnet
mkdir -p logs "$OUT_DIR"

# ResNet environment variables
export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_FAST_BATCHES=1
export RESNET_CUTPOINTS=8,14,21

export ETA_MIN0=0.125
export ETA_MIN1=0.125
export R_K=10.0

# =============================================================================
# Validate arguments
# =============================================================================
PARAM=${1}
VALUE=${2}

if [ -z "$PARAM" ] || [ -z "$VALUE" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: sbatch run_one_value.sh <sigma|N> <value>"
    echo "  e.g. sbatch run_one_value.sh sigma 0.05"
    echo "       sbatch run_one_value.sh N 10"
    exit 1
fi

if [ "$PARAM" != "sigma" ] && [ "$PARAM" != "N" ]; then
    echo "ERROR: param must be 'sigma' or 'N', got '$PARAM'"
    exit 1
fi

# Set a descriptive job name visible in squeue
# (overrides the --job-name default above)
#SBATCH --job-name=sweep_${PARAM}_${VALUE}

echo "============================================================"
echo "Slurm job:   $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Param:       $PARAM = $VALUE"
echo "T:           $T"
echo "fixed_N:     $FIXED_N  (used only when param=sigma)"
echo "fixed_sigma: $FIXED_SIGMA  (used only when param=N)"
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
    --param               "$PARAM"                \
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

