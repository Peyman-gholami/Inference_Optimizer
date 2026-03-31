#!/bin/bash
# =============================================================================
# run_one_value_llm.sh  —  Run a single LLM sweep value as one Slurm job
# =============================================================================
#
# Usage
# -----
#   sbatch run_one_value_llm.sh sigma 0.05
#   sbatch run_one_value_llm.sh N 5
#
# Arguments
#   $1  param : 'sigma' or 'N'
#   $2  value : the parameter value to run (e.g. 0.05 or 5)
#
# Example — run all sigma values manually, one job each:
#   sbatch run_one_value_llm.sh sigma 0.001
#   sbatch run_one_value_llm.sh sigma 0.01
#   sbatch run_one_value_llm.sh sigma 0.05
#   sbatch run_one_value_llm.sh sigma 0.1
#   sbatch run_one_value_llm.sh sigma 0.5
#
# Example — run all N values:
#   sbatch run_one_value_llm.sh N 2
#   sbatch run_one_value_llm.sh N 3
#   sbatch run_one_value_llm.sh N 5
#   sbatch run_one_value_llm.sh N 10
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=llm_single
#SBATCH --output=logs/llm_single_%x_%j.out
#SBATCH --error=logs/llm_single_%x_%j.err

# =============================================================================
# CONFIG — edit before submitting
# =============================================================================
T=30
SEED=12
C_MIN=10.0
C_MAX=150.0
MU=3.0
EPSILON=0.1
FIXED_N=3               # N used when sweeping sigma   (LLM: keep small, e.g. 3)
FIXED_SIGMA=0.05        # sigma used when sweeping N
HOME_DIR=..
BASELINE_ALGORITHMS=none

# =============================================================================
# LLM environment variables
# =============================================================================
export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=unsloth/gemma-2b-it
export LLM_ACTIVATION_STRATEGY=topk_per_token   # topk_per_token | magnitude | random | quantization
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=0.25
export ETA_MIN1=0.25
export MMLU_FAST_SAMPLES_PER_SUBJECT=5
export MMLU_FULL_SAMPLES_PER_SUBJECT=10
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export R_K=10.0

OUT_DIR=${HOME_DIR}/outputs/test/llm
mkdir -p logs "$OUT_DIR"

# =============================================================================
# Validate arguments
# =============================================================================
PARAM=${1}
VALUE=${2}

if [ -z "$PARAM" ] || [ -z "$VALUE" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: sbatch run_one_value_llm.sh <sigma|N> <value>"
    echo "  e.g. sbatch run_one_value_llm.sh sigma 0.05"
    echo "       sbatch run_one_value_llm.sh N 5"
    exit 1
fi

if [ "$PARAM" != "sigma" ] && [ "$PARAM" != "N" ]; then
    echo "ERROR: param must be 'sigma' or 'N', got '$PARAM'"
    exit 1
fi

# =============================================================================
# Activate conda env
# =============================================================================
eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

echo "============================================================"
echo "Slurm job:   $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Backend:     LLM ($LLM_MODEL_NAME)"
echo "Param:       $PARAM = $VALUE"
echo "T:           $T"
echo "fixed_N:     $FIXED_N  (used only when param=sigma)"
echo "fixed_sigma: $FIXED_SIGMA  (used only when param=N)"
echo "Strategy:    $LLM_ACTIVATION_STRATEGY"
echo "Split mode:  $LLM_SPLIT_MODE"
echo "out_dir:     $OUT_DIR"
echo "============================================================"

# =============================================================================
# Run
# =============================================================================
python -u ${HOME_DIR}/run_one_sweep_value.py \
    --param               "$PARAM"                \
    --value               "$VALUE"                \
    --fixed_N             "$FIXED_N"              \
    --fixed_sigma         "$FIXED_SIGMA"          \
    --T                   "$T"                    \
    --seed                "$SEED"                 \
    --c_min               "$C_MIN"                \
    --c_max               "$C_MAX"                \
    --mu                  "$MU"                   \
    --epsilon             "$EPSILON"              \
    --out_dir             "$OUT_DIR"              \
    --baseline_algorithms "$BASELINE_ALGORITHMS"

EXIT_CODE=$?
echo "============================================================"
echo "Done. Exit code: $EXIT_CODE"
echo "Results saved to: $OUT_DIR"
echo "============================================================"
exit $EXIT_CODE
