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
#
#   # Run both and plot automatically (submits all jobs with dependencies):
#   bash launch_llm_sigma_sweep.sh
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
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=llm_sweep
#SBATCH --array=0-8%6
#SBATCH --output=logs/llm_sweep_%A_%a.out
#SBATCH --error=logs/llm_sweep_%A_%a.err

# =============================================================================
# CONFIG — edit before submitting
# =============================================================================
T=30
SEED=12
C_MIN=10.0
C_MAX=150.0
MU=3.0
EPSILON=0.1
FIXED_N=3               # N used during sigma sweep (LLM: 3 is the default)
FIXED_SIGMA=0.05        # sigma used during N sweep
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/llm_sweep
BASELINE_ALGORITHMS=none

# ------------------------------------------------------------------
# LLM environment
# ------------------------------------------------------------------
export INFERENCE_OPTIMIZER_BACKEND=llm
export COMPRESSOR_BACKEND=gpu       # cpu (compressors.py) | gpu (gpu_compressors.py)
                                    # Note: LLM backend uses LLMActivationCompressor
                                    # (GPU-native hooks) regardless of this setting.
                                    # COMPRESSOR_BACKEND applies only to MLP/ResNet
export LLM_MODEL_NAME=unsloth/gemma-2b-it   # use unsloth mirror (no gated token needed)
export LLM_ACTIVATION_STRATEGY=quantization # topk_per_token | magnitude | random | quantization | llmint8
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=0.25
export ETA_MIN1=0.25
export MMLU_FAST_SAMPLES_PER_SUBJECT=5
export MMLU_FULL_SAMPLES_PER_SUBJECT=10
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export R_K=10.0

# =============================================================================
# Task index -> (param, value) mapping
#
# Index  0-4 : sigma sweep  (fixed N = FIXED_N)
# Index  5-8 : N sweep      (fixed sigma = FIXED_SIGMA)
# =============================================================================
SIGMA_VALUES=(0.001 0.01 0.05 0.1 0.5)
N_VALUES=(2 3 5 10)

IDX=$SLURM_ARRAY_TASK_ID

if [ "$IDX" -le 4 ]; then
    PARAM=sigma
    VALUE=${SIGMA_VALUES[$IDX]}
    N_ACTUAL=$FIXED_N
    SIGMA_ACTUAL=$VALUE
else
    PARAM=N
    N_IDX=$(( IDX - 5 ))
    VALUE=${N_VALUES[$N_IDX]}
    N_ACTUAL=$VALUE
    SIGMA_ACTUAL=$FIXED_SIGMA
fi

mkdir -p logs "$OUT_DIR"

# =============================================================================
# Activate conda env
# =============================================================================
eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

# =============================================================================
# Print job info
# =============================================================================
# Runtime estimate: T slots × 2 algos × 2N evals × ~2 s/eval / 60
EVALS_PER_SLOT=$(( 2 * 2 * N_ACTUAL ))
EST_MIN=$(( T * EVALS_PER_SLOT * 2 / 60 ))

echo "============================================================"
echo "Slurm array job:  $SLURM_ARRAY_JOB_ID  task: $IDX"
echo "Node:             $SLURM_NODELIST"
echo "Backend:          LLM ($LLM_MODEL_NAME)"
echo "Param:            $PARAM = $VALUE"
echo "sigma:            $SIGMA_ACTUAL   N: $N_ACTUAL"
echo "T:                $T"
echo "Strategy:         $LLM_ACTIVATION_STRATEGY"
echo "Split mode:       $LLM_SPLIT_MODE"
echo "out_dir:          $OUT_DIR"
echo "Runtime estimate: ~${EST_MIN} min"
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
