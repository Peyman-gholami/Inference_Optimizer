#!/bin/bash
# =============================================================================
# run_param_sweep_llm_llama.sh
# Slurm batch script: run run_param_sweep.py for Llama-3.1-8B / MMLU
# =============================================================================
#
# Runs both sigma sweep and N sweep sequentially in one job.
# To run only one experiment, pass it as the first argument.
#
# Usage
# -----
#   # Both experiments (sigma + N sweep):
#   sbatch run_param_sweep_llm_llama.sh
#
#   # Only sigma sweep:
#   sbatch run_param_sweep_llm_llama.sh sigma
#
#   # Only N sweep:
#   sbatch run_param_sweep_llm_llama.sh N
#
# Note: for parallel execution use submit_llm_sweep.sh with
#       LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --job-name=llm_llama_sweep
#SBATCH --output=logs/llm_llama_sweep_%j.out
#SBATCH --error=logs/llm_llama_sweep_%j.err

# =============================================================================
# Experiment selection
# =============================================================================
EXPERIMENTS=${1:-sigma,N}

# =============================================================================
# CONFIG — edit before submitting
# =============================================================================
export LLM_ACTIVATION_STRATEGY=topk_per_token   # topk_per_token | magnitude | random | quantization | llmint8

T=30
SEED=12
C_MIN=10.0
C_MAX=150.0
MU=3.0
EPSILON=0.1
FIXED_N=3
FIXED_SIGMA=0.05
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/llm_sweep/llama_3_1_8b/${LLM_ACTIVATION_STRATEGY}
BASELINE_ALGORITHMS=none

# Sweep values
SIGMA_VALUES=0.001,0.01,0.05,0.1,0.5
N_VALUES=2,3,5,10

# =============================================================================
# LLM environment variables — Llama-3.1-8B
# =============================================================================
export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=0.25
export ETA_MIN1=0.25
export MMLU_FAST_SAMPLES_PER_SUBJECT=5
export MMLU_FULL_SAMPLES_PER_SUBJECT=10
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export R_K=10.0

# =============================================================================
# Setup
# =============================================================================

mkdir -p logs "$OUT_DIR"

# =============================================================================
# Activate conda env
# =============================================================================

eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap


echo "============================================================"
echo "Slurm job:    $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Backend:      LLM ($LLM_MODEL_NAME)"
echo "Experiments:  $EXPERIMENTS"
echo "T:            $T"
echo "fixed_N:      $FIXED_N  (sigma sweep)"
echo "fixed_sigma:  $FIXED_SIGMA  (N sweep)"
echo "sigma_values: $SIGMA_VALUES"
echo "N_values:     $N_VALUES"
echo "out_dir:      $OUT_DIR"
echo "Strategy:     $LLM_ACTIVATION_STRATEGY"
echo "Split mode:   $LLM_SPLIT_MODE"
echo "============================================================"

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
