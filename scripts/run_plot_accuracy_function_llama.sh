#!/bin/bash
# =============================================================================
# run_plot_accuracy_function_llama.sh
# Plots A_true, A_fast, and Gaussian-smoothed A_smooth for all 5 strategies.
#
# Per strategy cost:
#   n_grid=15 grid points × (1 A_true + 1 A_fast + smooth_M=30 A_fast) = 480 evals
#   ~77s/eval → ~10h for 5 strategies
#   → Run as parallel jobs, one per strategy
#
# Usage:  bash run_plot_accuracy_function_llama.sh
# =============================================================================

HOME_DIR=..
TIME_LIMIT="8:00:00"
N_GRID=15
SMOOTH_M=30
SIGMA=0.05
FAST_SAMPLES=50
FULL_SAMPLES=100

mkdir -p logs

# Per-strategy sigma — use the values determined from Stein tests
declare -A SIGMA_MAP
SIGMA_MAP[magnitude]="0.05"
SIGMA_MAP[topk_per_token]="0.05"
SIGMA_MAP[random]="0.05"
SIGMA_MAP[quantization]="0.20"
SIGMA_MAP[llmint8]="0.05"

STRATEGIES=(magnitude topk_per_token random quantization llmint8)

for STRATEGY in "${STRATEGIES[@]}"; do

    sbatch --partition=gpu \
           --gres=gpu:h200:1 \
           --nodes=1 --ntasks=1 \
           --cpus-per-task=4 --mem=48G \
           --time=${TIME_LIMIT} \
           --job-name=accfn_${STRATEGY} \
           --output=logs/accfn_${STRATEGY}_%j.out \
           --error=logs/accfn_${STRATEGY}_%j.err \
           --wrap="
mkdir -p logs
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=llama_3_1_8b
export LLM_MEASURED_PAYLOAD=prefill_raw
export LLM_MEASURED_BATCH=1
export MMLU_SEED=0

python ${HOME_DIR}/plot_accuracy_function.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --strategies ${STRATEGY} \
    --eta_min 0.25 \
    --n_grid ${N_GRID} \
    --sigma ${SIGMA_MAP[$STRATEGY]} \
    --smooth_M ${SMOOTH_M} \
    --fast_samples ${FAST_SAMPLES} \
    --full_samples ${FULL_SAMPLES} \
    --out_dir ${HOME_DIR}/outputs/accuracy_function/llama
"
    echo "Submitted: ${STRATEGY}  sigma=${SIGMA_MAP[$STRATEGY]}"

done

echo ""
echo "Submitted 5 jobs. Monitor with: squeue -u \$USER"
echo "Plots saved to outputs/accuracy_function/llama/"
