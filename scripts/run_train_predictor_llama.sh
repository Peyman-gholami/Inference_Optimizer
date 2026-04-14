#!/bin/bash
# =============================================================================
# run_train_predictor_llama.sh
# Fits AccuracyPredictor for each Llama strategy in parallel.
#
# One-time offline cost per strategy (~2-9h depending on n_grid).
# After this, PredictorGradientOracle replaces SteinGradientOracle
# and experiments run in minutes instead of hours.
#
# Usage:  bash run_train_predictor_llama.sh
# =============================================================================

HOME_DIR=..
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ETA_MIN=0.0
FULL_SAMPLES=100
FAST_SAMPLES=50

mkdir -p logs
mkdir -p ${HOME_DIR}/models/predictors

# Per-strategy n_grid: quantization needs finer grid to capture discrete jumps
declare -A N_GRID
N_GRID[topk_per_token]=15
N_GRID[magnitude]=15
N_GRID[random]=15
N_GRID[quantization]=20
N_GRID[llmint8]=15

# llmint8 excluded — accuracy is flat, predictor would be trivially constant

#STRATEGIES=(topk_per_token magnitude random quantization)
STRATEGIES=(llmint8)

for STRATEGY in "${STRATEGIES[@]}"; do

    GRID=${N_GRID[$STRATEGY]}
    # Estimated time: n_grid^2 * 77s (per eval) / 3600
    # topk/mag/rand: 15^2 * 77 / 3600 ≈ 4.8h → use 6h
    # quantization:  20^2 * 77 / 3600 ≈ 8.6h → use 10h (> cluster limit, use 8h + resume)
    if [ "${STRATEGY}" == "topk_per_token" ]; then
        TIME="6:00:00"
    else
        TIME="8:00:00"
    fi

    sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=48G \
        --time=${TIME} \
        --job-name=predictor_${STRATEGY} \
        --output=logs/predictor_${STRATEGY}_%j.out \
        --error=logs/predictor_${STRATEGY}_%j.err \
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

python ${HOME_DIR}/train_accuracy_predictor.py \
    --model_name  '${MODEL}' \
    --strategy    ${STRATEGY} \
    --eta_min     ${ETA_MIN} \
    --n_grid      ${GRID} \
    --n_repeat    1 \
    --full_samples ${FULL_SAMPLES} \
    --fast_samples ${FAST_SAMPLES} \
    --grid_eta_min 0.0\
    --epochs      3000 \
    --mono_weight 0.1 \
    --save_path   ${HOME_DIR}/models/predictors/llama_${STRATEGY}_etamin${ETA_MIN}_grid${GRID}.pt
"
    echo "Submitted: ${STRATEGY}  n_grid=${GRID}  time=${TIME}"

done

echo ""
echo "All predictor training jobs submitted. Monitor with: squeue -u \$USER"
echo "Models will be saved to: ${HOME_DIR}/models/predictors/"
