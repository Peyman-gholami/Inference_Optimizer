#!/bin/bash
# =============================================================================
# run_train_predictor_llama_5shots.sh
# Fits AccuracyPredictor for each Llama strategy in parallel.
#
# One-time offline cost per strategy (~2-9h depending on n_grid).
# After this, PredictorGradientOracle replaces SteinGradientOracle
# and experiments run in minutes instead of hours.
#
# Usage:  bash run_train_predictor_llama_5shots.sh
# =============================================================================

HOME_DIR=..
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ETA_MIN=0.1
FULL_SAMPLES=100
FAST_SAMPLES=50

# 0 = zero-shot (default), 5 = standard 5-shot
# NOTE: predictors must be retrained when N_SHOT changes.
#       Experiment runs must use the same MMLU_N_SHOT value.
N_SHOT=5

mkdir -p logs
mkdir -p ${HOME_DIR}/models/predictors

# Per-strategy n_grid: reduced from 15/20 to 10/12 for 5-shot
# (5-shot evals are ~3-5x slower due to longer prompts)
declare -A N_GRID
N_GRID[topk_per_token]=10
N_GRID[magnitude]=10
N_GRID[random]=10
N_GRID[llmint8]=10
N_GRID[quantization]=12

# llmint8 excluded — accuracy is flat, predictor would be trivially constant

#STRATEGIES=(topk_per_token quantization llmint8)
STRATEGIES=(llmint8)

for STRATEGY in "${STRATEGIES[@]}"; do

    GRID=${N_GRID[$STRATEGY]}
    # Estimated time with 5-shot (~3-5x longer per eval):
    # topk/mag/rand: 10^2 * 300s / 3600 ≈ 8.3h → use 8h time limit
    # quantization:  12^2 * 300s / 3600 ≈ 12h  → use 8h + checkpoint/resume
    if [ "${STRATEGY}" == "quantization" ]; then
        TIME="8:00:00"
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
export MMLU_N_SHOT=${N_SHOT}

python ${HOME_DIR}/train_accuracy_predictor.py \
    --model_name  '${MODEL}' \
    --strategy    ${STRATEGY} \
    --eta_min     ${ETA_MIN} \
    --n_grid      ${GRID} \
    --n_repeat    1 \
    --full_samples ${FULL_SAMPLES} \
    --fast_samples ${FAST_SAMPLES} \
    --epochs      3000 \
    --mono_weight 0.1 \
    --grid_eta_min 0.1\
    --n_shot      ${N_SHOT} \
    --save_path   ${HOME_DIR}/models/predictors/llama_${STRATEGY}_etamin${ETA_MIN}_grid${GRID}_${N_SHOT}shot.pt
"
    echo "Submitted: ${STRATEGY}  n_grid=${GRID}  time=${TIME}"

done

echo ""
echo "All predictor training jobs submitted. Monitor with: squeue -u \$USER"
echo "Models will be saved to: ${HOME_DIR}/models/predictors/"
