#!/bin/bash
# =============================================================================
# run_mc_fill.sh
#
# Standalone adaptive-fill pass (mc_concavity.py fill) on already-collected
# ray sets, topping up any floor level c that has < min_box_samples usable
# rays. Safe to resubmit: cmd_fill checks each floor before loading the model
# and skips floors (or the whole job) that are already satisfied.
#
# Usage: bash scripts/run_mc_fill.sh
# =============================================================================
set -e
HOME_DIR=.
MIN_BOX_SAMPLES=100
FLOOR_STEP=0.1
TIME="08:00:00"
MEM="48G"

mkdir -p logs

submit() {
    local NAME=$1 MODEL=$2 DATASET=$3 METRIC=$4 STRATEGY=$5 RAYS_JSON=$6 \
          OUT_DIR=$7 TOL=$8 EXTRA=$9
    echo "Submitting fill: ${NAME}"
    sbatch --parsable \
        --partition=gpu --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
        --job-name=mcfill_${NAME} \
        --output=logs/mcfill_${NAME}_%j.out \
        --error=logs/mcfill_${NAME}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy
python ${HOME_DIR}/experiments/mc_concavity.py fill \\
    --rays_json ${RAYS_JSON} \\
    --model ${MODEL} --dataset ${DATASET} --metric ${METRIC} --strategy ${STRATEGY} \\
    --tol ${TOL} --min_box_samples ${MIN_BOX_SAMPLES} --adaptive_floor_step ${FLOOR_STEP} \\
    ${EXTRA} \\
    --out_dir ${OUT_DIR}"
}

PPL_EXTRA="--max_texts 128 --max_length 512 --batch_size 1"
ACC_EXTRA="--samples_per_subject 20 --n_shot 5 --max_length 512 --batch_size 16"

OUT=outputs/mc_concavity

submit gemma2b_topk_c4   google/gemma-2b sharegpt perplexity topk_per_token \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token/cuts4_merged_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"

submit gemma2b_reserve_c4 google/gemma-2b sharegpt perplexity llmint8_reserve \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve/cuts4_merged_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"

submit gemma7b_topk_c4   google/gemma-7b sharegpt perplexity topk_per_token \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/topk_per_token/cuts4_merged_rays.json \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"

submit gemma7b_reserve_c4 google/gemma-7b sharegpt perplexity llmint8_reserve \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/llmint8_reserve/cuts4_merged_rays.json \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"

submit llama_wikitext_topk_c4 meta-llama/Llama-3.1-8B wikitext perplexity topk_per_token \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/topk_per_token/cuts4_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"

submit llama_wikitext_reserve_c4 meta-llama/Llama-3.1-8B wikitext perplexity llmint8_reserve \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/llmint8_reserve/cuts4_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"

submit llama_mmlu_topk_c4 meta-llama/Llama-3.1-8B mmlu accuracy topk_per_token \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/topk_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy 0.02 "${ACC_EXTRA}"

submit llama_mmlu_reserve_c4 meta-llama/Llama-3.1-8B mmlu accuracy llmint8_reserve \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/reserve_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy 0.02 "${ACC_EXTRA}"

echo "Submitted 8 fill jobs (gemma-2b cuts7 topk fill deferred — resubmit once a slot frees, cluster caps at 9 in-flight jobs)."
