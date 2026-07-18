#!/bin/bash
# =============================================================================
# run_mc_fill_chain_to07.sh
#
# Chains 8h `mc_concavity.py fill --floor_max 0.7` jobs via SLURM job-dependency
# (afterany) so each task keeps resubmitting itself until every floor up to 0.7
# reaches min_box_samples usable rays. Safe to resubmit: cmd_fill resumes from
# <stem>_fill.partial.json and exits immediately once satisfied, so trailing
# links in the chain are cheap no-ops.
#
# Covers all task/strategy combos EXCEPT llama_mmlu_reserve_c4, which already
# has a fill chain running (job 8068270, targeting the old default floor_max=0.9,
# a superset of 0.7) — do not resubmit it here to avoid two jobs writing the
# same *_fill.partial.json.
#
# Usage: bash scripts/run_mc_fill_chain_to07.sh
# =============================================================================
set -e
HOME_DIR=.
MIN_BOX_SAMPLES=100
FLOOR_STEP=0.1
FLOOR_MAX=0.7
TIME="08:00:00"
MEM="48G"
MAX_LINKS=3
# HF cache lives on scratch: the /projects HF_HOME is over its group disk quota,
# so model downloads (e.g. gemma-2b, not pre-cached) fail with Errno 122 there.
# Override with HF_CACHE=/some/path if needed.
HF_CACHE="${HF_CACHE:-/scratch/sirera.m/hf_cache}"

mkdir -p logs

submit_chain() {
    local NAME=$1 MODEL=$2 DATASET=$3 METRIC=$4 STRATEGY=$5 RAYS_JSON=$6 \
          OUT_DIR=$7 TOL=$8 EXTRA=$9
    local JOBID=""
    for i in $(seq 1 ${MAX_LINKS}); do
        local DEPARG=()
        if [ -n "${JOBID}" ]; then
            DEPARG=(--dependency=afterany:${JOBID})
        fi
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=gpu:a100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            "${DEPARG[@]}" \
            --job-name=mcfillchain07_${NAME} \
            --output=logs/mcfillchain07_${NAME}_link${i}_%j.out \
            --error=logs/mcfillchain07_${NAME}_link${i}_%j.err \
            --wrap="
set -e
echo \"[wrap] link ${i} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
export HF_HOME=${HF_CACHE}
export HF_TOKEN=\$(cat ${HF_CACHE}/token 2>/dev/null)
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy
python ${HOME_DIR}/experiments/mc_concavity.py fill \\
    --rays_json ${RAYS_JSON} \\
    --model ${MODEL} --dataset ${DATASET} --metric ${METRIC} --strategy ${STRATEGY} \\
    --tol ${TOL} --min_box_samples ${MIN_BOX_SAMPLES} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} \\
    ${EXTRA} \\
    --out_dir ${OUT_DIR}")
        echo "  link ${i}: ${NAME} -> job ${JOBID}"
    done
}

PPL_EXTRA="--max_texts 128 --max_length 512 --batch_size 1"
ACC_EXTRA="--samples_per_subject 20 --n_shot 5 --max_length 512 --batch_size 16"

OUT=outputs/mc_concavity

# TASKS: space-separated subset of task names to submit this invocation
# (default: the first batch). Pass TASKS="all" to submit everything at once,
# or TASKS="name1 name2" to pick specific ones.
TASKS="${TASKS:-gemma2b_topk_c4 gemma2b_reserve_c4 gemma2b_topk_c7}"

want() {
    [ "${TASKS}" = "all" ] && return 0
    for t in ${TASKS}; do [ "$t" = "$1" ] && return 0; done
    return 1
}

want gemma2b_topk_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: gemma2b_topk_c4"
submit_chain gemma2b_topk_c4 google/gemma-2b sharegpt perplexity topk_per_token \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token/cuts4_merged_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"
}

want gemma2b_reserve_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: gemma2b_reserve_c4"
submit_chain gemma2b_reserve_c4 google/gemma-2b sharegpt perplexity llmint8_reserve \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve/cuts4_merged_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"
}

want gemma2b_topk_c7 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: gemma2b_topk_c7"
submit_chain gemma2b_topk_c7 google/gemma-2b sharegpt perplexity topk_per_token \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token/cuts7_merged_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"
}

want gemma2b_reserve_c7 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: gemma2b_reserve_c7"
submit_chain gemma2b_reserve_c7 google/gemma-2b sharegpt perplexity llmint8_reserve \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve/cuts7_merged_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"
}

want gemma7b_topk_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: gemma7b_topk_c4"
submit_chain gemma7b_topk_c4 google/gemma-7b sharegpt perplexity topk_per_token \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/topk_per_token/cuts4_merged_rays.json \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"
}

want gemma7b_reserve_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: gemma7b_reserve_c4"
submit_chain gemma7b_reserve_c4 google/gemma-7b sharegpt perplexity llmint8_reserve \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/llmint8_reserve/cuts4_merged_rays.json \
    ${OUT}/google_gemma-7b/sharegpt_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"
}

want llama_wikitext_topk_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: llama_wikitext_topk_c4"
submit_chain llama_wikitext_topk_c4 meta-llama/Llama-3.1-8B wikitext perplexity topk_per_token \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/topk_per_token/cuts4_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/topk_per_token 0.001 "${PPL_EXTRA}"
}

want llama_wikitext_reserve_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: llama_wikitext_reserve_c4"
submit_chain llama_wikitext_reserve_c4 meta-llama/Llama-3.1-8B wikitext perplexity llmint8_reserve \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/llmint8_reserve/cuts4_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/llmint8_reserve 0.001 "${PPL_EXTRA}"
}

want llama_mmlu_topk_c4 && {
echo "Chaining fill (floor_max=${FLOOR_MAX}) for: llama_mmlu_topk_c4"
submit_chain llama_mmlu_topk_c4 meta-llama/Llama-3.1-8B mmlu accuracy topk_per_token \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/topk_merged_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy 0.02 "${ACC_EXTRA}"
}

echo "SKIPPED llama_mmlu_reserve_c4 (already running as chain 8068270, floor_max=0.9 default)."

echo "Done submitting chains. (TASKS=${TASKS})"
