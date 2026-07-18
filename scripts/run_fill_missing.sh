#!/bin/bash
# =============================================================================
# run_fill_missing.sh  (2026-07-17)
#
# Fills the two LLM concavity groups that the density check
# (`check_ray_coverage.py --discover`) flagged as under-covered and that are
# actually fixable with the in-repo `mc_concavity.py fill`:
#
#   1. llama_mmlu_reserve_c4  -- genuinely thin (70 unique usable rays at floor
#      0.7 < 100). Its prior fill silently no-op'd because cmd_fill was
#      double-counting the persisted fill_rays against the _fill.partial.json;
#      that dedup bug is now fixed in experiments/mc_concavity.py, so the resume
#      correctly sees floors 0.5/0.6/0.7 under-filled.
#   2. gemma2b_topk_c7  -- base rebuilt: cuts7_merged_rays.json is CORRUPT
#      (n_cuts=7 but 4-dim data); cuts7/cuts7_clean_base_rays.json is the real
#      450 unique 7-dim rays. Very thin at high floors (0 usable at 0.7).
#
# Same 3-link afterany chain + scratch HF cache as run_mc_fill_chain_to07.sh.
# fill resumes from <stem>_fill.partial.json and exits once every floor to 0.7
# has >= MIN usable rays, so trailing links are cheap no-ops.
#
# NOT covered here (need separate handling, see the handoff notes):
#   - t5 sst2 accuracy/ppl_score : the flant5 harness has no `fill` subcommand.
#   - resnet (topk/quant/llmint8): different backend (10 directional rays).
#
# Usage: bash scripts/run_fill_missing.sh
# =============================================================================
set -e
HOME_DIR=.
MIN=100
FLOOR_STEP=0.1
FLOOR_MAX=0.7
TIME="08:00:00"
MEM="48G"
MAX_LINKS=3
HF_CACHE="${HF_CACHE:-/scratch/sirera.m/hf_cache}"
mkdir -p logs

OUT=outputs/mc_concavity
PPL_EXTRA="--max_texts 128 --max_length 512 --batch_size 1"
ACC_EXTRA="--samples_per_subject 20 --n_shot 5 --max_length 512 --batch_size 16"

submit_chain() {
    local NAME=$1 MODEL=$2 DATASET=$3 METRIC=$4 STRATEGY=$5 TOL=$6 \
          RAYS_JSON=$7 OUT_DIR=$8 EXTRA=$9
    local JOBID=""
    for i in $(seq 1 ${MAX_LINKS}); do
        local DEPARG=()
        [ -n "${JOBID}" ] && DEPARG=(--dependency=afterany:${JOBID})
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=gpu:a100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            "${DEPARG[@]}" \
            --job-name=fillmiss_${NAME} \
            --output=logs/fillmiss_${NAME}_link${i}_%j.out \
            --error=logs/fillmiss_${NAME}_link${i}_%j.err \
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
    --tol ${TOL} --min_box_samples ${MIN} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} \\
    ${EXTRA} \\
    --out_dir ${OUT_DIR}")
        echo "  link ${i}: ${NAME} -> job ${JOBID}"
    done
}

echo "Submitting fill chain: llama_mmlu_reserve_c4"
submit_chain llama_mmlu_reserve_c4 meta-llama/Llama-3.1-8B mmlu accuracy llmint8_reserve 0.02 \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/meta-llama__Llama-3.1-8B__mmlu__accuracy__llmint8_reserve__cuts4_rays.json \
    ${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy "${ACC_EXTRA}"

echo "Submitting fill chain: gemma2b_topk_c7 (clean rebuilt base -> fresh cuts7_fill/)"
submit_chain gemma2b_topk_c7 google/gemma-2b sharegpt perplexity topk_per_token 0.001 \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token/cuts7/cuts7_clean_base_rays.json \
    ${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token/cuts7_fill "${PPL_EXTRA}"

echo "Done. Monitor: squeue -u \$USER ; tail logs/fillmiss_*.out"
