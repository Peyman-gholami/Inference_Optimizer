#!/bin/bash
# =============================================================================
# run_fill_t5.sh  (2026-07-18)
#
# Adaptive-fill pass for the Flan-T5-base / SST-2 concavity rays, which the
# density check flagged as thin at high eta-floors (usable rays drop <100 above
# floor ~0.4 for accuracy, ~0.5 for ppl_score). The flant5 harness gained a
# `fill` subcommand (flant5_sst2_concavity_test/flan_t5_sst2_concavity.py fill)
# that mirrors mc_concavity.py fill: it checks per-floor coverage up to
# FLOOR_MAX and, only if under-filled, loads the model and samples extra rays
# inside the high-floor sub-cubes, resuming from <stem>_fill.partial.json.
#
# T5 is topk-ONLY (no llmint8/quantization rays exist), so only two fills:
# accuracy and ppl_score.
#
# Dataset: the harness's HF loader (load_dataset("glue","sst2")) fails in the
# `easy` env, so we pass a pre-materialized local validation jsonl (the full
# 872-example split, matching the original max_samples=0 run) via --dataset_path.
# Model flan-t5-base downloads on demand to the scratch HF cache (compute nodes
# have network).
#
# Args mirror the original runs' summary.json: n_points 7, max_input_length 128,
# max_samples 0 (full split), seed 0, tol read from each rays.json (0.01 acc /
# 0.001 ppl). Same 3-link afterany chain + scratch HF cache as
# run_fill_missing.sh; trailing links are cheap no-ops once every floor to
# FLOOR_MAX has >= MIN usable rays.
#
# Usage: bash scripts/run_fill_t5.sh
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

T5=flant5_sst2_concavity_test/flan_t5_sst2_concavity.py
DATASET=flant5_sst2_concavity_test/data/sst2_validation.jsonl
OUT=outputs/mc_concavity/google__flan-t5-base
COMMON="--dataset_path ${DATASET} --n_points 7 --max_input_length 128 --max_samples 0 --seed 0"

submit_chain() {
    local NAME=$1 RAYS_JSON=$2 OUT_DIR=$3
    local JOBID=""
    for i in $(seq 1 ${MAX_LINKS}); do
        local DEPARG=()
        [ -n "${JOBID}" ] && DEPARG=(--dependency=afterany:${JOBID})
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=gpu:a100:1 \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            "${DEPARG[@]}" \
            --job-name=fillt5_${NAME} \
            --output=logs/fillt5_${NAME}_link${i}_%j.out \
            --error=logs/fillt5_${NAME}_link${i}_%j.err \
            --wrap="
set -e
echo \"[wrap] link ${i} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
export HF_HOME=${HF_CACHE}
export HF_TOKEN=\$(cat ${HF_CACHE}/token 2>/dev/null)
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy
python ${HOME_DIR}/${T5} fill \\
    --rays_json ${RAYS_JSON} \\
    --out_dir ${OUT_DIR} \\
    --min_box_samples ${MIN} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} \\
    ${COMMON}")
        echo "  link ${i}: ${NAME} -> job ${JOBID}"
    done
}

echo "Submitting fill chain: t5_sst2_accuracy"
submit_chain t5_sst2_accuracy \
    ${OUT}/sst2_accuracy/topk/cuts3/google__flan-t5-base__sst2__accuracy__topk__cuts3_rays.json \
    ${OUT}/sst2_accuracy/topk/cuts3

echo "Submitting fill chain: t5_sst2_ppl_score"
submit_chain t5_sst2_ppl_score \
    ${OUT}/sst2_ppl_score/topk/cuts3/google__flan-t5-base__sst2__ppl_score__topk__cuts3_rays.json \
    ${OUT}/sst2_ppl_score/topk/cuts3

echo "Done. Monitor: squeue -u \$USER ; tail logs/fillt5_*.out"
