#!/bin/bash
# =============================================================================
# run_mc_fill_chain_t5.sh
#
# Adaptive-fill chains for the FLAN-T5 / SST-2 concavity runs (topk, cuts=3,
# metrics: accuracy + ppl_score). Same chain pattern as
# run_mc_fill_chain_to07.sh: 8h `fill --floor_max 0.7` jobs linked via SLURM
# afterany dependencies; fill resumes from <stem>_fill.partial.json and exits
# immediately once every floor up to 0.7 has >= MIN_BOX_SAMPLES usable rays,
# so trailing links are cheap no-ops.
#
# Coverage deficit as of 2026-07-16 (240 rays each, no fill_rays yet;
# python experiments/check_ray_coverage.py <rays.json>):
#   accuracy : floor 0.4 -> 90, 0.5 -> 43, 0.6 -> 10, 0.7 -> 3   usable
#   ppl_score: floor 0.4 -> 108, 0.5 -> 55, 0.6 -> 18, 0.7 -> 8  usable
#
# !! The T5/SST-2 evaluation harness is NOT in this repo (the original runs
# came from an external T5 variant of mc_concavity.py — this repo's version
# only supports wikitext/sharegpt/mmlu). Point T5_MC_SCRIPT at that script
# (it must provide the same `fill` subcommand) and T5_ENV at its conda env:
#   T5_MC_SCRIPT=/path/to/mc_concavity_t5.py bash scripts/run_mc_fill_chain_t5.sh
#
# Fill args mirror the original T5 run (from *_summary.json): n_points=7,
# seed=0, tol=0.01, split=validation, max_input_length=128, max_samples=0
# (full 872-example split), strategy topk on 3 encoder transfer points.
# =============================================================================
set -e
HOME_DIR=.
T5_MC_SCRIPT="${T5_MC_SCRIPT:?set T5_MC_SCRIPT to the T5 variant of mc_concavity.py (not in this repo)}"
T5_ENV="${T5_ENV:-easy}"
MIN_BOX_SAMPLES=100
FLOOR_STEP=0.1
FLOOR_MAX=0.7
TIME="08:00:00"
MEM="48G"
MAX_LINKS=3
GRES="${GRES:-gpu:a100:1}"   # flan-t5-base is small; GRES=gpu:v100:1 also fine

OUT=outputs/mc_concavity/google__flan-t5-base
T5_EXTRA="${T5_EXTRA:---max_samples 0 --max_input_length 128}"

mkdir -p logs

submit_chain() {
    local NAME=$1 METRIC=$2 RAYS_JSON=$3 OUT_DIR=$4
    local JOBID=""
    for i in $(seq 1 ${MAX_LINKS}); do
        local DEPARG=()
        if [ -n "${JOBID}" ]; then
            DEPARG=(--dependency=afterany:${JOBID})
        fi
        JOBID=$(sbatch --parsable \
            --partition=gpu --gres=${GRES} \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
            "${DEPARG[@]}" \
            --job-name=mcfillt5_${NAME} \
            --output=logs/mcfillt5_${NAME}_link${i}_%j.out \
            --error=logs/mcfillt5_${NAME}_link${i}_%j.err \
            --wrap="
set -e
echo \"[wrap] link ${i} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate ${T5_ENV}
# Stop test: skip the link entirely if coverage is already satisfied.
if python ${HOME_DIR}/experiments/check_ray_coverage.py ${RAYS_JSON} \
      --min ${MIN_BOX_SAMPLES} --floor_max ${FLOOR_MAX} --floor_step ${FLOOR_STEP}; then
    echo \"[wrap] coverage already satisfied, exiting.\" >&2
    exit 0
fi
python ${T5_MC_SCRIPT} fill \\
    --rays_json ${RAYS_JSON} \\
    --model google/flan-t5-base --dataset sst2 --metric ${METRIC} \\
    --strategy topk --n_points 7 --seed 0 --tol 0.01 \\
    --min_box_samples ${MIN_BOX_SAMPLES} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} \\
    ${T5_EXTRA} \\
    --out_dir ${OUT_DIR}
python ${HOME_DIR}/experiments/check_ray_coverage.py ${RAYS_JSON} \
    --min ${MIN_BOX_SAMPLES} --floor_max ${FLOOR_MAX} --floor_step ${FLOOR_STEP} || true")
        echo "  link ${i}: ${NAME} -> job ${JOBID}"
    done
}

echo "Chaining T5 fill (floor_max=${FLOOR_MAX}) for: t5_sst2_accuracy"
submit_chain t5_sst2_accuracy accuracy \
    ${OUT}/sst2_accuracy/topk/cuts3/google__flan-t5-base__sst2__accuracy__topk__cuts3_rays.json \
    ${OUT}/sst2_accuracy/topk/cuts3

echo "Chaining T5 fill (floor_max=${FLOOR_MAX}) for: t5_sst2_ppl_score"
submit_chain t5_sst2_ppl_score ppl_score \
    ${OUT}/sst2_ppl_score/topk/cuts3/google__flan-t5-base__sst2__ppl_score__topk__cuts3_rays.json \
    ${OUT}/sst2_ppl_score/topk/cuts3

echo "Done submitting T5 fill chains."
