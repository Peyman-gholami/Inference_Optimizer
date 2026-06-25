#!/bin/bash
# =============================================================================
# run_mc_more_points.sh
#
# Add MORE concavity samples per task, targeted at the sparse high-eta-floor
# regime (the region where the Jensen floor-sweep in experiments/jensen_concavity.py
# runs out of usable rays and marks points with an "x"). Three levers, all of
# which densify exactly where data is thin, without biasing the low-floor
# estimate:
#
#   1. FRESH SEED (=7)  -> a new independent batch of isotropic Phase-1 rays.
#      mc_concavity.py has no resume and seed-0/1 regenerate identical rays, so a
#      new seed is the only way to genuinely add rays. They merge with the
#      existing rays in jensen_concavity.py (content-hash dedup across all files).
#
#   2. FINER t-GRID  (--n_points 7 -> 13).  At a high floor only the grid points
#      with min(eta) >= c survive; a ray needs >= 3 survivors to yield a Jensen
#      triple. Doubling the points per ray keeps many more rays "usable" at high
#      floors -> directly shrinks the crossed (low-data) tail of the sweep.
#
#   3. BIGGER PHASE-2 (--phase2_rays 150).  Phase 2 resamples rays *inside* the
#      [eta_min, 1]^n candidate sub-cube, i.e. precisely the high-eta region the
#      crosses live in. jensen_concavity.py already folds phase2_rays in.
#
# Safety: jensen_concavity.py now harvests *_phase1/_phase2.partial.json
# checkpoints too, so a reserve job (slow, ~3x topk) that overruns the 8h gpu
# walltime mid-phase still contributes every completed ray.
#
# Outputs land in NEW dirs (cuts<N>_more) so existing seed-0 results are
# untouched; re-run experiments/jensen_concavity.py afterwards to fold them in.
#
# Usage:  bash scripts/run_mc_more_points.sh
# =============================================================================

HOME_DIR=..

# ── Monte-Carlo parameters (denser than run_mc_concavity.sh) ───────────────
N_RAYS=120          # fresh isotropic Phase-1 rays (unbiased, new seed)
N_POINTS=13         # finer t-grid -> more points survive a high floor
PHASE2_RAYS=150     # targeted resampling inside the upper sub-cube (the crosses)
TARGET=0.95
MIN_USABLE=30
SEED=7

# perplexity eval
MAX_TEXTS=128
MAX_LENGTH=512
BATCH_SIZE=1
PPL_TOL=0.001

# mmlu eval
SAMPLES_PER_SUBJECT=20
N_SHOT=5
ACC_TOL=0.02

STRATEGIES=("topk_per_token" "llmint8_reserve")

TIME="08:00:00"
MEM="48G"
OUT_BASE="${HOME_DIR}/outputs/mc_concavity"
# =============================================================================

mkdir -p logs

# submit MODEL DATASET METRIC TOL EXTRA_ARGS NCUTS_CSV
submit() {
    local MODEL=$1 DATASET=$2 METRIC=$3 TOL=$4 EXTRA=$5 NCUTS_CSV=$6
    local MODEL_SLUG; MODEL_SLUG=$(echo "${MODEL}" | tr '/' '_')

    local PY_CMDS=""
    IFS=',' read -ra NCUTS <<< "${NCUTS_CSV}"
    for N in "${NCUTS[@]}"; do
        local OUT_DIR="${OUT_BASE}/${MODEL_SLUG}/${DATASET}_${METRIC}/${STRATEGY}/cuts${N}_more"
        PY_CMDS="${PY_CMDS}
mkdir -p ${OUT_DIR}
python ${HOME_DIR}/experiments/mc_concavity.py run \\
    --model ${MODEL} --dataset ${DATASET} --metric ${METRIC} \\
    --strategy ${STRATEGY} --n_cuts ${N} \\
    --n_rays ${N_RAYS} --n_points ${N_POINTS} --phase2_rays ${PHASE2_RAYS} \\
    --target ${TARGET} --min_usable ${MIN_USABLE} --tol ${TOL} --seed ${SEED} \\
    ${EXTRA} \\
    --out_dir ${OUT_DIR}
"
    done

    echo "Submitting: ${MODEL} ${DATASET}/${METRIC} ${STRATEGY} cuts={${NCUTS_CSV}} seed=${SEED}"
    sbatch --parsable \
        --partition=gpu --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
        --job-name=mcmore_${MODEL_SLUG}_${DATASET}_${STRATEGY} \
        --output=logs/mcmore_${MODEL_SLUG}_${DATASET}_${STRATEGY}_%j.out \
        --error=logs/mcmore_${MODEL_SLUG}_${DATASET}_${STRATEGY}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy
${PY_CMDS}"
}

PPL_EXTRA="--max_texts ${MAX_TEXTS} --max_length ${MAX_LENGTH} --batch_size ${BATCH_SIZE}"
ACC_EXTRA="--samples_per_subject ${SAMPLES_PER_SUBJECT} --n_shot ${N_SHOT} --max_length ${MAX_LENGTH} --batch_size 16"

for STRATEGY in "${STRATEGIES[@]}"; do
    submit "google/gemma-2b"          sharegpt perplexity "${PPL_TOL}" "${PPL_EXTRA}" "4,7"
    submit "google/gemma-7b"          sharegpt perplexity "${PPL_TOL}" "${PPL_EXTRA}" "4"
    submit "meta-llama/Llama-3.1-8B"  mmlu     accuracy   "${ACC_TOL}" "${ACC_EXTRA}" "4"
    submit "meta-llama/Llama-3.1-8B"  wikitext perplexity "${PPL_TOL}" "${PPL_EXTRA}" "4"
done

echo ""
echo "All jobs submitted. Monitor: squeue -u \$USER"
echo "New rays land under ${OUT_BASE}/<model>/<dataset>_<metric>/<strategy>/cuts<N>_more/"
echo "Fold them into the plots with:"
echo "  python ${HOME_DIR}/experiments/jensen_concavity.py"
