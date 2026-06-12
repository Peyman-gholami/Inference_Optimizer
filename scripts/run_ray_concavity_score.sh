#!/bin/bash
# =============================================================================
# run_ray_concavity_score.sh
#
# Ray-based concavity test for the per-sample perplexity-ratio score:
#
#     score(s) = min{ 1, exp(Hbar_ref(s) - Hbar_comp(s)) }
#              = min{ 1, PPL_ref(s) / PPL_comp(s) }     # per held-out span
#
# evaluated along rays in eta-space with N cut-points (default 5).
#
# One SLURM job per (model, strategy).  Models: Llama-3.1-8B and Gemma-7B.
# Strategies: topk_per_token and llmint8.  Wall-clock limit: 8h.
#
# Usage:  bash run_ray_concavity_score.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# PARAMETERS
# =============================================================================
MODELS=("meta-llama/Llama-3.1-8B" "google/gemma-7b")
# llmint8_reserve won the scheme comparison (concave, graceful). Run it side by
# side with topk_per_token on identical rays so the two can be plotted together.
STRATEGIES=("llmint8_reserve" "topk_per_token")

# Run each (model, strategy) at both 5 and 8 activation cut-points. Each cut
# count needs its own rays file because every ray vector must have length
# == n_cuts (see load_rays() in ray_concavity_score.py). Both runs happen in a
# single SLURM job (sequentially), so this stays at 4 jobs total (2 models x
# 2 strategies), not 8.
N_CUTS_LIST=(5 8)   # activation cut-points (eta-vector dimension) to sweep
declare -A RAYS_FOR_CUTS=(
    [5]="${HOME_DIR}/experiments/rays_interesting_5.json"
    [8]="${HOME_DIR}/experiments/rays_interesting_8.json"
)

N_POINTS=16         # evaluation points along each ray (t in [0,1]); dense to resolve the knee
MAX_TEXTS=256       # WikiText-2 held-out paragraphs per evaluation
MAX_LENGTH=512
BATCH_SIZE=1
N_CHORDS=3          # random midpoint-chord concavity tests per ray (0 to skip)

TIME="08:00:00"     # two runs (5-cut then 8-cut) per job
MEM="48G"
OUT_BASE="${HOME_DIR}/outputs/ray_concavity_score"
# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

submit() {
    local MODEL=$1
    local STRATEGY=$2

    local MODEL_SLUG
    MODEL_SLUG=$(echo "${MODEL}" | tr '/' '_')

    # Build the per-cut python invocations for this job. Each cut count writes
    # to its own cuts<N>/ subdir so the (identically named) sweep CSVs and
    # summaries from the 5- and 8-cut runs don't overwrite each other.
    local PY_CMDS=""
    for N in "${N_CUTS_LIST[@]}"; do
        local RAYS_JSON="${RAYS_FOR_CUTS[$N]}"
        local OUT_DIR="${OUT_BASE}/${MODEL_SLUG}/${STRATEGY}/cuts${N}"
        echo "  cuts=${N}  rays=${RAYS_JSON}  out=${OUT_DIR}"
        PY_CMDS="${PY_CMDS}
mkdir -p ${OUT_DIR}
python ${HOME_DIR}/experiments/ray_concavity_score.py \\
    --models ${MODEL} \\
    --strategies ${STRATEGY} \\
    --n_cuts ${N} \\
    --n_points ${N_POINTS} \\
    --max_texts ${MAX_TEXTS} \\
    --max_length ${MAX_LENGTH} \\
    --batch_size ${BATCH_SIZE} \\
    --n_chords ${N_CHORDS} \\
    --out_dir ${OUT_DIR} \\
    --rays_json ${RAYS_JSON}
"
    done

    echo "Submitting ray-score concavity: model=${MODEL}  strategy=${STRATEGY}  cuts={${N_CUTS_LIST[*]}}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME} \
        --job-name=rayscore_${MODEL_SLUG}_${STRATEGY} \
        --output=logs/rayscore_${MODEL_SLUG}_${STRATEGY}_%j.out \
        --error=logs/rayscore_${MODEL_SLUG}_${STRATEGY}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID started on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy

# llmint8 uses the GPU-native compressor in src/core/gpu_compressors.py.
export COMPRESSOR_BACKEND=gpu
${PY_CMDS}")
    echo "  Submitted: job ${JOB}"
}

# =============================================================================
# SUBMIT JOBS  (one per model x strategy)
# =============================================================================
for MODEL in "${MODELS[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do
        submit "${MODEL}" "${STRATEGY}"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Per-run outputs under ${OUT_BASE}/<model>/<strategy>/cuts<N>/:"
echo "  <model>__<strategy>_summary.json   (all rays: scores + concavity tests)"
echo "  <model>__<strategy>_rays.png       (mean-score vs t per ray)"
echo "  <model>__<strategy>__<ray>_sweep.csv"
