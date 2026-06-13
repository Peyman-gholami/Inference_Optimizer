#!/bin/bash
# =============================================================================
# run_mc_mmlu_more.sh
#
# Accumulate MORE Monte-Carlo concavity rays for the Llama-3.1-8B MMLU task.
#
# mc_concavity.py has NO resume: a relaunch always loops rays from index 0, and
# seed=0 regenerates the *same* rays. So to genuinely add new rays we:
#   - use a fresh SEED (=1) -> independent random rays (also a fresh MMLU sample
#     draw, which is fine: each ray's concavity is judged on its own internally-
#     consistent y-curve, so seed-0 and seed-1 rays merge into one larger sample);
#   - write to a SEPARATE out_dir (cuts4_seed1) so the existing seed-0 partials
#     are preserved;
#   - set --phase2_rays 0 so the whole 8h walltime goes to isotropic Phase-1
#     rays (the full-cube probe that feeds global concavity + the scalar-floor
#     sweep) instead of the targeted Phase-2 detour;
#   - bump --batch_size 8 -> 16. Eval is only 120 prompts @ max_len 1744, so 16
#     fits a 40GB A100 comfortably (logits ~7GB + 16GB weights). Accuracy is
#     batch-invariant (last-token index uses attention_mask.sum-1), so results
#     are unchanged. For reserve this also cuts the per-forward mixed-precision
#     hook-call count in half.
#
# The gpu partition caps walltime at 8h. topk (~134 s/ray) finishes 150 rays in
# ~5.6h and writes a clean summary. reserve (~401 s/ray, 3x slower) will likely
# be killed mid-Phase-1, but the every-5-ray checkpoint (_phase1.partial.json)
# captures every completed ray for merging.
#
# Merge afterwards (model-free, CPU):
#   python experiments/mc_concavity.py etamin --rays_json <merged seed0+seed1 rays>
# (wrap the two partials' ray lists into one {"rays":[...], "n_cuts":4} file).
#
# Usage:  bash scripts/run_mc_mmlu_more.sh
# =============================================================================

HOME_DIR=..

MODEL="meta-llama/Llama-3.1-8B"
MODEL_SLUG="meta-llama_Llama-3.1-8B"
N_CUTS=4
N_RAYS=150          # Phase-1 rays (topk completes; reserve harvests via checkpoint)
N_POINTS=7
PHASE2_RAYS=0       # skip Phase-2: spend all budget on isotropic Phase-1 rays
TARGET=0.95
MIN_USABLE=30
ACC_TOL=0.02
SAMPLES_PER_SUBJECT=20
N_SHOT=5
MAX_LENGTH=512
BATCH_SIZE=16
SEED=1

TIME="08:00:00"
MEM="48G"
OUT_BASE="${HOME_DIR}/outputs/mc_concavity"

mkdir -p logs

for STRATEGY in "topk_per_token" "llmint8_reserve"; do
    OUT_DIR="${OUT_BASE}/${MODEL_SLUG}/mmlu_accuracy/${STRATEGY}/cuts${N_CUTS}_seed${SEED}"
    echo "Submitting: ${MODEL} mmlu/accuracy ${STRATEGY} cuts${N_CUTS} seed=${SEED} batch=${BATCH_SIZE}"
    sbatch --parsable \
        --partition=gpu --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
        --job-name=mcmmlu_more_${STRATEGY} \
        --output=logs/mcmmlu_more_${STRATEGY}_%j.out \
        --error=logs/mcmmlu_more_${STRATEGY}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy
mkdir -p ${OUT_DIR}
python ${HOME_DIR}/experiments/mc_concavity.py run \\
    --model ${MODEL} --dataset mmlu --metric accuracy \\
    --strategy ${STRATEGY} --n_cuts ${N_CUTS} \\
    --n_rays ${N_RAYS} --n_points ${N_POINTS} --phase2_rays ${PHASE2_RAYS} \\
    --target ${TARGET} --min_usable ${MIN_USABLE} --tol ${ACC_TOL} \\
    --samples_per_subject ${SAMPLES_PER_SUBJECT} --n_shot ${N_SHOT} \\
    --max_length ${MAX_LENGTH} --batch_size ${BATCH_SIZE} --seed ${SEED} \\
    --out_dir ${OUT_DIR}"
done

echo ""
echo "Submitted. Monitor: squeue -u \$USER"
echo "New rays land under ${OUT_BASE}/${MODEL_SLUG}/mmlu_accuracy/<strategy>/cuts${N_CUTS}_seed${SEED}/"
