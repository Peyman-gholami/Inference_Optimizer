#!/bin/bash
# =============================================================================
# run_directions_only.sh
#
# Runs ONLY Test 3 (directional / ray profiles) from analyze_concavity.py.
# Much faster than the full run — skips Tests 1 & 2 entirely.
#
# Usage:  bash run_directions_only.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# PARAMETERS
# =============================================================================

N_GRID=20         # points along each ray
N_REPEAT=5        # repeats per point (for noise reduction)
N_DIRECTIONS=4    # number of rays (first always goes 0->1, rest are random)
RAY_START_MAX=0.25  # starts sampled from [0, RAY_START_MAX] per link
RAY_END_MIN=0.75    # ends sampled from [RAY_END_MIN, RAY_END_MAX] per link
RAY_END_MAX=1.0

MODEL="meta-llama/Llama-3.1-8B-Instruct"

TIME_RESNET="0:20:00"
TIME_LLM="1:30:00"
MEM="32G"

OUT_BASE="${HOME_DIR}/outputs/directions_only1"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

# ── ResNet ────────────────────────────────────────────────────────────────────
submit_resnet() {
    local COMP=$1
    local EXTRA_FLAGS="${2:-}"
    local OUT_DIR="${OUT_BASE}/resnet_${COMP}"

    echo "Submitting ResNet directions-only: compressor=${COMP} ${EXTRA_FLAGS}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_RESNET} \
        --job-name=directions_resnet_${COMP} \
        --output=logs/directions_resnet_${COMP}_%j.out \
        --error=logs/directions_resnet_${COMP}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID started on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_CUTPOINTS=8,14,21
export RESNET_FAST_BATCHES=1
export COMPRESSOR_TYPE=${COMP}
export COMPRESSOR_BACKEND=gpu

python ${HOME_DIR}/analyze_concavity.py \\
    --backend resnet \\
    --compressor ${COMP} \\
    --compressor_backend gpu \\
    --n_grid ${N_GRID} \\
    --n_repeat ${N_REPEAT} \\
    --n_directions ${N_DIRECTIONS} \\
    --ray_start_max ${RAY_START_MAX} \\
    --ray_end_min ${RAY_END_MIN} \\
    --ray_end_max ${RAY_END_MAX} \\
    --out_dir ${OUT_DIR} \\
    --only_directions \\
    ${EXTRA_FLAGS}
")
    echo "  Submitted: job ${JOB}"
}

# ── LLM ───────────────────────────────────────────────────────────────────────
submit_llm() {
    local STRATEGY=$1
    local EXTRA_FLAGS="${2:-}"
    local OUT_DIR="${OUT_BASE}/llm_${STRATEGY}"

    echo "Submitting LLM directions-only: strategy=${STRATEGY} ${EXTRA_FLAGS}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_LLM} \
        --job-name=directions_llm_${STRATEGY} \
        --output=logs/directions_llm_${STRATEGY}_%j.out \
        --error=logs/directions_llm_${STRATEGY}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID started on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=${MODEL}
export LLM_ACTIVATION_STRATEGY=${STRATEGY}
export LLM_SPLIT_MODE=thirds
export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=llama_3_1_8b
export LLM_MEASURED_PAYLOAD=prefill_raw
export LLM_MEASURED_BATCH=1
export MMLU_FAST_SAMPLES_PER_SUBJECT=30
export MMLU_FULL_SAMPLES_PER_SUBJECT=50
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export MMLU_N_SHOT=5

python analyze_concavity_directions_only.py \
    --replot_directions outputs/directions_only1/resnet_topk/directional_results.json \
    --out_dir outputs/directions_only1/resnet_topk_replot

python ${HOME_DIR}/analyze_concavity_directions_only.py \\
    --backend llm \\
    --compressor ${STRATEGY} \\
    --n_grid ${N_GRID} \\
    --n_repeat ${N_REPEAT} \\
    --n_directions ${N_DIRECTIONS} \\
    --ray_start_max ${RAY_START_MAX} \\
    --ray_end_min ${RAY_END_MIN} \\
    --ray_end_max ${RAY_END_MAX} \\
    --out_dir ${OUT_DIR} \\
    --only_directions \\
    ${EXTRA_FLAGS}
")
    echo "  Submitted: job ${JOB}"
}

# =============================================================================
# SUBMIT JOBS
# =============================================================================

echo "=== ResNet directions-only ==="
submit_resnet topk
#submit_resnet quantization --quantization_aware
#submit_resnet llmint8

echo ""
echo "=== LLM directions-only ==="
#submit_llm topk_per_token
#submit_llm quantization --quantization_aware

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  ${OUT_BASE}/resnet_topk/directional_results.json"
echo "  ${OUT_BASE}/resnet_topk/directional_profiles.pdf"