#!/bin/bash
# =============================================================================
# run_concavity_analysis.sh
#
# Runs the concavity analysis to automatically detect per-link eta_min^true.
# Submit one job per compressor strategy.
#
# Usage:  bash run_concavity_analysis.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# PARAMETERS
# =============================================================================

# ResNet parameters (fast — use for initial analysis)
RESNET_N_GRID=20          # points per axis (20^1 * num_links = 40 evals per test)
RESNET_N_REPEAT=5         # repeats per point (for noise reduction)
RESNET_N_DIRECTIONS=10    # random rays for directional test
RESNET_NOISE_THRESH=0.005 # min slope to be considered increasing
RESNET_CONC_THRESH=0.005  # max D2 still considered concave
RESNET_SAFETY=0.05        # safety margin added to detected eta_min

# LLM parameters (slow — use smaller grid)
LLM_N_GRID=12
LLM_N_REPEAT=1
LLM_N_DIRECTIONS=5
LLM_NOISE_THRESH=0.01
LLM_CONC_THRESH=0.01
LLM_SAFETY=0.05

MODEL="meta-llama/Llama-3.1-8B-Instruct"

TIME_RESNET="1:00:00"
TIME_LLM="4:00:00"
MEM="32G"

OUT_BASE="${HOME_DIR}/outputs/concavity_analysis"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

# ── ResNet ────────────────────────────────────────────────────────────────────
submit_resnet() {
    local COMP=$1
    local EXTRA_FLAGS="${2:-}"   # optional extra flags e.g. --quantization_aware
    local OUT_DIR="${OUT_BASE}/resnet_${COMP}"

    echo "Submitting ResNet concavity analysis: compressor=${COMP} ${EXTRA_FLAGS}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_RESNET} \
        --job-name=concavity_resnet_${COMP} \
        --output=logs/concavity_resnet_${COMP}_%j.out \
        --error=logs/concavity_resnet_${COMP}_%j.err \
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
    --n_grid ${RESNET_N_GRID} \\
    --n_repeat ${RESNET_N_REPEAT} \\
    --n_directions ${RESNET_N_DIRECTIONS} \\
    --noise_threshold ${RESNET_NOISE_THRESH} \\
    --concavity_threshold ${RESNET_CONC_THRESH} \\
    --safety_margin ${RESNET_SAFETY} \\
    --out_dir ${OUT_DIR} \\
    ${EXTRA_FLAGS}
")
    echo "  Submitted: job ${JOB}"
}

# ── LLM ───────────────────────────────────────────────────────────────────────
submit_llm() {
    local STRATEGY=$1
    local EXTRA_FLAGS="${2:-}"
    local OUT_DIR="${OUT_BASE}/llm_${STRATEGY}"

    echo "Submitting LLM concavity analysis: strategy=${STRATEGY} ${EXTRA_FLAGS}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_LLM} \
        --job-name=concavity_llm_${STRATEGY} \
        --output=logs/concavity_llm_${STRATEGY}_%j.out \
        --error=logs/concavity_llm_${STRATEGY}_%j.err \
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

python ${HOME_DIR}/analyze_concavity.py \\
    --backend llm \\
    --compressor ${STRATEGY} \\
    --n_grid ${LLM_N_GRID} \\
    --n_repeat ${LLM_N_REPEAT} \\
    --n_directions ${LLM_N_DIRECTIONS} \\
    --noise_threshold ${LLM_NOISE_THRESH} \\
    --concavity_threshold ${LLM_CONC_THRESH} \\
    --safety_margin ${LLM_SAFETY} \\
    --out_dir ${OUT_DIR} \\
    ${EXTRA_FLAGS}
")
    echo "  Submitted: job ${JOB}"
}

# =============================================================================
# SUBMIT JOBS
# =============================================================================

echo "=== ResNet concavity analysis ==="
#submit_resnet topk
submit_resnet quantization --quantization_aware
#submit_resnet llmint8   # skip — flat accuracy landscape

echo ""
echo "=== LLM concavity analysis ==="
#submit_llm topk_per_token
#submit_llm quantization --quantization_aware
# submit_llm magnitude     # uncomment as needed
# submit_llm random

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  ${OUT_BASE}/resnet_topk/concavity_results.json"
echo "  ${OUT_BASE}/resnet_quantization/concavity_results.json"
echo "  ${OUT_BASE}/llm_topk_per_token/concavity_results.json"
echo "  ${OUT_BASE}/llm_quantization/concavity_results.json"
echo ""
echo "Key output: 'global_eta_min' in each JSON is the recommended"
echo "eta_min to use for --eta_min and --grid_eta_min in training."
