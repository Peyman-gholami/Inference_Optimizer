#!/bin/bash
# =============================================================================
# run_accuracy_3d.sh
#
# Submits jobs to generate 3D accuracy surface plots for the paper.
# Each job samples a 2D grid of (eta_1, eta_2) with eta_0 fixed at 1.0,
# fits a KNN regressor, computes the Stein-smoothed surface, and produces
# four publication-quality plots.
#
# Usage:  bash run_accuracy_3d.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# PARAMETERS
# =============================================================================

# Grid sampling
N_GRID=20          # points per axis (total = N_GRID^2 = 400 points)
N_REPEAT=3         # repeated evaluations per point (averaged, reduces noise)
ETA_LO=0.0         # lower bound of eta sweep

# KNN regressor
KNN_K=8            # neighbours (try 5, 8, 12 — larger = smoother)

# Stein smoothing (matches optimizer parameters)
STEIN_SIGMA=0.05
STEIN_N=10         # MC samples per point for Stein surface
STEIN_REPEAT=1

OUT_BASE="${HOME_DIR}/outputs/accuracy_3d"

TIME_RESNET="2:00:00"
TIME_LLM="8:00:00"
MEM="32G"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

# ── ResNet ────────────────────────────────────────────────────────────────────
submit_resnet() {
    local COMP=$1
    local EXTRA="${2:-}"
    local OUT_DIR="${OUT_BASE}/resnet_${COMP}"

    echo "Submitting ResNet 3D plot: compressor=${COMP}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_RESNET} \
        --job-name=acc3d_resnet_${COMP} \
        --output=logs/acc3d_resnet_${COMP}_%j.out \
        --error=logs/acc3d_resnet_${COMP}_%j.err \
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

python ${HOME_DIR}/plot_accuracy_3d.py \\
    --backend resnet \\
    --compressor ${COMP} \\
    --compressor_backend gpu \\
    --n_grid ${N_GRID} \\
    --n_repeat ${N_REPEAT} \\
    --eta_lo ${ETA_LO} \\
    --knn_k ${KNN_K} \\
    --stein_sigma ${STEIN_SIGMA} \\
    --stein_n ${STEIN_N} \\
    --stein_repeat ${STEIN_REPEAT} \\
    --out_dir ${OUT_DIR} \\
    ${EXTRA}
")
    echo "  Submitted: job ${JOB}"
}

# ── LLM ───────────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"

submit_llm() {
    local STRATEGY=$1
    local EXTRA="${2:-}"
    local OUT_DIR="${OUT_BASE}/llm_${STRATEGY}"

    echo "Submitting LLM 3D plot: strategy=${STRATEGY}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_LLM} \
        --job-name=acc3d_llm_${STRATEGY} \
        --output=logs/acc3d_llm_${STRATEGY}_%j.out \
        --error=logs/acc3d_llm_${STRATEGY}_%j.err \
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

python ${HOME_DIR}/plot_accuracy_3d.py \\
    --backend llm \\
    --compressor ${STRATEGY} \\
    --n_grid ${N_GRID} \\
    --n_repeat ${N_REPEAT} \\
    --eta_lo ${ETA_LO} \\
    --knn_k ${KNN_K} \\
    --stein_sigma ${STEIN_SIGMA} \\
    --stein_n ${STEIN_N} \\
    --stein_repeat ${STEIN_REPEAT} \\
    --out_dir ${OUT_DIR} \\
    ${EXTRA}
")
    echo "  Submitted: job ${JOB}"
}

# =============================================================================
# SUBMIT
# =============================================================================

echo "=== ResNet 3D accuracy plots ==="
#submit_resnet topk
submit_resnet quantization
#submit_resnet llmint8

# Uncomment for LLM (slow — use saved ResNet data to test first)
# echo ""
# echo "=== LLM 3D accuracy plots ==="
# submit_llm topk_per_token
# submit_llm quantization

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "To replot with different KNN k (no resampling):"
echo "  python plot_accuracy_3d.py \\"
echo "      --load_data ${OUT_BASE}/resnet_topk/grid_data.npz \\"
echo "      --knn_k 12 \\"
echo "      --out_dir ${OUT_BASE}/resnet_topk_k12"
echo ""
echo "To overlay eta_min from concavity analysis:"
echo "  python plot_accuracy_3d.py \\"
echo "      --load_data ${OUT_BASE}/resnet_topk/grid_data.npz \\"
echo "      --eta_min_json ${HOME_DIR}/outputs/concavity_analysis/resnet_topk/concavity_results.json \\"
echo "      --out_dir ${OUT_BASE}/resnet_topk_with_etamin"
