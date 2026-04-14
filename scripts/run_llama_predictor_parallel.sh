#!/bin/bash
# =============================================================================
# run_llama_predictor_parallel.sh
#
# Runs Llama experiments using PredictorGradientOracle instead of Stein.
# Requires pre-trained predictor models (from train_accuracy_predictor.py).
#
# Submits one job per strategy, all in parallel.
#
# Usage:  bash run_llama_predictor_parallel.sh
# =============================================================================

HOME_DIR=..
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# =============================================================================
# PARAMETERS
# =============================================================================

C_MIN=10000000
C_MAX=40000000
R_K=10
T=50
ETA_MIN=0.1

# No-CSI — per-strategy mu (scaled to accuracy range)
# These are set per-strategy inside submit_experiment()

EPSILON=0.1
FAST_SAMPLES=50
FULL_SAMPLES=100
GRAD_N=5       # still used by Stein fallback, ignored by predictor oracle

# Predictor model directory
PREDICTOR_DIR="${HOME_DIR}/models/predictors"

# Cone projection — set to 1 to project gradient onto upward cone
PROJECT_UPWARD=1
CONE_MARGIN=-0.01   # 0 = hard projection; >0 = soft (e.g. 0.02)

OUT_PREFIX="${HOME_DIR}/outputs/single_experiments/group2_predictor/llama"

TIME_LIMIT="4:00:00"
MEM="48G"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

# Common environment block (no PREDICTOR_PATH or MU here — set per strategy)
COMMON_ENV_BASE="
mkdir -p logs
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=${MODEL}
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=${ETA_MIN}
export ETA_MIN1=${ETA_MIN}
export MMLU_FAST_SAMPLES_PER_SUBJECT=${FAST_SAMPLES}
export MMLU_FULL_SAMPLES_PER_SUBJECT=${FULL_SAMPLES}
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=llama_3_1_8b
export LLM_MEASURED_PAYLOAD=prefill_raw
export LLM_MEASURED_BATCH=1
export LLM_GRAD_N=${GRAD_N}
export LLM_GRAD_PROJECT_UPWARD=${PROJECT_UPWARD}
export LLM_GRAD_CONE_MARGIN=${CONE_MARGIN}
"

# =============================================================================
# submit_experiment STRATEGY PREDICTOR_PATH MU [SUFFIX]
# =============================================================================
submit_experiment() {
    local STRATEGY=$1
    local PREDICTOR_PATH=$2
    local MU=$3
    local SUFFIX=${4:-${STRATEGY}}
    local OUT_DIR="${OUT_PREFIX}/${SUFFIX}"

    echo ""
    echo "Submitting: ${STRATEGY}"
    echo "  predictor = ${PREDICTOR_PATH}"
    echo "  mu=${MU}  eta_min=${ETA_MIN}  T=${T}"
    echo "  project_upward=${PROJECT_UPWARD}  cone_margin=${CONE_MARGIN}"
    echo "  out: ${OUT_DIR}"

    if [ ! -f "${PREDICTOR_PATH}" ]; then
        echo "  WARNING: predictor file not found: ${PREDICTOR_PATH}"
        echo "  Skipping ${STRATEGY}."
        return 1
    fi

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_LIMIT} \
        --job-name=llama_pred_${SUFFIX} \
        --output=logs/llama_pred_${SUFFIX}_%j.out \
        --error=logs/llama_pred_${SUFFIX}_%j.err \
        --wrap="${COMMON_ENV_BASE}
export LLM_ACTIVATION_STRATEGY=${STRATEGY}
export LLM_GRAD_SIGMA=0.05
export LLM_GRAD_PREDICTOR_PATH=${PREDICTOR_PATH}

python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_activation_strategy ${STRATEGY} \
    --llm_model_name ${MODEL} \
    --compressor_backend gpu \
    --eta_min ${ETA_MIN} \
    --no_csi_mu ${MU} \
    --no_csi_epsilon ${EPSILON} \
    --R_k ${R_K} \
    --c_min ${C_MIN} --c_max ${C_MAX} --T ${T} \
    --out_dir ${OUT_DIR}
")
    echo "  Submitted: job ${JOB}"
}

# =============================================================================
# EXPERIMENTS
# Per-strategy mu is scaled to accuracy range (from diagnostic results):
#   topk_per_token : range=0.043 -> mu=0.1
#   quantization   : range=0.317 -> mu=0.5
# =============================================================================

echo "PREDICTOR_DIR=${PREDICTOR_DIR}"

# ── topk_per_token ────────────────────────────────────────────────────────────
submit_experiment \
    "topk_per_token" \
    "${PREDICTOR_DIR}/llama_topk_per_token_etamin0.0_grid15.pt" \
    0.1

# ── quantization ─────────────────────────────────────────────────────────────
#submit_experiment \
#    "quantization" \
#    "${PREDICTOR_DIR}/llama_quantization_etamin0.0_grid20.pt" \
#    0.5

# ── Add more strategies here once predictors are trained ─────────────────────
# submit_experiment \
#     "magnitude" \
#     "${PREDICTOR_DIR}/llama_magnitude_etamin${ETA_MIN}_grid15.pt" \
#     0.5

# submit_experiment \
#     "random" \
#     "${PREDICTOR_DIR}/llama_random_etamin${ETA_MIN}_grid15.pt" \
#     3.0

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Output dirs:"
echo "  ${OUT_PREFIX}/topk_per_token"
echo "  ${OUT_PREFIX}/quantization"
