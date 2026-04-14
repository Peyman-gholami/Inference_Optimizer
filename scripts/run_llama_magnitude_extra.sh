#!/bin/bash
# =============================================================================
# run_llama_magnitude_extra.sh
#
# Submits 2 jobs for the eta_min=0 llmint8 run:
#
#   Job 1 — main run: eta_min=0, --baselines none  (~2.2h)
#   Job 2 — replay all 6 baselines                 (~7.7h, starts after Job 1)
#
# Usage:  bash run_llama_magnitude_extra.sh
# =============================================================================

HOME_DIR=..
FAST_SAMPLES=50
FULL_SAMPLES=100
GRAD_N=5
SIGMA=0.10
T=50
STRATEGY=magnitude

OUT_DIR="${HOME_DIR}/outputs/single_experiments/group2_rerun3_extra/llama/${STRATEGY}"

mkdir -p logs

COMMON_ENV="
mkdir -p logs
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=0.0
export ETA_MIN1=0.0
export MMLU_FAST_SAMPLES_PER_SUBJECT=${FAST_SAMPLES}
export MMLU_FULL_SAMPLES_PER_SUBJECT=${FULL_SAMPLES}
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=llama_3_1_8b
export LLM_MEASURED_PAYLOAD=prefill_raw
export LLM_MEASURED_BATCH=1
export LLM_GRAD_SIGMA=${SIGMA}
"

# =============================================================================
# Job 1 — main run
# =============================================================================
JOB1=$(sbatch --parsable \
    --partition=gpu --gres=gpu:h200:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G \
    --time=8:00:00 \
    --job-name=magnitude_main \
    --output=logs/magnitude_main_%j.out \
    --error=logs/magnitude_main_%j.err \
    --wrap="${COMMON_ENV}
python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_activation_strategy ${STRATEGY} \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --compressor_backend gpu \
    --eta_min 0.4 \
    --no_csi_mu 0.1 \
    --llm_grad_N ${GRAD_N} \
    --c_min 10000000 --c_max 40000000 --T ${T} \
    --baselines none \
    --out_dir ${OUT_DIR}
")
echo "Submitted Job 1 (main, eta_min=0): ${JOB1} for strategy: ${STRATEGY}"

# =============================================================================
# Job 2 — replay baselines (starts after Job 1 completes successfully)
# =============================================================================
JOB2=$(sbatch --parsable \
    --partition=gpu --gres=gpu:h200:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G \
    --time=8:00:00 \
    --dependency=afterok:${JOB1} \
    --job-name=magnitude_replay \
    --output=logs/magnitude_replay_%j.out \
    --error=logs/magnitude_replay_%j.err \
    --wrap="${COMMON_ENV}
python ${HOME_DIR}/replay_baselines.py \
    --out_dir ${OUT_DIR} \
    --baselines max_compression,no_compression,uniform_compression,myopic,conservative,moving_average
")
echo "Submitted Job 2 (replay, depends on ${JOB1}): ${JOB2}"

echo ""
echo "Job dependency graph:"
echo "  Job 1 (${JOB1}) ──► Job 2 (${JOB2})"
echo ""
echo "Monitor with: squeue -u \$USER"
