#!/bin/bash
# =============================================================================
# run_llama_quant_extra_resume.sh
#
# Resumes the  main run from checkpoint (stopped at t=25),
# then replays all 6 baselines after completion.
#
# Key fixes vs previous script:
#   1. --resume flag restores from checkpoint.npz (t=25 → t=49)
#   2. --dependency=afterany instead of afterok so the replay job
#      starts even if the main job exits via time limit again
#   3. replay_baselines.py checks if main run is complete before merging
#
# Job timeline:
#   Job 1 — resume main from t=25  (~3.2h for remaining 24 slots)
#   Job 2 — replay 6 baselines     (~7.7h, starts after Job 1 exits)
#
# Usage:  bash run_llama_quant_extra_resume.sh
# =============================================================================

HOME_DIR=..
FAST_SAMPLES=50
FULL_SAMPLES=100
GRAD_N=5
SIGMA=0.2
T=50

OUT_DIR="${HOME_DIR}/outputs/single_experiments/group2_rerun3_extra/llama/quantization"

mkdir -p logs

COMMON_ENV="
mkdir -p logs
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=0.25
export ETA_MIN1=0.25
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
# Job 1 — resume main run from checkpoint
# =============================================================================
JOB1=$(sbatch --parsable \
    --partition=gpu --gres=gpu:h200:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G \
    --time=8:00:00 \
    --job-name=quantization_resume \
    --output=logs/quantization_resume_%j.out \
    --error=logs/quantization_resume_%j.err \
    --wrap="${COMMON_ENV}
python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_activation_strategy quantization \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --compressor_backend gpu \
    --eta_min 0.4 \
    --no_csi_mu 0.1 \
    --llm_grad_N ${GRAD_N} \
    --c_min 10000000 --c_max 40000000 --T ${T} \
    --baselines none \
    --out_dir ${OUT_DIR} \
    --resume
")
echo "Submitted Job 1 (resume from checkpoint): ${JOB1}"

# =============================================================================
# Job 2 — replay baselines
# --dependency=afterany: starts after Job 1 exits for ANY reason
# (completed, timeout, or failed). replay_baselines.py will check
# that T_done=50 before merging — if main is still incomplete it
# will print a warning and exit cleanly without corrupting results.
# =============================================================================
JOB2=$(sbatch --parsable \
    --partition=gpu --gres=gpu:h200:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G \
    --time=8:00:00 \
    --dependency=afterany:${JOB1} \
    --job-name=quantization_replay \
    --output=logs/quantization_replay_%j.out \
    --error=logs/quantization_replay_%j.err \
    --wrap="${COMMON_ENV}
python ${HOME_DIR}/replay_baselines.py \
    --out_dir ${OUT_DIR} \
    --baselines max_compression,no_compression,uniform_compression,myopic,conservative,moving_average \
    --expected_T ${T}
")
echo "Submitted Job 2 (replay, depends on ${JOB1}): ${JOB2}"

echo ""
echo "Job dependency graph:"
echo "  Job 1 (${JOB1}) ──► Job 2 (${JOB2})   [afterany]"
echo ""
echo "Monitor with: squeue -u \$USER"
