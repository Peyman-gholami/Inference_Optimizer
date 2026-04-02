#!/bin/bash
#
# Usage
# -----
#   sbatch run_s_e_g4_llm.sh
#   # or interactively on a GPU node:
#   bash run_s_e_g4_llm.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=run_experiment
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

mkdir -p logs

# =============================================================================
# Activate conda env
# =============================================================================
eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

HOME_DIR=..

export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export LLM_SPLIT_MODE=thirds
export ETA_MIN0=0.25
export ETA_MIN1=0.25
export MMLU_FAST_SAMPLES_PER_SUBJECT=5
export MMLU_FULL_SAMPLES_PER_SUBJECT=10
export MMLU_BATCH_SIZE=8
export MMLU_SEED=0
export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=llama_3_1_8b
export LLM_MEASURED_PAYLOAD=prefill_raw
export LLM_MEASURED_BATCH=1

python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_activation_strategy topk_per_token \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --c_min 10000000 --c_max 40000000 --T 30 \
    --no_csi_mu 0.5 --T 30 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group4/llama/no_csi_mu_0_5

python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_activation_strategy topk_per_token \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --c_min 10000000 --c_max 40000000 --T 30 \
    --no_csi_mu 3.0 --T 30 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group4/llama/no_csi_mu_3_0

python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_activation_strategy topk_per_token \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --c_min 10000000 --c_max 40000000 --T 30 \
    --no_csi_mu 10.0 --T 30 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group4/llama/no_csi_mu_10_0

