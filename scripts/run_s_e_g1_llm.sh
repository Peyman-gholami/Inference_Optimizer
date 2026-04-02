#!/bin/bash
#
# Usage
# -----
#   sbatch run_s_e_resnet.sh
#   # or interactively on a GPU node:
#   bash run_s_e_resnet.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=llama_3_1_8b
export LLM_MEASURED_PAYLOAD=prefill_raw   # or decode_raw depending on your scenario
export LLM_MEASURED_BATCH=1

# topk_per_token | magnitude | random | quantization | llmint8

# topk_per_token
# Tight
python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_activation_strategy topk_per_token \
    --c_min 6000000 --c_max 15000000 --T 30 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group1/llama/topk_per_token/tigth

# Transitional
python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_activation_strategy topk_per_token \
    --c_min 10000000 --c_max 40000000 --T 30 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group1/llama/topk_per_token/transitional

# Loose
python ${HOME_DIR}/run_experiment.py --backend llm \
    --llm_model_name meta-llama/Llama-3.1-8B-Instruct \
    --llm_activation_strategy topk_per_token \
    --c_min 30000000 --c_max 100000000 --T 30 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group1/llama/topk_per_token/loose

