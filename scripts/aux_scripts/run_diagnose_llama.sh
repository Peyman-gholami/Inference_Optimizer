#!/bin/bash
# =============================================================================
# run_diagnose_llama.sh
# Run diagnose_llm_accuracy.py for Llama-3.1-8B.
#
# This takes ~30-60 minutes and tells you whether each strategy's
# accuracy function has enough signal for the gradient oracle to work,
# BEFORE committing to a full 7h experiment.
#
# Usage:  sbatch run_diagnose_llama.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --job-name=diagnose_llama
#SBATCH --output=logs/diagnose_llama_%j.out
#SBATCH --error=logs/diagnose_llama_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

HOME_DIR=..
    #--strategies quantization llmint8 topk_per_token magnitude random \

python ${HOME_DIR}/diagnose_llm_accuracy.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --strategies  topk_per_token magnitude random \
    --eta_values 0.25 0.5 0.75 1.0 \
    --eta_min 0.25 \
    --fast_samples 50 \
    --full_samples 100 \
    --grad_sigma 0.05 \
    --grad_N 5 \
    --noise_repeats 3 \
    --out_dir ${HOME_DIR}/outputs/diagnosis/llama
