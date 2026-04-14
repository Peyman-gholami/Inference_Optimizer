#!/bin/bash
# =============================================================================
# run_diagnose_llama_stein.sh
# Fast Stein gradient test at eta_min for random and quantization.
#
# Tests 8 sigma values [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# at eta_min and compares Stein gradient vs finite-difference gradient.
#
# This directly answers:
#   1. Is the gradient near zero at eta_min? (FD gradient)
#   2. Does the Stein oracle correctly estimate it? (agreement with FD)
#   3. What is the minimum sigma needed? (first sigma with non-zero gradient)
#
# Expected runtime: ~30-45 min (2 strategies × 8 sigmas × few evaluations)
#
# Usage:  sbatch run_diagnose_llama_stein.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --job-name=diagnose_stein
#SBATCH --output=logs/diagnose_stein_%j.out
#SBATCH --error=logs/diagnose_stein_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

HOME_DIR=..

python ${HOME_DIR}/diagnose_llm_accuracy.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --strategies magnitude topk_per_token llmint8 \
    --eta_values 0.25 0.5 0.75 1.0 \
    --eta_min 0.25 \
    --fast_samples 50 \
    --full_samples 50 \
    --grad_sigma 0.05 \
    --grad_N 10 \
    --noise_repeats 1 \
    --out_dir ${HOME_DIR}/outputs/diagnosis/llama_stein
