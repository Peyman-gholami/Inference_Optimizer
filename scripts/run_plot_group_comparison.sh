#!/bin/bash
#
# Usage
# -----
#   sbatch run_plot_group_comparison.sh
# =============================================================================

#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --job-name=plot_group_comparison
#SBATCH --output=logs/group_comparison_%j.out
#SBATCH --error=logs/group_comparison_%j.err

mkdir -p logs

HOME_DIR=..

# Group 1
python ${HOME_DIR}/plot_group_comparison.py \
    --runs  ${HOME_DIR}/outputs/single_experiments/group1/resnet/topk/loose \
            ${HOME_DIR}/outputs/single_experiments/group1/resnet/topk/tigth \
            ${HOME_DIR}/outputs/single_experiments/group1/resnet/topk/transitional \
    --labels "Loose" "Tight" "Transitional" \
    --title "ResNet Group 1: Channel Regime — Compressor: (gpu-native) topk" \
    --algorithms "CSI-aware (optimal)" "No-CSI (Alg1) + LCB(90%)" \
    --out_dir ${HOME_DIR}/outputs/comparisons/resnet_group1

python ${HOME_DIR}/plot_group_comparison.py \
    --runs  ${HOME_DIR}/outputs/single_experiments/group1/llama/topk_per_token/loose \
            ${HOME_DIR}/outputs/single_experiments/group1/llama/topk_per_token/tigth \
            ${HOME_DIR}/outputs/single_experiments/group1/llama/topk_per_token/transitional \
    --algorithms "CSI-aware (optimal)" "No-CSI (Alg1) + LCB(90%)" \
    --labels "Loose" "Tight" "Transitional" \
    --title "Llama Group 1: Channel Regime — Compressor: topk_per_token" \
    --out_dir ${HOME_DIR}/outputs/comparisons/llama_group1

# Group 2
python ${HOME_DIR}/plot_group_comparison.py \
    --runs  ${HOME_DIR}/outputs/single_experiments/group2/resnet/topk \
            ${HOME_DIR}/outputs/single_experiments/group2/resnet/quantization \
            ${HOME_DIR}/outputs/single_experiments/group2/resnet/llmint8 \
    --labels "TopK" "Quantization" "LLMInt8" \
    --title "ResNet Group 2: Compressor comparison — c~U[60000,300000]" \
    --algorithms "CSI-aware (optimal)" "No-CSI (Alg1) + LCB(90%)" \
    --out_dir ${HOME_DIR}/outputs/comparisons/resnet_group1


python ${HOME_DIR}/plot_group_comparison.py \
    --runs ${HOME_DIR}/outputs/single_experiments/group2/llama/topk_per_token \
            ${HOME_DIR}/outputs/single_experiments/group2/llama/magnitude \
            ${HOME_DIR}/outputs/single_experiments/group2/llama/random \
            ${HOME_DIR}/outputs/single_experiments/group2/llama/quantization \
            ${HOME_DIR}/outputs/single_experiments/group2/llama/llmint8 \
    --algorithms "CSI-aware (optimal)" "No-CSI (Alg1) + LCB(90%)" \
    --labels "topk_per_token" "magnitude" "random" "quantization" "llmint8" \
    --title "Llama-3.1-8B Group 2: Strategy comparison — c~U[10M,40M]" \
    --out_dir ${HOME_DIR}/outputs/comparisons/llama_group2
