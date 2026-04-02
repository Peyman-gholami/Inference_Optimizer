#!/bin/bash
#
# Usage
# -----
#   sbatch run_s_e_g6_resnet.sh
#   # or interactively on a GPU node:
#   bash run_s_e_g6_resnet.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
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

export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=../models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=../data
export RESNET_CUTPOINTS=8,14,21
export RESNET_FAST_BATCHES=1
export ETA_MIN0=0.125
export ETA_MIN1=0.125


export COMPRESSOR_TYPE=topk
export COMPRESSOR_BACKEND=gpu

python ${HOME_DIR}/run_experiment.py --backend resnet --compressor topk \
    --compressor_backend gpu \
    --c_min 60000 --c_max 300000 --T 50 \
    --R_k 5.0 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group4/resnet/r_k_5_0

python ${HOME_DIR}/run_experiment.py --backend resnet --compressor topk \
    --compressor_backend gpu \
    --c_min 60000 --c_max 300000 --T 50 \
    --R_k 10.0 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group4/resnet/r_k_10_0

python ${HOME_DIR}/run_experiment.py --backend resnet --compressor topk \
    --compressor_backend gpu \
    --c_min 60000 --c_max 300000 --T 50 \
    --R_k 20.0 \
    --out_dir ${HOME_DIR}/outputs/single_experiments/group4/resnet/r_k_20_0