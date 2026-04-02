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
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_CUTPOINTS=8,14,21
export RESNET_FAST_BATCHES=1
export ETA_MIN0=0.125
export ETA_MIN1=0.125

# TopK 
# Tight
python -u ${HOME_DIR}/run_experiment.py \
        --backend resnet \
        --compressor topk \
        --compressor_backend gpu \
        --c_min 40000 --c_max 100000 \
        --T 50 \
        --out_dir ${HOME_DIR}/outputs/single_experiments/group1/resnet/topk/tigth

# Transitional
python -u ${HOME_DIR}/run_experiment.py \
        --backend resnet \
        --compressor topk \
        --compressor_backend gpu \
        --c_min 60000 --c_max 300000 \
        --T 50 \
        --out_dir ${HOME_DIR}/outputs/single_experiments/group1/resnet/topk/transitional

# Loose
python -u ${HOME_DIR}/run_experiment.py \
        --backend resnet \
        --compressor topk \
        --compressor_backend gpu \
        --c_min 200000 --c_max 600000 \
        --T 50 \
        --out_dir ${HOME_DIR}/outputs/single_experiments/group1/resnet/topk/loose
