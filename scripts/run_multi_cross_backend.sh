#!/bin/bash
#
# Usage
# -----
#
#   sbatch run_multi_cross_backend.sh
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=multi_cross_backend
#SBATCH --output=logs/multi_cross_backend_%j.out
#SBATCH --error=logs/multi_cross_backend_%j.err

mkdir -p logs

# =============================================================================
# Activate conda env
# =============================================================================
eval "$(conda shell.bash hook)"
conda deactivate
conda activate newcap

COMP=llmint8  # topk quantization llmint8
MLP_ETA_MIN=0.125
RESNET_ETA_MIN=0.125
C_MIN=60000 
C_MAX=300000
EPSILON=0.1
MU=3.0

HOME_DIR=..
T=50
SEED=12
MLP_SIGMA=0.05
MLP_GRAD_N=50
MLP_R_K=100
RESNET_SIGMA=0.05
RESNET_GRAD_N=50
RESNET_R_K=10

# COMMON
export COMPRESSOR_BACKEND=gpu

# MLP
export MNIST_DATA_ROOT=${HOME_DIR}/data
export MLP_GRAD_SIGMA=${MLP_SIGMA}
export MLP_GRAD_N=${MLP_GRAD_N}

# RESNET
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_CUTPOINTS=8,14,21
export RESNET_FAST_BATCHES=1
export RESNET_GRAD_SIGMA=${RESNET_SIGMA}
export RESNET_GRAD_N=${RESNET_GRAD_N}

#Example: ResNet (task 1) + MLP (task 2):
        #--task_weights  \
        #--no_csi_J \
        #--baselines \
python -u ${HOME_DIR}/run_experiment_multi_cross_backend.py \
        --task_backends resnet mlp \
        --task_compressors ${COMP} ${COMP} \
        --task_eta_mins ${RESNET_ETA_MIN} ${MLP_ETA_MIN} \
        --R_k ${RESNET_R_K} ${MLP_R_K} \
        --compressor_backend gpu \
        --c_min ${C_MIN} --c_max ${C_MAX} \
        --no_csi_mu ${MU} --no_csi_epsilon ${EPSILON} \
        --T ${T} --seed ${SEED} \
        --out_dir ${HOME_DIR}/outputs/multi_experiments/resnet_mlp/${COMP}



