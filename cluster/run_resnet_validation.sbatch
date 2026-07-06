#!/bin/bash
# CIFAR-10 / ResNet-56 compression-aware validation on a GPU compute node.
# Submit from the login node:   sbatch cluster/run_resnet_validation.sbatch
#SBATCH --job-name=pretrain-resnet
#SBATCH --account=sioannidis
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home/davarakis.t/pretrain_runs/%x-%j.out

set -euo pipefail
module load anaconda3/2024.06
source "$HOME/envs/pretrain/bin/activate"

cd "$HOME/Inference_Optimizer"
# offline on compute nodes: use the hub cache / data we pre-fetched on the login node
export TORCH_HOME="$HOME/Inference_Optimizer/.torch"
export HF_HUB_OFFLINE=1

echo "host: $(hostname)"
nvidia-smi || true
python -c "import torch; print('cuda avail:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# device-agnostic code auto-selects cuda here
python pretrain_demo_resnet.py --model cifar10_resnet56 --steps 1000 \
    --n_channels 32 --control --lr 1e-4
echo "JOB DONE $(date)"
