"""Pre-download CIFAR-10 + pretrained CIFAR ResNet weights on a machine WITH
internet (the Explorer login node), so SLURM compute-node jobs --- which have no
outbound internet --- can run fully offline.

Caches into:
  * $TORCH_HOME/hub   (torch.hub pretrained ResNet weights)
  * ./data            (CIFAR-10)

Run from ~/Inference_Optimizer on the login node:
    export TORCH_HOME=$HOME/Inference_Optimizer/.torch
    python cluster/prefetch_assets.py
"""
import os
import torch
from torchvision import datasets

print("TORCH_HOME =", os.environ.get("TORCH_HOME", "(unset -> ~/.cache/torch)"))

# CIFAR-10 train + test
datasets.CIFAR10("./data", train=True, download=True)
datasets.CIFAR10("./data", train=False, download=True)
print("cached CIFAR-10")

# pretrained CIFAR ResNets we may use
for name in ["cifar10_resnet20", "cifar10_resnet56"]:
    torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True, trust_repo=True)
    print("cached", name)

print("prefetch done")
