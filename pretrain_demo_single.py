"""
Demo: CSI-aware single-task compression-aware pretraining (Algorithm 1, J=0).

Trains two MNIST MLPs:
  (1) standard      -- ordinary training at eta = 1 (no compression);
  (2) compression-aware -- SGD with a freshly channel-sampled eta*(c) per minibatch.

Both are then evaluated under the SAME deployment channel distribution P_c, where
the controller applies eta*(c) = min(1, c_i/(R a_i)).  The compression-aware model
should retain higher accuracy under deployment compression, at little clean-accuracy
cost --- the point of pretraining for the deployed compression regime.

Run from the repo root:
    .venv/bin/python pretrain_demo_single.py --steps 1500
"""

import argparse

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.core.stein_oracles import MLP
from src.core.accuracy_fn import CompressedModel
from src.pretraining.compression_aware import (
    ChannelSampler,
    compression_aware_pretrain,
    standard_pretrain,
    deployed_accuracy,
)


def get_loaders(batch_size=128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=tf),
                       batch_size=batch_size, shuffle=True)
    test = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=tf),
                      batch_size=512, shuffle=False)
    return train, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--n_channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    train_loader, test_loader = get_loaders()

    # MLP has 2 ReLU cut-points; activation sizes a_i (units transmitted per hop).
    d = 2
    a = np.array([128.0, 64.0])          # output dims of the two ReLU layers
    eta_min = np.array([0.1, 0.1])
    R = args.R

    # Channel distribution: c_i log-uniform so eta*_i = min(1, c_i/(R a_i)) ranges
    # over [eta_min, 1] -- i.e. the deployment regime involves real compression.
    c_min = R * a * eta_min          # eta* hits the floor
    c_max = R * a * 1.2              # eta* saturates at 1 part of the time
    sampler = ChannelSampler(d, c_min, c_max, seed=args.seed)

    common = dict(a=a, R=R, eta_min=eta_min, sampler=sampler, device=device)

    print("=" * 70)
    print("Training STANDARD baseline (eta = 1) ...")
    cm_std = CompressedModel(MLP().to(device), torch.ones(d))
    standard_pretrain(cm_std, train_loader, steps=args.steps, lr=args.lr, device=device)

    print("=" * 70)
    print("Training COMPRESSION-AWARE model (sampled eta*(c)) ...")
    cm_ca = CompressedModel(MLP().to(device), torch.ones(d))
    compression_aware_pretrain(cm_ca, train_loader, steps=args.steps, lr=args.lr, **common)

    print("=" * 70)
    print("Evaluating under the deployment channel distribution P_c ...")
    res_std = deployed_accuracy(cm_std, test_loader, n_channels=args.n_channels, **common)
    res_ca = deployed_accuracy(cm_ca, test_loader, n_channels=args.n_channels, **common)

    print("\n" + "=" * 70)
    print(f"{'model':<22}{'clean (eta=1)':>16}{'deployed E_c[acc]':>22}")
    print("-" * 70)
    print(f"{'standard':<22}{res_std['clean_acc']:>16.4f}"
          f"{res_std['deployed_acc_mean']:>16.4f} ± {res_std['deployed_acc_std']:.3f}")
    print(f"{'compression-aware':<22}{res_ca['clean_acc']:>16.4f}"
          f"{res_ca['deployed_acc_mean']:>16.4f} ± {res_ca['deployed_acc_std']:.3f}")
    print("=" * 70)
    gain = res_ca["deployed_acc_mean"] - res_std["deployed_acc_mean"]
    print(f"Deployed-accuracy gain from compression-aware pretraining: {gain:+.4f}")


if __name__ == "__main__":
    main()
