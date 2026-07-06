"""
P_c-aggressiveness sweep on MNIST (cheap frontier for the deck).

For a grid of uniform-eta priors  eta* ~ Uniform[lo, 1]  (lo small = aggressive
compression, lo near 1 = mild), we:
  * train ONE standard model at eta=1 (P_c-independent baseline);
  * for each lo, train a compression-aware model under Uniform[lo,1];
  * evaluate both under that same Uniform[lo,1].

Plots clean (eta=1) and deployed E_c[acc] vs lo for both, exposing the
clean/deployed tradeoff as a function of how aggressive P_c is.

Run from repo root:
    .venv/bin/python sweep_pc_mnist.py --steps 1500
"""

import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.core.stein_oracles import MLP
from src.core.accuracy_fn import CompressedModel
from src.pretraining.compression_aware import (
    UniformEtaChannelSampler, compression_aware_pretrain, standard_pretrain,
    deployed_accuracy,
)


def get_loaders(batch_size=128):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
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

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")
    train_loader, test_loader = get_loaders()

    d = 2
    a = np.array([128.0, 64.0])
    eta_min = np.array([0.1, 0.1])
    R = args.R
    los = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # one standard (eta=1) baseline, reused at every P_c
    print("Training standard baseline (eta=1) ...")
    cm_std = CompressedModel(MLP().to(device), torch.ones(d))
    standard_pretrain(cm_std, train_loader, steps=args.steps, lr=args.lr,
                      device=device, log_every=0)

    rows = []
    for lo in los:
        sampler_ca = UniformEtaChannelSampler(d, lo, 1.0, R, a, seed=args.seed)
        cm_ca = CompressedModel(MLP().to(device), torch.ones(d))
        compression_aware_pretrain(cm_ca, train_loader, a=a, R=R, eta_min=eta_min,
                                   sampler=sampler_ca, steps=args.steps, lr=args.lr,
                                   device=device, log_every=0)
        ev = dict(a=a, R=R, eta_min=eta_min, device=device, n_channels=args.n_channels)
        # evaluate both under THIS P_c (fresh sampler, same seed for fair comparison)
        r_ca = deployed_accuracy(cm_ca, test_loader,
                                 sampler=UniformEtaChannelSampler(d, lo, 1.0, R, a, seed=args.seed+1), **ev)
        r_std = deployed_accuracy(cm_std, test_loader,
                                  sampler=UniformEtaChannelSampler(d, lo, 1.0, R, a, seed=args.seed+1), **ev)
        rows.append((lo, r_std["clean_acc"], r_std["deployed_acc_mean"], r_std["deployed_acc_std"],
                     r_ca["clean_acc"], r_ca["deployed_acc_mean"], r_ca["deployed_acc_std"]))
        print(f"lo={lo:.1f}  std[clean={r_std['clean_acc']:.3f} dep={r_std['deployed_acc_mean']:.3f}]  "
              f"ca[clean={r_ca['clean_acc']:.3f} dep={r_ca['deployed_acc_mean']:.3f}±{r_ca['deployed_acc_std']:.3f}]")

    rows = np.array(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "presentation", "figs", "pc_sweep.csv")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savetxt(out, rows, delimiter=",",
               header="lo,std_clean,std_dep,std_dep_std,ca_clean,ca_dep,ca_dep_std", comments="")
    print("wrote", out)


if __name__ == "__main__":
    main()
