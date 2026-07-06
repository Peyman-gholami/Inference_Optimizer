"""
Demo: multi-task DECOUPLED compression-aware pretraining on MNIST.

K=2 symmetric tasks (two MNIST MLPs) share a 2-link chain. Per channel, the MobiHoc
CSIAwareMultiTaskOptimizer water-fills the comm shares -> joint eta*(c); each task
takes an independent compression-aware SGD step at its eta*_k. By symmetry the split
is equal, so each task effectively runs at the contention-reduced capacity c/K.

Compares decoupled compression-aware training vs standard (eta=1), both evaluated
under the same shared-bandwidth deployment distribution P_c.

Run from the repo root:
    .venv/bin/python pretrain_demo_multi.py --steps 1500
"""

import argparse

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.core.stein_oracles import MLP
from src.core.accuracy_fn import CompressedModel
from src.pretraining.compression_aware import ChannelSampler
from src.pretraining.multitask_decoupled import (
    build_symmetric_tasks, decoupled_pretrain, standard_pretrain_multi,
    deployed_accuracy_multi,
)
from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer


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
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--n_channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")
    train_loader, test_loader = get_loaders()

    K = args.K
    d = 2
    a = np.array([128.0, 64.0])
    eta_min = np.array([0.1, 0.1])
    R = args.R

    tasks, M = build_symmetric_tasks(K, a, eta_min, R=R)
    opt = CSIAwareMultiTaskOptimizer(M, tasks)

    # Shared-link capacities: c_i so the per-task (equal-split) eta*_i = min(1, c_i/(K R a_i))
    # spans [eta_min, 1]  ->  c_i in [K R a_i eta_min, 1.2 K R a_i].
    c_min = K * R * a * eta_min
    c_max = 1.2 * K * R * a
    sampler = ChannelSampler(d, c_min, c_max, seed=args.seed)
    print(f"K={K} symmetric tasks, M={M} nodes, d={d} shared links, a={a.astype(int)}")

    print("=" * 72)
    print("Training STANDARD baseline (eta=1) ...")
    cms_std = [CompressedModel(MLP().to(device), torch.ones(d)) for _ in range(K)]
    standard_pretrain_multi(cms_std, train_loader, steps=args.steps, lr=args.lr, device=device)

    print("=" * 72)
    print("Training DECOUPLED compression-aware (joint eta*(c) per channel) ...")
    cms_ca = [CompressedModel(MLP().to(device), torch.ones(d)) for _ in range(K)]
    decoupled_pretrain(cms_ca, train_loader, opt, sampler=sampler,
                       steps=args.steps, lr=args.lr, device=device)

    print("=" * 72)
    print("Evaluating under the shared-bandwidth deployment distribution P_c ...")
    common = dict(opt=opt, sampler=sampler, n_channels=args.n_channels, device=device)
    res_std = deployed_accuracy_multi(cms_std, test_loader, **common)
    res_ca = deployed_accuracy_multi(cms_ca, test_loader, **common)

    print("\n" + "=" * 72)
    print(f"{'model':<22}{'clean (per task)':>22}{'deployed E_c[acc] (per task)':>30}")
    print("-" * 72)
    def fmt(r):
        clean = ", ".join(f"{v:.3f}" for v in r["clean_per_task"])
        dep = ", ".join(f"{m:.3f}±{s:.3f}" for m, s in
                        zip(r["deployed_mean_per_task"], r["deployed_std_per_task"]))
        return clean, dep
    cs, ds = fmt(res_std); print(f"{'standard':<22}{cs:>22}{ds:>30}")
    cc, dc = fmt(res_ca);  print(f"{'decoupled comp-aware':<22}{cc:>22}{dc:>30}")
    print("-" * 72)
    print(f"aggregate deployed E_c[acc]:  standard={res_std['deployed_mean_agg']:.4f}   "
          f"comp-aware={res_ca['deployed_mean_agg']:.4f}   "
          f"gain={res_ca['deployed_mean_agg']-res_std['deployed_mean_agg']:+.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
