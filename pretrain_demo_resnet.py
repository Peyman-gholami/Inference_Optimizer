"""
Demo: CSI-aware single-task compression-aware *fine-tuning* on CIFAR-10 / ResNet.

We download a pretrained CIFAR-10 ResNet (avoiding the base pretraining) and then:
  (baseline)     evaluate it off-the-shelf under the deployment channel P_c;
  (comp-aware)   fine-tune a copy under the channel-induced eta*(c) and re-evaluate;
  (--control)    optionally also fine-tune a copy at eta=1 for the same budget,
                 to separate "compression-aware" from "just more training".

Cut-points are the layer1/layer2 feature-map boundaries (d=2). Compression is
per-sample top-k. Runs on GPU automatically (cuda -> mps -> cpu).

Run from the repo root, e.g.:
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python pretrain_demo_resnet.py \
        --model cifar10_resnet56 --steps 1000 --n_channels 32
"""

import argparse

import numpy as np
import torch

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, compression_aware_finetune, standard_finetune,
    deployed_accuracy_resnet, load_test_tensors,
)
from src.pretraining.compression_aware import ChannelSampler, UniformEtaChannelSampler


def build(model_name, device):
    """Load a fresh pretrained model, wrap it, return (cm, a)."""
    model = load_pretrained_cifar_resnet(model_name).to(device)
    cm, a = make_compressed_resnet(model)
    return cm.to(device), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cifar10_resnet56",
                   help="cifar10_resnet20/32/44/56")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--eta_min", type=float, default=0.1)
    p.add_argument("--pc", choices=["loguniform", "uniform_eta"], default="loguniform",
                   help="deployment channel prior: log-uniform on c, or uniform eta*")
    p.add_argument("--eta_lo", type=float, default=0.3,
                   help="lower bound of uniform-eta prior (used when --pc uniform_eta)")
    p.add_argument("--eta_hi", type=float, default=1.0,
                   help="upper bound of uniform-eta prior")
    p.add_argument("--n_channels", type=int, default=32)
    p.add_argument("--device", default=None, help="force cuda/mps/cpu")
    p.add_argument("--control", action="store_true",
                   help="also run a standard (eta=1) fine-tune control arm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}   model: {args.model}")

    train_loader, test_loader = get_cifar10_loaders()
    x_test, y_test = load_test_tensors(test_loader)

    # --- channel setup (uses the feature-map sizes a discovered from the model) ---
    cm_probe, a = build(args.model, device)
    d = cm_probe.d
    eta_min = np.full(d, args.eta_min)
    R = args.R
    if args.pc == "loguniform":
        c_min = R * a * eta_min          # eta* hits the floor
        c_max = R * a * 1.2              # eta* saturates at 1 part of the time
        sampler = ChannelSampler(d, c_min, c_max, seed=args.seed)
        pc_desc = f"log-uniform on c  (eta* log-uniform on [{args.eta_min},1] + atom at 1)"
    else:  # uniform_eta -- milder, less heavy-end-biased prior
        sampler = UniformEtaChannelSampler(d, args.eta_lo, args.eta_hi, R, a, seed=args.seed)
        pc_desc = f"uniform eta* on [{args.eta_lo},{args.eta_hi}]"
    print(f"cut-points d={d}, feature sizes a={a.astype(int)}, eta_min={args.eta_min}")
    print(f"P_c: {pc_desc}")

    common = dict(a=a, R=R, eta_min=eta_min, sampler=sampler, device=device)
    ev = dict(a=a, R=R, eta_min=eta_min, sampler=sampler, device=device,
              n_channels=args.n_channels)

    # --- baseline: off-the-shelf pretrained model under P_c ---
    print("\n[1] Evaluating OFF-THE-SHELF pretrained model under P_c ...")
    res_base = deployed_accuracy_resnet(cm_probe, x_test, y_test, **ev)

    # --- compression-aware fine-tuning ---
    print(f"\n[2] Compression-aware fine-tuning ({args.steps} steps) ...")
    cm_ca, _ = build(args.model, device)
    compression_aware_finetune(cm_ca, train_loader, steps=args.steps, lr=args.lr, **common)
    res_ca = deployed_accuracy_resnet(cm_ca, x_test, y_test, **ev)

    res_std = None
    if args.control:
        print(f"\n[3] Standard (eta=1) fine-tuning control ({args.steps} steps) ...")
        cm_std, _ = build(args.model, device)
        standard_finetune(cm_std, train_loader, steps=args.steps, lr=args.lr, device=device)
        res_std = deployed_accuracy_resnet(cm_std, x_test, y_test, **ev)

    # --- report ---
    print("\n" + "=" * 74)
    print(f"{'model':<28}{'clean (eta=1)':>16}{'deployed E_c[acc]':>24}")
    print("-" * 74)
    def row(name, r):
        print(f"{name:<28}{r['clean_acc']:>16.4f}"
              f"{r['deployed_acc_mean']:>16.4f} ± {r['deployed_acc_std']:.3f}")
    row("off-the-shelf", res_base)
    if res_std is not None:
        row("standard fine-tune", res_std)
    row("compression-aware", res_ca)
    print("=" * 74)
    gain = res_ca["deployed_acc_mean"] - res_base["deployed_acc_mean"]
    print(f"Deployed-accuracy gain vs off-the-shelf: {gain:+.4f}")
    if res_std is not None:
        gain2 = res_ca["deployed_acc_mean"] - res_std["deployed_acc_mean"]
        print(f"Deployed-accuracy gain vs standard fine-tune: {gain2:+.4f}")


if __name__ == "__main__":
    main()
