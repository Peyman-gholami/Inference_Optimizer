"""
Off-the-shelf accuracy-vs-eta sweep for each ResNet, to make the retraining-induced
monotonicity loss apparent. Pristine pretrained models should be MONOTONE (best at
eta=1); a narrow-band compression-aware retrain inverts the curve.

Curves: off-the-shelf ResNet-20, off-the-shelf ResNet-56, and (for contrast) a
narrow-band-retrained ResNet-20. Saves presentation/figs/eta_offshelf.png.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python eta_offshelf_sweep.py
"""
import os
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.accuracy_fn import CompressedModel
from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet, load_test_tensors,
    topk_sparsify_per_sample,
)


def build(name, device):
    m = load_pretrained_cifar_resnet(name).to(device)
    return CompressedModel(m, torch.ones(2), cut_modules=[m.layer1, m.layer2],
                           sparsify_fn=topk_sparsify_per_sample).to(device)


@torch.no_grad()
def sweep(cm, xs, ys, grid, device):
    cm.eval(); out = []
    for e in grid:
        cm.set_eta(torch.tensor([e, e], dtype=torch.float32, device=device))
        c = 0
        for i in range(0, len(xs), 512):
            c += (cm(xs[i:i+512]).argmax(1) == ys[i:i+512]).sum().item()
        out.append(c / len(xs))
    return np.array(out)


def finetune_narrow(cm, train_loader, device, steps=800, lr=1e-4, lo=0.1, hi=0.2):
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(cm.parameters(), lr=lr)
    data = cycle(train_loader); cm.train()
    for _ in range(steps):
        eta = rng.uniform(lo, hi, size=2)
        cm.set_eta(torch.tensor(eta, dtype=torch.float32, device=device))
        x, y = next(data); x, y = x.to(device), y.to(device)
        opt.zero_grad(); F.cross_entropy(cm(x), y).backward(); opt.step()
    return cm


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = select_device()
    train_loader, test_loader = get_cifar10_loaders()
    x_test, y_test = load_test_tensors(test_loader)
    xs, ys = x_test[:2000].to(device), y_test[:2000].to(device)
    grid = np.round(np.linspace(0.1, 1.0, 13), 3)

    print("off-the-shelf ResNet-20 ..."); off20 = sweep(build("cifar10_resnet20", device), xs, ys, grid, device)
    print("off-the-shelf ResNet-56 ..."); off56 = sweep(build("cifar10_resnet56", device), xs, ys, grid, device)
    print("narrow-band retrained ResNet-20 ...")
    r20n = build("cifar10_resnet20", device); finetune_narrow(r20n, train_loader, device)
    ret20 = sweep(r20n, xs, ys, grid, device)

    for nm, a in [("off r20", off20), ("off r56", off56), ("retrained r20", ret20)]:
        print(f"  {nm}: eta=0.1 -> {a[0]:.3f},  eta=1 -> {a[-1]:.3f},  peak {a.max():.3f}@eta={grid[a.argmax()]}")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(grid, off20, "o-", color="#2c6fbb", label="ResNet-20 off-the-shelf")
    ax.plot(grid, off56, "s-", color="#2ca02c", label="ResNet-56 off-the-shelf")
    ax.plot(grid, ret20, "^--", color="#e07b39", label="ResNet-20 retrained (narrow band)")
    ax.axvline(1.0, ls=":", c="r")
    ax.set_xlabel(r"$\eta$ (with $\eta_1=\eta_2$)"); ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs $\\eta$: off-the-shelf is monotone; retraining inverts it")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "presentation", "figs", "eta_offshelf.png"))
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
