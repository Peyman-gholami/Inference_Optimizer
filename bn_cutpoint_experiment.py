"""
Does a cut-point normalization fix the monotonicity break?

Reproduces the collapse in a controlled single-model setting (ResNet-20) and tests the
recommendation. Three conditions, each fine-tuned then swept over eta (diagonal eta1=eta2):

  A. baseline top-k,          NARROW-band training  eta~U[0.1,0.2]   (the pathology)
  B. cut-point RENORM top-k,  NARROW-band training                   (the proposed fix)
  C. baseline top-k,          FULL-range training   eta~U[0.1,1.0]   (control: not the norm)

If the normalization works, B's accuracy-vs-eta curve should flatten (eta=1 recovers)
relative to A; C shows how much of the fix is simply "don't train narrow-band".

Saves presentation/figs/cutpoint_norm.png.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python bn_cutpoint_experiment.py
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
    topk_sparsify_per_sample, topk_sparsify_per_sample_renorm,
)

STEPS, LR = int(os.environ.get("BN_STEPS", "800")), 1e-4
rng = np.random.default_rng(0)


def build_cm(sparsify_fn, device):
    model = load_pretrained_cifar_resnet("cifar10_resnet20").to(device)
    cm = CompressedModel(model, torch.ones(2), cut_modules=[model.layer1, model.layer2],
                         sparsify_fn=sparsify_fn)
    return cm.to(device)


def finetune(cm, train_loader, eta_lo, eta_hi, device):
    opt = torch.optim.Adam(cm.parameters(), lr=LR)
    data = cycle(train_loader)
    cm.train()
    for _ in range(STEPS):
        eta = rng.uniform(eta_lo, eta_hi, size=2)
        cm.set_eta(torch.tensor(eta, dtype=torch.float32, device=device))
        x, y = next(data)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(cm(x), y).backward()
        opt.step()
    return cm


@torch.no_grad()
def sweep(cm, xs, ys, grid, device):
    cm.eval()
    out = []
    for e in grid:
        cm.set_eta(torch.tensor([e, e], dtype=torch.float32, device=device))
        correct = 0
        for i in range(0, len(xs), 512):
            correct += (cm(xs[i:i+512]).argmax(1) == ys[i:i+512]).sum().item()
        out.append(correct / len(xs))
    return np.array(out)


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = select_device()
    train_loader, test_loader = get_cifar10_loaders()
    x_test, y_test = load_test_tensors(test_loader)
    xs, ys = x_test[:2000].to(device), y_test[:2000].to(device)
    grid = np.round(np.linspace(0.1, 1.0, 13), 3)

    conds = [
        ("A: baseline, narrow train",       topk_sparsify_per_sample,        (0.1, 0.2)),
        ("B: cut-point renorm, narrow",     topk_sparsify_per_sample_renorm, (0.1, 0.2)),
        ("C: baseline, full-range train",   topk_sparsify_per_sample,        (0.1, 1.0)),
    ]
    results = {}
    for name, fn, (lo, hi) in conds:
        print(f"=== {name} ===")
        cm = build_cm(fn, device)
        finetune(cm, train_loader, lo, hi, device)
        acc = sweep(cm, xs, ys, grid, device)
        results[name] = acc
        print(f"  acc(eta=0.15)~{acc[0]:.3f}(lo)  acc(eta=1)={acc[-1]:.3f}  peak={acc.max():.3f}@eta={grid[acc.argmax()]}")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    styles = {"A: baseline, narrow train": ("o-", "#2c6fbb"),
              "B: cut-point renorm, narrow": ("s-", "#e07b39"),
              "C: baseline, full-range train": ("^--", "#888888")}
    for name, acc in results.items():
        st, c = styles[name]
        ax.plot(grid, acc, st, color=c, label=name)
    ax.axvline(1.0, ls=":", c="r")
    ax.set_xlabel(r"$\eta$ (with $\eta_1=\eta_2$)"); ax.set_ylabel("test accuracy")
    ax.set_title("Cut-point normalization vs the monotonicity break (ResNet-20)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "presentation", "figs", "cutpoint_norm.png"))
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
