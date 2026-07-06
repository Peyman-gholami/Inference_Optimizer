"""
Confirm the monotonicity break. Reproduce the asymmetric multi-task training
(ResNet-56 + ResNet-20, w=[1.5,1], same seed/config that starved ResNet-20 to the
eta floor), then sweep the comp-aware ResNet-20's TRUE accuracy over the (eta1,eta2)
grid. Expect a peak at low eta (where it was trained) and a dip at eta=1, vs
ResNet-56 which should stay monotone-ish.

Saves presentation/figs/eta_sweep_starved.png and prints key points.
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python sweep_eta_curve.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.compression_aware import ChannelSampler
from src.pretraining.multitask_live import build_tasks_with_profiles, decoupled_pretrain_live
from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = select_device()
    names = ["cifar10_resnet56", "cifar10_resnet20"]
    w_list, R_list = [1.5, 1.0], [1.0, 1.0]

    train_loader, test_loader = get_cifar10_loaders()
    x_test, y_test = load_test_tensors(test_loader)
    x_meas, y_meas = x_test[:2000], y_test[:2000]

    cms, a = [], None
    for m in names:
        cm, a = make_compressed_resnet(load_pretrained_cifar_resnet(m).to(device))
        cms.append(cm.to(device))
    d = cms[0].d; K = len(cms); eta_min = np.full(d, 0.1)
    c_min, c_max = K * a * eta_min, 1.2 * K * a
    sampler = ChannelSampler(d, c_min, c_max, seed=0)
    profs = [LiveAccuracyProfile(cm, x_meas, y_meas, d, device) for cm in cms]
    tasks, M = build_tasks_with_profiles(profs, a, eta_min, R_list, w_list)
    opt = CSIAwareMultiTaskOptimizer(M, tasks)

    print("training (reproducing the starvation) ...")
    decoupled_pretrain_live(cms, train_loader, opt, profs, sampler=sampler,
                            steps=1000, lr=1e-4, device=device, refit_every=100, log_every=200)

    xs, ys = x_test[:2000].to(device), y_test[:2000].to(device)

    @torch.no_grad()
    def acc(cm, e):
        cm.eval()
        cm.set_eta(torch.tensor(e, dtype=torch.float32, device=device))
        correct = 0
        for i in range(0, len(xs), 512):
            correct += (cm(xs[i:i+512]).argmax(1) == ys[i:i+512]).sum().item()
        return correct / len(xs)

    grid = np.round(np.linspace(0.1, 1.0, 10), 3)
    H = np.array([[acc(cms[1], [e1, e2]) for e2 in grid] for e1 in grid])  # rows=eta1, cols=eta2

    diag = np.round(np.linspace(0.1, 1.0, 13), 3)
    d20 = [acc(cms[1], [e, e]) for e in diag]
    d56 = [acc(cms[0], [e, e]) for e in diag]

    clean20 = acc(cms[1], [1.0, 1.0]); dep20 = acc(cms[1], [0.1, 0.25])
    peak = np.unravel_index(np.argmax(H), H.shape)
    print(f"ResNet-20  clean(1,1)={clean20:.4f}   acc(0.1,0.25)={dep20:.4f}   "
          f"grid-peak acc={H.max():.4f} at (eta1={grid[peak[0]]},eta2={grid[peak[1]]})")

    fig, (axh, axd) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    im = axh.imshow(H, origin="lower", extent=[grid[0], grid[-1], grid[0], grid[-1]],
                    aspect="auto", cmap="viridis")
    axh.set_xlabel(r"$\eta_2$ (layer2)"); axh.set_ylabel(r"$\eta_1$ (layer1)")
    axh.set_title("comp-aware ResNet-20: accuracy over $(\\eta_1,\\eta_2)$")
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    axh.scatter([1.0], [1.0], c="red", marker="*", s=160, label="clean $(1,1)$")
    axh.scatter([0.25], [0.1], c="white", marker="o", s=70, edgecolors="k",
                label="deployed $\\sim[0.1,0.25]$")
    axh.legend(loc="upper left", fontsize=8)

    axd.plot(diag, d20, "o-", color="#2c6fbb", label="ResNet-20 (starved)")
    axd.plot(diag, d56, "s-", color="#888888", label="ResNet-56")
    axd.axvline(1.0, ls=":", c="r")
    axd.set_xlabel(r"$\eta$ (with $\eta_1=\eta_2$)"); axd.set_ylabel("test accuracy")
    axd.set_title("diagonal sweep"); axd.legend(fontsize=9); axd.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "presentation", "figs", "eta_sweep_starved.png"))
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
