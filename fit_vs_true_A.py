"""
fit_vs_true_A.py
================
What the controller believes vs. what is true, for a starved ResNet-20.

`LiveAccuracyProfile` fits the monotone-concave family

    A_hat(eta) = c0 * prod_i (1 - alpha_i (1 - eta_i)^2),   alpha_i in [0, 0.999)

which is increasing in every eta_i and maximal at eta = 1. After narrow-band
training (eta ~ U[0.1, 0.2], the starvation regime) the TRUE accuracy is
inverted: it peaks near the trained band and falls at eta = 1, because BN stats
and post-cut weights have specialised to sparse top-k activations, so eta = 1 is
out of distribution.

The fit family cannot represent a low-eta peak. Worse, it fails in a specific
way: c0 is measured at eta = 1 -- the WORST point -- and then any measured
accuracy above c0 gives 1 - acc/c0 < 0, so alpha is clipped to 0 and A_hat
collapses to a flat line at the clean accuracy.

This script measures both curves on the SAME marginals the fit actually uses
(sweep cut i, hold the others at 1) and plots them together.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python fit_vs_true_A.py
  BN_STEPS=800 python fit_vs_true_A.py --out ../presentation/figs/fit_vs_true_A.png
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.accuracy_fn import CompressedModel
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    load_test_tensors, topk_sparsify_per_sample,
)

# Validated categorical slots (see presentation/make_figs_coupling.py).
C_TRUE = "#2a78d6"     # slot 1 -- the truth
C_FIT = "#eb6834"      # slot 2 -- what the controller believes
C_REF = "#8d8d8d"      # recessive -- the pristine model, for reference
INK = "#333333"

CUT_NAMES = ["cut-point 1 (layer1)", "cut-point 2 (layer2)"]


def build_cm(device):
    model = load_pretrained_cifar_resnet("cifar10_resnet20").to(device)
    cm = CompressedModel(model, torch.ones(2), cut_modules=[model.layer1, model.layer2],
                         sparsify_fn=topk_sparsify_per_sample)
    return cm.to(device)


def finetune(cm, train_loader, eta_lo, eta_hi, device, steps, lr=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(cm.parameters(), lr=lr)
    data = cycle(train_loader)
    cm.train()
    for t in range(steps):
        eta = rng.uniform(eta_lo, eta_hi, size=2)
        cm.set_eta(torch.tensor(eta, dtype=torch.float32, device=device))
        x, y = next(data)
        opt.zero_grad()
        F.cross_entropy(cm(x.to(device)), y.to(device)).backward()
        opt.step()
        if (t + 1) % 100 == 0:
            print(f"    step {t+1}/{steps}", flush=True)
    cm.eval()
    return cm


@torch.no_grad()
def marginal_true_A(cm, xs, ys, cut, grid, device, bs=512):
    """True top-1 accuracy sweeping ONE cut-point, others held at 1.

    This is exactly the slice LiveAccuracyProfile fits alpha_i on, so the two
    curves are directly comparable rather than two different cross-sections.
    """
    cm.eval()
    out = []
    for e in grid:
        eta = np.ones(2)
        eta[cut] = e
        cm.set_eta(torch.tensor(eta, dtype=torch.float32, device=device))
        correct = 0
        for i in range(0, len(xs), bs):
            correct += (cm(xs[i:i + bs]).argmax(1) == ys[i:i + bs]).sum().item()
        out.append(correct / len(xs))
    return np.array(out)


def marginal_fitted_A(prof: LiveAccuracyProfile, cut, grid):
    """A_hat on the same marginal: c0 * (1 - alpha_cut (1 - eta)^2)."""
    return np.array([prof.A(np.where(np.arange(prof.d) == cut, e, 1.0)) for e in grid])


def fitted_on(c0, alpha, cut, grid):
    """A_hat marginal from saved parameters (no model needed)."""
    return np.array([c0 * (1.0 - alpha[cut] * (1.0 - e) ** 2) for e in grid])


def render(grid, true, fit, ref, eta_grid, c0, alpha, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 2.7), sharey=True)
    for c, ax in enumerate(axes):
        ax.plot(grid, ref[c], "--", color=C_REF, lw=1.8, zorder=2)
        ax.plot(grid, true[c], "-", color=C_TRUE, lw=2.2, zorder=4,
                marker="o", ms=3.5, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(grid, fit[c], "-", color=C_FIT, lw=2.2, zorder=3)
        # the six points the fit was actually given
        gp = np.asarray(eta_grid)
        ax.plot(gp, fitted_on(c0, alpha, c, gp), "o", color=C_FIT, ms=5.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#bbbbbb")
        ax.tick_params(colors=INK, labelsize=9)
        # title folded into the axis label: frees vertical space and stops the
        # series labels colliding with it at this aspect ratio
        ax.set_xlabel(rf"$\eta_{c+1}$ --- {CUT_NAMES[c]}, other held at 1",
                      fontsize=9.5, color=INK)
    axes[0].set_ylabel("top-1 accuracy", fontsize=9.5, color=INK)

    # direct labels on the left panel only; identity is never colour-alone
    ax = axes[0]
    at = lambda frac: min(len(grid) - 1, max(0, int(round(frac * (len(grid) - 1)))))
    i_t, i_f, i_r = at(0.15), at(0.50), at(0.55)
    ax.annotate("true $A$ (starved)", xy=(grid[i_t], true[0][i_t]), xytext=(14, 6),
                textcoords="offset points", color=C_TRUE, fontsize=9, fontweight="bold")
    ax.annotate(r"fitted $\hat{A}$", xy=(grid[i_f], fit[0][i_f]), xytext=(6, -18),
                textcoords="offset points", color=C_FIT, fontsize=9, fontweight="bold")
    ax.annotate("pristine model", xy=(grid[i_r], ref[0][i_r]), xytext=(-6, -15),
                textcoords="offset points", color="#6f6f6f", fontsize=8.5, ha="center")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"\nwrote {out}")

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=int(os.environ.get("BN_STEPS", "800")),
                   help="Narrow-band fine-tuning steps (the starvation regime).")
    p.add_argument("--eta_lo", type=float, default=0.1)
    p.add_argument("--eta_hi", type=float, default=0.2)
    p.add_argument("--n_sweep", type=int, default=25)
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--out", default="../presentation/figs/fit_vs_true_A.png")
    p.add_argument("--json_out", default="outputs/fit_vs_true/fit_vs_true_A.json")
    p.add_argument("--load_json", default="",
                   help="Replot from a previous run's JSON (no retraining).")
    args = p.parse_args()

    if args.load_json:
        d = json.loads(Path(args.load_json).read_text())
        grid = np.array(d["grid"])
        true = [np.array(v) for v in d["true_A"]]
        fit = [np.array(v) for v in d["fitted_A"]]
        ref = [np.array(v) for v in d["pristine_A"]]
        render(grid, true, fit, ref, np.array(d["eta_grid"]), d["c0"],
               np.array(d["alpha"]), Path(args.out))
        return

    torch.manual_seed(0)
    np.random.seed(0)
    device = select_device()
    train_loader, test_loader = get_cifar10_loaders()
    xt, yt = load_test_tensors(test_loader)
    xs, ys = xt[:args.n_eval].to(device), yt[:args.n_eval].to(device)
    # the profile fits on its own held-out slice, as in the real pipeline
    xm, ym = xt[args.n_eval:args.n_eval + 1000].to(device), yt[args.n_eval:args.n_eval + 1000].to(device)

    grid = np.linspace(args.eta_lo, 1.0, args.n_sweep)

    print(f"[1/3] pristine ResNet-20 reference sweep", flush=True)
    cm_ref = build_cm(device)
    ref = [marginal_true_A(cm_ref, xs, ys, c, grid, device) for c in (0, 1)]
    del cm_ref

    print(f"[2/3] narrow-band fine-tune  eta ~ U[{args.eta_lo},{args.eta_hi}]  "
          f"({args.steps} steps)", flush=True)
    cm = build_cm(device)
    finetune(cm, train_loader, args.eta_lo, args.eta_hi, device, args.steps)

    print("[3/3] true A vs fitted A-hat on the fit's own marginals", flush=True)
    prof = LiveAccuracyProfile(cm, xm, ym, 2, device)     # mode='hard' = the real pipeline
    prof.refit()
    true = [marginal_true_A(cm, xs, ys, c, grid, device) for c in (0, 1)]
    fit = [marginal_fitted_A(prof, c, grid) for c in (0, 1)]

    print(f"\n  fitted c0 = {prof.c0:.4f}   alpha = {np.round(prof.alpha, 4)}")
    for c in (0, 1):
        print(f"  {CUT_NAMES[c]}:  true A peak {true[c].max():.4f} at "
              f"eta={grid[int(true[c].argmax())]:.2f}   true A(1)={true[c][-1]:.4f}   "
              f"fitted A-hat(1)={fit[c][-1]:.4f}")

    render(grid, true, fit, ref, np.asarray(prof.eta_grid), float(prof.c0),
           prof.alpha, Path(args.out))


    jp = Path(args.json_out)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({
        "config": vars(args) | {"device": str(device)},
        "c0": float(prof.c0), "alpha": prof.alpha.tolist(),
        "eta_grid": np.asarray(prof.eta_grid).tolist(),
        "grid": grid.tolist(),
        "true_A": [t.tolist() for t in true],
        "fitted_A": [f.tolist() for f in fit],
        "pristine_A": [r.tolist() for r in ref],
    }, indent=2))
    print(f"wrote {jp}")


if __name__ == "__main__":
    main()
