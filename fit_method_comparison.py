"""
fit_method_comparison.py
========================
Compare A_hat surrogate families at three points in a model's life.

Stages (r56 + r20 sharing bandwidth, so "coupled" is meaningful):
  base        off-the-shelf, before any pretraining
  decoupled   after N compression-aware steps, g = direct
  coupled     after N steps from the SAME base, g = direct + J^T v  (A+B)

At each stage we measure the TRUE A(eta) on a common grid per task, fit every
family to that same data, and report the metrics the MobiHoc paper uses for
Table 7 (appendix: "RMSE, R^2, and the inference latency of a single prediction
call, since online optimization is sensitive to prediction overhead"), plus the
two diagnostics that table lacks:

  RMSE          in-sample, on the fit grid
  R^2           1 - SSE/SST, reported "--" when sd(A) < 0.01: with a near
                constant target R^2 is meaningless (Table 7 has -105348 and
                -inf entries for exactly this reason)
  sd(A)         so the reader can see when R^2 is unusable
  max eig H     concavity -- <= 0 means concave on the box. Water-filling
                assumes this and the product family does not guarantee it
  monotone      nondecreasing in every eta_i
  fit / predict fit cost once, and microseconds per A() call (the paper's
                "Inf. (ms)" -- SLSQP calls A and gradA many times per solve)

NOTE the training itself is unchanged -- it still uses `LiveAccuracyProfile`
as the controller. This experiment asks which family best *describes* A at each
stage; wiring a different family into the controller needs `SurrogateCoupled`
to become family-aware, which is a separate step.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python fit_method_comparison.py --steps 120
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pretraining.accuracy_fits import build_fit
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.compression_aware import ChannelSampler
from src.pretraining.coupled import (
    SurrogateCoupled, kkt_adjoint_reduced, stein_grad, stein_hybrid_basis,
)
from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)

BATCH, LR = 128, 1e-4
N_V, N_M, SIGMA, RCOND = 32, 20, 0.05, 1e-3
REFIT = 25
W = [1.0, 1.0]
R = [1.0, 1.5]
NAMES = ["cifar10_resnet56", "cifar10_resnet20"]
A_ARR = np.array([[16384.0, 8192.0], [16384.0, 8192.0]])
ETA_MIN = 0.1

# fits to report in the table; the first four are also plotted
FIT_ORDER = ["two_stage", "joint", "separable", "net", "joint_guarded"]
PLOT_FITS = ["two_stage", "joint", "separable", "net"]
# validated categorical slots 1-4 (adjacent gates: CVD dE 9.1, normal 22.9);
# dash patterns are the secondary encoding, since two slots carry a contrast WARN
STYLE = {
    "two_stage": ("#2a78d6", (0, (1, 1.4)), "two-stage (current)"),
    "joint":     ("#eb6834", (0, (4, 1.6)), "joint"),
    "separable": ("#1baf7a", (0, (6, 1.4, 1.4, 1.4)), "separable concave"),
    "net":       ("#eda100", "-", "concave net"),
}
INK = "#333333"


def r2(y, yhat, sd_floor=0.01):
    y = np.asarray(y, float)
    if y.std() < sd_floor:
        return None                      # meaningless: no variance to explain
    sse = float(((yhat - y) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=120, help="Steps per training arm.")
    p.add_argument("--n_fit", type=int, default=6, help="Fit grid: n_fit^2 points.")
    p.add_argument("--n_plot", type=int, default=25, help="Dense marginal for plotting.")
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--net_epochs", type=int, default=2500)
    p.add_argument("--out_json", default="outputs/fit_compare/fit_method_comparison.json")
    p.add_argument("--out_fig", default="../presentation/figs/fit_method_comparison.png")
    p.add_argument("--render_only", action="store_true",
                   help="With --load_json: re-render from stored curves, no refit.")
    p.add_argument("--load_json", default="",
                   help="Refit and replot from a previous run's measured curves "
                        "(no training, no model evaluation). Use with --net_epochs "
                        "to re-assess the net at a different budget.")
    args = p.parse_args()

    if args.load_json:
        if args.render_only:
            d = json.loads(Path(args.load_json).read_text())
            print_table(d["results"])
            render(d["results"], np.array(d["plot_grid"]), Path(args.out_fig))
        else:
            replot(args)
        return

    torch.manual_seed(0)
    np.random.seed(0)
    dev = select_device()
    train_loader, test = get_cifar10_loaders()
    xt, yt = load_test_tensors(test)
    x_meas, y_meas = xt[:1000].to(dev), yt[:1000].to(dev)
    x_ev, y_ev = xt[1000:1000 + args.n_eval].to(dev), yt[1000:1000 + args.n_eval].to(dev)

    ref = float(2 * np.exp(np.mean(np.log(
        [R[k] * A_ARR[k, i] for k in range(2) for i in range(2)]))))

    cms = [make_compressed_resnet(load_pretrained_cifar_resnet(m).to(dev),
                                  model_name=m)[0].to(dev) for m in NAMES]
    for cm in cms:
        cm.eval()
    profs = [LiveAccuracyProfile(cms[k], x_meas, y_meas, 2, dev) for k in range(2)]
    opts = [torch.optim.Adam(cms[k].parameters(), lr=LR) for k in range(2)]
    base_snap = {"m": [copy.deepcopy(c.state_dict()) for c in cms],
                 "o": [copy.deepcopy(o.state_dict()) for o in opts]}

    ax = np.linspace(ETA_MIN, 1.0, args.n_fit)
    FIT_GRID = np.array(np.meshgrid(ax, ax)).reshape(2, -1).T          # (n^2, 2)
    PLOT_G = np.linspace(ETA_MIN, 1.0, args.n_plot)

    def set_grad(cm, flat):
        i = 0
        for prm in cm.parameters():
            n = prm.numel()
            prm.grad = flat[i:i + n].reshape(prm.shape).clone()
            i += n

    @torch.no_grad()
    def true_A(k, eta):
        cms[k].set_eta(torch.tensor(eta, dtype=torch.float32, device=dev))
        correct = 0
        for i in range(0, len(x_ev), 512):
            correct += (cms[k](x_ev[i:i + 512]).argmax(1) == y_ev[i:i + 512]).sum().item()
        return correct / len(x_ev)

    def ce_on(k, xb, yb, e):
        with torch.no_grad():
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            return float(F.cross_entropy(cms[k](xb), yb))

    def flat_grad(scalar, cm):
        ps = list(cm.parameters())
        g = torch.autograd.grad(scalar, ps, allow_unused=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(q)).reshape(-1)
                          for gi, q in zip(g, ps)]).detach().cpu().numpy()

    def gradA_on(k, xb, yb, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        pr = torch.softmax(cms[k](xb), 1)
        return flat_grad(W[k] * pr[torch.arange(len(yb)), yb].mean(), cms[k])

    def direct_on(k, xb, yb, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        return flat_grad(F.cross_entropy(cms[k](xb), yb), cms[k])

    def train(mode, n):
        sampler = ChannelSampler(2, np.array([0.1 * ref] * 2),
                                 np.array([1.2 * ref] * 2), seed=1, kind="loguniform")
        data = cycle(train_loader)
        t0 = time.time()
        for step in range(n):
            if step % REFIT == 0:
                for q in profs:
                    q.refit()
            alpha = np.concatenate([q.alpha for q in profs])
            w_eff = [W[k] * float(profs[k].c0) for k in range(2)]
            prob = SurrogateCoupled(w=w_eff, R=R, a=A_ARR, c=sampler(), eta_min=ETA_MIN)
            eta = prob.solve(alpha)
            xb, yb = next(data)
            xb, yb = xb.to(dev), yb.to(dev)
            bases, adj = {}, None
            if mode == "coupled":
                active, free = prob.active_set(eta)
                Hm = prob.hess_eta(eta, alpha)
                v = np.zeros(4)
                for k in range(2):
                    ek = eta[k * 2:k * 2 + 2]
                    v[k * 2:k * 2 + 2] = stein_grad(
                        lambda e: ce_on(k, xb, yb, e), ek, SIGMA, N_V, seed=step)
                    bases[k] = stein_hybrid_basis(
                        lambda e: gradA_on(k, xb, yb, e), ek, SIGMA, N_M, seed=step)
                adj = kkt_adjoint_reduced(Hm, prob.B, active, free, 4, v, rcond=RCOND)
            for k in range(2):
                ek = eta[k * 2:k * 2 + 2]
                g = direct_on(k, xb, yb, ek)
                if mode == "coupled":
                    g = g - bases[k] @ adj[k * 2:k * 2 + 2]
                set_grad(cms[k], torch.tensor(g, dtype=torch.float32, device=dev))
                opts[k].step()
                opts[k].zero_grad()
                cms[k].eval()
            if (step + 1) % 20 == 0:
                print(f"    [{mode}] {step+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    def assess(stage):
        """Measure true A, fit every family, collect metrics + plot curves."""
        out = {}
        for k in range(2):
            t0 = time.time()
            y_fit = np.array([true_A(k, e) for e in FIT_GRID])
            y_marg = np.array([true_A(k, [e, 1.0]) for e in PLOT_G])
            print(f"    task {k}: measured {len(FIT_GRID)}+{len(PLOT_G)} points "
                  f"({time.time()-t0:.0f}s)  A in [{y_fit.min():.3f},{y_fit.max():.3f}]",
                  flush=True)
            rec = {"y_fit": y_fit.tolist(), "y_marg": y_marg.tolist(), "fits": {}}
            for name in FIT_ORDER:
                f = build_fit(name, 2, ETA_MIN)
                if name == "net":
                    f.epochs = args.net_epochs
                t1 = time.time()
                f.fit(FIT_GRID, y_fit)
                t_fit = time.time() - t1
                yhat = f.A_many(FIT_GRID)
                t2 = time.time()
                for _ in range(200):
                    f.A(np.array([0.5, 0.5]))
                t_pred = (time.time() - t2) / 200 * 1e6        # microseconds per call
                eig, where = f.max_hess_eig(n=7)
                rec["fits"][name] = {
                    "rmse": float(np.sqrt(np.mean((yhat - y_fit) ** 2))),
                    "r2": r2(y_fit, yhat),
                    "sd_A": float(y_fit.std()),
                    "max_eig_H": float(eig), "eig_at": list(where),
                    "monotone": bool(f.is_monotone(n=7)),
                    "fit_s": float(t_fit), "predict_us": float(t_pred),
                    "describe": f.describe(),
                    "curve": f.A_many(np.stack([PLOT_G, np.ones_like(PLOT_G)], 1)).tolist(),
                }
            out[f"task{k}"] = rec
        return out

    results = {}
    print("\n===== stage: base (off-the-shelf) =====", flush=True)
    results["base"] = assess("base")

    print(f"\n===== stage: decoupled ({args.steps} steps) =====", flush=True)
    train("decoupled", args.steps)
    results["decoupled"] = assess("decoupled")

    print(f"\n===== stage: coupled ({args.steps} steps from base) =====", flush=True)
    for k in range(2):
        cms[k].load_state_dict(copy.deepcopy(base_snap["m"][k]))
        opts[k].load_state_dict(copy.deepcopy(base_snap["o"][k]))
        cms[k].eval()
    for q in profs:
        q.refit()
    train("coupled", args.steps)
    results["coupled"] = assess("coupled")

    # ----------------------------- table -----------------------------------
    print("\n" + "=" * 100)
    hdr = (f"{'stage':<11}{'task':<10}{'fit':<24}{'RMSE':>9}{'R^2':>11}{'sd(A)':>8}"
           f"{'maxeigH':>10}{'mono':>6}{'fit s':>8}{'us/call':>9}")
    print(hdr)
    for stage in ("base", "decoupled", "coupled"):
        for k in range(2):
            for name in FIT_ORDER:
                m = results[stage][f"task{k}"]["fits"][name]
                r2s = "--" if m["r2"] is None else f"{m['r2']:.4f}"
                print(f"{stage:<11}{NAMES[k].replace('cifar10_',''):<10}{name:<24}"
                      f"{m['rmse']:>9.4f}{r2s:>11}{m['sd_A']:>8.4f}"
                      f"{m['max_eig_H']:>+10.3f}{'y' if m['monotone'] else 'N':>6}"
                      f"{m['fit_s']:>8.2f}{m['predict_us']:>9.1f}")
        print("-" * 100)

    jp = Path(args.out_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"config": vars(args), "plot_grid": PLOT_G.tolist(),
                              "fit_grid": FIT_GRID.tolist(), "names": NAMES,
                              "results": results}, indent=2))
    print(f"\nwrote {jp}")
    render(results, PLOT_G, Path(args.out_fig))


def refit_stage(rec, fit_grid, plot_g, net_epochs):
    """Re-fit every family to ALREADY MEASURED accuracies (no model needed)."""
    y_fit = np.array(rec["y_fit"])
    out = {"y_fit": rec["y_fit"], "y_marg": rec["y_marg"], "fits": {}}
    for name in FIT_ORDER:
        f = build_fit(name, 2, ETA_MIN)
        if name == "net":
            f.epochs = net_epochs
            f.lr = 0.02 if net_epochs >= 20000 else 0.05
        t1 = time.time()
        f.fit(fit_grid, y_fit)
        t_fit = time.time() - t1
        yhat = f.A_many(fit_grid)
        t2 = time.time()
        for _ in range(200):
            f.A(np.array([0.5, 0.5]))
        t_pred = (time.time() - t2) / 200 * 1e6
        eig, where = f.max_hess_eig(n=7)
        out["fits"][name] = {
            "rmse": float(np.sqrt(np.mean((yhat - y_fit) ** 2))),
            "r2": r2(y_fit, yhat), "sd_A": float(y_fit.std()),
            "max_eig_H": float(eig), "eig_at": list(where),
            "monotone": bool(f.is_monotone(n=7)),
            "fit_s": float(t_fit), "predict_us": float(t_pred),
            "describe": f.describe(),
            "curve": f.A_many(np.stack([plot_g, np.ones_like(plot_g)], 1)).tolist(),
        }
    return out


def replot(args):
    d = json.loads(Path(args.load_json).read_text())
    fit_grid = np.array(d["fit_grid"])
    plot_g = np.array(d["plot_grid"])
    results = {}
    for stage in ("base", "decoupled", "coupled"):
        results[stage] = {}
        for k in range(2):
            print(f"  refit {stage}/task{k} (net at {args.net_epochs} epochs)...",
                  flush=True)
            results[stage][f"task{k}"] = refit_stage(
                d["results"][stage][f"task{k}"], fit_grid, plot_g, args.net_epochs)
    print_table(results)
    jp = Path(args.out_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"config": vars(args), "plot_grid": plot_g.tolist(),
                              "fit_grid": fit_grid.tolist(), "names": NAMES,
                              "results": results}, indent=2))
    print(f"\nwrote {jp}")
    render(results, plot_g, Path(args.out_fig))


def print_table(results):
    print("\n" + "=" * 100)
    hdr = (f"{'stage':<11}{'task':<10}{'fit':<24}{'RMSE':>9}{'R^2':>11}{'sd(A)':>8}"
           f"{'maxeigH':>10}{'mono':>6}{'fit s':>8}{'us/call':>9}")
    print(hdr)
    for stage in ("base", "decoupled", "coupled"):
        for k in range(2):
            for name in FIT_ORDER:
                m = results[stage][f"task{k}"]["fits"][name]
                r2s = "--" if m["r2"] is None else f"{m['r2']:.4f}"
                print(f"{stage:<11}{NAMES[k].replace('cifar10_',''):<10}{name:<24}"
                      f"{m['rmse']:>9.4f}{r2s:>11}{m['sd_A']:>8.4f}"
                      f"{m['max_eig_H']:>+10.3f}{'y' if m['monotone'] else 'N':>6}"
                      f"{m['fit_s']:>8.2f}{m['predict_us']:>9.1f}")
        print("-" * 100)


def render(results, pg, out):
    stages = ["base", "decoupled", "coupled"]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 3.5), sharex=True)
    for r in range(2):                      # row = task
        for c, stage in enumerate(stages):  # col = stage
            ax = axes[r][c]
            rec = results[stage][f"task{r}"]
            ax.plot(pg, rec["y_marg"], "-", color="#222222", lw=2.4, zorder=6,
                    marker="o", ms=3.0, markeredgecolor="white", markeredgewidth=0.4)
            for name in PLOT_FITS:
                col, dash, _ = STYLE[name]
                ax.plot(pg, rec["fits"][name]["curve"], color=col, lw=1.9,
                        linestyle=dash, zorder=4)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#bbbbbb")
            ax.tick_params(colors=INK, labelsize=7.5)
            if r == 0:
                ax.set_title(stage, fontsize=10.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{NAMES[r].replace('cifar10_','')}\ntop-1 accuracy",
                              fontsize=9, color=INK)
            if r == 1:
                ax.set_xlabel(r"$\eta_1$   ($\eta_2$ held at 1)", fontsize=9.5, color=INK)
    # one legend, with the line style carrying identity alongside colour
    handles = [plt.Line2D([], [], color="#222222", lw=2.4, marker="o", ms=3.5,
                          label="true $A$")]
    handles += [plt.Line2D([], [], color=STYLE[n][0], lw=1.9, linestyle=STYLE[n][1],
                           label=STYLE[n][2]) for n in PLOT_FITS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
