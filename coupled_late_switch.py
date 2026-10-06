"""
coupled_late_switch.py
======================
Train DECOUPLED first, then switch to COUPLED -- does the coupling help once
theta is near a local optimum of the compressed loss?

Why
---
`coupled_vs_decoupled_train.py` ran both arms for 120 steps FROM THE PRETRAINED
INIT and found coupled strictly worse. But `alignment_test.py` then showed that
at the pretrained theta the coupling's alignment with grad F is not even
sign-consistent across Stein seeds (mean ~ 0 ~ the random control), while after
only 50 decoupled steps it becomes seed-consistent and reaches ~60% of the
direct direction's per-unit descent -- because the direct gradient's alignment
collapses ~6x as theta approaches an optimum while the coupling's grows.

So the original comparison switched the coupling on exactly where it is noise,
and paid for that over the whole trajectory. This script tests the other
schedule: warm up decoupled, then fork.

Design
------
One SHARED decoupled warm-up of `--warm` steps (a few epochs), then a snapshot
of BOTH the weights and the Adam state, then two arms of `--steps` steps each
from that identical snapshot:

    decoupled : keep using  g = direct
    coupled   : switch to   g = direct + (J^T v)   [A+B reliable coupling]

The fork means the switch is the ONLY difference between the arms -- same warm
theta, same optimizer moments, same channel stream, same controller.

Tracked: deployed E_c[acc], deployed E_c[CE] (the actual outer objective F),
and clean (eta=1) accuracy, on a held-out slice.

Usage
-----
  PYTORCH_ENABLE_MPS_FALLBACK=1 python coupled_late_switch.py \
      --warm 1200 --steps 60 --out outputs/late_switch/run.json

A coupled step costs ~50s on the M3 (Stein v + Stein-hybrid M per task), a
decoupled step ~2s, so runtime is dominated by --steps on the coupled arm.
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

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.compression_aware import ChannelSampler
from src.pretraining.coupled import (
    SurrogateCoupled, kkt_adjoint_reduced, stein_grad, stein_hybrid_basis,
)

# Matched to coupled_vs_decoupled_train.py so the two are comparable.
BATCH, LR = 128, 1e-4
N_V, N_M, SIGMA, RCOND = 32, 20, 0.05, 1e-3
REFIT, N_EVAL_CH = 25, 12
W = [1.0, 1.0]
R = [1.0, 1.5]
NAMES = ["cifar10_resnet56", "cifar10_resnet20"]
A_ARR = np.array([[16384.0, 8192.0], [16384.0, 8192.0]])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm", type=int, default=1200,
                   help="Shared decoupled warm-up steps (~391/epoch at batch 128).")
    p.add_argument("--steps", type=int, default=60, help="Steps per arm after the fork.")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--warm_eval_every", type=int, default=100)
    p.add_argument("--norm_match", action="store_true",
                   help="Rescale the coupled gradient to ||direct|| so the arms "
                        "differ only in DIRECTION, not step length. Without this, "
                        "the arm inherits Adam moments calibrated on ||direct|| "
                        "over the whole warm-up and then gets a gradient several "
                        "times larger -- the effective step blows up until Adam's "
                        "second moment catches up (beta2=0.999 ~ 1000 steps).")
    p.add_argument("--out", default="outputs/late_switch/run.json")
    args = p.parse_args()

    dev = select_device()
    train_loader, test = get_cifar10_loaders()
    xt, yt = load_test_tensors(test)
    x_meas, y_meas = xt[:1000].to(dev), yt[:1000].to(dev)
    x_ev, y_ev = xt[1000:3000].to(dev), yt[1000:3000].to(dev)

    ref = float(2 * np.exp(np.mean(np.log(
        [R[k] * A_ARR[k, i] for k in range(2) for i in range(2)]))))
    eval_ch = [ChannelSampler(2, np.array([0.1 * ref] * 2),
                              np.array([1.2 * ref] * 2), seed=7)()
               for _ in range(N_EVAL_CH)]

    torch.manual_seed(0)
    np.random.seed(0)
    cms = [make_compressed_resnet(load_pretrained_cifar_resnet(m).to(dev),
                                  model_name=m)[0].to(dev) for m in NAMES]
    for cm in cms:
        cm.eval()
    profs = [LiveAccuracyProfile(cms[k], x_meas, y_meas, 2, dev) for k in range(2)]
    opts = [torch.optim.Adam(cms[k].parameters(), lr=LR) for k in range(2)]

    def set_grad(cm, flat):
        i = 0
        for prm in cm.parameters():
            n = prm.numel()
            prm.grad = flat[i:i + n].reshape(prm.shape).clone()
            i += n

    def ce_on(k, xb, yb, e):
        with torch.no_grad():
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            return float(F.cross_entropy(cms[k](xb), yb))

    def flat_grad(scalar, cm):
        ps = list(cm.parameters())
        g = torch.autograd.grad(scalar, ps, allow_unused=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(prm)).reshape(-1)
                          for gi, prm in zip(g, ps)]).detach().cpu().numpy()

    def gradA_on(k, xb, yb, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        pr = torch.softmax(cms[k](xb), 1)
        return flat_grad(W[k] * pr[torch.arange(len(yb)), yb].mean(), cms[k])

    def direct_on(k, xb, yb, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        return flat_grad(F.cross_entropy(cms[k](xb), yb), cms[k])

    @torch.no_grad()
    def eval_at(k, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        logits = cms[k](x_ev)
        acc = (logits.argmax(1) == y_ev).float().mean().item()
        ce = float(F.cross_entropy(logits, y_ev))
        return acc, ce

    def evaluate():
        """Deployed E_c[acc], deployed E_c[CE] (= the outer objective F), clean acc."""
        for prf in profs:
            prf.refit()
        alpha = np.concatenate([prf.alpha for prf in profs])
        w_eff = [W[k] * float(profs[k].c0) for k in range(2)]
        accs, ces = [], []
        for c in eval_ch:
            ec = SurrogateCoupled(w=w_eff, R=R, a=A_ARR, c=c, eta_min=0.1).solve(alpha)
            pair = [eval_at(k, ec[k * 2:k * 2 + 2]) for k in range(2)]
            accs.append(np.mean([q[0] for q in pair]))
            ces.append(np.mean([q[1] for q in pair]))
        clean = np.mean([eval_at(k, [1.0, 1.0])[0] for k in range(2)])
        return float(np.mean(accs)), float(np.mean(ces)), float(clean)

    def train_steps(mode, n, sampler, data, step0, hist, eval_every, tag):
        for i in range(n):
            step = step0 + i
            if step % REFIT == 0:
                for prf in profs:
                    prf.refit()
            alpha = np.concatenate([prf.alpha for prf in profs])
            w_eff = [W[k] * float(profs[k].c0) for k in range(2)]
            prob = SurrogateCoupled(w=w_eff, R=R, a=A_ARR, c=sampler(), eta_min=0.1)
            eta = prob.solve(alpha)
            xb, yb = next(data)
            xb, yb = xb.to(dev), yb.to(dev)

            bases, adj = {}, None
            if mode == "coupled":
                active, free = prob.active_set(eta)
                H = prob.hess_eta(eta, alpha)
                v = np.zeros(4)
                for k in range(2):
                    ek = eta[k * 2:k * 2 + 2]
                    v[k * 2:k * 2 + 2] = stein_grad(
                        lambda e: ce_on(k, xb, yb, e), ek, SIGMA, N_V, seed=step)
                    bases[k] = stein_hybrid_basis(
                        lambda e: gradA_on(k, xb, yb, e), ek, SIGMA, N_M, seed=step)
                adj = kkt_adjoint_reduced(H, prob.B, active, free, 4, v, rcond=RCOND)

            nrm = {}
            for k in range(2):
                ek = eta[k * 2:k * 2 + 2]
                g = direct_on(k, xb, yb, ek)
                n_direct = float(np.linalg.norm(g))
                n_coup = 0.0
                if mode == "coupled":
                    jtv = -bases[k] @ adj[k * 2:k * 2 + 2]
                    n_coup = float(np.linalg.norm(jtv))
                    g = g + jtv
                    if args.norm_match:
                        # same step magnitude as the decoupled arm; direction only
                        g = g * (n_direct / (float(np.linalg.norm(g)) + 1e-12))
                nrm[k] = (n_direct, n_coup)
                set_grad(cms[k], torch.tensor(g, dtype=torch.float32, device=dev))
                opts[k].step()
                opts[k].zero_grad()
                cms[k].eval()

            if step % eval_every == 0 or i == n - 1:
                dep, depce, clean = evaluate()
                rec = {"step": step, "dep_acc": dep, "dep_ce": depce,
                       "clean": clean,
                       "norm_direct": [nrm[k][0] for k in range(2)],
                       "norm_coupling": [nrm[k][1] for k in range(2)]}
                hist.append(rec)
                ratio = ""
                if mode == "coupled":
                    rs = [nrm[k][1] / (nrm[k][0] + 1e-12) for k in range(2)]
                    ratio = f"  ||Jtv||/||direct||=[{rs[0]:.1f},{rs[1]:.1f}]"
                print(f"[{tag}] step {step:5d}  dep_acc={dep:.4f}  "
                      f"dep_CE={depce:.4f}  clean={clean:.4f}{ratio}", flush=True)

    # ---------------- shared decoupled warm-up ------------------------------
    sampler = ChannelSampler(2, np.array([0.1 * ref] * 2), np.array([1.2 * ref] * 2),
                             seed=1, kind="loguniform")
    data = cycle(train_loader)
    warm_hist = []
    print(f"===== SHARED DECOUPLED WARM-UP ({args.warm} steps "
          f"~ {args.warm/391:.1f} epochs) =====", flush=True)
    t0 = time.time()
    dep, depce, clean = evaluate()
    warm_hist.append({"step": 0, "dep_acc": dep, "dep_ce": depce, "clean": clean})
    print(f"[warmup] step     0  dep_acc={dep:.4f}  dep_CE={depce:.4f}  "
          f"clean={clean:.4f}", flush=True)
    train_steps("decoupled", args.warm, sampler, data, 0, warm_hist,
                args.warm_eval_every, "warmup")
    print(f"warm-up done in {(time.time()-t0)/60:.1f} min", flush=True)

    # snapshot weights AND Adam moments, plus the channel/data stream position
    snap = {
        "models": [copy.deepcopy(cm.state_dict()) for cm in cms],
        "opts": [copy.deepcopy(o.state_dict()) for o in opts],
        "sampler": copy.deepcopy(sampler),
    }

    def restore():
        for k in range(2):
            cms[k].load_state_dict(copy.deepcopy(snap["models"][k]))
            opts[k].load_state_dict(copy.deepcopy(snap["opts"][k]))
            cms[k].eval()
        for prf in profs:
            prf.refit()

    arms = {}
    for mode in ("decoupled", "coupled"):
        print(f"\n===== ARM: {mode.upper()} ({args.steps} steps from the warm "
              f"snapshot) =====", flush=True)
        restore()
        # identical channel and data streams for both arms
        arm_sampler = copy.deepcopy(snap["sampler"])
        arm_data = cycle(train_loader)
        hist = []
        dep, depce, clean = evaluate()
        hist.append({"step": args.warm, "dep_acc": dep, "dep_ce": depce, "clean": clean})
        t1 = time.time()
        train_steps(mode, args.steps, arm_sampler, arm_data, args.warm + 1, hist,
                    args.eval_every, mode)
        print(f"{mode} arm done in {(time.time()-t1)/60:.1f} min", flush=True)
        arms[mode] = hist

    # ---------------- summary ----------------------------------------------
    print("\n" + "=" * 72)
    print(f"fork at step {args.warm} ({args.warm/391:.1f} epochs of decoupled warm-up)")
    print(f"{'step':>7}{'decoupled dep/CE':>24}{'coupled dep/CE':>24}")
    dd = {h["step"]: h for h in arms["decoupled"]}
    cc = {h["step"]: h for h in arms["coupled"]}
    for s in sorted(set(dd) | set(cc)):
        a = dd.get(s)
        b = cc.get(s)
        astr = f"{a['dep_acc']:.4f} / {a['dep_ce']:.4f}" if a else "-"
        bstr = f"{b['dep_acc']:.4f} / {b['dep_ce']:.4f}" if b else "-"
        print(f"{s:>7}{astr:>24}{bstr:>24}")
    fa, fb = arms["decoupled"][-1], arms["coupled"][-1]
    print(f"\nfinal  decoupled: dep_acc={fa['dep_acc']:.4f}  dep_CE={fa['dep_ce']:.4f}")
    print(f"final  coupled  : dep_acc={fb['dep_acc']:.4f}  dep_CE={fb['dep_ce']:.4f}")
    better = ("COUPLED" if fb["dep_ce"] < fa["dep_ce"] - 1e-3 else
              "DECOUPLED" if fa["dep_ce"] < fb["dep_ce"] - 1e-3 else "TIE")
    print(f"-> on the outer objective (deployed CE), switching late is: {better}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"warm": args.warm, "steps": args.steps, "lr": LR, "batch": BATCH,
                   "norm_match": args.norm_match,
                   "N_V": N_V, "N_M": N_M, "sigma": SIGMA, "rcond": RCOND,
                   "refit": REFIT, "n_eval_ch": N_EVAL_CH, "W": W, "R": R,
                   "names": NAMES, "device": str(dev)},
        "warmup": warm_hist, "arms": arms,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
