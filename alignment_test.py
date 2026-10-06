"""
alignment_test.py
=================
Does the coupled direction actually point downhill? --- the step-size-free test.

Why this exists
---------------
`coupled_vs_decoupled_step.py` swept ONE shared step size over UNNORMALIZED
directions. Since ||J^T v|| ~ 2.35 ||direct|| at cos ~ 0.14, we have
||g_cpl|| ~ 2.5 ||g_dec||, so at every lr the coupled arm took a ~2.5x LONGER
step and its matched step length was never tested. "Coupled increases the loss"
may therefore have been "coupled overshoots".

The fix is to take step length out of the comparison entirely. We cannot form
the true hypergradient grad F -- that is the whole difficulty -- but we do not
need it. We only need its inner product with each candidate, and that is a
directional derivative of the deployed loss, which is a central difference:

    D(u) = [ F(theta + eps u) - F(theta - eps u) ] / (2 eps)   ,  ||u|| = 1

D(u) = <u, grad F>. We STEP ALONG -u (theta <- theta - s g), so a step of length
s changes F by  dF ~ -s * D(u).  A good descent direction therefore has D(u)
LARGE AND POSITIVE, and D alone ranks the directions for every step size at
once. This is the quantity the original test could not isolate.

F is the DEPLOYED loss: at each perturbed theta the accuracy profiles are refit
and eta*(c) is re-solved per channel, exactly as at deployment. That matters --
the coupling term J^T v exists precisely to anticipate the controller's
reaction, so freezing the allocation would test it against a target that has no
coupling by construction.

Candidates
----------
  decoupled      direct                      (the baseline)
  coupled        direct + J^T v              (A+B reliable coupling)
  coupling_only  J^T v                       (is the coupling alone downhill?)
  random         a random unit direction     (control: D should be ~ 0)

Outputs, per candidate:
  D(u)            directional derivative (step-size-free ranking)
  dF(eps)         one-sided F(theta - eps u) - F(theta), at MATCHED step length
                  -- the corrected version of the original payoff test
  ||g||, cos      direction norms and pairwise cosines, so the size of the
                  original confound is on the record

Usage
-----
  python alignment_test.py                       # pretrained theta
  python alignment_test.py --warmup 50           # also after a decoupled warm-up
  python alignment_test.py --stein_seeds 3       # error bar on the coupled rows
  python alignment_test.py --eps 1e-4,1e-3,1e-2  # linearity check
  python alignment_test.py --profile hard        # shows why the staircase breaks it
"""

from __future__ import annotations

import argparse
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

# Matched to coupled_vs_decoupled_step.py so the two are directly comparable.
NB = 200
N_V, N_M = 48, 32
SIGMA = 0.05
RCOND = 1e-3
W = [1.0, 1.0]
R = [1.0, 1.5]
NAMES = ["cifar10_resnet56", "cifar10_resnet20"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", default="3e-4,1e-3,3e-3,1e-2",
                   help="Step lengths (L2, in parameter space) for the sweep. "
                        "Directions are unit-norm, so eps IS the step length.")
    p.add_argument("--n_eval_channels", type=int, default=8)
    p.add_argument("--stein_seeds", type=int, default=1,
                   help="Repeat with different Stein seeds for an error bar on "
                        "the coupled/coupling_only rows.")
    p.add_argument("--warmup_schedule", default="0",
                   help="Cumulative decoupled-warm-up step counts at which to "
                        "re-measure, e.g. 0,25,50,100,200,400. Tests whether the "
                        "coupling's alignment grows as the direct gradient's "
                        "alignment collapses near the optimum.")
    p.add_argument("--profile", default="soft", choices=["soft", "hard"],
                   help="Profile measurement used inside F. 'soft' (default) = "
                        "mean correct-class softmax, smooth in theta. 'hard' = "
                        "argmax top-1, quantized to 1/200 here, which makes F a "
                        "staircase and the finite differences meaningless.")
    p.add_argument("--out", default="outputs/alignment/alignment_test.json")
    args = p.parse_args()

    eps_list = [float(e) for e in args.eps.split(",") if e.strip()]

    torch.manual_seed(0)
    np.random.seed(0)
    dev = select_device()
    train_loader, test = get_cifar10_loaders()
    xt, yt = load_test_tensors(test)
    x, y = xt[:NB].to(dev), yt[:NB].to(dev)
    train_it = cycle(train_loader)

    cms, a_list = [], []
    for m in NAMES:
        cm, a = make_compressed_resnet(load_pretrained_cifar_resnet(m).to(dev), model_name=m)
        cm.to(dev).eval()
        cms.append(cm)
        a_list.append(a)
    K, d = 2, 2
    A_arr = np.array(a_list)
    ref = K * float(np.exp(np.mean(np.log(
        [R[k] * A_arr[k, i] for k in range(K) for i in range(d)]))))
    profs = [LiveAccuracyProfile(cm, x, y, d, dev, mode=args.profile) for cm in cms]

    eval_sampler = ChannelSampler(d, np.array([0.1 * ref] * d),
                                  np.array([1.2 * ref] * d), seed=7)
    EVAL_C = [eval_sampler() for _ in range(args.n_eval_channels)]

    # ---------------- harness (same semantics as the payoff test) -------------
    def set_eta(k, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))

    def ce_val(k, e):
        with torch.no_grad():
            set_eta(k, e)
            return float(F.cross_entropy(cms[k](x), y))

    def flat_params(cm):
        return torch.cat([p.detach().reshape(-1) for p in cm.parameters()])

    def set_flat(cm, flat):
        i = 0
        for prm in cm.parameters():
            n = prm.numel()
            prm.data.copy_(flat[i:i + n].reshape(prm.shape))
            i += n

    def _flat_grad(scalar, cm):
        ps = list(cm.parameters())
        g = torch.autograd.grad(scalar, ps, allow_unused=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(prm)).reshape(-1)
                          for gi, prm in zip(g, ps)]).detach().cpu().numpy()

    def gradA_theta(k, e):
        set_eta(k, e)
        prob_p = torch.softmax(cms[k](x), 1)
        return _flat_grad(W[k] * prob_p[torch.arange(len(y)), y].mean(), cms[k])

    def direct_theta(k, e):
        set_eta(k, e)
        return _flat_grad(F.cross_entropy(cms[k](x), y), cms[k])

    def controller_state():
        for prf in profs:
            prf.refit()
        alpha = np.concatenate([prf.alpha for prf in profs])
        w_eff = [W[k] * float(profs[k].c0) for k in range(K)]
        prob = SurrogateCoupled(w=w_eff, R=R, a=A_arr,
                               c=np.array([0.5 * ref, 0.5 * ref]), eta_min=0.1)
        return prob, prob.solve(alpha), alpha, w_eff

    def deployed_loss():
        """F(theta): refit the profiles, re-solve eta*(c) per channel, mean CE.

        Deterministic given theta -- fixed eval batch and fixed channel set --
        so the finite differences carry no Monte-Carlo noise of their own.
        """
        for prf in profs:
            prf.refit()
        alpha = np.concatenate([prf.alpha for prf in profs])
        w_eff = [W[k] * float(profs[k].c0) for k in range(K)]
        tot = cnt = 0.0
        for c in EVAL_C:
            ec = SurrogateCoupled(w=w_eff, R=R, a=A_arr, c=c, eta_min=0.1).solve(alpha)
            for k in range(K):
                tot += ce_val(k, ec[k * 2:k * 2 + 2])
                cnt += 1
        return tot / cnt

    def compute_grads(seed_off=0):
        prob, eta, alpha, _ = controller_state()
        active, free = prob.active_set(eta)
        H = prob.hess_eta(eta, alpha)
        v = np.zeros(4)
        bases = {}
        for k in range(K):
            ek = eta[k * 2:k * 2 + 2]
            v[k * 2:k * 2 + 2] = stein_grad(lambda e: ce_val(k, e), ek, SIGMA,
                                            N_V, seed=k + 100 * seed_off)
            bases[k] = stein_hybrid_basis(lambda e: gradA_theta(k, e), ek, SIGMA,
                                          N_M, seed=k + 100 * seed_off)
        adj = kkt_adjoint_reduced(H, prob.B, active, free, 4, v, rcond=RCOND)
        direct, coupling = [], []
        for k in range(K):
            direct.append(direct_theta(k, eta[k * 2:k * 2 + 2]))
            coupling.append(-bases[k] @ adj[k * 2:k * 2 + 2])
        return direct, coupling

    # ---------------- direction algebra --------------------------------------
    def joint_norm(g):
        return float(np.sqrt(sum(float(np.dot(gk, gk)) for gk in g)))

    def normalize(g):
        n = joint_norm(g)
        return [gk / n for gk in g], n

    def joint_cos(g1, g2):
        num = sum(float(np.dot(a_, b_)) for a_, b_ in zip(g1, g2))
        return num / (joint_norm(g1) * joint_norm(g2) + 1e-30)

    # ---------------- the measurement ---------------------------------------
    def measure(label, seed_off=0):
        t0 = time.time()
        direct, coupling = compute_grads(seed_off)
        g_dec = direct
        g_cpl = [dk + ck for dk, ck in zip(direct, coupling)]

        rng = np.random.default_rng(1234 + seed_off)
        g_rnd = [rng.standard_normal(gk.size).astype(gk.dtype) for gk in g_dec]

        cands = {"decoupled": g_dec, "coupled": g_cpl,
                 "coupling_only": coupling, "random": g_rnd}

        norms = {k: joint_norm(v_) for k, v_ in cands.items()}
        F0 = deployed_loss()
        orig = [flat_params(cms[k]).clone() for k in range(K)]

        def move(u, s):
            for k in range(K):
                set_flat(cms[k], orig[k] - s * torch.tensor(
                    u[k], dtype=torch.float32, device=dev))

        def restore():
            for k in range(K):
                set_flat(cms[k], orig[k])

        out = {}
        for name, g in cands.items():
            u, _ = normalize(g)
            rec = {"norm": norms[name], "D": {}, "dF": {}}
            for eps in eps_list:
                move(u, +eps); f_minus = deployed_loss()   # theta - eps u
                move(u, -eps); f_plus = deployed_loss()    # theta + eps u
                restore()
                rec["D"][f"{eps:g}"] = (f_plus - f_minus) / (2 * eps)
                rec["dF"][f"{eps:g}"] = f_minus - F0
            out[name] = rec

        cos = {
            "direct_vs_coupling": joint_cos(direct, coupling),
            "decoupled_vs_coupled": joint_cos(g_dec, g_cpl),
        }
        ratio = norms["coupling_only"] / (norms["decoupled"] + 1e-30)

        print(f"\n=== {label}  (seed_off={seed_off})   F0 = {F0:.4f} "
              f"  [{time.time()-t0:.0f}s] ===")
        print(f"  ||direct||={norms['decoupled']:.4g}   "
              f"||J^T v||={norms['coupling_only']:.4g}   "
              f"ratio={ratio:.2f}   cos(direct,coupling)={cos['direct_vs_coupling']:+.3f}")
        print(f"  -> at a SHARED lr the coupled step was "
              f"{norms['coupled']/norms['decoupled']:.2f}x longer than the decoupled one")
        hdr = "".join(f"{'D@'+f'{e:g}':>13}" for e in eps_list)
        print(f"\n  directional derivative D = <u, grad F>  "
              f"(we step along -u, so LARGER POSITIVE = steeper descent)")
        print(f"  {'candidate':<15}{hdr}")
        for name in cands:
            row = "".join(f"{out[name]['D'][f'{e:g}']:>13.4f}" for e in eps_list)
            print(f"  {name:<15}{row}")

        # Non-smoothness diagnostic. D is linear in the direction, so
        # D_coupled * ||d+c|| should equal D_direct*||d|| + D_coupling*||c||.
        # Any gap means F is not locally linear at this eps -- active-set
        # switches in the water-fill and the profile refit make F piecewise.
        print(f"\n  additivity check (gap => F not locally linear at that eps)")
        print(f"  {'eps':<15}{'measured':>13}{'predicted':>13}{'gap':>13}")
        for e in eps_list:
            kk = f"{e:g}"
            pred = (out['decoupled']['D'][kk] * norms['decoupled']
                    + out['coupling_only']['D'][kk] * norms['coupling_only']
                    ) / norms['coupled']
            meas = out['coupled']['D'][kk]
            print(f"  {kk:<15}{meas:>13.4f}{pred:>13.4f}{meas-pred:>13.4f}")
        hdr2 = "".join(f"{'dF@'+f'{e:g}':>13}" for e in eps_list)
        print(f"\n  matched step length (one-sided dF; the corrected payoff test) "
              f"-- more negative = better")
        print(f"  {'candidate':<15}{hdr2}")
        for name in cands:
            row = "".join(f"{out[name]['dF'][f'{e:g}']:>13.4f}" for e in eps_list)
            print(f"  {name:<15}{row}")

        return {"label": label, "seed_off": seed_off, "F0": F0,
                "norms": norms, "cos": cos, "coupling_ratio": ratio,
                "candidates": out}

    schedule = sorted({int(t) for t in args.warmup_schedule.split(",") if t.strip()})
    opts = [torch.optim.Adam(cms[k].parameters(), lr=1e-4) for k in range(K)]

    results = []
    done_steps = 0
    for target in schedule:
        if target > done_steps:
            n = target - done_steps
            print(f"\n... {n} decoupled compression-aware steps "
                  f"({done_steps} -> {target}) ...", flush=True)
            eta_w = controller_state()[1]
            for i in range(n):
                if i % 25 == 0:
                    eta_w = controller_state()[1]
                for k in range(K):
                    xb, yb = next(train_it)
                    xb, yb = xb.to(dev), yb.to(dev)
                    set_eta(k, eta_w[k * 2:k * 2 + 2])
                    cms[k].train()
                    opts[k].zero_grad()
                    F.cross_entropy(cms[k](xb), yb).backward()
                    opts[k].step()
                for k in range(K):
                    cms[k].eval()
            done_steps = target
        label = "pretrained theta" if target == 0 else f"after {target} decoupled steps"
        for sd in range(args.stein_seeds):
            results.append(measure(label, sd))
            results[-1]["steps"] = target

    # ---------------- verdict -----------------------------------------------
    # Read the verdict in the RELIABLE window, not at the smallest eps. F has a
    # noise floor (SLSQP tolerances, active-set switches), and a central
    # difference divides by eps, so the SMALLEST eps is the NOISIEST. The
    # additivity gap is the diagnostic: pick the eps whose gap is smallest
    # relative to the measured D.
    def reliable_eps(rs):
        best, best_rel = eps_list[-1], float("inf")
        for e in eps_list:
            kk = f"{e:g}"
            rels = []
            for r in rs:
                cand = r["candidates"]
                pred = (cand["decoupled"]["D"][kk] * r["norms"]["decoupled"]
                        + cand["coupling_only"]["D"][kk] * r["norms"]["coupling_only"]
                        ) / r["norms"]["coupled"]
                meas = cand["coupled"]["D"][kk]
                rels.append(abs(meas - pred) / (abs(meas) + 1e-12))
            rel = float(np.mean(rels))
            if rel < best_rel:
                best, best_rel = e, rel
        return best, best_rel

    print("\n" + "=" * 72)
    for label in dict.fromkeys(r["label"] for r in results):
        rs = [r for r in results if r["label"] == label]
        eps_r, rel = reliable_eps(rs)
        eps0 = f"{eps_r:g}"
        dec = np.array([r["candidates"]["decoupled"]["D"][eps0] for r in rs])
        cpl = np.array([r["candidates"]["coupled"]["D"][eps0] for r in rs])
        sd = f" +/- {cpl.std():.4f}" if len(rs) > 1 else ""
        verdict = ("COUPLED is better aligned" if cpl.mean() > dec.mean() + 1e-4
                   else "DECOUPLED is better aligned" if dec.mean() > cpl.mean() + 1e-4
                   else "TIE")
        cpo = np.array([r["candidates"]["coupling_only"]["D"][eps0] for r in rs])
        rnd = np.array([r["candidates"]["random"]["D"][eps0] for r in rs])
        print(f"{label}  [reliable eps={eps0}, additivity gap {rel*100:.1f}%]")
        print(f"    D_decoupled     = {dec.mean():>8.3f}")
        print(f"    D_coupled       = {cpl.mean():>8.3f}{sd}   ->  {verdict}")
        print(f"    D_coupling_only = {cpo.mean():>8.3f}   "
              f"(vs random control {rnd.mean():+.3f})")
        print(f"    -> per unit length the coupling carries "
              f"{abs(cpo.mean())/(abs(dec.mean())+1e-12)*100:.0f}% of the "
              f"direct direction's descent, while taking up "
              f"{rs[0]['norms']['coupling_only']/rs[0]['norms']['coupled']*100:.0f}% "
              f"of the coupled direction's norm")
    print("(D = <u, grad F>; we step along -u, so LARGER POSITIVE = steeper "
          "descent per unit step; step-size-free)")

    if len(schedule) > 1:
        print("\n" + "=" * 72)
        print("TRAJECTORY  (D at the per-point reliable eps; mean over Stein seeds)")
        print(f"  {'steps':>7}{'F0':>9}{'D_direct':>11}{'D_coupled':>11}"
              f"{'D_coupling':>12}{'D_random':>10}{'coup/direct':>13}")
        for t in schedule:
            rs = [r for r in results if r["steps"] == t]
            if not rs:
                continue
            e, _ = reliable_eps(rs)
            kk = f"{e:g}"
            g = lambda c: float(np.mean([r["candidates"][c]["D"][kk] for r in rs]))
            dd, dc, dp, dr = g("decoupled"), g("coupled"), g("coupling_only"), g("random")
            print(f"  {t:>7}{rs[0]['F0']:>9.3f}{dd:>11.3f}{dc:>11.3f}"
                  f"{dp:>12.3f}{dr:>10.3f}{dp/(abs(dd)+1e-12):>13.2f}")
        print("  coup/direct rising while D_direct falls => the crossover is real")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"eps": eps_list, "n_eval_channels": args.n_eval_channels,
                   "stein_seeds": args.stein_seeds,
                   "warmup_schedule": args.warmup_schedule,
                   "N_V": N_V, "N_M": N_M, "sigma": SIGMA, "rcond": RCOND,
                   "W": W, "R": R, "names": NAMES, "nb": NB,
                   "profile": args.profile, "device": str(dev)},
        "results": results,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
