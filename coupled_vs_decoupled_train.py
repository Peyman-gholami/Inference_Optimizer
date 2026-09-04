"""
FULL coupled-vs-decoupled TRAINING comparison (r56+r20, shared bandwidth).

Both arms start from the same pretrained init and train with Adam under the same
controller (surrogate water-fill -> eta*). Per step, per task, on a fresh minibatch:
  decoupled:  g = direct = grad_theta CE(x; eta*_k)
  coupled:    g = direct + (J^T v)_k   [A+B: analytic surrogate H, truncated null-space
              adjoint; v = grad_eta CE via Stein; M^T a via the Stein-hybrid]
The surrogate profile A_k is re-fit every REFIT steps. We track deployed accuracy
E_c[acc] and clean (eta=1) accuracy, and report both trajectories.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python coupled_vs_decoupled_train.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from itertools import cycle

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.compression_aware import ChannelSampler
from src.pretraining.coupled import (
    SurrogateCoupled, kkt_adjoint_reduced, stein_grad, stein_hybrid_basis,
)

STEPS, BATCH, LR = 120, 128, 1e-4
N_V, N_M, SIGMA, RCOND = 32, 20, 0.05, 1e-3
REFIT, EVAL_EVERY, N_EVAL_CH = 25, 20, 12
W = [1.0, 1.0]; R = [1.0, 1.5]
NAMES = ["cifar10_resnet56", "cifar10_resnet20"]


def main():
    dev = select_device()
    train_loader, test = get_cifar10_loaders()
    xt, yt = load_test_tensors(test)
    x_meas, y_meas = xt[:1000].to(dev), yt[:1000].to(dev)         # profile fit
    x_ev, y_ev = xt[1000:3000].to(dev), yt[1000:3000].to(dev)     # deployed/clean eval
    A_arr = np.array([[16384.0, 8192.0], [16384.0, 8192.0]])
    ref = float(2 * np.exp(np.mean(np.log([R[k] * A_arr[k, i] for k in range(2) for i in range(2)]))))
    eval_ch = [ChannelSampler(2, np.array([0.1*ref]*2), np.array([1.2*ref]*2), seed=7)() for _ in range(N_EVAL_CH)]

    def set_grad(cm, flat):
        i = 0
        for p in cm.parameters():
            n = p.numel(); p.grad = flat[i:i+n].reshape(p.shape).clone(); i += n

    def run_arm(mode):
        torch.manual_seed(0); np.random.seed(0)
        cms = [make_compressed_resnet(load_pretrained_cifar_resnet(m).to(dev), model_name=m)[0].to(dev)
               for m in NAMES]
        for cm in cms: cm.eval()
        profs = [LiveAccuracyProfile(cms[k], x_meas, y_meas, 2, dev) for k in range(2)]
        opts = [torch.optim.Adam(cms[k].parameters(), lr=LR) for k in range(2)]
        sampler = ChannelSampler(2, np.array([0.1*ref]*2), np.array([1.2*ref]*2), seed=1, kind="loguniform")
        data = cycle(train_loader)

        def ce_on(k, xb, yb, e):
            with torch.no_grad():
                cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
                return float(F.cross_entropy(cms[k](xb), yb))

        def flat_grad(scalar, cm):
            ps = list(cm.parameters())
            g = torch.autograd.grad(scalar, ps, allow_unused=True)
            return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                              for gi, p in zip(g, ps)]).detach().cpu().numpy()

        def gradA_on(k, xb, yb, e):
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            p = torch.softmax(cms[k](xb), 1)
            return flat_grad(W[k] * p[torch.arange(len(yb)), yb].mean(), cms[k])

        def direct_on(k, xb, yb, e):
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            return flat_grad(F.cross_entropy(cms[k](xb), yb), cms[k])

        @torch.no_grad()
        def acc_at(k, e):
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            return (cms[k](x_ev).argmax(1) == y_ev).float().mean().item()

        def evaluate():
            for p in profs: p.refit()
            alpha = np.concatenate([p.alpha for p in profs]); w_eff = [W[k]*float(profs[k].c0) for k in range(2)]
            dep = []
            for c in eval_ch:
                ec = SurrogateCoupled(w=w_eff, R=R, a=A_arr, c=c, eta_min=0.1).solve(alpha)
                dep.append(np.mean([acc_at(k, ec[k*2:k*2+2]) for k in range(2)]))
            clean = np.mean([acc_at(k, [1.0, 1.0]) for k in range(2)])
            return float(np.mean(dep)), float(clean)

        hist = []
        for step in range(STEPS):
            if step % REFIT == 0:
                for p in profs: p.refit()
            alpha = np.concatenate([p.alpha for p in profs]); w_eff = [W[k]*float(profs[k].c0) for k in range(2)]
            prob = SurrogateCoupled(w=w_eff, R=R, a=A_arr, c=sampler(), eta_min=0.1)
            eta = prob.solve(alpha)
            xb, yb = next(data); xb, yb = xb.to(dev), yb.to(dev)
            adj = None
            if mode == "coupled":
                active, free = prob.active_set(eta); H = prob.hess_eta(eta, alpha)
                v = np.zeros(4)
                bases = {}
                for k in range(2):
                    ek = eta[k*2:k*2+2]
                    v[k*2:k*2+2] = stein_grad(lambda e: ce_on(k, xb, yb, e), ek, SIGMA, N_V, seed=step)
                    bases[k] = stein_hybrid_basis(lambda e: gradA_on(k, xb, yb, e), ek, SIGMA, N_M, seed=step)
                adj = kkt_adjoint_reduced(H, prob.B, active, free, 4, v, rcond=RCOND)
            for k in range(2):
                ek = eta[k*2:k*2+2]
                g = direct_on(k, xb, yb, ek)
                if mode == "coupled":
                    g = g - bases[k] @ adj[k*2:k*2+2]
                set_grad(cms[k], torch.tensor(g, dtype=torch.float32, device=dev))
                opts[k].step(); opts[k].zero_grad()
                cms[k].eval()
            if step % EVAL_EVERY == 0 or step == STEPS - 1:
                dep, clean = evaluate()
                hist.append((step, dep, clean))
                print(f"[{mode}] step {step:4d}  deployed_acc={dep:.4f}  clean={clean:.4f}", flush=True)
        return hist

    print("========== DECOUPLED ==========")
    h_dec = run_arm("decoupled")
    print("========== COUPLED (A+B) ==========")
    h_cpl = run_arm("coupled")

    print("\n================ SUMMARY ================")
    print(f"{'step':>6}{'  decoupled dep/clean':>24}{'  coupled dep/clean':>22}")
    dd = {s: (d, c) for s, d, c in h_dec}; cc = {s: (d, c) for s, d, c in h_cpl}
    for s in sorted(set(dd) | set(cc)):
        d = dd.get(s); c = cc.get(s)
        ds = f"{d[0]:.3f}/{d[1]:.3f}" if d else "-"
        cs = f"{c[0]:.3f}/{c[1]:.3f}" if c else "-"
        print(f"{s:>6}{ds:>24}{cs:>22}")


if __name__ == "__main__":
    main()
