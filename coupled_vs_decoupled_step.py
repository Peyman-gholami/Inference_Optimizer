"""
Payoff test: does a COUPLED step beat a DECOUPLED step at reducing the deployed loss
E_c[l]?  Using the A+B (reliable) coupled gradient.

At a given theta: g_dec_k = grad_theta CE (direct);  g_cpl_k = direct + (J^T v)_k
(J from the analytic surrogate H via kkt_adjoint_reduced; v=grad_eta CE and the mixed
term by Stein).  For a grid of lr and g in {decoupled, coupled}: step theta <- theta - lr*g,
re-solve eta*(c) for a fixed set of channels, measure mean CE, restore.  Coupled "helps"
if it lowers deployed loss more.  Repeated at pretrained theta and after a short decoupled
warm-up.
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

NB = 200; N_V, N_M = 48, 32; SIGMA = 0.05; RCOND = 1e-3
W = [1.0, 1.0]; R = [1.0, 1.5]
NAMES = ["cifar10_resnet56", "cifar10_resnet20"]
LRS = [3e-3, 1e-2, 3e-2, 1e-1]


def main():
    torch.manual_seed(0); np.random.seed(0)
    dev = select_device()
    train_loader, test = get_cifar10_loaders()
    xt, yt = load_test_tensors(test)
    x, y = xt[:NB].to(dev), yt[:NB].to(dev)
    train_it = cycle(train_loader)

    cms, a_list = [], []
    for m in NAMES:
        cm, a = make_compressed_resnet(load_pretrained_cifar_resnet(m).to(dev), model_name=m)
        cm.to(dev).eval(); cms.append(cm); a_list.append(a)
    K, d = 2, 2
    A_arr = np.array(a_list)
    ref = K * float(np.exp(np.mean(np.log([R[k] * A_arr[k, i] for k in range(K) for i in range(d)]))))
    profs = [LiveAccuracyProfile(cm, x, y, d, dev) for cm in cms]
    # fixed eval channels (physical, log-uniform) for the deployed-loss measurement
    eval_sampler = ChannelSampler(d, np.array([0.1 * ref] * d), np.array([1.2 * ref] * d), seed=7)
    EVAL_C = [eval_sampler() for _ in range(8)]

    def set_eta(k, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))

    def ce_val(k, e):
        with torch.no_grad():
            set_eta(k, e); return float(F.cross_entropy(cms[k](x), y))

    def flat_params(cm): return torch.cat([p.detach().reshape(-1) for p in cm.parameters()])

    def set_flat(cm, flat):
        i = 0
        for p in cm.parameters():
            n = p.numel(); p.data.copy_(flat[i:i+n].reshape(p.shape)); i += n

    def _flat_grad(scalar, cm):
        ps = list(cm.parameters())
        g = torch.autograd.grad(scalar, ps, allow_unused=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                          for gi, p in zip(g, ps)]).detach().cpu().numpy()

    def gradA_theta(k, e):
        set_eta(k, e); p = torch.softmax(cms[k](x), 1)
        return _flat_grad(W[k] * p[torch.arange(len(y)), y].mean(), cms[k])

    def direct_theta(k, e):
        set_eta(k, e); return _flat_grad(F.cross_entropy(cms[k](x), y), cms[k])

    def controller_state():
        for p in profs: p.refit()
        alpha = np.concatenate([p.alpha for p in profs])
        w_eff = [W[k] * float(profs[k].c0) for k in range(K)]
        prob = SurrogateCoupled(w=w_eff, R=R, a=A_arr, c=np.array([0.5 * ref, 0.5 * ref]), eta_min=0.1)
        eta = prob.solve(alpha)
        return prob, eta, alpha, w_eff

    def deployed_loss():
        for p in profs: p.refit()
        alpha = np.concatenate([p.alpha for p in profs])
        w_eff = [W[k] * float(profs[k].c0) for k in range(K)]
        tot = 0.0; cnt = 0
        for c in EVAL_C:
            prob_c = SurrogateCoupled(w=w_eff, R=R, a=A_arr, c=c, eta_min=0.1)
            ec = prob_c.solve(alpha)
            for k in range(K):
                tot += ce_val(k, ec[k*2:k*2+2]); cnt += 1
        return tot / cnt

    def compute_grads():
        prob, eta, alpha, w_eff = controller_state()
        active, free = prob.active_set(eta)
        H = prob.hess_eta(eta, alpha)
        v = np.zeros(4); g_dec, g_cpl = [], []
        bases = {}
        for k in range(K):
            ek = eta[k*2:k*2+2]
            v[k*2:k*2+2] = stein_grad(lambda e: ce_val(k, e), ek, SIGMA, N_V, seed=k)
            bases[k] = stein_hybrid_basis(lambda e: gradA_theta(k, e), ek, SIGMA, N_M, seed=k)
        a = kkt_adjoint_reduced(H, prob.B, active, free, 4, v, rcond=RCOND)
        for k in range(K):
            direct = direct_theta(k, eta[k*2:k*2+2])
            jtv = -bases[k] @ a[k*2:k*2+2]
            g_dec.append(direct); g_cpl.append(direct + jtv)
        return g_dec, g_cpl

    def step_test(label):
        g_dec, g_cpl = compute_grads()
        F0 = deployed_loss()
        orig = [flat_params(cms[k]).clone() for k in range(K)]
        print(f"\n=== {label}:  deployed loss F0 = {F0:.4f} ===")
        print(f"{'lr':>8}{'dF decoupled':>16}{'dF coupled':>14}{'   winner':>10}")
        for lr in LRS:
            res = {}
            for name, g in [("dec", g_dec), ("cpl", g_cpl)]:
                for k in range(K):
                    set_flat(cms[k], orig[k] - lr * torch.tensor(g[k], dtype=torch.float32, device=dev))
                res[name] = deployed_loss() - F0
                for k in range(K): set_flat(cms[k], orig[k])
            win = "coupled" if res["cpl"] < res["dec"] - 1e-4 else ("decoupled" if res["dec"] < res["cpl"] - 1e-4 else "tie")
            print(f"{lr:>8.0e}{res['dec']:>16.4f}{res['cpl']:>14.4f}{win:>10}")

    step_test("pretrained theta")

    # short decoupled warm-up, then re-test at the moved theta
    print("\n... 50 decoupled compression-aware warm-up steps ...")
    opts = [torch.optim.Adam(cms[k].parameters(), lr=1e-4) for k in range(K)]
    for s in range(50):
        if s % 25 == 0:
            _, eta_w, alpha_w, _ = controller_state()
        for k in range(K):
            xb, yb = next(train_it); xb, yb = xb.to(dev), yb.to(dev)
            set_eta(k, eta_w[k*2:k*2+2]); cms[k].train()
            opts[k].zero_grad(); F.cross_entropy(cms[k](xb), yb).backward(); opts[k].step()
        for k in range(K): cms[k].eval()
    step_test("after 50 decoupled steps")


if __name__ == "__main__":
    main()
