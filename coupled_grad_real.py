"""
One REAL-model coupled hypergradient on the r56+r20 multi-task setup, to confirm the
Stein-hybrid is feasible (Stein over the LOW-DIM eta only; autograd for theta once per
sample) and to measure the coupling magnitude.

Per task k (eta_k in R^2):
  H_k = w_k * Stein-Hessian_eta(soft-acc A_k)      [forward-only, eta-Stein]
  v_k = Stein-grad_eta(CE loss)                    [forward-only, eta-Stein]
Joint adjoint a solves the KKT with block-diag H + shared-link G (active set from the
controller). Then coupling grad for theta_k = -(M_k^T a_k) via the Stein-hybrid
  M_k^T a_k = (1/sigma) E_{z in R^2}[(a_k^T z) grad_theta_k (w_k A_k)(theta_k, eta_k*+sigma z)]
(z sampled in eta-space; grad_theta_k by autograd). Compare to direct = grad_theta CE.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python coupled_grad_real.py
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.coupled import (
    SurrogateCoupled, kkt_adjoint, stein_hessian, stein_grad, stein_hybrid_Mta,
)

NB = 200          # eval batch
N_H, N_V, N_M = 150, 64, 40     # eta-Stein sample counts (low-dim eta -> small)
SIGMA = 0.05
W = [1.0, 1.0]; R = [1.0, 1.5]
NAMES = ["cifar10_resnet56", "cifar10_resnet20"]


def main():
    torch.manual_seed(0); np.random.seed(0)
    dev = select_device()
    _, test = get_cifar10_loaders()
    xt, yt = load_test_tensors(test)
    x, y = xt[:NB].to(dev), yt[:NB].to(dev)

    cms, a_list = [], []
    for m in NAMES:
        cm, a = make_compressed_resnet(load_pretrained_cifar_resnet(m).to(dev), model_name=m)
        cm.to(dev).eval(); cms.append(cm); a_list.append(a)
    K, d = 2, 2

    def A_val(k, e):                     # soft-accuracy value (forward only)
        with torch.no_grad():
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            p = torch.softmax(cms[k](x), 1)
            return float(p[torch.arange(len(y)), y].mean())

    def ce_val(k, e):
        with torch.no_grad():
            cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
            return float(F.cross_entropy(cms[k](x), y))

    def _flat_grad(scalar, cm):
        g = torch.autograd.grad(scalar, list(cm.parameters()))
        return torch.cat([gi.reshape(-1) for gi in g]).detach().cpu().numpy()

    def gradA_theta(k, e):               # grad_theta ( w_k * soft-acc )
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        p = torch.softmax(cms[k](x), 1)
        Ak = W[k] * p[torch.arange(len(y)), y].mean()
        return _flat_grad(Ak, cms[k])

    def direct_theta(k, e):              # grad_theta CE loss
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        return _flat_grad(F.cross_entropy(cms[k](x), y), cms[k])

    # controller: fit surrogate profiles -> alpha, water-fill eta* at a contended channel
    profs = [LiveAccuracyProfile(cm, x, y, d, dev) for cm in cms]
    alpha = np.concatenate([p.alpha for p in profs])
    A_arr = np.array(a_list)
    demands = np.array([R[k] * A_arr[k, i] for k in range(K) for i in range(d)])
    ref = K * float(np.exp(np.mean(np.log(demands))))
    c = np.array([0.5 * ref, 0.5 * ref])          # mid channel -> contention
    prob = SurrogateCoupled(w=W, R=R, a=A_arr, c=c, eta_min=0.1)
    eta = prob.solve(alpha)
    active_links, free = prob.active_set(eta)
    print(f"eta*={np.round(eta,3)}  active_links={active_links}  free={list(free)}")

    t0 = time.time()
    H = np.zeros((4, 4)); v = np.zeros(4)
    for k in range(K):
        ek = eta[k * 2:k * 2 + 2]
        H[k*2:k*2+2, k*2:k*2+2] = W[k] * stein_hessian(lambda e: A_val(k, e), ek, SIGMA, N_H)
        v[k*2:k*2+2] = stein_grad(lambda e: ce_val(k, e), ek, SIGMA, N_V)
    a_adj = kkt_adjoint(H, prob.B, active_links, free, 4, v)

    print(f"\n{'task':<22}{'||direct||':>12}{'||coupling||':>14}{'ratio':>9}{'cos':>8}")
    for k in range(K):
        ek = eta[k*2:k*2+2]; ak = a_adj[k*2:k*2+2]
        coupling = -stein_hybrid_Mta(lambda e: gradA_theta(k, e), ek, ak, SIGMA, N_M)
        direct = direct_theta(k, ek)
        cos = float(coupling @ direct / (np.linalg.norm(coupling) * np.linalg.norm(direct) + 1e-12))
        print(f"{NAMES[k]:<22}{np.linalg.norm(direct):>12.4f}{np.linalg.norm(coupling):>14.4f}"
              f"{np.linalg.norm(coupling)/np.linalg.norm(direct):>9.3f}{cos:>8.3f}")
    print(f"\none coupled gradient (both tasks): {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
