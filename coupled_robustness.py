"""
Robustness of the real-model coupling: is the large, near-orthogonal coupling a real
signal or H^-1 / Stein noise? Diagnostics (r56+r20, fixed theta, one contended channel):
  (1) seed-to-seed consistency: recompute the coupling with M independent Stein seeds;
      report mean pairwise cosine of the coupling vectors (near 1 = robust, near 0 = noise).
  (2) H eigenvalues (per task) -> conditioning of the adjoint a = H^-1 v.
  (3) damping sweep: a(lambda) with H+lambda*I; ||coupling|| & cos(direct,coupling) vs lambda
      (free, since M^T a = basis @ a is linear in a).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.coupled import (
    SurrogateCoupled, kkt_adjoint, stein_hessian, stein_grad, stein_hybrid_basis,
)

NB = 200
N_H, N_V, N_M = 120, 48, 32
SIGMA = 0.05
SEEDS = [0, 1, 2, 3]
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

    def A_val(k, e):
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

    def gradA_theta(k, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        p = torch.softmax(cms[k](x), 1)
        return _flat_grad(W[k] * p[torch.arange(len(y)), y].mean(), cms[k])

    def direct_theta(k, e):
        cms[k].set_eta(torch.tensor(e, dtype=torch.float32, device=dev))
        return _flat_grad(F.cross_entropy(cms[k](x), y), cms[k])

    profs = [LiveAccuracyProfile(cm, x, y, d, dev) for cm in cms]
    alpha = np.concatenate([p.alpha for p in profs])
    A_arr = np.array(a_list)
    demands = np.array([R[k] * A_arr[k, i] for k in range(K) for i in range(d)])
    ref = K * float(np.exp(np.mean(np.log(demands))))
    prob = SurrogateCoupled(w=W, R=R, a=A_arr, c=np.array([0.5 * ref, 0.5 * ref]), eta_min=0.1)
    eta = prob.solve(alpha)
    active_links, free = prob.active_set(eta)
    print(f"eta*={np.round(eta,3)}  active={active_links}  free={list(free)}")

    direct = [direct_theta(k, eta[k*2:k*2+2]) for k in range(K)]

    couplings = {k: [] for k in range(K)}      # per-task list of coupling vectors over seeds
    Hkeep = {k: None for k in range(K)}
    for s in SEEDS:
        Hfull = np.zeros((4, 4)); v = np.zeros(4); bases = {}
        for k in range(K):
            ek = eta[k*2:k*2+2]
            Hk = W[k] * stein_hessian(lambda e: A_val(k, e), ek, SIGMA, N_H, seed=10*s+k)
            Hfull[k*2:k*2+2, k*2:k*2+2] = Hk
            if Hkeep[k] is None: Hkeep[k] = Hk
            v[k*2:k*2+2] = stein_grad(lambda e: ce_val(k, e), ek, SIGMA, N_V, seed=20*s+k)
            bases[k] = stein_hybrid_basis(lambda e: gradA_theta(k, e), ek, SIGMA, N_M, seed=30*s+k)
        a_adj = kkt_adjoint(Hfull, prob.B, active_links, free, 4, v)
        for k in range(K):
            couplings[k].append(-bases[k] @ a_adj[k*2:k*2+2])
        if s == 0:
            seed0 = (Hfull.copy(), v.copy(), {k: bases[k] for k in range(K)})
        print(f"  seed {s} done")

    print("\n--- seed-to-seed robustness (M={} seeds) ---".format(len(SEEDS)))
    for k in range(K):
        Cs = np.array(couplings[k])                       # (M, p)
        norms = np.linalg.norm(Cs, axis=1)
        # mean pairwise cosine
        U = Cs / (norms[:, None] + 1e-12)
        M = len(SEEDS); cs = [U[i] @ U[j] for i in range(M) for j in range(i+1, M)]
        cos_dir = float((Cs.mean(0) @ direct[k]) / (np.linalg.norm(Cs.mean(0)) * np.linalg.norm(direct[k]) + 1e-12))
        eig = np.linalg.eigvalsh(Hkeep[k])
        print(f"{NAMES[k]}: ||coupling||={norms.mean():.3f}±{norms.std():.3f}  "
              f"mean pairwise cos(seeds)={np.mean(cs):+.3f}  cos(mean,direct)={cos_dir:+.3f}  "
              f"H eig={np.round(eig,4)}")

    print("\n--- damping sweep (seed 0):  a=(H+lam*I)^-1-adjoint ---")
    Hfull, v, bases = seed0
    scale = np.mean(np.abs(np.linalg.eigvalsh(Hfull))) + 1e-9
    print(f"{'lam/scale':>10}{'  ResNet-56 ||c|| (cos)':>26}{'  ResNet-20 ||c|| (cos)':>26}")
    for lr in [0.0, 0.03, 0.1, 0.3, 1.0]:
        a_adj = kkt_adjoint(Hfull + lr * scale * np.eye(4), prob.B, active_links, free, 4, v)
        row = f"{lr:>10.2f}"
        for k in range(K):
            c = -bases[k] @ a_adj[k*2:k*2+2]
            cos = float(c @ direct[k] / (np.linalg.norm(c) * np.linalg.norm(direct[k]) + 1e-12))
            row += f"{np.linalg.norm(c):>16.3f} ({cos:+.2f})"
        print(row)


if __name__ == "__main__":
    main()
