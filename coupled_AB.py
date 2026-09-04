"""
A+B fix for the coupled adjoint: (A) use the analytic, well-conditioned surrogate Hessian
H = grad^2_eta (sum_k w_k c0_k prod_i(1-alpha(1-eta)^2))  [deterministic, no Stein noise],
and (B) invert it with a TRUNCATED pseudo-inverse in the active-constraint null space
(kkt_adjoint_reduced). The real-model mixed term (Stein-hybrid basis) and v=grad_eta CE
(Stein) are unchanged. Re-runs the robustness diagnostics to compare vs the Stein-H baseline
(seed-to-seed cos: r20 0.03 / r56 0.79; damping swing ~30x).
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
    SurrogateCoupled, kkt_adjoint, kkt_adjoint_reduced, stein_grad, stein_hybrid_basis,
)

NB = 200; N_V, N_M = 48, 32; SIGMA = 0.05
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

    # controller + ANALYTIC surrogate Hessian (A): inner obj = sum_k w_k c0_k * surrogate
    profs = [LiveAccuracyProfile(cm, x, y, d, dev) for cm in cms]
    alpha = np.concatenate([p.alpha for p in profs])
    c0 = np.array([p.c0 for p in profs])
    w_eff = [W[k] * float(c0[k]) for k in range(K)]
    A_arr = np.array(a_list)
    demands = np.array([R[k] * A_arr[k, i] for k in range(K) for i in range(d)])
    ref = K * float(np.exp(np.mean(np.log(demands))))
    prob = SurrogateCoupled(w=w_eff, R=R, a=A_arr, c=np.array([0.5 * ref, 0.5 * ref]), eta_min=0.1)
    eta = prob.solve(alpha)
    active_links, free = prob.active_set(eta)
    H = prob.hess_eta(eta, alpha)                       # analytic, deterministic
    print(f"eta*={np.round(eta,3)}  active={active_links}  free={list(free)}")
    for k in range(K):
        print(f"  {NAMES[k]}: surrogate H_kk eig = {np.round(np.linalg.eigvalsh(H[k*2:k*2+2, k*2:k*2+2]),4)}  (c0={c0[k]:.3f})")

    # sanity: reduced adjoint (no truncation) matches the direct KKT solve on this H
    v0 = np.array([0.3, -0.2, 0.5, -0.1])
    a_red = kkt_adjoint_reduced(H, prob.B, active_links, free, 4, v0, rcond=1e-12)
    a_old = kkt_adjoint(H, prob.B, active_links, free, 4, v0)
    print(f"  [sanity] ||reduced - direct adjoint|| = {np.linalg.norm(a_red-a_old):.2e}")

    direct = [direct_theta(k, eta[k*2:k*2+2]) for k in range(K)]
    couplings = {k: [] for k in range(K)}
    seed0 = None
    for s in SEEDS:
        v = np.zeros(4); bases = {}
        for k in range(K):
            ek = eta[k*2:k*2+2]
            v[k*2:k*2+2] = stein_grad(lambda e: ce_val(k, e), ek, SIGMA, N_V, seed=20*s+k)
            bases[k] = stein_hybrid_basis(lambda e: gradA_theta(k, e), ek, SIGMA, N_M, seed=30*s+k)
        a = kkt_adjoint_reduced(H, prob.B, active_links, free, 4, v, rcond=1e-3)
        for k in range(K):
            couplings[k].append(-bases[k] @ a[k*2:k*2+2])
        if s == 0: seed0 = (v.copy(), {k: bases[k] for k in range(K)})
        print(f"  seed {s} done")

    print("\n--- A+B: seed-to-seed robustness (surrogate H + truncated inverse) ---")
    for k in range(K):
        Cs = np.array(couplings[k]); norms = np.linalg.norm(Cs, axis=1)
        U = Cs / (norms[:, None] + 1e-12); M = len(SEEDS)
        cs = [U[i] @ U[j] for i in range(M) for j in range(i+1, M)]
        cos_dir = float((Cs.mean(0) @ direct[k]) / (np.linalg.norm(Cs.mean(0)) * np.linalg.norm(direct[k]) + 1e-12))
        ratio = norms.mean() / np.linalg.norm(direct[k])
        print(f"{NAMES[k]}: ||coupling||={norms.mean():.3f}±{norms.std():.3f}  ratio={ratio:.3f}  "
              f"mean pairwise cos(seeds)={np.mean(cs):+.3f}  cos(mean,direct)={cos_dir:+.3f}")

    print("\n--- rcond sweep (seed 0) ---")
    v, bases = seed0
    print(f"{'rcond':>10}{'  r56 ||c|| (cos)':>22}{'  r20 ||c|| (cos)':>22}")
    for rc in [1e-1, 1e-2, 1e-3, 1e-4]:
        a = kkt_adjoint_reduced(H, prob.B, active_links, free, 4, v, rcond=rc)
        row = f"{rc:>10.0e}"
        for k in range(K):
            c = -bases[k] @ a[k*2:k*2+2]
            cos = float(c @ direct[k] / (np.linalg.norm(c) * np.linalg.norm(direct[k]) + 1e-12))
            row += f"{np.linalg.norm(c):>13.3f} ({cos:+.2f})"
        print(row)


if __name__ == "__main__":
    main()
