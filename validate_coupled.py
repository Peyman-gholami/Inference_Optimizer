"""
Validate the coupled-stage KKT-adjoint sensitivity against finite differences, across
the regimes that matter: a saturated shared link (the coupling), a floored task (pinned
variable -> J row = 0), and an all-clean case (J ~ 0). theta := alpha here.
"""
import numpy as np
from src.pretraining.coupled import SurrogateCoupled


def fd_jacobian(prob, alpha, eps=1e-4):
    n = prob.n
    J = np.zeros((n, n))
    for q in range(n):
        ap, am = alpha.copy(), alpha.copy()
        ap[q] += eps; am[q] -= eps
        J[:, q] = (prob.solve(ap) - prob.solve(am)) / (2 * eps)
    return J


def run(name, w, R, a, c, alpha):
    prob = SurrogateCoupled(w=w, R=R, a=a, c=c, eta_min=0.1)
    alpha = np.asarray(alpha, float).reshape(-1)
    eta = prob.solve(alpha)
    links, free = prob.active_set(eta)
    Jimp = prob.implicit_J(eta, alpha)
    Jfd = fd_jacobian(prob, alpha)
    v = np.arange(1, prob.n + 1, dtype=float) / prob.n     # arbitrary outer grad d loss/d eta
    err = np.abs(Jimp - Jfd).max()
    jtv_err = np.abs(Jimp.T @ v - Jfd.T @ v).max()
    print(f"\n=== {name} ===")
    print(f"  eta*         = {np.round(eta, 4)}")
    print(f"  active links = {links}   free vars = {list(free)}")
    print(f"  max|J_implicit - J_fd|      = {err:.2e}")
    print(f"  max|J^T v (impl) - J^T v(fd)| = {jtv_err:.2e}   -> {'OK' if jtv_err < 1e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    a = [[1.0, 1.0], [1.0, 1.0]]     # 2 tasks, 2 links
    # A: saturated shared link, interior box (the coupling regime)
    run("A saturated link (coupling)", w=[1.0, 1.0], R=[1.0, 1.5], a=a, c=[1.5, 1.5],
        alpha=[[0.9, 0.9], [0.9, 0.9]])
    # B: task 1 down-weighted + tight capacity -> floored at eta_min (pinned variable)
    run("B floored low-weight task", w=[1.0, 0.05], R=[1.0, 1.5], a=a, c=[0.7, 0.7],
        alpha=[[0.9, 0.9], [0.9, 0.9]])
    # C: abundant capacity -> all eta*=1 (upper bound), J ~ 0
    run("C all-clean (eta*=1)", w=[1.0, 1.0], R=[1.0, 1.5], a=a, c=[100.0, 100.0],
        alpha=[[0.9, 0.9], [0.9, 0.9]])
