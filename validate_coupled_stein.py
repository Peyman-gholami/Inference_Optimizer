"""
End-to-end validation of the FULL coupled hypergradient with Stein oracles.

Tiny theta-parametrized testbed (2 tasks, 2 links, theta in R^2), with a deliberate
INNER/OUTER mismatch so the coupling term is nonzero:

    alpha_k(theta) = 0.5 + 0.4 tanh(theta_k)
    A_k(theta,eta) = prod_i (1 - alpha_k(theta)(1-eta_{k,i})^2)
    inner (controller): eta*(theta) = argmax_eta  sum_k w_k A_k   s.t. B eta<=1, box   (w=[1,1])
    outer (training):   F(theta)    = -sum_k u_k A_k(theta, eta*)                        (u=[1,3])

Because u != w, v = d(outer)/d eta is NOT annihilated at eta* -> J^T v != 0.
    dF/dtheta = direct + J^T v,   J = d eta*/d theta (from the INNER problem).

Compare FD (truth) vs analytic-coupled vs Stein-coupled vs decoupled(direct only).
"""
import numpy as np
import torch

from src.pretraining.coupled import (
    SurrogateCoupled, kkt_adjoint, stein_hessian, stein_hybrid_Mta,
)

torch.set_default_dtype(torch.double)
W_INNER = [1.0, 1.0]        # controller's QoS weights (inner objective)
U_OUTER = [1.0, 3.0]        # deployment weights (outer loss)  -> mismatch
R = [1.0, 1.5]
A = [[1.0, 1.0], [1.0, 1.0]]
C = [1.5, 1.5]
THETA0 = np.array([0.3, -0.2])


def alpha_of_theta(theta_t):
    ak = 0.5 + 0.4 * torch.tanh(theta_t)
    return ak.repeat_interleave(2)


def Phi(theta_t, eta_t, weights):
    al = alpha_of_theta(theta_t).reshape(2, 2)
    e = eta_t.reshape(2, 2)
    Ak = torch.prod(1.0 - al * (1.0 - e) ** 2, dim=1)
    return (torch.tensor(weights) * Ak).sum()


def phi_val(theta, eta, weights):
    return float(Phi(torch.tensor(theta), torch.tensor(eta), weights))


def grad_theta(theta, eta, weights):
    th = torch.tensor(theta, requires_grad=True)
    g, = torch.autograd.grad(Phi(th, torch.tensor(eta), weights), th)
    return g.numpy()


def grad_eta(theta, eta, weights):
    e = torch.tensor(eta, requires_grad=True)
    g, = torch.autograd.grad(Phi(torch.tensor(theta), e, weights), e)
    return g.numpy()


def alpha_np(theta):
    return alpha_of_theta(torch.tensor(theta)).numpy()


def jac_alpha_theta(theta):
    sech2 = 1.0 - np.tanh(theta) ** 2
    Jac = np.zeros((4, 2))
    for k in range(2):
        for i in range(2):
            Jac[k * 2 + i, k] = 0.4 * sech2[k]
    return Jac


def main():
    prob = SurrogateCoupled(w=W_INNER, R=R, a=A, c=C, eta_min=0.1)   # inner problem
    alpha0 = alpha_np(THETA0)
    eta = prob.solve(alpha0)
    active_links, free = prob.active_set(eta)
    n = prob.n
    print(f"eta*={np.round(eta,4)}  active_links={active_links}  free={list(free)}")

    v = -grad_eta(THETA0, eta, U_OUTER)                 # d(outer loss)/d eta
    direct = -grad_theta(THETA0, eta, U_OUTER)          # d(outer)/d theta at fixed eta*
    Jac = jac_alpha_theta(THETA0)

    # analytic coupled (J from the inner problem, chained through alpha(theta))
    Jtv_theta = Jac.T @ prob.Jt_v(eta, alpha0, v)
    coupled_analytic = direct + Jtv_theta

    # FD ground truth
    eps = 1e-5
    fd = np.zeros(2)
    for j in range(2):
        tp, tm = THETA0.copy(), THETA0.copy(); tp[j] += eps; tm[j] -= eps
        Fp = -phi_val(tp, prob.solve(alpha_np(tp)), U_OUTER)
        Fm = -phi_val(tm, prob.solve(alpha_np(tm)), U_OUTER)
        fd[j] = (Fp - Fm) / (2 * eps)

    # Stein coupled: H = Stein-Hessian of the INNER objective; M^T a via Stein-hybrid
    H = stein_hessian(lambda e: phi_val(THETA0, e, W_INNER), eta, sigma=0.05, N=8000)
    a = kkt_adjoint(H, prob.B, active_links, free, n, v)
    Mta = stein_hybrid_Mta(lambda e: grad_theta(THETA0, e, W_INNER), eta, a, sigma=0.05, N=4000)
    Jtv_stein = -Mta
    coupled_stein = direct + Jtv_stein

    def line(name, g):
        print(f"  {name:<20} {np.round(g,5)}   ||.-FD||={np.linalg.norm(g-fd):.2e}")

    print("\ndF/dtheta estimates:")
    print(f"  {'FD (truth)':<20} {np.round(fd,5)}")
    line("analytic coupled", coupled_analytic)
    line("Stein coupled", coupled_stein)
    line("decoupled (direct)", direct)
    print(f"\n  coupling term  J^T v:  analytic={np.round(Jtv_theta,5)}  Stein={np.round(Jtv_stein,5)}")
    print(f"  ||J^T v|| = {np.linalg.norm(Jtv_theta):.4f}  -> decoupled misses this much")


if __name__ == "__main__":
    main()
