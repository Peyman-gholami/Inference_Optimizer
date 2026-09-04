"""
Full COUPLED multi-task hypergradient --- core (surrogate + FD-validatable).

Inner problem in eta (per channel c), K tasks sharing d links:
    max_eta  Phi(eta) = sum_k w_k A_k(eta_k),   A_k = prod_i (1 - alpha_{k,i}(1-eta_{k,i})^2)
    s.t.     B eta <= 1   (per-link throughput; B[i,(k,i)] = R_k a_{k,i}/c_i)
             eta_min <= eta_{k,i} <= 1

We want J = d eta*/d theta (here theta = the surrogate params alpha) via the implicit
function theorem applied to the KKT system, restricted to the ACTIVE set, and the
coupled hypergradient term J^T v (v = d(outer loss)/d eta). Bound-pinned variables
contribute J = 0 (strict complementarity); the coupling lives in the free variables tied
by active (saturated) link constraints.

This module is the surrogate core: H = d^2Phi/deta^2 and M = d^2Phi/deta dtheta are exact
(torch autograd on the tiny surrogate), so J can be finite-difference-validated. Later,
for the real model, H comes from a Stein-Hessian and M^T a from the Stein-hybrid.
"""

import numpy as np
import torch
from scipy.optimize import minimize

# NB: do not set a global default dtype here -- the surrogate stays float64 because its
# inputs are numpy float64, and a global double default breaks MPS (float32-only) callers.


class SurrogateCoupled:
    def __init__(self, w, R, a, c, eta_min=0.1):
        self.w = np.asarray(w, float)          # (K,)
        self.R = np.asarray(R, float)          # (K,)
        self.a = np.asarray(a, float)          # (K, d)
        self.c = np.asarray(c, float)          # (d,)
        self.K, self.d = self.a.shape
        self.n = self.K * self.d
        self.eta_min = float(eta_min)
        B = np.zeros((self.d, self.n))         # link-throughput coefficients
        for k in range(self.K):
            for i in range(self.d):
                B[i, k * self.d + i] = self.R[k] * self.a[k, i] / self.c[i]
        self.B = B

    # ---- surrogate objective and its derivatives (torch autograd) ----
    def _phi(self, eta_t, alpha_t):
        eta = eta_t.reshape(self.K, self.d)
        alpha = alpha_t.reshape(self.K, self.d)
        Ak = torch.prod(1.0 - alpha * (1.0 - eta) ** 2, dim=1)
        return (torch.tensor(self.w) * Ak).sum()

    def phi(self, eta, alpha):
        return float(self._phi(torch.tensor(eta), self._al(alpha)))

    @staticmethod
    def _al(alpha):
        return torch.tensor(np.asarray(alpha, float).reshape(-1))

    def grad_eta(self, eta, alpha):
        e = torch.tensor(eta, requires_grad=True)
        g, = torch.autograd.grad(self._phi(e, self._al(alpha)), e)
        return g.numpy()

    def hess_eta(self, eta, alpha):
        al = self._al(alpha)
        H = torch.autograd.functional.hessian(lambda e: self._phi(e, al), torch.tensor(eta))
        return H.numpy()

    def mixed_eta_theta(self, eta, alpha):
        """M[p,q] = d^2 Phi / d eta_p d alpha_q  ->  (n, n)."""
        e = torch.tensor(eta)

        def geta(al_):
            ee = e.clone().requires_grad_(True)
            g, = torch.autograd.grad(self._phi(ee, al_), ee, create_graph=True)
            return g
        return torch.autograd.functional.jacobian(geta, self._al(alpha)).numpy()

    # ---- inner solve (water-fill in eta) ----
    def solve(self, alpha):
        alpha = np.asarray(alpha, float)
        cons = [{"type": "ineq",
                 "fun": lambda x: 1.0 - self.B @ x,
                 "jac": lambda x: -self.B}]
        bnds = [(self.eta_min, 1.0)] * self.n
        res = minimize(lambda x: -self.phi(x, alpha), np.full(self.n, self.eta_min),
                       jac=lambda x: -self.grad_eta(x, alpha), method="SLSQP",
                       bounds=bnds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        return res.x

    def active_set(self, eta, atol=1e-4):
        g = self.B @ eta
        active_links = [i for i in range(self.d) if g[i] > 1.0 - atol]
        at_bound = np.array([(eta[p] <= self.eta_min + atol) or (eta[p] >= 1.0 - atol)
                             for p in range(self.n)])
        free = np.where(~at_bound)[0]
        return active_links, free

    # ---- implicit sensitivity J = d eta*/d theta ----
    def implicit_J(self, eta, alpha):
        active_links, free = self.active_set(eta)
        H = self.hess_eta(eta, alpha)                 # d^2 Phi / d eta^2  (neg. def.)
        M = self.mixed_eta_theta(eta, alpha)          # d^2 Phi / d eta d theta
        nf = len(free)
        Hff = H[np.ix_(free, free)]
        Mf = M[np.ix_(free, range(self.n))]           # (nf, n)
        Baf = self.B[np.ix_(active_links, free)] if active_links else np.zeros((0, nf))
        keep = [r for r in range(Baf.shape[0]) if np.any(np.abs(Baf[r]) > 1e-12)]
        Baf = Baf[keep] if keep else np.zeros((0, nf))
        m = Baf.shape[0]
        # KKT of the maximization Lagrangian L = Phi - lam^T (B eta - 1):
        #   H dEtaF - Baf^T dLam = -Mf dTheta ;   Baf dEtaF = 0
        W = np.block([[Hff, -Baf.T], [Baf, np.zeros((m, m))]])
        rhs = np.vstack([-Mf, np.zeros((m, self.n))])
        sol = np.linalg.solve(W, rhs)
        J = np.zeros((self.n, self.n))
        J[free, :] = sol[:nf, :]
        return J

    def Jt_v(self, eta, alpha, v):
        return self.implicit_J(eta, alpha).T @ np.asarray(v, float)


# ===========================================================================
# Shared reduced-KKT adjoint + Stein oracles (for the real-model coupled path)
# ===========================================================================
def kkt_adjoint(H, B, active_links, free, n, v):
    """Adjoint a (full n-vector, 0 on pinned vars) with  J^T v = -M^T a.
    W = [[H_ff, -B_af^T],[B_af, 0]];  a_free = top block of solve(W^T, [v_free; 0]).
    (Requires only H and the active set --- H may come from a Stein-Hessian.)"""
    free = list(free)
    nf = len(free)
    Hff = H[np.ix_(free, free)]
    Baf = B[np.ix_(active_links, free)] if active_links else np.zeros((0, nf))
    keep = [r for r in range(Baf.shape[0]) if np.any(np.abs(Baf[r]) > 1e-12)]
    Baf = Baf[keep] if keep else np.zeros((0, nf))
    m = Baf.shape[0]
    W = np.block([[Hff, -Baf.T], [Baf, np.zeros((m, m))]])
    rhs = np.concatenate([np.asarray(v, float)[free], np.zeros(m)])
    sol = np.linalg.solve(W.T, rhs)
    a = np.zeros(n)
    a[free] = sol[:nf]
    return a


def kkt_adjoint_reduced(H, B, active_links, free, n, v, rcond=1e-3, damp=0.0):
    """Adjoint a with J^T v = -M^T a, computed in the active-constraint NULL SPACE using a
    TRUNCATED pseudo-inverse of the reduced Hessian (options A+B):
        Z = null(B_af);   a_free = Z (Z^T H_ff Z + damp I)^{+}_{rcond} Z^T v_free.
    Robust to a small / indefinite H (drops eigen-directions below rcond*max|eig|). With
    rcond->0, damp=0 and a well-conditioned H it matches `kkt_adjoint`."""
    free = list(free); nf = len(free)
    Hff = np.asarray(H, float)[np.ix_(free, free)]
    v_free = np.asarray(v, float)[free]
    Baf = np.asarray(B, float)[np.ix_(active_links, free)] if active_links else np.zeros((0, nf))
    keep = [r for r in range(Baf.shape[0]) if np.any(np.abs(Baf[r]) > 1e-12)]
    Baf = Baf[keep] if keep else np.zeros((0, nf))
    if Baf.shape[0] > 0:                      # Z spans the tangent of the active constraints
        _, s, vt = np.linalg.svd(Baf)
        rank = int((s > 1e-10).sum())
        Z = vt[rank:].T
    else:
        Z = np.eye(nf)
    if Z.shape[1] == 0:                       # no free tangent directions -> J = 0
        return np.zeros(n)
    Hr = Z.T @ Hff @ Z + damp * np.eye(Z.shape[1])
    Hr = 0.5 * (Hr + Hr.T)
    a_free = Z @ (np.linalg.pinv(Hr, rcond=rcond, hermitian=True) @ (Z.T @ v_free))
    a = np.zeros(n); a[free] = a_free
    return a


def stein_grad(f, x, sigma=0.05, N=2000, seed=0):
    """Antithetic Gaussian-smoothing gradient of scalar f: R^n -> R at x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); n = x.size; g = np.zeros(n)
    for z in rng.standard_normal((N, n)):
        g += (f(x + sigma * z) - f(x - sigma * z)) / (2 * sigma) * z
    return g / N


def stein_hessian(f, x, sigma=0.05, N=4000, seed=1):
    """Antithetic Stein Hessian of scalar f: R^n -> R at x (symmetrized)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); n = x.size; H = np.zeros((n, n)); f0 = f(x)
    for z in rng.standard_normal((N, n)):
        val = f(x + sigma * z) + f(x - sigma * z) - 2 * f0
        H += val * (np.outer(z, z) - np.eye(n))
    H = H / (2 * sigma * sigma * N)
    return 0.5 * (H + H.T)


def stein_hybrid_basis(grad_theta_at_eta, eta, sigma=0.05, N=40, seed=2):
    """Return the mixed 'basis' B (p x n) with column i approx M^T e_i =
    d/dtheta ( d Phi / d eta_i ), via the Stein-hybrid. Then M^T a = B @ a for ANY a
    (one sampling pass serves every adjoint / damping level, since M^T a is linear in a)."""
    rng = np.random.default_rng(seed)
    eta = np.asarray(eta, float); n = eta.size
    basis = None
    for z in rng.standard_normal((N, n)):
        g = (grad_theta_at_eta(eta + sigma * z) - grad_theta_at_eta(eta - sigma * z)) / (2 * sigma)
        contrib = np.outer(g, z)          # (p, n): column i = g * z_i
        basis = contrib if basis is None else basis + contrib
    return basis / N


def stein_hybrid_Mta(grad_theta_at_eta, eta, a, sigma=0.05, N=2000, seed=2):
    """Mixed term  M^T a = d/dtheta ( a^T grad_eta Phi )  via the Stein-hybrid:
    Stein in eta, autograd in theta.  grad_theta_at_eta(eta) -> d Phi/d theta (R^p)."""
    rng = np.random.default_rng(seed)
    eta = np.asarray(eta, float); a = np.asarray(a, float); n = eta.size
    acc = None
    for z in rng.standard_normal((N, n)):
        w = (a @ z) / (2 * sigma)
        term = w * (grad_theta_at_eta(eta + sigma * z) - grad_theta_at_eta(eta - sigma * z))
        acc = term if acc is None else acc + term
    return acc / N
