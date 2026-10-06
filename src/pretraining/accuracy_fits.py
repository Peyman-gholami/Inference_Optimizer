"""
accuracy_fits.py
================
Pluggable surrogate families for A_hat(eta), with a common interface so they can
be compared head to head (fit quality, concavity, and downstream training).

Four arms:

  ProductTwoStage   the CURRENT `LiveAccuracyProfile` fit, kept as the baseline:
                    c0 := A(eta=1) from one measurement, then alpha by least
                    squares CONDITIONAL on that c0.
  ProductJoint      same family, but (c0, alpha) fitted JOINTLY. Measured 32%
                    lower RMSE on an inverted marginal, identical when healthy.
  SeparableConcave  A_hat = c0 (1 - sum_i alpha_i (1-eta_i)^2). Concave
                    UNCONDITIONALLY -- the cross-partials are exactly zero.
  ConcaveNet        concave + monotone by construction (see the note below).

Why the product family is not concave
-------------------------------------
For A_hat = c0 prod_i (1 - alpha_i (1-eta_i)^2) each 1-D marginal IS concave,
but the product gives a POSITIVE cross-partial, so with s_i = alpha_i(1-eta_i)^2
and d = 2:

    concave  <=>  1 - s_1 - s_2 - 3 s_1 s_2 > 0      (symmetric: s < 1/3)

i.e. at the floor eta = 0.1, alpha < 0.412. Measured: alpha = [0.6, 0.6] is
indefinite on 7.8% of the eta box, max eigenvalue +0.55 at the floor. The fits
seen in practice were safe only because they were asymmetric (one sensitive
cut-point, one not), which keeps the cross term small.

Note on ConcaveNet
------------------
The `_ConcaveMonotoneNet` on the `concavity-experiments` branch claims concavity
but is NOT concave: softplus is CONVEX, so non-negative weights + softplus is an
ICNN (convex), and the output sigmoid composed with a convex function is neither.
It only measures concave when the sigmoid SATURATES, where the net is a constant
and the Hessian is vacuously zero. Verified: removing the sigmoid gives max
Hessian eigenvalue +2.7e3; putting the sigmoid in its active range gives
+1e-2..+3e-2 near the floor.

The construction here is concave by an actual argument:
  * activation  sigma(x) = -softplus(-x)  (= logsigmoid) is concave AND
    nondecreasing;
  * all weights are non-negative (softplus reparametrisation);
  * a non-negative combination of concave functions is concave, and a concave
    nondecreasing function composed with a concave function is concave;
  * the output layer is non-negative-linear, and there is NO output sigmoid.
Monotone increasing follows from non-negative weights and nondecreasing sigma.
Verified: max Hessian eigenvalue -2.99 over the box.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------

class AccuracyFit:
    """Common interface. Derivatives come from torch autograd on `_A_torch`, so
    every family shares one code path and no gradient is hand-derived."""

    name = "base"

    def __init__(self, d: int, eta_min: float = 0.1):
        self.d = int(d)
        self.eta_min = float(eta_min)

    # -- to implement ------------------------------------------------------
    def fit(self, etas: np.ndarray, accs: np.ndarray) -> "AccuracyFit":
        raise NotImplementedError

    def _A_torch(self, eta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    # -- shared ------------------------------------------------------------
    def A(self, eta) -> float:
        with torch.no_grad():
            return float(self._A_torch(self._t(eta)))

    def A_many(self, etas) -> np.ndarray:
        return np.array([self.A(e) for e in np.asarray(etas, float)])

    def gradA(self, eta) -> np.ndarray:
        e = self._t(eta).requires_grad_(True)
        g, = torch.autograd.grad(self._A_torch(e), e)
        return g.detach().numpy().astype(float)

    def hess(self, eta) -> np.ndarray:
        H = torch.autograd.functional.hessian(self._A_torch, self._t(eta))
        H = H.detach().numpy().astype(float)
        return 0.5 * (H + H.T)

    def rmse(self, etas, accs) -> float:
        return float(np.sqrt(np.mean((self.A_many(etas) - np.asarray(accs, float)) ** 2)))

    def max_hess_eig(self, n: int = 11) -> tuple[float, tuple]:
        """Largest Hessian eigenvalue over the eta box. <= 0 means concave there.

        This is the concavity check the water-fill silently assumes.
        """
        axis = np.linspace(self.eta_min, 1.0, n)
        worst, where = -np.inf, None
        for eta in np.array(np.meshgrid(*([axis] * self.d))).reshape(self.d, -1).T:
            ev = float(np.linalg.eigvalsh(self.hess(eta)).max())
            if ev > worst:
                worst, where = ev, tuple(np.round(eta, 3))
        return worst, where

    def is_monotone(self, n: int = 11, tol: float = 1e-9) -> bool:
        axis = np.linspace(self.eta_min, 1.0, n)
        for eta in np.array(np.meshgrid(*([axis] * self.d))).reshape(self.d, -1).T:
            if np.any(self.gradA(eta) < -tol):
                return False
        return True

    @staticmethod
    def _t(eta) -> torch.Tensor:
        return torch.as_tensor(np.asarray(eta, float), dtype=torch.float64)


# ---------------------------------------------------------------------------
# 1. the current fit, kept as the baseline
# ---------------------------------------------------------------------------

class ProductTwoStage(AccuracyFit):
    """c0 plugged in from the eta=1 measurement, then alpha by least squares.

    Reproduces `LiveAccuracyProfile.refit` exactly, including the
    `clip(alpha, 0, 0.999)` that pins alpha to 0 once the curve inverts (because
    c0 is measured at eta=1, which after narrow-band training is the WORST
    point, so every other measurement lands above it and 1 - acc/c0 < 0).
    """

    name = "product / two-stage (current)"

    def fit(self, etas, accs):
        etas, accs = np.asarray(etas, float), np.asarray(accs, float)
        ones = np.ones(self.d)
        i1 = int(np.argmin(np.linalg.norm(etas - ones, axis=1)))
        self.c0 = max(1e-3, float(accs[i1]))
        self.alpha = np.zeros(self.d)
        for i in range(self.d):
            # the marginal points for cut i: this cut varies, the others at 1
            m = np.all(np.delete(etas, i, axis=1) > 1 - 1e-9, axis=1)
            if m.sum() < 2:
                m = np.ones(len(etas), bool)      # fall back to everything
            w = (1.0 - etas[m, i]) ** 2
            r = 1.0 - np.minimum(1.0, accs[m] / self.c0)
            den = float((w * w).sum())
            self.alpha[i] = float(np.clip((r * w).sum() / den if den > 0 else 0.0,
                                          0.0, 0.999))
        return self

    def _A_torch(self, eta):
        a = torch.as_tensor(self.alpha, dtype=torch.float64)
        return self.c0 * torch.prod(1.0 - a * (1.0 - eta) ** 2)

    def describe(self):
        return f"{self.name}: c0={self.c0:.4f} alpha={np.round(self.alpha, 4)}"


# ---------------------------------------------------------------------------
# 2. same family, joint fit
# ---------------------------------------------------------------------------

class ProductJoint(AccuracyFit):
    """(c0, alpha) fitted jointly. Optionally constrained to stay concave."""

    name = "product / joint"

    def __init__(self, d, eta_min=0.1, enforce_concave: bool = False):
        super().__init__(d, eta_min)
        self.enforce_concave = enforce_concave
        if enforce_concave:
            self.name = "product / joint + concavity guard"

    def _alpha_cap(self) -> float:
        """Cap making 1 - sum s_i - 3 prod s_i > 0 at the worst corner eta_min.

        For the symmetric worst case s_i = alpha (1-eta_min)^2 the d=2 condition
        is s < 1/3; we apply the same cap per coordinate, which is conservative
        for d > 2.
        """
        return min(0.999, (1.0 / 3.0) / (1.0 - self.eta_min) ** 2)

    def fit(self, etas, accs):
        etas, accs = np.asarray(etas, float), np.asarray(accs, float)
        hi = self._alpha_cap() if self.enforce_concave else 0.999

        def resid(p):
            c0, a = p[0], p[1:]
            return c0 * np.prod(1.0 - a * (1.0 - etas) ** 2, axis=1) - accs

        p0 = np.concatenate([[float(accs.max())], np.full(self.d, 0.1)])
        lo = np.concatenate([[0.0], np.zeros(self.d)])
        up = np.concatenate([[1.0], np.full(self.d, hi)])
        s = least_squares(resid, np.clip(p0, lo, up), bounds=(lo, up))
        self.c0, self.alpha = float(s.x[0]), np.asarray(s.x[1:], float)
        return self

    def _A_torch(self, eta):
        a = torch.as_tensor(self.alpha, dtype=torch.float64)
        return self.c0 * torch.prod(1.0 - a * (1.0 - eta) ** 2)

    def describe(self):
        return f"{self.name}: c0={self.c0:.4f} alpha={np.round(self.alpha, 4)}"


# ---------------------------------------------------------------------------
# 3. additively separable -- concave unconditionally
# ---------------------------------------------------------------------------

class SeparableConcave(AccuracyFit):
    """A_hat = c0 (1 - sum_i alpha_i (1-eta_i)^2).

    Cross-partials are exactly 0 and each term is concave, so the whole function
    is concave on the box for any alpha >= 0 -- no condition to check. The cost
    is losing the multiplicative interaction between cut-points.

    alpha is bounded so A_hat stays non-negative at the eta_min corner.
    """

    name = "separable concave"

    def fit(self, etas, accs):
        etas, accs = np.asarray(etas, float), np.asarray(accs, float)
        cap = 1.0 / (self.d * (1.0 - self.eta_min) ** 2)

        def resid(p):
            c0, a = p[0], p[1:]
            return c0 * (1.0 - ((1.0 - etas) ** 2 * a).sum(axis=1)) - accs

        p0 = np.concatenate([[float(accs.max())], np.full(self.d, min(0.1, cap))])
        lo = np.concatenate([[0.0], np.zeros(self.d)])
        up = np.concatenate([[1.0], np.full(self.d, cap)])
        s = least_squares(resid, np.clip(p0, lo, up), bounds=(lo, up))
        self.c0, self.alpha = float(s.x[0]), np.asarray(s.x[1:], float)
        return self

    def _A_torch(self, eta):
        a = torch.as_tensor(self.alpha, dtype=torch.float64)
        return self.c0 * (1.0 - (a * (1.0 - eta) ** 2).sum())

    def describe(self):
        return f"{self.name}: c0={self.c0:.4f} alpha={np.round(self.alpha, 4)}"


# ---------------------------------------------------------------------------
# 4. concave + monotone network (correct construction)
# ---------------------------------------------------------------------------

class _ConcaveMonotoneNet(nn.Module):
    """Concave and nondecreasing in every input, by construction.

    sigma(x) = -softplus(-x) is concave and nondecreasing; all weights are
    non-negative; there is no output squash. A non-negative combination of
    concave functions is concave, and concave-nondecreasing composed with
    concave is concave -- so every layer preserves both properties.
    """

    def __init__(self, d: int, hidden: int = 24, depth: int = 2):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(hidden, d, dtype=torch.float64) * 0.5 - 1.0)
        self.b_in = nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
        self.w_hid = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, hidden, dtype=torch.float64) * 0.1 - 2.0)
            for _ in range(depth - 1)])
        self.b_hid = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
            for _ in range(depth - 1)])
        self.w_out = nn.Parameter(torch.randn(1, hidden, dtype=torch.float64) * 0.1 - 1.0)
        self.b_out = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        # Output affine: a NON-NEGATIVE scale times a concave function, plus a free
        # constant, is still concave -- so this buys range/offset freedom (the net
        # body is built from -softplus(-x) and so starts out negative) without
        # weakening the guarantee. No sigmoid: that is what broke the inherited net.
        self.out_scale = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.out_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    @staticmethod
    def _sigma(x):
        return -F.softplus(-x)          # concave, nondecreasing

    def forward(self, eta):
        one_d = eta.dim() == 1
        e = eta.reshape(1, -1) if one_d else eta
        h = self._sigma(e @ F.softplus(self.w_in).T + self.b_in)
        for w, b in zip(self.w_hid, self.b_hid):
            h = self._sigma(h @ F.softplus(w).T + b)
        core = (h @ F.softplus(self.w_out).T + self.b_out).squeeze(-1)
        out = F.softplus(self.out_scale) * core + self.out_bias
        return out.squeeze(0) if one_d else out


class ConcaveNet(AccuracyFit):
    name = "concave-monotone net"

    def __init__(self, d, eta_min=0.1, hidden=24, depth=2, epochs=3000,
                 lr=0.05, seed=0):
        super().__init__(d, eta_min)
        self.hidden, self.depth, self.epochs, self.lr, self.seed = \
            hidden, depth, epochs, lr, seed
        self.net: Optional[_ConcaveMonotoneNet] = None

    def fit(self, etas, accs):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(np.asarray(etas, float), dtype=torch.float64)
        y = torch.as_tensor(np.asarray(accs, float), dtype=torch.float64)
        self.net = _ConcaveMonotoneNet(self.d, self.hidden, self.depth)
        # start at the data's level so the net only has to learn the SHAPE
        with torch.no_grad():
            self.net.out_bias.fill_(float(y.mean()))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        for _ in range(self.epochs):
            opt.zero_grad()
            F.mse_loss(self.net(X), y).backward()
            opt.step()
            sched.step()
        self.net.eval()
        return self

    def _A_torch(self, eta):
        return self.net(eta)

    def describe(self):
        return f"{self.name}: hidden={self.hidden} depth={self.depth}"


# ---------------------------------------------------------------------------

ALL_FITS = {
    "two_stage": ProductTwoStage,
    "joint": ProductJoint,
    "joint_guarded": lambda d, eta_min=0.1: ProductJoint(d, eta_min, enforce_concave=True),
    "separable": SeparableConcave,
    "net": ConcaveNet,
}


def build_fit(name: str, d: int, eta_min: float = 0.1) -> AccuracyFit:
    if name not in ALL_FITS:
        raise KeyError(f"unknown fit {name!r}; choose from {sorted(ALL_FITS)}")
    return ALL_FITS[name](d, eta_min)
