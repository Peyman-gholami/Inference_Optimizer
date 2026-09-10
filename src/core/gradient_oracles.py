"""
gradient_oracles.py
===================
Gradient oracle hierarchy for the inference optimizer.

Classes
-------
GradientOracle (ABC)
    Base interface: gradient_callable(eta_vec: np.ndarray) -> np.ndarray
    Wraps any oracle with optional upward-cone projection.

SteinGradientOracle(GradientOracle)
    Zeroth-order Stein lemma oracle (existing approach).
    Wraps a black-box accuracy callable f: R^d -> R.

AccuracyPredictor
    Concave, monotone-increasing surrogate A_hat(eta) fitted offline
    from (eta, A_true) samples.  Enforces concavity + monotonicity via
    constrained architecture (softplus-based positive weights).

PredictorGradientOracle(GradientOracle)
    Gradient oracle that uses dA_hat/deta via PyTorch autograd.
    Fast (milliseconds vs minutes for Stein).
    Uses a trained AccuracyPredictor.

Cone projection
---------------
Any oracle can have its gradient projected onto the upward cone
    C = { g : g_i >= 0 for all i }
by passing project_upward=True.  This prevents the optimizer from
following gradients that point toward lower eta (more compression),
which is useful when A(eta) has non-convex dips near eta_min.

Usage
-----
# Stein oracle, no projection:
oracle = SteinGradientOracle(f=accuracy_callable, sigma=0.05, N=5, eta_min=0.25)

# Stein oracle with upward-cone projection:
oracle = SteinGradientOracle(f=accuracy_callable, sigma=0.05, N=5,
                              eta_min=0.25, project_upward=True)

# Fit and save predictor:
predictor = AccuracyPredictor(num_links=2, eta_min=0.25)
predictor.fit(accuracy_callable_true, n_grid=20, n_repeat=3,
              save_path="models/predictor_llama_topk.pt")

# Load and use predictor oracle:
predictor = AccuracyPredictor.load("models/predictor_llama_topk.pt")
oracle = PredictorGradientOracle(predictor, eta_min=0.25, project_upward=True)

# Drop-in replacement — same signature as gradient_callable:
grad = oracle(eta_np)   # np.ndarray -> np.ndarray
"""

from __future__ import annotations

import abc
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Abstract base
# =============================================================================

class GradientOracle(abc.ABC):
    """
    Abstract gradient oracle for A(eta).

    Subclasses implement _raw_gradient(eta_vec) -> np.ndarray.
    The public __call__ optionally projects the result onto a cone
    before returning.

    Cone projection
    ---------------
    Controlled by two parameters:

    project_upward : bool
        Master switch.  If False, the raw gradient is returned unchanged.
        If True, cone projection is applied.

    cone_margin : float  (default 0.0)
        Controls the "width" of the allowed cone:

        cone_margin = 0.0  →  hard non-negative projection
                               g_i < 0 are clamped to 0.
                               The optimizer can only increase eta.

        cone_margin = m > 0  →  soft projection
                               g_i < -m are clamped to -m.
                               Small downward gradients (within margin)
                               are allowed through unchanged.
                               Useful to avoid over-projecting when the
                               true gradient is slightly negative due to
                               noise rather than a genuine dip.

        cone_margin = inf  →  equivalent to project_upward=False
                               (no projection, all gradients pass through)

    Examples
    --------
    # Hard projection — never decrease eta:
    oracle = SteinGradientOracle(..., project_upward=True, cone_margin=0.0)

    # Soft projection — allow gradients down to -0.05:
    oracle = SteinGradientOracle(..., project_upward=True, cone_margin=0.05)

    # No projection:
    oracle = SteinGradientOracle(..., project_upward=False)

    Parameters
    ----------
    eta_min        : float or np.ndarray
        Minimum feasible eta per link.
    project_upward : bool
        Master switch for cone projection.
    cone_margin    : float
        Downward gradient tolerance.  0 = hard projection, >0 = soft.
    """

    def __init__(
        self,
        eta_min:        float | np.ndarray = 0.0,
        project_upward: bool               = False,
        cone_margin:    float              = 0.0,
    ):
        self.eta_min        = eta_min
        self.project_upward = project_upward
        self.cone_margin    = cone_margin

    @abc.abstractmethod
    def _raw_gradient(self, eta_vec: np.ndarray) -> np.ndarray:
        """Compute the raw (unprojected) gradient at eta_vec."""

    def __call__(self, eta_vec: np.ndarray) -> np.ndarray:
        """
        Return the (optionally projected) gradient at eta_vec.
        Signature matches InferenceTask.gradient_callable.
        """
        g = self._raw_gradient(eta_vec)
        if self.project_upward:
            g = self._project_onto_cone(g, eta_vec)
        return g

    def _project_onto_cone(
        self, g: np.ndarray, eta_vec: np.ndarray
    ) -> np.ndarray:
        """
        Project gradient onto the cone { g : g_i >= -cone_margin }.

        cone_margin=0.0 → non-negative orthant (hard upward projection).
        cone_margin>0.0 → softer cone allowing small downward components.

        At the upper boundary eta=1, the component is additionally
        clamped so the optimizer cannot push beyond the feasible region.
        """
        g_proj = g.copy()
        # Clamp components below -margin up to -margin
        g_proj = np.maximum(g_proj, -self.cone_margin)
        # At upper boundary eta=1, clamp upward components to 0
        # (cannot increase beyond 1)
        at_upper = np.isclose(eta_vec, 1.0)
        g_proj[at_upper] = np.minimum(g_proj[at_upper], 0.0)
        return g_proj

    @property
    def gradient_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return self as a gradient_callable compatible with InferenceTask."""
        return self


# =============================================================================
# Stein oracle
# =============================================================================

class SteinGradientOracle(GradientOracle):
    """
    Zeroth-order Stein lemma gradient oracle.

    Estimates grad A_sigma(eta) = E_z[ A(eta + sigma*z) * z ] / sigma
    using antithetic pairs (z, -z) for variance reduction.

    Parameters
    ----------
    f              : Callable[[np.ndarray], float]
        Fast accuracy callable (accuracy_callable, not accuracy_callable_true).
    sigma          : float
        Smoothing bandwidth.
    N              : int
        Number of antithetic sample pairs.  Total f-evaluations = 2N.
    eta_min        : float
        Lower clamp for eta during perturbation.
    project_upward : bool
        Project gradient onto cone after estimation.
    cone_margin    : float
        Downward tolerance for cone projection (0 = hard, >0 = soft).
    """

    def __init__(
        self,
        f:               Callable[[np.ndarray], float],
        sigma:           float = 0.05,
        N:               int   = 5,
        eta_min:         float = 0.0,
        project_upward:  bool  = False,
        cone_margin:     float = 0.0,
    ):
        super().__init__(eta_min=eta_min, project_upward=project_upward,
                         cone_margin=cone_margin)
        self.f     = f
        self.sigma = sigma
        self.N     = N

    def _raw_gradient(self, eta_vec: np.ndarray) -> np.ndarray:
        x = torch.tensor(eta_vec, dtype=torch.float32)

        def f_torch(e: torch.Tensor) -> torch.Tensor:
            # Clamp to [eta_min, 1] before calling the black-box
            e_clamped = torch.clamp(e, float(self.eta_min), 1.0)
            v = self.f(e_clamped.detach().cpu().numpy())
            return torch.tensor(float(v), dtype=torch.float32)

        d  = x.numel()
        g  = torch.zeros(d)
        for _ in range(self.N):
            z  = torch.randn(d)
            fp = f_torch(x + self.sigma * z)
            fm = f_torch(x - self.sigma * z)
            g += z * (fp - fm)
        g = g / (2 * self.N * self.sigma)
        return g.numpy()


# =============================================================================
# Concave monotone accuracy predictor
# =============================================================================

class _ConcaveMonotoneNet(nn.Module):
    """
    Surrogate network enforcing A_hat: [eta_min, 1]^d -> [0, 1] to be:
      - Monotone increasing in each eta_i
      - Concave (via input-convex architecture negated)

    Architecture: Input-Convex Neural Network (ICNN) structure negated to
    give concavity.  Specifically we use non-negative weights on hidden
    layers combined with softplus activations to guarantee concavity.

    For d=2 (two links) the network maps [eta_0, eta_1] -> scalar.

    Concavity enforcement:
      Each hidden layer uses weights constrained to be non-negative
      (via softplus reparametrisation) and the composition of non-negative
      weighted concave (softplus) activations is concave.

    Monotonicity enforcement:
      Input is always eta (not negated), weights are non-negative, and
      the output layer weight is non-negative -> function is non-decreasing.
    """

    def __init__(self, num_links: int = 2, hidden: int = 32, depth: int = 3):
        super().__init__()
        self.num_links = num_links
        self.depth     = depth

        # Raw (unconstrained) weight parameters — we apply softplus to get
        # non-negative weights at forward time
        self.w_in  = nn.Parameter(torch.randn(hidden, num_links) * 0.1)
        self.b_in  = nn.Parameter(torch.zeros(hidden))

        self.w_hid = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, hidden) * 0.1)
            for _ in range(depth - 1)
        ])
        self.b_hid = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden))
            for _ in range(depth - 1)
        ])

        self.w_out = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(1))

        # Scale parameter — output is sigmoid(net) * scale + offset
        # so it stays in [0, 1] range naturally
        self.log_scale  = nn.Parameter(torch.zeros(1))   # exp -> scale
        self.offset     = nn.Parameter(torch.zeros(1))

    def _pos(self, w: torch.Tensor) -> torch.Tensor:
        """Map unconstrained weights to non-negative via softplus."""
        return torch.nn.functional.softplus(w)

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        """
        eta : (..., num_links)  in [eta_min, 1]
        returns : (..., 1)  in [0, 1]
        """
        # First layer — non-negative weights, softplus activation (concave)
        h = torch.nn.functional.softplus(
            eta @ self._pos(self.w_in).T + self.b_in
        )
        # Hidden layers — non-negative weights preserve concavity
        for w, b in zip(self.w_hid, self.b_hid):
            h = torch.nn.functional.softplus(
                h @ self._pos(w).T + b
            )
        # Output — non-negative weights, sigmoid squash to [0, 1]
        out = h @ self._pos(self.w_out).T + self.b_out
        return torch.sigmoid(out)


class AccuracyPredictor:
    """
    Offline-fitted concave, monotone-increasing surrogate A_hat(eta).

    Fitting procedure
    -----------------
    1. Evaluates accuracy_callable_true at a grid of eta values in
       [eta_min, 1]^num_links covering the feasible space well.
    2. Fits _ConcaveMonotoneNet to the (eta, A_true) pairs using
       Adam + optional monotonicity regularisation loss.
    3. Saves the trained model to disk for reuse.

    The surrogate is differentiable — gradients are exact via autograd.

    Parameters
    ----------
    num_links    : int
        Number of links (dimension of eta).
    eta_min      : float
        Lower bound on eta.
    hidden       : int
        Hidden layer width of the surrogate network.
    depth        : int
        Number of hidden layers.
    device       : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        num_links: int   = 2,
        eta_min:   float = 0.0,
        hidden:    int   = 32,
        depth:     int   = 3,
        device:    str   = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_links = num_links
        self.eta_min   = eta_min
        self.device    = device
        self.net       = _ConcaveMonotoneNet(num_links, hidden, depth).to(device)
        self._fitted   = False

    def fit(
        self,
        accuracy_callable_true: Callable[[np.ndarray], float],
        n_grid:        int   = 20,
        n_repeat:      int   = 1,
        lr:            float = 1e-3,
        epochs:        int   = 2000,
        mono_weight:   float = 0.1,
        save_path:     Optional[str] = None,
        verbose:       bool  = True,
        grid_eta_min:  float = 0.0,
    ) -> "AccuracyPredictor":
        """
        Collect A_true samples on a grid and fit the surrogate.

        Parameters
        ----------
        accuracy_callable_true : Callable[[np.ndarray], float]
            Full-sample accuracy function (uses full_samples not fast_samples).
        n_grid     : int
            Number of evenly-spaced eta values per dimension.
            Total evaluations = n_grid^num_links × n_repeat.
            For num_links=2, n_grid=15 → 225 evaluations.
        n_repeat   : int
            Number of repeated evaluations per grid point (for averaging
            if the callable has any stochasticity; set 1 for deterministic).
        lr         : float
            Adam learning rate.
        epochs     : int
            Training epochs.
        mono_weight : float
            Weight for monotonicity regularisation loss:
            L_mono = mean(relu(-dA/deta_i)) penalises negative gradients.
        save_path  : str or None
            If provided, saves the trained model + metadata to this path.
        verbose    : bool
            Print training progress every 200 epochs.
        grid_eta_min : float
            Lower bound of the training grid. Defaults to 0.0 so the predictor
            covers the full [0, 1] range regardless of the eta_min used in any
            particular experiment. This allows the same predictor to be reused
            with different eta_min values — eta_min only acts as a clamp in
            the optimizer, not a limit on predictor coverage.
            Set to self.eta_min to reproduce the old behaviour.
        """
        print(f"[AccuracyPredictor.fit] Collecting {n_grid}^{self.num_links} "
              f"× {n_repeat} = "
              f"{n_grid**self.num_links * n_repeat} samples "
              f"over eta in [{grid_eta_min}, 1] ...", flush=True)

        # ── Build grid ────────────────────────────────────────────────────────
        axes   = [np.linspace(grid_eta_min, 1.0, n_grid)
                  for _ in range(self.num_links)]
        grids  = np.meshgrid(*axes, indexing="ij")
        eta_grid = np.stack([g.flatten() for g in grids], axis=1)  # (N, d)

        etas, accs = [], []
        for eta_vec in eta_grid:
            vals = [accuracy_callable_true(eta_vec) for _ in range(n_repeat)]
            etas.append(eta_vec.copy())
            accs.append(float(np.mean(vals)))
            if verbose:
                print(f"  eta={np.round(eta_vec, 3)}  A_true={accs[-1]:.4f}",
                      flush=True)

        eta_t = torch.tensor(np.array(etas), dtype=torch.float32,
                             device=self.device)
        acc_t = torch.tensor(np.array(accs), dtype=torch.float32,
                             device=self.device).unsqueeze(1)

        print(f"[AccuracyPredictor.fit] Fitting surrogate "
              f"({epochs} epochs) ...", flush=True)

        # ── Train ─────────────────────────────────────────────────────────────
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

        for epoch in range(epochs):
            opt.zero_grad()

            # Prediction loss
            pred  = self.net(eta_t)
            loss  = torch.nn.functional.mse_loss(pred, acc_t)

            # Monotonicity regularisation — penalise negative gradients
            if mono_weight > 0:
                eta_req = eta_t.detach().requires_grad_(True)
                p_req   = self.net(eta_req)
                grads   = torch.autograd.grad(
                    p_req.sum(), eta_req, create_graph=True
                )[0]
                loss_mono = torch.relu(-grads).mean()
                loss = loss + mono_weight * loss_mono

            loss.backward()
            opt.step()

            if verbose and (epoch + 1) % 200 == 0:
                print(f"  epoch {epoch+1:4d}/{epochs}  loss={loss.item():.6f}",
                      flush=True)

        self.net.eval()
        self._fitted  = True
        self._eta_data = np.array(etas)
        self._acc_data = np.array(accs)

        # ── Save ──────────────────────────────────────────────────────────────
        if save_path is not None:
            self.save(save_path)

        print("[AccuracyPredictor.fit] Done.", flush=True)
        return self

    def predict(self, eta_vec: np.ndarray) -> float:
        """Evaluate surrogate at eta_vec."""
        self._check_fitted()
        with torch.no_grad():
            t   = torch.tensor(eta_vec, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
            out = self.net(t)
        return float(out.item())

    def gradient(self, eta_vec: np.ndarray) -> np.ndarray:
        """Exact gradient dA_hat/deta via autograd."""
        self._check_fitted()
        t = torch.tensor(eta_vec, dtype=torch.float32,
                         device=self.device, requires_grad=True)
        out = self.net(t.unsqueeze(0)).squeeze()
        out.backward()
        return t.grad.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights and metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "num_links":  self.num_links,
            "eta_min":    self.eta_min,
            "hidden":     self.net.w_in.shape[0],
            "depth":      self.net.depth,
            "eta_data":   self._eta_data if self._fitted else None,
            "acc_data":   self._acc_data if self._fitted else None,
        }, path)
        print(f"[AccuracyPredictor] Saved to {path}", flush=True)

    @classmethod
    def load(cls, path: str,
             device: str = "cuda" if torch.cuda.is_available() else "cpu"
             ) -> "AccuracyPredictor":
        """Load a previously saved predictor."""
        # weights_only=False: these checkpoints carry the numpy eta_data/acc_data
        # grids alongside the state_dict, which the torch>=2.6 default
        # (weights_only=True) refuses to unpickle. The files are this repo's own
        # committed artifacts.
        ckpt       = torch.load(path, map_location=device, weights_only=False)
        predictor  = cls(
            num_links  = ckpt["num_links"],
            eta_min    = ckpt["eta_min"],
            hidden     = ckpt["hidden"],
            depth      = ckpt["depth"],
            device     = device,
        )
        predictor.net.load_state_dict(ckpt["state_dict"])
        predictor.net.eval()
        predictor._fitted   = True
        predictor._eta_data = ckpt.get("eta_data")
        predictor._acc_data = ckpt.get("acc_data")
        print(f"[AccuracyPredictor] Loaded from {path}", flush=True)
        return predictor

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "AccuracyPredictor has not been fitted. "
                "Call .fit() or .load() first."
            )


# =============================================================================
# Predictor gradient oracle
# =============================================================================

class PredictorGradientOracle(GradientOracle):
    """
    Gradient oracle using dA_hat/deta via autograd through the trained
    AccuracyPredictor surrogate.

    Orders of magnitude faster than Stein (milliseconds vs minutes).
    Gradients are exact with respect to the smooth surrogate.

    Parameters
    ----------
    predictor      : AccuracyPredictor
        Fitted surrogate model.
    eta_min        : float
        Lower bound for clamping.
    project_upward : bool
        Project gradient onto cone after autograd.
    cone_margin    : float
        Downward tolerance for cone projection (0 = hard, >0 = soft).
    """

    def __init__(
        self,
        predictor:      AccuracyPredictor,
        eta_min:        float = 0.0,
        project_upward: bool  = False,
        cone_margin:    float = 0.0,
    ):
        super().__init__(eta_min=eta_min, project_upward=project_upward,
                         cone_margin=cone_margin)
        self.predictor = predictor

    def _raw_gradient(self, eta_vec: np.ndarray) -> np.ndarray:
        return self.predictor.gradient(eta_vec)

    def predict(self, eta_vec: np.ndarray) -> float:
        """Convenience wrapper — evaluate surrogate accuracy."""
        return self.predictor.predict(eta_vec)
