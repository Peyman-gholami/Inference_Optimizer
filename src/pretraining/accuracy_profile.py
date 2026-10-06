"""
LiveAccuracyProfile: a cheap, θ-tracking A_k(η) oracle for the multi-task controller.

`CSIAwareMultiTaskOptimizer` calls A_k(η) and grad_A_k(η) many times per SLSQP solve,
per channel, per step -- far too often to evaluate the model inside the solve. So we
fit a cheap *analytic* concave surrogate to the model's TRUE top-k accuracy and refit
it periodically from freshly measured points (every N steps), so it tracks the live θ:

    A_k(η) = c0 * Π_i (1 - α_i (1 - η_i)^2)

  - c0    = measured clean accuracy at η = 1
  - α_i   ∈ [0,1): per-cut-point compression sensitivity, least-squares-fit to a
            marginal sweep of cut i (others held at 1)

The fit gives the optimizer an analytic A_k and grad_A_k (no model eval inside SLSQP),
while the periodic refit keeps it tied to the current weights. Measurement uses the
repo's value-only accuracy semantics (no_grad + argmax) on a fixed subset.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.accuracy_fn import CompressedModel


class LiveAccuracyProfile:
    """Fit A_k(eta) = c0 * prod_i (1 - alpha_i (1-eta_i)^2) to the live model.

    `mode` selects what "accuracy" is measured:

      "hard"  top-1 accuracy (argmax).  The deployment metric, and the default
              so existing callers are unchanged.  Note it is QUANTIZED to
              1/len(x_meas) -- with a 200-sample subset, steps of 0.005 -- so
              c0 and alpha are STEP functions of theta, and therefore so are
              eta*(theta) and any deployed loss computed through them.
      "soft"  mean correct-class softmax probability (the codebase's soft
              A-tilde, as used by the Stein H and the Stein-hybrid M).  Smooth
              in theta.

    Use "soft" whenever theta is being differentiated or finite-differenced
    THROUGH the controller: the coupled hypergradient J = d eta*/d theta is
    derived for a profile that responds smoothly to theta, so validating it
    against the "hard" profile finite-differences a staircase, which has no
    gradient to compare against.
    """

    def __init__(self, cm: CompressedModel, x_meas: torch.Tensor, y_meas: torch.Tensor,
                 d: int, device, eta_grid=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85),
                 mode: str = "hard"):
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.cm = cm
        self.x = x_meas.to(device)
        self.y = y_meas.to(device)
        self.d = d
        self.device = device
        self.eta_grid = np.asarray(eta_grid, float)
        self.mode = mode
        self.c0 = 1.0
        self.alpha = np.full(d, 0.5)
        self.refit()

    @torch.no_grad()
    def _measure(self, eta_vec) -> float:
        """Measured accuracy at eta_vec on the measurement subset.

        Saves/restores the model's eta and train/eval mode.  See `mode`.
        """
        was_training = self.cm.training
        old_eta = self.cm.eta.clone()
        self.cm.eval()
        self.cm.set_eta(torch.as_tensor(eta_vec, dtype=torch.float32, device=self.device))
        logits = self.cm(self.x)
        if self.mode == "hard":
            val = (logits.argmax(1) == self.y).float().mean().item()
        else:
            p = torch.softmax(logits, 1)
            val = p[torch.arange(len(self.y), device=self.device), self.y].mean().item()
        self.cm.set_eta(old_eta)
        self.cm.train(was_training)
        return val

    def refit(self) -> "LiveAccuracyProfile":
        ones = np.ones(self.d)
        self.c0 = max(1e-3, self._measure(ones))
        for i in range(self.d):
            # marginal sweep of cut i with the others at 1; least-squares for alpha_i:
            #   1 - acc/c0 ≈ alpha_i (1-eta_i)^2   ->   min_alpha Σ (r - alpha w)^2,  w=(1-eta)^2
            num = den = 0.0
            for e in self.eta_grid:
                eta = ones.copy(); eta[i] = e
                r = 1.0 - min(1.0, self._measure(eta) / self.c0)
                w = (1.0 - e) ** 2
                num += r * w
                den += w * w
            self.alpha[i] = float(np.clip(num / den if den > 0 else 0.0, 0.0, 0.999))
        return self

    # --- analytic oracles handed to InferenceTask ---
    def A(self, eta) -> float:
        eta = np.asarray(eta, float)
        return float(self.c0 * np.prod(1 - self.alpha * (1 - eta) ** 2))

    def gradA(self, eta) -> np.ndarray:
        eta = np.asarray(eta, float)
        terms = 1 - self.alpha * (1 - eta) ** 2
        return np.array([self.c0 * 2 * self.alpha[j] * (1 - eta[j]) * np.prod(np.delete(terms, j))
                         for j in range(self.d)])

    def true_accuracy(self, eta) -> float:
        """Direct measured accuracy (for A_k_true / fit diagnostics)."""
        return self._measure(np.asarray(eta, float))
