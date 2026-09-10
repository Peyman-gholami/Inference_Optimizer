"""
stein_smoothing.py
==================
The function the Stein oracle actually sees.

Stein's identity does not estimate A(eta).  It estimates the gradient of the
Gaussian-smoothed accuracy

    A_sigma(eta) = E_z[ A(eta + sigma z) ],     z ~ N(0, I_L)

which is the only well-defined first-order object here: with hard top-k the
kept coordinate set changes only at k = ceil(eta * a), so A is piecewise
constant in eta and has no classical derivative.  For a = 16384 the staircase
step is ~6e-5, so the paper's sigma = 0.05 averages over ~800 discontinuities.

This module measures how far A_sigma sits from A, and what that costs the
controller that maximises it.  Three families of quantity:

  bias      rmse / max / signed mean of (A_sigma - A) over an eta grid
  decision  regret A(eta*_A) - A(eta*_sigma) under the linear price family
            traversed by the No-CSI primal step
  variance  seed-to-seed spread of the Stein gradient at the scenario's own N

The smoothing is Monte-Carlo in the full L-dimensional eta space.  A 1-D
convolution of an axis profile is NOT the same object: smoothing along eta_1
requires A at perturbed eta_2, eta_3 as well.

Cost note
---------
Estimating A_sigma at one point costs `n_draws` accuracy evaluations, so the
whole grid is `n_grid**L * len(sigmas) * n_draws` evaluations.  The variance of
the estimate is  Var_z[A]/M + p(1-p)/(n_eval * M), so at a fixed image budget
many draws on a small evaluation subset beats few draws on the full test set.
Use the backend's fast callable (RESNET_FAST_BATCHES) for A_sigma and the true
callable (full test set) for the A baseline.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

AccuracyFn = Callable[[np.ndarray], float]

# Below this, a spread in accuracy is float round-off rather than signal:
# accuracy is correct/total, so the smallest real step is 1/n_eval >= 1e-5.
_CONST_TOL = 1e-9


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SmoothingConfig:
    """Everything that defines one (task, codec) column block of the table."""

    num_links: int
    sigmas: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2)
    n_grid: int = 6                 # points per eta axis -> n_grid**num_links
    eta_min: float = 0.1            # lower end of the swept box
    eta_max: float = 1.0
    n_draws: int = 200              # MC draws per (eta, sigma) for A_sigma
    clip_lo: float = 0.0            # feasible box the perturbation is clipped to
    clip_hi: float = 1.0
    seed: int = 0

    # gradient-stability probe
    grad_N: int = 20                # antithetic pairs, matches the scenario
    grad_seeds: int = 8
    grad_points: int = 5

    # regret probe
    n_prices: int = 24
    price_lo: float = 1e-3
    price_hi: float = 3.0
    link_sizes: Optional[Sequence[float]] = None   # a_i, sets the price shape

    # Explicit axis values, overriding the linspace. Needed for the codecs that
    # snap eta to a fixed ladder (quantization: 0.0625 INT2, 0.125 INT4,
    # 0.25 INT8, 0.5 FP16, else passthrough) -- a uniform grid lands between
    # rungs, where the codec is a no-op, and measures nothing.
    axis_values: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        self.sigmas = [float(s) for s in self.sigmas]
        if self.link_sizes is not None:
            self.link_sizes = [float(a) for a in self.link_sizes]
        if self.axis_values is not None:
            self.axis_values = [float(v) for v in self.axis_values]
            self.n_grid = len(self.axis_values)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def build_eta_grid(cfg: SmoothingConfig) -> np.ndarray:
    """Full tensor grid over the eta box, shape (n_grid**L, L).

    A tensor grid (rather than a space-filling sample) is deliberate: the
    regret metric needs an argmax over a common candidate set, and a product
    grid makes that argmax reproducible and easy to refine via --n_grid.
    """
    axis = (np.asarray(cfg.axis_values, dtype=float)
            if cfg.axis_values is not None
            else np.linspace(cfg.eta_min, cfg.eta_max, cfg.n_grid))
    return np.array(list(itertools.product(*([axis] * cfg.num_links))), dtype=float)


# ---------------------------------------------------------------------------
# Gaussian smoothing  A_sigma(eta) = E_z[A(eta + sigma z)]
# ---------------------------------------------------------------------------

def smoothed_value(
    acc_fn: AccuracyFn,
    eta: np.ndarray,
    sigma: float,
    n_draws: int,
    rng: np.random.Generator,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
    draws: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Monte-Carlo estimate of A_sigma(eta).

    Returns (mean, standard error).  `draws` lets the caller pass a fixed
    (n_draws, L) matrix of standard normals so that every sigma -- and every
    grid point -- is evaluated on common random numbers; that makes the
    sigma-to-sigma comparison paired and removes most of the MC noise from the
    *differences*, which is what the table reports.
    """
    z = rng.standard_normal((n_draws, eta.size)) if draws is None else draws
    pert = np.clip(eta[None, :] + sigma * z, clip_lo, clip_hi)
    vals = np.fromiter((acc_fn(p) for p in pert), dtype=float, count=pert.shape[0])
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size))


def smooth_grid(
    acc_fn: AccuracyFn,
    grid: np.ndarray,
    cfg: SmoothingConfig,
    progress_every: int = 25,
) -> Dict[float, Dict[str, np.ndarray]]:
    """A_sigma over the whole grid, for every sigma in cfg.sigmas.

    Common random numbers are shared across sigmas *and* grid points.
    """
    rng = np.random.default_rng(cfg.seed)
    draws = rng.standard_normal((cfg.n_draws, cfg.num_links))

    out: Dict[float, Dict[str, np.ndarray]] = {}
    for sigma in cfg.sigmas:
        vals = np.empty(len(grid))
        ses = np.empty(len(grid))
        t0 = time.time()
        for i, eta in enumerate(grid):
            vals[i], ses[i] = smoothed_value(
                acc_fn, eta, sigma, cfg.n_draws, rng,
                cfg.clip_lo, cfg.clip_hi, draws=draws,
            )
            if progress_every and (i + 1) % progress_every == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"    sigma={sigma:<5g} {i+1}/{len(grid)}  "
                      f"({rate:.2f} pts/s)", flush=True)
        out[sigma] = {"value": vals, "se": ses}
        print(f"  sigma={sigma:<5g} done in {time.time()-t0:.1f}s  "
              f"mean|A_s - .|se={ses.mean():.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Stein gradient and its seed-to-seed stability
# ---------------------------------------------------------------------------

def stein_gradient(
    acc_fn: AccuracyFn,
    eta: np.ndarray,
    sigma: float,
    N: int,
    rng: np.random.Generator,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
) -> np.ndarray:
    """Antithetic Stein estimate of grad A_sigma(eta).

    g = 1/(2 N sigma) sum_i z_i [ A(eta + sigma z_i) - A(eta - sigma z_i) ]

    Same estimator as src/core/stein_oracles.gradient_oracle, restated on
    numpy so it can wrap any value-only accuracy callable.
    """
    g = np.zeros_like(eta, dtype=float)
    for _ in range(N):
        z = rng.standard_normal(eta.size)
        plus = np.clip(eta + sigma * z, clip_lo, clip_hi)
        minus = np.clip(eta - sigma * z, clip_lo, clip_hi)
        g += z * (acc_fn(plus) - acc_fn(minus))
    return g / (2.0 * N * sigma)


def gradient_stability(
    acc_fn: AccuracyFn,
    points: np.ndarray,
    sigma: float,
    cfg: SmoothingConfig,
) -> Dict[str, float]:
    """Seed-to-seed spread of the Stein gradient at the scenario's own N.

    rel_sd  = || sd across seeds || / || mean across seeds ||   (norm-wise)
    cos     = mean pairwise cosine between per-seed gradients

    This is the variance half of the sigma trade-off: Var[g] ~ 1/(N sigma^2),
    so it moves opposite to the bias columns.
    """
    rel_sds: List[float] = []
    cosines: List[float] = []

    for eta in points:
        grads = np.array([
            stein_gradient(
                acc_fn, eta, sigma, cfg.grad_N,
                np.random.default_rng(cfg.seed + 1000 * s),
                cfg.clip_lo, cfg.clip_hi,
            )
            for s in range(cfg.grad_seeds)
        ])
        mean_g = grads.mean(axis=0)
        denom = np.linalg.norm(mean_g)
        if denom > 1e-12:
            rel_sds.append(float(np.linalg.norm(grads.std(axis=0, ddof=1)) / denom))

        norms = np.linalg.norm(grads, axis=1)
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                if norms[i] > 1e-12 and norms[j] > 1e-12:
                    cosines.append(float(grads[i] @ grads[j] / (norms[i] * norms[j])))

    return {
        "rel_sd": float(np.mean(rel_sds)) if rel_sds else float("nan"),
        "cos_seed": float(np.mean(cosines)) if cosines else float("nan"),
    }


# ---------------------------------------------------------------------------
# Decision-level cost: regret under the No-CSI price family
# ---------------------------------------------------------------------------

def price_vector(cfg: SmoothingConfig, lam: float) -> np.ndarray:
    """Linear price p on eta at multiplier lam.

    The No-CSI primal step maximises  A(eta) - mu * lam * D(eta, c_hat)  with
    D linear in eta and slope a_i / c_hat_i.  Both Stein scenarios draw every
    link's capacity from one shared range, so c_hat_i is a common scalar and
    the price shape is p ~ a / ||a||_1.  Sweeping lam therefore traverses the
    same family of inner solutions as sweeping (mu, lam, c_hat), which is why
    regret is reported as a sweep rather than at one operating point.
    """
    if cfg.link_sizes is None:
        a = np.ones(cfg.num_links)
    else:
        a = np.asarray(cfg.link_sizes, dtype=float)
    return lam * a / a.sum()


def argmax_under_price(
    values: np.ndarray, grid: np.ndarray, price: np.ndarray
) -> int:
    """Index of the grid point maximising  f(eta) - p . eta."""
    return int(np.argmax(values - grid @ price))


def regret_curve(
    a_ref: np.ndarray,
    a_smooth: np.ndarray,
    grid: np.ndarray,
    cfg: SmoothingConfig,
) -> Dict[str, object]:
    """Cost of letting the controller maximise A_sigma instead of A.

    Regret is measured in the controller's OWN objective  Phi(eta) = A - p.eta,
    not in raw accuracy:

        regret(lam) = Phi(eta*_A) - Phi(eta*_sigma) >= 0

    which is non-negative by construction because eta*_A maximises Phi.  The
    raw accuracy difference A(eta*_A) - A(eta*_sigma) is reported alongside as
    `acc_gap` but is signed: the smoothed choice can buy more accuracy at a
    higher delay price, so on its own it is not a cost.

    Both argmaxes run over the same candidate grid, so the number is quantised
    by cfg.n_grid -- refine the grid to sharpen it.
    """
    lams = np.geomspace(cfg.price_lo, cfg.price_hi, cfg.n_prices)
    regrets = np.empty(len(lams))
    acc_gaps = np.empty(len(lams))
    for i, lam in enumerate(lams):
        p = price_vector(cfg, float(lam))
        obj = a_ref - grid @ p
        i_true = argmax_under_price(a_ref, grid, p)
        i_smooth = argmax_under_price(a_smooth, grid, p)
        regrets[i] = float(obj[i_true] - obj[i_smooth])
        acc_gaps[i] = float(a_ref[i_true] - a_ref[i_smooth])
    return {
        "lambdas": lams.tolist(),
        "regret": regrets.tolist(),
        "mean": float(regrets.mean()),
        "max": float(regrets.max()),
        "acc_gap_mean": float(acc_gaps.mean()),
        "acc_gap_max": float(np.abs(acc_gaps).max()),
        "frac_changed": float(np.mean(regrets > 1e-12)),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(
    a_ref: np.ndarray,
    smoothed: Dict[float, Dict[str, np.ndarray]],
    grid: np.ndarray,
    cfg: SmoothingConfig,
    stability: Optional[Dict[float, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """One dict per sigma with every number the table needs.

    `a_ref` MUST be A measured on the same evaluation subset used to build
    A_sigma.  The backend's fast callable scores a fixed 200-image slice, whose
    accuracy differs from the full test set by a per-eta offset of order
    sqrt(p(1-p)/n) ~ 0.02 -- the same size as the sigma=0.05 bias we are trying
    to resolve, and NOT something more MC draws would average away.  Using a
    common reference cancels it exactly.  Pass the full-test-set A separately
    for reporting accuracy levels.

    `rmse_debiased` removes the Monte-Carlo variance floor: the measured
    mean-square difference is (true bias)^2 + se^2, so subtracting mean(se^2)
    recovers the bias itself rather than bias inflated by finite n_draws.
    """
    out: Dict[str, Dict[str, float]] = {}
    for sigma in cfg.sigmas:
        a_s = smoothed[sigma]["value"]
        se = smoothed[sigma]["se"]
        diff = a_s - a_ref
        mse = float(np.mean(diff ** 2))
        var_mc = float(np.mean(se ** 2))
        reg = regret_curve(a_ref, a_s, grid, cfg)

        # R^2 = 1 - SS_res/SS_tot, to mirror Table 7. Note the normaliser is
        # Var[A] over the grid, so this is meaningless the moment A is close to
        # constant -- which is exactly the llmint8 regime, and what produced
        # Table 7's -105348 and -inf entries. Reported faithfully rather than
        # suppressed: -inf here is the honest reading of "A has no variance".
        ss_tot = float(np.mean((a_ref - a_ref.mean()) ** 2))

        # Accuracy is a ratio of counts, so any real variation is at least
        # 1/n_eval. Anything below _CONST_TOL is float round-off from summing
        # identical values, and must NOT be used as an R^2 normaliser -- doing
        # so divides one rounding error by another and invents a finite-looking
        # number (llmint8 produced R^2 = -7.18 from sd(A) = 1.1e-16 that way).
        def _r2(num: float) -> float:
            if ss_tot <= _CONST_TOL ** 2:
                # A is constant: R^2 is -inf if the model still errs (Table 7's
                # case), and undefined 0/0 if it does not (ours).
                return float("-inf") if num > _CONST_TOL ** 2 else float("nan")
            return 1.0 - num / ss_tot

        r2 = _r2(mse)
        r2_deb = _r2(max(mse - var_mc, 0.0))

        row = {
            "sigma": float(sigma),
            "rmse": float(np.sqrt(mse)),
            "rmse_debiased": float(np.sqrt(max(mse - var_mc, 0.0))),
            "r2": r2,
            "r2_debiased": r2_deb,
            "sd_a": float(np.sqrt(ss_tot)),
            "max_abs_bias": float(np.max(np.abs(diff))),
            "mean_bias": float(diff.mean()),
            "regret_mean": reg["mean"],
            "regret_max": reg["max"],
            "acc_gap_mean": reg["acc_gap_mean"],
            "frac_decision_changed": reg["frac_changed"],
            "mc_se": float(np.mean(se)),
        }
        if stability and sigma in stability:
            row.update(stability[sigma])
        out[f"{sigma:g}"] = row
    return out
