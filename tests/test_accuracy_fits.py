"""Tests for the A_hat surrogate families (src/pretraining/accuracy_fits.py).

Covers the four things worth asserting:
  1. recovery      -- can each family recover parameters it generated itself?
  2. concavity     -- which families are concave BY CONSTRUCTION vs conditionally?
  3. monotonicity  -- all families must be nondecreasing in every eta_i
  4. the pathology -- a CHARACTERISATION test: on an inverted curve the current
                      two-stage fit collapses to a flat line. If a change fixes
                      that, this test fails loudly and we update it on purpose.
"""

import numpy as np
import pytest

from src.pretraining.accuracy_fits import (
    ProductTwoStage, ProductJoint, SeparableConcave, ConcaveNet, build_fit, ALL_FITS,
)

ETA_MIN = 0.1
D = 2


def grid(n=7, eta_min=ETA_MIN, d=D):
    ax = np.linspace(eta_min, 1.0, n)
    return np.array(np.meshgrid(*([ax] * d))).reshape(d, -1).T


def product_A(etas, c0, alpha):
    return c0 * np.prod(1.0 - np.asarray(alpha) * (1.0 - etas) ** 2, axis=1)


def separable_A(etas, c0, alpha):
    return c0 * (1.0 - ((1.0 - etas) ** 2 * np.asarray(alpha)).sum(axis=1))


# ---------------------------------------------------------------- recovery

def test_product_joint_recovers_its_own_parameters():
    etas = grid(9)
    c0, alpha = 0.93, np.array([0.45, 0.12])
    f = ProductJoint(D, ETA_MIN).fit(etas, product_A(etas, c0, alpha))
    assert f.c0 == pytest.approx(c0, abs=1e-6)
    assert f.alpha == pytest.approx(alpha, abs=1e-6)


def test_separable_recovers_its_own_parameters():
    etas = grid(9)
    c0, alpha = 0.90, np.array([0.30, 0.20])
    f = SeparableConcave(D, ETA_MIN).fit(etas, separable_A(etas, c0, alpha))
    assert f.c0 == pytest.approx(c0, abs=1e-6)
    assert f.alpha == pytest.approx(alpha, abs=1e-6)


def test_joint_beats_two_stage_on_a_monotone_curve():
    """The joint fit can never be worse -- it optimises the same objective over a
    superset (c0 is free rather than pinned to the eta=1 measurement)."""
    etas = grid(9)
    y = product_A(etas, 0.93, [0.45, 0.12]) + 0.01 * np.sin(17 * etas.sum(axis=1))
    ts = ProductTwoStage(D, ETA_MIN).fit(etas, y)
    jt = ProductJoint(D, ETA_MIN).fit(etas, y)
    assert jt.rmse(etas, y) <= ts.rmse(etas, y) + 1e-9


# --------------------------------------------------------------- concavity

def test_separable_is_concave_unconditionally():
    """Cross-partials are exactly zero, so no alpha can break concavity."""
    etas = grid(7)
    rng = np.random.default_rng(0)
    for _ in range(8):
        f = SeparableConcave(D, ETA_MIN)
        f.c0 = 1.0
        f.alpha = rng.uniform(0, 1.0 / (D * (1 - ETA_MIN) ** 2), size=D)
        worst, _ = f.max_hess_eig(n=7)
        assert worst <= 1e-9, f"separable not concave at alpha={f.alpha}"


def test_product_family_is_only_conditionally_concave():
    """The documented failure: alpha=[0.6,0.6] is indefinite near the eta floor,
    while the asymmetric fits seen in practice are safe."""
    unsafe = ProductJoint(D, ETA_MIN)
    unsafe.c0, unsafe.alpha = 1.0, np.array([0.6, 0.6])
    worst, where = unsafe.max_hess_eig(n=9)
    assert worst > 0, "expected the product family to be indefinite at alpha=0.6"
    assert where == pytest.approx((ETA_MIN, ETA_MIN), abs=1e-6)

    safe = ProductJoint(D, ETA_MIN)
    safe.c0, safe.alpha = 1.0, np.array([0.548, 0.106])   # a real measured fit
    assert safe.max_hess_eig(n=9)[0] <= 1e-9


def test_concavity_guard_keeps_the_product_fit_concave():
    """With the guard, alpha is capped so the d=2 condition holds at the floor."""
    etas = grid(9)
    y = product_A(etas, 1.0, [0.9, 0.9])          # data that wants a large alpha
    g = build_fit("joint_guarded", D, ETA_MIN).fit(etas, y)
    assert g.max_hess_eig(n=9)[0] <= 1e-9
    assert np.all(g.alpha <= (1.0 / 3.0) / (1 - ETA_MIN) ** 2 + 1e-9)


def test_concave_net_is_concave_and_the_icnn_shape_is_not():
    """Regression guard for the inherited bug: softplus + output sigmoid is NOT
    concave (softplus is convex -> ICNN -> convex; sigmoid of convex is
    neither). Our activation -softplus(-x) is concave and nondecreasing."""
    import torch
    import torch.nn.functional as Fn
    from src.pretraining.accuracy_fits import _ConcaveMonotoneNet

    torch.manual_seed(0)
    net = _ConcaveMonotoneNet(D, hidden=8, depth=2)
    worst = -np.inf
    for eta in grid(7):
        H = torch.autograd.functional.hessian(
            net, torch.as_tensor(eta, dtype=torch.float64))
        H = H.detach().numpy()
        worst = max(worst, np.linalg.eigvalsh(0.5 * (H + H.T)).max())
    assert worst <= 1e-9, f"concave net has positive curvature {worst}"

    # and the activation really is concave & nondecreasing
    x = torch.linspace(-4, 4, 41, dtype=torch.float64)
    s = _ConcaveMonotoneNet._sigma(x)
    assert torch.all(s[1:] - s[:-1] >= -1e-12)                       # nondecreasing
    assert torch.all(s[2:] - 2 * s[1:-1] + s[:-2] <= 1e-9)           # concave


# ------------------------------------------------------------- monotonicity

@pytest.mark.parametrize("name", ["two_stage", "joint", "separable"])
def test_all_analytic_families_are_monotone(name):
    etas = grid(7)
    y = product_A(etas, 0.9, [0.4, 0.15])
    assert build_fit(name, D, ETA_MIN).fit(etas, y).is_monotone(n=7)


# ----------------------------------------------------- the known pathology

def _inverted_curve(etas):
    """Accuracy that PEAKS at low eta and falls at eta=1 -- the starved r20
    shape (true peak .872 at eta1=.17, falling to .673 at eta1=1)."""
    e1 = etas[:, 0]
    return 0.872 - 0.20 * (e1 - 0.17) ** 2 / (1.0 - 0.17) ** 2 * 1.0 - 0.0 * etas[:, 1]


def test_two_stage_collapses_on_an_inverted_curve():
    """CHARACTERISATION. c0 is measured at eta=1 -- the worst point -- so every
    other measurement lands above it, 1-acc/c0 < 0, and alpha clips to 0."""
    etas = grid(9)
    y = _inverted_curve(etas)
    f = ProductTwoStage(D, ETA_MIN).fit(etas, y)
    assert f.alpha == pytest.approx(np.zeros(D), abs=1e-12), "expected alpha->0"
    vals = f.A_many(etas)
    assert vals.std() == pytest.approx(0.0, abs=1e-12), "expected a flat A_hat"


def test_joint_fit_improves_the_level_but_stays_flat():
    """The joint fit lifts c0 off the eta=1 value but cannot fix the SHAPE --
    a concave-increasing family still cannot represent a low-eta peak."""
    etas = grid(9)
    y = _inverted_curve(etas)
    ts = ProductTwoStage(D, ETA_MIN).fit(etas, y)
    jt = ProductJoint(D, ETA_MIN).fit(etas, y)
    assert jt.rmse(etas, y) < ts.rmse(etas, y)      # strictly better
    assert jt.c0 > ts.c0                            # by raising the level
    # still flat: assert the FUNCTION is constant to within 1 accuracy point,
    # rather than alpha being bitwise 0 (least_squares leaves it at ~1e-5)
    assert jt.A_many(etas).std() < 0.01
    assert np.all(jt.alpha < 1e-3)


def test_every_registered_fit_runs_end_to_end():
    etas = grid(5)
    y = product_A(etas, 0.9, [0.4, 0.15])
    for name in ALL_FITS:
        f = build_fit(name, D, ETA_MIN)
        if isinstance(f, ConcaveNet):
            f.epochs = 150                           # keep the test fast
        f.fit(etas, y)
        assert np.isfinite(f.A(np.ones(D)))
        assert np.all(np.isfinite(f.gradA(np.full(D, 0.5))))
        assert np.all(np.isfinite(f.hess(np.full(D, 0.5))))
        assert f.describe()
