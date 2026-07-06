"""
Compression-aware pretraining --- CSI-aware single task (the J_{eta*} = 0 case).

Background
----------
At deployment the controller chooses, for a channel realization c, the closed-form
optimal compression of a single CSI-aware task,

    eta_i*(c) = clip( min(1, c_i / (R a_i)),  eta_i^min,  1 ),

i.e. exactly `CSIAwareSingleTaskOptimizer.optimize`.  This is a function of the
channel only --- it does NOT depend on the weights theta --- so the inner Jacobian
J_{eta*} = d eta* / d theta is zero and the hypergradient collapses to

    grad_theta F(theta) = E_{c ~ P_c} E_{(x,y)} [ grad_theta loss( f_theta(x; eta*(c)), y ) ].

That is ordinary SGD in which every minibatch is pushed through the compressed
forward map at a freshly channel-sampled eta*(c).  No accuracy surrogate and no
Stein oracle are needed for *training* (they are still used at deployment).

Implementation note
-------------------
We reuse `CompressedModel` (top-k hooks).  The hook multiplies each activation by a
top-k mask, so it is differentiable in theta (gradient flows through the kept
entries); only eta is detached.  Unlike `AccuracyFunction` (which runs under
`no_grad` and returns argmax accuracy), here we run a normal forward + cross-entropy
+ backward, which is all the J=0 case requires.
"""

from __future__ import annotations

from itertools import cycle
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.accuracy_fn import CompressedModel


# ---------------------------------------------------------------------------
# Channel distribution  P_c
# ---------------------------------------------------------------------------
class ChannelSampler:
    """Samples per-link physical capacities c = (c_1, ..., c_d) i.i.d. on [c_min, c_max].

    kind="loguniform" (default, aggressive: more mass at low capacity) or "uniform"
    (milder). The capacities are physical and task-independent -- each task normalizes
    them itself via c_i/(R_k a_{k,i}) -- so with different R_k / a_{k,i} the induced
    compression differs across tasks/links (no scale-free reduction as in single-task).
    """

    def __init__(self, d: int, c_min, c_max, seed: Optional[int] = 0,
                 kind: str = "loguniform"):
        assert kind in ("loguniform", "uniform")
        self.d = d
        self.kind = kind
        self.c_min = np.broadcast_to(np.asarray(c_min, dtype=float), (d,)).copy()
        self.c_max = np.broadcast_to(np.asarray(c_max, dtype=float), (d,)).copy()
        assert np.all(self.c_min > 0) and np.all(self.c_max >= self.c_min)
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        u = self.rng.uniform(0.0, 1.0, size=self.d)
        if self.kind == "uniform":
            return self.c_min + u * (self.c_max - self.c_min)
        log_c = np.log(self.c_min) + u * (np.log(self.c_max) - np.log(self.c_min))
        return np.exp(log_c)


class UniformEtaChannelSampler:
    """Channel engineered so the induced compression eta* is *uniform* on
    [eta_lo, eta_hi] per cut-point -- a milder, less heavy-end-biased prior than
    the log-uniform-on-capacity `ChannelSampler`.

    It returns capacities c_i = eta_i * R * a_i, so the unchanged controller
    csi_aware_eta_star(c) = clip(min(1, c/(R a)), eta_min, 1) recovers eta_i.
    (Requires eta_lo >= eta_min so the floor never binds.)
    """

    def __init__(self, d: int, eta_lo: float, eta_hi: float, R: float, a,
                 seed: Optional[int] = 0):
        self.d = d
        self.eta_lo = float(eta_lo)
        self.eta_hi = float(eta_hi)
        self.R = float(R)
        self.a = np.broadcast_to(np.asarray(a, dtype=float), (d,)).copy()
        assert 0.0 < self.eta_lo <= self.eta_hi <= 1.0
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> np.ndarray:
        eta = self.rng.uniform(self.eta_lo, self.eta_hi, size=self.d)
        return eta * self.R * self.a


# ---------------------------------------------------------------------------
# CSI-aware closed-form selection  eta*(c)   (controller, single task)
# ---------------------------------------------------------------------------
def csi_aware_eta_star(c: np.ndarray, a: np.ndarray, R: float,
                       eta_min: np.ndarray) -> np.ndarray:
    """eta_i* = clip( min(1, c_i / (R a_i)), eta_i^min, 1 ).  Channel-only (J=0)."""
    c = np.asarray(c, dtype=float)
    a = np.asarray(a, dtype=float)
    eta_min = np.asarray(eta_min, dtype=float)
    eta = np.minimum(1.0, c / (R * a))
    return np.clip(eta, eta_min, 1.0)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def _set_eta(cm: CompressedModel, eta: np.ndarray, device) -> None:
    cm.set_eta(torch.as_tensor(eta, dtype=torch.float32, device=device))


def compression_aware_pretrain(
    cm: CompressedModel,
    train_loader: DataLoader,
    *,
    a: np.ndarray,
    R: float,
    eta_min: np.ndarray,
    sampler: ChannelSampler,
    steps: int = 2000,
    lr: float = 1e-3,
    device=None,
    log_every: int = 200,
) -> CompressedModel:
    """Algorithm 1: SGD with a freshly channel-sampled eta*(c) per minibatch."""
    device = device or next(cm.parameters()).device
    cm.to(device).train()
    opt = torch.optim.Adam(cm.parameters(), lr=lr)
    data = cycle(train_loader)

    for step in range(1, steps + 1):
        c = sampler()
        eta = csi_aware_eta_star(c, a, R, eta_min)
        _set_eta(cm, eta, device)

        x, y = next(data)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(cm(x), y)
        loss.backward()
        opt.step()

        if log_every and (step % log_every == 0 or step == 1):
            print(f"[compression-aware] step {step:5d}  "
                  f"eta={np.round(eta, 3)}  loss={loss.item():.4f}")
    return cm


def standard_pretrain(
    cm: CompressedModel,
    train_loader: DataLoader,
    *,
    steps: int = 2000,
    lr: float = 1e-3,
    device=None,
    log_every: int = 200,
) -> CompressedModel:
    """Baseline: ordinary training with no compression (eta = 1)."""
    device = device or next(cm.parameters()).device
    cm.to(device).train()
    _set_eta(cm, np.ones(cm.d), device)
    opt = torch.optim.Adam(cm.parameters(), lr=lr)
    data = cycle(train_loader)

    for step in range(1, steps + 1):
        x, y = next(data)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(cm(x), y)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"[standard]          step {step:5d}  loss={loss.item():.4f}")
    return cm


# ---------------------------------------------------------------------------
# Evaluation under the deployment channel distribution
# ---------------------------------------------------------------------------
@torch.no_grad()
def _accuracy_at_eta(cm: CompressedModel, x, y, eta: np.ndarray, device) -> float:
    _set_eta(cm, eta, device)
    cm.eval()
    return (cm(x).argmax(1) == y).float().mean().item()


@torch.no_grad()
def deployed_accuracy(
    cm: CompressedModel,
    test_loader: DataLoader,
    *,
    a: np.ndarray,
    R: float,
    eta_min: np.ndarray,
    sampler: ChannelSampler,
    n_channels: int = 64,
    device=None,
) -> dict:
    """Monte-Carlo estimate of E_{c~P_c}[ accuracy at eta*(c) ], plus the clean
    (eta = 1) accuracy, on the full test set."""
    device = device or next(cm.parameters()).device
    xs, ys = zip(*list(test_loader))
    x = torch.cat(xs).to(device)
    y = torch.cat(ys).to(device)

    clean = _accuracy_at_eta(cm, x, y, np.ones(cm.d), device)

    accs = []
    for _ in range(n_channels):
        eta = csi_aware_eta_star(sampler(), a, R, eta_min)
        accs.append(_accuracy_at_eta(cm, x, y, eta, device))
    accs = np.array(accs)
    return {
        "clean_acc": clean,
        "deployed_acc_mean": float(accs.mean()),
        "deployed_acc_std": float(accs.std()),
        "n_channels": n_channels,
    }
