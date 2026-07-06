"""
Multi-task DECOUPLED compression-aware pretraining.

K tasks share bandwidth on a linear chain. Per channel c the MobiHoc
`CSIAwareMultiTaskOptimizer` water-fills the per-link comm shares -> a *joint*
eta*(c) with a per-task eta*_k. Each task's model is then updated by INDEPENDENT
compression-aware SGD at its own eta*_k -- no cross-task coupling gradient (the
decoupled approximation; the coupling J = d eta*/d theta is dropped).

For SYMMETRIC tasks the water-filling splits each shared link equally, so
    eta*_{k,i}(c) = min(1, c_i / (K R a_i))
i.e. exactly K single-task problems at the contention-reduced capacity c/K
(verified against the optimizer). The controller is kept in the loop so the same
code generalizes to the asymmetric case -- where A_k(theta) genuinely matters and
should be re-estimated from the live model rather than frozen.
"""

from __future__ import annotations

from itertools import cycle
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.task import InferenceTask
from ..core.accuracy_fn import CompressedModel
from ..optimizers.csi_aware import CSIAwareMultiTaskOptimizer


# ---------------------------------------------------------------------------
# Concave accuracy surrogate A_k(eta) = prod_i (1 - alpha_i (1-eta_i)^2)
# (drives the controller's water-filling; for symmetric tasks the allocation is
#  equal-split regardless of its shape)
# ---------------------------------------------------------------------------
def make_surrogate_A(alpha):
    alpha = np.asarray(alpha, float)

    def A(eta):
        eta = np.asarray(eta, float)
        return float(np.prod(1 - alpha * (1 - eta) ** 2))

    def gradA(eta):
        eta = np.asarray(eta, float)
        terms = 1 - alpha * (1 - eta) ** 2
        return np.array([2 * alpha[j] * (1 - eta[j]) * np.prod(np.delete(terms, j))
                         for j in range(len(eta))])

    return A, gradA


def build_symmetric_tasks(K: int, a, eta_min, R: float = 1.0, w: float = 1.0,
                          alpha: float = 0.9, tau: float = 0.05):
    """K identical tasks, each spanning the whole chain of M = d+1 nodes (so they
    share all d links). Returns (tasks, M)."""
    d = len(a)
    M = d + 1
    a_dict = {i: float(a[i]) for i in range(d)}
    eta_min_dict = {i: float(eta_min[i]) for i in range(d)}
    tau_dict = {i: float(tau) for i in range(M)}
    A, gA = make_surrogate_A([alpha] * d)
    Rc = lambda t: R
    tasks = [InferenceTask(task_id=k, b_k=0, L_k=M, tau=tau_dict, a=a_dict,
                           eta_min=eta_min_dict, R_k_callable=Rc, w_k=w,
                           accuracy_callable=A, accuracy_callable_true=A,
                           gradient_callable=gA)
             for k in range(K)]
    return tasks, M


def joint_eta_star(opt: CSIAwareMultiTaskOptimizer, c: np.ndarray, K: int):
    """Run the controller; return list[k] -> eta vector (length d), or None if
    the slot is infeasible."""
    res = opt.optimize(0, np.asarray(c, float))
    if res is None:
        return None
    return [np.asarray(res[k]['eta'], float) for k in range(K)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _set_eta(cm, eta, device):
    cm.set_eta(torch.as_tensor(eta, dtype=torch.float32, device=device))


def decoupled_pretrain(cms: List[CompressedModel], train_loader: DataLoader,
                       opt: CSIAwareMultiTaskOptimizer, *, sampler, steps=1500,
                       lr=1e-3, device=None, log_every=300) -> List[CompressedModel]:
    """Per step: sample c -> joint eta*(c) (water-filled) -> each task takes an
    independent compression-aware SGD step at its eta*_k."""
    K = len(cms)
    device = device or next(cms[0].parameters()).device
    for cm in cms:
        cm.to(device).train()
    optims = [torch.optim.Adam(cm.parameters(), lr=lr) for cm in cms]
    data = cycle(train_loader)

    for step in range(1, steps + 1):
        eta = joint_eta_star(opt, sampler(), K)
        if eta is None:               # infeasible channel slot -> skip
            continue
        for k, cm in enumerate(cms):
            _set_eta(cm, eta[k], device)
            x, y = next(data)
            x, y = x.to(device), y.to(device)
            optims[k].zero_grad()
            loss = F.cross_entropy(cm(x), y)
            loss.backward()
            optims[k].step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"[decoupled] step {step:5d}  eta_k={[np.round(e,3).tolist() for e in eta]}")
    return cms


def standard_pretrain_multi(cms: List[CompressedModel], train_loader: DataLoader, *,
                            steps=1500, lr=1e-3, device=None, log_every=300):
    """Baseline: each task trained at eta = 1 (no compression)."""
    device = device or next(cms[0].parameters()).device
    for cm in cms:
        cm.to(device).train()
        _set_eta(cm, np.ones(cm.d), device)
    optims = [torch.optim.Adam(cm.parameters(), lr=lr) for cm in cms]
    data = cycle(train_loader)
    for step in range(1, steps + 1):
        for k, cm in enumerate(cms):
            x, y = next(data)
            x, y = x.to(device), y.to(device)
            optims[k].zero_grad()
            loss = F.cross_entropy(cm(x), y)
            loss.backward()
            optims[k].step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"[standard]  step {step:5d}")
    return cms


# ---------------------------------------------------------------------------
# Evaluation under the shared-bandwidth deployment distribution
# ---------------------------------------------------------------------------
@torch.no_grad()
def deployed_accuracy_multi(cms: List[CompressedModel], test_loader: DataLoader,
                            opt: CSIAwareMultiTaskOptimizer, *, sampler,
                            n_channels=64, device=None) -> dict:
    """E_{c~P_c}[ per-task accuracy at the joint eta*(c) ] + clean (eta=1) per task."""
    K = len(cms)
    device = device or next(cms[0].parameters()).device
    xs, ys = zip(*list(test_loader))
    x = torch.cat(xs).to(device)
    y = torch.cat(ys).to(device)
    for cm in cms:
        cm.eval()

    def acc(cm, eta):
        _set_eta(cm, eta, device)
        return (cm(x).argmax(1) == y).float().mean().item()

    clean = [acc(cms[k], np.ones(cms[k].d)) for k in range(K)]
    per_task = [[] for _ in range(K)]
    for _ in range(n_channels):
        eta = joint_eta_star(opt, sampler(), K)
        if eta is None:
            continue
        for k in range(K):
            per_task[k].append(acc(cms[k], eta[k]))
    per_task = [np.array(v) for v in per_task]
    return {
        "clean_per_task": clean,
        "deployed_mean_per_task": [float(v.mean()) for v in per_task],
        "deployed_std_per_task": [float(v.std()) for v in per_task],
        "deployed_mean_agg": float(np.mean([v.mean() for v in per_task])),
        "n_channels": n_channels,
    }
