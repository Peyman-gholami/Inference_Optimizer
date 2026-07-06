"""
Multi-task DECOUPLED pretraining with a LIVE, theta-tracking A_k oracle.

Same decoupled scheme as `multitask_decoupled`, but the controller's per-task
accuracy A_k(eta)/grad_A_k(eta) come from a `LiveAccuracyProfile` (measure-and-fit)
that is refit every `refit_every` steps -> A_k tracks the current weights. Used for
the ASYMMETRIC ResNet/CIFAR experiment, where the tasks' accuracy curves differ and
the water-filling allocation genuinely diverges between them.
"""

from __future__ import annotations

from itertools import cycle
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from ..core.task import InferenceTask
from ..optimizers.csi_aware import CSIAwareMultiTaskOptimizer
from .accuracy_profile import LiveAccuracyProfile
from .multitask_decoupled import joint_eta_star, _set_eta


def build_tasks_with_profiles(profiles: List[LiveAccuracyProfile], a_list, eta_min,
                              R_list, w_list, tau: float = 0.05):
    """Build InferenceTasks whose A_k/grad_A_k are backed by each task's live profile.
    `a_list` gives per-task activation sizes (each length d); all tasks span the chain
    of M=d+1 nodes (share all d links)."""
    K = len(profiles)
    d = len(a_list[0])
    M = d + 1
    tasks = []
    for k, prof in enumerate(profiles):
        tasks.append(InferenceTask(
            task_id=k, b_k=0, L_k=M,
            tau={i: float(tau) for i in range(M)},
            a={i: float(a_list[k][i]) for i in range(d)},
            eta_min={i: float(eta_min[i]) for i in range(d)},
            R_k_callable=(lambda t, r=float(R_list[k]): r),
            w_k=float(w_list[k]),
            accuracy_callable=prof.A,
            accuracy_callable_true=prof.true_accuracy,
            gradient_callable=prof.gradA,
        ))
    return tasks, M


def decoupled_pretrain_live(cms, train_loader, opt: CSIAwareMultiTaskOptimizer,
                            profiles: List[LiveAccuracyProfile], *, sampler,
                            steps=1000, lr=1e-4, device=None, refit_every=100,
                            log_every=100):
    """Per step: (refit profiles if due) -> water-fill joint eta*(c) with the live A_k
    -> each task takes an independent compression-aware step at its eta*_k."""
    K = len(cms)
    device = device or next(cms[0].parameters()).device
    for cm in cms:
        cm.to(device).train()
    optims = [torch.optim.Adam(cm.parameters(), lr=lr) for cm in cms]
    data = cycle(train_loader)
    for step in range(1, steps + 1):
        if (step - 1) % refit_every == 0:
            for p in profiles:
                p.refit()
        eta = joint_eta_star(opt, sampler(), K)
        if eta is None:
            continue
        for k, cm in enumerate(cms):
            _set_eta(cm, eta[k], device)
            x, y = next(data)
            x, y = x.to(device), y.to(device)
            optims[k].zero_grad()
            F.cross_entropy(cm(x), y).backward()
            optims[k].step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"[live] step {step:5d}  eta_k={[np.round(e,3).tolist() for e in eta]}  "
                  f"alpha={[np.round(p.alpha,2).tolist() for p in profiles]}  "
                  f"c0={[round(p.c0,3) for p in profiles]}")
    return cms


@torch.no_grad()
def deployed_accuracy_multi_resnet(cms, x_all, y_all, opt: CSIAwareMultiTaskOptimizer,
                                   profiles: List[LiveAccuracyProfile], *, sampler,
                                   n_channels=24, device=None, batch_size=512):
    """Refit profiles to the deployed models, then per channel water-fill the joint
    eta*(c) and measure each task's accuracy. Batched over the test set."""
    K = len(cms)
    device = device or next(cms[0].parameters()).device
    for p in profiles:
        p.refit()
    for cm in cms:
        cm.eval()

    def acc(cm, eta):
        _set_eta(cm, eta, device)
        correct = 0
        for i in range(0, len(x_all), batch_size):
            xb = x_all[i:i + batch_size].to(device)
            yb = y_all[i:i + batch_size].to(device)
            correct += (cm(xb).argmax(1) == yb).sum().item()
        return correct / len(x_all)

    clean = [acc(cms[k], np.ones(cms[k].d)) for k in range(K)]
    per = [[] for _ in range(K)]
    etas = []
    for _ in range(n_channels):
        eta = joint_eta_star(opt, sampler(), K)
        if eta is None:
            continue
        etas.append(eta)
        for k in range(K):
            per[k].append(acc(cms[k], eta[k]))
    per = [np.array(v) for v in per]
    mean_eta = [np.mean([e[k] for e in etas], axis=0).tolist() for k in range(K)] if etas else None
    return {
        "clean_per_task": clean,
        "deployed_mean_per_task": [float(v.mean()) for v in per],
        "deployed_std_per_task": [float(v.std()) for v in per],
        "mean_eta_per_task": mean_eta,
        "fitted_alpha": [p.alpha.tolist() for p in profiles],
    }
