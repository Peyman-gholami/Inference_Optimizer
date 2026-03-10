"""
Demo: All Four Optimizer Types with InferenceTaskCallables

This script demonstrates the four optimizer configurations:
  1. CSI-Aware Single Task   - Perfect channel state (c_t), one task, closed-form optimal
  2. CSI-Aware Multi-Task    - Perfect channel state (c_t), multiple tasks, convex solver
  3. No-CSI Single Task     - Estimated channel (c_hat), one task, dual descent
  4. No-CSI Multi-Task      - Estimated channel (c_hat), multiple tasks, BCD dual descent

All tasks use InferenceTaskCallables (MLP on MNIST with Stein gradient oracle),
as shown in src/core/task_callables.py.

Usage:
  python main.py

Requires: torch, torchvision, numpy, scipy (see requirements.txt)
"""

import os
import sys

# Ensure src/core is on path so task_callables can import accuracy_fn, toy_A, stein_oracles
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Core components
from src.core.task import InferenceTask
from src.core.task_callables import InferenceTaskCallables
from src.optimizers import (
    CSIAwareSingleTaskOptimizer,
    CSIAwareMultiTaskOptimizer,
    NoCSISingleTaskOptimizer,
    NoCSIMultiTaskOptimizer,
)

# ---------------------------------------------------------------------------
# Setup: Train MLP, build InferenceTaskCallables (shared across demos)
# ---------------------------------------------------------------------------

def setup_model_and_callables():
    """
    Trains a small MLP on MNIST and returns (api, flat_theta) for building tasks.
    api.accuracy_callable(eta_np) -> float, api.gradient_callable(eta_np) -> np.ndarray
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Data loaders
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=256,
        shuffle=False,
    )

    # Train MLP (3 epochs)
    from stein_oracles import MLP, get_flat_params
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training MLP (3 epochs)...")
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)

    # 2 cut-points (links) -> 3-node chain topology
    eta_init = torch.ones(2)
    from accuracy_fn import CompressedModel
    cm = CompressedModel(model, eta_init)
    api = InferenceTaskCallables(
        cm,
        test_loader,
        sigma=0.05,
        N=50,
        n_samples=512,
        flat_theta=flat_theta,
        device=device,
    )

    return api, flat_theta


def make_task(task_id: int, api: InferenceTaskCallables, w_k: float = 1.0) -> InferenceTask:
    """
    Builds an InferenceTask from InferenceTaskCallables.

    Topology: 3-node chain (L_k=3) — nodes 0, 1, 2 connected by links 0, 1.
    The eta vector has one entry per link (compression ratio for that link).

    Parameters
    ----------
    task_id : int
        Unique identifier for this task. Used as key in optimizer output dict.
    api : InferenceTaskCallables
        Provides accuracy_callable(eta_np) -> float and gradient_callable(eta_np) -> np.ndarray.
    w_k : float
        Weight in the multi-task objective. Higher w_k = higher priority for accuracy.

    Task Parameters (must match the optimizer's feasibility conditions)
    ----------------------------------------------------------------
    b_k : int
        First node index of this task's path. b_k=0 means the task starts at node 0.
    L_k : int
        Number of nodes in the chain. L_k=3 -> nodes 0,1,2 and links 0,1.
    tau : dict[int, float]
        Compute time (seconds) per node. tau[i] = time to run computation at node i.
        Small values (e.g. 0.01) keep the problem feasible.
    a : dict[int, float]
        Size (bytes or bits) per link. a[i] = data size sent over link i.
        Used in delay constraint: delay on link i >= (a[i] * eta[i]) / c_i.
    eta_min : dict[int, float]
        Minimum compression ratio per link. eta[i] in [eta_min[i], 1.0].
        Lower eta = more compression = less bandwidth but lower accuracy.
    R_k_callable : callable t -> float
        Target throughput R_k(t) packets/s. Max allowed delay = 1/R_k(t).
    """
    return InferenceTask(
        task_id=task_id,
        b_k=0,       # task starts at node 0
        L_k=3,       # 3 nodes (0,1,2) -> 2 links (0,1)
        tau={0: 0.01, 1: 0.01, 2: 0.01},   # compute time per node (small = feasible)
        a={0: 10.0, 1: 10.0},              # data size per link
        eta_min={0: 0.1, 1: 0.1},          # min compression ratio per link (eta in [0.1, 1])
        R_k_callable=lambda t: 10.0,       # target throughput 10 -> max delay 0.1s
        w_k=w_k,                            # weight in multi-task objective
        accuracy_callable=api.accuracy_callable,
        gradient_callable=api.gradient_callable,
    )


# ---------------------------------------------------------------------------
# Demo 1: CSI-Aware Single Task
# ---------------------------------------------------------------------------

def demo_csi_aware_single_task(api):
    """
    CSI-Aware Single Task: Uses perfect channel state information (c_t).
    Closed-form optimal solution (Theorem 3.2). Returns None if infeasible.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: CSI-Aware Single Task Optimizer")
    print("=" * 60)
    print("  Perfect CSI (c_t), one task, closed-form solution")
    print("  eta_i* = min(1, c_i / (R * a_i))")

    task = make_task(task_id=1, api=api)
    optimizer = CSIAwareSingleTaskOptimizer(M=3, tasks=[task])

    # Feasible: high capacity -> optimizer can allocate up to 1.0
    c_t = np.array([100.0, 100.0])
    res = optimizer.optimize(t=0, c_t=c_t)

    if res is None:
        print("  -> Infeasible (channel too weak or compute bottleneck)")
        return

    eta, s_comp, s_comm = res[1]["eta"], res[1]["s_comp"], res[1]["s_comm"]
    acc = task.A_k(eta)
    print(f"  c_t      = {c_t}")
    print(f"  eta*     = {eta}")
    print(f"  s_comp   = {s_comp}")
    print(f"  s_comm   = {s_comm}")
    print(f"  A(eta*)  = {acc:.4f}")


# ---------------------------------------------------------------------------
# Demo 2: CSI-Aware Multi-Task
# ---------------------------------------------------------------------------

def demo_csi_aware_multi_task(api):
    """
    CSI-Aware Multi-Task: Perfect CSI, multiple tasks sharing the network.
    Convex optimization over communication shares (Theorem 3.4).
    """
    print("\n" + "=" * 60)
    print("DEMO 2: CSI-Aware Multi-Task Optimizer")
    print("=" * 60)
    print("  Perfect CSI (c_t), multiple tasks, convex solver (SLSQP)")
    print("  Maximizes sum of weighted accuracies under link capacity constraints")

    task1 = make_task(task_id=1, api=api, w_k=1.0)
    task2 = make_task(task_id=2, api=api, w_k=1.0)
    optimizer = CSIAwareMultiTaskOptimizer(M=3, tasks=[task1, task2])

    # Feasible capacity for two tasks
    c_t = np.array([80.0, 80.0])
    res = optimizer.optimize(t=0, c_t=c_t)

    if res is None:
        print("  -> Infeasible")
        return

    for tid in [1, 2]:
        eta = res[tid]["eta"]
        s_comm = res[tid]["s_comm"]
        acc = task1.A_k(eta) if tid == 1 else task2.A_k(eta)
        print(f"  Task {tid}: eta = {eta}, s_comm = {s_comm}, A = {acc:.4f}")

    # Verify link capacity: sum of s_comm on each link <= 1
    total_link0 = res[1]["s_comm"][0] + res[2]["s_comm"][0]
    total_link1 = res[1]["s_comm"][1] + res[2]["s_comm"][1]
    print(f"  Link utilization: [{total_link0:.4f}, {total_link1:.4f}] (must be <= 1.0)")


# ---------------------------------------------------------------------------
# Demo 3: No-CSI Single Task
# ---------------------------------------------------------------------------

def demo_no_csi_single_task(api):
    """
    No-CSI Single Task: Uses estimated channel (c_hat), not actual c_t.
    Algorithm 1: epigraph-based dual descent. Learns from actual delays via update_dual.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: No-CSI Single Task Optimizer")
    print("=" * 60)
    print("  Estimated channel (c_hat), one task, dual descent")
    print("  update_dual(t, actual_delays) adjusts lambda from real delays")

    task = make_task(task_id=1, api=api)
    optimizer = NoCSISingleTaskOptimizer(M=3, tasks=[task], mu=1.0, epsilon=0.1)

    c_hat = np.array([50.0, 50.0])

    # Empty queue -> maximize accuracy (eta -> 1)
    optimizer.lambda_t = 0.0
    res = optimizer.optimize(t=0, c_hat=c_hat)
    print(f"  [Empty queue] lambda=0: eta = {res[1]['eta']}")

    # Simulate delay violation -> lambda grows
    optimizer.update_dual(t=0, actual_delays={1: 0.5})
    print(f"  After delay 0.5s: lambda = {optimizer.lambda_t:.2f}")

    # Congested queue -> compress to eta_min
    optimizer.lambda_t = 11.0
    res = optimizer.optimize(t=1, c_hat=c_hat)
    print(f"  [Congested] lambda=11: eta = {res[1]['eta']}")


# ---------------------------------------------------------------------------
# Demo 4: No-CSI Multi-Task
# ---------------------------------------------------------------------------

def demo_no_csi_multi_task(api):
    """
    No-CSI Multi-Task: Estimated channel, multiple tasks. Algorithm 2: BCD (Block
    Coordinate Descent) alternating between resource allocation (Phase A) and
    configuration (Phase B). Each task has its own dual queue lambda_k.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: No-CSI Multi-Task Optimizer")
    print("=" * 60)
    print("  Estimated channel (c_hat), multiple tasks, BCD with dual queues")
    print("  Each task has lambda_k; update_dual(t, actual_delays) updates queues")

    task1 = make_task(task_id=1, api=api, w_k=1.0)
    task2 = make_task(task_id=2, api=api, w_k=1.0)
    optimizer = NoCSIMultiTaskOptimizer(M=3, tasks=[task1, task2], mu=1.0, epsilon=0.1, J=10)

    c_hat = np.array([50.0, 50.0])

    # Symmetric queues
    res = optimizer.optimize(t=0, c_hat=c_hat)
    print(f"  [Symmetric queues]")
    for tid in [1, 2]:
        print(f"    Task {tid}: eta = {res[tid]['eta']}, s_comm = {res[tid]['s_comm']}")

    # Asymmetric: Task 1 congested, Task 2 healthy
    optimizer.lambda_k[1] = 50.0
    optimizer.lambda_k[2] = 0.1
    res = optimizer.optimize(t=1, c_hat=c_hat)
    print(f"  [Asymmetric: Task1 congested]")
    for tid in [1, 2]:
        print(f"    Task {tid} (lambda={optimizer.lambda_k[tid]}): eta = {res[tid]['eta']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Inference Optimizer Demo – Four Optimizer Types with InferenceTaskCallables")
    api, _ = setup_model_and_callables()

    demo_csi_aware_single_task(api)
    demo_csi_aware_multi_task(api)
    demo_no_csi_single_task(api)
    demo_no_csi_multi_task(api)

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
