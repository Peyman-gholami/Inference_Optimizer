"""
Demo: Optimizers and Baselines with InferenceTaskCallables.

This script demonstrates:
  1. CSI-Aware Single Task optimizer
  2. CSI-Aware Multi-Task optimizer
  3. No-CSI Single Task optimizer
  4. No-CSI Multi-Task optimizer
  5. Single-task baselines used by compare_to_baselines.py
  6. Multi-task baselines used by compare_to_baselines_multi.py

By default, tasks use InferenceTaskCallables (MLP on MNIST with Stein gradient oracle),
By default, tasks use InferenceTaskCallables (MLP on MNIST with Stein gradient oracle),
as shown in src/core/task_callables.py.

Optional LLM + MMLU backend: set INFERENCE_OPTIMIZER_BACKEND=llm (see README_LLM.md).

Optional LLM + MMLU backend: set INFERENCE_OPTIMIZER_BACKEND=llm (see README_LLM.md).

Usage:
  python main.py

Requires: torch, torchvision, numpy, scipy (see requirements.txt)
"""

import os
import sys
from typing import Any
from typing import Any

# Ensure src/core is on path so task_callables can import accuracy_fn, toy_A, stein_oracles
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

import numpy as np

# Core components
from src.core.task import InferenceTask
from src.core.task_callables import InferenceTaskCallables
from src.core.llm_plug_in import is_llm_backend, maybe_measured_tau_a
from src.core.task_handler import get_task
from src.optimizers import (
    CSIAwareSingleTaskOptimizer,
    CSIAwareMultiTaskOptimizer,
    NoCSISingleTaskOptimizer,
    NoCSIMultiTaskOptimizer,
)
from src.optimizers.baseline import (
    MaxCompressionSingleTaskBaseline,
    NoCompressionSingleTaskBaseline,
    UniformCompressionSingleTaskBaseline,
    MyopicSingleTaskBaseline,
    ConservativeSingleTaskBaseline,
    MovingAverageSingleTaskBaseline,
    LCBSingleTaskBaseline,
    MaxCompressionMultiTaskBaseline,
    NoCompressionMultiTaskBaseline,
    StaticEqualShareMultiTaskBaseline,
    ProportionalResourceAllocationMultiTaskBaseline,
    StrictPriorityGreedyMultiTaskBaseline,
    DecoupledEqualSplitStochasticDescentMultiTaskBaseline,
    QueueProportionalHeuristicMultiTaskBaseline,
    HistoricalAverageCertaintyEquivalenceMultiTaskBaseline,
    LCBCertaintyEquivalenceMultiTaskBaseline,
)
from src.optimizers.estimators import (
    LastObservationEstimator,
    RunningMinEstimator,
    MovingAverageEstimator,
    MeanEstimator,
)

# ---------------------------------------------------------------------------
# Setup: Train MLP, build InferenceTaskCallables (shared across demos)
# ---------------------------------------------------------------------------

def setup_model_and_callables():
    """
    Returns (api, flat_theta) for building tasks.

    Default: train a small MLP on MNIST (api implements accuracy/gradient callables).

    LLM mode: set INFERENCE_OPTIMIZER_BACKEND=llm — uses MMLU + activation compression
    (see src/core/llm_task_callables.py). flat_theta is None in that case.
    """
    if is_llm_backend():
        from src.core.llm_task_callables import build_llm_backend_from_env

        eta_min0 = float(os.environ.get("ETA_MIN0", "0.25"))
        eta_min1 = float(os.environ.get("ETA_MIN1", "0.25"))
        eta_min = min(eta_min0, eta_min1)
        api = build_llm_backend_from_env(eta_min=eta_min)
        return api, None

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
        flat_theta=None,   # model already has trained weights — no swap needed (saves time)
        device=device,
    )

    return api, flat_theta


def make_task(task_id: int, api: Any, w_k: float = 1.0) -> InferenceTask:
    """
    Builds an InferenceTask from InferenceTaskCallables.

    Topology: 3-node chain (L_k=3) — nodes 0, 1, 2 connected by links 0, 1.
    The eta vector has one entry per link (compression ratio for that link).

    Parameters
    ----------
    task_id : int
        Unique identifier for this task. Used as key in optimizer output dict.
    api : InferenceTaskCallables | LLMInferenceTaskCallables
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

    Environment (optional)
    ----------------------
    ETA_MIN0, ETA_MIN1 — per-link eta floors (default 0.1 / 0.05 for MNIST;
    0.25 / 0.25 for LLM backend when vars are unset).
    R_K or MEASURED_R_K — constant target throughput for toy/measured setups.
    With LLM_USE_MEASURED_TAU_A=1, tau and a come from measured_llm_profile tables
    (see README_LLM.md).
    """
    if is_llm_backend():
        eta_min0 = float(os.environ.get("ETA_MIN0", "0.25"))
        eta_min1 = float(os.environ.get("ETA_MIN1", "0.25"))
    else:
        eta_min0 = float(os.environ.get("ETA_MIN0", "0.1"))
        eta_min1 = float(os.environ.get("ETA_MIN1", "0.05"))

    measured = maybe_measured_tau_a()
    if measured is not None:
        tau, a = measured
    else:
        tau = {0: 0.01, 1: 0.01, 2: 0.01}
        a = {0: 10.0, 1: 20.0}

    r_k = float(os.environ.get("R_K", os.environ.get("MEASURED_R_K", "10.0")))

    return InferenceTask(
        task_id=task_id,
        b_k=0,       # task starts at node 0
        L_k=3,       # 3 nodes (0,1,2) -> 2 links (0,1)
        tau=tau,
        a=a,
        eta_min={0: eta_min0, 1: eta_min1},
        R_k_callable=lambda t, r=r_k: r,
        w_k=w_k,                            # weight in multi-task objective
        accuracy_callable=api.accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=api.gradient_callable,
    )

    MeanMinusZStdLCB,
)

DEFAULT_TASK_NAME = "toy_mlp_mnist"

# ---------------------------------------------------------------------------
# Demo 1: CSI-Aware Single Task
# ---------------------------------------------------------------------------

def demo_csi_aware_single_task():
    """
    CSI-Aware Single Task: Uses perfect channel state information (c_t).
    Closed-form optimal solution (Theorem 3.2). Returns None if infeasible.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: CSI-Aware Single Task Optimizer")
    print("=" * 60)
    print("  Perfect CSI (c_t), one task, closed-form solution")
    print("  eta_i* = min(1, c_i / (R * a_i))")

    task = get_task(DEFAULT_TASK_NAME, task_id=1)
    optimizer = CSIAwareSingleTaskOptimizer(M=3, tasks=[task])

    # Feasible: high capacity -> optimizer can allocate up to 1.0
    c_t = np.array([30.0, 100.0])
    res = optimizer.optimize(t=0, c_t=c_t)

    if res is None:
        print("  -> Infeasible (channel too weak or compute bottleneck)")
        return

    eta, s_comp, s_comm = res[1]["eta"], res[1]["s_comp"], res[1]["s_comm"]
    acc = task.A_k_true(eta)
    print(f"  c_t      = {c_t}")
    print(f"  eta*     = {eta}")
    print(f"  s_comp   = {s_comp}")
    print(f"  s_comm   = {s_comm}")
    print(f"  A(eta*)  = {acc:.4f}")


# ---------------------------------------------------------------------------
# Demo 2: CSI-Aware Multi-Task
# ---------------------------------------------------------------------------

def demo_csi_aware_multi_task():
    """
    CSI-Aware Multi-Task: Perfect CSI, multiple tasks sharing the network.
    Convex optimization over communication shares (Theorem 3.4).
    """
    print("\n" + "=" * 60)
    print("DEMO 2: CSI-Aware Multi-Task Optimizer")
    print("=" * 60)
    print("  Perfect CSI (c_t), multiple tasks, convex solver (SLSQP)")
    print("  Maximizes sum of weighted accuracies under link capacity constraints")

    task1 = get_task(DEFAULT_TASK_NAME, task_id=1, w_k=1.0)
    task2 = get_task(DEFAULT_TASK_NAME, task_id=2, w_k=1.0)
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
        s_comp = res[tid]["s_comp"]
        acc = task1.A_k_true(eta) if tid == 1 else task2.A_k_true(eta)
        print(f"  Task {tid}: eta = {eta}, s_comm = {s_comm}, s_comp = {s_comp}, A = {acc:.4f}")

    # Verify link capacity: sum of s_comm on each link <= 1
    total_link0 = res[1]["s_comm"][0] + res[2]["s_comm"][0]
    total_link1 = res[1]["s_comm"][1] + res[2]["s_comm"][1]
    print(f"  Link utilization: [{total_link0:.4f}, {total_link1:.4f}] (must be <= 1.0)")


# ---------------------------------------------------------------------------
# Demo 3: No-CSI Single Task
# ---------------------------------------------------------------------------

def demo_no_csi_single_task():
    """
    No-CSI Single Task: Uses estimated channel (c_hat), not actual c_t.
    Algorithm 1: epigraph-based dual descent. Learns from actual delays via update_dual.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: No-CSI Single Task Optimizer")
    print("=" * 60)
    print("  Estimated channel (c_hat), one task, dual descent")
    print("  update_dual(t, actual_delays) adjusts lambda from real delays")

    task = get_task(DEFAULT_TASK_NAME, task_id=1)
    optimizer = NoCSISingleTaskOptimizer(M=3, tasks=[task], mu=1.0, epsilon=0.1)

    c_hat = np.array([50.0, 50.0])

    # Empty queue -> maximize accuracy (eta -> 1)
    optimizer.lambda_t = 0.0
    res = optimizer.optimize(t=0, c_hat=c_hat)
    print(f"  [Empty queue] lambda=0:")
    print(f"    eta = {res[1]['eta']}, s_comm = {res[1]['s_comm']}, s_comp = {res[1]['s_comp']}")

    # Simulate delay violation -> lambda grows
    optimizer.update_dual(t=0, actual_delays={1: 0.5})
    print(f"  After delay 0.5s: lambda = {optimizer.lambda_t:.2f}")

    # Congested queue -> compress to eta_min
    optimizer.lambda_t = 11.0
    res = optimizer.optimize(t=1, c_hat=c_hat)
    print(f"  [Congested] lambda=11:")
    print(f"    eta = {res[1]['eta']}, s_comm = {res[1]['s_comm']}, s_comp = {res[1]['s_comp']}")


# ---------------------------------------------------------------------------
# Demo 4: No-CSI Multi-Task
# ---------------------------------------------------------------------------

def demo_no_csi_multi_task():
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

    task1 = get_task(DEFAULT_TASK_NAME, task_id=1, w_k=1.0)
    task2 = get_task(DEFAULT_TASK_NAME, task_id=2, w_k=1.0)
    optimizer = NoCSIMultiTaskOptimizer(M=3, tasks=[task1, task2], mu=1.0, epsilon=0.1, J=10)

    c_hat = np.array([50.0, 50.0])

    # Symmetric queues
    res = optimizer.optimize(t=0, c_hat=c_hat)
    print(f"  [Symmetric queues]")
    for tid in [1, 2]:
        print(f"    Task {tid}: eta = {res[tid]['eta']}, s_comm = {res[tid]['s_comm']}, s_comp = {res[tid]['s_comp']}")

    # Dual update: simulate actual delays (Task 1 violated, Task 2 under target)
    # Target delay = 1/R = 0.1s. lambda_k grows if D_act > 0.1, shrinks if D_act < 0.1
    optimizer.update_dual(t=0, actual_delays={1: 0.5, 2: 0.05})
    print(f"  After update_dual(t=0, {{1: 0.5, 2: 0.05}}):")
    for tid in [1, 2]:
        print(f"    Task {tid}: lambda_k = {optimizer.lambda_k[tid]:.2f}")

    # Asymmetric: Task 1 congested (high lambda), Task 2 healthy (low lambda)
    res = optimizer.optimize(t=1, c_hat=c_hat)
    print(f"  [After dual update: Task1 congested, Task2 healthy]")
    for tid in [1, 2]:
        r = res[tid]
        print(f"    Task {tid} (lambda={optimizer.lambda_k[tid]:.2f}): eta = {r['eta']}, s_comm = {r['s_comm']}, s_comp = {r['s_comp']}")


# ---------------------------------------------------------------------------
# Demo 5: Single-Task Baselines (from compare_to_baselines.py)
# ---------------------------------------------------------------------------

def demo_single_task_baselines():
    print("\n" + "=" * 60)
    print("DEMO 5: Single-Task Baselines")
    print("=" * 60)
    print("  Grouped exactly like compare_to_baselines.py")

    task = get_task(DEFAULT_TASK_NAME, task_id=1)
    c_t = np.array([50.0, 70.0])

    print("\n  CSI-aware baselines (instantiate -> set c_t -> optimize):")

    max_comp = MaxCompressionSingleTaskBaseline(M=3, tasks=[task])
    res = max_comp.optimize(t=0, c_t=c_t)
    eta = res[1]["eta"]
    print(f"    CSI-Aware Baseline: max compression -> eta = {eta}, A = {task.A_k_true(eta):.4f}")

    no_comp = NoCompressionSingleTaskBaseline(M=3, tasks=[task])
    res = no_comp.optimize(t=0, c_t=c_t)
    eta = res[1]["eta"]
    print(f"    CSI-Aware Baseline: no compression  -> eta = {eta}, A = {task.A_k_true(eta):.4f}")

    uniform_comp = UniformCompressionSingleTaskBaseline(M=3, tasks=[task])
    res = uniform_comp.optimize(t=0, c_t=c_t)
    eta = res[1]["eta"]
    print(f"    CSI-Aware Baseline: uniform compression  -> eta = {eta}, A = {task.A_k_true(eta):.4f}")

    print("\n  No-CSI baselines (instantiate -> estimate c_hat -> optimize):")
    print("    Note: single-task No-CSI baselines are estimator-based; they do not use dual queues.")

    # Myopic
    myopic_est = LastObservationEstimator(num_links=2, warmup_value=40.0)
    myopic_est.update(c_t)
    c_hat_myopic = myopic_est.estimate(1)
    myopic = MyopicSingleTaskBaseline(M=3, tasks=[task])
    res = myopic.optimize(t=1, c_hat=c_hat_myopic)
    eta = res[1]["eta"]
    print(f"    No-CSI Baseline: myopic        c_hat={c_hat_myopic} -> eta = {eta}")

    # Conservative
    cons_est = RunningMinEstimator(num_links=2, warmup_value=40.0)
    cons_est.update(c_t)
    c_hat_cons = cons_est.estimate(1)
    conservative = ConservativeSingleTaskBaseline(M=3, tasks=[task])
    res = conservative.optimize(t=1, c_hat=c_hat_cons)
    eta = res[1]["eta"]
    print(f"    No-CSI Baseline: conservative  c_hat={c_hat_cons} -> eta = {eta}")

    # Moving average
    ma_est = MovingAverageEstimator(num_links=2, window=5, warmup_value=40.0)
    ma_est.update(c_t)
    c_hat_ma = ma_est.estimate(1)
    moving_avg = MovingAverageSingleTaskBaseline(M=3, tasks=[task])
    res = moving_avg.optimize(t=1, c_hat=c_hat_ma)
    eta = res[1]["eta"]
    print(f"    No-CSI Baseline: moving average c_hat={c_hat_ma} -> eta = {eta}")

    # LCB
    lcb_est = MeanMinusZStdLCB(
        num_links=2, warmup_value=40.0, z=1.2815515655446004, warmup=1
    )
    lcb_est.update(c_t)
    c_hat_lcb = lcb_est.estimate(1)
    lcb_baseline = LCBSingleTaskBaseline(M=3, tasks=[task])
    res = lcb_baseline.optimize(t=1, c_hat=c_hat_lcb)
    eta = res[1]["eta"]
    print(f"    No-CSI Baseline: LCB           c_hat={c_hat_lcb} -> eta = {eta}")


# ---------------------------------------------------------------------------
# Demo 6: Multi-Task Baselines (from compare_to_baselines_multi.py)
# ---------------------------------------------------------------------------

def demo_multi_task_baselines():
    print("\n" + "=" * 60)
    print("DEMO 6: Multi-Task Baselines")
    print("=" * 60)
    print("  Grouped exactly like compare_to_baselines_multi.py")

    task1 = get_task(DEFAULT_TASK_NAME, task_id=1, w_k=1.0)
    task2 = get_task(DEFAULT_TASK_NAME, task_id=2, w_k=2.0)
    tasks = [task1, task2]
    c_t = np.array([80.0, 80.0])
    c_hat = np.array([70.0, 65.0])

    print("\n  CSI-aware baselines (instantiate -> set c_t -> optimize):")

    max_comp = MaxCompressionMultiTaskBaseline(M=3, tasks=tasks)
    res = max_comp.optimize(t=0, c_t=c_t)
    print(f"    CSI-Aware Baseline: max compression -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    no_comp = NoCompressionMultiTaskBaseline(M=3, tasks=tasks)
    res = no_comp.optimize(t=0, c_t=c_t)
    print(f"    CSI-Aware Baseline: no compression  -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    equal_share = StaticEqualShareMultiTaskBaseline(M=3, tasks=tasks)
    res = equal_share.optimize(t=0, c_t=c_t)
    print(f"    CSI-Aware Baseline: equal-share     -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    proportional = ProportionalResourceAllocationMultiTaskBaseline(M=3, tasks=tasks)
    res = proportional.optimize(t=0, c_t=c_t)
    print(f"    CSI-Aware Baseline: proportional    -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    strict_priority = StrictPriorityGreedyMultiTaskBaseline(M=3, tasks=tasks)
    res = strict_priority.optimize(t=0, c_t=c_t)
    print(f"    CSI-Aware Baseline: strict priority -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    print("\n  No-CSI baselines (instantiate -> set c_hat -> optimize -> update_dual):")

    decoupled = DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
        M=3, tasks=tasks, mu=1.0, epsilon=0.1
    )
    res = decoupled.optimize(t=0, c_hat=c_hat)
    print(f"    No-CSI Baseline: decoupled equal-split -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")
    decoupled.update_dual(t=0, actual_delays={1: 0.50, 2: 0.05})
    print(f"      after update_dual: lambda1={decoupled.lambda_k[1]:.2f}, lambda2={decoupled.lambda_k[2]:.2f}")
    res = decoupled.optimize(t=1, c_hat=c_hat)
    print(f"      re-optimize(t=1): Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    queue_prop = QueueProportionalHeuristicMultiTaskBaseline(
        M=3, tasks=tasks, mu=1.0, epsilon=0.1
    )
    res = queue_prop.optimize(t=0, c_hat=c_hat)
    print(f"    No-CSI Baseline: lambda-proportional  -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")
    queue_prop.update_dual(t=0, actual_delays={1: 0.50, 2: 0.05})
    print(f"      after update_dual: lambda1={queue_prop.lambda_k[1]:.2f}, lambda2={queue_prop.lambda_k[2]:.2f}")
    res = queue_prop.optimize(t=1, c_hat=c_hat)
    print(f"      re-optimize(t=1): Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}")

    hist_mean = MeanEstimator(num_links=2, warmup_value=40.0, warmup=3)
    hist_mean.update(c_t)
    moving_average_c_hat = hist_mean.estimate(1)
    certainty_eq = HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(M=3, tasks=tasks)
    res = certainty_eq.optimize(t=1, c_hat=moving_average_c_hat)
    print(
        "    No-CSI Baseline: moving average"
        f" -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}"
    )
    print("      note: this baseline does not maintain dual queues (no update_dual).")

    lcb_est = MeanMinusZStdLCB(
        num_links=2, warmup_value=40.0, z=1.2815515655446004, warmup=1
    )
    lcb_est.update(c_t)
    lcb_c_hat = lcb_est.estimate(1)
    lcb_multi = LCBCertaintyEquivalenceMultiTaskBaseline(M=3, tasks=tasks)
    res = lcb_multi.optimize(t=1, c_hat=lcb_c_hat)
    print(
        "    No-CSI Baseline: LCB"
        f" -> Task1 eta={res[1]['eta']}, Task2 eta={res[2]['eta']}"
    )
    print("      note: this baseline does not maintain dual queues (no update_dual).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Inference Optimizer Demo – Optimizers + Baselines with InferenceTaskCallables")

    demo_csi_aware_single_task()
    demo_csi_aware_multi_task()
    demo_no_csi_single_task()
    demo_no_csi_multi_task()
    demo_single_task_baselines()
    demo_multi_task_baselines()

    print("\n" + "=" * 60)
    print("All demos completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
