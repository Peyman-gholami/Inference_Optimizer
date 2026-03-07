import pytest
import numpy as np

from src.core.task import InferenceTask
from src.optimizers import (
    CSIAwareSingleTaskOptimizer,
    CSIAwareMultiTaskOptimizer,
    NoCSISingleTaskOptimizer,
    NoCSIMultiTaskOptimizer
)

# ==========================================
# FIXTURES: DUMMY DATA FOR TESTING
# ==========================================

@pytest.fixture
def flexible_task_factory():
    """
    Creates a customizable task. Defaults to a 3-node task starting at node 0 
    with a standard linear accuracy sum(eta).
    """
    def _make_task(
        task_id: int, 
        b_k: int = 0, 
        L_k: int = 3, 
        acc_multiplier: float = 1.0
    ):
        # Dynamically generate nodes and links based on b_k and L_k
        nodes = list(range(b_k, b_k + L_k))
        links = list(range(b_k, b_k + L_k - 1))
        
        tau = {i: 0.01 for i in nodes}
        a = {i: 10.0 for i in links}
        eta_min = {i: 0.1 for i in links}
        
        # Custom accuracy: A = multiplier * sum(eta)
        def acc_callable(eta_vec: np.ndarray) -> float:
            return float(acc_multiplier * np.sum(eta_vec))
        
        # Gradient is an array filled with the multiplier
        def grad_callable(eta_vec: np.ndarray) -> np.ndarray:
            return np.full_like(eta_vec, acc_multiplier)

        return InferenceTask(
            task_id=task_id,
            b_k=b_k,
            L_k=L_k,
            tau=tau,
            a=a,
            eta_min=eta_min,
            R_k_callable=lambda t: 10.0,
            w_k=1.0,
            accuracy_callable=acc_callable,
            gradient_callable=grad_callable
        )
    return _make_task



def test_csi_aware_multi_task_heterogeneous(flexible_task_factory):
    """
    Verifies the "Per-Link Water-Filling with a QoS Floor" behavior[cite: 214].
    Task 2 has a steeper gradient, so the solver should starve Task 1 on their 
    shared link down to its absolute minimum viable bandwidth .
    """
    # Task 1: Starts at 0, length 3 (Links 0, 1). Accuracy multiplier = 1.0
    task1 = flexible_task_factory(task_id=1, b_k=0, L_k=3, acc_multiplier=1.0)
    
    # Task 2: Starts at 1, length 2 (Link 1 only). Accuracy multiplier = 2.0
    task2 = flexible_task_factory(task_id=2, b_k=1, L_k=2, acc_multiplier=2.0)
    
    optimizer = CSIAwareMultiTaskOptimizer(M=3, tasks=[task1, task2])

    # Moderate capacity: c = 50.0 for both links
    # Minimum bandwidth required for a task: s_min = (a * eta_min * R) / c 
    # s_min = (10.0 * 0.1 * 10.0) / 50.0 = 0.2
    c_t = np.array([200.0, 50.0])
    
    res = optimizer.optimize(t=0, c_t=c_t)
    print("res:", res)
    assert res is not None, "System should be feasible"

    s_comm_t1 = res[1]['s_comm']
    s_comm_t2 = res[2]['s_comm']

    print("\n--- HETEROGENEOUS TASK OUTPUT ---")
    print(f"Task 1 (Links 0, 1) S_COMM: {s_comm_t1}")
    print(f"Task 2 (Link 1 only) S_COMM: {s_comm_t2}")
    print("---------------------------------")

    # --- ASSERTIONS FOR LINK 0 ---
    # Only Task 1 is on Link 0, so it should get the full bandwidth (1.0)
    assert np.isclose(s_comm_t1[0], .5)

    # --- ASSERTIONS FOR LINK 1 ---
    # Task 1 and Task 2 share Link 1.
    # Because Task 2's marginal utility is double, the solver will attempt to 
    # starve Task 1. However, the constraint halts the reduction at s_min (0.2) .
    # Task 1 gets 0.2, Task 2 gets the remaining 0.8.
    assert np.isclose(s_comm_t1[1], 0.2, atol=1e-3), "Task 1 was not starved to its QoS floor!"
    assert np.isclose(s_comm_t2[0], 0.8, atol=1e-3), "Task 2 did not consume the remaining bandwidth!"
    
    # Verify exact array shapes mapping to physical links
    assert len(s_comm_t1) == 2  # Link 0 and 1
    assert len(s_comm_t2) == 1  # Link 1 only


def test_csi_aware_multi_task_exponential_accuracy():
    """
    Tests the optimizer using a realistic, non-linear exponential accuracy function:
    A(eta) = A_max - sum( omega_i * exp(-beta_i * (eta_i - eta_min_i)) )
    """
    def make_exponential_task(task_id, b_k, L_k, A_max, omega, beta_params):
        nodes = list(range(b_k, b_k + L_k))
        links = list(range(b_k, b_k + L_k - 1))
        
        tau = {i: 0.01 for i in nodes}
        a = {i: 10.0 for i in links}
        eta_min = {i: 0.1 for i in links}
        
        def acc_callable(eta_vec: np.ndarray) -> float:
            penalty = 0.0
            for idx, link_i in enumerate(links):
                # Penalty term: omega_i * e^(-beta_i * (eta_i - eta_min_i))
                exponent = -beta_params[link_i] * (eta_vec[idx] - eta_min[link_i])
                penalty += omega[link_i] * np.exp(exponent)
            return float(A_max - penalty)
        
        def grad_callable(eta_vec: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(eta_vec)
            for idx, link_i in enumerate(links):
                # Derivative: omega_i * beta_i * e^(-beta_i * (eta_i - eta_min_i))
                exponent = -beta_params[link_i] * (eta_vec[idx] - eta_min[link_i])
                grad[idx] = omega[link_i] * beta_params[link_i] * np.exp(exponent)
            return grad
            
        return InferenceTask(
            task_id=task_id, b_k=b_k, L_k=L_k, tau=tau, a=a, eta_min=eta_min,
            R_k_callable=lambda t: 10.0, w_k=1.0, 
            accuracy_callable=acc_callable, gradient_callable=grad_callable
        )

    # Task 1: Balanced sensitivity across both links
    omega_t1 = {0: 0.5, 1: 0.5}
    beta_t1 = {0: 4.0, 1: 4.0}
    task1 = make_exponential_task(1, 0, 3, A_max=1.0, omega=omega_t1, beta_params=beta_t1)
    
    # Task 2: Highly sensitive to compression on Link 1 (Higher omega and beta)
    omega_t2 = {0: 0.2, 1: 0.8}
    beta_t2 = {0: 2.0, 1: 8.0}
    task2 = make_exponential_task(2, 0, 3, A_max=1.0, omega=omega_t2, beta_params=beta_t2)

    optimizer = CSIAwareMultiTaskOptimizer(M=3, tasks=[task1, task2])
    
    # Restrict channel capacity to force them to fight for bandwidth.
    # c_t = 30.0 -> s_min = (10 * 0.1 * 10) / 30 = 0.333 for each task.
    c_t = np.array([30.0, 30.0])
    
    res = optimizer.optimize(t=0, c_t=c_t)
    assert res is not None, "Optimization failed!"
    
    s_comm_t1 = res[1]['s_comm']
    s_comm_t2 = res[2]['s_comm']
    
    print("\n--- EXPONENTIAL ACCURACY MULTI-TASK OUTPUT ---")
    for task_id in [1, 2]:
        acc_val = task1.A_k(res[1]['eta']) if task_id == 1 else task2.A_k(res[2]['eta'])
        print(f"Task {task_id}:")
        print(f"  ETA:    {res[task_id]['eta']}")
        print(f"  S_COMM: {res[task_id]['s_comm']}")
        print(f"  S_COMP: {res[task_id]['s_comp']}")
        print(f"  FINAL ACCURACY: {acc_val:.4f} / 1.0000")
    print("----------------------------------------------")

    # 1. Verify strict link capacity constraints
    for link_idx in range(2):
        total_s = s_comm_t1[link_idx] + s_comm_t2[link_idx]
        assert total_s <= 1.0001, f"Link {link_idx} capacity exceeded: {total_s}"
        
    # 2. Verify non-linear water-filling logic
    # Because Task 2 has omega=0.8 and beta=8.0 on Link 1, its marginal gradient 
    # is drastically steeper than Task 1's. The optimizer MUST allocate more 
    # bandwidth to Task 2 on Link 1 to maximize the sum of accuracies.
    assert s_comm_t2[1] > s_comm_t1[1], "Failed: Task 2 should have won more bandwidth on Link 1!"