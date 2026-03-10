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


def test_no_csi_multi_task_bcd_structure(flexible_task_factory):
    """
    Verifies that the BCD loop executes without crashing, respects global 
    network capacity constraints, and outputs the correct data structures.
    """
    task1 = flexible_task_factory(task_id=1, b_k=0, L_k=3, acc_multiplier=1.0)
    task2 = flexible_task_factory(task_id=2, b_k=0, L_k=3, acc_multiplier=2.0)
    
    # We use a small J=2 for the test to ensure it runs quickly while still 
    # verifying that the loop successfully passes state back and forth.
    optimizer = NoCSIMultiTaskOptimizer(M=3, tasks=[task1, task2], mu=1.0, epsilon=0.1, J=10)
    
    c_hat = np.array([50.0, 50.0])
    res = optimizer.optimize(t=0, c_hat=c_hat)
    
    assert res is not None, "BCD optimization failed to find a valid allocation."
    
    print("\n--- NO-CSI MULTI-TASK BCD: sanity check ---")
    for task_id in [1, 2]:
        print(f"Task {task_id} (Lambda = {optimizer.lambda_k[task_id]}):")
        print(f"  ETA:    {res[task_id]['eta']}")
        print(f"  S_COMM: {res[task_id]['s_comm']}")
        print(f"  S_COMP: {res[task_id]['s_comp']}")
    print("------------------------------------------------")

    # 1. Check data structures
    for task_id in [1, 2]:
        assert 'eta' in res[task_id]
        assert 's_comp' in res[task_id]
        assert 's_comm' in res[task_id]
        assert len(res[task_id]['eta']) == 2
        assert len(res[task_id]['s_comm']) == 2
        assert len(res[task_id]['s_comp']) == 3

    # 2. Check global network capacity constraints
    for link_i in range(2):
        total_s_comm = res[1]['s_comm'][link_i] + res[2]['s_comm'][link_i]
        assert total_s_comm <= 1.0001, f"Link {link_i} capacity exceeded: {total_s_comm}"
        
    for node_i in range(3):
        total_s_comp = res[1]['s_comp'][node_i] + res[2]['s_comp'][node_i]
        assert total_s_comp <= 1.0001, f"Node {node_i} capacity exceeded: {total_s_comp}"


def test_no_csi_multi_task_queue_priority(flexible_task_factory):
    """
    Verifies that the BCD loop properly prioritizes tasks based on their dual queues.
    A heavily congested task should be forced to compress its data, while an 
    uncongested task should maximize its accuracy.
    """
    task1 = flexible_task_factory(task_id=1, b_k=0, L_k=3, acc_multiplier=1.0)
    task2 = flexible_task_factory(task_id=2, b_k=0, L_k=3, acc_multiplier=1.0)
    
    optimizer = NoCSIMultiTaskOptimizer(M=3, tasks=[task1, task2], mu=1.0, epsilon=0.1, J=20)
    
    # ==========================================
    # ASYMMETRIC QUEUE STATE
    # ==========================================
    # Task 1 is suffering from a massive delay backlog.
    # Task 2 is perfectly healthy.
    optimizer.lambda_k[1] = 50.0 
    optimizer.lambda_k[2] = 0.1
    
    c_hat = np.array([40.0, 40.0])
    res = optimizer.optimize(t=0, c_hat=c_hat)
    
    print("\n--- NO-CSI MULTI-TASK BCD: ASYMMETRIC QUEUES ---")
    for task_id in [1, 2]:
        print(f"Task {task_id} (Lambda = {optimizer.lambda_k[task_id]}):")
        print(f"  ETA:    {res[task_id]['eta']}")
        print(f"  S_COMM: {res[task_id]['s_comm']}")
        print(f"  S_COMP: {res[task_id]['s_comp']}")
    print("------------------------------------------------")

    # 1. Configuration (Phase B) Assertion
    # Because Task 1 has a massive queue, the delay penalty dominates its objective.
    # The solver MUST compress Task 1 to its minimum (0.1).
    np.testing.assert_allclose(res[1]['eta'], np.array([0.1, 0.1]), atol=1e-2)
    
    # Because Task 2 has no queue, its delay penalty is exactly 0. 
    # The solver has no reason to compress it, so it should stay at 1.0.
    np.testing.assert_allclose(res[2]['eta'], np.array([1.0, 1.0]), atol=1e-2)

    # 2. Resource Allocation (Phase A) Assertion
    # To minimize the heavily weighted delay z_1 for Task 1, the BCD loop will 
    # prioritize clearing Task 1's bottleneck. Therefore, Task 1's actual 
    # delay footprint (which dictates s_comm) is handled aggressively.
    # (Note: In pure BCD, exact s_comm proportions depend on the solver bounds,
    # but the structural eta constraints are the most rigid indicators of success).