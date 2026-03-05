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
def dummy_task_factory():
    """
    Creates a simple 3-node, 2-link task.
    Accuracy is simply the sum of the compression factors (linear).
    Gradient is an array of 1s.
    """
    def _make_task(task_id: int):
        # A simple linear accuracy: A = sum(eta)
        def acc_callable(eta_vec: np.ndarray) -> float:
            return float(np.sum(eta_vec))
        
        # Gradient of sum(eta) is just an array of 1.0s
        def grad_callable(eta_vec: np.ndarray) -> np.ndarray:
            return np.ones_like(eta_vec)

        return InferenceTask(
            task_id=task_id,
            b_k=0,
            L_k=3, # 3 nodes -> 2 links (links 0 and 1)
            tau={0: 0.01, 1: 0.01, 2: 0.01},
            a={0: 10.0, 1: 10.0},
            eta_min={0: 0.1, 1: 0.1},
            R_k_callable=lambda t: 10.0, # 10 throughput
            w_k=1.0,
            accuracy_callable=acc_callable,
            gradient_callable=grad_callable
        )
    return _make_task


# ==========================================
# TEST 1: CSI-AWARE SINGLE TASK (CLOSED FORM)
# ==========================================

def test_csi_aware_single_task_feasibility_and_optimization(dummy_task_factory):
    """
    Verifies Lemma 3.1 (Feasibility) and Theorem 3.2 (Closed-Form Solution)[cite: 88, 105].
    Target throughput R(t) = 10.0 -> Max allowed delay = 0.1s.
    """
    task = dummy_task_factory(task_id=1)
    print("eta_min:", task.eta_min)
    optimizer = CSIAwareSingleTaskOptimizer(M=3, tasks=[task])

    # ==========================================
    # CASE 1: FEASIBLE (High Channel Capacity)
    # ==========================================
    # c = 20.0 -> min comm delay = (10.0 * 0.1) / 20.0 = 0.05s <= 0.1s (Feasible!)
    c_t_feasible = np.array([20.0, 20.0])
    
    # 1. Check strict feasibility method
    assert optimizer.check_feasibility(t=0, c_t=c_t_feasible) is True
    
    # 2. Check optimization output
    res_feasible = optimizer.optimize(t=0, c_t=c_t_feasible)
    assert res_feasible is not None
    
    # Formula: min(1.0, 20.0 / (10.0 * 10.0)) = min(1.0, 0.2) = 0.2
    np.testing.assert_array_equal(res_feasible[1]['eta'], np.array([0.2, 0.2]))
    np.testing.assert_array_equal(res_feasible[1]['s_comp'], np.array([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(res_feasible[1]['s_comm'], np.array([1.0, 1.0]))

    # ==========================================
    # CASE 2: INFEASIBLE (Comm Bottleneck)
    # ==========================================
    # c = 5.0 -> min comm delay = (10.0 * 0.1) / 5.0 = 0.2s > 0.1s (Infeasible!)
    c_t_infeasible_comm = np.array([5.0, 5.0])
    
    # 1. Feasibility check must fail
    assert optimizer.check_feasibility(t=0, c_t=c_t_infeasible_comm) is False
    
    # 2. Optimization must gracefully return None without running the solver
    res_infeasible = optimizer.optimize(t=0, c_t=c_t_infeasible_comm)
    assert res_infeasible is None

    # ==========================================
    # CASE 3: INFEASIBLE (Compute Bottleneck)
    # ==========================================
    # We alter the task to have a massive compute requirement on node 1.
    # tau = 0.5s > 0.1s max delay (Infeasible!)
    task.tau[1] = 0.5 
    
    # Even if the channel is perfect (c=1000), the node is too slow.
    c_t_perfect = np.array([1000.0, 1000.0])
    
    assert optimizer.check_feasibility(t=0, c_t=c_t_perfect) is False
    assert optimizer.optimize(t=0, c_t=c_t_perfect) is None




def test_csi_aware_multi_task_feasibility(dummy_task_factory):
    """
    Verifies Lemma 3.3: Multi-Task Feasibility Condition .
    """
    task1 = dummy_task_factory(task_id=1)
    task2 = dummy_task_factory(task_id=2)
    optimizer = CSIAwareMultiTaskOptimizer(M=3, tasks=[task1, task2])

    # ==========================================
    # CASE 1: FEASIBLE 
    # ==========================================
    # Both tasks need R=10.0. 
    # Min comm delay per task: (a * eta_min * R) / c = (10 * 0.1 * 10) / 100 = 0.1
    # Total link utilization = 0.1 + 0.1 = 0.2 <= 1.0 (Feasible!)
    c_t_feasible = np.array([100.0, 100.0])
    assert optimizer.check_feasibility(t=0, c_t=c_t_feasible) is True
    
    # ==========================================
    # CASE 2: INFEASIBLE (Comm Bottleneck)
    # ==========================================
    # c = 15.0 -> Min bandwidth per task = 10 / 15 = 0.66
    # Total link utilization = 0.66 + 0.66 = 1.33 > 1.0 (Infeasible!)
    c_t_infeasible_comm = np.array([15.0, 15.0])
    assert optimizer.check_feasibility(t=0, c_t=c_t_infeasible_comm) is False
    assert optimizer.optimize(t=0, c_t=c_t_infeasible_comm) is None

    # ==========================================
    # CASE 3: INFEASIBLE (Compute Bottleneck)
    # ==========================================
    # Default tau = 0.01, R = 10 -> node util = 0.1 per task (Total = 0.2).
    # Let's spike task 1's compute requirement on Node 1.
    task1.tau[1] = 0.091  # 0.091 * 10 = 0.91 node utilization
    # Total node 1 utilization = 0.91 (Task 1) + 0.1 (Task 2) = 1.01 > 1.0 (Infeasible!)
    
    assert optimizer.check_feasibility(t=0, c_t=c_t_feasible) is False



def test_csi_aware_multi_task_optimization(dummy_task_factory):
    """
    Verifies Theorem 3.4: Convex reduction and resource allocation [cite: 178-190].
    """
    task1 = dummy_task_factory(task_id=1)
    task2 = dummy_task_factory(task_id=2)
    optimizer = CSIAwareMultiTaskOptimizer(M=3, tasks=[task1, task2])

    # Give them moderate capacity so they have to share the links
    c_t = np.array([500.0, 50.0])
    
    res = optimizer.optimize(t=0, c_t=c_t)
    assert res is not None, "System should be feasible"

    s_comm_t1 = res[1]['s_comm']
    s_comm_t2 = res[2]['s_comm']
    
    # --- PRINT OUTPUT FOR DEBUGGING (-s flag) ---
    print("\n--- MULTI-TASK OPTIMIZER OUTPUT ---")
    for task_id in [1, 2]:
        print(f"Task {task_id}:")
        print(f"  ETA:    {res[task_id]['eta']}")
        print(f"  S_COMP: {res[task_id]['s_comp']}")
        print(f"  S_COMM: {res[task_id]['s_comm']}")
    print("-----------------------------------")

    # 1. Verify decoupled computation shares (s_comp = tau * R) 
    # tau = 0.01, R = 10.0 -> s_comp should be exactly 0.1
    np.testing.assert_allclose(res[1]['s_comp'], np.array([0.1, 0.1, 0.1]))
    np.testing.assert_allclose(res[2]['s_comp'], np.array([0.1, 0.1, 0.1]))

    # 2. Verify strict link capacity constraints (Sum of s_comm <= 1.0) [cite: 137-138]
    for link_idx in range(2):
        total_bandwidth_used = s_comm_t1[link_idx] + s_comm_t2[link_idx]
        assert total_bandwidth_used <= 1.0001, f"Link {link_idx} capacity exceeded!"

    # 3. Verify eta calculation matches Theorem 3.4 [cite: 182]
    # eta = min(1.0, (c / (R * a)) * s_comm)
    beta = 500.0 / (10.0 * 10.0) # c / (R * a) = 0.5
    expected_eta_t1 = np.minimum(1.0, beta * s_comm_t1)
    
    np.testing.assert_allclose(res[1]['eta'], expected_eta_t1)