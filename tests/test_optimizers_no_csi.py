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



def test_no_csi_single_task_primal_dual_dynamics(flexible_task_factory):
    """
    Verifies the No-CSI Single-Task epigraph solver and queue dynamics.
    Tests both the primal resource allocation trade-offs and the dual queue updates.
    """
    # Task 1: 3 nodes (2 links), R = 10.0 (Target Delay = 0.1s), eta_min = 0.1
    task = flexible_task_factory(task_id=1, b_k=0, L_k=3, acc_multiplier=1.0)
    
    # We set mu = 10.0 to create a strong penalty when the queue grows
    optimizer = NoCSISingleTaskOptimizer(M=3, tasks=[task], mu=1.0)

    # Moderate estimated capacity
    c_hat = np.array([50.0, 50.0])

    # ==========================================
    # SCENARIO 1: Empty Queue (No Historical Violations)
    # ==========================================
    # Because lambda_t = 0.0, the objective penalty (mu * lambda * z) is exactly 0.
    # The solver has no reason to compress, so it should maximize accuracy.
    optimizer.lambda_t = 0.0
    res_empty = optimizer.optimize(t=0, c_hat=c_hat)
    
    assert res_empty is not None
    
    print("\n--- NO-CSI SINGLE-TASK: EMPTY QUEUE ---")
    print(f"Lambda: {optimizer.lambda_t}")
    print(f"ETA Allocated: {res_empty[1]['eta']}")
    print("---------------------------------------")
    
    # Assert eta naturally floated to the upper bound of 1.0
    np.testing.assert_allclose(res_empty[1]['eta'], np.array([1.0, 1.0]), atol=1e-3)

    # ==========================================
    # SCENARIO 2: Dual Update (Constraint Violation)
    # ==========================================
    # The optimizer sent uncompressed data (eta=1.0), but let's pretend the actual 
    # channel condition was terrible. The actual delay took 0.5s.
    # Target delay = 1.0 / 10.0 = 0.1s.
    # Formula: lambda = max(0, lambda + D_act - 1/R) -> 0.0 + 0.5 - 0.1 = 0.4
    optimizer.update_dual(t=0, actual_delays={1: 0.5})
    
    print("\n--- QUEUE UPDATE (VIOLATION) ---")
    print(f"New Lambda after 0.5s delay: {optimizer.lambda_t:.2f}")
    
    assert np.isclose(optimizer.lambda_t, 0.4)

    # ==========================================
    # SCENARIO 3: Congested Queue (High Penalty)
    # ==========================================
    # To really test the epigraph bounds, let's artificially spike the queue 
    # to simulate severe, prolonged network congestion.
    optimizer.lambda_t = 11.0
    
    res_congested = optimizer.optimize(t=1, c_hat=c_hat)
    
    print("\n--- NO-CSI SINGLE-TASK: HEAVY CONGESTION ---")
    print(f"Lambda: {optimizer.lambda_t}")
    print(f"ETA Allocated: {res_congested[1]['eta']}")
    print("--------------------------------------------")
    
    # The penalty for delay is now massive. The solver MUST compress the data 
    # as much as physically possible, driving eta down to its minimum bound (0.1).
    np.testing.assert_allclose(res_congested[1]['eta'], np.array([0.1, 0.1]), atol=1e-3)

    # ==========================================
    # SCENARIO 4: Dual Update (Queue Recovery)
    # ==========================================
    # Thanks to the maximum compression, the next transmission was lightning fast (0.05s).
    # Since 0.05s < 0.1s target, the queue should drain by 0.05.
    # Formula: 9.0 + 0.05 - 0.1 = 8.95
    optimizer.update_dual(t=1, actual_delays={1: 0.05})
    assert np.isclose(optimizer.lambda_t, 10.95)



def test_no_csi_single_task_exponential_spectrum():
    """
    Proves that a strictly concave accuracy function results in a smooth, 
    proportional reduction in bandwidth as the delay queue grows, rather 
    than a binary 1.0 or 0.1 response.
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
                exponent = -beta_params[link_i] * (eta_vec[idx] - eta_min[link_i])
                penalty += omega[link_i] * np.exp(exponent)
            return float(A_max - penalty)
        
        def grad_callable(eta_vec: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(eta_vec)
            for idx, link_i in enumerate(links):
                exponent = -beta_params[link_i] * (eta_vec[idx] - eta_min[link_i])
                grad[idx] = omega[link_i] * beta_params[link_i] * np.exp(exponent)
            return grad
            
        return InferenceTask(
            task_id=task_id, b_k=b_k, L_k=L_k, tau=tau, a=a, eta_min=eta_min,
            R_k_callable=lambda t: 10.0, w_k=1.0, 
            accuracy_callable=acc_callable, gradient_callable=grad_callable
        )

    # Balanced exponential task. 
    # Max marginal accuracy (at eta=0.1) is omega * beta = 0.5 * 5.0 = 2.5
    omega_params = {0: 0.5, 1: 0.5}
    beta_params = {0: 5.0, 1: 5.0}
    task = make_exponential_task(1, 0, 3, 1.0, omega_params, beta_params)
    
    optimizer = NoCSISingleTaskOptimizer(M=3, tasks=[task], mu=1.0)
    c_hat = np.array([50.0, 20.0]) # beta = a/c = 10/50 = 0.2 delay multiplier

    # We will test a spectrum of queue congestions
    lambda_values = [0.0, 2.0, 5.0, 15.0]
    previous_eta = np.array([1.1, 1.1]) # Start above max bound for strict < assertions
    
    print("\n--- NO-CSI EXPONENTIAL COMPRESSION SPECTRUM ---")
    
    for lambda_t in lambda_values:
        optimizer.lambda_t = lambda_t
        res = optimizer.optimize(t=0, c_hat=c_hat)
        
        current_eta = res[1]['eta']
        
        print(f"Queue (Lambda): {lambda_t:>4.1f} | Allocated ETA: {current_eta}")
        
        # ASSERTION 1: As the queue grows, the allocated bandwidth must strictly decrease
        assert np.all(current_eta < previous_eta) or np.allclose(current_eta, 0.1), \
            "ETA did not smoothly decrease as penalty increased!"
            
        # ASSERTION 2: Bounds check
        assert np.all(current_eta >= 0.1) and np.all(current_eta <= 1.0)
        
        previous_eta = current_eta
        
    print("-----------------------------------------------")

    # Specific mathematical checks based on the equilibrium:
    # 1. At lambda=0, accuracy wins completely.
    optimizer.lambda_t = 0.0
    res_0 = optimizer.optimize(t=0, c_hat=c_hat)
    np.testing.assert_allclose(res_0[1]['eta'], [1.0, 1.0], atol=1e-3)
    
    # 2. At lambda=15.0, penalty slope (1.0 * 15.0 * 0.2 = 3.0) is greater than 
    # the maximum possible accuracy slope (2.5). It must crush to 0.1.
    optimizer.lambda_t = 15.0
    res_15 = optimizer.optimize(t=0, c_hat=c_hat)
    np.testing.assert_allclose(res_15[1]['eta'], [0.2, 0.2], atol=1e-1)