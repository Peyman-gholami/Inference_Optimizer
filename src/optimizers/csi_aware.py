import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional

from src.optimizers.base import BaseOptimizer
from src.core.task import InferenceTask

class CSIAwareSingleTaskOptimizer(BaseOptimizer):
    """
    Implements the closed-form optimal solution for a single task with perfect CSI (Problem 1).
    """
    def check_feasibility(self, t: int, c_t: np.ndarray) -> bool:
        """
        Implements Lemma 3.1 (Single-Task Feasibility Condition).
        """
        # In a single-task setup, we assume self.tasks[0] is the only task
        task = self.tasks[0]
        R_t = task.get_R_k(t)

        # Condition 3: Check computation delay bounds
        for i in range(task.b_k, task.e_k + 1):
            if task.tau[i] > 1.0 / R_t:
                return False

        # Condition 4: Check communication delay bounds
        for i in range(task.b_k, task.e_k):
            if (task.a[i] * task.eta_min[i]) / c_t[i] > 1.0 / R_t:
                return False

        return True

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Implements Theorem 3.2 (Closed-Form Optimal Solution).
        """
        if not self.check_feasibility(t, c_t):
            return None

        task = self.tasks[0]
        R_t = task.get_R_k(t)

        eta = {}
        s_comp = {}
        s_comm = {}

        # Computation processing logic (Nodes)
        for i in range(task.b_k, task.e_k + 1):
            s_comp[i] = 1.0  # Receives full node capacity

        # Communication transmission logic (Links)
        for i in range(task.b_k, task.e_k):
            s_comm[i] = 1.0  # Receives full link capacity
            
            # Apply Theorem 3.2: \eta_i^*(t) = min(1, c_i(t) / (R(t) * a_i))
            eta_opt = min(1.0, c_t[i] / (R_t * task.a[i]))
            eta[i] = eta_opt

        # Format into the expected nested dictionary
        return {
            task.task_id: {
                'eta': np.array([eta[i] for i in range(task.b_k, task.e_k)]),
                's_comp': np.array([s_comp[i] for i in range(task.b_k, task.e_k + 1)]),
                's_comm': np.array([s_comm[i] for i in range(task.b_k, task.e_k)])
            }
        }


class CSIAwareMultiTaskOptimizer(BaseOptimizer):
    """
    Implements the convex optimization solver for multiple tasks with perfect CSI (Problem 2).
    """
    def check_feasibility(self, t: int, c_t: np.ndarray) -> bool:
        """
        Implements Lemma 3.3 (Multi-Task Feasibility Condition).
        """
        # Condition (14): Check computation sums at every node i in [M]
        for i in range(self.M):
            node_comp_sum = 0.0
            for task in self.tasks:
                if task.is_active_at_node(i):
                    node_comp_sum += task.tau[i] * task.get_R_k(t)
            if node_comp_sum > 1.0:
                return False

        # Condition (15): Check communication sums at every link i in [M-1]
        for i in range(self.M - 1):
            link_comm_sum = 0.0
            for task in self.tasks:
                if task.is_active_on_link(i):
                    R_k = task.get_R_k(t)
                    link_comm_sum += (task.a[i] * task.eta_min[i] * R_k) / c_t[i]
            if link_comm_sum > 1.0:
                return False

        return True

    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Reduces to a convex optimization problem over s^{comm}(t) (Theorem 3.4).
        Solves using SciPy SLSQP and piecewise chain-rule KKT gradients.
        """
        if not self.check_feasibility(t, c_t):
            return None

        # 1. Map active link-task pairs to a 1D flat array for SciPy
        # active_vars[idx] = (link_i, task_k)
        active_vars = []
        bounds = []
        x0 = []
        
        for task in self.tasks:
            R_k = task.get_R_k(t)
            for i in range(task.b_k, task.e_k):
                active_vars.append((i, task))
                
                # Minimum viable bandwidth s_{i,k}^{min}(t)
                s_min = (task.a[i] * task.eta_min[i] * R_k) / c_t[i]
                bounds.append((s_min, 1.0))
                x0.append(s_min) # Start strictly feasible

        N_vars = len(active_vars)

        # 2. Define the Objective and Gradient
        def objective(s_flat: np.ndarray):
            total_obj = 0.0
            grad_flat = np.zeros(N_vars)
            
            # Reconstruct the s_comm dictionary grouped by task to evaluate A_k
            task_s_comm = {task.task_id: {} for task in self.tasks}
            task_eta = {task.task_id: {} for task in self.tasks}
            
            for idx, (i, task) in enumerate(active_vars):
                task_s_comm[task.task_id][i] = s_flat[idx]
            
            # Evaluate objective and gradients per task
            for task in self.tasks:
                R_k = task.get_R_k(t)
                eta_vec = []
                grad_map = {} # Maps link i to its chain-rule derivative factor
                
                for i in range(task.b_k, task.e_k):
                    s_val = task_s_comm[task.task_id][i]
                    beta = c_t[i] / (R_k * task.a[i])
                    
                    # Calculate eta_{i,k} (Theorem 3.4)
                    eta_val = min(1.0, beta * s_val)
                    eta_vec.append(eta_val)
                    task_eta[task.task_id][i] = eta_val
                    
                    # Piecewise chain rule derivative logic
                    if np.isclose(eta_val, 1.0):
                        grad_map[i] = 0.0
                    else:
                        grad_map[i] = beta
                
                eta_array = np.array(eta_vec)
                
                # Black-box calls
                acc = task.A_k(eta_array)
                grad_acc = task.grad_A_k(eta_array)
                
                total_obj += task.w_k * acc
                
                # Apply chain rule and map back to flat array
                vec_idx = 0
                for i in range(task.b_k, task.e_k):
                    # Find where this (i, task) lives in the flat array
                    flat_idx = active_vars.index((i, task))
                    
                    # Chain rule: w_k * (d_A/d_eta) * (d_eta/d_s)
                    d_eta_ds = grad_map[i]
                    grad_flat[flat_idx] = task.w_k * grad_acc[vec_idx] * d_eta_ds
                    vec_idx += 1

            # SciPy MINIMIZES, so return negative objective and negative gradient
            return -total_obj, -grad_flat

        # 3. Define the Link Capacity Constraints (Sum of s <= 1 per link)
        constraints = []
        for link_i in range(self.M - 1):
            # Check if any tasks use this link to avoid empty constraints
            if any(var[0] == link_i for var in active_vars):
                # The function must be >= 0, so 1.0 - sum(s) >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, l_i=link_i: 1.0 - sum(
                        x[idx] for idx, var in enumerate(active_vars) if var[0] == l_i
                    )
                })

        # 4. Solve the convex optimization problem
        result = minimize(
            fun=objective,
            x0=np.array(x0),
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints
        )

        # 5. Unpack results into the standardized output format
        final_allocations = {}
        for task in self.tasks:
            R_k = task.get_R_k(t)
            
            # The computation variables s^{comp} are decoupled and calculated directly
            s_comp_opt = np.array([task.tau[i] * R_k for i in range(task.b_k, task.e_k + 1)])
            
            s_comm_opt = []
            eta_opt = []
            
            for i in range(task.b_k, task.e_k):
                flat_idx = active_vars.index((i, task))
                s_val = result.x[flat_idx]
                s_comm_opt.append(s_val)
                
                # Recalculate final eta
                eta_val = min(1.0, (c_t[i] / (R_k * task.a[i])) * s_val)
                eta_opt.append(eta_val)

            final_allocations[task.task_id] = {
                'eta': np.array(eta_opt),
                's_comp': s_comp_opt,
                's_comm': np.array(s_comm_opt)
            }

        return final_allocations