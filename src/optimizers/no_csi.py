import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, List

from src.optimizers.base import BaseOptimizer
from src.core.task import InferenceTask

class NoCSISingleTaskOptimizer(BaseOptimizer):
    """
    Implements Algorithm 1: Estimated Stochastic Dual Descent for a Single Task .
    """
    def __init__(self, M: int, tasks: List[InferenceTask], mu: float):
        super().__init__(M, tasks)
        self.mu = mu  # Tradeoff parameter \mu > 0 [cite: 245]
        self.lambda_val = 0.0  # Initialize the scalar Lagrange multiplier \lambda_1 = 0 [cite: 246]

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Primal Step: Selects \eta(t) to maximize A(\eta) - \mu * \lambda_t * \hat{D}(\eta, \hat{c}(t)) [cite: 254-255].
        """
        task = self.tasks[0]
        
        # In a single-task setup, compute and comm resources are fully available
        s_comp_opt = np.ones(task.L_k)
        s_comm_opt = np.ones(task.L_k - 1)
        
        # Calculate the baseline compute delays \tau_i
        tau_delays = [task.tau[i] for i in range(task.b_k, task.e_k + 1)]
        max_tau = max(tau_delays) if tau_delays else 0.0

        # Optimization variables: x = [\eta_1, ..., \eta_{L-1}, z]
        # where z represents the estimated bottleneck delay \hat{D} [cite: 228]
        N_eta = task.L_k - 1
        
        def objective(x: np.ndarray):
            eta_vec = x[:N_eta]
            z = x[-1]
            
            acc = task.A_k(eta_vec)
            grad_acc = task.grad_A_k(eta_vec)
            
            # Objective: Maximize A(\eta) - \mu * \lambda * z -> Minimize -A(\eta) + \mu * \lambda * z
            obj = -acc + self.mu * self.lambda_val * z
            
            # Gradients
            grad = np.zeros_like(x)
            grad[:N_eta] = -grad_acc
            grad[-1] = self.mu * self.lambda_val
            
            return obj, grad

        bounds = []
        x0 = []
        for i in range(task.b_k, task.e_k):
            bounds.append((task.eta_min[i], 1.0))
            x0.append(1.0)
            
        # Bound for z: It must be at least the maximum computation delay
        bounds.append((max_tau, None))
        x0.append(max_tau + 0.1)

        # Constraint: z >= (a_i * \eta_i) / \hat{c}_i  =>  z - (a_i / \hat{c}_i) * \eta_i >= 0 [cite: 228]
        constraints = []
        for idx, i in enumerate(range(task.b_k, task.e_k)):
            beta = task.a[i] / c_hat[i]
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, beta=beta, idx=idx: x[-1] - beta * x[idx],
                'jac': lambda x, beta=beta, idx=idx: np.array(
                    [-beta if j == idx else (1.0 if j == N_eta else 0.0) for j in range(len(x))]
                )
            })

        result = minimize(
            fun=objective,
            x0=np.array(x0),
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints
        )

        eta_opt = result.x[:N_eta]

        return {
            task.task_id: {
                'eta': eta_opt,
                's_comp': s_comp_opt,
                's_comm': s_comm_opt
            }
        }

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        """
        Dual Step: Update the Lagrange multiplier based on actual realized constraint violation [cite: 258-259].
        """
        task = self.tasks[0]
        actual_delay = actual_delays[task.task_id]
        R_t = task.get_R_k(t)
        
        # \lambda_{t+1} = \max\{0, \lambda_t + D_{act}(t) - 1/R(t)\} [cite: 259]
        self.lambda_val = max(0.0, self.lambda_val + actual_delay - (1.0 / R_t))


class NoCSIMultiTaskOptimizer(BaseOptimizer):
    """
    Implements Algorithm 2: Multi-Task Estimated Stochastic Dual Descent .
    """
    def __init__(self, M: int, tasks: List[InferenceTask], mu: float, J: int = 5):
        super().__init__(M, tasks)
        self.mu = mu  # Tradeoff parameter [cite: 328]
        self.J = J    # Maximum Block Coordinate Descent (BCD) iterations [cite: 328]
        # Initialize dual variables for all tasks to 0 [cite: 329]
        self.lambda_k = {task.task_id: 0.0 for task in tasks}

    def optimize(self, t: int, c_hat: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Primal Step: Uses Block Coordinate Descent to alternate between Phase A and Phase B [cite: 335-343].
        """
        # Initialize configurations \eta to 1.0 (uncompressed) to start
        eta_current = {
            task.task_id: {i: 1.0 for i in range(task.b_k, task.e_k)}
            for task in self.tasks
        }
        
        s_comp_current = {task.task_id: {} for task in self.tasks}
        s_comm_current = {task.task_id: {} for task in self.tasks}

        # BCD Loop [cite: 340]
        for j in range(self.J):
            
            # ==========================================================
            # Phase A: Resource Optimization (Fix \eta, solve for s via z) [cite: 341]
            # Formulated entirely in terms of the delay variables z_k [cite: 356-361]
            # ==========================================================
            def phase_a_objective(z_flat: np.ndarray):
                obj = sum(self.mu * self.lambda_k[task.task_id] * z_flat[idx] 
                          for idx, task in enumerate(self.tasks))
                grad = np.array([self.mu * self.lambda_k[task.task_id] for task in self.tasks])
                return obj, grad

            phase_a_constraints = []
            
            # Node computation capacity constraints: \sum (\tau_{i,k} / z_k) <= 1 [cite: 359]
            for i in range(self.M):
                active_tasks_idx = [idx for idx, task in enumerate(self.tasks) if task.is_active_at_node(i)]
                if active_tasks_idx:
                    def comp_constraint(z, node_i=i, tasks_idx=active_tasks_idx):
                        return 1.0 - sum(self.tasks[idx].tau[node_i] / z[idx] for idx in tasks_idx)
                    phase_a_constraints.append({'type': 'ineq', 'fun': comp_constraint})

            # Link communication capacity constraints: \sum (a_{i,k}\eta_{i,k} / (\hat{c}_i z_k)) <= 1 [cite: 360]
            for i in range(self.M - 1):
                active_tasks_idx = [idx for idx, task in enumerate(self.tasks) if task.is_active_on_link(i)]
                if active_tasks_idx:
                    def comm_constraint(z, link_i=i, tasks_idx=active_tasks_idx):
                        return 1.0 - sum(
                            (self.tasks[idx].a[link_i] * eta_current[self.tasks[idx].task_id][link_i]) 
                            / (c_hat[link_i] * z[idx]) 
                            for idx in tasks_idx
                        )
                    phase_a_constraints.append({'type': 'ineq', 'fun': comm_constraint})

            # Bounds for z_k: Must be > 0. We initialize with conservative estimates.
            z0 = [1.0] * self.K
            z_bounds = [(1e-5, None) for _ in range(self.K)]

            res_a = minimize(fun=phase_a_objective, x0=np.array(z0), method='SLSQP', 
                             jac=True, bounds=z_bounds, constraints=phase_a_constraints)
            
            z_opt = res_a.x
            
            # Extract s_comp and s_comm from the optimized z variables [cite: 353]
            for idx, task in enumerate(self.tasks):
                z_k = z_opt[idx]
                for i in range(task.b_k, task.e_k + 1):
                    s_comp_current[task.task_id][i] = min(1.0, task.tau[i] / z_k)
                for i in range(task.b_k, task.e_k):
                    s_comm_current[task.task_id][i] = min(1.0, (task.a[i] * eta_current[task.task_id][i]) / (c_hat[i] * z_k))

            # ==========================================================
            # Phase B: Configuration Optimization (Fix s, solve for \eta) [cite: 342]
            # ==========================================================
            # This cleanly decouples per task since resources s are fixed.
            for task in self.tasks:
                N_eta = task.L_k - 1
                
                def phase_b_objective(x: np.ndarray):
                    eta_vec = x[:N_eta]
                    z = x[-1]
                    
                    acc = task.A_k(eta_vec)
                    grad_acc = task.grad_A_k(eta_vec)
                    
                    obj = -(task.w_k * acc - self.mu * self.lambda_k[task.task_id] * z)
                    
                    grad = np.zeros_like(x)
                    grad[:N_eta] = -task.w_k * grad_acc
                    grad[-1] = self.mu * self.lambda_k[task.task_id]
                    return obj, grad

                phase_b_bounds = []
                x0_b = []
                for i in range(task.b_k, task.e_k):
                    phase_b_bounds.append((task.eta_min[i], 1.0))
                    x0_b.append(eta_current[task.task_id][i])
                
                # Minimum bound for z based on fixed compute resources [cite: 353]
                min_z_comp = max([task.tau[i] / s_comp_current[task.task_id][i] 
                                  for i in range(task.b_k, task.e_k + 1)])
                phase_b_bounds.append((min_z_comp, None))
                x0_b.append(z_opt[self.tasks.index(task)])

                phase_b_constraints = []
                for idx, i in enumerate(range(task.b_k, task.e_k)):
                    # z >= (a_{i,k} * \eta_{i,k}) / (\hat{c}_i * s^{comm}_{i,k}) [cite: 353]
                    beta = task.a[i] / (c_hat[i] * s_comm_current[task.task_id][i])
                    phase_b_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, beta=beta, idx=idx: x[-1] - beta * x[idx]
                    })

                res_b = minimize(fun=phase_b_objective, x0=np.array(x0_b), method='SLSQP', 
                                 jac=True, bounds=phase_b_bounds, constraints=phase_b_constraints)
                
                # Update \eta for the next BCD iteration
                for idx, i in enumerate(range(task.b_k, task.e_k)):
                    eta_current[task.task_id][i] = res_b.x[idx]

        # Pack the final operational decisions \eta^{(J)} and s^{(J)} [cite: 343-344]
        final_allocations = {}
        for task in self.tasks:
            final_allocations[task.task_id] = {
                'eta': np.array([eta_current[task.task_id][i] for i in range(task.b_k, task.e_k)]),
                's_comp': np.array([s_comp_current[task.task_id][i] for i in range(task.b_k, task.e_k + 1)]),
                's_comm': np.array([s_comm_current[task.task_id][i] for i in range(task.b_k, task.e_k)])
            }

        return final_allocations

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        """
        Dual Step: Update the Lagrange multipliers based on actual realized constraint violations, 
        freezing the queues for inactive tasks[cite: 346].
        """
        for task in self.tasks:
            # We assume all tasks in self.tasks are active in this slot (k \in \mathcal{K}(t))
            if task.task_id in actual_delays:
                D_act = actual_delays[task.task_id]
                R_t = task.get_R_k(t)
                
                # \lambda_k(t+1) = \max\{0, \lambda_k(t) + D_{act,k}(t) - 1/R_k(t)\} [cite: 347]
                self.lambda_k[task.task_id] = max(0.0, self.lambda_k[task.task_id] + D_act - (1.0 / R_t))