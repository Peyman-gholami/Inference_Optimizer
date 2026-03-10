from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np

from src.core.task import InferenceTask

class BaseOptimizer(ABC):
    """
    Abstract base class for all distributed inference optimizers.
    """
    def __init__(self, M: int, tasks: List[InferenceTask]):
        """
        Initializes the optimizer with the physical system topology and task list.
        
        Args:
            M (int): The number of computational nodes arranged in a linear topology. 
            tasks (List[InferenceTask]): The list of distinct inference tasks running on the system.
        """
        self.M = M
        self.tasks = tasks
        self.K = len(tasks)

    def check_feasibility(self, t: int, c_t: np.ndarray) -> bool:
        """
        Evaluates whether a valid resource allocation exists.
        
        By default, this returns True (used by No-CSI algorithms which handle 
        violations via long-term dual queues). 
        CSI-Aware subclasses MUST override this to implement strict per-slot checks.
        """
        return True

    @abstractmethod
    def optimize(self, t: int, c_t: np.ndarray) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """
        Determines the optimal resource allocations for a given time slot.
        Implementations MUST call `check_feasibility` before attempting to solve.
        
        Args:
            t (int): The current time slot.
            c_t (np.ndarray): The channel capacities at time slot t.
                              
        Returns:
            Optional[Dict[int, Dict[str, np.ndarray]]]: 
                A nested dictionary mapping each task_id to its specific configuration.
                Returns None if the problem is infeasible for the current time slot.
        """
        pass

    def scale_allocations_to_unit_sum(
        self, allocations: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Scale s_comp and s_comm so that at each node (resp. link) the sum across
        tasks equals 1. E.g. if two tasks each have s_comp=0.1 at a node (sum=0.2),
        both become 0.5 at that node. Leaves eta unchanged.
        """
        if not allocations:
            return allocations

        M = self.M
        comp_sum_per_node = np.zeros(M)
        comm_sum_per_link = np.zeros(M - 1)

        for task in self.tasks:
            tid = task.task_id
            if tid not in allocations:
                continue
            s_comp = allocations[tid]["s_comp"]
            s_comm = allocations[tid]["s_comm"]
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                comp_sum_per_node[node_i] += s_comp[local_idx]
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                comm_sum_per_link[link_i] += s_comm[local_idx]

        result = {}
        for task in self.tasks:
            tid = task.task_id
            if tid not in allocations:
                continue
            s_comp = np.array(allocations[tid]["s_comp"], dtype=float)
            s_comm = np.array(allocations[tid]["s_comm"], dtype=float)
            for local_idx, node_i in enumerate(range(task.b_k, task.e_k + 1)):
                total = comp_sum_per_node[node_i]
                if total > 0:
                    s_comp[local_idx] /= total
            for local_idx, link_i in enumerate(range(task.b_k, task.e_k)):
                total = comm_sum_per_link[link_i]
                if total > 0:
                    s_comm[local_idx] /= total
            result[tid] = {
                **allocations[tid],
                "s_comp": s_comp,
                "s_comm": s_comm,
            }
        return result

    def update_dual(self, t: int, actual_delays: Dict[int, float]):
        """
        Updates the dual variables (Lagrange multipliers) based on the true 
        experienced bottleneck delay. 
        
        Args:
            t (int): The current time slot.
            actual_delays (Dict[int, float]): A mapping of task_id to the actual 
                                              bottleneck delay experienced.
        """
        pass