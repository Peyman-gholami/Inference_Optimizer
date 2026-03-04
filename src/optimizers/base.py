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