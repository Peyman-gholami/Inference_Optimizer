from typing import Callable, Dict, Optional
import numpy as np

class InferenceTask:
    """
    Represents a distributed inference task k spanning a sequence of interconnected nodes.
    """
    def __init__(
        self,
        task_id: int,
        b_k: int,
        L_k: int,
        tau: Dict[int, float],
        a: Dict[int, float],
        eta_min: Dict[int, float],
        R_k_callable: Callable,   # signature: (t: int) -> float
                                  # OR for llmint8: (t: int, c_t: np.ndarray) -> np.ndarray
                                  # where return is per-link effective R_k array
        w_k: float,
        accuracy_callable: Callable[[np.ndarray], float],
        accuracy_callable_true: Callable[[np.ndarray], float],
        gradient_callable: Callable[[np.ndarray], np.ndarray],
    ):
        # Topology Assignments
        self.task_id = task_id   
        self.b_k = b_k           
        self.L_k = L_k           
        self.e_k = b_k + L_k - 1 
        
        # Resource Requirements 
        self.tau = tau           
        self.a = a               
        self.eta_min = eta_min   
        
        # Objective and QoS
        self._R_k_callable = R_k_callable 
        self.w_k = w_k           
        
        # Black-Box Callables
        self._accuracy_callable = accuracy_callable
        self._gradient_callable = gradient_callable
        self._accuracy_callable_true = accuracy_callable_true
        
    def get_R_k(self, t: int, c_t: Optional[np.ndarray] = None) -> float:
        """
        Retrieves the target throughput R_k(t) for a specific time slot t.

        For standard strategies, returns a scalar R_k independent of c_t.

        For llmint8, the effective throughput requirement depends on the
        current channel capacity because the INT8 floor payload occupies
        a fixed fraction of the channel regardless of eta:

            D_actual = a_i*(1+eta)/(2*c_i)
                     = a_i*eta/(2*c_i) + a_i/(2*c_i)

        Absorbing the fixed term into the delay target:

            1/R_k_eff = 1/R_k - a_i/(2*c_i)
            R_k_eff   = R_k / (1 - a_i*R_k/(2*c_i))

        The R_k_callable for llmint8 tasks accepts (t, c_t) and returns
        this per-link effective R_k array. The caller (optimizer) must
        pass c_t when calling get_R_k for such tasks.

        If c_t is None and the callable requires it, a static fallback
        R_k is returned (using c_t = infinity, i.e. no INT8 floor penalty).
        """
        try:
            if c_t is not None:
                return self._R_k_callable(t, c_t)
            else:
                return self._R_k_callable(t)
        except TypeError:
            # callable only accepts t — return static R_k
            return self._R_k_callable(t)

    def A_k(self, eta_vec: np.ndarray) -> float:
        """
        Evaluates and returns ONLY the scalar accuracy value.
        """
        return self._accuracy_callable(eta_vec)
        
    def grad_A_k(self, eta_vec: np.ndarray) -> np.ndarray:
        """
        Evaluates and returns ONLY the gradient vector with respect to \eta.
        """
        return self._gradient_callable(eta_vec)
    
    def A_k_true(self, eta_vec: np.ndarray) -> float:
        """
        Evaluates and returns ONLY the scalar accuracy value.
        """
        return self._accuracy_callable_true(eta_vec)
        
    def is_active_at_node(self, i: int) -> bool:
        """
        Checks if task k has a computation stage at node i.
        """
        return self.b_k <= i <= self.e_k
        
    def is_active_on_link(self, i: int) -> bool:
        """
        Checks if task k transmits intermediate data over link i.
        """
        return self.b_k <= i < self.e_k