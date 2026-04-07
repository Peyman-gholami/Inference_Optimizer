"""
Resolve task family by name and build :class:`~src.core.task.InferenceTask` instances.

Setup (model training, :class:`~src.core.task_callables.InferenceTaskCallables`, etc.) lives
in ``src/core/task_instances/*.py``. This module caches that result per task name so scripts
only pass the task name to :func:`get_task`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .task import InferenceTask
from .task_callables import InferenceTaskCallables
from .task_instances import toy_mlp_mnist

_REGISTRY: Dict[str, Any] = {
    "toy_mlp_mnist": toy_mlp_mnist,
}

_API_CACHE: Dict[str, Tuple[InferenceTaskCallables, Any]] = {}


def register_task(name: str, module: Any) -> None:
    _REGISTRY[name] = module


def registered_task_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_task_api_cache(name: Optional[str] = None) -> None:
    """Drop cached setup for one task name, or all names (e.g. for tests)."""
    if name is None:
        _API_CACHE.clear()
    else:
        _API_CACHE.pop(name, None)


def _module(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown task {name!r}. Registered: {', '.join(registered_task_names())}"
        )
    return _REGISTRY[name]


def _ensure_setup(name: str) -> Tuple[InferenceTaskCallables, Any]:
    if name not in _API_CACHE:
        mod = _module(name)
        _API_CACHE[name] = mod.setup_model_and_callables()
    return _API_CACHE[name]


def setup_task_api(name: str) -> Tuple[InferenceTaskCallables, Any]:
    """Return cached ``(api, extra)`` from the task instance module, creating it if needed."""
    return _ensure_setup(name)


def get_task(
    name: str,
    *,
    task_id: int = 1,
    w_k: float = 1.0,
    R_k: float = 10.0,
) -> InferenceTask:
    """
    Return a configured :class:`~src.core.task.InferenceTask` for ``name``.

    The task family's setup runs at most once per process per ``name``; further calls reuse
    the same callables so multiple tasks can share one trained model without passing ``api``.
    """
    api, _ = _ensure_setup(name)
    return _module(name).make_task(task_id=task_id, api=api, w_k=w_k, R_k=R_k)
