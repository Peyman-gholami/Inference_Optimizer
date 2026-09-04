"""
compressor_registry.py  —  Central compressor dispatch for the Inference Optimizer
===================================================================================

Single source of truth for the COMPRESSOR_BACKEND environment variable.
All modules that need a compressor instance import from here instead of
duplicating the try/except import logic and backend-selection code.

Environment variable
--------------------
    COMPRESSOR_BACKEND=cpu   (default)  use compressors.py      — CPU round-trip
    COMPRESSOR_BACKEND=gpu              use gpu_compressors.py  — GPU-native, no CPU transfer

The variable is read once at module import time.  Changing it after import
has no effect.

Public API
----------
    get_compressor(name)      -> compressor instance (CPU or GPU depending on backend)
    is_gpu_backend()          -> bool
    is_gpu_compressor(obj)    -> bool  (True if obj is a GPUBaseCompressor instance)
    COMPRESSOR_BACKEND        -> str   ('cpu' or 'gpu')

Usage
-----
    from compressor_registry import get_compressor, is_gpu_compressor, COMPRESSOR_BACKEND

    comp = get_compressor('topk')          # returns TopKCompressor or GPUTopKCompressor
    comp = get_compressor('llmint8')       # returns LLMInt8Compressor or GPULLMInt8Compressor

    if is_gpu_compressor(comp):
        # GPU-native path — tensor stays on device
        ...
    else:
        # CPU path — move to CPU, compress, move back
        ...
"""

import os

# ============================================================================
# CPU compressors  (compressors.py)
# ============================================================================
try:
    from compressors import (
        BaseCompressor,
        TopKCompressor,
        get_compressor as _cpu_get_compressor,
    )
    _CPU_AVAILABLE = True
except ImportError:
    _CPU_AVAILABLE = False
    BaseCompressor    = object   # dummy so isinstance checks don't crash
    TopKCompressor    = None
    _cpu_get_compressor = None

# ============================================================================
# GPU-native compressors  (gpu_compressors.py)
# ============================================================================
try:
    from gpu_compressors import (
        GPUBaseCompressor,
        get_gpu_compressor as _gpu_get_compressor,
    )
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE    = False
    GPUBaseCompressor = None
    _gpu_get_compressor = None

# ============================================================================
# Read COMPRESSOR_BACKEND once at import time
# ============================================================================
COMPRESSOR_BACKEND: str = os.environ.get('COMPRESSOR_BACKEND', 'cpu').strip().lower()

if COMPRESSOR_BACKEND not in ('cpu', 'gpu'):
    raise ValueError(
        f"COMPRESSOR_BACKEND='{COMPRESSOR_BACKEND}' is invalid. "
        f"Choose from: cpu, gpu."
    )

if COMPRESSOR_BACKEND == 'gpu' and not _GPU_AVAILABLE:
    raise ImportError(
        "COMPRESSOR_BACKEND=gpu requested but gpu_compressors.py could not be "
        "imported.  Ensure gpu_compressors.py is on sys.path."
    )

if COMPRESSOR_BACKEND == 'cpu' and not _CPU_AVAILABLE:
    raise ImportError(
        "COMPRESSOR_BACKEND=cpu but compressors.py could not be imported.  "
        "Ensure compressors.py is on sys.path."
    )

# ============================================================================
# Public helpers
# ============================================================================

def is_gpu_backend() -> bool:
    """Return True if COMPRESSOR_BACKEND=gpu is active."""
    return COMPRESSOR_BACKEND == 'gpu'


def is_gpu_compressor(obj) -> bool:
    """
    Return True if *obj* is a GPU-native compressor instance
    (i.e. a subclass of GPUBaseCompressor from gpu_compressors.py).
    Safe to call even when gpu_compressors.py is not available.
    """
    if not _GPU_AVAILABLE or GPUBaseCompressor is None:
        return False
    return isinstance(obj, GPUBaseCompressor)


def get_compressor(name: str):
    """
    Instantiate a compressor by name, respecting COMPRESSOR_BACKEND.

    Parameters
    ----------
    name : str
        One of: 'topk', 'quantization' / 'quant', 'llmint8' / 'llm_int8'

    Returns
    -------
    BaseCompressor or GPUBaseCompressor instance depending on COMPRESSOR_BACKEND.

    Raises
    ------
    ValueError  if *name* is not recognised.
    ImportError if the required library (compressors.py / gpu_compressors.py)
                is not available.

    Notes
    -----
    Callers that need to know which execution path will be used can call
    is_gpu_compressor(result) after receiving the instance, or check
    is_gpu_backend() upfront.
    """
    if COMPRESSOR_BACKEND == 'gpu':
        if not _GPU_AVAILABLE:
            raise ImportError(
                "COMPRESSOR_BACKEND=gpu but gpu_compressors.py is not importable."
            )
        return _gpu_get_compressor(name)
    else:
        if not _CPU_AVAILABLE:
            raise ImportError(
                "COMPRESSOR_BACKEND=cpu but compressors.py is not importable."
            )
        return _cpu_get_compressor(name)


def describe_compressor(instance, module_tag: str = '') -> str:
    """
    Return a human-readable one-line description of a compressor instance.

    Parameters
    ----------
    instance   : compressor object with a get_name() method
    module_tag : optional prefix shown in brackets, e.g. 'ResNet backend'

    Returns
    -------
    str suitable for print(..., flush=True)

    Example output
    --------------
    [ResNet backend] compressor=LLMInt8              backend=cpu  path=CPU (compress→cpu→decompress→device)
    [ResNet backend] compressor=GPULLMInt8           backend=gpu  path=GPU-native (no CPU round-trip)
    """
    backend_actual = 'gpu' if is_gpu_compressor(instance) else 'cpu'
    execution_path = (
        'GPU-native (no CPU round-trip)'
        if backend_actual == 'gpu'
        else 'CPU (compress→cpu→decompress→device)'
    )
    name = instance.get_name() if hasattr(instance, 'get_name') else type(instance).__name__
    tag  = f"[{module_tag}] " if module_tag else ''
    return (
        f"{tag}compressor={name:<20} "
        f"backend={backend_actual}  "
        f"path={execution_path}"
    )
