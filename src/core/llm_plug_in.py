"""
Env-driven hooks for the optional LLM task backend.

When unset, the rest of the repo behaves as before (MNIST + toy τ, a).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple


def is_llm_backend() -> bool:
    return os.environ.get("INFERENCE_OPTIMIZER_BACKEND", "mnist").strip().lower() == "llm"


def is_resnet_backend() -> bool:
    return os.environ.get("INFERENCE_OPTIMIZER_BACKEND", "mnist").strip().lower() == "resnet"


def llm_use_measured_tau_a() -> bool:
    v = os.environ.get("LLM_USE_MEASURED_TAU_A", "").strip().lower()
    return v in {"1", "true", "yes"}


def maybe_measured_tau_a() -> Optional[Tuple[Dict[int, float], Dict[int, float]]]:
    """
    If LLM_USE_MEASURED_TAU_A is truthy, return (tau, a) from measured tables.

    Env:
      LLM_MEASURED_MODEL: gemma_2b | gemma_7b (optional; inferred from LLM_MODEL_NAME)
      LLM_MEASURED_BATCH: 1 | 2 | 4 (default 1)
      LLM_MEASURED_PAYLOAD: prefill_raw | prefill_serial | decode_raw | decode_serial
    """
    if not llm_use_measured_tau_a():
        return None

    from src.core.measured_llm_profile import measured_tau_and_a

    model = os.environ.get("LLM_MEASURED_MODEL", "").strip().lower()
    if model not in {"gemma_2b", "gemma_7b"}:
        mid = os.environ.get("LLM_MODEL_NAME", "").lower()
        if "gemma-2b" in mid or "gemma_2b" in mid:
            model = "gemma_2b"
        elif "7b" in mid:
            model = "gemma_7b"
        else:
            model = "gemma_2b"

    batch = int(os.environ.get("LLM_MEASURED_BATCH", "1"))
    payload = os.environ.get("LLM_MEASURED_PAYLOAD", "prefill_raw").strip().lower()
    if payload not in {"prefill_raw", "prefill_serial", "decode_raw", "decode_serial"}:
        payload = "prefill_raw"

    tau, a = measured_tau_and_a(model, batch, payload)  # type: ignore[arg-type]
    print(
        f"[measured LLM profile] model={model} batch={batch} payload={payload} "
        f"tau(s)={tau} a(bytes)={a}"
    )
    return tau, a
