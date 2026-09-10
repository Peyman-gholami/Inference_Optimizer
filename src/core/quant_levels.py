"""
quant_levels.py
===============
Single source of truth for the eta -> quantization-level mapping.

Both compressors.py (CPU) and gpu_compressors.py (GPU) select a bit-width from
the scalar eta.  They used to do it independently and disagreed, which is how
the CPU path ended up non-monotone.  Neither module imports the other, so the
rule lives here and both import it.

The ladder
----------
Quantization can only deliver four payload ratios.  eta is the *requested*
ratio (compressed size / original size), and is continuous.

    ratio     bit-width   compression
    0.5       FP16        2x
    0.25      INT8        4x
    0.125     INT4        8x
    0.0625    INT2        16x

Mapping modes
-------------
floor  (default)  largest supported ratio <= eta.
                  Monotone in eta, and the delivered payload never exceeds the
                  requested budget -- which matters because the controller's
                  delay model assumes the link carries a_i * eta_i bytes.
                  eta below the finest rung clamps to INT2.

ceil   (legacy GPU)  smallest supported ratio >= eta.  Monotone, but overshoots
                  the budget: eta = 0.3 is delivered at ratio 0.5, i.e. 67%
                  more traffic than the delay model was told to expect.

exact  (legacy CPU)  match a rung to within 1e-6, otherwise fall back to FP16.
                  NOT monotone: requesting eta = 0.1 (aggressive) returns FP16,
                  the *highest* quality setting.  Kept only to reproduce results
                  generated before the fix.

Selectable with QUANT_ETA_MAPPING=floor|ceil|exact.

Why it matters beyond correctness
---------------------------------
Under `exact`, the set of eta values the codec responds to is four points of
half-width 1e-6.  Any continuous perturbation -- a Stein probe, a finite
difference -- misses it with probability ~1 and reads back the fallback, so the
measured accuracy is constant and every gradient estimate is exactly zero.
`floor` gives each level a positive-measure interval, which restores the signal.
"""

from __future__ import annotations

import os
from typing import List, Tuple

# (payload ratio, method tag), ascending by ratio.
QUANT_LEVELS: Tuple[Tuple[float, str], ...] = (
    (0.0625, "int2"),
    (0.125, "int4"),
    (0.25, "int8"),
    (0.5, "fp16"),
)

_TOL = 1e-6
_VALID_MODES = ("floor", "ceil", "exact")


def mapping_mode() -> str:
    mode = os.environ.get("QUANT_ETA_MAPPING", "floor").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"QUANT_ETA_MAPPING='{mode}' is invalid. Choose from: "
            f"{', '.join(_VALID_MODES)}."
        )
    return mode


def select_quant_level(eta: float, mode: str | None = None) -> str:
    """Return the method tag ('int2' | 'int4' | 'int8' | 'fp16') for *eta*."""
    mode = mode or mapping_mode()
    eta = float(eta)

    if mode == "exact":
        for ratio, tag in QUANT_LEVELS:
            if abs(eta - ratio) < _TOL:
                return tag
        return "fp16"

    if mode == "ceil":
        for ratio, tag in QUANT_LEVELS:
            if eta < ratio + _TOL:
                return tag
        return "fp16"

    # floor: largest ratio <= eta, clamped to the finest rung below the ladder
    tag = QUANT_LEVELS[0][1]
    for ratio, candidate in QUANT_LEVELS:
        if eta >= ratio - _TOL:
            tag = candidate
    return tag


def delivered_ratio(eta: float, mode: str | None = None) -> float:
    """Payload ratio actually delivered for *eta* -- what the delay model sees."""
    tag = select_quant_level(eta, mode)
    return next(r for r, t in QUANT_LEVELS if t == tag)


def describe(mode: str | None = None) -> str:
    mode = mode or mapping_mode()
    edges: List[str] = []
    for ratio, tag in QUANT_LEVELS:
        edges.append(f"{tag}@{ratio:g}")
    return f"quant eta-mapping='{mode}' over [{', '.join(edges)}]"
