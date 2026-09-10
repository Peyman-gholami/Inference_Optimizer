"""
mixed_precision.py
==================
Single source of truth for the eta -> mixed-precision mapping used by the
LLM.int8-style compressor, and the companion to quant_levels.py.

The rule, and why it is stated this way
---------------------------------------
The controller's delay model assumes a link carries ``a_i * eta_i`` bytes, so
eta must be the *payload fraction*. This module defines eta that way: eta is
the target average bit-width relative to the FP16 activation baseline,

    b = 16 * eta        bits per element

and the two-band split is then solved for the mixing fraction that hits b:

    f_high = (b - lo_bits) / (hi_bits - lo_bits)

which delivers f_high*hi + (1-f_high)*lo = b exactly. Payload fraction is
therefore b/16 = eta by construction, for any eta in [0, 1].

This is the inverse of parameterising by the FP16-outlier fraction. With an
outlier fraction f the payload is a*(1+f)/2, which bottoms out at 0.5*a and
does not equal a*f -- so a delay model of a_i*eta_i is wrong unless eta is
mapped through f = 2*eta - 1 (the single fp16/int8 rung of the ladder below).
The older outlier-fraction compressors are still reachable as
``llmint8_outlier`` for reproducing runs that used them; they are NOT
compatible with the a*eta delay model.

Byte accounting
---------------
``achieved_bits`` reports the average bit-width actually delivered after
n_high is rounded to a whole number of elements. It deliberately does NOT
count the ~1 bit/element band mask: eta then stays exactly the delay model's
parameter, matching quant_levels.py's invariant that delivered payload never
exceeds the requested budget. A real encoder's mask (and any serialisation
overhead) is out of scope here -- measure it if you need wire-accurate bytes.

Schemes
-------
``adjacent``     brackets b between neighbouring ladder rungs, so only one
                 precision boundary moves at a time. Delivers exactly b.
``fp16_reserve`` keeps the high band at FP16 always (protected outliers, in
                 the spirit of LLM.int8()), stepping the low band down the
                 ladder. f_high is floored at _FP16_RESERVE_FLOOR so the
                 outliers are never left unprotected, which costs a <0.3%
                 upward deviation of the delivered ratio from eta.

Select with LLMINT8_SCHEME=adjacent|fp16_reserve (default fp16_reserve).

The ladder implementation below is by mqsirera, moved here verbatim from
llm_compression.py so that resnet_task_callables.py can use it without
importing the transformers-dependent LLM module.
"""

import os

import torch

SCHEME_ADJACENT = "adjacent"
SCHEME_RESERVE  = "fp16_reserve"
_SCHEMES        = (SCHEME_ADJACENT, SCHEME_RESERVE)


def default_scheme() -> str:
    """Scheme used when a caller does not name one. LLMINT8_SCHEME overrides."""
    s = os.environ.get("LLMINT8_SCHEME", SCHEME_RESERVE).strip().lower()
    if s not in _SCHEMES:
        raise ValueError(
            f"LLMINT8_SCHEME={s!r} is not one of {_SCHEMES}."
        )
    return s


# ---------------------------------------------------------------------------
# Mixed-precision activation compression where eta IS the compression ratio.
#
# Two strategies share the machinery below. In both, eta is the target average
# bit-width relative to the FP16 activation baseline (target_bits = 16 * eta),
# so eta is a true compression ratio on the same [0, 1] axis as topk_per_token.
# Elements are split by magnitude into a high-precision and a low-precision
# band; the largest-magnitude ("outlier") elements always get high precision.
#
#   llmint8_adjacent : the two bands are *neighbouring* rungs of the ladder
#                      (fp16/int8, int8/int4, int4/int2, int2/drop). Only the
#                      boundary between two adjacent precisions ever moves, so
#                      composition is continuous across eta -> smoothest curve.
#   llmint8_reserve  : the high band is *always* FP16 (protected outliers); the
#                      low band steps down int8 -> int4 -> int2 -> drop as eta
#                      decreases. Keeps a permanent FP16 slice at the cost of a
#                      composition jump each time the low band changes rung.
# ---------------------------------------------------------------------------

# (name, bits) rungs ordered low -> high.
_PRECISION_LADDER = [("drop", 0), ("int2", 2), ("int4", 4), ("int8", 8),
                     ("fp16", 16), ("fp32", 32)]
_QUANT_LEVELS = {"int8": 127.0, "int4": 7.0, "int2": 1.0}

# Minimum FP16-outlier fraction for the fp16_reserve scheme. Without this, f_high
# evaluates to exactly 0 at the phase-bottom etas (0.5 -> all int8, 0.25 -> all int4,
# 0.125 -> all int2): the scheme then keeps NO outliers in FP16 and the unprotected
# low-precision activations collapse the model (score -> 0 at those exact etas).
# Flooring guarantees outliers are always FP16-protected, in the spirit of LLM.int8()'s
# ~0.1-1% outlier rate. Costs a negligible (<0.3%) deviation of eta from the exact ratio.
_FP16_RESERVE_FLOOR = 0.005


def quantize_band(x: torch.Tensor, precision: str) -> torch.Tensor:
    """Deterministic symmetric per-group abs-max (de)quantization.

    Returns dequantized float32 values with the same shape as `x`. Deterministic
    rounding (no stochastic noise) keeps the eta->score landscape smooth, which
    the concavity analysis relies on.
    """
    if precision == "fp32":
        return x
    if precision == "fp16":
        return x.half().float()
    if precision == "drop":
        return torch.zeros_like(x)
    levels = _QUANT_LEVELS[precision]
    scale = (x.abs().max() / levels).clamp(min=1e-8)
    return (x / scale).round().clamp(-levels, levels) * scale


def select_mixed_precision(eta: float, scheme: str, base_bits: int = 16):
    """Pick (low_precision, high_precision, f_high) for target bits b = base_bits*eta.

    `f_high` is the fraction of (largest-magnitude) elements placed in the
    high-precision band; the rest go in the low-precision band.

    `base_bits` is the width of the uncompressed activation, so eta = 1 means
    "no compression" whatever the dtype: 16 for FP16 hidden states (the
    original LLM setting), 32 for the FP32 activations the vision backend
    produces. Anchoring to the dtype is what makes the delivered payload
    fraction equal eta, which the a_i * eta_i delay model requires.
    """
    b = float(base_bits) * float(eta)
    rungs = [(n, k) for n, k in _PRECISION_LADDER if k <= base_bits]
    high_name, high_bits = rungs[-1]
    if scheme == SCHEME_RESERVE:
        # Low band is the largest rung strictly below base_bits that fits b.
        low_name, low_bits = rungs[0]
        for n, k in rungs[:-1]:
            if b >= k:
                low_name, low_bits = n, k
        f_high = ((b - low_bits) / (high_bits - low_bits)
                  if high_bits > low_bits else 1.0)
        # Never drop the high-precision outlier protection entirely
        # (see _FP16_RESERVE_FLOOR).
        f_high = max(f_high, _FP16_RESERVE_FLOOR)
        return low_name, high_name, f_high
    # adjacent-rung: bracket b between neighbouring ladder rungs.
    for i in range(len(rungs) - 1):
        lo_name, lo_bits = rungs[i]
        hi_name, hi_bits = rungs[i + 1]
        if lo_bits <= b <= hi_bits:
            f_high = (b - lo_bits) / (hi_bits - lo_bits)
            return lo_name, hi_name, f_high
    return high_name, high_name, 1.0  # b >= base_bits


def mixed_precision_quantize(hidden: torch.Tensor, eta: float, scheme: str,
                             base_bits: int = None) -> torch.Tensor:
    """Two-tier mixed-precision activation quantization (see module note above)."""
    original_dtype = hidden.dtype
    if base_bits is None:
        base_bits = 8 * hidden.element_size()
    flat = hidden.reshape(-1).float()
    N = flat.numel()

    low_prec, high_prec, f_high = select_mixed_precision(eta, scheme, base_bits)
    f_high = min(max(f_high, 0.0), 1.0)
    n_high = int(round(f_high * N))

    if n_high <= 0:
        out = quantize_band(flat, low_prec)
    elif n_high >= N:
        out = quantize_band(flat, high_prec)
    else:
        absf = flat.abs()
        # Threshold above which the top-n_high magnitudes lie (GPU-native).
        thresh = torch.kthvalue(absf, N - n_high).values
        hi_mask = absf > thresh
        out = torch.empty_like(flat)
        out[hi_mask] = quantize_band(flat[hi_mask], high_prec)
        out[~hi_mask] = quantize_band(flat[~hi_mask], low_prec)

    return out.reshape_as(hidden).to(original_dtype)


def achieved_bits(eta: float, scheme: str, n_elements: int,
                  base_bits: int = 16) -> float:
    """Average bit-width per element actually delivered at *eta*.

    Mirrors mixed_precision_quantize's rounding of n_high, so byte counts
    agree with what the compressor really produced. Excludes the band mask
    (see the module docstring).
    """
    low_prec, high_prec, f_high = select_mixed_precision(eta, scheme, base_bits)
    f_high = min(max(f_high, 0.0), 1.0)
    bits   = dict(_PRECISION_LADDER)
    if n_elements <= 0:
        return 0.0
    n_high = int(round(f_high * n_elements))
    n_high = min(max(n_high, 0), n_elements)
    return (n_high * bits[high_prec]
            + (n_elements - n_high) * bits[low_prec]) / n_elements


def payload_bytes(eta: float, scheme: str, n_elements: int,
                  base_bits: int = 16) -> int:
    """Bytes delivered for *n_elements* activations at *eta*."""
    import math
    return math.ceil(
        n_elements * achieved_bits(eta, scheme, n_elements, base_bits) / 8.0)
