from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def detect_llm_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Best-effort decoder-layer detection across common HF architectures.
    We hook decoder blocks at indices to approximate "split points".
    """
    # LLaMA / Gemma-like
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if isinstance(layers, nn.ModuleList):
            return layers

    # Some models expose layers directly
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers

    # GPT-NeoX-like
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layers = model.gpt_neox.layers
        if isinstance(layers, nn.ModuleList):
            return layers

    # GPT-2 like
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        if isinstance(layers, nn.ModuleList):
            return layers

    raise ValueError("Could not detect decoder layers for this HF architecture.")


def _topk_mask_by_abs(flat_abs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns a boolean mask keeping the top-k magnitudes.
    This is used for magnitude sparsification.
    """
    if k <= 0:
        return torch.zeros_like(flat_abs, dtype=torch.bool)
    if k >= flat_abs.numel():
        return torch.ones_like(flat_abs, dtype=torch.bool)
    # Use threshold from top-k so we can build a mask in O(numel).
    vals = flat_abs.topk(k).values
    thresh = vals.min()
    return flat_abs >= thresh


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
_PRECISION_LADDER = [("drop", 0), ("int2", 2), ("int4", 4), ("int8", 8), ("fp16", 16)]
_QUANT_LEVELS = {"int8": 127.0, "int4": 7.0, "int2": 1.0}

# Minimum FP16-outlier fraction for the fp16_reserve scheme. Without this, f_high
# evaluates to exactly 0 at the phase-bottom etas (0.5 -> all int8, 0.25 -> all int4,
# 0.125 -> all int2): the scheme then keeps NO outliers in FP16 and the unprotected
# low-precision activations collapse the model (score -> 0 at those exact etas).
# Flooring guarantees outliers are always FP16-protected, in the spirit of LLM.int8()'s
# ~0.1-1% outlier rate. Costs a negligible (<0.3%) deviation of eta from the exact ratio.
_FP16_RESERVE_FLOOR = 0.005


def _quantize_band(x: torch.Tensor, precision: str) -> torch.Tensor:
    """Deterministic symmetric per-group abs-max (de)quantization.

    Returns dequantized float32 values with the same shape as `x`. Deterministic
    rounding (no stochastic noise) keeps the eta->score landscape smooth, which
    the concavity analysis relies on.
    """
    if precision == "fp16":
        return x.half().float()
    if precision == "drop":
        return torch.zeros_like(x)
    levels = _QUANT_LEVELS[precision]
    scale = (x.abs().max() / levels).clamp(min=1e-8)
    return (x / scale).round().clamp(-levels, levels) * scale


def _mixed_precision_select(eta: float, scheme: str):
    """Pick (low_precision, high_precision, f_high) for target bits b = 16*eta.

    `f_high` is the fraction of (largest-magnitude) elements placed in the
    high-precision band; the rest go in the low-precision band.
    """
    b = 16.0 * float(eta)
    if scheme == "fp16_reserve":
        if   b >= 8.0: low_bits = 8;  low = "int8"
        elif b >= 4.0: low_bits = 4;  low = "int4"
        elif b >= 2.0: low_bits = 2;  low = "int2"
        else:          low_bits = 0;  low = "drop"
        f_high = (b - low_bits) / (16.0 - low_bits)
        # Never drop FP16 outlier protection entirely (see _FP16_RESERVE_FLOOR).
        f_high = max(f_high, _FP16_RESERVE_FLOOR)
        return low, "fp16", f_high
    # adjacent-rung: bracket b between neighbouring ladder rungs.
    for i in range(len(_PRECISION_LADDER) - 1):
        lo_name, lo_bits = _PRECISION_LADDER[i]
        hi_name, hi_bits = _PRECISION_LADDER[i + 1]
        if lo_bits <= b <= hi_bits:
            f_high = (b - lo_bits) / (hi_bits - lo_bits)
            return lo_name, hi_name, f_high
    return "fp16", "fp16", 1.0  # b >= 16


def _mixed_precision_quantize(hidden: torch.Tensor, eta: float, scheme: str) -> torch.Tensor:
    """Two-tier mixed-precision activation quantization (see module note above)."""
    original_dtype = hidden.dtype
    flat = hidden.reshape(-1).float()
    N = flat.numel()

    low_prec, high_prec, f_high = _mixed_precision_select(eta, scheme)
    f_high = min(max(f_high, 0.0), 1.0)
    n_high = int(round(f_high * N))

    if n_high <= 0:
        out = _quantize_band(flat, low_prec)
    elif n_high >= N:
        out = _quantize_band(flat, high_prec)
    else:
        absf = flat.abs()
        # Threshold above which the top-n_high magnitudes lie (GPU-native).
        thresh = torch.kthvalue(absf, N - n_high).values
        hi_mask = absf > thresh
        out = torch.empty_like(flat)
        out[hi_mask] = _quantize_band(flat[hi_mask], high_prec)
        out[~hi_mask] = _quantize_band(flat[~hi_mask], low_prec)

    return out.reshape_as(hidden).to(original_dtype)


@dataclass(frozen=True)
class ActivationCompressionConfig:
    strategy: str  # magnitude | topk_per_token | random | quantization | llmint8
    # Cut-points in "layer index" space where hooks are attached.
    layer_indices: Sequence[int]
    # Seed used to make random compression deterministic across eta evaluations.
    random_seed: int = 1234


class LLMActivationCompressor:
    """
    Registers forward hooks at given decoder-layer indices and applies activation
    sparsification controlled by `eta` (a vector, one eta per hooked layer).

    The compressor keeps hooks installed; only `eta` is updated in-place.

    Strategies
    ----------
    magnitude        : top-k by absolute value, flat over the full tensor
    topk_per_token   : top-k per token position along hidden_dim
    random           : Bernoulli mask with deterministic seed
    quantization     : symmetric uniform quantisation snapped to nearest bitwidth
    llmint8          : GPU-native LLM.int8()-style mixed-precision via
                       GPULLMInt8Compressor (gpu_compressors.py).
                       eta is used as the outlier_ratio (top-eta fraction kept
                       in FP16; remaining values quantised to INT8).
                       Requires COMPRESSOR_BACKEND=gpu and gpu_compressors.py.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ActivationCompressionConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.layers = detect_llm_decoder_layers(model)

        self.d = len(config.layer_indices)
        if self.d <= 0:
            raise ValueError("ActivationCompressionConfig.layer_indices must be non-empty.")

        # Persist eta as floats for cheap .item()-free access in hooks.
        self._eta: List[float] = [1.0 for _ in range(self.d)]

        # GPU-native LLMInt8 compressor — instantiated once and reused across
        # all hook calls to avoid repeated object creation in the forward pass.
        self._gpu_llmint8 = None
        if config.strategy == "llmint8":
            try:
                from .gpu_compressors import GPULLMInt8Compressor
                self._gpu_llmint8 = GPULLMInt8Compressor()
                print(
                    "[LLMActivationCompressor] strategy=llmint8  "
                    "backend=gpu  path=GPU-native (no CPU round-trip)",
                    flush=True,
                )
            except ImportError:
                raise ImportError(
                    "LLM_ACTIVATION_STRATEGY=llmint8 requires gpu_compressors.py. "
                    "Ensure gpu_compressors.py is on sys.path and "
                    "COMPRESSOR_BACKEND=gpu is set."
                )

        handles: List[torch.utils.hooks.RemovableHandle] = []
        for local_i, layer_idx in enumerate(config.layer_indices):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise ValueError(
                    f"layer index {layer_idx} out of range (n_layers={len(self.layers)})"
                )
            handles.append(self.layers[layer_idx].register_forward_hook(self._make_hook(local_i)))
        self._handles = handles

    def remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def set_eta(self, eta: torch.Tensor) -> None:
        if eta.numel() != self.d:
            raise ValueError(f"Expected eta of length {self.d}, got {eta.numel()}")
        # Clamp to [0,1] to keep sparsifiers well-behaved.
        eta_cpu = eta.detach().to(torch.float32).cpu()
        self._eta = [float(torch.clamp(v, 0.0, 1.0).item()) for v in eta_cpu]

    def _compress(self, hidden: torch.Tensor, eta: float) -> torch.Tensor:
        if eta >= 1.0:
            return hidden
        if eta <= 0.0:
            return torch.zeros_like(hidden)

        if self.config.strategy == "llmint8":
            # GPU-native LLM.int8()-style mixed-precision compression.
            # eta is used as the outlier_ratio: top-eta fraction of elements
            # are kept in FP16, the rest are quantised to INT8.
            # The tensor stays on its device throughout — no CPU round-trip.
            # Clamp eta to [0, 1] to satisfy GPULLMInt8Compressor's quantile range.
            original_dtype = hidden.dtype
            outlier_ratio = float(max(0.0, min(1.0, eta)))
            params = [outlier_ratio, 'fp16', 'int8']
            comp_data = self._gpu_llmint8.compress(hidden, params)
            restored = self._gpu_llmint8.decompress(comp_data, device=hidden.device)
            # GPULLMInt8Compressor dequantises to float32. Cast back to the
            # model's working dtype (float16 for Gemma/Llama on GPU) so the
            # next decoder layer receives the expected dtype.
            return restored.to(original_dtype)

        if self.config.strategy == "llmint8_adjacent":
            # eta = compression ratio; adjacent ladder rungs, smoothest curve.
            return _mixed_precision_quantize(hidden, eta, "adjacent")

        if self.config.strategy == "llmint8_reserve":
            # eta = compression ratio; FP16-protected outliers, descending base.
            return _mixed_precision_quantize(hidden, eta, "fp16_reserve")

        if self.config.strategy == "quantization":
            # Fake (post-training) activation quantization controlled by eta.
            #
            # We interpret eta as a "compression ratio" relative to the original
            # activation bit-width:
            #   - if original dtype is FP16/BF16 (16-bit): eta≈1.0->16-bit, 0.5->8-bit, 0.25->4-bit
            #   - if original dtype is FP32 (32-bit):       eta≈1.0->32-bit, 0.5->16-bit, 0.25->8-bit
            #
            # User requirement: "clamp eta to closest compression", e.g. eta=0.50 -> 8-bit
            # (in the common FP16 case).
            base_bits = int(hidden.element_size() * 8)
            if base_bits <= 0:
                return hidden

            candidates = [base_bits, max(1, base_bits // 2), max(1, base_bits // 4)]
            # Convert candidates to compression ratios in eta-space.
            ratios = [float(b) / float(base_bits) for b in candidates]
            # Snap eta to the closest ratio, then quantize to the corresponding bitwidth.
            closest_idx = int(np.argmin([abs(float(eta) - r) for r in ratios]))
            target_bits = int(candidates[closest_idx])

            # If the chosen bitwidth equals the original, do nothing.
            if target_bits >= base_bits:
                return hidden

            # Symmetric uniform quantization with per-tensor scale.
            qmax = float((1 << (target_bits - 1)) - 1)
            x = hidden.to(torch.float32)
            max_abs = float(x.abs().max().item())
            if max_abs == 0.0:
                return hidden
            scale = max_abs / qmax
            x_q = torch.round(x / scale).clamp(-qmax, qmax) * scale
            return x_q.to(hidden.dtype)

        if self.config.strategy == "magnitude":
            flat = hidden.flatten()
            k = max(1, int(eta * flat.numel()))
            mask = _topk_mask_by_abs(flat.abs(), k=k)
            return (flat * mask.to(flat.dtype)).reshape_as(hidden)

        if self.config.strategy == "topk_per_token":
            # Keep top-k hidden dimensions per token position.
            # Expected hidden: (batch, seq_len, hidden_dim) -> mask over last dim.
            D = hidden.size(-1)
            k = max(1, int(eta * D))
            abs_h = hidden.abs()
            _, idx = abs_h.topk(k, dim=-1)
            mask = torch.zeros_like(hidden)
            mask.scatter_(-1, idx, 1.0)
            return hidden * mask

        if self.config.strategy == "random":
            # Bernoulli mask with deterministic generator seed.
            g = torch.Generator(device=hidden.device)
            g.manual_seed(int(self.config.random_seed))
            mask = torch.rand(hidden.shape, generator=g, device=hidden.device) < float(eta)
            return hidden * mask.to(hidden.dtype)

        raise ValueError(f"Unknown activation compression strategy: {self.config.strategy}")

    def _make_hook(self, local_i: int):
        def hook(_module: nn.Module, _inputs, output):
            # HF decoder blocks typically return:
            #   - hidden_states tensor, or
            #   - tuple(hidden_states, past_key_value, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                compressed = self._compress(hidden, self._eta[local_i])
                return (compressed, *output[1:])
            return self._compress(output, self._eta[local_i])

        return hook

