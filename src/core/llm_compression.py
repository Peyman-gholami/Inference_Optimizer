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
# The ladder now lives in mixed_precision.py so the ResNet backend can use it
# without importing this transformers-dependent module. eta is the target
# average bit-width relative to FP16 (target_bits = 16 * eta), so the
# delivered payload fraction equals eta.
# ---------------------------------------------------------------------------
from .mixed_precision import (
    SCHEME_ADJACENT,
    SCHEME_RESERVE,
    default_scheme,
    mixed_precision_quantize,
)


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
    llmint8          : LLM.int8()-style two-band mixed precision via the
                       mixed_precision.py ladder. eta is the payload fraction
                       (target_bits = 16 * eta), so the a_i * eta_i delay model
                       holds. Scheme from LLMINT8_SCHEME, default fp16_reserve.
    llmint8_adjacent : the ladder pinned to neighbouring rungs.
    llmint8_reserve  : the ladder pinned to FP16-protected outliers.
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

        # 'llmint8' resolves to the mixed-precision ladder, whose scheme is
        # fixed once here (LLMINT8_SCHEME) rather than per hook call.
        self._llmint8_scheme = None
        if config.strategy == "llmint8":
            self._llmint8_scheme = default_scheme()
            print(
                f"[LLMActivationCompressor] strategy=llmint8  "
                f"scheme={self._llmint8_scheme}  eta=payload fraction "
                f"(target_bits=16*eta)",
                flush=True,
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
            # eta = payload fraction; the ladder derives the two-band split.
            # mixed_precision_quantize preserves the input dtype, so the next
            # decoder layer receives what it expects.
            return mixed_precision_quantize(hidden, eta, self._llmint8_scheme)

        if self.config.strategy == "llmint8_adjacent":
            # eta = compression ratio; adjacent ladder rungs, smoothest curve.
            return mixed_precision_quantize(hidden, eta, SCHEME_ADJACENT)

        if self.config.strategy == "llmint8_reserve":
            # eta = compression ratio; FP16-protected outliers, descending base.
            return mixed_precision_quantize(hidden, eta, SCHEME_RESERVE)

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

