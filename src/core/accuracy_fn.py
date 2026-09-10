"""
CompressedModel: nn.Module with persistent hooks, eta updated in-place.
AccuracyFunction: measures accuracy on a (compressed) model, optionally
                  swapping theta in-place and restoring afterwards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .stein_oracles import MLP, get_flat_params, set_flat_params
from .toy_A import grad_oracle

# Optional: import the unified compressor library.
# Falls back gracefully if compressors.py is not on the path.
try:
    from .compressors import BaseCompressor, TopKCompressor, get_compressor as _get_compressor
    _COMPRESSORS_AVAILABLE = True
except ImportError:
    _COMPRESSORS_AVAILABLE = False
    BaseCompressor = object          # dummy so type hints don't crash

# Central compressor registry — reads COMPRESSOR_BACKEND once at import time.
# Provides get_compressor(), is_gpu_compressor(), is_gpu_backend(),
# describe_compressor(), and the COMPRESSOR_BACKEND constant.
from .compressor_registry import (
    get_compressor      as _registry_get_compressor,
    is_gpu_compressor   as _is_gpu_compressor,
    is_gpu_backend      as _is_gpu_backend,
    describe_compressor as _describe_compressor,
    COMPRESSOR_BACKEND  as _COMPRESSOR_BACKEND,
)


# ─────────────────────────────────────────────
# 1.  Compressor helpers
# ─────────────────────────────────────────────

def _make_compressor(compressor):
    """
    Resolve *compressor* to a compressor instance, respecting COMPRESSOR_BACKEND.

    COMPRESSOR_BACKEND=cpu (default)  ->  compressors.py  (CPU round-trip)
    COMPRESSOR_BACKEND=gpu            ->  gpu_compressors.py  (GPU-native)

    Accepted values for *compressor*
    ---------------------------------
    None / 'topk'           -> TopKCompressor  (default)
    'quantization'          -> QuantizationCompressor
    BaseCompressor instance -> used as-is (CPU path)
    GPUBaseCompressor inst. -> used as-is (GPU path)

    'llmint8'               -> LadderInt8Compressor
    'llmint8_outlier'       -> rejected (see below)

    'llmint8' is accepted here: the mixed_precision.py ladder flattens the
    tensor and makes no layout assumption, so it applies to MLP activations
    as readily as to transformer hidden states.  'llmint8_outlier' remains
    excluded — its row-wise AbsMax scaling assumes a transformer
    [batch, seq_len, hidden_dim] layout, whereas MLP activations are 1-D
    vectors after each ReLU.
    """
    if not _COMPRESSORS_AVAILABLE and _COMPRESSOR_BACKEND == 'cpu':
        instance = _BuiltinTopKCompressor()
        print(_describe_compressor(instance, 'MLP backend') +
              '  (fallback — compressors.py not found)', flush=True)
        return instance

    if isinstance(compressor, str):
        name = compressor.strip().lower()
        if name in ('llmint8_outlier', 'llm_int8_outlier'):
            raise ValueError(
                f"COMPRESSOR_TYPE='{compressor}' is not supported for the MLP backend. "
                f"Choose from: topk, quantization, llmint8.  "
                f"(The outlier-fraction LLMInt8 assumes transformer hidden states; "
                f"use 'llmint8' for the layout-agnostic mixed-precision ladder.)"
            )
        instance = _registry_get_compressor(name)
    elif compressor is None:
        instance = _registry_get_compressor('topk')
    else:
        instance = compressor

    print(_describe_compressor(instance, 'MLP backend'), flush=True)
    return instance


class _BuiltinTopKCompressor:
    """
    Minimal fallback used when compressors.py is not importable.
    Matches the BaseCompressor interface but does not require the library.
    """
    def compress(self, tensor, compression_params):
        eta = float(compression_params)
        if eta >= 1.0:
            return {'passthrough': tensor}
        flat = tensor.flatten()
        k = max(1, int(eta * flat.numel()))
        thresh = flat.abs().topk(k).values.min()
        sparse = (flat * (flat.abs() >= thresh)).reshape(tensor.shape)
        return {'passthrough': sparse}

    def decompress(self, compressed, device='cpu'):
        return compressed['passthrough'].to(device)

    def get_compressed_size(self, compressed):
        t = compressed['passthrough']
        return t.numel() * t.element_size()

    def get_name(self):
        return "TopK(builtin)"


def _compress_decompress(compressor, x: torch.Tensor, eta: float) -> torch.Tensor:
    """
    Round-trip a tensor through compress/decompress using *eta* as the
    compression parameter.  Returns a tensor on the same device as *x*.

    For TopKCompressor / GPUTopKCompressor:
        eta is the keep-fraction (k_ratio).
    For QuantizationCompressor / GPUQuantizationCompressor:
        eta is mapped to the nearest supported bit-width level
        (0.5->FP16, 0.25->INT8, 0.125->INT4, 0.0625->INT2).
    For LLMInt8Compressor / GPULLMInt8Compressor:
        eta is used as the outlier ratio, with FP16 outliers and INT8
        regular values as defaults.

    GPU-native compressors (GPUBaseCompressor subclasses) operate entirely
    on the input tensor's device — no .cpu() transfer is performed.
    CPU compressors (BaseCompressor subclasses) use the original CPU
    round-trip path.
    """
    # Clamp eta to [0, 1] — the Stein oracle perturbs eta with x ± sigma*z
    # which can push it outside this range. LLMInt8 uses eta as an
    # outlier_ratio passed to torch.quantile() which requires [0, 1].
    eta = max(0.0, min(1.0, eta))

    if eta >= 1.0:
        return x                              # no compression needed

    device = x.device

    # Every compressor takes the scalar eta. For llmint8 the mixed_precision.py
    # ladder reads it as target bit-width / 16, so the delivered payload
    # fraction is eta and no remap is needed here.
    params = eta

    if _is_gpu_compressor(compressor):
        # GPU-native path — tensor stays on device throughout
        compressed = compressor.compress(x.detach(), params)
        return compressor.decompress(compressed, device=device)
    else:
        # CPU path — original behaviour
        tensor_cpu = x.detach().cpu()
        compressed = compressor.compress(tensor_cpu, params)
        restored   = compressor.decompress(compressed, device='cpu')
        return restored.to(device)


# ─────────────────────────────────────────────
# 2.  CompressedModel
#     Registers hooks once at construction.
#     eta is updated in-place via set_eta().
# ─────────────────────────────────────────────

class CompressedModel(nn.Module):
    """
    Wraps base_model with persistent activation-compression hooks.
    Hooks are registered once and never removed — update eta in-place.

    Parameters
    ----------
    base_model  : nn.Module
    eta         : Tensor (d,) -- initial compression ratios (one per cut-point)
    cut_modules : list of nn.Module -- layers after which to compress.
                  Defaults to all ReLU/Tanh/GELU/Sigmoid layers.
    compressor  : None | str | BaseCompressor
                  Compression backend to use at every cut-point.
                  - None / 'topk'      -> TopKCompressor  (default)
                  - 'quantization'     -> QuantizationCompressor
                  - 'llmint8'          -> LLMInt8Compressor
                  - BaseCompressor obj -> used directly

    Usage
    -----
    cm = CompressedModel(model, eta)                        # TopK (default)
    cm = CompressedModel(model, eta, compressor='quantization')
    cm.set_eta(new_eta)     # update in-place, no new hooks
    out = cm(x)
    """

    def __init__(self, base_model: nn.Module, eta: torch.Tensor,
                 cut_modules=None, compressor=None):
        super().__init__()
        self.base_model = base_model
        self.eta        = eta.clone()
        self.compressor = _make_compressor(compressor)

        if cut_modules is None:
            cut_modules = [m for m in base_model.modules()
                           if isinstance(m, (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid))]
        self.cut_modules = cut_modules
        self.d           = len(cut_modules)
        assert len(eta) == self.d, \
            f"eta length {len(eta)} does not match {self.d} cut-points"

        # register hooks once — they read from self.eta at forward time
        for i, m in enumerate(self.cut_modules):
            m.register_forward_hook(self._make_hook(i))

    def _make_hook(self, i: int):
        def hook(module, input, output):
            return _compress_decompress(self.compressor, output, self.eta[i].item())
        return hook

    def set_eta(self, eta: torch.Tensor):
        assert len(eta) == self.d
        self.eta = eta.clone()

    def set_compressor(self, compressor):
        """
        Swap the compressor at runtime without re-registering hooks.
        Accepts the same values as the constructor's compressor parameter.
        """
        self.compressor = _make_compressor(compressor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


# ─────────────────────────────────────────────
# 3.  AccuracyFunction
# ─────────────────────────────────────────────

class AccuracyFunction:
    """
    Measures classification accuracy of a CompressedModel on a test set.

    Parameters
    ----------
    cm          : CompressedModel -- model with hooks already installed
    test_loader : DataLoader
    n_samples   : int or None -- random subset size for speed (None = full set)
    device      : torch.device

    Usage
    -----
    A = AccuracyFunction(cm, test_loader, n_samples=512)

    A(eta)                   # evaluate at eta, current theta
    A(eta, flat_theta=th)    # evaluate at eta and th, restores theta after
    """

    def __init__(self, cm: CompressedModel, test_loader: DataLoader,
                 n_samples=None, device=None):
        self.cm          = cm
        self.test_loader = test_loader
        self.n_samples   = n_samples
        self.device      = device or next(cm.parameters()).device

        if n_samples is not None:
            xs, ys = zip(*list(test_loader))
            self._all_x = torch.cat(xs)
            self._all_y = torch.cat(ys)

    def _sample_batch(self):
        idx = torch.randperm(len(self._all_x))[:self.n_samples]
        return self._all_x[idx], self._all_y[idx]

    def _eval(self) -> torch.Tensor:
        self.cm.eval()
        with torch.no_grad():
            if self.n_samples is not None:
                x, y    = self._sample_batch()
                x, y    = x.to(self.device), y.to(self.device)
                correct = (self.cm(x).argmax(1) == y).sum().item()
                total   = len(y)
            else:
                correct = total = 0
                for x, y in self.test_loader:
                    x, y    = x.to(self.device), y.to(self.device)
                    correct += (self.cm(x).argmax(1) == y).sum().item()
                    total   += y.size(0)
        return torch.tensor(correct / total)

    def __call__(self, eta: torch.Tensor,
                 flat_theta: torch.Tensor = None) -> torch.Tensor:
        self.cm.set_eta(eta)

        if flat_theta is not None:
            original = get_flat_params(self.cm.base_model)
            set_flat_params(self.cm.base_model, flat_theta)
            try:
                acc = self._eval()
            finally:
                set_flat_params(self.cm.base_model, original)
        else:
            acc = self._eval()

        return acc


# ─────────────────────────────────────────────
# 4.  Demo
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True,  download=True, transform=tf),
        batch_size=128, shuffle=True)
    test_loader  = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=tf),
        batch_size=256, shuffle=False)

    # -- train --
    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training MLP on MNIST (3 epochs)...")
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)

    # construct once
    eta_full = torch.ones(2)   # MLP has 2 activation layers
    cm       = CompressedModel(model, eta_full)
    A        = AccuracyFunction(cm, test_loader, device=device)
    A_fast   = AccuracyFunction(cm, test_loader, n_samples=512, device=device)
    d        = cm.d
    print(f"Cut-points: {d}")

    # -- basic checks --
    eta_heavy = torch.full((d,), 0.1)
    print(f"\nNo compression    (eta=1.0): {A(eta_full):.4f}")
    print(f"Heavy compression (eta=0.1): {A(eta_heavy):.4f}")

    # verify theta restore
    _ = A(eta_full, flat_theta=torch.zeros_like(flat_theta))
    print(f"After dummy theta, acc restored: {A(eta_full):.4f}")

    # -- sweep eta_0 --
    print("\nSweeping eta_0 (others=1.0):")
    for v in torch.linspace(0, 1, 11).tolist():
        eta = eta_full.clone(); eta[0] = v
        acc = A_fast(eta)
        bar = "█" * int(acc * 30)
        print(f"  eta_0={v:.1f}  acc={acc:.4f}  {bar}")

    # -- Stein gradient w.r.t. eta --
    print("\nStein gradient of A w.r.t. eta (sigma=0.05, N=50):")
    def f_eta(e):   return A_fast(e, flat_theta=flat_theta)
    g = grad_oracle(f_eta, eta_full * 0.8, sigma=0.05, N=50)
    print(f"  grad_eta A = {g.numpy().round(4)}")
"""
    # -- Stein gradient w.r.t. theta --
    print("\nStein gradient of A w.r.t. theta (sigma=0.01, N=100):")
    eta_mid = torch.full((d,), 0.8)
    def f_theta(th): return A_fast(eta_mid, flat_theta=th)
    g_th = grad_oracle(f_theta, flat_theta, sigma=0.01, N=100)
    print(f"  ||grad_theta A|| = {g_th.norm():.4f}")
"""

if __name__ == "__main__":
    main()