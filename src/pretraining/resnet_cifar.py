"""
CSI-aware single-task compression-aware pretraining on CIFAR-10 / ResNet.

Same J_{eta*}=0 algorithm as `compression_aware.py`, but:
  * the model is a *pretrained* CIFAR-10 ResNet (we adapt it rather than train
    from scratch -- "compression-aware fine-tuning");
  * the cut-points are residual-stage boundaries (feature maps), not dense MLP
    activations, so compression is top-k over each sample's feature map;
  * it is device-agnostic (cuda -> mps -> cpu) so the same code runs on the M3
    GPU locally and on a CUDA cluster.

Cut-points (d=2): outputs of `layer1` (16xHxW) and `layer2` (32xH/2xW/2) -- the
two points where, in the distributed system, an activation tensor would be
compressed and shipped to the next node before the final stage + classifier.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ..core.accuracy_fn import CompressedModel
from .compression_aware import ChannelSampler, csi_aware_eta_star

# chenyaofo/pytorch-cifar-models normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def select_device(prefer: Optional[str] = None) -> torch.device:
    """cuda -> mps -> cpu, unless `prefer` forces a choice."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-sample top-k sparsification (compression of a conv feature map)
# ---------------------------------------------------------------------------
def topk_sparsify_per_sample(x: torch.Tensor, eta: float) -> torch.Tensor:
    """Keep, *for each sample in the batch*, the ceil(eta * n) largest-magnitude
    entries of its (flattened) feature map; zero the rest.

    Unlike `topk_sparsify` (which flattens the whole batch together), this matches
    deployment: every inference compresses its own activation independently.
    Differentiable in theta (gradient flows through the kept entries); eta detached.
    """
    if eta >= 1.0:
        return x
    if eta <= 0.0:
        return torch.zeros_like(x)
    b = x.shape[0]
    flat = x.reshape(b, -1)                       # (B, F)
    n = flat.shape[1]
    k = max(1, int(eta * n))
    # per-row threshold = k-th largest magnitude (topk is MPS/CUDA-supported)
    kth = flat.abs().topk(k, dim=1).values[:, -1:]   # (B, 1)
    mask = (flat.abs() >= kth)
    return (flat * mask).reshape_as(x)


def topk_sparsify_per_sample_renorm(x: torch.Tensor, eta: float) -> torch.Tensor:
    """Top-k as above, but then RESCALE each sample's surviving activations so the
    compressed tensor keeps the SAME per-sample L2 energy as the full one:

        x_out = (x * mask) * ||x|| / ||x * mask||

    This is a per-sample, eta-adaptive normalization at the cut-point: whatever fraction
    survives, the downstream layer receives a tensor of consistent energy -> less
    covariate shift across eta (in particular eta=1, where it is a no-op). Better suited
    to top-k than a plain trained BN (whose running stats would calibrate to one eta).
    """
    if eta >= 1.0:
        return x
    if eta <= 0.0:
        return torch.zeros_like(x)
    b = x.shape[0]
    flat = x.reshape(b, -1)
    n = flat.shape[1]
    k = max(1, int(eta * n))
    kth = flat.abs().topk(k, dim=1).values[:, -1:]
    kept = flat * (flat.abs() >= kth)
    scale = flat.norm(dim=1, keepdim=True) / (kept.norm(dim=1, keepdim=True) + 1e-6)
    return (kept * scale).reshape_as(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_cifar10_loaders(root: str = "./data", batch_size: int = 128,
                        augment: bool = True):
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_tf = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
         if augment else []) + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    train = DataLoader(datasets.CIFAR10(root, train=True, download=True, transform=train_tf),
                       batch_size=batch_size, shuffle=True, num_workers=2)
    test = DataLoader(datasets.CIFAR10(root, train=False, download=True, transform=test_tf),
                      batch_size=512, shuffle=False, num_workers=2)
    return train, test


# ---------------------------------------------------------------------------
# Pretrained model + cut-points
# ---------------------------------------------------------------------------
def load_pretrained_cifar_resnet(name: str = "cifar10_resnet56") -> nn.Module:
    """Download a pretrained CIFAR-10 ResNet (chenyaofo/pytorch-cifar-models)."""
    return torch.hub.load("chenyaofo/pytorch-cifar-models", name,
                          pretrained=True, trust_repo=True)


# per-cut-point activation sizes a_i = C*H*W, cached to disk per model name
ACT_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "activation_sizes.json")


def _probe_cut_sizes(model: nn.Module, cut_modules) -> np.ndarray:
    """Determine a_i from the model itself: run one dummy forward and read the
    activation shape at each cut-point (C*H*W)."""
    sizes = {}
    handles = [m.register_forward_hook(
        lambda mod, i, o, j=j: sizes.__setitem__(j, int(np.prod(o.shape[1:]))))
        for j, m in enumerate(cut_modules)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    for h in handles:
        h.remove()
    model.train(was_training)
    return np.array([sizes[j] for j in range(len(cut_modules))], dtype=float)


def cached_activation_sizes(model_name: str, model: nn.Module, cut_modules,
                            cache_file: str = ACT_CACHE_FILE) -> np.ndarray:
    """a_i for `model_name`, loaded from the JSON cache if present else probed once
    and saved -- so we never recompute a model's cut-point sizes."""
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    if model_name in cache:
        return np.array(cache[model_name], dtype=float)
    a = _probe_cut_sizes(model, cut_modules)
    cache[model_name] = a.tolist()
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    return a


def make_compressed_resnet(model: nn.Module, eta_init=None, model_name: Optional[str] = None):
    """Wrap a CIFAR ResNet with compression at the layer1 / layer2 outputs (d=2),
    using per-sample top-k. Returns (cm, a); if `model_name` is given, a_i is cached."""
    cut_modules = [model.layer1, model.layer2]
    if eta_init is None:
        eta_init = torch.ones(len(cut_modules))
    cm = CompressedModel(model, eta_init, cut_modules=cut_modules,
                         sparsify_fn=topk_sparsify_per_sample)
    a = (cached_activation_sizes(model_name, model, cut_modules) if model_name
         else _probe_cut_sizes(model, cut_modules))
    return cm, a


# ---------------------------------------------------------------------------
# Training (compression-aware fine-tuning) -- device-agnostic
# ---------------------------------------------------------------------------
def _set_eta(cm: CompressedModel, eta: np.ndarray, device) -> None:
    cm.set_eta(torch.as_tensor(eta, dtype=torch.float32, device=device))


def compression_aware_finetune(cm, train_loader, *, a, R, eta_min, sampler,
                               steps=1000, lr=1e-4, device=None, log_every=100):
    """SGD with a freshly channel-sampled eta*(c) per minibatch (Algorithm 1),
    starting from pretrained weights."""
    import torch.nn.functional as F
    from itertools import cycle
    device = device or next(cm.parameters()).device
    cm.to(device).train()
    opt = torch.optim.Adam(cm.parameters(), lr=lr)
    data = cycle(train_loader)
    for step in range(1, steps + 1):
        eta = csi_aware_eta_star(sampler(), a, R, eta_min)
        _set_eta(cm, eta, device)
        x, y = next(data)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(cm(x), y)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"[comp-aware ft] step {step:5d}  eta={np.round(eta,3)}  loss={loss.item():.4f}")
    return cm


def standard_finetune(cm, train_loader, *, steps=1000, lr=1e-4, device=None, log_every=100):
    """Control arm: continue training at eta=1 (no compression) for the same budget."""
    import torch.nn.functional as F
    from itertools import cycle
    device = device or next(cm.parameters()).device
    cm.to(device).train()
    _set_eta(cm, np.ones(cm.d), device)
    opt = torch.optim.Adam(cm.parameters(), lr=lr)
    data = cycle(train_loader)
    for step in range(1, steps + 1):
        x, y = next(data)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(cm(x), y)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"[standard ft]   step {step:5d}  loss={loss.item():.4f}")
    return cm


# ---------------------------------------------------------------------------
# Evaluation under the deployment channel distribution (batched for memory)
# ---------------------------------------------------------------------------
@torch.no_grad()
def deployed_accuracy_resnet(cm, x_all, y_all, *, a, R, eta_min, sampler,
                             n_channels=32, device=None, batch_size=512):
    """E_{c~P_c}[accuracy at eta*(c)] (Monte-Carlo) + clean (eta=1) accuracy,
    evaluated in batches so the full test set need not be resident at once."""
    device = device or next(cm.parameters()).device
    cm.eval()

    def acc_at(eta):
        _set_eta(cm, eta, device)
        correct = 0
        for i in range(0, len(x_all), batch_size):
            xb = x_all[i:i + batch_size].to(device)
            yb = y_all[i:i + batch_size].to(device)
            correct += (cm(xb).argmax(1) == yb).sum().item()
        return correct / len(x_all)

    clean = acc_at(np.ones(cm.d))
    accs = np.array([acc_at(csi_aware_eta_star(sampler(), a, R, eta_min))
                     for _ in range(n_channels)])
    return {
        "clean_acc": clean,
        "deployed_acc_mean": float(accs.mean()),
        "deployed_acc_std": float(accs.std()),
        "n_channels": n_channels,
    }


def load_test_tensors(test_loader):
    """Materialize the test set once (CPU) for fast repeated evaluation."""
    xs, ys = zip(*list(test_loader))
    return torch.cat(xs), torch.cat(ys)
