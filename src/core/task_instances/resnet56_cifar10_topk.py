from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import resnet20

from ..accuracy_fn import CompressedModel
from ..task import InferenceTask
from ..task_callables import InferenceTaskCallables


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHECKPOINT_PATH = _ROOT / "assets" / "resnet56-4bfd9763.th"
_DEFAULT_TRACE_PATH = _ROOT / "assets" / "resnet56_cifar10_15Mbps_scenario_trace_topk.json"
_DEFAULT_DATA_ROOT = _ROOT / "data"
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_FAST_SAMPLES = 100
_DEFAULT_SIGMA = 0.05
_DEFAULT_N = 5
_DEFAULT_ETA_MIN = 0.125


def _env_path(name: str, default_path: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default_path


def _load_profile(trace_path: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = payload["trace"]
    activation_bytes = np.asarray(trace["activation_bytes"], dtype=float)
    tau_compute = np.asarray(trace["tau_compute_node_sec"], dtype=float)

    a_med = np.median(activation_bytes, axis=0)
    tau_med = np.median(tau_compute, axis=0)

    a = {int(idx): float(a_med[idx]) for idx in range(a_med.shape[0])}
    tau = {int(idx): float(tau_med[idx]) for idx in range(tau_med.shape[0])}
    return tau, a


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = resnet20.resnet56()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(normalized)
    model.to(device)
    model.eval()
    return model


def _build_test_loader(data_root: Path, batch_size: int = 100, download: bool = False) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = datasets.CIFAR10(root=str(data_root), train=False, download=download, transform=tf)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False)


def setup_model_and_callables():
    device = torch.device(os.environ.get("RESNET56_PHASE1_DEVICE", _DEFAULT_DEVICE))
    checkpoint_path = _env_path("RESNET56_PHASE1_CHECKPOINT", _DEFAULT_CHECKPOINT_PATH)
    data_root = _env_path("RESNET56_PHASE1_DATA_ROOT", _DEFAULT_DATA_ROOT)
    fast_samples = int(os.environ.get("RESNET56_PHASE1_FAST_SAMPLES", str(_DEFAULT_FAST_SAMPLES)))
    sigma = float(os.environ.get("RESNET56_PHASE1_SIGMA", str(_DEFAULT_SIGMA)))
    stein_n = int(os.environ.get("RESNET56_PHASE1_STEIN_N", str(_DEFAULT_N)))
    download = os.environ.get("RESNET56_PHASE1_DOWNLOAD", "0").strip() in {"1", "true", "True"}

    model = _load_model(checkpoint_path, device)
    test_loader = _build_test_loader(data_root=data_root, batch_size=100, download=download)

    # Three transport links in a 4-node chain: after layer1, layer2, layer3.
    cut_modules = [model.layer1, model.layer2, model.layer3]
    eta_init = torch.ones(len(cut_modules))
    cm = CompressedModel(model, eta_init, cut_modules=cut_modules)
    api = InferenceTaskCallables(
        cm,
        test_loader,
        sigma=sigma,
        N=stein_n,
        n_samples=fast_samples,
        flat_theta=None,
        device=device,
    )

    extra = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "data_root": str(data_root),
        "fast_samples": int(fast_samples),
    }
    return api, extra


def make_task(
    task_id: int,
    api: InferenceTaskCallables,
    w_k: float = 1.0,
    R_k: float = 10.0,
) -> InferenceTask:
    trace_path = _env_path("RESNET56_PHASE1_TRACE_PATH", _DEFAULT_TRACE_PATH)
    tau, a = _load_profile(trace_path)
    eta_floor = float(os.environ.get("RESNET56_PHASE1_ETA_MIN", str(_DEFAULT_ETA_MIN)))

    return InferenceTask(
        task_id=task_id,
        b_k=0,
        L_k=4,
        tau=tau,
        a=a,
        eta_min={0: eta_floor, 1: eta_floor, 2: eta_floor},
        R_k_callable=lambda t, _R=R_k: float(_R),
        w_k=w_k,
        accuracy_callable=api.accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=api.gradient_callable,
    )
