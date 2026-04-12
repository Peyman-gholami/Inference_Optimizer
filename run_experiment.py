"""
run_experiment.py — Configurable single-task experiment runner
==============================================================

Runs all algorithms and baselines for a fully-specified experimental setting
and produces comparison plots where all methods are displayed together.

Each experimental setting is defined by:
  - backend            : which model (mlp | resnet | llm)
  - compressor         : topk | quantization | llmint8  (ResNet/MLP)
  - compressor_backend : cpu | gpu
  - activation_strategy: LLM compression strategy
  - llm_model_name     : HuggingFace model ID (LLM only)
  - c_dist             : channel capacity distribution (uniform; min/max parameterised)
  - tau                : per-node compute times  (derived from api)
  - eta_min            : per-link compression floor
  - R_k                : target throughput (constant)
  - T, seed            : horizon and random seed
  - no_csi_mu          : No-CSI penalty weight
  - no_csi_epsilon     : No-CSI dual variable floor

All parameters — including those at their default values — are printed at
startup and saved in experiment_setting.json so every run is fully reproducible.

Usage
-----
  # MLP / MNIST, default setting:
  python run_experiment.py --backend mlp --T 150 --seed 0

  # ResNet, TopK compressor:
  python run_experiment.py --backend resnet --compressor topk \\
      --c_min 500000 --c_max 5000000 --T 50

  # ResNet, LLMInt8, GPU-native compressor:
  python run_experiment.py --backend resnet --compressor llmint8 \\
      --compressor_backend gpu --c_min 500000 --c_max 5000000 --T 50

  # LLM / Gemma-2B, topk_per_token:
  python run_experiment.py --backend llm \\
      --llm_activation_strategy topk_per_token \\
      --llm_model_name unsloth/gemma-2b-it --T 30

  # LLM / Llama-3.1-8B, GPU-native llmint8:
  python run_experiment.py --backend llm \\
      --llm_activation_strategy llmint8 \\
      --llm_model_name meta-llama/Llama-3.1-8B-Instruct --T 30

  # Override eta_min and R_k:
  python run_experiment.py --backend resnet --eta_min 0.1 --R_k 5.0

  # Select specific baselines:
  python run_experiment.py --backend mlp \\
      --baselines max_compression,no_compression,myopic

  # Skip baselines entirely:
  python run_experiment.py --backend mlp --baselines none

Output
------
  <out_dir>/
    experiment_setting.json   — full setting record (all parameters)
    accuracy_over_time.png    — cumulative avg accuracy, all methods
    eta_over_time.png         — cumulative avg eta per link, all methods
    delay_over_invR_time.png  — cumulative avg D_k/(1/R_k), all methods
    tradeoff.png              — accuracy vs delay scatter at t = T-1
    simulation_results.npz    — raw time series for post-processing
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

from src.core.task import InferenceTask
from src.optimizers.base import BaseOptimizer
from src.optimizers.csi_aware import CSIAwareSingleTaskOptimizer
from src.optimizers.no_csi import NoCSISingleTaskOptimizer
from src.optimizers.estimators import (
    LastObservationEstimator,
    MeanMinusZStdLCB,
    MovingAverageEstimator,
    RunningMinEstimator,
)
from src.optimizers.baseline import (
    MaxCompressionSingleTaskBaseline,
    NoCompressionSingleTaskBaseline,
    UniformCompressionSingleTaskBaseline,
    MyopicSingleTaskBaseline,
    ConservativeSingleTaskBaseline,
    MovingAverageSingleTaskBaseline,
)

# ============================================================================
# Experiment setting dataclass
# ============================================================================

@dataclass
class ExperimentSetting:
    """
    Full specification of one experimental setting.
    Serialisable to JSON so it can be embedded in every plot title.
    """
    # ── Core arguments ───────────────────────────────────────────────────────
    backend:             str            # mlp | resnet | llm
    compressor:          str            # topk | quantization | llmint8 (ResNet/MLP)
    compressor_backend:  str            # cpu | gpu
    c_min:               float
    c_max:               float
    c_dist:              str            # uniform
    R_k:                 float
    eta_min:             float
    T:                   int
    seed:                int
    no_csi_mu:           float
    no_csi_epsilon:      float

    # ── LLM-specific arguments ───────────────────────────────────────────────
    llm_model_name:          str   = ""      # e.g. unsloth/gemma-2b-it
    llm_activation_strategy: str   = ""      # topk_per_token | magnitude | random | quantization | llmint8
    llm_split_mode:          str   = "thirds"
    llm_grad_sigma:          float = 0.05
    llm_grad_N:              int   = 3
    llm_fast_samples:        int   = 5       # MMLU_FAST_SAMPLES_PER_SUBJECT
    llm_full_samples:        int   = 10      # MMLU_FULL_SAMPLES_PER_SUBJECT
    llm_batch_size:          int   = 8

    # ── ResNet-specific arguments ─────────────────────────────────────────────
    resnet_cutpoints:    str   = "8,14,21"
    resnet_fast_batches: int   = 1
    resnet_grad_sigma:   float = 0.05
    resnet_grad_N:       int   = 20

    # ── MLP-specific arguments ────────────────────────────────────────────────
    mlp_sigma:           float = 0.05
    mlp_N:               int   = 50
    mlp_n_samples:       int   = 512

    # ── Derived after backend is loaded — filled by build_task() ─────────────
    num_links:           int  = 0
    L_k:                 int  = 0
    a:                   Dict[int, float] = field(default_factory=dict)
    tau:                 Dict[int, float] = field(default_factory=dict)
    eta_min_per_link:    Dict[int, float] = field(default_factory=dict)

    def short_label(self) -> str:
        """One-line string for plot subtitles."""
        if self.backend == "llm":
            comp_str = self.llm_activation_strategy or self.compressor
            model_short = self.llm_model_name.split("/")[-1] if self.llm_model_name else "llm"
            return (
                f"LLM({model_short}) | {comp_str}/{self.compressor_backend} | "
                f"c~U[{self.c_min:.0f},{self.c_max:.0f}] | "
                f"R_k={self.R_k} | η_min={self.eta_min} | "
                f"T={self.T} seed={self.seed}"
            )
        return (
            f"{self.backend.upper()} | {self.compressor}/{self.compressor_backend} | "
            f"c~U[{self.c_min:.0f},{self.c_max:.0f}] | "
            f"R_k={self.R_k} | η_min={self.eta_min} | "
            f"T={self.T} seed={self.seed}"
        )

    def detail_label(self) -> str:
        """Multi-line string for the annotation block on every plot."""
        if self.backend == "llm":
            model_short = self.llm_model_name.split("/")[-1] if self.llm_model_name else "llm"
            comp_str    = self.llm_activation_strategy or self.compressor
            lines = [
                f"LLM({model_short})  strategy={comp_str}/{self.compressor_backend}  split={self.llm_split_mode}",
                f"c ~ Uniform[{self.c_min:.0f}, {self.c_max:.0f}]   R_k={self.R_k}   T={self.T}   seed={self.seed}",
                f"η_min={self.eta_min}   μ={self.no_csi_mu}   ε={self.no_csi_epsilon}",
                f"σ={self.llm_grad_sigma}   N={self.llm_grad_N}   fast_samples/subj={self.llm_fast_samples}",
                f"links={self.num_links}   a={[f'{self.a[i]:.0f}' for i in range(self.num_links)]}",
                f"τ={[f'{self.tau[i]:.4f}' for i in range(self.L_k)]}",
            ]
        elif self.backend == "resnet":
            lines = [
                f"ResNet56  compressor={self.compressor}/{self.compressor_backend}  cutpoints={self.resnet_cutpoints}",
                f"c ~ Uniform[{self.c_min:.0f}, {self.c_max:.0f}]   R_k={self.R_k}   T={self.T}   seed={self.seed}",
                f"η_min={self.eta_min}   μ={self.no_csi_mu}   ε={self.no_csi_epsilon}",
                f"σ={self.resnet_grad_sigma}   N={self.resnet_grad_N}   fast_batches={self.resnet_fast_batches}",
                f"links={self.num_links}   a={[f'{self.a[i]:.0f}' for i in range(self.num_links)]}",
                f"τ={[f'{self.tau[i]:.6f}' for i in range(self.L_k)]}",
            ]
        else:  # mlp
            lines = [
                f"MLP/MNIST  compressor={self.compressor}/{self.compressor_backend}",
                f"c ~ Uniform[{self.c_min:.0f}, {self.c_max:.0f}]   R_k={self.R_k}   T={self.T}   seed={self.seed}",
                f"η_min={self.eta_min}   μ={self.no_csi_mu}   ε={self.no_csi_epsilon}",
                f"σ={self.mlp_sigma}   N={self.mlp_N}   n_samples={self.mlp_n_samples}",
                f"links={self.num_links}   a={[f'{self.a[i]:.0f}' for i in range(self.num_links)]}",
                f"τ={[f'{self.tau[i]:.4f}' for i in range(self.L_k)]}",
            ]
        return "\n".join(lines)

    def full_printout(self, out_dir: str, baselines: list) -> None:
        """
        Print every parameter to stdout, grouped by section.
        Parameters that are at their default values are marked with (default).
        Called once at startup before any model is loaded, and again after
        topology is derived (when num_links > 0).
        """
        W = 65
        print("=" * W, flush=True)
        print("run_experiment.py  —  Full experiment configuration", flush=True)
        print("=" * W, flush=True)

        # ── Common ────────────────────────────────────────────────────────────
        print("  [Common]", flush=True)
        print(f"    backend            = {self.backend}", flush=True)
        print(f"    T                  = {self.T}"
              + ("  (default)" if self.T == 150 else ""), flush=True)
        print(f"    seed               = {self.seed}"
              + ("  (default)" if self.seed == 12 else ""), flush=True)
        print(f"    c_dist             = {self.c_dist}  "
              f"[{self.c_min:.0f}, {self.c_max:.0f}]", flush=True)
        print(f"    R_k                = {self.R_k}"
              + ("  (default)" if self.R_k == 10.0 else ""), flush=True)
        print(f"    eta_min            = {self.eta_min}", flush=True)
        print(f"    no_csi_mu          = {self.no_csi_mu}"
              + ("  (default)" if self.no_csi_mu == 3.0 else ""), flush=True)
        print(f"    no_csi_epsilon     = {self.no_csi_epsilon}"
              + ("  (default)" if self.no_csi_epsilon == 0.1 else ""), flush=True)

        # ── LLM backend ───────────────────────────────────────────────────────
        if self.backend == "llm":
            print("  [LLM]", flush=True)
            print(f"    model              = {self.llm_model_name}", flush=True)
            print(f"    activation_strategy= {self.llm_activation_strategy}"
                  + ("  (default)" if self.llm_activation_strategy == "topk_per_token"
                     else ""), flush=True)
            print(f"    compressor_backend = {self.compressor_backend}"
                  + ("  (default)" if self.compressor_backend == "cpu" else ""), flush=True)
            print(f"    split_mode         = {self.llm_split_mode}"
                  + ("  (default)" if self.llm_split_mode == "thirds" else ""), flush=True)
            print(f"    grad_sigma         = {self.llm_grad_sigma}"
                  + ("  (default)" if self.llm_grad_sigma == 0.05 else ""), flush=True)
            print(f"    grad_N             = {self.llm_grad_N}"
                  + ("  (default)" if self.llm_grad_N == 3 else ""), flush=True)
            print(f"    fast_samples/subj  = {self.llm_fast_samples}"
                  + ("  (default)" if self.llm_fast_samples == 5 else ""), flush=True)
            print(f"    full_samples/subj  = {self.llm_full_samples}"
                  + ("  (default)" if self.llm_full_samples == 10 else ""), flush=True)
            print(f"    batch_size         = {self.llm_batch_size}"
                  + ("  (default)" if self.llm_batch_size == 8 else ""), flush=True)

        # ── ResNet backend ────────────────────────────────────────────────────
        elif self.backend == "resnet":
            print("  [Compressor]", flush=True)
            print(f"    compressor         = {self.compressor}"
                  + ("  (default)" if self.compressor == "topk" else ""), flush=True)
            print(f"    compressor_backend = {self.compressor_backend}"
                  + ("  (default)" if self.compressor_backend == "cpu" else ""), flush=True)
            print("  [ResNet]", flush=True)
            print(f"    cutpoints          = {self.resnet_cutpoints}"
                  + ("  (default)" if self.resnet_cutpoints == "8,14,21" else ""), flush=True)
            print(f"    fast_batches       = {self.resnet_fast_batches}"
                  + ("  (default)" if self.resnet_fast_batches == 1 else ""), flush=True)
            print(f"    grad_sigma         = {self.resnet_grad_sigma}"
                  + ("  (default)" if self.resnet_grad_sigma == 0.05 else ""), flush=True)
            print(f"    grad_N             = {self.resnet_grad_N}"
                  + ("  (default)" if self.resnet_grad_N == 20 else ""), flush=True)

        # ── MLP backend ───────────────────────────────────────────────────────
        else:
            print("  [Compressor]", flush=True)
            print(f"    compressor         = {self.compressor}"
                  + ("  (default)" if self.compressor == "topk" else ""), flush=True)
            print(f"    compressor_backend = {self.compressor_backend}"
                  + ("  (default)" if self.compressor_backend == "cpu" else ""), flush=True)
            print("  [MLP]", flush=True)
            print(f"    grad_sigma         = {self.mlp_sigma}"
                  + ("  (default)" if self.mlp_sigma == 0.05 else ""), flush=True)
            print(f"    grad_N             = {self.mlp_N}"
                  + ("  (default)" if self.mlp_N == 50 else ""), flush=True)
            print(f"    n_samples          = {self.mlp_n_samples}"
                  + ("  (default)" if self.mlp_n_samples == 512 else ""), flush=True)

        # ── Topology (filled after backend loads) ─────────────────────────────
        if self.num_links > 0:
            print("  [Topology — derived from model]", flush=True)
            print(f"    L_k                = {self.L_k}", flush=True)
            print(f"    num_links          = {self.num_links}", flush=True)
            print(f"    a (bytes/sample)   = "
                  f"{[f'{self.a[i]:.0f}' for i in range(self.num_links)]}",
                  flush=True)
            print(f"    tau (s)            = "
                  f"{[f'{self.tau[i]:.6f}' for i in range(self.L_k)]}",
                  flush=True)
            print(f"    eta_min_per_link   = "
                  f"{[self.eta_min_per_link[i] for i in range(self.num_links)]}",
                  flush=True)

        # ── Run ───────────────────────────────────────────────────────────────
        print("  [Run]", flush=True)
        print(f"    baselines          = {baselines or '(none)'}", flush=True)
        print(f"    out_dir            = {out_dir}", flush=True)
        print("=" * W, flush=True)


# ============================================================================
# Backend loading
# ============================================================================

def _set_env(key: str, value: str):
    """Set an env var only if not already set by the caller."""
    if key not in os.environ:
        os.environ[key] = value


def load_backend(s: ExperimentSetting):
    """
    Load the api and derive topology from the environment + setting.
    Returns (api, flat_theta_or_None).
    Sets COMPRESSOR_TYPE, COMPRESSOR_BACKEND, and LLM_ACTIVATION_STRATEGY
    in the environment before importing backend modules.
    """
    os.environ['COMPRESSOR_TYPE']    = s.compressor
    os.environ['COMPRESSOR_BACKEND'] = s.compressor_backend

    from src.core.llm_plug_in import is_llm_backend, is_resnet_backend

    if is_llm_backend():
        # Push LLM-specific settings into env so build_llm_backend_from_env picks them up
        if s.llm_activation_strategy:
            os.environ['LLM_ACTIVATION_STRATEGY'] = s.llm_activation_strategy
        if s.llm_model_name:
            os.environ['LLM_MODEL_NAME'] = s.llm_model_name
        os.environ.setdefault('LLM_SPLIT_MODE',                  s.llm_split_mode)
        os.environ.setdefault('LLM_GRAD_SIGMA',                   str(s.llm_grad_sigma))
        os.environ.setdefault('LLM_GRAD_N',                       str(s.llm_grad_N))
        os.environ.setdefault('MMLU_FAST_SAMPLES_PER_SUBJECT',    str(s.llm_fast_samples))
        os.environ.setdefault('MMLU_FULL_SAMPLES_PER_SUBJECT',    str(s.llm_full_samples))
        os.environ.setdefault('MMLU_BATCH_SIZE',                  str(s.llm_batch_size))

        # Reflect actual env values back into setting (env may have been set by caller)
        s.llm_model_name          = os.environ.get('LLM_MODEL_NAME', s.llm_model_name)
        s.llm_activation_strategy = os.environ.get('LLM_ACTIVATION_STRATEGY', s.llm_activation_strategy)
        s.llm_split_mode          = os.environ.get('LLM_SPLIT_MODE', s.llm_split_mode)
        s.llm_grad_sigma          = float(os.environ.get('LLM_GRAD_SIGMA', str(s.llm_grad_sigma)))
        s.llm_grad_N              = int(os.environ.get('LLM_GRAD_N', str(s.llm_grad_N)))
        s.llm_fast_samples        = int(os.environ.get('MMLU_FAST_SAMPLES_PER_SUBJECT', str(s.llm_fast_samples)))
        s.llm_full_samples        = int(os.environ.get('MMLU_FULL_SAMPLES_PER_SUBJECT', str(s.llm_full_samples)))
        s.llm_batch_size          = int(os.environ.get('MMLU_BATCH_SIZE', str(s.llm_batch_size)))

        from src.core.llm_task_callables import build_llm_backend_from_env
        api = build_llm_backend_from_env(eta_min=s.eta_min)
        return api, None

    if is_resnet_backend():
        # Reflect ResNet env values into setting
        s.resnet_cutpoints    = os.environ.get('RESNET_CUTPOINTS',    s.resnet_cutpoints)
        s.resnet_fast_batches = int(os.environ.get('RESNET_FAST_BATCHES', str(s.resnet_fast_batches)))
        s.resnet_grad_sigma   = float(os.environ.get('RESNET_GRAD_SIGMA',  str(s.resnet_grad_sigma)))
        s.resnet_grad_N       = int(os.environ.get('RESNET_GRAD_N',        str(s.resnet_grad_N)))

        from src.core.resnet_task_callables import build_resnet_backend_from_env
        api = build_resnet_backend_from_env()
        return api, None

    # MLP / MNIST default — reflect sigma/N into setting
    s.mlp_sigma     = float(os.environ.get('MLP_GRAD_SIGMA', str(s.mlp_sigma)))
    s.mlp_N         = int(os.environ.get('MLP_GRAD_N',       str(s.mlp_N)))
    s.mlp_n_samples = int(os.environ.get('MLP_N_SAMPLES',    str(s.mlp_n_samples)))

    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from src.core.task_callables import InferenceTaskCallables
    from src.core.stein_oracles import MLP, get_flat_params
    from accuracy_fn import CompressedModel

    torch.manual_seed(s.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_root = os.environ.get("MNIST_DATA_ROOT", "./data")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST(mnist_root, train=True,  download=True, transform=tf),
        batch_size=128, shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(mnist_root, train=False, download=True, transform=tf),
        batch_size=256, shuffle=False,
    )

    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("[run_experiment] Training MLP (3 epochs)...", flush=True)
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)
    eta_init   = torch.ones(2, device=device)
    cm = CompressedModel(model, eta_init, compressor=s.compressor)

    class DeviceLoader:
        def __init__(self, loader, dev):
            self._loader = loader; self._dev = dev
        def __iter__(self):
            for x, y in self._loader:
                yield x.to(self._dev), y.to(self._dev)
        def __len__(self): return len(self._loader)
        def __getattr__(self, name): return getattr(self._loader, name)

    api = InferenceTaskCallables(
        cm, DeviceLoader(test_loader, device),
        sigma=s.mlp_sigma, N=s.mlp_N, n_samples=s.mlp_n_samples,
        flat_theta=None, device=device,
        compressor=s.compressor,
    )
    return api, flat_theta


# ============================================================================
# Task builder
# ============================================================================

def build_task(s: ExperimentSetting, api) -> InferenceTask:
    """
    Build an InferenceTask from the api and setting.
    Fills in s.num_links, s.L_k, s.a, s.tau, s.eta_min_per_link in-place
    so the setting record is complete before plotting.
    """
    from src.core.llm_plug_in import is_resnet_backend

    R_k_callable = lambda t, r=s.R_k: r

    if is_resnet_backend():
        num_links = api.num_links
        num_nodes = num_links + 1
        raw_a     = api.measured_a()
        a         = {i: float(raw_a[i]) for i in range(num_links)}
        tau       = {i: 0.001 for i in range(num_nodes)}
        eta_min_d = {i: s.eta_min for i in range(num_links)}

        s.num_links        = num_links
        s.L_k              = num_nodes
        s.a                = a
        s.tau              = tau
        s.eta_min_per_link = eta_min_d

        print(f"[run_experiment] ResNet: L_k={num_nodes}  num_links={num_links}", flush=True)
        print(f"[run_experiment] a (bytes/sample) = {[f'{a[i]:.0f}' for i in range(num_links)]}", flush=True)

        return InferenceTask(
            task_id=1, b_k=0, L_k=num_nodes,
            tau=tau, a=a,
            eta_min=eta_min_d,
            R_k_callable=R_k_callable,
            w_k=1.0,
            accuracy_callable=api.accuracy_callable,
            accuracy_callable_true=api.accuracy_callable_true,
            gradient_callable=api.gradient_callable,
        )

    # MLP / LLM — 2 links, toy a unless measured profile available
    from src.core.llm_plug_in import maybe_measured_tau_a
    measured = maybe_measured_tau_a()
    if measured is not None:
        tau, a = measured
    else:
        # MLP defaults: activations are [batch,128] and [batch,64] at FP32
        # Use per-sample byte sizes: 128*4=512, 64*4=256
        tau = {0: 0.01, 1: 0.01, 2: 0.01}
        a   = {0: 512.0, 1: 256.0}

    num_links = 2
    num_nodes = 3
    eta_min_d = {0: s.eta_min, 1: s.eta_min}

    s.num_links        = num_links
    s.L_k              = num_nodes
    s.a                = {k: float(v) for k, v in a.items()}
    s.tau              = {k: float(v) for k, v in tau.items()}
    s.eta_min_per_link = eta_min_d

    return InferenceTask(
        task_id=1, b_k=0, L_k=num_nodes,
        tau=tau, a=a,
        eta_min=eta_min_d,
        R_k_callable=R_k_callable,
        w_k=1.0,
        accuracy_callable=api.accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=api.gradient_callable,
    )


# ============================================================================
# Algorithm factory
# ============================================================================

def build_algorithms(
    task: InferenceTask,
    s: ExperimentSetting,
    baselines: List[str],
) -> Tuple[List[Tuple[str, BaseOptimizer]],
           Optional[object], Optional[object], Optional[object],
           object]:
    """
    Build all algorithm instances.
    Returns (algos, myopic_est, cons_est, ma_est, lcb).
    """
    M         = task.L_k
    num_links = task.L_k - 1

    lcb = MeanMinusZStdLCB(
        num_links=num_links,
        warmup_value=s.c_min,
        z=1.2815515655446004,
        warmup=5,
    )

    algos: List[Tuple[str, BaseOptimizer]] = [
        ("CSI-aware (optimal)",
         CSIAwareSingleTaskOptimizer(M=M, tasks=[task])),
        ("No-CSI (Alg1) + LCB(90%)",
         NoCSISingleTaskOptimizer(M=M, tasks=[task],
                                  mu=s.no_csi_mu,
                                  epsilon=s.no_csi_epsilon)),
    ]

    baseline_catalog = {
        "max_compression":    ("Baseline: max compression",
                               MaxCompressionSingleTaskBaseline(M=M, tasks=[task])),
        "no_compression":     ("Baseline: no compression",
                               NoCompressionSingleTaskBaseline(M=M, tasks=[task])),
        "uniform_compression":("Baseline: uniform compression",
                               UniformCompressionSingleTaskBaseline(M=M, tasks=[task])),
        "myopic":             ("Baseline: myopic",
                               MyopicSingleTaskBaseline(M=M, tasks=[task])),
        "conservative":       ("Baseline: conservative",
                               ConservativeSingleTaskBaseline(M=M, tasks=[task])),
        "moving_average":     ("Baseline: moving average",
                               MovingAverageSingleTaskBaseline(M=M, tasks=[task])),
    }

    selected_set = set(baselines)
    for key in baselines:
        if key not in baseline_catalog:
            raise ValueError(
                f"Unknown baseline '{key}'. "
                f"Valid: {sorted(baseline_catalog.keys())}"
            )
        algos.append(baseline_catalog[key])

    myopic_est = (LastObservationEstimator(num_links=num_links, warmup_value=s.c_min)
                  if "myopic" in selected_set else None)
    cons_est   = (RunningMinEstimator(num_links=num_links, warmup_value=s.c_min)
                  if "conservative" in selected_set else None)
    ma_est     = (MovingAverageEstimator(num_links=num_links, window=5, warmup_value=s.c_min)
                  if "moving_average" in selected_set else None)

    return algos, myopic_est, cons_est, ma_est, lcb


# ============================================================================
# Simulation loop
# ============================================================================

def _save_checkpoint(
    t: int,
    out_dir: str,
    algo_names: List[str],
    times: np.ndarray,
    c_t_hist: np.ndarray,
    acc_hist: Dict,
    delay_ratio_hist: Dict,
    eta_hist: Dict,
    lambda_t: float = 0.0,
) -> None:
    """Save a partial checkpoint after slot t so cancelled jobs are not lost."""
    os.makedirs(out_dir, exist_ok=True)
    T_done = t + 1
    algo_names_arr = np.array(algo_names, dtype=str)
    np.savez_compressed(
        os.path.join(out_dir, "checkpoint.npz"),
        T_done          = np.array([T_done], dtype=int),
        lambda_t        = np.array([lambda_t], dtype=float),
        times           = times[:T_done],
        c_t_hist        = c_t_hist[:T_done],
        algo_names      = algo_names_arr,
        acc_hist        = np.stack([acc_hist[n][:T_done]           for n in algo_names]),
        delay_ratio_hist= np.stack([delay_ratio_hist[n][:T_done]   for n in algo_names]),
        eta_hist        = np.stack([eta_hist[n][:T_done]           for n in algo_names]),
    )
    print(f"  [checkpoint] saved at t={t}  lambda_t={lambda_t:.4f}"
          f" → {out_dir}/checkpoint.npz", flush=True)


def load_checkpoint(out_dir: str) -> Optional[Dict]:
    """
    Load a checkpoint from <out_dir>/checkpoint.npz.
    Returns a dict with keys:
      T_done, lambda_t, c_t_hist, algo_names,
      acc_hist, delay_ratio_hist, eta_hist
    or None if no checkpoint exists.
    """
    path = os.path.join(out_dir, "checkpoint.npz")
    if not os.path.exists(path):
        return None
    data = dict(np.load(path, allow_pickle=True))
    algo_names = list(data["algo_names"])
    T_done     = int(data["T_done"][0])
    lambda_t   = float(data["lambda_t"][0]) if "lambda_t" in data else 0.0

    # Reconstruct per-algo dicts
    acc_hist         = {n: data["acc_hist"][i]         for i, n in enumerate(algo_names)}
    delay_ratio_hist = {n: data["delay_ratio_hist"][i] for i, n in enumerate(algo_names)}
    eta_hist_raw     = data["eta_hist"]   # shape (A, T_done, L)

    print(f"[load_checkpoint] Loaded checkpoint from {path}", flush=True)
    print(f"  T_done={T_done}  lambda_t={lambda_t:.4f}  "
          f"algos={algo_names}", flush=True)

    return dict(
        T_done           = T_done,
        lambda_t         = lambda_t,
        c_t_hist         = data["c_t_hist"],      # (T_done, num_links)
        algo_names       = algo_names,
        acc_hist         = acc_hist,
        delay_ratio_hist = delay_ratio_hist,
        eta_hist_raw     = eta_hist_raw,
    )


def run_simulation(
    task: InferenceTask,
    algos: List[Tuple[str, BaseOptimizer]],
    s: ExperimentSetting,
    lcb, myopic_est, cons_est, ma_est,
    out_dir: str = "",
    checkpoint_every: int = 5,
    checkpoint: Optional[Dict] = None,
) -> Dict:
    """
    Run T time slots and return all history dicts.

    Checkpointing
    -------------
    Every `checkpoint_every` slots a partial checkpoint is written to
    <out_dir>/checkpoint.npz so that jobs cancelled by Slurm do not lose
    all their work.  A SIGTERM handler (sent by Slurm before cancellation)
    also triggers an immediate checkpoint save.

    Resume
    ------
    If `checkpoint` is provided (loaded via load_checkpoint()), the simulation
    resumes from T_done, restoring all history arrays, the RNG state, the
    online estimators, and the No-CSI dual variable lambda_t.
    """
    import signal

    T         = s.T
    num_links = task.L_k - 1
    times     = np.arange(T)
    algo_names= [n for n, _ in algos]

    # ── Initialise or restore history arrays ──────────────────────────────────
    if checkpoint is not None:
        T_done   = checkpoint["T_done"]
        lambda_t = checkpoint["lambda_t"]
        print(f"[run_simulation] Resuming from t={T_done} "
              f"(lambda_t={lambda_t:.4f})", flush=True)

        # Restore full-size arrays and fill completed slots from checkpoint
        acc_hist         = {n: np.full(T, np.nan) for n in algo_names}
        delay_ratio_hist = {n: np.full(T, np.nan) for n in algo_names}
        eta_hist         = {n: np.full((T, num_links), np.nan) for n in algo_names}
        c_t_hist         = np.full((T, num_links), np.nan)

        c_t_hist[:T_done] = checkpoint["c_t_hist"]
        for i, name in enumerate(algo_names):
            acc_hist[name][:T_done]          = checkpoint["acc_hist"][name]
            delay_ratio_hist[name][:T_done]  = checkpoint["delay_ratio_hist"][name]
            eta_hist[name][:T_done]          = checkpoint["eta_hist_raw"][i]

        # Restore RNG by replaying T_done draws (same seed → same sequence)
        rng = np.random.default_rng(s.seed)
        for _ in range(T_done):
            rng.uniform(low=s.c_min, high=s.c_max, size=num_links)

        # Restore online estimators by replaying observed c_t values
        for t_past in range(T_done):
            c_past = c_t_hist[t_past]
            lcb.update(c_past)
            if myopic_est: myopic_est.update(c_past)
            if cons_est:   cons_est.update(c_past)
            if ma_est:     ma_est.update(c_past)

        # Restore No-CSI dual variable
        no_csi_opt = next((opt for name, opt in algos
                           if name == "No-CSI (Alg1) + LCB(90%)"), None)
        if no_csi_opt is not None:
            no_csi_opt.lambda_t = max(no_csi_opt.epsilon, lambda_t)
            print(f"[run_simulation] Restored lambda_t={no_csi_opt.lambda_t:.4f}",
                  flush=True)

        t_start = T_done
    else:
        rng              = np.random.default_rng(s.seed)
        acc_hist         = {n: np.full(T, np.nan) for n, _ in algos}
        delay_ratio_hist = {n: np.full(T, np.nan) for n, _ in algos}
        eta_hist         = {n: np.full((T, num_links), np.nan) for n, _ in algos}
        c_t_hist         = np.full((T, num_links), np.nan)
        lambda_t         = s.no_csi_epsilon
        t_start          = 0

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    _last_t    = [t_start - 1]
    _lambda_t  = [lambda_t]

    def _sigterm_handler(signum, frame):
        t_done = _last_t[0]
        if t_done >= 0 and out_dir:
            print(f"\n[run_simulation] SIGTERM at t={t_done} — saving checkpoint.",
                  flush=True)
            _save_checkpoint(t_done, out_dir, algo_names, times,
                             c_t_hist, acc_hist, delay_ratio_hist, eta_hist,
                             lambda_t=_lambda_t[0])
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Simulation loop ───────────────────────────────────────────────────────
    for t in range(t_start, T):
        c_t = rng.un