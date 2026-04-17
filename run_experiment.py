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
from typing import Callable, Dict, List, Optional, Tuple

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

def _llmint8_correct_a_and_eta_min(
    a: dict, eta_min: float, is_llmint8: bool
) -> tuple:
    """
    For llmint8 the transmitted payload is a_i*(1+eta)/2 bytes.
    We rewrite this as a_eff*eta where a_eff = a_i/2, and absorb the
    constant INT8-floor term a_i/(2c) into the delay target via an
    effective R_k that depends on c_t each slot (see make_llmint8_R_k_callable).

    This function returns:
      a_eff    = a_i / 2  (halved payload coefficient)
      eta_min_eff = max(eta_min, 0.5)  (pure INT8 floor)

    The R_k correction is handled separately by make_llmint8_R_k_callable.
    """
    if not is_llmint8:
        return a, eta_min
    a_corr       = {k: v / 2.0 for k, v in a.items()}
    eta_min_corr = max(eta_min, 0.5)
    print("[build_task] llmint8: a_eff=a/2, eta_min=max(eta_min,0.5)", flush=True)
    print(f"  a: {[f'{a[k]:.0f}' for k in sorted(a)]} -> "
          f"{[f'{a_corr[k]:.0f}' for k in sorted(a_corr)]}", flush=True)
    print(f"  eta_min: {eta_min} -> {eta_min_corr}", flush=True)
    return a_corr, eta_min_corr


def make_llmint8_R_k_callable(
    R_k: float,
    a:   dict,
) -> Callable:
    """
    Build R_k_callable for llmint8 that corrects for the INT8 floor.

    The actual delay for llmint8 is:
        D = a_i*(1+eta)/(2*c_i) = (a_i/2)*eta/c_i + a_i/(2*c_i)
                                 = a_eff*eta/c_i  +  floor_i(c_i)

    Absorbing the floor into the target:
        D - 1/R_k = a_eff*eta/c_i - [1/R_k - a_i/(2*c_i)]
                  = a_eff*eta/c_i - 1/R_k_eff_i(c_i)

    where:
        1/R_k_eff_i = 1/R_k - a_i/(2*c_i)
        R_k_eff_i   = R_k / (1 - a_i*R_k/(2*c_i))

    Returns a callable (t, c_t) -> float  (scalar, using link 0 as bottleneck)
    or per-link array if needed.

    If 1/R_k - a_i/(2*c_i) <= 0 for any link, the constraint is infeasible
    regardless of eta (INT8 floor alone exceeds target delay) — returns inf.
    """
    a_fp16 = {k: v * 2.0 for k, v in a.items()}  # recover original a from a_eff=a/2

    def R_k_callable(t: int, c_t: np.ndarray = None) -> float:
        if c_t is None:
            # Static fallback — ignore floor correction
            return R_k
        # Per-link effective R_k: use minimum (most constraining link)
        R_eff_vals = []
        for i, c_i in enumerate(c_t):
            a_i     = a_fp16.get(i, list(a_fp16.values())[0])
            inv_R   = 1.0 / R_k
            floor_i = a_i / (2.0 * c_i) if c_i > 0 else float("inf")
            slack   = inv_R - floor_i
            if slack <= 0:
                return float("inf")   # infeasible — floor exceeds target
            R_eff_vals.append(1.0 / slack)
        # Return the minimum effective R_k (tightest constraint across links)
        return float(min(R_eff_vals))

    return R_k_callable


def build_task(s: ExperimentSetting, api) -> InferenceTask:
    """
    Build an InferenceTask from the api and setting.
    Fills in s.num_links, s.L_k, s.a, s.tau, s.eta_min_per_link in-place
    so the setting record is complete before plotting.
    """
    from src.core.llm_plug_in import is_resnet_backend

    if is_resnet_backend():
        num_links = api.num_links
        num_nodes = num_links + 1
        raw_a     = api.measured_a()
        a         = {i: float(raw_a[i]) for i in range(num_links)}
        tau       = {i: 0.001 for i in range(num_nodes)}

        is_llmint8 = (s.compressor.lower() == "llmint8")
        a, eta_min_eff = _llmint8_correct_a_and_eta_min(a, s.eta_min, is_llmint8)
        eta_min_d  = {i: eta_min_eff for i in range(num_links)}
        if is_llmint8:
            R_k_callable = make_llmint8_R_k_callable(s.R_k, a)
        else:
            R_k_callable = lambda t, *_, r=s.R_k: r

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
        tau = {0: 0.01, 1: 0.01, 2: 0.01}
        a   = {0: 512.0, 1: 256.0}

    is_llmint8 = (s.llm_activation_strategy.lower() == "llmint8")
    a, eta_min_eff = _llmint8_correct_a_and_eta_min(
        {k: float(v) for k, v in a.items()}, s.eta_min, is_llmint8
    )
    if is_llmint8:
        R_k_callable = make_llmint8_R_k_callable(s.R_k, a)
    else:
        R_k_callable = lambda t, *_, r=s.R_k: r

    num_links = 2
    num_nodes = 3
    eta_min_d = {0: eta_min_eff, 1: eta_min_eff}

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
        c_t = rng.uniform(low=s.c_min, high=s.c_max, size=num_links).astype(float)
        c_t_hist[t] = c_t
        print(f"  t={t:3d}  c_t={np.round(c_t, 1)}", flush=True)

        for name, opt in algos:
            if name == "No-CSI (Alg1) + LCB(90%)":
                allocations = opt.optimize(t=t, c_hat=lcb.estimate(t))
            elif name == "Baseline: myopic" and myopic_est:
                allocations = opt.optimize(t=t, c_hat=myopic_est.estimate(t))
            elif name == "Baseline: conservative" and cons_est:
                allocations = opt.optimize(t=t, c_hat=cons_est.estimate(t))
            elif name == "Baseline: moving average" and ma_est:
                allocations = opt.optimize(t=t, c_hat=ma_est.estimate(t))
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                # Infeasible slot: fall back to maximum compression (η = η_min).
                # This ensures all algorithms are evaluated over the same T slots
                # so cumulative averages are comparable. Without this, CSI-aware
                # would silently skip infeasible slots and nanmean over a smaller
                # subset, making its delay average appear lower than baselines
                # that always return an allocation.
                eta_fallback = np.array(
                    [task.eta_min[i] for i in range(task.b_k, task.e_k)],
                    dtype=float
                )
                s_comp_fb = np.ones(task.L_k, dtype=float)
                s_comm_fb = np.ones(task.L_k - 1, dtype=float)
                allocations = {
                    task.task_id: {
                        "eta":    eta_fallback,
                        "s_comp": s_comp_fb,
                        "s_comm": s_comm_fb,
                    }
                }

            eta = allocations[task.task_id]["eta"]
            acc_hist[name][t] = float(task.A_k_true(eta))

            residuals = opt.compute_residual_delay(t, allocations, c_t)
            residual  = float(residuals[task.task_id])
            R_k       = float(task.get_R_k(t, c_t))

            # R_k = inf means the llmint8 INT8 floor alone exceeds the delay
            # target at this channel realisation (structurally infeasible slot).
            # Fall back to the static nominal R_k so delay metrics remain finite.
            if not np.isfinite(R_k):
                R_k = float(task.get_R_k(t))   # c_t=None → static fallback

            inv_R     = 1.0 / R_k
            delay     = residual + inv_R
            delay_ratio_hist[name][t] = delay / inv_R

            eta_hist[name][t] = np.array(eta, dtype=float)

            if name == "No-CSI (Alg1) + LCB(90%)":
                opt.update_dual(t=t, actual_delays={task.task_id: delay}, c_t=c_t)
                _lambda_t[0] = opt.lambda_t   # track for checkpoint

        lcb.update(c_t)
        if myopic_est: myopic_est.update(c_t)
        if cons_est:   cons_est.update(c_t)
        if ma_est:     ma_est.update(c_t)

        _last_t[0] = t

        if out_dir and checkpoint_every > 0 and (t + 1) % checkpoint_every == 0:
            _save_checkpoint(t, out_dir, algo_names, times,
                             c_t_hist, acc_hist, delay_ratio_hist, eta_hist,
                             lambda_t=_lambda_t[0])

    # ── Cumulative averages ───────────────────────────────────────────────────
    acc_avg         = {}
    delay_ratio_avg = {}
    eta_avg         = {}
    for name in algo_names:
        acc_avg[name]         = np.array([np.nanmean(acc_hist[name][:t+1])
                                          for t in range(T)])
        delay_ratio_avg[name] = np.array([np.nanmean(delay_ratio_hist[name][:t+1])
                                          for t in range(T)])
        eta_avg[name]         = np.array([np.nanmean(eta_hist[name][:t+1], axis=0)
                                          for t in range(T)])

    return dict(
        times=times, c_t_hist=c_t_hist,
        acc_hist=acc_hist, delay_ratio_hist=delay_ratio_hist, eta_hist=eta_hist,
        acc_avg=acc_avg, delay_ratio_avg=delay_ratio_avg, eta_avg=eta_avg,
        algo_names=algo_names,
    )


# ============================================================================
# Plotting
# ============================================================================

def _setting_text(fig, s: ExperimentSetting, fontsize: int = 6.5):
    """
    Add the experiment setting as a small text block below the plot area.
    Placed outside the axes so it never overlaps data or legend.
    """
    fig.text(
        0.01, 0.01, s.detail_label(),
        fontsize=fontsize,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='grey', alpha=0.85),
        family='monospace',
    )


def _markers():
    return ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]


def plot_results(
    task: InferenceTask,
    hist: Dict,
    s: ExperimentSetting,
    out_dir: str,
) -> Dict[str, str]:
    """
    Produce all four comparison plots.
    Returns dict mapping plot name -> file path.
    """
    os.makedirs(out_dir, exist_ok=True)

    algo_names      = hist["algo_names"]
    times           = hist["times"]
    acc_avg         = hist["acc_avg"]
    delay_ratio_avg = hist["delay_ratio_avg"]
    eta_avg         = hist["eta_avg"]
    T               = s.T
    num_links       = task.L_k - 1

    markers   = _markers()
    marker_map= {n: markers[i % len(markers)] for i, n in enumerate(algo_names)}
    markevery = max(1, T // 10)
    paths     = {}

    # Extra bottom margin to accommodate the setting text box below the plot.
    # Extra right margin to accommodate the external legend.
    BOTTOM  = 0.22   # fraction of figure height reserved at bottom
    RIGHT   = 0.72   # axes right edge (rest is legend space)

    # ── Plot 1: Cumulative average accuracy ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in algo_names:
        ax.plot(times, acc_avg[name], label=name, linewidth=1.5,
                marker=marker_map[name], markevery=markevery, markersize=5)
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg accuracy  (1/t) Σ A_k(η(s))")
    ax.set_title(f"Accuracy over time\n{s.short_label()}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "accuracy_over_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["accuracy"] = p

    # ── Plot 2: Cumulative average eta per link ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in algo_names:
        for li in range(num_links):
            ax.plot(times, eta_avg[name][:, li],
                    label=f"{name} (link {li})",
                    linewidth=1.2,
                    marker=marker_map[name], markevery=markevery, markersize=4)
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg η")
    ax.set_title(f"Compression ratio η over time\n{s.short_label()}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "eta_over_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["eta"] = p

    # ── Plot 3: Cumulative average delay / (1/R_k) ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in algo_names:
        ax.plot(times, delay_ratio_avg[name], label=name, linewidth=1.5,
                marker=marker_map[name], markevery=markevery, markersize=5)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary (D_k = 1/R_k)")
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg  D_k / (1/R_k)")
    ax.set_title(f"Delay ratio over time\n{s.short_label()}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "delay_over_invR_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["delay"] = p

    # ── Plot 4: Accuracy vs delay scatter at t = T-1 ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in algo_names:
        x = float(acc_avg[name][-1])
        y = float(delay_ratio_avg[name][-1])
        ax.scatter(x, y, marker=marker_map[name], s=100, label=name, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, clip_on=True)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Cumulative avg accuracy at t = T-1")
    ax.set_ylabel("Cumulative avg  D_k / (1/R_k)  at t = T-1")
    ax.set_title(f"Accuracy vs delay trade-off at t = T-1\n{s.short_label()}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s, fontsize=6)
    p = os.path.join(out_dir, "tradeoff.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["tradeoff"] = p

    return paths


# ============================================================================
# Save results
# ============================================================================

def save_results(
    task: InferenceTask,
    hist: Dict,
    s: ExperimentSetting,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    algo_names = hist["algo_names"]

    # Experiment setting JSON
    setting_dict = asdict(s)
    setting_dict["a"]              = {str(k): v for k, v in s.a.items()}
    setting_dict["tau"]            = {str(k): v for k, v in s.tau.items()}
    setting_dict["eta_min_per_link"]= {str(k): v for k, v in s.eta_min_per_link.items()}
    with open(os.path.join(out_dir, "experiment_setting.json"), "w") as f:
        json.dump(setting_dict, f, indent=2, sort_keys=True)

    # Raw time-series as npz
    np.savez_compressed(
        os.path.join(out_dir, "simulation_results.npz"),
        times           = hist["times"],
        c_t_hist        = hist["c_t_hist"],
        algo_names      = np.array(algo_names, dtype=str),
        acc_avg         = np.stack([hist["acc_avg"][n]         for n in algo_names]),
        delay_ratio_avg = np.stack([hist["delay_ratio_avg"][n] for n in algo_names]),
        eta_avg         = np.stack([hist["eta_avg"][n]         for n in algo_names]),
        acc_hist        = np.stack([hist["acc_hist"][n]        for n in algo_names]),
        delay_ratio_hist= np.stack([hist["delay_ratio_hist"][n]for n in algo_names]),
    )


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Run all algorithms + baselines for one experimental setting."
    )

    # Backend
    p.add_argument("--backend", type=str, default="mlp",
                   choices=["mlp", "resnet", "llm"],
                   help="Model backend.  Reads env vars for ResNet/LLM details.")

    # Compressor — for ResNet and MLP
    p.add_argument("--compressor", type=str, default="topk",
                   choices=["topk", "quantization", "llmint8"],
                   help="Compression algorithm for ResNet/MLP.  "
                        "(llmint8 not supported for MLP.)")
    p.add_argument("--compressor_backend", type=str, default="cpu",
                   choices=["cpu", "gpu"],
                   help="cpu = compressors.py round-trip;  gpu = gpu_compressors.py.")

    # LLM-specific
    p.add_argument("--llm_activation_strategy", type=str, default=None,
                   choices=["topk_per_token", "magnitude", "random",
                            "quantization", "llmint8", None],
                   help="LLM activation compression strategy.  "
                        "Overrides LLM_ACTIVATION_STRATEGY env var.  "
                        "For llmint8, also set --compressor_backend gpu.")
    p.add_argument("--llm_model_name", type=str, default=None,
                   help="HuggingFace model ID for LLM backend.  "
                        "Overrides LLM_MODEL_NAME env var.  "
                        "e.g. unsloth/gemma-2b-it or meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--llm_grad_N",     type=int,   default=None,
                   help="Stein N for LLM backend (default: 3).")
    p.add_argument("--llm_fast_samples", type=int, default=None,
                   help="MMLU samples per subject for fast eval (default: 5).")

    # MLP-specific
    p.add_argument("--mlp_sigma", type=float, default=None,
                   help="Stein sigma for MLP backend (default: 0.05).")
    p.add_argument("--mlp_N",     type=int,   default=None,
                   help="Stein N for MLP backend (default: 50).")

    # Channel distribution
    p.add_argument("--c_min", type=float, default=None,
                   help="Channel capacity lower bound.  "
                        "Default: 10 (MLP), 500000 (ResNet/LLM).")
    p.add_argument("--c_max", type=float, default=None,
                   help="Channel capacity upper bound.  "
                        "Default: 150 (MLP), 5000000 (ResNet/LLM).")

    # Task parameters
    p.add_argument("--R_k",     type=float, default=10.0,
                   help="Constant target throughput R_k(t).")
    p.add_argument("--eta_min", type=float, default=None,
                   help="η floor on every link.  "
                        "Default: 0.05 (MLP), 0.125 (ResNet), 0.25 (LLM).")

    # No-CSI algorithm parameters
    p.add_argument("--no_csi_mu",      type=float, default=3.0)
    p.add_argument("--no_csi_epsilon", type=float, default=0.1)

    # Sweep / run parameters
    p.add_argument("--T",    type=int, default=150)
    p.add_argument("--seed", type=int, default=12)

    # Resume
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from checkpoint.npz in --out_dir. "
                        "Loads history, restores RNG, estimators, and lambda_t, "
                        "then continues from T_done to T.")

    # Output
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory.  "
                        "Default: outputs/experiments/<backend>_<compressor>_<seed>")

    # Baselines
    p.add_argument(
        "--baselines", type=str,
        default="max_compression,no_compression,uniform_compression,"
                "myopic,conservative,moving_average",
        help="Comma-separated baseline keys, or 'none'."
    )

    args = p.parse_args()

    # Set backend env var before importing any backend modules
    backend_env = {"mlp": "mnist", "resnet": "resnet", "llm": "llm"}[args.backend]
    os.environ.setdefault("INFERENCE_OPTIMIZER_BACKEND", backend_env)

    # Apply defaults that depend on backend
    backend = args.backend
    c_min   = args.c_min   or (10.0   if backend == "mlp" else 500_000.0)
    c_max   = args.c_max   or (150.0  if backend == "mlp" else 5_000_000.0)
    eta_min = (args.eta_min if args.eta_min is not None else
               (0.05  if backend == "mlp" else
                0.125 if backend == "resnet" else 0.25))

    # LLM activation strategy: CLI arg > env var > default
    llm_strategy = (args.llm_activation_strategy
                    or os.environ.get("LLM_ACTIVATION_STRATEGY", "topk_per_token"))
    llm_model    = (args.llm_model_name
                    or os.environ.get("LLM_MODEL_NAME", "unsloth/gemma-2b-it"))

    # Parse baselines
    bl_arg    = args.baselines.strip().lower()
    baselines = [] if bl_arg == "none" else [b.strip() for b in bl_arg.split(",") if b.strip()]

    # Build setting record — all parameters visible
    s = ExperimentSetting(
        backend            = backend,
        compressor         = args.compressor,
        compressor_backend = args.compressor_backend,
        c_min              = c_min,
        c_max              = c_max,
        c_dist             = "uniform",
        R_k                = args.R_k,
        eta_min            = eta_min,
        T                  = args.T,
        seed               = args.seed,
        no_csi_mu          = args.no_csi_mu,
        no_csi_epsilon     = args.no_csi_epsilon,
        # LLM
        llm_model_name          = llm_model,
        llm_activation_strategy = llm_strategy,
        llm_split_mode          = os.environ.get("LLM_SPLIT_MODE", "thirds"),
        llm_grad_sigma          = float(os.environ.get("LLM_GRAD_SIGMA", "0.05")),
        llm_grad_N              = args.llm_grad_N or int(os.environ.get("LLM_GRAD_N", "3")),
        llm_fast_samples        = args.llm_fast_samples or int(os.environ.get("MMLU_FAST_SAMPLES_PER_SUBJECT", "5")),
        llm_full_samples        = int(os.environ.get("MMLU_FULL_SAMPLES_PER_SUBJECT", "10")),
        llm_batch_size          = int(os.environ.get("MMLU_BATCH_SIZE", "8")),
        # ResNet
        resnet_cutpoints    = os.environ.get("RESNET_CUTPOINTS", "8,14,21"),
        resnet_fast_batches = int(os.environ.get("RESNET_FAST_BATCHES", "1")),
        resnet_grad_sigma   = float(os.environ.get("RESNET_GRAD_SIGMA", "0.05")),
        resnet_grad_N       = int(os.environ.get("RESNET_GRAD_N", "20")),
        # MLP
        mlp_sigma    = args.mlp_sigma or float(os.environ.get("MLP_GRAD_SIGMA", "0.05")),
        mlp_N        = args.mlp_N     or int(os.environ.get("MLP_GRAD_N", "50")),
        mlp_n_samples= int(os.environ.get("MLP_N_SAMPLES", "512")),
    )

    # For LLM + llmint8, force compressor_backend=gpu
    if backend == "llm" and llm_strategy == "llmint8" and args.compressor_backend == "cpu":
        print("[run_experiment] WARNING: llmint8 strategy requires gpu backend. "
              "Setting compressor_backend=gpu.", flush=True)
        s.compressor_backend = "gpu"

    out_dir = args.out_dir or os.path.join(
        "outputs", "experiments",
        f"{backend}_{args.compressor if backend != 'llm' else llm_strategy}"
        f"_{args.compressor_backend}_seed{args.seed}"
    )

    # ── Print full configuration before loading any model ────────────────────
    s.full_printout(out_dir, baselines)

    api, _ = load_backend(s)
    task   = build_task(s, api)

    # ── Print again after topology is derived (a, tau, eta_min_per_link filled) ─
    print("\n[run_experiment] Topology derived — updated configuration:", flush=True)
    s.full_printout(out_dir, baselines)

    algos, myopic_est, cons_est, ma_est, lcb = build_algorithms(task, s, baselines)

    print(f"[run_experiment] Algorithms ({len(algos)}):", flush=True)
    for name, _ in algos:
        print(f"  • {name}", flush=True)
    print(flush=True)

    # ── Load checkpoint if resuming ───────────────────────────────────────────
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(out_dir)
        if checkpoint is None:
            print(f"[run_experiment] WARNING: --resume set but no checkpoint found in "
                  f"{out_dir}. Starting from t=0.", flush=True)
        else:
            T_done = checkpoint["T_done"]
            if T_done >= s.T:
                print(f"[run_experiment] Checkpoint is already complete "
                      f"(T_done={T_done} >= T={s.T}). Nothing to do.", flush=True)
                return
            print(f"[run_experiment] Resuming from t={T_done} → t={s.T-1}.",
                  flush=True)

    hist = run_simulation(task, algos, s, lcb, myopic_est, cons_est, ma_est,
                          out_dir=out_dir, checkpoint_every=1,
                          checkpoint=checkpoint)

    save_results(task, hist, s, out_dir)
    paths = plot_results(task, hist, s, out_dir)

    print("\n" + "=" * 65, flush=True)
    print("Done.", flush=True)
    for label, path in paths.items():
        print(f"  {label:<12} -> {path}", flush=True)
    print(f"  {'setting':<12} -> {os.path.join(out_dir, 'experiment_setting.json')}", flush=True)
    print(f"  {'raw data':<12} -> {os.path.join(out_dir, 'simulation_results.npz')}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
