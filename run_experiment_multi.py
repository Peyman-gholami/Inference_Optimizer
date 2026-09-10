"""
run_experiment_multi.py — Configurable multi-task experiment runner
====================================================================

Runs all multi-task algorithms and baselines for a fully-specified experimental
setting and produces comparison plots where all methods are displayed together.

Mirrors run_experiment.py exactly in structure and interface, but solves the
multi-task problem: K concurrent inference pipelines sharing the same network,
each with its own compression vector eta_k, resource shares, payload sizes a_k,
weight w_k, and deadline R_k.

Key differences from run_experiment.py
---------------------------------------
  - Uses CSIAwareMultiTaskOptimizer  (Theorem 3.4, convex SLSQP)
    and NoCSIMultiTaskOptimizer      (Algorithm 2, BCD + per-task duals)
  - Eight multi-task baselines instead of six single-task baselines
  - Accuracy metric: weighted average (1/W) * sum_k w_k * A_k(eta_k)
  - Delay metric:    task-average    (1/K) * sum_k D_k / (1/R_k)
  - K tasks with heterogeneous payload sizes a_{i,k} and weights w_k
  - Per-task eta history saved in npz (shape A x T x K x L)
  - Extra plot: per-task delay over time

Task heterogeneity
------------------
  By default tasks are made heterogeneous by assigning different per-link
  payload sizes:
    Task 1 (w=1): a = {0: a0*0.5, 1: a0*1.5}   (light link-0, heavy link-1)
    Task 2 (w=2): a = {0: a0*1.5, 1: a0*0.5}   (heavy link-0, light link-1)
    Task k (w=k): interpolated between the two extremes

  where a0 = a[0] measured from the backend (real activation size).
  Override with --heterogeneous_a false to give all tasks identical a values.

Cross-backend mode
------------------
  Use --task_backends to assign a different backend to each task.
  Each task gets its own model, compressor, and accuracy callable.
  All tasks share the same network (same c_t per slot) and the same
  num_links / L_k topology (3 nodes, 2 links).

  Example: ResNet (task 1) + Llama-3.1-8B (task 2):
    python run_experiment_multi.py \\
        --task_backends resnet llm \\
        --task_compressors topk topk_per_token \\
        --task_llm_models "" "meta-llama/Llama-3.1-8B-Instruct" \\
        --c_min 500000 --c_max 5000000 --T 50

  In cross-backend mode:
    - Each task's a[i] is derived independently from its own backend
    - eta_min can be set per-task via --task_eta_mins
    - --backend, --compressor, --llm_* still set the defaults for any
      task that does not override them

Usage
-----
  # ResNet, 2 tasks, transitional regime:
  python run_experiment_multi.py --backend resnet --K 2 \\
      --c_min 60000 --c_max 300000 --T 50

  # ResNet, 3 tasks:
  python run_experiment_multi.py --backend resnet --K 3 \\
      --c_min 60000 --c_max 300000 --T 50

  # LLM Llama-3.1-8B, 2 tasks:
  python run_experiment_multi.py --backend llm --K 2 \\
      --llm_model_name meta-llama/Llama-3.1-8B-Instruct \\
      --llm_activation_strategy topk_per_token \\
      --c_min 10000000 --c_max 40000000 --T 30

  # Cross-backend: ResNet (task 1) + Llama-3.1-8B (task 2)
  python run_experiment_multi.py \\
      --task_backends resnet llm \\
      --task_compressors topk topk_per_token \\
      --task_llm_models "" "meta-llama/Llama-3.1-8B-Instruct" \\
      --c_min 500000 --c_max 5000000 --T 50

  # No baselines (fastest):
  python run_experiment_multi.py --backend resnet --K 2 \\
      --c_min 60000 --c_max 300000 --baselines none --T 50

Output
------
  <out_dir>/
    experiment_setting_multi.json  — full setting record
    accuracy_over_time.png         — cumulative avg weighted accuracy
    delay_over_invR_time.png       — cumulative avg (1/K) sum_k D_k/(1/R_k)
    delay_per_task_time.png        — per-task delay ratio over time
    eta_over_time.png              — cumulative avg eta per task per link
    tradeoff.png                   — weighted accuracy vs avg delay scatter
    simulation_results_multi.npz   — raw time series
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
from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer
from src.optimizers.no_csi import NoCSIMultiTaskOptimizer
from src.optimizers.estimators import MeanEstimator, MeanMinusZStdLCB
from src.optimizers.baseline import (
    MaxCompressionMultiTaskBaseline,
    NoCompressionMultiTaskBaseline,
    StaticEqualShareMultiTaskBaseline,
    ProportionalResourceAllocationMultiTaskBaseline,
    StrictPriorityGreedyMultiTaskBaseline,
    DecoupledEqualSplitStochasticDescentMultiTaskBaseline,
    QueueProportionalHeuristicMultiTaskBaseline,
    HistoricalAverageCertaintyEquivalenceMultiTaskBaseline,
)


# ============================================================================
# Experiment setting dataclass
# ============================================================================

@dataclass
class MultiTaskExperimentSetting:
    """
    Full specification of one multi-task experimental setting.
    Serialisable to JSON and printed at startup with (default) markers.
    """
    # ── Core ──────────────────────────────────────────────────────────────────
    backend:             str
    compressor:          str
    compressor_backend:  str
    c_min:               float
    c_max:               float
    c_dist:              str
    R_k_per_task:        List[float]
    eta_min:             float
    T:                   int
    seed:                int
    K:                   int            # number of tasks
    no_csi_mu:           float
    no_csi_epsilon:      float
    no_csi_J:            int            # BCD inner iterations
    heterogeneous_a:     bool           # whether to vary a_{i,k} across tasks

    # ── LLM-specific ──────────────────────────────────────────────────────────
    llm_model_name:          str   = ""
    llm_activation_strategy: str   = ""
    llm_split_mode:          str   = "thirds"
    llm_grad_sigma:          float = 0.05
    llm_grad_N:              int   = 3
    llm_fast_samples:        int   = 5
    llm_full_samples:        int   = 10
    llm_batch_size:          int   = 8

    # ── ResNet-specific ───────────────────────────────────────────────────────
    resnet_cutpoints:    str   = "8,14,21"
    resnet_fast_batches: int   = 1
    resnet_grad_sigma:   float = 0.05
    resnet_grad_N:       int   = 20

    # ── MLP-specific ──────────────────────────────────────────────────────────
    mlp_sigma:           float = 0.05
    mlp_N:               int   = 50
    mlp_n_samples:       int   = 512

    # ── Cross-backend mode ────────────────────────────────────────────────────
    # When non-empty, each entry specifies the backend for task k.
    # Length must equal K.  Empty list = single-backend mode (all tasks use `backend`).
    task_backends:    List[str] = field(default_factory=list)
    # Per-task compressor/strategy strings (length K or empty for single-backend)
    task_compressors: List[str] = field(default_factory=list)
    # Per-task LLM model names (length K or empty; "" = use llm_model_name default)
    task_llm_models:  List[str] = field(default_factory=list)
    # Per-task eta_min overrides (length K or empty)
    task_eta_mins:    List[float] = field(default_factory=list)
    # Per-task weight overrides w_k (length K or empty).
    # Default: w_k = float(task_id) i.e. task 1 → 1.0, task 2 → 2.0, etc.
    task_weight_overrides: List[float] = field(default_factory=list)
    num_links:           int  = 0
    L_k:                 int  = 0
    a_base:              Dict[int, float] = field(default_factory=dict)
    tau:                 Dict[int, float] = field(default_factory=dict)
    task_a:              Dict[int, Dict[int, float]] = field(default_factory=dict)
    task_weights:        Dict[int, float] = field(default_factory=dict)

    def _is_cross_backend(self) -> bool:
        return len(self.task_backends) > 0

    def _r_k_str(self) -> str:
        """Compact R_k display: single value if all equal, list otherwise."""
        vals = self.R_k_per_task
        if not vals:
            return "?"
        if len(set(vals)) == 1:
            return str(vals[0])
        return "[" + ",".join(str(v) for v in vals) + "]"

    def short_label(self) -> str:
        r_str = self._r_k_str()
        if self._is_cross_backend():
            backends_str = "+".join(self.task_backends)
            compressors_str = "+".join(self.task_compressors) if self.task_compressors else "mixed"
            return (
                f"CROSS({backends_str}) | {compressors_str} | "
                f"K={self.K} | c~U[{self.c_min:.0f},{self.c_max:.0f}] | "
                f"R_k={r_str} | T={self.T} seed={self.seed}"
            )
        if self.backend == "llm":
            model_short = self.llm_model_name.split("/")[-1] if self.llm_model_name else "llm"
            comp_str    = self.llm_activation_strategy or self.compressor
            return (
                f"LLM({model_short}) | {comp_str}/{self.compressor_backend} | "
                f"K={self.K} | c~U[{self.c_min:.0f},{self.c_max:.0f}] | "
                f"R_k={r_str} | η_min={self.eta_min} | T={self.T} seed={self.seed}"
            )
        return (
            f"{self.backend.upper()} | {self.compressor}/{self.compressor_backend} | "
            f"K={self.K} | c~U[{self.c_min:.0f},{self.c_max:.0f}] | "
            f"R_k={r_str} | η_min={self.eta_min} | T={self.T} seed={self.seed}"
        )

    def detail_label(self) -> str:
        r_str = self._r_k_str()
        if self._is_cross_backend():
            lines = [
                f"CROSS-BACKEND  K={self.K}  c~U[{self.c_min:.0f},{self.c_max:.0f}]"
                f"  R_k={r_str}  T={self.T}  seed={self.seed}",
                f"μ={self.no_csi_mu}  ε={self.no_csi_epsilon}  J={self.no_csi_J}",
            ]
            for k_idx in range(self.K):
                bk = self.task_backends[k_idx] if k_idx < len(self.task_backends) else "?"
                ck = self.task_compressors[k_idx] if k_idx < len(self.task_compressors) else "?"
                mk = self.task_llm_models[k_idx] if k_idx < len(self.task_llm_models) else ""
                em = self.task_eta_mins[k_idx] if k_idx < len(self.task_eta_mins) else self.eta_min
                rk = self.R_k_per_task[k_idx] if k_idx < len(self.R_k_per_task) else "?"
                model_info = f"  model={mk.split('/')[-1]}" if mk else ""
                lines.append(f"  task{k_idx+1}: backend={bk}  compressor={ck}{model_info}  η_min={em}  R_k={rk}")
        elif self.backend == "llm":
            model_short = self.llm_model_name.split("/")[-1] if self.llm_model_name else "llm"
            comp_str    = self.llm_activation_strategy or self.compressor
            lines = [
                f"LLM({model_short})  strategy={comp_str}/{self.compressor_backend}  split={self.llm_split_mode}",
                f"K={self.K}  c~U[{self.c_min:.0f},{self.c_max:.0f}]  R_k={r_str}  T={self.T}  seed={self.seed}",
                f"η_min={self.eta_min}  μ={self.no_csi_mu}  ε={self.no_csi_epsilon}  J={self.no_csi_J}",
                f"σ={self.llm_grad_sigma}  N={self.llm_grad_N}  fast_samples/subj={self.llm_fast_samples}",
            ]
        elif self.backend == "resnet":
            lines = [
                f"ResNet56  compressor={self.compressor}/{self.compressor_backend}  cutpoints={self.resnet_cutpoints}",
                f"K={self.K}  c~U[{self.c_min:.0f},{self.c_max:.0f}]  R_k={r_str}  T={self.T}  seed={self.seed}",
                f"η_min={self.eta_min}  μ={self.no_csi_mu}  ε={self.no_csi_epsilon}  J={self.no_csi_J}",
                f"σ={self.resnet_grad_sigma}  N={self.resnet_grad_N}  fast_batches={self.resnet_fast_batches}",
            ]
        else:
            lines = [
                f"MLP/MNIST  compressor={self.compressor}/{self.compressor_backend}",
                f"K={self.K}  c~U[{self.c_min:.0f},{self.c_max:.0f}]  R_k={r_str}  T={self.T}  seed={self.seed}",
                f"η_min={self.eta_min}  μ={self.no_csi_mu}  ε={self.no_csi_epsilon}  J={self.no_csi_J}",
                f"σ={self.mlp_sigma}  N={self.mlp_N}  n_samples={self.mlp_n_samples}",
            ]
        if self.num_links > 0:
            lines.append(
                f"links={self.num_links}  a_base={[f'{self.a_base[i]:.0f}' for i in range(self.num_links)]}"
            )
            for tid, a_k in self.task_a.items():
                w  = self.task_weights.get(tid, 1.0)
                rk = self.R_k_per_task[tid - 1] if (tid - 1) < len(self.R_k_per_task) else "?"
                lines.append(
                    f"  task{tid}(w={w:.0f}, R_k={rk}): a={[f'{a_k[i]:.0f}' for i in range(self.num_links)]}"
                )
        return "\n".join(lines)

    def full_printout(self, out_dir: str, baselines: list) -> None:
        W = 65
        print("=" * W, flush=True)
        print("run_experiment_multi.py  —  Full experiment configuration", flush=True)
        print("=" * W, flush=True)

        print("  [Common]", flush=True)
        print(f"    backend            = {self.backend}", flush=True)
        print(f"    K (num tasks)      = {self.K}", flush=True)
        print(f"    T                  = {self.T}"
              + ("  (default)" if self.T == 150 else ""), flush=True)
        print(f"    seed               = {self.seed}"
              + ("  (default)" if self.seed == 12 else ""), flush=True)
        print(f"    c_dist             = {self.c_dist}  [{self.c_min:.0f}, {self.c_max:.0f}]",
              flush=True)
        print(f"    R_k_per_task       = {self.R_k_per_task}"
              + ("  (all equal)" if len(set(self.R_k_per_task)) == 1 else ""), flush=True)
        print(f"    eta_min            = {self.eta_min}", flush=True)
        print(f"    heterogeneous_a    = {self.heterogeneous_a}"
              + ("  (default)" if self.heterogeneous_a else ""), flush=True)

        print("  [No-CSI Multi-Task (Alg2)]", flush=True)
        print(f"    no_csi_mu          = {self.no_csi_mu}"
              + ("  (default)" if self.no_csi_mu == 3.0 else ""), flush=True)
        print(f"    no_csi_epsilon     = {self.no_csi_epsilon}"
              + ("  (default)" if self.no_csi_epsilon == 0.1 else ""), flush=True)
        print(f"    no_csi_J           = {self.no_csi_J}"
              + ("  (default)" if self.no_csi_J == 10 else ""), flush=True)

        if self._is_cross_backend():
            print("  [Cross-backend tasks]", flush=True)
            for k_idx in range(self.K):
                bk  = self.task_backends[k_idx]   if k_idx < len(self.task_backends)   else "?"
                ck  = self.task_compressors[k_idx] if k_idx < len(self.task_compressors) else "?"
                mk  = self.task_llm_models[k_idx]  if k_idx < len(self.task_llm_models)  else ""
                em  = self.task_eta_mins[k_idx]    if k_idx < len(self.task_eta_mins)    else self.eta_min
                print(f"    task {k_idx+1}: backend={bk}  compressor={ck}  "
                      f"η_min={em}"
                      + (f"  model={mk}" if mk else ""), flush=True)
        elif self.backend == "llm":
            print(f"    activation_strategy= {self.llm_activation_strategy}"
                  + ("  (default)" if self.llm_activation_strategy == "topk_per_token" else ""),
                  flush=True)
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

        if self.num_links > 0:
            print("  [Topology — derived]", flush=True)
            print(f"    L_k                = {self.L_k}", flush=True)
            print(f"    num_links          = {self.num_links}", flush=True)
            print(f"    a_base (bytes)     = "
                  f"{[f'{self.a_base[i]:.0f}' for i in range(self.num_links)]}", flush=True)
            print(f"    tau (s)            = "
                  f"{[f'{self.tau[i]:.6f}' for i in range(self.L_k)]}", flush=True)
            for tid in sorted(self.task_a.keys()):
                a_k = self.task_a[tid]
                w   = self.task_weights.get(tid, 1.0)
                print(f"    task {tid} (w={w:.0f}): a = "
                      f"{[f'{a_k[i]:.0f}' for i in range(self.num_links)]}", flush=True)

        print("  [Run]", flush=True)
        print(f"    baselines          = {baselines or '(none)'}", flush=True)
        print(f"    out_dir            = {out_dir}", flush=True)
        print("=" * W, flush=True)


# ============================================================================
# Backend loading
# ============================================================================

def load_single_backend(
    backend:             str,
    compressor:          str,
    compressor_backend:  str,
    eta_min:             float,
    s:                   "MultiTaskExperimentSetting",
    llm_model_name:      str = "",
    llm_activation_strategy: str = "",
):
    """
    Load one backend and return its api object.
    Sets environment variables appropriately for the given backend.
    Does NOT permanently modify env vars used by other backends.
    """
    # Determine effective LLM strategy / model for this task
    effective_strategy = llm_activation_strategy or s.llm_activation_strategy or "topk_per_token"
    effective_model    = llm_model_name or s.llm_model_name or "unsloth/gemma-2b-it"
    effective_cb       = compressor_backend
    if backend == "llm" and effective_strategy == "llmint8":
        effective_cb = "gpu"

    if backend == "resnet":
        os.environ['INFERENCE_OPTIMIZER_BACKEND'] = 'resnet'
        os.environ['COMPRESSOR_TYPE']    = compressor
        os.environ['COMPRESSOR_BACKEND'] = effective_cb
        from src.core.resnet_task_callables import build_resnet_backend_from_env
        api = build_resnet_backend_from_env()
        return api

    if backend == "llm":
        os.environ['INFERENCE_OPTIMIZER_BACKEND'] = 'llm'
        os.environ['COMPRESSOR_TYPE']    = compressor
        os.environ['COMPRESSOR_BACKEND'] = effective_cb
        os.environ['LLM_ACTIVATION_STRATEGY'] = effective_strategy
        os.environ['LLM_MODEL_NAME']          = effective_model
        os.environ.setdefault('LLM_SPLIT_MODE',               s.llm_split_mode)
        os.environ.setdefault('LLM_GRAD_SIGMA',                str(s.llm_grad_sigma))
        os.environ.setdefault('LLM_GRAD_N',                    str(s.llm_grad_N))
        os.environ.setdefault('MMLU_FAST_SAMPLES_PER_SUBJECT', str(s.llm_fast_samples))
        os.environ.setdefault('MMLU_FULL_SAMPLES_PER_SUBJECT', str(s.llm_full_samples))
        os.environ.setdefault('MMLU_BATCH_SIZE',               str(s.llm_batch_size))
        from src.core.llm_task_callables import build_llm_backend_from_env
        api = build_llm_backend_from_env(eta_min=eta_min)
        return api

    # MLP fallback
    import torch, torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from src.core.task_callables import InferenceTaskCallables
    from src.core.stein_oracles import MLP
    from src.core.accuracy_fn import CompressedModel

    os.environ['INFERENCE_OPTIMIZER_BACKEND'] = 'mnist'
    os.environ['COMPRESSOR_TYPE']    = compressor
    os.environ['COMPRESSOR_BACKEND'] = effective_cb

    torch.manual_seed(s.seed)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_root = os.environ.get("MNIST_DATA_ROOT", "./data")
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        datasets.MNIST(mnist_root, train=True,  download=True, transform=tf),
        batch_size=128, shuffle=True)
    test_loader  = DataLoader(
        datasets.MNIST(mnist_root, train=False, download=True, transform=tf),
        batch_size=256, shuffle=False)
    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[run_experiment_multi] Training MLP for task (3 epochs)...", flush=True)
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    eta_init = torch.ones(2, device=device)
    cm  = CompressedModel(model, eta_init, compressor=compressor)

    class _DL:
        def __init__(self, l, d): self._l = l; self._d = d
        def __iter__(self):
            for x, y in self._l: yield x.to(self._d), y.to(self._d)
        def __len__(self): return len(self._l)

    api = InferenceTaskCallables(
        cm, _DL(test_loader, device),
        sigma=s.mlp_sigma, N=s.mlp_N, n_samples=s.mlp_n_samples,
        flat_theta=None, device=device, compressor=compressor)
    return api


def load_all_backends(s: MultiTaskExperimentSetting) -> List:
    """
    Load one api per task.
    In single-backend mode all tasks share the same api instance.
    In cross-backend mode each task gets its own api instance.
    Returns a list of length K.
    """
    if not s._is_cross_backend():
        # Single-backend: load once, share across all K tasks
        os.environ['COMPRESSOR_TYPE']    = s.compressor
        os.environ['COMPRESSOR_BACKEND'] = s.compressor_backend
        backend_env = {"mlp": "mnist", "resnet": "resnet", "llm": "llm"}[s.backend]
        os.environ['INFERENCE_OPTIMIZER_BACKEND'] = backend_env

        if s.backend == "llm":
            if s.llm_activation_strategy:
                os.environ['LLM_ACTIVATION_STRATEGY'] = s.llm_activation_strategy
            if s.llm_model_name:
                os.environ['LLM_MODEL_NAME'] = s.llm_model_name
            os.environ.setdefault('LLM_SPLIT_MODE',               s.llm_split_mode)
            os.environ.setdefault('LLM_GRAD_SIGMA',                str(s.llm_grad_sigma))
            os.environ.setdefault('LLM_GRAD_N',                    str(s.llm_grad_N))
            os.environ.setdefault('MMLU_FAST_SAMPLES_PER_SUBJECT', str(s.llm_fast_samples))
            os.environ.setdefault('MMLU_FULL_SAMPLES_PER_SUBJECT', str(s.llm_full_samples))
            os.environ.setdefault('MMLU_BATCH_SIZE',               str(s.llm_batch_size))
            s.llm_model_name          = os.environ.get('LLM_MODEL_NAME', s.llm_model_name)
            s.llm_activation_strategy = os.environ.get('LLM_ACTIVATION_STRATEGY', s.llm_activation_strategy)
            from src.core.llm_task_callables import build_llm_backend_from_env
            api = build_llm_backend_from_env(eta_min=s.eta_min)
        elif s.backend == "resnet":
            s.resnet_cutpoints    = os.environ.get('RESNET_CUTPOINTS',    s.resnet_cutpoints)
            s.resnet_fast_batches = int(os.environ.get('RESNET_FAST_BATCHES', str(s.resnet_fast_batches)))
            s.resnet_grad_sigma   = float(os.environ.get('RESNET_GRAD_SIGMA',  str(s.resnet_grad_sigma)))
            s.resnet_grad_N       = int(os.environ.get('RESNET_GRAD_N',        str(s.resnet_grad_N)))
            from src.core.resnet_task_callables import build_resnet_backend_from_env
            api = build_resnet_backend_from_env()
        else:
            api = load_single_backend("mlp", s.compressor, s.compressor_backend,
                                      s.eta_min, s)
        return [api] * s.K

    # Cross-backend: load each task's backend independently
    print("[run_experiment_multi] Cross-backend mode — loading one backend per task",
          flush=True)
    apis = []
    for k_idx in range(s.K):
        bk  = s.task_backends[k_idx]   if k_idx < len(s.task_backends)   else s.backend
        ck  = s.task_compressors[k_idx] if k_idx < len(s.task_compressors) else s.compressor
        mk  = s.task_llm_models[k_idx]  if k_idx < len(s.task_llm_models)  else s.llm_model_name
        em  = s.task_eta_mins[k_idx]    if k_idx < len(s.task_eta_mins)    else s.eta_min
        cb  = "gpu" if (bk == "llm" and ck == "llmint8") else s.compressor_backend
        print(f"  Loading task {k_idx+1}: backend={bk}  compressor={ck}  "
              f"eta_min={em}" + (f"  model={mk}" if mk else ""), flush=True)
        api = load_single_backend(bk, ck, cb, em, s,
                                  llm_model_name=mk,
                                  llm_activation_strategy=ck if bk == "llm" else "")
        apis.append(api)
    return apis


# ============================================================================
# Task builder
# ============================================================================

def build_tasks(
    s:    MultiTaskExperimentSetting,
    apis: List,                        # length K — one api per task
) -> List[InferenceTask]:
    """
    Build K InferenceTask objects, one api per task.
    In single-backend mode apis is [same_api] * K.
    In cross-backend mode each api is different.
    Fills s.num_links, s.L_k, s.a_base, s.tau, s.task_a, s.task_weights in-place.
    """
    R_k_callable = lambda t, *_, r=s.R_k_per_task[0]: r  # default for task 0; overridden per-task below

    # ── Derive topology from task 0's api (all tasks share same L_k / num_links) ─
    api0   = apis[0]
    bk0    = s.task_backends[0] if s._is_cross_backend() else s.backend

    if bk0 == "resnet":
        num_links = api0.num_links
        num_nodes = num_links + 1
        raw_a0    = api0.measured_a()
        a_base    = {i: float(raw_a0[i]) for i in range(num_links)}
        tau       = {i: 0.001 for i in range(num_nodes)}
    else:
        from src.core.llm_plug_in import maybe_measured_tau_a
        measured = maybe_measured_tau_a()
        if measured is not None:
            tau, a_base = measured
            a_base = {k: float(v) for k, v in a_base.items()}
            tau    = {k: float(v) for k, v in tau.items()}
        else:
            tau    = {0: 0.01, 1: 0.01, 2: 0.01}
            a_base = {0: 512.0, 1: 256.0}
        num_links = 2
        num_nodes = 3

    s.num_links = num_links
    s.L_k       = num_nodes

    s.a_base    = a_base
    s.tau       = tau

    tasks = []
    for k_idx, api in enumerate(apis):
        task_id = k_idx + 1
        w_k = (s.task_weight_overrides[k_idx]
               if k_idx < len(s.task_weight_overrides)
               else float(task_id))

        # Per-task a[i]: in cross-backend mode, derive from each task's own api.
        # In single-backend mode, apply the heterogeneous scaling as before.
        bk = s.task_backends[k_idx] if s._is_cross_backend() and k_idx < len(s.task_backends) \
             else s.backend

        if s._is_cross_backend():
            # Each task gets its real activation size from its own model
            if bk == "resnet":
                raw_a = api.measured_a()
                a_k   = {i: float(raw_a[i]) for i in range(num_links)}
            else:
                from src.core.llm_plug_in import maybe_measured_tau_a
                measured_k = maybe_measured_tau_a()
                if measured_k is not None:
                    _, a_k_raw = measured_k
                    a_k = {i: float(a_k_raw[i]) for i in range(num_links)}
                else:
                    a_k = {0: 512.0, 1: 256.0}
            # tau is shared (topology is fixed)
        else:
            # Single-backend heterogeneous scaling
            def _task_a(k_idx: int, K: int, a_base: Dict[int, float]) -> Dict[int, float]:
                if not s.heterogeneous_a or K == 1:
                    return dict(a_base)
                frac = k_idx / max(1, K - 1)
                scales = []
                for i in range(num_links):
                    if i % 2 == 0:
                        scales.append(0.5 + frac * 1.0)
                    else:
                        scales.append(1.5 - frac * 1.0)
                return {i: a_base[i] * scales[i] for i in range(num_links)}
            a_k = _task_a(k_idx, s.K, a_base)

        # eta_min: per-task override in cross-backend mode, else global
        em = s.task_eta_mins[k_idx] if (s._is_cross_backend()
                                         and k_idx < len(s.task_eta_mins)) else s.eta_min

        eta_min_d = {i: em for i in range(num_links)}

        r_k = s.R_k_per_task[k_idx] if k_idx < len(s.R_k_per_task) else s.R_k_per_task[-1]
        R_k_callable = lambda t, *_, r=r_k: r

        task = InferenceTask(
            task_id=task_id, b_k=0, L_k=num_nodes,
            tau=tau, a=a_k,
            eta_min=eta_min_d,
            R_k_callable=R_k_callable,
            w_k=w_k,
            accuracy_callable=api.accuracy_callable,
            accuracy_callable_true=api.accuracy_callable_true,
            gradient_callable=api.gradient_callable,
        )
        tasks.append(task)
        s.task_a[task_id]       = a_k
        s.task_weights[task_id] = w_k

    print(f"[run_experiment_multi] Built {s.K} tasks:", flush=True)
    for task in tasks:
        print(f"  task {task.task_id} (w={task.w_k:.0f}): "
              f"a={[f'{task.a[i]:.0f}' for i in range(num_links)]}", flush=True)

    return tasks


# ============================================================================
# Cross-backend helpers
# ============================================================================

def _load_one_backend(
    backend:            str,
    compressor:         str,
    compressor_backend: str,
    eta_min:            float,
    s:                  MultiTaskExperimentSetting,
) -> object:
    """
    Load a single backend api for use in cross-backend mode.
    Sets the relevant env vars, imports the backend, and returns the api object.
    The caller is responsible for restoring env vars if needed.
    """
    os.environ['COMPRESSOR_TYPE']    = compressor
    os.environ['COMPRESSOR_BACKEND'] = compressor_backend
    os.environ['INFERENCE_OPTIMIZER_BACKEND'] = (
        "resnet" if backend == "resnet" else
        "llm"    if backend == "llm"    else
        "mnist"
    )

    if backend == "resnet":
        from src.core.resnet_task_callables import build_resnet_backend_from_env
        return build_resnet_backend_from_env()

    if backend == "llm":
        from src.core.llm_task_callables import build_llm_backend_from_env
        return build_llm_backend_from_env(eta_min=eta_min)

    # MLP
    import torch, torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from src.core.task_callables import InferenceTaskCallables
    from src.core.stein_oracles import MLP
    from src.core.accuracy_fn import CompressedModel

    torch.manual_seed(s.seed)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_root = os.environ.get("MNIST_DATA_ROOT", "./data")
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        datasets.MNIST(mnist_root, train=True,  download=True, transform=tf),
        batch_size=128, shuffle=True)
    test_loader  = DataLoader(
        datasets.MNIST(mnist_root, train=False, download=True, transform=tf),
        batch_size=256, shuffle=False)
    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[run_experiment_multi] Training MLP for task (3 epochs)...", flush=True)
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    eta_init = torch.ones(2, device=device)
    cm  = CompressedModel(model, eta_init, compressor=compressor)

    class _DL:
        def __init__(self, l, d): self._l = l; self._d = d
        def __iter__(self):
            for x, y in self._l: yield x.to(self._d), y.to(self._d)
        def __len__(self): return len(self._l)

    return InferenceTaskCallables(
        cm, _DL(test_loader, device),
        sigma=s.mlp_sigma, N=s.mlp_N, n_samples=s.mlp_n_samples,
        flat_theta=None, device=device, compressor=compressor)


def _derive_a_tau_for_backend(backend: str, api: object) -> Tuple[Dict, Dict, int, int]:
    """
    Derive a_base, tau, num_links, num_nodes from a loaded api object.
    Returns (a_base, tau, num_links, num_nodes).
    """
    from src.core.llm_plug_in import maybe_measured_tau_a

    if backend == "resnet":
        num_links = api.num_links
        num_nodes = num_links + 1
        raw_a     = api.measured_a()
        a_base    = {i: float(raw_a[i]) for i in range(num_links)}
        tau       = {i: 0.001 for i in range(num_nodes)}
        return a_base, tau, num_links, num_nodes

    # LLM or MLP
    measured = maybe_measured_tau_a()
    if measured is not None:
        tau, a_base = measured
        a_base = {k: float(v) for k, v in a_base.items()}
        tau    = {k: float(v) for k, v in tau.items()}
    else:
        tau    = {0: 0.01, 1: 0.01, 2: 0.01}
        a_base = {0: 512.0, 1: 256.0}
    return a_base, tau, 2, 3


def build_tasks_cross(s: MultiTaskExperimentSetting) -> List[InferenceTask]:
    """
    Cross-backend task builder.
    Loads one api per task according to s.task_backends / s.task_compressors /
    s.task_llm_models / s.task_eta_mins, derives per-task a[i] from the real
    model activations, and builds one InferenceTask per task.

    The shared topology (num_links, L_k, tau) is taken from task 1.
    All tasks must produce the same number of links — this is enforced by
    the fixed 3-node / 2-link topology used throughout the codebase.
    """
    R_k_callable = lambda t, *_, r=s.R_k_per_task[0]: r  # placeholder; overridden per-task below
    tasks        = []
    ref_num_links = None
    ref_num_nodes = None
    ref_tau       = None

    for k_idx in range(s.K):
        task_id  = k_idx + 1
        backend  = s.task_backends[k_idx]
        comp     = s.task_compressors[k_idx] if k_idx < len(s.task_compressors) else s.compressor
        cb       = "gpu" if (backend == "llm" and "llmint8" in comp) else s.compressor_backend
        eta_min_k= s.task_eta_mins[k_idx] if k_idx < len(s.task_eta_mins) else s.eta_min
        llm_model= s.task_llm_models[k_idx] if k_idx < len(s.task_llm_models) else s.llm_model_name

        # Set LLM model env var for this task if needed
        if backend == "llm" and llm_model:
            os.environ['LLM_MODEL_NAME']          = llm_model
            os.environ['LLM_ACTIVATION_STRATEGY'] = comp
            # Propagate shared LLM settings
            os.environ.setdefault('LLM_SPLIT_MODE',               s.llm_split_mode)
            os.environ.setdefault('LLM_GRAD_SIGMA',                str(s.llm_grad_sigma))
            os.environ.setdefault('LLM_GRAD_N',                    str(s.llm_grad_N))
            os.environ.setdefault('MMLU_FAST_SAMPLES_PER_SUBJECT', str(s.llm_fast_samples))
            os.environ.setdefault('MMLU_FULL_SAMPLES_PER_SUBJECT', str(s.llm_full_samples))
            os.environ.setdefault('MMLU_BATCH_SIZE',               str(s.llm_batch_size))

        print(f"\n[run_experiment_multi] Loading backend for task {task_id}:", flush=True)
        print(f"  backend={backend}  compressor={comp}/{cb}  eta_min={eta_min_k}"
              + (f"  model={llm_model}" if backend == "llm" else ""), flush=True)

        api = _load_one_backend(backend, comp, cb, eta_min_k, s)

        a_base, tau, num_links, num_nodes = _derive_a_tau_for_backend(backend, api)

        # Enforce consistent topology across tasks
        if ref_num_links is None:
            ref_num_links = num_links
            ref_num_nodes = num_nodes
            ref_tau       = tau
            s.num_links   = num_links
            s.L_k         = num_nodes
            s.tau         = tau
            s.a_base      = a_base      # store task-1 a_base as reference
        else:
            if num_links != ref_num_links:
                raise ValueError(
                    f"Task {task_id} backend '{backend}' produces {num_links} links "
                    f"but task 1 produced {ref_num_links}. "
                    f"All tasks must share the same 3-node / 2-link topology."
                )

        # Per-task a: use the real measured values directly (no scaling in cross-backend)
        # The a[i] values already differ enormously between backends
        # (ResNet: ~65KB, Llama: ~4MB) which is the natural heterogeneity.
        a_k     = dict(a_base)
        w_k     = (s.task_weight_overrides[k_idx]
                   if k_idx < len(s.task_weight_overrides)
                   else float(task_id))
        eta_min = {i: eta_min_k for i in range(num_links)}

        r_k = s.R_k_per_task[k_idx] if k_idx < len(s.R_k_per_task) else s.R_k_per_task[-1]
        task_R_k_callable = lambda t, *_, r=r_k: r

        task = InferenceTask(
            task_id=task_id, b_k=0, L_k=num_nodes,
            tau=ref_tau, a=a_k,
            eta_min=eta_min,
            R_k_callable=task_R_k_callable,
            w_k=w_k,
            accuracy_callable=api.accuracy_callable,
            accuracy_callable_true=api.accuracy_callable_true,
            gradient_callable=api.gradient_callable,
        )
        tasks.append(task)
        s.task_a[task_id]       = a_k
        s.task_weights[task_id] = w_k

        print(f"  task {task_id} (w={w_k:.0f}): "
              f"a={[f'{a_k[i]:.0f}' for i in range(num_links)]}", flush=True)

    return tasks

def build_algorithms(
    tasks: List[InferenceTask],
    s:     MultiTaskExperimentSetting,
    baselines: List[str],
) -> Tuple[List[Tuple[str, BaseOptimizer]], Optional[object], Optional[object]]:
    M         = tasks[0].L_k
    num_links = M - 1

    lcb = MeanMinusZStdLCB(
        num_links=num_links, warmup_value=(s.c_min + s.c_max) / 2.0,
        z=1.2815515655446004, warmup=5,
    )
    hist_mean = None

    algos: List[Tuple[str, BaseOptimizer]] = [
        ("CSI-aware (optimal)",
         CSIAwareMultiTaskOptimizer(M=M, tasks=tasks)),
        ("No-CSI (Alg2) + LCB",
         NoCSIMultiTaskOptimizer(M=M, tasks=tasks,
                                  mu=s.no_csi_mu,
                                  epsilon=s.no_csi_epsilon,
                                  J=s.no_csi_J)),
    ]

    baseline_catalog = {
        "max_compression":      ("Baseline: max compression",
                                  MaxCompressionMultiTaskBaseline(M=M, tasks=tasks)),
        "no_compression":       ("Baseline: no compression",
                                  NoCompressionMultiTaskBaseline(M=M, tasks=tasks)),
        "equal_share":          ("Baseline: equal-share",
                                  StaticEqualShareMultiTaskBaseline(M=M, tasks=tasks)),
        "proportional":         ("Baseline: proportional",
                                  ProportionalResourceAllocationMultiTaskBaseline(M=M, tasks=tasks)),
        "strict_priority":      ("Baseline: strict priority",
                                  StrictPriorityGreedyMultiTaskBaseline(M=M, tasks=tasks)),
        "decoupled_equal_split":("No-CSI Baseline: decoupled equal-split",
                                  DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
                                      M=M, tasks=tasks,
                                      mu=s.no_csi_mu, epsilon=s.no_csi_epsilon)),
        "queue_proportional":   ("No-CSI Baseline: queue-proportional",
                                  QueueProportionalHeuristicMultiTaskBaseline(
                                      M=M, tasks=tasks,
                                      mu=s.no_csi_mu, epsilon=s.no_csi_epsilon)),
        "certainty_equivalence":("No-CSI Baseline: certainty equivalence",
                                  HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(
                                      M=M, tasks=tasks)),
    }

    selected_set = set(baselines)
    for key in baselines:
        if key not in baseline_catalog:
            raise ValueError(
                f"Unknown baseline '{key}'. Valid: {sorted(baseline_catalog.keys())}"
            )
        algos.append(baseline_catalog[key])

    if "certainty_equivalence" in selected_set:
        hist_mean = MeanEstimator(num_links=num_links, warmup_value=s.c_min, warmup=5)

    return algos, lcb, hist_mean


# ============================================================================
# Simulation loop
# ============================================================================

def _no_csi_names() -> set:
    return {
        "No-CSI (Alg2) + LCB",
        "No-CSI Baseline: decoupled equal-split",
        "No-CSI Baseline: queue-proportional",
    }


def run_simulation(
    tasks:     List[InferenceTask],
    algos:     List[Tuple[str, BaseOptimizer]],
    s:         MultiTaskExperimentSetting,
    lcb,
    hist_mean,
) -> Dict:
    rng       = np.random.default_rng(s.seed)
    T         = s.T
    K         = len(tasks)
    num_links = tasks[0].L_k - 1
    times     = np.arange(T)
    algo_names= [n for n, _ in algos]

    # Weighted accuracy and average delay over tasks
    acc_hist        = {n: np.full(T, np.nan) for n in algo_names}
    delay_ratio_hist= {n: np.full(T, np.nan) for n in algo_names}

    # Per-task accuracy: shape (A, T, K)
    per_task_acc    = {n: np.full((T, K), np.nan) for n in algo_names}

    # Per-task delay ratio: shape (T, K)
    per_task_delay  = {n: np.full((T, K), np.nan) for n in algo_names}

    # Per-task eta: shape (A, T, K, L)
    eta_hist        = {n: np.full((T, K, num_links), np.nan) for n in algo_names}

    c_t_hist = np.full((T, num_links), np.nan)

    for t in range(T):
        c_t = rng.uniform(low=s.c_min, high=s.c_max, size=num_links).astype(float)
        c_t_hist[t] = c_t
        print(f"  t={t:3d}  c_t={np.round(c_t, 1)}", flush=True)

        for name, opt in algos:
            # Choose c_hat vs c_t
            if name in _no_csi_names():
                allocations = opt.optimize(t=t, c_hat=lcb.estimate(t))
            elif name == "No-CSI Baseline: certainty equivalence" and hist_mean:
                allocations = opt.optimize(t=t, c_hat=hist_mean.estimate(t))
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                # Infeasible slot: fall back to maximum compression per task.
                # Same rationale as single-task: ensures all algorithms are
                # evaluated over the same T slots for fair comparison.
                allocations = {}
                for task in tasks:
                    eta_fb    = np.array(
                        [task.eta_min[i] for i in range(task.b_k, task.e_k)],
                        dtype=float
                    )
                    allocations[task.task_id] = {
                        "eta":    eta_fb,
                        "s_comp": np.ones(task.L_k, dtype=float),
                        "s_comm": np.ones(task.L_k - 1, dtype=float),
                    }
            wacc = 0.0
            wsum = 0.0
            for k_idx, task in enumerate(tasks):
                eta    = allocations[task.task_id]["eta"]
                a_k    = float(task.A_k_true(eta))
                wacc  += float(task.w_k) * a_k
                wsum  += float(task.w_k)
                eta_hist[name][t, task.task_id - 1]    = np.array(eta, dtype=float)
                per_task_acc[name][t, k_idx]           = a_k
            acc_hist[name][t] = wacc / wsum if wsum > 0 else 0.0

            # Per-task and average delay ratio
            residuals = opt.compute_residual_delay(t, allocations, c_t)
            ratios    = []
            for k_idx, task in enumerate(tasks):
                residual = float(residuals[task.task_id])
                R_k      = float(task.get_R_k(t))
                inv_R    = 1.0 / R_k
                delay    = residual + inv_R
                ratio    = delay / inv_R
                ratios.append(ratio)
                per_task_delay[name][t, k_idx] = ratio
            delay_ratio_hist[name][t] = float(np.mean(ratios))

            # Dual update
            if name in _no_csi_names():
                actual_delays = {}
                for task in tasks:
                    res  = float(residuals[task.task_id])
                    R_k  = float(task.get_R_k(t))
                    actual_delays[task.task_id] = res + 1.0 / R_k
                opt.update_dual(t=t, actual_delays=actual_delays)

        lcb.update(c_t)
        if hist_mean:
            hist_mean.update(c_t)

    # Cumulative averages
    acc_avg         = {n: np.array([np.nanmean(acc_hist[n][:t+1])         for t in range(T)]) for n in algo_names}
    delay_ratio_avg = {n: np.array([np.nanmean(delay_ratio_hist[n][:t+1]) for t in range(T)]) for n in algo_names}
    per_task_avg    = {n: np.array([np.nanmean(per_task_delay[n][:t+1,:], axis=0) for t in range(T)]) for n in algo_names}
    eta_avg         = {n: np.array([np.nanmean(eta_hist[n][:t+1,:,:], axis=0) for t in range(T)]) for n in algo_names}

    per_task_acc_avg= {n: np.array([np.nanmean(per_task_acc[n][:t+1,:], axis=0) for t in range(T)]) for n in algo_names}

    return dict(
        times=times, c_t_hist=c_t_hist,
        acc_hist=acc_hist, delay_ratio_hist=delay_ratio_hist,
        per_task_delay=per_task_delay, per_task_acc=per_task_acc,
        acc_avg=acc_avg, delay_ratio_avg=delay_ratio_avg,
        per_task_avg=per_task_avg, per_task_acc_avg=per_task_acc_avg,
        eta_avg=eta_avg, eta_hist=eta_hist,
        algo_names=algo_names,
    )


# ============================================================================
# Plotting
# ============================================================================

def _setting_text(fig, s: MultiTaskExperimentSetting, fontsize: float = 6.5):
    fig.text(
        0.01, 0.01, s.detail_label(),
        fontsize=fontsize,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='grey', alpha=0.85),
        family='monospace',
    )


def _markers():
    return ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]


def plot_results(
    tasks:    List[InferenceTask],
    hist:     Dict,
    s:        MultiTaskExperimentSetting,
    out_dir:  str,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    algo_names      = hist["algo_names"]
    times           = hist["times"]
    acc_avg         = hist["acc_avg"]
    delay_ratio_avg = hist["delay_ratio_avg"]
    per_task_avg    = hist["per_task_avg"]
    eta_avg         = hist["eta_avg"]
    T               = s.T
    K               = len(tasks)
    num_links       = tasks[0].L_k - 1

    markers   = _markers()
    marker_map= {n: markers[i % len(markers)] for i, n in enumerate(algo_names)}
    markevery = max(1, T // 10)
    paths     = {}

    BOTTOM = 0.25
    RIGHT  = 0.72

    # ── Plot 1: Weighted accuracy ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in algo_names:
        ax.plot(times, acc_avg[name], label=name, linewidth=1.5,
                marker=marker_map[name], markevery=markevery, markersize=5)
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg weighted accuracy  (1/W) Σ_k w_k A_k(η_k)")
    ax.set_title(f"Weighted accuracy over time\n{s.short_label()}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "accuracy_over_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["accuracy"] = p

    # ── Plot 2: Average delay ratio ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in algo_names:
        ax.plot(times, delay_ratio_avg[name], label=name, linewidth=1.5,
                marker=marker_map[name], markevery=markevery, markersize=5)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg  (1/K) Σ_k D_k / (1/R_k)")
    ax.set_title(f"Average delay ratio over time\n{s.short_label()}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "delay_over_invR_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["delay"] = p

    # ── Plot 3: Per-task delay (main algorithms only) ─────────────────────────
    main_algos = [n for n in algo_names
                  if "csi-aware" in n.lower() or "no-csi" in n.lower()]
    task_colors = plt.cm.tab10(np.linspace(0, 0.9, K))
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in main_algos:
        ls = "-" if "csi-aware" in name.lower() else "--"
        for k_idx, task in enumerate(tasks):
            ax.plot(times, per_task_avg[name][:, k_idx],
                    label=f"{name} task {task.task_id} (w={task.w_k:.0f})",
                    linewidth=1.3, linestyle=ls,
                    color=task_colors[k_idx],
                    marker=marker_map[name], markevery=markevery, markersize=4)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg  D_k / (1/R_k)  per task")
    ax.set_title(f"Per-task delay ratio (main algorithms)\n{s.short_label()}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "delay_per_task_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["per_task_delay"] = p

    # ── Plot 4: Eta (main algorithms only) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in main_algos:
        ls = "-" if "csi-aware" in name.lower() else "--"
        for k_idx, task in enumerate(tasks):
            for li in range(num_links):
                ax.plot(times, eta_avg[name][:, k_idx, li],
                        label=f"{name} task {task.task_id} link {li}",
                        linewidth=1.2, linestyle=ls,
                        color=task_colors[k_idx],
                        marker=marker_map[name], markevery=markevery, markersize=4)
    ax.set_xlabel("Time slot t")
    ax.set_ylabel("Cumulative avg η")
    ax.set_title(f"Compression ratio η per task per link (main algs)\n{s.short_label()}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=7, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=BOTTOM, right=RIGHT)
    _setting_text(fig, s)
    p = os.path.join(out_dir, "eta_over_time.png")
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    paths["eta"] = p

    # ── Plot 5: Trade-off scatter ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in algo_names:
        x = float(acc_avg[name][-1])
        y = float(delay_ratio_avg[name][-1])
        ax.scatter(x, y, marker=marker_map[name], s=100, label=name, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, clip_on=True)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Feasibility boundary")
    ax.set_xlabel("Cumulative avg weighted accuracy at t = T-1")
    ax.set_ylabel("Cumulative avg  (1/K) Σ_k D_k/(1/R_k)  at t = T-1")
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
    tasks:    List[InferenceTask],
    hist:     Dict,
    s:        MultiTaskExperimentSetting,
    out_dir:  str,
):
    os.makedirs(out_dir, exist_ok=True)

    algo_names = hist["algo_names"]
    K          = len(tasks)

    setting_dict = asdict(s)
    setting_dict["a_base"]       = {str(k): v for k, v in s.a_base.items()}
    setting_dict["tau"]          = {str(k): v for k, v in s.tau.items()}
    setting_dict["task_a"]       = {str(tid): {str(i): v for i, v in a.items()}
                                    for tid, a in s.task_a.items()}
    setting_dict["task_weights"] = {str(k): v for k, v in s.task_weights.items()}
    with open(os.path.join(out_dir, "experiment_setting_multi.json"), "w") as f:
        json.dump(setting_dict, f, indent=2, sort_keys=True)

    np.savez_compressed(
        os.path.join(out_dir, "simulation_results_multi.npz"),
        times               = hist["times"],
        c_t_hist            = hist["c_t_hist"],
        algo_names          = np.array(algo_names, dtype=str),
        task_ids            = np.array([t.task_id for t in tasks], dtype=int),
        acc_avg             = np.stack([hist["acc_avg"][n]          for n in algo_names]),
        delay_ratio_avg     = np.stack([hist["delay_ratio_avg"][n]  for n in algo_names]),
        per_task_avg        = np.stack([hist["per_task_avg"][n]     for n in algo_names]),
        per_task_acc_avg    = np.stack([hist["per_task_acc_avg"][n] for n in algo_names]),
        eta_avg             = np.stack([hist["eta_avg"][n]          for n in algo_names]),
        acc_hist            = np.stack([hist["acc_hist"][n]         for n in algo_names]),
        delay_ratio_hist    = np.stack([hist["delay_ratio_hist"][n] for n in algo_names]),
        K                   = np.array([K], dtype=int),
    )


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Run all multi-task algorithms + baselines for one experimental setting."
    )

    p.add_argument("--backend", type=str, default="mlp",
                   choices=["mlp", "resnet", "llm"],
                   help="Backend for all tasks in single-backend mode. "
                        "Ignored when --task_backends is set.")
    p.add_argument("--K", type=int, default=2,
                   help="Number of concurrent tasks.  "
                        "Inferred from --task_backends length if provided.")

    # Cross-backend arguments
    p.add_argument("--task_backends", nargs="+", default=None,
                   help="Per-task backend list, e.g. --task_backends resnet llm. "
                        "Activates cross-backend mode. Length must equal K.")
    p.add_argument("--task_compressors", nargs="+", default=None,
                   help="Per-task compressor/strategy list, "
                        "e.g. --task_compressors topk topk_per_token.")
    p.add_argument("--task_llm_models", nargs="+", default=None,
                   help="Per-task LLM model names. Use '' for non-LLM tasks. "
                        "e.g. --task_llm_models '' 'meta-llama/Llama-3.1-8B-Instruct'")
    p.add_argument("--task_eta_mins", nargs="+", type=float, default=None,
                   help="Per-task eta_min overrides. "
                        "e.g. --task_eta_mins 0.125 0.25")

    # Compressor — ResNet / MLP
    p.add_argument("--compressor",         type=str, default="topk",
                   choices=["topk", "quantization", "llmint8"])
    p.add_argument("--compressor_backend", type=str, default="cpu",
                   choices=["cpu", "gpu"])

    # LLM-specific
    p.add_argument("--llm_activation_strategy", type=str, default=None,
                   choices=["topk_per_token","magnitude","random","quantization","llmint8",None])
    p.add_argument("--llm_model_name",    type=str,  default=None)
    p.add_argument("--llm_grad_N",        type=int,  default=None)
    p.add_argument("--llm_fast_samples",  type=int,  default=None)

    # MLP-specific
    p.add_argument("--mlp_sigma", type=float, default=None)
    p.add_argument("--mlp_N",     type=int,   default=None)

    # Channel
    p.add_argument("--c_min", type=float, default=None)
    p.add_argument("--c_max", type=float, default=None)

    # Task parameters
    p.add_argument(
        "--R", type=float, default=10.0,
        help="Default deadline R for all tasks when --R_k is not set. "
             "Ignored if --R_k provides per-task values.")
    p.add_argument(
        "--R_k", nargs="+", type=float, default=None,
        help="Per-task deadlines R_k (length 1 or K). "
             "One value repeats for all K tasks; otherwise supply exactly K values. "
             "Example: --R_k 10 20  (task 1 R=10, task 2 R=20). "
             "Overrides --R.")
    p.add_argument("--eta_min", type=float, default=None)
    p.add_argument("--heterogeneous_a", type=lambda x: x.lower() != "false",
                   default=True,
                   help="Vary a_{i,k} across tasks (default: true). "
                        "Pass 'false' to give all tasks identical a.")

    # No-CSI multi-task parameters
    p.add_argument("--no_csi_mu",      type=float, default=3.0)
    p.add_argument("--no_csi_epsilon", type=float, default=0.1)
    p.add_argument("--no_csi_J",       type=int,   default=10,
                   help="BCD inner iterations for No-CSI multi-task (default: 10).")

    p.add_argument(
        "--task_weights", nargs="*", type=float, default=None,
        help="Per-task objective weights w_k (length K). "
             "Default: w_k = float(task_id), i.e. task 1 weight 1.0, task 2 weight 2.0. "
             "Example: --task_weights 1.0 3.0",
    )

    # Run
    p.add_argument("--T",    type=int, default=150)
    p.add_argument("--seed", type=int, default=12)

    # Output
    p.add_argument("--out_dir", type=str, default=None)

    # Baselines
    p.add_argument(
        "--baselines", type=str,
        default="max_compression,no_compression,equal_share,proportional,"
                "strict_priority,decoupled_equal_split,queue_proportional,"
                "certainty_equivalence",
        help="Comma-separated baseline keys, or 'none'. "
             "Options: max_compression, no_compression, equal_share, proportional, "
             "strict_priority, decoupled_equal_split, queue_proportional, "
             "certainty_equivalence.",
    )

    args = p.parse_args()

    # Backend env var
    backend_env = {"mlp": "mnist", "resnet": "resnet", "llm": "llm"}[args.backend]
    os.environ.setdefault("INFERENCE_OPTIMIZER_BACKEND", backend_env)

    backend = args.backend
    c_min   = args.c_min   or (10.0  if backend == "mlp" else 500_000.0)
    c_max   = args.c_max   or (150.0 if backend == "mlp" else 5_000_000.0)
    eta_min = (args.eta_min if args.eta_min is not None else
               (0.05  if backend == "mlp" else
                0.125 if backend == "resnet" else 0.25))

    llm_strategy = (args.llm_activation_strategy
                    or os.environ.get("LLM_ACTIVATION_STRATEGY", "topk_per_token"))
    llm_model    = (args.llm_model_name
                    or os.environ.get("LLM_MODEL_NAME", "unsloth/gemma-2b-it"))

    bl_arg    = args.baselines.strip().lower()
    baselines = [] if bl_arg == "none" else [b.strip() for b in bl_arg.split(",") if b.strip()]

    # ── Resolve per-task R_k list ─────────────────────────────────────────────
    # Mirrors compare_to_baselines_multi.py: one value repeats for all K, or
    # exactly K values must be supplied.
    K_for_rk = len(args.task_backends) if args.task_backends else args.K
    if args.R_k is not None:
        if len(args.R_k) == 1:
            R_k_per_task = [float(args.R_k[0])] * K_for_rk
        elif len(args.R_k) == K_for_rk:
            R_k_per_task = [float(v) for v in args.R_k]
        else:
            raise ValueError(
                f"--R_k: provide 1 value (repeated for all {K_for_rk} tasks) or "
                f"exactly {K_for_rk} values; got {len(args.R_k)}: {args.R_k}"
            )
    else:
        R_k_per_task = [float(args.R)] * K_for_rk

    # ── Cross-backend validation ──────────────────────────────────────────────
    cross_backend = args.task_backends is not None
    if cross_backend:
        K = len(args.task_backends)
        if args.K != 2 and args.K != K:
            print(f"[run_experiment_multi] --K overridden to {K} by --task_backends.",
                  flush=True)
        args.K = K
        # Re-expand R_k_per_task now that we know K from task_backends
        if len(R_k_per_task) == 1:
            R_k_per_task = R_k_per_task * K
        elif len(R_k_per_task) != K:
            raise ValueError(
                f"--R_k: got {len(R_k_per_task)} values but --task_backends implies K={K}"
            )
        # Fill defaults for per-task lists
        task_compressors = (args.task_compressors or
                            [(args.compressor if tb != "llm" else "topk_per_token")
                             for tb in args.task_backends])
        task_llm_models  = (args.task_llm_models or
                            [("" if tb != "llm" else llm_model)
                             for tb in args.task_backends])
        task_eta_mins    = (args.task_eta_mins or
                            [(0.125 if tb == "resnet" else
                              0.25  if tb == "llm"    else
                              0.05)
                             for tb in args.task_backends])
        if len(task_compressors) != K:
            raise ValueError(f"--task_compressors must have {K} entries, got {len(task_compressors)}")
        if len(task_llm_models)  != K:
            raise ValueError(f"--task_llm_models must have {K} entries, got {len(task_llm_models)}")
    else:
        task_compressors = []
        task_llm_models  = []
        task_eta_mins    = []

    s = MultiTaskExperimentSetting(
        backend            = backend,
        compressor         = args.compressor,
        compressor_backend = args.compressor_backend,
        c_min              = c_min,
        c_max              = c_max,
        c_dist             = "uniform",
        R_k_per_task       = R_k_per_task,
        eta_min            = eta_min,
        T                  = args.T,
        seed               = args.seed,
        K                  = args.K,
        no_csi_mu          = args.no_csi_mu,
        no_csi_epsilon     = args.no_csi_epsilon,
        no_csi_J           = args.no_csi_J,
        heterogeneous_a    = args.heterogeneous_a,
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
        # Cross-backend
        task_backends    = args.task_backends or [],
        task_compressors = task_compressors,
        task_llm_models  = task_llm_models,
        task_eta_mins    = task_eta_mins,
        task_weight_overrides = args.task_weights or [],
    )

    # Auto-force gpu for LLM + llmint8
    if backend == "llm" and llm_strategy == "llmint8" and args.compressor_backend == "cpu":
        print("[run_experiment_multi] llmint8 requires gpu backend — forcing.", flush=True)
        s.compressor_backend = "gpu"

    if cross_backend:
        backends_str = "_".join(args.task_backends)
        comps_str    = "_".join(task_compressors)
        out_dir = args.out_dir or os.path.join(
            "outputs", "experiments_multi",
            f"cross_{backends_str}_{comps_str}_K{args.K}_seed{args.seed}"
        )
    else:
        out_dir = args.out_dir or os.path.join(
            "outputs", "experiments_multi",
            f"{backend}_K{args.K}_{args.compressor if backend != 'llm' else llm_strategy}"
            f"_{s.compressor_backend}_seed{args.seed}"
        )

    # ── Print configuration before loading ───────────────────────────────────
    s.full_printout(out_dir, baselines)

    # ── Load backends and build tasks ─────────────────────────────────────────
    apis  = load_all_backends(s)
    tasks = build_tasks(s, apis)

    # ── Print again with topology filled in ──────────────────────────────────
    print("\n[run_experiment_multi] Topology derived — updated configuration:", flush=True)
    s.full_printout(out_dir, baselines)

    algos, lcb, hist_mean = build_algorithms(tasks, s, baselines)

    print(f"\n[run_experiment_multi] Algorithms ({len(algos)}):", flush=True)
    for name, _ in algos:
        print(f"  • {name}", flush=True)
    print(flush=True)

    hist  = run_simulation(tasks, algos, s, lcb, hist_mean)

    save_results(tasks, hist, s, out_dir)
    paths = plot_results(tasks, hist, s, out_dir)

    print("\n" + "=" * 65, flush=True)
    print("Done.", flush=True)
    for label, path in paths.items():
        print(f"  {label:<18} -> {path}", flush=True)
    print(f"  {'setting':<18} -> "
          f"{os.path.join(out_dir, 'experiment_setting_multi.json')}", flush=True)
    print(f"  {'raw data':<18} -> "
          f"{os.path.join(out_dir, 'simulation_results_multi.npz')}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
