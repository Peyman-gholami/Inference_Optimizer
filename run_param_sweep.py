"""
Parameter sweep experiments over sigma and N.

Experiment 1 (sigma sweep):
    N=50 fixed, sigma in [0.001, 0.01, 0.05, 0.1, 0.5]
    -> 4 summary diagrams across all sigma values

Experiment 2 (N sweep):
    sigma=0.05 fixed, N in [10, 25, 50, 75, 100]
    -> 4 summary diagrams across all N values

For each experiment the 4 summary diagrams are:
    1. Cumulative average accuracy over time  (one curve per param value)
    2. Cumulative average delay/(1/R_k) over time  (one curve per param value)
    3. Bar chart: final accuracy at t=T-1 vs param value
    4. Bar chart: final delay/(1/R_k) at t=T-1 vs param value

Each individual run also saves its own 4 per-run plots (same as compare_to_baselines.py).

Usage:
    python run_param_sweep.py
    python run_param_sweep.py --T 100 --seed 12
    python run_param_sweep.py --baseline_algorithms none
    python run_param_sweep.py --out_dir outputs/sweep

Run Experiments
    # Both experiments (default)
    python run_param_sweep.py --baseline_algorithms none

    # Only sigma sweep
    python run_param_sweep.py --baseline_algorithms none --experiments sigma

    # Only N sweep
    python run_param_sweep.py --baseline_algorithms none --experiments N

    # With baselines included
    python run_param_sweep.py --experiments sigma,N
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup (mirrors compare_to_baselines.py)
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

# ---------------------------------------------------------------------------
# Patched setup_model_and_callables that accepts sigma and N
# ---------------------------------------------------------------------------

def setup_model_and_callables_patched(sigma: float = 0.05, N: int = 50):
    """
    Backend-aware version of main.setup_model_and_callables() that also
    accepts sigma and N so they can be varied across sweep experiments.

    Respects INFERENCE_OPTIMIZER_BACKEND:
      (unset / 'mnist') -> MLP on MNIST  (sigma and N are used)
      'resnet'          -> ResNet56 on CIFAR-10  (sigma and N override env defaults)
      'llm'             -> LLM/MMLU  (sigma and N override env defaults)
    """
    from src.core.llm_plug_in import is_llm_backend, is_resnet_backend

    # ------------------------------------------------------------------ #
    # LLM backend                                                          #
    # ------------------------------------------------------------------ #
    if is_llm_backend():
        from src.core.llm_task_callables import build_llm_backend_from_env
        eta_min = min(
            float(os.environ.get("ETA_MIN0", "0.25")),
            float(os.environ.get("ETA_MIN1", "0.25")),
        )
        api = build_llm_backend_from_env(eta_min=eta_min)
        # Override Stein oracle params with sweep values
        api.config = api.config.__class__(
            **{**api.config.__dict__, 'grad_sigma': sigma, 'grad_N': N}
        )
        print(f"  [LLM backend] sigma={sigma}, N={N}")
        return api, None

    # ------------------------------------------------------------------ #
    # ResNet backend                                                       #
    # ------------------------------------------------------------------ #
    if is_resnet_backend():
        from src.core.resnet_task_callables import build_resnet_backend_from_env
        # build_resnet_backend_from_env reads RESNET_GRAD_SIGMA / RESNET_GRAD_N
        # from env; override them here with the sweep values.
        os.environ["RESNET_GRAD_SIGMA"] = str(sigma)
        os.environ["RESNET_GRAD_N"]     = str(N)
        api = build_resnet_backend_from_env()
        print(f"  [ResNet backend] sigma={sigma}, N={N}")
        return api, None

    # ------------------------------------------------------------------ #
    # Default: MLP on MNIST                                                #
    # ------------------------------------------------------------------ #
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from src.core.task_callables import InferenceTaskCallables

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=256,
        shuffle=False,
    )

    from stein_oracles import MLP, get_flat_params
    model = MLP().to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"  Training MLP (3 epochs) for sigma={sigma}, N={N} ...")
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt_model.step()

    flat_theta = get_flat_params(model)

    eta_init = torch.ones(2, device=device)
    from accuracy_fn import CompressedModel

    compressor_type = os.environ.get("COMPRESSOR_TYPE", "topk").strip().lower()
    cm = CompressedModel(model, eta_init, compressor=compressor_type)

    # Wrap test_loader so every batch is pre-moved to device.
    class DeviceLoader:
        def __init__(self, loader, dev):
            self._loader = loader
            self._dev = dev
        def __iter__(self):
            for x, y in self._loader:
                yield x.to(self._dev), y.to(self._dev)
        def __len__(self):
            return len(self._loader)
        def __getattr__(self, name):
            return getattr(self._loader, name)

    device_test_loader = DeviceLoader(test_loader, device)

    api = InferenceTaskCallables(
        cm,
        device_test_loader,
        sigma=sigma,
        N=N,
        n_samples=512,
        flat_theta=None,
        device=device,
        compressor=compressor_type,
    )
    return api, flat_theta


# ---------------------------------------------------------------------------
# Import the rest from compare_to_baselines (simulation logic + helpers)
# ---------------------------------------------------------------------------
from compare_to_baselines import (   # noqa: E402
    _compute_accuracy,
    _compute_delay_and_ratio,
)
from main import make_task  # noqa: E402

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


# ---------------------------------------------------------------------------
# Single-run simulation (same logic as run_compare, but returns raw histories)
# ---------------------------------------------------------------------------

def run_single(
    T: int,
    seed: int,
    c_min: float,
    c_max: float,
    sigma: float,
    N: int,
    out_dir: str,
    baseline_algorithms: Optional[List[str]],
    no_csi_mu: float,
    no_csi_epsilon: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Tuple[str, BaseOptimizer]]]:
    """
    Runs the full simulation for one (sigma, N) combination.

    Returns
    -------
    acc_avg : dict[algo_name -> cumulative-avg accuracy array of length T]
    delay_ratio_avg : dict[algo_name -> cumulative-avg delay-ratio array of length T]
    algos : list of (name, optimizer) so callers know the algorithm list
    """
    rng = np.random.default_rng(seed)

    api, _ = setup_model_and_callables_patched(sigma=sigma, N=N)
    task = make_task(task_id=1, api=api)

    # Derive topology from the task itself so this works for any backend.
    # MLP/LLM: L_k=3 (3 nodes, 2 links).  ResNet default: L_k=4 (4 nodes, 3 links).
    M = task.L_k
    num_links = task.L_k - 1

    # ---- Override c_min/c_max for backends with real payload sizes -----------
    # The default c_min=10, c_max=150 were designed for the MLP toy values
    # (a[i] ~ 10-20).  For ResNet, a[i] is in bytes (~tens of thousands), so
    # those defaults make every CSI-aware slot infeasible.
    # We derive feasibility-aware bounds from the task's own a, eta_min, R_k.
    #   c_min_link = a[i] * eta_min[i] * R_k   (minimum for feasibility)
    #   c_min      = max over links             (tightest constraint)
    #   c_max      = c_min * 10                 (one decade of headroom)
    R_k = task.get_R_k(0)
    c_min_per_link = [
        task.a[task.b_k + i] * task.eta_min[task.b_k + i] * R_k
        for i in range(num_links)
    ]
    c_min_feasible = max(c_min_per_link)
    if c_min_feasible > c_min:
        # Backend has real payload sizes — override the passed-in values
        c_min = c_min_feasible * 1.1    # just above feasibility floor
        c_max = c_min_feasible * 10.0   # one decade of range
        print(f"  [run_single] c_min/c_max overridden for backend: "
              f"c_min={c_min:.1f}, c_max={c_max:.1f} "
              f"(derived from a={[task.a[task.b_k+i] for i in range(num_links)]}, "
              f"eta_min={[task.eta_min[task.b_k+i] for i in range(num_links)]}, "
              f"R_k={R_k})", flush=True)

    # ---- Build algorithm list (same as run_compare) -------------------------
    algos: List[Tuple[str, BaseOptimizer]] = [
        ("CSI-aware (optimal)", CSIAwareSingleTaskOptimizer(M=M, tasks=[task]))
    ]

    baseline_catalog = {
        "max_compression": (
            "Baseline: max compression",
            MaxCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "no_compression": (
            "Baseline: no compression",
            NoCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "uniform_compression": (
            "Baseline: uniform compression",
            UniformCompressionSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "myopic": (
            "Baseline: myopic",
            MyopicSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "conservative": (
            "Baseline: conservative",
            ConservativeSingleTaskBaseline(M=M, tasks=[task]),
        ),
        "moving_average": (
            "Baseline: moving average",
            MovingAverageSingleTaskBaseline(M=M, tasks=[task]),
        ),
    }
    selected_baselines = (
        list(baseline_catalog.keys()) if baseline_algorithms is None else baseline_algorithms
    )
    for baseline_name in selected_baselines:
        algo_entry = baseline_catalog.get(baseline_name)
        if algo_entry is None:
            valid = ", ".join(sorted(baseline_catalog.keys()))
            raise ValueError(
                f"Unknown baseline algorithm '{baseline_name}'. Valid options: {valid}."
            )
        algos.append(algo_entry)

    lcb = MeanMinusZStdLCB(
        num_links=num_links,
        warmup_value=c_min,
        z=1.2815515655446004,
        warmup=1,
    )
    no_csi = NoCSISingleTaskOptimizer(M=M, tasks=[task], mu=no_csi_mu, epsilon=no_csi_epsilon)
    algos.append(("No-CSI (Alg1) + LCB(90%)", no_csi))

    selected_set = set(selected_baselines)
    myopic_est = (
        LastObservationEstimator(num_links=num_links, warmup_value=c_min)
        if "myopic" in selected_set else None
    )
    cons_est = (
        RunningMinEstimator(num_links=num_links, warmup_value=c_min)
        if "conservative" in selected_set else None
    )
    ma_est = (
        MovingAverageEstimator(num_links=num_links, window=5, warmup_value=c_min)
        if "moving_average" in selected_set else None
    )

    # ---- Storage ------------------------------------------------------------
    times = np.arange(T)
    acc_hist: Dict[str, np.ndarray] = {
        name: np.full(T, np.nan, dtype=float) for name, _ in algos
    }
    delay_ratio_hist: Dict[str, np.ndarray] = {
        name: np.full(T, np.nan, dtype=float) for name, _ in algos
    }
    eta_hist: Dict[str, np.ndarray] = {
        name: np.full((T, task.L_k - 1), np.nan, dtype=float) for name, _ in algos
    }

    # ---- Simulation loop ----------------------------------------------------
    for t in range(T):
        print(f"  [sigma={sigma}, N={N}] t={t}", flush=True)
        c_t = rng.uniform(low=c_min, high=c_max, size=num_links).astype(float)

        for name, opt in algos:
            if name == "No-CSI (Alg1) + LCB(90%)":
                c_hat = lcb.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "Baseline: myopic":
                c_hat = myopic_est.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "Baseline: conservative":
                c_hat = cons_est.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            elif name == "Baseline: moving average":
                c_hat = ma_est.estimate(t)
                allocations = opt.optimize(t=t, c_hat=c_hat)
            else:
                allocations = opt.optimize(t=t, c_t=c_t)

            if allocations is None:
                continue

            acc_hist[name][t] = _compute_accuracy(task, allocations)
            delay, delay_over_invR = _compute_delay_and_ratio(opt, task, t, allocations, c_t)
            delay_ratio_hist[name][t] = delay_over_invR
            eta_hist[name][t, :] = np.array(allocations[task.task_id]["eta"], dtype=float)

            if name == "No-CSI (Alg1) + LCB(90%)":
                opt.update_dual(t=t, actual_delays={task.task_id: delay})

        lcb.update(c_t)
        if myopic_est is not None:
            myopic_est.update(c_t)
        if cons_est is not None:
            cons_est.update(c_t)
        if ma_est is not None:
            ma_est.update(c_t)

    # ---- Cumulative averages ------------------------------------------------
    acc_avg: Dict[str, np.ndarray] = {}
    delay_ratio_avg: Dict[str, np.ndarray] = {}
    eta_avg: Dict[str, np.ndarray] = {}
    for name, _ in algos:
        acc_avg[name] = np.array(
            [np.nanmean(acc_hist[name][: t + 1]) for t in range(T)], dtype=float
        )
        delay_ratio_avg[name] = np.array(
            [np.nanmean(delay_ratio_hist[name][: t + 1]) for t in range(T)], dtype=float
        )
        eta_avg[name] = np.array(
            [np.nanmean(eta_hist[name][: t + 1, :], axis=0) for t in range(T)], dtype=float
        )

    # ---- Per-run plots (mirrors the 4 plots in compare_to_baselines.py) -----
    os.makedirs(out_dir, exist_ok=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    marker_map = {name: markers[i % len(markers)] for i, (name, _) in enumerate(algos)}
    markevery = max(1, T // 10)

    # Per-run plot 1: accuracy over time
    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(times, acc_avg[name], label=name, linewidth=1.5,
                 marker=marker_map[name], markevery=markevery, markersize=5)
    plt.xlabel("Time slot t")
    plt.ylabel("Average Accuracy (1/t) * sum A_k(eta(s))")
    plt.title(f"Cumulative avg accuracy | sigma={sigma}, N={N}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_over_time.png"), dpi=180)
    plt.close()

    # Per-run plot 2: eta over time
    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        for link_idx in range(task.L_k - 1):
            plt.plot(times, eta_avg[name][:, link_idx],
                     label=f"{name} (eta[{link_idx}])", linewidth=1.2,
                     marker=marker_map[name], markevery=markevery, markersize=4)
    plt.xlabel("Time slot t")
    plt.ylabel("Cumulative avg eta")
    plt.title(f"Cumulative avg compression | sigma={sigma}, N={N}")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eta_over_time.png"), dpi=180)
    plt.close()

    # Per-run plot 3: delay/invR over time
    plt.figure(figsize=(12, 6))
    for name, _ in algos:
        plt.plot(times, delay_ratio_avg[name], label=name, linewidth=1.5,
                 marker=marker_map[name], markevery=markevery, markersize=5)
    plt.xlabel("Time slot t")
    plt.ylabel("Average D_k / (1/R_k)")
    plt.title(f"Cumulative avg delay/(1/R_k) | sigma={sigma}, N={N}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delay_over_invR_over_time.png"), dpi=180)
    plt.close()

    # Per-run plot 4: accuracy vs delay trade-off at t=T-1
    plt.figure(figsize=(8, 6))
    for name, _ in algos:
        plt.scatter(float(acc_avg[name][-1]), float(delay_ratio_avg[name][-1]),
                    marker=marker_map[name], s=80, label=name)
    plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.8)
    plt.xlabel("Cumulative avg accuracy at t=T-1")
    plt.ylabel("Cumulative avg D_k/(1/R_k) at t=T-1")
    plt.title(f"Accuracy vs feasibility | sigma={sigma}, N={N}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeoff_acc_vs_delay_over_invR_Tminus1.png"), dpi=180)
    plt.close()

    return acc_avg, delay_ratio_avg, algos


# ---------------------------------------------------------------------------
# Summary plots for one experiment (4 diagrams across all param values)
# ---------------------------------------------------------------------------

def _linestyles():
    """Cycle through distinct linestyles for param-sweep curves."""
    return ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def plot_summary(
    param_values: List,
    param_name: str,          # "sigma" or "N"
    all_acc_avg: List[Dict[str, np.ndarray]],
    all_delay_avg: List[Dict[str, np.ndarray]],
    all_algos: List[List[Tuple[str, BaseOptimizer]]],
    T: int,
    out_dir: str,
    focus_algo: str = "No-CSI (Alg1) + LCB(90%)",
    ref_algo: str = "CSI-aware (optimal)",
) -> None:
    """
    Produces the 4 summary diagrams for a sweep experiment.

    Diagram 1: Accuracy over time — side-by-side subplots
        Left : No-CSI,     one curve per param value
        Right: CSI-aware,  one curve per param value (shared y-axis)
    Diagram 2: Delay/(1/R_k) over time — same side-by-side layout
        Left : No-CSI
        Right: CSI-aware  (both panels share the feasibility=1 reference line)
    Diagram 3: Bar chart — final accuracy at t=T-1 per param value (all algos)
    Diagram 4: Bar chart — final delay/(1/R_k) at t=T-1 per param value (all algos)
    """
    os.makedirs(out_dir, exist_ok=True)
    times = np.arange(T)
    ls_cycle = _linestyles()
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    markevery = max(1, T // 10)
    param_markers = ["o", "s", "^", "D", "v"]

    def _draw_side_by_side(
        metric_all: List[Dict[str, np.ndarray]],
        ylabel: str,
        suptitle: str,
        filename: str,
        add_hline: bool = False,
    ) -> None:
        """
        Side-by-side figure: left subplot = focus_algo, right = ref_algo.
        Both share the same y-axis scale. One curve per param value in each panel.
        """
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2,
            figsize=(18, 6),
            sharey=True,
        )

        for i, (pval, metric_avg) in enumerate(zip(param_values, metric_all)):
            color  = colors[i % len(colors)]
            marker = param_markers[i % len(param_markers)]
            ls     = ls_cycle[i % len(ls_cycle)]
            label  = f"{param_name}={pval}"
            kw = dict(linewidth=2.0, linestyle=ls, color=color,
                      marker=marker, markevery=markevery, markersize=5)

            if focus_algo in metric_avg:
                ax_left.plot(times, metric_avg[focus_algo], label=label, **kw)
            if ref_algo in metric_avg:
                ax_right.plot(times, metric_avg[ref_algo], label=label, **kw)

        for ax, algo_label in ((ax_left, focus_algo), (ax_right, ref_algo)):
            if add_hline:
                ax.axhline(1.0, color="black", linewidth=1.2,
                           linestyle=":", alpha=0.8, label="feasibility=1")
            ax.set_xlabel("Time slot t")
            ax.set_title(algo_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=1, fontsize=8)
            # Guard: if this panel has no valid data (all-NaN), set explicit
            # axis limits so sharey does not inherit a degenerate scale.
            lines = ax.get_lines()
            has_data = any(
                np.any(np.isfinite(line.get_ydata())) for line in lines
                if len(line.get_ydata()) > 0
            )
            if not has_data:
                ax.set_xlim(0, T)
                ax.set_ylim(0, 1)
                ax.text(0.5, 0.5, "No feasible slots",
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='grey')

        ax_left.set_ylabel(ylabel)
        fig.suptitle(f"{suptitle}  (varying {param_name})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=180)
        plt.close()

    # ---- Diagram 1: accuracy over time (side-by-side) -----------------------
    _draw_side_by_side(
        metric_all=all_acc_avg,
        ylabel="Cumulative avg accuracy (1/t) * sum A_k(eta(s))",
        suptitle="Accuracy over time",
        filename=f"summary_accuracy_over_time_vs_{param_name}.png",
        add_hline=False,
    )
    print(f"  Saved summary diagram 1 (accuracy over time vs {param_name})", flush=True)

    # ---- Diagram 2: delay/invR over time (side-by-side) ---------------------
    _draw_side_by_side(
        metric_all=all_delay_avg,
        ylabel="Cumulative avg D_k / (1/R_k)",
        suptitle="Delay / (1/R_k) over time",
        filename=f"summary_delay_over_time_vs_{param_name}.png",
        add_hline=True,
    )
    print(f"  Saved summary diagram 2 (delay over time vs {param_name})", flush=True)

    # ---- Diagrams 3 & 4: bar charts (all algos) -----------------------------
    algo_names = [name for name, _ in all_algos[0]]

    final_acc: Dict[str, List[float]] = {n: [] for n in algo_names}
    final_delay: Dict[str, List[float]] = {n: [] for n in algo_names}
    for acc_avg, delay_avg in zip(all_acc_avg, all_delay_avg):
        for n in algo_names:
            final_acc[n].append(float(np.nan if n not in acc_avg else acc_avg[n][-1]))
            final_delay[n].append(float(np.nan if n not in delay_avg else delay_avg[n][-1]))

    x = np.arange(len(param_values))
    bar_width = 0.8 / max(len(algo_names), 1)

    # ---- Diagram 3: final accuracy bar chart --------------------------------
    fig, ax = plt.subplots(figsize=(max(10, len(param_values) * 2), 6))
    for i, name in enumerate(algo_names):
        offset = (i - len(algo_names) / 2 + 0.5) * bar_width
        edgecolor = "black" if name in (focus_algo, ref_algo) else "none"
        linewidth = 1.2 if name in (focus_algo, ref_algo) else 0
        ax.bar(
            x + offset,
            final_acc[name],
            width=bar_width * 0.9,
            label=name,
            color=colors[i % len(colors)],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel(param_name)
    ax.set_ylabel("Cumulative avg accuracy at t=T-1")
    ax.set_title(f"Final accuracy vs {param_name}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"summary_final_accuracy_vs_{param_name}.png"), dpi=180)
    plt.close()
    print(f"  Saved summary diagram 3 (final accuracy bar chart vs {param_name})", flush=True)

    # ---- Diagram 4: final delay bar chart -----------------------------------
    fig, ax = plt.subplots(figsize=(max(10, len(param_values) * 2), 6))
    for i, name in enumerate(algo_names):
        offset = (i - len(algo_names) / 2 + 0.5) * bar_width
        edgecolor = "black" if name in (focus_algo, ref_algo) else "none"
        linewidth = 1.2 if name in (focus_algo, ref_algo) else 0
        ax.bar(
            x + offset,
            final_delay[name],
            width=bar_width * 0.9,
            label=name,
            color=colors[i % len(colors)],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.8, label="feasibility=1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel(param_name)
    ax.set_ylabel("Cumulative avg D_k/(1/R_k) at t=T-1")
    ax.set_title(f"Final delay/(1/R_k) vs {param_name}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"summary_final_delay_vs_{param_name}.png"), dpi=180)
    plt.close()
    print(f"  Saved summary diagram 4 (final delay bar chart vs {param_name})", flush=True)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    param_name: str,
    param_values: List,
    fixed_sigma: float,
    fixed_N: int,
    T: int,
    seed: int,
    c_min: float,
    c_max: float,
    base_out_dir: str,
    baseline_algorithms: Optional[List[str]],
    no_csi_mu: float,
    no_csi_epsilon: float,
) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT: varying {param_name} = {param_values}", flush=True)
    print(f"{'='*60}", flush=True)

    all_acc_avg = []
    all_delay_avg = []
    all_algos_list = []

    for pval in param_values:
        sigma = pval if param_name == "sigma" else fixed_sigma
        N = pval if param_name == "N" else fixed_N

        run_label = f"{param_name}={pval}"
        run_out_dir = os.path.join(base_out_dir, run_label.replace("=", "_").replace(".", "p"))

        print(f"\n--- Run: {run_label} ---", flush=True)
        acc_avg, delay_avg, algos = run_single(
            T=T,
            seed=seed,
            c_min=c_min,
            c_max=c_max,
            sigma=sigma,
            N=N,
            out_dir=run_out_dir,
            baseline_algorithms=baseline_algorithms,
            no_csi_mu=no_csi_mu,
            no_csi_epsilon=no_csi_epsilon,
        )
        all_acc_avg.append(acc_avg)
        all_delay_avg.append(delay_avg)
        all_algos_list.append(algos)

    summary_dir = os.path.join(base_out_dir, "summary")
    plot_summary(
        param_values=param_values,
        param_name=param_name,
        all_acc_avg=all_acc_avg,
        all_delay_avg=all_delay_avg,
        all_algos=all_algos_list,
        T=T,
        out_dir=summary_dir,
    )
    print(f"\nSummary diagrams saved to: {summary_dir}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run sigma-sweep and N-sweep experiments."
    )
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--c_min", type=float, default=10.0,
                   help="Channel min (MLP default). Auto-overridden for ResNet/LLM based on task payload sizes.")
    p.add_argument("--c_max", type=float, default=150.0,
                   help="Channel max (MLP default). Auto-overridden for ResNet/LLM based on task payload sizes.")
    p.add_argument("--mu", type=float, default=3.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="outputs/param_sweep")
    p.add_argument(
        "--baseline_algorithms",
        type=str,
        default="none",
        help=(
            "Comma-separated list of baselines to include, or 'none' to skip all. "
            "Options: max_compression, no_compression, uniform_compression, "
            "myopic, conservative, moving_average."
        ),
    )
    p.add_argument(
        "--experiments",
        type=str,
        default="sigma,N",
        help="Which experiments to run. Comma-separated subset of: sigma, N.",
    )
    p.add_argument(
        "--sigma_values",
        type=str,
        default=None,
        help=(
            "Comma-separated sigma values for sigma sweep. "
            "MLP default: 0.001,0.01,0.05,0.1,0.5  "
            "ResNet recommended: 0.001,0.01,0.05,0.1,0.5 (same, but use small N)"
        ),
    )
    p.add_argument(
        "--N_values",
        type=str,
        default=None,
        help=(
            "Comma-separated N values for N sweep. "
            "MLP default: 10,25,50,75,100  "
            "ResNet recommended: 2,5,10,20  (each eval is ~1s on GPU)"
        ),
    )
    p.add_argument(
        "--fixed_N",
        type=int,
        default=None,
        help=(
            "Fixed N used during sigma sweep. "
            "MLP default: 50.  ResNet recommended: 5  "
            "(2N forward passes per slot — keep small to avoid hour-long waits)"
        ),
    )
    p.add_argument(
        "--fixed_sigma",
        type=float,
        default=0.05,
        help="Fixed sigma used during N sweep (default: 0.05).",
    )
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # Detect backend and set sensible defaults                             #
    # ------------------------------------------------------------------ #
    backend = os.environ.get("INFERENCE_OPTIMIZER_BACKEND", "mnist").strip().lower()
    is_resnet = backend == "resnet"
    is_llm    = backend == "llm"

    # Fixed N for sigma sweep: user override > backend default
    if args.fixed_N is not None:
        fixed_N = args.fixed_N
    elif is_resnet or is_llm:
        fixed_N = 5       # ResNet/LLM: each eval is expensive
    else:
        fixed_N = 50      # MLP: evals are fast

    # Sigma sweep values
    if args.sigma_values is not None:
        sigma_values = [float(x.strip()) for x in args.sigma_values.split(",") if x.strip()]
    else:
        sigma_values = [0.001, 0.01, 0.05, 0.1, 0.5]   # same for all backends

    # N sweep values
    if args.N_values is not None:
        N_values = [int(x.strip()) for x in args.N_values.split(",") if x.strip()]
    elif is_resnet or is_llm:
        N_values = [2, 5, 10, 20]    # ResNet/LLM: each N adds ~2s per time slot
    else:
        N_values = [10, 25, 50, 75, 100]

    # ------------------------------------------------------------------ #
    # Print configuration summary (flush=True for Slurm log visibility)   #
    # ------------------------------------------------------------------ #
    print("=" * 60, flush=True)
    print("run_param_sweep.py  —  Configuration", flush=True)
    print("=" * 60, flush=True)
    print(f"  Backend               : {backend}", flush=True)
    print(f"  T (time slots)        : {args.T}", flush=True)
    print(f"  seed                  : {args.seed}", flush=True)
    print(f"  c_min / c_max         : {args.c_min} / {args.c_max}", flush=True)
    print(f"  mu / epsilon          : {args.mu} / {args.epsilon}", flush=True)
    print(f"  out_dir               : {args.out_dir}", flush=True)
    print(f"  baseline_algorithms   : {args.baseline_algorithms}", flush=True)
    print(f"  experiments           : {args.experiments}", flush=True)
    print(f"  sigma_values (sweep)  : {sigma_values}", flush=True)
    print(f"  N_values (sweep)      : {N_values}", flush=True)
    print(f"  fixed_N (sigma sweep) : {fixed_N}", flush=True)
    print(f"  fixed_sigma (N sweep) : {args.fixed_sigma}", flush=True)
    print("", flush=True)
    print("  Environment variables:", flush=True)
    _ENV_VARS = [
        "INFERENCE_OPTIMIZER_BACKEND",
        "RESNET_CHECKPOINT",
        "RESNET_DATA_ROOT",
        "RESNET_CUTPOINTS",
        "RESNET_BATCH_SIZE",
        "RESNET_FAST_BATCHES",
        "RESNET_GRAD_SIGMA",
        "RESNET_GRAD_N",
        "COMPRESSOR_TYPE",
        "ETA_MIN0",
        "ETA_MIN1",
        "R_K",
        "LLM_MODEL_NAME",
        "LLM_ACTIVATION_STRATEGY",
        "MMLU_FAST_SAMPLES_PER_SUBJECT",
    ]
    for var in _ENV_VARS:
        val = os.environ.get(var, "(not set)")
        print(f"    {var:<35} = {val}", flush=True)
    print("=" * 60, flush=True)

    # ------------------------------------------------------------------ #
    # Runtime estimate                                                     #
    # ------------------------------------------------------------------ #
    if is_resnet:
        algos_count = 2  # CSI-aware + No-CSI (no baselines by default)
        evals_per_slot = algos_count * 2 * fixed_N   # 2N per gradient call
        secs_per_eval = 1.5   # rough estimate: 1 batch, 3 partitions, topk
        est_hours_per_run = (args.T * evals_per_slot * secs_per_eval) / 3600
        n_runs = len(sigma_values) if "sigma" in args.experiments else 0
        n_runs += len(N_values) if "n" in args.experiments.lower() else 0
        print(f"  ⏱  ResNet time estimate per run : ~{est_hours_per_run:.1f} h  "
              f"(T={args.T}, fixed_N={fixed_N}, ~{secs_per_eval}s/eval)", flush=True)
        print(f"  ⏱  Total runs                  : {n_runs}  "
              f"=> ~{n_runs * est_hours_per_run:.1f} h sequential  "
              f"(use --parallel mode to run each as a separate Slurm job)", flush=True)
        print("", flush=True)

    baseline_arg = args.baseline_algorithms.strip().lower()
    baseline_algorithms: Optional[List[str]] = (
        [] if baseline_arg == "none"
        else [s.strip() for s in args.baseline_algorithms.split(",") if s.strip()]
    )

    experiments_to_run = {s.strip().lower() for s in args.experiments.split(",") if s.strip()}

    common = dict(
        T=args.T,
        seed=args.seed,
        c_min=args.c_min,
        c_max=args.c_max,
        baseline_algorithms=baseline_algorithms,
        no_csi_mu=args.mu,
        no_csi_epsilon=args.epsilon,
    )

    # ------------------------------------------------------------------
    # Experiment 1: sigma sweep
    # ------------------------------------------------------------------
    if "sigma" in experiments_to_run:
        run_experiment(
            param_name="sigma",
            param_values=sigma_values,
            fixed_sigma=args.fixed_sigma,
            fixed_N=fixed_N,
            base_out_dir=os.path.join(args.out_dir, "exp1_sigma_sweep"),
            **common,
        )

    # ------------------------------------------------------------------
    # Experiment 2: N sweep
    # ------------------------------------------------------------------
    if "n" in experiments_to_run:
        run_experiment(
            param_name="N",
            param_values=N_values,
            fixed_sigma=args.fixed_sigma,
            fixed_N=fixed_N,
            base_out_dir=os.path.join(args.out_dir, "exp2_N_sweep"),
            **common,
        )

    print("\nAll experiments completed.", flush=True)


if __name__ == "__main__":
    main()
