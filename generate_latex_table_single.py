"""
generate_latex_table_single.py
=====================================================================
Reads the output of run_experiment.py and generates a LaTeX table
in the single-task paper format (Table 7), plus a parameter label.

Reads from <out_dir>:
  - simulation_results.npz
  - experiment_setting.json

Metrics at t = T-1 (final cumulative average):
  - Avg. Accuracy:   acc_avg[algo, -1]
  - Avg. Delay (ms): delay_ratio_avg[algo, -1] × (1/R_k) × 1000
  - Delay Ratio:     delay_ratio_avg[algo, -1]

Usage
-----
  python generate_latex_table_single.py \\
      --out_dir outputs/single_experiments/group2_rerun/llama/random

  python generate_latex_table_single.py \\
      --out_dir outputs/single_experiments/group2_rerun/llama/random \\
      --label OF-L2rr2-rand \\
      --t_idx -1
"""

import argparse
import json
import os
import sys

import numpy as np

# ── Algorithm name → (LaTeX display name, is_ours) ───────────────────────────
ALGO_LATEX = {
    "CSI-aware (optimal)":          (r"\textbf{Ours: CSI-Aware (Optimal)}",    True),
    "No-CSI (Alg1) + LCB(90%)":    (r"\textbf{Ours: No-CSI (Alg.~1 + LCB)}", True),
    "Baseline: max compression":    (r"\textsf{Baseline: max compression}",     False),
    "Baseline: no compression":     (r"\textsf{Baseline: no compression}",      False),
    "Baseline: uniform compression":(r"\textsf{Baseline: uniform compression}", False),
    "Baseline: myopic":             (r"\textsf{No-CSI: myopic}",                False),
    "Baseline: conservative":       (r"\textsf{No-CSI: conservative}",          False),
    "Baseline: moving average":     (r"\textsf{No-CSI: moving average}",        False),
}

# Order baselines appear in the table
BASELINE_ORDER = [
    "Baseline: max compression",
    "Baseline: no compression",
    "Baseline: uniform compression",
    "Baseline: myopic",
    "Baseline: conservative",
    "Baseline: moving average",
]

MAIN_ORDER = [
    "CSI-aware (optimal)",
    "No-CSI (Alg1) + LCB(90%)",
]


# ── Compression label ─────────────────────────────────────────────────────────
def compression_label(setting: dict, algo_name: str) -> str:
    backend = setting.get("backend", "")
    if "no compression" in algo_name.lower():
        return "None"
    if backend == "llm":
        strat = setting.get("llm_activation_strategy", "")
        return {
            "topk_per_token": "Top-$k$",
            "magnitude":      "Magnitude",
            "random":         "Random",
            "quantization":   "Quant.",
            "llmint8":        "LLM.int8",
        }.get(strat, strat)
    comp = setting.get("compressor", "topk")
    return {"topk": "Top-$k$", "quantization": "Quant.",
            "llmint8": "LLM.int8"}.get(comp, comp)


# ── Parameter label ───────────────────────────────────────────────────────────
def make_param_label(setting: dict, T: int) -> str:
    backend  = setting.get("backend", "?")
    c_min    = setting.get("c_min", 0)
    c_max    = setting.get("c_max", 0)
    R_k      = setting.get("R_k", 10)
    eta_min  = setting.get("eta_min", "?")
    seed     = setting.get("seed", "?")
    mu       = setting.get("no_csi_mu", "?")
    eps      = setting.get("no_csi_epsilon", "?")

    a_dict   = setting.get("a", {})
    a_vals   = [f"{v:.0f}" for v in a_dict.values()]
    tau_dict = setting.get("tau", {})
    tau_vals = [f"{v:.4f}" for v in tau_dict.values()]

    if backend == "llm":
        model  = setting.get("llm_model_name", "?").split("/")[-1]
        strat  = setting.get("llm_activation_strategy", "?")
        fast   = setting.get("llm_fast_samples", "?")
        full   = setting.get("llm_full_samples", "?")
        sigma  = setting.get("llm_grad_sigma", "?")
        N      = setting.get("llm_grad_N", "?")
        backend_str = (f"LLM ({model}, {strat}) $|$ "
                       f"fast\\_samples={fast}/subj $|$ full\\_samples={full}/subj $|$ "
                       f"$\\sigma$={sigma} $|$ $N$={N}")
    elif backend == "resnet":
        cuts  = setting.get("resnet_cutpoints", "?")
        sigma = setting.get("resnet_grad_sigma", "?")
        N     = setting.get("resnet_grad_N", "?")
        backend_str = f"ResNet-56 (cutpoints={cuts}) $|$ $\\sigma$={sigma} $|$ $N$={N}"
    else:
        sigma = setting.get("mlp_sigma", "?")
        N     = setting.get("mlp_N", "?")
        backend_str = f"MLP/MNIST $|$ $\\sigma$={sigma} $|$ $N$={N}"

    lines = [
        backend_str,
        (f"$T$={T} $|$ seed={seed} $|$ "
         f"$c \\sim \\mathcal{{U}}[{c_min:.2e},\\,{c_max:.2e}]$ $|$ "
         f"$R_k$={R_k} $|$ $\\eta_{{\\min}}$={eta_min} $|$ "
         f"$\\mu$={mu} $|$ $\\varepsilon$={eps}"),
        f"$a$ (bytes)$=[{',\\,'.join(a_vals)}]$ $|$ "
        f"$\\tau$ (s)$=[{',\\,'.join(tau_vals)}]$",
    ]
    return r" \\ ".join(lines)


# ── Format one table row ──────────────────────────────────────────────────────
def fmt_row(name: str, acc: float, delay_ms: float, excess_delay_ms: float,
            delay_ratio: float, comp: str, bold: bool) -> str:
    latex_name, _ = ALGO_LATEX.get(name, (r"\textsf{" + name + "}", False))
    b = r"\textbf" if bold else ""

    acc_str    = f"{b}{{{acc:.4f}}}"            if bold else f"{acc:.4f}"
    delay_str  = f"{b}{{{delay_ms:.2f}}}"       if bold else f"{delay_ms:.2f}"
    excess_str = f"{b}{{{excess_delay_ms:.2f}}}" if bold else f"{excess_delay_ms:.2f}"
    dr_str     = f"{b}{{{delay_ratio:.4f}}}"    if bold else f"{delay_ratio:.4f}"
    comp_str   = f"{b}{{{comp}}}"               if bold else comp

    return (f"  {latex_name} & {comp_str} & "
            f"{acc_str} & {delay_str} & {excess_str} & {dr_str} \\\\")


def make_caption(setting: dict, T: int, t_actual: int, label: str) -> str:
    backend  = setting.get("backend", "?")
    c_min    = setting.get("c_min", 0)
    c_max    = setting.get("c_max", 0)
    R_k      = setting.get("R_k", 10)
    eta_min  = setting.get("eta_min", "?")
    seed     = setting.get("seed", "?")
    mu       = setting.get("no_csi_mu", "?")
    eps      = setting.get("no_csi_epsilon", "?")

    if backend == "llm":
        model   = setting.get("llm_model_name", "?").split("/")[-1]
        strat   = setting.get("llm_activation_strategy", "?")
        fast    = setting.get("llm_fast_samples", "?")
        full    = setting.get("llm_full_samples", "?")
        sigma   = setting.get("llm_grad_sigma", "?")
        N       = setting.get("llm_grad_N", "?")
        n_subj  = 6  # default MMLU subjects
        fast_total = int(fast) * n_subj if str(fast).isdigit() else f"{fast}\\times 6"
        full_total = int(full) * n_subj if str(full).isdigit() else f"{full}\\times 6"
        backend_str = (
            f"Model: {model}, strategy: {strat}. "
            f"Accuracy oracle: {fast_total} samples/eval (gradient), "
            f"{full_total} samples/eval (plot). "
            f"Stein $\\sigma={sigma}$, $N={N}$."
        )
    elif backend == "resnet":
        cuts    = setting.get("resnet_cutpoints", "?")
        sigma   = setting.get("resnet_grad_sigma", "?")
        N       = setting.get("resnet_grad_N", "?")
        fast_b  = setting.get("resnet_fast_batches", 1)
        batch_sz= setting.get("resnet_batch_size", 100)
        fast_total = int(fast_b) * int(batch_sz)
        full_total = "10{,}000"
        backend_str = (
            f"ResNet-56, cutpoints: {cuts}. "
            f"Accuracy oracle: {fast_total} samples/eval (gradient, "
            f"{fast_b}~batch(es) $\\times$ {batch_sz}), "
            f"${full_total}$ samples/eval (plot, full test set). "
            f"Stein $\\sigma={sigma}$, $N={N}$."
        )
    else:
        sigma   = setting.get("mlp_sigma", "?")
        N       = setting.get("mlp_N", "?")
        n_samp  = setting.get("mlp_n_samples", "?")
        backend_str = (
            f"MLP/MNIST. "
            f"Accuracy oracle: {n_samp} samples/eval. "
            f"Stein $\\sigma={sigma}$, $N={N}$."
        )

    return (
        f"Single-task performance averaged over $t=1,\\ldots,{t_actual+1}$ slots "
        f"({label}). "
        f"{backend_str} "
        f"Channel: $c\\sim\\mathcal{{U}}[{c_min:.2e},\\,{c_max:.2e}]$~B/s, "
        f"$T={T}$, seed$={seed}$, $R_k={R_k}$, "
        f"$\\eta_{{\\min}}={eta_min}$, $\\mu={mu}$, $\\varepsilon={eps}$. "
        r"Delay Ratio $\leq 1$ indicates QoS satisfied."
    )


# ── Reproducibility table row (Table 5 style) ─────────────────────────────────

def generate_repro_row(setting: dict, T: int, algo_names: list,
                       cfg_id: str, task_ids: str,
                       testbed_id: str = "--",
                       t_duration: str = "--",
                       w_k: str = "--") -> str:
    """
    Generate one row for the offline reproducibility table (Table 5).

    Columns:
      Cfg. ID | Task IDs | K | Testbed ID | Samples | Batch | T | T_duration |
      Split bounds | Node map | A_k | tau_k | a_k | eta_min_k |
      Comp. algs. | Channel descriptors | R_k | w_k

    Parameters
    ----------
    cfg_id      : e.g. "OF-L2rr2-rand"
    task_ids    : e.g. "[Lang-5]"
    testbed_id  : e.g. "C" or "--" for simulation
    t_duration  : wall-clock duration string e.g. "6.4h" or "--"
    w_k         : task weight string e.g. "[1.0]" or "--"
    """
    backend = setting.get("backend", "?")
    K       = 1
    R_k     = setting.get("R_k", 10.0)
    c_min   = setting.get("c_min", 0)
    c_max   = setting.get("c_max", 0)
    eta_min = setting.get("eta_min", "?")
    seed    = setting.get("seed", "?")

    # ── Samples and batch ─────────────────────────────────────────────────────
    if backend == "llm":
        full     = setting.get("llm_full_samples", "?")
        n_subj   = 6
        samples  = f"[{int(full) * n_subj}]" if str(full).isdigit() else f"[{full}\\times6]"
        batch    = f"[{setting.get('llm_batch_size', 8)}]"
    elif backend == "resnet":
        batch_sz = setting.get("resnet_batch_size", 100)
        fast_b   = setting.get("resnet_fast_batches", 1)
        samples  = f"[{int(fast_b) * int(batch_sz)}]"
        batch    = f"[{batch_sz}]"
    else:
        samples  = f"[{setting.get('mlp_n_samples', 512)}]"
        batch    = "--"

    # ── Topology ──────────────────────────────────────────────────────────────
    L_k       = setting.get("L_k", 3)
    num_nodes = L_k

    # Split bounds
    if backend == "resnet":
        cuts_str = setting.get("resnet_cutpoints", "8,14,21")
        cuts = [int(c) for c in cuts_str.split(",")]
        bounds, prev = [], 0
        for c in cuts:
            bounds.append(f"[{prev},{c-1}]")
            prev = c
        bounds.append(f"[{prev},55]")
        split_bounds = "[" + ",".join(bounds) + "]"
    elif backend == "llm":
        model_name = setting.get("llm_model_name", "")
        if "llama" in model_name.lower():
            split_bounds = "[[0,10],[11,21]]"
        elif "gemma-2" in model_name.lower():
            split_bounds = "[[0,8],[9,17]]"
        else:
            split_bounds = "[[0,5],[6,11]]"
    else:
        split_bounds = "--"

    # Node map
    node_map = "[" + ",".join(f"[{i+1}]" for i in range(num_nodes)) + "]"

    # ── A_k metric ────────────────────────────────────────────────────────────
    if backend == "llm":
        metric = "[MMLU-Acc]"
    elif backend == "resnet":
        metric = "[Top-1-Acc]"
    else:
        metric = "[Acc]"

    # ── tau_k ─────────────────────────────────────────────────────────────────
    tau_dict = setting.get("tau", {})
    tau_vals = [f"{v:.4f}" for v in tau_dict.values()]
    tau_str  = "[[" + ",".join(tau_vals) + "]]"

    # ── a_k ───────────────────────────────────────────────────────────────────
    def fmt_bytes(v):
        if v >= 1e6:
            exp = int(f"{v:.2e}".split("e")[1])
            m   = v / (10 ** exp)
            if abs(m - round(m)) < 0.01:
                return f"${int(round(m))}{{\\times}}10^{{{exp}}}$"
            return f"${m:.1f}{{\\times}}10^{{{exp}}}$"
        return str(int(v))

    a_dict = setting.get("a", {})
    a_str  = "[[" + ",".join(fmt_bytes(v) for v in a_dict.values()) + "]]"

    # ── eta_min_k ─────────────────────────────────────────────────────────────
    eta_min_dict = setting.get("eta_min_per_link", {})
    if eta_min_dict:
        vals = list(eta_min_dict.values())
        unique = list(set(vals))
        eta_str = f"[[{unique[0]}]]" if len(unique) == 1 else \
                  "[[" + ",".join(str(v) for v in vals) + "]]"
    else:
        eta_str = f"[[{eta_min}]]"

    # ── Compression algorithms ────────────────────────────────────────────────
    if backend == "llm":
        strat = setting.get("llm_activation_strategy", "?")
        comp_map = {"topk_per_token": "topk", "magnitude": "magnitude",
                    "random": "random", "quantization": "quant", "llmint8": "llmint8"}
        comp_algs = f"[{comp_map.get(strat, strat)}]"
    else:
        comp_algs = f"[{setting.get('compressor', 'topk')}]"

    # ── Channel descriptors ───────────────────────────────────────────────────
    def sci(v):
        if v == 0: return "0"
        exp = int(f"{v:.2e}".split("e")[1])
        m   = v / (10 ** exp)
        if abs(m - 1.0) < 0.01:
            return f"$10^{{{exp}}}$"
        if abs(m - round(m)) < 0.01:
            return f"${int(round(m))}{{\\times}}10^{{{exp}}}$"
        return f"${m:.1f}{{\\times}}10^{{{exp}}}$"

    channel = (f"uniform-iid, "
               f"$c\\sim\\mathcal{{U}}[{sci(c_min)},{sci(c_max)}]$, "
               f"seed$={seed}$")

    # ── Assemble row ──────────────────────────────────────────────────────────
    cols = [
        cfg_id, task_ids, str(K), testbed_id,
        samples, batch, str(T), t_duration,
        split_bounds, node_map, metric,
        tau_str, a_str, eta_str, comp_algs,
        channel, f"[{R_k:.0f}]", w_k,
    ]
    return " & ".join(cols) + r" \\"


def main():
    p = argparse.ArgumentParser(
        description="Generate LaTeX single-task table from run_experiment.py output."
    )
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory with simulation_results.npz + experiment_setting.json")
    p.add_argument("--t_idx", type=int, default=-1,
                   help="Time index to read metrics from (-1 = last slot).")
    p.add_argument("--label", type=str, default=None,
                   help="Short label for table caption and \\label{}.")
    p.add_argument("--repro_row", action="store_true", default=False,
                   help="Also print a reproducibility table row (Table 5 style).")
    p.add_argument("--cfg_id",      type=str, default=None,
                   help="Config ID for repro row, e.g. OF-L2rr2-rand.")
    p.add_argument("--task_ids",    type=str, default="[Lang-5]",
                   help="Task ID string for repro row, e.g. [Lang-5].")
    p.add_argument("--testbed_id",  type=str, default="--",
                   help="Testbed ID for repro row.")
    p.add_argument("--t_duration",  type=str, default="--",
                   help="Wall-clock duration string for repro row, e.g. 6.4h.")
    p.add_argument("--w_k",         type=str, default="--",
                   help="Task weight string for repro row, e.g. [1.0].")
    args = p.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    npz_path  = os.path.join(args.out_dir, "simulation_results.npz")
    json_path = os.path.join(args.out_dir, "experiment_setting.json")

    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found.", file=sys.stderr); sys.exit(1)
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found.", file=sys.stderr); sys.exit(1)

    data = dict(np.load(npz_path, allow_pickle=True))
    with open(json_path) as f:
        setting = json.load(f)

    algo_names      = list(data["algo_names"])
    T               = len(data["times"])
    t_idx           = args.t_idx
    t_actual        = T + t_idx if t_idx < 0 else t_idx

    acc_avg_arr         = data["acc_avg"]          # (A, T)
    delay_ratio_avg_arr = data["delay_ratio_avg"]  # (A, T)

    R_k   = float(setting.get("R_k", 10.0))
    inv_R = 1.0 / R_k  # target delay in seconds

    # ── Build metric dict ─────────────────────────────────────────────────────
    metrics = {}
    for i, name in enumerate(algo_names):
        acc              = float(acc_avg_arr[i, t_idx])
        delay_ratio      = float(delay_ratio_avg_arr[i, t_idx])
        delay_ms         = delay_ratio * inv_R * 1000.0
        excess_delay_ms  = (delay_ratio - 1.0) * inv_R * 1000.0
        comp             = compression_label(setting, name)
        metrics[name] = dict(acc=acc, delay_ms=delay_ms,
                             excess_delay_ms=excess_delay_ms,
                             delay_ratio=delay_ratio, comp=comp)

    # ── Build table ───────────────────────────────────────────────────────────
    label       = args.label or os.path.basename(args.out_dir.rstrip("/"))
    param_label = make_param_label(setting, T)

    # Order rows: known baselines first, then main algorithms,
    # then any unknown names appended at the end
    known   = set(BASELINE_ORDER + MAIN_ORDER)
    extras  = [n for n in algo_names if n not in known]
    bl_rows = [n for n in BASELINE_ORDER if n in algo_names]
    mn_rows = [n for n in MAIN_ORDER     if n in algo_names]

    table = []
    table.append(r"% " + "=" * 70)
    table.append(r"% Auto-generated by generate_latex_table_single.py")
    table.append(f"% out_dir: {args.out_dir}")
    table.append(f"% t_idx:   {args.t_idx} (t={t_actual})")
    table.append(r"% " + "=" * 70)
    table.append(r"")
    table.append(r"% Parameter label — copy into caption or footnote:")
    table.append(r"% \footnotesize{" + param_label + r"}")
    table.append(r"")
    table.append(r"\begin{table}[t]")
    table.append(r"\centering")
    caption = make_caption(setting, T, t_actual, label)
    table.append(f"\\caption{{{caption}}}")
    table.append(f"\\label{{tab:single:{label}}}")
    table.append(r"\small")
    table.append(r"\resizebox{\columnwidth}{!}{%")
    table.append(r"\begin{tabular}{@{}ll cccc@{}}")
    table.append(r"\toprule")
    table.append(
        r"\textbf{Policy} & \textbf{Compression} & "
        r"\textbf{Avg.~Accuracy ($A$)} & "
        r"\textbf{Avg.~Delay (ms)} & "
        r"\textbf{Excess Delay (ms)} & "
        r"\textbf{Delay Ratio} \\"
    )
    table.append(r"\midrule")

    # Baseline rows
    for name in bl_rows:
        m = metrics[name]
        table.append(fmt_row(name, m["acc"], m["delay_ms"], m["excess_delay_ms"],
                             m["delay_ratio"], m["comp"], bold=False))

    # Any extra unknown baselines
    for name in extras:
        m = metrics[name]
        table.append(fmt_row(name, m["acc"], m["delay_ms"], m["excess_delay_ms"],
                             m["delay_ratio"], m["comp"], bold=False))

    table.append(r"\midrule")

    # Main algorithm rows
    for name in mn_rows:
        m = metrics[name]
        table.append(fmt_row(name, m["acc"], m["delay_ms"], m["excess_delay_ms"],
                             m["delay_ratio"], m["comp"], bold=True))

    table.append(r"\midrule")
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"}")
    table.append(r"\end{table}")

    latex_str = "\n".join(table)
    print(latex_str)

    # Save to file
    out_path = os.path.join(args.out_dir, "latex_table.tex")
    with open(out_path, "w") as f:
        f.write(latex_str + "\n")
    print(f"\n% Saved to: {out_path}", file=sys.stderr)

    # ── Reproducibility row ───────────────────────────────────────────────────
    if args.repro_row:
        cfg_id = args.cfg_id or label
        print(f"\n% Reproducibility table row (Table 5):", file=sys.stderr)
        repro = generate_repro_row(
            setting     = setting,
            T           = T,
            algo_names  = algo_names,
            cfg_id      = cfg_id,
            task_ids    = args.task_ids,
            testbed_id  = args.testbed_id,
            t_duration  = args.t_duration,
            w_k         = args.w_k,
        )
        print("\n% --- Repro row ---")
        print(repro)
        # Append to latex_table.tex
        with open(out_path, "a") as f:
            f.write("\n% --- Repro row (Table 5) ---\n")
            f.write(repro + "\n")
        print(f"% Repro row appended to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
