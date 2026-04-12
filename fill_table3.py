"""
fill_table3.py
=============================================================================
Fills Table 3 (offline single-task results) and generates a compact
reproducibility parameter table for a given scenario group.

A scenario is a set of runs with identical parameters but different
compression methods (Top-k / Quantization / LLM.int8).

Usage
-----
  # Single scenario:
  python fill_table3.py \\
      --scenario "OF-Lang-accuracy" \\
      --topk  outputs/.../llama/topk_per_token \\
      --quant outputs/.../llama/quantization \\
      --int8  outputs/.../llama/llmint8

  # Multiple scenarios in one call (repeating the flag set):
  python fill_table3.py \\
      --scenario "OF-Vision" \\
      --topk  outputs/.../resnet/topk \\
      --quant outputs/.../resnet/quantization \\
      --int8  outputs/.../resnet/llmint8 \\
      --scenario "OF-Lang-accuracy" \\
      --topk  outputs/.../llama/topk_per_token \\
      --quant outputs/.../llama/quantization \\
      --int8  outputs/.../llama/llmint8

  # Save output to file:
  python fill_table3.py ... --out table3.tex

Outputs
-------
  1. Table 3 rows filled in (acc, delay_ms, delay_ratio per algorithm per compression)
  2. Compact reproducibility parameter table (one row per scenario)
"""

import argparse
import json
import os
import sys
import numpy as np

# ── Algorithm order matching Table 3 column groups ───────────────────────────
# Table 3 column order: Max | None | Uniform | Myopic | Consv. | Mov-avg | Ours-CSI | Ours-NoCSI
ALGO_COL_ORDER = [
    "Baseline: max compression",
    "Baseline: no compression",
    "Baseline: uniform compression",
    "Baseline: myopic",
    "Baseline: conservative",
    "Baseline: moving average",
    "CSI-aware (optimal)",
    "No-CSI (Alg1) + LCB(90%)",
]

# Short header labels matching the table
ALGO_SHORT = {
    "Baseline: max compression":    "Max",
    "Baseline: no compression":     "None",
    "Baseline: uniform compression":"Uniform",
    "Baseline: myopic":             "Myopic",
    "Baseline: conservative":       "Consv.",
    "Baseline: moving average":     "Mov.-avg",
    "CSI-aware (optimal)":          "Ours-CSI",
    "No-CSI (Alg1) + LCB(90%)":    "Ours-NoCSI",
}

COMP_KEYS   = ["topk", "quant", "int8"]   # column sub-headers T Q I8
METRICS     = ["utility", "acc", "delay_ms", "excess_delay_ms", "delay_ratio"]
METRIC_LABELS = {
    "utility":          "Avg.~utility",
    "acc":              "Avg.~acc./qual.",
    "delay_ms":         "Avg.~delay (ms)",
    "excess_delay_ms":  "Excess delay (ms)",
    "delay_ratio":      "Delay ratio",
}


# ── Load one run ──────────────────────────────────────────────────────────────

def load_run(out_dir: str) -> tuple:
    """Load simulation_results.npz and experiment_setting.json from out_dir."""
    npz_path  = os.path.join(out_dir, "simulation_results.npz")
    json_path = os.path.join(out_dir, "experiment_setting.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing: {npz_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing: {json_path}")

    data = dict(np.load(npz_path, allow_pickle=True))
    with open(json_path) as f:
        setting = json.load(f)
    return data, setting


def extract_metrics(data: dict, setting: dict, t_idx: int = -1) -> dict:
    """
    Extract acc, delay_ms, delay_ratio for every algorithm at t_idx.
    Returns dict: algo_name → {acc, delay_ms, delay_ratio}
    """
    algo_names      = list(data["algo_names"])
    acc_avg_arr     = data["acc_avg"]          # (A, T)
    dr_avg_arr      = data["delay_ratio_avg"]  # (A, T)
    R_k             = float(setting.get("R_k", 10.0))
    inv_R           = 1.0 / R_k

    out = {}
    for i, name in enumerate(algo_names):
        acc              = float(acc_avg_arr[i, t_idx])
        delay_ratio      = float(dr_avg_arr[i, t_idx])
        delay_ms         = delay_ratio * inv_R * 1000.0
        excess_delay_ms  = (delay_ratio - 1.0) * inv_R * 1000.0
        # For single-task K=1, w_k=1: aggregate utility = acc exactly.
        # Stored separately so multi-task extension can override this.
        utility          = acc
        out[name] = dict(utility=utility, acc=acc, delay_ms=delay_ms,
                         excess_delay_ms=excess_delay_ms,
                         delay_ratio=delay_ratio)
    return out


# ── Compact reproducibility parameter table ───────────────────────────────────

def sci(v: float) -> str:
    """Format float as LaTeX sci notation."""
    if v == 0: return "0"
    exp = int(f"{v:.2e}".split("e")[1])
    m   = v / (10 ** exp)
    if abs(m - 1.0) < 0.01:
        return f"$10^{{{exp}}}$"
    if abs(m - round(m)) < 0.01:
        return f"${int(round(m))}{{\\times}}10^{{{exp}}}$"
    return f"${m:.1f}{{\\times}}10^{{{exp}}}$"


def fmt_bytes(v: float) -> str:
    if v >= 1e6:
        exp = int(f"{v:.2e}".split("e")[1])
        m   = v / (10 ** exp)
        if abs(m - round(m)) < 0.01:
            return f"${int(round(m))}{{\\times}}10^{{{exp}}}$"
        return f"${m:.1f}{{\\times}}10^{{{exp}}}$"
    return str(int(v))


def make_repro_row(scenario: str, setting: dict, T: int) -> str:
    """
    One row for the compact reproducibility table.
    Columns: Scenario | Backend | c_min | c_max | R_k | T | seed |
             η_min | μ | ε | σ | N | Samples (grad/plot) | τ (s) | a (bytes)
    """
    backend = setting.get("backend", "?")
    c_min   = setting.get("c_min", 0)
    c_max   = setting.get("c_max", 0)
    R_k     = setting.get("R_k", 10.0)
    seed    = setting.get("seed", "?")
    eta_min = setting.get("eta_min", "?")
    mu      = setting.get("no_csi_mu", "?")
    eps     = setting.get("no_csi_epsilon", "?")

    # Backend-specific fields
    if backend == "llm":
        model  = setting.get("llm_model_name", "?").split("/")[-1]
        sigma  = setting.get("llm_grad_sigma", "?")
        N      = setting.get("llm_grad_N", "?")
        fast   = setting.get("llm_fast_samples", "?")
        full   = setting.get("llm_full_samples", "?")
        n_subj = 6
        fast_total = int(fast) * n_subj if str(fast).isdigit() else f"{fast}\\times6"
        full_total = int(full) * n_subj if str(full).isdigit() else f"{full}\\times6"
        samples = f"{fast_total} / {full_total}"
        backend_str = f"LLM ({model})"
    elif backend == "resnet":
        model  = "ResNet-56"
        sigma  = setting.get("resnet_grad_sigma", "?")
        N      = setting.get("resnet_grad_N", "?")
        fast_b = setting.get("resnet_fast_batches", 1)
        bsz    = setting.get("resnet_batch_size", 100)
        fast_total = int(fast_b) * int(bsz)
        samples = f"{fast_total} / 10{{,}}000"
        backend_str = "ResNet-56"
    else:
        sigma  = setting.get("mlp_sigma", "?")
        N      = setting.get("mlp_N", "?")
        n_samp = setting.get("mlp_n_samples", 512)
        samples = f"{n_samp} / {n_samp}"
        backend_str = "MLP/MNIST"

    # tau and a
    tau_dict = setting.get("tau", {})
    tau_str  = "[" + ", ".join(f"{v:.4f}" for v in tau_dict.values()) + "]"

    a_dict  = setting.get("a", {})
    a_str   = "[" + ", ".join(fmt_bytes(v) for v in a_dict.values()) + "]"

    cols = [
        scenario, backend_str,
        sci(c_min), sci(c_max), f"{R_k:.0f}",
        str(T), str(seed),
        str(eta_min), str(mu), str(eps),
        str(sigma), str(N),
        samples,
        tau_str, a_str,
    ]
    return " & ".join(cols) + r" \\"


def make_repro_table(repro_rows: list) -> str:
    """Wrap repro rows in a full table environment."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Offline single-task experiment parameters. "
                 r"Samples column shows gradient-oracle / plot evaluations. "
                 r"Compression method varies per column group in Table~\ref{tab:results-offline-single}.}")
    lines.append(r"\label{tab:repro-offline-single}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(
        r"\begin{tabular}{@{}"
        r"l l "           # Scenario, Backend
        r"ccc "           # c_min, c_max, R_k
        r"cc "            # T, seed
        r"ccc "           # eta_min, mu, eps
        r"cc "            # sigma, N
        r"c "             # Samples
        r"cc"             # tau, a
        r"@{}}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Scenario} & \textbf{Backend} & "
        r"$\bm{c_{\min}}$ & $\bm{c_{\max}}$ & $\bm{R_k}$ & "
        r"$\bm{T}$ & \textbf{seed} & "
        r"$\bm{\eta_{\min}}$ & $\bm{\mu}$ & $\bm{\varepsilon}$ & "
        r"$\bm{\sigma}$ & $\bm{N}$ & "
        r"\textbf{Samples (grad/plot)} & "
        r"$\bm{\tau}$ (s) & $\bm{a}$ (bytes) \\"
    )
    lines.append(r"\midrule")
    for row in repro_rows:
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ── Table 3 filling ───────────────────────────────────────────────────────────

def fmt_val(v: float, metric: str) -> str:
    if metric in ("acc", "utility"):
        return f"{v:.4f}"
    elif metric in ("delay_ms", "excess_delay_ms"):
        return f"{v:.2f}"
    else:  # delay_ratio
        return f"{v:.3f}"


def fill_table3_scenario(
    scenario:    str,
    runs:        dict,
    t_idx:       int = -1,
) -> list:
    """
    Generate 4 data rows (acc, delay_ms, excess_delay_ms, delay_ratio)
    for one scenario in Table 3.
    Each row has 24 cells: 8 algorithms × 3 compression methods.
    Missing runs show '--'.
    """
    rows = []
    is_first = True
    n_rows = len(METRICS)

    for metric in METRICS:
        cells = []
        for algo in ALGO_COL_ORDER:
            for comp in COMP_KEYS:
                metrics_dict = runs.get(comp, {})
                algo_metrics = metrics_dict.get(algo, None)
                if algo_metrics is None:
                    cells.append("--")
                else:
                    v = algo_metrics[metric]
                    cells.append(fmt_val(v, metric))

        # Bold best value in Ours-CSI and Ours-NoCSI groups
        ours_start   = len(ALGO_COL_ORDER) * len(COMP_KEYS) - 2 * len(COMP_KEYS)
        best_indices = set()
        if metric in ("acc", "utility"):
            # Higher is better
            for group_start in [ours_start, ours_start + len(COMP_KEYS)]:
                group_vals = []
                for j in range(len(COMP_KEYS)):
                    try:
                        group_vals.append((float(cells[group_start + j]), group_start + j))
                    except ValueError:
                        pass
                if group_vals:
                    best_indices.add(max(group_vals, key=lambda x: x[0])[1])
        elif metric in ("delay_ratio", "delay_ms", "excess_delay_ms"):
            # Lower is better (closer to 1.0 for ratio, closest to 0 for excess/delay)
            for group_start in [ours_start, ours_start + len(COMP_KEYS)]:
                group_vals = []
                for j in range(len(COMP_KEYS)):
                    try:
                        group_vals.append((float(cells[group_start + j]), group_start + j))
                    except ValueError:
                        pass
                if group_vals:
                    best_indices.add(min(group_vals, key=lambda x: x[0])[1])

        formatted = []
        for i, c in enumerate(cells):
            if i in best_indices and c != "--":
                formatted.append(f"\\textbf{{{c}}}")
            else:
                formatted.append(c)

        metric_label = METRIC_LABELS[metric]
        if is_first:
            prefix = (f"\\multirow{{{n_rows}}}{{*}}"
                      f"{{\\shortstack[l]{{{scenario}}}}} & {metric_label}")
            is_first = False
        else:
            prefix = f" & {metric_label}"

        rows.append(f"  {prefix} & " + " & ".join(formatted) + r" \\")

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fill Table 3 and generate compact repro table for "
                    "grouped single-task experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scenario",  action="append", default=[],
                   help="Scenario label (repeat for multiple scenarios).")
    p.add_argument("--topk",      action="append", default=[],
                   help="out_dir for Top-k run (repeat for each scenario).")
    p.add_argument("--quant",     action="append", default=[],
                   help="out_dir for Quantization run (repeat for each scenario).")
    p.add_argument("--int8",      action="append", default=[],
                   help="out_dir for LLM.int8 run (repeat for each scenario).")
    p.add_argument("--t_idx",     type=int, default=-1,
                   help="Time index (-1 = last slot).")
    p.add_argument("--out",       type=str, default=None,
                   help="Output .tex file (default: print to stdout).")
    args = p.parse_args()

    # Validate: all lists must have the same length
    n = len(args.scenario)
    if not n:
        print("ERROR: at least one --scenario is required.", file=sys.stderr)
        sys.exit(1)
    for name, lst in [("--topk", args.topk), ("--quant", args.quant),
                       ("--int8", args.int8)]:
        if len(lst) != n:
            print(f"ERROR: {name} must be specified {n} time(s) to match "
                  f"{n} --scenario(s).", file=sys.stderr)
            sys.exit(1)

    # ── Load all runs ─────────────────────────────────────────────────────────
    scenario_data = []   # list of (scenario, runs_dict, setting, T)
    for i in range(n):
        scenario  = args.scenario[i]
        comp_dirs = {"topk": args.topk[i], "quant": args.quant[i],
                     "int8": args.int8[i]}
        runs      = {}
        setting   = None
        T         = None

        for comp, out_dir in comp_dirs.items():
            if not out_dir or out_dir == "--":
                runs[comp] = {}
                continue
            try:
                data, s   = load_run(out_dir)
                T         = len(data["times"])
                if setting is None:
                    setting = s   # use first available run's setting for repro table
                runs[comp] = extract_metrics(data, s, args.t_idx)
                print(f"[load] {scenario}/{comp}: T={T}, "
                      f"algos={list(data['algo_names'][:2])}...",
                      file=sys.stderr)
            except FileNotFoundError as e:
                print(f"[warn] {scenario}/{comp}: {e}", file=sys.stderr)
                runs[comp] = {}

        scenario_data.append((scenario, runs, setting, T or 0))

    # ── Generate Table 3 rows ─────────────────────────────────────────────────
    table3_lines = []
    repro_rows   = []

    for scenario, runs, setting, T in scenario_data:
        rows = fill_table3_scenario(scenario, runs, args.t_idx)
        table3_lines.extend(rows)
        table3_lines.append(r"\midrule")

        if setting is not None:
            repro_rows.append(make_repro_row(scenario, setting, T))

    # Remove trailing \midrule
    if table3_lines and table3_lines[-1] == r"\midrule":
        table3_lines.pop()

    # ── Assemble Table 3 block ────────────────────────────────────────────────
    t_actual = "T"  # generic since T may differ across scenarios

    table3 = []
    table3.append(r"% ============================================================")
    table3.append(r"% Table 3 rows — paste inside \begin{tabular} environment")
    table3.append(r"% ============================================================")
    table3.extend(table3_lines)

    # ── Assemble compact repro table ──────────────────────────────────────────
    repro_table = make_repro_table(repro_rows)

    output = "\n".join(table3) + "\n\n" + repro_table

    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print(f"Saved to: {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
