"""
diagnose_llm_accuracy.py — Diagnostic probe for LLM accuracy function
======================================================================

Loads the LLM backend once and evaluates A(eta) at a grid of eta values
for each requested strategy. Also estimates the Stein gradient magnitude
to check whether the gradient oracle can detect accuracy differences.

This script answers the question "is there a meaningful A(eta) curve
before committing to a full T-slot experiment?" in minutes rather than hours.

What is tested
--------------
For each (strategy, eta) pair:
  - accuracy_callable_true(eta)  : full-sample accuracy (used in plots)
  - accuracy_callable(eta)       : fast-sample accuracy (used in gradient oracle)
  - gradient_callable(eta)       : Stein gradient estimate (direction of improvement)

Additionally, for each strategy:
  - Noise estimate: evaluate accuracy_callable(eta) N_repeat times at the
    same eta and report std — tells you if the sample size is large enough
    to produce a stable estimate.

Usage
-----
  # All strategies, default eta grid:
  python diagnose_llm_accuracy.py

  # Specific strategies:
  python diagnose_llm_accuracy.py --strategies quantization llmint8

  # Custom eta grid:
  python diagnose_llm_accuracy.py --eta_values 0.25 0.5 0.75 1.0

  # More samples for noise estimate:
  python diagnose_llm_accuracy.py --noise_repeats 5

  # Full example matching the quantization rerun settings:
  export LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
  export LLM_ACTIVATION_STRATEGY=quantization
  export MMLU_FAST_SAMPLES_PER_SUBJECT=100
  export MMLU_FULL_SAMPLES_PER_SUBJECT=200
  export MMLU_BATCH_SIZE=8
  export MMLU_SEED=0
  python diagnose_llm_accuracy.py \\
      --strategies quantization llmint8 topk_per_token \\
      --eta_values 0.25 0.5 0.75 1.0 \\
      --eta_min 0.25 \\
      --noise_repeats 3

Output
------
  Prints a table per strategy showing A(eta) for each eta value,
  gradient magnitude, and noise std. Saves results to
  <out_dir>/diagnosis_results.json for later reference.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))


# ============================================================================
# Core diagnostic
# ============================================================================

def probe_strategy(
    strategy:       str,
    model_name:     str,
    eta_values:     List[float],
    eta_min:        float,
    fast_samples:   int,
    full_samples:   int,
    batch_size:     int,
    grad_sigma:     float,
    grad_N:         int,
    noise_repeats:  int,
    seed:           int,
    **kwargs,
) -> Dict:
    """
    Load the backend for one strategy and evaluate A(eta) across eta_values.
    Returns a dict of results for JSON serialisation.
    """
    print(f"\n{'='*65}", flush=True)
    print(f"Strategy: {strategy}", flush=True)
    print(f"  model          = {model_name}", flush=True)
    print(f"  eta_min        = {eta_min}", flush=True)
    print(f"  fast_samples   = {fast_samples}/subj", flush=True)
    print(f"  full_samples   = {full_samples}/subj", flush=True)
    print(f"  grad_sigma     = {grad_sigma}  grad_N = {grad_N}", flush=True)
    print(f"  noise_repeats  = {noise_repeats}", flush=True)
    print(f"  eta grid       = {eta_values}", flush=True)
    print(f"{'='*65}", flush=True)

    # Set env vars for this strategy
    os.environ['LLM_MODEL_NAME']              = model_name
    os.environ['LLM_ACTIVATION_STRATEGY']     = strategy
    os.environ['LLM_SPLIT_MODE']              = 'thirds'
    os.environ['MMLU_FAST_SAMPLES_PER_SUBJECT'] = str(fast_samples)
    os.environ['MMLU_FULL_SAMPLES_PER_SUBJECT'] = str(full_samples)
    os.environ['MMLU_BATCH_SIZE']             = str(batch_size)
    os.environ['MMLU_SEED']                   = str(seed)
    os.environ['LLM_GRAD_SIGMA']              = str(grad_sigma)
    os.environ['LLM_GRAD_N']                  = str(grad_N)
    os.environ['INFERENCE_OPTIMIZER_BACKEND'] = 'llm'

    from src.core.llm_task_callables import build_llm_backend_from_env
    api = build_llm_backend_from_env(eta_min=eta_min)

    results = {
        "strategy":     strategy,
        "model_name":   model_name,
        "eta_min":      eta_min,
        "fast_samples": fast_samples,
        "full_samples": full_samples,
        "grad_sigma":   grad_sigma,
        "grad_N":       grad_N,
        "eta_sweep":    [],
        "noise":        [],
    }

    # ── A(eta) sweep ──────────────────────────────────────────────────────────
    print(f"\n  {'eta_vec':<20} {'A_true':>8} {'A_fast':>8} {'grad_mag':>10} {'time(s)':>8}", flush=True)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*8}", flush=True)

    for eta_val in eta_values:
        # Use the same eta on both links
        eta_vec = np.array([eta_val, eta_val], dtype=float)

        t0 = time.time()
        a_true = api.accuracy_callable_true(eta_vec)
        a_fast = api.accuracy_callable(eta_vec)
        grad   = api.gradient_callable(eta_vec)
        grad_mag = float(np.linalg.norm(grad))
        elapsed = time.time() - t0

        print(f"  [{eta_val:.3f}, {eta_val:.3f}]       "
              f"{a_true:8.4f} {a_fast:8.4f} {grad_mag:10.6f} {elapsed:8.1f}",
              flush=True)

        results["eta_sweep"].append({
            "eta":      eta_val,
            "a_true":   float(a_true),
            "a_fast":   float(a_fast),
            "grad":     grad.tolist(),
            "grad_mag": grad_mag,
            "time_s":   elapsed,
        })

    # ── Noise estimate at eta_min ─────────────────────────────────────────────
    print(f"\n  Noise estimate at eta=[{eta_min:.3f}, {eta_min:.3f}] "
          f"({noise_repeats} repeats of accuracy_callable):", flush=True)

    eta_vec_min = np.array([eta_min, eta_min], dtype=float)
    noise_vals  = []
    for i in range(noise_repeats):
        v = api.accuracy_callable(eta_vec_min)
        noise_vals.append(float(v))
        print(f"    repeat {i+1}: A_fast = {v:.4f}", flush=True)

    noise_mean = float(np.mean(noise_vals))
    noise_std  = float(np.std(noise_vals))
    print(f"  → mean={noise_mean:.4f}  std={noise_std:.4f}  "
          f"cv={noise_std/max(noise_mean,1e-8):.4f}", flush=True)

    results["noise"] = {
        "eta":    eta_min,
        "values": noise_vals,
        "mean":   noise_mean,
        "std":    noise_std,
        "cv":     noise_std / max(noise_mean, 1e-8),
    }

    # ── Stein gradient test at eta_min ────────────────────────────────────────
    # Directly test whether the gradient oracle can see a signal from eta_min
    # by evaluating at several sigma values. For each sigma, we also manually
    # compute the finite-difference gradient to check if the Stein estimate
    # agrees with the true numerical gradient.
    print(f"\n  Stein gradient test at eta=[{eta_min:.3f}, {eta_min:.3f}]:", flush=True)
    print(f"  {'sigma':<8} {'grad_mag':>10} {'fd_grad_mag':>12} {'agreement':>10}",
          flush=True)
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10}", flush=True)

    eta_vec_min  = np.array([eta_min, eta_min], dtype=float)
    a_at_min     = api.accuracy_callable(eta_vec_min)
    stein_results = []

    for test_sigma in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        # Temporarily override grad_sigma by patching the config
        original_sigma = api.config.grad_sigma
        # We call gradient_callable which uses the configured sigma internally.
        # To test different sigmas we call the Stein oracle directly.
        import sys
        sys.path.insert(0, _root)
        from src.core.toy_A import grad_oracle
        import torch

        def _f(e):
            acc = api._accuracy_for_eta(e, true=False)
            return torch.tensor(acc, dtype=torch.float32)

        eta_t = torch.tensor(eta_vec_min, dtype=torch.float32)
        g_stein = grad_oracle(_f, eta_t, sigma=test_sigma, N=grad_N)
        g_stein_np  = g_stein.detach().cpu().numpy()
        stein_mag   = float(np.linalg.norm(g_stein_np))

        # Finite-difference gradient at eta_min with step = test_sigma
        fd_grad = np.zeros(2)
        for dim in range(2):
            eta_plus = eta_vec_min.copy()
            eta_plus[dim] = min(1.0, eta_vec_min[dim] + test_sigma)
            a_plus = api.accuracy_callable(eta_plus)
            fd_grad[dim] = (a_plus - a_at_min) / test_sigma
        fd_mag = float(np.linalg.norm(fd_grad))

        # Agreement: cosine similarity between Stein and FD gradients
        if stein_mag > 1e-8 and fd_mag > 1e-8:
            cosine = float(np.dot(g_stein_np, fd_grad) / (stein_mag * fd_mag))
            agreement = f"{cosine:+.3f}"
        else:
            agreement = "N/A (zero)"

        print(f"  {test_sigma:<8.3f} {stein_mag:>10.6f} {fd_mag:>12.6f} {agreement:>10}",
              flush=True)
        stein_results.append({
            "sigma": test_sigma,
            "stein_grad": g_stein_np.tolist(),
            "stein_mag": stein_mag,
            "fd_grad": fd_grad.tolist(),
            "fd_mag": fd_mag,
            "agreement": agreement,
        })

    # Find smallest sigma where both Stein and FD give non-zero gradient
    useful = [(r["sigma"], r["stein_mag"], r["fd_mag"])
              for r in stein_results
              if r["stein_mag"] > 0.005 and r["fd_mag"] > 0.005]
    if useful:
        best_sigma, best_stein, best_fd = useful[0]
        print(f"\n  → Smallest sigma with detectable gradient: {best_sigma:.3f} "
              f"(Stein={best_stein:.6f}, FD={best_fd:.6f})", flush=True)
    else:
        print(f"\n  → No sigma produced detectable gradient at eta_min={eta_min:.3f}.",
              flush=True)
        print(f"     Gradient is flat at eta_min regardless of sigma.", flush=True)

    results["stein_test"] = stein_results
    fine_grid_results = []
    if kwargs.get("eta_fine_grid", False):
        fine_vals = list(np.linspace(eta_min, min(1.0, eta_min + 0.3), 20))
        print(f"\n  Fine grid [{eta_min:.3f} → {eta_min+0.3:.3f}] "
              f"(20 points) — gradient detection:", flush=True)
        print(f"  {'eta':<8} {'A_fast':>8} {'grad[0]':>10} {'grad[1]':>10} {'grad_mag':>10}",
              flush=True)
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}", flush=True)
        for eta_val in fine_vals:
            eta_vec  = np.array([eta_val, eta_val], dtype=float)
            a_fast   = api.accuracy_callable(eta_vec)
            grad     = api.gradient_callable(eta_vec)
            grad_mag = float(np.linalg.norm(grad))
            print(f"  {eta_val:.3f}    {a_fast:8.4f} {grad[0]:10.6f} {grad[1]:10.6f} {grad_mag:10.6f}",
                  flush=True)
            fine_grid_results.append({
                "eta": eta_val, "a_fast": float(a_fast),
                "grad": grad.tolist(), "grad_mag": grad_mag,
            })
        # Find first eta where gradient is meaningfully non-zero
        threshold = 0.01
        first_nonzero = next(
            (r for r in fine_grid_results if r["grad_mag"] > threshold), None
        )
        if first_nonzero:
            print(f"\n  First eta with grad_mag > {threshold}: "
                  f"eta={first_nonzero['eta']:.3f}  grad_mag={first_nonzero['grad_mag']:.6f}",
                  flush=True)
            print(f"  → Minimum useful sigma: {first_nonzero['eta'] - eta_min:.3f} "
                  f"(to reach this point from eta_min={eta_min})", flush=True)
        else:
            print(f"\n  No eta in grid has grad_mag > {threshold}. "
                  f"Gradient is flat throughout [{eta_min:.3f}, {eta_min+0.3:.3f}].", flush=True)

    results["fine_grid"] = fine_grid_results
    a_values = [r["a_true"] for r in results["eta_sweep"]]
    a_range  = max(a_values) - min(a_values)
    print(f"\n  Summary:", flush=True)
    print(f"    A_true range across eta grid: {min(a_values):.4f} → {max(a_values):.4f}  "
          f"(Δ = {a_range:.4f})", flush=True)
    print(f"    Noise std at eta_min:         {noise_std:.4f}", flush=True)

    if a_range < noise_std and noise_std > 0:
        print(f"    ⚠ WARNING: accuracy range ({a_range:.4f}) < noise std ({noise_std:.4f}).", flush=True)
        print(f"      The gradient oracle CANNOT reliably detect accuracy differences.", flush=True)
        print(f"      Increase fast_samples or choose a different strategy/eta_min.", flush=True)
    elif noise_std == 0.0:
        print(f"    ✓ OK: noise std = 0 (deterministic eval). "
              f"Accuracy range = {a_range:.4f}.", flush=True)
    elif a_range < 2 * noise_std:
        print(f"    ⚠ CAUTION: accuracy range is only {a_range/noise_std:.1f}× the noise std.", flush=True)
        print(f"      Gradient signal is weak. Consider increasing fast_samples.", flush=True)
    else:
        print(f"    ✓ OK: accuracy range is {a_range/noise_std:.1f}× the noise std.", flush=True)
        print(f"      Gradient signal should be detectable.", flush=True)

    results["summary"] = {
        "a_true_min":   min(a_values),
        "a_true_max":   max(a_values),
        "a_true_range": a_range,
        "noise_std":    noise_std,
        "snr":          a_range / max(noise_std, 1e-8),
        "verdict":      "OK" if a_range >= 2 * noise_std else
                        "CAUTION" if a_range >= noise_std else "WARNING",
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Diagnose LLM accuracy function before committing to full experiment."
    )
    p.add_argument(
        "--strategies", nargs="+",
        default=["quantization", "topk_per_token", "llmint8", "magnitude", "random"],
        help="Strategies to probe.",
    )
    p.add_argument(
        "--eta_values", nargs="+", type=float,
        default=[0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        help="eta values to evaluate (same value on both links).",
    )
    p.add_argument(
        "--eta_min", type=float, default=0.25,
        help="eta_min passed to the backend (clamp floor).",
    )
    p.add_argument(
        "--model_name", type=str,
        default=os.environ.get("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        help="HuggingFace model ID.",
    )
    p.add_argument(
        "--fast_samples", type=int,
        default=int(os.environ.get("MMLU_FAST_SAMPLES_PER_SUBJECT", "50")),
        help="MMLU samples per subject for fast (gradient oracle) eval.",
    )
    p.add_argument(
        "--full_samples", type=int,
        default=int(os.environ.get("MMLU_FULL_SAMPLES_PER_SUBJECT", "100")),
        help="MMLU samples per subject for full (plot) eval.",
    )
    p.add_argument(
        "--batch_size", type=int,
        default=int(os.environ.get("MMLU_BATCH_SIZE", "8")),
    )
    p.add_argument(
        "--grad_sigma", type=float,
        default=float(os.environ.get("LLM_GRAD_SIGMA", "0.05")),
        help="Stein oracle sigma.",
    )
    p.add_argument(
        "--grad_N", type=int,
        default=int(os.environ.get("LLM_GRAD_N", "5")),
        help="Stein oracle N (number of perturbations).",
    )
    p.add_argument(
        "--eta_fine_grid", action="store_true", default=False,
        help="In addition to --eta_values, evaluate a fine grid of 20 points "
             "between eta_min and eta_min+0.3 to locate where gradient becomes non-zero.",
    )
    p.add_argument(
        "--noise_repeats", type=int, default=3,
        help="Number of repeated fast evaluations at eta_min to estimate noise.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
    )
    p.add_argument(
        "--out_dir", type=str, default="outputs/diagnosis",
        help="Directory to save diagnosis_results.json.",
    )

    args = p.parse_args()

    print("=" * 65, flush=True)
    print("diagnose_llm_accuracy.py", flush=True)
    print(f"  model          = {args.model_name}", flush=True)
    print(f"  strategies     = {args.strategies}", flush=True)
    print(f"  eta grid       = {args.eta_values}", flush=True)
    print(f"  eta_min        = {args.eta_min}", flush=True)
    print(f"  fast_samples   = {args.fast_samples}/subj", flush=True)
    print(f"  full_samples   = {args.full_samples}/subj", flush=True)
    print(f"  grad_sigma     = {args.grad_sigma}  grad_N = {args.grad_N}", flush=True)
    print(f"  noise_repeats  = {args.noise_repeats}", flush=True)
    print("=" * 65, flush=True)

    all_results = []
    for strategy in args.strategies:
        result = probe_strategy(
            strategy      = strategy,
            model_name    = args.model_name,
            eta_values    = args.eta_values,
            eta_min       = args.eta_min,
            fast_samples  = args.fast_samples,
            full_samples  = args.full_samples,
            batch_size    = args.batch_size,
            grad_sigma    = args.grad_sigma,
            grad_N        = args.grad_N,
            noise_repeats = args.noise_repeats,
            seed          = args.seed,
            eta_fine_grid = args.eta_fine_grid,
        )
        all_results.append(result)

    # ── Final cross-strategy summary ─────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Strategy':<20} {'A(eta_min)':>10} {'A(1.0)':>8} "
          f"{'Range':>8} {'Noise':>8} {'SNR':>6} {'Verdict'}", flush=True)
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*10}", flush=True)

    for r in all_results:
        sweep  = r["eta_sweep"]
        a_min  = next((s["a_true"] for s in sweep if abs(s["eta"] - r["eta_min"]) < 1e-6),
                      sweep[0]["a_true"])
        a_max  = sweep[-1]["a_true"]   # last entry = eta=1.0 (no compression)
        rng    = r["summary"]["a_true_range"]
        noise  = r["summary"]["noise_std"]
        snr    = r["summary"]["snr"]
        verdict= r["summary"]["verdict"]
        print(f"  {r['strategy']:<20} {a_min:>10.4f} {a_max:>8.4f} "
              f"{rng:>8.4f} {noise:>8.4f} {snr:>6.2f} {verdict}", flush=True)

    print(f"\n  Interpretation:", flush=True)
    print(f"    SNR ≥ 2 → OK:      gradient oracle can detect accuracy differences", flush=True)
    print(f"    SNR 1-2 → CAUTION: signal is weak, increase fast_samples", flush=True)
    print(f"    SNR < 1 → WARNING: noise exceeds signal, optimizer is blind", flush=True)

    # Save JSON
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "diagnosis_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
