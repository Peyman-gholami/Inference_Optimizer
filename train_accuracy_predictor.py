"""
train_accuracy_predictor.py
===========================
Fits an AccuracyPredictor surrogate for a given LLM strategy and saves it.

This is a one-time offline cost per (model, strategy) combination.
Once saved, PredictorGradientOracle can replace SteinGradientOracle
in any experiment — gradient computation drops from ~385s/slot to <1s.

Usage
-----
  # Fit for Llama + topk_per_token:
  python train_accuracy_predictor.py \\
      --model_name meta-llama/Llama-3.1-8B-Instruct \\
      --strategy   topk_per_token \\
      --eta_min    0.25 \\
      --n_grid     15 \\
      --n_repeat   1 \\
      --save_path  models/predictor_llama_topk.pt

  # Fit for Llama + quantization (larger grid for discrete function):
  python train_accuracy_predictor.py \\
      --model_name meta-llama/Llama-3.1-8B-Instruct \\
      --strategy   quantization \\
      --eta_min    0.25 \\
      --n_grid     20 \\
      --epochs     3000 \\
      --save_path  models/predictor_llama_quant.pt

Grid size guide (num_links=2):
  n_grid=10 →  100 evals  → ~2.1h  (quick test)
  n_grid=15 →  225 evals  → ~4.8h
  n_grid=20 →  400 evals  → ~8.6h  (use --n_grid 20 for quantization)

Slurm: use a 10h job for n_grid=20, 6h for n_grid=15.
"""

import argparse
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src", "core"))

from gradient_oracles import AccuracyPredictor


def main():
    p = argparse.ArgumentParser(
        description="Fit and save AccuracyPredictor surrogate for one strategy."
    )
    p.add_argument("--model_name",  type=str, required=True)
    p.add_argument("--strategy",    type=str, required=True,
                   choices=["topk_per_token", "magnitude", "random",
                             "quantization", "llmint8"])
    p.add_argument("--eta_min",     type=float, default=0.25,
                   help="eta_min used by the optimizer (clamp floor). "
                        "Does NOT control the training grid range — "
                        "use --grid_eta_min for that.")
    p.add_argument("--grid_eta_min", type=float, default=0.0,
                   help="Lower bound of the training grid (default 0.0). "
                        "Using 0.0 means the predictor covers [0,1] and can "
                        "be reused with any eta_min value at experiment time.")
    p.add_argument("--n_grid",      type=int,   default=15,
                   help="Grid points per link (total evals = n_grid^num_links).")
    p.add_argument("--n_repeat",    type=int,   default=1,
                   help="Repeated evals per grid point (1 for deterministic).")
    p.add_argument("--hidden",      type=int,   default=32)
    p.add_argument("--depth",       type=int,   default=3)
    p.add_argument("--epochs",      type=int,   default=2000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--mono_weight", type=float, default=0.1,
                   help="Monotonicity regularisation weight.")
    p.add_argument("--save_path",   type=str,   default=None,
                   help="Path to save the fitted predictor (.pt file).")
    p.add_argument("--fast_samples",type=int,   default=50)
    p.add_argument("--full_samples",type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--n_shot",      type=int,   default=0,
                   help="Number of few-shot examples (0=zero-shot, 5=standard 5-shot). "
                        "Must match MMLU_N_SHOT used at experiment time.")
    args = p.parse_args()

    # ── Set env vars for backend ──────────────────────────────────────────────
    import os as _os
    _os.environ["LLM_MODEL_NAME"]               = args.model_name
    _os.environ["LLM_ACTIVATION_STRATEGY"]      = args.strategy
    _os.environ["LLM_SPLIT_MODE"]               = "thirds"
    _os.environ["MMLU_FAST_SAMPLES_PER_SUBJECT"] = str(args.fast_samples)
    _os.environ["MMLU_FULL_SAMPLES_PER_SUBJECT"] = str(args.full_samples)
    _os.environ["MMLU_BATCH_SIZE"]              = str(args.batch_size)
    _os.environ["MMLU_SEED"]                    = str(args.seed)
    _os.environ["MMLU_N_SHOT"]                  = str(args.n_shot)
    _os.environ["INFERENCE_OPTIMIZER_BACKEND"]  = "llm"
    _os.environ.setdefault("LLM_USE_MEASURED_TAU_A", "1")

    # ── Load backend ──────────────────────────────────────────────────────────
    from src.core.llm_task_callables import build_llm_backend_from_env
    print(f"[train_accuracy_predictor] Loading backend ...", flush=True)
    print(f"  model    = {args.model_name}", flush=True)
    print(f"  strategy = {args.strategy}", flush=True)
    print(f"  eta_min      = {args.eta_min}  (optimizer clamp floor)", flush=True)
    print(f"  grid_eta_min = {args.grid_eta_min}  (training grid lower bound)", flush=True)
    print(f"  n_grid   = {args.n_grid}  (total evals = {args.n_grid**2} "
          f"× {args.n_repeat} = {args.n_grid**2 * args.n_repeat})", flush=True)
    api = build_llm_backend_from_env(eta_min=args.eta_min)

    # ── Default save path ─────────────────────────────────────────────────────
    if args.save_path is None:
        model_short  = args.model_name.split("/")[-1].replace("-", "_").lower()
        args.save_path = (f"models/predictor_{model_short}_{args.strategy}"
                          f"_etamin{args.eta_min}_grid{args.n_grid}"
                          f"_{args.n_shot}shot.pt")
    os.makedirs(os.path.dirname(args.save_path)
                if os.path.dirname(args.save_path) else ".",
                exist_ok=True)

    # ── Fit predictor ─────────────────────────────────────────────────────────
    predictor = AccuracyPredictor(
        num_links = 2,
        eta_min   = args.eta_min,
        hidden    = args.hidden,
        depth     = args.depth,
    )
    predictor.fit(
        accuracy_callable_true = api.accuracy_callable_true,
        n_grid       = args.n_grid,
        n_repeat     = args.n_repeat,
        lr           = args.lr,
        epochs       = args.epochs,
        mono_weight  = args.mono_weight,
        save_path    = args.save_path,
        verbose      = True,
        grid_eta_min = args.grid_eta_min,
    )

    # ── Held-out evaluation ───────────────────────────────────────────────────
    # Evaluate on a separate held-out grid (different points from training grid)
    # to get unbiased estimates of RMSE, R², and MAE.
    print("\n" + "="*60, flush=True)
    print("[validation] Held-out evaluation on fresh grid points", flush=True)
    print("="*60, flush=True)

    # Build held-out grid at n_grid+1 points (offset so they fall between
    # training points and are never seen during fitting)
    n_held = max(8, args.n_grid - 2)
    held_lo = args.grid_eta_min
    axes_held = [np.linspace(held_lo + 0.5/(n_held-1), 1.0 - 0.5/(n_held-1),
                             n_held)
                 for _ in range(2)]
    grids_held  = np.meshgrid(*axes_held, indexing="ij")
    eta_held    = np.stack([g.flatten() for g in grids_held], axis=1)

    print(f"  Evaluating {len(eta_held)} held-out points ...", flush=True)

    a_true_vals = []
    a_pred_vals = []
    for eta_vec in eta_held:
        a_true_vals.append(float(api.accuracy_callable_true(eta_vec)))
        a_pred_vals.append(float(predictor.predict(eta_vec)))

    a_true_arr = np.array(a_true_vals)
    a_pred_arr = np.array(a_pred_vals)

    residuals   = a_pred_arr - a_true_arr
    rmse        = float(np.sqrt(np.mean(residuals**2)))
    mae         = float(np.mean(np.abs(residuals)))
    ss_res      = float(np.sum(residuals**2))
    ss_tot      = float(np.sum((a_true_arr - a_true_arr.mean())**2))
    r2          = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    max_err     = float(np.max(np.abs(residuals)))
    bias        = float(np.mean(residuals))   # signed: + = over-predicts

    print(f"\n  Held-out metrics ({len(eta_held)} points):", flush=True)
    print(f"    RMSE     = {rmse:.6f}", flush=True)
    print(f"    MAE      = {mae:.6f}", flush=True)
    print(f"    Max err  = {max_err:.6f}", flush=True)
    print(f"    Bias     = {bias:+.6f}  (+ = over-predicts, - = under-predicts)", flush=True)
    print(f"    R²       = {r2:.6f}  (1.0 = perfect fit)", flush=True)

    # Accuracy range on held-out set (sanity check of landscape coverage)
    a_range = float(a_true_arr.max() - a_true_arr.min())
    print(f"    A_true range = [{a_true_arr.min():.4f}, {a_true_arr.max():.4f}]"
          f"  (range={a_range:.4f})", flush=True)
    print(f"    A_pred range = [{a_pred_arr.min():.4f}, {a_pred_arr.max():.4f}]",
          flush=True)

    # Relative RMSE as fraction of accuracy range (most interpretable metric)
    if a_range > 1e-6:
        rel_rmse = rmse / a_range
        print(f"    Rel. RMSE    = {rel_rmse:.3f}  "
              f"(RMSE / A_range, lower is better)", flush=True)

    # Monotonicity check: fraction of held-out pairs where predictor
    # respects the correct ordering (A(η_high) >= A(η_low))
    print(f"\n  Monotonicity check:", flush=True)
    n_pairs   = 0
    n_correct = 0
    rng_check = np.random.default_rng(42)
    sample_idx = rng_check.choice(len(eta_held),
                                   size=min(500, len(eta_held)),
                                   replace=False)
    for i in sample_idx:
        for j in sample_idx:
            if i >= j:
                continue
            # Check if eta_held[i] <= eta_held[j] componentwise
            if np.all(eta_held[i] <= eta_held[j]) and not np.all(eta_held[i] == eta_held[j]):
                n_pairs += 1
                if a_pred_arr[i] <= a_pred_arr[j]:
                    n_correct += 1
    if n_pairs > 0:
        mono_frac = n_correct / n_pairs
        print(f"    {n_correct}/{n_pairs} ordered pairs correct  "
              f"({mono_frac*100:.1f}%)", flush=True)
    else:
        print("    (no ordered pairs found in sample)", flush=True)

    # Per-point breakdown at key eta values
    print(f"\n  Point-by-point at key eta values:", flush=True)
    key_etas = [
        np.array([args.grid_eta_min, args.grid_eta_min]),
        np.array([0.25, 0.25]),
        np.array([0.5, 0.5]),
        np.array([0.75, 0.75]),
        np.array([1.0, 1.0]),
    ]
    for eta_vec in key_etas:
        eta_vec = np.clip(eta_vec, args.grid_eta_min, 1.0)
        a_t = float(api.accuracy_callable_true(eta_vec))
        a_p = float(predictor.predict(eta_vec))
        g   = predictor.gradient(eta_vec)
        err = a_p - a_t
        print(f"    eta={np.round(eta_vec, 2)}  "
              f"A_true={a_t:.4f}  A_pred={a_p:.4f}  "
              f"err={err:+.4f}  grad={np.round(g, 4)}", flush=True)

    print(f"\nPredictor saved to: {args.save_path}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
