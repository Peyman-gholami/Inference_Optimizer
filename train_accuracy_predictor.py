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
                          f"_etamin{args.eta_min}_grid{args.n_grid}.pt")
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

    # ── Quick validation ──────────────────────────────────────────────────────
    import numpy as np
    print("\n[validation] Predictor vs A_true at a few test points:", flush=True)
    test_etas = [
        np.array([args.eta_min, args.eta_min]),
        np.array([0.5, 0.5]),
        np.array([0.75, 0.75]),
        np.array([1.0, 1.0]),
    ]
    for eta_vec in test_etas:
        eta_vec = np.clip(eta_vec, args.eta_min, 1.0)
        a_true  = api.accuracy_callable_true(eta_vec)
        a_pred  = predictor.predict(eta_vec)
        g       = predictor.gradient(eta_vec)
        print(f"  eta={np.round(eta_vec, 2)}  "
              f"A_true={a_true:.4f}  A_pred={a_pred:.4f}  "
              f"grad={np.round(g, 4)}", flush=True)

    print(f"\nPredictor saved to: {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
