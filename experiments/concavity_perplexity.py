"""
Concavity test for a perplexity-based metric across local HF LLMs.

For each model we install activation-compression hooks at two cut-points
(matching the optimizer's split layout), sweep eta in [eta_min, 1] along
the diagonal eta_0 = eta_1 = eta, and measure WikiText-2 perplexity.

Metric:
    rel_ppl(eta) = PPL(eta) / PPL_baseline      # PPL_baseline = PPL at eta=1 (uncompressed)
    score(eta)   = - log( rel_ppl(eta) )        # = log(PPL_base) - log(PPL(eta))

score is the concave-friendly transform (higher = better, 0 at baseline, negative below).

We then check concavity two ways:
  (1) discrete second differences along the eta grid
  (2) random chord / midpoint test: f((a+b)/2) >= 0.5*(f(a)+f(b)) - tol

Outputs (per model) under experiments/results/:
  - {model_slug}_sweep.csv
  - {model_slug}_summary.json
  - {model_slug}_curve.png
  - all_summary.json  (aggregate)

Example:
  python experiments/concavity_perplexity.py \
      --models gpt2 google/gemma-2b \
      --strategy topk_per_token \
      --etas 0.25 0.4 0.55 0.7 0.85 1.0 \
      --max_texts 64 --max_length 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

# Allow running as a top-level script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.llm_compression import (  # noqa: E402
    ActivationCompressionConfig,
    LLMActivationCompressor,
    detect_llm_decoder_layers,
)


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────

def load_wikitext_texts(max_texts: int, min_chars: int = 200) -> List[str]:
    """Load WikiText-2 test split and return up to `max_texts` non-trivial paragraphs."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts: List[str] = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            texts.append(t)
            if len(texts) >= max_texts:
                break
    if not texts:
        raise RuntimeError("No suitable WikiText-2 paragraphs found.")
    return texts


# ──────────────────────────────────────────────────────────────────────────
# Perplexity
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def corpus_perplexity(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int = 1,
) -> tuple[float, float]:
    """Returns (perplexity, mean_log_likelihood_per_token).

    Mean log-likelihood is in nats per token.  PPL = exp(-mean_LL).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # (B, S, V)

        # Shift for next-token loss.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().to(torch.float32)

        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        tgt_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        # Mask out padding positions.
        tgt_logp = tgt_logp * shift_mask

        nll = -tgt_logp.sum().item()
        n_tok = shift_mask.sum().item()
        total_nll += float(nll)
        total_tokens += float(n_tok)

    mean_ll = -total_nll / max(total_tokens, 1.0)  # nats per token
    ppl = math.exp(-mean_ll)
    return ppl, mean_ll


# ──────────────────────────────────────────────────────────────────────────
# Sweep + concavity
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    eta: float
    ppl: float
    rel_ppl: float  # PPL(eta) / PPL_baseline
    score: float    # -log(rel_ppl) = log(PPL_base) - log(PPL(eta))
    seconds: float


def _eval_ppl_at(
    eta_scalar: float, *, model, tokenizer, compressor, texts,
    device, max_length, batch_size,
) -> tuple[float, float, float]:
    """Returns (perplexity, mean_log_likelihood_per_token, seconds_elapsed)."""
    compressor.set_eta(torch.tensor([eta_scalar, eta_scalar], dtype=torch.float32))
    t0 = time.time()
    ppl, mean_ll = corpus_perplexity(
        model, tokenizer, texts,
        device=device, max_length=max_length, batch_size=batch_size,
    )
    return float(ppl), float(mean_ll), time.time() - t0


def run_sweep(
    *,
    model,
    tokenizer,
    compressor: LLMActivationCompressor,
    texts: Sequence[str],
    etas: Sequence[float],
    baseline_ppl: float,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[SweepPoint]:
    pts: List[SweepPoint] = []
    log_base = math.log(baseline_ppl)
    for eta_scalar in etas:
        ppl, _, secs = _eval_ppl_at(
            eta_scalar, model=model, tokenizer=tokenizer, compressor=compressor,
            texts=texts, device=device, max_length=max_length, batch_size=batch_size,
        )
        rel_ppl = ppl / baseline_ppl
        score = log_base - math.log(ppl)  # = -log(rel_ppl)
        pts.append(SweepPoint(
            eta=float(eta_scalar), ppl=ppl, rel_ppl=rel_ppl,
            score=score, seconds=secs,
        ))
        print(f"    eta={eta_scalar:.3f}  ppl={ppl:.3f}  rel_ppl={rel_ppl:.4f}  "
              f"score={score:+.4f}  ({secs:.1f}s)", flush=True)
    return pts


def second_difference_test(pts: List[SweepPoint]) -> dict:
    """Concavity on a non-uniform grid: a function is concave iff the slopes
    (f(x_{i+1}) - f(x_i)) / (x_{i+1} - x_i) are non-increasing in i.
    We count violations and return the worst margin.
    """
    xs = [p.eta for p in pts]
    ys = [p.score for p in pts]
    slopes = []
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        slopes.append((ys[i + 1] - ys[i]) / dx if dx != 0 else 0.0)

    violations = 0
    worst = 0.0  # slope_next - slope_prev; should be <= 0 for concavity
    for i in range(len(slopes) - 1):
        diff = slopes[i + 1] - slopes[i]
        if diff > worst:
            worst = diff
        if diff > 0:
            violations += 1

    return {
        "n_pairs": max(0, len(slopes) - 1),
        "violations": violations,
        "worst_slope_increase": worst,  # > 0 means non-concave between those points
        "slopes": slopes,
    }


def random_chord_test(
    *, model, tokenizer, compressor, texts,
    baseline_ppl: float,
    device, max_length, batch_size,
    eta_min: float, eta_max: float = 1.0,
    n_chords: int = 6, tol: float = 1e-3, rng_seed: int = 0,
) -> dict:
    """For each random pair (a, b) in [eta_min, eta_max], evaluate the
    baseline-relative score at a, b, midpoint m=(a+b)/2.
        score(eta) = -log( PPL(eta) / PPL_baseline )
    Concave  =>  score(m) >= 0.5*(score(a)+score(b)) - tol.
    """
    rng = np.random.default_rng(rng_seed)
    log_base = math.log(baseline_ppl)
    results = []
    passes = 0
    for _ in range(n_chords):
        a, b = rng.uniform(eta_min, eta_max, size=2)
        a, b = sorted([float(a), float(b)])
        if b - a < 0.02:
            b = min(eta_max, a + 0.05)
        m = 0.5 * (a + b)

        scores = {}
        for tag, eta_val in [("a", a), ("b", b), ("m", m)]:
            ppl, _, _ = _eval_ppl_at(
                eta_val, model=model, tokenizer=tokenizer, compressor=compressor,
                texts=texts, device=device, max_length=max_length, batch_size=batch_size,
            )
            scores[tag] = log_base - math.log(ppl)

        chord = 0.5 * (scores["a"] + scores["b"])
        margin = scores["m"] - chord  # >= -tol means concave
        ok = margin >= -tol
        passes += int(ok)
        results.append({"a": a, "b": b, "m": m, **scores,
                        "chord": chord, "margin": margin, "concave": ok})
        print(f"    chord a={a:.3f} b={b:.3f} m={m:.3f}  margin={margin:+.4f}  "
              f"{'OK' if ok else 'VIOLATION'}", flush=True)
    return {"n_chords": n_chords, "passes": passes, "tol": tol, "results": results}


# ──────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────

def save_curve_plot(pts: List[SweepPoint], path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip plot: {e})")
        return

    xs = [p.eta for p in pts]
    ys = [p.score for p in pts]
    rel = [p.rel_ppl for p in pts]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, ys, marker="o", color="tab:blue", label="score = -log(PPL/PPL_base)")
    ax1.axhline(0.0, color="tab:blue", linestyle=":", alpha=0.4)
    ax1.set_xlabel("eta (compression ratio)")
    ax1.set_ylabel("score (-log rel_ppl)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(xs, rel, marker="x", color="tab:red", alpha=0.6, label="rel_ppl")
    ax2.axhline(1.0, color="tab:red", linestyle=":", alpha=0.4)
    ax2.set_ylabel("relative perplexity (PPL/PPL_base)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Per-model driver
# ──────────────────────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    *,
    texts: Sequence[str],
    etas: Sequence[float],
    strategy: str,
    split_mode: str,
    max_length: int,
    batch_size: int,
    n_chords: int,
    out_dir: Path,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== {model_name} ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    layers = detect_llm_decoder_layers(model)
    n_layers = len(layers)
    if split_mode == "thirds":
        cut1 = n_layers // 3
        cut2 = (2 * n_layers) // 3
    else:  # half
        cut1 = n_layers // 2
        cut2 = min(n_layers - 1, cut1 + 1)
    cut1 = int(np.clip(cut1, 0, n_layers - 1))
    cut2 = int(np.clip(cut2, 0, n_layers - 1))
    print(f"  layers={n_layers}, cuts=({cut1},{cut2}), strategy={strategy}", flush=True)

    compressor = LLMActivationCompressor(
        model,
        ActivationCompressionConfig(
            strategy=strategy, layer_indices=[cut1, cut2], random_seed=0,
        ),
    )

    try:
        print("  -- baseline (eta=1.0, uncompressed) --", flush=True)
        baseline_ppl, baseline_ll, baseline_secs = _eval_ppl_at(
            1.0, model=model, tokenizer=tokenizer, compressor=compressor,
            texts=texts, device=device, max_length=max_length, batch_size=batch_size,
        )
        print(f"    baseline_ppl={baseline_ppl:.3f}  baseline_LL={baseline_ll:+.4f}  "
              f"({baseline_secs:.1f}s)", flush=True)

        print("  -- sweep --", flush=True)
        pts = run_sweep(
            model=model, tokenizer=tokenizer, compressor=compressor,
            texts=texts, etas=etas, baseline_ppl=baseline_ppl, device=device,
            max_length=max_length, batch_size=batch_size,
        )

        print("  -- second-difference concavity test --", flush=True)
        sd = second_difference_test(pts)
        print(f"    violations={sd['violations']}/{sd['n_pairs']}  "
              f"worst_slope_increase={sd['worst_slope_increase']:+.4f}", flush=True)

        if n_chords > 0:
            print("  -- random chord (midpoint) test --", flush=True)
            chord = random_chord_test(
                model=model, tokenizer=tokenizer, compressor=compressor,
                texts=texts, baseline_ppl=baseline_ppl, device=device,
                max_length=max_length, batch_size=batch_size,
                eta_min=min(etas), eta_max=max(etas),
                n_chords=n_chords,
            )
        else:
            chord = {"n_chords": 0, "passes": 0, "results": []}
    finally:
        compressor.remove_hooks()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    slug = model_name.replace("/", "__")

    # CSV
    csv_path = out_dir / f"{slug}_sweep.csv"
    with csv_path.open("w") as f:
        f.write("eta,perplexity,rel_ppl,score_neg_log_rel_ppl,seconds\n")
        for p in pts:
            f.write(f"{p.eta},{p.ppl},{p.rel_ppl},{p.score},{p.seconds}\n")

    # JSON
    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "cuts": [cut1, cut2],
        "strategy": strategy,
        "split_mode": split_mode,
        "etas": list(etas),
        "baseline_ppl": baseline_ppl,
        "baseline_mean_ll": baseline_ll,
        "sweep": [asdict(p) for p in pts],
        "second_difference": sd,
        "chord": chord,
    }
    (out_dir / f"{slug}_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    save_curve_plot(pts, out_dir / f"{slug}_curve.png",
                    title=f"{model_name}  ({strategy}, cuts={cut1},{cut2})")

    return summary


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

LOCAL_DEFAULT_MODELS = [
    "gpt2",
    "google/gemma-2b",
    "google/gemma-7b",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-3.1-8B",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=LOCAL_DEFAULT_MODELS,
                   help="HF model ids (must be in your HF cache).")
    p.add_argument("--strategy", default="topk_per_token",
                   choices=["topk_per_token", "magnitude", "random", "quantization"])
    p.add_argument("--split_mode", default="thirds", choices=["thirds", "half"])
    p.add_argument("--etas", nargs="+", type=float,
                   default=[0.25, 0.40, 0.55, 0.70, 0.85, 1.00])
    p.add_argument("--max_texts", type=int, default=64,
                   help="WikiText-2 paragraphs used per evaluation.")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_chords", type=int, default=4,
                   help="Random midpoint-chord tests per model (0 to skip).")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "experiments" / "results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    etas = sorted(set(float(e) for e in args.etas))
    print(f"Using etas: {etas}")
    print(f"Loading WikiText-2 ({args.max_texts} paragraphs, max_length={args.max_length})…")
    texts = load_wikitext_texts(args.max_texts)
    print(f"  -> {len(texts)} paragraphs.")

    aggregate = {"args": vars(args), "models": {}}
    for m in args.models:
        try:
            summary = run_model(
                m,
                texts=texts, etas=etas,
                strategy=args.strategy, split_mode=args.split_mode,
                max_length=args.max_length, batch_size=args.batch_size,
                n_chords=args.n_chords, out_dir=out_dir,
            )
            aggregate["models"][m] = {
                "baseline_ppl": summary["baseline_ppl"],
                "violations": summary["second_difference"]["violations"],
                "n_pairs": summary["second_difference"]["n_pairs"],
                "worst_slope_increase": summary["second_difference"]["worst_slope_increase"],
                "chord_passes": summary["chord"]["passes"],
                "n_chords": summary["chord"]["n_chords"],
                "scores": [p["score"] for p in summary["sweep"]],
                "rel_ppls": [p["rel_ppl"] for p in summary["sweep"]],
                "ppls": [p["ppl"] for p in summary["sweep"]],
            }
        except Exception as e:
            print(f"  !! {m} failed: {type(e).__name__}: {e}", flush=True)
            aggregate["models"][m] = {"error": f"{type(e).__name__}: {e}"}

    (out_dir / "all_summary.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\nDone. Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
