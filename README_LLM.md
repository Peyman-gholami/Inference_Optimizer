# LLM / MMLU task backend (plug-in)

The default demos use MNIST + `InferenceTaskCallables`. To swap in **compressed-activation MMLU** as the accuracy/gradient oracle, set one environment variable; everything else (`main.py`, `compare_to_baselines_multi.py`, optimizers) stays the same.

## Quick start

Install extras (see `requirements.txt`: `transformers`, `datasets`, `accelerate`), then:

```bash
export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=google/gemma-2b-it   # or another causal LM on Hugging Face

python main.py
```

For a shorter run, reduce MMLU size:

```bash
export MMLU_FAST_SAMPLES_PER_SUBJECT=2
export MMLU_FULL_SAMPLES_PER_SUBJECT=4
export MMLU_BATCH_SIZE=2
python main.py
```

## Multi-task baseline comparison

```bash
export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_MODEL_NAME=google/gemma-2b-it

python compare_to_baselines_multi.py --T 30 --K 2 --baseline_algorithms none
```

(Add baselines back when runtime allows; LLM eval per optimizer step is heavy.)

## Measured \(\tau\) and link payloads \(a\)

To use the bundled **Gemma-2B / Gemma-7B** tables in `src/core/measured_llm_profile.py` instead of toy \(\tau, a\):

```bash
export INFERENCE_OPTIMIZER_BACKEND=llm
export LLM_USE_MEASURED_TAU_A=1
export LLM_MEASURED_MODEL=gemma_2b    # or gemma_7b
export LLM_MEASURED_BATCH=1           # 1, 2, or 4
export LLM_MEASURED_PAYLOAD=prefill_raw   # or prefill_serial, decode_raw, decode_serial
```

If `LLM_MEASURED_MODEL` is omitted, the model is inferred from `LLM_MODEL_NAME` when possible.

You may need to tune channel bounds and \(R_k\) so the problem stays feasible, e.g.:

```bash
export R_K=5
python compare_to_baselines_multi.py --c_min 1e5 --c_max 5e6 --T 20
```

(Interpret `c_t` in the same units as `a` in the tables, e.g. bytes/s if \(a\) is in bytes.)

## Main environment variables

| Variable | Role |
|----------|------|
| `INFERENCE_OPTIMIZER_BACKEND` | `mnist` (default) or `llm` |
| `LLM_MODEL_NAME` | Hugging Face model id |
| `LLM_ACTIVATION_STRATEGY` | e.g. `topk_per_token`, `magnitude`, … |
| `LLM_SPLIT_MODE` | `thirds` or `half` (where activations are compressed) |
| `ETA_MIN0`, `ETA_MIN1` | Per-link minimum compression; default `0.25` in LLM mode |
| `R_K` / `MEASURED_R_K` | Target throughput \(R_k\) when using a constant schedule |
| `LLM_USE_MEASURED_TAU_A` | `1` / `true` → use measured \(\tau, a\) |
| `MMLU_*` | Subjects, sample counts, batch size, seed (see `build_llm_backend_from_env` in `src/core/llm_task_callables.py`) |

## Code entry points

- `src/core/llm_plug_in.py` — env switches shared by `main.py` and comparisons
- `src/core/llm_task_callables.py` — `LLMInferenceTaskCallables` + `build_llm_backend_from_env`
- `src/core/measured_llm_profile.py` — table-driven \(\tau, a\)
