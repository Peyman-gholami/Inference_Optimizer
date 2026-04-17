Phase 1: ResNet56 + CIFAR10 + topk

This directory is a standalone copy of `Inference_Optimizer-main-peyman` for
Phase 1 integration only. The original directory is untouched.

What was added:

- `src/core/task_instances/resnet56_cifar10_topk.py`
  - registers a new Peyman task family
  - uses real ResNet56+CIFAR10 accuracy evaluation
  - uses Peyman's `CompressedModel`, `AccuracyFunction`, and Stein gradient path
- `run_phase1_resnet_compare.py`
  - convenience launcher that derives `c_min/c_max` from the copied 15 Mbps trace
- `assets/resnet56-4bfd9763.th`
  - ResNet56 checkpoint
- `assets/resnet56_cifar10_15Mbps_scenario_trace_topk.json`
  - copied 15 Mbps scenario trace used to derive fixed `tau` and `a`
- `assets/resnet56_cifar10_15Mbps_trace_summary_topk.json`
  - copied summary reference
- `assets/raw_accuracy_resnet56_llmint8_eta_mapping.json`
  - local llmint8 eta-to-codec mapping so llmint8 mode does not depend on the parent workspace
- `compressors.py`
  - local activation compressor implementations used by the ResNet task instance
- `resnet20.py`
  - copied locally so this Phase 1 directory is self-contained

Task registration:

- task name: `resnet56_cifar10_topk`

Recommended run:

```powershell
python run_phase1_resnet_compare.py --R 5 --mu 6 --epsilon 0.1 --T 100
```

Direct Peyman script run:

```powershell
python compare_to_baselines.py --task resnet56_cifar10_topk --M 4 --R 5 --mu 6 --epsilon 0.1 --c_min 1500000 --c_max 1500000
```

Notes:

- `tau` and `a` are derived from medians of the copied 15 Mbps scenario trace.
- `run_phase1_resnet_compare.py` directly replays `trace.channel_true_bps` from the copied scenario.
- The direct `compare_to_baselines.py` command above is only a fallback path; it does not replay the trace.
- Default Phase 1 Stein settings:
  - `RESNET56_PHASE1_FAST_SAMPLES=100`
  - `RESNET56_PHASE1_STEIN_N=5`
  - `RESNET56_PHASE1_SIGMA=0.05`
- CIFAR10 is expected under `data/` in this directory unless overridden with:
  - `RESNET56_PHASE1_DATA_ROOT`
- You can override defaults with environment variables:
  - `RESNET56_PHASE1_DEVICE`
  - `RESNET56_PHASE1_CHECKPOINT`
  - `RESNET56_PHASE1_TRACE_PATH`
  - `RESNET56_PHASE1_FAST_SAMPLES`
  - `RESNET56_PHASE1_SIGMA`
  - `RESNET56_PHASE1_STEIN_N`
  - `RESNET56_PHASE1_ETA_MIN`
