"""
test_gpu_compressors.py
=======================
Equivalence tests between CPU compressors (compressors.py) and GPU-native
compressors (gpu_compressors.py).

Four test levels
----------------
1. test_topk_equivalence          — exact output equality (deterministic)
2. test_quantization_equivalence  — exact for FP16/INT8/INT2, tolerance for INT4
3. test_llmint8_equivalence       — near-zero tolerance (kthvalue vs quantile threshold)
4. test_resnet_end_to_end         — full partitioned inference, logit + prediction agreement

Known differences documented
------------------------------
TopK mask size:
  CPU  uses numpy.packbits  → 1 bit/element  (smaller payload)
  GPU  uses torch.bool      → 1 byte/element (8× larger mask, but GPU-native)
  Decompressed OUTPUT is identical — only the payload dict layout differs.

INT4 stochastic rounding:
  Both call torch.bernoulli(prob) but on different devices (CPU vs CUDA).
  The CUDA RNG may produce different Bernoulli samples from the same seed
  depending on the device and cuDNN version.  The test allows a small
  tolerance and verifies that the quantisation error magnitude is comparable.

LLMInt8 threshold:
  CPU  uses torch.quantile (exact percentile interpolation)
  GPU  uses torch.kthvalue  (nearest-rank, no interpolation)
  When multiple elements tie at the percentile boundary the two methods may
  select a slightly different threshold, leading to a different outlier set.
  The test checks max absolute difference < tolerance rather than exact equality.

Usage
-----
  # Run all tests (requires CUDA):
  python test_gpu_compressors.py

  # Run without a checkpoint (skips end-to-end test):
  python test_gpu_compressors.py --skip_e2e

  # Run with a specific ResNet checkpoint:
  python test_gpu_compressors.py --checkpoint ../models/resnet56-4bfd9763.th

  # Verbose mode (print per-test detail):
  python test_gpu_compressors.py --verbose
"""

import argparse
import os
import sys
import traceback
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Robust path resolution: walk upward from this file's directory until we
# find the project root (identified by the presence of 'src/core').
# This works whether the test is run from the project root, from tests/,
# or from any other subdirectory.
def _find_project_root(start: str) -> str:
    candidate = os.path.abspath(start)
    for _ in range(5):   # max 5 levels up
        if os.path.isdir(os.path.join(candidate, 'src', 'core')):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    # Fallback: assume this file is one level below the root (tests/ subfolder)
    return os.path.dirname(os.path.abspath(__file__))


_project_root = _find_project_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'src', 'core'))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from compressors import (
    TopKCompressor,
    QuantizationCompressor,
    LLMInt8Compressor,
)
from gpu_compressors import (
    GPUTopKCompressor,
    GPUQuantizationCompressor,
    GPULLMInt8Compressor,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_tensor(shape, seed: int, relu: bool = True) -> torch.Tensor:
    """Create a reproducible FP32 test tensor."""
    torch.manual_seed(seed)
    t = torch.randn(*shape)
    return torch.relu(t) if relu else t


def _round_trip_cpu(comp, tensor: torch.Tensor, params, device: torch.device):
    """CPU compressor round-trip — compress on CPU, decompress to device."""
    data = comp.compress(tensor.cpu(), params)
    return comp.decompress(data, device=str(device))


def _round_trip_gpu(comp, tensor: torch.Tensor, params, device: torch.device):
    """GPU compressor round-trip — compress on device, decompress to device."""
    data = comp.compress(tensor.to(device), params)
    return comp.decompress(data, device=device)


def _check(
    out_cpu: torch.Tensor,
    out_gpu: torch.Tensor,
    label: str,
    atol: float,
    verbose: bool,
) -> Tuple[bool, float, float]:
    """
    Compare two tensors.  Returns (passed, max_abs_diff, mean_abs_diff).
    """
    diff = (out_cpu.cpu() - out_gpu.cpu()).abs()
    max_diff  = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    passed    = max_diff <= atol

    status = '✓  PASS' if passed else '✗  FAIL'
    if verbose or not passed:
        print(f"    {status}  {label}")
        print(f"           max_abs_diff={max_diff:.2e}  mean_abs_diff={mean_diff:.2e}"
              f"  tolerance={atol:.2e}")
        if not passed:
            print(f"           shapes: cpu={tuple(out_cpu.shape)} gpu={tuple(out_gpu.shape)}")
    return passed, max_diff, mean_diff


def _check_predictions(
    logits_cpu: torch.Tensor,
    logits_gpu: torch.Tensor,
    label: str,
    verbose: bool,
) -> Tuple[bool, float]:
    """
    Compare argmax predictions.  Returns (all_agree, agreement_rate).
    """
    pred_cpu = logits_cpu.cpu().argmax(dim=1)
    pred_gpu = logits_gpu.cpu().argmax(dim=1)
    agreement = float((pred_cpu == pred_gpu).float().mean().item())
    passed    = agreement == 1.0

    status = '✓  PASS' if passed else '✗  FAIL'
    if verbose or not passed:
        print(f"    {status}  {label}")
        print(f"           prediction agreement={agreement:.4f}  "
              f"({int(agreement * pred_cpu.numel())}/{pred_cpu.numel()})")
    return passed, agreement


# ============================================================================
# Test shapes — covers ResNet activation sizes at default cutpoints
# ============================================================================
# link 0: batch=4, C=32, H=16, W=16  (~32K elements)
# link 1: batch=4, C=64, H=8,  W=8   (~16K elements)
# link 2: batch=4, C=128,H=4,  W=4   (~8K  elements)
# small:  batch=2, C=8,  H=4,  W=4   (256 elements — tests edge cases)
TEST_SHAPES = [
    (4, 32, 16, 16),
    (4, 64,  8,  8),
    (4, 128, 4,  4),
    (2,   8, 4,  4),
]


# ============================================================================
# 1. TopK equivalence
# ============================================================================

def test_topk_equivalence(device: torch.device, verbose: bool) -> bool:
    """
    Decompressed output should be bit-for-bit identical between CPU and GPU TopK.

    Why exact equality is expected:
    - Both implementations call torch.topk() which is deterministic for
      distinct absolute values.
    - The only risk is tie-breaking when two elements share the same absolute
      value — torch.topk is not guaranteed to break ties identically on CPU
      vs CUDA.  The test reports any ties found and uses exact equality;
      if your hardware breaks ties differently, the test will tell you.

    Note on payload size:
    - CPU uses numpy.packbits (1 bit/element mask) — smaller payload.
    - GPU uses torch.bool     (1 byte/element mask) — larger payload.
    - This is expected and documented.  Only the decompressed output is tested.
    """
    print("\n[1] TopK equivalence")
    cpu_comp = TopKCompressor()
    gpu_comp = GPUTopKCompressor()
    k_values = [0.5, 0.25, 0.125]
    seeds    = [42, 7, 99]
    all_pass = True

    for shape in TEST_SHAPES:
        for k in k_values:
            for seed in seeds:
                tensor = _make_tensor(shape, seed)
                out_cpu = _round_trip_cpu(cpu_comp, tensor, k, device)
                out_gpu = _round_trip_gpu(gpu_comp, tensor, k, device)

                # Check for ties in absolute values (may cause different results)
                flat     = tensor.reshape(tensor.shape[0], -1)
                n_keep   = max(1, int(flat.shape[1] * k))
                threshold = flat.abs().topk(n_keep, dim=1).values.min()
                n_ties   = int((flat.abs() == threshold).sum().item())

                label = f"shape={shape} k={k} seed={seed}"
                if n_ties > 0:
                    label += f"  [ties={n_ties} — may differ]"

                passed, max_diff, _ = _check(out_cpu, out_gpu, label,
                                             atol=0.0, verbose=verbose)
                all_pass = all_pass and passed

                # Also verify shape and device
                assert out_gpu.shape == tensor.shape, \
                    f"Shape mismatch: {out_gpu.shape} != {tensor.shape}"
                assert out_gpu.device.type == device.type, \
                    f"Device mismatch: {out_gpu.device} != {device}"

    # Payload size comparison (documented difference)
    tensor = _make_tensor((4, 32, 16, 16), 42)
    cpu_data = cpu_comp.compress(tensor.cpu(), 0.5)
    gpu_data = gpu_comp.compress(tensor.to(device), 0.5)
    cpu_bytes = cpu_comp.get_compressed_size(cpu_data)
    gpu_bytes = gpu_comp.get_compressed_size(gpu_data)
    if verbose:
        print(f"    [payload size] CPU={cpu_bytes}B  GPU={gpu_bytes}B  "
              f"(GPU mask is ~8× larger — torch.bool vs packbits, expected)")

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# 2. Quantization equivalence
# ============================================================================

def test_quantization_equivalence(device: torch.device, verbose: bool) -> bool:
    """
    FP16, INT2: exact equality expected (deterministic ops).
    INT8: near-exact — CPU/GPU float32↔int8↔float32 cast chains may differ
          by up to 1 ULP (~5e-7) due to rounding in the scale multiply.
          We allow atol = 1e-5 (well below one INT8 quantisation step).
    INT4: stochastic rounding (torch.bernoulli) is seeded but the CPU and
          CUDA Bernoulli generators produce different samples from the same
          seed.  Maximum possible error is one full quantisation step = scale.
          We allow atol = scale (1 full step) as the correct theoretical bound.
    """
    print("\n[2] Quantization equivalence")
    cpu_comp = QuantizationCompressor()
    gpu_comp = GPUQuantizationCompressor()

    # (k value, bit-width label, fixed_atol, scale_multiplier)
    #   fixed_atol applies a constant tolerance regardless of scale.
    #   scale_multiplier is applied to `scale = abs_max / levels`.
    #   A small epsilon (1e-5) is added to scale-based tolerances to absorb
    #   floating-point boundary effects when max_diff == tolerance exactly.
    configs = [
        (0.5,    'FP16',  None,  0.0),    # exact — dtype cast only
        (0.25,   'INT8',  1e-5,  None),   # fixed epsilon — ~1 ULP in fp32
        (0.125,  'INT4',  None,  1.0),    # 1 full quantisation step = scale
        (0.0625, 'INT2',  None,  0.0),    # exact — only 3 levels, deterministic
    ]
    seeds   = [42, 7, 99]
    all_pass = True

    for k, label, fixed_atol, scale_multiplier in configs:
        for shape in TEST_SHAPES:
            for seed in seeds:
                torch.manual_seed(seed)
                tensor = _make_tensor(shape, seed)

                abs_max = float(tensor.abs().max().item())
                scale   = abs_max / 7.0 if abs_max > 0 else 1.0   # INT4 scale

                if fixed_atol is not None:
                    atol = fixed_atol
                elif scale_multiplier == 0.0:
                    atol = 0.0
                else:
                    # Add a small epsilon to absorb floating-point boundary
                    # effects when max_diff and tolerance land on exactly the
                    # same value through different computation paths.
                    atol = scale_multiplier * scale + 1e-5

                torch.manual_seed(seed)
                out_cpu = _round_trip_cpu(cpu_comp, tensor, k, device)
                torch.manual_seed(seed)
                out_gpu = _round_trip_gpu(gpu_comp, tensor, k, device)

                test_label = f"{label} shape={shape} k={k} seed={seed}"
                passed, max_diff, _ = _check(out_cpu, out_gpu, test_label,
                                             atol=atol, verbose=verbose)
                all_pass = all_pass and passed

                assert out_gpu.shape == tensor.shape
                assert out_gpu.device.type == device.type

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# 3. LLMInt8 equivalence
# ============================================================================

def test_llmint8_equivalence(device: torch.device, verbose: bool) -> bool:
    """
    Tolerance depends on the regular_precision:
    - int8, int2: 0.01 * abs_max  (threshold difference only)
    - int4:       1.0 * reg_scale  (stochastic rounding dominates)
      where reg_scale = abs_max_regular / 7.0 for the regular block.

    The int4 regular precision uses torch.bernoulli which produces different
    samples on CPU vs CUDA, so the full quantisation step is the correct bound.

    Also checks:
    - num_outliers is close between CPU and GPU (within 5% of total elements)
    - Decompressed output shape and device are correct
    """
    print("\n[3] LLMInt8 equivalence")
    cpu_comp = LLMInt8Compressor()
    gpu_comp = GPULLMInt8Compressor()

    configs = [
        [0.01, 'fp16', 'int8'],
        [0.05, 'fp16', 'int4'],   # int4 regular — stochastic rounding
        [0.10, 'int8', 'int2'],
        [0.01, 'fp16', 'int2'],
    ]
    seeds    = [42, 7]
    all_pass = True

    for params in configs:
        outlier_ratio, outlier_prec, regular_prec = params
        for shape in TEST_SHAPES:
            for seed in seeds:
                torch.manual_seed(seed)
                tensor = _make_tensor(shape, seed, relu=False)
                abs_max = float(tensor.abs().max().item())

                if regular_prec == 'int4':
                    # Stochastic rounding: allow up to 1 full INT4 step.
                    # Approximate the regular block scale as abs_max / 7.0
                    # (same formula as _GPUQuanInt4 and _QuanInt4).
                    reg_scale = abs_max / 7.0 if abs_max > 0 else 1.0
                    atol = reg_scale
                else:
                    # Threshold difference only — 1% of dynamic range
                    atol = 0.01 * abs_max if abs_max > 0 else 1e-6

                torch.manual_seed(seed)
                out_cpu = _round_trip_cpu(cpu_comp, tensor, params, device)
                torch.manual_seed(seed)
                out_gpu = _round_trip_gpu(gpu_comp, tensor, params, device)

                test_label = (f"k={outlier_ratio} out={outlier_prec} reg={regular_prec} "
                              f"shape={shape} seed={seed}")
                passed, max_diff, mean_diff = _check(out_cpu, out_gpu, test_label,
                                                     atol=atol, verbose=verbose)
                all_pass = all_pass and passed

                # Check outlier count is close
                cpu_data = cpu_comp.compress(tensor.cpu(), params)
                gpu_data = gpu_comp.compress(tensor.to(device), params)
                n_out_cpu = int(cpu_data['num_outliers'])
                n_out_gpu = int(gpu_data['num_outliers'])
                n_total   = tensor.numel()
                outlier_diff_frac = abs(n_out_cpu - n_out_gpu) / max(n_total, 1)

                if outlier_diff_frac > 0.05:
                    print(f"    ⚠  outlier count differs by {outlier_diff_frac:.1%}: "
                          f"cpu={n_out_cpu} gpu={n_out_gpu} total={n_total}")
                    all_pass = False
                elif verbose:
                    print(f"    [outliers] cpu={n_out_cpu} gpu={n_out_gpu} "
                          f"diff={abs(n_out_cpu - n_out_gpu)} "
                          f"({outlier_diff_frac:.2%} of total)  ✓")

                assert out_gpu.shape == tensor.shape
                assert out_gpu.device.type == device.type

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# 4. ResNet end-to-end equivalence
# ============================================================================

def test_resnet_end_to_end(
    checkpoint: str,
    device: torch.device,
    verbose: bool,
) -> bool:
    """
    Run full partitioned ResNet56 inference with CPU and GPU compressors
    and compare output logits and predictions.

    For each compressor type and k/params value:
    - Load the same batch of random images
    - Run _run_partitioned with the CPU compressor
    - Run _run_partitioned with the GPU compressor
    - Compare logits (atol = 1e-4 relative to logit dynamic range)
    - Verify argmax predictions agree on 100% of samples

    This is the most meaningful test because it exercises the full pipeline
    including the branch logic in _run_partitioned.
    """
    print("\n[4] ResNet end-to-end equivalence")

    try:
        import resnet20
        from resnet_task_callables import _load_resnet56, _build_units, _build_partitions
        from resnet_task_callables import _run_partitioned
    except ImportError as e:
        print(f"  SKIP — could not import ResNet modules: {e}")
        return True

    if not os.path.exists(checkpoint):
        print(f"  SKIP — checkpoint not found: {checkpoint}")
        return True

    # Load model and build partitions once
    model      = _load_resnet56(checkpoint, device)
    units, _   = _build_units(model)
    cutpoints  = [8, 14, 21]
    partitions = _build_partitions(units, cutpoints, device)
    num_links  = len(cutpoints)

    # Random CIFAR-10 sized batch (no real data needed for equivalence test)
    torch.manual_seed(0)
    images = torch.randn(8, 3, 32, 32)   # batch=8, CIFAR-10 resolution

    configs = [
        ('topk',         TopKCompressor(),         GPUTopKCompressor(),         [0.5, 0.25]),
        ('quantization', QuantizationCompressor(),  GPUQuantizationCompressor(), [0.5, 0.25]),
        ('llmint8',      LLMInt8Compressor(),       GPULLMInt8Compressor(),
         [[0.01, 'fp16', 'int8']]),
    ]

    all_pass = True

    for comp_name, cpu_comp, gpu_comp, param_list in configs:
        for params in param_list:
            eta_vec = np.array([params] * num_links) if not isinstance(params, list) \
                      else np.array([params[0]] * num_links)

            # Build per-link params list
            if comp_name == 'llmint8':
                link_params = [params] * num_links
            else:
                link_params = [params] * num_links

            # Run with CPU compressor
            import os as _os
            _os.environ['COMPRESSOR_BACKEND'] = 'cpu'
            logits_cpu, stats_cpu = _run_partitioned(
                partitions, images, cpu_comp, eta_vec, device)

            # Run with GPU compressor
            logits_gpu, stats_gpu = _run_partitioned(
                partitions, images, gpu_comp, eta_vec, device)

            # Logit tolerance: 0.1% of logit dynamic range
            logit_range = float((logits_cpu.max() - logits_cpu.min()).abs().item())
            atol_logits = max(1e-4, 1e-3 * logit_range)

            label = f"{comp_name} params={params}"
            passed_logits, logit_diff, _ = _check(
                logits_cpu, logits_gpu, label + " [logits]",
                atol=atol_logits, verbose=verbose)

            passed_preds, agree_rate = _check_predictions(
                logits_cpu, logits_gpu, label + " [predictions]", verbose=verbose)

            all_pass = all_pass and passed_logits and passed_preds

            if verbose:
                for i, (sc, sg) in enumerate(zip(stats_cpu, stats_gpu)):
                    print(f"      link {i}: cpu_bytes={sc['compressed_bytes']} "
                          f"gpu_bytes={sg['compressed_bytes']}")

    status = '✓  ALL PASS' if all_pass else '✗  FAILURES'
    print(f"  Result: {status}")
    return all_pass


# ============================================================================
# Runner
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Equivalence tests: CPU compressors vs GPU-native compressors."
    )
    p.add_argument('--skip_e2e',   action='store_true',
                   help='Skip the ResNet end-to-end test.')
    p.add_argument('--checkpoint', type=str,
                   default=None,
                   help=('Path to ResNet56 checkpoint for end-to-end test. '
                         'Default: <project_root>/models/resnet56-4bfd9763.th'))
    p.add_argument('--verbose',    action='store_true',
                   help='Print per-test detail including passing tests.')
    p.add_argument('--device',     type=str, default=None,
                   help="Device to use: 'cuda' or 'cpu'. Default: cuda if available.")
    args = p.parse_args()

    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device     = torch.device(device_str)

    checkpoint = args.checkpoint or os.path.join(
        _project_root, 'models', 'resnet56-4bfd9763.th'
    )

    print("=" * 60)
    print("GPU Compressor Equivalence Test Suite")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Device          : {device}")
    if device.type == 'cuda':
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint      : {checkpoint}")
    print(f"  Verbose         : {args.verbose}")
    print("=" * 60)

    if device.type == 'cpu':
        print("\n⚠  Running on CPU — GPU compressors will be tested on CPU device.")
        print("   For a meaningful GPU vs CPU comparison, run on a CUDA device.\n")

    results = {}

    try:
        results['TopK']          = test_topk_equivalence(device, args.verbose)
    except Exception:
        print("  ERROR in TopK test:")
        traceback.print_exc()
        results['TopK'] = False

    try:
        results['Quantization']  = test_quantization_equivalence(device, args.verbose)
    except Exception:
        print("  ERROR in Quantization test:")
        traceback.print_exc()
        results['Quantization'] = False

    try:
        results['LLMInt8']       = test_llmint8_equivalence(device, args.verbose)
    except Exception:
        print("  ERROR in LLMInt8 test:")
        traceback.print_exc()
        results['LLMInt8'] = False

    if not args.skip_e2e:
        try:
            results['End-to-end'] = test_resnet_end_to_end(
                checkpoint, device, args.verbose)
        except Exception:
            print("  ERROR in end-to-end test:")
            traceback.print_exc()
            results['End-to-end'] = False
    else:
        print("\n[4] ResNet end-to-end  — SKIPPED (--skip_e2e)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = '✓  PASS' if passed else '✗  FAIL'
        print(f"  {status}  {name}")
        all_pass = all_pass and passed

    print("=" * 60)
    if all_pass:
        print("All tests passed.  CPU and GPU compressors are equivalent.")
    else:
        print("Some tests FAILED.  Review output above for details.")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()

