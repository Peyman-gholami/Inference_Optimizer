"""
Measured LLM inference profiles for realistic `tau` (per-node compute) and `a` (link payloads).

Sources: user-provided tables for Gemma-2B / Gemma-7B / Llama-3.1-8B
  - Per-stage latency @ seq_len=256 (ms)
  - Inter-stage activation sizes @ prefill_len=512 FP16 (KB raw/serialized) and decode (B mean)

Topology mapping (3 nodes, 3 stages, 2 links):
  Node 0 = stage 0, link 0 carries activations after stage 0 (split layer = first split row)
  Node 1 = stage 1, link 1 carries activations after stage 1 (split layer = second split row)
  Node 2 = stage 2

Units in returned dict:
  tau[i]: seconds
  a[i]: bytes (for use with c_t in bytes/s if you interpret c_t that way)

Layer partitioning (thirds split mode):
  Gemma-2B    (18 layers): 6 / 6 / 6   (exact thirds)
  Gemma-7B    (28 layers): 9 / 9 / 10
  Llama-3.1-8B(32 layers): 10 / 11 / 11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

PayloadMode = Literal["prefill_raw", "prefill_serial", "decode_raw", "decode_serial"]


@dataclass(frozen=True)
class _StageSplitProfile:
    """stage latencies in ms for stages 0,1,2; two split rows (link0, link1)."""

    latency_ms: Tuple[float, float, float]
    # Prefill: KB (as in table). Decode: mean bytes (as in table).
    prefill_raw_kb: Tuple[float, float]
    prefill_serial_kb: Tuple[float, float]
    decode_raw_b: Tuple[float, float]
    decode_serial_b: Tuple[float, float]


# batch -> profile
_GEMMA_2B: Dict[int, _StageSplitProfile] = {
    1: _StageSplitProfile(
        latency_ms=(6.79, 6.81, 6.75),
        prefill_raw_kb=(2048.0, 2048.0),
        prefill_serial_kb=(2049.2, 2049.2),
        decode_raw_b=(4096.0, 4096.0),
        decode_serial_b=(5276.0, 5276.0),
    ),
    2: _StageSplitProfile(
        latency_ms=(7.86, 7.82, 7.79),
        prefill_raw_kb=(4096.0, 4096.0),
        prefill_serial_kb=(4097.1, 4097.1),
        decode_raw_b=(8192.0, 8192.0),
        decode_serial_b=(9372.0, 9372.0),
    ),
    4: _StageSplitProfile(
        latency_ms=(10.80, 10.77, 10.75),
        prefill_raw_kb=(8192.0, 8192.0),
        prefill_serial_kb=(8193.1, 8193.1),
        decode_raw_b=(16384.0, 16384.0),
        decode_serial_b=(17564.0, 17564.0),
    ),
}

_GEMMA_7B: Dict[int, _StageSplitProfile] = {
    1: _StageSplitProfile(
        latency_ms=(12.91, 12.92, 14.30),
        prefill_raw_kb=(3072.0, 3072.0),
        prefill_serial_kb=(3073.2, 3073.2),
        decode_raw_b=(6144.0, 6144.0),
        decode_serial_b=(7324.0, 7324.0),
    ),
    2: _StageSplitProfile(
        latency_ms=(16.59, 16.55, 18.37),
        prefill_raw_kb=(6144.0, 6144.0),
        prefill_serial_kb=(6145.2, 6145.2),
        decode_raw_b=(12288.0, 12288.0),
        decode_serial_b=(13468.0, 13468.0),
    ),
    4: _StageSplitProfile(
        latency_ms=(29.40, 29.40, 32.54),
        prefill_raw_kb=(12288.0, 12288.0),
        prefill_serial_kb=(12289.1, 12289.1),
        decode_raw_b=(24576.0, 24576.0),
        decode_serial_b=(25756.0, 25756.0),
    ),
}

# Llama-3.1-8B: 32 layers, hidden_dim=4096, split at layers 10 and 21 (thirds).
# Activation size per link = batch * seq_len * hidden_dim * bytes_per_element
#   FP16: batch=1, seq=512, hidden=4096 -> 1 * 512 * 4096 * 2 = 4,194,304 B = 4096 KB
# Stage latencies estimated proportionally to model size vs Gemma-7B
# (8B params / 7B params ≈ 1.14×), rounded to reflect H200 measurements.
# Replace with your own measured values once you have profiled the model.
_LLAMA_3_1_8B: Dict[int, _StageSplitProfile] = {
    1: _StageSplitProfile(
        latency_ms=(14.70, 16.20, 14.80),   # PLACEHOLDER — measure on H200
        prefill_raw_kb=(4096.0, 4096.0),     # 1 * 512 * 4096 * 2 bytes / 1024
        prefill_serial_kb=(4097.5, 4097.5),  # raw + small serialisation header
        decode_raw_b=(8192.0, 8192.0),       # 1 * 1 * 4096 * 2 bytes
        decode_serial_b=(9372.0, 9372.0),    # PLACEHOLDER
    ),
    2: _StageSplitProfile(
        latency_ms=(19.10, 21.00, 19.20),   # PLACEHOLDER
        prefill_raw_kb=(8192.0, 8192.0),
        prefill_serial_kb=(8193.5, 8193.5),
        decode_raw_b=(16384.0, 16384.0),
        decode_serial_b=(17564.0, 17564.0),  # PLACEHOLDER
    ),
    4: _StageSplitProfile(
        latency_ms=(33.50, 36.80, 33.70),   # PLACEHOLDER
        prefill_raw_kb=(16384.0, 16384.0),
        prefill_serial_kb=(16385.5, 16385.5),
        decode_raw_b=(32768.0, 32768.0),
        decode_serial_b=(33948.0, 33948.0),  # PLACEHOLDER
    ),
}

_MODEL_TABLE = {
    "gemma_2b":     _GEMMA_2B,
    "gemma_7b":     _GEMMA_7B,
    "llama_3_1_8b": _LLAMA_3_1_8B,
}


def _kb_to_bytes(kb: float) -> float:
    return float(kb) * 1024.0


def measured_tau_and_a(
    model: Literal["gemma_2b", "gemma_7b", "llama_3_1_8b"],
    batch: int,
    payload_mode: PayloadMode,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Returns (tau, a) for InferenceTask with L_k=3, nodes 0..2, links 0..1.

    Supported models
    ----------------
    gemma_2b      : Gemma-2B   (18 layers, hidden=2048)
    gemma_7b      : Gemma-7B   (28 layers, hidden=3072)
    llama_3_1_8b  : Llama-3.1-8B (32 layers, hidden=4096)
                    NOTE: latency values are placeholders — replace with
                    measurements from your H200 cluster before using
                    LLM_USE_MEASURED_TAU_A=1 with this model.
    """
    if model not in _MODEL_TABLE:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Choose one of: {sorted(_MODEL_TABLE.keys())}."
        )
    table = _MODEL_TABLE[model]
    if batch not in table:
        raise ValueError(
            f"No measured profile for model={model} batch={batch}. "
            f"Choose one of {sorted(table.keys())}."
        )

    p = table[batch]
    tau = {
        0: p.latency_ms[0] / 1000.0,
        1: p.latency_ms[1] / 1000.0,
        2: p.latency_ms[2] / 1000.0,
    }

    if payload_mode == "prefill_raw":
        a = {0: _kb_to_bytes(p.prefill_raw_kb[0]), 1: _kb_to_bytes(p.prefill_raw_kb[1])}
    elif payload_mode == "prefill_serial":
        a = {0: _kb_to_bytes(p.prefill_serial_kb[0]), 1: _kb_to_bytes(p.prefill_serial_kb[1])}
    elif payload_mode == "decode_raw":
        a = {0: float(p.decode_raw_b[0]), 1: float(p.decode_raw_b[1])}
    elif payload_mode == "decode_serial":
        a = {0: float(p.decode_serial_b[0]), 1: float(p.decode_serial_b[1])}
    else:
        raise ValueError(f"Unknown payload_mode: {payload_mode}")

    return tau, a
