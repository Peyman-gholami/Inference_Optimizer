"""
Measured LLM inference profiles for realistic `tau` (per-node compute) and `a` (link payloads).

Sources: user-provided tables for Gemma-2B / Gemma-7B
  - Per-stage latency @ seq_len=256 (ms)
  - Inter-stage activation sizes @ prefill_len=512 FP16 (KB raw/serialized) and decode (B mean)

Topology mapping (3 nodes, 3 stages, 2 links):
  Node 0 = stage 0, link 0 carries activations after stage 0 (split layer = first split row)
  Node 1 = stage 1, link 1 carries activations after stage 1 (split layer = second split row)
  Node 2 = stage 2

Units in returned dict:
  tau[i]: seconds
  a[i]: bytes (for use with c_t in bytes/s if you interpret c_t that way)
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


def _kb_to_bytes(kb: float) -> float:
    return float(kb) * 1024.0


def measured_tau_and_a(
    model: Literal["gemma_2b", "gemma_7b"],
    batch: int,
    payload_mode: PayloadMode,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Returns (tau, a) for InferenceTask with L_k=3, nodes 0..2, links 0..1.
    """
    table = _GEMMA_2B if model == "gemma_2b" else _GEMMA_7B
    if batch not in table:
        raise ValueError(f"No measured profile for batch={batch}. Choose one of {sorted(table.keys())}.")

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
