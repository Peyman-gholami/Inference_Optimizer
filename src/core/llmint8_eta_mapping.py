from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _as_float_list(value: Any) -> List[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def validate_llmint8_mapping_entries(entries: Any, *, num_links: int) -> List[Dict[str, Any]]:
    """
    Normalize llmint8 mapping entries and keep only valid rows.

    Expected per-entry schema:
      - eta or eta_vec: list[float] length == num_links
      - compression_params_list (or aliases): list[float] length == num_links
    """
    if not isinstance(entries, list):
        return []

    normalized: List[Dict[str, Any]] = []
    expected = int(num_links)
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        eta = _as_float_list(raw.get("eta", raw.get("eta_vec")))
        params = _as_float_list(
            raw.get("compression_params_list", raw.get("compression_params", raw.get("params")))
        )
        if eta is None or params is None:
            continue
        if len(eta) != expected or len(params) != expected:
            continue
        normalized.append({**raw, "eta": eta, "compression_params_list": params})
    return normalized


def _bytes_per_value(precision: str) -> float:
    p = str(precision).strip().lower()
    if p == "fp16":
        return 2.0
    if p == "int8":
        return 1.0
    if p == "int4":
        return 0.5
    if p == "int2":
        return 0.25
    raise ValueError(f"Unsupported precision: {precision!r}")


def _analytic_llmint8_params_from_eta(
    eta_value: float,
    *,
    outlier_precision: str,
) -> List[Any]:
    """
    Map optimizer eta (0..1) to llmint8 params [k, outlier_precision, regular_precision]
    using a simple payload-size model (no lookup table).

    Approximate per-element compressed bytes:
      mask_bits + regular + outlier_delta
      = 0.125 + (1-k)*b_reg + k*b_outlier
    Original FP32 bytes per element = 4.0
    Target bytes per element        = 4.0 * eta
    """
    eta = float(np.clip(float(eta_value), 0.0, 1.0))
    target_bpe = 4.0 * eta
    mask_bpe = 0.125
    b_outlier = _bytes_per_value(outlier_precision)

    # Bulk ("starved") path: regular activations use int2 for maximum compression.
    candidate_regular = ["int2"]
    k_min, k_max = 0.001, 0.20

    best: Dict[str, Any] | None = None
    for regular in candidate_regular:
        b_regular = _bytes_per_value(regular)
        denom = b_outlier - b_regular
        if abs(denom) < 1e-12:
            k_star = k_min
        else:
            k_star = (target_bpe - mask_bpe - b_regular) / denom
        k = float(np.clip(k_star, k_min, k_max))

        achieved_bpe = mask_bpe + (1.0 - k) * b_regular + k * b_outlier
        error = abs(achieved_bpe - target_bpe)
        candidate = {"k": k, "regular_precision": regular, "error": error}
        if best is None or candidate["error"] < best["error"]:
            best = candidate

    assert best is not None
    return [best["k"], str(outlier_precision), str(best["regular_precision"])]


def resolve_codec_execution_plan(
    *,
    codec_name: str,
    eta: List[float],
    outlier_precision: str = "fp16",
    regular_precision: str = "int4",
    llmint8_mapping_entries: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Map optimizer eta to llmint8 runtime compression parameters.

    Strategy:
      1) If mapping table is empty, passthrough eta.
      2) Else nearest-neighbor in eta-space to capture measured overhead behavior.
    """
    codec = str(codec_name).strip().lower()
    eta_vec = np.asarray([float(x) for x in eta], dtype=float).reshape(-1)

    if codec != "llmint8":
        return {"compression_params_list": eta_vec.tolist()}

    entries = list(llmint8_mapping_entries or [])
    if not entries:
        return {
            "codec_name": codec,
            "outlier_precision": str(outlier_precision),
            "regular_precision": None,
            "compression_params_list": [
                _analytic_llmint8_params_from_eta(v, outlier_precision=str(outlier_precision))
                for v in eta_vec.tolist()
            ],
        }

    best_params: List[float] | None = None
    best_dist = float("inf")
    for entry in entries:
        e = np.asarray(entry.get("eta", []), dtype=float).reshape(-1)
        p = entry.get("compression_params_list", [])
        if e.shape != eta_vec.shape or not isinstance(p, list) or len(p) != int(eta_vec.size):
            continue
        dist = float(np.linalg.norm(e - eta_vec))
        if dist < best_dist:
            best_dist = dist
            best_params = [float(x) for x in p]

    if best_params is None:
        best_params = [
            _analytic_llmint8_params_from_eta(v, outlier_precision=str(outlier_precision))
            for v in eta_vec.tolist()
        ]

    return {
        "codec_name": codec,
        "outlier_precision": str(outlier_precision),
        "regular_precision": str(regular_precision),
        "compression_params_list": best_params,
    }
