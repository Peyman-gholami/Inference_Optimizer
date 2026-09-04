"""Analysis utilities that sit on top of the inference-optimizer backends."""

from .stein_smoothing import (
    SmoothingConfig,
    build_eta_grid,
    smoothed_value,
    smooth_grid,
    stein_gradient,
    gradient_stability,
    price_vector,
    argmax_under_price,
    regret_curve,
    summarize,
)

__all__ = [
    "SmoothingConfig",
    "build_eta_grid",
    "smoothed_value",
    "smooth_grid",
    "stein_gradient",
    "gradient_stability",
    "price_vector",
    "argmax_under_price",
    "regret_curve",
    "summarize",
]
