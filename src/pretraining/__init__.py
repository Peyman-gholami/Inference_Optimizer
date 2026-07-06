"""
Model-pretraining algorithms (extension of the inference optimizer).

These optimize the *model weights* theta so the model performs well under the
compression policy the deployment-time controller selects, rather than treating
theta as fixed (as the rest of the repo does).

Currently implemented:
  - compression_aware: CSI-aware single task (J_{eta*} = 0), Algorithm 1.
"""

from .compression_aware import (
    ChannelSampler,
    UniformEtaChannelSampler,
    csi_aware_eta_star,
    compression_aware_pretrain,
    standard_pretrain,
    deployed_accuracy,
)

__all__ = [
    "ChannelSampler",
    "UniformEtaChannelSampler",
    "csi_aware_eta_star",
    "compression_aware_pretrain",
    "standard_pretrain",
    "deployed_accuracy",
]
