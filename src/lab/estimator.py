"""Parameter and VRAM estimate for the shared spatial Q network.

The count matches SpatialQNetwork, including biases. VRAM assumes each learner
family keeps an online net, a target net, and two Adam moments, then adds a
fixed factor for activations. Two families means IQL and QMIX are both resident.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.core.experiment_protocol import ACTION_COUNT, EGO_FEATURE_DIM
from src.lab.controls import MODES, LabControls

BYTES_PER_COPY = 4
COPIES_PER_FAMILY = 4
ACTIVATION_FACTOR = 1.5
WARN_MB = 1024.0


def learner_families(algorithm_mode: str) -> int:
    groups = MODES.get(algorithm_mode, ())
    return sum(1 for name in groups if name in ("iql", "qmix"))


def count_spatial_params(
    conv_channels: Tuple[int, int],
    head_dim: int,
    ego_hidden: int,
    pool_size: int,
    ego_dim: int = EGO_FEATURE_DIM,
    action_dim: int = ACTION_COUNT,
) -> int:
    width_in, width_out = int(conv_channels[0]), int(conv_channels[1])
    conv1 = 3 * width_in * 9 + width_in
    conv2 = width_in * width_out * 9 + width_out
    ego = ego_dim * int(ego_hidden) + int(ego_hidden)
    pooled = width_out * int(pool_size) * int(pool_size)
    head1 = (pooled + int(ego_hidden)) * int(head_dim) + int(head_dim)
    head2 = int(head_dim) * action_dim + action_dim
    return int(conv1 + conv2 + ego + head1 + head2)


def estimate_for_controls(controls: LabControls) -> Dict[str, Any]:
    params = count_spatial_params(
        (controls.conv_in, controls.conv_out),
        controls.head_dim,
        controls.ego_hidden,
        controls.pool_size,
    )
    families = learner_families(controls.algorithm_mode)
    if families == 0:
        vram_mb = 0.0
    else:
        vram_mb = (
            params * BYTES_PER_COPY * COPIES_PER_FAMILY * families / (1024 ** 2) * ACTIVATION_FACTOR
        )
    return {
        "params": params,
        "vram_mb": round(vram_mb, 1),
        "warn": vram_mb > WARN_MB,
        "families": families,
    }
