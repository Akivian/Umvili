"""
Shared observation encoder.

Learning agents consume this vector and nothing else. Length mismatches raise;
they are never truncated or zero-padded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.experiment_protocol import EGO_FEATURE_DIM, GLOBAL_POOL, GLOBAL_STATE_DIM


class ObservationSpecError(ValueError):
    """Raised when an encoded observation does not match the protocol."""


def local_channels(environment: Any, x: int, y: int, vision_range: int) -> np.ndarray:
    """Return sugar, spice, and hazard channels with shape (3, V, V)."""
    vision = int(vision_range)
    if vision < 1:
        raise ObservationSpecError("vision_range must be >= 1")
    view = vision * 2 + 1
    channels = np.zeros((3, view, view), dtype=np.float32)
    size = int(environment.size)
    max_sugar = max(float(getattr(environment, "max_sugar", 10.0)), 1e-6)
    max_spice = max(float(getattr(environment, "max_spice", 6.0)), 1e-6)
    # 与 ActionCodec 同一套环面：站在边上向左看，看到的是地图另一侧，而不是被夹回原地。
    offsets = np.arange(-vision, vision + 1)
    xs = (offsets + int(x)) % size
    ys = (offsets + int(y)) % size
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    channels[0] = np.asarray(environment.sugar_map, dtype=np.float32)[grid_x, grid_y] / max_sugar
    spice_map = getattr(environment, "spice_map", None)
    hazard_map = getattr(environment, "hazard_map", None)
    if spice_map is not None:
        channels[1] = np.asarray(spice_map, dtype=np.float32)[grid_x, grid_y] / max_spice
    if hazard_map is not None:
        channels[2] = np.asarray(hazard_map, dtype=np.float32)[grid_x, grid_y]
    return np.nan_to_num(channels, nan=0.0, posinf=1.0, neginf=0.0)


def ego_features(agent: Any) -> np.ndarray:
    features = np.array(
        [
            min(max(float(agent.sugar) / 50.0, 0.0), 1.0),
            min(float(agent.age) / 1000.0, 1.0),
            min(max(float(agent.metabolism_rate) / 3.0, 0.0), 1.0),
            min(max(float(agent.vision_range) / 8.0, 0.0), 1.0),
        ],
        dtype=np.float32,
    )
    if features.shape != (EGO_FEATURE_DIM,):
        raise ObservationSpecError("ego feature width drifted from the protocol")
    return features


def encode_local_state(environment: Any, agent: Any) -> np.ndarray:
    channels = local_channels(environment, agent.x, agent.y, agent.vision_range)
    state = np.concatenate([channels.reshape(-1), ego_features(agent)])
    expected = 3 * channels.shape[1] * channels.shape[2] + EGO_FEATURE_DIM
    if state.shape != (expected,):
        raise ObservationSpecError(f"local state length {state.shape[0]} != {expected}")
    return state.astype(np.float32, copy=False)


def require_local_state(state: np.ndarray, state_dim: int) -> np.ndarray:
    vector = np.asarray(state, dtype=np.float32).reshape(-1)
    if vector.shape != (int(state_dim),):
        raise ObservationSpecError(
            f"local state length {vector.shape[0]} != protocol state_dim {state_dim}"
        )
    return vector


def _block_pool(values: np.ndarray, out: int, normalizer: float) -> np.ndarray:
    size = int(values.shape[0])
    pooled = np.zeros((out, out), dtype=np.float32)
    scale = max(float(normalizer), 1e-6)
    for i in range(out):
        x0 = int(i * size / out)
        x1 = max(int((i + 1) * size / out), x0 + 1)
        for j in range(out):
            y0 = int(j * size / out)
            y1 = max(int((j + 1) * size / out), y0 + 1)
            pooled[i, j] = float(values[x0:x1, y0:y1].mean()) / scale
    return pooled


def encode_global_state(environment: Any, alive_agents: int | None = None) -> np.ndarray:
    """Downsampled resource maps plus an occupancy grid for the mixer."""
    size = max(int(getattr(environment, "size", 1)), 1)
    max_sugar = max(float(getattr(environment, "max_sugar", 10.0)), 1e-6)
    max_spice = max(float(getattr(environment, "max_spice", 6.0)), 1e-6)
    sugar = getattr(environment, "sugar_map", np.zeros((size, size), dtype=np.float32))
    spice = getattr(environment, "spice_map", np.zeros_like(sugar))
    hazard = getattr(environment, "hazard_map", np.zeros_like(sugar))
    channels = np.concatenate(
        [
            _block_pool(np.asarray(sugar), GLOBAL_POOL, max_sugar).reshape(-1),
            _block_pool(np.asarray(spice), GLOBAL_POOL, max_spice).reshape(-1),
            _block_pool(np.asarray(hazard), GLOBAL_POOL, 1.0).reshape(-1),
        ]
    )
    occupancy = np.zeros((GLOBAL_POOL, GLOBAL_POOL), dtype=np.float32)
    positions = getattr(environment, "agent_positions", ()) or ()
    for x, y in positions:
        ix = min(int(x) * GLOBAL_POOL // size, GLOBAL_POOL - 1)
        iy = min(int(y) * GLOBAL_POOL // size, GLOBAL_POOL - 1)
        occupancy[ix, iy] += 1.0
    if occupancy.max() > 0:
        occupancy = occupancy / occupancy.max()
    features = np.concatenate([channels, occupancy.reshape(-1)]).astype(np.float32)
    if features.shape != (GLOBAL_STATE_DIM,):
        raise ObservationSpecError(
            f"global state length {features.shape[0]} != {GLOBAL_STATE_DIM}"
        )
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
