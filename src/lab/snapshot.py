"""JSON-safe views of one simulation step.

Maps are stored as [x, y] and drawn with x horizontal. The encoded image is
row-major, so image row is y and image column is x.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Iterable, List

import numpy as np

from src.core.agent_base import AgentStatus


def encode_maps(
    sugar_map: Any,
    spice_map: Any,
    hazard_map: Any,
    max_sugar: float,
    max_spice: float,
) -> Dict[str, Any]:
    sugar = np.asarray(sugar_map, dtype=np.float32)
    height_x, width_y = sugar.shape
    spice = np.zeros_like(sugar) if spice_map is None else np.asarray(spice_map, dtype=np.float32)
    hazard = np.zeros_like(sugar) if hazard_map is None else np.asarray(hazard_map, dtype=np.float32)
    rgb = np.zeros((width_y, height_x, 3), dtype=np.uint8)
    rgb[:, :, 0] = _channel(sugar, max_sugar).T
    rgb[:, :, 1] = _channel(spice, max_spice).T
    rgb[:, :, 2] = _channel(hazard, 1.0).T
    return {
        "width": int(height_x),
        "height": int(width_y),
        "rgb": base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode("ascii"),
    }


def sugar_rates(agents: Iterable[Any]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for agent in agents:
        if not _is_alive(agent):
            continue
        age = int(getattr(agent, "age", 0))
        if age <= 0:
            continue
        collected = float(getattr(agent, "total_collected", 0.0))
        name = _type_name(agent)
        buckets.setdefault(name, []).append(collected / age)
    return {name: float(np.mean(values)) for name, values in buckets.items()}


def frame_from_simulation(simulation: Any, *, include_map: bool) -> Dict[str, Any]:
    protocol = getattr(simulation, "protocol", None)
    metrics = getattr(simulation, "current_metrics", None)
    training = getattr(simulation, "training_metrics", {}) or {}
    loss = {}
    epsilons = []
    for name, row in training.items():
        loss[str(name)] = float(getattr(row, "recent_loss", 0.0) or 0.0)
        rate = float(getattr(row, "exploration_rate", 0.0) or 0.0)
        if rate > 0.0 or str(name) in ("independent_q_learning", "qmix"):
            epsilons.append(rate)
    agents = []
    if include_map:
        for agent in getattr(getattr(simulation, "agent_manager", None), "agents", []):
            if not _is_alive(agent):
                continue
            agents.append(
                {
                    "x": int(agent.x),
                    "y": int(agent.y),
                    "type": _type_name(agent),
                }
            )
    environment = getattr(simulation, "environment", None)
    payload: Dict[str, Any] = {
        "step": int(getattr(simulation, "step_count", 0)),
        "hash": protocol.protocol_hash() if protocol is not None else "",
        "state": getattr(getattr(simulation, "state", None), "value", "ready"),
        "epsilon": float(np.mean(epsilons)) if epsilons else 0.0,
        "sugar": sugar_rates(getattr(getattr(simulation, "agent_manager", None), "agents", [])),
        "loss": loss,
        "alive": int(getattr(metrics, "alive_agents", 0) or 0),
        "agents": agents,
    }
    if include_map and environment is not None:
        payload["map"] = encode_maps(
            getattr(environment, "sugar_map"),
            getattr(environment, "spice_map", None),
            getattr(environment, "hazard_map", None),
            float(getattr(environment, "max_sugar", 10.0) or 10.0),
            float(getattr(environment, "max_spice", 6.0) or 6.0),
        )
    return payload


def _channel(values: np.ndarray, scale: float) -> np.ndarray:
    denom = scale if scale > 1e-8 else 1.0
    scaled = np.clip(np.asarray(values, dtype=np.float32) / denom, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _is_alive(agent: Any) -> bool:
    status = getattr(agent, "status", None)
    if status is None:
        return True
    return getattr(status, "value", status) == AgentStatus.ALIVE.value


def _type_name(agent: Any) -> str:
    kind = getattr(agent, "agent_type", "")
    return str(getattr(kind, "value", kind))
