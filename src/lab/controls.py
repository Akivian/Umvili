"""Slim controls for one comparison run.

The page only submits the fields that change the next conclusion. Every other
protocol field stays on the dataclass default, and still enters the hash.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Tuple

from src.core.agent_factory import AgentTypeConfig, DistributionType, PositionConfig
from src.core.experiment_protocol import (
    BodySpec,
    EnvironmentFingerprint,
    ExperimentProtocol,
    MatchedHyperparameters,
    ProtocolError,
)

MODES: Dict[str, Tuple[str, ...]] = {
    "iql_only": ("iql",),
    "qmix_only": ("qmix",),
    "iql_qmix": ("iql", "qmix"),
    "baseline_only": ("rule_based",),
    "mixed": ("iql", "qmix", "rule_based"),
}

PRESETS: Dict[str, Dict[str, int]] = {
    "tiny": {"conv_in": 32, "conv_out": 64, "head_dim": 128, "ego_hidden": 64, "pool_size": 4},
    "default": {"conv_in": 128, "conv_out": 256, "head_dim": 1024, "ego_hidden": 256, "pool_size": 8},
    "wide": {"conv_in": 192, "conv_out": 384, "head_dim": 1536, "ego_hidden": 384, "pool_size": 8},
}

LIMITS: Dict[str, Tuple[float, float]] = {
    "agents_per_group": (1, 40),
    "conv_in": (8, 512),
    "conv_out": (8, 512),
    "head_dim": (32, 2048),
    "ego_hidden": (16, 1024),
    "pool_size": (1, 16),
    "learning_rate": (1e-5, 1e-2),
    "gamma": (0.8, 0.999),
    "epsilon_end": (0.0, 0.5),
    "batch_size": (8, 256),
    "learning_starts": (1, 10000),
    "grid_size": (20, 160),
    "sugar_growth_rate": (0.0, 1.0),
    "vision_range": (1, 8),
}

METRIC_NAMES = {
    "iql": "independent_q_learning",
    "qmix": "qmix",
    "rule_based": "rule_based",
}


class ControlError(ValueError):
    """Raised when a submitted control cannot become a protocol."""


@dataclass
class LabControls:
    algorithm_mode: str = "iql_qmix"
    agents_per_group: int = 8
    conv_in: int = 128
    conv_out: int = 256
    head_dim: int = 1024
    ego_hidden: int = 256
    pool_size: int = 8
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_end: float = 0.05
    batch_size: int = 32
    learning_starts: int = 128
    grid_size: int = 80
    sugar_growth_rate: float = 0.1
    vision_range: int = 5
    spice_enabled: bool = True
    hazard_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def default_controls() -> LabControls:
    return LabControls()


def controls_meta() -> Dict[str, Any]:
    return {
        "defaults": default_controls().to_dict(),
        "presets": PRESETS,
        "modes": [
            {"id": "iql_only", "label": "IQL"},
            {"id": "qmix_only", "label": "QMIX"},
            {"id": "iql_qmix", "label": "IQL + QMIX"},
            {"id": "baseline_only", "label": "规则基线"},
            {"id": "mixed", "label": "混合"},
        ],
        "limits": {key: {"min": low, "max": high} for key, (low, high) in LIMITS.items()},
    }


def controls_from_dict(raw: Mapping[str, Any] | None) -> LabControls:
    data = dict(raw or {})
    controls = default_controls()
    for item in fields(controls):
        if item.name not in data or data[item.name] is None:
            continue
        setattr(controls, item.name, _coerce(item.name, data[item.name]))
    _validate(controls)
    return controls


def build_protocol(controls: LabControls) -> ExperimentProtocol:
    _validate(controls)
    protocol = ExperimentProtocol(
        environment=EnvironmentFingerprint(
            grid_size=int(controls.grid_size),
            sugar_growth_rate=float(controls.sugar_growth_rate),
            spice_enabled=bool(controls.spice_enabled),
            hazard_enabled=bool(controls.hazard_enabled),
        ),
        body=BodySpec(vision_range=int(controls.vision_range)),
        matched=MatchedHyperparameters(
            learning_rate=float(controls.learning_rate),
            gamma=float(controls.gamma),
            epsilon_end=float(controls.epsilon_end),
            batch_size=int(controls.batch_size),
            learning_starts=int(controls.learning_starts),
            conv_channels=(int(controls.conv_in), int(controls.conv_out)),
            head_dim=int(controls.head_dim),
            ego_hidden=int(controls.ego_hidden),
            pool_size=int(controls.pool_size),
            network_type="dqn",
        ),
        algorithm_mode=controls.algorithm_mode,
    )
    try:
        protocol.validate()
    except ProtocolError as exc:
        raise ControlError(str(exc)) from exc
    return protocol


def build_reset_config(controls: LabControls) -> Dict[str, Any]:
    """Config dict accepted by MARLSimulation.reset."""
    protocol = build_protocol(controls)
    env = protocol.environment
    groups = MODES[controls.algorithm_mode]
    position = PositionConfig(distribution=DistributionType.UNIFORM)
    agent_configs = []
    for agent_type in groups:
        config: Dict[str, Any] = {}
        if agent_type == "rule_based":
            config = {"movement_strategy": "greedy"}
        agent_configs.append(
            AgentTypeConfig(
                agent_type=agent_type,
                count=int(controls.agents_per_group),
                config=config,
                position_config=position,
            )
        )
    return {
        "grid_size": env.grid_size,
        "cell_size": 8,
        "initial_agents": int(controls.agents_per_group) * len(groups),
        "sugar_growth_rate": env.sugar_growth_rate,
        "max_sugar": env.max_sugar,
        "agent_configs": agent_configs,
        "protocol": protocol.to_dict(),
        "spice_growth_rate": env.spice_growth_rate,
        "max_spice": env.max_spice,
        "hazard_decay_rate": env.hazard_decay_rate,
        "hazard_target_fraction": env.hazard_target_fraction,
        "resource_enabled": {
            "sugar": env.sugar_enabled,
            "spice": env.spice_enabled,
            "hazard": env.hazard_enabled,
        },
    }


def active_metric_types(controls: LabControls) -> Tuple[str, ...]:
    return tuple(METRIC_NAMES[name] for name in MODES[controls.algorithm_mode])


def _validate(controls: LabControls) -> None:
    if controls.algorithm_mode not in MODES:
        raise ControlError(f"unknown algorithm_mode: {controls.algorithm_mode}")
    for name, (low, high) in LIMITS.items():
        value = getattr(controls, name)
        if value < low or value > high:
            raise ControlError(f"{name} must be between {low} and {high}")


def _coerce(name: str, value: Any) -> Any:
    if name in ("spice_enabled", "hazard_enabled"):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if name == "algorithm_mode":
        return str(value)
    integer_fields = {
        "agents_per_group",
        "conv_in",
        "conv_out",
        "head_dim",
        "ego_hidden",
        "pool_size",
        "batch_size",
        "learning_starts",
        "grid_size",
        "vision_range",
    }
    try:
        if name in integer_fields:
            return int(value)
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ControlError(f"{name} is not a number") from exc
