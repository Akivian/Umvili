"""
Experiment protocol

One run has one frozen protocol. The hash covers the locked layer and the
matched layer. The algorithm slot (IQL versus QMIX) is intentionally excluded,
so two curves with the same hash differ only in the update rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


GLOBAL_STATE_DIM = 64
GLOBAL_POOL = 4
EGO_FEATURE_DIM = 4
ACTION_COUNT = 8

# N, NE, E, SE, S, SW, W, NW. Index order is part of the action contract.
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)


class ProtocolError(ValueError):
    """Raised when a protocol contradicts the comparison contract."""


@dataclass(frozen=True)
class RewardEvent:
    """Explicit transition facts. Reward is not inferred from the sugar delta."""

    sugar_collected: float
    spice_collected: float
    hazard_damage: float
    metabolism_cost: float
    died: bool
    prev_position: Tuple[int, int]
    position: Tuple[int, int]
    visited_positions: Tuple[Tuple[int, int], ...]
    environment_size: int
    age: int = 0
    total_collected: float = 0.0
    current_sugar: float = 0.0

    @staticmethod
    def visited_from(positions: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted((int(x), int(y)) for x, y in positions))


@dataclass(frozen=True)
class RewardSpec:
    sugar_weight: float = 1.0
    spice_weight: float = 5.0
    hazard_weight: float = 1.0
    metabolism_weight: float = 1.0
    death_penalty: float = 5.0
    exploration_enabled: bool = False
    exploration_reward: float = 0.5
    wealth_enabled: bool = False
    wealth_threshold_1: float = 25.0
    wealth_reward_1: float = 0.1
    wealth_threshold_2: float = 50.0
    wealth_reward_2: float = 0.2
    efficiency_enabled: bool = False
    efficiency_multiplier: float = 0.1
    efficiency_min_age: int = 10
    movement_cost_enabled: bool = False
    movement_cost: float = 0.05
    boundary_penalty_enabled: bool = False
    boundary_penalty: float = 0.01
    clip_abs: float = 10.0


@dataclass(frozen=True)
class BodySpec:
    vision_range: int = 5
    metabolism_rate: float = 0.8
    initial_sugar: float = 15.0


@dataclass(frozen=True)
class EnvironmentFingerprint:
    grid_size: int = 80
    sugar_growth_rate: float = 0.1
    max_sugar: float = 10.0
    spice_growth_rate: float = 0.02
    max_spice: float = 6.0
    hazard_decay_rate: float = 0.0
    hazard_target_fraction: float = 0.33
    sugar_enabled: bool = True
    spice_enabled: bool = True
    hazard_enabled: bool = True


@dataclass(frozen=True)
class MatchedHyperparameters:
    """Shared by every learning algorithm in the run."""

    learning_rate: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.01
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.998
    hidden_dims: Tuple[int, ...] = (128, 64)
    batch_size: int = 32
    replay_buffer_size: int = 10000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    network_type: str = "dqn"
    train_frequency: int = 4
    target_update_frequency: int = 100
    learning_starts: int = 128
    episode_length: int = 256


@dataclass(frozen=True)
class ExperimentProtocol:
    environment: EnvironmentFingerprint = field(default_factory=EnvironmentFingerprint)
    body: BodySpec = field(default_factory=BodySpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    matched: MatchedHyperparameters = field(default_factory=MatchedHyperparameters)
    algorithm_mode: str = "iql_qmix"

    @property
    def local_state_dim(self) -> int:
        view = self.body.vision_range * 2 + 1
        return 3 * view * view + EGO_FEATURE_DIM

    @property
    def global_state_dim(self) -> int:
        return GLOBAL_STATE_DIM

    def validate(self) -> None:
        if self.body.vision_range < 1:
            raise ProtocolError("vision_range must be >= 1")
        if self.matched.network_type not in ("dqn", "dueling", "noisy"):
            raise ProtocolError(f"unknown network_type: {self.matched.network_type}")
        if self.algorithm_mode == "iql_qmix" and self.matched.network_type != "dqn":
            raise ProtocolError(
                "iql_qmix keeps the network family at dqn; "
                "Dueling and Noisy are IQL-only ablations"
            )
        if len(DIRECTIONS) != ACTION_COUNT:
            raise ProtocolError("action contract must contain 8 directions")

    def hash_payload(self) -> Dict[str, Any]:
        """Locked layer plus matched layer. Algorithm name is not included."""
        return {
            "environment": asdict(self.environment),
            "body": asdict(self.body),
            "reward": asdict(self.reward),
            "matched": asdict(self.matched),
        }

    def protocol_hash(self) -> str:
        encoded = json.dumps(
            _canonicalize(self.hash_payload()),
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        data = self.hash_payload()
        data["algorithm_mode"] = self.algorithm_mode
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentProtocol":
        matched_raw = dict(data.get("matched") or {})
        hidden = matched_raw.get("hidden_dims", (128, 64))
        matched_raw["hidden_dims"] = tuple(int(v) for v in hidden)
        protocol = cls(
            environment=EnvironmentFingerprint(**dict(data.get("environment") or {})),
            body=BodySpec(**dict(data.get("body") or {})),
            reward=RewardSpec(**dict(data.get("reward") or {})),
            matched=MatchedHyperparameters(**matched_raw),
            algorithm_mode=str(data.get("algorithm_mode", "iql_qmix")),
        )
        protocol.validate()
        return protocol


def default_protocol(
    grid_size: int = 80,
    sugar_growth_rate: float = 0.1,
    max_sugar: float = 10.0,
    spice_growth_rate: float = 0.02,
    max_spice: float = 6.0,
    hazard_decay_rate: float = 0.0,
    hazard_target_fraction: float = 0.33,
    sugar_enabled: bool = True,
    spice_enabled: bool = True,
    hazard_enabled: bool = True,
    algorithm_mode: str = "iql_qmix",
    matched: MatchedHyperparameters | None = None,
    reward: RewardSpec | None = None,
) -> ExperimentProtocol:
    protocol = ExperimentProtocol(
        environment=EnvironmentFingerprint(
            grid_size=int(grid_size),
            sugar_growth_rate=float(sugar_growth_rate),
            max_sugar=float(max_sugar),
            spice_growth_rate=float(spice_growth_rate),
            max_spice=float(max_spice),
            hazard_decay_rate=float(hazard_decay_rate),
            hazard_target_fraction=float(hazard_target_fraction),
            sugar_enabled=bool(sugar_enabled),
            spice_enabled=bool(spice_enabled),
            hazard_enabled=bool(hazard_enabled),
        ),
        reward=reward or RewardSpec(),
        matched=matched or MatchedHyperparameters(),
        algorithm_mode=algorithm_mode,
    )
    protocol.validate()
    return protocol


class ActionCodec:
    """Single interpreter of the 8-direction toroidal action contract."""

    def __init__(self, environment_size: int):
        if environment_size <= 0:
            raise ProtocolError("environment_size must be positive")
        self.environment_size = int(environment_size)

    @property
    def action_count(self) -> int:
        return ACTION_COUNT

    def index_to_position(self, x: int, y: int, action_idx: int) -> Tuple[int, int]:
        if not 0 <= int(action_idx) < ACTION_COUNT:
            raise ProtocolError(f"action index {action_idx} is outside 0..{ACTION_COUNT - 1}")
        dx, dy = DIRECTIONS[int(action_idx)]
        size = self.environment_size
        return (int(x) + dx) % size, (int(y) + dy) % size

    def legal_targets(self, x: int, y: int) -> List[Tuple[int, int]]:
        return [self.index_to_position(x, y, idx) for idx in range(ACTION_COUNT)]


def epsilon_greedy_policy(
    q_values: Sequence[float] | None,
    epsilon: float,
    action_dim: int = ACTION_COUNT,
):
    import numpy as np

    if q_values is None or len(q_values) != action_dim:
        return np.ones(action_dim, dtype=np.float64) / action_dim
    policy = np.ones(action_dim, dtype=np.float64) * (float(epsilon) / action_dim)
    best = int(np.argmax(np.asarray(q_values)))
    policy[best] += 1.0 - float(epsilon)
    return policy


def policy_entropy_bits(policy) -> float:
    import numpy as np

    probs = np.asarray(policy, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    return str(value)
