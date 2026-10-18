"""Tests for the fair-comparison protocol."""

import unittest

import numpy as np
import torch

from src.core.experiment_protocol import (
    ActionCodec,
    MatchedHyperparameters,
    RewardEvent,
    RewardSpec,
    default_protocol,
)
from src.core.observation_encoder import (
    ObservationSpecError,
    encode_global_state,
    encode_local_state,
    local_channels,
)
from src.core.reward_calculator import RewardCalculator
from src.marl.losses import weighted_mse
from src.marl.networks import QMIXMixingNetwork, SpatialQNetwork
from src.marl.replay_buffer import Experience, PriorityReplayBuffer


class _MapEnv:
    def __init__(self, size=10):
        self.size = size
        self.max_sugar = 10.0
        self.max_spice = 6.0
        self.sugar_map = np.ones((size, size), dtype=np.float32)
        self.spice_map = np.zeros((size, size), dtype=np.float32)
        self.hazard_map = np.zeros((size, size), dtype=np.float32)
        self.hazard_map[0, 0] = 1.0
        self.total_sugar = float(self.sugar_map.sum())
        self.avg_sugar = float(self.sugar_map.mean())
        self.total_spice = 0.0
        self.step = 3
        self.alive_agent_count = 4
        self.hazard_active_threshold = 0.3


class _Agent:
    def __init__(self):
        self.x = 1
        self.y = 1
        self.sugar = 15.0
        self.age = 2
        self.metabolism_rate = 0.8
        self.vision_range = 5


def _event(**overrides):
    payload = dict(
        sugar_collected=1.0,
        spice_collected=2.0,
        hazard_damage=0.5,
        metabolism_cost=0.8,
        died=False,
        prev_position=(0, 0),
        position=(1, 0),
        visited_positions=((0, 0),),
        environment_size=20,
        age=3,
        total_collected=4.0,
        current_sugar=12.0,
    )
    payload.update(overrides)
    return RewardEvent(**payload)


class ProtocolTests(unittest.TestCase):
    def test_hash_ignores_algorithm_slot_and_changes_with_ablation(self):
        left = default_protocol(algorithm_mode="iql_qmix")
        right = default_protocol(algorithm_mode="qmix_only")
        self.assertEqual(left.protocol_hash(), right.protocol_hash())

        explored = default_protocol(reward=RewardSpec(exploration_enabled=True))
        self.assertNotEqual(left.protocol_hash(), explored.protocol_hash())

        dueling = default_protocol(
            algorithm_mode="iql_only",
            matched=MatchedHyperparameters(network_type="dueling"),
        )
        self.assertNotEqual(left.protocol_hash(), dueling.protocol_hash())

    def test_joint_comparison_rejects_non_dqn_family(self):
        with self.assertRaises(Exception):
            default_protocol(
                algorithm_mode="iql_qmix",
                matched=MatchedHyperparameters(network_type="dueling"),
            )

    def test_same_event_same_reward(self):
        event = _event(died=True)
        iql_reward = RewardCalculator.from_spec(RewardSpec()).calculate_from_event(event)
        qmix_reward = RewardCalculator.from_spec(RewardSpec()).calculate_from_event(event)
        self.assertEqual(iql_reward["reward"], qmix_reward["reward"])
        expected = 1.0 + 2.0 * 5.0 - 0.5 - 0.8 - 5.0
        self.assertAlmostEqual(iql_reward["reward"], expected)

    def test_encoder_length_and_global_width(self):
        env = _MapEnv()
        agent = _Agent()
        state = encode_local_state(env, agent)
        self.assertEqual(state.shape, (3 * 11 * 11 + 4,))
        channels = local_channels(env, agent.x, agent.y, agent.vision_range)
        self.assertEqual(channels.shape, (3, 11, 11))
        self.assertGreater(channels[2].sum(), 0.0)
        global_state = encode_global_state(env)
        self.assertEqual(global_state.shape, (64,))
        with self.assertRaises(ObservationSpecError):
            local_channels(env, 0, 0, 0)

    def test_action_codec_wraps_and_has_eight_targets(self):
        codec = ActionCodec(10)
        self.assertEqual(codec.index_to_position(0, 0, 6), (9, 0))
        targets = codec.legal_targets(0, 0)
        self.assertEqual(len(targets), 8)
        self.assertEqual(len(set(targets)), 8)

    def test_mixing_network_is_monotonic(self):
        torch.manual_seed(0)
        net = QMIXMixingNetwork(num_agents=3, state_dim=8, mixing_hidden_dim=16)
        agent_qs = torch.randn(4, 3)
        states = torch.randn(4, 8)
        base = net(agent_qs, states)
        bumped = agent_qs.clone()
        bumped[:, 1] = bumped[:, 1] + 0.75
        after = net(bumped, states)
        self.assertTrue(torch.all(after + 1e-5 >= base))

    def test_joint_replay_keeps_agent_axis(self):
        buffer = PriorityReplayBuffer(capacity=32)
        for _ in range(8):
            buffer.add(Experience(
                state=np.zeros((3, 6), dtype=np.float32),
                action=np.array([0, 1, 2]),
                reward=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                next_state=np.zeros((3, 6), dtype=np.float32),
                done=np.array([False, False, True]),
                global_state=np.zeros(64, dtype=np.float32),
                next_global_state=np.zeros(64, dtype=np.float32),
            ))
        batch = buffer.sample(4)
        self.assertEqual(batch["actions"].shape, (4, 3))
        self.assertEqual(batch["rewards"].shape, (4, 3))
        self.assertEqual(batch["states"].shape, (4, 3, 6))

    def test_spatial_q_reads_local_grid(self):
        net = SpatialQNetwork(view_size=11, action_dim=8, ego_dim=4)
        flat = torch.zeros(2, 3 * 11 * 11 + 4)
        q_values = net(flat)
        self.assertEqual(tuple(q_values.shape), (2, 8))

    def test_weighted_mse_drops_zero_weight_sample(self):
        predictions = torch.tensor([0.0, 10.0])
        targets = torch.tensor([0.0, 0.0])
        weights = torch.tensor([1.0, 0.0])
        loss = weighted_mse(predictions, targets, weights)
        self.assertAlmostEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
