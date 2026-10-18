"""Tests for the slim experiment-lab controls."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from src.core.experiment_protocol import (
    BodySpec,
    MatchedHyperparameters,
    RewardSpec,
    default_protocol,
)
from src.lab.controls import (
    ControlError,
    build_protocol,
    build_reset_config,
    controls_from_dict,
    default_controls,
)
from src.lab.estimator import count_spatial_params, estimate_for_controls
from src.lab.snapshot import encode_maps, frame_from_simulation
from src.marl.networks import SpatialQNetwork


class LabControlTests(unittest.TestCase):
    def test_defaults_match_the_comparison_protocol(self):
        protocol = build_protocol(default_controls())
        self.assertEqual(protocol.protocol_hash(), default_protocol().protocol_hash())
        self.assertEqual(protocol.matched.tau, MatchedHyperparameters().tau)
        self.assertEqual(protocol.matched.epsilon_start, MatchedHyperparameters().epsilon_start)
        self.assertEqual(protocol.matched.epsilon_decay, MatchedHyperparameters().epsilon_decay)
        self.assertEqual(protocol.matched.per_alpha, MatchedHyperparameters().per_alpha)
        self.assertEqual(protocol.matched.replay_buffer_size, MatchedHyperparameters().replay_buffer_size)
        self.assertEqual(protocol.matched.train_frequency, MatchedHyperparameters().train_frequency)
        self.assertEqual(protocol.matched.network_type, "dqn")
        self.assertEqual(protocol.reward, RewardSpec())
        self.assertEqual(protocol.body.metabolism_rate, BodySpec().metabolism_rate)

    def test_network_width_changes_hash_and_algorithm_mode_does_not(self):
        baseline = build_protocol(default_controls())
        wider = build_protocol(replace(default_controls(), head_dim=512))
        solo = build_protocol(replace(default_controls(), algorithm_mode="iql_only"))
        self.assertNotEqual(baseline.protocol_hash(), wider.protocol_hash())
        self.assertEqual(baseline.protocol_hash(), solo.protocol_hash())

    def test_unspecified_fields_stay_on_defaults_when_loading_a_partial_form(self):
        controls = controls_from_dict({"learning_rate": 1e-3})
        protocol = build_protocol(controls)
        self.assertAlmostEqual(protocol.matched.learning_rate, 1e-3)
        self.assertEqual(protocol.matched.batch_size, MatchedHyperparameters().batch_size)
        self.assertEqual(protocol.matched.conv_channels, (128, 256))

    def test_even_split_and_validation(self):
        config = build_reset_config(default_controls())
        counts = {item.agent_type: item.count for item in config["agent_configs"]}
        self.assertEqual(counts, {"iql": 8, "qmix": 8})
        baseline = build_reset_config(replace(default_controls(), algorithm_mode="baseline_only", agents_per_group=5))
        self.assertEqual([item.agent_type for item in baseline["agent_configs"]], ["rule_based"])
        self.assertEqual(baseline["agent_configs"][0].count, 5)
        self.assertEqual(config["protocol"]["matched"]["conv_channels"], (128, 256))
        with self.assertRaises(ControlError):
            controls_from_dict({"vision_range": 0})

    def test_parameter_count_matches_the_network(self):
        channels = (8, 16)
        head_dim, ego_hidden, pool_size = 32, 16, 2
        expected = count_spatial_params(channels, head_dim, ego_hidden, pool_size)
        network = SpatialQNetwork(
            view_size=5,
            action_dim=8,
            ego_dim=4,
            conv_channels=channels,
            head_dim=head_dim,
            ego_hidden=ego_hidden,
            pool_size=pool_size,
        )
        actual = sum(parameter.numel() for parameter in network.parameters())
        self.assertEqual(expected, actual)

    def test_vram_scales_with_learner_families(self):
        one = estimate_for_controls(replace(default_controls(), algorithm_mode="iql_only"))
        two = estimate_for_controls(default_controls())
        none = estimate_for_controls(replace(default_controls(), algorithm_mode="baseline_only"))
        self.assertEqual(none["vram_mb"], 0.0)
        self.assertAlmostEqual(two["vram_mb"], one["vram_mb"] * 2, places=3)
        self.assertFalse(two["warn"])

    def test_snapshot_is_json_and_uses_image_coordinates(self):
        sugar = np.zeros((4, 4), dtype=np.float32)
        sugar[1, 3] = 10.0
        encoded = encode_maps(sugar, None, None, max_sugar=10.0, max_spice=6.0)
        raw = np.frombuffer(__import__("base64").b64decode(encoded["rgb"]), dtype=np.uint8)
        image = raw.reshape(encoded["height"], encoded["width"], 3)
        self.assertEqual(int(image[3, 1, 0]), 255)
        self.assertEqual(int(image[1, 3, 0]), 0)

        agent = SimpleNamespace(
            status=None,
            age=4,
            total_collected=8.0,
            agent_type="rule_based",
            x=1,
            y=2,
        )
        simulation = SimpleNamespace(
            step_count=2,
            protocol=build_protocol(default_controls()),
            state=SimpleNamespace(value="running"),
            current_metrics=SimpleNamespace(alive_agents=1),
            training_metrics={},
            environment=SimpleNamespace(
                sugar_map=sugar,
                spice_map=None,
                hazard_map=None,
                max_sugar=10.0,
                max_spice=6.0,
            ),
            agent_manager=SimpleNamespace(agents=[agent]),
        )
        frame = frame_from_simulation(simulation, include_map=True)
        json.dumps(frame)
        self.assertAlmostEqual(frame["sugar"]["rule_based"], 2.0)
        self.assertEqual(frame["agents"], [{"x": 1, "y": 2, "type": "rule_based"}])
        self.assertIn("map", frame)


if __name__ == "__main__":
    unittest.main()
