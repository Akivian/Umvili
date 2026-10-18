"""Parameter-shared Independent Q-Learning.

One network and one replay buffer serve every IQL agent. Each transition
still boots from that agent's own action and reward, so the update rule
stays independent of the joint action. Sharing is what makes the sample
count comparable to QMIX.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from src.core.experiment_protocol import EGO_FEATURE_DIM
from src.marl.losses import weighted_mse
from src.marl.networks import SpatialQNetwork
from src.marl.replay_buffer import Experience, PriorityReplayBuffer


class IQLTrainer:
    def __init__(self, state_dim: int, action_dim: int, view_size: int, config: Optional[Dict[str, Any]] = None):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.view_size = int(view_size)
        self.config = config or {}
        self.batch_size = int(self.config.get("batch_size", 32))
        self.gamma = float(self.config.get("gamma", 0.99))
        self.learning_rate = float(self.config.get("learning_rate", 5e-4))
        self.tau = float(self.config.get("tau", 0.01))
        self.learning_starts = int(self.config.get("learning_starts", 128))
        self.train_frequency = int(self.config.get("train_frequency", 4))
        self.target_update_frequency = int(self.config.get("target_update_frequency", 100))
        self.gradient_steps = 0
        self.last_target_update = 0

        self.q_network = SpatialQNetwork(self.view_size, self.action_dim, EGO_FEATURE_DIM)
        self.target_q_network = SpatialQNetwork(self.view_size, self.action_dim, EGO_FEATURE_DIM)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, eps=1e-8)
        self.replay_buffer = PriorityReplayBuffer(
            capacity=int(self.config.get("replay_buffer_size", 10000)),
            alpha=float(self.config.get("per_alpha", 0.6)),
            beta=float(self.config.get("per_beta", 0.4)),
        )
        self.training_stats = {"losses": [], "q_values": [], "td_errors": []}

    def store(self, experience: Dict[str, Any]) -> None:
        self.replay_buffer.add(
            Experience(
                state=np.asarray(experience["state"], dtype=np.float32),
                action=int(experience["action"]),
                reward=float(experience["reward"]),
                next_state=np.asarray(experience["next_state"], dtype=np.float32),
                done=bool(experience["done"]),
                agent_id=experience.get("agent_id"),
            )
        )

    def train_step(self, env_step: int = 0) -> Optional[float]:
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None
        if int(env_step) % self.train_frequency != 0:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long).view(-1, 1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32).view(-1, 1)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.bool).view(-1, 1)
        weights = batch["weights"].view(-1, 1)
        if states.shape[1] != self.state_dim or actions.shape[0] != states.shape[0]:
            return None

        current_q = self.q_network(states).gather(1, actions.clamp(0, self.action_dim - 1))
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_q_network(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (~dones)
        td = (target - current_q).abs().squeeze(1)
        loss = weighted_mse(current_q.squeeze(1), target.squeeze(1), weights.squeeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        self.replay_buffer.update_priorities(batch["indices"], td.detach().cpu().tolist())
        self.gradient_steps += 1
        if self.gradient_steps - self.last_target_update >= self.target_update_frequency:
            self._soft_update()
            self.last_target_update = self.gradient_steps
        self.training_stats["losses"].append(float(loss.item()))
        self.training_stats["q_values"].append(float(current_q.mean().item()))
        self.training_stats["td_errors"].extend(td.detach().cpu().tolist())
        return float(loss.item())

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target, online in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target.data.mul_(1.0 - self.tau).add_(online.data, alpha=self.tau)

    def get_training_stats(self) -> Dict[str, Any]:
        losses = self.training_stats["losses"]
        return {
            "avg_loss": float(np.mean(losses[-20:])) if losses else 0.0,
            "recent_loss": float(losses[-1]) if losses else 0.0,
            "avg_q_value": float(np.mean(self.training_stats["q_values"][-20:])) if self.training_stats["q_values"] else 0.0,
            "training_steps": self.gradient_steps,
        }
