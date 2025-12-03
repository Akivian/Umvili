"""
独立Q学习智能体实现

基于：Tan, M. (1993). Multi-agent reinforcement learning: Independent vs. cooperative agents.
这是多智能体强化学习中最基础的算法，每个智能体独立学习自己的Q函数。

特性：
- ε-贪婪探索策略
- 目标网络稳定训练
- 优先经验回放
- 可配置的网络架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import random
import time
from dataclasses import dataclass
import logging

from src.core.agent_base import LearningAgent, ObservationSpace, AgentType, AgentStatus
from src.core.reward_calculator import RewardCalculator, RewardConfig
from src.marl.networks import QNetwork, DuelingQNetwork, NoisyQNetwork
from src.marl.replay_buffer import Experience, PriorityReplayBuffer
from src.config.simulation_config import SimulationConfig
# 向后兼容：使用默认值
GRID_SIZE = 80


@dataclass
class IQLConfig:
    """IQL智能体配置数据类"""
    # 网络配置
    state_dim: int = 64
    action_dim: int = 8
    hidden_dims: List[int] = None
    network_type: str = "dqn"  # dqn, dueling, noisy
    use_double_dqn: bool = True
    
    # 学习参数
    learning_rate: float = 0.001
    gamma: float = 0.95
    tau: float = 0.01  # 目标网络软更新参数
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # 训练参数
    batch_size: int = 32
    replay_buffer_size: int = 10000
    learning_starts: int = 1000
    train_frequency: int = 4
    target_update_frequency: int = 100
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class IQLAgent(LearningAgent):
    """
    独立Q学习智能体
    
    每个智能体独立学习自己的Q函数，不考虑其他智能体的存在。
    这是多智能体强化学习中最简单的算法，适合作为基准。
    """
    
    def __init__(self, 
                 x: int, 
                 y: int, 
                 agent_id: int,
                 config: Optional[IQLConfig] = None,
                 name: Optional[str] = None):
        
        """
        初始化IQL智能体
        
        Args:
            x, y, agent_id: 基础参数
            config: IQL配置
            name: 智能体名称
        """
        # Algorithm-specific configuration (keep separate from BaseAgent base_config)
        self.config = config or IQLConfig()
        
        # BaseAgent uses its own BaseAgentConfig internally; we don't override it here.
        # LearningAgent will pass remaining args/kwargs through to BaseAgent.
        super().__init__(
            x=x, y=y, agent_id=agent_id, 
            agent_type=AgentType.IQL,
            learning_rate=self.config.learning_rate,
            discount_factor=self.config.gamma,
            name=name or f"iql_{agent_id}"
        )
        
        # Q学习参数
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.epsilon = self.config.epsilon_start
        self.tau = self.config.tau
        
        # 训练状态
        self.training_step = 0
        self.last_state = None
        self.last_action_idx = None  # 动作索引（整数），避免与BaseAgent的last_action（位置tuple）冲突
        self.last_experience = None
        self.learning_steps = 0
        
        # 环境信息（将在第一次观察时更新）
        self.environment_size = GRID_SIZE
        self._environment_size_initialized = False
        
        # 动作映射（8个方向 + 静止）
        self.directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

        # === 状态记录用于奖励计算 ===
        self._last_sugar = self.sugar  # 记录上次糖量
        self._last_position = (self.x, self.y)  # 记录上次位置
        self.visited_positions = set([(self.x, self.y)])  # 探索记录
        self._sugar_collected_this_step = 0.0  # 当前步骤收集的糖量
        
        # 初始化奖励计算器
        reward_config = RewardConfig(
            sugar_collection_multiplier=2.0,
            sugar_consumption_penalty=0.1,
            exploration_reward=0.5,
            survival_reward=0.02,
            efficiency_multiplier=0.1,
            wealth_threshold_1=25.0,
            wealth_reward_1=0.1,
            wealth_threshold_2=50.0,
            wealth_reward_2=0.2,
            movement_cost=0.05,
            death_penalty=5.0,
            boundary_penalty=0.01,
            min_reward=-5.0,
            max_reward=5.0
        )
        self.reward_calculator = RewardCalculator(reward_config)
        
        # 初始化网络
        self._initialize_networks()
        
        # 经验回放
        self.replay_buffer = PriorityReplayBuffer(
            capacity=self.config.replay_buffer_size
        )
        
        # 训练统计
        self.training_stats = {
            'q_values': [],
            'losses': [],
            'td_errors': [],
            'exploration_rate': [],
            'rewards': []  # 添加rewards键
        }
        
        self.logger.info(f"初始化IQL智能体 {self.name} "
                        f"(网络: {self.config.network_type}, "
                        f"动作数: {self.action_dim})")
    
    def _initialize_networks(self) -> None:
        """初始化Q网络和目标网络"""
        network_args = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.config.hidden_dims
        }
        
        # 验证网络参数
        if self.state_dim <= 0:
            raise ValueError(f"状态维度必须大于0，得到: {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"动作维度必须大于0，得到: {self.action_dim}")
        if self.config.hidden_dims is None or len(self.config.hidden_dims) == 0:
            self.config.hidden_dims = [128, 64]
            self.logger.warning("隐藏层维度未指定，使用默认值 [128, 64]")
        
        # 选择网络类型
        if self.config.network_type == "dueling":
            network_class = DuelingQNetwork
        elif self.config.network_type == "noisy":
            network_class = NoisyQNetwork
        else:
            network_class = QNetwork
        
        try:
            self.q_network = network_class(**network_args)
            self.target_network = network_class(**network_args)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # 设置目标网络为评估模式（不需要梯度）
            self.target_network.eval()
            
            # 优化器
            self.optimizer = optim.Adam(
                self.q_network.parameters(), 
                lr=self.config.learning_rate,
                eps=1e-8  # 添加数值稳定性
            )
            
            # 损失函数
            self.loss_fn = nn.MSELoss(reduction='none')  # 使用'none'以便应用权重
            
            self.logger.info(
                f"网络初始化成功: 类型={self.config.network_type}, "
                f"状态维度={self.state_dim}, 动作维度={self.action_dim}, "
                f"隐藏层={self.config.hidden_dims}"
            )
        except Exception as e:
            self.logger.error(f"网络初始化失败: {e}", exc_info=True)
            raise
    
    def decide_action(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        ε-贪婪策略选择动作
        
        Args:
            observation: 观察空间
            
        Returns:
            目标位置 (x, y)
        """
        try:
            # 处理观察为状态向量
            state = self._process_observation(observation)
            
            # 验证状态维度
            if len(state) != self.state_dim:
                self.logger.warning(
                    f"智能体 {self.agent_id} 状态维度不匹配: "
                    f"{len(state)} != {self.state_dim}，使用随机动作"
                )
                action_idx = random.randint(0, self.action_dim - 1)
            else:
                # 探索 vs 利用
                if random.random() < self.epsilon:
                    # 随机探索
                    action_idx = random.randint(0, self.action_dim - 1)
                    self.logger.debug(f"智能体 {self.agent_id} 随机探索: 动作 {action_idx}")
                else:
                    # 利用Q值选择最佳动作
                    if self.q_network is None:
                        # 网络未初始化，随机选择
                        action_idx = random.randint(0, self.action_dim - 1)
                        self.logger.warning(f"智能体 {self.agent_id} Q网络未初始化，使用随机动作")
                    else:
                        with torch.no_grad():
                            # 确保状态是2D张量
                            if isinstance(state, np.ndarray):
                                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            else:
                                state_tensor = torch.FloatTensor([state]).unsqueeze(0)
                            
                            # 验证状态张量维度
                            if state_tensor.size(1) != self.state_dim:
                                self.logger.warning(
                                    f"智能体 {self.agent_id} 状态张量维度不匹配: "
                                    f"{state_tensor.size(1)} != {self.state_dim}"
                                )
                                action_idx = random.randint(0, self.action_dim - 1)
                            else:
                                q_values = self.q_network(state_tensor)
                                
                                # 验证Q值输出维度
                                if q_values.size(1) != self.action_dim:
                                    self.logger.error(
                                        f"智能体 {self.agent_id} Q值维度错误: "
                                        f"{q_values.size(1)} != {self.action_dim}"
                                    )
                                    action_idx = random.randint(0, self.action_dim - 1)
                                else:
                                    action_idx = q_values.argmax().item()
                                    
                                    # 确保动作索引在有效范围内
                                    action_idx = max(0, min(action_idx, self.action_dim - 1))
                                    
                                    # 记录Q值统计
                                    max_q = q_values.max().item()
                                    self.training_stats['q_values'].append(max_q)
                                
                                self.logger.debug(f"智能体 {self.agent_id} 利用: "
                                                f"动作 {action_idx}, Q值 {max_q:.3f}")
            
            # 转换为环境动作
            action = self._action_idx_to_position(action_idx)
            
            # 准备经验数据（在update方法中完善并存储）
            # 注意：使用last_action_idx而不是last_action，避免与BaseAgent的last_action（位置tuple）冲突
            if self.last_state is not None and self.last_action_idx is not None:
                # 确保last_state和state都是numpy数组且维度正确
                last_state = np.array(self.last_state, dtype=np.float32)
                current_state = np.array(state, dtype=np.float32)
                
                # 确保维度匹配
                if len(last_state) != self.state_dim:
                    last_state = np.zeros(self.state_dim, dtype=np.float32)
                if len(current_state) != self.state_dim:
                    current_state = np.zeros(self.state_dim, dtype=np.float32)
                
                # 确保last_action_idx是整数
                action_idx_value = self.last_action_idx
                if isinstance(action_idx_value, tuple):
                    # 如果意外是tuple，记录错误并使用默认值
                    self.logger.error(
                        f"智能体 {self.agent_id} last_action_idx是tuple而不是整数: {action_idx_value}，"
                        f"使用随机动作索引"
                    )
                    action_idx_value = random.randint(0, self.action_dim - 1)
                elif not isinstance(action_idx_value, (int, np.integer)):
                    # 如果不是整数类型，尝试转换
                    try:
                        action_idx_value = int(action_idx_value)
                    except (ValueError, TypeError):
                        self.logger.error(
                            f"智能体 {self.agent_id} 无法将last_action_idx转换为整数: {action_idx_value}，"
                            f"使用随机动作索引"
                        )
                        action_idx_value = random.randint(0, self.action_dim - 1)
                
                # 确保动作索引在有效范围内
                action_idx_value = max(0, min(int(action_idx_value), self.action_dim - 1))
                
                experience = Experience(
                    state=last_state,
                    action=int(action_idx_value),  # 确保是整数
                    reward=0.0,  # 在update中设置实际奖励
                    next_state=current_state,
                    done=False,
                    agent_id=self.agent_id,
                    timestamp=time.time()
                )
                self.last_experience = experience
            
            # 更新状态记录（确保是numpy数组）
            self.last_state = np.array(state, dtype=np.float32)
            self.last_action_idx = int(action_idx)  # 存储动作索引（整数）
            
            return action
            
        except Exception as e:
            self.logger.error(f"智能体 {self.agent_id} 决策失败: {e}", exc_info=True)
            import traceback
            self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
            # 失败时返回当前位置（不移动）
            return self.x, self.y
    
    def update(self, environment: Any) -> bool:
        """
        更新智能体状态并学习
        
        Args:
            environment: 环境对象
            
        Returns:
            是否存活
        """
        try:
            # 记录更新前的状态（用于奖励计算）
            prev_sugar = self._last_sugar
            prev_position = self._last_position
            
            # 调用父类更新逻辑（移动、收集糖、新陈代谢）
            is_alive = super().update(environment)
            
            # 完善并存储经验数据
            if self.last_experience is not None:
                # 计算奖励（基于实际变化）
                reward = self._calculate_reward()
                self.last_experience.reward = reward
                
                # 如果智能体死亡，标记为终止状态
                if not is_alive:
                    self.last_experience.done = True
                
                # 添加到经验回放
                self.replay_buffer.add(self.last_experience)
                self.logger.debug(f"智能体 {self.agent_id} 存储经验, 奖励: {reward:.3f}")
                self.last_experience = None
            
            # 学习（基于配置的频率）
            if self.training_step % self.config.train_frequency == 0:
                self.learn({})
            
            # 更新探索率（在learn中也会更新，但这里确保即使不学习也更新）
            self._update_epsilon()
            
            # 递增训练步数（只在update中递增，避免重复计数）
            self.training_step += 1
            
            return is_alive
            
        except Exception as e:
            self.logger.error(f"智能体 {self.agent_id} 更新失败: {e}", exc_info=True)
            # 更新失败时标记为死亡
            self.status = AgentStatus.DEAD
            return False

    def _update_epsilon(self) -> None:
        """更新探索率"""
        self.epsilon = max(
            self.config.epsilon_end, 
            self.epsilon * self.config.epsilon_decay
        )
        self.training_stats['exploration_rate'].append(self.epsilon)
        
        # 定期记录探索率
        if self.training_step % 100 == 0:
            self.logger.debug(f"智能体 {self.agent_id} 探索率: {self.epsilon:.3f}")
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        从经验中学习
        
        Args:
            experience: 经验数据（在IQL中主要使用回放缓冲区）
        """
        # 检查是否开始学习
        if len(self.replay_buffer) < max(self.config.learning_starts, self.config.batch_size):
            return
        
        # 定期学习
        if self.training_step % self.config.train_frequency == 0:
            loss, td_errors = self._update_network()
            
            if loss is not None:
                self.training_stats['losses'].append(loss)
                self.training_stats['td_errors'].extend(td_errors)
        
        # 定期更新目标网络
        if self.training_step % self.config.target_update_frequency == 0:
            self._update_target_network()
        
        # 衰减探索率（在update中也会更新，这里确保即使不学习也更新）
        self.epsilon = max(
            self.config.epsilon_end, 
            self.epsilon * self.config.epsilon_decay
        )
        self.training_stats['exploration_rate'].append(self.epsilon)
        
        # 重置噪声网络的噪声（如果使用）
        if self.config.network_type == "noisy":
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # 注意：training_step在update中递增，这里只递增learning_steps
        self.learning_steps += 1
    
    def _update_network(self) -> Tuple[Optional[float], List[float]]:
        """更新Q网络"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None, []
        
        try:
            # 从回放缓冲区采样
            batch = self.replay_buffer.sample(self.config.batch_size)
            
            # 转换数据为张量，确保正确的形状和类型
            states = torch.FloatTensor(batch['states'])
            actions = torch.LongTensor(batch['actions'])
            rewards = torch.FloatTensor(batch['rewards'])
            next_states = torch.FloatTensor(batch['next_states'])
            dones = torch.BoolTensor(batch['dones'])
            weights = batch['weights']
            indices = batch['indices']
            
            # 输入验证和维度修复
            batch_size = states.size(0)
            
            # 确保states是2D: [batch_size, state_dim]
            if states.dim() == 1:
                states = states.unsqueeze(0)
                batch_size = 1
            elif states.dim() > 2:
                states = states.view(batch_size, -1)
            
            # 确保actions是1D: [batch_size]
            if actions.dim() == 0:
                actions = actions.unsqueeze(0)
            elif actions.dim() > 1:
                actions = actions.squeeze()
            
            # 确保actions在有效范围内
            actions = torch.clamp(actions, 0, self.action_dim - 1)
            
            # 确保rewards是1D: [batch_size]
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            elif rewards.dim() > 1:
                rewards = rewards.squeeze()
            
            # 确保next_states是2D: [batch_size, state_dim]
            if next_states.dim() == 1:
                next_states = next_states.unsqueeze(0)
            elif next_states.dim() > 2:
                next_states = next_states.view(batch_size, -1)
            
            # 确保dones是1D: [batch_size]
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
            elif dones.dim() > 1:
                dones = dones.squeeze()
            
            # 验证状态维度
            if states.size(1) != self.state_dim:
                self.logger.warning(
                    f"状态维度不匹配: 期望 {self.state_dim}, 得到 {states.size(1)}. "
                    f"将调整状态维度"
                )
                # 如果状态维度不匹配，尝试调整
                if states.size(1) > self.state_dim:
                    states = states[:, :self.state_dim]
                    if next_states.size(1) > self.state_dim:
                        next_states = next_states[:, :self.state_dim]
                else:
                    # 填充零
                    padding = torch.zeros(batch_size, self.state_dim - states.size(1), device=states.device)
                    states = torch.cat([states, padding], dim=1)
                    if next_states.size(1) < self.state_dim:
                        next_padding = torch.zeros(batch_size, self.state_dim - next_states.size(1), device=next_states.device)
                        next_states = torch.cat([next_states, next_padding], dim=1)
            
            # 计算当前Q值
            q_values = self.q_network(states)  # [batch_size, action_dim]
            
            # 确保actions是正确的形状用于gather
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)  # [batch_size, 1]
            
            # 验证actions维度
            if actions.size(0) != batch_size:
                self.logger.error(f"动作批次大小不匹配: {actions.size(0)} != {batch_size}")
                return None, []
            
            current_q_values = q_values.gather(1, actions)  # [batch_size, 1]
            
            # 计算目标Q值
            with torch.no_grad():
                if self.config.use_double_dqn:
                    # Double DQN: 使用在线网络选择动作，目标网络评估
                    next_q_values_online = self.q_network(next_states)  # [batch_size, action_dim]
                    next_actions = next_q_values_online.argmax(1, keepdim=True)  # [batch_size, 1]
                    next_q_values = self.target_network(next_states).gather(1, next_actions)  # [batch_size, 1]
                else:
                    # 标准DQN: 直接使用目标网络的最大值
                    next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]  # [batch_size, 1]
                
                # 确保rewards和dones的形状正确
                if rewards.dim() == 1:
                    rewards = rewards.unsqueeze(1)  # [batch_size, 1]
                if dones.dim() == 1:
                    dones = dones.unsqueeze(1)  # [batch_size, 1]
                
                # 计算目标Q值，考虑终止状态
                target_q_values = rewards + (
                    self.discount_factor * next_q_values * (~dones)
                )
            
            # 计算TD误差和损失
            td_errors = (target_q_values - current_q_values).abs().squeeze()  # [batch_size]
            
            # 确保weights的形状正确
            if isinstance(weights, torch.Tensor):
                if weights.dim() == 1:
                    weights = weights.unsqueeze(1)  # [batch_size, 1]
                elif weights.dim() == 0:
                    weights = weights.unsqueeze(0).unsqueeze(0)
            else:
                weights = torch.ones(batch_size, 1)
            
            # 计算加权损失
            loss = (weights * self.loss_fn(current_q_values, target_q_values)).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            
            self.optimizer.step()
            
            # 更新优先级（基于TD误差）
            td_errors_np = td_errors.detach().cpu().numpy()
            if td_errors_np.ndim == 0:
                td_errors_np = np.array([td_errors_np])
            
            # 确保indices和td_errors_np的长度匹配
            if len(indices) != len(td_errors_np):
                self.logger.warning(
                    f"优先级更新时索引和误差长度不匹配: {len(indices)} != {len(td_errors_np)}"
                )
                min_len = min(len(indices), len(td_errors_np))
                indices = indices[:min_len]
                td_errors_np = td_errors_np[:min_len]
            
            self.replay_buffer.update_priorities(indices, td_errors_np.tolist())
            
            # 记录训练统计
            self.training_stats['losses'].append(loss.item())
            self.training_stats['td_errors'].extend(td_errors_np.tolist())
            
            # 定期记录训练进度
            if self.training_step % 100 == 0:
                self.logger.info(
                    f"智能体 {self.agent_id} 训练步数: {self.training_step}, "
                    f"损失: {loss.item():.4f}, "
                    f"平均Q值: {current_q_values.mean().item():.4f}, "
                    f"平均TD误差: {td_errors.mean().item():.4f}"
                )
            
            return loss.item(), td_errors_np.tolist()
            
        except Exception as e:
            self.logger.error(f"网络更新失败: {str(e)}", exc_info=True)
            import traceback
            self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return None, []
    
    def _update_target_network(self) -> None:
        """更新目标网络（软更新）"""
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def _calculate_reward(self) -> float:
        """
        计算奖励值（使用统一的奖励计算器）
        
        Returns:
            奖励值
        """
        current_pos = (self.x, self.y)
        
        # 使用统一的奖励计算器
        reward_result = self.reward_calculator.calculate_reward(
            prev_sugar=self._last_sugar,
            current_sugar=self.sugar,
            prev_position=self._last_position,
            current_position=current_pos,
            status=self.status,
            age=self.age,
            total_collected=self.total_collected,
            visited_positions=self.visited_positions,
            environment_size=self.environment_size,
            component_breakdown=False
        )
        
        reward = reward_result['reward']
        
        # 更新探索记录（奖励计算器不修改visited_positions）
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)
            self.logger.debug(f"智能体 {self.agent_id} 探索新位置 {current_pos}")
        
        # 记录糖收集量
        sugar_change = self.sugar - self._last_sugar
        if sugar_change > 0:
            self._sugar_collected_this_step = sugar_change
        
        # 记录奖励用于分析
        self.training_stats['rewards'].append(reward)
        
        # 更新状态记录
        self._last_sugar = self.sugar
        self._last_position = current_pos
        
        return reward
    
    def _process_observation(self, observation: ObservationSpace) -> np.ndarray:
        """
        处理观察为神经网络输入（优化版本，带错误处理）
        
        Args:
            observation: 观察空间
            
        Returns:
            处理后的状态向量
        """
        try:
            # 更新环境大小（如果可用）
            if hasattr(observation, 'environment_size') and observation.environment_size > 0:
                if not self._environment_size_initialized or self.environment_size != observation.environment_size:
                    self.environment_size = observation.environment_size
                    self._environment_size_initialized = True
                    self.logger.debug(f"智能体 {self.agent_id} 环境大小更新为: {self.environment_size}")
            
            features = []
            
            # 1. 局部视野特征（糖分布）
            if hasattr(observation, 'local_view') and observation.local_view is not None:
                local_view = observation.local_view.flatten()
                max_sugar = getattr(self.config, 'max_sugar', 10.0)
                if max_sugar <= 0:
                    max_sugar = 1.0
                local_view = local_view / max_sugar
                # 检查NaN和Inf
                local_view = np.nan_to_num(local_view, nan=0.0, posinf=1.0, neginf=0.0)
                features.extend(local_view)
            else:
                # 如果local_view不存在，使用零向量
                view_size = (self.vision_range * 2 + 1) ** 2
                features.extend([0.0] * view_size)
            
            # 2. 智能体内部状态
            agent_features = [
                max(0.0, min(1.0, self.sugar / 50.0)),  # 归一化糖量，限制在[0,1]
                min(self.age / 1000.0, 1.0),  # 归一化年龄，上限1000
                max(0.0, min(1.0, self.metabolism_rate / 3.0)),  # 归一化新陈代谢
                max(0.0, min(1.0, self.vision_range / 8.0)),  # 归一化视野范围
                min(len(self.visited_positions) / 200.0, 1.0),  # 探索进度归一化
                min(self.total_collected / 500.0, 1.0)  # 总收集糖量归一化
            ]
            features.extend(agent_features)
            
            # 3. 全局环境特征
            if hasattr(observation, 'global_stats') and observation.global_stats:
                global_stats = observation.global_stats
                global_features = [
                    min(global_stats.get('avg_sugar', 0) / 10.0, 1.0),
                    min(global_stats.get('total_sugar', 0) / 5000.0, 1.0),
                    min(len(global_stats.get('agents_by_type', {})) / 20.0, 1.0),
                    max(0.0, min(1.0, global_stats.get('agent_diversity', 0)))  # 智能体多样性
                ]
                features.extend(global_features)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 4. 位置和移动特征
            grid_center = self.environment_size // 2
            pos_features = [
                max(0.0, min(1.0, self.x / self.environment_size)),  # 绝对位置x
                max(0.0, min(1.0, self.y / self.environment_size)),  # 绝对位置y
                max(-1.0, min(1.0, (self.x - grid_center) / self.environment_size)),  # 相对中心位置x
                max(-1.0, min(1.0, (self.y - grid_center) / self.environment_size)),  # 相对中心位置y
            ]
            features.extend(pos_features)
            
            # 5. 时间特征（周期性）
            if hasattr(observation, 'step'):
                step = observation.step
                # 添加周期性时间特征，帮助学习时间模式
                features.extend([
                    np.sin(2 * np.pi * step / 100),
                    np.cos(2 * np.pi * step / 100),
                    min(step / 10000.0, 1.0)  # 线性时间特征，限制在[0,1]
                ])
            else:
                features.extend([0.0, 1.0, 0.0])
            
            # 转换为numpy数组
            state = np.array(features, dtype=np.float32)
            
            # 检查NaN和Inf
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                self.logger.warning(f"智能体 {self.agent_id} 状态包含NaN/Inf，已修复")
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 维度处理 - 确保状态维度匹配网络输入
            state_len = len(state)
            if state_len != self.state_dim:
                if state_len > self.state_dim:
                    # 截断到指定维度
                    state = state[:self.state_dim]
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"状态向量被截断: {state_len} -> {self.state_dim} "
                            f"(智能体 {self.agent_id})"
                        )
                elif state_len < self.state_dim:
                    # 填充到指定维度
                    padding = np.zeros(self.state_dim - state_len, dtype=np.float32)
                    state = np.concatenate([state, padding])
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"状态向量被填充: {state_len} -> {self.state_dim} "
                            f"(智能体 {self.agent_id})"
                        )
            
            # 最终验证输出维度
            if len(state) != self.state_dim:
                self.logger.error(
                    f"状态维度错误: {len(state)} != {self.state_dim} "
                    f"(智能体 {self.agent_id})，使用零向量"
                )
                state = np.zeros(self.state_dim, dtype=np.float32)
            
            return state
            
        except Exception as e:
            self.logger.error(f"智能体 {self.agent_id} 状态处理失败: {e}", exc_info=True)
            # 返回零向量作为后备
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def _action_idx_to_position(self, action_idx: int) -> Tuple[int, int]:
        """
        将动作索引转换为位置坐标
        
        Args:
            action_idx: 动作索引
            
        Returns:
            目标位置 (x, y)
        """
        if 0 <= action_idx < len(self.directions):
            dx, dy = self.directions[action_idx]
        else:
            dx, dy = 0, 0  # 默认不动
        
        new_x = (self.x + dx) % self.environment_size
        new_y = (self.y + dy) % self.environment_size
        
        return new_x, new_y
    
    def get_q_values(self, observation: ObservationSpace) -> np.ndarray:
        """
        获取Q值（实现抽象方法）
        
        Args:
            observation: 观察空间
            
        Returns:
            Q值数组
        """
        try:
            state = self._process_observation(observation)
            
            # 验证状态维度
            if len(state) != self.state_dim:
                self.logger.warning(
                    f"获取Q值时状态维度不匹配: {len(state)} != {self.state_dim}"
                )
                return np.zeros(self.action_dim, dtype=np.float32)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # 验证张量维度
                if state_tensor.size(1) != self.state_dim:
                    self.logger.error(f"状态张量维度错误: {state_tensor.size(1)} != {self.state_dim}")
                    return np.zeros(self.action_dim, dtype=np.float32)
                
                q_values = self.q_network(state_tensor)
                
                # 验证Q值维度
                if q_values.size(1) != self.action_dim:
                    self.logger.error(f"Q值维度错误: {q_values.size(1)} != {self.action_dim}")
                    return np.zeros(self.action_dim, dtype=np.float32)
                
                return q_values.numpy().flatten()
        except Exception as e:
            self.logger.error(f"获取Q值失败: {e}", exc_info=True)
            return np.zeros(self.action_dim, dtype=np.float32)
    
    def get_policy(self, observation: ObservationSpace) -> np.ndarray:
        """
        获取策略（实现抽象方法）
        
        Args:
            observation: 观察空间
            
        Returns:
            策略概率分布
        """
        # 对于Q学习，策略是基于Q值的epsilon-贪婪
        q_values = self.get_q_values(observation)
        
        # 创建epsilon-贪婪策略
        policy = np.ones(self.action_dim) * (self.epsilon / self.action_dim)
        best_action = np.argmax(q_values)
        policy[best_action] += (1.0 - self.epsilon)
        
        return policy
    
    def get_q_value_map(self, environment: Any, sample_density: int = 2) -> np.ndarray:
        """
        获取整个环境的Q值热图
        
        Args:
            environment: 环境对象
            sample_density: 采样密度（每隔多少个格子采样一次，1表示全部采样）
            
        Returns:
            Q值热图，形状为 (grid_size, grid_size)，值为最大Q值
        """
        if self.q_network is None:
            return np.zeros((self.environment_size, self.environment_size), dtype=np.float32)
        
        try:
            q_map = np.zeros((self.environment_size, self.environment_size), dtype=np.float32)
            
            # 保存当前位置
            original_x, original_y = self.x, self.y
            
            # 遍历环境中的每个位置（按采样密度）
            for x in range(0, self.environment_size, sample_density):
                for y in range(0, self.environment_size, sample_density):
                    # 临时设置智能体位置
                    self.x, self.y = x, y
                    
                    # 获取该位置的观察
                    observation = self.observe(environment)
                    
                    # 计算Q值
                    q_values = self.get_q_values(observation)
                    
                    # 记录最大Q值
                    q_map[x, y] = np.max(q_values) if len(q_values) > 0 else 0.0
                    
                    # 如果采样密度>1，填充周围区域
                    if sample_density > 1:
                        for dx in range(sample_density):
                            for dy in range(sample_density):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.environment_size and 0 <= ny < self.environment_size:
                                    q_map[nx, ny] = q_map[x, y]
            
            # 恢复原始位置
            self.x, self.y = original_x, original_y
            
            return q_map
        except Exception as e:
            self.logger.error(f"获取Q值热图失败: {e}", exc_info=True)
            # 恢复位置
            self.x, self.y = original_x, original_y
            return np.zeros((self.environment_size, self.environment_size), dtype=np.float32)
    
    def get_network_state(self, observation: ObservationSpace) -> Dict[str, Any]:
        """
        获取网络内部状态（隐藏层激活、策略分布等）
        
        Args:
            observation: 观察空间
            
        Returns:
            包含网络内部状态的字典
        """
        if self.q_network is None:
            return {
                'hidden_activations': [],
                'q_values': np.zeros(self.action_dim),
                'policy': np.ones(self.action_dim) / self.action_dim,
                'network_initialized': False
            }
        
        try:
            state = self._process_observation(observation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 获取Q值
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                q_values_np = q_values.numpy().flatten()
            
            # 获取隐藏层激活（如果网络支持）
            hidden_activations = []
            if hasattr(self.q_network, 'network'):
                # 遍历网络层，提取激活值
                x = state_tensor
                for i, layer in enumerate(self.q_network.network):
                    if isinstance(layer, (nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Tanh)):
                        x = layer(x)
                        if isinstance(layer, nn.Linear):
                            # 记录线性层的输出（激活前）
                            hidden_activations.append(x.detach().numpy().flatten())
            
            # 计算策略分布
            policy = self.get_policy(observation)
            
            return {
                'hidden_activations': hidden_activations,
                'q_values': q_values_np,
                'policy': policy,
                'network_initialized': True,
                'epsilon': self.epsilon,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }
        except Exception as e:
            self.logger.error(f"获取网络状态失败: {e}", exc_info=True)
            return {
                'hidden_activations': [],
                'q_values': np.zeros(self.action_dim),
                'policy': np.ones(self.action_dim) / self.action_dim,
                'network_initialized': False,
                'error': str(e)
            }
    
    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        if len(self.training_stats['q_values']) == 0:
            avg_q = 0.0
            avg_loss = 0.0
        else:
            avg_q = np.mean(self.training_stats['q_values'][-100:])  # 最近100步
            avg_loss = np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0.0
        
        return {
            'agent_id': self.agent_id,
            'training_steps': self.training_step,
            'learning_steps': self.learning_steps,
            'epsilon': self.epsilon,
            'avg_q_value': avg_q,
            'avg_loss': avg_loss,
            'replay_buffer_size': len(self.replay_buffer),
            'exploration_rate': self.epsilon
        }
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        
        self.logger.info(f"模型已从 {filepath} 加载")
    
    def reset(self, x: int, y: int) -> None:
        """重置智能体状态"""
        super().reset(x, y)
        
        # 重置IQL特定状态
        self.last_state = None
        self.last_action_idx = None  # 使用last_action_idx而不是last_action
        self.last_experience = None
        self._last_sugar = self.sugar
        self._last_position = (self.x, self.y)
        self._sugar_collected_this_step = 0.0
        
        # 重置探索记录（保留初始位置）
        self.visited_positions = set([(x, y)])
        
        # 重置探索率（可选，根据需求决定）
        # self.epsilon = self.config.epsilon_start
        
        self.logger.info(f"智能体 {self.agent_id} 已重置到位置 ({x}, {y})")

    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        info = {
            'q_network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'target_network_parameters': sum(p.numel() for p in self.target_network.parameters()),
            'network_type': self.config.network_type,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        # 添加梯度信息
        total_grad_norm = 0.0
        for p in self.q_network.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        info['gradient_norm'] = total_grad_norm ** 0.5
        
        return info

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """获取详细指标（增强版，包含完整诊断信息）"""
        base_info = super().get_agent_info()
        training_info = self.get_training_info()
        network_info = self.get_network_info()
        
        # 组合所有信息
        detailed_info = {
            **base_info,
            **training_info,
            **network_info,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_capacity': self.config.replay_buffer_size,
            'replay_buffer_utilization': len(self.replay_buffer) / self.config.replay_buffer_size if self.config.replay_buffer_size > 0 else 0.0,
            'exploration_rate': self.epsilon,
            'visited_positions_count': len(self.visited_positions),
            'exploration_ratio': len(self.visited_positions) / (self.environment_size ** 2) if self.environment_size > 0 else 0.0,
            'learning_enabled': len(self.replay_buffer) >= self.config.learning_starts,
            'config': {
                'learning_rate': self.config.learning_rate,
                'gamma': self.config.gamma,
                'epsilon': self.epsilon,
                'batch_size': self.config.batch_size,
                'train_frequency': self.config.train_frequency
            }
        }
        
        return detailed_info