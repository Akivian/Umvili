"""
QMIX多智能体强化学习算法实现

基于：Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (2018)
通过混合网络将个体Q值组合为联合Q值，满足单调性约束，实现集中式训练分布式执行。
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
from src.marl.networks import QMIXAgentNetwork, QMIXMixingNetwork
from src.marl.replay_buffer import Experience, PriorityReplayBuffer
from src.config.simulation_config import SimulationConfig
# 向后兼容：使用默认值
GRID_SIZE = 80


@dataclass
class QMIXConfig:
    """QMIX算法配置数据类"""
    # 网络配置
    state_dim: int = 128
    action_dim: int = 8
    agent_hidden_dims: List[int] = None
    mixing_hidden_dim: int = 32
    
    # 学习参数
    learning_rate: float = 0.0005
    gamma: float = 0.99
    tau: float = 0.005  # 目标网络软更新参数
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999
    
    # 训练参数
    batch_size: int = 32
    replay_buffer_size: int = 10000
    learning_starts: int = 1000
    train_frequency: int = 4
    target_update_frequency: int = 200
    
    def __post_init__(self):
        if self.agent_hidden_dims is None:
            self.agent_hidden_dims = [64, 64]


class QMIXAgent(LearningAgent):
    """
    QMIX智能体
    
    特性：
    - 集中式训练，分布式执行
    - 单调值函数分解
    - 联合动作值函数学习
    """
    
    def __init__(self, 
                 x: int, 
                 y: int, 
                 agent_id: int,
                 config: Optional[QMIXConfig] = None,
                 trainer: Optional[Any] = None,
                 name: Optional[str] = None):
        """
        初始化QMIX智能体
        
        Args:
            x, y, agent_id: 基础参数
            config: QMIX配置
            trainer: QMIX训练器实例
            name: 智能体名称
        """
        # Algorithm-specific configuration
        self.config = config or QMIXConfig()
        
        # BaseAgent uses its own BaseAgentConfig internally; we don't override it here.
        super().__init__(
            x=x, y=y, agent_id=agent_id, 
            agent_type=AgentType.QMIX,
            learning_rate=self.config.learning_rate,
            discount_factor=self.config.gamma,
            name=name or f"qmix_{agent_id}"
        )
        
        # QMIX特定参数
        self.state_dim = self.config.state_dim
        self.action_dim = self.action_space.get_action_count() if self.action_space else 8
        
        # 训练器集成
        self.trainer = trainer  # 保存训练器引用

        # 探索策略
        self.epsilon = self.config.epsilon_start
        
        # 训练状态
        self.training_step = 0
        self.last_state = None
        # 注意：BaseAgent.last_action 用于记录环境中的目标坐标 (x, y)
        # QMIX 自身需要一个“离散动作索引”，不能与 BaseAgent.last_action 混用
        self.last_action_idx = None  # 上一步的离散动作索引（0 ~ action_dim-1）
        self.last_global_state = None  # 新增：存储全局状态
        
        # 环境信息
        self.environment_size = GRID_SIZE
        
        # 动作映射（8个方向）
        self.directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        # 经验缓冲区（用于临时存储）
        self.pending_experience = None

        # 注意：网络将在QMIX训练器中统一初始化
        self.q_network = None
        self.target_q_network = None
        
        # 训练统计
        self.training_stats = {
            'q_values': [],
            'losses': [],
            'td_errors': [],
            'exploration_rate': []
        }

        # 状态记录
        self._last_sugar = self.sugar
        self._last_position = (self.x, self.y)
        self.visited_positions = set([(self.x, self.y)])  # 探索记录
        
        # 初始化奖励计算器（QMIX使用稍不同的配置）
        reward_config = RewardConfig(
            sugar_collection_multiplier=1.0,  # QMIX使用较小的乘数
            sugar_consumption_penalty=0.3,
            exploration_reward=0.3,
            survival_reward=0.1,
            efficiency_multiplier=0.2,
            wealth_threshold_1=25.0,
            wealth_reward_1=0.0,  # QMIX不使用财富奖励
            wealth_threshold_2=50.0,
            wealth_reward_2=0.0,
            movement_cost=0.05,
            death_penalty=2.0,  # QMIX使用较小的死亡惩罚
            boundary_penalty=0.0,  # QMIX不使用边界惩罚
            min_reward=-2.0,
            max_reward=2.0
        )
        self.reward_calculator = RewardCalculator(reward_config)
        
        self.logger.info(f"初始化QMIX智能体 {self.name}")
    
    def set_trainer(self, trainer: Any) -> None:
        """
        设置训练器（用于后期绑定）
        
        Args:
            trainer: QMIX训练器实例
        """
        self.trainer = trainer
        self.logger.info(f"智能体 {self.agent_id} 绑定训练器")

    def set_networks(self, q_network: nn.Module, target_q_network: nn.Module) -> None:
        """
        设置网络（由训练器调用）
        
        Args:
            q_network: 在线Q网络
            target_q_network: 目标Q网络
        """
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.logger.debug(f"智能体 {self.agent_id} 网络已设置")
    
    def decide_action(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        ε-贪婪策略选择动作，并准备经验数据
        
        Args:
            observation: 观察空间
            
        Returns:
            目标位置 (x, y)
        """
        # 处理观察为状态向量
        state = self._process_observation(observation)
        global_state = self._get_global_state(observation)
        
        # 探索 vs 利用
        if random.random() < self.epsilon:
            # 随机探索
            action_idx = random.randint(0, self.action_dim - 1)
            self.logger.debug(f"QMIX智能体 {self.agent_id} 随机探索: 动作 {action_idx}")
        else:
            # 利用Q值选择最佳动作
            if self.q_network is not None:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax().item()
                
                self.logger.debug(f"QMIX智能体 {self.agent_id} 利用: "
                                f"动作 {action_idx}, Q值 {q_values.max().item():.3f}")
            else:
                # 网络未初始化，随机选择
                action_idx = random.randint(0, self.action_dim - 1)
                self.logger.debug(f"QMIX智能体 {self.agent_id} 网络未初始化，随机选择: 动作 {action_idx}")
        
        # 转换为环境动作
        action = self._action_idx_to_position(action_idx)
        
        # 准备经验数据（在update方法中完善并发送给训练器）
        # 这里使用上一步的离散动作索引，而不是 BaseAgent.last_action（坐标）
        if self.last_state is not None and self.last_action_idx is not None:
            self.pending_experience = {
                'agent_id': self.agent_id,
                'state': self.last_state,
                'action': self.last_action_idx,
                'reward': 0.0,  # 在update中设置实际奖励
                'next_state': state,
                'done': False,
                'global_state': self.last_global_state,
                'next_global_state': global_state,
                'timestamp': time.time()
            }
        
        # 更新状态记录（保存离散动作索引）
        self.last_state = state
        self.last_action_idx = action_idx
        self.last_global_state = global_state
        
        return action
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        学习函数 - QMIX智能体不直接学习，由训练器统一训练
        
        Args:
            experience: 经验数据（在QMIX中由训练器处理）
        """
        # QMIX是集中式训练，个体智能体不直接进行梯度更新
        # 但可以记录一些统计信息
        if 'reward' in experience:
            self.training_stats['td_errors'].append(experience.get('td_error', 0))
        
        self.training_step += 1
    
    def get_last_experience(self) -> Optional[Dict[str, Any]]:
        """获取上一步的经验数据"""
        if self.last_state is not None and self.last_action_idx is not None:
            return {
                'agent_id': self.agent_id,
                'state': self.last_state,
                'action': self.last_action_idx,
                'timestamp': time.time()
            }
        return None
    
    def update(self, environment: Any) -> bool:
        """
        更新智能体状态，收集经验并发送给训练器
        
        Args:
            environment: 环境对象
            
        Returns:
            是否存活
        """
        try:
            if self.status != AgentStatus.ALIVE:
                return False
            
            # 记录更新前的状态（用于奖励计算）
            prev_sugar = self._last_sugar
            prev_position = self._last_position
            
            # 调用父类更新逻辑（移动、收集糖、新陈代谢）
            is_alive = super().update(environment)
            
            # 计算奖励
            reward = self._calculate_reward(prev_sugar, prev_position)
            
            # 完善并发送经验数据
            if self.pending_experience is not None:
                self.pending_experience['reward'] = reward
                self.pending_experience['done'] = not is_alive
                
                # 发送给训练器
                if self.trainer and hasattr(self.trainer, 'store_individual_experience'):
                    self.trainer.store_individual_experience(self.pending_experience)
                    self.logger.debug(f"智能体 {self.agent_id} 发送经验到训练器, 奖励: {reward:.3f}")
                else:
                    self.logger.warning(f"智能体 {self.agent_id} 无法发送经验: 训练器未设置或方法不存在")
                
                self.pending_experience = None
            
            # 更新探索率
            self.update_epsilon()
            
            # 更新状态记录
            self._last_sugar = self.sugar
            self._last_position = (self.x, self.y)
            
            return is_alive
            
        except Exception as e:
            self.logger.error(f"QMIX智能体 {self.agent_id} 更新失败: {e}", exc_info=True)
            # 更新失败时标记为死亡
            self.status = AgentStatus.DEAD
            return False

    def _calculate_reward(self, prev_sugar: float, prev_position: Tuple[int, int]) -> float:
        """
        计算奖励值（使用统一的奖励计算器）
        
        Args:
            prev_sugar: 上一步的糖量
            prev_position: 上一步的位置
            
        Returns:
            奖励值
        """
        current_pos = (self.x, self.y)
        
        # 使用统一的奖励计算器
        # 获取本次收集的spice量（从agent的harvest记录中）
        spice_collected = getattr(self, '_last_harvest_spice', 0.0)
        
        reward_result = self.reward_calculator.calculate_reward(
            prev_sugar=prev_sugar,
            current_sugar=self.sugar,
            prev_position=prev_position,
            current_position=current_pos,
            status=self.status,
            spice_collected=spice_collected,
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
        
        return reward
    
    def update_epsilon(self) -> None:
        """更新探索率"""
        self.epsilon = max(
            self.config.epsilon_end, 
            self.epsilon * self.config.epsilon_decay
        )
        self.training_stats['exploration_rate'].append(self.epsilon)
    
    def _process_observation(self, observation: ObservationSpace) -> np.ndarray:
        """
        处理观察为神经网络输入
        
        Args:
            observation: 原始观察
            
        Returns:
            处理后的状态向量
        """
        try:
            features = []
            
            # 1. 局部视野特征（糖分布）
            if hasattr(observation, 'local_view') and observation.local_view is not None:
                local_view = observation.local_view.flatten()
                # 归一化局部视野（使用默认最大值，如果配置中没有）
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
                min(self.age / 500.0, 1.0),  # 年龄归一化，上限500
                max(0.0, min(1.0, self.metabolism_rate / 3.0)),  # 归一化新陈代谢
                max(0.0, min(1.0, self.vision_range / 8.0)),  # 归一化视野范围
                min(len(self.visited_positions) / 100.0, 1.0)  # 探索进度归一化
            ]
            features.extend(agent_features)
            
            # 3. 相对位置特征
            grid_center = self.environment_size // 2 if self.environment_size > 0 else 40
            rel_x = max(-1.0, min(1.0, (self.x - grid_center) / max(1, self.environment_size)))
            rel_y = max(-1.0, min(1.0, (self.y - grid_center) / max(1, self.environment_size)))
            features.extend([rel_x, rel_y])
            
            # 4. 时间特征（周期性）
            if hasattr(observation, 'step'):
                step = observation.step
                # 添加周期性时间特征
                features.extend([
                    np.sin(2 * np.pi * step / 100),
                    np.cos(2 * np.pi * step / 100)
                ])
            else:
                features.extend([0.0, 1.0])
            
            # 转换为固定维度状态向量
            state = np.array(features, dtype=np.float32)
            
            # 检查NaN和Inf
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                self.logger.warning(f"智能体 {self.agent_id} 状态包含NaN/Inf，已修复")
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 维度处理
            state_len = len(state)
            if state_len != self.state_dim:
                if state_len > self.state_dim:
                    # 截断到指定维度
                    state = state[:self.state_dim]
                    # 为了避免日志过于嘈杂，这里只在 DEBUG 级别输出详细说明，
                    # 和 IQLAgent 的行为保持一致，正常运行时不会反复刷 WARNING。
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
            
            # 验证输出维度
            if len(state) != self.state_dim:
                self.logger.error(
                    f"状态维度错误: {len(state)} != {self.state_dim}，使用零向量"
                )
                state = np.zeros(self.state_dim, dtype=np.float32)
            
            return state
            
        except Exception as e:
            self.logger.error(f"智能体 {self.agent_id} 状态处理失败: {e}", exc_info=True)
            # 返回零向量作为后备
            return np.zeros(self.state_dim, dtype=np.float32)

    def _get_global_state(self, observation: ObservationSpace) -> np.ndarray:
        """
        获取全局状态表示（用于混合网络）
        
        Args:
            observation: 观察空间
            
        Returns:
            全局状态向量
        """
        try:
            global_features = []
            
            # 1. 环境统计信息
            if hasattr(observation, 'global_stats') and observation.global_stats:
                stats = observation.global_stats
                global_features.extend([
                    min(stats.get('total_sugar', 0) / 5000.0, 1.0),  # 环境总糖量归一化
                    min(stats.get('avg_sugar', 0) / 10.0, 1.0),      # 平均糖量归一化
                    max(0.0, min(1.0, stats.get('agent_diversity', 0)))  # 智能体多样性
                ])
            else:
                global_features.extend([0.0, 0.0, 0.0])
            
            # 2. 智能体群体统计
            if hasattr(observation, 'global_stats') and observation.global_stats and 'agents_by_type' in observation.global_stats:
                type_counts = observation.global_stats['agents_by_type']
                total_agents = sum(type_counts.values())
                
                # 各类型智能体比例
                for agent_type in ['rule_based', 'iql', 'qmix', 'conservative', 'exploratory', 'adaptive']:
                    count = type_counts.get(agent_type, 0)
                    proportion = count / max(1, total_agents)
                    global_features.append(max(0.0, min(1.0, proportion)))
            else:
                # 如果没有统计信息，使用零向量
                global_features.extend([0.0] * 6)
            
            # 3. 时间信息
            if hasattr(observation, 'step'):
                step = observation.step
                global_features.extend([
                    min(step / 10000.0, 1.0),  # 步数归一化
                    np.sin(2 * np.pi * step / 1000),
                    np.cos(2 * np.pi * step / 1000)
                ])
            else:
                global_features.extend([0.0, 0.0, 1.0])
            
            # 转换为numpy数组
            global_state = np.array(global_features, dtype=np.float32)
            
            # 检查NaN和Inf
            if np.any(np.isnan(global_state)) or np.any(np.isinf(global_state)):
                self.logger.warning(f"智能体 {self.agent_id} 全局状态包含NaN/Inf，已修复")
                global_state = np.nan_to_num(global_state, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 确保维度一致性
            if len(global_state) < self.state_dim:
                padding = np.zeros(self.state_dim - len(global_state), dtype=np.float32)
                global_state = np.concatenate([global_state, padding])
            elif len(global_state) > self.state_dim:
                global_state = global_state[:self.state_dim]
            
            return global_state
            
        except Exception as e:
            self.logger.error(f"智能体 {self.agent_id} 获取全局状态失败: {e}", exc_info=True)
            # 返回零向量作为后备
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def _action_idx_to_position(self, action_idx: int) -> Tuple[int, int]:
        """
        将动作索引转换为位置坐标
        
        Args:
            action_idx: 动作索引 (0-7对应8个方向)
            
        Returns:
            目标位置 (x, y)
        """
        # 验证动作索引有效性
        if not (0 <= action_idx < len(self.directions)):
            self.logger.warning(f"无效动作索引: {action_idx}, 使用默认动作")
            action_idx = 0  # 默认动作
        
        dx, dy = self.directions[action_idx]
        
        # 计算新位置（考虑边界环绕）
        new_x = (self.x + dx) % self.environment_size
        new_y = (self.y + dy) % self.environment_size
        
        # 记录移动决策
        self.logger.debug(f"智能体 {self.agent_id} 从 ({self.x}, {self.y}) 移动到 ({new_x}, {new_y})")
        
        return new_x, new_y
    
    def get_q_values(self, observation: ObservationSpace) -> np.ndarray:
        """
        获取Q值（实现抽象方法）
        
        Args:
            observation: 观察空间
            
        Returns:
            Q值数组
        """
        if self.q_network is None:
            # 网络未初始化，返回随机值
            return np.random.randn(self.action_dim)
        
        state = self._process_observation(observation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
        return q_values.numpy().flatten()
    
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
    
    def get_policy(self, observation: ObservationSpace) -> np.ndarray:
        """
        获取策略（实现抽象方法）
        
        Args:
            observation: 观察空间
            
        Returns:
            策略概率分布
        """
        # 对于QMIX，策略是基于Q值的epsilon-贪婪
        q_values = self.get_q_values(observation)
        
        # 创建epsilon-贪婪策略
        policy = np.ones(self.action_dim) * (self.epsilon / self.action_dim)
        best_action = np.argmax(q_values)
        policy[best_action] += (1.0 - self.epsilon)
        
        return policy

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return len(self.directions)
    
    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        return {
            'agent_id': self.agent_id,
            'epsilon': self.epsilon,
            'training_steps': self.training_step,
            'visited_positions': len(self.visited_positions),
            'exploration_rate': self.epsilon,
            'status': self.status.value
        }
    
    def reset(self, x: int, y: int) -> None:
        """重置智能体状态"""
        super().reset(x, y)
        
        # 重置QMIX特定状态
        self.last_state = None
        self.last_action_idx = None  # 离散动作索引
        self.last_global_state = None
        self.pending_experience = None
        self.epsilon = self.config.epsilon_start
        
        # 重置探索记录（保留初始位置）
        self.visited_positions = set([(x, y)])
        
        # 重置状态记录
        self._last_sugar = self.sugar
        self._last_position = (self.x, self.y)
        
        self.logger.debug(f"QMIX智能体 {self.agent_id} 已重置")