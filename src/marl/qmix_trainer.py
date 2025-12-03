"""
QMIX训练器
负责集中式训练所有QMIX智能体
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import random
import time
import logging

from src.marl.networks import QMIXAgentNetwork, QMIXMixingNetwork
from src.marl.replay_buffer import Experience, PriorityReplayBuffer


class QMIXTrainer:
    """
    QMIX训练器 - 集中式训练所有QMIX智能体
    
    基于：Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (2018)
    """
    
    def __init__(self, 
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any] = None):
        """
        初始化QMIX训练器
        
        Args:
            num_agents: 智能体数量
            state_dim: 状态维度
            action_dim: 动作维度
            config: 训练配置
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 配置参数
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 32)
        self.gamma = self.config.get('gamma', 0.99)
        self.learning_rate = self.config.get('learning_rate', 0.0005)
        self.tau = self.config.get('tau', 0.005)
        self.target_update_frequency = self.config.get('target_update_frequency', 200)
        self.learning_starts = self.config.get('learning_starts', 1000)
        
        # 网络配置
        self.agent_hidden_dims = self.config.get('agent_hidden_dims', [64, 64])
        self.mixing_hidden_dim = self.config.get('mixing_hidden_dim', 32)
        
        # 初始化网络
        self._initialize_networks()
        
        # 经验回放
        self.replay_buffer = PriorityReplayBuffer(
            capacity=self.config.get('replay_buffer_size', 10000)
        )
        
        # 将网络移动到设备（必须在网络初始化之后）
        self._move_networks_to_device()
        
        # 智能体经验临时存储
        self.agent_experiences: Dict[int, Dict[str, Any]] = {}
        self.global_state_history: deque = deque(maxlen=1000)
        self.current_global_state: Optional[np.ndarray] = None
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.last_target_update = 0
        
        # 训练统计
        self.training_stats = {
            'losses': [],
            'q_values': [],
            'td_errors': [],
            'mixing_losses': [],
            'individual_losses': []
        }
        
        # 日志（必须在任何会使用logger的方法之前初始化）
        self.logger = logging.getLogger('QMIXTrainer')

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"QMIX训练器初始化: {num_agents}个智能体, 设备: {self.device}")
    
    def _initialize_networks(self) -> None:
        """初始化所有网络"""
        # 个体Q网络（所有智能体共享参数）
        self.q_network = QMIXAgentNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.agent_hidden_dims
        )
        
        self.target_q_network = QMIXAgentNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.agent_hidden_dims
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # 混合网络
        self.mixing_network = QMIXMixingNetwork(
            num_agents=self.num_agents,
            state_dim=self.state_dim,  # 全局状态维度
            mixing_hidden_dim=self.mixing_hidden_dim
        )
        
        self.target_mixing_network = QMIXMixingNetwork(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            mixing_hidden_dim=self.mixing_hidden_dim
        )
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + list(self.mixing_network.parameters()),
            lr=self.learning_rate
        )
        
        # 损失函数
        self.loss_fn = nn.MSELoss()

    def _move_networks_to_device(self) -> None:
        """将网络移动到指定设备"""
        # 确保网络已初始化
        if not hasattr(self, 'q_network') or self.q_network is None:
            raise RuntimeError("网络未初始化，无法移动到设备")
        
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.mixing_network.to(self.device)
        self.target_mixing_network.to(self.device)
        
        # 该方法在 __init__ 早期就会被调用，因此要防御性地访问 logger
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(f"网络已移动到设备: {self.device}")
    
    def store_individual_experience(self, experience: Dict[str, Any]) -> None:
        """
        存储单个智能体的经验
        
        Args:
            experience: 单个智能体的经验字典，包含：
                - agent_id: 智能体ID
                - state: 个体状态
                - action: 动作
                - reward: 奖励
                - next_state: 下一状态
                - done: 终止标志
                - global_state: 全局状态
                - next_global_state: 下一全局状态
        """
        try:
            agent_id = experience.get('agent_id')
            if agent_id is None:
                self.logger.error("经验缺少agent_id字段")
                return
            
            # 验证经验数据
            if not self._validate_experience(experience):
                self.logger.warning(f"智能体 {agent_id} 的经验验证失败，已跳过")
                return
            
            # 存储个体经验（覆盖旧的经验，确保使用最新的）
            self.agent_experiences[agent_id] = experience
            
            # 如果收集到所有智能体的经验，形成联合经验
            if len(self.agent_experiences) == self.num_agents:
                self._form_joint_experience()
        except Exception as e:
            self.logger.error(f"存储智能体经验失败: {e}", exc_info=True)

    def _validate_experience(self, experience: Dict[str, Any]) -> bool:
        """
        验证经验数据的有效性
        
        Args:
            experience: 经验数据
            
        Returns:
            是否有效
        """
        required_fields = ['state', 'action', 'reward', 'next_state', 'done']
        
        # 检查必需字段
        for field in required_fields:
            if field not in experience:
                self.logger.error(f"经验缺少必需字段: {field}")
                return False
        
        # 检查状态是否是numpy数组
        if not isinstance(experience['state'], np.ndarray):
            self.logger.error(f"状态不是numpy数组: {type(experience['state'])}")
            return False
        
        # 检查状态维度（允许1D数组）
        state_shape = experience['state'].shape
        if len(state_shape) != 1:
            self.logger.error(f"状态应该是1D数组，得到形状: {state_shape}")
            return False
        
        if state_shape[0] != self.state_dim:
            # 允许维度不匹配，但会记录警告（在_form_joint_experience中修复）
            self.logger.debug(
                f"状态维度不匹配: {state_shape[0]} != {self.state_dim}，"
                f"将在形成联合经验时修复"
            )
        
        # 检查动作类型和范围
        action = experience['action']
        if not isinstance(action, (int, np.integer)):
            try:
                action = int(action)
            except (ValueError, TypeError):
                self.logger.error(f"动作无法转换为整数: {action}")
                return False
        
        if not (0 <= action < self.action_dim):
            self.logger.warning(
                f"动作超出范围: {action} not in [0, {self.action_dim})，"
                f"将裁剪到有效范围"
            )
            experience['action'] = max(0, min(int(action), self.action_dim - 1))
        
        # 检查next_state
        if not isinstance(experience['next_state'], np.ndarray):
            self.logger.error(f"下一状态不是numpy数组: {type(experience['next_state'])}")
            return False
        
        return True

    def store_global_experience(self, global_state: np.ndarray) -> None:
        """
        存储全局状态
        
        Args:
            global_state: 全局状态向量
        """
        if global_state.shape != (self.state_dim,):
            self.logger.warning(f"全局状态维度不匹配: {global_state.shape} != ({self.state_dim},)")
            # 尝试调整维度
            if len(global_state) < self.state_dim:
                padding = np.zeros(self.state_dim - len(global_state))
                global_state = np.concatenate([global_state, padding])
            else:
                global_state = global_state[:self.state_dim]
        
        self.current_global_state = global_state
        self.global_state_history.append(global_state)

    def _form_joint_experience(self) -> None:
        """
        形成联合经验并存入回放缓冲区
        """
        try:
            # 按智能体ID排序以确保一致性
            agent_ids = sorted(self.agent_experiences.keys())
            
            # 检查是否所有智能体都有经验
            # 注意：如果智能体死亡，可能无法收集到所有智能体的经验
            # 在这种情况下，我们仍然可以形成部分联合经验，或者跳过
            if len(agent_ids) != self.num_agents:
                self.logger.warning(
                    f"智能体经验不完整: {len(agent_ids)}/{self.num_agents}。"
                    f"可能有些智能体已死亡或未更新。跳过此次联合经验形成。"
                )
                # 清空当前经验，等待下一轮
                self.agent_experiences.clear()
                return
            
            # 提取联合经验组件
            states, actions, rewards, next_states, dones = [], [], [], [], []
            global_state = None
            next_global_state = None
            
            for agent_id in agent_ids:
                exp = self.agent_experiences[agent_id]
                
                # 验证经验数据
                if not isinstance(exp['state'], np.ndarray):
                    self.logger.error(f"智能体 {agent_id} 的状态不是numpy数组")
                    self.agent_experiences.clear()
                    return
                
                # 确保状态维度正确
                if len(exp['state']) != self.state_dim:
                    self.logger.warning(
                        f"智能体 {agent_id} 状态维度不匹配: "
                        f"{len(exp['state'])} != {self.state_dim}，尝试修复"
                    )
                    # 修复状态维度
                    if len(exp['state']) > self.state_dim:
                        exp['state'] = exp['state'][:self.state_dim]
                        exp['next_state'] = exp['next_state'][:self.state_dim] if len(exp['next_state']) > self.state_dim else exp['next_state']
                    else:
                        padding = np.zeros(self.state_dim - len(exp['state']), dtype=np.float32)
                        exp['state'] = np.concatenate([exp['state'], padding])
                        if len(exp['next_state']) < self.state_dim:
                            next_padding = np.zeros(self.state_dim - len(exp['next_state']), dtype=np.float32)
                            exp['next_state'] = np.concatenate([exp['next_state'], next_padding])
                
                states.append(exp['state'])
                actions.append(exp['action'])
                rewards.append(exp['reward'])
                next_states.append(exp['next_state'])
                dones.append(exp['done'])
                
                # 使用第一个智能体的全局状态（假设所有智能体相同）
                if global_state is None:
                    global_state = exp.get('global_state')
                    next_global_state = exp.get('next_global_state', global_state)  # 后备
                    
                    # 验证全局状态
                    if global_state is not None and len(global_state) != self.state_dim:
                        self.logger.warning(
                            f"全局状态维度不匹配: {len(global_state)} != {self.state_dim}，尝试修复"
                        )
                        if len(global_state) > self.state_dim:
                            global_state = global_state[:self.state_dim]
                            if next_global_state is not None and len(next_global_state) > self.state_dim:
                                next_global_state = next_global_state[:self.state_dim]
                        else:
                            padding = np.zeros(self.state_dim - len(global_state), dtype=np.float32)
                            global_state = np.concatenate([global_state, padding])
                            if next_global_state is not None and len(next_global_state) < self.state_dim:
                                next_padding = np.zeros(self.state_dim - len(next_global_state), dtype=np.float32)
                                next_global_state = np.concatenate([next_global_state, next_padding])
            
            # 如果没有全局状态，使用当前存储的
            if global_state is None and self.current_global_state is not None:
                global_state = self.current_global_state
                # 估计下一全局状态（简化）
                next_global_state = self._estimate_next_global_state()
            
            # 创建联合经验
            joint_experience = Experience(
                state=np.array(states),  # [num_agents, state_dim]
                action=np.array(actions),  # [num_agents]
                reward=np.array(rewards),  # [num_agents]
                next_state=np.array(next_states),  # [num_agents, state_dim]
                done=np.array(dones),  # [num_agents]
                global_state=global_state if global_state is not None else np.zeros(self.state_dim),
                next_global_state=next_global_state if next_global_state is not None else np.zeros(self.state_dim),
                timestamp=time.time()
            )
            
            # 存入回放缓冲区
            self.replay_buffer.add(joint_experience)
            
            # 清空临时存储
            self.agent_experiences.clear()
            
            self.logger.debug(f"形成联合经验，缓冲区大小: {len(self.replay_buffer)}")
            
        except Exception as e:
            self.logger.error(f"形成联合经验失败: {str(e)}")
            # 清空有问题的经验
            self.agent_experiences.clear()

    def _estimate_next_global_state(self) -> np.ndarray:
        """
        估计下一全局状态（简化实现）
        
        Returns:
            估计的下一全局状态
        """
        if len(self.global_state_history) > 0:
            return self.global_state_history[-1]  # 使用最新状态
        return np.zeros(self.state_dim)
    
    def train_step(self) -> Optional[float]:
        """
        执行QMIX训练步骤
        
        Returns:
            损失值（如果进行了训练）
        """
        # 检查是否开始训练
        if (len(self.replay_buffer) < max(self.batch_size, self.learning_starts) or 
            self.training_step % 4 != 0):  # 每4步训练一次
            return None
        
        try:
            # 从回放缓冲区采样
            batch = self.replay_buffer.sample(self.batch_size)
            
            # 转换数据为张量并移动到设备
            states = torch.FloatTensor(batch['states']).to(self.device)  # [batch_size, num_agents, state_dim]
            actions = torch.LongTensor(batch['actions']).to(self.device)  # [batch_size, num_agents]
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)  # [batch_size, num_agents]
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)  # [batch_size, num_agents, state_dim]
            dones = torch.BoolTensor(batch['dones']).to(self.device)  # [batch_size, num_agents]
            global_states = torch.FloatTensor(batch['global_state']).to(self.device)  # [batch_size, state_dim]
            next_global_states = torch.FloatTensor(batch['next_global_state']).to(self.device)  # [batch_size, state_dim]
            weights = batch['weights'].to(self.device)  # [batch_size]
            
            batch_size = states.size(0)
            
            # 验证批次维度
            if states.dim() != 3 or states.size(1) != self.num_agents or states.size(2) != self.state_dim:
                self.logger.error(
                    f"状态批次维度错误: 期望 [batch_size, {self.num_agents}, {self.state_dim}], "
                    f"得到 {list(states.shape)}"
                )
                return None
            
            if actions.dim() != 2 or actions.size(1) != self.num_agents:
                self.logger.error(
                    f"动作批次维度错误: 期望 [batch_size, {self.num_agents}], "
                    f"得到 {list(actions.shape)}"
                )
                return None
            
            if rewards.dim() != 2 or rewards.size(1) != self.num_agents:
                self.logger.error(
                    f"奖励批次维度错误: 期望 [batch_size, {self.num_agents}], "
                    f"得到 {list(rewards.shape)}"
                )
                return None
            
            if next_states.dim() != 3 or next_states.size(1) != self.num_agents or next_states.size(2) != self.state_dim:
                self.logger.error(
                    f"下一状态批次维度错误: 期望 [batch_size, {self.num_agents}, {self.state_dim}], "
                    f"得到 {list(next_states.shape)}"
                )
                return None
            
            if global_states.dim() != 2 or global_states.size(1) != self.state_dim:
                self.logger.error(
                    f"全局状态批次维度错误: 期望 [batch_size, {self.state_dim}], "
                    f"得到 {list(global_states.shape)}"
                )
                return None
            
            # === 计算当前Q值 ===
            current_q_values = []
            for agent_idx in range(self.num_agents):
                # 获取每个智能体的状态和动作
                agent_states = states[:, agent_idx, :]  # [batch_size, state_dim]
                agent_actions = actions[:, agent_idx]  # [batch_size]
                
                # 确保动作在有效范围内
                agent_actions = torch.clamp(agent_actions, 0, self.action_dim - 1)
                
                # 计算当前Q值
                agent_q = self.q_network(agent_states)  # [batch_size, action_dim]
                agent_current_q = agent_q.gather(1, agent_actions.unsqueeze(1))  # [batch_size, 1]
                current_q_values.append(agent_current_q)
            
            # 组合所有智能体的Q值 [batch_size, num_agents]
            current_q_values = torch.cat(current_q_values, dim=1)
            
            # 通过混合网络计算联合Q值
            current_q_tot = self.mixing_network(current_q_values, global_states)  # [batch_size]
            
            # === 计算目标Q值 ===
            with torch.no_grad():
                next_q_values = []
                for agent_idx in range(self.num_agents):
                    agent_next_states = next_states[:, agent_idx, :]  # [batch_size, state_dim]
                    
                    # 使用目标网络计算下一状态的最大Q值
                    next_agent_q = self.target_q_network(agent_next_states)  # [batch_size, action_dim]
                    next_agent_max_q = next_agent_q.max(1, keepdim=True)[0]  # [batch_size, 1]
                    next_q_values.append(next_agent_max_q)
                
                # 组合所有智能体的下一Q值 [batch_size, num_agents]
                next_q_values = torch.cat(next_q_values, dim=1)
                
                # 通过目标混合网络计算联合目标Q值
                next_q_tot = self.target_mixing_network(next_q_values, next_global_states)  # [batch_size]
                
                # 计算目标联合Q值
                # 使用所有智能体的奖励和 [batch_size]
                joint_rewards = rewards.sum(dim=1)
                # 所有智能体都结束时才视为终止 [batch_size]
                joint_dones = dones.all(dim=1)
                
                target_q_tot = joint_rewards + (self.gamma * next_q_tot * ~joint_dones)
            
            # === 计算损失 ===
            td_errors = (target_q_tot - current_q_tot).abs().detach().cpu().numpy()
            
            # 确保weights是1D张量
            if weights.dim() > 1:
                weights = weights.squeeze()
            if weights.dim() == 0:
                weights = weights.unsqueeze(0)
            
            # 确保weights和loss的维度匹配
            if weights.size(0) != batch_size:
                self.logger.warning(
                    f"权重维度不匹配: {weights.size(0)} != {batch_size}，使用均匀权重"
                )
                weights = torch.ones(batch_size, device=self.device)
            
            # 计算加权损失
            loss_per_sample = self.loss_fn(current_q_tot, target_q_tot)
            weighted_loss = (weights * loss_per_sample).mean()
            
            # === 反向传播 ===
            self.optimizer.zero_grad()
            weighted_loss.backward()
            
            # 梯度裁剪
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(self.q_network.parameters()) + list(self.mixing_network.parameters()),
                max_norm=10.0
            )
            
            self.optimizer.step()
            
            # === 更新优先级 ===
            # 确保td_errors是列表格式
            if isinstance(td_errors, np.ndarray):
                if td_errors.ndim == 0:
                    td_errors = [float(td_errors)]
                else:
                    td_errors = td_errors.tolist()
            elif not isinstance(td_errors, list):
                td_errors = [float(td_errors)]
            
            self.replay_buffer.update_priorities(batch['indices'], td_errors)
            
            # === 定期更新目标网络 ===
            if self.training_step - self.last_target_update >= self.target_update_frequency:
                self._soft_update_target_networks()
                self.last_target_update = self.training_step
            
            # === 记录统计信息 ===
            self.training_stats['losses'].append(weighted_loss.item())
            self.training_stats['q_values'].append(current_q_tot.mean().item())
            self.training_stats['td_errors'].extend(td_errors.tolist())
            self.training_stats['mixing_losses'].append(weighted_loss.item())
            
            self.training_step += 1
            
            if self.training_step % 100 == 0:
                self.logger.info(
                    f"训练步数: {self.training_step}, "
                    f"损失: {weighted_loss.item():.4f}, "
                    f"平均Q值: {current_q_tot.mean().item():.4f}, "
                    f"梯度范数: {total_norm:.4f}"
                )
            
            return weighted_loss.item()
            
        except Exception as e:
            self.logger.error(f"QMIX训练失败: {str(e)}", exc_info=True)
            return None
    
    def _soft_update_target_networks(self) -> None:
        """软更新目标网络"""
        with torch.no_grad():
            # 更新个体Q网络
            for target_param, param in zip(self.target_q_network.parameters(), 
                                        self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
            
            # 更新混合网络
            for target_param, param in zip(self.target_mixing_network.parameters(), 
                                        self.mixing_network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
        
        self.logger.debug(f"目标网络已更新 (τ={self.tau})")

    def hard_update_target_networks(self) -> None:
        """硬更新目标网络（完全复制）"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.logger.info("目标网络已硬更新")
    
    def get_networks_for_agent(self) -> Tuple[nn.Module, nn.Module]:
        """
        获取智能体使用的网络
        
        Returns:
            (q_network, target_q_network) 元组
        """
        return self.q_network, self.target_q_network

    def sync_agent_networks(self, agents: List[Any]) -> None:
        """
        同步所有智能体的网络
        
        Args:
            agents: QMIX智能体列表
        """
        for agent in agents:
            if hasattr(agent, 'set_networks'):
                agent.set_networks(self.q_network, self.target_q_network)
        
        self.logger.info(f"已同步 {len(agents)} 个智能体的网络")

    def get_training_stats(self) -> Dict[str, Any]:
        """获取详细的训练统计"""
        stats = {
            'training_steps': self.training_step,
            'episodes': self.episode_count,
            'replay_buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.capacity,
            'pending_experiences': len(self.agent_experiences),
        }
        
        # 计算滑动平均
        if self.training_stats['losses']:
            stats.update({
                'avg_loss': np.mean(self.training_stats['losses'][-100:]),
                'avg_q_value': np.mean(self.training_stats['q_values'][-100:]),
                'recent_loss': self.training_stats['losses'][-1] if self.training_stats['losses'] else 0,
                'recent_q_value': self.training_stats['q_values'][-1] if self.training_stats['q_values'] else 0,
            })
        
        if self.training_stats['td_errors']:
            stats['avg_td_error'] = np.mean(self.training_stats['td_errors'][-100:])
        
        # 添加网络统计
        stats.update(self._get_network_stats())
        
        return stats

    def _get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计"""
        stats = {}
        
        # Q网络参数统计
        q_params = list(self.q_network.parameters())
        if q_params:
            stats['q_network_params'] = sum(p.numel() for p in q_params)
            stats['q_network_grad_norm'] = self._calculate_gradient_norm(q_params)
        
        # 混合网络参数统计
        mixing_params = list(self.mixing_network.parameters())
        if mixing_params:
            stats['mixing_network_params'] = sum(p.numel() for p in mixing_params)
            stats['mixing_network_grad_norm'] = self._calculate_gradient_norm(mixing_params)
        
        return stats

    def _calculate_gradient_norm(self, parameters) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """获取详细诊断信息"""
        diagnostics = {
            'training': self.get_training_stats(),
            'networks': {
                'q_network': str(self.q_network),
                'mixing_network': str(self.mixing_network),
                'device': str(self.device)
            },
            'configuration': {
                'num_agents': self.num_agents,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'tau': self.tau
            }
        }
        return diagnostics
    
    def save_models(self, filepath: str) -> None:
        """
        保存模型（简化版本）
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'mixing_network_state_dict': self.mixing_network.state_dict(),
            'target_mixing_network_state_dict': self.target_mixing_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"QMIX模型已保存到: {filepath}")

    def load_models(self, filepath: str) -> None:
        """
        加载模型（简化版本）
        """
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.mixing_network.load_state_dict(checkpoint['mixing_network_state_dict'])
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_step = checkpoint['training_step']
        
        self.logger.info(f"QMIX模型已从 {filepath} 加载")

