"""
经验回放系统

实现高效的经验存储和采样机制，支持：
- 优先经验回放 (Prioritized Experience Replay)
- 多智能体经验管理
- 轨迹存储和采样
- 内存优化管理
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from collections import deque
import heapq
import logging


@dataclass
class Experience:
    """经验数据类"""
    state: np.ndarray
    action: Any  # 支持int或np.ndarray（用于多智能体）
    reward: Any  # 支持float或np.ndarray（用于多智能体）
    next_state: np.ndarray
    done: Any  # 支持bool或np.ndarray（用于多智能体）
    priority: float = 1.0
    timestamp: float = 0.0
    agent_id: Optional[int] = None
    global_state: Optional[np.ndarray] = None  # QMIX需要的全局状态
    next_global_state: Optional[np.ndarray] = None  # QMIX需要的下一全局状态


class PriorityReplayBuffer:
    """
    优先经验回放缓冲区
    
    基于：Schaul et al. "Prioritized Experience Replay" (2015)
    通过TD误差优先级提高学习效率。
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-5):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=均匀，1=完全优先)
            beta: 重要性采样权重参数
            beta_increment: beta的增量
            epsilon: 防止零优先级的小常数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 存储结构
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.size = 0
        
        # 最大优先级（用于新经验）
        self.max_priority = 1.0

        self.logger = logging.getLogger('PriorityReplayBuffer')
        
    def add(self, experience: Experience) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验对象
        """
        # 设置初始优先级（使用最大优先级）
        priority = self.max_priority ** self.alpha
        
        if len(self.buffer) < self.capacity:
            # 缓冲区未满，直接添加
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # 缓冲区已满，替换最旧的经验（FIFO）
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        # 更新位置指针
        self.position = (self.position + 1) % self.capacity
        
        # 更新大小
        self.size = len(self.buffer)
        
        # 更新最大优先级
        self.max_priority = max(self.max_priority, priority)
        
        self.logger.debug(f"经验已添加，缓冲区大小: {self.size}")
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            包含经验批次和权重的字典
        """
        if len(self.buffer) == 0:
            raise ValueError("缓冲区为空，无法采样")
        
        # 确保不采样超过缓冲区大小
        batch_size = min(batch_size, len(self.buffer))
        
        if batch_size == 0:
            raise ValueError("批次大小必须大于0")
        
        # 计算采样概率
        priorities_array = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities_array / priorities_array.sum()
        
        # 采样索引（如果batch_size等于缓冲区大小，使用replace=True避免错误）
        replace = batch_size > len(self.buffer) // 2  # 如果采样超过一半，允许重复
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=replace)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化
        
        # 收集经验
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'weights': torch.FloatTensor(weights),
            'indices': indices,
            'global_state': [],
            'next_global_state': []
        }
        
        for idx in indices:
            exp = self.buffer[idx]
            batch['states'].append(exp.state)
            batch['actions'].append(exp.action)
            batch['rewards'].append(exp.reward)
            batch['next_states'].append(exp.next_state)
            batch['dones'].append(exp.done)
            # 处理全局状态（如果存在）
            if hasattr(exp, 'global_state') and exp.global_state is not None:
                batch['global_state'].append(exp.global_state)
            else:
                # 如果没有全局状态，使用零向量（维度根据第一个状态推断）
                if len(batch['states']) > 0:
                    state_dim = len(batch['states'][0]) if isinstance(batch['states'][0], np.ndarray) else 128
                else:
                    state_dim = 128
                batch['global_state'].append(np.zeros(state_dim, dtype=np.float32))
            
            if hasattr(exp, 'next_global_state') and exp.next_global_state is not None:
                batch['next_global_state'].append(exp.next_global_state)
            else:
                # 如果没有下一全局状态，使用零向量
                if len(batch['next_states']) > 0:
                    state_dim = len(batch['next_states'][0]) if isinstance(batch['next_states'][0], np.ndarray) else 128
                else:
                    state_dim = 128
                batch['next_global_state'].append(np.zeros(state_dim, dtype=np.float32))
        
        # 转换为numpy数组，确保正确的数据类型和形状
        batch['states'] = np.array(batch['states'], dtype=np.float32)
        
        # 处理actions - 确保是整数数组
        actions_array = np.array(batch['actions'])
        if actions_array.dtype != np.int64 and actions_array.dtype != np.int32:
            # 如果是浮点数，转换为整数
            actions_array = actions_array.astype(np.int64)
        # 确保是1D数组
        if actions_array.ndim > 1:
            actions_array = actions_array.flatten()
        batch['actions'] = actions_array
        
        # 处理rewards - 确保是浮点数数组
        rewards_array = np.array(batch['rewards'])
        if rewards_array.dtype != np.float32 and rewards_array.dtype != np.float64:
            rewards_array = rewards_array.astype(np.float32)
        if rewards_array.ndim > 1:
            rewards_array = rewards_array.flatten()
        batch['rewards'] = rewards_array
        
        batch['next_states'] = np.array(batch['next_states'], dtype=np.float32)
        
        # 处理dones - 确保是布尔数组
        dones_array = np.array(batch['dones'])
        if dones_array.dtype != bool:
            dones_array = dones_array.astype(bool)
        if dones_array.ndim > 1:
            dones_array = dones_array.flatten()
        batch['dones'] = dones_array
        # 处理全局状态数组
        if batch['global_state']:
            batch['global_state'] = np.array(batch['global_state'])
        else:
            # 如果没有全局状态，创建默认零向量
            state_dim = len(batch['states'][0]) if len(batch['states']) > 0 else 128
            batch['global_state'] = np.zeros((batch_size, state_dim), dtype=np.float32)
        
        if batch['next_global_state']:
            batch['next_global_state'] = np.array(batch['next_global_state'])
        else:
            state_dim = len(batch['next_states'][0]) if len(batch['next_states']) > 0 else 128
            batch['next_global_state'] = np.zeros((batch_size, state_dim), dtype=np.float32)
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        self.logger.debug(f"采样 {batch_size} 条经验")
        
        return batch
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        更新经验的优先级
        
        Args:
            indices: 经验索引
            priorities: 新的优先级（通常是TD误差）
        """
        for idx, priority in zip(indices, priorities):
            # 添加小常数防止零优先级
            self.priorities[idx] = (abs(priority) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        return self.size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if self.size == 0:
            return {}
        
        priorities = self.priorities[:self.size]
        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'avg_priority': np.mean(priorities),
            'max_priority': np.max(priorities),
            'min_priority': np.min(priorities)
        }


class MultiAgentReplayBuffer:
    """
    多智能体经验回放缓冲区
    
    为每个智能体维护单独的经验缓冲区，
    支持集中式和分布式经验管理。
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffers: Dict[int, PriorityReplayBuffer] = {}
        self.logger = logging.getLogger('MultiAgentReplayBuffer')
    
    def add(self, agent_id: int, experience: Experience) -> None:
        """为指定智能体添加经验"""
        if agent_id not in self.buffers:
            self.buffers[agent_id] = PriorityReplayBuffer(self.capacity)
            self.logger.info(f"为智能体 {agent_id} 创建新的经验缓冲区")
        
        self.buffers[agent_id].add(experience)
    
    def sample(self, agent_id: int, batch_size: int) -> Optional[Dict[str, Any]]:
        """从指定智能体的缓冲区采样"""
        if agent_id not in self.buffers or len(self.buffers[agent_id]) == 0:
            self.logger.warning(f"智能体 {agent_id} 的缓冲区为空或不存在")
            return None
        
        return self.buffers[agent_id].sample(batch_size)
    
    def sample_all(self, batch_size: int) -> Dict[int, Dict[str, Any]]:
        """从所有智能体采样相同大小的批次"""
        batches = {}
        for agent_id, buffer in self.buffers.items():
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                if batch is not None:
                    batches[agent_id] = batch
        
        self.logger.debug(f"从 {len(batches)} 个智能体采样经验")
        return batches
    
    def update_priorities(self, agent_id: int, indices: List[int], priorities: List[float]) -> None:
        """更新指定智能体的经验优先级"""
        if agent_id in self.buffers:
            self.buffers[agent_id].update_priorities(indices, priorities)
        else:
            self.logger.warning(f"尝试更新不存在的智能体 {agent_id} 的优先级")
    
    def get_agent_buffer(self, agent_id: int) -> Optional[PriorityReplayBuffer]:
        """获取指定智能体的缓冲区"""
        return self.buffers.get(agent_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        stats = {
            'total_agents': len(self.buffers),
            'total_experiences': sum(len(buffer) for buffer in self.buffers.values()),
            'agent_stats': {}
        }
        
        for agent_id, buffer in self.buffers.items():
            stats['agent_stats'][agent_id] = {
                'buffer_size': len(buffer),
                'capacity': buffer.capacity,
                'utilization': len(buffer) / buffer.capacity
            }
        
        return stats
    
    def __len__(self) -> int:
        """返回总经验数"""
        return sum(len(buffer) for buffer in self.buffers.values())


class TrajectoryBuffer:
    """
    轨迹缓冲区
    
    存储完整的智能体轨迹，用于策略梯度方法。
    """
    
    def __init__(self):
        self.trajectories: Dict[int, List[Experience]] = {}
    
    def start_trajectory(self, agent_id: int) -> None:
        """开始新的轨迹"""
        self.trajectories[agent_id] = []
    
    def add_step(self, agent_id: int, experience: Experience) -> None:
        """添加轨迹步骤"""
        if agent_id not in self.trajectories:
            self.start_trajectory(agent_id)
        
        self.trajectories[agent_id].append(experience)
    
    def end_trajectory(self, agent_id: int) -> List[Experience]:
        """结束轨迹并返回"""
        if agent_id in self.trajectories:
            trajectory = self.trajectories[agent_id]
            del self.trajectories[agent_id]
            return trajectory
        return []
    
    def get_trajectory(self, agent_id: int) -> List[Experience]:
        """获取当前轨迹"""
        return self.trajectories.get(agent_id, [])