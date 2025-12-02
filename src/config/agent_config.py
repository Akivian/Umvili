"""
Agent Configuration

Configuration classes for all agent types.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.agent_base import BaseAgentConfig
from src.config.defaults import DEFAULT_AGENT_CONFIGS


# Re-export BaseAgentConfig for convenience
__all__ = ['BaseAgentConfig', 'RuleBasedAgentConfig', 'IQLConfig', 'QMIXConfig', 'AgentDefaultConfigs']


@dataclass
class RuleBasedAgentConfig:
    """
    规则型智能体配置数据类
    
    用于RuleBasedAgent及其子类（Conservative, Exploratory, Adaptive）。
    """
    vision_range: int = 5
    metabolism_rate: float = 1.0
    initial_sugar: float = 15.0
    movement_strategy: str = "greedy"  # greedy, conservative, exploratory
    exploration_rate: float = 0.1  # 探索概率
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.vision_range < 1:
            raise ValueError("vision_range must be >= 1")
        if self.metabolism_rate < 0:
            raise ValueError("metabolism_rate must be >= 0")
        if self.initial_sugar < 0:
            raise ValueError("initial_sugar must be >= 0")
        if self.movement_strategy not in ['greedy', 'conservative', 'exploratory']:
            raise ValueError(f"movement_strategy must be one of ['greedy', 'conservative', 'exploratory'], got {self.movement_strategy}")
        if not (0 <= self.exploration_rate <= 1):
            raise ValueError("exploration_rate must be between 0 and 1")
    
    def to_base_config(self) -> BaseAgentConfig:
        """
        转换为BaseAgentConfig
        
        Returns:
            BaseAgentConfig对象
        """
        return BaseAgentConfig(
            vision_range=self.vision_range,
            metabolism_rate=self.metabolism_rate,
            initial_sugar=self.initial_sugar
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RuleBasedAgentConfig':
        """从字典创建配置对象"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})


@dataclass
class IQLConfig:
    """
    IQL智能体配置数据类
    
    用于独立Q学习智能体。
    """
    # 基础属性
    vision_range: int = 5
    metabolism_rate: float = 0.8
    initial_sugar: float = 15.0
    
    # 网络配置
    state_dim: int = 64
    action_dim: int = 8
    hidden_dims: Optional[List[int]] = None
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
        """后处理：设置默认值"""
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("epsilon_end <= epsilon_start and both in [0, 1]")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IQLConfig':
        """从字典创建配置对象"""
        # 处理hidden_dims（可能是列表或None）
        if 'hidden_dims' in config_dict and isinstance(config_dict['hidden_dims'], list):
            config_dict = config_dict.copy()
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})


@dataclass
class QMIXConfig:
    """
    QMIX算法配置数据类
    
    用于QMIX智能体。
    """
    # 基础属性
    vision_range: int = 5
    metabolism_rate: float = 0.8
    initial_sugar: float = 15.0
    
    # 网络配置
    state_dim: int = 128
    action_dim: int = 8
    agent_hidden_dims: Optional[List[int]] = None
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
        """后处理：设置默认值"""
        if self.agent_hidden_dims is None:
            self.agent_hidden_dims = [64, 64]
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("epsilon_end <= epsilon_start and both in [0, 1]")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QMIXConfig':
        """从字典创建配置对象"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})


# Agent default configurations dictionary
AgentDefaultConfigs = DEFAULT_AGENT_CONFIGS

