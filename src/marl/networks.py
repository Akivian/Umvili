"""
神经网络组件

包含各种强化学习算法所需的神经网络架构。
采用模块化设计，支持灵活的架构配置。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
import math


class MLP(nn.Module):
    """
    多层感知机基础模块
    
    支持灵活的层配置、激活函数和归一化层。
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = None,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 use_batch_norm: bool = False,
                 output_activation: Optional[str] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # 激活函数映射
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "selu": nn.SELU()
        }
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 批量归一化
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 激活函数
            if activation in activations:
                layers.append(activations[activation])
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 输出激活函数
        if output_activation and output_activation in activations:
            layers.append(activations[output_activation])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，适合ReLU激活函数
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    # 使用小的正偏置，避免死神经元
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QNetwork(MLP):
    """
    Q值网络
    
    用于学习状态-动作值函数。
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = None,
                 **kwargs):
        super().__init__(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN网络
    
    基于：Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
    分别学习状态价值和动作优势，提高策略评估能力。
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # 共享特征层
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_network(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 组合价值和优势
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class PolicyNetwork(MLP):
    """
    策略网络
    
    用于策略梯度方法，输出动作概率分布。
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = None,
                 **kwargs):
        super().__init__(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            output_activation="softmax",  # 输出动作概率
            **kwargs
        )


class ValueNetwork(MLP):
    """
    价值函数网络
    
    用于学习状态价值函数。
    """
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dims: List[int] = None,
                 **kwargs):
        super().__init__(
            input_dim=state_dim,
            output_dim=1,  # 输出单一价值
            hidden_dims=hidden_dims,
            **kwargs
        )


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络
    
    共享特征提取层，分别输出策略和价值。
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # 共享特征层
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Actor (策略)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (价值)
        self.critic = nn.Linear(prev_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_network(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


class MixingNetwork(nn.Module):
    """
    混合网络 (用于QMIX)
    
    基于：Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (2018)
    将个体Q值混合为联合Q值，满足单调性约束。
    """
    
    def __init__(self, 
                 num_agents: int, 
                 state_dim: int, 
                 mixing_dim: int = 32):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_dim = mixing_dim
        
        # 超网络生成混合权重
        self.hyper_w1 = nn.Linear(state_dim, num_agents * mixing_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_dim)
        
        # 偏置
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """初始化超网络权重"""
        for module in [self.hyper_w1, self.hyper_w2, self.hyper_b1, self.hyper_b2]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            agent_qs: 个体Q值 [batch_size, num_agents]
            states: 全局状态 [batch_size, state_dim]
            
        Returns:
            联合Q值 [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        
        # 第一层权重和偏置
        w1 = torch.abs(self.hyper_w1(states))  # 绝对值确保单调性
        w1 = w1.view(-1, self.num_agents, self.mixing_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.mixing_dim)
        
        # 第二层权重和偏置
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.mixing_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)
        
        # 前向传播
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # [batch, 1, mixing_dim]
        q_total = torch.bmm(hidden, w2) + b2  # [batch, 1, 1]
        
        return q_total.squeeze(-1).squeeze(-1)  # [batch_size]


class NoisyLinear(nn.Module):
    """
    噪声线性层
    
    基于：Fortunato et al. "Noisy Networks for Exploration" (2017)
    通过参数噪声实现更高效的探索。
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # 噪声参数
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """重置可学习参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成缩放噪声"""
        noise = torch.randn(size)
        return noise.sign().mul(noise.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（带噪声）"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """
    带噪声网络的Q网络
    
    使用噪声层替代传统线性层，实现更高效的探索。
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        prev_dim = state_dim
        
        # 隐藏层使用噪声线性层
        for hidden_dim in hidden_dims:
            layers.extend([
                NoisyLinear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # 输出层也使用噪声线性层
        layers.append(NoisyLinear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def reset_noise(self):
        """重置所有噪声层的噪声"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class QMIXAgentNetwork(nn.Module):
    """
    QMIX个体智能体网络
    每个智能体有独立的Q网络
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QMIXMixingNetwork(nn.Module):
    """
    QMIX混合网络
    将个体Q值混合为联合Q值，保证单调性
    """
    
    def __init__(self, 
                 num_agents: int,
                 state_dim: int,
                 mixing_hidden_dim: int = 32):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        
        # 超网络生成混合权重
        self.hyper_w1 = nn.Linear(state_dim, num_agents * mixing_hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_hidden_dim)
        
        # 超网络生成偏置
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 初始化超网络权重
        for module in [self.hyper_w1, self.hyper_w2]:
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
        
        for module in [self.hyper_b1, self.hyper_b2]:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            agent_qs: 个体Q值 [batch_size, num_agents]
            states: 全局状态 [batch_size, state_dim]
            
        Returns:
            联合Q值 [batch_size]
        """
        batch_size = agent_qs.size(0)
        
        # 输入验证
        if agent_qs.dim() != 2:
            raise ValueError(f"agent_qs应该是2维张量，得到 {agent_qs.dim()} 维")
        if states.dim() != 2:
            raise ValueError(f"states应该是2维张量，得到 {states.dim()} 维")
        if agent_qs.size(1) != self.num_agents:
            raise ValueError(f"agent_qs第二维应该是 {self.num_agents}，得到 {agent_qs.size(1)}")
        if states.size(1) != self.state_dim:
            raise ValueError(f"states第二维应该是 {self.state_dim}，得到 {states.size(1)}")
        
        try:
            # 第一层权重和偏置（使用绝对值保证单调性）
            w1 = torch.abs(self.hyper_w1(states))  # [batch_size, num_agents * mixing_hidden_dim]
            w1 = w1.view(-1, self.num_agents, self.mixing_hidden_dim)  # [batch_size, num_agents, mixing_hidden_dim]
            
            b1 = self.hyper_b1(states)  # [batch_size, mixing_hidden_dim]
            b1 = b1.view(-1, 1, self.mixing_hidden_dim)  # [batch_size, 1, mixing_hidden_dim]
            
            # 第二层权重和偏置
            w2 = torch.abs(self.hyper_w2(states))  # [batch_size, mixing_hidden_dim]
            w2 = w2.view(-1, self.mixing_hidden_dim, 1)  # [batch_size, mixing_hidden_dim, 1]
            
            b2 = self.hyper_b2(states)  # [batch_size, 1]
            b2 = b2.view(-1, 1, 1)  # [batch_size, 1, 1]
            
            # 前向传播
            # [batch_size, 1, num_agents] * [batch_size, num_agents, mixing_hidden_dim]
            hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # [batch_size, 1, mixing_hidden_dim]
            
            # [batch_size, 1, mixing_hidden_dim] * [batch_size, mixing_hidden_dim, 1]
            q_total = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]
            
            return q_total.squeeze(-1).squeeze(-1)  # [batch_size]
            
        except Exception as e:
            raise RuntimeError(f"QMIX混合网络前向传播失败: {str(e)}")
        
    