"""
高度优化的Simulation类 - 专注于核心模拟逻辑
移除了冗余组件，简化了架构，专注于MARL沙盘的核心功能
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import logging

from src.core.agent_base import BaseAgent, AgentStatus, AgentType
from src.core.agent_factory import get_agent_factory
from src.core.environment import SugarEnvironment
from src.config.simulation_config import SimulationConfig
# 向后兼容：使用默认值
GRID_SIZE = 80
CELL_SIZE = 10


class SimulationState(Enum):
    """简化的模拟状态枚举"""
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class SimulationMetrics:
    """核心指标数据类"""
    step: int = 0
    total_agents: int = 0
    alive_agents: int = 0
    avg_sugar: float = 0.0
    avg_age: float = 0.0
    total_environment_sugar: float = 0.0
    agent_diversity: float = 0.0
    agents_by_type: Dict[str, int] = field(default_factory=dict)
    avg_sugar_by_type: Dict[str, float] = field(default_factory=dict)
    # 按类型统计的动作分布：{agent_type: {action_idx: count}}
    action_distribution_by_type: Dict[str, Dict[int, int]] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    agent_type: str
    avg_loss: float = 0.0
    avg_q_value: float = 0.0
    avg_td_error: float = 0.0
    exploration_rate: float = 0.0
    training_steps: int = 0
    sample_count: int = 0
    recent_loss: float = 0.0
    recent_q_value: float = 0.0
    # 奖励相关统计（用于 Reward Trend 可视化）
    avg_reward: float = 0.0
    recent_reward: float = 0.0


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    fps: float = 0.0
    step_time: float = 0.0
    agent_update_time: float = 0.0
    memory_usage_mb: float = 0.0


class AgentManager:
    """
    简化的智能体管理器
    专注于智能体的生命周期管理
    """
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        # 用于分配全局唯一的 agent_id（即使中途有删除）
        self._agent_id_counter = 0
    
    def add_agent(self, agent: BaseAgent) -> None:
        """添加智能体"""
        self.agents.append(agent)
        # 确保计数器始终大于当前已存在的最大 ID
        self._agent_id_counter = max(self._agent_id_counter, agent.agent_id + 1)
    
    def remove_agent(self, agent: BaseAgent) -> None:
        """移除智能体"""
        if agent in self.agents:
            self.agents.remove(agent)
    
    def update_agents(self, environment) -> None:
        """更新所有智能体状态"""
        dead_agents = []
        
        for agent in self.agents:
            if agent.status != AgentStatus.ALIVE:
                continue
            
            try:
                is_alive = agent.update(environment)
                if not is_alive:
                    dead_agents.append(agent)
            except Exception as e:
                logging.warning(f"智能体 {agent.agent_id} 更新失败: {e}")
                dead_agents.append(agent)
        
        # 移除死亡智能体
        for agent in dead_agents:
            self.remove_agent(agent)
    
    def clear(self) -> None:
        """清空所有智能体"""
        self.agents.clear()
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """按类型获取智能体"""
        return [agent for agent in self.agents if agent.agent_type.value == agent_type]
    
    def __len__(self) -> int:
        return len(self.agents)

    def next_id(self) -> int:
        """获取下一个唯一的智能体 ID"""
        agent_id = self._agent_id_counter
        self._agent_id_counter += 1
        return agent_id


class MARLSimulation:
    """
    核心MARL模拟类
    
    特性：
    - 高度专注：只处理核心模拟逻辑
    - 性能优化：最小化不必要的计算
    - 模块化设计：清晰的职责分离
    - 易于扩展：支持新的智能体类型和算法
    """
    
    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        cell_size: int = CELL_SIZE,
        initial_agents: int = 50,
        sugar_growth_rate: float = 0.1,
        max_sugar: float = 10.0,
        agent_configs: Optional[List[Any]] = None
    ):
        """
        初始化模拟
        
        Args:
            grid_size: 网格大小
            cell_size: 单元格大小
            initial_agents: 初始智能体数量
            sugar_growth_rate: 糖生长速率
            max_sugar: 最大糖量
            agent_configs: 智能体配置列表
        """
        # 基础配置
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.initial_agents = initial_agents
        self.sugar_growth_rate = sugar_growth_rate
        self.max_sugar = max_sugar
        
        # 状态管理
        self.state = SimulationState.READY
        self.step_count = 0
        self.start_time = time.time()
        
        # 核心组件
        self.environment: Optional[SugarEnvironment] = None
        self.agent_manager = AgentManager()
        self.agent_factory = get_agent_factory()
        # Ensure factory uses the same grid size as this simulation
        try:
            self.agent_factory.set_grid_size(self.grid_size)
        except Exception:
            # Fail silently if factory doesn't support this (for safety)
            pass
        
        # 数据管理
        self.current_metrics = SimulationMetrics()
        self.metrics_history: Deque[SimulationMetrics] = deque(maxlen=500)
        self.performance_metrics = PerformanceMetrics()
        self.training_metrics: Dict[str, TrainingMetrics] = {}
        
        # MARL训练器支持
        self.marl_trainers: Dict[str, Any] = {}
        
        # 初始化 - 确保环境大小与配置一致
        self._initialize_environment()
        self._initialize_agents(agent_configs)
        
        # 性能监控
        self._frame_times: Deque[float] = deque(maxlen=60)
        self._last_frame_time = time.time()
        
        logging.info(f"MARL模拟系统初始化完成: 网格大小={grid_size}, 智能体数量={len(self.agent_manager)}")
    
    def _initialize_environment(self, env_config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化环境
        
        Args:
            env_config: 可选的环境配置字典，包含：
                - spice_growth_rate: Spice 生长速率
                - max_spice: Spice 最大值
                - hazard_decay_rate: Hazard 衰减速率
                - hazard_target_fraction: Hazard 目标覆盖比例
                - resource_enabled: 资源开关字典 {'sugar': bool, 'spice': bool, 'hazard': bool}
        """
        # 默认环境参数
        spice_growth_rate = 0.02
        max_spice = 6.0
        hazard_decay_rate = 0.01
        hazard_target_fraction = 0.09
        resource_enabled = {'sugar': True, 'spice': True, 'hazard': True}
        
        # 从配置中获取参数
        if env_config:
            spice_growth_rate = env_config.get('spice_growth_rate', spice_growth_rate)
            max_spice = env_config.get('max_spice', max_spice)
            hazard_decay_rate = env_config.get('hazard_decay_rate', hazard_decay_rate)
            hazard_target_fraction = env_config.get('hazard_target_fraction', hazard_target_fraction)
            resource_enabled = env_config.get('resource_enabled', resource_enabled)
        
        # 存储资源开关状态（供环境使用）
        self.resource_enabled = resource_enabled
        
        # 创建环境（如果资源被禁用，相关参数会被忽略）
        # 注意：SugarEnvironment 不接受 hazard_target_fraction 参数，它是环境的内部属性
        self.environment = SugarEnvironment(
            size=self.grid_size,
            growth_rate=self.sugar_growth_rate,
            max_sugar=self.max_sugar,
            spice_growth_rate=spice_growth_rate if resource_enabled.get('spice', True) else 0.0,
            max_spice=max_spice if resource_enabled.get('spice', True) else 0.0,
            hazard_decay_rate=hazard_decay_rate if resource_enabled.get('hazard', True) else 0.0,
            # hazard_penalty_factor 和 hazard_damage_per_step 使用默认值
        )
        
        # 如果提供了 hazard_target_fraction，在创建环境后设置它（如果环境支持）
        if resource_enabled.get('hazard', True) and hasattr(self.environment, 'hazard_target_fraction'):
            self.environment.hazard_target_fraction = hazard_target_fraction
        
        # 如果资源被禁用，清空对应的资源图
        if not resource_enabled.get('sugar', True):
            self.environment.sugar_map.fill(0.0)
        if not resource_enabled.get('spice', True):
            self.environment.spice_map.fill(0.0)
        if not resource_enabled.get('hazard', True):
            self.environment.hazard_map.fill(0.0)
    
    def _initialize_agents(self, agent_configs: Optional[List[Any]] = None) -> None:
        """初始化智能体"""
        self.agent_manager.clear()
        
        if agent_configs:
            # 使用配置创建智能体
            agents = self.agent_factory.create_agents(
                agent_configs,
                generation_name="initial_population"
            )
            for agent in agents:
                self.agent_manager.add_agent(agent)
        else:
            # 创建默认智能体群体
            self._create_default_agents()
        
        logging.info(f"初始化 {len(self.agent_manager)} 个智能体")
    
    def _create_default_agents(self) -> None:
        """创建默认智能体群体 - 修复配置问题"""
        from src.core.agent_factory import AgentTypeConfig, PositionConfig, DistributionType
        
        # 创建多样化的智能体群体 - 确保数量正确
        # 如果initial_agents为0，使用默认值50
        total_agents = max(self.initial_agents, 50) if self.initial_agents == 0 else min(self.initial_agents, 100)
        
        default_configs = [
            AgentTypeConfig(
                agent_type="rule_based",
                count=max(1, total_agents // 2),  # 确保至少1个
                config={
                    "vision_range": min(4, max(1, self.grid_size // 10)),  # 根据网格大小调整视野，至少为1
                    "metabolism_rate": 1.0,
                    "movement_strategy": "greedy"
                },
                position_config=PositionConfig(distribution=DistributionType.UNIFORM)
            ),
            AgentTypeConfig(
                agent_type="conservative",
                count=max(1, total_agents // 4),  # 确保至少1个
                config={
                    "vision_range": min(3, max(1, self.grid_size // 10)),
                    "metabolism_rate": 0.5,
                    "movement_strategy": "conservative"
                },
                position_config=PositionConfig(distribution=DistributionType.UNIFORM)
            ),
            AgentTypeConfig(
                agent_type="exploratory",
                count=max(1, total_agents - (total_agents // 2) - (total_agents // 4)),  # 剩余数量
                config={
                    "vision_range": min(6, max(1, self.grid_size // 8)),
                    "metabolism_rate": 0.8,
                    "movement_strategy": "exploratory"
                },
                position_config=PositionConfig(distribution=DistributionType.UNIFORM)
            )
        ]
        
        agents = self.agent_factory.create_agents(default_configs, "default_population")
        for agent in agents:
            self.agent_manager.add_agent(agent)
    
    def register_trainer(self, agent_type: str, trainer: Any) -> None:
        """
        注册MARL训练器
        
        Args:
            agent_type: 智能体类型
            trainer: 训练器实例
        """
        self.marl_trainers[agent_type] = trainer
        logging.info(f"注册训练器 for {agent_type}: {trainer.__class__.__name__}")
    
    def start(self) -> None:
        """启动模拟"""
        if self.state == SimulationState.READY:
            self.state = SimulationState.RUNNING
            self.start_time = time.time()
            logging.info("模拟启动")
    
    def pause(self) -> None:
        """暂停模拟"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            logging.info("模拟暂停")
    
    def resume(self) -> None:
        """恢复模拟"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            logging.info("模拟恢复")
    
    def reset(self, new_config: Optional[Dict[str, Any]] = None) -> None:
        """
        重置模拟
        
        Args:
            new_config: 可选的新配置字典，支持以下字段：
                - grid_size: 网格大小
                - cell_size: 单元格大小
                - initial_agents: 初始智能体数量
                - sugar_growth_rate: Sugar 生长速率
                - max_sugar: Sugar 最大值
                - agent_configs: 智能体配置列表（AgentTypeConfig 对象列表）
                - spice_growth_rate: Spice 生长速率
                - max_spice: Spice 最大值
                - hazard_decay_rate: Hazard 衰减速率
                - hazard_target_fraction: Hazard 目标覆盖比例
                - resource_enabled: 资源开关字典 {'sugar': bool, 'spice': bool, 'hazard': bool}
        """
        logging.info("重置模拟")
        
        # 暂停当前模拟（如果正在运行）
        was_running = self.state == SimulationState.RUNNING
        if was_running:
            self.state = SimulationState.PAUSED
        
        # 提取配置
        env_config = {}
        agent_configs = None
        
        if new_config:
            # 更新基础配置
            self._update_config(new_config)
            
            # 提取环境配置
            env_config = {
                'spice_growth_rate': new_config.get('spice_growth_rate', 0.02),
                'max_spice': new_config.get('max_spice', 6.0),
                'hazard_decay_rate': new_config.get('hazard_decay_rate', 0.01),
                'hazard_target_fraction': new_config.get('hazard_target_fraction', 0.09),
                'resource_enabled': new_config.get('resource_enabled', {
                    'sugar': True,
                    'spice': True,
                    'hazard': True
                })
            }
            
            # 提取智能体配置
            agent_configs = new_config.get('agent_configs', None)
        
        # 重置状态
        self.step_count = 0
        self.metrics_history.clear()
        self.training_metrics.clear()
        
        # 清空训练器（如果需要重新创建）
        # 注意：这里不清空，让外部代码决定是否重新创建训练器
        
        # 重新初始化环境
        self._initialize_environment(env_config)
        
        # Sync factory grid size with (potentially updated) simulation grid size
        try:
            self.agent_factory.set_grid_size(self.grid_size)
        except Exception:
            pass
        
        # 重新初始化智能体
        self._initialize_agents(agent_configs)
        
        # 如果需要，重新注册训练器（由外部代码处理）
        # 这里可以添加回调机制，让可视化系统知道需要重新创建训练器
        
        # 自动重新开始模拟，提升交互体验
        if was_running:
            self.state = SimulationState.RUNNING
        else:
            self.state = SimulationState.READY
        
        self.start_time = time.time()
        logging.info(f"模拟重置完成: grid_size={self.grid_size}, agents={len(self.agent_manager)}")
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config: 配置字典，只更新模拟对象已有的属性
        """
        # 基础配置字段
        basic_fields = ['grid_size', 'cell_size', 'initial_agents', 'sugar_growth_rate', 'max_sugar']
        
        for key, value in config.items():
            if key in basic_fields and hasattr(self, key):
                setattr(self, key, value)
    
    def update(self) -> bool:
        """
        更新模拟状态
        
        Returns:
            是否成功更新
        """
        if self.state != SimulationState.RUNNING:
            return False
        
        update_start = time.time()
        
        try:
            # 环境更新
            self._update_environment()
            
            # 智能体更新
            self._update_agents()
            
            # MARL训练（如果启用）
            self._update_marl_training()
            
            # 收集指标
            self._collect_metrics()
            
            # 性能监控
            self._update_performance_metrics(update_start)
            
            self.step_count += 1
            return True
            
        except Exception as e:
            logging.error(f"模拟更新失败: {e}")
            return False
    
    def _update_environment(self) -> None:
        """更新环境状态"""
        if hasattr(self.environment, 'grow_back'):
            self.environment.grow_back()
    
    def _update_agents(self) -> None:
        """更新所有智能体，并记录耗时"""
        update_start = time.time()
        self.agent_manager.update_agents(self.environment)
        self.performance_metrics.agent_update_time = (time.time() - update_start) * 1000
    
    def _update_marl_training(self) -> None:
        """更新MARL训练"""
        if not self.marl_trainers or self.step_count % 4 != 0:  # 每4步训练一次
            return
        
        for agent_type, trainer in self.marl_trainers.items():
            try:
                if hasattr(trainer, 'train_step'):
                    trainer.train_step()
            except Exception as e:
                logging.warning(f"{agent_type} 训练失败: {e}")
    
    def _collect_metrics(self) -> None:
        """收集模拟指标"""
        metrics = SimulationMetrics()
        metrics.step = self.step_count
        
        agents = self.agent_manager.agents
        
        if not agents:
            # 没有智能体时的默认值
            return
        
        # 基础统计
        metrics.total_agents = len(agents)
        metrics.alive_agents = len([a for a in agents if a.status == AgentStatus.ALIVE])
        
        # 计算平均值
        sugar_values = [a.sugar for a in agents]
        age_values = [a.age for a in agents]
        
        metrics.avg_sugar = np.mean(sugar_values) if sugar_values else 0.0
        metrics.avg_age = np.mean(age_values) if age_values else 0.0
        
        # 环境统计
        if hasattr(self.environment, 'total_sugar'):
            metrics.total_environment_sugar = self.environment.total_sugar
        
        # 类型统计和多样性
        self._calculate_diversity_metrics(metrics, agents)
        # 动作分布统计
        self._collect_action_distribution(metrics, agents)
        
        # 收集训练指标
        self._collect_training_metrics()
        
        # 更新当前指标和历史
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
    
    def _calculate_diversity_metrics(self, metrics: SimulationMetrics, agents: List[BaseAgent]) -> None:
        """计算多样性指标"""
        type_counts = {}
        type_sugars = {}
        
        for agent in agents:
            agent_type = agent.agent_type.value
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
            
            if agent_type not in type_sugars:
                type_sugars[agent_type] = []
            type_sugars[agent_type].append(agent.sugar)
        
        metrics.agents_by_type = type_counts
        
        # 计算类型平均糖量
        for agent_type, sugars in type_sugars.items():
            metrics.avg_sugar_by_type[agent_type] = np.mean(sugars)
        
        # 计算多样性（香农熵）
        total_agents = len(agents)
        diversity = 0.0
        for count in type_counts.values():
            proportion = count / total_agents
            if proportion > 0:
                diversity -= proportion * np.log(proportion)
        
        metrics.agent_diversity = diversity

    def _collect_action_distribution(self, metrics: SimulationMetrics, agents: List[BaseAgent]) -> None:
        """
        统计按类型划分的动作分布（快照级，不追踪时间序列）。
        用于行为可视化中的动作频率面板。
        """
        action_dist: Dict[str, Dict[int, int]] = {}

        for agent in agents:
            # 仅统计存活智能体的当前动作倾向
            if agent.status != AgentStatus.ALIVE:
                continue

            agent_type = agent.agent_type.value

            # IQL 使用 last_action_idx，QMIX 使用 last_action（索引）
            action_idx = None
            if hasattr(agent, "last_action_idx"):
                action_idx = getattr(agent, "last_action_idx")
            elif hasattr(agent, "last_action"):
                action_idx = getattr(agent, "last_action")

            # 过滤无效动作值
            if action_idx is None:
                continue
            try:
                action_idx = int(action_idx)
            except (TypeError, ValueError):
                continue
            if action_idx < 0:
                continue

            if agent_type not in action_dist:
                action_dist[agent_type] = {}
            action_dist[agent_type][action_idx] = action_dist[agent_type].get(action_idx, 0) + 1

        metrics.action_distribution_by_type = action_dist
    
    def _collect_training_metrics(self) -> None:
        """收集训练指标"""
        self.training_metrics.clear()
        
        agents = self.agent_manager.agents
        
        # 按类型聚合训练指标
        type_metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        # 从智能体收集训练信息
        for agent in agents:
            if not hasattr(agent, 'get_training_info'):
                continue
            
            try:
                agent_type = agent.agent_type.value
                training_info = agent.get_training_info()
                
                if agent_type not in type_metrics:
                    type_metrics[agent_type] = []
                
                # 基础训练信息（loss / q / td / epsilon 等）
                type_metrics[agent_type].append(training_info)

                # 额外：从智能体内部的 training_stats 中提取奖励信息
                # 这样可以在不修改具体 Agent 实现的前提下，为可视化提供 Reward Trend 数据
                stats = getattr(agent, 'training_stats', None)
                if isinstance(stats, dict):
                    rewards = stats.get('rewards') or stats.get('reward_history')
                    if rewards:
                        try:
                            rewards_arr = np.array(rewards, dtype=float)
                            if rewards_arr.size > 0:
                                recent_reward = float(rewards_arr[-1])
                                # 使用一个滑动窗口计算平均奖励，避免过长历史导致平滑过度
                                window = rewards_arr[-100:] if rewards_arr.size > 100 else rewards_arr
                                avg_reward = float(window.mean())
                                type_metrics[agent_type].append({
                                    'avg_reward': avg_reward,
                                    'recent_reward': recent_reward,
                                })
                        except Exception as e:
                            logging.debug(f"处理智能体 {agent.agent_id} 奖励统计失败: {e}")
            except Exception as e:
                logging.debug(f"收集智能体 {agent.agent_id} 训练信息失败: {e}")
                continue
        
        # 从训练器收集统计信息（QMIX等集中式训练）
        for agent_type, trainer in self.marl_trainers.items():
            try:
                if hasattr(trainer, 'get_training_stats'):
                    stats = trainer.get_training_stats()
                    
                    if agent_type not in type_metrics:
                        type_metrics[agent_type] = []
                    
                    # 将训练器统计转换为与智能体统计一致的格式
                    trainer_info = {
                        'avg_loss': stats.get('avg_loss', stats.get('recent_loss', 0.0)),
                        'avg_q_value': stats.get('avg_q_value', stats.get('recent_q_value', 0.0)),
                        'avg_td_error': stats.get('avg_td_error', 0.0),
                        'training_steps': stats.get('training_steps', 0),
                        'exploration_rate': 0.0  # 训练器不直接提供探索率
                    }
                    type_metrics[agent_type].append(trainer_info)
            except Exception as e:
                logging.debug(f"收集训练器 {agent_type} 统计信息失败: {e}")
                continue
        
        # 聚合每个类型的指标
        for agent_type, info_list in type_metrics.items():
            if not info_list:
                continue
            
            # 计算平均值（不再强制 > 0 过滤，避免真实为 0 的值被丢弃）
            losses = [info.get('avg_loss', info.get('recent_loss', 0.0)) for info in info_list]
            q_values = [info.get('avg_q_value', info.get('recent_q_value', 0.0)) for info in info_list]
            td_errors = [info.get('avg_td_error', 0.0) for info in info_list]
            exploration_rates = [
                info.get('exploration_rate', info.get('epsilon', 0.0))
                for info in info_list
            ]
            training_steps = [info.get('training_steps', 0) for info in info_list]
            # 奖励统计（平均奖励 + 最近一次奖励）
            reward_values = [
                info.get('avg_reward', info.get('recent_reward', 0.0))
                for info in info_list
                if ('avg_reward' in info) or ('recent_reward' in info)
            ]

            avg_reward = float(np.mean(reward_values)) if reward_values else 0.0
            recent_reward = 0.0
            if reward_values:
                # 从 info_list 逆序找到最近一条包含奖励信息的记录
                for info in reversed(info_list):
                    if ('recent_reward' in info) or ('avg_reward' in info):
                        recent_reward = float(info.get('recent_reward', info.get('avg_reward', 0.0)))
                        break
            
            # 获取最近的值（用于实时显示）
            recent_loss = info_list[-1].get('avg_loss', info_list[-1].get('recent_loss', 0.0)) if info_list else 0.0
            recent_q_value = info_list[-1].get('avg_q_value', info_list[-1].get('recent_q_value', 0.0)) if info_list else 0.0
            
            metrics = TrainingMetrics(
                agent_type=agent_type,
                avg_loss=np.mean(losses) if losses else 0.0,
                avg_q_value=np.mean(q_values) if q_values else 0.0,
                avg_td_error=np.mean(td_errors) if td_errors else 0.0,
                exploration_rate=np.mean(exploration_rates) if exploration_rates else 0.0,
                training_steps=max(training_steps) if training_steps else 0,
                sample_count=len(info_list),
                recent_loss=recent_loss,
                recent_q_value=recent_q_value,
                avg_reward=avg_reward,
                recent_reward=recent_reward,
            )
            
            self.training_metrics[agent_type] = metrics
    
    def _update_performance_metrics(self, update_start: float) -> None:
        """更新性能指标"""
        # 计算帧率
        current_time = time.time()
        frame_time = (current_time - self._last_frame_time) * 1000  # 毫秒
        self._frame_times.append(frame_time)
        self._last_frame_time = current_time
        
        # 更新性能指标
        self.performance_metrics.step_time = (current_time - update_start) * 1000
        self.performance_metrics.fps = 1000.0 / np.mean(self._frame_times) if self._frame_times else 0
        
        # 简单的内存使用估计（实际项目中可以使用psutil）
        self.performance_metrics.memory_usage_mb = len(self.agent_manager.agents) * 0.1  # 估算值
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """
        获取模拟数据 - 供可视化系统使用
        
        Returns:
            包含所有模拟数据的字典
        """
        return {
            'state': self.state.value,
            'step_count': self.step_count,
            'running_time': time.time() - self.start_time,
            'metrics': self._serialize_metrics(self.current_metrics),
            'environment': self._get_environment_data(),
            'agents': self._get_agents_data(),
            'performance': self._get_performance_data(),
            'configuration': self._get_configuration_data(),
            'training_metrics': self._serialize_training_metrics(),
            'simulation': self  # 添加simulation对象引用，用于网络状态可视化
        }
    
    def _serialize_metrics(self, metrics: SimulationMetrics) -> Dict[str, Any]:
        """序列化指标数据"""
        return {
            'step': metrics.step,
            'total_agents': metrics.total_agents,
            'alive_agents': metrics.alive_agents,
            'avg_sugar': metrics.avg_sugar,
            'avg_age': metrics.avg_age,
            'total_environment_sugar': metrics.total_environment_sugar,
            'agent_diversity': metrics.agent_diversity,
            'agents_by_type': metrics.agents_by_type,
            'avg_sugar_by_type': metrics.avg_sugar_by_type,
            'action_distribution_by_type': metrics.action_distribution_by_type,
        }
    
    def _get_environment_data(self) -> Dict[str, Any]:
        """获取环境数据"""
        if self.environment is None:
            # Fallback if environment not initialized
            sugar_map = np.zeros((self.grid_size, self.grid_size))
            spice_map = np.zeros_like(sugar_map)
            hazard_map = np.zeros_like(sugar_map)
        else:
            sugar_map = getattr(self.environment, 'sugar_map', None)
            if sugar_map is None:
                sugar_map = np.zeros((self.grid_size, self.grid_size))
            # Ensure it's a numpy array
            if not isinstance(sugar_map, np.ndarray):
                sugar_map = np.array(sugar_map)
            
            spice_map = getattr(self.environment, 'spice_map', None)
            if spice_map is None:
                spice_map = np.zeros_like(sugar_map)
            if not isinstance(spice_map, np.ndarray):
                spice_map = np.array(spice_map)
            
            hazard_map = getattr(self.environment, 'hazard_map', None)
            if hazard_map is None:
                hazard_map = np.zeros_like(sugar_map)
            if not isinstance(hazard_map, np.ndarray):
                hazard_map = np.array(hazard_map)
        
        return {
            'sugar_map': sugar_map,
            'spice_map': spice_map,
            'hazard_map': hazard_map,
            'total_sugar': getattr(self.environment, 'total_sugar', 0) if self.environment else 0,
            'avg_sugar': getattr(self.environment, 'avg_sugar', 0) if self.environment else 0,
            'total_spice': getattr(self.environment, 'total_spice', 0) if self.environment else 0,
            'avg_spice': getattr(self.environment, 'avg_spice', 0) if self.environment else 0,
            'total_hazard': getattr(self.environment, 'total_hazard', 0) if self.environment else 0,
            'grid_size': self.grid_size,
            'cell_size': self.cell_size
        }
    
    def _get_agents_data(self) -> List[Dict[str, Any]]:
        """获取智能体数据"""
        agents_data = []
        for agent in self.agent_manager.agents:
            # Only include alive agents for visualization
            if agent.status == AgentStatus.ALIVE:
                agents_data.append({
                    'id': agent.agent_id,
                    'type': agent.agent_type.value,
                    'x': int(agent.x),  # Ensure integer coordinates
                    'y': int(agent.y),
                    'sugar': float(agent.sugar),
                    'age': int(agent.age),
                    'status': agent.status.value,
                    'vision_range': int(agent.vision_range),
                    'metabolism_rate': float(agent.metabolism_rate),
                    'total_collected': float(agent.total_collected)
                })
        return agents_data
    
    def _get_performance_data(self) -> Dict[str, float]:
        """获取性能数据"""
        return {
            'fps': self.performance_metrics.fps,
            'step_time': self.performance_metrics.step_time,
            'agent_update_time': self.performance_metrics.agent_update_time,
            'memory_usage_mb': self.performance_metrics.memory_usage_mb
        }
    
    def _get_configuration_data(self) -> Dict[str, Any]:
        """获取配置数据"""
        return {
            'grid_size': self.grid_size,
            'cell_size': self.cell_size,
            'sugar_growth_rate': self.sugar_growth_rate,
            'max_sugar': self.max_sugar,
            'marl_trainers': list(self.marl_trainers.keys())
        }
    
    def _serialize_training_metrics(self) -> Dict[str, Dict[str, Any]]:
        """序列化训练指标数据"""
        serialized = {}
        for agent_type, metrics in self.training_metrics.items():
            serialized[agent_type] = {
                'avg_loss': metrics.avg_loss,
                'avg_q_value': metrics.avg_q_value,
                'avg_td_error': metrics.avg_td_error,
                'exploration_rate': metrics.exploration_rate,
                'training_steps': metrics.training_steps,
                'sample_count': metrics.sample_count,
                'recent_loss': metrics.recent_loss,
                'recent_q_value': metrics.recent_q_value,
                'avg_reward': metrics.avg_reward,
                'recent_reward': metrics.recent_reward,
            }
        return serialized
    
    def add_agent(self, agent_type: str, x: int, y: int, **kwargs) -> BaseAgent:
        """
        动态添加智能体
        
        Args:
            agent_type: 智能体类型
            x, y: 位置坐标
            **kwargs: 额外参数
            
        Returns:
            创建的智能体
        """
        # 使用 AgentManager 提供的计数器，确保 ID 全局唯一
        agent_id = self.agent_manager.next_id()
        agent = self.agent_factory.create_agent(
            agent_type=agent_type,
            x=x, y=y,
            agent_id=agent_id,
            **kwargs
        )
        self.agent_manager.add_agent(agent)
        return agent
    
    def remove_agent(self, agent_id: int) -> bool:
        """
        移除指定智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否成功移除
        """
        for agent in self.agent_manager.agents:
            if agent.agent_id == agent_id:
                self.agent_manager.remove_agent(agent)
                return True
        return False
    
    def get_agent_info(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """
        获取指定智能体的详细信息
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体信息字典
        """
        for agent in self.agent_manager.agents:
            if agent.agent_id == agent_id:
                return agent.get_agent_info()
        return None
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """获取详细指标报告"""
        base_data = self.get_simulation_data()
        
        # 添加历史数据统计
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # 最近10步
            base_data['recent_trends'] = {
                'avg_sugar_trend': [m.avg_sugar for m in recent_metrics],
                'population_trend': [m.total_agents for m in recent_metrics],
                'diversity_trend': [m.agent_diversity for m in recent_metrics]
            }
        
        return base_data
    
    def __repr__(self) -> str:
        return (f"MARLSimulation(state={self.state.value}, steps={self.step_count}, "
                f"agents={len(self.agent_manager.agents)}, grid={self.grid_size})")


class SimulationFactory:
    """
    模拟工厂 - 用于创建预配置的模拟实例
    """
    
    @staticmethod
    def create_comparative_simulation() -> MARLSimulation:
        """创建算法对比模拟"""
        from src.core.agent_factory import AgentTypeConfig, PositionConfig, DistributionType
        from src.config.defaults import DEFAULT_AGENT_CONFIGS
        from src.marl.qmix_trainer import QMIXTrainer
        
        # 使用默认配置构造 QMIX 参数
        qmix_defaults = DEFAULT_AGENT_CONFIGS.get('qmix', {})
        qmix_config = {
            "vision_range": qmix_defaults.get("vision_range", 5),
            "metabolism_rate": qmix_defaults.get("metabolism_rate", 0.8),
            "state_dim": qmix_defaults.get("state_dim", 128),
            "action_dim": qmix_defaults.get("action_dim", 8),
            "agent_hidden_dims": qmix_defaults.get("agent_hidden_dims", [64, 64]),
            "mixing_hidden_dim": qmix_defaults.get("mixing_hidden_dim", 32),
            "learning_rate": qmix_defaults.get("learning_rate", 0.0005),
            "gamma": qmix_defaults.get("gamma", 0.99),
            "tau": qmix_defaults.get("tau", 0.005),
            "epsilon_start": qmix_defaults.get("epsilon_start", 1.0),
            "epsilon_end": qmix_defaults.get("epsilon_end", 0.05),
            "epsilon_decay": qmix_defaults.get("epsilon_decay", 0.999),
            "batch_size": qmix_defaults.get("batch_size", 32),
            "replay_buffer_size": qmix_defaults.get("replay_buffer_size", 10000),
            "learning_starts": qmix_defaults.get("learning_starts", 1000),
            "train_frequency": qmix_defaults.get("train_frequency", 4),
            "target_update_frequency": qmix_defaults.get("target_update_frequency", 200),
        }

        agent_configs = [
            AgentTypeConfig(
                agent_type="rule_based",
                count=20,
                config={"vision_range": 4, "metabolism_rate": 1.0},
                position_config=PositionConfig(distribution=DistributionType.CLUSTERED)
            ),
            AgentTypeConfig(
                agent_type="iql", 
                count=20,
                config={"vision_range": 5, "metabolism_rate": 0.8},
                position_config=PositionConfig(distribution=DistributionType.CLUSTERED)
            ),
            AgentTypeConfig(
                agent_type="qmix",
                count=20,
                config=qmix_config,
                position_config=PositionConfig(distribution=DistributionType.CLUSTERED)
            ),
            AgentTypeConfig(
                agent_type="conservative",
                count=15,
                config={"vision_range": 3, "metabolism_rate": 0.5},
                position_config=PositionConfig(distribution=DistributionType.CLUSTERED)
            )
        ]
        
        sim = MARLSimulation(
            grid_size=60,
            initial_agents=0,  # 使用自定义配置
            agent_configs=agent_configs
        )

        # 为QMIX智能体创建集中式训练器并注册，以便可视化其训练数据
        qmix_agents = sim.agent_manager.get_agents_by_type("qmix")
        if qmix_agents:
            sample_agent = qmix_agents[0]
            state_dim = getattr(sample_agent, "state_dim", qmix_config["state_dim"])
            action_dim = getattr(sample_agent, "action_dim", qmix_config["action_dim"])

            qmix_trainer = QMIXTrainer(
                num_agents=len(qmix_agents),
                state_dim=state_dim,
                action_dim=action_dim,
                config=qmix_config
            )

            for agent in qmix_agents:
                if hasattr(agent, "set_trainer"):
                    agent.set_trainer(qmix_trainer)

            qmix_trainer.sync_agent_networks(qmix_agents)
            sim.register_trainer("qmix", qmix_trainer)

        return sim
    
    @staticmethod
    def create_marl_training_simulation() -> MARLSimulation:
        """创建MARL训练模拟"""
        from src.core.agent_factory import AgentTypeConfig, PositionConfig, DistributionType
        from src.config.defaults import DEFAULT_AGENT_CONFIGS
        from src.marl.qmix_trainer import QMIXTrainer
        
        # IQL 和 QMIX 共同参与训练对比
        iql_config = {
            "vision_range": 5,
            "metabolism_rate": 1.0,
            "learning_rate": 0.001,
            "epsilon_start": 1.0
        }
        # 使用默认QMIX配置作为基础
        qmix_defaults = DEFAULT_AGENT_CONFIGS.get('qmix', {})
        qmix_config = {
            "vision_range": qmix_defaults.get("vision_range", 5),
            "metabolism_rate": qmix_defaults.get("metabolism_rate", 0.8),
            "state_dim": qmix_defaults.get("state_dim", 128),
            "action_dim": qmix_defaults.get("action_dim", 8),
            "agent_hidden_dims": qmix_defaults.get("agent_hidden_dims", [64, 64]),
            "mixing_hidden_dim": qmix_defaults.get("mixing_hidden_dim", 32),
            "learning_rate": qmix_defaults.get("learning_rate", 0.0005),
            "gamma": qmix_defaults.get("gamma", 0.99),
            "tau": qmix_defaults.get("tau", 0.005),
            "epsilon_start": qmix_defaults.get("epsilon_start", 1.0),
            "epsilon_end": qmix_defaults.get("epsilon_end", 0.05),
            "epsilon_decay": qmix_defaults.get("epsilon_decay", 0.999),
            "batch_size": qmix_defaults.get("batch_size", 32),
            "replay_buffer_size": qmix_defaults.get("replay_buffer_size", 10000),
            "learning_starts": qmix_defaults.get("learning_starts", 1000),
            "train_frequency": qmix_defaults.get("train_frequency", 4),
            "target_update_frequency": qmix_defaults.get("target_update_frequency", 200),
        }

        agent_configs = [
            AgentTypeConfig(
                agent_type="iql",
                count=24,
                config=iql_config,
                position_config=PositionConfig(distribution=DistributionType.UNIFORM)
            ),
            AgentTypeConfig(
                agent_type="qmix",
                count=24,
                config=qmix_config,
                position_config=PositionConfig(distribution=DistributionType.UNIFORM)
            )
        ]
        
        sim = MARLSimulation(
            grid_size=80,
            initial_agents=0,
            agent_configs=agent_configs
        )

        # 为QMIX智能体创建集中式训练器，并注册到模拟中
        qmix_agents = sim.agent_manager.get_agents_by_type("qmix")
        if qmix_agents:
            # 从第一个QMIX智能体推断状态和动作维度
            sample_agent = qmix_agents[0]
            state_dim = getattr(sample_agent, "state_dim", qmix_config["state_dim"])
            action_dim = getattr(sample_agent, "action_dim", qmix_config["action_dim"])

            qmix_trainer = QMIXTrainer(
                num_agents=len(qmix_agents),
                state_dim=state_dim,
                action_dim=action_dim,
                config=qmix_config
            )

            # 将训练器绑定到所有QMIX智能体
            for agent in qmix_agents:
                if hasattr(agent, "set_trainer"):
                    agent.set_trainer(qmix_trainer)

            # 同步网络参数到所有QMIX智能体
            qmix_trainer.sync_agent_networks(qmix_agents)

            # 在模拟中注册训练器，便于统一训练和指标收集
            sim.register_trainer("qmix", qmix_trainer)

        return sim
    
    @staticmethod
    def create_high_performance_simulation() -> MARLSimulation:
        """创建高性能模拟（大量智能体）"""
        return MARLSimulation(
            grid_size=100,
            initial_agents=200,
            sugar_growth_rate=0.15,
            max_sugar=15.0
        )