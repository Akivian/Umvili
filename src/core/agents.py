"""
智能体实现模块

该模块包含基于BaseAgent的具体智能体实现，包括规则型智能体和其他算法智能体。
所有智能体都遵循统一的接口和设计规范。

设计目标：
1. 保持与现有代码的兼容性
2. 提供清晰的算法实现
3. 支持多种行为策略
4. 便于扩展和测试
"""

from typing import Tuple, Dict, Any, Optional, List
import random
import numpy as np
from dataclasses import dataclass

from src.core.agent_base import (
    BaseAgent, LearningAgent, EvolutionaryAgent, 
    AgentType, AgentStatus, ObservationSpace, ActionSpace,
    AgentMetrics, BaseAgentConfig
)
from src.config.ui_config import COLORS
from src.config.simulation_config import SimulationConfig
# 向后兼容：使用默认值
CELL_SIZE = 10
GRID_SIZE = 80


@dataclass
class RuleBasedAgentConfig:
    """规则型智能体配置数据类"""
    vision_range: int = 5
    metabolism_rate: float = 1.0
    movement_strategy: str = "greedy"  # greedy, conservative, exploratory
    exploration_rate: float = 0.1  # 探索概率


class RuleBasedAgent(BaseAgent):
    """
    基于规则的智能体
    
    实现传统的Sugarscape规则：
    - 在视野范围内寻找糖最多的位置
    - 移动到目标位置并收集糖
    - 消耗新陈代谢维持生命
    """

    def __init__(self, 
                 x: int, 
                 y: int, 
                 agent_id: int,
                 config: Optional[RuleBasedAgentConfig] = None,
                 name: Optional[str] = None):
        """
        初始化规则型智能体
        
        Args:
            x: 初始x坐标
            y: 初始y坐标
            agent_id: 智能体唯一标识符
            config: 智能体配置
            name: 智能体名称
        """
        self.config = config or RuleBasedAgentConfig()
        
        # Create base config from rule-based config
        base_config = BaseAgentConfig(
            vision_range=self.config.vision_range,
            metabolism_rate=self.config.metabolism_rate
        )
        
        super().__init__(
            x=x,
            y=y,
            agent_id=agent_id,
            agent_type=AgentType.RULE_BASED,
            base_config=base_config,
            name=name
        )
        
        # 规则型智能体特定属性
        self.movement_history = []
        self.sugar_encounters = []  # 遇到的糖量记录
        
        # 初始化动作空间（允许在视野范围内移动）
        self.action_space = ActionSpace(
            environment_size=GRID_SIZE,
            max_movement=self.vision_range
        )

    def decide_action(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        基于规则的决策函数
        
        根据配置的策略选择移动目标：
        - greedy: 选择糖最多的位置
        - conservative: 考虑移动成本
        - exploratory: 随机探索
        
        Args:
            observation: 观察空间对象
            
        Returns:
            目标位置 (target_x, target_y)
        """
        if self.config.movement_strategy == "greedy":
            return self._greedy_movement(observation)
        elif self.config.movement_strategy == "conservative":
            return self._conservative_movement(observation)
        elif self.config.movement_strategy == "exploratory":
            return self._exploratory_movement(observation)
        else:
            return self._greedy_movement(observation)

    def learn(self, experience: Dict[str, Any]) -> None:
        """
        学习函数 - 规则型智能体不学习，但记录经验
        
        Args:
            experience: 经验数据
        """
        # 规则型智能体不进行机器学习，但可以记录经验用于分析
        if 'reward' in experience and 'state' in experience:
            self.sugar_encounters.append({
                'step': experience.get('step', 0),
                'sugar_collected': experience.get('reward', 0),
                'position': (self.x, self.y)
            })

    def _greedy_movement(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        贪婪移动策略 - 选择糖最多的位置
        
        Args:
            observation: 观察空间对象
            
        Returns:
            目标位置 (target_x, target_y)
        """
        best_x, best_y = self.x, self.y
        max_sugar = -1.0
        env_size = observation.environment_size
        
        # 在视野范围内搜索糖最多的位置
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                if dx == 0 and dy == 0:
                    continue  # 排除当前位置
                    
                # 使用环境实际大小进行边界处理
                new_x = max(0, min(env_size - 1, self.x + dx))
                new_y = max(0, min(env_size - 1, self.y + dy))
                
                # 获取该位置的糖量
                sugar_here = self._estimate_sugar_at(new_x, new_y, observation)
                
                if sugar_here > max_sugar:
                    max_sugar = sugar_here
                    best_x, best_y = new_x, new_y
        
        return best_x, best_y

    def _conservative_movement(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        保守移动策略 - 考虑移动距离和糖量
        
        Args:
            observation: 观察空间对象
            
        Returns:
            目标位置 (target_x, target_y)
        """
        best_x, best_y = self.x, self.y
        best_score = -float('inf')
        env_size = observation.environment_size
        
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                # 使用环境实际大小进行边界处理
                new_x = max(0, min(env_size - 1, self.x + dx))
                new_y = max(0, min(env_size - 1, self.y + dy))
                
                sugar_here = self._estimate_sugar_at(new_x, new_y, observation)
                distance = abs(dx) + abs(dy)  # 曼哈顿距离
                
                # 计算得分：糖量 - 移动成本
                score = sugar_here - (distance * 0.1)
                
                if score > best_score:
                    best_score = score
                    best_x, best_y = new_x, new_y
        
        return best_x, best_y

    def _exploratory_movement(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        探索性移动策略 - 结合贪婪和随机探索
        
        Args:
            observation: 观察空间对象
            
        Returns:
            目标位置 (target_x, target_y)
        """
        env_size = observation.environment_size
        
        # 按概率进行随机探索
        if random.random() < self.config.exploration_rate:
            # 随机选择视野内的一个位置
            dx = random.randint(-self.vision_range, self.vision_range)
            dy = random.randint(-self.vision_range, self.vision_range)
            # 使用环境实际大小进行边界处理
            new_x = max(0, min(env_size - 1, self.x + dx))
            new_y = max(0, min(env_size - 1, self.y + dy))
            return new_x, new_y
        else:
            # 使用贪婪策略
            return self._greedy_movement(observation)

    def _estimate_sugar_at(self, x: int, y: int, observation: ObservationSpace) -> float:
        """
        估计指定位置的糖量（优化版，带错误处理）
        
        Args:
            x: 目标x坐标
            y: 目标y坐标
            observation: 观察空间对象
            
        Returns:
            估计的糖量
        """
        try:
            # 从局部视野中获取糖量信息
            if not hasattr(observation, 'local_view') or observation.local_view is None:
                return 0.0
            
            local_view = observation.local_view
            view_size = local_view.shape[0] if len(local_view.shape) > 0 else 0
            
            if view_size == 0:
                return 0.0
            
            # 计算局部坐标
            local_x = x - self.x + self.vision_range
            local_y = y - self.y + self.vision_range
            
            # 边界检查
            if (0 <= local_x < view_size and 
                0 <= local_y < view_size):
                sugar_value = float(local_view[local_x, local_y])
                # 检查NaN和Inf
                if np.isnan(sugar_value) or np.isinf(sugar_value):
                    return 0.0
                return max(0.0, sugar_value)
            else:
                # 如果不在当前视野内，返回保守估计
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"智能体 {self.agent_id} 估计糖量失败: {e}")
            return 0.0

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        获取详细性能指标
        
        Returns:
            包含详细指标的字典
        """
        base_metrics = super().get_agent_info()
        
        # 添加规则型智能体特有指标
        detailed_metrics = {
            **base_metrics,
            'movement_strategy': self.config.movement_strategy,
            'exploration_rate': self.config.exploration_rate,
            'total_movements': len(self.movement_history),
            'average_sugar_per_move': (
                self.total_collected / len(self.movement_history) 
                if self.movement_history else 0
            ),
            'sugar_encounter_history': self.sugar_encounters[-10:]  # 最近10次糖遇到记录
        }
        
        return detailed_metrics

    def __repr__(self) -> str:
        return (f"RuleBasedAgent(id={self.agent_id}, name={self.name}, "
                f"strategy={self.config.movement_strategy}, "
                f"pos=({self.x}, {self.y}), sugar={self.sugar:.1f})")


class ConservativeAgent(RuleBasedAgent):
    """
    保守型智能体
    
    特别关注生存，优先选择低风险移动
    """

    def __init__(self, x: int, y: int, agent_id: int,
                 config: Optional[RuleBasedAgentConfig] = None,
                 name: Optional[str] = None):
        # 允许外部传入配置，否则使用保守型默认配置
        config = config or RuleBasedAgentConfig(
            movement_strategy="conservative",
            vision_range=4,  # 较小的视野范围
            metabolism_rate=0.8  # 较低的新陈代谢
        )
        super().__init__(x, y, agent_id, config, name or f"conservative_{agent_id}")
        # 设置正确的智能体类型
        self.agent_type = AgentType.CONSERVATIVE


class ExploratoryAgent(RuleBasedAgent):
    """
    探索型智能体
    
    喜欢探索新区域，具有较高的探索率
    """

    def __init__(self, x: int, y: int, agent_id: int,
                 config: Optional[RuleBasedAgentConfig] = None,
                 name: Optional[str] = None):
        # 允许外部传入配置，否则使用探索型默认配置
        config = config or RuleBasedAgentConfig(
            movement_strategy="exploratory",
            exploration_rate=0.3,  # 较高的探索率
            vision_range=6,        # 较大的视野范围
            metabolism_rate=1.2    # 较高的新陈代谢
        )
        super().__init__(x, y, agent_id, config, name or f"exploratory_{agent_id}")
        # 设置正确的智能体类型
        self.agent_type = AgentType.EXPLORATORY


class AdaptiveAgent(RuleBasedAgent):
    """
    自适应智能体
    
    根据当前状态调整策略
    """

    def __init__(self, x: int, y: int, agent_id: int,
                 config: Optional[RuleBasedAgentConfig] = None,
                 name: Optional[str] = None):
        # 自适应智能体默认使用贪婪策略配置，后续根据经验自适应调整
        config = config or RuleBasedAgentConfig(
            movement_strategy="greedy",
            vision_range=5,
            metabolism_rate=1.0
        )
        super().__init__(x, y, agent_id, config, name=name or f"adaptive_{agent_id}")
        # 设置正确的智能体类型
        self.agent_type = AgentType.ADAPTIVE
        self.risk_tolerance = 0.5  # 风险容忍度
        self.last_sugar_trend = 0  # 糖量趋势

    def decide_action(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        自适应决策 - 根据当前状态调整策略
        
        Args:
            observation: 观察空间对象
            
        Returns:
            目标位置 (target_x, target_y)
        """
        # 根据糖量水平调整策略
        if self.sugar < 10:  # 低糖状态，采取保守策略
            return self._conservative_movement(observation)
        elif self.sugar > 30:  # 高糖状态，可以冒险探索
            return self._exploratory_movement(observation)
        else:  # 正常状态，使用贪婪策略
            return self._greedy_movement(observation)

    def learn(self, experience: Dict[str, Any]) -> None:
        """
        自适应学习 - 根据结果调整风险容忍度
        
        Args:
            experience: 经验数据
        """
        super().learn(experience)
        
        # 根据收集的糖量调整风险容忍度
        if 'reward' in experience:
            reward = experience['reward']
            if reward > 0:
                # 成功收集糖，稍微增加风险容忍度
                self.risk_tolerance = min(0.8, self.risk_tolerance + 0.01)
            else:
                # 没有收集到糖，降低风险容忍度
                self.risk_tolerance = max(0.2, self.risk_tolerance - 0.01)


# 保持向后兼容的工厂函数
def create_legacy_agent(x: int, y: int, vision: int = 5, metabolism: int = 1) -> RuleBasedAgent:
    """
    创建与传统Agent兼容的智能体
    
    Args:
        x: x坐标
        y: y坐标
        vision: 视野范围
        metabolism: 新陈代谢速率
        
    Returns:
        规则型智能体实例
    """
    config = RuleBasedAgentConfig(
        vision_range=vision,
        metabolism_rate=metabolism,
        movement_strategy="greedy"
    )
    return RuleBasedAgent(x, y, agent_id=hash(f"{x}_{y}"), config=config)


# 智能体类型注册表
AGENT_REGISTRY = {
    "rule_based": RuleBasedAgent,
    "conservative": ConservativeAgent,
    "exploratory": ExploratoryAgent,
    "adaptive": AdaptiveAgent
}


def create_agent(agent_type: str, x: int, y: int, agent_id: int, **kwargs) -> BaseAgent:
    """
    创建指定类型的智能体
    
    Args:
        agent_type: 智能体类型
        x: x坐标
        y: y坐标
        agent_id: 智能体ID
        **kwargs: 额外参数
        
    Returns:
        智能体实例
        
    Raises:
        ValueError: 当智能体类型未知时
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"未知的智能体类型: {agent_type}。可用类型: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(x, y, agent_id, **kwargs)