"""
Agent Base Module - Core Foundation for MARL Platform

This module defines the unified interface and base functionality for all agents,
providing a solid foundation for building multi-algorithm comparison platforms.

Design Principles:
1. Open-Closed Principle: Open for extension, closed for modification
2. Dependency Inversion: High-level modules depend on abstractions
3. Interface Segregation: Clients should not depend on unused interfaces
4. Single Responsibility: Each class has one reason to change

Author: MARL Platform Team
Version: 3.0.0
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List, Protocol
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

# 向后兼容：使用默认值
CELL_SIZE = 10


class AgentType(Enum):
    """Agent Type Enumeration"""
    RULE_BASED = "rule_based"
    IQL = "independent_q_learning"
    QMIX = "qmix"
    MAPPO = "mappo"
    EVOLUTIONARY = "evolutionary"
    CONSERVATIVE = "conservative"
    EXPLORATORY = "exploratory"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


class AgentStatus(Enum):
    """Agent Status Enumeration"""
    ALIVE = "alive"
    DEAD = "dead"
    INACTIVE = "inactive"


@dataclass
class AgentMetrics:
    """Agent Performance Metrics Data Class"""
    agent_id: int
    agent_type: AgentType
    age: int = 0
    sugar: float = 0.0
    total_collected: float = 0.0
    max_sugar: float = 0.0
    moves_made: int = 0
    status: AgentStatus = AgentStatus.ALIVE
    efficiency: float = 0.0  # Sugar collection efficiency = total_collected / age
    last_update_time: float = 0.0
    
    def update_efficiency(self) -> None:
        """Update efficiency metric"""
        if self.age > 0:
            self.efficiency = self.total_collected / self.age
        else:
            self.efficiency = 0.0


@dataclass
class BaseAgentConfig:
    """Base Configuration for All Agents"""
    vision_range: int = 5
    metabolism_rate: float = 1.0
    initial_sugar: float = 15.0
    min_sugar_for_survival: float = 0.0
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.vision_range < 1:
            raise ValueError("vision_range must be >= 1")
        if self.metabolism_rate < 0:
            raise ValueError("metabolism_rate must be >= 0")
        if self.initial_sugar < 0:
            raise ValueError("initial_sugar must be >= 0")


class ObservationSpace:
    """Observation Space Encapsulation Class"""
    
    def __init__(self, 
                 local_view: np.ndarray, 
                 global_stats: Dict[str, Any], 
                 agent_position: Tuple[int, int], 
                 step: int,
                 environment_size: int):
        """
        Initialize observation space
        
        Args:
            local_view: Local environment observation (2D array)
            global_stats: Global statistics dictionary
            agent_position: Current agent position (x, y)
            step: Current time step
            environment_size: Size of the environment grid
        """
        self.local_view = local_view
        self.global_stats = global_stats
        self.agent_position = agent_position
        self.step = step
        self.environment_size = environment_size
        
    def flatten(self) -> np.ndarray:
        """Flatten observation space to 1D array (for neural network input)"""
        local_flat = self.local_view.flatten()
        global_features = np.array([
            self.global_stats.get('total_sugar', 0),
            self.global_stats.get('avg_sugar', 0),
            self.agent_position[0] / self.environment_size,  # Normalized
            self.agent_position[1] / self.environment_size,  # Normalized
            self.step / 10000.0  # Normalized step
        ])
        return np.concatenate([local_flat, global_features])
    
    def get_local_sugar_at(self, relative_x: int, relative_y: int) -> float:
        """Get sugar value at relative position in local view"""
        view_size = self.local_view.shape[0]
        center = view_size // 2
        x_idx = center + relative_x
        y_idx = center + relative_y
        
        if 0 <= x_idx < view_size and 0 <= y_idx < view_size:
            return float(self.local_view[x_idx, y_idx])
        return 0.0
    
    def __repr__(self) -> str:
        return f"ObservationSpace(pos={self.agent_position}, step={self.step}, size={self.environment_size})"


class ActionSpace:
    """Action Space Encapsulation Class"""
    
    def __init__(self, environment_size: int, max_movement: int = 1):
        """
        Initialize action space
        
        Args:
            environment_size: Size of the environment grid
            max_movement: Maximum movement distance in one step
        """
        self.environment_size = environment_size
        self.max_movement = max_movement
        self.actions = self._generate_actions()
        
    def _generate_actions(self) -> List[Tuple[int, int]]:
        """Generate all possible movement actions"""
        actions = []
        for dx in range(-self.max_movement, self.max_movement + 1):
            for dy in range(-self.max_movement, self.max_movement + 1):
                if dx == 0 and dy == 0:
                    continue  # Exclude staying in place
                actions.append((dx, dy))
        return actions
    
    def get_action_count(self) -> int:
        """Get action space size"""
        return len(self.actions)
    
    def get_action_by_index(self, index: int) -> Tuple[int, int]:
        """Get action by index"""
        if 0 <= index < len(self.actions):
            return self.actions[index]
        raise IndexError(f"Action index {index} out of range [0, {len(self.actions)})")
    
    def get_index_by_action(self, action: Tuple[int, int]) -> int:
        """Get index by action"""
        try:
            return self.actions.index(action)
        except ValueError:
            raise ValueError(f"Action {action} not in action space")
    
    def is_valid_action(self, action: Tuple[int, int]) -> bool:
        """Check if action is valid"""
        return action in self.actions


class EnvironmentProtocol(Protocol):
    """Protocol defining environment interface"""
    size: int
    step: int
    sugar_map: np.ndarray
    
    def harvest(self, x: int, y: int) -> float:
        """Harvest sugar at position"""
        ...
    
    @property
    def total_sugar(self) -> float:
        """Total sugar in environment"""
        ...
    
    @property
    def avg_sugar(self) -> float:
        """Average sugar in environment"""
        ...


class BaseAgent(ABC):
    """
    Abstract Base Class for All Agents
    
    Provides unified interface and base functionality for all agent implementations.
    Follows object-oriented design principles, supporting multiple algorithm implementations and comparisons.
    """
    
    # Class constants
    INITIAL_SUGAR = 15.0
    MIN_SUGAR_FOR_SURVIVAL = 0.0
    
    def __init__(self, 
                 x: int, 
                 y: int, 
                 agent_id: int, 
                 agent_type: AgentType,
                 base_config: Optional[BaseAgentConfig] = None,
                 name: Optional[str] = None):
        """
        Initialize agent
        
        Args:
            x: Initial x coordinate
            y: Initial y coordinate
            agent_id: Unique agent identifier
            agent_type: Agent type
            config: Agent configuration (optional)
            name: Agent name (optional)
        """
        # Validate inputs
        if x < 0 or y < 0:
            raise ValueError(f"Invalid initial position: ({x}, {y})")
        
        # Configuration (stored as base_config to avoid clashing with algorithm-specific configs)
        self.base_config = base_config or BaseAgentConfig()
        self.base_config.validate()
        
        # Position attributes
        self.x = x
        self.y = y
        self.previous_x = x
        self.previous_y = y
        
        # Identity attributes
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}_{agent_id}"
        
        # Physiological attributes
        self.vision_range = max(1, self.base_config.vision_range)
        self.metabolism_rate = max(0.1, self.base_config.metabolism_rate)
        self.sugar = self.base_config.initial_sugar
        self.age = 0
        self.max_sugar = self.sugar
        self.total_collected = 0.0
        
        # State management
        self.status = AgentStatus.ALIVE
        self.last_action: Optional[Tuple[int, int]] = None
        self.action_history: List[Dict[str, Any]] = []
        self.creation_time = time.time()
        
        # Performance metrics
        self.metrics = AgentMetrics(
            agent_id=agent_id,
            agent_type=agent_type,
            sugar=self.sugar,
            max_sugar=self.max_sugar,
            last_update_time=self.creation_time
        )
        
        # Action space (to be initialized by subclasses)
        self.action_space: Optional[ActionSpace] = None
        
        # Logger
        self.logger = logging.getLogger(f'{self.__class__.__name__}_{agent_id}')
    
    @abstractmethod
    def decide_action(self, observation: ObservationSpace) -> Tuple[int, int]:
        """
        Decision function - must be implemented by subclasses
        
        Args:
            observation: Observation space object
            
        Returns:
            Target position (target_x, target_y)
        """
        pass
    
    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        Learning function - must be implemented by subclasses
        
        Args:
            experience: Experience data containing state, action, reward, next_state, etc.
        """
        pass
    
    def observe(self, environment: EnvironmentProtocol) -> ObservationSpace:
        """
        Observe environment
        
        Args:
            environment: Environment object conforming to EnvironmentProtocol
            
        Returns:
            Observation space object
        """
        # Get local view
        local_view = self._get_local_view(environment)
        
        # Get global statistics
        global_stats = self._get_global_stats(environment)
        
        return ObservationSpace(
            local_view=local_view,
            global_stats=global_stats,
            agent_position=(self.x, self.y),
            step=getattr(environment, 'step', 0),
            environment_size=environment.size
        )
    
    def update(self, environment: EnvironmentProtocol) -> bool:
        """
        Update agent state
        
        Args:
            environment: Environment object
            
        Returns:
            Whether agent is still alive
        """
        if self.status != AgentStatus.ALIVE:
            return False
            
        # Observe environment
        observation = self.observe(environment)
        
        # Make decision
        target_x, target_y = self.decide_action(observation)
        
        # Ensure target position is within valid range
        target_x = self._clamp_coordinate(target_x, environment.size)
        target_y = self._clamp_coordinate(target_y, environment.size)
        
        # Record action
        self.last_action = (target_x, target_y)
        self.action_history.append({
            'step': getattr(environment, 'step', 0),
            'from': (self.x, self.y),
            'to': (target_x, target_y),
            'sugar_before': self.sugar
        })
        
        # Move
        self.previous_x, self.previous_y = self.x, self.y
        self.x, self.y = target_x, target_y
        
        # Harvest sugar
        collected = self._harvest(environment)
        self.sugar += collected
        self.total_collected += collected
        
        # Metabolism consumption
        self.sugar -= self.metabolism_rate
        self.age += 1
        
        # Update max sugar record
        self.max_sugar = max(self.max_sugar, self.sugar)
        
        # Check survival status
        if self.sugar <= self.base_config.min_sugar_for_survival:
            self.status = AgentStatus.DEAD
            self.metrics.status = AgentStatus.DEAD
        
        # Update performance metrics
        self._update_metrics()
        
        return self.status == AgentStatus.ALIVE
    
    def _clamp_coordinate(self, coord: int, max_size: int) -> int:
        """
        Clamp coordinate to valid range with error handling
        
        Args:
            coord: Coordinate value
            max_size: Maximum valid size
            
        Returns:
            Clamped coordinate
        """
        if max_size <= 0:
            self.logger.warning(f"Invalid max_size: {max_size}, using 1")
            max_size = 1
        return max(0, min(max_size - 1, int(coord)))
    
    def get_performance_metrics(self) -> AgentMetrics:
        """
        Get performance metrics
        
        Returns:
            AgentMetrics object
        """
        return self.metrics
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information
        
        Returns:
            Dictionary containing all agent information
        """
        return {
            'id': self.agent_id,
            'name': self.name,
            'type': self.agent_type.value,
            'position': (self.x, self.y),
            'sugar': self.sugar,
            'age': self.age,
            'vision_range': self.vision_range,
            'metabolism_rate': self.metabolism_rate,
            'status': self.status.value,
            'total_collected': self.total_collected,
            'max_sugar': self.max_sugar,
            'efficiency': self.metrics.efficiency
        }
    
    def reset(self, x: int, y: int) -> None:
        """
        Reset agent state
        
        Args:
            x: New x coordinate
            y: New y coordinate
        """
        self.x = x
        self.y = y
        self.previous_x = x
        self.previous_y = y
        self.sugar = self.base_config.initial_sugar
        self.age = 0
        self.max_sugar = self.sugar
        self.total_collected = 0.0
        self.status = AgentStatus.ALIVE
        self.last_action = None
        self.action_history.clear()
        self._update_metrics()
    
    def _get_local_view(self, environment: EnvironmentProtocol) -> np.ndarray:
        """Get local view with proper boundary handling"""
        size = self.vision_range * 2 + 1
        local_view = np.zeros((size, size))
        
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                view_x = dx + self.vision_range
                view_y = dy + self.vision_range
                
                # Calculate environment coordinates with boundary clamping
                env_x = self._clamp_coordinate(self.x + dx, environment.size)
                env_y = self._clamp_coordinate(self.y + dy, environment.size)
                
                local_view[view_x, view_y] = environment.sugar_map[env_x, env_y]
        
        return local_view
    
    def _get_global_stats(self, environment: EnvironmentProtocol) -> Dict[str, Any]:
        """Get global statistics"""
        stats = {}
        
        if hasattr(environment, 'total_sugar'):
            stats['total_sugar'] = environment.total_sugar
        if hasattr(environment, 'avg_sugar'):
            stats['avg_sugar'] = environment.avg_sugar
        
        return stats
    
    def _harvest(self, environment: EnvironmentProtocol) -> float:
        """
        Harvest sugar with error handling
        
        Args:
            environment: Environment object
            
        Returns:
            Harvested sugar amount
        """
        try:
            if hasattr(environment, 'harvest'):
                # 确保坐标在有效范围内
                x = self._clamp_coordinate(self.x, environment.size)
                y = self._clamp_coordinate(self.y, environment.size)
                return environment.harvest(x, y)
            return 0.0
        except Exception as e:
            self.logger.warning(f"智能体 {self.agent_id} 收获糖失败: {e}")
            return 0.0
    
    def _update_metrics(self) -> None:
        """Update performance metrics"""
        self.metrics.age = self.age
        self.metrics.sugar = self.sugar
        self.metrics.total_collected = self.total_collected
        self.metrics.max_sugar = self.max_sugar
        self.metrics.moves_made = len(self.action_history)
        self.metrics.status = self.status
        self.metrics.last_update_time = time.time()
        self.metrics.update_efficiency()
    
    def get_serializable_data(self) -> Dict[str, Any]:
        """
        Get serializable agent data
        For use by simulation system
        """
        return {
            'id': self.agent_id,
            'type': self.agent_type.value,
            'x': self.x,
            'y': self.y,
            'sugar': self.sugar,
            'age': self.age,
            'status': self.status.value,
            'vision_range': self.vision_range,
            'metabolism_rate': self.metabolism_rate,
            'total_collected': self.total_collected,
            'max_sugar': self.max_sugar,
            'efficiency': self.metrics.efficiency
        }
    
    @classmethod
    def get_agent_type_color(cls, agent_type: AgentType) -> Tuple[int, int, int]:
        """
        Get color for agent type
        Uses configuration colors if available
        """
        # 延迟导入以避免循环导入
        from src.config.ui_config import COLORS
        
        color_map = {
            AgentType.RULE_BASED: COLORS.get('AGENT_RULE_BASED', (31, 119, 180)),
            AgentType.IQL: COLORS.get('AGENT_IQL', (255, 127, 14)),
            AgentType.QMIX: COLORS.get('AGENT_QMIX', (44, 160, 44)),
            AgentType.CONSERVATIVE: COLORS.get('AGENT_CONSERVATIVE', (214, 39, 40)),
            AgentType.EXPLORATORY: COLORS.get('AGENT_EXPLORATORY', (148, 103, 189)),
            AgentType.ADAPTIVE: COLORS.get('AGENT_ADAPTIVE', (140, 86, 75)),
        }
        return color_map.get(agent_type, (200, 200, 200))
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.agent_id}, name={self.name}, "
                f"pos=({self.x}, {self.y}), sugar={self.sugar:.1f}, status={self.status.value})")
    
    def __str__(self) -> str:
        return (f"{self.name} - Position: ({self.x}, {self.y}), "
                f"Sugar: {self.sugar:.1f}, Age: {self.age}, Status: {self.status.value}")


class LearningAgent(BaseAgent, ABC):
    """
    Abstract Base Class for Learning Agents
    
    Provides additional infrastructure for agents that need learning capabilities.
    """
    
    def __init__(self, 
                 *args, 
                 learning_rate: float = 0.01, 
                 discount_factor: float = 0.99, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experience_buffer: List[Dict[str, Any]] = []
        self.training_step = 0
        
    @abstractmethod
    def get_q_values(self, observation: ObservationSpace) -> np.ndarray:
        """Get Q-values (for value-based algorithms)"""
        pass
    
    @abstractmethod
    def get_policy(self, observation: ObservationSpace) -> np.ndarray:
        """Get policy (for policy gradient algorithms)"""
        pass
    
    def store_experience(self, experience: Dict[str, Any]) -> None:
        """Store experience to replay buffer"""
        self.experience_buffer.append(experience)
        # Limit buffer size (prevent memory overflow)
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
    
    def sample_experience(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample from experience replay buffer"""
        if batch_size >= len(self.experience_buffer):
            return self.experience_buffer.copy()
        
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        return [self.experience_buffer[i] for i in indices]
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get training status information
        For use by simulation system for monitoring
        """
        return {
            'training_steps': getattr(self, 'training_step', 0),
            'learning_rate': getattr(self, 'learning_rate', 0),
            'epsilon': getattr(self, 'epsilon', 0) if hasattr(self, 'epsilon') else 0,
            'buffer_size': len(self.experience_buffer),
            'q_values': getattr(self, 'last_q_values', None)
        }


class EvolutionaryAgent(BaseAgent, ABC):
    """
    Abstract Base Class for Evolutionary Agents
    
    Provides additional infrastructure for agents that need evolutionary capabilities.
    """
    
    def __init__(self, *args, mutation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_rate = mutation_rate
        self.genome = self._initialize_genome()
        self.fitness = 0.0
        
    @abstractmethod
    def _initialize_genome(self) -> Dict[str, Any]:
        """Initialize genome"""
        pass
    
    @abstractmethod
    def mutate(self) -> None:
        """Perform mutation operation"""
        pass
    
    def calculate_fitness(self) -> float:
        """Calculate fitness"""
        # Base fitness function, can be overridden
        survival_bonus = self.age * 0.1
        sugar_bonus = self.total_collected * 0.5
        efficiency_bonus = self.metrics.efficiency * 10
        
        return survival_bonus + sugar_bonus + efficiency_bonus
    
    def crossover(self, other: 'EvolutionaryAgent') -> Dict[str, Any]:
        """Crossover with another agent to produce new genome"""
        child_genome = {}
        
        for key in self.genome:
            if np.random.random() < 0.5:
                child_genome[key] = self.genome[key]
            else:
                child_genome[key] = other.genome[key]
                
        return child_genome
