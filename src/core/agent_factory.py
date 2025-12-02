"""
Agent Factory Module - High-Quality Agent Creation System

This module provides agent creation, management, and configuration functionality,
supporting dynamic registration and instantiation of multiple agent types.
Uses Factory Pattern, Registry Pattern, and Strategy Pattern for flexible agent generation.

Design Principles:
1. Open-Closed Principle: Support extension of new agent types without modifying factory code
2. Single Responsibility: Factory only handles creation, configuration separated from creation logic
3. Dependency Inversion: Depends on abstractions (BaseAgent) rather than concrete implementations
4. Configuration-driven: Support agent creation through config files or dictionaries

Author: MARL Platform Team
Version: 3.0.0
"""

from typing import Dict, Any, List, Optional, Type, Union, Callable, Protocol, Tuple
import random
import inspect
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import threading

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import json
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

from src.core.agent_base import BaseAgent, AgentType, AgentStatus, BaseAgentConfig
from src.core.agents import (
    RuleBasedAgent, ConservativeAgent, ExploratoryAgent, AdaptiveAgent,
    RuleBasedAgentConfig, AGENT_REGISTRY
)
from src.config.simulation_config import SimulationConfig
# 向后兼容：使用默认值
GRID_SIZE = 80

try:
    from src.marl.iql_agent import IQLAgent, IQLConfig
    from src.marl.qmix_agent import QMIXAgent, QMIXConfig
    MARL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MARL module import failed: {e}")
    MARL_AVAILABLE = False


# ============================================================================
# Exception Classes
# ============================================================================

class FactoryError(Exception):
    """Base exception for factory-related errors"""
    pass


class AgentCreationError(FactoryError):
    """Exception raised when agent creation fails"""
    pass


class AgentConfigError(FactoryError):
    """Exception raised when agent configuration is invalid"""
    pass


# ============================================================================
# Configuration Classes
# ============================================================================

class DistributionType(Enum):
    """Agent distribution type enumeration"""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    CLUSTERED = "clustered"
    CUSTOM = "custom"


@dataclass
class PositionConfig:
    """Position configuration data class"""
    distribution: DistributionType = DistributionType.UNIFORM
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    cluster_centers: List[Tuple[int, int]] = field(default_factory=list)
    custom_positions: List[Tuple[int, int]] = field(default_factory=list)
    std_deviation: float = 20.0  # Standard deviation for Gaussian distribution


@dataclass
class AgentTypeConfig:
    """Agent type configuration data class"""
    agent_type: str
    count: int
    config: Dict[str, Any] = field(default_factory=dict)
    position_config: PositionConfig = field(default_factory=PositionConfig)


@dataclass
class GenerationConfig:
    """Agent generation configuration data class"""
    name: str
    description: str = ""
    agent_configs: List[AgentTypeConfig] = field(default_factory=list)
    total_agents: int = 0


# ============================================================================
# Position Generation Strategies
# ============================================================================

class PositionGenerationStrategy(Protocol):
    """Protocol for position generation strategies"""
    def generate(self, count: int, grid_size: int) -> List[Tuple[int, int]]:
        """Generate positions"""
        ...


class UniformPositionStrategy:
    """Uniform distribution position generation strategy"""
    
    def generate(self, count: int, grid_size: int) -> List[Tuple[int, int]]:
        """Generate uniformly distributed positions"""
        return [
            (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            for _ in range(count)
        ]


class GaussianPositionStrategy:
    """Gaussian distribution position generation strategy"""
    
    def __init__(self, center_x: int, center_y: int, std_dev: float):
        self.center_x = center_x
        self.center_y = center_y
        self.std_dev = std_dev
    
    def generate(self, count: int, grid_size: int) -> List[Tuple[int, int]]:
        """Generate Gaussian-distributed positions"""
        positions = []
        for _ in range(count):
            x = int(random.gauss(self.center_x, self.std_dev))
            y = int(random.gauss(self.center_y, self.std_dev))
            # Clamp to grid bounds
            x = max(0, min(grid_size - 1, x))
            y = max(0, min(grid_size - 1, y))
            positions.append((x, y))
        return positions


class ClusteredPositionStrategy:
    """Clustered distribution position generation strategy"""
    
    def __init__(self, cluster_centers: List[Tuple[int, int]], std_dev: float):
        self.cluster_centers = cluster_centers
        self.std_dev = std_dev
    
    def generate(self, count: int, grid_size: int) -> List[Tuple[int, int]]:
        """Generate clustered positions"""
        if not self.cluster_centers:
            # Fallback to single Gaussian at center
            strategy = GaussianPositionStrategy(grid_size // 2, grid_size // 2, self.std_dev)
            return strategy.generate(count, grid_size)
        
        positions = []
        clusters_count = len(self.cluster_centers)
        base_count = count // clusters_count
        
        for i, center in enumerate(self.cluster_centers):
            cluster_count = base_count + (1 if i < count % clusters_count else 0)
            strategy = GaussianPositionStrategy(center[0], center[1], self.std_dev)
            cluster_positions = strategy.generate(cluster_count, grid_size)
            positions.extend(cluster_positions)
        
        return positions


class CustomPositionStrategy:
    """Custom position generation strategy"""
    
    def __init__(self, custom_positions: List[Tuple[int, int]]):
        self.custom_positions = custom_positions
    
    def generate(self, count: int, grid_size: int) -> List[Tuple[int, int]]:
        """Generate custom positions"""
        return self.custom_positions[:count]


# ============================================================================
# Agent Creator Protocol
# ============================================================================

class AgentCreator(Protocol):
    """Protocol for agent creation functions"""
    def __call__(self, x: int, y: int, agent_id: int, **kwargs) -> BaseAgent:
        """Create agent"""
        ...


# ============================================================================
# Agent Factory
# ============================================================================

class AgentFactory:
    """
    Agent Factory Class
    
    Provides agent creation, configuration, and management functionality,
    supporting batch generation and custom configurations.
    Uses thread-safe singleton pattern to ensure global unique factory instance.
    """
    
    _instance: Optional['AgentFactory'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Thread-safe singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AgentFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize factory"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    # Core registries
                    self._registry: Dict[str, Type[BaseAgent]] = {}
                    self._custom_creators: Dict[str, AgentCreator] = {}
                    self._generation_history: List[GenerationConfig] = []
                    # Grid size used for position generation; can be overridden by simulation
                    self.grid_size: int = GRID_SIZE
                    # Logger and registry init
                    self._logger = self._setup_logger()
                    self._initialize_registry()
                    self._initialized = True
    
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_grid_size(self, grid_size: int) -> None:
        """
        Set grid size used for position generation.
        
        This should be called by the simulation so that agent positions
        are always generated within the actual environment bounds.
        """
        try:
            grid_size_int = int(grid_size)
            if grid_size_int <= 0:
                raise ValueError("grid_size must be positive")
            self.grid_size = grid_size_int
            if hasattr(self, "_logger"):
                self._logger.info(f"AgentFactory grid_size set to {self.grid_size}")
        except Exception as e:
            if hasattr(self, "_logger"):
                self._logger.warning(f"Failed to set AgentFactory grid_size ({grid_size}): {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('AgentFactory')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_registry(self) -> None:
        """Initialize agent registry"""
        # Clear registry to avoid duplicates
        self._registry.clear()
        self._custom_creators.clear()
        
        # Register from existing registry
        self._registry.update(AGENT_REGISTRY)
        
        # Register base agent types
        agent_types_to_register = [
            ("rule_based", RuleBasedAgent),
            ("conservative", ConservativeAgent),
            ("exploratory", ExploratoryAgent),
            ("adaptive", AdaptiveAgent)
        ]
        
        for agent_type, agent_class in agent_types_to_register:
            if agent_type not in self._registry:
                self.register_agent_type(agent_type, agent_class)
        
        # Register MARL agent types (if available)
        if MARL_AVAILABLE:
            try:
                if "iql" not in self._registry:
                    self.register_agent_type("iql", IQLAgent, 
                                            custom_creator=self._create_iql_agent_creator())
                
                if "qmix" not in self._registry:
                    self.register_agent_type("qmix", QMIXAgent, 
                                            custom_creator=self._create_qmix_agent_creator())
                
                self._logger.info("Registered MARL agents: iql, qmix")
            except Exception as e:
                self._logger.warning(f"MARL agent registration failed: {e}")
        
        self._logger.info(f"Agent factory initialized, registered {len(self._registry)} agent types")
    
    def _create_iql_agent_creator(self) -> AgentCreator:
        """Create IQL agent creator function"""
        def creator(x: int, y: int, agent_id: int, **kwargs) -> IQLAgent:
            """IQL agent creator"""
            # Create IQLConfig from kwargs
            iql_config = IQLConfig(
                state_dim=kwargs.get('state_dim', 64),
                action_dim=kwargs.get('action_dim', 8),
                hidden_dims=kwargs.get('hidden_dims', [128, 64]),
                network_type=kwargs.get('network_type', 'dqn'),
                use_double_dqn=kwargs.get('use_double_dqn', True),
                learning_rate=kwargs.get('learning_rate', 0.001),
                gamma=kwargs.get('gamma', 0.95),
                tau=kwargs.get('tau', 0.01),
                epsilon_start=kwargs.get('epsilon_start', 1.0),
                epsilon_end=kwargs.get('epsilon_end', 0.01),
                epsilon_decay=kwargs.get('epsilon_decay', 0.995),
                batch_size=kwargs.get('batch_size', 32),
                replay_buffer_size=kwargs.get('replay_buffer_size', 10000),
                learning_starts=kwargs.get('learning_starts', 1000),
                train_frequency=kwargs.get('train_frequency', 4),
                target_update_frequency=kwargs.get('target_update_frequency', 100)
            )
            
            agent = IQLAgent(
                x=x,
                y=y,
                agent_id=agent_id,
                config=iql_config,
                name=kwargs.get('name', f"iql_{agent_id}")
            )

            return agent
        
        return creator
    
    def _create_qmix_agent_creator(self) -> AgentCreator:
        """Create QMIX agent creator function"""
        def creator(x: int, y: int, agent_id: int, **kwargs) -> QMIXAgent:
            """QMIX agent creator"""
            # Extract trainer if provided
            trainer = kwargs.pop('trainer', None)
            
            # Create QMIXConfig from kwargs
            qmix_config = QMIXConfig(
                state_dim=kwargs.get('state_dim', 128),
                action_dim=kwargs.get('action_dim', 8),
                agent_hidden_dims=kwargs.get('agent_hidden_dims', [64, 64]),
                mixing_hidden_dim=kwargs.get('mixing_hidden_dim', 32),
                learning_rate=kwargs.get('learning_rate', 0.0005),
                gamma=kwargs.get('gamma', 0.99),
                tau=kwargs.get('tau', 0.005),
                epsilon_start=kwargs.get('epsilon_start', 1.0),
                epsilon_end=kwargs.get('epsilon_end', 0.05),
                epsilon_decay=kwargs.get('epsilon_decay', 0.999),
                batch_size=kwargs.get('batch_size', 32),
                replay_buffer_size=kwargs.get('replay_buffer_size', 10000),
                learning_starts=kwargs.get('learning_starts', 1000),
                train_frequency=kwargs.get('train_frequency', 4),
                target_update_frequency=kwargs.get('target_update_frequency', 200)
            )
            
            # Extract base agent config parameters
            vision_range = kwargs.pop('vision_range', 5)
            metabolism_rate = kwargs.pop('metabolism_rate', 1.0)
            
            # Create QMIX agent
            agent = QMIXAgent(
                x=x,
                y=y,
                agent_id=agent_id,
                config=qmix_config,
                trainer=trainer,
                name=kwargs.get('name', f"qmix_{agent_id}")
            )
            
            # Set vision_range and metabolism_rate if needed
            agent.vision_range = vision_range
            agent.metabolism_rate = metabolism_rate
            
            return agent
        
        return creator
    
    def register_agent_type(self, 
                          agent_type: str, 
                          agent_class: Type[BaseAgent],
                          custom_creator: Optional[AgentCreator] = None) -> None:
        """
        Register new agent type
        
        Args:
            agent_type: Agent type identifier
            agent_class: Agent class
            custom_creator: Custom creator function (optional)
            
        Raises:
            AgentConfigError: When agent class is invalid
        """
        if agent_type in self._registry:
            self._logger.warning(f"Agent type '{agent_type}' already registered, skipping duplicate registration")
            return
        
        if not issubclass(agent_class, BaseAgent):
            raise AgentConfigError(f"Agent class must inherit from BaseAgent")
        
        self._registry[agent_type] = agent_class
        
        if custom_creator:
            self._custom_creators[agent_type] = custom_creator
        
        self._logger.info(f"Registered agent type: {agent_type} -> {agent_class.__name__}")
    
    def unregister_agent_type(self, agent_type: str) -> None:
        """
        Unregister agent type
        
        Args:
            agent_type: Agent type to unregister
        """
        if agent_type in self._registry:
            del self._registry[agent_type]
            if agent_type in self._custom_creators:
                del self._custom_creators[agent_type]
            self._logger.info(f"Unregistered agent type: {agent_type}")
        else:
            self._logger.warning(f"Attempted to unregister unregistered agent type: {agent_type}")
    
    def get_available_agent_types(self) -> List[str]:
        """
        Get list of available agent types
        
        Returns:
            List of agent type identifiers
        """
        return list(self._registry.keys())
    
    def create_agent(self,
                    agent_type: str,
                    x: int,
                    y: int,
                    agent_id: int,
                    name: Optional[str] = None,
                    **kwargs) -> BaseAgent:
        """
        Create single agent
        
        Args:
            agent_type: Agent type identifier
            x: x coordinate
            y: y coordinate
            agent_id: Agent ID
            name: Agent name (optional)
            **kwargs: Additional parameters passed to agent constructor
            
        Returns:
            Created agent instance
            
        Raises:
            AgentCreationError: When agent creation fails
        """
        try:
            if agent_type not in self._registry:
                available_types = self.get_available_agent_types()
                raise AgentCreationError(
                    f"Unknown agent type: '{agent_type}'. Available types: {available_types}"
                )
            
            # Use custom creator if available
            if agent_type in self._custom_creators:
                agent = self._custom_creators[agent_type](x, y, agent_id, name=name, **kwargs)
            else:
                # Use standard constructor
                agent_class = self._registry[agent_type]
                constructor_params = inspect.signature(agent_class.__init__).parameters
                
                # Special handling for RuleBasedAgent and its subclasses
                if agent_type in ["rule_based", "conservative", "exploratory", "adaptive"]:
                    from src.core.agents import RuleBasedAgentConfig
                    # Create RuleBasedAgentConfig from kwargs
                    rule_config = RuleBasedAgentConfig(
                        vision_range=kwargs.get('vision_range', 5),
                        metabolism_rate=kwargs.get('metabolism_rate', 1.0),
                        movement_strategy=kwargs.get('movement_strategy', 'greedy'),
                        exploration_rate=kwargs.get('exploration_rate', 0.1)
                    )
                    # Filter valid constructor parameters
                    valid_kwargs = {
                        k: v for k, v in kwargs.items() 
                        if k in constructor_params and k not in ['x', 'y', 'agent_id', 'name', 
                                                                  'vision_range', 'metabolism_rate', 
                                                                  'movement_strategy', 'exploration_rate']
                    }
                    agent = agent_class(
                        x=x,
                        y=y,
                        agent_id=agent_id,
                        config=rule_config,
                        name=name,
                        **valid_kwargs
                    )
                else:
                    # Filter valid constructor parameters
                    valid_kwargs = {
                        k: v for k, v in kwargs.items() 
                        if k in constructor_params and k not in ['x', 'y', 'agent_id', 'name']
                    }
                    
                    agent = agent_class(
                        x=x,
                        y=y,
                        agent_id=agent_id,
                        name=name,
                        **valid_kwargs
                    )
            
            self._logger.debug(f"Created agent: {agent_type} ID={agent_id} at ({x}, {y})")
            return agent
            
        except Exception as e:
            error_msg = f"Agent creation failed (type: {agent_type}, ID: {agent_id}): {str(e)}"
            self._logger.error(error_msg)
            raise AgentCreationError(error_msg) from e
    
    def _get_position_strategy(self, position_config: PositionConfig) -> PositionGenerationStrategy:
        """Get position generation strategy based on configuration"""
        if position_config.distribution == DistributionType.UNIFORM:
            return UniformPositionStrategy()
        elif position_config.distribution == DistributionType.GAUSSIAN:
            center_x = position_config.center_x or GRID_SIZE // 2
            center_y = position_config.center_y or GRID_SIZE // 2
            return GaussianPositionStrategy(center_x, center_y, position_config.std_deviation)
        elif position_config.distribution == DistributionType.CLUSTERED:
            cluster_centers = position_config.cluster_centers or [
                (GRID_SIZE//3, GRID_SIZE//3), 
                (2*GRID_SIZE//3, 2*GRID_SIZE//3)
            ]
            return ClusteredPositionStrategy(cluster_centers, position_config.std_deviation)
        elif position_config.distribution == DistributionType.CUSTOM:
            return CustomPositionStrategy(position_config.custom_positions)
        else:
            self._logger.warning(f"Unknown distribution type: {position_config.distribution}, using uniform")
            return UniformPositionStrategy()
    
    def create_agents(self,
                     agent_configs: List[AgentTypeConfig],
                     generation_name: str = "default",
                     description: str = "",
                     trainers: Optional[Dict[str, Any]] = None) -> List[BaseAgent]:
        """
        Batch create agents
        
        Args:
            agent_configs: List of agent configurations
            generation_name: Generation batch name
            description: Batch description
            trainers: Trainer dictionary {agent_type: trainer}
            
        Returns:
            List of created agents
        """
        all_agents = []
        agent_id_counter = 0
        
        for agent_config in agent_configs:
            agents = self._create_agent_type_batch(agent_config, agent_id_counter, trainers or {})
            all_agents.extend(agents)
            agent_id_counter += len(agents)
        
        # Record generation history
        generation_config = GenerationConfig(
            name=generation_name,
            description=description,
            agent_configs=agent_configs,
            total_agents=len(all_agents)
        )
        self._generation_history.append(generation_config)
        
        self._logger.info(
            f"Batch creation complete: {generation_name} - {len(all_agents)} agents"
        )
        
        return all_agents
    
    def _create_agent_type_batch(self,
                           agent_config: AgentTypeConfig,
                           start_id: int,
                           trainers: Dict[str, Any]) -> List[BaseAgent]:
        """
        Create batch of single agent type
        
        Args:
            agent_config: Agent type configuration
            start_id: Starting ID
            trainers: Trainer dictionary
            
        Returns:
            List of created agents
        """
        agents = []
        
        # Generate positions using strategy pattern
        position_strategy = self._get_position_strategy(agent_config.position_config)
        # Use factory's grid_size (aligned with simulation environment) instead of global GRID_SIZE
        grid_size = self.grid_size or GRID_SIZE
        positions = position_strategy.generate(agent_config.count, grid_size)
        
        for i in range(agent_config.count):
            agent_id = start_id + i
            if i < len(positions):
                x, y = positions[i]
            else:
                # Fallback: sample uniformly within current grid bounds
                x, y = (random.randint(0, grid_size - 1),
                        random.randint(0, grid_size - 1))
            
            try:
                # Prepare creation parameters
                creation_kwargs = agent_config.config.copy()
                
                # Extract nested config if present
                if 'config' in creation_kwargs and isinstance(creation_kwargs['config'], dict):
                    nested_config = creation_kwargs.pop('config')
                    creation_kwargs.update(nested_config)
                
                # Add trainer if provided
                if agent_config.agent_type in trainers:
                    creation_kwargs['trainer'] = trainers[agent_config.agent_type]
                    self._logger.debug(f"Setting trainer for agent {agent_id}")
                
                # Create agent
                agent = self.create_agent(
                    agent_type=agent_config.agent_type,
                    x=x,
                    y=y,
                    agent_id=agent_id,
                    **creation_kwargs
                )
                agents.append(agent)
                
            except AgentCreationError as e:
                self._logger.warning(f"Skipping failed agent {agent_id}: {str(e)}")
                continue
        
        self._logger.info(
            f"Created {len(agents)} {agent_config.agent_type} agents"
        )
        
        return agents
    
    def create_from_config_file(self, config_path: Union[str, Path]) -> List[BaseAgent]:
        """
        Create agents from configuration file
        
        Args:
            config_path: Configuration file path
            
        Returns:
            List of created agents
            
        Raises:
            AgentConfigError: When config file format is invalid or read fails
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise AgentConfigError(f"Configuration file does not exist: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        raise AgentConfigError("YAML support not available. Install pyyaml.")
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    if not JSON_AVAILABLE:
                        raise AgentConfigError("JSON support not available.")
                    config_data = json.load(f)
                else:
                    raise AgentConfigError(f"Unsupported config file format: {config_path.suffix}")
            
            return self.create_from_config_dict(config_data)
            
        except Exception as e:
            error_msg = f"Failed to create agents from config file: {str(e)}"
            self._logger.error(error_msg)
            raise AgentConfigError(error_msg) from e
    
    def create_from_config_dict(self, config: Dict[str, Any]) -> List[BaseAgent]:
        """
        Create agents from configuration dictionary
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of created agents
        """
        generation_name = config.get('generation_name', 'from_config')
        description = config.get('description', '')
        agent_configs = []
        
        for agent_type_config in config.get('agents', []):
            agent_type = agent_type_config['type']
            count = agent_type_config['count']
            
            # Parse position configuration
            pos_config_data = agent_type_config.get('position', {})
            position_config = PositionConfig(
                distribution=DistributionType(pos_config_data.get('distribution', 'uniform')),
                center_x=pos_config_data.get('center_x'),
                center_y=pos_config_data.get('center_y'),
                cluster_centers=pos_config_data.get('cluster_centers', []),
                custom_positions=pos_config_data.get('custom_positions', []),
                std_deviation=pos_config_data.get('std_deviation', 20.0)
            )
            
            # Create agent type configuration
            agent_config = AgentTypeConfig(
                agent_type=agent_type,
                count=count,
                config=agent_type_config.get('config', {}),
                position_config=position_config
            )
            agent_configs.append(agent_config)
        
        return self.create_agents(agent_configs, generation_name, description)
    
    def get_generation_history(self) -> List[GenerationConfig]:
        """Get generation history"""
        return self._generation_history.copy()
    
    def clear_generation_history(self) -> None:
        """Clear generation history"""
        self._generation_history.clear()
        self._logger.info("Generation history cleared")
    
    def __repr__(self) -> str:
        return (f"AgentFactory(registered_types={len(self._registry)}, "
                f"generations={len(self._generation_history)})")


# ============================================================================
# Global Factory Instance
# ============================================================================

_global_factory: Optional[AgentFactory] = None
_factory_lock = threading.Lock()

def get_agent_factory() -> AgentFactory:
    """
    Get global agent factory instance
    
    Returns:
        Global AgentFactory instance
    """
    global _global_factory
    if _global_factory is None:
        with _factory_lock:
            if _global_factory is None:
                _global_factory = AgentFactory()
    return _global_factory


# ============================================================================
# Convenience Functions
# ============================================================================

def create_agents_simple(agent_type: str, count: int, **kwargs) -> List[BaseAgent]:
    """
    Simplified agent creation function (for quick testing and simple scenarios)
    
    Args:
        agent_type: Agent type
        count: Count
        **kwargs: Additional parameters
        
    Returns:
        List of created agents
    """
    factory = get_agent_factory()
    
    agent_config = AgentTypeConfig(
        agent_type=agent_type,
        count=count,
        config=kwargs
    )
    
    return factory.create_agents([agent_config], "simple_batch", "Simplified batch creation")
