"""
Configuration Builder Module

Provides functionality to build simulation and agent configurations from UI parameters.
Supports validation, error handling, and flexible configuration construction.

Design Principles:
- Type Safety: Strong type hints and validation
- Error Handling: Comprehensive validation with clear error messages
- Extensibility: Easy to add new configuration options
- Backward Compatibility: Works with existing configuration system

Author: MARL Platform Team
Version: 1.0.0
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.config.simulation_config import SimulationConfig
from src.config.agent_config import IQLConfig, QMIXConfig, RuleBasedAgentConfig
from src.core.agent_factory import AgentTypeConfig, PositionConfig, DistributionType
from src.config.defaults import (
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_AGENT_CONFIGS
)

logger = logging.getLogger(__name__)


class ConfigBuilderError(Exception):
    """Base exception for configuration builder errors"""
    pass


class ConfigValidationError(ConfigBuilderError):
    """Exception raised when configuration validation fails"""
    pass


class ConfigBuilder:
    """
    Configuration Builder
    
    Builds simulation and agent configurations from UI parameters.
    Provides validation and error handling.
    """
    
    # Algorithm combination modes
    ALGORITHM_MODES = {
        'iql_only': 'iql_only',
        'qmix_only': 'qmix_only',
        'iql_qmix': 'iql_qmix',
        'baseline_only': 'baseline_only',
        'mixed': 'mixed',
        'custom': 'custom'
    }
    
    # Resource types
    RESOURCE_TYPES = ['sugar', 'spice', 'hazard']
    
    def __init__(self):
        """Initialize configuration builder"""
        self.logger = logger
    
    def collect_ui_params(self, ui_controls: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect parameters from UI controls
        
        Args:
            ui_controls: Dictionary mapping control IDs to their values
            
        Returns:
            Dictionary of collected parameters organized by category
        """
        try:
            params = {
                'simulation': self._collect_simulation_params(ui_controls),
                'environment': self._collect_environment_params(ui_controls),
                'algorithm_combination': self._collect_algorithm_combination(ui_controls),
                'agents': self._collect_agent_params(ui_controls),
                'resource_enabled': self._collect_resource_enabled(ui_controls)
            }
            
            self.logger.debug(f"Collected UI parameters: {params}")
            return params
            
        except Exception as e:
            self.logger.error(f"Failed to collect UI parameters: {e}", exc_info=True)
            raise ConfigBuilderError(f"Failed to collect UI parameters: {e}")
    
    def _collect_simulation_params(self, ui_controls: Dict[str, Any]) -> Dict[str, Any]:
        """Collect simulation parameters"""
        return {
            'grid_size': ui_controls.get('grid_size', DEFAULT_SIMULATION_CONFIG['grid_size']),
            'cell_size': ui_controls.get('cell_size', DEFAULT_SIMULATION_CONFIG['cell_size']),
            'initial_agents': ui_controls.get('total_agents', DEFAULT_SIMULATION_CONFIG['initial_agents']),
        }
    
    def _collect_environment_params(self, ui_controls: Dict[str, Any]) -> Dict[str, Any]:
        """Collect environment parameters"""
        return {
            'sugar_growth_rate': ui_controls.get('sugar_growth_rate', DEFAULT_SIMULATION_CONFIG['sugar_growth_rate']),
            'max_sugar': ui_controls.get('max_sugar', DEFAULT_SIMULATION_CONFIG['max_sugar']),
            'spice_growth_rate': ui_controls.get('spice_growth_rate', 0.02),
            'max_spice': ui_controls.get('max_spice', 6.0),
            'hazard_decay_rate': ui_controls.get('hazard_decay_rate', 0.01),
            'hazard_target_fraction': ui_controls.get('hazard_target_fraction', 0.09),
        }
    
    def _collect_algorithm_combination(self, ui_controls: Dict[str, Any]) -> str:
        """Collect algorithm combination mode"""
        mode = ui_controls.get('algorithm_combination', 'iql_qmix')
        if mode not in self.ALGORITHM_MODES.values():
            self.logger.warning(f"Unknown algorithm mode: {mode}, using default")
            mode = 'iql_qmix'
        return mode
    
    def _collect_agent_params(self, ui_controls: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Collect agent parameters for each agent type"""
        algorithm_mode = self._collect_algorithm_combination(ui_controls)
        
        agent_params = {}
        
        # Collect IQL parameters
        if algorithm_mode in ['iql_only', 'iql_qmix', 'mixed', 'custom']:
            agent_params['iql'] = self._collect_iql_params(ui_controls)
        
        # Collect QMIX parameters
        if algorithm_mode in ['qmix_only', 'iql_qmix', 'mixed', 'custom']:
            agent_params['qmix'] = self._collect_qmix_params(ui_controls)
        
        # Collect rule-based parameters
        if algorithm_mode in ['baseline_only', 'mixed', 'custom']:
            agent_params['rule_based'] = self._collect_rule_based_params(ui_controls)
            agent_params['conservative'] = self._collect_rule_based_params(ui_controls, 'conservative')
            agent_params['exploratory'] = self._collect_rule_based_params(ui_controls, 'exploratory')
            agent_params['adaptive'] = self._collect_rule_based_params(ui_controls, 'adaptive')
        
        return agent_params
    
    def _collect_iql_params(self, ui_controls: Dict[str, Any]) -> Dict[str, Any]:
        """Collect IQL agent parameters"""
        defaults = DEFAULT_AGENT_CONFIGS.get('iql', {})
        
        # Get agent counts (for custom mode)
        count = ui_controls.get('iql_count', 20)
        
        # Learning parameters
        learning_rate = ui_controls.get('iql_learning_rate', defaults.get('learning_rate', 0.001))
        gamma = ui_controls.get('iql_gamma', defaults.get('gamma', 0.95))
        
        # Exploration parameters
        epsilon_start = ui_controls.get('iql_epsilon_start', defaults.get('epsilon_start', 1.0))
        epsilon_end = ui_controls.get('iql_epsilon_end', defaults.get('epsilon_end', 0.01))
        epsilon_decay = ui_controls.get('iql_epsilon_decay', defaults.get('epsilon_decay', 0.995))
        
        # Training parameters
        batch_size = ui_controls.get('iql_batch_size', defaults.get('batch_size', 32))
        replay_buffer_size = ui_controls.get('iql_replay_buffer_size', defaults.get('replay_buffer_size', 10000))
        train_frequency = ui_controls.get('iql_train_frequency', defaults.get('train_frequency', 4))
        target_update_frequency = ui_controls.get('iql_target_update_frequency', defaults.get('target_update_frequency', 100))
        
        # Network parameters
        hidden_dims_str = ui_controls.get('iql_hidden_dims', None)
        if hidden_dims_str:
            try:
                hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(',')]
            except (ValueError, AttributeError):
                hidden_dims = defaults.get('hidden_dims', [128, 64])
        else:
            hidden_dims = defaults.get('hidden_dims', [128, 64])
        
        network_type = ui_controls.get('iql_network_type', defaults.get('network_type', 'dqn'))
        
        # Base parameters
        vision_range = ui_controls.get('iql_vision_range', defaults.get('vision_range', 5))
        metabolism_rate = ui_controls.get('iql_metabolism_rate', defaults.get('metabolism_rate', 0.8))
        initial_sugar = ui_controls.get('iql_initial_sugar', defaults.get('initial_sugar', 15.0))
        
        return {
            'count': count,
            'vision_range': vision_range,
            'metabolism_rate': metabolism_rate,
            'initial_sugar': initial_sugar,
            'state_dim': defaults.get('state_dim', 64),
            'action_dim': defaults.get('action_dim', 8),
            'hidden_dims': hidden_dims,
            'network_type': network_type,
            'use_double_dqn': defaults.get('use_double_dqn', True),
            'learning_rate': learning_rate,
            'gamma': gamma,
            'tau': defaults.get('tau', 0.01),
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            'learning_starts': defaults.get('learning_starts', 1000),
            'train_frequency': train_frequency,
            'target_update_frequency': target_update_frequency,
        }
    
    def _collect_qmix_params(self, ui_controls: Dict[str, Any]) -> Dict[str, Any]:
        """Collect QMIX agent parameters"""
        defaults = DEFAULT_AGENT_CONFIGS.get('qmix', {})
        
        # Get agent counts (for custom mode)
        count = ui_controls.get('qmix_count', 20)
        
        # Learning parameters
        learning_rate = ui_controls.get('qmix_learning_rate', defaults.get('learning_rate', 0.0005))
        gamma = ui_controls.get('qmix_gamma', defaults.get('gamma', 0.99))
        
        # Exploration parameters
        epsilon_start = ui_controls.get('qmix_epsilon_start', defaults.get('epsilon_start', 1.0))
        epsilon_end = ui_controls.get('qmix_epsilon_end', defaults.get('epsilon_end', 0.05))
        epsilon_decay = ui_controls.get('qmix_epsilon_decay', defaults.get('epsilon_decay', 0.999))
        
        # Training parameters
        batch_size = ui_controls.get('qmix_batch_size', defaults.get('batch_size', 32))
        replay_buffer_size = ui_controls.get('qmix_replay_buffer_size', defaults.get('replay_buffer_size', 10000))
        train_frequency = ui_controls.get('qmix_train_frequency', defaults.get('train_frequency', 4))
        target_update_frequency = ui_controls.get('qmix_target_update_frequency', defaults.get('target_update_frequency', 200))
        
        # Network parameters
        agent_hidden_dims_str = ui_controls.get('qmix_agent_hidden_dims', None)
        if agent_hidden_dims_str:
            try:
                agent_hidden_dims = [int(x.strip()) for x in agent_hidden_dims_str.split(',')]
            except (ValueError, AttributeError):
                agent_hidden_dims = defaults.get('agent_hidden_dims', [64, 64])
        else:
            agent_hidden_dims = defaults.get('agent_hidden_dims', [64, 64])
        
        mixing_hidden_dim = ui_controls.get('qmix_mixing_hidden_dim', defaults.get('mixing_hidden_dim', 32))
        
        # Base parameters
        vision_range = ui_controls.get('qmix_vision_range', defaults.get('vision_range', 5))
        metabolism_rate = ui_controls.get('qmix_metabolism_rate', defaults.get('metabolism_rate', 0.8))
        initial_sugar = ui_controls.get('qmix_initial_sugar', defaults.get('initial_sugar', 15.0))
        
        return {
            'count': count,
            'vision_range': vision_range,
            'metabolism_rate': metabolism_rate,
            'initial_sugar': initial_sugar,
            'state_dim': defaults.get('state_dim', 128),
            'action_dim': defaults.get('action_dim', 8),
            'agent_hidden_dims': agent_hidden_dims,
            'mixing_hidden_dim': mixing_hidden_dim,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'tau': defaults.get('tau', 0.005),
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            'learning_starts': defaults.get('learning_starts', 1000),
            'train_frequency': train_frequency,
            'target_update_frequency': target_update_frequency,
        }
    
    def _collect_rule_based_params(self, ui_controls: Dict[str, Any], agent_type: str = 'rule_based') -> Dict[str, Any]:
        """Collect rule-based agent parameters"""
        defaults = DEFAULT_AGENT_CONFIGS.get(agent_type, DEFAULT_AGENT_CONFIGS.get('rule_based', {}))
        
        # Get agent counts (for custom mode)
        count_key = f'{agent_type}_count'
        count = ui_controls.get(count_key, 10)
        
        # Base parameters
        vision_range = ui_controls.get(f'{agent_type}_vision_range', defaults.get('vision_range', 4))
        metabolism_rate = ui_controls.get(f'{agent_type}_metabolism_rate', defaults.get('metabolism_rate', 1.0))
        initial_sugar = ui_controls.get(f'{agent_type}_initial_sugar', defaults.get('initial_sugar', 15.0))
        movement_strategy = defaults.get('movement_strategy', 'greedy')
        exploration_rate = defaults.get('exploration_rate', 0.1)
        
        return {
            'count': count,
            'vision_range': vision_range,
            'metabolism_rate': metabolism_rate,
            'initial_sugar': initial_sugar,
            'movement_strategy': movement_strategy,
            'exploration_rate': exploration_rate,
        }
    
    def _collect_resource_enabled(self, ui_controls: Dict[str, Any]) -> Dict[str, bool]:
        """Collect resource type enabled flags"""
        return {
            'sugar': ui_controls.get('resource_sugar_enabled', True),
            'spice': ui_controls.get('resource_spice_enabled', True),
            'hazard': ui_controls.get('resource_hazard_enabled', True),
        }
    
    def build_simulation_config(self, params: Dict[str, Any]) -> SimulationConfig:
        """
        Build SimulationConfig from parameters
        
        Args:
            params: Parameters dictionary from collect_ui_params
            
        Returns:
            SimulationConfig object
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            sim_params = params.get('simulation', {})
            env_params = params.get('environment', {})
            
            # Merge simulation and environment parameters
            config_dict = {
                'grid_size': sim_params.get('grid_size', DEFAULT_SIMULATION_CONFIG['grid_size']),
                'cell_size': sim_params.get('cell_size', DEFAULT_SIMULATION_CONFIG['cell_size']),
                'initial_agents': sim_params.get('initial_agents', DEFAULT_SIMULATION_CONFIG['initial_agents']),
                'sugar_growth_rate': env_params.get('sugar_growth_rate', DEFAULT_SIMULATION_CONFIG['sugar_growth_rate']),
                'max_sugar': env_params.get('max_sugar', DEFAULT_SIMULATION_CONFIG['max_sugar']),
                'max_steps': DEFAULT_SIMULATION_CONFIG.get('max_steps', 10000),
            }
            
            # Validate before creating
            config = SimulationConfig.from_dict(config_dict)
            is_valid, error_msg = config.validate()
            
            if not is_valid:
                raise ConfigValidationError(f"Simulation configuration validation failed: {error_msg}")
            
            self.logger.info(f"Built simulation config: grid_size={config.grid_size}, agents={config.initial_agents}")
            return config
            
        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            self.logger.error(f"Failed to build simulation config: {e}", exc_info=True)
            raise ConfigBuilderError(f"Failed to build simulation config: {e}")
    
    def build_agent_configs(self, params: Dict[str, Any]) -> List[AgentTypeConfig]:
        """
        Build list of AgentTypeConfig from parameters
        
        Args:
            params: Parameters dictionary from collect_ui_params
            
        Returns:
            List of AgentTypeConfig objects
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            algorithm_mode = params.get('algorithm_combination', 'iql_qmix')
            agent_params = params.get('agents', {})
            
            agent_configs = []
            
            # Determine agent distribution
            distribution_str = params.get('agent_distribution', 'uniform')
            distribution_map = {
                'uniform': DistributionType.UNIFORM,
                'clustered': DistributionType.CLUSTERED,
                'random': DistributionType.UNIFORM,  # Random uses uniform for now
            }
            distribution = distribution_map.get(distribution_str, DistributionType.UNIFORM)
            position_config = PositionConfig(distribution=distribution)
            
            # Build IQL configs
            if 'iql' in agent_params:
                iql_params = agent_params['iql']
                count = iql_params.pop('count', 20)
                if count > 0:
                    agent_configs.append(AgentTypeConfig(
                        agent_type='iql',
                        count=count,
                        config=iql_params,
                        position_config=position_config
                    ))
            
            # Build QMIX configs
            if 'qmix' in agent_params:
                qmix_params = agent_params['qmix']
                count = qmix_params.pop('count', 20)
                if count > 0:
                    agent_configs.append(AgentTypeConfig(
                        agent_type='qmix',
                        count=count,
                        config=qmix_params,
                        position_config=position_config
                    ))
            
            # Build rule-based configs
            rule_based_types = ['rule_based', 'conservative', 'exploratory', 'adaptive']
            for agent_type in rule_based_types:
                if agent_type in agent_params:
                    type_params = agent_params[agent_type]
                    count = type_params.pop('count', 10)
                    if count > 0:
                        agent_configs.append(AgentTypeConfig(
                            agent_type=agent_type,
                            count=count,
                            config=type_params,
                            position_config=position_config
                        ))
            
            # Validate agent configs
            total_agents = sum(config.count for config in agent_configs)
            if total_agents == 0:
                raise ConfigValidationError("No agents configured. At least one agent type must have count > 0")
            
            self.logger.info(f"Built {len(agent_configs)} agent type configs, total agents: {total_agents}")
            return agent_configs
            
        except Exception as e:
            if isinstance(e, ConfigValidationError):
                raise
            self.logger.error(f"Failed to build agent configs: {e}", exc_info=True)
            raise ConfigBuilderError(f"Failed to build agent configs: {e}")
    
    def build_full_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build full configuration dictionary for simulation.reset()
        
        Args:
            params: Parameters dictionary from collect_ui_params
            
        Returns:
            Full configuration dictionary compatible with simulation.reset()
        """
        try:
            sim_config = self.build_simulation_config(params)
            agent_configs = self.build_agent_configs(params)
            
            # Build full config dict
            full_config = {
                'grid_size': sim_config.grid_size,
                'cell_size': sim_config.cell_size,
                'initial_agents': sim_config.initial_agents,
                'sugar_growth_rate': sim_config.sugar_growth_rate,
                'max_sugar': sim_config.max_sugar,
                'agent_configs': agent_configs,
                # Environment parameters
                'spice_growth_rate': params.get('environment', {}).get('spice_growth_rate', 0.02),
                'max_spice': params.get('environment', {}).get('max_spice', 6.0),
                'hazard_decay_rate': params.get('environment', {}).get('hazard_decay_rate', 0.01),
                'hazard_target_fraction': params.get('environment', {}).get('hazard_target_fraction', 0.09),
                'resource_enabled': params.get('resource_enabled', {
                    'sugar': True,
                    'spice': True,
                    'hazard': True
                }),
            }
            
            self.logger.info("Built full configuration dictionary")
            return full_config
            
        except Exception as e:
            self.logger.error(f"Failed to build full config: {e}", exc_info=True)
            raise ConfigBuilderError(f"Failed to build full config: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate simulation config
            sim_config = self.build_simulation_config(config)
            is_valid, error_msg = sim_config.validate()
            if not is_valid:
                return False, error_msg
            
            # Validate agent configs
            agent_configs = self.build_agent_configs(config)
            total_agents = sum(cfg.count for cfg in agent_configs)
            
            # Check total agents against grid size
            max_recommended = sim_config.grid_size * 2
            if total_agents > max_recommended:
                return False, f"Total agents ({total_agents}) exceeds recommended maximum ({max_recommended}) for grid size {sim_config.grid_size}"
            
            if total_agents == 0:
                return False, "No agents configured"
            
            # Validate environment parameters
            env_params = config.get('environment', {})
            if env_params.get('sugar_growth_rate', 0.1) < 0 or env_params.get('sugar_growth_rate', 0.1) > 1:
                return False, "sugar_growth_rate must be between 0 and 1"
            
            if env_params.get('max_sugar', 10.0) < 1 or env_params.get('max_sugar', 10.0) > 100:
                return False, "max_sugar must be between 1 and 100"
            
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"Configuration validation error: {error_msg}")
            return False, error_msg

