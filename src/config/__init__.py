"""
Configuration Module - Unified Configuration Management System

This module provides a centralized configuration management system for the MARL platform.
All configuration classes and utilities are organized by domain for clarity and maintainability.

Structure:
- app_config: Application-level configuration
- simulation_config: Simulation environment configuration
- agent_config: Agent-specific configurations
- ui_config: UI and visualization configuration
- config_loader: Configuration loading and validation utilities
- defaults: Default configuration values
"""

from src.config.app_config import ApplicationConfig, ApplicationState
from src.config.simulation_config import SimulationConfig, EnvironmentConfig
from src.config.agent_config import (
    BaseAgentConfig, 
    RuleBasedAgentConfig,
    IQLConfig,
    QMIXConfig,
    AgentDefaultConfigs
)
from src.config.ui_config import (
    UIConfig,
    ColorScheme,
    FontConfig,
    WindowConfig
)
from src.config.config_loader import (
    ConfigLoader,
    load_config_from_file,
    save_config_to_file,
    merge_configs
)
from src.config.defaults import (
    DEFAULT_APP_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_AGENT_CONFIGS,
    DEFAULT_UI_CONFIG
)

__all__ = [
    # Application
    'ApplicationConfig',
    'ApplicationState',
    # Simulation
    'SimulationConfig',
    'EnvironmentConfig',
    # Agent
    'BaseAgentConfig',
    'RuleBasedAgentConfig',
    'IQLConfig',
    'QMIXConfig',
    'AgentDefaultConfigs',
    # UI
    'UIConfig',
    'ColorScheme',
    'FontConfig',
    'WindowConfig',
    # Loader
    'ConfigLoader',
    'load_config_from_file',
    'save_config_to_file',
    'merge_configs',
    # Defaults
    'DEFAULT_APP_CONFIG',
    'DEFAULT_SIMULATION_CONFIG',
    'DEFAULT_AGENT_CONFIGS',
    'DEFAULT_UI_CONFIG',
]

