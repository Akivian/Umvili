"""
Default Configuration Values

Centralized default configuration values for all components.
These defaults are used when no configuration is provided or when merging configurations.
"""

from typing import Dict, Any

# ============================================================================
# Application Defaults
# ============================================================================

DEFAULT_APP_CONFIG: Dict[str, Any] = {
    'simulation_type': 'comparative',
    'log_level': 'INFO',
    'log_file': 'marl_simulation.log',
    'show_fps': True,
    'show_debug_info': False,
    'enable_vsync': False,
}

# ============================================================================
# Simulation Defaults
# ============================================================================

DEFAULT_SIMULATION_CONFIG: Dict[str, Any] = {
    'grid_size': 80,
    'cell_size': 10,
    'initial_agents': 50,
    'sugar_growth_rate': 0.1,
    'max_sugar': 10.0,
    'max_steps': 10000,
}

# ============================================================================
# Agent Defaults
# ============================================================================

DEFAULT_AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'rule_based': {
        'vision_range': 4,
        'metabolism_rate': 1.0,
        'initial_sugar': 15.0,
        'movement_strategy': 'greedy',
        'exploration_rate': 0.1,
    },
    'conservative': {
        'vision_range': 4,
        'metabolism_rate': 0.8,
        'initial_sugar': 15.0,
        'movement_strategy': 'conservative',
        'exploration_rate': 0.05,
    },
    'exploratory': {
        'vision_range': 6,
        'metabolism_rate': 1.2,
        'initial_sugar': 15.0,
        'movement_strategy': 'exploratory',
        'exploration_rate': 0.3,
    },
    'adaptive': {
        'vision_range': 5,
        'metabolism_rate': 1.0,
        'initial_sugar': 15.0,
        'movement_strategy': 'greedy',
        'exploration_rate': 0.1,
    },
    'iql': {
        'vision_range': 5,
        'metabolism_rate': 0.8,
        'initial_sugar': 15.0,
        'state_dim': 64,
        'action_dim': 8,
        'hidden_dims': [128, 64],
        'network_type': 'dqn',
        'use_double_dqn': True,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'tau': 0.01,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'replay_buffer_size': 10000,
        'learning_starts': 1000,
        'train_frequency': 4,
        'target_update_frequency': 100,
    },
    'qmix': {
        'vision_range': 5,
        'metabolism_rate': 0.8,
        'initial_sugar': 15.0,
        'state_dim': 128,
        'action_dim': 8,
        'agent_hidden_dims': [64, 64],
        'mixing_hidden_dim': 32,
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'tau': 0.005,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.999,
        'batch_size': 32,
        'replay_buffer_size': 10000,
        'learning_starts': 1000,
        'train_frequency': 4,
        'target_update_frequency': 200,
    },
}

# ============================================================================
# UI Defaults
# ============================================================================

DEFAULT_UI_CONFIG: Dict[str, Any] = {
    'window': {
        'width': 1400,
        'height': 900,
        'fps': 60,
        'title': 'MARL沙盘平台 - 多智能体强化学习模拟',
        'enable_vsync': False,
    },
    'font': {
        'TITLE': 24,
        'HEADING': 20,
        'BODY': 16,
        'SMALL': 14,
        'TINY': 12,
    },
}

