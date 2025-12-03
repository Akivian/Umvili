"""
Configuration Module - Backward Compatibility Layer

This module provides backward compatibility for the old configuration system.
All new code should use src.config instead.

DEPRECATED: This module is maintained for backward compatibility only.
Please use src.config for new code.
"""

# Import from new config system
from src.config.ui_config import (
    COLORS,
    FONT_SIZES,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    GRID_SIZE,
    CELL_SIZE
)
from src.config.simulation_config import SimulationConfig
from src.config.defaults import (
    DEFAULT_SIMULATION_CONFIG as DEFAULT_CONFIG,
    DEFAULT_AGENT_CONFIGS as AGENT_DEFAULT_CONFIGS
)

# Re-export for backward compatibility
__all__ = [
    'COLORS',
    'FONT_SIZES',
    'WINDOW_WIDTH',
    'WINDOW_HEIGHT',
    'FPS',
    'GRID_SIZE',
    'CELL_SIZE',
    'DEFAULT_CONFIG',
    'AGENT_DEFAULT_CONFIGS',
]