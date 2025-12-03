# Architecture Documentation

## Overview

Umvili follows a modular architecture designed for extensibility and maintainability. The project is organized into clear, well-defined modules with single responsibilities.

## Project Structure

```
Umvili/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/                  # Configuration files
│   └── default.json         # Default configuration
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md     # This file
│   ├── CONFIGURATION.md    # Configuration guide
│   ├── DEVELOPMENT.md      # Development guide
│   └── zh/                 # Chinese documentation
└── src/                     # Source code
    ├── config/              # Configuration management
    │   ├── __init__.py
    │   ├── app_config.py
    │   ├── simulation_config.py
    │   ├── agent_config.py
    │   ├── ui_config.py
    │   ├── config_loader.py
    │   └── defaults.py
    ├── core/                # Core modules
    │   ├── __init__.py
    │   ├── agent_base.py
    │   ├── agents.py
    │   ├── agent_factory.py
    │   ├── environment.py
    │   ├── simulation.py
    │   └── reward_calculator.py
    ├── marl/                # MARL algorithms
    │   ├── __init__.py
    │   ├── iql_agent.py
    │   ├── qmix_agent.py
    │   ├── qmix_trainer.py
    │   ├── networks.py
    │   └── replay_buffer.py
    └── utils/               # Utilities
        ├── __init__.py
        ├── config.py
        ├── logging_config.py
        └── visualization.py
```

## Module Responsibilities

### 1. Configuration Module (`src/config/`)

**Purpose**: Centralized configuration management with validation and file loading support.

**Key Components**:
- `app_config.py`: Application-level configuration (simulation type, logging, UI options)
- `simulation_config.py`: Simulation environment configuration (grid size, sugar parameters)
- `agent_config.py`: Agent configuration for all agent types
- `ui_config.py`: UI and visualization configuration (window, fonts, colors)
- `config_loader.py`: Configuration loader supporting JSON/YAML formats
- `defaults.py`: Centralized default configuration values

**Design Principles**:
- Type safety using `dataclass`
- Configuration validation
- Support for configuration merging (default → file → user → CLI)
- Backward compatibility

### 2. Core Module (`src/core/`)

**Purpose**: Core functionality and base infrastructure.

**Key Components**:
- `agent_base.py`: Base agent class and interface definitions
- `agents.py`: Rule-based agent implementations (RuleBased, Conservative, Exploratory, Adaptive)
- `agent_factory.py`: Agent creation and management factory
- `environment.py`: Sugar environment implementation
- `simulation.py`: Simulation engine core logic
- `reward_calculator.py`: Unified reward calculation system

**Design Patterns**:
- Factory Pattern: Agent creation
- Strategy Pattern: Different agent behaviors
- Observer Pattern: Event handling

### 3. MARL Module (`src/marl/`)

**Purpose**: Multi-agent reinforcement learning algorithm implementations.

**Key Components**:
- `iql_agent.py`: Independent Q-Learning algorithm
- `qmix_agent.py`: QMIX algorithm implementation
- `qmix_trainer.py`: Centralized QMIX trainer
- `networks.py`: Neural network architectures (DQN, Dueling, Noisy, QMIX)
- `replay_buffer.py`: Experience replay system with prioritized replay

**Algorithms Supported**:
- **IQL**: Independent Q-Learning
- **QMIX**: Value-based MARL algorithm

### 4. Utilities Module (`src/utils/`)

**Purpose**: Common utilities and helper functions.

**Key Components**:
- `config.py`: Configuration compatibility layer
- `logging_config.py`: Logging system configuration
- `visualization.py`: Comprehensive visualization system including:
  - `MultiLineChart`: Multi-series real-time charting component
  - `AcademicVisualizationSystem`: Main visualization system with multi-view support
  - `AgentDistributionPanel`: Agent type and distribution visualization
  - `ActionDistributionPanel`: Action frequency and behavior analysis
  - View system: Overview, Training, Behavior, Debug views

## Design Principles

### 1. Single Responsibility Principle
Each module and class has a clear, single responsibility.

### 2. Open-Closed Principle
Easy to extend with new agent types and algorithms without modifying existing code.

### 3. Dependency Inversion
High-level modules depend on abstractions (interfaces) rather than concrete implementations.

### 4. Configuration-Driven
Behavior is controlled through configuration files, making the system flexible and testable.

## Configuration Architecture

### Configuration Hierarchy

```
Full Configuration
├── app (ApplicationConfig)
│   ├── simulation_type
│   ├── log_level
│   └── show_fps
├── simulation (SimulationConfig)
│   ├── grid_size
│   ├── cell_size
│   └── initial_agents
├── ui (UIConfig)
│   ├── window (WindowConfig)
│   ├── font (FontConfig)
│   └── color_scheme (ColorScheme)
└── agents (Dict[str, Dict])
    ├── rule_based
    ├── iql
    └── qmix
```

### Configuration Loading Priority

1. **Default Configuration** (`defaults.py`)
2. **File Configuration** (`config/*.json` or `config/*.yaml`)
3. **User Configuration** (dictionary passed in code)
4. **Command-line Arguments** (argparse in `main.py`)

## Extension Guide

### Adding a New Agent Type

1. Create agent class inheriting from `BaseAgent` in `src/core/agents.py`
2. Register in `agent_factory.py`
3. Add configuration in `src/config/agent_config.py`
4. Update `defaults.py` with default values

### Adding a New MARL Algorithm

1. Create algorithm implementation in `src/marl/`
2. Implement required interfaces from `agent_base.py`
3. Register in `agent_factory.py`
4. Add configuration support

### Adding New Configuration

1. Create configuration class in `src/config/`
2. Add default values in `defaults.py`
3. Update `config_loader.py` to handle loading
4. Export in `__init__.py`

## Backward Compatibility

- Old import paths are maintained in `src/utils/config.py`
- Legacy configuration formats are automatically converted
- Gradual migration path for existing code

## Visualization System

### Multi-View Interface

The visualization system supports four distinct views:

1. **Overview**: Environment statistics and agent type distribution
2. **Training**: Real-time training metrics (Loss, Q-value, TD Error, Exploration Rate)
3. **Behavior**: Behavior analysis (Action Distribution, Reward Trends, Policy Entropy)
4. **Debug**: Performance metrics and debugging information

### Chart Components

- **MultiLineChart**: Supports multiple data series with dynamic line management, sliding windows, and performance optimizations
- **Layout Management**: Grid-based chart positioning to prevent overlaps
- **Update Frequency Control**: Configurable update rates for different chart types

### Data Collection

- Training metrics collected from agents and trainers
- Action distribution tracking per algorithm type
- Real-time performance monitoring
- Historical data with configurable window sizes

## Performance Considerations

- Efficient agent creation using factory pattern
- Optimized visualization rendering with update frequency control
- Sliding window data management (200 data points)
- Configurable performance metrics
- Support for high-performance simulation modes

## Testing Strategy

- Unit tests for individual modules
- Integration tests for agent interactions
- Configuration validation tests
- Performance benchmarks

---

For more details, see:
- [Configuration Guide](CONFIGURATION.md)
- [Development Guide](DEVELOPMENT.md)

