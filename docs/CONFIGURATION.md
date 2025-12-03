# Configuration Guide

## Overview

Umvili uses a unified configuration management system that supports multiple configuration sources with a clear priority hierarchy.

## Configuration Sources

The configuration system supports the following sources (in priority order):

1. **Default Configuration** - Built-in defaults in `src/config/defaults.py`
2. **File Configuration** - JSON/YAML files in `config/` directory
3. **User Configuration** - Dictionary passed programmatically
4. **Command-line Arguments** - Override specific settings

## Configuration Structure

### Application Configuration (`app`)

Controls application-level settings:

```json
{
  "app": {
    "simulation_type": "comparative",
    "log_level": "INFO",
    "show_fps": true
  }
}
```

**Options**:
- `simulation_type`: `"default"`, `"comparative"`, `"training"`, `"performance"`
- `log_level`: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- `show_fps`: Boolean to display FPS counter

### Simulation Configuration (`simulation`)

Controls the simulation environment:

```json
{
  "simulation": {
    "grid_size": 80,
    "cell_size": 10,
    "initial_agents": 50,
    "sugar_growth_rate": 0.1,
    "max_sugar": 10.0
  }
}
```

**Options**:
- `grid_size`: Grid dimensions (integer)
- `cell_size`: Cell size in pixels (integer)
- `initial_agents`: Number of agents at start (integer)
- `sugar_growth_rate`: Sugar regeneration rate (float)
- `max_sugar`: Maximum sugar per cell (float)

### UI Configuration (`ui`)

Controls visualization and UI:

```json
{
  "ui": {
    "window": {
      "width": 1400,
      "height": 900,
      "fps": 60,
      "title": "Umvili - MARL Platform",
      "enable_vsync": true
    },
    "font": {
      "family": "Arial",
      "size": 14
    }
  }
}
```

### Agent Configuration (`agents`)

Configures individual agent types:

```json
{
  "agents": {
    "rule_based": {
      "vision_range": 5,
      "metabolism_rate": 1.0,
      "movement_strategy": "greedy"
    },
    "iql": {
      "learning_rate": 0.001,
      "gamma": 0.95,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01
    },
    "qmix": {
      "learning_rate": 0.0005,
      "gamma": 0.99,
      "mixing_hidden_dim": 32
    }
  }
}
```

## Usage Examples

### Using Default Configuration

```python
from src.config import (
    ApplicationConfig,
    SimulationConfig,
    UIConfig
)

app_config = ApplicationConfig.default()
sim_config = SimulationConfig.default()
ui_config = UIConfig.default()
```

### Loading from File

```python
from src.config import ConfigLoader

# Load full configuration
full_config = ConfigLoader.load_full_config('config/custom.json')
configs = ConfigLoader.create_config_objects(full_config)

app_config = configs['app']
sim_config = configs['simulation']
ui_config = configs['ui']
```

### Programmatic Configuration

```python
from src.config import ApplicationConfig

# Create from dictionary
app_config = ApplicationConfig.from_dict({
    'simulation_type': 'training',
    'log_level': 'DEBUG'
})

# Merge with additional settings
app_config = app_config.merge({'show_fps': False})
```

### Command-line Configuration

```bash
# Override specific settings
python main.py --simulation-type training --grid-size 100 --agents 200
```

## Configuration File Formats

### JSON Format

```json
{
  "app": {
    "simulation_type": "comparative",
    "log_level": "INFO"
  },
  "simulation": {
    "grid_size": 80,
    "initial_agents": 50
  }
}
```

### YAML Format

```yaml
app:
  simulation_type: comparative
  log_level: INFO

simulation:
  grid_size: 80
  initial_agents: 50
```

## Configuration Validation

All configuration classes support validation:

```python
app_config = ApplicationConfig(...)
is_valid, error_msg = app_config.validate()

if not is_valid:
    raise ValueError(f"Configuration error: {error_msg}")
```

## Best Practices

1. **Use Configuration Files**: For complex configurations, use JSON/YAML files
2. **Validate Early**: Validate configuration at application startup
3. **Type Safety**: Leverage dataclass type hints
4. **Centralized Defaults**: Keep all defaults in `defaults.py`
5. **Documentation**: Document custom configuration options

## Advanced Configuration

### Configuration Merging

Configurations are merged in priority order:

```python
# Default config
default_config = ApplicationConfig.default()

# File config (overrides defaults)
file_config = ConfigLoader.load_app_config('config/custom.json')

# User config (overrides file config)
user_config = ApplicationConfig.from_dict({'log_level': 'DEBUG'})

# Final config (user > file > default)
final_config = default_config.merge(file_config).merge(user_config)
```

### Environment-Specific Configuration

Create different configuration files for different environments:

- `config/development.json` - Development settings
- `config/production.json` - Production settings
- `config/testing.json` - Testing settings

## Troubleshooting

### Common Issues

1. **Configuration not loading**: Check file path and format
2. **Validation errors**: Review error messages for invalid values
3. **Type errors**: Ensure configuration matches expected types

### Debugging

Enable debug logging to see configuration loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

For more details, see:
- [Architecture Documentation](ARCHITECTURE.md)
- [Development Guide](DEVELOPMENT.md)

