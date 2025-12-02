# 配置管理使用指南

## 概述

项目现在使用统一的配置管理系统，所有配置集中在`src/config/`目录下，支持从文件加载、验证和合并。

## 配置结构

### 配置层次

```
完整配置
├── app (ApplicationConfig)
│   ├── simulation_type: 模拟类型
│   ├── log_level: 日志级别
│   ├── show_fps: 显示FPS
│   └── ...
├── simulation (SimulationConfig)
│   ├── grid_size: 网格大小
│   ├── cell_size: 单元格大小
│   ├── initial_agents: 初始智能体数量
│   └── ...
├── ui (UIConfig)
│   ├── window (WindowConfig)
│   │   ├── width: 窗口宽度
│   │   ├── height: 窗口高度
│   │   ├── fps: 帧率
│   │   └── ...
│   ├── font (FontConfig)
│   └── color_scheme (ColorScheme)
└── agents (Dict[str, Dict])
    ├── rule_based
    ├── iql
    ├── qmix
    └── ...
```

## 使用方式

### 1. 使用默认配置

```python
from src.config import (
    ApplicationConfig,
    SimulationConfig,
    UIConfig
)

# 创建默认配置
app_config = ApplicationConfig.default()
sim_config = SimulationConfig.default()
ui_config = UIConfig.default()
```

### 2. 从文件加载配置

```python
from src.config import ConfigLoader

# 加载完整配置
full_config = ConfigLoader.load_full_config('config/custom.json')
configs = ConfigLoader.create_config_objects(full_config)

app_config = configs['app']
sim_config = configs['simulation']
ui_config = configs['ui']
```

### 3. 在代码中创建配置

```python
from src.config import ApplicationConfig, SimulationConfig

# 从字典创建
app_config = ApplicationConfig.from_dict({
    'simulation_type': 'training',
    'log_level': 'DEBUG'
})

# 合并配置
app_config = app_config.merge({'show_fps': False})
```

### 4. 在main.py中使用

```python
# 使用配置文件
app = MARLApplication(config_path='config/custom.json')

# 使用代码配置
app = MARLApplication(config={
    'simulation': {'grid_size': 100},
    'app': {'simulation_type': 'training'}
})

# 混合使用
app = MARLApplication(
    config={'simulation': {'initial_agents': 200}},
    config_path='config/base.json'
)
```

## 配置文件格式

### JSON格式示例

```json
{
  "app": {
    "simulation_type": "comparative",
    "log_level": "INFO",
    "show_fps": true
  },
  "simulation": {
    "grid_size": 80,
    "cell_size": 10,
    "initial_agents": 50,
    "sugar_growth_rate": 0.1,
    "max_sugar": 10.0
  },
  "ui": {
    "window": {
      "width": 1400,
      "height": 900,
      "fps": 60,
      "title": "MARL沙盘平台"
    }
  }
}
```

### YAML格式示例

```yaml
app:
  simulation_type: comparative
  log_level: INFO
  show_fps: true

simulation:
  grid_size: 80
  cell_size: 10
  initial_agents: 50

ui:
  window:
    width: 1400
    height: 900
    fps: 60
```

## 配置验证

所有配置类都支持验证：

```python
app_config = ApplicationConfig(...)
is_valid, error_msg = app_config.validate()
if not is_valid:
    print(f"配置错误: {error_msg}")
```

## 配置合并

配置按以下优先级合并：

1. **默认配置** (`defaults.py`)
2. **文件配置** (`config/*.json` 或 `config/*.yaml`)
3. **用户配置** (代码中传入的字典)
4. **命令行参数** (main.py中的argparse)

## 向后兼容

旧的导入路径仍然有效：

```python
# 旧方式（仍然有效）
from src.utils.config import COLORS, FONT_SIZES, GRID_SIZE

# 新方式（推荐）
from src.config.ui_config import COLORS, FONT_SIZES
from src.config.simulation_config import SimulationConfig
```

## 最佳实践

1. **使用配置文件**: 对于复杂配置，使用JSON/YAML文件
2. **验证配置**: 在创建对象后验证配置
3. **使用类型提示**: 利用dataclass的类型安全特性
4. **集中管理**: 所有配置相关代码放在`src/config/`目录
5. **文档化**: 为自定义配置添加注释

## 扩展配置

### 添加新的配置字段

1. 在相应的配置类中添加字段
2. 在`defaults.py`中添加默认值
3. 在`validate()`方法中添加验证逻辑
4. 更新文档

### 添加新的配置类型

1. 在`src/config/`目录创建新文件
2. 定义配置类（使用dataclass）
3. 在`__init__.py`中导出
4. 在`config_loader.py`中添加加载逻辑

