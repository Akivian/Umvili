# 项目结构说明

## 目录结构

```
δ-me13/
├── main.py                          # 主程序入口
├── requirements.txt                 # 依赖包列表
├── README.md                        # 项目说明文档
├── OPTIMIZATION_SUMMARY.md          # 优化总结文档
├── PROJECT_STRUCTURE.md             # 本文件：项目结构说明
│
├── config/                          # 配置文件目录（新增）
│   └── default.json                 # 默认配置文件
│
├── src/                             # 源代码目录
│   ├── __init__.py
│   │
│   ├── config/                      # 配置管理模块（新增，重构）
│   │   ├── __init__.py              # 统一导出所有配置类
│   │   ├── defaults.py              # 默认配置值
│   │   ├── app_config.py            # 应用配置
│   │   ├── simulation_config.py    # 模拟配置
│   │   ├── agent_config.py          # 智能体配置
│   │   ├── ui_config.py             # UI/可视化配置
│   │   └── config_loader.py          # 配置加载器（JSON/YAML）
│   │
│   ├── core/                        # 核心模块
│   │   ├── __init__.py
│   │   ├── agent_base.py            # 智能体基类
│   │   ├── agents.py                # 规则型智能体实现
│   │   ├── agent_factory.py         # 智能体工厂
│   │   ├── environment.py           # 环境类
│   │   ├── simulation.py            # 模拟引擎
│   │   └── reward_calculator.py     # 奖励计算器（新增）
│   │
│   ├── marl/                        # MARL算法模块
│   │   ├── __init__.py
│   │   ├── iql_agent.py             # IQL算法实现
│   │   ├── qmix_agent.py            # QMIX算法实现
│   │   ├── qmix_trainer.py          # QMIX训练器
│   │   ├── networks.py              # 神经网络架构
│   │   └── replay_buffer.py        # 经验回放缓冲区
│   │
│   └── utils/                       # 工具模块
│       ├── __init__.py
│       ├── config.py                # 配置兼容层（向后兼容）
│       ├── logging_config.py        # 日志配置
│       └── visualization.py         # 可视化系统
│
└── marl_simulation.log              # 日志文件（运行时生成）
```

## 模块职责说明

### 1. config/ - 配置管理模块（新增）

**职责**: 统一管理所有配置，支持从文件加载和验证

- **app_config.py**: 应用级配置（模拟类型、日志、UI显示选项）
- **simulation_config.py**: 模拟环境配置（网格大小、糖参数等）
- **agent_config.py**: 智能体配置（所有智能体类型的配置类）
- **ui_config.py**: UI配置（窗口、字体、颜色方案）
- **config_loader.py**: 配置加载器（支持JSON/YAML，配置合并）
- **defaults.py**: 默认配置值集中定义

### 2. core/ - 核心模块

**职责**: 提供核心功能和基础架构

- **agent_base.py**: 智能体基类和接口定义
- **agents.py**: 规则型智能体实现（RuleBased, Conservative, Exploratory, Adaptive）
- **agent_factory.py**: 智能体创建和管理工厂
- **environment.py**: 糖环境实现
- **simulation.py**: 模拟引擎核心逻辑
- **reward_calculator.py**: 统一奖励计算系统

### 3. marl/ - MARL算法模块

**职责**: 实现多智能体强化学习算法

- **iql_agent.py**: 独立Q学习算法
- **qmix_agent.py**: QMIX算法
- **qmix_trainer.py**: QMIX集中式训练器
- **networks.py**: 神经网络架构（DQN, Dueling, Noisy, QMIX等）
- **replay_buffer.py**: 经验回放系统（优先经验回放）

### 4. utils/ - 工具模块

**职责**: 提供通用工具和辅助功能

- **config.py**: 配置兼容层（保持向后兼容）
- **logging_config.py**: 日志系统配置
- **visualization.py**: 可视化系统（UI渲染、图表等）

## 配置管理架构

### 配置层次结构

```
完整配置
├── app (ApplicationConfig)
│   ├── simulation_type
│   ├── log_level
│   ├── show_fps
│   └── ...
├── simulation (SimulationConfig)
│   ├── grid_size
│   ├── cell_size
│   ├── initial_agents
│   └── ...
├── ui (UIConfig)
│   ├── window (WindowConfig)
│   │   ├── width
│   │   ├── height
│   │   ├── fps
│   │   └── ...
│   ├── font (FontConfig)
│   └── color_scheme (ColorScheme)
└── agents (Dict[str, Dict])
    ├── rule_based
    ├── iql
    ├── qmix
    └── ...
```

### 配置加载优先级

1. **默认配置** (`defaults.py`)
2. **文件配置** (`config/*.json` 或 `config/*.yaml`)
3. **用户配置** (代码中传入的字典)
4. **命令行参数** (main.py中的argparse)

### 配置文件格式

支持JSON和YAML两种格式：

```json
{
  "app": {
    "simulation_type": "comparative",
    "log_level": "INFO"
  },
  "simulation": {
    "grid_size": 80,
    "initial_agents": 50
  },
  "ui": {
    "window": {
      "width": 1400,
      "height": 900,
      "fps": 60
    }
  }
}
```

## 设计原则

### 1. 单一职责原则
- 每个模块和类都有明确的单一职责
- 配置管理独立于业务逻辑

### 2. 开闭原则
- 易于扩展新的配置类型
- 易于添加新的智能体算法

### 3. 依赖倒置原则
- 高层模块依赖配置抽象，不依赖具体实现

### 4. 配置集中管理
- 所有配置集中在`src/config/`目录
- 支持从文件加载和保存
- 支持配置验证和合并

## 使用示例

### 从文件加载配置

```python
from src.config import ConfigLoader

# 加载配置文件
config = ConfigLoader.load_full_config('config/custom.json')
configs = ConfigLoader.create_config_objects(config)

app_config = configs['app']
sim_config = configs['simulation']
ui_config = configs['ui']
```

### 使用默认配置

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

### 命令行使用

```bash
# 使用默认配置
python main.py

# 使用配置文件
python main.py --config config/custom.json

# 使用命令行参数
python main.py --simulation-type training --grid-size 100 --agents 200
```

## 向后兼容性

- `src/utils/config.py` 保留作为兼容层
- 所有旧的导入路径仍然有效
- 旧的配置格式会被自动转换

## 扩展指南

### 添加新的配置类型

1. 在`src/config/`目录创建新的配置类
2. 在`defaults.py`中添加默认值
3. 在`config_loader.py`中添加加载逻辑
4. 在`__init__.py`中导出

### 添加新的智能体配置

1. 在`agent_config.py`中添加配置类
2. 在`defaults.py`的`DEFAULT_AGENT_CONFIGS`中添加默认值
3. 在`agent_factory.py`中注册

## 文件组织优势

1. **清晰性**: 每个模块职责明确，易于理解
2. **可维护性**: 配置集中管理，易于修改和扩展
3. **可测试性**: 模块化设计便于单元测试
4. **可扩展性**: 易于添加新功能和算法
5. **类型安全**: 使用dataclass提供类型提示和验证

