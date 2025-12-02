# Delta-ME13: MARL沙盘式算法演算对比平台

一个基于多智能体强化学习(MARL)的沙盘式算法演算对比平台，支持多种智能体算法的实时可视化对比。

## 项目简介

Delta-ME13是一个用于研究和对比不同多智能体强化学习算法的可视化平台。平台支持：

- **多种智能体算法**：IQL、QMIX、规则型智能体等
- **实时可视化**：动态展示智能体行为和学习过程
- **算法对比**：同时运行多种算法进行性能对比
- **灵活配置**：支持JSON/YAML配置文件

## 功能特性

- 🎯 **多算法支持**：IQL、QMIX、规则型智能体（保守型、探索型、自适应型）
- 📊 **实时可视化**：动态展示智能体位置、糖分布、学习曲线
- ⚙️ **灵活配置**：支持配置文件、命令行参数等多种配置方式
- 🔄 **算法对比**：支持同时运行多种算法进行性能对比
- 📈 **性能监控**：实时监控FPS、学习进度等性能指标

## 项目结构

```
δ-me13/
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包列表
├── config/                 # 配置文件目录
│   └── default.json        # 默认配置文件
└── src/                    # 源代码目录
    ├── config/             # 配置管理模块
    ├── core/               # 核心模块（智能体、环境、模拟）
    ├── marl/               # MARL算法模块
    └── utils/              # 工具模块
```

详细的项目结构说明请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- Pygame 2.1.0+

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/Tim-He9/Delta-ME13.git
cd Delta-ME13
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
# 使用默认配置运行
python main.py

# 使用配置文件运行
python main.py --config config/default.json

# 指定模拟类型和参数
python main.py --simulation-type comparative --grid-size 100 --agents 200
```

### 配置说明

详细配置说明请参考 [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)

## 支持的智能体类型

- **rule_based**: 基于规则的智能体
- **conservative**: 保守型智能体
- **exploratory**: 探索型智能体
- **adaptive**: 自适应智能体
- **iql**: 独立Q学习算法
- **qmix**: QMIX算法

## 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [配置指南](CONFIGURATION_GUIDE.md)
- [架构改进说明](ARCHITECTURE_IMPROVEMENTS.md)
- [优化总结](OPTIMIZATION_SUMMARY.md)
- [Context优化指南](CONTEXT_OPTIMIZATION_GUIDE.md)

## 开发

### 代码规范

- 遵循PEP 8代码风格
- 使用类型提示
- 完整的文档字符串

### 贡献

欢迎提交Issue和Pull Request！

## 许可证

[待添加许可证信息]

## 作者

Tim-He9

