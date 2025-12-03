# Umvili - 中文文档

[English](../README.md) | 中文

## 概述

Umvili 是一个基于多智能体强化学习(MARL)的沙盘式算法演算对比平台，支持多种智能体算法的实时可视化对比。

## 主要特性

- 🎯 **多算法支持**：IQL、QMIX、规则型智能体（保守型、探索型、自适应型）
- 📊 **实时可视化**：动态展示智能体位置、糖分布、学习曲线
- ⚙️ **灵活配置**：支持配置文件、命令行参数等多种配置方式
- 🔄 **算法对比**：支持同时运行多种算法进行性能对比
- 📈 **性能监控**：实时监控FPS、学习进度等性能指标

## 快速开始

### 安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/Akivian/Umvili.git
   cd Umvili
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行程序**
   ```bash
   python main.py
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

## 支持的智能体类型

- `rule_based`: 基于规则的智能体
- `conservative`: 保守型智能体
- `exploratory`: 探索型智能体
- `adaptive`: 自适应智能体
- `iql`: 独立Q学习算法
- `qmix`: QMIX算法

## 项目结构

```
Umvili/
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

## 文档

- [架构文档](../ARCHITECTURE.md) - 项目结构和设计
- [配置指南](../CONFIGURATION.md) - 配置管理说明
- [开发指南](../DEVELOPMENT.md) - 开发设置和规范

## 配置说明

项目支持灵活的配置方式：

- **配置文件**：JSON/YAML格式（见 `config/default.json`）
- **命令行参数**：覆盖特定设置
- **代码配置**：程序化配置

详细配置说明请参考 [配置指南](../CONFIGURATION.md)。

## 开发

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型提示
- 完整的文档字符串

### 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件。

## 作者

**Akivian**

- GitHub: [@Akivian](https://github.com/Akivian)

---

**注意**：本项目正在积极开发中，功能和API可能会发生变化。

