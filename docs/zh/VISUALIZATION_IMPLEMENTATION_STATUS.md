# 智能体训练可视化实现状态

## 📋 已完成工作（当前状态）

### ✅ 阶段1：数据收集基础设施（已完成）

#### 1.1 数据结构扩展
- ✅ 在 `src/core/simulation.py` 中添加了 `TrainingMetrics` 数据类
  - 包含：平均损失、平均Q值、TD误差、探索率、训练步数等
  - 支持按智能体类型聚合数据

#### 1.2 训练数据收集
- ✅ 实现了 `_collect_training_metrics()` 方法
  - 从IQL智能体收集训练信息（通过 `get_training_info()`）
  - 从QMIX训练器收集训练统计（通过 `get_training_stats()`）
  - 从QMIX智能体收集训练信息
  - 按智能体类型自动聚合数据

#### 1.3 数据暴露
- ✅ 在 `get_simulation_data()` 中添加了 `training_metrics` 字段
- ✅ 实现了 `_serialize_training_metrics()` 方法，将训练指标序列化为字典格式

#### 验收结果
- ✅ 代码通过语法检查
- ✅ 数据结构设计合理
- ✅ 支持IQL和QMIX两种智能体类型
- ✅ 错误处理完善（使用try-except保护）

---

### ✅ 阶段2：基础图表实现 & 集成（已完成）

#### 2.1 多线图表组件
**文件**: `src/utils/visualization.py`

- ✅ 创建 `MultiLineChart` 类，支持多条曲线显示
- ✅ 每条曲线可以有不同的颜色和标签（IQL / QMIX / Rule-Based 等）
- ✅ 支持动态添加/移除曲线、控制可见性
- ✅ 支持滑动窗口、更新频率控制、数据统计（均值、方差等）

#### 2.2 训练指标图表
**文件**: `src/utils/visualization.py`

- ✅ 在 `AcademicVisualizationSystem._initialize_training_charts()` 中添加训练图表：
  - `Training Loss`
  - `Q-Value Trend`
  - `TD Error`
  - `Exploration Rate`
- ✅ 在 `_update_training_charts()` 中更新训练数据，按智能体类型（IQL、QMIX）分别绘制多条曲线
- ✅ 使用统一颜色方案区分算法类型：
  - IQL：橙色 (`AGENT_IQL`)
  - QMIX：绿色 (`AGENT_QMIX`)
  - Rule-Based / Conservative / Exploratory 等：各自固定颜色

#### 2.3 Agent 类型分布可视化增强
**文件**: `src/utils/visualization.py` (`AgentDistributionPanel`)

- ✅ 显示各类型智能体数量 + 占比（%）
- ✅ 对学习型算法类型（IQL / QMIX）增加 `[RL]` 标记
- ✅ 集成平均糖量 `avg_sugar_by_type`，在图例中展示每种类型的平均资源状态
- ✅ 图表按数量动态排序，实时反映类型结构变化

---

## 📊 数据流示意图（当前实现）

```
智能体/训练器
    ↓
get_training_info() / get_training_stats()
    ↓
_collect_training_metrics()
    ↓
TrainingMetrics (按类型聚合)
    ↓
_serialize_training_metrics()
    ↓
get_simulation_data()['training_metrics']
    ↓
`AcademicVisualizationSystem._update_training_charts()`
    ↓
`MultiLineChart.add_data_point_conditional()`
    ↓
右侧训练图表实时显示（Loss / Q / TD / Exploration）
```

---

## 🔍 当前数据结构

### TrainingMetrics
```python
@dataclass
class TrainingMetrics:
    agent_type: str                    # 智能体类型 (如 'iql', 'qmix')
    avg_loss: float = 0.0              # 平均损失
    avg_q_value: float = 0.0           # 平均Q值
    avg_td_error: float = 0.0          # 平均TD误差
    exploration_rate: float = 0.0      # 探索率 (ε)
    training_steps: int = 0             # 训练步数
    sample_count: int = 0              # 样本数量
    recent_loss: float = 0.0           # 最近损失值
    recent_q_value: float = 0.0        # 最近Q值
```

### 数据格式（序列化后）
```python
{
    'iql': {
        'avg_loss': 0.123,
        'avg_q_value': 5.67,
        'avg_td_error': 0.045,
        'exploration_rate': 0.15,
        'training_steps': 1000,
        'sample_count': 30,
        'recent_loss': 0.120,
        'recent_q_value': 5.70
    },
    'qmix': {
        'avg_loss': 0.234,
        'avg_q_value': 8.90,
        ...
    }
}
```

---

## 🎯 使用示例

### 在可视化系统中获取并更新训练数据（当前实现概念）

```python
def _update_training_charts(self, simulation_data: Dict[str, Any]) -> None:
    training_metrics = simulation_data.get('training_metrics', {})
    step = simulation_data.get('step_count', 0)

    # 颜色与标签映射：确保 IQL / QMIX 等算法类型使用固定颜色和图例
    label_map = {'iql': 'IQL', 'independent_q_learning': 'IQL', 'qmix': 'QMIX', 'rule_based': 'Rule-Based'}

    # 训练损失
    for agent_type, metrics in training_metrics.items():
        agent_label = label_map.get(agent_type, agent_type.upper())
        if 'loss' in self.training_charts and 'recent_loss' in metrics:
            self.training_charts['loss'].add_data_point_conditional(
                agent_label, metrics['recent_loss'], step
            )

    # Q值趋势 / TD 误差 / 探索率 同理更新...
```

---

## 📝 注意事项

1. **数据更新频率**: 训练图表目前默认每5/10步更新一次，可通过 `update_frequency` 调整
2. **数据验证**: 已添加异常处理，但建议在可视化层也添加数据验证
3. **性能考虑**: 大量智能体时，数据收集可能影响性能，需要监控
4. **向后兼容**: 新功能不影响现有功能，`training_metrics` 字段可选

---

## 🎯 下一阶段规划概览

结合当前实现和窗口空间限制，下一阶段的重点目标为：

1. **右侧 Panel 视图化重构**
   - 引入「Overview / Training / Behavior / Debug」四个视图，使用简单 Tab 切换。
   - Overview：只展示 `Simulation Statistics` + `Agent Types & Distribution`。
   - Training：只展示四张训练曲线（Loss / Q / TD / Exploration）。
   - Behavior（预留）：后续挂载动作分布、奖励分布、策略熵等图表。

2. **图表布局抽象**
   - 用统一的「图表槽位布局器」管理每个视图中的图表 `rect`，避免硬编码坐标。
   - Training 视图中采用 2×2 网格布局：上层 Loss + Q，下层 TD + Exploration，保证高度与可读性。

3. **行为与策略级可视化（下一阶段核心功能）**
   - 动作分布面板：按算法类型展示最近 N 步动作选择频率。
   - 奖励趋势图：展示 IQL / QMIX 的平均奖励随时间变化。
   - 策略熵曲线：衡量策略确定性，配合探索率一起判断收敛情况。

---

## 🛠 详细实现路线（开发视角）

### 路线 A：视图与布局系统（推荐优先）

1. **引入视图状态**
   - 在 `AcademicVisualizationSystem` 中增加成员：
     - `self.active_view: str = "training"`（默认 training）
     - `self.views = ["overview", "training", "behavior", "debug"]`
   - 在控制面板顶部或右上角绘制 Tab 文本按钮，并在 `handle_event` 中根据点击切换 `active_view`。

2. **抽象图表槽位布局**
   - 在 `visualization.py` 中新增简单布局函数：
     ```python
     def build_panel_grid(x, y, width, row_count, col_count, row_spacing, col_spacing, height=None) -> List[pygame.Rect]:
         ...
     ```
   - Training 视图调用该函数生成 2×2 槽位，将现有四张训练图表挂载到对应 `rect`。
   - 保持 `MultiLineChart.rect` 与槽位同步，减少手写坐标。

3. **按视图绘制内容**
   - 在 `AcademicVisualizationSystem.draw()` 中：
     - 根据 `active_view` 决定是否绘制训练图表 / overview 图表。
     - 当前实现可以先支持 `overview` 与 `training` 两个视图，其它预留。

### 路线 B：行为与策略可视化

1. **数据收集扩展**
   - 在 IQL / QMIX 智能体中，记录最近 N 步动作索引与奖励：
     - 可以在 `LearningAgent` 基类增加一个轻量的 `action_history` / `reward_history` 缓冲。
   - 在 `MARLSimulation._collect_training_metrics()` 中增加：
     - 每类算法的动作频次统计（最近窗口）。
     - 每类算法的平均/最近奖励。
     - 可选：策略熵（基于动作频次归一化分布计算）。

2. **Behavior 视图图表实现**
   - `ActionDistributionPanel`：
     - 接收 `{agent_type: {action_idx: count}}` 结构，绘制条形图或分组柱状图。
   - `RewardTrendChart`：
     - 使用 `MultiLineChart` 显示 `avg_reward` / `recent_reward`。
   - `PolicyEntropyChart`：
     - 使用 `MultiLineChart` 显示 IQL / QMIX 的策略熵随时间变化。

3. **性能与更新策略**
   - 行为统计不需要每步更新，可在 `step % 10 == 0` 时刷新一次，降低开销。
   - 训练与行为图表共享滑动窗口（例如最近 200 个时间点）。

---

## 🔗 相关文档

- [完整实现方案](../VISUALIZATION_ENHANCEMENT_PLAN.md) - 最新规划与阶段划分
- [集成完成报告](./VISUALIZATION_INTEGRATION_COMPLETE.md) - 当前图表结构与特性

---

**最后更新**: 2024年（当前日期）  
**整体状态**: 阶段1–3已完成，阶段4（布局/性能优化）进行中，阶段5（行为策略可视化）为下一阶段核心目标。

