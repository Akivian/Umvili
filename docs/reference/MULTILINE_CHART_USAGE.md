# MultiLineChart 使用指南

## 概述

`MultiLineChart` 是一个功能强大的多线实时图表组件，专为MARL平台的可视化需求设计。它支持多条曲线同时显示，每条曲线有独立的颜色、标签和可见性控制。

## 核心特性

### ✅ 已实现功能

1. **多曲线支持**
   - 动态添加/移除曲线
   - 每条曲线独立的颜色和标签
   - 曲线可见性控制

2. **性能优化**
   - 滑动窗口限制数据点数量（默认200点）
   - 可配置的更新频率（减少不必要的更新）
   - 数据采样优化绘制性能

3. **可视化功能**
   - 自动Y轴缩放
   - 美观的网格线
   - 图例显示（带最新值）
   - 可选的数据点标记
   - 可选的平滑曲线

4. **错误处理**
   - 自动过滤无效数据（NaN、Inf）
   - 完善的异常处理
   - 空数据状态处理

5. **统计功能**
   - 获取单条曲线统计信息
   - 获取所有曲线统计信息

## 基本使用

### 1. 创建图表

```python
from src.utils.visualization import MultiLineChart, AcademicFontManager

font_manager = AcademicFontManager()

# 创建多线图表
chart = MultiLineChart(
    x=100,                    # X坐标
    y=100,                    # Y坐标
    width=600,                # 宽度
    height=300,               # 高度
    title="训练损失",          # 标题
    y_label="Loss",          # Y轴标签
    font_manager=font_manager, # 字体管理器
    max_points=200,           # 最大数据点数
    update_frequency=1,       # 更新频率（每N步更新）
    show_legend=True,        # 显示图例
    show_points=False,        # 显示数据点
    line_width=2,            # 线条宽度
    smooth_lines=False       # 平滑曲线
)
```

### 2. 添加曲线

```python
# 添加IQL损失曲线
chart.add_line("IQL", color=COLORS['AGENT_IQL'])

# 添加QMIX损失曲线（自动分配颜色）
chart.add_line("QMIX")

# 添加自定义颜色曲线
chart.add_line("Custom", color=(255, 0, 0))
```

### 3. 添加数据点

```python
# 方式1：直接添加（每次调用都添加）
chart.add_data_point("IQL", loss_value, step_count)

# 方式2：条件添加（根据更新频率）
chart.add_data_point_conditional("IQL", loss_value, step_count)
```

### 4. 绘制图表

```python
# 在主循环中绘制
chart.draw(screen)
```

## 高级用法

### 控制曲线可见性

```python
# 隐藏曲线
chart.set_line_visible("IQL", False)

# 显示曲线
chart.set_line_visible("IQL", True)
```

### 清空数据

```python
# 清空单条曲线
chart.clear_line("IQL")

# 清空所有曲线
chart.clear_all()
```

### 移除曲线

```python
# 移除曲线
chart.remove_line("IQL")
```

### 获取统计信息

```python
# 获取单条曲线统计
stats = chart.get_statistics("IQL")
# 返回: {'label': 'IQL', 'count': 100, 'min': 0.1, 'max': 2.5, 'mean': 0.8, 'std': 0.3, 'latest': 0.75}

# 获取所有曲线统计
all_stats = chart.get_statistics()
```

## 在可视化系统中集成

### 示例：在AcademicVisualizationSystem中使用

```python
class AcademicVisualizationSystem:
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # 初始化训练指标图表
        self.training_charts = self._initialize_training_charts(panel_x, panel_width)
    
    def _initialize_training_charts(self, panel_x: int, panel_width: int):
        """初始化训练指标图表"""
        chart_width = panel_width - 30
        chart_height = 120
        chart_spacing = 10
        
        charts = {}
        
        # 损失函数图表
        base_y = self.control_panel.rect.bottom + 20
        charts['loss'] = MultiLineChart(
            panel_x + 15,
            base_y,
            chart_width,
            chart_height,
            "训练损失",
            "Loss",
            self.font_manager,
            max_points=200,
            update_frequency=5,  # 每5步更新一次
            show_legend=True,
            show_points=False
        )
        
        # 添加IQL和QMIX曲线
        charts['loss'].add_line("IQL", color=COLORS['AGENT_IQL'])
        charts['loss'].add_line("QMIX", color=COLORS['AGENT_QMIX'])
        
        # Q值趋势图表
        charts['q_value'] = MultiLineChart(
            panel_x + 15,
            base_y + chart_height + chart_spacing,
            chart_width,
            chart_height,
            "Q值趋势",
            "Q Value",
            self.font_manager,
            max_points=200,
            update_frequency=5,
            show_legend=True
        )
        charts['q_value'].add_line("IQL", color=COLORS['AGENT_IQL'])
        charts['q_value'].add_line("QMIX", color=COLORS['AGENT_QMIX'])
        
        return charts
    
    def _update_training_charts(self, simulation_data: Dict[str, Any]):
        """更新训练图表数据"""
        training_metrics = simulation_data.get('training_metrics', {})
        
        step = simulation_data.get('step_count', 0)
        
        # 更新损失图表
        if 'loss' in self.training_charts:
            for agent_type, metrics in training_metrics.items():
                if 'recent_loss' in metrics:
                    self.training_charts['loss'].add_data_point_conditional(
                        agent_type.upper(),
                        metrics['recent_loss'],
                        step
                    )
        
        # 更新Q值图表
        if 'q_value' in self.training_charts:
            for agent_type, metrics in training_metrics.items():
                if 'recent_q_value' in metrics:
                    self.training_charts['q_value'].add_data_point_conditional(
                        agent_type.upper(),
                        metrics['recent_q_value'],
                        step
                    )
    
    def draw(self, screen: pygame.Surface, simulation_data: Dict[str, Any]) -> None:
        """绘制整个可视化系统"""
        # ... 现有绘制代码 ...
        
        # 更新训练图表数据
        self._update_training_charts(simulation_data)
        
        # 绘制训练图表
        for chart in self.training_charts.values():
            chart.draw(screen)
```

## 配置选项

### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `x` | int | - | 图表X坐标 |
| `y` | int | - | 图表Y坐标 |
| `width` | int | - | 图表宽度 |
| `height` | int | - | 图表高度 |
| `title` | str | - | 图表标题 |
| `y_label` | str | - | Y轴标签 |
| `font_manager` | AcademicFontManager | - | 字体管理器 |
| `max_points` | int | 200 | 每条曲线最大数据点数 |
| `update_frequency` | int | 1 | 更新频率（每N步更新） |
| `show_legend` | bool | True | 是否显示图例 |
| `show_points` | bool | False | 是否显示数据点 |
| `line_width` | int | 2 | 线条宽度（像素） |
| `smooth_lines` | bool | False | 是否使用平滑曲线 |

### 性能调优建议

1. **更新频率**
   - 对于快速变化的指标，使用 `update_frequency=5` 或更高
   - 对于慢速变化的指标，可以使用 `update_frequency=1`

2. **数据点数量**
   - 默认200点适合大多数场景
   - 如果需要更长的历史，可以增加到500
   - 注意：更多数据点会影响绘制性能

3. **平滑曲线**
   - `smooth_lines=True` 会略微影响性能
   - 建议只在需要时启用

4. **数据点标记**
   - `show_points=True` 会增加绘制负担
   - 建议只在调试时启用

## 颜色方案

组件使用预定义的颜色方案（色盲友好）：

- `CHART_LINE_1`: 蓝色 (31, 119, 180)
- `CHART_LINE_2`: 橙色 (255, 127, 14)
- `CHART_LINE_3`: 绿色 (44, 160, 44)
- `CHART_LINE_4`: 红色 (214, 39, 40)
- `CHART_LINE_5`: 紫色 (148, 103, 189)
- `AGENT_IQL`: 橙色 (255, 127, 14)
- `AGENT_QMIX`: 绿色 (44, 160, 44)

## 错误处理

组件内置完善的错误处理：

- 自动过滤 NaN 和 Inf 值
- 处理空数据状态
- 异常时显示占位符
- 无效标签时静默失败

## 最佳实践

1. **初始化时添加所有曲线**
   ```python
   # 在初始化时添加所有需要的曲线
   chart.add_line("IQL")
   chart.add_line("QMIX")
   ```

2. **使用条件更新**
   ```python
   # 使用 add_data_point_conditional 减少更新频率
   chart.add_data_point_conditional("IQL", value, step)
   ```

3. **合理设置数据点数量**
   ```python
   # 根据需求设置 max_points
   # 200点 ≈ 最近200步的数据
   chart = MultiLineChart(..., max_points=200)
   ```

4. **按需显示图例**
   ```python
   # 如果空间有限，可以隐藏图例
   chart = MultiLineChart(..., show_legend=False)
   ```

## 示例：完整的训练指标可视化

```python
# 初始化
loss_chart = MultiLineChart(
    100, 100, 600, 300,
    "训练损失", "Loss", font_manager,
    max_points=200, update_frequency=5
)
loss_chart.add_line("IQL", COLORS['AGENT_IQL'])
loss_chart.add_line("QMIX", COLORS['AGENT_QMIX'])

# 在主循环中
def update_and_draw(simulation_data):
    step = simulation_data['step_count']
    training_metrics = simulation_data.get('training_metrics', {})
    
    # 更新数据
    for agent_type, metrics in training_metrics.items():
        if 'recent_loss' in metrics:
            loss_chart.add_data_point_conditional(
                agent_type.upper(),
                metrics['recent_loss'],
                step
            )
    
    # 绘制
    loss_chart.draw(screen)
```

## 未来扩展

可能的功能扩展：

1. **交互功能**
   - 鼠标悬停显示数值
   - 点击切换曲线可见性
   - 缩放和平移

2. **更多图表类型**
   - 面积图
   - 柱状图
   - 散点图

3. **数据导出**
   - 导出为CSV
   - 导出为图像

4. **高级统计**
   - 移动平均
   - 置信区间
   - 趋势线

---

**版本**: 1.0.0  
**最后更新**: 2024年

