# Umvili 多智能体强化学习平台 - 项目进度与规划

## 📊 当前项目状态

### ✅ 已完成功能（Phase 1-6, Phase 7部分）

#### 1. 核心可视化系统
- ✅ **MultiLineChart 组件**：支持多线实时绘制、动态曲线管理、图例显示
- ✅ **视图系统**：Overview / Training / Behavior / Debug / Network 五个视图，Tab 切换
- ✅ **训练指标可视化**：
  - Training Loss（按算法类型：IQL / QMIX）
  - Q-Value Trend
  - TD Error
  - Exploration Rate
- ✅ **行为策略可视化**：
  - Action Distribution（动作分布，带语义说明和 Top-3 摘要）
  - Reward Trend（奖励趋势曲线）
  - Policy Entropy（策略熵曲线）
- ✅ **Agent 类型分布面板**：数量、占比、平均糖量、RL 标记

#### 2. 数据收集基础设施
- ✅ `TrainingMetrics` 数据类（损失、Q值、TD误差、探索率、奖励）
- ✅ `_collect_training_metrics()` 方法（从 IQL/QMIX 智能体和训练器收集）
- ✅ 训练数据序列化与暴露（`get_simulation_data()`）

#### 3. 布局与性能优化
- ✅ 图表布局抽象（`_build_chart_grid`）
- ✅ 垂直分区系统（避免组件重叠）
- ✅ 更新频率控制（训练图表每5步，探索率每10步）
- ✅ 滑动窗口限制（最近200个数据点）

#### 4. 环境复杂度升级（Phase 6）✅
- ✅ **多资源类型系统**：
  - **Sugar（糖）**：基础资源，全地图分布，两个主要高地，持续再生
  - **Spice（香料）**：高价值稀缺资源，在特定小区域集中生成，消耗殆尽后随机重新生成
  - **Hazard（危险区）**：深血红色危险区域，从随机核心点开始，无规则向外扩散，形成连片危险区
- ✅ **动态资源系统**：
  - Spice 动态生成：在1-2个随机中心点周围极小范围内生成，消耗殆尽后延迟80步重新生成
  - Hazard 动态扩散：从核心点开始，每步从边缘随机向外扩散，老区域缓慢衰减解除
  - 资源竞争机制：Spice 奖励倍率5.0，鼓励智能体竞争抢夺
- ✅ **Hazard 危险机制**：
  - 危险区内资源完全清空且不再生成
  - 误入危险区的智能体每步损失大量 sugar（hazard_damage_per_step = 4.0）
  - 危险区稳定覆盖约9%地图，比 spice 区更大，形成明显的危险地带
  - 深血红色可视化，营造危险氛围
- ✅ **多资源可视化**：
  - Sugar：绿色渐变背景
  - Spice：金黄色半透明叠加（70%不透明度）
  - Hazard：深血红色半透明叠加（78%不透明度）
  - 清晰的绘制顺序和层次，避免视觉混乱

#### 5. 深度学习状态监测（Phase 7部分）✅
- ✅ **Q值热图可视化**：
  - `QValueHeatmapPanel` 组件，在环境地图上叠加显示 Q 值
  - 性能优化：增量更新、智能采样、缓存机制
  - 可切换开关，支持在 Behavior 视图中开启/关闭
- ✅ **网络内部状态可视化**：
  - `NetworkStatePanel` 组件，显示策略分布、策略熵、Q值统计
  - 网络参数统计（参数数量、梯度范数）
  - 集成到 Network 视图

---

## 🎯 下一阶段核心目标

### Phase 7: 深度学习状态监测（剩余部分，优先级：高）

#### 7.1 网络内部状态可视化（部分完成）
- ⚠️ **注意力热图**：需要网络架构支持注意力机制（当前未实现）
- ✅ **Q值热图**：已完成，在环境地图上叠加显示 Q 值
- ✅ **策略网络可视化**：已完成，显示策略分布、策略熵、Q值统计

#### 7.2 学习过程深度分析
- **梯度流分析**：显示训练过程中梯度的流动和变化
- **权重分布可视化**：网络权重的分布直方图、权重变化趋势
- **经验回放分析**：
  - Replay Buffer 中样本的分布（状态空间覆盖度）
  - 优先级回放的权重分布
  - 样本年龄分布

#### 7.3 多智能体协作分析
- **协作指标**：
  - 智能体之间的空间相关性（是否聚集）
  - 动作协调性（是否同时选择互补动作）
  - 资源分配公平性（Gini 系数）
- **通信可视化**：如果实现通信机制，可视化消息传递网络

**剩余实现路径**：
1. 如果未来引入注意力机制，添加 `get_attention_map()` 方法
2. 增强梯度流分析：显示训练过程中梯度的流动和变化
3. 权重分布可视化：网络权重的分布直方图、权重变化趋势
4. 经验回放分析：Replay Buffer 中样本的分布、优先级回放的权重分布

---

### Phase 8: 交互式实验配置系统（优先级：高）

**核心目标**：实现无需修改源代码即可在可视化窗口内灵活配置环境、智能体、算法和模型参数，显著提升项目可玩性和研究迭代效率。

---

#### 8.1 算法组合选择器（核心功能）

**位置**：Control Panel 顶部 / Experiment Tab 第一区域

**功能**：
- **下拉菜单/按钮组**：选择算法组合模式
  - `IQL Only`：仅独立Q学习智能体
  - `QMIX Only`：仅QMIX智能体
  - `IQL + QMIX`：两种算法同时运行对比
  - `Rule-based Baseline`：仅规则型智能体（保守型、探索型、自适应型）
  - `Mixed (IQL + QMIX + Baseline)`：混合模式，包含所有类型
  - `Custom`：自定义组合（手动选择每种类型的数量）

**实现要点**：
- 选择后立即更新智能体配置，调用 `simulation.reset(new_config)`
- 保持当前环境配置不变（除非用户明确修改）
- 显示当前选择的算法组合状态

---

#### 8.2 环境配置面板

**位置**：Experiment Tab / Control Panel 扩展区域

**功能模块**：

##### 8.2.1 基础环境参数
- **网格大小（Grid Size）**：
  - 滑块：10-200（步长：10）
  - 输入框：精确输入
  - 实时预览：显示当前网格大小
  
- **资源密度控制**：
  - **Sugar 生长速率**：滑块 0.0-1.0（步长：0.01）
  - **Sugar 最大值**：滑块 1.0-100.0（步长：1.0）
  - **Spice 生长速率**：滑块 0.0-0.1（步长：0.001）
  - **Spice 最大值**：滑块 1.0-20.0（步长：0.5）
  - **Hazard 衰减速率**：滑块 0.0-0.1（步长：0.001）
  - **Hazard 目标覆盖比例**：滑块 0.05-0.25（步长：0.01）

##### 8.2.2 资源类型开关
- **复选框组**：
  - ☑ Sugar（基础资源，默认开启）
  - ☑ Spice（高价值资源，默认开启）
  - ☑ Hazard（危险区域，默认开启）
- 关闭某个资源类型后，该资源在地图上不再生成，相关奖励机制也关闭

##### 8.2.3 智能体数量配置
- **总智能体数量**：滑块 10-500（步长：10）
- **按类型分配**（当选择 Custom 组合时）：
  - IQL 数量：滑块 0-200
  - QMIX 数量：滑块 0-200
  - Rule-based 数量：滑块 0-200
  - Conservative 数量：滑块 0-100
  - Exploratory 数量：滑块 0-100
  - Adaptive 数量：滑块 0-100
- **智能分布模式**：
  - 下拉菜单：Uniform（均匀分布）/ Clustered（聚集分布）/ Random（随机分布）

---

#### 8.3 算法超参数配置面板

**位置**：Experiment Tab 第二区域

**功能模块**：

##### 8.3.1 学习参数（Learning Parameters）
- **学习率（Learning Rate）**：
  - 档位选择：低（0.0001）/ 中（0.001）/ 高（0.01）/ 自定义
  - 自定义模式：输入框 0.0001-0.1（步长：0.0001）
  
- **折扣因子（Gamma）**：
  - 滑块：0.5-0.99（步长：0.01）
  - 显示当前值：`γ = 0.95`

##### 8.3.2 探索参数（Exploration Parameters）
- **Epsilon 起始值（Epsilon Start）**：
  - 滑块：0.0-1.0（步长：0.01）
  - 默认：1.0（完全探索）
  
- **Epsilon 终止值（Epsilon End）**：
  - 滑块：0.0-0.5（步长：0.01）
  - 默认：0.01（几乎完全利用）
  
- **Epsilon 衰减率（Epsilon Decay）**：
  - 滑块：0.9-0.9999（步长：0.0001）
  - 显示衰减曲线预览（可选）

##### 8.3.3 训练参数（Training Parameters）
- **Replay Buffer 大小**：
  - 滑块：1000-50000（步长：1000）
  - 显示内存占用估算
  
- **Batch Size**：
  - 下拉菜单：16 / 32 / 64 / 128 / 256
  - 或输入框：自定义 8-512
  
- **训练频率（Train Frequency）**：
  - 滑块：1-10（步长：1）
  - 说明：每 N 步进行一次训练更新
  
- **目标网络更新频率（Target Update Frequency）**：
  - 滑块：10-1000（步长：10）
  - 说明：每 N 步更新一次目标网络

##### 8.3.4 网络结构参数（Network Architecture）
- **隐藏层维度（Hidden Dimensions）**：
  - 预设选择：`[64]` / `[128]` / `[256]` / `[64, 64]` / `[128, 128]` / `[256, 256]`
  - 自定义模式：输入框（格式：`64,128,64`）
  
- **网络类型（Network Type）**：
  - 下拉菜单：DQN / Dueling DQN / Noisy DQN

##### 8.3.5 智能体基础参数（Agent Base Parameters）
- **视野范围（Vision Range）**：
  - 滑块：1-20（步长：1）
  - 说明：智能体观察范围
  
- **新陈代谢速率（Metabolism Rate）**：
  - 滑块：0.1-5.0（步长：0.1）
  - 说明：每步消耗的 sugar 量
  
- **初始 Sugar**：
  - 滑块：0-50（步长：1）
  - 说明：智能体初始生命值

---

#### 8.4 带参数的 Reset 功能

**核心机制**：
- **"Apply & Reset" 按钮**：
  - 收集所有 UI 配置参数
  - 构建配置字典
  - 调用 `simulation.reset(new_config)`
  - 整个系统按新配置重新初始化

**配置传递流程**：
```
UI 参数收集 → ConfigBuilder → SimulationConfig + AgentConfigs → simulation.reset(config)
```

**Reset 行为**：
1. 暂停当前模拟
2. 清空所有智能体和环境状态
3. 应用新配置重新初始化环境
4. 按新配置创建智能体
5. 重置所有可视化图表和数据
6. 自动恢复运行状态

**配置验证**：
- 在 Reset 前验证所有参数的有效性
- 显示错误提示（如：智能体数量过多、参数范围错误）
- 阻止无效配置的应用

---

#### 8.5 配置管理功能

##### 8.5.1 预设配置（Preset Configurations）
- **保存当前配置**：
  - 按钮："Save Current Config"
  - 弹出对话框：输入配置名称
  - 保存为 JSON 文件到 `config/presets/` 目录
  
- **加载预设配置**：
  - 下拉菜单：显示所有已保存的预设
  - 选择后自动填充所有 UI 控件
  - 点击 "Apply & Reset" 应用
  
- **预设配置类型**：
  - `Quick Start (IQL)`：快速开始 IQL 实验
  - `Quick Start (QMIX)`：快速开始 QMIX 实验
  - `Comparison Mode`：对比模式（IQL + QMIX）
  - `Baseline Only`：仅规则型智能体
  - `High Complexity`：高复杂度环境（多资源、大网格）
  - `Low Complexity`：低复杂度环境（单资源、小网格）

##### 8.5.2 配置导入/导出
- **导出配置**：
  - 按钮："Export Config"
  - 保存为 JSON 文件，用户选择保存位置
  
- **导入配置**：
  - 按钮："Import Config"
  - 文件选择对话框，加载 JSON 配置
  - 自动填充所有 UI 控件

##### 8.5.3 配置对比（多配置同时运行）
- **功能**：在同一个窗口中同时运行多个配置，对比结果
- **实现方式**：
  - 创建多个 `MARLSimulation` 实例
  - 每个实例使用不同的配置
  - 在可视化窗口中并排显示多个环境视图
  - 图表中同时显示多个配置的训练曲线（不同颜色）
- **限制**：最多同时运行 2-4 个配置（性能考虑）

---

#### 8.6 UI 组件设计

##### 8.6.1 Experiment Tab（新视图）
- **位置**：在现有视图（Overview / Training / Behavior / Debug / Network）基础上新增
- **布局**：
  - 顶部：算法组合选择器
  - 左侧：环境配置面板（可折叠）
  - 中间：算法超参数配置面板（可折叠）
  - 右侧：配置管理面板（保存/加载/导入/导出）
  - 底部：操作按钮区（Apply & Reset / Cancel）

##### 8.6.2 Control Panel 扩展
- **扩展方式**：
  - 在现有 Control Panel 下方添加"Quick Config"区域
  - 提供最常用的配置选项（算法组合、智能体数量、网格大小）
  - 详细配置在 Experiment Tab 中

##### 8.6.3 参数控件组件库
- **Slider（滑块）**：
  - 显示当前值
  - 显示最小值/最大值
  - 支持步长设置
  - 实时更新预览
  
- **InputBox（输入框）**：
  - 数值验证
  - 范围检查
  - 格式化显示
  
- **Dropdown（下拉菜单）**：
  - 预设选项
  - 自定义选项支持
  
- **Checkbox（复选框）**：
  - 资源类型开关
  - 功能开关
  
- **Button Group（按钮组）**：
  - 算法组合选择
  - 互斥选择

---

#### 8.7 技术实现路线

##### 阶段 1：核心配置系统（2-3天）
1. **创建配置构建器（ConfigBuilder）**：
   - `src/utils/config_builder.py`
   - 从 UI 控件收集参数
   - 构建 `SimulationConfig` 和 `AgentConfig` 对象
   - 验证配置有效性

2. **扩展 `simulation.reset()` 方法**：
   - 支持完整的配置字典
   - 处理环境参数更新
   - 处理智能体配置更新
   - 处理算法参数更新

3. **创建参数控件组件库**：
   - `src/utils/ui_components.py`
   - `Slider`、`InputBox`、`Dropdown`、`Checkbox`、`ButtonGroup` 类
   - 统一的样式和交互逻辑

##### 阶段 2：Experiment Tab 实现（3-4天）
1. **创建 Experiment Tab 视图**：
   - 在 `AcademicVisualizationSystem` 中添加 "experiment" 视图
   - 实现 Tab 切换逻辑

2. **实现算法组合选择器**：
   - 下拉菜单/按钮组组件
   - 选择后更新智能体配置

3. **实现环境配置面板**：
   - 网格大小、资源参数滑块
   - 资源类型开关复选框
   - 智能体数量配置

4. **实现算法超参数面板**：
   - 学习参数、探索参数、训练参数控件
   - 网络结构配置

##### 阶段 3：配置管理和 Reset 功能（2-3天）
1. **实现配置序列化/反序列化**：
   - JSON 格式保存/加载
   - 预设配置管理
   - 导入/导出功能

2. **实现带参数的 Reset**：
   - "Apply & Reset" 按钮
   - 配置收集和验证
   - 调用 `simulation.reset(new_config)`
   - 重置可视化状态

3. **实现配置对比功能**（可选）：
   - 多实例模拟支持
   - 并排可视化

##### 阶段 4：Control Panel 扩展和优化（1-2天）
1. **扩展 Control Panel**：
   - 添加 "Quick Config" 区域
   - 常用配置快速访问

2. **UI 优化**：
   - 响应式布局
   - 折叠面板支持
   - 参数验证提示
   - 配置预览

3. **错误处理和用户反馈**：
   - 参数验证错误提示
   - 配置应用成功/失败提示
   - 加载状态显示

---

#### 8.8 技术架构设计

##### 8.8.1 配置数据流
```
UI Controls → ConfigBuilder → Config Objects → simulation.reset(config) → Reinitialize
```

##### 8.8.2 核心类设计

**ConfigBuilder**：
```python
class ConfigBuilder:
    def collect_ui_params(self, ui_controls: Dict) -> Dict[str, Any]
    def build_simulation_config(self, params: Dict) -> SimulationConfig
    def build_agent_configs(self, params: Dict) -> List[AgentTypeConfig]
    def validate_config(self, config: Dict) -> Tuple[bool, Optional[str]]
```

**ExperimentConfigPanel**：
```python
class ExperimentConfigPanel:
    def __init__(self, rect: pygame.Rect, font_manager: FontManager)
    def draw(self, screen: pygame.Surface)
    def handle_event(self, event: pygame.event.Event) -> bool
    def get_config(self) -> Dict[str, Any]
    def load_config(self, config: Dict[str, Any]) -> None
```

**UIComponent 基类**：
```python
class UIComponent:
    def draw(self, screen: pygame.Surface)
    def handle_event(self, event: pygame.event.Event) -> bool
    def get_value(self) -> Any
    def set_value(self, value: Any) -> None
```

##### 8.8.3 配置存储格式
```json
{
  "simulation": {
    "grid_size": 60,
    "cell_size": 10,
    "initial_agents": 100
  },
  "environment": {
    "sugar_growth_rate": 0.1,
    "max_sugar": 10.0,
    "spice_growth_rate": 0.02,
    "max_spice": 6.0,
    "hazard_decay_rate": 0.01,
    "hazard_target_fraction": 0.09,
    "resource_enabled": {
      "sugar": true,
      "spice": true,
      "hazard": true
    }
  },
  "algorithm_combination": "iql_qmix",
  "agents": {
    "iql": {
      "count": 30,
      "learning_rate": 0.001,
      "gamma": 0.95,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01,
      "epsilon_decay": 0.995,
      "replay_buffer_size": 10000,
      "batch_size": 32,
      "hidden_dims": [64, 64]
    },
    "qmix": {
      "count": 30,
      "learning_rate": 0.001,
      "gamma": 0.95,
      "epsilon_start": 1.0,
      "epsilon_end": 0.01,
      "epsilon_decay": 0.995,
      "replay_buffer_size": 10000,
      "batch_size": 32
    }
  }
}
```

---

#### 8.9 验收标准

- [ ] **算法组合选择**：能够快速切换不同的算法组合，无需重启程序
- [ ] **环境参数调整**：网格大小、资源参数能够实时调整并生效
- [ ] **超参数配置**：所有学习参数、探索参数、训练参数都能通过 UI 配置
- [ ] **带参数 Reset**：修改配置后点击 "Apply & Reset" 能够立即应用新配置
- [ ] **配置保存/加载**：能够保存当前配置为预设，并能加载预设配置
- [ ] **配置导入/导出**：能够导出配置为 JSON 文件，并能从 JSON 文件导入
- [ ] **参数验证**：无效参数能够被检测并提示错误
- [ ] **UI 响应性**：所有控件响应流畅，无卡顿
- [ ] **向后兼容**：新功能不影响现有功能，现有配置仍能正常工作

---

#### 8.10 预期效果

**可玩性提升**：
- 用户可以在可视化窗口内快速尝试不同的算法组合
- 无需修改代码即可调整所有重要参数
- 快速对比不同配置的效果

**研究迭代效率提升**：
- 快速切换实验配置，无需重启程序
- 保存常用配置为预设，一键加载
- 配置对比功能支持同时运行多个实验

**用户体验提升**：
- 直观的 UI 控件，清晰的参数说明
- 实时预览和验证
- 错误提示和操作反馈

---

### Phase 9: 数据导出与分析工具（优先级：中）

#### 9.1 数据导出
- **CSV 导出**：训练指标、环境统计、智能体状态历史
- **图像导出**：当前可视化窗口截图、图表单独导出（PNG/SVG）
- **视频导出**：录制模拟过程为视频（MP4）
- **JSON 导出**：完整模拟状态快照（可用于回放）

#### 9.2 数据分析工具
- **统计摘要**：自动生成实验报告（平均性能、收敛速度、稳定性指标）
- **对比分析**：多组实验的对比图表（箱线图、误差带图）
- **趋势分析**：长期趋势拟合、周期性检测

#### 9.3 回放系统
- **状态回放**：从保存的快照恢复模拟状态
- **时间轴控制**：快进、倒退、跳转到指定步数
- **对比回放**：同时播放多个实验的对比视频

**实现路径**：
1. 实现 `DataExporter` 类（CSV、图像、视频导出）
2. 在控制面板添加“Export”按钮和菜单
3. 实现 `ReplaySystem` 用于状态回放
4. 创建 `AnalysisTool` 用于离线数据分析

---

### Phase 10: 高级可视化增强（优先级：中-低）

#### 10.1 交互式图表
- **鼠标悬停**：显示精确数值、时间戳
- **缩放和平移**：鼠标滚轮缩放、拖拽平移
- **曲线选择**：点击图例切换曲线可见性
- **数据点标记**：标记重要事件（训练开始、参数变更等）

#### 10.2 3D 可视化（可选）
- **3D 环境视图**：地形高度、资源分布的三维展示
- **3D 轨迹**：智能体在状态空间中的轨迹可视化
- **3D 价值函数**：Q 值函数的三维表面图

#### 10.3 智能体视角可视化
- **第一人称视角**：跟随某个智能体，显示其观察到的环境
- **决策过程可视化**：显示智能体选择动作的推理过程
- **策略对比**：并排显示不同算法在同一状态下的决策差异

**实现路径**：
1. 为 `MultiLineChart` 添加交互事件处理（鼠标事件）
2. 实现缩放/平移逻辑
3. 可选：集成 3D 渲染库（如 PyOpenGL）用于 3D 可视化

---

## 📅 实现时间线

| 阶段 | 状态 | 预计时间 | 累计时间 | 优先级 |
|------|------|---------|---------|--------|
| Phase 6: 增强环境模拟 | ✅ 已完成 | 8-12 小时 | 8-12 小时 | 高 |
| Phase 7: 深度学习状态监测 | 🔄 部分完成 | 10-15 小时 | 18-27 小时 | 高 |
| Phase 8: 交互式实验配置 | ⏳ 待开始 | 20-30 小时 | 38-57 小时 | 高 |
| Phase 9: 数据导出与分析 | ⏳ 待开始 | 4-6 小时 | 42-63 小时 | 中 |
| Phase 10: 高级可视化增强 | ⏳ 待开始 | 6-10 小时 | 48-73 小时 | 中-低 |

**已完成**：Phase 6 全部 + Phase 7 核心部分（Q值热图、网络状态可视化）  
**当前阶段**：Phase 8 - 交互式实验配置系统（详细规划已完成）  
**剩余工作量**：约 30-46 小时（约 1-1.5 周全职开发）

---

## 🛠 技术实现要点

### 架构设计原则
1. **模块化**：每个新功能作为独立模块，通过接口集成
2. **可扩展性**：新环境类型、新可视化组件易于添加
3. **性能优先**：交互式操作不阻塞主循环，使用异步/后台线程
4. **向后兼容**：新功能不影响现有功能

### 关键技术选型
- **环境扩展**：基于 `SugarEnvironment` 继承，使用组合模式添加新特性
- **可视化增强**：扩展 `AcademicVisualizationSystem`，新增子视图和交互组件
- **配置管理**：使用 `dataclass` + JSON/YAML 序列化
- **数据导出**：使用 `pandas` 处理 CSV，`PIL`/`matplotlib` 处理图像，`opencv-python` 处理视频

### 性能优化策略
- **懒加载**：复杂可视化组件按需创建
- **数据采样**：大数据集自动降采样显示
- **缓存机制**：计算结果缓存，避免重复计算
- **异步导出**：数据导出在后台线程进行，不阻塞 UI

---

## 🎯 短期行动计划（接下来 2-3 周）

### Week 1: Phase 8 核心配置系统
1. **Day 1-2**：创建配置构建器和参数控件组件库
   - 实现 `ConfigBuilder` 类
   - 创建 `UIComponent` 基类和子类（Slider、InputBox、Dropdown、Checkbox、ButtonGroup）
   - 扩展 `simulation.reset()` 方法支持完整配置

2. **Day 3-4**：实现 Experiment Tab 基础框架
   - 在可视化系统中添加 "experiment" 视图
   - 实现 Tab 切换和布局
   - 实现算法组合选择器

3. **Day 5**：实现环境配置面板
   - 网格大小、资源参数滑块
   - 资源类型开关
   - 智能体数量配置

### Week 2: Phase 8 超参数配置和 Reset 功能
1. **Day 1-2**：实现算法超参数配置面板
   - 学习参数、探索参数、训练参数控件
   - 网络结构配置
   - 智能体基础参数

2. **Day 3**：实现带参数的 Reset 功能
   - "Apply & Reset" 按钮
   - 配置收集和验证
   - 调用 `simulation.reset(new_config)`
   - 重置可视化状态

3. **Day 4-5**：实现配置管理功能
   - 配置序列化/反序列化（JSON）
   - 预设配置保存/加载
   - 配置导入/导出

### Week 3: Phase 8 优化和 Phase 7 剩余部分
1. **Day 1**：Control Panel 扩展和 UI 优化
   - 添加 "Quick Config" 区域
   - 响应式布局和折叠面板
   - 参数验证和错误提示

2. **Day 2-3**：完成 Phase 7 剩余部分（可选）
   - 梯度流分析
   - 权重分布可视化
   - 经验回放分析

3. **Day 4-5**：测试和优化
   - 功能测试
   - 性能优化
   - 用户体验优化

---

## 📝 验收标准

### Phase 6: 环境增强 ✅
- [x] 支持至少 3 种资源类型（Sugar / Spice / Hazard）
- [x] 资源分布可视化清晰（绿色糖、金黄色香料、深血红色危险区）
- [x] 智能体能够平衡多种资源（Spice 高奖励，Hazard 高惩罚）
- [x] 性能影响 < 10% FPS 下降（已验证）

### Phase 7: 深度监测（部分完成）
- [x] Q 值热图实时更新（已完成，带性能优化）
- [x] 网络状态可视化清晰易懂（策略分布、熵、Q值统计）
- [ ] 协作指标计算准确（待实现）
- [x] 不影响训练性能（已验证）

### Phase 8: 交互配置
- [ ] 参数调整界面友好
- [ ] 配置保存/加载功能正常
- [ ] 支持至少 2 个配置同时对比
- [ ] 参数修改有明确反馈

### Phase 9: 数据导出
- [ ] CSV 导出格式正确
- [ ] 图像导出质量良好
- [ ] 视频导出流畅（可选）
- [ ] 导出功能不阻塞 UI

### Phase 10: 高级可视化
- [ ] 图表交互流畅
- [ ] 缩放/平移功能正常
- [ ] 3D 可视化（如果实现）性能可接受

---

## 🔗 相关文档

- [架构设计](../ARCHITECTURE.md)
- [配置指南](../CONFIGURATION.md)
- [开发指南](../DEVELOPMENT.md)
- [MultiLineChart 使用指南](../MULTILINE_CHART_USAGE.md)

---

**最后更新**：2025年1月  
**文档版本**：3.0.0  
**项目状态**：Phase 1-6 已完成，Phase 7 部分完成（Q值热图、网络状态可视化），Phase 8-10 规划中

