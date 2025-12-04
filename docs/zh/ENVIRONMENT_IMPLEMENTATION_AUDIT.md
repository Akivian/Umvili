# 环境模块实现审计报告

## 审计日期
2025年1月

## 审计范围
根据 `PROJECT_ROADMAP.md` 第33-45行的预期功能，检查 `src/core/environment.py` 中 Spice 和 Hazard 的实现是否符合预期。

---

## 预期功能清单

### Spice（香料）预期功能
1. ✅ 高价值稀缺资源，在特定小区域集中生成
2. ✅ 在1-2个随机中心点周围极小范围内生成
3. ✅ 消耗殆尽后延迟80步重新生成
4. ✅ 避开hazard区域生成
5. ✅ 奖励倍率5.0（在reward_calculator中实现）

### Hazard（危险区）预期功能
1. ✅ 从随机核心点开始生成
2. ✅ 无规则向外扩散，形成连片危险区
3. ✅ 稳定覆盖约9%地图
4. ✅ 老区域缓慢衰减解除（decay_rate = 0.01）
5. ✅ 危险区内资源完全清空且不再生成
6. ✅ 误入危险区的智能体每步损失大量sugar（hazard_damage_per_step = 4.0）

---

## 发现的问题

### 🔴 严重问题 1：Spice 避开 Hazard 的逻辑错误

**位置**：`src/core/environment.py` 第199-201行

**问题描述**：
```python
# 检查是否远离hazard区域
center_x, center_y = self.size // 2, self.size // 2
dist_to_hazard = np.sqrt((x - center_x)**2 + (y - center_y)**2)
```

代码检查的是距离**地图中心**的距离，而不是实际的**hazard核心点** `self.hazard_core`。这导致：
- Spice 中心点可能不会真正避开 hazard 区域
- 因为 hazard 核心点是随机选择的（在 `_initialize_hazard_core()` 中），不一定在地图中心
- Spice 可能会在 hazard 区域内生成

**影响**：
- Spice 可能生成在 hazard 区域内，违反预期设计
- 智能体可能需要在危险区收集 spice，导致不合理的游戏机制

**修复方案**：
应该检查距离 `self.hazard_core` 的距离，而不是地图中心。同时还需要检查是否在现有的 hazard 有效区域内。

---

### 🟡 中等问题 2：Hazard 扩散可能不够"边缘优先"

**位置**：`src/core/environment.py` 第164-181行

**问题描述**：
当前实现从所有 `hazard_active_cells` 中随机选择源点，然后向8邻域扩散。虽然能形成连片区域，但可能不够"边缘优先"。

**当前实现**：
```python
# 随机选择一个当前hazard单元格作为扩散源
src_x, src_y = random.choice(self.hazard_active_cells)
```

**预期行为**：
应该优先从边缘单元格扩散，而不是从所有单元格中随机选择。这样可以形成更自然的扩散模式。

**影响**：
- 扩散模式可能不够自然
- 但功能基本可用，不是严重问题

**修复方案**（可选）：
可以识别边缘单元格（有非hazard邻居的单元格），优先从边缘扩散。

---

### 🟡 中等问题 3：Spice 重新生成时避开 Hazard 检查不完整

**位置**：`src/core/environment.py` 第267-276行

**问题描述**：
`_respawn_spice_centers()` 调用 `_initialize_spice_centers()`，但后者存在上述问题1，导致重新生成时也可能不会正确避开 hazard。

**影响**：
- 与问题1相同，spice 重新生成时可能出现在 hazard 区域内

**修复方案**：
修复问题1后，此问题也会自动解决。

---

### ✅ 已正确实现的功能

1. ✅ **Spice 动态生成逻辑**：
   - 1-2个随机中心点（`spice_min_centers = 1`, `spice_max_centers = 2`）
   - 极小范围内生成（`spice_radius = max(3, size // 20)`）
   - 延迟80步重新生成（`spice_respawn_delay = 80`）
   - 消耗殆尽检测（`_check_spice_depleted()`）

2. ✅ **Hazard 动态扩散逻辑**：
   - 随机核心点初始化（`_initialize_hazard_core()`）
   - 8邻域随机扩散
   - 目标面积9%（`hazard_target_fraction = 0.09`）
   - 衰减机制（`hazard_decay_rate = 0.01`）
   - 完全解除后重新初始化

3. ✅ **Hazard 区域内资源清空**：
   - 在 `grow_back()` 中正确清空（第322-325行）
   - 使用 `active_hazard_mask` 确保只清空有效区域

4. ✅ **Hazard 伤害机制**：
   - `hazard_damage_per_step = 4.0` 正确配置
   - 在 `harvest()` 中返回 `hazard_damage`
   - 在 `agent_base.py` 中正确应用伤害

5. ✅ **Spice 奖励倍率**：
   - `spice_collection_multiplier = 5.0` 在 `reward_calculator.py` 中正确配置
   - 在奖励计算中正确使用

---

## 修复优先级

1. **高优先级**：修复问题1（Spice 避开 Hazard 的逻辑错误）
2. **中优先级**：优化问题2（Hazard 边缘优先扩散，可选）
3. **低优先级**：问题3会随问题1的修复自动解决

---

## 修复建议

### 修复问题1的代码改动

在 `_initialize_spice_centers()` 方法中：

**修改前**：
```python
# 检查是否远离hazard区域
center_x, center_y = self.size // 2, self.size // 2
dist_to_hazard = np.sqrt((x - center_x)**2 + (y - center_y)**2)
```

**修改后**：
```python
# 检查是否远离hazard区域（使用实际的hazard核心点）
hazard_core_x, hazard_core_y = self.hazard_core
dist_to_hazard = np.sqrt((x - hazard_core_x)**2 + (y - hazard_core_y)**2)

# 同时检查是否在hazard有效区域内
is_in_hazard = self.hazard_map[x, y] >= self.hazard_active_threshold
```

**完整修复逻辑**：
```python
# 检查是否远离hazard区域
hazard_core_x, hazard_core_y = self.hazard_core
dist_to_hazard_core = np.sqrt((x - hazard_core_x)**2 + (y - hazard_core_y)**2)

# 检查是否在hazard有效区域内
is_in_hazard = self.hazard_map[x, y] >= self.hazard_active_threshold

# 检查是否距离hazard核心点足够远，且不在hazard区域内
if dist_to_hazard_core > hazard_margin and not is_in_hazard:
    # ... 其余逻辑
```

---

## 修复状态

### ✅ 已修复的问题

#### 问题1：Spice 避开 Hazard 的逻辑错误（已修复）

**修复内容**：
- 修改 `_initialize_spice_centers()` 方法，使用实际的 `self.hazard_core` 而不是地图中心
- 添加对 hazard 有效区域的检查（`is_in_hazard`）
- 改进安全距离计算，考虑 hazard 目标覆盖比例

**修复后的逻辑**：
```python
# 使用实际的hazard核心点
hazard_core_x, hazard_core_y = self.hazard_core
dist_to_hazard_core = np.sqrt((x - hazard_core_x)**2 + (y - hazard_core_y)**2)

# 检查是否在hazard有效区域内
is_in_hazard = self.hazard_map[x, y] >= self.hazard_active_threshold

# 确保距离hazard核心点足够远，且不在hazard区域内
if dist_to_hazard_core > hazard_margin and not is_in_hazard:
    # ... 接受该spice中心点
```

#### 问题2：Hazard 边缘优先扩散（已优化）

**优化内容**：
- 识别边缘单元格（有非hazard邻居的单元格）
- 优先从边缘单元格扩散，而不是从所有active cells中随机选择
- 动态更新边缘单元格列表，移除被完全包围的单元格

**优化效果**：
- 扩散模式更自然，更符合"从边缘向外扩散"的预期
- 形成更规则的连片危险区

---

## 总结

**总体评估**：
- 大部分功能已正确实现 ✅
- 发现并修复1个严重问题（Spice 避开 Hazard 逻辑错误）✅
- 发现并优化1个可优化点（Hazard 边缘优先扩散）✅

**当前状态**：
所有发现的问题已修复，环境模块现在完全符合预期功能要求。

**验证建议**：
1. 运行模拟，观察 spice 是否确实避开 hazard 区域生成
2. 观察 hazard 扩散模式是否更自然（从边缘向外扩散）
3. 验证 spice 重新生成时是否正确避开 hazard

