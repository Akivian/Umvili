"""
修复数组越界问题的环境类
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import pygame
import random
from src.config.ui_config import COLORS
# 向后兼容：使用默认值
CELL_SIZE = 10

class SugarEnvironment:
    """
    多资源糖环境（sugar / spice / hazard） - 修复边界检查并支持动态资源。
    
    设计目标：
    - 在保持原有 sugar-only 逻辑兼容的前提下，引入额外资源图：spice_map, hazard_map；
    - 提供基础的资源再生、耗尽与危险区动态衰减；
    - 为后续在观察空间和奖励函数中区分不同资源类型打下基础。
    """
    
    def __init__(
        self,
        size: int,
        growth_rate: float = 0.1,
        max_sugar: float = 10.0,
        spice_growth_rate: float = 0.02,  # 降低生成速度，让spice更稀缺
        max_spice: float = 6.0,  # 降低最大值，增强稀缺性
        hazard_decay_rate: float = 0.01,  # hazard缓慢衰减，保证区域有一定持续时间
        hazard_penalty_factor: float = 3.0,  # hazard对资源收益的惩罚系数
        hazard_damage_per_step: float = 4.0,  # agent每步在hazard中损失的sugar（按强度缩放）
    ):
        self.size = size
        # 资源图
        self.sugar_map = np.zeros((size, size), dtype=np.float32)
        self.spice_map = np.zeros((size, size), dtype=np.float32)
        self.hazard_map = np.zeros((size, size), dtype=np.float32)
        self.max_sugar = float(max_sugar)
        self.max_spice = float(max_spice)
        self.growth_rate = float(growth_rate)
        self.spice_growth_rate = float(spice_growth_rate)
        self.hazard_decay_rate = float(hazard_decay_rate)
        # hazard在奖励/资源上的惩罚强度（用于net_gain计算）
        self.hazard_penalty_factor = float(hazard_penalty_factor)
        # hazard对agent生命值（sugar）的直接伤害（用于饿死/快速死亡机制）
        self.hazard_damage_per_step = float(hazard_damage_per_step)
        
        # Spice动态生成相关参数
        self.spice_centers: List[Tuple[int, int]] = []  # 当前spice生成中心点列表（1-2个）
        self.spice_radius = max(3, self.size // 20)  # 每个中心点的生成半径（极小范围，3-5个格子）
        self.spice_depleted_step: int = -1  # 记录spice被消耗殆尽的时间步（-1表示未耗尽）
        self.spice_respawn_delay = 80  # 重新生成前的等待时间（步数）
        self.spice_min_centers = 1  # 最少中心点数
        self.spice_max_centers = 2  # 最多中心点数
        
        # Hazard 动态扩散相关参数
        self.hazard_active_threshold = 0.4  # 强度大于此值视为“有效hazard”
        self.hazard_target_fraction = 0.09  # hazard区域目标占据的地图比例（约9%，缩小到原来的3/5）
        self.hazard_spread_attempts = max(3, self.size // 6)  # 每步尝试扩散的单元数
        self.hazard_active_cells: List[Tuple[int, int]] = []  # 当前有效的hazard单元格
        self.hazard_core: Tuple[int, int] = (self.size // 2, self.size // 2)  # 初始核心点占位
        
        # 统计属性（按资源类型区分，便于后续可视化）
        self.total_sugar = 0.0
        self.avg_sugar = 0.0
        self.total_spice = 0.0
        self.avg_spice = 0.0
        self.total_hazard = 0.0
        
        # 时间步（用于观察空间）
        self.step = 0
        
        # 创建资源/危险分布（只初始化sugar与空的spice/hazard）
        self._create_sugar_hills()
        
        # 初始化hazard核心点及初始区域
        self._initialize_hazard_core()
        
        # 初始化spice中心点（随机选择1-2个，避开hazard区域）
        self._initialize_spice_centers()
        
        # 统计历史（保留原 total_sugar_history 接口）
        self.total_sugar_history = deque(maxlen=200)
        self.update_stats()
    
    def _create_sugar_hills(self):
        """
        创建糖/香料高地。
        
        - sugar: 两个主要高地，类似经典 Sugarscape；
        - spice: 初始为空，由spice中心点逻辑动态生成；
        - hazard: 仅初始化为0，由hazard动态扩散逻辑管理。
        """
        # Sugar: 两个高地
        center1 = (self.size // 3, self.size // 3)
        center2 = (2 * self.size // 3, 2 * self.size // 3)
        
        for x in range(self.size):
            for y in range(self.size):
                # Sugar hills（两个高地）
                dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
                dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)
                sugar1 = max(0.0, self.max_sugar - dist1 / 5.0)
                sugar2 = max(0.0, self.max_sugar - dist2 / 5.0)
                self.sugar_map[x, y] = min(self.max_sugar, sugar1 + sugar2)
                
                # Spice: 初始化为空（将在_spice逻辑中动态生成）
                self.spice_map[x, y] = 0.0
                
                # Hazard: 初始全部为0，由hazard动态逻辑控制
                self.hazard_map[x, y] = 0.0

    # ------------------------------------------------------------------
    # Hazard 初始化与动态扩散
    # ------------------------------------------------------------------
    def _initialize_hazard_core(self) -> None:
        """在模拟开始时随机选择一个核心点位生成初始hazard。"""
        # 在地图任意位置随机选一个核心点作为hazard起点
        core_x = random.randint(self.size // 4, 3 * self.size // 4)
        core_y = random.randint(self.size // 4, 3 * self.size // 4)
        self.hazard_core = (core_x, core_y)
        
        # 清空hazard_map并设置核心点为最大强度
        self.hazard_map.fill(0.0)
        self.hazard_map[core_x, core_y] = 1.0
        self.hazard_active_cells = [(core_x, core_y)]
    
    def _update_hazard(self) -> None:
        """
        更新hazard区域：
        - 老的hazard区域缓慢衰减并逐步解除；
        - 从当前有效hazard区域边缘随机向外扩散，形成不规则连片区域；
        - hazard区域内的资源被清空且不会再生成，直到区域危险解除。
        """
        # 1) 衰减已有hazard（老区域慢慢解除）
        if self.hazard_decay_rate > 0:
            self.hazard_map *= (1.0 - self.hazard_decay_rate)
            self.hazard_map = np.clip(self.hazard_map, 0.0, 1.0)
        
        # 2) 重新计算“有效hazard单元格”
        active_mask = self.hazard_map >= self.hazard_active_threshold
        active_indices = np.argwhere(active_mask)
        self.hazard_active_cells = [(int(x), int(y)) for x, y in active_indices]
        
        # 如果hazard区域完全解除，则重新随机选择核心点
        if not self.hazard_active_cells:
            self._initialize_hazard_core()
            active_mask = self.hazard_map >= self.hazard_active_threshold
            active_indices = np.argwhere(active_mask)
            self.hazard_active_cells = [(int(x), int(y)) for x, y in active_indices]
        
        # 3) 从当前hazard区域边缘向外扩散（形成不规则连片区域）
        # 目标总面积（单元格数）
        hazard_target_cells = int(self.hazard_target_fraction * self.size * self.size)
        current_active_count = len(self.hazard_active_cells)
        
        if current_active_count < hazard_target_cells and self.hazard_active_cells:
            attempts = 0
            max_attempts = self.hazard_spread_attempts * 3  # 允许一些失败尝试
            while attempts < max_attempts and current_active_count < hazard_target_cells:
                attempts += 1
                # 随机选择一个当前hazard单元格作为扩散源
                src_x, src_y = random.choice(self.hazard_active_cells)
                # 在8邻域中随机选择一个方向扩散
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = src_x + dx, src_y + dy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                
                # 如果邻居尚未是有效hazard，则以较高强度激活它
                if self.hazard_map[nx, ny] < self.hazard_active_threshold:
                    self.hazard_map[nx, ny] = 1.0
                    self.hazard_active_cells.append((nx, ny))
                    current_active_count += 1
                    if current_active_count >= hazard_target_cells:
                        break

    
    def _initialize_spice_centers(self) -> None:
        """初始化spice生成中心点（随机选择1-2个，避开hazard区域）"""
        num_centers = random.randint(self.spice_min_centers, self.spice_max_centers)
        self.spice_centers = []
        
        # 避开hazard区域（中心区域）
        hazard_region_size = self.size // 4
        hazard_margin = hazard_region_size // 2 + self.spice_radius + 5
        
        for _ in range(num_centers):
            attempts = 0
            while attempts < 50:  # 最多尝试50次
                x = random.randint(self.spice_radius, self.size - self.spice_radius - 1)
                y = random.randint(self.spice_radius, self.size - self.spice_radius - 1)
                
                # 检查是否远离hazard区域
                center_x, center_y = self.size // 2, self.size // 2
                dist_to_hazard = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if dist_to_hazard > hazard_margin:
                    # 检查是否与已有中心点距离足够远
                    too_close = False
                    for existing_center in self.spice_centers:
                        dist = np.sqrt((x - existing_center[0])**2 + (y - existing_center[1])**2)
                        if dist < self.spice_radius * 3:  # 至少相距3倍半径
                            too_close = True
                            break
                    
                    if not too_close:
                        self.spice_centers.append((x, y))
                        break
                
                attempts += 1
        
        # 在中心点周围生成spice
        self._generate_spice_at_centers()
    
    def _generate_spice_at_centers(self) -> None:
        """在当前的spice中心点周围生成spice（极小范围内）"""
        for center_x, center_y in self.spice_centers:
            # 只在中心点周围的小范围内生成
            for dx in range(-self.spice_radius, self.spice_radius + 1):
                for dy in range(-self.spice_radius, self.spice_radius + 1):
                    x = center_x + dx
                    y = center_y + dy
                    
                    # 边界检查
                    if not (0 <= x < self.size and 0 <= y < self.size):
                        continue
                    
                    # 计算距离
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= self.spice_radius:
                        # 使用陡峭的衰减曲线，让spice更集中
                        normalized_dist = dist / self.spice_radius
                        spice_val = self.max_spice * (1.0 - normalized_dist ** 2)
                        # 只在当前值为0或很小的地方生成（避免重复叠加）
                        if self.spice_map[x, y] < 0.1:
                            self.spice_map[x, y] = max(0.0, spice_val)
    
    def _check_spice_depleted(self) -> bool:
        """检查spice是否被消耗殆尽（所有中心点周围的spice都接近0）"""
        if not self.spice_centers:
            return True
        
        for center_x, center_y in self.spice_centers:
            # 检查中心点周围小范围内的spice总量
            local_spice = 0.0
            for dx in range(-self.spice_radius, self.spice_radius + 1):
                for dy in range(-self.spice_radius, self.spice_radius + 1):
                    x = center_x + dx
                    y = center_y + dy
                    if 0 <= x < self.size and 0 <= y < self.size:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= self.spice_radius:
                            local_spice += self.spice_map[x, y]
            
            # 如果某个中心点周围还有spice，则未耗尽
            if local_spice > 0.5:  # 阈值，允许少量残留
                return False
        
        return True
    
    def _respawn_spice_centers(self) -> None:
        """重新随机选择spice中心点并生成spice"""
        # 清空现有spice
        self.spice_map.fill(0.0)
        
        # 重新初始化中心点
        self._initialize_spice_centers()
        
        # 重置耗尽标记
        self.spice_depleted_step = -1
    
    def grow_back(self):
        """资源再生与危险区衰减"""
        # Sugar按速率再生（全地图）
        self.sugar_map += self.growth_rate
        self.sugar_map = np.clip(self.sugar_map, 0.0, self.max_sugar)
        
        # Spice动态生成逻辑
        if self.spice_centers:
            # 检查spice是否耗尽
            if self._check_spice_depleted():
                if self.spice_depleted_step < 0:
                    # 刚耗尽，记录时间步
                    self.spice_depleted_step = self.step
                else:
                    # 已耗尽，检查是否到了重新生成的时间
                    if self.step - self.spice_depleted_step >= self.spice_respawn_delay:
                        # 重新随机选择中心点并生成spice
                        self._respawn_spice_centers()
            else:
                # Spice未耗尽，只在中心点周围的小范围内再生
                for center_x, center_y in self.spice_centers:
                    for dx in range(-self.spice_radius, self.spice_radius + 1):
                        for dy in range(-self.spice_radius, self.spice_radius + 1):
                            x = center_x + dx
                            y = center_y + dy
                            
                            if not (0 <= x < self.size and 0 <= y < self.size):
                                continue
                            
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= self.spice_radius:
                                # 只在中心点周围再生
                                self.spice_map[x, y] += self.spice_growth_rate
                                self.spice_map[x, y] = np.clip(
                                    self.spice_map[x, y], 0.0, self.max_spice
                                )
        else:
            # 如果没有中心点，初始化
            self._initialize_spice_centers()
        
        # Hazard 动态更新（衰减 + 扩散）
        self._update_hazard()
        
        # 在hazard区域内清空所有资源（糖和spice不会在hazard内存在或再生）
        active_hazard_mask = self.hazard_map >= self.hazard_active_threshold
        if np.any(active_hazard_mask):
            self.sugar_map[active_hazard_mask] = 0.0
            self.spice_map[active_hazard_mask] = 0.0
        
        self.step += 1  # 更新时间步
        self.update_stats()
        
    def harvest(self, x: int, y: int) -> Dict[str, float]:
        """
        收获指定位置的资源，返回详细的资源信息。
        
        Args:
            x, y: 位置坐标
            
        Returns:
            包含sugar, spice, hazard, net_gain的字典
        """
        # 确保坐标在有效范围内
        if not (0 <= x < self.size and 0 <= y < self.size):
            return {'sugar': 0.0, 'spice': 0.0, 'hazard': 0.0, 'net_gain': 0.0}
            
        sugar = float(self.sugar_map[x, y])
        spice = float(self.spice_map[x, y])
        hazard = float(self.hazard_map[x, y])
        
        # 收获后清空该格资源（危险值保留，由衰减过程处理）
        self.sugar_map[x, y] = 0.0
        self.spice_map[x, y] = 0.0
        
        # 危险惩罚（与本地 hazard 程度成正比）
        penalty = hazard * self.hazard_penalty_factor
        net_gain = max(0.0, sugar + spice - penalty)
        
        # Hazard 对 agent 的直接生命损耗（由agent在_update中应用）
        hazard_damage = hazard * self.hazard_damage_per_step
        
        self.update_stats()
        return {
            'sugar': sugar,
            'spice': spice,
            'hazard': hazard,
            'net_gain': net_gain,       # 保持向后兼容
            'hazard_damage': hazard_damage,
        }
    
    def update_stats(self):
        """更新环境统计"""
        self.total_sugar = float(np.sum(self.sugar_map))
        self.avg_sugar = float(np.mean(self.sugar_map))
        self.total_spice = float(np.sum(self.spice_map))
        self.avg_spice = float(np.mean(self.spice_map))
        self.total_hazard = float(np.sum(self.hazard_map))
        self.total_sugar_history.append(self.total_sugar)
    
    def get_serializable_data(self) -> Dict[str, Any]:
        """获取可序列化的环境数据"""
        return {
            'sugar_map': self.sugar_map,
            'spice_map': self.spice_map,
            'hazard_map': self.hazard_map,
            'total_sugar': self.total_sugar,
            'avg_sugar': self.avg_sugar,
            'total_spice': self.total_spice,
            'avg_spice': self.avg_spice,
            'total_hazard': self.total_hazard,
            'size': self.size,
            'max_sugar': self.max_sugar,
            'growth_rate': self.growth_rate,
            'max_spice': self.max_spice,
            'spice_growth_rate': self.spice_growth_rate
        }
    
    def draw(self, screen: pygame.Surface) -> None:
        """绘制糖分布"""
        # 绘制糖单元格
        for x in range(self.size):
            for y in range(self.size):
                sugar = self.sugar_map[x, y]
                if sugar > 0:
                    # 根据糖量选择颜色
                    if sugar < 3:
                        color = COLORS['SUGAR_LOW']
                    elif sugar < 7:
                        color = COLORS['SUGAR_MEDIUM']
                    else:
                        color = COLORS['SUGAR_HIGH']
                    
                    pygame.draw.rect(screen, color, 
                                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # 绘制细网格线
        for x in range(0, self.size * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(screen, COLORS['CHART_GRID'], (x, 0), (x, self.size * CELL_SIZE), 1)
        for y in range(0, self.size * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(screen, COLORS['CHART_GRID'], (0, y), (self.size * CELL_SIZE, y), 1)