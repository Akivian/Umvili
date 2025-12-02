"""
修复数组越界问题的环境类
"""

import numpy as np
from typing import Dict, Any
from collections import deque
import pygame
from src.config.ui_config import COLORS
# 向后兼容：使用默认值
CELL_SIZE = 10

class SugarEnvironment:
    """糖环境 - 修复边界检查"""
    
    def __init__(self, size: int, growth_rate: float = 0.1, max_sugar: float = 10.0):
        self.size = size
        self.sugar_map = np.zeros((size, size))
        self.max_sugar = max_sugar
        self.growth_rate = growth_rate
        
        # 统计属性
        self.total_sugar = 0.0
        self.avg_sugar = 0.0
        
        # 时间步（用于观察空间）
        self.step = 0
        
        # 创建糖分布
        self._create_sugar_hills()
        
        # 统计历史
        self.total_sugar_history = deque(maxlen=200)
        self.update_stats()
    
    def _create_sugar_hills(self):
        """创建两个糖高地"""
        center1 = (self.size // 3, self.size // 3)
        center2 = (2 * self.size // 3, 2 * self.size // 3)
        
        for x in range(self.size):
            for y in range(self.size):
                dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
                dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2)
                sugar1 = max(0, self.max_sugar - dist1 / 5)
                sugar2 = max(0, self.max_sugar - dist2 / 5)
                self.sugar_map[x, y] = min(self.max_sugar, sugar1 + sugar2)
    
    def grow_back(self):
        """糖再生"""
        self.sugar_map += self.growth_rate
        self.sugar_map = np.clip(self.sugar_map, 0, self.max_sugar)
        self.step += 1  # 更新时间步
        self.update_stats()
        
    def harvest(self, x: int, y: int) -> float:
        """
        收获指定位置的糖 - 修复边界检查
        
        Args:
            x, y: 位置坐标
            
        Returns:
            收获的糖量
        """
        # 确保坐标在有效范围内
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0.0
            
        sugar = self.sugar_map[x, y]
        self.sugar_map[x, y] = 0
        self.update_stats()
        return sugar
    
    def update_stats(self):
        """更新环境统计"""
        self.total_sugar = np.sum(self.sugar_map)
        self.avg_sugar = np.mean(self.sugar_map)
        self.total_sugar_history.append(self.total_sugar)
    
    def get_serializable_data(self) -> Dict[str, Any]:
        """获取可序列化的环境数据"""
        return {
            'sugar_map': self.sugar_map,
            'total_sugar': self.total_sugar,
            'avg_sugar': self.avg_sugar,
            'size': self.size,
            'max_sugar': self.max_sugar,
            'growth_rate': self.growth_rate
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