"""
Academic Professional Visualization System for MARL Platform

A high-quality, modular visualization system designed for multi-agent reinforcement
learning research. Features academic-style color schemes, real-time data visualization,
and clear agent type/status representation.

Design Principles:
- Academic Professional: Color schemes inspired by scientific publications
- Modular Architecture: Clear separation of concerns
- Real-time Performance: Optimized for smooth rendering
- Accessibility: Colorblind-friendly palettes
- Extensibility: Easy to add new visualization components

Author: MARL Platform Team
Version: 3.0.0
"""

import pygame
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import math
from types import SimpleNamespace

from src.config.ui_config import COLORS, FONT_SIZES
from src.config.simulation_config import SimulationConfig
# 向后兼容：使用默认值
GRID_SIZE = 80
CELL_SIZE = 10


@dataclass
class UIMetrics:
    """UI Metrics Data Structure"""
    step: int = 0
    total_agents: int = 0
    alive_agents: int = 0
    avg_sugar: float = 0.0
    avg_age: float = 0.0
    total_sugar: float = 0.0
    diversity: float = 0.0
    fps: float = 0.0
    state: str = "unknown"
    performance: Dict[str, float] = field(default_factory=dict)
    agents_by_type: Dict[str, int] = field(default_factory=dict)
    avg_sugar_by_type: Dict[str, float] = field(default_factory=dict)


class AcademicFontManager:
    """Academic-style Font Manager with reliable font loading"""
    
    def __init__(self):
        pygame.font.init()
        self.fonts = {}
        self._initialize_fonts()
    
    def _initialize_fonts(self):
        """Initialize all fonts with fallback options"""
        font_names = [
            'Arial', 'Helvetica', 'Tahoma', 'Verdana', 
            'DejaVu Sans', 'Liberation Sans', 'Calibri'
        ]
        
        for size_name, size in FONT_SIZES.items():
            font_created = False
            for font_name in font_names:
                try:
                    self.fonts[size_name] = pygame.font.SysFont(font_name, size)
                    # Test rendering
                    test_surface = self.fonts[size_name].render("Test", True, COLORS['TEXT_PRIMARY'])
                    if test_surface.get_width() > 0:
                        font_created = True
                        break
                except:
                    continue
            
            # Fallback to default font
            if not font_created:
                self.fonts[size_name] = pygame.font.Font(None, size)
    
    def get_font(self, size_name: str) -> pygame.font.Font:
        """Get font by size name"""
        return self.fonts.get(size_name, self.fonts['BODY'])
    
    def render_text(self, text: str, size_name: str = 'BODY', 
                   color: Tuple = COLORS['TEXT_PRIMARY']) -> pygame.Surface:
        """Render text with error handling"""
        try:
            if text is None:
                text = ""
            text = str(text)
            font = self.get_font(size_name)
            return font.render(text, True, color)
        except Exception as e:
            print(f"Text rendering failed: {e}")
            return pygame.Surface((1, 1))


class AgentDistributionPanel:
    """
    Agent 类型分布面板（图例 + 条形图 + 比例信息）。

    设计目标：
    - 动态反映当前环境中各类型智能体的数量及占比
    - 支持区分 MARL 算法类（IQL/QMIX等）与规则/策略类智能体
    - 显示更加丰富的统计信息，而不仅仅是静态的几根柱子
    """
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 font_manager: AcademicFontManager):
        self.rect = pygame.Rect(x, y, width, height)
        self.font_manager = font_manager
        self.padding = {'left': 12, 'right': 12, 'top': 26, 'bottom': 20}
        
        self.agent_type_colors = {
            'rule_based': COLORS['AGENT_RULE_BASED'],
            'independent_q_learning': COLORS['AGENT_IQL'],
            'iql': COLORS['AGENT_IQL'],
            'qmix': COLORS['AGENT_QMIX'],
            'conservative': COLORS['AGENT_CONSERVATIVE'],
            'exploratory': COLORS['AGENT_EXPLORATORY'],
            'adaptive': COLORS['AGENT_ADAPTIVE'],
        }
        self.agent_type_labels = {
            'rule_based': 'Rule-Based',
            'independent_q_learning': 'IQL',
            'iql': 'IQL',
            'qmix': 'QMIX',
            'conservative': 'Conservative',
            'exploratory': 'Exploratory',
            'adaptive': 'Adaptive',
        }

        # 区分学习型算法与非学习型（用于图例标记）
        self.learning_types = {'iql', 'independent_q_learning', 'qmix'}
    
    def draw(self,
             screen: pygame.Surface,
             counts: Dict[str, int],
             avg_sugar_by_type: Optional[Dict[str, float]] = None) -> None:
        """
        绘制类型分布面板。

        Args:
            counts: 各类型智能体数量，例如 {'iql': 20, 'qmix': 20, 'rule_based': 20}
            avg_sugar_by_type: 可选，各类型平均糖量，用于在图例中展示额外状态信息
        """
        try:
            # Background
            pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
            
            total_agents = sum(counts.values()) if counts else 0

            # Title（包含智能体总数）
            title_text = "Agent Types & Distribution"
            if total_agents > 0:
                title_text += f"  (Total: {total_agents})"
            title_surface = self.font_manager.render_text(
                title_text, 'SMALL', COLORS['TEXT_PRIMARY']
            )
            screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
            
            if not counts or total_agents == 0:
                empty_text = self.font_manager.render_text("No data", 'TINY', COLORS['TEXT_MUTED'])
                text_rect = empty_text.get_rect(center=self.rect.center)
                screen.blit(empty_text, text_rect)
                return
            
            # 按数量从大到小排序，避免“看起来死”的静态顺序
            sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            
            # Legend on the left
            legend_x = self.rect.x + self.padding['left']
            legend_y = self.rect.y + self.padding['top']
            legend_line_h = 18
            for i, (agent_type, value) in enumerate(sorted_items):
                if value == 0:
                    continue
                color = self.agent_type_colors.get(agent_type, COLORS['GRAY'])
                label = self.agent_type_labels.get(agent_type, agent_type.title())
                # 占比（%）
                proportion = (value / total_agents) * 100 if total_agents > 0 else 0.0

                # 额外状态信息：平均糖量（如果提供）
                extra = ""
                if avg_sugar_by_type and agent_type in avg_sugar_by_type:
                    extra = f", AvgSugar: {avg_sugar_by_type[agent_type]:.1f}"

                # 标识是否为学习型算法
                is_learning = agent_type in self.learning_types
                algo_tag = " [RL]" if is_learning else ""

                text = f"{label}{algo_tag}: {value} ({proportion:.1f}%){extra}"
                
                indicator_rect = pygame.Rect(legend_x, legend_y + i * legend_line_h, 10, 10)
                pygame.draw.rect(screen, color, indicator_rect)
                pygame.draw.rect(screen, COLORS['LEGEND_BORDER'], indicator_rect, 1)
                
                text_surface = self.font_manager.render_text(text, 'TINY', COLORS['TEXT_PRIMARY'])
                screen.blit(text_surface, (legend_x + 16, legend_y - 2 + i * legend_line_h))
            
            # Bar chart on the right
            chart_left = self.rect.x + self.rect.width // 2 + 5
            chart_area = pygame.Rect(
                chart_left,
                self.rect.y + self.padding['top'] + 4,
                self.rect.width - (chart_left - self.rect.x) - self.padding['right'],
                self.rect.height - self.padding['top'] - self.padding['bottom'],
            )
            
            max_value = max(counts.values())
            num_bars = len([v for _, v in sorted_items if v > 0])
            bar_spacing = 6
            bar_width = max(10, (chart_area.width - (num_bars - 1) * bar_spacing) // max(num_bars, 1))
            
            x = chart_area.x
            for agent_type, value in sorted_items:
                if value == 0:
                    continue
                bar_height = int((value / max_value) * chart_area.height)
                bar_rect = pygame.Rect(
                    x,
                    chart_area.bottom - bar_height,
                    bar_width,
                    bar_height,
                )
                color = self.agent_type_colors.get(agent_type, COLORS['GRAY'])
                pygame.draw.rect(screen, color, bar_rect)
                pygame.draw.rect(screen, COLORS['PANEL_BORDER'], bar_rect, 1)
                
                # Value label above bar
                value_surface = self.font_manager.render_text(str(value), 'TINY', COLORS['TEXT_PRIMARY'])
                value_rect = value_surface.get_rect(center=(bar_rect.centerx, bar_rect.y - 8))
                screen.blit(value_surface, value_rect)
                
                x += bar_width + bar_spacing
        
        except Exception as e:
            print(f"Agent distribution panel drawing failed: {e}")


class MultiLineChart:
    """
    多线实时图表组件 - 支持多条曲线同时显示
    
    特性：
    - 支持多条曲线，每条曲线有独立的颜色和标签
    - 动态添加/移除曲线
    - 性能优化：数据采样、滑动窗口
    - 美观的图例显示
    - 自动Y轴缩放
    - 支持数据点标记
    - 完善的错误处理
    """
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 title: str, y_label: str, font_manager: AcademicFontManager,
                 max_points: int = 200, update_frequency: int = 1,
                 show_legend: bool = True, show_points: bool = False,
                 line_width: int = 2, smooth_lines: bool = False):
        """
        初始化多线图表
        
        Args:
            x, y: 图表位置
            width, height: 图表尺寸
            title: 图表标题
            y_label: Y轴标签
            font_manager: 字体管理器
            max_points: 每条曲线最大数据点数
            update_frequency: 更新频率（每N步更新一次）
            show_legend: 是否显示图例
            show_points: 是否显示数据点
            line_width: 线条宽度
            smooth_lines: 是否使用平滑曲线
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.y_label = y_label
        self.font_manager = font_manager
        self.max_points = max_points
        self.update_frequency = update_frequency
        self.show_legend = show_legend
        self.show_points = show_points
        self.line_width = line_width
        self.smooth_lines = smooth_lines
        
        # 曲线数据存储：{label: {'data': deque, 'timestamps': deque, 'color': tuple}}
        self.lines: Dict[str, Dict[str, Any]] = {}
        
        # 默认颜色列表（色盲友好）
        self.default_colors = [
            COLORS['CHART_LINE_1'],  # Blue
            COLORS['CHART_LINE_2'],  # Orange
            COLORS['CHART_LINE_3'],  # Green
            COLORS['CHART_LINE_4'],  # Red
            COLORS['CHART_LINE_5'],  # Purple
            COLORS['AGENT_IQL'],     # Orange (IQL)
            COLORS['AGENT_QMIX'],    # Green (QMIX)
        ]
        self.color_index = 0
        
        # 更新计数器
        self.update_counter = 0
        
        # 图表内边距
        self.padding = {
            'left': 50,
            'right': 10,
            'top': 28,
            'bottom': 30
        }
        
        # 图例区域（如果显示）
        if self.show_legend:
            self.padding['right'] = 120  # 为图例预留空间
    
    def add_line(self, label: str, color: Optional[Tuple[int, int, int]] = None) -> bool:
        """
        添加一条新曲线
        
        Args:
            label: 曲线标签
            color: 曲线颜色（如果为None则自动分配）
            
        Returns:
            是否成功添加
        """
        if label in self.lines:
            return False  # 标签已存在
        
        if color is None:
            color = self.default_colors[self.color_index % len(self.default_colors)]
            self.color_index += 1
        
        self.lines[label] = {
            'data': deque(maxlen=self.max_points),
            'timestamps': deque(maxlen=self.max_points),
            'color': color,
            'visible': True,
            'min_val': float('inf'),
            'max_val': float('-inf')
        }
        return True
    
    def remove_line(self, label: str) -> bool:
        """
        移除一条曲线
        
        Args:
            label: 曲线标签
            
        Returns:
            是否成功移除
        """
        if label in self.lines:
            del self.lines[label]
            return True
        return False
    
    def set_line_visible(self, label: str, visible: bool) -> bool:
        """
        设置曲线可见性
        
        Args:
            label: 曲线标签
            visible: 是否可见
            
        Returns:
            是否成功设置
        """
        if label in self.lines:
            self.lines[label]['visible'] = visible
            return True
        return False
    
    def add_data_point(self, label: str, value: float, timestamp: int) -> bool:
        """
        添加数据点
        
        Args:
            label: 曲线标签
            value: 数据值
            timestamp: 时间戳
            
        Returns:
            是否成功添加
        """
        if label not in self.lines:
            return False
        
        # 验证数据有效性
        try:
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return False
        except (ValueError, TypeError):
            return False
        
        line = self.lines[label]
        line['data'].append(value)
        line['timestamps'].append(int(timestamp))
        
        # 更新最小/最大值（用于快速范围计算）
        if value < line['min_val']:
            line['min_val'] = value
        if value > line['max_val']:
            line['max_val'] = value
        
        return True
    
    def add_data_point_conditional(self, label: str, value: float, timestamp: int) -> bool:
        """
        条件添加数据点（根据更新频率）
        
        Args:
            label: 曲线标签
            value: 数据值
            timestamp: 时间戳
            
        Returns:
            是否实际添加了数据点
        """
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:
            return self.add_data_point(label, value, timestamp)
        return False
    
    def clear_line(self, label: str) -> bool:
        """
        清空指定曲线的数据
        
        Args:
            label: 曲线标签
            
        Returns:
            是否成功清空
        """
        if label in self.lines:
            self.lines[label]['data'].clear()
            self.lines[label]['timestamps'].clear()
            self.lines[label]['min_val'] = float('inf')
            self.lines[label]['max_val'] = float('-inf')
            return True
        return False
    
    def clear_all(self) -> None:
        """清空所有曲线数据"""
        for label in self.lines:
            self.clear_line(label)
    
    def get_line_count(self) -> int:
        """获取曲线数量"""
        return len(self.lines)
    
    def get_data_range(self) -> Tuple[float, float]:
        """
        计算所有可见曲线的数据范围
        
        Returns:
            (min_value, max_value)
        """
        all_values = []
        for line in self.lines.values():
            if line['visible'] and len(line['data']) > 0:
                all_values.extend(line['data'])
        
        if not all_values:
            return (0.0, 1.0)
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        # 处理退化情况（所有值相同）
        if min_val == max_val:
            delta = 1.0 if min_val == 0 else abs(min_val) * 0.1
            return (min_val - delta, max_val + delta)
        
        # 添加10%边距
        margin = (max_val - min_val) * 0.1
        return (min_val - margin, max_val + margin)
    
    def draw(self, screen: pygame.Surface) -> None:
        """绘制图表"""
        try:
            # 检查是否有可见数据
            visible_lines = [line for line in self.lines.values() if line['visible'] and len(line['data']) > 0]
            
            if not visible_lines:
                self._draw_empty_chart(screen)
                return
            
            # 计算图表区域
            chart_area = pygame.Rect(
                self.rect.x + self.padding['left'],
                self.rect.y + self.padding['top'],
                self.rect.width - self.padding['left'] - self.padding['right'],
                self.rect.height - self.padding['top'] - self.padding['bottom']
            )
            
            # 绘制背景
            pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
            
            # 绘制标题
            title_surface = self.font_manager.render_text(self.title, 'SMALL', COLORS['TEXT_PRIMARY'])
            screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
            
            # 计算数据范围
            min_val, max_val = self.get_data_range()
            val_range = max_val - min_val or 1.0
            
            # 绘制网格
            self._draw_grid(screen, chart_area, min_val, max_val)
            
            # 绘制所有曲线
            for label, line in self.lines.items():
                if line['visible'] and len(line['data']) > 0:
                    self._draw_line(screen, chart_area, label, line, min_val, val_range)
            
            # 绘制坐标轴标签
            self._draw_axes_labels(screen, chart_area, min_val, max_val)
            
            # 绘制图例
            if self.show_legend:
                self._draw_legend(screen, chart_area)
                
        except Exception as e:
            print(f"MultiLineChart drawing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_empty_chart(self, screen: pygame.Surface) -> None:
        """绘制空图表占位符"""
        pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
        pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
        
        title_surface = self.font_manager.render_text(self.title, 'SMALL', COLORS['TEXT_PRIMARY'])
        screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
        
        empty_text = self.font_manager.render_text("等待数据...", 'TINY', COLORS['TEXT_MUTED'])
        text_rect = empty_text.get_rect(center=self.rect.center)
        screen.blit(empty_text, text_rect)
    
    def _draw_grid(self, screen: pygame.Surface, chart_area: pygame.Rect,
                   min_val: float, max_val: float) -> None:
        """绘制网格线"""
        num_lines = 5
        
        # 水平网格线
        for i in range(num_lines + 1):
            y = chart_area.y + (i * chart_area.height // num_lines)
            pygame.draw.line(screen, COLORS['CHART_GRID'],
                           (chart_area.x, y), (chart_area.right, y), 1)
        
        # 垂直网格线（可选，减少数量以提高性能）
        num_vertical = 4
        for i in range(num_vertical + 1):
            x = chart_area.x + (i * chart_area.width // num_vertical)
            pygame.draw.line(screen, COLORS['CHART_GRID'],
                           (x, chart_area.y), (x, chart_area.bottom), 1)
    
    def _draw_line(self, screen: pygame.Surface, chart_area: pygame.Rect,
                   label: str, line: Dict[str, Any], min_val: float, val_range: float) -> None:
        """绘制单条曲线"""
        data = line['data']
        color = line['color']
        
        if len(data) < 2:
            return
        
        # 计算点坐标
        points = []
        for i, value in enumerate(data):
            x = chart_area.x + (i / (len(data) - 1)) * chart_area.width
            y = chart_area.bottom - ((value - min_val) / val_range) * chart_area.height
            points.append((int(x), int(y)))
        
        if len(points) < 2:
            return
        
        # 绘制平滑曲线（如果启用）
        if self.smooth_lines and len(points) > 2:
            self._draw_smooth_line(screen, points, color)
        else:
            # 绘制折线
            pygame.draw.lines(screen, color, False, points, self.line_width)
        
        # 绘制数据点（如果启用）
        if self.show_points:
            point_sample = max(1, len(points) // 20)  # 采样以减少绘制负担
            for point in points[::point_sample]:
                pygame.draw.circle(screen, color, point, 3)
                pygame.draw.circle(screen, COLORS['WHITE'], point, 1)
    
    def _draw_smooth_line(self, screen: pygame.Surface, points: List[Tuple[int, int]],
                         color: Tuple[int, int, int]) -> None:
        """
        绘制平滑曲线（使用简单的线性插值）
        
        注意：真正的平滑曲线需要更复杂的算法（如样条插值），
        这里使用简化的方法以提高性能。
        """
        if len(points) < 2:
            return
        
        # 对于少量点，直接绘制折线
        if len(points) <= 10:
            pygame.draw.lines(screen, color, False, points, self.line_width)
            return
        
        # 使用更密集的点来模拟平滑曲线
        smoothed_points = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            # 添加起点
            smoothed_points.append(p1)
            
            # 在两点之间插入中间点（简化平滑）
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            smoothed_points.append((mid_x, mid_y))
        
        # 添加最后一个点
        smoothed_points.append(points[-1])
        
        # 绘制平滑后的曲线
        pygame.draw.lines(screen, color, False, smoothed_points, self.line_width)
    
    def _draw_axes_labels(self, screen: pygame.Surface, chart_area: pygame.Rect,
                         min_val: float, max_val: float) -> None:
        """绘制坐标轴标签"""
        num_labels = 5
        
        # Y轴标签
        for i in range(num_labels + 1):
            value = min_val + (max_val - min_val) * (num_labels - i) / num_labels
            y = chart_area.y + (i * chart_area.height // num_labels)
            
            # 格式化数值
            if abs(value) < 0.01:
                value_text = "0"
            elif abs(value) < 1:
                value_text = f"{value:.3f}"
            elif abs(value) < 100:
                value_text = f"{value:.2f}"
            else:
                value_text = f"{value:.1f}"
            
            label_surface = self.font_manager.render_text(value_text, 'TINY', COLORS['TEXT_SECONDARY'])
            screen.blit(label_surface, (self.rect.x + 5, y - 8))
        
        # Y轴标签（单位）
        y_label_surface = self.font_manager.render_text(self.y_label, 'TINY', COLORS['TEXT_SECONDARY'])
        # 简单放置，不旋转（旋转需要额外处理）
        screen.blit(y_label_surface, (self.rect.x + 5, self.rect.y + self.rect.height // 2))
    
    def _draw_legend(self, screen: pygame.Surface, chart_area: pygame.Rect) -> None:
        """绘制图例"""
        visible_lines = [(label, line) for label, line in self.lines.items() if line['visible']]
        
        if not visible_lines:
            return
        
        # 图例位置（图表右侧）
        legend_x = chart_area.right + 10
        legend_y = chart_area.y
        legend_item_height = 18
        legend_width = self.rect.width - chart_area.right - 15
        
        # 绘制图例背景（可选）
        legend_rect = pygame.Rect(
            legend_x - 5,
            legend_y - 5,
            legend_width,
            len(visible_lines) * legend_item_height + 10
        )
        pygame.draw.rect(screen, COLORS['LEGEND_BG'], legend_rect)
        pygame.draw.rect(screen, COLORS['LEGEND_BORDER'], legend_rect, 1)
        
        # 绘制图例项
        for i, (label, line) in enumerate(visible_lines):
            item_y = legend_y + i * legend_item_height
            
            # 颜色指示器
            indicator_rect = pygame.Rect(legend_x, item_y, 12, 12)
            pygame.draw.rect(screen, line['color'], indicator_rect)
            pygame.draw.rect(screen, COLORS['LEGEND_BORDER'], indicator_rect, 1)
            
            # 标签文本
            # 截断过长的标签
            display_label = label if len(label) <= 12 else label[:10] + ".."
            label_surface = self.font_manager.render_text(display_label, 'TINY', COLORS['TEXT_PRIMARY'])
            screen.blit(label_surface, (legend_x + 16, item_y - 2))
            
            # 显示最新值（可选）
            if len(line['data']) > 0:
                latest_value = line['data'][-1]
                value_text = f"{latest_value:.2f}"
                value_surface = self.font_manager.render_text(value_text, 'TINY', COLORS['TEXT_SECONDARY'])
                value_x = legend_x + legend_width - value_surface.get_width() - 5
                screen.blit(value_surface, (value_x, item_y - 2))
    
    def get_statistics(self, label: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            label: 曲线标签（如果为None则返回所有曲线的统计）
            
        Returns:
            统计信息字典
        """
        if label is not None:
            if label not in self.lines:
                return {}
            line = self.lines[label]
            data = list(line['data'])
            if not data:
                return {}
            
            return {
                'label': label,
                'count': len(data),
                'min': min(data),
                'max': max(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'latest': data[-1] if data else None
            }
        else:
            # 返回所有曲线的统计
            stats = {}
            for label in self.lines:
                stats[label] = self.get_statistics(label)
            return stats


class QValueHeatmapPanel:
    """
    Q值热图可视化面板（性能优化版）
    
    在环境地图上叠加显示每个位置的Q值，直观展示价值函数。
    采用增量更新、缓存机制和智能采样来优化性能。
    """
    
    def __init__(self, x: int, y: int, width: int, height: int, font_manager: AcademicFontManager):
        self.rect = pygame.Rect(x, y, width, height)
        self.font_manager = font_manager
        self.padding = {'left': 12, 'right': 12, 'top': 32, 'bottom': 28}
        
        # Q值热图数据缓存（按算法类型）
        self.q_maps: Dict[str, np.ndarray] = {}
        self.update_counter = 0
        self.update_frequency = 50  # 每50步更新一次（降低计算频率）
        
        # 增量更新机制：每次只更新部分区域
        self.incremental_update = True
        self.update_region_index = 0  # 当前更新的区域索引
        self.update_regions = 4  # 将环境分为4个区域，每次更新一个
        
        # 预渲染的surface缓存（避免每帧重新计算）
        self.cached_overlay: Optional[pygame.Surface] = None
        self.cached_q_map: Optional[np.ndarray] = None
        self.cached_agent_type: Optional[str] = None
        
        # 智能采样：只在智能体附近详细计算
        self.smart_sampling = True
        self.detailed_radius = 10  # 智能体周围详细计算的半径
        self.coarse_sample_density = 4  # 远离智能体区域的采样密度
        
        # 显示开关
        self.enabled = False  # 默认关闭，需要用户手动开启
        
        # 开关按钮rect（在draw时设置）
        self.toggle_rect: Optional[pygame.Rect] = None
        
    def update(self, agents_by_type: Dict[str, List[Any]], environment: Any) -> None:
        """
        更新Q值热图数据（性能优化版）
        
        采用增量更新策略：每次只更新部分区域，避免一次性计算整个环境。
        """
        if not self.enabled:
            return
        
        self.update_counter += 1
        
        # 为每种学习型算法计算Q值热图
        for agent_type, agents in agents_by_type.items():
            if not agents or agent_type not in ['iql', 'independent_q_learning', 'qmix']:
                continue
            
            # 使用第一个智能体作为代表
            representative_agent = None
            for agent in agents:
                if hasattr(agent, 'get_q_value_map') and hasattr(agent, 'q_network') and agent.q_network is not None:
                    representative_agent = agent
                    break
            
            if representative_agent is None:
                continue
            
            try:
                # 初始化Q值图（如果不存在）
                if agent_type not in self.q_maps:
                    grid_size = getattr(environment, 'size', 80)
                    self.q_maps[agent_type] = np.zeros((grid_size, grid_size), dtype=np.float32)
                
                q_map = self.q_maps[agent_type]
                grid_size = q_map.shape[0]
                
                # 增量更新策略
                if self.incremental_update and self.update_counter % (self.update_frequency // self.update_regions) == 0:
                    # 每次只更新一个区域
                    region_size = grid_size // self.update_regions
                    start_x = (self.update_region_index % self.update_regions) * region_size
                    end_x = min(start_x + region_size, grid_size)
                    
                    # 更新该区域的Q值
                    self._update_region(representative_agent, environment, q_map, start_x, end_x, grid_size)
                    
                    # 移动到下一个区域
                    self.update_region_index = (self.update_region_index + 1) % self.update_regions
                elif self.update_counter % self.update_frequency == 0:
                    # 完整更新（每N次增量更新后进行一次完整更新）
                    if self.smart_sampling:
                        # 智能采样：在智能体附近详细计算，远离区域粗略采样
                        self._update_with_smart_sampling(representative_agent, environment, q_map, agents)
                    else:
                        # 标准采样
                        full_map = representative_agent.get_q_value_map(environment, sample_density=3)
                        if full_map is not None and full_map.size > 0:
                            self.q_maps[agent_type] = full_map
                
                # 清除缓存（如果数据已更新）
                if agent_type != self.cached_agent_type:
                    self.cached_overlay = None
                    self.cached_q_map = None
                    self.cached_agent_type = agent_type
                    
            except Exception as e:
                print(f"Error updating Q-value map for {agent_type}: {e}")
                continue
    
    def _update_region(self, agent: Any, environment: Any, q_map: np.ndarray, 
                      start_x: int, end_x: int, grid_size: int) -> None:
        """更新指定区域的Q值"""
        original_x, original_y = agent.x, agent.y
        
        try:
            for x in range(start_x, end_x):
                for y in range(0, grid_size, 3):  # 采样密度3
                    agent.x, agent.y = x, y
                    observation = agent.observe(environment)
                    q_values = agent.get_q_values(observation)
                    q_map[x, y] = np.max(q_values) if len(q_values) > 0 else 0.0
                    
                    # 填充周围区域
                    for dx in range(3):
                        for dy in range(3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                q_map[nx, ny] = q_map[x, y]
        finally:
            agent.x, agent.y = original_x, original_y
    
    def _update_with_smart_sampling(self, agent: Any, environment: Any, 
                                   q_map: np.ndarray, agents: List[Any]) -> None:
        """使用智能采样更新Q值图：在智能体附近详细计算，远离区域粗略采样"""
        grid_size = q_map.shape[0]
        original_x, original_y = agent.x, agent.y
        
        try:
            # 计算智能体的平均位置（用于确定详细计算区域）
            agent_positions = [(a.x, a.y) for a in agents if hasattr(a, 'x') and hasattr(a, 'y')]
            if not agent_positions:
                return
            
            avg_x = int(np.mean([pos[0] for pos in agent_positions]))
            avg_y = int(np.mean([pos[1] for pos in agent_positions]))
            
            # 遍历环境
            for x in range(0, grid_size, self.coarse_sample_density):
                for y in range(0, grid_size, self.coarse_sample_density):
                    # 计算到智能体平均位置的距离
                    dist = np.sqrt((x - avg_x)**2 + (y - avg_y)**2)
                    
                    # 在智能体附近使用更密集的采样
                    if dist < self.detailed_radius:
                        sample_density = 1  # 详细采样
                    else:
                        sample_density = self.coarse_sample_density  # 粗略采样
                    
                    if x % sample_density == 0 and y % sample_density == 0:
                        agent.x, agent.y = x, y
                        observation = agent.observe(environment)
                        q_values = agent.get_q_values(observation)
                        q_val = np.max(q_values) if len(q_values) > 0 else 0.0
                        
                        # 填充周围区域
                        for dx in range(sample_density):
                            for dy in range(sample_density):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                    q_map[nx, ny] = q_val
        finally:
            agent.x, agent.y = original_x, original_y
    
    def draw(self, screen: pygame.Surface, grid_size: int, cell_size: int, 
             env_x: int = 0, env_y: int = 0) -> None:
        """
        绘制Q值热图叠加在环境地图上（使用缓存优化性能）
        
        Args:
            screen: Pygame surface
            grid_size: 网格大小
            cell_size: 单元格大小
            env_x, env_y: 环境地图的左上角坐标
        """
        # 即使未启用，也显示开关按钮（让用户可以开启）
        # 在右上角显示图例（包含开关按钮）
        legend_x = env_x + grid_size * cell_size - 150
        legend_y = env_y + 10
        legend_rect = pygame.Rect(legend_x, legend_y, 140, 90)  # 增加高度以容纳开关
        pygame.draw.rect(screen, COLORS['CHART_BG'], legend_rect)
        pygame.draw.rect(screen, COLORS['PANEL_BORDER'], legend_rect, 1)
        
        # 图例标题
        agent_type_label = "IQL/QMIX"  # 默认标签
        if self.q_maps:
            for agent_type in self.q_maps.keys():
                agent_type_label = 'IQL' if 'iql' in agent_type else 'QMIX'
                break
        
        title_text = self.font_manager.render_text(
            f"Q-Value Heatmap ({agent_type_label})", 'TINY', COLORS['TEXT_PRIMARY']
        )
        screen.blit(title_text, (legend_x + 5, legend_y + 5))
        
        # 开关按钮（始终显示）
        toggle_x = legend_x + 5
        toggle_y = legend_y + 20
        toggle_width = 130
        toggle_height = 20
        self.toggle_rect = pygame.Rect(toggle_x, toggle_y, toggle_width, toggle_height)
        
        # 按钮状态颜色
        mouse_pos = pygame.mouse.get_pos()
        if self.toggle_rect.collidepoint(mouse_pos):
            toggle_bg = COLORS['BUTTON_HOVER']
        else:
            toggle_bg = COLORS['BUTTON_NORMAL']
        
        if self.enabled:
            toggle_text = "● Hide Heatmap"
            toggle_text_color = COLORS['SUCCESS']
        else:
            toggle_text = "○ Show Heatmap"
            toggle_text_color = COLORS['TEXT_SECONDARY']
        
        pygame.draw.rect(screen, toggle_bg, self.toggle_rect, border_radius=3)
        pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.toggle_rect, 1, border_radius=3)
        
        toggle_text_surface = self.font_manager.render_text(toggle_text, 'TINY', toggle_text_color)
        text_rect = toggle_text_surface.get_rect(center=self.toggle_rect.center)
        screen.blit(toggle_text_surface, text_rect)
        
        # 如果未启用，不绘制热图，只显示开关
        if not self.enabled or not self.q_maps:
            return
        
        try:
            # 选择第一个可用的Q值热图
            q_map = None
            agent_type_label = None
            current_agent_type = None
            for agent_type, map_data in self.q_maps.items():
                if map_data is not None and map_data.size > 0:
                    q_map = map_data
                    agent_type_label = 'IQL' if 'iql' in agent_type else 'QMIX'
                    current_agent_type = agent_type
                    break
            
            if q_map is None:
                return
            
            # 检查缓存是否有效
            cache_valid = (self.cached_overlay is not None and 
                          self.cached_q_map is not None and
                          np.array_equal(self.cached_q_map, q_map) and
                          self.cached_agent_type == current_agent_type)
            
            if not cache_valid:
                # 重新计算并缓存overlay
                self._rebuild_overlay(q_map, grid_size, cell_size)
                self.cached_q_map = q_map.copy()
                self.cached_agent_type = current_agent_type
            
            # 使用缓存的overlay绘制
            if self.cached_overlay is not None:
                screen.blit(self.cached_overlay, (env_x, env_y))
            
            # 颜色条和数值标签（仅在启用且有数据时显示）
            if self.enabled and self.q_maps:
                bar_width = 120
                bar_height = 8
                bar_x = legend_x + 10
                bar_y = legend_y + 45
                for i in range(bar_width):
                    val = i / bar_width
                    if val < 0.5:
                        r, g, b = 0, int(255 * val * 2), int(255 * (1 - val * 2))
                    else:
                        r, g, b = int(255 * (val - 0.5) * 2), int(255 * (1 - (val - 0.5) * 2)), 0
                    pygame.draw.line(
                        screen,
                        (r, g, b),
                        (bar_x + i, bar_y),
                        (bar_x + i, bar_y + bar_height)
                    )
                
                # 数值标签（使用缓存的值）
                q_min = getattr(self, 'cached_q_min', 0.0)
                q_max = getattr(self, 'cached_q_max', 0.0)
                min_text = self.font_manager.render_text(
                    f"Min: {q_min:.2f}", 'TINY', COLORS['TEXT_SECONDARY']
                )
                max_text = self.font_manager.render_text(
                    f"Max: {q_max:.2f}", 'TINY', COLORS['TEXT_SECONDARY']
                )
                screen.blit(min_text, (bar_x, bar_y + bar_height + 5))
                screen.blit(max_text, (bar_x + bar_width - max_text.get_width(), bar_y + bar_height + 5))
            
        except Exception as e:
            print(f"Error drawing Q-value heatmap: {e}")
    
    def _rebuild_overlay(self, q_map: np.ndarray, grid_size: int, cell_size: int) -> None:
        """重建overlay surface（仅在数据变化时调用）"""
        # 归一化Q值到[0, 1]范围用于颜色映射
        q_min, q_max = np.min(q_map), np.max(q_map)
        if q_max - q_min < 1e-6:
            normalized_q = np.zeros_like(q_map)
        else:
            normalized_q = (q_map - q_min) / (q_max - q_min)
        
        # 使用半透明叠加显示Q值
        alpha = 120  # 透明度（0-255）
        
        # 创建临时surface用于叠加
        overlay = pygame.Surface((grid_size * cell_size, grid_size * cell_size))
        overlay.set_alpha(alpha)
        overlay.fill((0, 0, 0, 0))  # 透明背景
        
        # 优化：使用NumPy批量计算颜色，然后一次性绘制
        # 创建颜色数组
        colors = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        # 批量计算颜色
        mask_low = normalized_q < 0.5
        mask_high = ~mask_low
        
        # 低值区域（蓝色到绿色）
        colors[mask_low, 0] = 0
        colors[mask_low, 1] = (normalized_q[mask_low] * 2 * 255).astype(np.uint8)
        colors[mask_low, 2] = ((1 - normalized_q[mask_low] * 2) * 255).astype(np.uint8)
        
        # 高值区域（绿色到红色）
        colors[mask_high, 0] = ((normalized_q[mask_high] - 0.5) * 2 * 255).astype(np.uint8)
        colors[mask_high, 1] = ((1 - (normalized_q[mask_high] - 0.5) * 2) * 255).astype(np.uint8)
        colors[mask_high, 2] = 0
        
        # 绘制（优化：只绘制非零区域）
        for x in range(grid_size):
            for y in range(grid_size):
                if x < q_map.shape[0] and y < q_map.shape[1]:
                    color = tuple(colors[x, y])
                    pygame.draw.rect(
                        overlay,
                        color,
                        (y * cell_size, x * cell_size, cell_size, cell_size)
                    )
        
        self.cached_overlay = overlay
        self.cached_q_min = q_min
        self.cached_q_max = q_max


class NetworkStatePanel:
    """
    网络内部状态可视化面板
    
    显示策略网络对不同状态的动作概率分布、隐藏层激活模式等。
    """
    
    def __init__(self, x: int, y: int, width: int, height: int, font_manager: AcademicFontManager):
        self.rect = pygame.Rect(x, y, width, height)
        self.font_manager = font_manager
        self.padding = {'left': 12, 'right': 12, 'top': 32, 'bottom': 28}
        
        # 网络状态数据缓存
        self.network_states: Dict[str, Dict[str, Any]] = {}
        self.update_counter = 0
        self.update_frequency = 10  # 每10步更新一次
        
    def update(self, agents_by_type: Dict[str, List[Any]], environment: Any) -> None:
        """更新网络状态数据"""
        self.update_counter += 1
        if self.update_counter % self.update_frequency != 0:
            return
        
        self.network_states.clear()
        
        # 为每种学习型算法获取网络状态
        for agent_type, agents in agents_by_type.items():
            if not agents or agent_type not in ['iql', 'independent_q_learning', 'qmix']:
                continue
            
            # 使用第一个智能体作为代表
            representative_agent = None
            for agent in agents:
                if hasattr(agent, 'get_network_state') and hasattr(agent, 'q_network') and agent.q_network is not None:
                    representative_agent = agent
                    break
            
            if representative_agent is None:
                continue
            
            try:
                # 获取当前观察
                observation = representative_agent.observe(environment)
                
                # 获取网络状态
                network_state = representative_agent.get_network_state(observation)
                if network_state:
                    self.network_states[agent_type] = network_state
            except Exception as e:
                print(f"Error getting network state for {agent_type}: {e}")
                continue
    
    def draw(self, screen: pygame.Surface) -> None:
        """绘制网络状态面板"""
        try:
            pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
            
            # 标题
            title_surface = self.font_manager.render_text(
                "Network Internal State", 'SMALL', COLORS['TEXT_PRIMARY']
            )
            screen.blit(title_surface, (self.rect.x + self.padding['left'], self.rect.y + 8))
            
            if not self.network_states:
                empty_text = self.font_manager.render_text(
                    "No network state data", 'TINY', COLORS['TEXT_MUTED']
                )
                text_rect = empty_text.get_rect(center=self.rect.center)
                screen.blit(empty_text, text_rect)
                return
            
            # 绘制每个算法的网络状态
            y_offset = self.rect.y + self.padding['top'] + 10
            row_height = 80
            
            for agent_type, state in self.network_states.items():
                if y_offset + row_height > self.rect.y + self.rect.height - self.padding['bottom']:
                    break
                
                # 算法标签
                agent_label = 'IQL' if 'iql' in agent_type else 'QMIX'
                label_color = COLORS['AGENT_IQL'] if 'iql' in agent_type else COLORS['AGENT_QMIX']
                
                label_text = self.font_manager.render_text(
                    f"{agent_label} Network State", 'TINY', label_color
                )
                screen.blit(label_text, (self.rect.x + self.padding['left'], y_offset))
                
                # 策略分布条形图
                if 'policy' in state and len(state['policy']) > 0:
                    policy = state['policy']
                    bar_width = 15
                    bar_spacing = 2
                    chart_x = self.rect.x + self.padding['left']
                    chart_y = y_offset + 20
                    chart_width = self.rect.width - self.padding['left'] - self.padding['right']
                    chart_height = 30
                    
                    max_policy = np.max(policy) if len(policy) > 0 else 1.0
                    
                    for i, prob in enumerate(policy):
                        if i * (bar_width + bar_spacing) > chart_width:
                            break
                        
                        bar_height = int((prob / max_policy) * chart_height) if max_policy > 0 else 0
                        bar_x = chart_x + i * (bar_width + bar_spacing)
                        bar_y = chart_y + chart_height - bar_height
                        
                        # 使用算法颜色
                        pygame.draw.rect(
                            screen,
                            label_color,
                            (bar_x, bar_y, bar_width, bar_height)
                        )
                        pygame.draw.rect(
                            screen,
                            COLORS['PANEL_BORDER'],
                            (bar_x, bar_y, bar_width, bar_height),
                            1
                        )
                    
                    # 策略熵值
                    if len(policy) > 0:
                        entropy = -np.sum(policy * np.log(policy + 1e-10))
                        entropy_text = self.font_manager.render_text(
                            f"Policy Entropy: {entropy:.3f}", 'TINY', COLORS['TEXT_SECONDARY']
                        )
                        screen.blit(entropy_text, (chart_x, chart_y + chart_height + 5))
                
                # Q值信息
                if 'q_values' in state and len(state['q_values']) > 0:
                    q_values = state['q_values']
                    max_q = np.max(q_values)
                    min_q = np.min(q_values)
                    avg_q = np.mean(q_values)
                    
                    q_info_text = self.font_manager.render_text(
                        f"Q: avg={avg_q:.2f}, max={max_q:.2f}, min={min_q:.2f}",
                        'TINY',
                        COLORS['TEXT_SECONDARY']
                    )
                    screen.blit(q_info_text, (self.rect.x + self.padding['left'], y_offset + 60))
                
                y_offset += row_height + 10
            
        except Exception as e:
            print(f"Error drawing network state panel: {e}")


class ActionDistributionPanel:
    """
    动作分布可视化面板。

    输入数据结构：
        action_dist = {
            'iql': {0: 10, 1: 5, 2: 1, ...},
            'qmix': {0: 3, 2: 7, ...},
            ...
        }

    面板设计：
    - 左侧：图例（算法类型 + 总计动作数 + 全局占比 + 策略熵 + Top‑3 actions）
    - 右侧：按动作索引分组的组合柱状图，每个动作组内分算法类型绘制，
      使用「相对频率」而非绝对次数，突出策略偏好。
    """

    def __init__(self, x: int, y: int, width: int, height: int, font_manager: AcademicFontManager):
        self.rect = pygame.Rect(x, y, width, height)
        self.font_manager = font_manager
        self.padding = {'left': 12, 'right': 12, 'top': 32, 'bottom': 28}

        self.agent_type_colors = {
            'iql': COLORS['AGENT_IQL'],
            'independent_q_learning': COLORS['AGENT_IQL'],
            'qmix': COLORS['AGENT_QMIX'],
            'rule_based': COLORS['AGENT_RULE_BASED'],
        }
        self.agent_type_labels = {
            'iql': 'IQL',
            'independent_q_learning': 'IQL',
            'qmix': 'QMIX',
            'rule_based': 'Rule-Based',
        }

        # 默认的离散动作语义映射（可根据具体环境在后续迭代中扩展 / 替换）
        # 这里只提供简短的英文标签，避免中文渲染问题。
        self.action_semantic_map: Dict[int, str] = {
            0: "Move Up",
            1: "Move Right",
            2: "Collect",
            3: "Move Left",
            4: "Move Down",
            5: "Stay",
            6: "Explore",
            7: "Share",
        }

    def _format_action_label(self, action_idx: int) -> str:
        """构造带语义的动作标签，例如 'A0: Move Up'"""
        name = self.action_semantic_map.get(action_idx)
        if name:
            return f"A{action_idx}: {name}"
        return f"A{action_idx}"

    def draw(self, screen: pygame.Surface, action_dist: Dict[str, Dict[int, int]]) -> None:
        try:
            pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)

            # 标题 + 简短说明：一眼看出在展示什么
            title_surface = self.font_manager.render_text(
                "Action Distribution (per algorithm)", 'SMALL', COLORS['TEXT_PRIMARY']
            )
            screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 4))

            subtitle_surface = self.font_manager.render_text(
                "Each group = one discrete action; bar height = relative selection frequency per algorithm.",
                'TINY',
                COLORS['TEXT_SECONDARY'],
            )
            screen.blit(subtitle_surface, (self.rect.x + 5, self.rect.y + 18))

            if not action_dist:
                empty_text = self.font_manager.render_text("No action data", 'TINY', COLORS['TEXT_MUTED'])
                text_rect = empty_text.get_rect(center=self.rect.center)
                screen.blit(empty_text, text_rect)
                return

            # 计算每种类型的总动作数，并收集所有动作索引
            type_totals: Dict[str, int] = {}
            action_indices: set[int] = set()
            for agent_type, counts in action_dist.items():
                if not counts:
                    continue
                total = sum(int(c) for c in counts.values())
                if total <= 0:
                    continue
                type_totals[agent_type] = total
                action_indices.update(counts.keys())

            if not type_totals or not action_indices:
                empty_text = self.font_manager.render_text("No action data", 'TINY', COLORS['TEXT_MUTED'])
                text_rect = empty_text.get_rect(center=self.rect.center)
                screen.blit(empty_text, text_rect)
                return

            sorted_types = sorted(type_totals.items(), key=lambda kv: kv[0])
            sorted_actions = sorted(action_indices)

            # 如果动作维度很多，只显示最具代表性的 Top-K 动作，防止过度拥挤
            max_actions_to_show = 12
            if len(sorted_actions) > max_actions_to_show:
                action_global_totals: Dict[int, int] = {}
                for idx in sorted_actions:
                    action_global_totals[idx] = sum(
                        action_dist.get(t, {}).get(idx, 0) for t in type_totals.keys()
                    )
                top_actions = sorted(
                    sorted_actions,
                    key=lambda idx: action_global_totals.get(idx, 0),
                    reverse=True
                )[:max_actions_to_show]
                sorted_actions = sorted(top_actions)

            # 左侧图例
            legend_x = self.rect.x + self.padding['left']
            legend_y = self.rect.y + self.padding['top']
            legend_line_h = 18
            total_actions_all = sum(type_totals.values())

            # 预先计算每个算法的策略熵（基于动作分布的离散熵，单位 bit）
            entropies: Dict[str, float] = {}
            for agent_type, total in type_totals.items():
                counts = action_dist.get(agent_type, {})
                if not counts or total <= 0:
                    continue
                probs = []
                for c in counts.values():
                    c = int(c)
                    if c > 0:
                        probs.append(c / total)
                if not probs:
                    continue
                entropy = 0.0
                for p in probs:
                    # 使用 log2，便于直观解释“比特数”
                    entropy -= p * math.log(p + 1e-12, 2)
                entropies[agent_type] = entropy

            for i, (agent_type, total) in enumerate(sorted_types):
                color = self.agent_type_colors.get(agent_type, COLORS['GRAY'])
                label = self.agent_type_labels.get(agent_type, agent_type.title())
                proportion = (total / total_actions_all) * 100 if total_actions_all > 0 else 0.0
                entropy_text = ""
                if agent_type in entropies:
                    entropy_text = f", H={entropies[agent_type]:.2f}"

                # 计算该算法下最常用的 Top‑3 动作（按相对频率）
                counts = action_dist.get(agent_type, {})
                top_actions_text = ""
                if counts and total > 0:
                    # (action_idx, freq) 按 freq 降序排序
                    freqs = []
                    for idx, c in counts.items():
                        c_int = int(c)
                        if c_int <= 0:
                            continue
                        freqs.append((idx, c_int / total))
                    freqs.sort(key=lambda kv: kv[1], reverse=True)
                    if freqs:
                        top_k = freqs[:3]
                        parts = []
                        for a_idx, f in top_k:
                            label_str = self._format_action_label(a_idx)
                            parts.append(f"{label_str} ({f*100:.1f}%)")
                        top_actions_text = "; ".join(parts)

                main_text = f"{label}: {total} ({proportion:.1f}%){entropy_text}"

                indicator_rect = pygame.Rect(legend_x, legend_y + i * legend_line_h, 10, 10)
                pygame.draw.rect(screen, color, indicator_rect)
                pygame.draw.rect(screen, COLORS['LEGEND_BORDER'], indicator_rect, 1)

                text_surface = self.font_manager.render_text(main_text, 'TINY', COLORS['TEXT_PRIMARY'])
                screen.blit(text_surface, (legend_x + 16, legend_y - 2 + i * legend_line_h))

                if top_actions_text:
                    summary_surface = self.font_manager.render_text(
                        f"Top-3 actions: {top_actions_text}", 'TINY', COLORS['TEXT_SECONDARY']
                    )
                    screen.blit(summary_surface, (legend_x + 16, legend_y + 8 + i * legend_line_h))

            # 右侧组合柱状图区域
            chart_left = self.rect.x + self.rect.width // 2 + 5
            chart_area = pygame.Rect(
                chart_left,
                self.rect.y + self.padding['top'] + 4,
                self.rect.width - (chart_left - self.rect.x) - self.padding['right'],
                self.rect.height - self.padding['top'] - self.padding['bottom'],
            )

            num_actions = len(sorted_actions)
            num_types = len(sorted_types)
            if num_actions <= 0 or num_types <= 0:
                return

            group_width = chart_area.width / num_actions
            # 给每个动作组留一点间隔
            group_inner_width = group_width * 0.8
            bar_width = max(4, int(group_inner_width / num_types))

            # 找出全局最大「相对频率」用于归一化（更具可比性）
            max_value = 0.0
            for agent_type, total in type_totals.items():
                counts = action_dist.get(agent_type, {})
                if not counts or total <= 0:
                    continue
                for idx, c in counts.items():
                    if idx not in sorted_actions:
                        continue
                    c = int(c)
                    if c <= 0:
                        continue
                    value = c / total
                    if value > max_value:
                        max_value = value

            if max_value <= 0:
                return

            # 在柱状图区绘制简单的 Y 轴参考线（0%、50%、100%），帮助理解“高度代表频率”
            # 这里原本使用了 COLORS['GRID_MINOR'] / COLORS['GRID_MAJOR']，但这些键在全局调色板中不存在，
            # 会导致 KeyError。改为复用已有的网格和边框颜色，既保持风格统一，又避免运行时错误。
            for frac, label in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
                y = chart_area.bottom - int(frac * chart_area.height)
                # 中间参考线使用图表网格颜色，两端（0%、100%）使用面板边框颜色以强调边界
                color = COLORS['CHART_GRID'] if frac not in (0.0, 1.0) else COLORS['PANEL_BORDER']
                pygame.draw.line(screen, color, (chart_area.x, y), (chart_area.right, y), 1)

                tick_text = self.font_manager.render_text(label, 'TINY', COLORS['TEXT_SECONDARY'])
                screen.blit(tick_text, (chart_area.x - 28, y - 6))

            # 为每个动作索引画一组条形
            for action_i, action_idx in enumerate(sorted_actions):
                group_x_center = chart_area.x + (action_i + 0.5) * group_width
                # 每种类型的条从左到右排列
                start_x = group_x_center - (group_inner_width / 2)

                for t_i, (agent_type, _) in enumerate(sorted_types):
                    counts = action_dist.get(agent_type, {})
                    raw = int(counts.get(action_idx, 0))
                    if raw <= 0:
                        continue

                    total = type_totals.get(agent_type, 0)
                    if total <= 0:
                        continue

                    value = raw / total
                    height_ratio = value / max_value
                    bar_height = int(height_ratio * chart_area.height)
                    x = int(start_x + t_i * bar_width)
                    y = chart_area.bottom - bar_height

                    bar_rect = pygame.Rect(x, y, bar_width, bar_height)
                    color = self.agent_type_colors.get(agent_type, COLORS['GRAY'])
                    pygame.draw.rect(screen, color, bar_rect)
                    pygame.draw.rect(screen, COLORS['PANEL_BORDER'], bar_rect, 1)

                # 在组底部标记动作索引 + 简短语义
                idx_label = self._format_action_label(action_idx)
                idx_text = self.font_manager.render_text(idx_label, 'TINY', COLORS['TEXT_SECONDARY'])
                idx_rect = idx_text.get_rect(center=(group_x_center, chart_area.bottom + 8))
                screen.blit(idx_text, idx_rect)

            # X / Y 轴含义说明
            axis_x_text = self.font_manager.render_text(
                "Discrete action IDs A0, A1, A2, ... (with semantics)", 'TINY', COLORS['TEXT_MUTED']
            )
            screen.blit(
                axis_x_text,
                (chart_area.x, self.rect.bottom - self.padding['bottom'] + 10),
            )

            axis_y_text = self.font_manager.render_text(
                "Relative frequency (selection ratio within each algorithm)", 'TINY', COLORS['TEXT_MUTED']
            )
            screen.blit(
                axis_y_text,
                (chart_area.x - 4, chart_area.y - 16),
            )

        except Exception as e:
            print(f"Action distribution panel drawing failed: {e}")

class RealTimeChart:
    """Real-time Line Chart Component for Academic Visualization"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 title: str, y_label: str, font_manager: AcademicFontManager,
                 color: Tuple[int, int, int] = COLORS['CHART_LINE_1'],
                 max_points: int = 200):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.y_label = y_label
        self.font_manager = font_manager
        self.color = color
        self.max_points = max_points
        
        # Data storage
        self.data: Deque[float] = deque(maxlen=max_points)
        self.timestamps: Deque[int] = deque(maxlen=max_points)
        
        # Chart padding
        self.padding = {
            'left': 45,
            'right': 8,
            'top': 24,
            'bottom': 26
        }
    
    def add_data_point(self, value: float, timestamp: int) -> None:
        """Add a new data point"""
        # Ensure value is a valid number
        try:
            value = float(value)
            if not (np.isnan(value) or np.isinf(value)):
                self.data.append(value)
                self.timestamps.append(int(timestamp))
        except (ValueError, TypeError):
            # Skip invalid values
            pass
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the chart"""
        if len(self.data) < 2:
            self._draw_empty_chart(screen)
            return
        
        try:
            # Chart area
            chart_area = pygame.Rect(
                self.rect.x + self.padding['left'],
                self.rect.y + self.padding['top'],
                self.rect.width - self.padding['left'] - self.padding['right'],
                self.rect.height - self.padding['top'] - self.padding['bottom']
            )
            
            # Background
            pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
            
            # Title
            title_surface = self.font_manager.render_text(self.title, 'SMALL', COLORS['TEXT_PRIMARY'])
            screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
            
            # Calculate data range with padding to avoid cramped lines
            min_val = min(self.data)
            max_val = max(self.data)
            if min_val == max_val:
                # Degenerate case: expand symmetrically around value
                delta = 1.0 if min_val == 0 else abs(min_val) * 0.1
                min_val = min_val - delta
                max_val = max_val + delta
            else:
                # Add 10% padding on both sides
                margin = (max_val - min_val) * 0.1
                min_val -= margin
                max_val += margin
            
            val_range = max_val - min_val or 1.0
            
            # Draw grid lines
            self._draw_grid(screen, chart_area, min_val, max_val)
            
            # Draw data line
            self._draw_data_line(screen, chart_area, min_val, val_range)
            
            # Draw axes labels
            self._draw_axes_labels(screen, chart_area, min_val, max_val)
            
        except Exception as e:
            print(f"Chart drawing failed: {e}")
    
    def _draw_empty_chart(self, screen: pygame.Surface) -> None:
        """Draw empty chart placeholder"""
        pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
        pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
        
        title_surface = self.font_manager.render_text(self.title, 'SMALL', COLORS['TEXT_PRIMARY'])
        screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
        
        empty_text = self.font_manager.render_text("Collecting data...", 'TINY', COLORS['TEXT_MUTED'])
        text_rect = empty_text.get_rect(center=self.rect.center)
        screen.blit(empty_text, text_rect)
    
    def _draw_grid(self, screen: pygame.Surface, chart_area: pygame.Rect, 
                   min_val: float, max_val: float) -> None:
        """Draw grid lines"""
        # Horizontal grid lines
        num_lines = 5
        for i in range(num_lines + 1):
            y = chart_area.y + (i * chart_area.height // num_lines)
            pygame.draw.line(screen, COLORS['CHART_GRID'],
                           (chart_area.x, y), (chart_area.right, y), 1)
    
    def _draw_data_line(self, screen: pygame.Surface, chart_area: pygame.Rect,
                       min_val: float, val_range: float) -> None:
        """Draw data line"""
        if len(self.data) < 2:
            return
        
        points = []
        for i, value in enumerate(self.data):
            x = chart_area.x + (i / (len(self.data) - 1)) * chart_area.width
            y = chart_area.bottom - ((value - min_val) / val_range) * chart_area.height
            points.append((int(x), int(y)))
        
        if len(points) > 1:
            # Draw line
            pygame.draw.lines(screen, self.color, False, points, 2)
            
            # Draw points
            for point in points[::max(1, len(points)//20)]:  # Sample points for performance
                pygame.draw.circle(screen, self.color, point, 2)
    
    def _draw_axes_labels(self, screen: pygame.Surface, chart_area: pygame.Rect,
                         min_val: float, max_val: float) -> None:
        """Draw axis labels"""
        # Y-axis labels
        num_labels = 5
        for i in range(num_labels + 1):
            value = min_val + (max_val - min_val) * (num_labels - i) / num_labels
            y = chart_area.y + (i * chart_area.height // num_labels)
            
            value_text = f"{value:.1f}"
            label_surface = self.font_manager.render_text(value_text, 'TINY', COLORS['TEXT_SECONDARY'])
            screen.blit(label_surface, (self.rect.x + 5, y - 8))
        
        # Y-axis label
        y_label_surface = self.font_manager.render_text(self.y_label, 'TINY', COLORS['TEXT_SECONDARY'])
        # Rotate 90 degrees would require additional surface manipulation
        screen.blit(y_label_surface, (self.rect.x + 5, self.rect.y + self.rect.height // 2))




class MARLSimulationRenderer:
    """MARL Simulation Renderer - Academic Professional Style"""
    
    def __init__(self, grid_size: int, cell_size: int):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.font_manager = AcademicFontManager()
        
        # Agent type color mapping
        self.agent_type_colors = {
            'rule_based': COLORS['AGENT_RULE_BASED'],
            'independent_q_learning': COLORS['AGENT_IQL'],
            'iql': COLORS['AGENT_IQL'],
            'qmix': COLORS['AGENT_QMIX'],
            'conservative': COLORS['AGENT_CONSERVATIVE'],
            'exploratory': COLORS['AGENT_EXPLORATORY'],
            'adaptive': COLORS['AGENT_ADAPTIVE'],
        }
    
    def get_screen_info(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        screen_width = self.grid_size * self.cell_size + 500  # Panel width
        screen_height = max(900, self.grid_size * self.cell_size)
        return (screen_width, screen_height)
    
    def draw_environment(self, screen: pygame.Surface, sugar_map: np.ndarray) -> None:
        """Draw environment with academic color scheme"""
        if sugar_map is None:
            return
        
        try:
            rows, cols = sugar_map.shape
            self.grid_size = rows
            
            # Get max sugar for normalization (if needed)
            max_sugar = float(np.max(sugar_map)) if sugar_map.size > 0 else 10.0
            
            # Draw sugar distribution with gradient
            for x in range(rows):
                for y in range(cols):
                    sugar = float(sugar_map[x, y])
                    if sugar <= 0:
                        # Draw empty cell with very light background
                        color = COLORS.get('BACKGROUND', (250, 250, 250))
                    else:
                        # Normalize sugar value for color calculation
                        normalized_sugar = sugar / max_sugar if max_sugar > 0 else 0
                        # Scale to actual max (10.0)
                        actual_sugar = normalized_sugar * 10.0
                        color = self._get_sugar_color(actual_sugar)
                    
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(screen, color, rect)
            
            # Draw subtle grid
            self._draw_grid(screen, rows, cols)
            
        except Exception as e:
            print(f"Environment drawing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_sugar_color(self, sugar: float) -> Tuple[int, int, int]:
        """Get sugar color using academic sequential scheme with better gradient"""
        max_sugar = 10.0  # Default max sugar
        sugar = max(0.0, min(sugar, max_sugar))  # Clamp sugar value
        
        # More granular color interpolation for better gradient visibility
        if sugar < 1.0:
            # Very low sugar - very light green
            ratio = sugar / 1.0
            return self._interpolate_color(COLORS.get('BACKGROUND', (250, 250, 250)), 
                                         COLORS.get('SUGAR_LOW', (237, 248, 233)), ratio)
        elif sugar < 2.5:
            # Low sugar
            ratio = (sugar - 1.0) / 1.5
            return self._interpolate_color(COLORS.get('SUGAR_LOW', (237, 248, 233)), 
                                         COLORS.get('SUGAR_MEDIUM', (186, 228, 179)), ratio)
        elif sugar < 5.0:
            # Medium sugar
            ratio = (sugar - 2.5) / 2.5
            return self._interpolate_color(COLORS.get('SUGAR_MEDIUM', (186, 228, 179)), 
                                         COLORS.get('SUGAR_HIGH', (116, 196, 118)), ratio)
        elif sugar < 7.5:
            # High sugar
            ratio = (sugar - 5.0) / 2.5
            return self._interpolate_color(COLORS.get('SUGAR_HIGH', (116, 196, 118)), 
                                         COLORS.get('SUGAR_MAX', (35, 139, 69)), ratio)
        else:
            # Maximum sugar
            return COLORS.get('SUGAR_MAX', (35, 139, 69))
    
    def _interpolate_color(self, color1: Tuple, color2: Tuple, ratio: float) -> Tuple[int, int, int]:
        """Color interpolation"""
        ratio = max(0, min(1, ratio))
        return (
            int(color1[0] + (color2[0] - color1[0]) * ratio),
            int(color1[1] + (color2[1] - color1[1]) * ratio),
            int(color1[2] + (color2[2] - color1[2]) * ratio)
        )
    
    def _draw_grid(self, screen: pygame.Surface, rows: int, cols: int) -> None:
        """Draw subtle grid lines"""
        grid_color = COLORS['CHART_GRID']
        width_px = rows * self.cell_size
        height_px = cols * self.cell_size
        
        # Draw every 5th line for performance
        for i in range(0, rows + 1, 5):
            pygame.draw.line(
                screen, grid_color,
                (i * self.cell_size, 0),
                (i * self.cell_size, height_px),
                1
            )
        for j in range(0, cols + 1, 5):
            pygame.draw.line(
                screen, grid_color,
                (0, j * self.cell_size),
                (width_px, j * self.cell_size),
                1
            )
        
        # Border
        border_rect = pygame.Rect(0, 0, width_px, height_px)
        pygame.draw.rect(screen, COLORS['PANEL_BORDER'], border_rect, 2)
    
    def draw_agents(self, screen: pygame.Surface, agents: List[Any]) -> None:
        """Draw all agents with clear type and status indication"""
        try:
            agents_by_type: Dict[str, List[Any]] = {}
            for agent_data in agents:
                normalized_agent = self._normalize_agent(agent_data)
                if normalized_agent is None:
                    continue
                
                agent_type = getattr(normalized_agent, 'agent_type', None)
                type_key = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
                if type_key not in agents_by_type:
                    agents_by_type[type_key] = []
                agents_by_type[type_key].append(normalized_agent)
            
            # Draw by type for better organization
            for agent_type, type_agents in agents_by_type.items():
                for agent in type_agents:
                    self._draw_agent(screen, agent, agent_type)
                    
        except Exception as e:
            print(f"Agent drawing failed: {e}")
    
    def _normalize_agent(self, agent: Any) -> Optional[Any]:
        """Normalize agent data to consistent format"""
        if agent is None:
            return None
        
        # Native object (from simulation.agent_manager.agents)
        if hasattr(agent, 'status') and hasattr(agent, 'x') and hasattr(agent, 'y'):
            status_value = getattr(agent.status, 'value', getattr(agent.status, None))
            if status_value is not None and status_value != 'alive':
                return None
            return agent
        
        # Dictionary format (from simulation.get_simulation_data)
        if isinstance(agent, dict):
            status = agent.get('status', 'alive')
            if status != 'alive':
                return None
            
            # Ensure required fields exist
            if 'x' not in agent or 'y' not in agent:
                return None
            
            agent_type = agent.get('type', 'rule_based')
            return SimpleNamespace(
                x=agent.get('x', 0),
                y=agent.get('y', 0),
                sugar=agent.get('sugar', 0),
                age=agent.get('age', 0),
                agent_type=SimpleNamespace(value=agent_type),
                status=SimpleNamespace(value='alive')
            )
        
        return None
    
    def _draw_agent(self, screen: pygame.Surface, agent: Any, agent_type: str) -> None:
        """Draw single agent with academic style"""
        try:
            # Get position - handle both int and float coordinates
            agent_x = int(getattr(agent, 'x', 0))
            agent_y = int(getattr(agent, 'y', 0))
            
            # Calculate screen position
            x = agent_x * self.cell_size + self.cell_size // 2
            y = agent_y * self.cell_size + self.cell_size // 2
            
            # Ensure position is within screen bounds
            if x < 0 or y < 0 or x >= screen.get_width() or y >= screen.get_height():
                return
            
            # Base size
            base_radius = max(4, self.cell_size // 3)  # Larger radius for visibility
            sugar = float(getattr(agent, 'sugar', 0))
            
            # Size based on wealth (subtle)
            size_multiplier = 1.0 + min(sugar / 50.0, 1.0) * 0.3
            radius = int(base_radius * size_multiplier)
            radius = max(3, min(radius, self.cell_size // 2))  # Clamp radius
            
            # Get type color
            base_color = self.agent_type_colors.get(agent_type, COLORS.get('GRAY', (128, 128, 128)))
            
            # Draw agent circle with better visibility
            pygame.draw.circle(screen, base_color, (x, y), radius)
            
            # Draw border (darker for contrast)
            border_color = self._darken_color(base_color, 0.6)
            pygame.draw.circle(screen, border_color, (x, y), radius, 2)
            
            # Draw status indicator (small inner circle for wealth)
            if sugar > 20:
                inner_radius = max(2, radius // 3)
                pygame.draw.circle(screen, COLORS.get('AGENT_WEALTHY', (0, 123, 255)), (x, y), inner_radius)
            
        except Exception as e:
            print(f"Single agent drawing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _darken_color(self, color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Darken color"""
        return (
            max(0, int(color[0] * factor)),
            max(0, int(color[1] * factor)),
            max(0, int(color[2] * factor))
        )


class AcademicControlPanel:
    """Academic-style Control Panel"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.font_manager = AcademicFontManager()
        self.buttons = self._create_buttons()
        
    def _create_buttons(self) -> Dict[str, Dict]:
        """Create control buttons"""
        button_width = 180
        button_height = 35
        start_x = self.rect.x + 15
        start_y = self.rect.y + 50
        
        buttons = {
            'play_pause': {
                'rect': pygame.Rect(start_x, start_y, button_width, button_height),
                'text': 'Pause',
                'active': False,
            },
            'reset': {
                'rect': pygame.Rect(start_x, start_y + 45, button_width, button_height),
                'text': 'Reset',
                'active': False,
            },
        }
        return buttons
    
    def draw(self, screen: pygame.Surface, metrics: UIMetrics) -> None:
        """Draw control panel"""
        try:
            # Panel background
            pygame.draw.rect(screen, COLORS['PANEL_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
            
            # Title
            title = self.font_manager.render_text("MARL Simulation Control", 'HEADING', COLORS['TEXT_PRIMARY'])
            screen.blit(title, (self.rect.x + 15, self.rect.y + 15))
            
            # Draw buttons
            self._draw_buttons(screen)
            
            # Draw statistics
            self._draw_statistics(screen, metrics)
            
            # Draw status
            self._draw_status(screen, metrics)
            
        except Exception as e:
            print(f"Control panel drawing failed: {e}")
    
    def _draw_buttons(self, screen: pygame.Surface) -> None:
        """Draw buttons"""
        mouse_pos = pygame.mouse.get_pos()
        
        for name, button in self.buttons.items():
            rect = button['rect']
            
            if button['active']:
                color = COLORS['BUTTON_ACTIVE']
                text_color = COLORS['WHITE']
            elif rect.collidepoint(mouse_pos):
                color = COLORS['BUTTON_HOVER']
                text_color = COLORS['TEXT_PRIMARY']
            else:
                color = COLORS['BUTTON_NORMAL']
                text_color = COLORS['TEXT_PRIMARY']
            
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], rect, 1, border_radius=4)
            
            text_surface = self.font_manager.render_text(button['text'], 'BODY', text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)
    
    def _draw_statistics(self, screen: pygame.Surface, metrics: UIMetrics) -> None:
        """Draw statistics section"""
        stats_start_y = self.rect.y + 150
        
        # Section title
        title = self.font_manager.render_text("Simulation Statistics", 'BODY', COLORS['TEXT_PRIMARY'])
        screen.blit(title, (self.rect.x + 15, stats_start_y))
        
        # Two-column statistics layout to avoid being covered by charts
        stats_left = [
            f"Step: {metrics.step}",
            f"Total Agents: {metrics.total_agents}",
            f"Alive Agents: {metrics.alive_agents}",
            f"Avg Sugar: {metrics.avg_sugar:.2f}",
        ]
        stats_right = [
            f"Avg Age: {metrics.avg_age:.1f}",
            f"Total Sugar: {metrics.total_sugar:.0f}",
            f"Diversity: {metrics.diversity:.3f}",
            f"FPS: {metrics.fps:.1f}",
        ]
        
        col_spacing = 170
        row_spacing = 20
        base_y = stats_start_y + 30
        
        # Left column
        for i, stat in enumerate(stats_left):
            text_surface = self.font_manager.render_text(stat, 'SMALL', COLORS['TEXT_PRIMARY'])
            screen.blit(text_surface, (self.rect.x + 20, base_y + i * row_spacing))
        
        # Right column
        for i, stat in enumerate(stats_right):
            text_surface = self.font_manager.render_text(stat, 'SMALL', COLORS['TEXT_PRIMARY'])
            screen.blit(
                text_surface,
                (self.rect.x + 20 + col_spacing, base_y + i * row_spacing),
            )
    
    def _draw_status(self, screen: pygame.Surface, metrics: UIMetrics) -> None:
        """Draw status bar"""
        status_y = self.rect.y + self.rect.height - 35
        
        # Status color
        if metrics.state == 'running':
            status_color = COLORS['SUCCESS']
            status_text = "Running"
        elif metrics.state == 'paused':
            status_color = COLORS['WARNING']
            status_text = "Paused"
        else:
            status_color = COLORS['ERROR']
            status_text = "Stopped"
        
        status_rect = pygame.Rect(self.rect.x + 10, status_y, self.rect.width - 20, 25)
        pygame.draw.rect(screen, status_color, status_rect, border_radius=4)
        pygame.draw.rect(screen, COLORS['PANEL_BORDER'], status_rect, 1, border_radius=4)
        
        text_surface = self.font_manager.render_text(f"Status: {status_text}", 'SMALL', COLORS['WHITE'])
        text_rect = text_surface.get_rect(center=status_rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event: pygame.event.Event, simulation: Any) -> bool:
        """Handle UI events"""
        try:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                
                for name, button in self.buttons.items():
                    if button['rect'].collidepoint(mouse_pos):
                        if name == 'play_pause':
                            if simulation.state.value == "running":
                                simulation.pause()
                                button['text'] = 'Resume'
                            else:
                                simulation.resume()
                                button['text'] = 'Pause'
                            return True
                        elif name == 'reset':
                            simulation.reset()
                            return True
            
            return False
        except Exception as e:
            print(f"Event handling failed: {e}")
            return False


class AcademicVisualizationSystem:
    """Academic Professional Visualization System"""
    
    def __init__(self, screen_width: int, screen_height: int, grid_size: int, cell_size: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Initialize components
        self.simulation_renderer = MARLSimulationRenderer(grid_size, cell_size)
        self.font_manager = AcademicFontManager()
        
        # Calculate panel position
        panel_x = grid_size * cell_size + 10
        panel_width = screen_width - panel_x - 10
        panel_height = screen_height - 20
        self.control_panel = AcademicControlPanel(panel_x, 10, panel_width, panel_height)

        # 视图系统：Overview / Training / Behavior / Debug
        self.views = ["overview", "training", "behavior", "debug"]
        self.active_view = "training"  # 默认以训练视图为主
        self.view_tabs = []  # List[Tuple[view_id, rect]]
        self.view_tab_rects: Dict[str, pygame.Rect] = {}

        # 统一的训练图表配置（性能相关）
        self.training_chart_max_points = 200
        self.training_chart_update_freq = 5
        self.exploration_chart_update_freq = 10

        # Tab 相关尺寸（用于视图与图表区域的分区）
        self.view_tab_height = 22
        self.view_tab_margin = 6
        
        # Layout constants inside panel
        # Reserve top area of the panel for title, buttons and statistics
        self.panel_top_reserved = 260  # px from top of panel
        
        # 初始化视图 Tab
        self._initialize_view_tabs(panel_x)

        # 图表区域顶部：在统计区和 Tab 之下，避免任何遮挡
        self.charts_top = (
            self.control_panel.rect.y
            + self.panel_top_reserved
            + self.view_tab_height
            + self.view_tab_margin
        )

        # Initialize charts with a vertical stack layout
        # 使用MultiLineChart统一图表显示方式（当前常规图表在UI中隐藏）
        self.charts = self._initialize_charts(panel_x, panel_width)
        
        # Initialize training metrics charts
        self.training_charts = self._initialize_training_charts(panel_x, panel_width)
        
        # Behavior 视图：奖励趋势 & 策略熵曲线图
        behavior_chart_area_x = panel_x + 15
        behavior_chart_width = panel_width - 30
        behavior_chart_height = 100
        behavior_row_spacing = 8
        behavior_slots = self._build_chart_grid(
            start_x=behavior_chart_area_x,
            start_y=self.charts_top,
            total_width=behavior_chart_width,
            rows=2,
            cols=1,
            chart_height=behavior_chart_height,
            row_spacing=behavior_row_spacing,
            col_spacing=0,
        )
        self.behavior_charts: Dict[str, MultiLineChart] = {}
        if behavior_slots:
            reward_rect = behavior_slots[0]
            self.behavior_charts["reward_trend"] = MultiLineChart(
                reward_rect.x,
                reward_rect.y,
                reward_rect.width,
                reward_rect.height,
                "Reward Trend",
                "Avg / Recent Reward",
                self.font_manager,
                max_points=self.training_chart_max_points,
                update_frequency=self.training_chart_update_freq,
                show_legend=True,
                show_points=False,
                line_width=2,
            )
            self.behavior_charts["reward_trend"].add_line("IQL", color=COLORS["AGENT_IQL"])
            self.behavior_charts["reward_trend"].add_line("QMIX", color=COLORS["AGENT_QMIX"])
        if len(behavior_slots) >= 2:
            entropy_rect = behavior_slots[1]
            self.behavior_charts["policy_entropy"] = MultiLineChart(
                entropy_rect.x,
                entropy_rect.y,
                entropy_rect.width,
                entropy_rect.height,
                "Policy Entropy",
                "Entropy (bits)",
                self.font_manager,
                max_points=self.training_chart_max_points,
                update_frequency=self.training_chart_update_freq,
                show_legend=True,
                show_points=False,
                line_width=2,
            )
            self.behavior_charts["policy_entropy"].add_line("IQL", color=COLORS["AGENT_IQL"])
            self.behavior_charts["policy_entropy"].add_line("QMIX", color=COLORS["AGENT_QMIX"])
        
        # Combined agent-distribution panel 默认位置（实际绘制时不再依赖精确 rect.y）
        dist_y = self.control_panel.rect.y + self.panel_top_reserved + 4 * 140
        self.agent_distribution_panel = AgentDistributionPanel(
            panel_x + 15,
            dist_y,
            panel_width - 30,
            150,
            self.font_manager,
        )

        # 行为视图：动作分布面板（位于图表区域内，宽度与训练图表一致）
        self.action_distribution_panel = ActionDistributionPanel(
            panel_x + 15,
            self.charts_top,
            panel_width - 30,
            220,
            self.font_manager,
        )
        
        # Q值热图面板（叠加在环境地图上）
        self.q_value_heatmap = QValueHeatmapPanel(
            0, 0, 0, 0, self.font_manager  # 位置和大小在draw时动态确定
        )
        
        # Q值热图开关按钮（在Behavior视图的图例区域）
        self.q_heatmap_toggle_rect: Optional[pygame.Rect] = None
        
        # 网络状态面板（在Behavior视图中显示）
        network_state_y = self.charts_top + 220 + 12  # 在动作分布面板下方
        self.network_state_panel = NetworkStatePanel(
            panel_x + 15,
            network_state_y,
            panel_width - 30,
            200,
            self.font_manager,
        )
    
    def get_screen_info(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        return (self.screen_width, self.screen_height)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _build_chart_grid(
        self,
        start_x: int,
        start_y: int,
        total_width: int,
        rows: int,
        cols: int,
        chart_height: int,
        row_spacing: int = 12,
        col_spacing: int = 12,
    ) -> List[pygame.Rect]:
        """
        根据行列数生成图表槽位 Rect 列表（行优先顺序）。

        设计目标：
        - 在给定宽度区域内平均分配列宽，并预留列/行间距。
        - 统一 Training / Behavior 等视图内部的图表布局。
        """
        rects: List[pygame.Rect] = []
        if rows <= 0 or cols <= 0:
            return rects

        # 水平方向：平均分配列宽
        total_col_spacing = col_spacing * (cols - 1) if cols > 1 else 0
        col_width = max(40, (total_width - total_col_spacing) // cols)

        y = start_y
        for _row in range(rows):
            x = start_x
            for _col in range(cols):
                rects.append(pygame.Rect(x, y, col_width, chart_height))
                x += col_width + col_spacing
            y += chart_height + row_spacing

        return rects

    # ------------------------------------------------------------------
    # View Tabs
    # ------------------------------------------------------------------
    def _initialize_view_tabs(self, panel_x: int) -> None:
        """初始化右侧面板顶部的视图 Tab 按钮"""
        tab_labels = {
            "overview": "Overview",
            "training": "Training",
            "behavior": "Behavior",
            "debug": "Debug",
        }
        tab_width = 90
        tab_height = self.view_tab_height
        spacing = 4
        x = panel_x
        # 放在 Simulation Statistics 区域下方、训练图表上方（不遮挡统计或图表）
        y = self.control_panel.rect.y + self.panel_top_reserved + 4

        self.view_tabs.clear()
        self.view_tab_rects.clear()

        for view_id in self.views:
            rect = pygame.Rect(x, y, tab_width, tab_height)
            self.view_tabs.append((view_id, rect))
            self.view_tab_rects[view_id] = rect
            x += tab_width + spacing

    def _draw_view_tabs(self, screen: pygame.Surface) -> None:
        """绘制视图切换 Tab"""
        mouse_pos = pygame.mouse.get_pos()
        for view_id, rect in self.view_tabs:
            is_active = (view_id == self.active_view)

            if is_active:
                bg_color = COLORS['BUTTON_ACTIVE']
                text_color = COLORS['WHITE']
            elif rect.collidepoint(mouse_pos):
                bg_color = COLORS['BUTTON_HOVER']
                text_color = COLORS['TEXT_PRIMARY']
            else:
                bg_color = COLORS['BUTTON_NORMAL']
                text_color = COLORS['TEXT_PRIMARY']

            pygame.draw.rect(screen, bg_color, rect, border_radius=4)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], rect, 1, border_radius=4)

            label = view_id.capitalize()
            text_surface = self.font_manager.render_text(label, 'TINY', text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)
    
    def _initialize_charts(self, panel_x: int, panel_width: int) -> Dict[str, Any]:
        """
        初始化实时图表 - 使用MultiLineChart统一显示方式
        
        注意：常规图表（Population, Average Sugar, Diversity）已隐藏，
        训练图表将直接使用这些图表的位置。
        """
        # 返回空字典，不创建常规图表（已隐藏）
        return {}
    
    def _initialize_training_charts(self, panel_x: int, panel_width: int) -> Dict[str, MultiLineChart]:
        """
        初始化训练指标图表
        
        包括：
        - 损失函数曲线（按智能体类型）
        - Q值趋势（按智能体类型）
        - TD误差（按智能体类型，可选）
        - 探索率（按智能体类型，可选）
        
        注意：训练图表现在直接使用原来常规图表的位置（上移）
        """
        # Training 视图采用 2×2 网格布局：
        # ┌───────┬───────┐
        # │ Loss  │ Q     │
        # ├───────┼───────┤
        # │ TD    │ Eps   │
        # └───────┴───────┘
        chart_area_x = panel_x + 15
        chart_area_width = panel_width - 30
        chart_height = 120
        row_spacing = 12
        col_spacing = 16
        # 图表区域顶部由 charts_top 统一控制，避免与 Tab/统计信息重叠
        base_y = self.charts_top

        slots = self._build_chart_grid(
            start_x=chart_area_x,
            start_y=base_y,
            total_width=chart_area_width,
            rows=2,
            cols=2,
            chart_height=chart_height,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
        )

        charts: Dict[str, MultiLineChart] = {}
        if len(slots) < 4:
            # 安全防护：退回到单列布局
            slots = [
                pygame.Rect(chart_area_x, base_y + i * (chart_height + row_spacing),
                            chart_area_width, chart_height)
                for i in range(4)
            ]

        # 损失函数图表（左上）
        loss_rect = slots[0]
        charts['loss'] = MultiLineChart(
            loss_rect.x,
            loss_rect.y,
            loss_rect.width,
            loss_rect.height,
            "Training Loss",
            "Loss",
            self.font_manager,
            max_points=self.training_chart_max_points,
            update_frequency=self.training_chart_update_freq,
            show_legend=True,
            show_points=False,
            line_width=2
        )
        # 预添加常见的智能体类型曲线
        charts['loss'].add_line("IQL", color=COLORS['AGENT_IQL'])
        charts['loss'].add_line("QMIX", color=COLORS['AGENT_QMIX'])
        
        # Q值趋势图表（右上）
        q_rect = slots[1]
        charts['q_value'] = MultiLineChart(
            q_rect.x,
            q_rect.y,
            q_rect.width,
            q_rect.height,
            "Q-Value Trend",
            "Q Value",
            self.font_manager,
            max_points=self.training_chart_max_points,
            update_frequency=self.training_chart_update_freq,
            show_legend=True,
            show_points=False,
            line_width=2
        )
        charts['q_value'].add_line("IQL", color=COLORS['AGENT_IQL'])
        charts['q_value'].add_line("QMIX", color=COLORS['AGENT_QMIX'])
        
        # TD误差图表（左下，可选）
        td_rect = slots[2]
        charts['td_error'] = MultiLineChart(
            td_rect.x,
            td_rect.y,
            td_rect.width,
            td_rect.height,
            "TD Error",
            "TD Error",
            self.font_manager,
            max_points=self.training_chart_max_points,
            update_frequency=self.training_chart_update_freq,
            show_legend=True,
            show_points=False,
            line_width=2
        )
        charts['td_error'].add_line("IQL", color=COLORS['AGENT_IQL'])
        charts['td_error'].add_line("QMIX", color=COLORS['AGENT_QMIX'])
        # 默认隐藏TD误差图表（可通过配置启用）
        # charts['td_error'].set_line_visible("IQL", False)
        # charts['td_error'].set_line_visible("QMIX", False)
        
        # 探索率图表（右下，可选）
        eps_rect = slots[3]
        charts['exploration_rate'] = MultiLineChart(
            eps_rect.x,
            eps_rect.y,
            eps_rect.width,
            eps_rect.height,
            "Exploration Rate",
            "Epsilon",
            self.font_manager,
            max_points=self.training_chart_max_points,
            update_frequency=self.exploration_chart_update_freq,  # 探索率变化较慢，更新频率可以更低
            show_legend=True,
            show_points=False,
            line_width=2
        )
        charts['exploration_rate'].add_line("IQL", color=COLORS['AGENT_IQL'])
        charts['exploration_rate'].add_line("QMIX", color=COLORS['AGENT_QMIX'])
        
        return charts
    
    def draw(self, screen: pygame.Surface, simulation_data: Dict[str, Any]) -> None:
        """Draw entire visualization system"""
        try:
            # Clear screen with academic background
            screen.fill(COLORS['BACKGROUND'])
            
            # Draw environment - ensure we get the actual sugar_map
            env_x, env_y = 0, 0
            if 'environment' in simulation_data:
                env_data = simulation_data['environment']
                sugar_map = env_data.get('sugar_map')
                if sugar_map is not None:
                    # Ensure it's a numpy array
                    if not isinstance(sugar_map, np.ndarray):
                        sugar_map = np.array(sugar_map)
                    self.simulation_renderer.draw_environment(screen, sugar_map)
                else:
                    # Fallback: create empty map
                    grid_size = env_data.get('grid_size', self.grid_size)
                    sugar_map = np.zeros((grid_size, grid_size))
                    self.simulation_renderer.draw_environment(screen, sugar_map)
            
            # Draw agents - ensure we have valid agent data
            if 'agents' in simulation_data:
                agents = simulation_data['agents']
                if agents and len(agents) > 0:
                    self.simulation_renderer.draw_agents(screen, agents)
            
            # Update and draw Q-value heatmap (overlay on environment) - only in behavior view
            if self.active_view == "behavior":
                simulation = simulation_data.get('simulation')
                if simulation is not None:
                    # Group agents by type
                    agents_by_type: Dict[str, List[Any]] = {}
                    for agent in simulation.agent_manager.agents:
                        agent_type = agent.agent_type.value
                        if agent_type not in agents_by_type:
                            agents_by_type[agent_type] = []
                        agents_by_type[agent_type].append(agent)
                    
                    environment = simulation.environment
                    if environment is not None:
                        # 更新Q值热图数据（即使未启用也更新，以便快速响应开关）
                        if self.q_value_heatmap.enabled:
                            self.q_value_heatmap.update(agents_by_type, environment)
                        # 绘制Q值热图和开关按钮（开关按钮始终显示）
                        self.q_value_heatmap.draw(screen, self.grid_size, self.cell_size, env_x, env_y)
            
            # Prepare UI metrics
            ui_metrics = self._prepare_ui_metrics(simulation_data)
            
            # Draw control panel（基础统计）
            self.control_panel.draw(screen, ui_metrics)

            # 绘制视图 Tab
            self._draw_view_tabs(screen)

            # 根据当前视图绘制对应内容
            if self.active_view == "training":
                # 训练视图：Loss / Q / TD / Exploration + 类型分布
                self._update_training_charts(simulation_data)
                for chart in self.training_charts.values():
                    chart.draw(screen)

                # 将 Agent 分布面板紧贴在训练图表下方（仍然属于“图表区域”之内）
                if 'metrics' in simulation_data and self.training_charts:
                    last_training_chart = max(self.training_charts.values(), key=lambda c: c.rect.bottom)
                    self.agent_distribution_panel.rect.y = last_training_chart.rect.bottom + 10
                    available_height = self.control_panel.rect.bottom - self.agent_distribution_panel.rect.y - 20
                    self.agent_distribution_panel.rect.height = max(140, min(180, available_height))

                    metrics = simulation_data['metrics']
                    agents_by_type = metrics.get('agents_by_type', {})
                    avg_sugar_by_type = metrics.get('avg_sugar_by_type', {})
                    self.agent_distribution_panel.draw(screen, agents_by_type, avg_sugar_by_type)

            elif self.active_view == "overview":
                # 概览视图：在“图表区域”顶部展示类型分布
                if 'metrics' in simulation_data:
                    top_y = self.charts_top
                    self.agent_distribution_panel.rect.y = top_y
                    available_height = self.control_panel.rect.bottom - top_y - 20
                    self.agent_distribution_panel.rect.height = max(140, min(200, available_height))

                    metrics = simulation_data['metrics']
                    agents_by_type = metrics.get('agents_by_type', {})
                    avg_sugar_by_type = metrics.get('avg_sugar_by_type', {})
                    self.agent_distribution_panel.draw(screen, agents_by_type, avg_sugar_by_type)

            elif self.active_view == "behavior":
                # 行为视图：Reward Trend + Policy Entropy + 动作分布 + 网络状态
                self._update_behavior_charts(simulation_data)

                last_bottom = self.charts_top
                for chart_id in ["reward_trend", "policy_entropy"]:
                    chart = self.behavior_charts.get(chart_id)
                    if chart is not None:
                        chart.draw(screen)
                        last_bottom = max(last_bottom, chart.rect.bottom)

                if 'metrics' in simulation_data:
                    metrics = simulation_data['metrics']
                    action_dist = metrics.get('action_distribution_by_type', {})

                    # 将动作分布面板放在行为图表下方
                    self.action_distribution_panel.rect.y = last_bottom + 10
                    available_height = self.control_panel.rect.bottom - self.action_distribution_panel.rect.y - 20
                    self.action_distribution_panel.rect.height = max(140, min(220, available_height))

                    self.action_distribution_panel.draw(screen, action_dist)
                    
                    # 更新并绘制网络状态面板（在动作分布面板下方）
                    # 获取智能体对象（需要从simulation中获取）
                    simulation = simulation_data.get('simulation')
                    if simulation is not None:
                        agents_by_type = {}
                        for agent in simulation.agent_manager.agents:
                            agent_type = agent.agent_type.value
                            if agent_type not in agents_by_type:
                                agents_by_type[agent_type] = []
                            agents_by_type[agent_type].append(agent)
                        
                        environment = simulation.environment
                        if environment is not None:
                            self.network_state_panel.update(agents_by_type, environment)
                    
                    self.network_state_panel.rect.y = self.action_distribution_panel.rect.bottom + 10
                    available_height = self.control_panel.rect.bottom - self.network_state_panel.rect.y - 20
                    self.network_state_panel.rect.height = max(150, min(200, available_height))
                    self.network_state_panel.draw(screen)

            elif self.active_view == "debug":
                # 调试视图：展示简单的性能信息，从 charts_top 开始绘制
                if 'performance' in simulation_data:
                    perf = simulation_data['performance']
                    debug_lines = [
                        f"FPS: {perf.get('fps', 0.0):.1f}",
                        f"Step Time (ms): {perf.get('step_time', 0.0):.2f}",
                        f"Agent Update Time (ms): {perf.get('agent_update_time', 0.0):.2f}",
                        f"Memory (est. MB): {perf.get('memory_usage_mb', 0.0):.1f}",
                    ]
                    base_y = self.charts_top
                    for i, line in enumerate(debug_lines):
                        text = self.font_manager.render_text(line, 'SMALL', COLORS['TEXT_PRIMARY'])
                        screen.blit(text, (self.control_panel.rect.x + 20, base_y + i * 22))
            
        except Exception as e:
            print(f"Visualization system drawing failed: {e}")
    
    def _prepare_ui_metrics(self, simulation_data: Dict[str, Any]) -> UIMetrics:
        """Prepare UI metrics from simulation data"""
        metrics = UIMetrics()
        
        metrics.step = simulation_data.get('step_count', 0)
        metrics.state = simulation_data.get('state', 'unknown')
        
        if 'metrics' in simulation_data:
            sim_metrics = simulation_data['metrics']
            metrics.total_agents = sim_metrics.get('total_agents', 0)
            metrics.alive_agents = sim_metrics.get('alive_agents', 0)
            metrics.avg_sugar = sim_metrics.get('avg_sugar', 0.0)
            metrics.avg_age = sim_metrics.get('avg_age', 0.0)
            metrics.total_sugar = sim_metrics.get('total_environment_sugar', 0.0)
            metrics.diversity = sim_metrics.get('agent_diversity', 0.0)
            metrics.agents_by_type = sim_metrics.get('agents_by_type', {})
            metrics.avg_sugar_by_type = sim_metrics.get('avg_sugar_by_type', {})
        
        if 'performance' in simulation_data:
            metrics.performance = simulation_data['performance']
            metrics.fps = simulation_data['performance'].get('fps', 0.0)
        
        return metrics
    
    def _update_charts(self, metrics: UIMetrics, simulation_data: Optional[Dict[str, Any]] = None) -> None:
        """
        更新常规图表数据
        
        使用MultiLineChart的统一接口，但保持单线显示（每个图表只有一条曲线）
        """
        # 更新人口图表
        if 'population' in self.charts:
            self.charts['population'].add_data_point_conditional('population', metrics.total_agents, metrics.step)
        
        # 更新平均糖量图表
        if 'avg_sugar' in self.charts:
            self.charts['avg_sugar'].add_data_point_conditional('avg_sugar', metrics.avg_sugar, metrics.step)
        
        # 更新多样性图表
        if 'diversity' in self.charts:
            self.charts['diversity'].add_data_point_conditional('diversity', metrics.diversity, metrics.step)
    
    def _update_training_charts(self, simulation_data: Dict[str, Any]) -> None:
        """
        更新训练指标图表数据
        
        从simulation_data中获取training_metrics并更新相应的图表
        """
        training_metrics = simulation_data.get('training_metrics', {})
        step = simulation_data.get('step_count', 0)
        
        if not training_metrics:
            return  # 没有训练数据时跳过
        
        # 颜色和标签映射：确保不同算法类型使用固定、可区分的颜色和标签
        label_map = {
            'iql': 'IQL',
            'qmix': 'QMIX',
            'independent_q_learning': 'IQL',
            'rule_based': 'Rule-Based',
        }
        color_map = {
            'iql': COLORS.get('AGENT_IQL', COLORS['CHART_LINE_2']),
            'independent_q_learning': COLORS.get('AGENT_IQL', COLORS['CHART_LINE_2']),
            'qmix': COLORS.get('AGENT_QMIX', COLORS['CHART_LINE_3']),
            'rule_based': COLORS.get('AGENT_RULE_BASED', COLORS['CHART_LINE_1']),
        }

        # 更新损失函数图表
        if 'loss' in self.training_charts:
            for agent_type, metrics in training_metrics.items():
                # 统一标签：例如 'iql' -> 'IQL'
                agent_label = label_map.get(agent_type, agent_type.upper())
                
                # 确保曲线存在（动态添加）
                if agent_label not in self.training_charts['loss'].lines:
                    # 根据智能体类型选择固定颜色，默认退回学术配色
                    color = color_map.get(agent_type, COLORS['CHART_LINE_1'])
                    self.training_charts['loss'].add_line(agent_label, color=color)
                
                # 只要存在有效值就更新（允许为0），由MultiLineChart内部负责NaN/Inf过滤
                if 'recent_loss' in metrics:
                    self.training_charts['loss'].add_data_point_conditional(
                        agent_label,
                        metrics['recent_loss'],
                        step
                    )
        
        # 更新Q值趋势图表
        if 'q_value' in self.training_charts:
            for agent_type, metrics in training_metrics.items():
                agent_label = label_map.get(agent_type, agent_type.upper())
                
                # 确保曲线存在
                if agent_label not in self.training_charts['q_value'].lines:
                    color = color_map.get(agent_type, COLORS['CHART_LINE_1'])
                    self.training_charts['q_value'].add_line(agent_label, color=color)
                
                if 'recent_q_value' in metrics:
                    self.training_charts['q_value'].add_data_point_conditional(
                        agent_label,
                        metrics['recent_q_value'],
                        step
                    )
        
        # 更新TD误差图表
        if 'td_error' in self.training_charts:
            for agent_type, metrics in training_metrics.items():
                agent_label = label_map.get(agent_type, agent_type.upper())
                
                # 确保曲线存在
                if agent_label not in self.training_charts['td_error'].lines:
                    color = color_map.get(agent_type, COLORS['CHART_LINE_1'])
                    self.training_charts['td_error'].add_line(agent_label, color=color)
                
                if 'avg_td_error' in metrics:
                    self.training_charts['td_error'].add_data_point_conditional(
                        agent_label,
                        metrics['avg_td_error'],
                        step
                    )
        
        # 更新探索率图表
        if 'exploration_rate' in self.training_charts:
            for agent_type, metrics in training_metrics.items():
                agent_label = label_map.get(agent_type, agent_type.upper())
                
                # 确保曲线存在
                if agent_label not in self.training_charts['exploration_rate'].lines:
                    color = color_map.get(agent_type, COLORS['CHART_LINE_1'])
                    self.training_charts['exploration_rate'].add_line(agent_label, color=color)
                
                if 'exploration_rate' in metrics:
                    self.training_charts['exploration_rate'].add_data_point_conditional(
                        agent_label,
                        metrics['exploration_rate'],
                        step
                    )
    
    def _update_behavior_charts(self, simulation_data: Dict[str, Any]) -> None:
        """
        更新行为视图相关图表：
        - Reward Trend：基于训练指标中的 avg_reward / recent_reward
        - Policy Entropy：基于当前动作分布计算离散策略熵
        """
        if not self.behavior_charts:
            return

        step = simulation_data.get("step_count", 0)
        training_metrics = simulation_data.get("training_metrics", {}) or {}
        metrics_block = simulation_data.get("metrics", {}) or {}
        action_dist = metrics_block.get("action_distribution_by_type", {}) or {}

        label_map = {
            "iql": "IQL",
            "independent_q_learning": "IQL",
            "qmix": "QMIX",
            "rule_based": "Rule-Based",
        }
        color_map = {
            "iql": COLORS.get("AGENT_IQL", COLORS["CHART_LINE_2"]),
            "independent_q_learning": COLORS.get("AGENT_IQL", COLORS["CHART_LINE_2"]),
            "qmix": COLORS.get("AGENT_QMIX", COLORS["CHART_LINE_3"]),
            "rule_based": COLORS.get("AGENT_RULE_BASED", COLORS["CHART_LINE_1"]),
        }

        # 1) 奖励趋势：使用 recent_reward（如无则退回 avg_reward）
        reward_chart = self.behavior_charts.get("reward_trend")
        if reward_chart and training_metrics:
            for agent_type, metrics in training_metrics.items():
                agent_label = label_map.get(agent_type, agent_type.upper())
                if agent_label not in reward_chart.lines:
                    color = color_map.get(agent_type, COLORS["CHART_LINE_1"])
                    reward_chart.add_line(agent_label, color=color)

                value = None
                if "recent_reward" in metrics:
                    value = metrics["recent_reward"]
                elif "avg_reward" in metrics:
                    value = metrics["avg_reward"]

                if value is not None:
                    reward_chart.add_data_point_conditional(agent_label, value, step)

        # 2) 策略熵：基于当前动作分布计算 H(a | type)
        entropy_chart = self.behavior_charts.get("policy_entropy")
        if entropy_chart and action_dist:
            for agent_type, counts in action_dist.items():
                if not counts:
                    continue
                total = sum(int(c) for c in counts.values())
                if total <= 0:
                    continue

                probs = []
                for c in counts.values():
                    c = int(c)
                    if c > 0:
                        probs.append(c / total)
                if not probs:
                    continue

                entropy = 0.0
                for p in probs:
                    entropy -= p * math.log(p + 1e-12, 2)

                agent_label = label_map.get(agent_type, agent_type.upper())
                if agent_label not in entropy_chart.lines:
                    color = color_map.get(agent_type, COLORS["CHART_LINE_1"])
                    entropy_chart.add_line(agent_label, color=color)

                entropy_chart.add_data_point_conditional(agent_label, entropy, step)
    
    def handle_event(self, event: pygame.event.Event, simulation: Any) -> bool:
        """Handle events（视图 Tab 点击 + 控制面板按钮 + Q值热图开关）"""
        try:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                
                # 先处理视图 Tab 点击
                for view_id, rect in self.view_tabs:
                    if rect.collidepoint(mouse_pos):
                        self.active_view = view_id
                        return True
                
                # 处理Q值热图开关（仅在Behavior视图）
                if self.active_view == "behavior":
                    if hasattr(self.q_value_heatmap, 'toggle_rect') and self.q_value_heatmap.toggle_rect is not None:
                        if self.q_value_heatmap.toggle_rect.collidepoint(mouse_pos):
                            self.q_value_heatmap.enabled = not self.q_value_heatmap.enabled
                            # 如果关闭，清除缓存
                            if not self.q_value_heatmap.enabled:
                                self.q_value_heatmap.cached_overlay = None
                                self.q_value_heatmap.q_maps.clear()
                            return True
                
            # 未命中 Tab 和开关，则交给控制面板处理
            return self.control_panel.handle_event(event, simulation)
        except Exception as e:
            print(f"Visualization system event handling failed: {e}")
            return False


# Backward compatibility aliases
ModernVisualizationSystem = AcademicVisualizationSystem
ModernFontManager = AcademicFontManager
ModernControlPanel = AcademicControlPanel
MARLSimulationRenderer = MARLSimulationRenderer
