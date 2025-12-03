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
    """Combined panel for Agent Types legend and distribution bars."""
    
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
    
    def draw(self, screen: pygame.Surface, data: Dict[str, int]) -> None:
        """Draw combined legend + bar chart."""
        try:
            # Background
            pygame.draw.rect(screen, COLORS['CHART_BG'], self.rect)
            pygame.draw.rect(screen, COLORS['PANEL_BORDER'], self.rect, 1)
            
            # Title
            title_surface = self.font_manager.render_text(
                "Agent Types & Distribution", 'SMALL', COLORS['TEXT_PRIMARY']
            )
            screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
            
            if not data or sum(data.values()) == 0:
                empty_text = self.font_manager.render_text("No data", 'TINY', COLORS['TEXT_MUTED'])
                text_rect = empty_text.get_rect(center=self.rect.center)
                screen.blit(empty_text, text_rect)
                return
            
            sorted_items = sorted(data.items())
            
            # Legend on the left
            legend_x = self.rect.x + self.padding['left']
            legend_y = self.rect.y + self.padding['top']
            legend_line_h = 18
            for i, (agent_type, value) in enumerate(sorted_items):
                if value == 0:
                    continue
                color = self.agent_type_colors.get(agent_type, COLORS['GRAY'])
                label = self.agent_type_labels.get(agent_type, agent_type.title())
                text = f"{label}: {value}"
                
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
            
            max_value = max(data.values())
            num_bars = len(data)
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
        
        # Layout constants inside panel
        # Reserve top area of the panel for title, buttons and statistics
        self.panel_top_reserved = 260  # px from top of panel
        
        # Initialize charts with a vertical stack layout
        self.charts = self._initialize_charts(panel_x, panel_width)
        
        # Combined agent-distribution panel below the last time-series chart
        last_chart = self.charts['diversity']
        dist_y = last_chart.rect.bottom + 10
        self.agent_distribution_panel = AgentDistributionPanel(
            panel_x + 15,
            dist_y,
            panel_width - 30,
            150,
            self.font_manager,
        )
    
    def get_screen_info(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        return (self.screen_width, self.screen_height)
    
    def _initialize_charts(self, panel_x: int, panel_width: int) -> Dict[str, RealTimeChart]:
        """Initialize real-time charts"""
        chart_width = panel_width - 30
        # Slightly taller charts so that Y-axis labels are easier to read
        chart_height = 95
        chart_spacing = 10
        
        charts = {}
        
        # Position charts in a vertical stack below the control-panel reserved area.
        #  - population
        #  - average sugar
        #  - diversity
        base_y = self.control_panel.rect.y + self.panel_top_reserved
        chart_configs = [
            ('population', "Population", "Count", COLORS['CHART_LINE_1'], base_y),
            ('avg_sugar', "Average Sugar", "Sugar", COLORS['CHART_LINE_2'],
             base_y + (chart_height + chart_spacing)),
            ('diversity', "Diversity", "Index", COLORS['CHART_LINE_3'],
             base_y + 2 * (chart_height + chart_spacing)),
        ]
        
        for chart_id, title, y_label, color, y_pos in chart_configs:
            charts[chart_id] = RealTimeChart(
                panel_x + 15,
                y_pos,
                chart_width,
                chart_height,
                title,
                y_label,
                self.font_manager,
                color,
            )
        
        return charts
    
    def draw(self, screen: pygame.Surface, simulation_data: Dict[str, Any]) -> None:
        """Draw entire visualization system"""
        try:
            # Clear screen with academic background
            screen.fill(COLORS['BACKGROUND'])
            
            # Draw environment - ensure we get the actual sugar_map
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
            
            # Prepare UI metrics
            ui_metrics = self._prepare_ui_metrics(simulation_data)
            
            # Draw control panel
            self.control_panel.draw(screen, ui_metrics)
            
            # Update and draw charts
            self._update_charts(ui_metrics)
            for chart in self.charts.values():
                chart.draw(screen)
            
            # Draw combined agent distribution panel
            if 'metrics' in simulation_data:
                agents_by_type = simulation_data['metrics'].get('agents_by_type', {})
                self.agent_distribution_panel.draw(screen, agents_by_type)
            
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
    
    def _update_charts(self, metrics: UIMetrics) -> None:
        """Update chart data"""
        self.charts['population'].add_data_point(metrics.total_agents, metrics.step)
        self.charts['avg_sugar'].add_data_point(metrics.avg_sugar, metrics.step)
        self.charts['diversity'].add_data_point(metrics.diversity, metrics.step)
    
    def handle_event(self, event: pygame.event.Event, simulation: Any) -> bool:
        """Handle events"""
        return self.control_panel.handle_event(event, simulation)


# Backward compatibility aliases
ModernVisualizationSystem = AcademicVisualizationSystem
ModernFontManager = AcademicFontManager
ModernControlPanel = AcademicControlPanel
MARLSimulationRenderer = MARLSimulationRenderer
