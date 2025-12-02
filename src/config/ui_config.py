"""
UI Configuration

Configuration for user interface, visualization, and display settings.
"""

from typing import Dict, Tuple, Any
from dataclasses import dataclass, field

from src.config.defaults import DEFAULT_UI_CONFIG


# ============================================================================
# Color Scheme
# ============================================================================

class ColorScheme:
    """
    颜色方案类
    
    集中管理所有UI颜色定义，采用学术风格配色方案。
    """
    
    # Base Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (220, 220, 220)
    DARK_GRAY = (80, 80, 80)
    
    # Background Colors - Academic Paper Style
    BACKGROUND = (250, 250, 250)  # Off-white paper background
    PANEL_BG = (255, 255, 255)     # Pure white panels
    PANEL_BORDER = (200, 200, 200) # Subtle gray borders
    PANEL_SHADOW = (240, 240, 240) # Subtle shadow
    
    # Sugar Environment - Sequential Color Scheme (Green)
    SUGAR_LOW = (237, 248, 233)    # Very light green
    SUGAR_MEDIUM = (186, 228, 179) # Medium green
    SUGAR_HIGH = (116, 196, 118)   # Dark green
    SUGAR_MAX = (35, 139, 69)      # Very dark green
    
    # Agent Type Colors - Categorical Color Scheme (Colorblind-friendly)
    # Based on Tableau and ColorBrewer palettes
    AGENT_RULE_BASED = (31, 119, 180)      # Blue
    AGENT_IQL = (255, 127, 14)             # Orange
    AGENT_QMIX = (44, 160, 44)            # Green
    AGENT_CONSERVATIVE = (214, 39, 40)     # Red
    AGENT_EXPLORATORY = (148, 103, 189)   # Purple
    AGENT_ADAPTIVE = (140, 86, 75)        # Brown
    
    # Agent Status Colors - Sequential
    AGENT_POOR = (255, 237, 213)          # Light orange
    AGENT_MEDIUM = (255, 193, 7)          # Amber
    AGENT_RICH = (40, 167, 69)            # Success green
    AGENT_WEALTHY = (0, 123, 255)         # Primary blue
    
    # UI Elements - Minimalist Academic Style
    BUTTON_NORMAL = (245, 245, 245)       # Light gray
    BUTTON_HOVER = (230, 230, 230)        # Medium gray
    BUTTON_ACTIVE = (0, 123, 255)         # Blue
    BUTTON_TEXT = (33, 37, 41)            # Dark text
    TEXT_PRIMARY = (33, 37, 41)           # Dark gray (almost black)
    TEXT_SECONDARY = (108, 117, 125)      # Medium gray
    TEXT_MUTED = (173, 181, 189)          # Light gray
    
    # Status Colors - Semantic
    SUCCESS = (40, 167, 69)               # Green
    WARNING = (255, 193, 7)               # Amber
    ERROR = (220, 53, 69)                 # Red
    INFO = (0, 123, 255)                  # Blue
    
    # Chart Colors - Academic Style
    CHART_GRID = (233, 236, 239)          # Very light gray
    CHART_AXIS = (108, 117, 125)          # Medium gray
    CHART_LINE_1 = (31, 119, 180)         # Blue
    CHART_LINE_2 = (255, 127, 14)         # Orange
    CHART_LINE_3 = (44, 160, 44)         # Green
    CHART_LINE_4 = (214, 39, 40)         # Red
    CHART_LINE_5 = (148, 103, 189)        # Purple
    CHART_BG = (255, 255, 255)            # White
    CHART_AREA_ALPHA = 0.3                # For area charts
    
    # Legend and Labels
    LEGEND_BG = (255, 255, 255)
    LEGEND_BORDER = (200, 200, 200)
    LABEL_BG = (255, 255, 255)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Tuple[int, int, int]]:
        """
        转换为字典格式（用于向后兼容）
        
        Returns:
            颜色字典
        """
        return {
            'BLACK': cls.BLACK,
            'WHITE': cls.WHITE,
            'GRAY': cls.GRAY,
            'LIGHT_GRAY': cls.LIGHT_GRAY,
            'DARK_GRAY': cls.DARK_GRAY,
            'BACKGROUND': cls.BACKGROUND,
            'PANEL_BG': cls.PANEL_BG,
            'PANEL_BORDER': cls.PANEL_BORDER,
            'PANEL_SHADOW': cls.PANEL_SHADOW,
            'SUGAR_LOW': cls.SUGAR_LOW,
            'SUGAR_MEDIUM': cls.SUGAR_MEDIUM,
            'SUGAR_HIGH': cls.SUGAR_HIGH,
            'SUGAR_MAX': cls.SUGAR_MAX,
            'AGENT_RULE_BASED': cls.AGENT_RULE_BASED,
            'AGENT_IQL': cls.AGENT_IQL,
            'AGENT_QMIX': cls.AGENT_QMIX,
            'AGENT_CONSERVATIVE': cls.AGENT_CONSERVATIVE,
            'AGENT_EXPLORATORY': cls.AGENT_EXPLORATORY,
            'AGENT_ADAPTIVE': cls.AGENT_ADAPTIVE,
            'AGENT_POOR': cls.AGENT_POOR,
            'AGENT_MEDIUM': cls.AGENT_MEDIUM,
            'AGENT_RICH': cls.AGENT_RICH,
            'AGENT_WEALTHY': cls.AGENT_WEALTHY,
            'BUTTON_NORMAL': cls.BUTTON_NORMAL,
            'BUTTON_HOVER': cls.BUTTON_HOVER,
            'BUTTON_ACTIVE': cls.BUTTON_ACTIVE,
            'BUTTON_TEXT': cls.BUTTON_TEXT,
            'TEXT_PRIMARY': cls.TEXT_PRIMARY,
            'TEXT_SECONDARY': cls.TEXT_SECONDARY,
            'TEXT_MUTED': cls.TEXT_MUTED,
            'SUCCESS': cls.SUCCESS,
            'WARNING': cls.WARNING,
            'ERROR': cls.ERROR,
            'INFO': cls.INFO,
            'CHART_GRID': cls.CHART_GRID,
            'CHART_AXIS': cls.CHART_AXIS,
            'CHART_LINE_1': cls.CHART_LINE_1,
            'CHART_LINE_2': cls.CHART_LINE_2,
            'CHART_LINE_3': cls.CHART_LINE_3,
            'CHART_LINE_4': cls.CHART_LINE_4,
            'CHART_LINE_5': cls.CHART_LINE_5,
            'CHART_BG': cls.CHART_BG,
            'CHART_AREA_ALPHA': cls.CHART_AREA_ALPHA,
            'LEGEND_BG': cls.LEGEND_BG,
            'LEGEND_BORDER': cls.LEGEND_BORDER,
            'LABEL_BG': cls.LABEL_BG,
        }


@dataclass
class FontConfig:
    """字体配置"""
    TITLE: int = 24
    HEADING: int = 20
    BODY: int = 16
    SMALL: int = 14
    TINY: int = 12
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return {
            'TITLE': self.TITLE,
            'HEADING': self.HEADING,
            'BODY': self.BODY,
            'SMALL': self.SMALL,
            'TINY': self.TINY,
        }


@dataclass
class WindowConfig:
    """窗口配置"""
    width: int = 1400
    height: int = 900
    fps: int = 60
    title: str = "MARL沙盘平台 - 多智能体强化学习模拟"
    enable_vsync: bool = False
    
    def validate(self) -> tuple[bool, str | None]:
        """验证窗口配置"""
        if self.width < 400 or self.width > 3840:
            return False, f"窗口宽度必须在400-3840之间，当前值: {self.width}"
        if self.height < 300 or self.height > 2160:
            return False, f"窗口高度必须在300-2160之间，当前值: {self.height}"
        if self.fps < 1 or self.fps > 120:
            return False, f"FPS必须在1-120之间，当前值: {self.fps}"
        return True, None


@dataclass
class UIConfig:
    """
    UI配置数据类
    
    包含所有UI和可视化相关的配置。
    """
    window: WindowConfig = field(default_factory=WindowConfig)
    font: FontConfig = field(default_factory=FontConfig)
    color_scheme: ColorScheme = field(default_factory=ColorScheme)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UIConfig':
        """从字典创建配置对象"""
        window_config = WindowConfig(**config_dict.get('window', {}))
        font_config = FontConfig(**config_dict.get('font', {}))
        
        return cls(
            window=window_config,
            font=font_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'window': {
                'width': self.window.width,
                'height': self.window.height,
                'fps': self.window.fps,
                'title': self.window.title,
                'enable_vsync': self.window.enable_vsync,
            },
            'font': self.font.to_dict(),
            'colors': self.color_scheme.to_dict(),
        }
    
    @classmethod
    def default(cls) -> 'UIConfig':
        """创建默认配置"""
        return cls.from_dict(DEFAULT_UI_CONFIG)
    
    def validate(self) -> tuple[bool, str | None]:
        """验证UI配置"""
        return self.window.validate()


# 向后兼容：导出COLORS字典
COLORS = ColorScheme.to_dict()

# 向后兼容：导出FONT_SIZES
FONT_SIZES = FontConfig().to_dict()

# 向后兼容：导出窗口常量
# 从嵌套的window字典中获取值，支持新旧两种格式
if 'window' in DEFAULT_UI_CONFIG:
    WINDOW_WIDTH = DEFAULT_UI_CONFIG['window'].get('width', 1400)
    WINDOW_HEIGHT = DEFAULT_UI_CONFIG['window'].get('height', 900)
    FPS = DEFAULT_UI_CONFIG['window'].get('fps', 60)
else:
    # 旧格式兼容
    WINDOW_WIDTH = DEFAULT_UI_CONFIG.get('window_width', 1400)
    WINDOW_HEIGHT = DEFAULT_UI_CONFIG.get('window_height', 900)
    FPS = DEFAULT_UI_CONFIG.get('fps', 60)

# 注意：GRID_SIZE和CELL_SIZE应该从SimulationConfig获取，这里仅为向后兼容
GRID_SIZE = 80
CELL_SIZE = 10

