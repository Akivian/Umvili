"""
Application Configuration

Application-level configuration including runtime settings, logging, and general preferences.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config.defaults import DEFAULT_APP_CONFIG


class ApplicationState(Enum):
    """应用程序状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class ApplicationConfig:
    """
    应用程序配置数据类
    
    包含应用程序级别的配置，如模拟类型、日志设置、UI显示选项等。
    """
    # 模拟类型配置
    simulation_type: str = "comparative"
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = "marl_simulation.log"
    
    # UI显示配置
    show_fps: bool = True
    show_debug_info: bool = False
    window_title: str = "MARL沙盘平台 - 多智能体强化学习模拟"
    
    # 性能配置
    enable_vsync: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ApplicationConfig':
        """
        从字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置对象
        """
        # 只使用有效的字段
        valid_fields = {k: v for k, v in config_dict.items() 
                       if k in cls.__dataclass_fields__}
        return cls(**valid_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {field.name: getattr(self, field.name) 
                for field in self.__dataclass_fields__.values()}
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        验证配置有效性
        
        Returns:
            (是否有效, 错误消息)
        """
        valid_simulation_types = ['default', 'comparative', 'training', 'performance']
        if self.simulation_type not in valid_simulation_types:
            return False, f"模拟类型必须是 {valid_simulation_types} 之一，当前值: {self.simulation_type}"
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            return False, f"日志级别必须是 {valid_log_levels} 之一，当前值: {self.log_level}"
        
        return True, None
    
    @classmethod
    def default(cls) -> 'ApplicationConfig':
        """
        创建默认配置
        
        Returns:
            默认配置对象
        """
        return cls.from_dict(DEFAULT_APP_CONFIG)
    
    def merge(self, other: Dict[str, Any]) -> 'ApplicationConfig':
        """
        合并其他配置
        
        Args:
            other: 要合并的配置字典
            
        Returns:
            合并后的新配置对象
        """
        current_dict = self.to_dict()
        current_dict.update(other)
        return self.from_dict(current_dict)

