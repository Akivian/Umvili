"""
Simulation Configuration

Configuration for simulation environment and core simulation parameters.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.config.defaults import DEFAULT_SIMULATION_CONFIG


@dataclass
class EnvironmentConfig:
    """
    环境配置
    
    定义糖环境的基本参数。
    """
    size: int = 80
    growth_rate: float = 0.1
    max_sugar: float = 10.0
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """验证环境配置"""
        if self.size < 10 or self.size > 200:
            return False, f"环境大小必须在10-200之间，当前值: {self.size}"
        
        if self.growth_rate < 0 or self.growth_rate > 1:
            return False, f"糖生长速率必须在0-1之间，当前值: {self.growth_rate}"
        
        if self.max_sugar < 1 or self.max_sugar > 100:
            return False, f"最大糖量必须在1-100之间，当前值: {self.max_sugar}"
        
        return True, None


@dataclass
class SimulationConfig:
    """
    模拟配置
    
    包含模拟系统的所有配置参数。
    """
    # 环境配置
    grid_size: int = 80
    cell_size: int = 10
    sugar_growth_rate: float = 0.1
    max_sugar: float = 10.0
    
    # 智能体配置
    initial_agents: int = 50
    
    # 模拟限制
    max_steps: int = 10000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """从字典创建配置对象"""
        valid_fields = {k: v for k, v in config_dict.items() 
                       if k in cls.__dataclass_fields__}
        return cls(**valid_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {field.name: getattr(self, field.name) 
                for field in self.__dataclass_fields__.values()}
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        验证配置有效性
        
        Returns:
            (是否有效, 错误消息)
        """
        if self.grid_size < 10 or self.grid_size > 200:
            return False, f"网格大小必须在10-200之间，当前值: {self.grid_size}"
        
        if self.cell_size < 5 or self.cell_size > 20:
            return False, f"单元格大小必须在5-20之间，当前值: {self.cell_size}"
        
        if self.initial_agents < 1 or self.initial_agents > 1000:
            return False, f"初始智能体数量必须在1-1000之间，当前值: {self.initial_agents}"
        
        if self.sugar_growth_rate < 0 or self.sugar_growth_rate > 1:
            return False, f"糖生长速率必须在0-1之间，当前值: {self.sugar_growth_rate}"
        
        if self.max_sugar < 1 or self.max_sugar > 100:
            return False, f"最大糖量必须在1-100之间，当前值: {self.max_sugar}"
        
        if self.max_steps < 1:
            return False, f"最大步数必须大于0，当前值: {self.max_steps}"
        
        # 额外的业务逻辑验证
        max_recommended_agents = self.grid_size * 2
        if self.initial_agents > max_recommended_agents:
            # 不返回错误，只是警告
            pass
        
        return True, None
    
    @classmethod
    def default(cls) -> 'SimulationConfig':
        """创建默认配置"""
        return cls.from_dict(DEFAULT_SIMULATION_CONFIG)
    
    def get_environment_config(self) -> EnvironmentConfig:
        """
        获取环境配置对象
        
        Returns:
            环境配置对象
        """
        return EnvironmentConfig(
            size=self.grid_size,
            growth_rate=self.sugar_growth_rate,
            max_sugar=self.max_sugar
        )
    
    def merge(self, other: Dict[str, Any]) -> 'SimulationConfig':
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

