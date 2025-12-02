"""
Configuration Loader

Utilities for loading, saving, and merging configurations from various sources.
Supports JSON and YAML formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.config.app_config import ApplicationConfig
from src.config.simulation_config import SimulationConfig
from src.config.ui_config import UIConfig
from src.config.defaults import (
    DEFAULT_APP_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_UI_CONFIG,
    DEFAULT_AGENT_CONFIGS
)


logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    配置加载器
    
    提供统一的配置加载、保存和合并功能。
    """
    
    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径（支持.json和.yaml）
            
        Returns:
            配置字典，如果加载失败返回None
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"配置文件不存在: {config_path}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        logger.error("YAML支持不可用，请安装pyyaml: pip install pyyaml")
                        return None
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                    return None
            
            logger.info(f"成功从文件加载配置: {config_path}")
            return config_data
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {config_path}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def save_to_file(config_dict: Dict[str, Any], 
                    config_path: Union[str, Path],
                    format: str = 'json') -> bool:
        """
        保存配置到文件
        
        Args:
            config_dict: 配置字典
            config_path: 配置文件路径
            format: 文件格式 ('json' 或 'yaml')
            
        Returns:
            是否保存成功
        """
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml' or config_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        logger.error("YAML支持不可用，使用JSON格式")
                        format = 'json'
                    
                    if YAML_AVAILABLE:
                        yaml.dump(config_dict, f, default_flow_style=False, 
                                 allow_unicode=True, sort_keys=False)
                    else:
                        json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {config_path}: {e}", exc_info=True)
            return False
    
    @staticmethod
    def load_full_config(config_path: Optional[Union[str, Path]] = None,
                        user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        加载完整配置（合并默认配置、文件配置和用户配置）
        
        Args:
            config_path: 配置文件路径（可选）
            user_config: 用户提供的配置字典（可选）
            
        Returns:
            合并后的完整配置字典
        """
        # 从默认配置开始
        full_config = {
            'app': DEFAULT_APP_CONFIG.copy(),
            'simulation': DEFAULT_SIMULATION_CONFIG.copy(),
            'ui': DEFAULT_UI_CONFIG.copy(),
        }
        
        # 加载文件配置
        if config_path:
            file_config = ConfigLoader.load_from_file(config_path)
            if file_config:
                full_config = merge_configs(full_config, file_config)
        
        # 合并用户配置
        if user_config:
            full_config = merge_configs(full_config, user_config)
        
        return full_config
    
    @staticmethod
    def create_config_objects(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从配置字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            包含配置对象的字典
        """
        configs = {}
        
        # 应用配置
        if 'app' in config_dict:
            configs['app'] = ApplicationConfig.from_dict(config_dict['app'])
        else:
            configs['app'] = ApplicationConfig.default()
        
        # 模拟配置
        if 'simulation' in config_dict:
            configs['simulation'] = SimulationConfig.from_dict(config_dict['simulation'])
        else:
            configs['simulation'] = SimulationConfig.default()
        
        # UI配置
        if 'ui' in config_dict:
            configs['ui'] = UIConfig.from_dict(config_dict['ui'])
        else:
            configs['ui'] = UIConfig.default()
        
        return configs


def load_config_from_file(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    从文件加载配置（便捷函数）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    return ConfigLoader.load_from_file(config_path)


def save_config_to_file(config_dict: Dict[str, Any], 
                       config_path: Union[str, Path],
                       format: str = 'json') -> bool:
    """
    保存配置到文件（便捷函数）
    
    Args:
        config_dict: 配置字典
        config_path: 配置文件路径
        format: 文件格式
        
    Returns:
        是否保存成功
    """
    return ConfigLoader.save_to_file(config_dict, config_path, format)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def create_default_config_file(config_path: Union[str, Path] = "config/default.json") -> bool:
    """
    创建默认配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        是否创建成功
    """
    default_config = {
        'app': DEFAULT_APP_CONFIG,
        'simulation': DEFAULT_SIMULATION_CONFIG,
        'ui': DEFAULT_UI_CONFIG,
        'agents': DEFAULT_AGENT_CONFIGS,
    }
    
    return ConfigLoader.save_to_file(default_config, config_path)

