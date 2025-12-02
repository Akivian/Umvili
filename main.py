"""
MARL沙盘平台 - 主程序入口

这是一个基于多智能体强化学习(MARL)的沙盘式算法演算对比平台。
支持多种智能体算法（IQL、QMIX、规则型等）的实时可视化对比。

设计原则：
- 单一职责：每个类和方法都有明确的职责
- 开闭原则：易于扩展新的模拟类型和功能
- 依赖注入：通过配置灵活控制行为
- 错误处理：完善的异常处理和资源清理
- 类型安全：完整的类型提示

作者：MARL Platform Team
版本：2.0.0
"""

import sys
import pygame
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback

# 在导入其他模块之前设置日志
from src.utils.logging_config import setup_logging
setup_logging()

from src.core.simulation import MARLSimulation, SimulationFactory, SimulationState
from src.utils.visualization import ModernVisualizationSystem
from src.config import (
    ApplicationConfig,
    SimulationConfig,
    UIConfig,
    ConfigLoader,
    DEFAULT_APP_CONFIG,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_UI_CONFIG
)


# 获取模块日志记录器
logger = logging.getLogger(__name__)


# ApplicationState is now imported from src.config
# This enum definition is removed to use the centralized config system


# ApplicationConfig is now imported from src.config
# This class definition is removed to use the centralized config system


class MARLApplication:
    """
    MARL沙盘平台主应用程序类
    
    负责应用程序的完整生命周期管理，包括：
    - 初始化和配置验证
    - 模拟和可视化系统的创建
    - 主循环管理
    - 事件处理
    - 资源清理
    
    特性：
    - 完善的错误处理和恢复机制
    - 资源自动管理
    - 配置验证
    - 性能监控
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化应用程序
        
        Args:
            config: 配置字典，如果为None则使用默认配置
            
        Raises:
            ValueError: 配置验证失败时
            RuntimeError: 初始化失败时
        """
        self.state = ApplicationState.INITIALIZING
        self.config = self._load_config(config)
        self._validate_config()
        
        # 核心组件（延迟初始化）
        self.simulation: Optional[MARLSimulation] = None
        self.visualization_system: Optional[ModernVisualizationSystem] = None
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # 运行状态
        self.running = False
        self.frame_count = 0
        self.start_time: Optional[float] = None
        
        # 性能统计
        self.fps_history: list[float] = []
        self.max_fps_history_size = 100
        
        logger.info("MARL应用程序对象创建完成")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> ApplicationConfig:
        """
        加载和合并配置
        
        Args:
            config: 用户提供的配置字典
            
        Returns:
            合并后的配置对象
        """
        # 从默认配置开始
        default_dict = DEFAULT_CONFIG.copy()
        default_dict.update({
            'simulation_type': 'comparative',
            'target_fps': FPS,
            'window_title': "MARL沙盘平台 - 多智能体强化学习模拟"
        })
        
        # 合并用户配置
        if config:
            default_dict.update(config)
        
        return ApplicationConfig.from_dict(default_dict)
    
    def _validate_config(self) -> None:
        """验证配置有效性"""
        # 验证应用配置
        is_valid, error_msg = self.app_config.validate()
        if not is_valid:
            raise ValueError(f"应用配置验证失败: {error_msg}")
        
        # 验证模拟配置
        is_valid, error_msg = self.sim_config.validate()
        if not is_valid:
            raise ValueError(f"模拟配置验证失败: {error_msg}")
        
        # 验证UI配置
        is_valid, error_msg = self.ui_config.validate()
        if not is_valid:
            raise ValueError(f"UI配置验证失败: {error_msg}")
        
        # 额外的业务逻辑验证
        max_agents = self.sim_config.grid_size * 2
        if self.sim_config.initial_agents > max_agents:
            logger.warning(f"智能体数量 {self.sim_config.initial_agents} 超过推荐值 {max_agents}")
            # 注意：dataclass是只读的，这里只记录警告，实际调整需要在创建配置时进行
    
    def initialize(self) -> bool:
        """
        初始化应用程序的所有组件
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("开始初始化应用程序组件...")
            
            # 初始化Pygame
            if not self._initialize_pygame():
                return False
            
            # 创建模拟实例
            if not self._create_simulation():
                return False
            
            # 创建可视化系统
            if not self._create_visualization():
                return False
            
            # 创建显示窗口
            if not self._create_display():
                return False
            
            self.state = ApplicationState.READY
            logger.info("应用程序初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            self.state = ApplicationState.ERROR
            return False
    
    def _initialize_pygame(self) -> bool:
        """初始化Pygame系统"""
        try:
            pygame.init()
            pygame.display.set_caption(self.ui_config.window.title)
            
            # 检查Pygame是否成功初始化
            if not pygame.get_init():
                logger.error("Pygame初始化失败")
                return False
            
            logger.info("Pygame初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"Pygame初始化异常: {e}", exc_info=True)
            return False
    
    def _create_simulation(self) -> bool:
        """
        创建模拟实例
        
        Returns:
            创建是否成功
        """
        try:
            logger.info(f"创建模拟实例 (类型: {self.app_config.simulation_type})")
            
            # 根据类型创建不同的模拟
            sim = self._create_simulation_by_type()
            
            # 应用配置覆盖
            self._apply_simulation_config(sim)
            
            self.simulation = sim
            logger.info(f"模拟实例创建成功: {sim}")
            return True
            
        except Exception as e:
            logger.error(f"创建模拟实例失败: {e}", exc_info=True)
            return False
    
    def _create_simulation_by_type(self) -> MARLSimulation:
        """根据配置类型创建模拟实例"""
        sim_type = self.app_config.simulation_type
        
        if sim_type == 'comparative':
            sim = SimulationFactory.create_comparative_simulation()
        elif sim_type == 'training':
            sim = SimulationFactory.create_marl_training_simulation()
        elif sim_type == 'performance':
            sim = SimulationFactory.create_high_performance_simulation()
        else:  # default
            sim = MARLSimulation(
                grid_size=self.sim_config.grid_size,
                cell_size=self.sim_config.cell_size,
                initial_agents=self.sim_config.initial_agents,
                sugar_growth_rate=self.sim_config.sugar_growth_rate,
                max_sugar=self.sim_config.max_sugar
            )
        
        return sim
    
    def _apply_simulation_config(self, sim: MARLSimulation) -> None:
        """
        同步模拟与应用配置，确保双方对实际网格尺寸认知一致。
        对于带自定义设置的预配置模拟（如comparative），以模拟自身配置为准。
        """
        # 以模拟环境的真实尺寸为权威来源
        env_size = getattr(getattr(sim, 'environment', None), 'size', None)
        if env_size:
            sim.grid_size = env_size
            # 注意：不能直接修改dataclass，这里只是同步内部状态
    
    def _create_visualization(self) -> bool:
        """
        创建可视化系统
        
        Returns:
            创建是否成功
        """
        try:
            if self.simulation is None:
                logger.error("无法创建可视化系统：模拟实例未初始化")
                return False
            
            logger.info("创建可视化系统...")
            
            grid_size = getattr(getattr(self.simulation, 'environment', None), 'size', None) \
                or getattr(self.simulation, 'grid_size', self.sim_config.grid_size)
            cell_size = getattr(self.simulation, 'cell_size', self.sim_config.cell_size)
            self.simulation.grid_size = grid_size  # 保持模拟对象内部一致

            simulation_width = grid_size * cell_size
            panel_width = 400
            
            self.visualization_system = ModernVisualizationSystem(
                screen_width=simulation_width + panel_width + 20,
                screen_height=max(800, simulation_width),
                grid_size=grid_size,
                cell_size=cell_size
            )
            
            logger.info("可视化系统创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建可视化系统失败: {e}", exc_info=True)
            return False
    
    def _create_display(self) -> bool:
        """
        创建显示窗口
        
        Returns:
            创建是否成功
        """
        try:
            if self.visualization_system is None:
                logger.error("无法创建显示窗口：可视化系统未初始化")
                return False
            
            screen_info = self.visualization_system.get_screen_info()
            # get_screen_info返回元组 (width, height)
            if isinstance(screen_info, tuple):
                width, height = screen_info
            else:
                # 兼容字典格式
                width = screen_info.get('width', 1200)
                height = screen_info.get('height', 800)
            
            # 创建窗口标志
            flags = 0
            if self.ui_config.window.enable_vsync:
                flags |= pygame.DOUBLEBUF
            
            self.screen = pygame.display.set_mode((width, height), flags)
            
            if self.screen is None:
                logger.error("无法创建显示窗口")
                return False
            
            # 创建时钟对象
            self.clock = pygame.time.Clock()
            
            logger.info(f"显示窗口创建成功: {width}x{height}")
            return True
            
        except Exception as e:
            logger.error(f"创建显示窗口失败: {e}", exc_info=True)
            return False
    
    def run(self) -> int:
        """
        运行主应用程序循环
        
        Returns:
            退出代码 (0表示成功，非0表示错误)
        """
        if self.state != ApplicationState.READY:
            logger.error(f"应用程序未就绪，当前状态: {self.state.value}")
            return 1
        
        if self.simulation is None or self.visualization_system is None or self.screen is None:
            logger.error("核心组件未初始化")
            return 1
        
        # 启动模拟
        try:
            self.simulation.start()
            self.running = True
            self.state = ApplicationState.RUNNING
            self.start_time = pygame.time.get_ticks() / 1000.0
            
            logger.info("开始主循环")
            
            # 主循环
            while self.running:
                exit_code = self._run_frame()
                if exit_code is not None:
                    return exit_code
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            return 0
        except Exception as e:
            logger.error(f"主循环异常: {e}", exc_info=True)
            self.state = ApplicationState.ERROR
            return 1
        finally:
            self._cleanup()
    
    def _run_frame(self) -> Optional[int]:
        """
        运行一帧
        
        Returns:
            如果需要退出，返回退出代码；否则返回None
        """
        try:
            # 处理事件
            if not self._handle_events():
                return 0  # 正常退出
            
            # 更新模拟
            if not self._update():
                return 1  # 更新失败
            
            # 渲染画面
            if not self._render():
                return 1  # 渲染失败
            
            # 控制帧率
            dt = self.clock.tick(self.ui_config.window.fps)
            self._update_performance_stats(dt)
            
            self.frame_count += 1
            return None
            
        except Exception as e:
            logger.error(f"帧处理异常: {e}", exc_info=True)
            return 1
    
    def _handle_events(self) -> bool:
        """
        处理所有事件
        
        Returns:
            是否继续运行
        """
        try:
            for event in pygame.event.get():
                # 退出事件
                if event.type == pygame.QUIT:
                    logger.info("收到退出事件")
                    return False
                
                # 将事件传递给可视化系统
                if self.visualization_system and self.simulation:
                    if self.visualization_system.handle_event(event, self.simulation):
                        continue  # 事件已被处理
                
                # 处理键盘事件
                if event.type == pygame.KEYDOWN:
                    if not self._handle_keydown(event):
                        return False  # 退出请求
            
            return True
            
        except Exception as e:
            logger.error(f"事件处理异常: {e}", exc_info=True)
            return True  # 继续运行，不因事件处理错误而退出
    
    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        """
        处理键盘按下事件
        
        Args:
            event: Pygame事件对象
            
        Returns:
            是否继续运行
        """
        if self.simulation is None:
            return True
        
        try:
            key = event.key
            
            # 空格键：暂停/继续
            if key == pygame.K_SPACE:
                if self.simulation.state == SimulationState.RUNNING:
                    self.simulation.pause()
                    self.state = ApplicationState.PAUSED
                    logger.info("模拟已暂停")
                else:
                    self.simulation.resume()
                    self.state = ApplicationState.RUNNING
                    logger.info("模拟已恢复")
            
            # R键：重置
            elif key == pygame.K_r:
                logger.info("重置模拟")
                self.simulation.reset()
                self.frame_count = 0
                self.start_time = pygame.time.get_ticks() / 1000.0
            
            # ESC键：退出
            elif key == pygame.K_ESCAPE:
                logger.info("用户请求退出")
                return False
            
            # F键：切换全屏（可选功能）
            elif key == pygame.K_f:
                self._toggle_fullscreen()
            
            return True
            
        except Exception as e:
            logger.error(f"键盘事件处理异常: {e}", exc_info=True)
            return True
    
    def _toggle_fullscreen(self) -> None:
        """切换全屏模式"""
        try:
            if self.screen is None:
                return
            
            current_flags = self.screen.get_flags()
            if current_flags & pygame.FULLSCREEN:
                # 退出全屏
                if self.visualization_system:
                    screen_info = self.visualization_system.get_screen_info()
                    if isinstance(screen_info, tuple):
                        width, height = screen_info
                    else:
                        width = screen_info.get('width', 1200)
                        height = screen_info.get('height', 800)
                else:
                    width, height = 1200, 800
                self.screen = pygame.display.set_mode((width, height))
            else:
                # 进入全屏
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            
            logger.info("切换全屏模式")
        except Exception as e:
            logger.warning(f"切换全屏失败: {e}")
    
    def _update(self) -> bool:
        """
        更新模拟状态
        
        Returns:
            更新是否成功
        """
        try:
            if self.simulation is None:
                return False
            
            # 只在运行状态时更新
            if self.simulation.state == SimulationState.RUNNING:
                if not self.simulation.update():
                    logger.warning("模拟更新返回False")
                    # 检查是否需要停止
                    if self.simulation.state == SimulationState.STOPPED:
                        logger.info("模拟已停止")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"更新模拟失败: {e}", exc_info=True)
            return False
    
    def _render(self) -> bool:
        """
        渲染画面
        
        Returns:
            渲染是否成功
        """
        try:
            if self.screen is None or self.simulation is None or self.visualization_system is None:
                return False
            
            # 获取模拟数据
            simulation_data = self.simulation.get_simulation_data()
            
            # 使用可视化系统绘制
            self.visualization_system.draw(self.screen, simulation_data)
            
            # 显示FPS（如果启用）
            if self.app_config.show_fps and self.clock:
                self._render_fps()
            
            # 更新显示
            pygame.display.flip()
            
            return True
            
        except Exception as e:
            logger.error(f"渲染失败: {e}", exc_info=True)
            return False
    
    def _render_fps(self) -> None:
        """在屏幕上渲染FPS信息"""
        try:
            if self.screen is None or self.clock is None:
                return
            
            fps = self.clock.get_fps()
            fps_text = f"FPS: {fps:.1f}"
            
            # 使用Pygame默认字体
            font = pygame.font.Font(None, 24)
            text_surface = font.render(fps_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            
        except Exception as e:
            logger.debug(f"渲染FPS失败: {e}")
    
    def _update_performance_stats(self, dt: float) -> None:
        """更新性能统计"""
        if self.clock:
            fps = self.clock.get_fps()
            self.fps_history.append(fps)
            if len(self.fps_history) > self.max_fps_history_size:
                self.fps_history.pop(0)
    
    def _cleanup(self) -> None:
        """清理资源"""
        logger.info("开始清理资源...")
        self.state = ApplicationState.SHUTTING_DOWN
        
        try:
            # 清理模拟
            if self.simulation:
                # 模拟对象会自动清理，这里可以添加额外的清理逻辑
                pass
            
            # 清理可视化系统
            if self.visualization_system:
                # 可视化系统会自动清理
                pass
            
            # 清理Pygame
            pygame.quit()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源时出错: {e}", exc_info=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        stats = {
            'frame_count': self.frame_count,
            'fps_history': self.fps_history.copy() if self.fps_history else [],
        }
        
        if self.fps_history:
            stats['avg_fps'] = sum(self.fps_history) / len(self.fps_history)
            stats['min_fps'] = min(self.fps_history)
            stats['max_fps'] = max(self.fps_history)
        
        if self.start_time:
            elapsed = pygame.time.get_ticks() / 1000.0 - self.start_time
            stats['elapsed_time'] = elapsed
            if elapsed > 0:
                stats['avg_fps_overall'] = self.frame_count / elapsed
        
        return stats


def load_config_from_file(config_path: str) -> Optional[Dict[str, Any]]:
    """
    从文件加载配置（便捷函数，使用新的配置加载器）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典，如果加载失败返回None
    """
    return ConfigLoader.load_from_file(config_path)


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'app': DEFAULT_APP_CONFIG,
        'simulation': DEFAULT_SIMULATION_CONFIG,
        'ui': DEFAULT_UI_CONFIG,
    }


def main() -> int:
    """
    主函数入口
    
    Returns:
        退出代码
    """
    try:
        # 解析命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='MARL沙盘平台 - 多智能体强化学习模拟')
        parser.add_argument('--config', type=str, help='配置文件路径（JSON或YAML）')
        parser.add_argument('--simulation-type', type=str, 
                          choices=['default', 'comparative', 'training', 'performance'],
                          help='模拟类型')
        parser.add_argument('--grid-size', type=int, help='网格大小')
        parser.add_argument('--agents', type=int, help='初始智能体数量')
        args = parser.parse_args()
        
        # 构建配置字典
        user_config = {}
        if args.simulation_type:
            user_config.setdefault('app', {})['simulation_type'] = args.simulation_type
        if args.grid_size:
            user_config.setdefault('simulation', {})['grid_size'] = args.grid_size
        if args.agents:
            user_config.setdefault('simulation', {})['initial_agents'] = args.agents
        
        # 创建应用程序（支持配置文件路径）
        app = MARLApplication(config=user_config if user_config else None, 
                             config_path=args.config)
        
        # 初始化
        if not app.initialize():
            logger.error("应用程序初始化失败")
            return 1
        
        # 运行
        exit_code = app.run()
        
        # 输出性能统计
        if logger.isEnabledFor(logging.INFO):
            stats = app.get_performance_stats()
            logger.info(f"性能统计: {stats}")
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        return 0
    except Exception as e:
        logger.critical(f"程序异常退出: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
