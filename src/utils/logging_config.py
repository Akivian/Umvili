"""
统一的日志配置模块。

此模块封装了日志初始化逻辑，确保：
- 只初始化一次，避免重复输出
- 同时输出到控制台和文件
- 可以通过参数自定义日志级别和文件路径
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "marl_simulation.log",
    propagate: bool = False
) -> None:
    """
    设置统一的日志配置。

    Args:
        level: 根日志级别
        log_file: 日志文件路径
        propagate: 是否向父日志器传播
    """
    root_logger = logging.getLogger()

    # 如果已经配置过，则先清理，避免重复输出
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 控制台输出
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 文件输出
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError as exc:
            root_logger.warning(f"无法写入日志文件 {log_file}: {exc}")

    root_logger.propagate = propagate

    # 设置子日志器级别
    logging.getLogger("AgentFactory").setLevel(level)
    logging.getLogger("MARLSimulation").setLevel(level)