"""
运行时日志环形缓冲，供 Debug 视图以终端方式展示。

与 logging 根记录器上的 UILogHandler 配合，复用与控制台相同的 Formatter 输出格式。
按来源分为三个竖向子窗口：Simulation / system、IQL、QMIX。
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Deque, Dict, List

_LOCK = threading.Lock()

# 与 Debug 三栏顺序一致：上 Simulation，中 IQL，下 QMIX
CHANNEL_SIMULATION = "simulation"
CHANNEL_IQL = "iql"
CHANNEL_QMIX = "qmix"

_CHANNEL_ORDER: tuple[str, ...] = (CHANNEL_SIMULATION, CHANNEL_IQL, CHANNEL_QMIX)
_CHANNELS: Dict[str, Deque[str]] = {
    CHANNEL_SIMULATION: deque(maxlen=800),
    CHANNEL_IQL: deque(maxlen=800),
    CHANNEL_QMIX: deque(maxlen=800),
}


def _route_channel(record: logging.LogRecord) -> str:
    name = record.name
    path = getattr(record, "pathname", "") or ""
    base = os.path.basename(path.replace("\\", "/"))
    if base in ("qmix_trainer.py", "qmix_agent.py"):
        return CHANNEL_QMIX
    if name.startswith("IQLAgent_") or name == "PriorityReplayBuffer":
        return CHANNEL_IQL
    if (
        name.startswith("QMIXAgent_")
        or name == "QMIXTrainer"
        or name == "MultiAgentReplayBuffer"
    ):
        return CHANNEL_QMIX
    return CHANNEL_SIMULATION


def get_ui_log_channel_snapshots() -> Dict[str, List[str]]:
    """各通道线程安全拷贝，供 UI 每帧读取。"""
    with _LOCK:
        return {ch: list(_CHANNELS[ch]) for ch in _CHANNEL_ORDER}


def clear_ui_log_buffer() -> None:
    with _LOCK:
        for d in _CHANNELS.values():
            d.clear()


class UILogHandler(logging.Handler):
    """将已格式化的日志行按来源写入对应环形缓冲。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record).rstrip("\n")
            if not msg:
                return
            ch = _route_channel(record)
            with _LOCK:
                _CHANNELS[ch].append(msg)
        except Exception:
            self.handleError(record)
