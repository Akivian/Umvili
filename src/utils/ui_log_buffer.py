"""
运行时日志环形缓冲，供 Debug 视图以终端方式展示。

与 logging 根记录器上的 UILogHandler 配合，复用与控制台相同的 Formatter 输出格式。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, List

_LOCK = threading.Lock()
_BUFFER: Deque[str] = deque(maxlen=2000)


def get_ui_log_buffer_snapshot() -> List[str]:
    """线程安全拷贝，供 UI 每帧读取。"""
    with _LOCK:
        return list(_BUFFER)


def clear_ui_log_buffer() -> None:
    with _LOCK:
        _BUFFER.clear()


class UILogHandler(logging.Handler):
    """将已格式化的日志行写入环形缓冲。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record).rstrip("\n")
            if not msg:
                return
            with _LOCK:
                _BUFFER.append(msg)
        except Exception:
            self.handleError(record)
