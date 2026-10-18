"""One local simulation, driven off the request thread.

Training stays on this thread so CUDA work is not split across the event loop.
The browser receives a snapshot about ten times a second, independent of how
many environment steps ran between frames.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from src.lab.controls import (
    ControlError,
    LabControls,
    active_metric_types,
    build_protocol,
    build_reset_config,
    controls_from_dict,
    default_controls,
)
from src.lab.estimator import estimate_for_controls
from src.lab.snapshot import frame_from_simulation

LOGGER = logging.getLogger(__name__)
HISTORY_CAP = 4000
FRAME_INTERVAL = 0.1


class LabSession:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._commands: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: List[asyncio.Queue] = []
        self._simulation: Any = None
        self._controls: LabControls = default_controls()
        self._applied_hash: Optional[str] = None
        self._speed = 1
        self._paused = False
        self._device = "cpu"
        self._error = ""
        self._status = "starting"
        self._sugar: Dict[str, List[Dict[str, float]]] = {}
        self._loss: Dict[str, List[Dict[str, float]]] = {}
        self._latest: Optional[Dict[str, Any]] = None
        self._replace_next = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="umvili-lab", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def preview(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        controls = controls_from_dict(raw)
        protocol = build_protocol(controls)
        digest = protocol.protocol_hash()
        with self._lock:
            applied = self._applied_hash
        return {
            "controls": controls.to_dict(),
            "protocol_hash": digest,
            "applied_hash": applied,
            "hash_changes": applied is not None and digest != applied,
            **estimate_for_controls(controls),
        }

    def apply(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("apply", raw)

    def control(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("control", raw)

    def export(self) -> Dict[str, Any]:
        return self._call("export", None)

    def public_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "error": self._error,
                "device": self._device,
                "speed": self._speed,
                "controls": self._controls.to_dict(),
                "hash": self._applied_hash,
                "frame": _without_points(self._latest),
            }

    def subscribe(self) -> asyncio.Queue:
        inbox: asyncio.Queue = asyncio.Queue(maxsize=4)
        with self._lock:
            self._subscribers.append(inbox)
            hello = {
                "kind": "hello",
                "status": self._status,
                "error": self._error,
                "device": self._device,
                "speed": self._speed,
                "controls": self._controls.to_dict(),
                "hash": self._applied_hash,
                "replace": self._history_unlocked(),
                "frame": _without_points(self._latest),
            }
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(_offer, inbox, hello)
        return inbox

    def unsubscribe(self, inbox: asyncio.Queue) -> None:
        with self._lock:
            if inbox in self._subscribers:
                self._subscribers.remove(inbox)

    def _call(self, name: str, payload: Any, timeout: float = 180) -> Dict[str, Any]:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("实验台还没有启动")
        box: Dict[str, Any] = {}
        done = threading.Event()
        self._commands.put((name, payload, box, done))
        if not done.wait(timeout):
            raise TimeoutError("模拟线程没有在时限内完成")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _run(self) -> None:
        try:
            self._boot()
        except Exception as exc:
            LOGGER.exception("lab boot failed")
            with self._lock:
                self._error = str(exc)
                self._status = "error"
            self._publish([])
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self._drain()
                points = self._burst()
                if self._simulation is not None or self._status == "error":
                    self._publish(points)
            except Exception as exc:
                LOGGER.exception("lab tick failed")
                with self._lock:
                    self._error = str(exc)
            remaining = FRAME_INTERVAL - (time.perf_counter() - started)
            if self._stop.wait(max(0.0, remaining)):
                break

    def _boot(self) -> None:
        import torch

        from src.core.simulation import MARLSimulation
        from src.marl.device import select_device

        device = select_device()
        if device.type == "cuda":
            label = "cuda · " + torch.cuda.get_device_name(0)
        else:
            label = "cpu"
        simulation = MARLSimulation(
            grid_size=self._controls.grid_size,
            initial_agents=self._controls.agents_per_group,
        )
        simulation.reset(build_reset_config(self._controls))
        simulation.start()
        digest = simulation.protocol.protocol_hash()
        with self._lock:
            self._simulation = simulation
            self._device = label
            self._applied_hash = digest
            self._status = "ready"
            self._error = ""
        self._replace_next = True

    def _drain(self) -> None:
        while True:
            try:
                name, payload, box, done = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                box["result"] = self._dispatch(name, payload)
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

    def _dispatch(self, name: str, payload: Any) -> Dict[str, Any]:
        if name == "apply":
            return self._apply(payload or {})
        if name == "control":
            return self._control(payload or {})
        if name == "export":
            return self._export()
        raise ControlError(f"unknown command: {name}")

    def _burst(self) -> List[Dict[str, Any]]:
        simulation = self._simulation
        if simulation is None or self._paused:
            return []
        if getattr(simulation.state, "value", "") != "running":
            return []
        points = []
        for _ in range(self._speed):
            if getattr(simulation.state, "value", "") != "running":
                break
            point = self._step_once()
            if point is not None:
                points.append(point)
        return points

    def _step_once(self) -> Optional[Dict[str, Any]]:
        if not self._simulation.update():
            return None
        sample = frame_from_simulation(self._simulation, include_map=False)
        point = {
            "step": int(sample["step"]),
            "sugar": sample["sugar"],
            "loss": sample["loss"],
        }
        with self._lock:
            _append(self._sugar, point["step"], point["sugar"])
            _append(self._loss, point["step"], point["loss"])
        return point

    def _apply(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if self._simulation is None:
            raise RuntimeError(self._error or "模拟还没有准备好")
        controls = controls_from_dict(raw)
        digest = build_protocol(controls).protocol_hash()
        with self._lock:
            old_hash = self._applied_hash
            if old_hash != digest:
                self._sugar.clear()
                self._loss.clear()
            else:
                active = set(active_metric_types(controls))
                for bucket in (self._sugar, self._loss):
                    for key in list(bucket):
                        if key in active:
                            del bucket[key]
        self._simulation.reset(build_reset_config(controls))
        if not self._paused:
            self._simulation.start()
        with self._lock:
            self._controls = controls
            self._applied_hash = digest
            self._error = ""
        self._replace_next = True
        return {
            "protocol_hash": digest,
            "hash_changed": old_hash != digest,
            **estimate_for_controls(controls),
        }

    def _control(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if self._simulation is None:
            raise RuntimeError(self._error or "模拟还没有准备好")
        action = str(raw.get("action", ""))
        if action == "pause":
            self._paused = True
            self._simulation.pause()
        elif action == "resume":
            self._paused = False
            self._simulation.resume()
        elif action == "reset":
            with self._lock:
                self._sugar.clear()
                self._loss.clear()
            self._simulation.reset(build_reset_config(self._controls))
            if not self._paused:
                self._simulation.start()
            self._replace_next = True
        elif action == "speed":
            try:
                speed = int(raw.get("speed", 1))
            except (TypeError, ValueError) as exc:
                raise ControlError("speed must be 1, 4, or 16") from exc
            if speed not in (1, 4, 16):
                raise ControlError("speed must be 1, 4, or 16")
            self._speed = speed
        else:
            raise ControlError(f"unknown action: {action}")
        return {"state": self._simulation.state.value, "speed": self._speed, "paused": self._paused}

    def _export(self) -> Dict[str, Any]:
        protocol = self._simulation.protocol if self._simulation is not None else None
        with self._lock:
            return {
                "protocol": protocol.to_dict() if protocol is not None else None,
                "protocol_hash": self._applied_hash,
                "controls": self._controls.to_dict(),
                "device": self._device,
                "series": self._history_unlocked(),
            }

    def _publish(self, points: List[Dict[str, Any]]) -> None:
        if self._simulation is None:
            frame: Dict[str, Any] = {
                "kind": "frame",
                "step": 0,
                "hash": "",
                "state": self._status,
                "epsilon": 0.0,
                "sugar": {},
                "loss": {},
                "alive": 0,
                "agents": [],
                "points": [],
                "error": self._error,
            }
            include_replace = False
        else:
            include_map = self._replace_next or self._latest is None or bool(points)
            frame = frame_from_simulation(self._simulation, include_map=include_map)
            frame["kind"] = "frame"
            frame["error"] = self._error
            include_replace = self._replace_next
            frame["points"] = [] if include_replace else points
            self._replace_next = False
        frame["speed"] = self._speed
        frame["device"] = self._device
        with self._lock:
            if include_replace:
                frame["replace"] = self._history_unlocked()
            else:
                frame["replace"] = None
            self._latest = frame
            subscribers = list(self._subscribers)
        self._fanout(subscribers, frame)

    def _history_unlocked(self) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        return {
            "sugar_per_step": _copy_series(self._sugar),
            "loss": _copy_series(self._loss),
        }

    def _fanout(self, subscribers: List[asyncio.Queue], frame: Dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        for inbox in subscribers:
            loop.call_soon_threadsafe(_offer, inbox, frame)


def _append(bucket: Dict[str, List[Dict[str, float]]], step: int, values: Dict[str, float]) -> None:
    for name, value in values.items():
        series = bucket.setdefault(str(name), [])
        series.append({"step": step, "value": float(value)})
        if len(series) > HISTORY_CAP:
            del series[: len(series) - HISTORY_CAP]


def _copy_series(bucket: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[Dict[str, float]]]:
    return {name: [dict(point) for point in rows] for name, rows in bucket.items()}


def _without_points(frame: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    copied = dict(frame)
    copied["points"] = []
    copied["replace"] = None
    return copied


def _offer(inbox: asyncio.Queue, frame: Dict[str, Any]) -> None:
    if inbox.full():
        try:
            inbox.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        inbox.put_nowait(frame)
    except asyncio.QueueFull:
        return
