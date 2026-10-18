"""Local experiment server.

Run with ``python -m src.lab``. The page is a static file; the simulation
stays in this process.
"""

from __future__ import annotations

import asyncio
import json
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.lab.controls import ControlError, controls_meta
from src.lab.session import LabSession

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION = LabSession()


@asynccontextmanager
async def _lifespan(app):
    SESSION.bind_loop(asyncio.get_running_loop())
    SESSION.start()
    if getattr(app.state, "open_browser", False):
        webbrowser.open(getattr(app.state, "url", "http://127.0.0.1:8765/"))
    try:
        yield
    finally:
        SESSION.stop()


def create_app():
    app = FastAPI(title="Umvili Lab", lifespan=_lifespan)

    @app.get("/api/meta")
    def meta():
        return controls_meta()

    @app.get("/api/state")
    def state():
        return SESSION.public_state()

    @app.post("/api/preview")
    def preview(body: dict):
        try:
            return SESSION.preview(body)
        except ControlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/apply")
    def apply(body: dict):
        try:
            return SESSION.apply(body)
        except ControlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/control")
    def control(body: dict):
        try:
            return SESSION.control(body)
        except ControlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/export")
    def export():
        try:
            payload = SESSION.export()
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        digest = payload.get("protocol_hash") or "run"
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        return Response(
            content=raw,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="umvili-{digest}.json"'},
        )

    @app.websocket("/ws")
    async def stream(socket: WebSocket):
        await socket.accept()
        inbox = SESSION.subscribe()
        try:
            while True:
                frame = await inbox.get()
                await socket.send_json(frame)
        except WebSocketDisconnect:
            return
        finally:
            SESSION.unsubscribe(inbox)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("缺少 uvicorn。请先安装：pip install fastapi uvicorn") from exc
    app = create_app()
    app.state.open_browser = open_browser
    app.state.url = f"http://{host}:{port}/"
    uvicorn.run(app, host=host, port=port, log_level="info")
