"""Loopback-only StudySync API and event stream for the local web interface."""

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from study_buddy_inference import DEFAULT_MODEL_PATH
from study_session import SessionController, SessionSnapshot


FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
logger = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=120)
    goal: str = Field(min_length=1, max_length=500)
    duration_minutes: float = Field(ge=1, le=240)


class EventBus:
    """Relay low-rate snapshots from the camera thread to WebSocket clients."""

    def __init__(self):
        self._lock = Lock()
        self._sequence = 0
        self._latest = None
        self._loop = None
        self._subscribers: dict[asyncio.Queue, str] = {}

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, snapshot: SessionSnapshot) -> None:
        with self._lock:
            self._sequence += 1
            event = {
                "session_id": snapshot.session_id,
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "snapshot",
                "payload": asdict(snapshot),
            }
            self._latest = event
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._fan_out, event)

    def current(self, snapshot: SessionSnapshot | None) -> dict:
        with self._lock:
            if self._latest is not None and snapshot is not None and (
                self._latest["session_id"] == snapshot.session_id
            ):
                return self._latest
            sequence = self._sequence
        return {
            "session_id": snapshot.session_id if snapshot else None,
            "sequence": sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "snapshot",
            "payload": asdict(snapshot) if snapshot else None,
        }

    def subscribe(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=16)
        self._subscribers[queue] = session_id
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.pop(queue, None)

    def _fan_out(self, event: dict) -> None:
        for queue, session_id in tuple(self._subscribers.items()):
            if event["session_id"] != session_id:
                continue
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)


class Presence:
    """Pause capture after the last browser connection has been gone briefly."""

    def __init__(self, controller: SessionController, grace_seconds: float):
        self.controller = controller
        self.grace_seconds = grace_seconds
        self.connections: dict[str, int] = {}
        self._task: asyncio.Task | None = None
        self._task_session_id: str | None = None

    def connected(self, session_id: str) -> None:
        self.connections[session_id] = self.connections.get(session_id, 0) + 1
        if self._task is not None and self._task_session_id == session_id:
            self._task.cancel()
            self._task = None
            self._task_session_id = None

    def disconnected(self, session_id: str) -> None:
        remaining = self.connections.get(session_id, 0) - 1
        if remaining > 0:
            self.connections[session_id] = remaining
        else:
            self.connections.pop(session_id, None)
        self.require_connection()

    def require_connection(self) -> None:
        snapshot = self.controller.snapshot()
        if (snapshot and snapshot.status == "running"
                and self.connections.get(snapshot.session_id, 0) == 0):
            if self._task is not None and self._task_session_id == snapshot.session_id:
                return
            if self._task is not None:
                self._task.cancel()
            self._task_session_id = snapshot.session_id
            self._task = asyncio.create_task(self._pause_later(snapshot.session_id))

    async def _pause_later(self, session_id: str) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
            snapshot = self.controller.snapshot()
            if (snapshot and snapshot.session_id == session_id
                    and snapshot.status == "running"
                    and self.connections.get(session_id, 0) == 0):
                try:
                    await asyncio.to_thread(self.controller.pause)
                except RuntimeError:
                    # A concurrent stop or pause won the race.
                    pass
                except TimeoutError:
                    logger.exception("Camera worker did not pause after browser disconnect")
        except asyncio.CancelledError:
            return
        finally:
            if self._task_session_id == session_id:
                self._task = None
                self._task_session_id = None

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            self._task_session_id = None


def create_app(
    *,
    controller: SessionController | None = None,
    static_dir: Path = FRONTEND_DIST,
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
    dev_origin: str | None = None,
    disconnect_grace_seconds: float = 15.0,
    demo_mode: bool = False,
) -> FastAPI:
    if disconnect_grace_seconds < 0:
        raise ValueError("disconnect_grace_seconds cannot be negative")
    if dev_origin:
        parsed = urlsplit(dev_origin)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.port is None or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("Development origin must be a loopback HTTP origin")
    bus = EventBus()
    controller = controller or SessionController()
    previous_observer = controller.on_update

    def publish(snapshot: SessionSnapshot) -> None:
        bus.publish(snapshot)
        if previous_observer is not None:
            previous_observer(snapshot)

    controller.on_update = publish
    presence = Presence(controller, disconnect_grace_seconds)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.attach(asyncio.get_running_loop())
        try:
            yield
        finally:
            await presence.shutdown()
            await asyncio.to_thread(controller.shutdown)

    app = FastAPI(title="StudySync local service", lifespan=lifespan)
    app.state.controller = controller

    def safe_origin(host: str | None, origin: str | None, fetch_site: str | None) -> bool:
        if not host or host.split(":", 1)[0].lower() not in allowed_hosts:
            return False
        if fetch_site == "cross-site":
            return False
        if origin and origin not in (f"http://{host}", dev_origin):
            return False
        return True

    @app.middleware("http")
    async def local_request_guard(request: Request, call_next):
        if not safe_origin(
            request.headers.get("host"),
            request.headers.get("origin"),
            request.headers.get("sec-fetch-site"),
        ):
            return JSONResponse({"detail": "Local origin required"}, status_code=403)
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                declared_size = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid request size"}, status_code=400)
            if declared_size > 4096:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
            if len(await request.body()) > 4096:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def require_token(x_studysync_token: str | None = Header(default=None)) -> None:
        if not x_studysync_token or not secrets.compare_digest(x_studysync_token, token):
            raise HTTPException(status_code=403, detail="Session token required")

    def matching_snapshot(session_id: str) -> SessionSnapshot:
        snapshot = controller.snapshot()
        if snapshot is None or snapshot.session_id != session_id:
            raise HTTPException(status_code=404, detail="Session not found")
        return snapshot

    @app.get("/health")
    def health():
        snapshot = controller.snapshot()
        return {
            "service": "ready",
            "model_present": DEFAULT_MODEL_PATH.is_file(),
            "key_configured": bool(os.environ.get("GEMINI_API_KEY")),
            "session_status": snapshot.status if snapshot else "idle",
            "demo_mode": demo_mode,
        }

    @app.get("/bootstrap")
    def bootstrap():
        return {"token": token}

    @app.get("/sessions/current", dependencies=[Depends(require_token)])
    def current_session():
        snapshot = controller.snapshot()
        return asdict(snapshot) if snapshot else None

    @app.post("/sessions", dependencies=[Depends(require_token)])
    async def start_session(body: SessionRequest):
        try:
            snapshot = await asyncio.to_thread(
                controller.start, body.subject, body.goal, body.duration_minutes
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        presence.require_connection()
        return asdict(snapshot)

    @app.get("/sessions/{session_id}", dependencies=[Depends(require_token)])
    def get_session(session_id: str):
        return asdict(matching_snapshot(session_id))

    @app.post("/sessions/{session_id}/pause", dependencies=[Depends(require_token)])
    async def pause_session(session_id: str):
        matching_snapshot(session_id)
        try:
            return asdict(await asyncio.to_thread(controller.pause))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume", dependencies=[Depends(require_token)])
    async def resume_session(session_id: str):
        matching_snapshot(session_id)
        try:
            snapshot = await asyncio.to_thread(controller.resume)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        presence.require_connection()
        return asdict(snapshot)

    @app.post("/sessions/{session_id}/stop", dependencies=[Depends(require_token)])
    async def stop_session(session_id: str):
        matching_snapshot(session_id)
        return asdict(await asyncio.to_thread(controller.stop))

    @app.websocket("/sessions/{session_id}/events")
    async def session_events(websocket: WebSocket, session_id: str):
        if not safe_origin(
            websocket.headers.get("host"),
            websocket.headers.get("origin"),
            websocket.headers.get("sec-fetch-site"),
        ):
            await websocket.close(code=1008)
            return
        snapshot = controller.snapshot()
        if snapshot is None or snapshot.session_id != session_id:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            first = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            supplied = first.get("token") if isinstance(first, dict) else None
            if not isinstance(supplied, str) or not secrets.compare_digest(supplied, token):
                await websocket.close(code=1008)
                return
            presence.connected(session_id)
            queue = bus.subscribe(session_id)
            try:
                await websocket.send_json(bus.current(controller.snapshot()))
                while True:
                    incoming = asyncio.create_task(websocket.receive())
                    outgoing = asyncio.create_task(queue.get())
                    done, pending = await asyncio.wait(
                        (incoming, outgoing), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if incoming in done:
                        message = incoming.result()
                        if message["type"] == "websocket.disconnect":
                            break
                    if outgoing in done:
                        await websocket.send_json(outgoing.result())
            finally:
                bus.unsubscribe(queue)
                presence.disconnected(session_id)
        except WebSocketDisconnect:
            return
        except ValueError:
            await websocket.close(code=1008)
        except asyncio.TimeoutError:
            await websocket.close(code=1008)

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


app = create_app()
