"""Loopback-only StudySync API and event stream for the local web interface."""

import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

import cv2
from anyio import EndOfStream
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from study_buddy_inference import DEFAULT_MODEL_PATH
from study_session import SessionController, SessionSnapshot
from study_tutor import (
    MAX_EXCERPT, MAX_TOPIC, MAX_FOLLOWUP_CONTEXT, MAX_REPLY,
    TutorError, TutorService, make_prompt, make_followup_prompt, prompt_digest,
)


FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
logger = logging.getLogger(__name__)


class SessionRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=120)
    goal: str = Field(min_length=1, max_length=500)
    duration_minutes: float | None = Field(default=None, ge=1, le=240)
    timer_mode: Literal["timer", "stopwatch"] = "timer"
    monitoring: bool = True


class HelpText(BaseModel):
    mode: str
    topic: str = Field(min_length=1, max_length=MAX_TOPIC)
    excerpt: str = Field(default="", max_length=MAX_EXCERPT)


class HelpRequest(HelpText):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    approved_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    offer_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class FollowUpText(BaseModel):
    guide_request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    guide_context: str = Field(min_length=1, max_length=MAX_FOLLOWUP_CONTEXT)
    reply: str = Field(min_length=1, max_length=MAX_REPLY)


class FollowUpRequest(FollowUpText):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    approved_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeedbackRequest(BaseModel):
    choice: str


class OfferActionRequest(BaseModel):
    action: str


class OfferClassesRequest(BaseModel):
    enabled: list[str] = Field(max_length=4)


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


class PreviewBus:
    """Encode small transient JPEGs only while a local viewer is subscribed."""

    def __init__(self):
        self._lock = Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[bytes]] = set()
        self._last_encoded = 0.0

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(queue)
        while not queue.empty():
            queue.get_nowait()

    def publish(self, frame) -> None:
        with self._lock:
            now = time.monotonic()
            if (not self._subscribers or self._loop is None
                    or now - self._last_encoded < 0.25):
                return
            self._last_encoded = now
            loop = self._loop
        height, width = frame.shape[:2]
        if width > 480:
            frame = cv2.resize(frame, (480, max(1, round(height * 480 / width))))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok:
            loop.call_soon_threadsafe(self._fan_out, encoded.tobytes())

    def _fan_out(self, jpeg: bytes) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for queue in subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(jpeg)


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
        if (snapshot and snapshot.status in ("running", "study_only")
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
                    and snapshot.status in ("running", "study_only")
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
    tutor: TutorService | None = None,
) -> FastAPI:
    if disconnect_grace_seconds < 0:
        raise ValueError("disconnect_grace_seconds cannot be negative")
    if dev_origin:
        parsed = urlsplit(dev_origin)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.port is None or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("Development origin must be a loopback HTTP origin")
    bus = EventBus()
    preview_bus = PreviewBus()
    controller = controller or SessionController()
    previous_observer = controller.on_update
    previous_preview = controller.on_preview

    def publish(snapshot: SessionSnapshot) -> None:
        bus.publish(snapshot)
        if previous_observer is not None:
            previous_observer(snapshot)

    controller.on_update = publish

    def publish_preview(frame) -> None:
        preview_bus.publish(frame)
        if previous_preview is not None:
            previous_preview(frame)

    controller.on_preview = publish_preview
    presence = Presence(controller, disconnect_grace_seconds)
    tutor = tutor or TutorService()
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.attach(asyncio.get_running_loop())
        preview_bus.attach(asyncio.get_running_loop())
        try:
            yield
        finally:
            await presence.shutdown()
            await asyncio.to_thread(controller.shutdown)

    app = FastAPI(title="StudySync local service", lifespan=lifespan)
    app.state.controller = controller
    app.state.tutor = tutor

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
            if declared_size > 16384:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
            if len(await request.body()) > 16384:
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

    def active_snapshot(session_id: str) -> SessionSnapshot:
        snapshot = matching_snapshot(session_id)
        if snapshot.status not in ("running", "study_only", "paused"):
            raise HTTPException(status_code=409, detail="Start an active study session first")
        return snapshot

    def help_prompt(body: HelpText) -> str:
        try:
            return make_prompt(body.mode, body.topic, body.excerpt)
        except TutorError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.message) from exc

    @app.get("/health")
    def health():
        snapshot = controller.snapshot()
        return {
            "service": "ready",
            "model_present": DEFAULT_MODEL_PATH.is_file(),
            "key_configured": bool(os.environ.get("GEMINI_API_KEY")),
            "tutor_ready": tutor.ready(),
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
                controller.start, body.subject, body.goal, body.duration_minutes,
                body.monitoring, body.timer_mode,
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

    @app.post("/sessions/{session_id}/help/preview", dependencies=[Depends(require_token)])
    def preview_help(session_id: str, body: HelpText):
        active_snapshot(session_id)
        prompt = help_prompt(body)
        return {"prompt": prompt, "digest": prompt_digest(prompt)}

    @app.post("/sessions/{session_id}/help", dependencies=[Depends(require_token)])
    async def request_help(session_id: str, body: HelpRequest):
        active_snapshot(session_id)
        if body.offer_id and not controller.offer_was_accepted(body.offer_id):
            raise HTTPException(status_code=409, detail="Accept the offer before using it for a guide")
        try:
            return await tutor.ask(
                session_id, body.request_id, body.mode, body.topic,
                body.excerpt, body.approved_digest,
            )
        except TutorError as exc:
            raise HTTPException(status_code=exc.status, detail={
                "code": exc.code, "message": exc.message,
            }) from exc

    @app.post("/sessions/{session_id}/followup/preview", dependencies=[Depends(require_token)])
    def preview_followup(session_id: str, body: FollowUpText):
        active_snapshot(session_id)
        if not tutor.has_completed(session_id, body.guide_request_id):
            raise HTTPException(status_code=409, detail="A completed guide is required for a reply")
        try:
            prompt = make_followup_prompt(body.guide_context, body.reply)
        except TutorError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.message) from exc
        return {"prompt": prompt, "digest": prompt_digest(prompt)}

    @app.post("/sessions/{session_id}/followup", dependencies=[Depends(require_token)])
    async def request_followup(session_id: str, body: FollowUpRequest):
        active_snapshot(session_id)
        try:
            return await tutor.ask_followup(
                session_id, body.guide_request_id, body.request_id,
                body.guide_context, body.reply, body.approved_digest,
            )
        except TutorError as exc:
            raise HTTPException(status_code=exc.status, detail={
                "code": exc.code, "message": exc.message,
            }) from exc

    @app.post("/sessions/{session_id}/feedback", dependencies=[Depends(require_token)])
    def submit_feedback(session_id: str, body: FeedbackRequest):
        matching_snapshot(session_id)
        if body.choice not in (
            "useful", "irrelevant", "too_easy", "too_hard", "incorrect", "wrong_cue"
        ):
            raise HTTPException(status_code=422, detail="Choose a valid feedback option")
        return tutor.feedback(session_id, body.choice)

    @app.post("/sessions/{session_id}/offers/{offer_id}/action", dependencies=[Depends(require_token)])
    def offer_action(session_id: str, offer_id: str, body: OfferActionRequest):
        active_snapshot(session_id)
        try:
            result = controller.act_offer(offer_id, body.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if body.action == "wrong_cue":
            tutor.feedback(session_id, "wrong_cue")
        return result

    @app.post("/sessions/{session_id}/offer-classes", dependencies=[Depends(require_token)])
    def offer_classes(session_id: str, body: OfferClassesRequest):
        active_snapshot(session_id)
        try:
            return asdict(controller.set_offer_classes(set(body.enabled)))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        except (WebSocketDisconnect, EndOfStream):
            return
        except ValueError:
            await websocket.close(code=1008)
        except asyncio.TimeoutError:
            await websocket.close(code=1008)

    @app.websocket("/sessions/{session_id}/preview")
    async def session_preview(websocket: WebSocket, session_id: str):
        if not safe_origin(
            websocket.headers.get("host"),
            websocket.headers.get("origin"),
            websocket.headers.get("sec-fetch-site"),
        ):
            await websocket.close(code=1008)
            return
        snapshot = controller.snapshot()
        if snapshot is None or snapshot.session_id != session_id or snapshot.status != "running":
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
            queue = preview_bus.subscribe()
            try:
                while True:
                    incoming = asyncio.create_task(websocket.receive())
                    outgoing = asyncio.create_task(queue.get())
                    done, pending = await asyncio.wait(
                        (incoming, outgoing), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if incoming in done and incoming.result()["type"] == "websocket.disconnect":
                        break
                    if outgoing in done:
                        current = controller.snapshot()
                        if (current is None or current.session_id != session_id
                                or current.status != "running"):
                            break
                        await websocket.send_bytes(outgoing.result())
            finally:
                preview_bus.unsubscribe(queue)
                presence.disconnected(session_id)
        except (WebSocketDisconnect, EndOfStream):
            return
        except (ValueError, asyncio.TimeoutError):
            await websocket.close(code=1008)

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


app = create_app()
