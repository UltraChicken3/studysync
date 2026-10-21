"""One local study session and one camera owner for the future web service."""

import logging
from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import monotonic
from typing import Callable, Literal
from uuid import uuid4

import cv2

from study_buddy_inference import InferenceResult, StudentStateClassifier


logger = logging.getLogger(__name__)
SessionStatus = Literal["running", "paused", "stopped", "error"]


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    subject: str
    goal: str
    duration_seconds: float
    elapsed_seconds: float
    status: SessionStatus
    capture_status: str | None
    tentative_class: str | None
    error: str | None


class SessionController:
    """Own capture in a background thread; never store or publish frames."""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        sample_interval: float = 0.5,
        camera_factory: Callable[[int], object] = cv2.VideoCapture,
        classifier_factory: Callable[[], object] = StudentStateClassifier,
        clock: Callable[[], float] = monotonic,
        on_update: Callable[[SessionSnapshot], None] | None = None,
    ):
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.camera_factory = camera_factory
        self.classifier_factory = classifier_factory
        self.clock = clock
        self.on_update = on_update
        self._lock = RLock()
        self._classifier = None
        self._worker: Thread | None = None
        self._stop_signal: Event | None = None
        self._session_id: str | None = None
        self._subject = ""
        self._goal = ""
        self._duration = 0.0
        self._elapsed = 0.0
        self._run_started: float | None = None
        self._status: SessionStatus = "stopped"
        self._capture_status: str | None = None
        self._tentative_class: str | None = None
        self._error: str | None = None

    def snapshot(self) -> SessionSnapshot | None:
        with self._lock:
            if self._session_id is None:
                return None
            elapsed = self._elapsed
            if self._run_started is not None:
                elapsed += max(0.0, self.clock() - self._run_started)
            return SessionSnapshot(
                self._session_id,
                self._subject,
                self._goal,
                self._duration,
                min(elapsed, self._duration),
                self._status,
                self._capture_status,
                self._tentative_class,
                self._error,
            )

    def start(self, subject: str, goal: str, duration_minutes: float) -> SessionSnapshot:
        subject, goal = subject.strip(), goal.strip()
        if not subject or len(subject) > 120 or not goal or len(goal) > 500:
            raise ValueError("Enter a subject and goal within the length limits")
        if not 1 <= duration_minutes <= 240:
            raise ValueError("Session duration must be between 1 and 240 minutes")
        with self._lock:
            if self._status in ("running", "paused") or (
                self._worker is not None and self._worker.is_alive()
            ):
                raise RuntimeError("A study session already owns the camera")
            self._session_id = uuid4().hex
            self._subject, self._goal = subject, goal
            self._duration = duration_minutes * 60
            self._elapsed = 0.0
            self._run_started = self.clock()
            self._status = "running"
            self._capture_status = None
            self._tentative_class = None
            self._error = None
            self._start_worker()
        self._publish()
        return self.snapshot()

    def pause(self) -> SessionSnapshot:
        with self._lock:
            if self._status != "running":
                raise RuntimeError("Only a running session can be paused")
            self._freeze_clock()
            self._status = "paused"
            self._capture_status = None
            self._tentative_class = None
            self._stop_signal.set()
            worker = self._worker
        self._join_worker(worker)
        self._publish()
        return self.snapshot()

    def resume(self) -> SessionSnapshot:
        with self._lock:
            if self._status != "paused":
                raise RuntimeError("Only a paused session can be resumed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("The previous camera worker is still stopping")
            self._run_started = self.clock()
            self._status = "running"
            self._start_worker()
        self._publish()
        return self.snapshot()

    def stop(self) -> SessionSnapshot:
        with self._lock:
            if self._session_id is None:
                raise RuntimeError("No study session exists")
            if self._status == "running":
                self._freeze_clock()
            self._status = "stopped"
            self._capture_status = None
            self._tentative_class = None
            if self._stop_signal is not None:
                self._stop_signal.set()
            worker = self._worker
        self._join_worker(worker)
        self._publish()
        return self.snapshot()

    def shutdown(self) -> None:
        if self.snapshot() is not None:
            self.stop()

    def _freeze_clock(self) -> None:
        if self._run_started is not None:
            self._elapsed += max(0.0, self.clock() - self._run_started)
            self._run_started = None

    def _start_worker(self) -> None:
        signal = Event()
        self._stop_signal = signal
        self._worker = Thread(target=self._capture_loop, args=(signal,), daemon=True)
        self._worker.start()

    @staticmethod
    def _join_worker(worker: Thread | None) -> None:
        if worker is not None:
            worker.join(timeout=10)
            if worker.is_alive():
                raise TimeoutError("Camera worker did not stop within ten seconds")

    def _capture_loop(self, signal: Event) -> None:
        camera = None
        failure = None
        completed = False
        try:
            if self._classifier is None:
                self._classifier = self.classifier_factory()
            if signal.is_set():
                return
            camera = self.camera_factory(self.camera_index)
            if not camera.isOpened():
                raise RuntimeError("Could not open the camera")
            while not signal.is_set():
                if self.snapshot().elapsed_seconds >= self._duration:
                    completed = True
                    break
                ok, frame = camera.read()
                frame = cv2.flip(frame, 1) if ok and frame is not None and frame.size else None
                result: InferenceResult = self._classifier.infer_frame(frame)
                if result.capture_status == "camera_error":
                    raise RuntimeError("Camera feed stopped")
                with self._lock:
                    if signal.is_set():
                        break
                    self._capture_status = result.capture_status
                    self._tentative_class = result.tentative_class
                self._publish()
                frame = None
                signal.wait(self.sample_interval)
        except Exception as exc:
            failure = str(exc) or type(exc).__name__
        finally:
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    logger.exception("Camera release failed")
            with self._lock:
                if not signal.is_set() and self._status == "running":
                    self._freeze_clock()
                    self._status = "stopped" if completed else "error"
                    self._capture_status = None
                    self._tentative_class = None
                    self._error = None if completed else failure
                    publish = True
                else:
                    publish = False
            if publish:
                self._publish()

    def _publish(self) -> None:
        if self.on_update is not None:
            try:
                self.on_update(self.snapshot())
            except Exception:
                # An observer must not retain the camera or kill inference.
                logger.exception("Session observer failed")
