"""One local study session and one camera owner for the future web service."""

import logging
from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import monotonic
from typing import Callable, Literal
from uuid import uuid4

import cv2

from study_buddy_inference import InferenceResult, StudentStateClassifier
from study_offer import OFFER_MODES, OfferPolicy


logger = logging.getLogger(__name__)
SessionStatus = Literal["running", "study_only", "paused", "stopped", "error"]
TimerMode = Literal["timer", "stopwatch"]


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    subject: str
    goal: str
    duration_seconds: float | None
    elapsed_seconds: float
    timer_mode: TimerMode
    status: SessionStatus
    capture_status: str | None
    tentative_class: str | None
    error: str | None
    offer: dict | None = None
    enabled_offers: tuple[str, ...] = ()


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
        on_preview: Callable[[object], None] | None = None,
        offer_policy_factory: Callable[[], OfferPolicy] = OfferPolicy,
    ):
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.camera_factory = camera_factory
        self.classifier_factory = classifier_factory
        self.clock = clock
        self.on_update = on_update
        self.on_preview = on_preview
        self.offer_policy_factory = offer_policy_factory
        self._lock = RLock()
        self._classifier = None
        self._worker: Thread | None = None
        self._stop_signal: Event | None = None
        self._session_id: str | None = None
        self._subject = ""
        self._goal = ""
        self._duration: float | None = 0.0
        self._timer_mode: TimerMode = "timer"
        self._monitoring_enabled = True
        self._elapsed = 0.0
        self._run_started: float | None = None
        self._status: SessionStatus = "stopped"
        self._capture_status: str | None = None
        self._tentative_class: str | None = None
        self._error: str | None = None
        self._offer_policy: OfferPolicy | None = None
        self._accepted_offer_ids: set[str] = set()

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
                min(elapsed, self._duration) if self._duration is not None else elapsed,
                self._timer_mode,
                self._status,
                self._capture_status,
                self._tentative_class,
                self._error,
                self._offer_policy.active.as_payload() if self._offer_policy and self._offer_policy.active else None,
                tuple(name for name in OFFER_MODES if self._offer_policy and name in self._offer_policy.enabled),
            )

    def start(self, subject: str, goal: str, duration_minutes: float | None,
              monitoring: bool = True, timer_mode: TimerMode = "timer") -> SessionSnapshot:
        subject, goal = subject.strip(), goal.strip()
        if not subject or len(subject) > 120 or not goal or len(goal) > 500:
            raise ValueError("Enter a subject and goal within the length limits")
        if timer_mode not in ("timer", "stopwatch"):
            raise ValueError("Choose timer or stopwatch")
        if timer_mode == "timer" and (duration_minutes is None or not 1 <= duration_minutes <= 240):
            raise ValueError("Session duration must be between 1 and 240 minutes")
        with self._lock:
            if self._status in ("running", "study_only", "paused") or (
                self._worker is not None and self._worker.is_alive()
            ):
                raise RuntimeError("A study session is already active")
            self._session_id = uuid4().hex
            self._subject, self._goal = subject, goal
            self._duration = duration_minutes * 60 if timer_mode == "timer" else None
            self._timer_mode = timer_mode
            self._monitoring_enabled = monitoring
            self._elapsed = 0.0
            self._run_started = self.clock()
            self._status = "running" if monitoring else "study_only"
            self._worker = None
            self._stop_signal = None
            self._capture_status = None
            self._tentative_class = None
            self._error = None
            self._offer_policy = self.offer_policy_factory()
            self._accepted_offer_ids.clear()
            if monitoring:
                self._start_worker()
            else:
                self._start_timer_worker()
        self._publish()
        return self.snapshot()

    def pause(self) -> SessionSnapshot:
        with self._lock:
            if self._status not in ("running", "study_only"):
                raise RuntimeError("Only an active session can be paused")
            self._freeze_clock()
            self._status = "paused"
            self._capture_status = None
            self._tentative_class = None
            if self._offer_policy:
                self._offer_policy.reset_evidence()
            if self._stop_signal is not None:
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
            self._status = "running" if self._monitoring_enabled else "study_only"
            if self._monitoring_enabled:
                self._start_worker()
            else:
                self._start_timer_worker()
        self._publish()
        return self.snapshot()

    def stop(self) -> SessionSnapshot:
        with self._lock:
            if self._session_id is None:
                raise RuntimeError("No study session exists")
            if self._run_started is not None:
                self._freeze_clock()
            self._status = "stopped"
            self._capture_status = None
            self._tentative_class = None
            if self._offer_policy:
                self._offer_policy.active = None
                self._offer_policy.reset_evidence()
            if self._stop_signal is not None:
                self._stop_signal.set()
            worker = self._worker
        self._join_worker(worker)
        self._publish()
        return self.snapshot()

    def shutdown(self) -> None:
        if self.snapshot() is not None:
            self.stop()

    def act_offer(self, offer_id: str, action: str) -> dict:
        with self._lock:
            if self._status not in ("running", "study_only", "paused") or self._offer_policy is None:
                raise RuntimeError("No active study session")
            try:
                offer = self._offer_policy.act(offer_id, action, self.clock())
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            if action == "accept":
                self._accepted_offer_ids.add(offer_id)
        self._publish()
        return {"action": action, "offer": offer.as_payload()}

    def offer_was_accepted(self, offer_id: str) -> bool:
        with self._lock:
            return offer_id in self._accepted_offer_ids

    def set_offer_classes(self, enabled: set[str]) -> SessionSnapshot:
        with self._lock:
            if self._status not in ("running", "study_only", "paused") or self._offer_policy is None:
                raise RuntimeError("No active study session")
            self._offer_policy.set_enabled(enabled)
        self._publish()
        return self.snapshot()

    def _freeze_clock(self) -> None:
        if self._run_started is not None:
            self._elapsed += max(0.0, self.clock() - self._run_started)
            self._run_started = None

    def _start_worker(self) -> None:
        signal = Event()
        self._stop_signal = signal
        self._worker = Thread(target=self._capture_loop, args=(signal,), daemon=True)
        self._worker.start()

    def _start_timer_worker(self) -> None:
        signal = Event()
        self._stop_signal = signal
        self._worker = Thread(target=self._timer_loop, args=(signal,), daemon=True)
        self._worker.start()

    def _timer_loop(self, signal: Event) -> None:
        while not signal.wait(0.5):
            with self._lock:
                if self._status != "study_only":
                    return
                if self._duration is not None and self.snapshot().elapsed_seconds >= self._duration:
                    self._freeze_clock()
                    self._status = "stopped"
                    complete = True
                else:
                    complete = False
            self._publish()
            if complete:
                return

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
                if self._duration is not None and self.snapshot().elapsed_seconds >= self._duration:
                    completed = True
                    break
                ok, frame = camera.read()
                frame = cv2.flip(frame, 1) if ok and frame is not None and frame.size else None
                result: InferenceResult = self._classifier.infer_frame(frame)
                if result.capture_status == "camera_error":
                    raise RuntimeError("Camera feed stopped")
                if frame is not None and self.on_preview is not None and not signal.is_set():
                    try:
                        self.on_preview(frame)
                    except Exception:
                        logger.exception("Camera preview failed")
                with self._lock:
                    if signal.is_set():
                        break
                    self._capture_status = result.capture_status
                    self._tentative_class = result.tentative_class
                    if self._offer_policy:
                        self._offer_policy.observe(
                            result.scores if result.capture_status == "ok" else None,
                            self.clock(),
                        )
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
                    if self._offer_policy:
                        self._offer_policy.active = None
                        self._offer_policy.reset_evidence()
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
