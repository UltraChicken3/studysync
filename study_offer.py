"""Provisional, time-based offer policy around the unchanged six-class model."""

from collections import deque
from dataclasses import dataclass, asdict
from math import isfinite
from uuid import uuid4

from study_buddy_logic import CLASS_NAMES, classify_probabilities


OFFER_MODES = {
    "Confused": ("explain", "Would a simpler explanation help?"),
    "Frustrated": ("steps", "Want to break this into smaller steps?"),
    "Bored": ("engage", "Want a different angle on this topic?"),
    "Drowsy": ("break", "Would a short break help?"),
}
DEFAULT_ENABLED = frozenset(("Confused", "Frustrated", "Bored"))


@dataclass(frozen=True)
class Offer:
    offer_id: str
    cue_class: str
    guide_mode: str
    message: str
    sample_count: int
    window_seconds: float

    def as_payload(self) -> dict:
        return asdict(self)


class OfferPolicy:
    """Use a rolling probability mean and one offer per sustained cue episode."""

    def __init__(
        self, *, window_seconds: float = 3.0, min_samples: int = 4,
        hold_seconds: float = 2.0, max_gap_seconds: float = 1.5,
        clear_seconds: float = 2.0, cooldown_seconds: float = 60.0,
        enabled: frozenset[str] = DEFAULT_ENABLED,
    ):
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.hold_seconds = hold_seconds
        self.max_gap_seconds = max_gap_seconds
        self.clear_seconds = clear_seconds
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.samples: deque[tuple[float, tuple[float, ...]]] = deque()
        self.last_valid_at: float | None = None
        self.candidate: str | None = None
        self.candidate_since: float | None = None
        self.clear_since: float | None = None
        self.armed = True
        self.last_offer_at = float("-inf")
        self.snoozed_until = float("-inf")
        self.active: Offer | None = None

    def set_enabled(self, enabled: set[str]) -> None:
        if not enabled <= OFFER_MODES.keys():
            raise ValueError("Unknown offer class")
        self.enabled = frozenset(enabled)
        if self.active and self.active.cue_class not in self.enabled:
            self.active = None
        self.reset_evidence()

    def reset_evidence(self) -> None:
        self.samples.clear()
        self.last_valid_at = None
        self.candidate = None
        self.candidate_since = None
        self.clear_since = None

    def observe(self, scores: tuple[float, ...] | None, now: float) -> Offer | None:
        if scores is None:
            if self.last_valid_at is not None and now - self.last_valid_at > self.max_gap_seconds:
                self.reset_evidence()
                self._clear(now)
            elif self.last_valid_at is None and self.clear_since is not None:
                self._clear(now)
            return None
        if len(scores) != len(CLASS_NAMES) or any(not isfinite(float(x)) for x in scores):
            raise ValueError("Expected six finite model scores")
        if self.last_valid_at is not None and now - self.last_valid_at > self.max_gap_seconds:
            self.reset_evidence()
        self.last_valid_at = now
        self.samples.append((now, tuple(float(x) for x in scores)))
        while self.samples and now - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()
        if len(self.samples) < self.min_samples:
            return None
        mean = tuple(sum(sample[index] for _, sample in self.samples) / len(self.samples)
                     for index in range(len(CLASS_NAMES)))
        decision = classify_probabilities(mean)
        if decision not in self.enabled:
            self._clear(now)
            return None
        self.clear_since = None
        if decision != self.candidate:
            self.candidate = decision
            self.candidate_since = now
        if (self.armed and self.active is None and now >= self.snoozed_until
                and now - self.candidate_since >= self.hold_seconds
                and now - self.last_offer_at >= self.cooldown_seconds):
            mode, message = OFFER_MODES[decision]
            self.active = Offer(
                uuid4().hex, decision, mode, message,
                len(self.samples), now - self.samples[0][0],
            )
            self.armed = False
            self.last_offer_at = now
            return self.active
        return None

    def _clear(self, now: float) -> None:
        self.candidate = None
        self.candidate_since = None
        if self.clear_since is None:
            self.clear_since = now
        if now - self.clear_since >= self.clear_seconds:
            self.armed = True

    def act(self, offer_id: str, action: str, now: float) -> Offer:
        if self.active is None or self.active.offer_id != offer_id:
            raise ValueError("Offer is no longer active")
        if action not in ("accept", "dismiss", "snooze", "wrong_cue"):
            raise ValueError("Unknown offer action")
        offer = self.active
        self.active = None
        if action == "snooze":
            self.snoozed_until = now + 300
        return offer
