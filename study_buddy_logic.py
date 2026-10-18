"""Small, dependency-free decisions used by the webcam application."""

from math import isfinite


CLASS_NAMES = (
    "Bored", "Confused", "Drowsy", "Focused", "Frustrated", "Looking Away"
)


def classify_probabilities(probabilities, min_probability=0.65, min_margin=0.15):
    """Return a tentative class only when the model is reasonably decisive."""
    scores = list(probabilities)
    if len(scores) != len(CLASS_NAMES) or any(not isfinite(float(x)) for x in scores):
        raise ValueError("Expected six finite model probabilities")
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if ranked[0][1] < min_probability or ranked[0][1] - ranked[1][1] < min_margin:
        return "Uncertain"
    return CLASS_NAMES[ranked[0][0]]


def help_mode(state):
    if state in ("Confused", "Frustrated"):
        return "confusion"
    if state == "Bored":
        return "boredom"
    return None


class CueTracker:
    """Offer help once per sustained cue, then wait for it to clear."""

    def __init__(self, hold_seconds=2.0, reset_seconds=2.0, cooldown_seconds=60.0):
        self.hold_seconds = hold_seconds
        self.reset_seconds = reset_seconds
        self.cooldown_seconds = cooldown_seconds
        self.candidate = None
        self.candidate_since = None
        self.clear_since = None
        self.last_offer_at = float("-inf")
        self.armed = True

    def observe(self, mode, now):
        if mode is None:
            self.candidate = None
            self.candidate_since = None
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= self.reset_seconds:
                self.armed = True
            return None

        self.clear_since = None
        if mode != self.candidate:
            self.candidate = mode
            self.candidate_since = now
        if (self.armed and now - self.candidate_since >= self.hold_seconds
                and now - self.last_offer_at >= self.cooldown_seconds):
            self.armed = False
            self.last_offer_at = now
            return mode
        return None
