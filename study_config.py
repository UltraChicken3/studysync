"""Read local Gemini settings without logging or committing credentials."""

import os
from pathlib import Path


LOCAL_CONFIG = Path(__file__).resolve().parent / ".env"
GEMINI_SETTINGS = frozenset({"GEMINI_API_KEY", "STUDYSYNC_GEMINI_FREE_TIER_CONFIRMED"})


def get_setting(name: str, config_path: Path = LOCAL_CONFIG) -> str | None:
    """Process environment wins over the ignored, plain key=value project file."""
    if name not in GEMINI_SETTINGS:
        raise ValueError("Unknown Gemini setting")
    if name in os.environ:
        return os.environ[name].strip()
    if not config_path.is_file():
        return None
    value = None
    for line in config_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() != name:
            continue
        candidate = candidate.strip()
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
            candidate = candidate[1:-1]
        value = candidate
    return value
