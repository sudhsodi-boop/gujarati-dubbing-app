"""Rotating Gemini API key manager.

Puts a key into a cooldown box when a request hits a rate limit (429 /
quota exhausted) so free-tier keys can be rotated transparently.

Usage:
    rotator = KeyRotator(["key1", "key2", "key3"])
    result = rotator.execute(lambda api_key: call_gemini(api_key, ...))
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_QUOTA_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "rate limit",
    "quota",
    "Quota exceeded",
)


class NoKeysAvailable(RuntimeError):
    pass


class KeyRotator:
    def __init__(self, keys: list[str], cooldown_s: float = 90.0):
        keys = [k.strip() for k in keys if k and k.strip()]
        if not keys:
            raise NoKeysAvailable("No API keys configured.")
        self._keys = keys
        self._cooldown_s = cooldown_s
        self._cooldown_until: dict[str, float] = {}
        self._cycle = itertools.cycle(keys)
        self._lock = threading.Lock()

    # -- key lifecycle ----------------------------------------------------
    def _next_key(self) -> str | None:
        now = time.time()
        with self._lock:
            # one full lap over the cycle looking for a healthy key
            for _ in range(len(self._keys)):
                key = next(self._cycle)
                if self._cooldown_until.get(key, 0.0) <= now:
                    return key
        return None

    def report_failure(self, key: str) -> None:
        with self._lock:
            self._cooldown_until[key] = time.time() + self._cooldown_s

    # -- main entry point -------------------------------------------------
    def execute(self, fn: Callable[[str], T], max_attempts: int | None = None) -> T:
        """Run fn(api_key), rotating keys on quota errors.

        Non-quota exceptions are re-raised immediately.
        """
        attempts = max_attempts or max(len(self._keys) * 2, 2)
        last_exc: Exception | None = None
        for _ in range(attempts):
            key = self._next_key()
            if key is None:
                # every key cooling down: wait for the soonest release
                soonest = min(self._cooldown_until.values())
                time.sleep(min(max(soonest - time.time(), 1.0), 30.0))
                continue
            try:
                return fn(key)
            except Exception as exc:  # noqa: BLE001 - deliberate broad catch
                msg = f"{type(exc).__name__}: {exc}"
                if any(m in msg for m in _QUOTA_MARKERS):
                    self.report_failure(key)
                    last_exc = exc
                    continue
                raise
        raise NoKeysAvailable(
            f"All keys are rate-limited. Last error: {last_exc}"
        )


def load_keys_from_text(text: str) -> list[str]:
    """Accept keys pasted one-per-line OR comma-separated."""
    raw = text.replace(",", "\n").splitlines()
    return [k.strip() for k in raw if k.strip() and not k.strip().startswith("#")]
