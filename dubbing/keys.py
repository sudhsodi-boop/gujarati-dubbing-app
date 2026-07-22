"""Rotating Gemini API key manager.

Puts a key into a cooldown box when a request hits a rate limit (429 /
quota exhausted) so free-tier keys can be rotated transparently.

Usage:
    rotator = KeyRotator(["key1", "key2", "key3"])
    result = rotator.execute(lambda api_key: call_gemini(api_key, ...))
"""

from __future__ import annotations

import re
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

# temporary blips (overloaded Google backend, timeouts) — retry, don't die
_TRANSIENT_MARKERS = (
    "503", "500", "UNAVAILABLE", "overloaded", "Try again",
    "timeout", "timed out", "DEADLINE", "temporarily", "Internal error",
    # malformed JSON from the model -> retry (usually with another key)
    "JSONDecodeError", "Expecting", "No JSON array",
    "Unterminated", "Extra data", "Invalid control character",
)


class NoKeysAvailable(RuntimeError):
    pass


class AllKeysQuotaExhausted(RuntimeError):
    """Raised when ALL keys have hit daily quota (429).
    Contains the estimated seconds until quota reset (from API retryDelay or heuristic).
    The job should pause and ask the user whether to wait."""
    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class KeyRotator:
    def __init__(self, keys: list[str], cooldown_s: float = 90.0):
        keys = [k.strip() for k in keys if k and k.strip()]
        if not keys:
            raise NoKeysAvailable("No API keys configured.")
        self._keys = keys
        self._cooldown_s = cooldown_s
        self._cooldown_until: dict[str, float] = {}
        self._uses: dict[str, int] = {}
        self._last_use: dict[str, float] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    def stats(self) -> dict[str, int]:
        """Successful calls per key (tail-masked) — proof of rotation."""
        return {f"…{k[-4:]}": self._uses.get(k, 0) for k in self._keys}

    def _next_key(self) -> str | None:
        """FAIR rotation (least-used-first):
        1. among ready keys → pick the one with fewest successful uses
        2. tie → the one idle the longest
        3. tie → lowest index (k1, k2, k3 … order)"""
        now = time.time()
        with self._lock:
            ready = [
                k for k in self._keys
                if self._cooldown_until.get(k, 0.0) <= now
            ]
            if not ready:
                return None
            ready.sort(key=lambda k: (self._uses.get(k, 0),
                                      self._last_use.get(k, 0.0),
                                      self._keys.index(k)))
            return ready[0]

    def soonest_ready_in(self) -> float:
        """Seconds until the nearest key leaves cooldown (0 if one is ready)."""
        now = time.time()
        with self._lock:
            for k in self._keys:
                if self._cooldown_until.get(k, 0.0) <= now:
                    return 0.0
            if not self._cooldown_until:
                return 0.0
            return max(0.0, min(self._cooldown_until.values()) - now)

    def report_failure(self, key: str) -> None:
        with self._lock:
            self._cooldown_until[key] = time.time() + self._cooldown_s

    def _extract_retry_delay(self, msg: str) -> float | None:
        """Parse 'retryDelay': '50s' or 'retry in 50 seconds' from error message."""
        # Try structured retryDelay first (from google.rpc.RetryInfo)
        m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", msg)
        if m:
            return float(m.group(1))
        # Try "retry in X seconds" or "Please retry in Xs"
        m = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r"Please retry in\s+(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    # -- main entry point -------------------------------------------------
    def execute(self, fn: Callable[[str], T], max_attempts: int | None = None) -> T:
        """Run fn(api_key): rotate keys on quota errors, RETRY on transient
        errors (503/overload/timeout). Only truly fatal errors propagate.
        
        If ALL keys hit quota (429), raises AllKeysQuotaExhausted with retry_after_s
        so the caller can decide whether to wait (hours) or abort."""
        attempts = max_attempts or max(len(self._keys) * 3, 4)
        last_exc: Exception | None = None
        quota_keys: set[str] = set()  # keys that have hit 429 in this call
        
        for _ in range(attempts):
            key = self._next_key()
            if key is None:
                # every key cooling down: wait for the soonest release
                soonest = min(self._cooldown_until.values())
                time.sleep(min(max(soonest - time.time(), 1.0), 30.0))
                continue
            try:
                result = fn(key)
                self._uses[key] = self._uses.get(key, 0) + 1
                self._last_use[key] = time.time()
                # Success clears this key's quota mark for this call
                quota_keys.discard(key)
                return result
            except Exception as exc:  # noqa: BLE001 - deliberate broad catch
                msg = f"{type(exc).__name__}: {exc}"
                # a FAILED key goes to the back of the queue: stamp its attempt
                # so fair least-used selection tries your OTHER keys next
                self._last_use[key] = time.time()
                if any(m in msg for m in _QUOTA_MARKERS):
                    self.report_failure(key)
                    quota_keys.add(key)
                    last_exc = exc
                    # If ALL keys have now hit quota, we're truly exhausted
                    if len(quota_keys) >= len(self._keys):
                        retry_after = self._extract_retry_delay(msg)
                        # If retry_after is short (< 5 min), just wait and retry once more
                        if retry_after is not None and retry_after < 300:
                            time.sleep(min(retry_after + 2, 300))
                            quota_keys.clear()  # give keys another chance after wait
                            continue
                        # Long wait (hours) — let caller decide
                        raise AllKeysQuotaExhausted(
                            f"All {len(self._keys)} keys hit daily quota. Last error: {exc}",
                            retry_after_s=retry_after
                        )
                    continue
                if any(m in msg for m in _TRANSIENT_MARKERS):
                    time.sleep(3.0)   # backend blip / bad JSON — retry (next attempt uses another key)
                    last_exc = exc
                    continue
                raise
        raise NoKeysAvailable(
            f"All keys busy/failing after {attempts} attempts. Last error: {last_exc}"
        )


def load_keys_from_text(text: str) -> list[str]:
    """Accept keys pasted one-per-line OR comma-separated."""
    raw = text.replace(",", "\n").splitlines()
    return [k.strip() for k in raw if k.strip() and not k.strip().startswith("#")]
