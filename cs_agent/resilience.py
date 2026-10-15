"""Circuit breaker for the judgment layer.

Jev is an external early-access service whose rate limits "adjust without
notice". Without this, every Jev error is a fail-safe handoff, so a Jev
outage would send 100% of conversations to human agents.

    closed     Jev answers. Errors and slow answers (> slow_ms) are strikes;
               an error falls back to the LLM judge for that same turn.
    open       after `failure_threshold` strikes in a row: Jev is not called,
               every turn uses the LLM judge, for `cooldown_s`.
    half-open  after the cooldown, one turn probes Jev: success closes the
               circuit, failure opens it for another cooldown.

Every judgment produced by the backup carries `degraded` (why), so the UI,
traces and logs show it. If the backup fails too, the error propagates and
the gate's fail-safe hands the turn to a human, as before.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .judge import JudgeError, Judgment

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class ResilientJudge:
    def __init__(self, primary, fallback=None, failure_threshold: int = 3, slow_ms: float = 3000.0,
                 cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.primary = primary
        self.fallback = fallback
        self.failure_threshold = failure_threshold
        self.slow_ms = slow_ms
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probing = False
        self._last_error = ""
        self._primary_calls = 0
        self._fallback_calls = 0

    # -- breaker bookkeeping (all under the lock) --

    def _try_primary(self) -> bool:
        with self._lock:
            if self._state == CLOSED:
                return True
            if self._state == OPEN and self.clock() - self._opened_at >= self.cooldown_s:
                self._state = HALF_OPEN
            if self._state == HALF_OPEN and not self._probing:
                self._probing = True      # exactly one probe at a time
                return True
            return False

    def _strike(self, error: str) -> None:
        with self._lock:
            self._failures += 1
            self._last_error = error
            if self._state == HALF_OPEN or self._failures >= self.failure_threshold:
                self._state, self._opened_at = OPEN, self.clock()
            self._probing = False

    def _success(self) -> None:
        with self._lock:
            self._state, self._failures, self._probing = CLOSED, 0, False

    # -- public --

    def judge(self, state: dict) -> Judgment:
        if not self._try_primary():
            return self._use_fallback(state, "熔断中")
        started = self.clock()
        with self._lock:
            self._primary_calls += 1
        try:
            judgment = self.primary.judge(state)
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"[:120]
            self._strike(reason)
            return self._use_fallback(state, reason)
        elapsed_ms = (self.clock() - started) * 1000
        if elapsed_ms > self.slow_ms:
            self._strike(f"太慢：{elapsed_ms:.0f}ms > {self.slow_ms:.0f}ms")  # answer still used
        else:
            self._success()
        return judgment

    def _use_fallback(self, state: dict, reason: str) -> Judgment:
        if self.fallback is None:
            raise JudgeError(f"主判断层不可用（{reason}），且没有备用判断层")
        with self._lock:
            self._fallback_calls += 1
        judgment = self.fallback.judge(state)     # its errors propagate -> gate fail-safe
        judgment.degraded = reason
        return judgment

    def status(self) -> dict:
        with self._lock:
            retry = max(0.0, self.cooldown_s - (self.clock() - self._opened_at)) if self._state == OPEN else None
            return {"state": self._state, "consecutive_failures": self._failures,
                    "failure_threshold": self.failure_threshold, "cooldown_s": self.cooldown_s,
                    "retry_in_s": retry, "last_error": self._last_error,
                    "primary_calls": self._primary_calls, "fallback_calls": self._fallback_calls}
