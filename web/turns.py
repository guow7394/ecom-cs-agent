"""One browser chat session.

The agent runs synchronously and may stop halfway to ask the customer to
confirm a refund. A web page can't be called back like a terminal input(),
so each turn runs on a worker thread:

    browser ── POST /messages ──► start_turn() ──► worker thread: agent.handle(text)
       ▲                              │                 │ session.emit(...)  (hooks)
       │  SSE: one event per step     ▼                 ▼
       └──────────────────────── queue.Queue ◄──── _on_event()

    confirm_fn (called by the permission gate on the worker thread)
       emits confirm_request, then waits on a threading.Event
       until POST /confirm calls resolve(), or the timeout says "no".
"""
from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, Optional

from cs_agent.agent import CustomerServiceAgent, build_judge, build_llm
from cs_agent.compact import Compactor, llm_summarizer
from cs_agent.config import default_needs_human_threshold
from cs_agent.judge import GatePolicy
from cs_agent.session import new_session

CONFIRM_TIMEOUT = 120.0
MAX_SESSIONS = 20
_END_OF_STREAM = object()


class TurnRejected(Exception):
    """The session can't take a message right now (a turn is running, or it was handed off)."""


class WebSession:
    def __init__(self, customer_id: str, judge: str, threshold: float, mock: bool,
                 llm=None, confirm_timeout: float = CONFIRM_TIMEOUT, shared_judge=None):
        self.id = uuid.uuid4().hex[:12]
        self.confirm_timeout = confirm_timeout
        # Fresh copy of the seeded memory per session, so every demo starts from the same data.
        self.memory_dir = tempfile.mkdtemp(prefix="cs-web-")
        session = new_session(customer_id, Path(self.memory_dir), confirm_fn=self._confirm)
        session.on_event = self._on_event
        llm = llm or build_llm(mock)
        self.agent = CustomerServiceAgent(
            llm, session, judge=shared_judge or build_judge(judge, llm, mock, resilient=True),
            policy=GatePolicy(threshold=threshold, needs_human_threshold=default_needs_human_threshold()),
            compactor=Compactor(llm_summarizer(llm)))
        self.ended = False
        self._turn_lock = threading.Lock()
        self._events: Optional[queue.Queue] = None
        self._pending: Dict[str, dict] = {}
        self._pending_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._turn_lock.locked()

    # -- hooks and confirmation (called on the worker thread) --

    def _on_event(self, event_type: str, data: dict) -> None:
        events = self._events
        if events is not None:
            events.put({"type": event_type, **data})

    def _confirm(self, text: str) -> bool:
        request_id = uuid.uuid4().hex[:8]
        waiter = {"answered": threading.Event(), "approved": False}
        with self._pending_lock:
            self._pending[request_id] = waiter
        self._on_event("confirm_request", {"request_id": request_id, "text": text})
        answered = waiter["answered"].wait(self.confirm_timeout)
        with self._pending_lock:
            self._pending.pop(request_id, None)
        approved = answered and waiter["approved"]
        self._on_event("confirm_resolved", {"request_id": request_id, "approved": approved,
                                            "timed_out": not answered})
        return approved

    def resolve(self, request_id: str, approve: bool) -> bool:
        """Answer a pending confirmation (called from the /confirm request thread)."""
        with self._pending_lock:
            waiter = self._pending.get(request_id)
        if waiter is None:
            return False
        waiter["approved"] = bool(approve)
        waiter["answered"].set()
        return True

    # -- turns --

    def start_turn(self, text: str) -> Iterator[dict]:
        if self.ended:
            raise TurnRejected("会话已转人工，请开新会话")
        if not self._turn_lock.acquire(blocking=False):
            raise TurnRejected("上一轮还没结束")
        events: queue.Queue = queue.Queue()
        self._events = events
        threading.Thread(target=self._run, args=(text, events), daemon=True).start()
        return self._drain(events)

    def _run(self, text: str, events: queue.Queue) -> None:
        started = time.perf_counter()
        try:
            result = self.agent.handle(text)
            ticket = (self.agent.session.handoff or {}).get("ticket_id") if result.route == "human" else None
            events.put({"type": "reply", "text": result.reply, "route": result.route,
                        "reason": result.reason, "ticket_id": ticket})
        except Exception as e:  # shown in the page; the session stays usable
            self._repair_history()
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self.ended = self.agent.session.handoff is not None
            self._events = None
            self._turn_lock.release()
            events.put({"type": "done", "elapsed_ms": round((time.perf_counter() - started) * 1000),
                        "ended": self.ended})
            events.put(_END_OF_STREAM)

    def _repair_history(self) -> None:
        # A failed LLM call leaves the customer's message unanswered; close the turn
        # so the next one still alternates user/assistant.
        messages = self.agent.messages
        if messages and messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": "（这一轮处理出错）"})

    @staticmethod
    def _drain(events: queue.Queue) -> Iterator[dict]:
        while True:
            item = events.get()
            if item is _END_OF_STREAM:
                return
            yield item

    def close(self) -> None:
        shutil.rmtree(self.memory_dir, ignore_errors=True)


class SessionRegistry:
    """In-memory sessions for one local user; the oldest are dropped past max_sessions."""

    def __init__(self, mock: bool, max_sessions: int = MAX_SESSIONS):
        self.mock = mock
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, WebSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._llm = None
        # One judge per backend, shared by every session, so the circuit breaker sees
        # all traffic: three sessions failing on Jev open it for everyone.
        self._judges: Dict[str, object] = {}

    def _shared_llm(self):
        # One client for all sessions (it is thread-safe); created on first use so the
        # server can start without keys and still report which judges are available.
        with self._lock:
            if self._llm is None:
                self._llm = build_llm(self.mock)
            return self._llm

    def _shared_judge(self, name: str):
        llm = self._shared_llm()
        with self._lock:
            if name not in self._judges:
                self._judges[name] = build_judge(name, llm, self.mock, resilient=True)
            return self._judges[name]

    def health(self) -> Dict[str, dict]:
        """Circuit-breaker status of each judge built so far (judges without a breaker are omitted)."""
        with self._lock:
            judges = dict(self._judges)
        return {name: j.status() for name, j in judges.items() if hasattr(j, "status")}

    def create(self, customer_id: str, judge: str, threshold: float) -> WebSession:
        session = WebSession(customer_id, judge, threshold, self.mock, llm=self._shared_llm(),
                             shared_judge=self._shared_judge(judge))
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > self.max_sessions:
                _, oldest = self._sessions.popitem(last=False)
                oldest.close()
        return session

    def get(self, session_id: str) -> Optional[WebSession]:
        with self._lock:
            return self._sessions.get(session_id)
