"""Request-level traces: what one customer turn did, retrievable by its request_id.

    handle(text)                         request_id = r-<12 hex>
      session.emit(...) ──► Trace.add()     every hook event, stamped t_ms from the turn start
      end of turn       ──► Trace.finish()  reply, route, error, total_ms, breakdown
                        ──► TraceStore.save()
                              memory: the latest max_items, for the web page
                              disk:   <dir>/YYYY-MM-DD.jsonl, one line per turn (gitignored)

The breakdown answers "where did this turn's time go": judge_ms is the
wall time until the judgment (including a failed Jev attempt before the
backup answered), llm_ms sums the agent-loop LLM rounds (front desk and
specialist), and the rest is tools, RAG and overhead.

Traces hold what the customer said and what the tools returned, so the
directory stays out of git.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def new_request_id() -> str:
    return f"r-{uuid.uuid4().hex[:12]}"


class Trace:
    def __init__(self, request_id: str, customer_id: str, text: str):
        self.request_id = request_id
        self.customer_id = customer_id
        self.text = text
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        self.events: List[dict] = []
        self.result: dict = {}

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self._t0) * 1000)

    def add(self, event_type: str, data: dict) -> None:
        self.events.append({"t_ms": self.elapsed_ms(), "type": event_type, **data})

    def finish(self, reply: str = "", route: str = "", reason: str = "", error: str = "") -> None:
        self.result = {"reply": reply, "route": route, "reason": reason, "error": error,
                       "total_ms": self.elapsed_ms()}

    def breakdown(self) -> dict:
        judgment = next((e for e in self.events if e["type"] == "judgment"), None)
        llm = [e for e in self.events if e["type"] == "llm"]
        return {
            "judge_backend": judgment.get("backend") or "" if judgment else "",
            "judge_ms": judgment["t_ms"] if judgment else 0,
            "degraded": judgment.get("degraded") or "" if judgment else "",
            "llm_rounds": len(llm),
            "llm_ms": sum(e.get("ms", 0) for e in llm),
            "tokens_in": sum(e.get("input_tokens") or 0 for e in llm),
            "tokens_out": sum(e.get("output_tokens") or 0 for e in llm),
            "tool_calls": sum(1 for e in self.events if e["type"] == "tool_start"),
        }

    def to_dict(self) -> dict:
        return {"request_id": self.request_id, "started_at": self.started_at, "customer_id": self.customer_id,
                "input": self.text, **self.result, "breakdown": self.breakdown(),
                "events": self.events}


def summary(trace: dict) -> dict:
    keys = ("request_id", "started_at", "customer_id", "route", "reason", "error", "total_ms")
    b = trace.get("breakdown", {})
    return {**{k: trace.get(k) for k in keys}, "input": trace.get("input", "")[:40],
            "judge_backend": b.get("judge_backend", ""), "degraded": b.get("degraded", "")}


class TraceStore:
    def __init__(self, directory: Optional[Path] = None, max_items: int = 200):
        self.directory = Path(directory) if directory else None
        self.max_items = max_items
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def save(self, trace: Trace) -> dict:
        record = trace.to_dict()
        with self._lock:
            self._items[trace.request_id] = record
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
            if self.directory:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"{trace.started_at[:10]}.jsonl"
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def get(self, request_id: str) -> Optional[dict]:
        with self._lock:
            if request_id in self._items:
                return self._items[request_id]
        return self._find_on_disk(request_id)

    def recent(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return list(reversed(self._items.values()))[:limit]

    def _find_on_disk(self, request_id: str) -> Optional[dict]:
        if not self.directory or not self.directory.exists():
            return None
        needle = f'"request_id": "{request_id}"'
        for path in sorted(self.directory.glob("*.jsonl"), reverse=True):
            for line in path.read_text(encoding="utf-8").splitlines():
                if needle in line:
                    return json.loads(line)
        return None
