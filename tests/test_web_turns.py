"""WebSession: one browser chat. Mock mode, no HTTP."""
import threading
import time
from pathlib import Path

import pytest

from web.turns import SessionRegistry, TurnRejected, WebSession

REFUND = "A1001 鞋子不想要了，七天无理由退了吧"   # mock agent -> specialist -> apply_refund -> confirmation


def run(ws, text, on_event=None):
    events = []
    for event in ws.start_turn(text):
        events.append(event)
        if on_event:
            on_event(event)
    return events


def types(events):
    return [e["type"] for e in events]


def first(events, type_, **match):
    return next(e for e in events if e["type"] == type_ and all(e.get(k) == v for k, v in match.items()))


def test_event_order_for_a_normal_turn():
    events = run(WebSession("C001", "jev", 0.5, mock=True), "A1002 的快递到哪了")
    assert types(events)[:4] == ["judgment", "gate", "tool_start", "tool_end"]
    assert types(events)[-2:] == ["reply", "done"]
    assert events[-2]["route"] == "agent" and events[-2]["ticket_id"] is None
    assert events[-1]["ended"] is False and events[-1]["elapsed_ms"] >= 0


def test_no_judge_means_no_judgment_events():
    events = run(WebSession("C001", "none", 0.5, mock=True), "A1002 的快递到哪了")
    assert "judgment" not in types(events) and "gate" not in types(events)


def test_confirmation_pauses_the_turn_until_resolved():
    ws = WebSession("C001", "none", 0.5, mock=True)

    def answer(event):
        if event["type"] == "confirm_request":
            assert "A1001" in event["text"] and "399" in event["text"]
            threading.Timer(0.05, ws.resolve, args=(event["request_id"], True)).start()

    events = run(ws, REFUND, answer)
    resolved = first(events, "confirm_resolved")
    assert (resolved["approved"], resolved["timed_out"]) == (True, False)
    assert first(events, "tool_end", name="apply_refund")["ok"] is True


def test_unanswered_confirmation_times_out_as_decline():
    events = run(WebSession("C001", "none", 0.5, mock=True, confirm_timeout=0.1), REFUND)
    resolved = first(events, "confirm_resolved")
    assert (resolved["approved"], resolved["timed_out"]) == (False, True)
    assert first(events, "tool_end", name="apply_refund")["ok"] is False


def test_abandoned_stream_still_finishes_and_frees_the_session():
    # the browser closed the tab while a confirmation was pending
    ws = WebSession("C001", "none", 0.5, mock=True, confirm_timeout=0.1)
    ws.start_turn(REFUND)                       # nobody reads the events
    deadline = time.time() + 3
    while ws.busy and time.time() < deadline:
        time.sleep(0.02)
    assert not ws.busy


def test_resolve_unknown_request():
    assert WebSession("C001", "none", 0.5, mock=True).resolve("nope", True) is False


def test_handoff_ends_the_session():
    ws = WebSession("C006", "jev", 0.5, mock=True)
    events = run(ws, "转人工！我要人工客服")
    assert events[-2]["route"] == "human" and events[-2]["ticket_id"].startswith("H")
    assert events[-1]["ended"] is True and ws.ended
    with pytest.raises(TurnRejected):
        ws.start_turn("还在吗")


def test_busy_session_rejects_a_second_turn():
    ws = WebSession("C001", "none", 0.5, mock=True, confirm_timeout=5)
    stream = ws.start_turn(REFUND)
    pending = next(e for e in stream if e["type"] == "confirm_request")
    with pytest.raises(TurnRejected):
        ws.start_turn("快递到哪了")
    ws.resolve(pending["request_id"], False)
    list(stream)
    assert not ws.busy


def test_errors_become_events_and_the_session_survives():
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("LLM down")

    ws = WebSession("C001", "none", 0.5, mock=True, llm=Boom())
    events = run(ws, "你好")
    assert types(events) == ["error", "done"] and "LLM down" in events[0]["message"]
    assert not ws.ended and not ws.busy
    assert ws.agent.messages[-1]["role"] == "assistant"   # history stays user/assistant alternating


def test_each_session_starts_from_fresh_data():
    a = WebSession("C001", "none", 0.5, mock=True)
    run(a, "A1002 的快递到哪了")
    b = WebSession("C001", "none", 0.5, mock=True)
    assert b.agent.session.memory.root != a.agent.session.memory.root
    assert b.agent.session.memory.load("C001")["preferences"]["鞋码"] == "37"


def test_registry_creates_finds_and_evicts_old_sessions():
    reg = SessionRegistry(mock=True, max_sessions=2)
    s1, s2, s3 = (reg.create("C001", "jev", 0.5) for _ in range(3))
    assert reg.get(s3.id) is s3 and reg.get(s2.id) is s2
    assert reg.get(s1.id) is None and not Path(s1.memory_dir).exists()
    assert reg.get("missing") is None
