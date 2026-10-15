"""Request-level traces: every turn gets a request_id, and its steps can be looked up afterwards."""
import json

import pytest

from cs_agent.agent import CustomerServiceAgent
from cs_agent.judge import GatePolicy, JudgeError, MockJudge
from cs_agent.llm import MockLLM
from cs_agent.trace import Trace, TraceStore, new_request_id


def make_agent(make_session, traces, judge=None, llm=None):
    return CustomerServiceAgent(llm or MockLLM(), make_session("C001"), judge=judge or MockJudge(),
                                policy=GatePolicy(0.5), traces=traces)


def test_every_turn_gets_a_trace_with_timed_steps(make_session):
    traces = TraceStore()
    agent = make_agent(make_session, traces)
    result = agent.handle("A1002 的快递到哪了")
    trace = traces.get(result.request_id)
    assert trace["route"] == "agent" and trace["reply"] == result.reply and trace["input"] == "A1002 的快递到哪了"
    kinds = [e["type"] for e in trace["events"]]
    assert kinds[:2] == ["judgment", "gate"] and "llm" in kinds and "tool_start" in kinds
    times = [e["t_ms"] for e in trace["events"]]
    assert times == sorted(times) and trace["total_ms"] >= times[-1]
    b = trace["breakdown"]
    assert b["llm_rounds"] >= 2 and b["tool_calls"] == kinds.count("tool_start") and b["judge_backend"] == "mock"


def test_handoff_turn_is_traced_without_llm_rounds(make_session):
    traces = TraceStore()
    result = make_agent(make_session, traces).handle("转人工！我要投诉")
    trace = traces.get(result.request_id)
    assert trace["route"] == "human" and trace["breakdown"]["llm_rounds"] == 0


def test_degraded_judgment_shows_in_the_breakdown(make_session):
    class Down:
        def judge(self, state):
            raise JudgeError("Jev HTTP 529")

    from cs_agent.resilience import ResilientJudge
    traces = TraceStore()
    result = make_agent(make_session, traces, judge=ResilientJudge(Down(), MockJudge())).handle("A1002 到哪了")
    assert "529" in traces.get(result.request_id)["breakdown"]["degraded"]


def test_a_crashed_turn_is_still_traced(make_session):
    class Boom(MockLLM):
        def create(self, *a, **k):
            raise RuntimeError("LLM down")

    traces = TraceStore()
    agent = make_agent(make_session, traces, llm=Boom())
    with pytest.raises(RuntimeError):
        agent.handle("A1002 的快递到哪了")
    trace = traces.get(agent.last_request_id)
    assert "LLM down" in trace["error"] and trace["route"] == ""


def test_events_still_reach_the_page(make_session):
    seen = []
    agent = make_agent(make_session, TraceStore())
    agent.session.on_event = lambda t, d: seen.append(t)
    agent.handle("A1002 的快递到哪了")
    assert seen[:2] == ["judgment", "gate"]
    assert agent.session.on_event is not None and "llm" in seen


def test_store_keeps_the_latest_and_persists_jsonl(tmp_path):
    store = TraceStore(tmp_path, max_items=2)
    ids = []
    for i in range(3):
        trace = Trace(new_request_id(), "C001", f"第{i}句")
        trace.finish(reply="好的", route="agent")
        store.save(trace)
        ids.append(trace.request_id)
    assert [t["request_id"] for t in store.recent()] == ids[:0:-1]        # newest first, oldest dropped
    assert store.get(ids[0])["input"] == "第0句"                          # still found on disk
    lines = [json.loads(l) for f in tmp_path.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert [l["request_id"] for l in lines] == ids
    assert TraceStore(tmp_path).get(ids[1])["input"] == "第1句"           # survives a restart
    assert store.get("r-nope") is None


def test_request_ids_are_unique():
    assert len({new_request_id() for _ in range(1000)}) == 1000
