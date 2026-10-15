"""Event hooks (learn-claude-code s04 idea): the agent reports each step so a UI can show it live."""
from cs_agent.agent import CustomerServiceAgent
from cs_agent.judge import GatePolicy, MockJudge
from cs_agent.llm import MockLLM
from cs_agent.tools import ToolBox


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event_type, data):
        self.events.append((event_type, data))

    def types(self):
        return [t for t, _ in self.events]


def test_session_emit_is_a_noop_by_default(make_session):
    make_session().emit("anything", x=1)


def test_toolbox_reports_start_and_end_for_success_and_failure(make_session):
    s = make_session("C001")
    s.on_event = rec = Recorder()
    box = ToolBox(s, by="specialist")
    box.run("query_order", {"order_id": "A1001"})
    box.run("query_order", {"order_id": "A9999"})
    assert rec.types() == ["tool_start", "tool_end", "tool_start", "tool_end"]
    assert rec.events[0][1] == {"name": "query_order", "by": "specialist", "input": {"order_id": "A1001"}}
    assert rec.events[1][1]["ok"] is True and "A1001" in rec.events[1][1]["output"]
    assert rec.events[3][1]["ok"] is False


def test_agent_reports_judgment_and_gate_before_tools(make_session):
    s = make_session("C001")
    s.on_event = rec = Recorder()
    CustomerServiceAgent(MockLLM(), s, judge=MockJudge(), policy=GatePolicy(0.5)).handle("A1002 的快递到哪了")
    assert rec.types()[:2] == ["judgment", "gate"] and "tool_start" in rec.types()
    judgment = rec.events[0][1]
    assert (judgment["intent"], judgment["intent_zh"]) == ("logistics_query", "查物流")
    assert {"confidence", "needs_human", "latency_ms", "backend"} <= set(judgment)
    assert rec.events[1][1] == {"route": "agent", "reason": "", "threshold": 0.5, "needs_human_threshold": 0.5}


def test_agent_reports_judge_errors(make_session):
    class Broken:
        def judge(self, state):
            raise RuntimeError("Jev down")

    s = make_session()
    s.on_event = rec = Recorder()
    CustomerServiceAgent(MockLLM(), s, judge=Broken()).handle("你好")
    assert rec.events[0] == ("judgment", {"error": "RuntimeError: Jev down"})
    assert rec.events[1][1]["route"] == "human"


def test_specialist_delegation_is_reported_around_its_own_tools(make_session):
    s = make_session("C005")
    s.on_event = rec = Recorder()
    CustomerServiceAgent(MockLLM(), s).handle("A5001 牛仔裤小了，想换31码")
    calls = [(t, d["name"], d["by"]) for t, d in rec.events if t.startswith("tool")]
    assert calls == [("tool_start", "after_sales_specialist", "main"),
                     ("tool_start", "create_exchange", "specialist"),
                     ("tool_end", "create_exchange", "specialist"),
                     ("tool_end", "after_sales_specialist", "main")]
