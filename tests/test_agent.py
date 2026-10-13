import pytest

from cs_agent.agent import CustomerServiceAgent
from cs_agent.judge import GatePolicy, Judgment, MockJudge
from cs_agent.llm import MockLLM


class FixedJudge:
    def __init__(self, intent, confidence):
        self.intent, self.confidence, self.states = intent, confidence, []

    def judge(self, state):
        self.states.append(state)
        return Judgment(self.intent, self.confidence, {self.intent: self.confidence}, "fixed")


class BrokenJudge:
    def judge(self, state):
        raise RuntimeError("Jev down")


class SpyLLM(MockLLM):
    def __init__(self):
        self.systems = []

    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=None):
        self.systems.append(system)
        return super().create(system, messages, tools, tool_choice, max_tokens)


class ExplodingLLM:
    def create(self, **kwargs):
        raise AssertionError("LLM must not be called when the gate hands off")


def make_agent(make_session, customer="C001", judge=None, llm=None, threshold=0.5):
    return CustomerServiceAgent(llm or MockLLM(), make_session(customer), judge=judge,
                                policy=GatePolicy(threshold=threshold))


def test_low_confidence_hands_off_without_calling_llm(make_session):
    agent = make_agent(make_session, judge=FixedJudge("logistics_query", 0.3), llm=ExplodingLLM())
    r = agent.handle("那个东西怎么回事")
    assert (r.route, r.reason) == ("human", "low_confidence")
    assert agent.session.handoff["by"] == "gate" and agent.session.handoff["ticket_id"] in r.reply
    assert r.tools == []


def test_human_intent_hands_off(make_session):
    r = make_agent(make_session, judge=MockJudge(), llm=ExplodingLLM()).handle("转人工！我要人工客服")
    assert (r.route, r.reason) == ("human", "human_intent")
    assert r.judgment.intent == "request_human"


def test_judge_failure_is_fail_safe(make_session):
    r = make_agent(make_session, judge=BrokenJudge(), llm=ExplodingLLM()).handle("快递到哪了")
    assert (r.route, r.reason) == ("human", "judge_error") and "Jev down" in r.error


def test_confident_turn_goes_to_agent_with_hint(make_session):
    llm = SpyLLM()
    r = make_agent(make_session, judge=MockJudge(), llm=llm).handle("A1002 的快递到哪了")
    assert r.route == "agent" and r.gate.route == "agent"
    assert [t["name"] for t in r.tools] == ["query_logistics"]
    assert "[判断层]" in llm.systems[0] and "logistics_query" in llm.systems[0]
    assert r.reply


def test_baseline_has_no_hint_and_can_self_escalate(make_session):
    llm = SpyLLM()
    r = make_agent(make_session, customer="C003", llm=llm).handle("再不解决我就去12315投诉")
    assert "[判断层]" not in llm.systems[0]
    assert (r.route, r.reason) == ("human", "agent_transfer")
    assert r.judgment is None


def test_exchange_is_delegated_to_specialist(make_session):
    r = make_agent(make_session, customer="C005").handle("A5001 牛仔裤小了，想换31码")
    assert [(t["name"], t["by"]) for t in r.tools] == [("after_sales_specialist", "main"),
                                                         ("create_exchange", "specialist")]
    assert r.tools[-1]["ok"] and r.route == "agent"


def test_history_reaches_judge_and_messages(make_session):
    judge = FixedJudge("exchange", 0.9)
    agent = make_agent(make_session, judge=judge)
    agent.preload_history([("customer", "我上周买的那双鞋"), ("agent", "查到 A1001，请问有什么问题？")])
    agent.handle("帮我换小一号吧")
    assert judge.states[0]["recent_conversation"][0] == {"from": "customer", "text": "我上周买的那双鞋"}
    assert [m["role"] for m in agent.messages[:3]] == ["user", "assistant", "user"]


class TruncatingLLM:
    def create(self, **kwargs):
        from cs_agent.llm import Block, LLMResponse
        return LLMResponse([Block("thinking", text="还在想……")], "max_tokens")


def test_truncated_reply_fails_safe_to_human(make_session):
    agent = make_agent(make_session, llm=TruncatingLLM())
    r = agent.respond("收到的东西跟描述不一样")
    assert (r.route, r.reason) == ("human", "truncated")
    assert agent.session.handoff["ticket_id"] in r.reply


def test_preload_history_rejects_bad_alternation(make_session):
    agent = make_agent(make_session)
    with pytest.raises(ValueError):
        agent.preload_history([("customer", "a"), ("customer", "b")])
