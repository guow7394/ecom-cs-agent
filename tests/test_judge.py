import json

import httpx
import pytest

from cs_agent.judge import (INTENTS, GatePolicy, Judgment, JudgeError, JevJudge, LLMJudge,
                            MockJudge, build_state, choice_confidence, gate)
from cs_agent.llm import Block, LLMResponse


def jev_answer(choice="logistics_query", confidence=0.91, needs_human=0.12, model="jev-1.13.0"):
    probs = {k: 0.0 for k in INTENTS}
    probs[choice] = 1.0
    return {"model": model, "answers": {
        "intent": {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probs},
        "needs_human": {"type": "noul", "noul": needs_human}},
        "usage": {"input_tokens": 300, "output_tokens": 20}}


STATE = build_state([("customer", "我买了双鞋"), ("agent", "请问有什么问题？")], "到哪了")


# -- confidence formula (docs.typesafe.ai/confidence) --

def test_choice_confidence_matches_jev_docs():
    assert choice_confidence({"a": 0.6, "b": 0.3, "c": 0.1}) == pytest.approx(0.4)
    assert choice_confidence({"a": 0.6, "b": 0.2, "c": 0.2}) == pytest.approx(0.4)
    assert choice_confidence({"billing": 0.88, "technical": 0.12, "sales": 0.0}) == pytest.approx(0.82, abs=0.01)
    assert choice_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)


def test_build_state_keeps_recent_turns_only():
    transcript = [("customer", str(i)) if i % 2 == 0 else ("agent", str(i)) for i in range(10)]
    state = build_state(transcript, "最新")
    assert state["latest_customer_message"] == "最新"
    assert [t["text"] for t in state["recent_conversation"]] == ["4", "5", "6", "7", "8", "9"]
    assert state["recent_conversation"][0]["from"] == "customer"


# -- Jev over HTTP --

def test_jev_request_shape_and_parsing():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=jev_answer())

    judge = JevJudge(api_key="k", model="jev-1.13.0", transport=httpx.MockTransport(handler))
    j = judge.judge(STATE)
    assert seen["path"] == "/v1/systemone" and seen["auth"] == "Bearer k"
    body = seen["body"]
    assert body["model"] == "jev-1.13.0" and body["state"] == STATE
    q = body["questions"]["intent"]
    assert q["type"] == "choice" and list(q["criteria"]) == list(INTENTS)
    assert (j.intent, j.confidence, j.backend) == ("logistics_query", 0.91, "jev:jev-1.13.0")
    assert sum(j.probabilities.values()) == pytest.approx(1.0)
    # second question, same call: evaluated in parallel by Jev, so no extra latency
    nq = body["questions"]["needs_human"]
    assert nq["type"] == "noul" and set(nq["criteria"]) == {"true", "false"}
    assert j.needs_human == 0.12


def test_jev_missing_needs_human_answer_is_an_error():
    bad = jev_answer()
    del bad["answers"]["needs_human"]
    judge = JevJudge(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bad)))
    with pytest.raises(JudgeError):
        judge.judge(STATE)


def test_jev_retries_on_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"detail": "slow down"})
        return httpx.Response(200, json=jev_answer())

    judge = JevJudge(api_key="k", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert judge.judge(STATE).intent == "logistics_query" and len(calls) == 2


def test_jev_gives_up_after_retries():
    judge = JevJudge(api_key="k", max_retries=2, sleep=lambda s: None,
                     transport=httpx.MockTransport(lambda r: httpx.Response(529, text="overloaded")))
    with pytest.raises(JudgeError, match="529"):
        judge.judge(STATE)


def test_jev_rejects_unknown_choice():
    bad = jev_answer()
    bad["answers"]["intent"]["choice"] = "pizza"
    judge = JevJudge(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=bad)))
    with pytest.raises(JudgeError):
        judge.judge(STATE)


def test_jev_requires_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        JevJudge()


# -- LLM self-reported confidence --

class OneShot:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_llm_judge_forces_tool_and_parses():
    llm = OneShot(LLMResponse([Block("tool_use", id="1", name="report_judgment",
                                     input={"intent": "exchange", "confidence": 1.3, "needs_human": -0.2,
                                            "reason": "r"})]))
    j = LLMJudge(llm).judge(STATE)
    assert llm.kwargs["tool_choice"] == {"type": "tool", "name": "report_judgment"}
    assert (j.intent, j.confidence, j.needs_human, j.backend) == ("exchange", 1.0, 0.0, "llm")
    assert "needs_human" in llm.kwargs["tools"][0]["input_schema"]["required"]


def test_llm_judge_missing_needs_human_is_an_error():
    llm = OneShot(LLMResponse([Block("tool_use", id="1", name="report_judgment",
                                     input={"intent": "exchange", "confidence": 0.9, "reason": "r"})]))
    with pytest.raises(JudgeError):
        LLMJudge(llm).judge(STATE)


def test_llm_judge_falls_back_to_json_in_text():
    llm = OneShot(LLMResponse([Block("text", text='好的 {"intent": "policy_question", "confidence": 0.7, "needs_human": 0.2}')]))
    assert LLMJudge(llm).judge(STATE).intent == "policy_question"


class RejectsForcedTool:
    """Some compatible endpoints refuse a forced tool_choice (e.g. with thinking on)."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tool_choice" in kwargs:
            raise RuntimeError("400 tool_choice not supported with thinking")
        return LLMResponse([Block("tool_use", id="1", name="report_judgment",
                                  input={"intent": "exchange", "confidence": 0.8, "needs_human": 0.3, "reason": "r"})])


def test_llm_judge_retries_without_forced_tool():
    llm = RejectsForcedTool()
    assert LLMJudge(llm).judge(STATE).intent == "exchange"
    assert "tool_choice" in llm.calls[0] and "tool_choice" not in llm.calls[1]


def test_llm_judge_leaves_room_for_thinking_and_flags_truncation():
    # deepseek-v4 thinks before calling the tool; at 400 tokens ambiguous cases got cut off
    llm = OneShot(LLMResponse([Block("thinking", text="The customer says...")], "max_tokens"))
    with pytest.raises(JudgeError, match="截断"):
        LLMJudge(llm).judge(STATE)
    assert llm.kwargs["max_tokens"] >= 4096


def test_llm_judge_rejects_garbage():
    with pytest.raises(JudgeError):
        LLMJudge(OneShot(LLMResponse([Block("text", text="不知道")]))).judge(STATE)


# -- Mock --

def test_mock_judge_is_confident_on_clear_and_unsure_on_vague():
    clear = MockJudge().judge(build_state([], "转人工！我要人工客服"))
    vague = MockJudge().judge(build_state([], "嗯……那个"))
    assert clear.intent == "request_human" and clear.confidence > 0.6
    assert vague.confidence < 0.2
    mixed = MockJudge().judge(build_state([], "先别发货，可能要改地址也可能不要了，还有上次的事怎么说"))
    simple = MockJudge().judge(build_state([], "A1002 的快递到哪了"))
    assert mixed.needs_human >= 0.5 > simple.needs_human


# -- gate --

def J(intent, conf, nh=None):
    return Judgment(intent=intent, confidence=conf, probabilities={}, backend="t", needs_human=nh)


@pytest.mark.parametrize("judgment,route,reason", [
    (J("logistics_query", 0.4), "human", "low_confidence"),
    (J("logistics_query", 0.55), "agent", ""),
    (J("return_refund", 0.55), "human", "low_confidence"),      # sensitive: 0.5 + 0.1
    (J("return_refund", 0.65), "agent", ""),
    (J("request_human", 0.95), "human", "human_intent"),
    (J("complaint_escalation", 0.9), "human", "human_intent"),
    (None, "human", "judge_error"),
    (J("return_refund", 0.95, nh=0.7), "human", "needs_judgment"),
    (J("return_refund", 0.95, nh=0.3), "agent", ""),
    (J("logistics_query", 0.4, nh=0.9), "human", "low_confidence"),  # first matching reason wins
])
def test_gate(judgment, route, reason):
    d = gate(judgment, GatePolicy(threshold=0.5))
    assert (d.route, d.reason) == (route, reason)


def test_needs_human_check_can_be_disabled():
    j = J("exchange", 0.95, nh=0.99)
    assert gate(j, GatePolicy(threshold=0.5, needs_human_threshold=None)).route == "agent"
    assert gate(j, GatePolicy(threshold=0.5)).reason == "needs_judgment"   # on by default


def test_effective_threshold_is_capped():
    p = GatePolicy(threshold=0.9, sensitive_margin=0.1)
    assert p.effective_threshold("exchange") == 0.95
    assert p.effective_threshold("order_status") == 0.9
