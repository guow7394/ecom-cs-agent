"""Drive the real anthropic SDK against a fake HTTP server: proves the
requests we build (SDK blocks round-tripped, tool results, forced tool
choice) are valid Messages API payloads, without needing a DeepSeek key."""
import json

import httpx

from cs_agent.agent import CustomerServiceAgent
from cs_agent.judge import GatePolicy, Judgment, LLMJudge, build_state
from cs_agent.llm import AnthropicLLM


def message(content, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "deepseek-test",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def fake_llm(responses, requests):
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))
    return AnthropicLLM(model="deepseek-test", base_url="http://fake.local", api_key="test",
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


class FixedJudge:
    def judge(self, state):
        return Judgment("logistics_query", 0.93, {}, "fixed")


def test_agent_turn_over_the_wire(make_session):
    requests = []
    llm = fake_llm([
        message([{"type": "text", "text": "我查一下"},
                 {"type": "tool_use", "id": "toolu_1", "name": "query_logistics",
                  "input": {"order_id": "A1002"}}], "tool_use"),
        message([{"type": "text", "text": "您的连衣裙已到广州，预计明天派送。"}]),
    ], requests)
    agent = CustomerServiceAgent(llm, make_session("C001"), judge=FixedJudge(), policy=GatePolicy(0.5))
    result = agent.handle("连衣裙到哪了")

    assert result.route == "agent" and "广州" in result.reply
    first, second = requests
    assert first["model"] == "deepseek-test" and first["temperature"] == 0.0
    assert "[判断层]" in first["system"]
    assert {t["name"] for t in first["tools"]} >= {"query_logistics", "after_sales_specialist",
                                                     "transfer_to_human"}
    # the SDK objects from the first response are sent back verbatim, followed by our tool_result
    assert second["messages"][1]["content"][1] == {"type": "tool_use", "id": "toolu_1",
                                                   "name": "query_logistics", "input": {"order_id": "A1002"}}
    tool_result = second["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert "到达广州" in tool_result["content"]


def test_llm_judge_over_the_wire():
    requests = []
    llm = fake_llm([message([{"type": "tool_use", "id": "toolu_9", "name": "report_judgment",
                              "input": {"intent": "exchange", "confidence": 0.7, "needs_human": 0.1, "reason": "换码"}}],
                            "tool_use")], requests)
    j = LLMJudge(llm).judge(build_state([], "想换大一码"))
    assert (j.intent, j.confidence) == ("exchange", 0.7)
    assert requests[0]["tool_choice"] == {"type": "tool", "name": "report_judgment"}
    assert json.loads(requests[0]["messages"][0]["content"])["latest_customer_message"] == "想换大一码"
