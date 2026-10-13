from cs_agent.llm import Block, LLMResponse, MockLLM
from cs_agent.loop import agent_loop
from cs_agent.subagent import run_specialist


class ScriptedLLM:
    """Returns canned responses in order and records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        return self.responses.pop(0)


def tool_call(name, args, id_="t1"):
    return LLMResponse([Block("tool_use", id=id_, name=name, input=args)], "tool_use")


def text(t):
    return LLMResponse([Block("text", text=t)])


def test_loop_runs_tools_until_text():
    llm = ScriptedLLM([tool_call("echo", {"x": 1}), text("done")])
    seen = []
    messages = [{"role": "user", "content": "hi"}]
    out = agent_loop(llm, "sys", messages, [], lambda n, a: seen.append((n, a)) or "ok")
    assert out.text == "done" and not out.exhausted
    assert seen == [("echo", {"x": 1})]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}


def test_loop_gives_up_after_max_rounds_and_keeps_alternation():
    llm = ScriptedLLM([tool_call("echo", {}, f"t{i}") for i in range(3)])
    messages = [{"role": "user", "content": "hi"}]
    out = agent_loop(llm, "sys", messages, [], lambda n, a: "ok", max_rounds=3)
    assert out.exhausted
    assert messages[-1]["role"] == "assistant"


def test_loop_reports_truncated_final_answer():
    llm = ScriptedLLM([LLMResponse([Block("thinking", text="hmm")], "max_tokens")])
    out = agent_loop(llm, "sys", [{"role": "user", "content": "hi"}], [], lambda n, a: "ok")
    assert out.truncated and out.text == ""


def test_anthropic_llm_default_budget_leaves_room_for_thinking():
    from cs_agent.llm import AnthropicLLM
    assert AnthropicLLM(model="m", api_key="k").max_tokens >= 8192


def test_specialist_runs_in_fresh_context_and_tags_calls(make_session):
    s = make_session("C005")
    llm = ScriptedLLM([tool_call("create_exchange", {"order_id": "A5001", "new_size": "31"}),
                       text("已为顾客把 A5001 换成 31 码")])
    result = run_specialist(llm, s, "顾客要把 A5001 牛仔裤换成 31 码")
    assert "31" in result
    assert llm.calls[0]["messages"][0] == {"role": "user", "content": "顾客要把 A5001 牛仔裤换成 31 码"}
    assert "售后专员" in llm.calls[0]["system"] and "裤子尺码" in llm.calls[0]["system"]
    assert s.tool_log[-1]["by"] == "specialist" and s.tool_log[-1]["ok"]


def test_mock_llm_picks_a_tool_and_then_answers():
    llm = MockLLM()
    tools = [{"name": "query_logistics"}, {"name": "query_order"}]
    r1 = llm.create("sys", [{"role": "user", "content": "A1002 的快递到哪了"}], tools)
    assert r1.content[0].type == "tool_use" and r1.content[0].name == "query_logistics"
    assert r1.content[0].input == {"order_id": "A1002"}
    msgs = [{"role": "user", "content": "A1002 的快递到哪了"},
            {"role": "assistant", "content": r1.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "到达广州"}]}]
    r2 = llm.create("sys", msgs, tools)
    assert r2.content[0].type == "text" and "到达广州" in r2.content[0].text
