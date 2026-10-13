from cs_agent.compact import Compactor, render_messages


def convo(n_turns, pad=200):
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"顾客第{i}句" + "啊" * pad})
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}",
                                                         "name": "query_order", "input": {}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                                    "content": "A1001 运动鞋"}]})
        msgs.append({"role": "assistant", "content": f"客服第{i}句"})
    return msgs


def test_under_limit_is_untouched():
    c = Compactor(lambda t: "S", max_chars=10 ** 6)
    msgs = convo(3)
    assert c.maybe_compact(msgs) is msgs


def test_over_limit_summarises_old_part_and_keeps_pairs():
    captured = []
    c = Compactor(lambda t: captured.append(t) or "摘要内容", max_chars=1000, keep_last=6)
    msgs = convo(6)
    out = c.maybe_compact(msgs)
    assert out[0]["role"] == "user" and "摘要内容" in out[0]["content"]
    assert out[1]["role"] == "assistant"
    kept = out[2:]
    assert kept[0]["role"] == "user" and isinstance(kept[0]["content"], str)
    assert len(kept) >= 6 and len(out) < len(msgs)
    # every tool_result still follows its tool_use
    for i, m in enumerate(kept):
        if m["role"] == "user" and isinstance(m["content"], list):
            prev = kept[i - 1]["content"]
            assert prev[0]["id"] == m["content"][0]["tool_use_id"]
    assert "顾客第0句" in captured[0]


def test_render_messages_labels_roles():
    text = render_messages(convo(1, pad=0))
    assert "顾客：顾客第0句" in text and "[调用 query_order]" in text and "客服：客服第0句" in text


def test_writes_transcript_before_compacting(tmp_path):
    c = Compactor(lambda t: "S", max_chars=500, keep_last=4, transcript_dir=tmp_path)
    c.maybe_compact(convo(5))
    assert len(list(tmp_path.glob("*.jsonl"))) == 1
