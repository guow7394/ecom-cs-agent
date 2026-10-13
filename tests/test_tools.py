from cs_agent.tools import BASE_TOOL_NAMES, SPECIALIST_TOOL_NAMES, TOOL_SCHEMAS, ToolBox
from tests.conftest import ConfirmRecorder


def test_schema_names_are_consistent():
    for name in BASE_TOOL_NAMES + SPECIALIST_TOOL_NAMES:
        assert TOOL_SCHEMAS[name]["name"] == name
    assert "apply_refund" not in BASE_TOOL_NAMES  # main agent must delegate refunds (s06)
    assert "transfer_to_human" not in SPECIALIST_TOOL_NAMES


def test_query_order_list_and_detail(make_session):
    box = ToolBox(make_session("C001"))
    listing = box.run("query_order", {})
    assert listing.ok and "A1001" in listing.content and "A1002" in listing.content
    detail = box.run("query_order", {"order_id": "A1001"})
    assert "签收 4 天" in detail.content
    assert not box.run("query_order", {"order_id": "A2001"}).ok  # someone else's order


def test_logistics_flags_stalled_parcel(make_session):
    box = ToolBox(make_session("C003"))
    out = box.run("query_logistics", {"order_id": "A3002"}).content
    assert "物流异常" in out
    box = ToolBox(make_session("C001"))
    assert "物流异常" not in box.run("query_logistics", {"order_id": "A1002"}).content


def test_register_urge_only_for_stalled_parcels(make_session):
    s = make_session("C003")
    res = ToolBox(s).run("register_urge", {"order_id": "A3002"})  # 4 days without an update
    assert res.ok and "24 小时" in res.content
    assert s.memory.load("C003")["after_sales"][-1]["type"] == "催件"
    box = ToolBox(make_session("C001"))
    fresh = box.run("register_urge", {"order_id": "A1002"})      # updated today
    assert not fresh.ok and "72 小时" in fresh.content
    assert not box.run("register_urge", {"order_id": "A1001"}).ok  # already delivered


def test_search_policy(make_session):
    out = ToolBox(make_session()).run("search_policy", {"query": "运费险"}).content
    assert "运费与运费险" in out


def test_refund_success_logs_and_writes_memory(make_session):
    s = make_session("C001")
    box = ToolBox(s, by="specialist")
    res = box.run("apply_refund", {"order_id": "A1001", "reason_type": "no_reason", "reason": "不喜欢"})
    assert res.ok
    assert s.tool_log[-1] == {**s.tool_log[-1], "name": "apply_refund", "ok": True, "by": "specialist"}
    assert s.memory.load("C001")["after_sales"][-1]["order_id"] == "A1001"


def test_refund_blocked_is_logged_as_not_ok(make_session):
    s = make_session("C002")
    res = ToolBox(s).run("apply_refund", {"order_id": "A2001", "reason_type": "no_reason"})
    assert not res.ok and s.tool_log[-1]["ok"] is False
    assert s.store.get_order("C002", "A2001")["status"] == "已签收"


def test_declined_confirmation_does_not_execute(make_session):
    s = make_session("C002", confirm_fn=ConfirmRecorder(False))
    res = ToolBox(s).run("change_address", {"order_id": "A2002", "new_address": "X"})
    assert not res.ok and s.store.get_order("C002", "A2002")["address"] != "X"


def test_exchange_and_preference(make_session):
    s = make_session("C005")
    box = ToolBox(s)
    assert box.run("create_exchange", {"order_id": "A5001", "new_size": "31"}).ok
    assert box.run("remember_preference", {"category": "裤子尺码", "value": "31"}).ok
    assert s.memory.load("C005")["preferences"]["裤子尺码"] == "31"


def test_transfer_to_human_sets_handoff(make_session):
    s = make_session("C006")
    res = ToolBox(s).run("transfer_to_human", {"reason": "顾客要求人工", "summary": "s"})
    assert res.ok and s.handoff["by"] == "agent" and s.handoff["reason"] == "顾客要求人工"


def test_unknown_tool_and_bad_args_do_not_raise(make_session):
    box = ToolBox(make_session())
    assert not box.run("rm_rf", {}).ok
    assert not box.run("query_logistics", {"wrong": 1}).ok
