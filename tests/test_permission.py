from tests.conftest import ConfirmRecorder


def test_non_sensitive_tool_passes_without_asking(make_session, confirm):
    s = make_session("C001")
    assert s.permission.check("C001", "query_order", {"order_id": "A1001"}) == (True, "")
    assert confirm.prompts == []


def test_gate1_hard_rule_blocks_big_refund_without_asking(make_session, confirm):
    s = make_session("C004")
    ok, msg = s.permission.check("C004", "apply_refund", {"order_id": "A4001", "reason_type": "no_reason"})
    assert not ok and "[硬规则]" in msg and "2000" in msg
    assert confirm.prompts == []


def test_gate2_business_rule_blocks_without_asking(make_session, confirm):
    s = make_session("C003")
    ok, msg = s.permission.check("C003", "apply_refund", {"order_id": "A3001", "reason_type": "no_reason"})
    assert not ok and "[业务规则]" in msg and "定制" in msg
    assert confirm.prompts == []


def test_gate3_customer_confirmation_text(make_session, confirm):
    s = make_session("C001")
    ok, _ = s.permission.check("C001", "apply_refund", {"order_id": "A1001", "reason_type": "no_reason"})
    assert ok
    assert "A1001" in confirm.prompts[0] and "399" in confirm.prompts[0]


def test_gate3_customer_declines(make_session):
    s = make_session("C002", confirm_fn=ConfirmRecorder(False))
    ok, msg = s.permission.check("C002", "change_address", {"order_id": "A2002", "new_address": "X"})
    assert not ok and "未确认" in msg
