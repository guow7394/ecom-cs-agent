import pytest

from eval.metrics import auroc, calibration, case_correct, materialize, separation, summarize

CASES = {
    "A": {"id": "A", "category": "standard", "gold_route": "agent", "gold_intents": ["logistics_query"],
          "expect_tools": ["query_logistics"], "forbid_success": []},
    "B": {"id": "B", "category": "trap", "gold_route": "agent", "gold_intents": ["return_refund"],
          "expect_tools": [], "forbid_success": ["apply_refund"]},
    "H": {"id": "H", "category": "explicit_human", "gold_route": "human", "gold_intents": ["request_human"],
          "expect_tools": [], "forbid_success": []},
}


def tool(name, ok=True):
    return {"name": name, "ok": ok, "by": "main"}


@pytest.mark.parametrize("cid,route,tools,expected", [
    ("A", "agent", [tool("query_logistics")], (True, True)),
    ("A", "agent", [tool("query_order")], (True, False)),          # missing expected tool
    ("A", "human", [], (False, False)),
    ("B", "agent", [tool("apply_refund", ok=False)], (True, True)),  # attempted but blocked: fine
    ("B", "agent", [tool("apply_refund", ok=True)], (True, False)),  # forbidden action executed
    ("H", "human", [], (True, True)),
    ("H", "agent", [], (False, False)),
    ("A", "error", [], (False, False)),
])
def test_case_correct(cid, route, tools, expected):
    assert case_correct(CASES[cid], route, tools) == expected


def judge_record(cid, intent, conf, agent=None, judge_error=""):
    judgment = None if judge_error else {"intent": intent, "confidence": conf, "probabilities": {},
                                          "backend": "t", "latency_ms": 1.0, "reason": ""}
    return {"case_id": cid, "arm": "jev", "judgment": judgment, "judge_error": judge_error,
            "agent": agent, "agent_error": ""}


def test_materialize_applies_threshold_offline():
    rec = judge_record("A", "logistics_query", 0.6,
                       agent={"route": "agent", "reason": "", "tools": [tool("query_logistics")]})
    low = materialize(rec, CASES["A"], 0.5)
    assert (low["route"], low["e2e_ok"], low["intent_ok"]) == ("agent", True, True)
    high = materialize(rec, CASES["A"], 0.7)
    assert (high["route"], high["reason"], high["e2e_ok"]) == ("human", "low_confidence", False)
    assert high["intent_ok"] is True  # intent accuracy does not depend on the threshold


def test_materialize_sensitive_margin():
    rec = judge_record("B", "return_refund", 0.55, agent={"route": "agent", "reason": "", "tools": []})
    assert materialize(rec, CASES["B"], 0.5, margin=0.1)["route"] == "human"
    assert materialize(rec, CASES["B"], 0.5, margin=0.0)["route"] == "agent"


def test_materialize_needs_human_variant_is_replayed_offline():
    rec = judge_record("B", "return_refund", 0.95, agent={"route": "agent", "reason": "", "tools": []})
    rec["judgment"]["needs_human"] = 0.7
    intent_only = materialize(rec, CASES["B"], 0.5)
    assert intent_only["route"] == "agent" and intent_only["needs_human"] == 0.7
    with_nh = materialize(rec, CASES["B"], 0.5, needs_human_threshold=0.5)
    assert (with_nh["route"], with_nh["reason"]) == ("human", "needs_judgment")


def test_auroc():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([], [0.3]) is None


def test_separation_ignores_human_intents():
    def r(gold, intent, conf, nh):
        return dict(row(gold, "x", True, conf=conf), intent=intent, needs_human=nh)
    rows = [r("human", "return_refund", 0.4, 0.8), r("human", "request_human", 1.0, 0.1),
            r("agent", "exchange", 0.99, 0.1), r("agent", "order_status", 0.9, 0.3)]
    s = separation(rows)
    assert (s["n_human"], s["n_agent"]) == (1, 2)
    assert s["auroc_confidence"] == 1.0 and s["auroc_needs_human"] == 1.0
    assert s["mean_needs_human_human"] == pytest.approx(0.8)
    assert s["mean_needs_human_agent"] == pytest.approx(0.2)


def test_materialize_judge_error_is_human():
    row = materialize(judge_record("H", None, None, judge_error="boom"), CASES["H"], 0.5)
    assert (row["route"], row["reason"], row["e2e_ok"], row["intent_ok"]) == ("human", "judge_error", True, None)


def test_materialize_baseline_and_agent_error():
    base = {"case_id": "A", "arm": "baseline", "judgment": None, "judge_error": "",
            "agent": {"route": "agent", "reason": "", "tools": [tool("query_logistics")]}, "agent_error": ""}
    assert materialize(base, CASES["A"], 0.5)["e2e_ok"] is True
    broken = dict(base, agent=None, agent_error="timeout")
    row = materialize(broken, CASES["A"], 0.5)
    assert (row["route"], row["e2e_ok"], row["error"]) == ("error", False, "timeout")


def row(gold, route, e2e, intent_ok=True, conf=0.9, reason=""):
    return {"case_id": "x", "category": "c", "gold_route": gold, "route": route, "reason": reason,
            "intent": "i", "confidence": conf, "intent_ok": intent_ok,
            "route_ok": gold == route, "e2e_ok": e2e, "error": "" if route != "error" else "e"}


def test_summarize():
    rows = [row("agent", "agent", True), row("agent", "agent", False), row("agent", "human", False, reason="low_confidence"),
            row("human", "human", True, reason="human_intent"), row("human", "agent", False, intent_ok=False)]
    s = summarize(rows)
    assert s["n"] == 5
    assert s["route_acc"] == pytest.approx(3 / 5)
    assert s["e2e_acc"] == pytest.approx(2 / 5)
    assert s["intent_acc"] == pytest.approx(4 / 5)
    assert s["handoff_rate"] == pytest.approx(2 / 5)
    assert s["missed_handoff"] == pytest.approx(1 / 2)
    assert s["false_handoff"] == pytest.approx(1 / 3)
    assert s["coverage"] == pytest.approx(3 / 5)
    assert s["auto_acc"] == pytest.approx(1 / 3)
    assert s["reasons"] == {"low_confidence": 1, "human_intent": 1}


def test_summarize_counts_fail_safe_handoffs_as_failures():
    rows = [row("human", "human", True, reason="judge_error"), row("human", "human", True, reason="truncated"),
            row("human", "human", True, reason="human_intent"), row("agent", "error", False)]
    assert summarize(rows)["failures"] == 3  # judge_error + truncated + hard error


def test_summarize_without_intents():
    s = summarize([dict(row("agent", "agent", True), intent_ok=None, confidence=None)])
    assert s["intent_acc"] is None


def test_calibration_buckets_and_ece():
    rows = [row("agent", "agent", True, intent_ok=True, conf=0.95),
            row("agent", "agent", True, intent_ok=False, conf=0.95),
            row("agent", "agent", True, intent_ok=False, conf=0.2)]
    bins, ece = calibration(rows)
    top = [b for b in bins if b["n"] == 2][0]
    assert top["accuracy"] == pytest.approx(0.5) and top["mean_confidence"] == pytest.approx(0.95)
    # ECE = 2/3*|0.5-0.95| + 1/3*|0-0.2|
    assert ece == pytest.approx(2 / 3 * 0.45 + 1 / 3 * 0.2)


def test_quality_is_attached_only_to_bot_handled_rows():
    rec = judge_record("A", "logistics_query", 0.6,
                       agent={"route": "agent", "reason": "", "tools": [tool("query_logistics")]})
    rec["quality"] = {"mean": 0.9, "passed": True}
    assert materialize(rec, CASES["A"], 0.5)["quality"] == {"mean": 0.9, "passed": True}
    assert materialize(rec, CASES["A"], 0.7)["quality"] is None        # gated to a human at 0.7


def test_summarize_quality():
    def q_row(route, quality):
        return dict(row("agent", route, True), quality=quality)
    rows = [q_row("agent", {"mean": 0.9, "passed": True}), q_row("agent", {"mean": 0.6, "passed": False}),
            q_row("agent", {"failed": True, "error": "x"}), q_row("human", None)]
    s = summarize(rows)
    assert (s["quality_n"], s["quality_failed"]) == (2, 1)
    assert s["quality_mean"] == pytest.approx(0.75) and s["quality_pass"] == pytest.approx(0.5)


def test_summarize_without_quality():
    s = summarize([dict(row("agent", "agent", True), quality=None)])
    assert s["quality_mean"] is None and s["quality_n"] == 0
