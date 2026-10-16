"""Bootstrap intervals and paired tests: how sure can 40 cases make us?"""
import json

import pytest

from eval.stats import analyze, main, mcnemar_p, render_stats


def rows(missed=(), false=(), n_human=20, n_agent=20):
    """One variant's rows: gold-human cases H00.., gold-agent cases A00..; the listed ids go wrong."""
    out = []
    for i in range(n_human):
        cid = f"H{i:02d}"
        route = "agent" if cid in missed else "human"
        out.append({"case_id": cid, "gold_route": "human", "route": route, "route_ok": route == "human",
                    "e2e_ok": route == "human"})
    for i in range(n_agent):
        cid = f"A{i:02d}"
        route = "human" if cid in false else "agent"
        out.append({"case_id": cid, "gold_route": "agent", "route": route, "route_ok": route == "agent",
                    "e2e_ok": route == "agent"})
    return out


SIX = [f"H{i:02d}" for i in range(6)]


def test_mcnemar_exact_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(6, 1) == pytest.approx(0.125)          # 2 * (1 + 7) / 2**7
    assert mcnemar_p(1, 6) == pytest.approx(0.125)
    assert mcnemar_p(10, 0) == pytest.approx(2 / 1024)


def test_interval_brackets_the_value_and_is_reproducible():
    data = {"baseline": rows(missed=SIX), "jev+nh": rows(missed=SIX[:3], false=["A00", "A01"])}
    a, b = analyze(data, n_boot=2000, seed=1), analyze(data, n_boot=2000, seed=1)
    assert a == b
    missed = a["intervals"]["baseline"]["missed_handoff"]
    assert missed["value"] == pytest.approx(0.3) and missed["n"] == 20
    assert (missed["lo"], missed["hi"]) == pytest.approx((0.1455, 0.5190), abs=1e-3)   # Wilson 6/20
    perfect = a["intervals"]["baseline"]["false_handoff"]
    assert perfect["value"] == perfect["lo"] == 0.0 and perfect["hi"] == pytest.approx(0.1611, abs=1e-3)


def test_identical_variants_show_no_difference():
    data = {"baseline": rows(missed=SIX), "jev+nh": rows(missed=SIX)}
    cmp = next(c for c in analyze(data, n_boot=500)["comparisons"] if c["metric"] == "missed_handoff")
    assert cmp["diff"] == cmp["lo"] == cmp["hi"] == 0.0 and cmp["p"] == 1.0


def test_a_clear_improvement_is_significant_and_a_small_one_is_not():
    many = [f"H{i:02d}" for i in range(10)]
    clear = analyze({"baseline": rows(missed=many), "jev+nh": rows()}, n_boot=2000)
    c = next(c for c in clear["comparisons"] if c["metric"] == "missed_handoff")
    assert c["diff"] == pytest.approx(-0.5) and c["hi"] < 0 and c["p"] < 0.01
    assert c["a_better"] == 10 and c["b_better"] == 0            # a = jev+nh fixes 10 of baseline's misses

    small = analyze({"baseline": rows(missed=SIX), "jev+nh": rows(missed=SIX[:3])}, n_boot=2000)
    s = next(c for c in small["comparisons"] if c["metric"] == "missed_handoff")
    assert s["p"] > 0.05                                       # 3 cases either way: could be chance


def test_default_comparisons_cover_the_questions_we_ask():
    data = {v: rows() for v in ("baseline", "jev", "jev+nh", "llm", "llm+nh")}
    pairs = {(c["a"], c["b"]) for c in analyze(data, n_boot=100)["comparisons"]}
    assert {("jev+nh", "baseline"), ("jev+nh", "jev"), ("jev+nh", "llm+nh")} <= pairs


def test_report_section_and_cli(tmp_path, capsys):
    data = {"baseline": rows(missed=SIX), "jev+nh": rows(missed=SIX[:1])}
    text = render_stats(analyze(data, n_boot=500))
    assert "统计显著性" in text and "McNemar" in text and "漏转率" in text
    (tmp_path / "summary.json").write_text(json.dumps({"rows": data}), encoding="utf-8")
    assert main([str(tmp_path), "--n-boot", "500"]) == 0
    assert "漏转率" in capsys.readouterr().out


def judged_rows(nh_human, nh_agent, conf_human, conf_agent):
    out = [{"case_id": f"H{i}", "gold_route": "human", "intent": "return_refund", "confidence": c, "needs_human": n}
           for i, (n, c) in enumerate(zip(nh_human, conf_human))]
    out += [{"case_id": f"A{i}", "gold_route": "agent", "intent": "return_refund", "confidence": c, "needs_human": n}
            for i, (n, c) in enumerate(zip(nh_agent, conf_agent))]
    return out + [{"case_id": "X", "gold_route": "human", "intent": "request_human", "confidence": 0.99,
                   "needs_human": 0.0}]            # intent forces a handoff: not a test of the signals


def test_separation_interval_matches_the_report_and_shows_the_gap():
    from eval.metrics import separation
    from eval.stats import separation_interval
    rows = judged_rows([0.9, 0.8, 0.7, 0.6] * 3, [0.1, 0.2, 0.3, 0.05] * 4,
                       [0.9, 0.95, 0.6, 0.99] * 3, [0.9, 0.7, 0.98, 0.95] * 4)
    sep = separation_interval(rows, n_boot=1000)
    assert sep["auroc_needs_human"] == pytest.approx(separation(rows)["auroc_needs_human"])
    assert sep["auroc_confidence"] == pytest.approx(separation(rows)["auroc_confidence"])
    assert sep["n_human"] == 12 and sep["auroc_needs_human"] == 1.0 and sep["diff_ci"][0] > 0
    assert separation_interval(rows[:3], n_boot=10) is None          # no bot-gold turns: nothing to separate
