"""Eval regression check: flag a variant whose metrics got worse than a baseline run."""
import copy
import json

import pytest

from eval.regression import compare, load_summary, main, render_check


def run(route_acc=0.875, missed=0.15, failures=0, quality=0.92, auroc_nh=0.93, cases=("H01", "H02", "H03"),
        **meta):
    metrics = {"n": len(cases), "route_acc": route_acc, "e2e_acc": route_acc, "intent_acc": 1.0,
               "missed_handoff": missed, "false_handoff": 0.1, "coverage": 0.5, "auto_acc": 0.9,
               "failures": failures, "errors": 0, "quality_mean": quality, "quality_pass": 0.9}
    rows = [{"case_id": c, "e2e_ok": True} for c in cases]
    return {"meta": {"tau_key": "0.5", "dataset": "dataset.jsonl", "mock": False, "model_id": "m",
                     "jev_model": "jev-1.13.0", "started_at": "2026-10-15 21:56:08", **meta},
            "summary": {"jev+nh": {"0.5": metrics}, "baseline": {"0.5": dict(metrics)}},
            "separation": {"jev": {"auroc_confidence": 0.74, "auroc_needs_human": auroc_nh}},
            "rows": {"jev+nh": rows, "baseline": copy.deepcopy(rows)}}


def names(items):
    return {(c["variant"], c["metric"]) for c in items}


def test_identical_runs_have_no_regressions():
    check = compare(run(), run())
    assert check["regressions"] == [] and check["improvements"] == [] and check["comparable"]


def test_small_wobble_is_tolerated_but_a_real_drop_is_flagged():
    assert compare(run(route_acc=0.86), run())["regressions"] == []              # -1.7% relative
    check = compare(run(route_acc=0.80), run())                                   # -8.6% relative
    assert ("jev+nh", "route_acc") in names(check["regressions"])


def test_rates_that_should_stay_low_use_percentage_points():
    assert compare(run(missed=0.18), run())["regressions"] == []                 # +3 pp
    assert ("jev+nh", "missed_handoff") in names(compare(run(missed=0.20), run())["regressions"])   # +5 pp
    assert ("jev+nh", "missed_handoff") in names(compare(run(missed=0.05), run())["improvements"])


def test_any_new_system_failure_is_a_regression():
    assert ("jev+nh", "failures") in names(compare(run(failures=1), run())["regressions"])


def test_separation_auroc_is_checked_per_judge():
    assert ("jev", "auroc_needs_human") in names(compare(run(auroc_nh=0.80), run())["regressions"])


def test_missing_quality_is_skipped_not_flagged():
    assert compare(run(quality=None), run())["regressions"] == []


def test_cases_that_flipped_are_listed():
    new = run()
    new["rows"]["jev+nh"][1]["e2e_ok"] = False
    assert compare(new, run())["flipped"]["jev+nh"] == {"broke": ["H02"], "fixed": []}


def test_a_different_dataset_is_not_comparable():
    check = compare(run(cases=("H01", "H02", "X9")), run())
    assert not check["comparable"] and any("用例" in w for w in check["warnings"])


def test_changed_setup_is_reported():
    check = compare(run(jev_model="jev-1.14.0", mock=True), run())
    assert any("jev-1.14.0" in w for w in check["warnings"]) and any("mock" in w.lower() for w in check["warnings"])


def test_report_section_and_cli(tmp_path, capsys):
    for name, summary in (("new", run(route_acc=0.70)), ("base", run())):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert load_summary(tmp_path / "base")["meta"]["tau_key"] == "0.5"
    text = render_check(compare(run(route_acc=0.70), run()))
    assert "回归检测" in text and "route_acc" in text
    assert main([str(tmp_path / "new"), str(tmp_path / "base")]) == 0
    assert main([str(tmp_path / "new"), str(tmp_path / "base"), "--fail-on-regression"]) == 1
    assert main([str(tmp_path / "base"), str(tmp_path / "base"), "--fail-on-regression"]) == 0
    assert "路由准确率" in capsys.readouterr().out
    broken = run()
    broken["rows"]["jev+nh"][0]["e2e_ok"] = False
    (tmp_path / "new" / "summary.json").write_text(json.dumps(broken), encoding="utf-8")
    main([str(tmp_path / "new"), str(tmp_path / "base")])
    assert "对→错 H01" in capsys.readouterr().out
