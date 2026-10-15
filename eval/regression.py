"""Regression check: did this run get worse than a baseline run?

    python3 -m eval.run_eval --dataset eval/holdout/dataset.jsonl --baseline eval/results/holdout-v1
    python3 -m eval.regression <new run dir> <baseline run dir> [--fail-on-regression]

For every variant both runs have, at each run's main tau:

    higher is better   route_acc e2e_acc intent_acc auto_acc quality_mean quality_pass
                       regression = relative drop of more than 5%
    should stay low    missed_handoff false_handoff
                       regression = rise of 5 percentage points or more (2 of 40 cases)
    failure counts     failures errors
                       regression = any increase
    separation         AUROC of confidence / needs_human, per judge: relative drop > 5%

The tolerances absorb the one-case wobble the same LLM gives run to run;
they are arguments, not constants, for a larger set. handoff_rate and
coverage are not checked: fewer handoffs is only good if they were right,
which route_acc and missed_handoff already say.

Runs are only comparable on the same cases: if the case ids differ the
check says so and its verdict should not be trusted. Changes to the setup
(model, Jev version, tau, mock) are listed, since they are often the
point of the comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from eval.report import ARM_LABEL, METRICS

HIGHER_BETTER = ["route_acc", "e2e_acc", "intent_acc", "auto_acc", "quality_mean", "quality_pass"]
LOWER_BETTER = ["missed_handoff", "false_handoff"]
COUNTS = ["failures", "errors"]
SEPARATION = ["auroc_confidence", "auroc_needs_human"]
SETUP_KEYS = ["model_id", "jev_model", "tau_key", "mock", "needs_human_threshold", "margin"]
LABEL = {key: label for key, label, _ in METRICS}
LABEL.update({"failures": "系统故障数", "errors": "判断层出错数",
              "auroc_confidence": "AUROC·意图置信度", "auroc_needs_human": "AUROC·需人工分"})


def load_summary(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))


def _verdict(metric: str, old: float, new: float, rel_tol: float, abs_tol: float) -> str:
    if metric in HIGHER_BETTER or metric in SEPARATION:
        change = (new - old) / old if old else new - old
        return "regression" if change < -rel_tol else "improvement" if change > rel_tol else "same"
    if metric in LOWER_BETTER:
        rise = new - old
        return "regression" if rise >= abs_tol - 1e-9 else "improvement" if rise <= -abs_tol + 1e-9 else "same"
    return "regression" if new > old else "improvement" if new < old else "same"


def _change(variant: str, metric: str, old, new, rel_tol: float, abs_tol: float) -> Optional[dict]:
    if old is None or new is None:   # e.g. quality not scored in one of the runs
        return None
    return {"variant": variant, "metric": metric, "label": LABEL.get(metric, metric), "old": old, "new": new,
            "delta": new - old, "verdict": _verdict(metric, old, new, rel_tol, abs_tol)}


def _case_ids(summary: dict) -> set:
    return {row["case_id"] for rows in summary.get("rows", {}).values() for row in rows}


def compare(new: dict, base: dict, rel_tol: float = 0.05, abs_tol: float = 0.05) -> dict:
    warnings: List[str] = []
    new_ids, base_ids = _case_ids(new), _case_ids(base)
    comparable = new_ids == base_ids
    if not comparable:
        warnings.append(f"两次评测的用例不同（新 {len(new_ids)} 条、基线 {len(base_ids)} 条、"
                        f"共有 {len(new_ids & base_ids)} 条），回归结论不可信")
    for key in SETUP_KEYS:
        old_v, new_v = base["meta"].get(key), new["meta"].get(key)
        if old_v != new_v:
            warnings.append(f"配置变了：{key} {old_v} → {new_v}")

    changes: List[dict] = []
    tau_new, tau_base = new["meta"]["tau_key"], base["meta"]["tau_key"]
    variants = [v for v in new["summary"] if v in base["summary"]]
    for variant in variants:
        now, before = new["summary"][variant][tau_new], base["summary"][variant][tau_base]
        for metric in HIGHER_BETTER + LOWER_BETTER + COUNTS:
            changes.append(_change(variant, metric, before.get(metric), now.get(metric), rel_tol, abs_tol))
    for judge in (j for j in new.get("separation", {}) if j in base.get("separation", {})):
        for metric in SEPARATION:
            changes.append(_change(judge, metric, base["separation"][judge].get(metric),
                                   new["separation"][judge].get(metric), rel_tol, abs_tol))
    changes = [c for c in changes if c]

    flipped: Dict[str, dict] = {}
    for variant in variants:
        before = {r["case_id"]: r.get("e2e_ok") for r in base.get("rows", {}).get(variant, [])}
        now = {r["case_id"]: r.get("e2e_ok") for r in new.get("rows", {}).get(variant, [])}
        shared = sorted(set(before) & set(now))
        flipped[variant] = {"broke": [c for c in shared if before[c] and not now[c]],
                            "fixed": [c for c in shared if not before[c] and now[c]]}

    return {"baseline": {"started_at": base["meta"].get("started_at"), "dataset": base["meta"].get("dataset")},
            "comparable": comparable, "warnings": warnings, "checked": len(changes),
            "regressions": [c for c in changes if c["verdict"] == "regression"],
            "improvements": [c for c in changes if c["verdict"] == "improvement"],
            "flipped": flipped}


def _fmt(metric: str, value) -> str:
    if metric in COUNTS:
        return str(value)
    if metric in SEPARATION or metric == "quality_mean":
        return f"{value:.3f}"
    return f"{value * 100:.1f}%"


def _row(c: dict) -> str:
    who = ARM_LABEL.get(c["variant"], c["variant"])
    return f"| {who} | {c['label']}（{c['metric']}） | {_fmt(c['metric'], c['old'])} | {_fmt(c['metric'], c['new'])} |"


def render_check(check: dict) -> str:
    base = check["baseline"]
    out = ["## 回归检测", "", f"基线：{base['started_at']} 的评测（{base['dataset']}），"
           f"共比较 {check['checked']} 项。"]
    out += [f"> ⚠ {w}" for w in check["warnings"]]
    out.append("")
    for title, items in (("变差的指标", check["regressions"]), ("变好的指标", check["improvements"])):
        out.append(f"**{title}：{len(items) or '无'}**")
        if items:
            out += ["", "| 组 | 指标 | 基线 | 本次 |", "|---|---|---|---|"] + [_row(c) for c in items]
        out.append("")
    broke = {v: f for v, f in check["flipped"].items() if f["broke"] or f["fixed"]}
    if broke:
        out.append("**端到端结果翻转的用例**（对→错 / 错→对）：")
        out.append("")
        for variant, f in broke.items():
            out.append(f"- {ARM_LABEL.get(variant, variant)}：对→错 {', '.join(f['broke']) or '—'}；"
                       f"错→对 {', '.join(f['fixed']) or '—'}")
        out.append("")
    return "\n".join(out)


def print_check(check: dict) -> None:
    print(f"\n回归检测（基线 {check['baseline']['started_at']}，比较 {check['checked']} 项）")
    for w in check["warnings"]:
        print(f"  ⚠ {w}")
    if not check["regressions"]:
        print("  没有变差的指标")
    for c in check["regressions"]:
        print(f"  ✗ {ARM_LABEL.get(c['variant'], c['variant'])} {c['label']}："
              f"{_fmt(c['metric'], c['old'])} → {_fmt(c['metric'], c['new'])}")
    for c in check["improvements"]:
        print(f"  ✓ {ARM_LABEL.get(c['variant'], c['variant'])} {c['label']}："
              f"{_fmt(c['metric'], c['old'])} → {_fmt(c['metric'], c['new'])}")
    for variant, f in check["flipped"].items():
        if f["broke"] or f["fixed"]:
            print(f"  ↔ {ARM_LABEL.get(variant, variant)} 用例翻转：对→错 {', '.join(f['broke']) or '—'}；"
                  f"错→对 {', '.join(f['fixed']) or '—'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="对比两次评测，找出变差的指标")
    p.add_argument("new", help="本次评测的结果目录")
    p.add_argument("baseline", help="基线评测的结果目录")
    p.add_argument("--rel-tol", type=float, default=0.05, help="越高越好的指标：相对下降超过多少算变差")
    p.add_argument("--abs-tol", type=float, default=0.05, help="漏转率/误转率：上升多少（绝对值）算变差")
    p.add_argument("--fail-on-regression", action="store_true", help="有变差的指标时退出码为 1（给 CI 用）")
    args = p.parse_args(argv)
    check = compare(load_summary(Path(args.new)), load_summary(Path(args.baseline)), args.rel_tol, args.abs_tol)
    print_check(check)
    return 1 if args.fail_on_regression and check["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
