"""How sure can 40 cases make us? Bootstrap intervals and paired tests over cases.

On the held-out set, 20 cases need a human and 20 do not: one case moves
route accuracy by 2.5 points and the missed-handoff rate by 5 points, so a
table of rates alone says more than the data does.

    intervals     95% Wilson interval for each rate (the usual choice for a
                  proportion on few cases; unlike a bootstrap it does not collapse
                  to 0–0 when a variant made no errors).
    comparisons   two variants on the same cases: an exact McNemar test that looks
                  only at the cases where the two disagree (if A fixes 6 of B's
                  errors and breaks 1, is 6 vs 1 more than a coin would give?),
                  plus a bootstrap interval for the difference. The McNemar p is
                  the verdict; the bootstrap interval is approximate on so few cases.
    separation    AUROC of intent confidence and of needs_human per judge, with
                  bootstrap intervals, and their difference on the same resamples.

Bootstraps resample cases with replacement 10,000 times (fixed seed),
separately within the "needs a human" and "bot can handle it" halves so
every resample keeps the design's balance.

    python3 -m eval.stats eval/results/holdout-v1        # for a finished run

No correction is made for testing several pairs, so read a p near 0.05 as
"maybe", not "yes".
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb, sqrt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from cs_agent.judge import HUMAN_INTENTS
from eval.metrics import auroc
from eval.report import ARM_LABEL

# metric -> (which half of the cases it is measured on, per-case indicator, label)
METRICS: Dict[str, Tuple[Optional[str], Callable[[dict], bool], str]] = {
    "route_acc": (None, lambda r: bool(r["route_ok"]), "路由准确率"),
    "e2e_acc": (None, lambda r: bool(r["e2e_ok"]), "端到端正确率"),
    "missed_handoff": ("human", lambda r: r["route"] == "agent", "漏转率"),
    "false_handoff": ("agent", lambda r: r["route"] == "human", "误转率"),
}
LOWER_IS_BETTER = {"missed_handoff", "false_handoff"}
DEFAULT_PAIRS = [("jev+nh", "baseline"), ("llm+nh", "baseline"), ("fusion+nh", "baseline"),
                 ("jev", "baseline"), ("llm", "baseline"),
                 ("jev+nh", "jev"), ("llm+nh", "llm"), ("jev+nh", "llm+nh")]


def mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar test: binomial(n = discordant pairs, 0.5)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def _percentile(sorted_values: List[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))]


def _mean(values: Sequence[int]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def analyze(rows_by_variant: Dict[str, List[dict]], pairs: Optional[Sequence[Tuple[str, str]]] = None,
            n_boot: int = 10000, seed: int = 0) -> dict:
    variants = list(rows_by_variant)
    by_id = {v: {r["case_id"]: r for r in rows_by_variant[v]} for v in variants}
    ids = sorted(set.intersection(*(set(m) for m in by_id.values())))
    gold = {cid: by_id[variants[0]][cid]["gold_route"] for cid in ids}
    halves = {half: [c for c in ids if gold[c] == half] for half in ("human", "agent")}
    # indicator[v][metric][case_id] in {0, 1}, only for the cases the metric is measured on
    indicator = {v: {m: {c: int(fn(by_id[v][c])) for c in ids if half is None or gold[c] == half}
                     for m, (half, fn, _) in METRICS.items()} for v in variants}
    pairs = [(a, b) for a, b in (pairs or DEFAULT_PAIRS) if a in by_id and b in by_id]

    def value(v: str, metric: str, sample: Dict[str, List[str]]) -> Optional[float]:
        half = METRICS[metric][0]
        cases = sample["human"] + sample["agent"] if half is None else sample[half]
        return _mean([indicator[v][metric][c] for c in cases])

    rng = random.Random(seed)
    boot = {(v, m): [] for v in variants for m in METRICS}
    for _ in range(n_boot):
        sample = {half: rng.choices(members, k=len(members)) if members else [] for half, members in halves.items()}
        for key in boot:
            boot[key].append(value(*key, sample))

    full = {"human": halves["human"], "agent": halves["agent"]}
    intervals: Dict[str, Dict[str, dict]] = {}
    for v in variants:
        intervals[v] = {}
        for m in METRICS:
            k, n = sum(indicator[v][m].values()), len(indicator[v][m])
            lo, hi = wilson(k, n)
            intervals[v][m] = {"value": value(v, m, full), "k": k, "n": n, "lo": lo, "hi": hi}

    comparisons = []
    for a, b in pairs:
        for m in METRICS:
            if intervals[a][m]["value"] is None or intervals[b][m]["value"] is None:
                continue
            diffs = sorted(x - y for x, y in zip(boot[(a, m)], boot[(b, m)]) if x is not None and y is not None)
            lower = m in LOWER_IS_BETTER   # "better" = indicator 0 for error rates, 1 for accuracies
            ia, ib = indicator[a][m], indicator[b][m]
            a_better = sum(1 for c in ia if (ia[c] < ib[c] if lower else ia[c] > ib[c]))
            b_better = sum(1 for c in ia if (ib[c] < ia[c] if lower else ib[c] > ia[c]))
            comparisons.append({"a": a, "b": b, "metric": m,
                                "a_value": intervals[a][m]["value"], "b_value": intervals[b][m]["value"],
                                "diff": intervals[a][m]["value"] - intervals[b][m]["value"],
                                "lo": _percentile(diffs, 0.025), "hi": _percentile(diffs, 0.975),
                                "a_better": a_better, "b_better": b_better, "p": mcnemar_p(a_better, b_better)})
    separation = {v: separation_interval(rows_by_variant[v], n_boot, seed)
                  for v in variants if "+" not in v and v != "baseline"}
    return {"n_cases": len(ids), "n_human": len(halves["human"]), "n_agent": len(halves["agent"]),
            "n_boot": n_boot, "seed": seed, "intervals": intervals, "comparisons": comparisons,
            "separation": {v: s for v, s in separation.items() if s}}


def separation_interval(rows: List[dict], n_boot: int = 10000, seed: int = 0) -> Optional[dict]:
    """Bootstrap intervals for eval.metrics.separation(): the same turns (intent does not force a
    handoff), confidence flipped so that higher = more likely to need a human."""
    judged = [r for r in rows if r.get("intent") is not None and r["intent"] not in HUMAN_INTENTS
              and r.get("needs_human") is not None]
    human = [(1 - r["confidence"], r["needs_human"]) for r in judged if r["gold_route"] == "human"]
    agent = [(1 - r["confidence"], r["needs_human"]) for r in judged if r["gold_route"] == "agent"]
    if not human or not agent:
        return None

    def both(h, a):
        return auroc([x[0] for x in h], [x[0] for x in a]), auroc([x[1] for x in h], [x[1] for x in a])

    rng = random.Random(seed)
    conf, nh, diff = [], [], []
    for _ in range(n_boot):
        c, n = both(rng.choices(human, k=len(human)), rng.choices(agent, k=len(agent)))
        conf.append(c), nh.append(n), diff.append(n - c)
    point_c, point_n = both(human, agent)

    def ci(values):
        values = sorted(values)
        return _percentile(values, 0.025), _percentile(values, 0.975)

    return {"n_human": len(human), "n_agent": len(agent),
            "auroc_confidence": point_c, "auroc_confidence_ci": ci(conf),
            "auroc_needs_human": point_n, "auroc_needs_human_ci": ci(nh),
            "diff": point_n - point_c, "diff_ci": ci(diff)}


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def _verdict(c: dict) -> str:
    if c["p"] < 0.05:
        return "显著"
    if c["p"] < 0.1:
        return "边缘"
    return "不显著"


def render_stats(stats: dict) -> str:
    out = ["## 统计显著性", "",
           f"共 {stats['n_cases']} 条用例（需人工 {stats['n_human']} 条、机器人可处理 {stats['n_agent']} 条）。"
           "各组比例的区间是 95% Wilson 区间。两组对比以精确 McNemar 检验为准（只看两组结论不同的用例），"
           f"差值区间来自 bootstrap（按两类用例分层重抽样 {stats['n_boot']} 次，种子 {stats['seed']}），样本这么少时只是近似。"
           "没有做多重比较校正，p 接近 0.05 时只能算「可能」。", ""]
    variants = list(stats["intervals"])
    header = ["组"] + [label for _, (_, _, label) in METRICS.items()]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for v in variants:
        cells = [f"{_pct(i['value'])}（{_pct(i['lo'])}–{_pct(i['hi'])}）" for i in stats["intervals"][v].values()]
        out.append(f"| {ARM_LABEL.get(v, v)} | " + " | ".join(cells) + " |")
    out += ["", "| 对比（A vs B） | 指标 | A | B | 差值（百分点，95% 区间） | A 更好 / B 更好（条） | p | 结论 |",
            "|---|---|---|---|---|---|---|---|"]
    for c in stats["comparisons"]:
        out.append(f"| {ARM_LABEL.get(c['a'], c['a'])} vs {ARM_LABEL.get(c['b'], c['b'])} | {METRICS[c['metric']][2]} | "
                   f"{_pct(c['a_value'])} | {_pct(c['b_value'])} | {_pp(c['diff'])}（{_pp(c['lo'])} ~ {_pp(c['hi'])}） | "
                   f"{c['a_better']} / {c['b_better']} | {c['p']:.3f} | {_verdict(c)} |")
    if stats.get("separation"):
        out += ["", "区分能力（AUROC，95% bootstrap 区间；只看意图本身不强制转人工的轮次）：", "",
                "| 判断层 | 样本（需人工 / 可处理） | 意图置信度 | 需人工分 | 需人工分 − 意图置信度 |", "|---|---|---|---|---|"]
        for v, sep in stats["separation"].items():
            def f(x, rng):
                return f"{x:.3f}（{rng[0]:.3f}–{rng[1]:.3f}）"
            d_lo, d_hi = sep["diff_ci"]
            out.append(f"| {ARM_LABEL.get(v, v).split('·')[0]} | {sep['n_human']} / {sep['n_agent']} | "
                       f"{f(sep['auroc_confidence'], sep['auroc_confidence_ci'])} | "
                       f"{f(sep['auroc_needs_human'], sep['auroc_needs_human_ci'])} | "
                       f"{sep['diff']:+.3f}（{d_lo:+.3f} ~ {d_hi:+.3f}） |")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="评测结果的置信区间与配对检验")
    p.add_argument("run_dir", help="评测结果目录（含 summary.json）")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    summary = json.loads((Path(args.run_dir) / "summary.json").read_text(encoding="utf-8"))
    print(render_stats(analyze(summary["rows"], n_boot=args.n_boot, seed=args.seed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
