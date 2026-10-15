"""Run the three-arm handoff experiment.

    python3 -m eval.run_eval                     # real run: needs DeepSeek + TypeSafe keys in .env
    python3 -m eval.run_eval --mock              # offline smoke run, numbers are meaningless
    python3 -m eval.run_eval --arms baseline,jev --cases T01,T19 --workers 2

For each (case, arm) the judge is called once and the agent at most once;
every threshold in --thresholds is then scored offline (see metrics.materialize).
Each judge arm is scored twice from the same records: "jev" / "llm" with the
intent-only gate, and "jev+nh" / "llm+nh" with the needs_human check added.
Outputs eval/results/<timestamp>/{results.jsonl, summary.json, report.md}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from cs_agent.agent import CustomerServiceAgent, build_judge, build_llm
from cs_agent.config import default_needs_human_threshold, default_threshold, load_env
from cs_agent.judge import GatePolicy, Judgment, build_state, gate
from cs_agent.session import new_session
from eval.metrics import DATASET_PATH, calibration, load_dataset, materialize, separation, summarize
from eval.quality import QualityJudge
from eval.report import ARM_LABEL, pct, render_report

ARMS = ("baseline", "jev", "llm", "fusion")
ARM_BACKEND = {"baseline": "none", "jev": "jev", "llm": "llm", "fusion": "fusion"}
# variant -> (arm whose records it replays, use the needs_human check?)
VARIANTS = {"baseline": ("baseline", False), "jev": ("jev", False), "jev+nh": ("jev", True),
            "llm": ("llm", False), "llm+nh": ("llm", True),
            "fusion": ("fusion", False), "fusion+nh": ("fusion", True)}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def needs_agent(judgment: Optional[Judgment], min_tau: float, margin: float) -> bool:
    """Would the agent run under the most permissive gate any variant replays
    (lowest threshold, needs_human check off)?"""
    policy = GatePolicy(min_tau, margin, needs_human_threshold=None)
    return gate(judgment, policy).route == "agent"


def run_case(case: dict, arm: str, llm, judge, min_tau: float, margin: float,
             quality_judge: Optional[QualityJudge] = None) -> dict:
    record = {"case_id": case["id"], "arm": arm, "judgment": None, "judge_error": "",
              "agent": None, "agent_error": "", "seconds": 0.0}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as memory_root:
        # Fresh store + seeded memory per case; the simulated customer confirms every action.
        session = new_session(case["customer_id"], Path(memory_root), confirm_fn=lambda _: True)
        agent = CustomerServiceAgent(llm, session)
        agent.preload_history([tuple(turn) for turn in case["history"]])
        judgment, run_agent = None, True
        if judge is not None:
            try:
                judgment = judge.judge(build_state(agent.transcript, case["message"]))
                record["judgment"] = judgment.to_dict()
            except Exception as e:
                record["judge_error"] = f"{type(e).__name__}: {e}"
            run_agent = needs_agent(judgment, min_tau, margin)
        if run_agent:
            try:
                r = agent.respond(case["message"], judgment)
                record["agent"] = {"route": r.route, "reason": r.reason, "reply": r.reply, "tools": r.tools}
            except Exception as e:
                record["agent_error"] = f"{type(e).__name__}: {e}"
            if quality_judge is not None and record["agent"] and record["agent"]["route"] == "agent":
                record["quality"] = quality_judge.score(case, record["agent"]["reply"], record["agent"]["tools"])
    record["seconds"] = round(time.perf_counter() - started, 2)
    return record


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="置信度驱动转人工 · 三组对比实验")
    p.add_argument("--arms", default=",".join(ARMS), help="baseline,jev,llm 的子集")
    p.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--tau", type=float, default=None, help="主阈值，默认 HANDOFF_THRESHOLD 或 0.5")
    p.add_argument("--margin", type=float, default=0.1, help="敏感意图额外加的阈值")
    p.add_argument("--needs-human", type=float, default=None,
                   help="needs_human 阈值，默认 NEEDS_HUMAN_THRESHOLD 或 0.5")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mock", action="store_true", help="离线：MockLLM + MockJudge，数字无意义")
    p.add_argument("--cases", default="", help="只跑这些 id，逗号分隔")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dataset", default=str(DATASET_PATH))
    p.add_argument("--no-quality", action="store_true", help="不做 LLM-as-Judge 回复质量评测（省一次模型调用/条）")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    load_env()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        sys.exit(f"unknown arms: {unknown}")
    tau = default_threshold() if args.tau is None else args.tau
    nh_threshold = default_needs_human_threshold() if args.needs_human is None else args.needs_human
    thresholds = sorted({round(float(t), 4) for t in args.thresholds.split(",")} | {tau})

    cases = load_dataset(Path(args.dataset))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]
    by_id = {c["id"]: c for c in cases}

    llm = build_llm(args.mock)
    judges = {arm: build_judge(ARM_BACKEND[arm], llm, args.mock) for arm in arms}
    quality_judge = None if args.no_quality else QualityJudge(llm)

    started_at = datetime.now()
    t0 = time.perf_counter()
    records = []
    jobs = [(case, arm) for arm in arms for case in cases]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_case, case, arm, llm, judges[arm], thresholds[0], args.margin, quality_judge)
                   for case, arm in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            records.append(rec)
            j = rec["judgment"]
            judged = f" {j['intent']}@{j['confidence']:.2f}" if j else ""
            problem = rec["judge_error"] or rec["agent_error"]
            print(f"[{done:>3}/{len(jobs)}] {rec['case_id']} {rec['arm']:<8}{judged}"
                  f"{' ERROR ' + problem[:80] if problem else ''} ({rec['seconds']}s)", file=sys.stderr)
    records.sort(key=lambda r: (arms.index(r["arm"]), r["case_id"]))

    variants = [v for v, (arm, _) in VARIANTS.items() if arm in arms]
    summary, rows_by_tau = {}, {}
    for variant in variants:
        arm, use_nh = VARIANTS[variant]
        mine = [r for r in records if r["arm"] == arm]
        nh = nh_threshold if use_nh else None
        rows_by_tau[variant] = {f"{t:g}": [materialize(r, by_id[r["case_id"]], t, args.margin, nh)
                                           for r in mine] for t in thresholds}
        summary[variant] = {t: summarize(rows) for t, rows in rows_by_tau[variant].items()}
    tau_key = f"{tau:g}"
    rows_main = {v: rows_by_tau[v][tau_key] for v in variants}
    judge_arms = [a for a in arms if a != "baseline"]
    calib = {a: calibration(rows_main[a]) for a in judge_arms}
    sep = {a: separation(rows_main[a]) for a in judge_arms}

    meta = {"started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.perf_counter() - t0, 1), "mock": args.mock, "arms": arms,
            "model_id": "MockLLM" if args.mock else os.getenv("MODEL_ID", "?"),
            "jev_model": "MockJudge" if args.mock else os.getenv("JEV_MODEL", "jev-latest"),
            "n_cases": len(cases), "tau": tau, "tau_key": tau_key, "margin": args.margin,
            "needs_human_threshold": nh_threshold, "thresholds": thresholds, "variants": variants,
            "dataset": Path(args.dataset).name, "quality": not args.no_quality,
            "holdout": Path(args.dataset).resolve() != DATASET_PATH.resolve()}

    out_dir = Path(args.out) if args.out else RESULTS_DIR / (
        started_at.strftime("%Y%m%d-%H%M%S") + ("-mock" if args.mock else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(
        {"meta": meta, "summary": summary,
         "calibration": {a: {"bins": b, "ece": e} for a, (b, e) in calib.items()},
         "separation": sep, "rows": rows_main}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(meta, summary, calib, sep, rows_main, cases),
                                       encoding="utf-8")

    print(("\n[MOCK 模式：数字无实验意义]" if args.mock else "") + f"\n主结果 τ={tau}")
    print(f"{'':<14}" + "".join(f"{ARM_LABEL[v]:<16}" for v in variants))
    for key, label in [("route_acc", "路由准确率"), ("e2e_acc", "端到端正确率"), ("intent_acc", "意图准确率"),
                       ("handoff_rate", "转人工率"), ("missed_handoff", "漏转率"), ("quality_mean", "回复质量"),
                       ("failures", "系统故障数")]:
        def fmt(value, key=key):
            if key == "failures":
                return str(value)
            return "—" if value is None else f"{value:.2f}" if key == "quality_mean" else pct(value)
        cells = [fmt(summary[v][tau_key][key]) for v in variants]
        print(f"{label:<12}" + "".join(f"{c:<18}" for c in cells))
    print(f"\n报告：{out_dir / 'report.md'}")
    return out_dir


if __name__ == "__main__":
    main()
