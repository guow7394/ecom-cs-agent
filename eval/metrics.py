"""Scoring for the handoff experiment. Pure functions, no I/O except loading.

A *record* is what run_eval stores per (case, arm): the judgment (if any)
and the agent's result (if the agent ran). A *row* is a record scored at
one threshold: materialize() replays the gate offline, which is exactly
what handle() would have done online, because gated turns never reach
the agent.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from cs_agent.judge import HUMAN_INTENTS, GatePolicy, Judgment, gate

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
CALIBRATION_EDGES = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
# Handoffs caused by the system failing rather than by a decision. They are routed
# to a human (fail safe), which can look "correct" on human-gold cases, so they
# must be reported next to accuracy, never hidden inside it.
FAILURE_REASONS = ("judge_error", "truncated", "loop_limit")


def load_dataset(path: Path = DATASET_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def case_correct(case: dict, route: str, tools: List[dict]) -> Tuple[bool, bool]:
    """(route_ok, e2e_ok). End-to-end also requires the right tool calls and
    that no forbidden action actually executed (an attempt blocked by the
    permission gate is fine)."""
    route_ok = route == case["gold_route"]
    if not route_ok:
        return False, False
    called = {t["name"] for t in tools}
    executed = {t["name"] for t in tools if t["ok"]}
    if set(case["forbid_success"]) & executed:
        return True, False
    if case["gold_route"] == "agent" and not set(case["expect_tools"]) <= called:
        return True, False
    return True, True


def materialize(record: dict, case: dict, tau: float, margin: float = 0.1,
                needs_human_threshold: Optional[float] = None) -> dict:
    """needs_human_threshold=None scores the intent-only gate; a number adds the
    needs_human check. Both replay the same judgment, so they are a clean ablation."""
    judgment = Judgment(**record["judgment"]) if record.get("judgment") else None
    route, reason, tools, error = None, "", [], ""
    if record["arm"] != "baseline":
        policy = GatePolicy(threshold=tau, sensitive_margin=margin,
                            needs_human_threshold=needs_human_threshold)
        decision = gate(judgment, policy)
        if decision.route == "human":
            route, reason = "human", decision.reason
    if route is None:
        agent = record.get("agent")
        if agent is None:
            route, error = "error", record.get("agent_error") or "agent result missing"
        else:
            route, reason, tools = agent["route"], agent["reason"], agent["tools"]
    route_ok, e2e_ok = case_correct(case, route, tools)
    return {
        "case_id": case["id"], "category": case["category"], "gold_route": case["gold_route"],
        "route": route, "reason": reason,
        "intent": judgment.intent if judgment else None,
        "confidence": judgment.confidence if judgment else None,
        "needs_human": judgment.needs_human if judgment else None,
        "intent_ok": (judgment.intent in case["gold_intents"]) if judgment else None,
        "route_ok": route_ok, "e2e_ok": e2e_ok, "error": error,
    }


def _rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    gold_human = [r for r in rows if r["gold_route"] == "human"]
    gold_agent = [r for r in rows if r["gold_route"] == "agent"]
    auto = [r for r in rows if r["route"] == "agent"]
    judged = [r for r in rows if r["intent_ok"] is not None]
    return {
        "n": n,
        "route_acc": _rate(sum(r["route_ok"] for r in rows), n),
        "e2e_acc": _rate(sum(r["e2e_ok"] for r in rows), n),
        "intent_acc": _rate(sum(r["intent_ok"] for r in judged), len(judged)),
        "handoff_rate": _rate(sum(r["route"] == "human" for r in rows), n),
        "missed_handoff": _rate(sum(r["route"] == "agent" for r in gold_human), len(gold_human)),
        "false_handoff": _rate(sum(r["route"] == "human" for r in gold_agent), len(gold_agent)),
        "coverage": _rate(len(auto), n),
        "auto_acc": _rate(sum(r["e2e_ok"] for r in auto), len(auto)),
        "errors": sum(r["route"] == "error" for r in rows),
        "failures": sum(r["route"] == "error" or r["reason"] in FAILURE_REASONS for r in rows),
        "reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
    }


def calibration(rows: Sequence[dict], edges: Sequence[float] = CALIBRATION_EDGES):
    """Intent accuracy per confidence bucket, plus expected calibration error."""
    judged = [r for r in rows if r["intent_ok"] is not None]
    bins, ece = [], 0.0
    for lo, hi in zip(edges, edges[1:]):
        last = hi == edges[-1]
        members = [r for r in judged if lo <= r["confidence"] < hi or (last and r["confidence"] == hi)]
        if members:
            acc = sum(r["intent_ok"] for r in members) / len(members)
            conf = sum(r["confidence"] for r in members) / len(members)
            ece += len(members) / len(judged) * abs(acc - conf)
        else:
            acc = conf = None
        bins.append({"lo": lo, "hi": hi, "n": len(members), "accuracy": acc, "mean_confidence": conf})
    return bins, (ece if judged else None)


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> Optional[float]:
    """P(a random positive scores higher than a random negative); ties count half.
    0.5 = no better than chance, 1.0 = perfect separation, independent of any threshold."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def separation(rows: Sequence[dict]) -> dict:
    """How well each judge signal separates human-gold from bot-gold turns, among
    turns whose intent does not already force a handoff (there, only the signals decide)."""
    judged = [r for r in rows if r["intent"] is not None and r["intent"] not in HUMAN_INTENTS]
    human = [r for r in judged if r["gold_route"] == "human"]
    agent = [r for r in judged if r["gold_route"] == "agent"]

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    has_nh = all(r["needs_human"] is not None for r in judged)
    return {
        "n_human": len(human), "n_agent": len(agent),
        "auroc_confidence": auroc([1 - r["confidence"] for r in human], [1 - r["confidence"] for r in agent]),
        "auroc_needs_human": auroc([r["needs_human"] for r in human],
                                   [r["needs_human"] for r in agent]) if has_nh else None,
        "mean_needs_human_human": mean(r["needs_human"] for r in human),
        "mean_needs_human_agent": mean(r["needs_human"] for r in agent),
    }
