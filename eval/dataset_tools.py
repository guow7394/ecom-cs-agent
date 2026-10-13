"""Tools for writing a new (held-out) test set in a spreadsheet.

    python3 -m eval.dataset_tools convert eval/holdout/cases.csv --out eval/holdout/dataset.jsonl
    python3 -m eval.dataset_tools blind   eval/holdout/cases.csv --out eval/holdout/for_labeler_b.csv
    python3 -m eval.dataset_tools agree   eval/holdout/cases.csv eval/holdout/labels_b.csv

CSV is UTF-8 (with BOM, so Excel shows Chinese correctly). See eval/holdout/编写指南.md.
Rows whose id starts with "EX" are examples and are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from cs_agent.judge import INTENTS
from cs_agent.store import Store
from cs_agent.tools import TOOL_SCHEMAS

HOLDOUT_DIR = Path(__file__).resolve().parent / "holdout"
TEMPLATE_PATH = HOLDOUT_DIR / "template.csv"
CSV_FIELDS = ["id", "category", "customer_id", "history", "message", "gold_route",
              "gold_intents", "expect_tools", "forbid_success", "note"]
BLIND_FIELDS = ["id", "customer_id", "history", "message", "gold_route"]
CATEGORIES = ("standard", "trap", "trap_human", "explicit_human", "ambiguous", "adversarial")
ROLES = {"顾客": "customer", "客服": "agent"}
EXAMPLE_PREFIX = "EX"
_LIST_SPLIT = re.compile(r"[/、,，;；]")
_TURN_SPLIT = re.compile(r"\s*\|\|\s*")
_ORDER_ID = re.compile(r"[A-Z]\d{4}")
_ROLE_PREFIX = re.compile(r"^([^:：]{1,6})[:：](.*)$", re.S)


def _read_rows(path: Path, include_examples: bool = False) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [{k: (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get("id") or r.get("message")]
    if not include_examples:
        rows = [r for r in rows if not r.get("id", "").upper().startswith(EXAMPLE_PREFIX)]
    return rows


def _split_list(text: str) -> List[str]:
    return [x.strip() for x in _LIST_SPLIT.split(text or "") if x.strip()]


def _parse_history(text: str) -> List[List[str]]:
    """"顾客：… || 客服：…" -> [["customer", "…"], ["agent", "…"]]; unknown roles are
    kept as written so validate_cases can point at them."""
    turns = []
    for part in _TURN_SPLIT.split(text or ""):
        part = part.strip()
        if not part:
            continue
        m = _ROLE_PREFIX.match(part)
        role, content = (m.group(1).strip(), m.group(2)) if m else ("?", part)
        turns.append([ROLES.get(role, role), content.strip()])
    return turns


def parse_csv(path: Path, include_examples: bool = False) -> List[dict]:
    return [{
        "id": r.get("id", ""),
        "category": r.get("category", ""),
        "customer_id": r.get("customer_id", "").upper(),
        "history": _parse_history(r.get("history", "")),
        "message": r.get("message", ""),
        "gold_intents": _split_list(r.get("gold_intents", "")),
        "gold_route": r.get("gold_route", "").lower(),
        "expect_tools": _split_list(r.get("expect_tools", "")),
        "forbid_success": _split_list(r.get("forbid_success", "")),
        "note": r.get("note", ""),
    } for r in _read_rows(Path(path), include_examples)]


def validate_cases(cases: List[dict]) -> List[str]:
    store = Store()
    errors = []
    seen = Counter(c["id"] for c in cases)
    for c in cases:
        cid = c["id"] or "（无 id）"

        def err(msg):
            errors.append(f"{cid}: {msg}")

        if seen[c["id"]] > 1:
            err("id 重复")
        if c["category"] not in CATEGORIES:
            err(f"未知类别 {c['category']!r}，可选：{'/'.join(CATEGORIES)}")
        if c["customer_id"] not in store.customers:
            err(f"未知顾客 {c['customer_id']!r}")
        if not c["message"].strip():
            err("message 为空")
        if c["gold_route"] not in ("agent", "human"):
            err(f"gold_route 只能是 agent 或 human，现在是 {c['gold_route']!r}")
        if not c["gold_intents"]:
            err("gold_intents 至少填一个")
        for intent in c["gold_intents"]:
            if intent not in INTENTS:
                err(f"未知意图 {intent!r}，可选：{'/'.join(INTENTS)}")
        for tool in c["expect_tools"] + c["forbid_success"]:
            if tool not in TOOL_SCHEMAS:
                err(f"未知工具 {tool!r}")
        if c["gold_route"] == "human" and c["expect_tools"]:
            err("gold_route=human 时不应填 expect_tools（转人工的样本不要求机器人调工具）")
        roles = [role for role, _ in c["history"]]
        if roles != ["customer", "agent"] * (len(roles) // 2) + (["customer"] if len(roles) % 2 else []):
            err("history 必须顾客/客服交替，从顾客开始，每句写成「顾客：…」或「客服：…」")
        if roles and roles[-1] != "agent":
            err("history 必须以客服结尾（最后一句顾客的话放在 message 里）")
        if c["customer_id"] in store.customers:
            text = c["message"] + " " + " ".join(t for _, t in c["history"])
            for oid in _ORDER_ID.findall(text):
                if store.get_order(c["customer_id"], oid) is None:
                    err(f"订单 {oid} 不属于顾客 {c['customer_id']}")
    return errors


def route_agreement(cases: List[dict], other: Dict[str, str]) -> dict:
    """Agreement on gold_route between labeler A (cases) and labeler B, with Cohen's kappa."""
    pairs = [(c["id"], c["gold_route"], other[c["id"]]) for c in cases if other.get(c["id"])]
    missing = [c["id"] for c in cases if not other.get(c["id"])]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agree_rate": None, "kappa": None, "disagreements": [], "missing": missing}
    po = sum(a == b for _, a, b in pairs) / n
    pa = sum(a == "human" for _, a, _ in pairs) / n
    pb = sum(b == "human" for _, _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    kappa = 0.0 if abs(kappa) < 1e-12 else kappa  # avoid printing -0.00
    return {"n": n, "agree_rate": po, "kappa": kappa, "missing": missing,
            "disagreements": [(i, a, b) for i, a, b in pairs if a != b]}


def _write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="留出测试集工具")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("convert", help="校验 CSV 并转成实验用的 jsonl")
    p.add_argument("csv")
    p.add_argument("--out", required=True)
    p = sub.add_parser("blind", help="生成给第二位标注者的盲标副本（去掉所有标注）")
    p.add_argument("csv")
    p.add_argument("--out", required=True)
    p = sub.add_parser("agree", help="计算两位标注者在 gold_route 上的一致性")
    p.add_argument("csv_a")
    p.add_argument("csv_b")
    args = parser.parse_args(argv)

    if args.cmd == "convert":
        cases = parse_csv(Path(args.csv))
        if not cases:
            print("没有样本（EX 开头的示例行会被跳过），未生成文件")
            return 1
        errors = validate_cases(cases)
        if errors:
            print(f"发现 {len(errors)} 处问题，未生成文件：", *errors, sep="\n  ")
            return 1
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
        routes = Counter(c["gold_route"] for c in cases)
        cats = Counter(c["category"] for c in cases)
        multi = sum(bool(c["history"]) for c in cases)
        print(f"✓ {len(cases)} 条 → {out}\n  去向：{dict(routes)}\n  类别：{dict(cats)}\n  多轮：{multi} 条")
        return 0

    if args.cmd == "blind":
        rows = _read_rows(Path(args.csv))
        _write_csv(Path(args.out), BLIND_FIELDS, [dict(r, gold_route="") for r in rows])
        print(f"✓ 盲标副本 {len(rows)} 条 → {args.out}（只保留 id/顾客/对话，gold_route 留空）")
        return 0

    cases = parse_csv(Path(args.csv_a))
    other = {r.get("id", ""): r.get("gold_route", "").lower() for r in _read_rows(Path(args.csv_b))}
    res = route_agreement(cases, other)
    if not res["n"]:
        print("两份文件没有可比较的样本")
        return 1
    print(f"比较 {res['n']} 条：一致率 {res['agree_rate']:.1%}，Cohen's kappa = {res['kappa']:.2f}"
          "（≥0.8 很好，0.6–0.8 可接受，<0.6 需要先统一标注规则）")
    for i, a, b in res["disagreements"]:
        print(f"  不一致 {i}: A={a} B={b}")
    if res["missing"]:
        print(f"  B 没有标注：{', '.join(res['missing'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
