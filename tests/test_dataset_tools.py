import csv
import json

import pytest

from eval.dataset_tools import (CSV_FIELDS, TEMPLATE_PATH, main, parse_csv, route_agreement,
                                validate_cases)
from eval.metrics import DATASET_PATH, load_dataset


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    return path


ROWS = [
    {"id": "EX1", "category": "standard", "customer_id": "C007", "message": "示例",
     "gold_route": "agent", "gold_intents": "order_status", "note": "示例行"},
    {"id": "H01", "category": "standard", "customer_id": "C009", "history": "",
     "message": "篮球鞋小了 想换44的", "gold_route": "agent", "gold_intents": "exchange",
     "expect_tools": "create_exchange", "forbid_success": "", "note": "签收第 7 天"},
    {"id": "H02", "category": "ambiguous", "customer_id": "C010",
     "history": "顾客：我那个炸锅 || 客服：查到 B0002 空气炸锅，请问有什么问题？",
     "message": "算了不想等了，退钱，顺便问下上回那条裙子还能不能换",
     "gold_route": "human", "gold_intents": "return_refund / exchange、complaint_escalation",
     "expect_tools": "", "forbid_success": "apply_refund", "note": "多诉求"},
]


def test_parse_csv(tmp_path):
    cases = parse_csv(write_csv(tmp_path / "c.csv", ROWS))
    assert [c["id"] for c in cases] == ["H01", "H02"]           # EX rows are skipped
    h2 = cases[1]
    assert h2["history"] == [["customer", "我那个炸锅"], ["agent", "查到 B0002 空气炸锅，请问有什么问题？"]]
    assert h2["gold_intents"] == ["return_refund", "exchange", "complaint_escalation"]
    assert h2["forbid_success"] == ["apply_refund"] and h2["expect_tools"] == []
    assert cases[0]["history"] == []
    assert validate_cases(cases) == []
    assert [c["id"] for c in parse_csv(tmp_path / "c.csv", include_examples=True)] == ["EX1", "H01", "H02"]


@pytest.mark.parametrize("change,fragment", [
    ({"gold_intents": "refund_please"}, "未知意图"),
    ({"message": "A1001 的鞋退了吧"}, "A1001"),                       # someone else's order
    ({"expect_tools": "create_exchange", "gold_route": "human"}, "不应填 expect_tools"),
    ({"history": "顾客：一 || 顾客：二"}, "顾客/客服交替"),
    ({"history": "顾客：只有一句"}, "以客服结尾"),
    ({"expect_tools": "rm_rf"}, "未知工具"),
    ({"category": "weird"}, "未知类别"),
    ({"gold_route": "bot"}, "gold_route"),
    ({"message": "  "}, "message 为空"),
    ({"customer_id": "C999"}, "未知顾客"),
])
def test_validate_catches_mistakes(tmp_path, change, fragment):
    rows = [dict(ROWS[1], **change)]
    errors = validate_cases(parse_csv(write_csv(tmp_path / "c.csv", rows)))
    assert any(fragment in e for e in errors), errors


def test_validate_catches_duplicate_ids(tmp_path):
    errors = validate_cases(parse_csv(write_csv(tmp_path / "c.csv", [ROWS[1], ROWS[1]])))
    assert any("重复" in e for e in errors)


def test_main_dataset_is_valid():
    assert validate_cases(load_dataset(DATASET_PATH)) == []


def test_template_examples_are_valid():
    assert parse_csv(TEMPLATE_PATH) == []                          # only example rows
    examples = parse_csv(TEMPLATE_PATH, include_examples=True)
    assert examples and validate_cases(examples) == []


def test_convert_roundtrip(tmp_path):
    src = write_csv(tmp_path / "c.csv", ROWS)
    out = tmp_path / "d.jsonl"
    assert main(["convert", str(src), "--out", str(out)]) == 0
    assert load_dataset(out) == parse_csv(src)


def test_convert_refuses_invalid_file(tmp_path):
    src = write_csv(tmp_path / "c.csv", [dict(ROWS[1], gold_intents="nope")])
    out = tmp_path / "d.jsonl"
    assert main(["convert", str(src), "--out", str(out)]) == 1
    assert not out.exists()


def test_blind_copy_hides_labels(tmp_path):
    src = write_csv(tmp_path / "c.csv", ROWS)
    out = tmp_path / "b.csv"
    assert main(["blind", str(src), "--out", str(out)]) == 0
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["H01", "H02"]
    assert set(rows[0]) == {"id", "customer_id", "history", "message", "gold_route"}
    assert all(r["gold_route"] == "" for r in rows)


def test_route_agreement_and_kappa():
    cases = [{"id": i, "gold_route": r} for i, r in zip("abcd", ["human", "human", "agent", "agent"])]
    other = {"a": "human", "b": "agent", "c": "agent", "d": "agent"}
    res = route_agreement(cases, other)
    # po = 0.75; pe = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75-0.5)/(1-0.5)
    assert res["n"] == 4 and res["agree_rate"] == 0.75 and res["kappa"] == pytest.approx(0.5)
    assert res["disagreements"] == [("b", "human", "agent")]
    assert route_agreement(cases, {"a": "human"})["missing"] == ["b", "c", "d"]


def test_convert_refuses_empty_file(tmp_path):
    out = tmp_path / "d.jsonl"
    assert main(["convert", str(TEMPLATE_PATH), "--out", str(out)]) == 1   # only EX rows
    assert not out.exists()


def test_kappa_is_never_negative_zero():
    cases = [{"id": i, "gold_route": r} for i, r in zip("abc", ["agent", "agent", "human"])]
    res = route_agreement(cases, {"a": "agent", "b": "agent", "c": "agent"})
    assert res["kappa"] == 0.0 and f"{res['kappa']:.2f}" == "0.00"
