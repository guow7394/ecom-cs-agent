"""HTTP layer of the web demo (FastAPI TestClient, mock mode)."""
import json

import pytest
from fastapi.testclient import TestClient

import web.server as server
from web.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app(mock=True))


def sse_events(response):
    return [json.loads(line[len("data: "):]) for line in response.iter_lines() if line.startswith("data: ")]


def new_session(client, **overrides):
    body = {"customer_id": "C001", "judge": "jev", "threshold": 0.5, **overrides}
    response = client.post("/api/sessions", json=body)
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def test_config(client):
    cfg = client.get("/api/config").json()
    assert cfg["mock"] is True and cfg["judges"] == {"jev": True, "llm": True, "none": True}
    assert cfg["default_threshold"] == 0.5


def test_customers_carry_orders_with_readable_notes(client):
    customers = client.get("/api/customers").json()
    assert len(customers) == 10
    c001 = next(c for c in customers if c["id"] == "C001")
    assert c001["name"] == "王女士" and "鞋码: 37" in c001["memory"]
    a1001 = next(o for o in c001["orders"] if o["order_id"] == "A1001")
    assert a1001 == {**a1001, "item": "轻跑运动鞋", "size": "37", "amount": 399, "note": "已签收 · 第 4 天"}
    notes = {o["order_id"]: o["note"] for c in customers for o in c["orders"]}
    assert notes["B0002"] == "物流 5 天未更新" and notes["A1002"] == "运输中"
    assert notes["B9002"] == "待发货 · 定制商品"


@pytest.mark.parametrize("body", [
    {"customer_id": "C999"},
    {"customer_id": "C001", "judge": "gpt"},
    {"customer_id": "C001", "threshold": 0.9},
    {"customer_id": "C001", "threshold": 0.1},
])
def test_session_validation(client, body):
    assert client.post("/api/sessions", json=body).status_code == 400


def test_slider_float_noise_is_accepted(client):
    new_session(client, threshold=0.30000000000000004)


def test_message_streams_events(client):
    sid = new_session(client)
    with client.stream("POST", f"/api/sessions/{sid}/messages", json={"text": "A1002 的快递到哪了"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r)
    assert [e["type"] for e in events][:2] == ["judgment", "gate"]
    assert events[-2]["type"] == "reply" and "（模拟回复）" in events[-2]["text"]
    assert events[-1]["type"] == "done"


def test_error_codes(client):
    assert client.post("/api/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    assert client.post("/api/sessions/nope/confirm", json={"request_id": "x", "approve": True}).status_code == 404
    sid = new_session(client)
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "   "}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/confirm", json={"request_id": "x", "approve": True}).status_code == 404


def test_ended_session_returns_409(client):
    sid = new_session(client, customer_id="C006")
    with client.stream("POST", f"/api/sessions/{sid}/messages", json={"text": "转人工！我要人工客服"}) as r:
        assert sse_events(r)[-1]["ended"] is True
    response = client.post(f"/api/sessions/{sid}/messages", json={"text": "还在吗"})
    assert response.status_code == 409 and "新会话" in response.json()["detail"]


def test_without_keys_no_judge_is_available(monkeypatch):
    monkeypatch.setattr(server, "load_env", lambda: None)
    for key in ("ANTHROPIC_API_KEY", "MODEL_ID", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    client = TestClient(create_app(mock=False))
    assert client.get("/api/config").json()["judges"] == {"jev": False, "llm": False, "none": False}
    response = client.post("/api/sessions", json={"customer_id": "C001", "judge": "none"})
    assert response.status_code == 400 and "--mock" in response.json()["detail"]


def test_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "售后客服小安" in page.text
    for asset in ("app.js", "style.css"):
        assert client.get(f"/static/{asset}").status_code == 200


def test_frontend_never_injects_html():
    # customer and model text must be inserted with textContent, never parsed as HTML
    js = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "outerHTML" not in js


def test_health_reports_the_breaker(client):
    assert client.get("/api/health").json() == {"judges": {}}            # nothing built yet
    new_session(client)
    jev = client.get("/api/health").json()["judges"]["jev"]
    assert jev["state"] == "closed" and jev["failure_threshold"] == 3
