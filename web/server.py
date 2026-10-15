"""Local demo server for the after-sales agent.

    python3 -m web.server            # real models, keys from .env
    python3 -m web.server --mock     # no keys needed; keyword mock, numbers mean nothing

Then open http://127.0.0.1:8000. Binds to localhost only: the API keys stay on this machine.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cs_agent.agent import PRODUCT_JUDGES
from cs_agent.config import DATA_DIR, default_needs_human_threshold, default_threshold, load_env
from cs_agent.memory import MemoryStore
from cs_agent.store import Store
from web.turns import SessionRegistry, TurnRejected

STATIC_DIR = Path(__file__).resolve().parent / "static"
JUDGES = PRODUCT_JUDGES
MIN_THRESHOLD, MAX_THRESHOLD = 0.3, 0.8
STALL_DAYS = 3


class NewSession(BaseModel):
    customer_id: str
    judge: str = "jev"
    threshold: float = 0.5


class Message(BaseModel):
    text: str


class Confirm(BaseModel):
    request_id: str
    approve: bool


def available_judges(mock: bool) -> dict:
    if mock:
        return {name: True for name in JUDGES}
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("MODEL_ID"))
    has_jev = bool(os.getenv("TYPESAFE_API_KEY"))
    return {"jev": has_llm and has_jev, "llm": has_llm, "none": has_llm}


def order_note(store: Store, order: dict) -> str:
    if order["status"] == "已签收":
        note = f"已签收 · 第 {store.days_since_delivery(order)} 天"
    elif order["status"] == "已发货":
        idle = store.idle_days(order["customer_id"], order["order_id"]) or 0
        note = f"物流 {idle} 天未更新" if idle >= STALL_DAYS else "运输中"
    else:
        note = order["status"]
    if order["item"].get("custom"):
        note += " · 定制商品"
    return note


def customers_payload() -> List[dict]:
    store = Store()
    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryStore(Path(tmp), DATA_DIR / "memory_seed")
        return [{
            "id": cid, "name": c["name"], "level": c["level"],
            "memory": memory.render(cid),
            "orders": [{"order_id": o["order_id"], "item": o["item"]["name"], "size": o["item"].get("size"),
                        "amount": o["amount"], "status": o["status"], "note": order_note(store, o)}
                       for o in sorted(store.list_orders(cid), key=lambda o: o["order_id"])],
        } for cid, c in store.customers.items()]


def create_app(mock: bool = False) -> FastAPI:
    load_env()
    app = FastAPI(title="售后客服小安 · 演示")
    registry = SessionRegistry(mock=mock)
    customer_ids = set(Store().customers)

    @app.get("/api/config")
    def config():
        return {"mock": mock, "judges": available_judges(mock), "default_threshold": default_threshold(),
                "needs_human_threshold": default_needs_human_threshold()}

    @app.get("/api/health")
    def health():
        return {"judges": registry.health()}

    @app.get("/api/customers")
    def customers():
        return customers_payload()

    @app.post("/api/sessions")
    def create_session(body: NewSession):
        if body.customer_id not in customer_ids:
            raise HTTPException(400, f"没有顾客 {body.customer_id}")
        if body.judge not in JUDGES:
            raise HTTPException(400, f"判断层只能是 {'/'.join(JUDGES)}")
        if not available_judges(mock)[body.judge]:
            raise HTTPException(400, "这个判断层缺少 API key：请在 .env 里配置，或用 --mock 启动")
        if not MIN_THRESHOLD - 1e-6 <= body.threshold <= MAX_THRESHOLD + 1e-6:
            raise HTTPException(400, f"阈值要在 {MIN_THRESHOLD} 到 {MAX_THRESHOLD} 之间")
        session = registry.create(body.customer_id, body.judge, round(body.threshold, 2))
        return {"session_id": session.id}

    @app.post("/api/sessions/{session_id}/messages")
    def send_message(session_id: str, body: Message):
        session = registry.get(session_id)
        if session is None:
            raise HTTPException(404, "会话不存在，请开新会话")
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "消息不能为空")
        try:
            events = session.start_turn(text)
        except TurnRejected as e:
            raise HTTPException(409, str(e))

        def stream():
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/sessions/{session_id}/confirm", status_code=204)
    def confirm(session_id: str, body: Confirm):
        session = registry.get(session_id)
        if session is None or not session.resolve(body.request_id, body.approve):
            raise HTTPException(404, "没有这条待确认的请求（可能已超时）")
        return Response(status_code=204)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="售后客服小安 · 本地演示")
    parser.add_argument("--mock", action="store_true", help="不需要 key 的离线模式（关键词规则，结果无意义）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    app = create_app(mock=args.mock)
    print(f"售后客服小安{'（MOCK 模式）' if args.mock else ''}：打开 http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
