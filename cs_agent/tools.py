"""s02 tool use: learn-claude-code's bash/read/write become after-sales tools.

Same shape as upstream: a list of JSON-schema tool definitions plus a
name -> handler dispatch. Handlers return ToolResult(ok, content) so the
evaluation can tell "refund attempted and blocked" from "refund executed".
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Iterable, List

from .permission import SENSITIVE_TOOLS
from .session import Session

STALL_DAYS = 3  # 72h without a tracking update counts as abnormal


@dataclass
class ToolResult:
    ok: bool
    content: str


def _schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required}}


_ORDER_ID = {"type": "string", "description": "订单号，如 A1001"}

TOOL_SCHEMAS = {s["name"]: s for s in [
    _schema("query_order", "查询当前顾客的订单。不传 order_id 时列出该顾客全部订单。",
            {"order_id": _ORDER_ID}, []),
    _schema("query_logistics", "查询订单的物流轨迹，并判断是否物流异常。",
            {"order_id": _ORDER_ID}, ["order_id"]),
    _schema("register_urge", "为物流异常（超过 72 小时没有更新）的包裹登记催件，快递公司 24 小时内反馈。",
            {"order_id": _ORDER_ID}, ["order_id"]),
    _schema("load_skill", "加载一份业务处理规范的全文（名字见系统提示里的规范目录）。",
            {"name": {"type": "string", "description": "规范名称，如 退货退款处理规范"}}, ["name"]),
    _schema("search_policy", "检索售后政策知识库（退货、换货、运费险、保修、定制品、发票等）。",
            {"query": {"type": "string", "description": "要查的问题"}}, ["query"]),
    _schema("apply_refund", "申请退款/退货（敏感操作：系统会校验规则并请顾客确认）。",
            {"order_id": _ORDER_ID,
             "reason_type": {"type": "string", "enum": ["no_reason", "quality"],
                             "description": "no_reason=无理由/不想要；quality=质量问题"},
             "reason": {"type": "string", "description": "顾客描述的原因"}},
            ["order_id", "reason_type"]),
    _schema("create_exchange", "申请换货（换尺码）（敏感操作：系统会校验规则并请顾客确认）。",
            {"order_id": _ORDER_ID,
             "new_size": {"type": "string", "description": "要换成的尺码，如 38、L"},
             "reason": {"type": "string"}},
            ["order_id", "new_size"]),
    _schema("change_address", "修改未发货订单的收货地址（敏感操作：系统会校验规则并请顾客确认）。",
            {"order_id": _ORDER_ID, "new_address": {"type": "string"}},
            ["order_id", "new_address"]),
    _schema("remember_preference", "记住顾客的长期偏好（如鞋码、上衣尺码），下次服务时使用。",
            {"category": {"type": "string", "description": "如 鞋码"},
             "value": {"type": "string", "description": "如 38"}},
            ["category", "value"]),
    _schema("transfer_to_human", "转接人工客服并创建工单。调用后由人工接手。",
            {"reason": {"type": "string", "description": "转人工原因"},
             "summary": {"type": "string", "description": "给人工客服的一句话情况摘要"}},
            ["reason"]),
]}

BASE_TOOL_NAMES = ["query_order", "query_logistics", "register_urge", "search_policy", "load_skill", "change_address",
                   "remember_preference", "transfer_to_human"]
SPECIALIST_TOOL_NAMES = ["query_order", "query_logistics", "search_policy", "load_skill",
                         "apply_refund", "create_exchange"]


def _brief(order: dict) -> str:
    item = order["item"]
    size = f" {item['size']}码" if item.get("size") else ""
    return f"{order['order_id']} {item['name']}{size} ¥{order['amount']} 【{order['status']}】"


class ToolBox:
    """Dispatches tool calls for one session; `by` tags who made the call."""

    def __init__(self, session: Session, by: str = "main"):
        self.session = session
        self.by = by

    @staticmethod
    def schemas(names: Iterable[str]) -> List[dict]:
        return [TOOL_SCHEMAS[n] for n in names]

    def run(self, name: str, args: dict) -> ToolResult:
        self.session.emit("tool_start", name=name, by=self.by, input=args or {})
        handler = getattr(self, f"_{name}", None)
        if name not in TOOL_SCHEMAS or handler is None:
            result = ToolResult(False, f"未知工具：{name}")
        else:
            try:
                result = handler(**(args or {}))
            except TypeError as e:
                result = ToolResult(False, f"参数错误：{e}")
            except Exception as e:  # tool bugs go back to the model, not up the stack
                result = ToolResult(False, f"工具出错：{type(e).__name__}: {e}")
        self.session.tool_log.append({"name": name, "input": args, "ok": result.ok,
                                      "by": self.by, "output": result.content[:300]})
        self.session.emit("tool_end", name=name, by=self.by, ok=result.ok, output=result.content[:300])
        return result

    # -- read-only tools --

    def _query_order(self, order_id: str = "") -> ToolResult:
        s = self.session
        if not order_id:
            orders = s.store.list_orders(s.customer_id)
            return ToolResult(True, "\n".join(_brief(o) for o in orders) or "该顾客没有订单")
        order = s.store.get_order(s.customer_id, order_id)
        if order is None:
            return ToolResult(False, f"未找到订单 {order_id}（只能查询本人订单）")
        info = {k: order[k] for k in ("order_id", "status", "amount", "created_at",
                                      "shipped_at", "delivered_at", "address", "carrier")}
        info["item"] = order["item"]
        days = s.store.days_since_delivery(order)
        note = f"\n（签收 {days} 天）" if days is not None else ""
        return ToolResult(True, json.dumps(info, ensure_ascii=False) + note)

    def _query_logistics(self, order_id: str) -> ToolResult:
        s = self.session
        order = s.store.get_order(s.customer_id, order_id)
        if order is None:
            return ToolResult(False, f"未找到订单 {order_id}（只能查询本人订单）")
        events = s.store.tracking(s.customer_id, order_id)
        if not events:
            return ToolResult(True, f"订单 {order['order_id']} 状态「{order['status']}」，暂无物流信息")
        lines = [f"{order['carrier']} {order['tracking_no']}"]
        lines += [f"{e['time']} {e['desc']}" for e in events]
        idle = s.store.idle_days(s.customer_id, order_id)
        if order["status"] == "已发货" and idle >= STALL_DAYS:
            lines.append(f"⚠ 距最近一次更新已 {idle} 天，属于物流异常，可用 register_urge 为顾客登记催件")
        return ToolResult(True, "\n".join(lines))

    def _register_urge(self, order_id: str) -> ToolResult:
        s = self.session
        order = s.store.get_order(s.customer_id, order_id)
        if order is None:
            return ToolResult(False, f"未找到订单 {order_id}（只能查询本人订单）")
        if order["status"] != "已发货":
            return ToolResult(False, f"订单状态为「{order['status']}」，只有运输中的包裹可以催件")
        idle = s.store.idle_days(s.customer_id, order_id)
        if idle is None or idle < STALL_DAYS:
            return ToolResult(False, f"最近一次物流更新在 {idle or 0} 天前，未超过 72 小时，暂不能催件")
        ticket = f"U{uuid.uuid4().hex[:6].upper()}"
        message = f"已为订单 {order['order_id']} 登记催件（工单 {ticket}），快递公司会在 24 小时内反馈"
        s.memory.add_after_sales(s.customer_id, {"date": s.store.today.isoformat(),
                                                 "order_id": order["order_id"], "type": "催件",
                                                 "detail": message})
        return ToolResult(True, message)

    def _search_policy(self, query: str) -> ToolResult:
        searcher = self.session.policy_searcher
        if searcher is None:
            hits, header = self.session.policies.search(query, k=3), ""
        else:
            hits, meta = searcher.search(query, k=3)
            header = (f"【检索】{len(meta['queries'])} 条查询 · 召回 {meta['recalled']} 段"
                      f"{' · LLM 重排' if meta['reranked'] else ''}\n\n")
        if not hits:
            return ToolResult(True, header + "未检索到相关政策")
        return ToolResult(True, header + "\n\n".join(h.render() for h in hits))

    def _load_skill(self, name: str) -> ToolResult:
        skills = self.session.skills
        skill = skills.get(name.strip()) if skills else None
        if skill is None:
            names = "、".join(s.name for s in skills.skills) if skills else "（无）"
            return ToolResult(False, f"没有规范「{name}」，可用的有：{names}")
        return ToolResult(True, f"# {skill.name}\n{skill.body}")

    def _remember_preference(self, category: str, value: str) -> ToolResult:
        s = self.session
        return ToolResult(True, s.memory.remember_preference(s.customer_id, category, value))

    def _transfer_to_human(self, reason: str, summary: str = "") -> ToolResult:
        ticket = f"H{uuid.uuid4().hex[:6].upper()}"
        self.session.handoff = {"by": "agent", "ticket_id": ticket, "reason": reason,
                                "summary": summary}
        return ToolResult(True, f"已创建人工工单 {ticket}。请告诉顾客人工客服会尽快接入。")

    # -- sensitive tools: permission gate first, then execute, then remember --

    def _guarded(self, tool: str, args: dict, execute, record_type: str) -> ToolResult:
        s = self.session
        assert tool in SENSITIVE_TOOLS
        allowed, why = s.permission.check(s.customer_id, tool, args)
        if not allowed:
            return ToolResult(False, why)
        message = execute()
        s.memory.add_after_sales(s.customer_id, {
            "date": s.store.today.isoformat(),
            "order_id": args["order_id"].upper(), "type": record_type, "detail": message})
        return ToolResult(True, message)

    def _apply_refund(self, order_id: str, reason_type: str = "no_reason", reason: str = "") -> ToolResult:
        args = {"order_id": order_id, "reason_type": reason_type}
        cid = self.session.customer_id
        return self._guarded("apply_refund", args,
                             lambda: self.session.store.do_refund(cid, order_id, reason), "退款")

    def _create_exchange(self, order_id: str, new_size: str, reason: str = "") -> ToolResult:
        args = {"order_id": order_id, "new_size": new_size}
        cid = self.session.customer_id
        return self._guarded("create_exchange", args,
                             lambda: self.session.store.do_exchange(cid, order_id, new_size), "换货")

    def _change_address(self, order_id: str, new_address: str) -> ToolResult:
        args = {"order_id": order_id, "new_address": new_address}
        cid = self.session.customer_id
        return self._guarded("change_address", args,
                             lambda: self.session.store.do_address_change(cid, order_id, new_address),
                             "改地址")
