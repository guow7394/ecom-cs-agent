"""s03 permission, translated to customer service.

Three gates run before any state-changing tool, in the same order as
learn-claude-code s03:

    Gate 1  hard rules      big refunds are never automated
    Gate 2  business rules  refund window, custom items, shipped orders ...
    Gate 3  confirmation    the *customer* must say yes (y/N)

Gates 1 and 2 fail closed without bothering the customer; only an action
that is actually allowed is put to them for confirmation.
"""
from __future__ import annotations

from typing import Callable, Tuple

from .store import Store

SENSITIVE_TOOLS = ("apply_refund", "create_exchange", "change_address")
MAX_AUTO_REFUND = 2000

ConfirmFn = Callable[[str], bool]


class PermissionGate:
    def __init__(self, store: Store, confirm_fn: ConfirmFn):
        self.store = store
        self.confirm_fn = confirm_fn

    def check(self, customer_id: str, tool: str, args: dict) -> Tuple[bool, str]:
        if tool not in SENSITIVE_TOOLS:
            return True, ""
        order = self.store.get_order(customer_id, args.get("order_id", ""))

        # Gate 1: hard rules
        if tool == "apply_refund" and order and order["amount"] > MAX_AUTO_REFUND:
            return False, (f"[硬规则] 退款金额 ¥{order['amount']} 超过 ¥{MAX_AUTO_REFUND}，"
                           "机器人不能办理，需转人工审核")

        # Gate 2: business rules
        ok, why = self._business_check(customer_id, tool, args)
        if not ok:
            return False, f"[业务规则] {why}"

        # Gate 3: customer confirmation
        if not self.confirm_fn(self.describe(tool, order, args)):
            return False, "[顾客确认] 顾客未确认，操作已取消"
        return True, why

    def _business_check(self, customer_id: str, tool: str, args: dict) -> Tuple[bool, str]:
        oid = args.get("order_id", "")
        if tool == "apply_refund":
            return self.store.check_refund(customer_id, oid, args.get("reason_type", "no_reason"))
        if tool == "create_exchange":
            return self.store.check_exchange(customer_id, oid, args.get("new_size", ""))
        return self.store.check_address_change(customer_id, oid)

    @staticmethod
    def describe(tool: str, order: dict, args: dict) -> str:
        name = order["item"]["name"]
        if tool == "apply_refund":
            return f"为订单 {order['order_id']}（{name}）申请退款 ¥{order['amount']}，原路退回"
        if tool == "create_exchange":
            return (f"订单 {order['order_id']}（{name}）由 {order['item'].get('size')} 码"
                    f"换成 {args.get('new_size')} 码")
        return f"订单 {order['order_id']}（{name}）收货地址改为：{args.get('new_address')}"
