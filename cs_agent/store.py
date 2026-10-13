"""Mock order / logistics / stock backend plus the after-sales business rules.

The rules live in code on purpose: the LLM can ask for a refund, but it
cannot talk its way past "7 days" or "custom items are final sale".
Every Store loads its own copy of the JSON files, so sessions never leak
state into each other (important for the parallel evaluation).
"""
from __future__ import annotations

import copy
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

from .config import DATA_DIR, today as config_today

NO_REASON_DAYS = 7
QUALITY_DAYS = 15
EXCHANGE_DAYS = 15

Check = Tuple[bool, str]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, data_dir: Path = DATA_DIR, today: Optional[date] = None):
        self.today = today or config_today()
        self.orders = {o["order_id"]: o for o in _load(data_dir / "orders.json")}
        self.logistics = _load(data_dir / "logistics.json")
        self.stock = _load(data_dir / "stock.json")
        self.customers = _load(data_dir / "customers.json")

    # -- queries --

    def get_order(self, customer_id: str, order_id: str) -> Optional[dict]:
        order = self.orders.get((order_id or "").strip().upper())
        if order is None or order["customer_id"] != customer_id:
            return None
        return order

    def list_orders(self, customer_id: str) -> list:
        mine = [o for o in self.orders.values() if o["customer_id"] == customer_id]
        return sorted(mine, key=lambda o: o["created_at"], reverse=True)

    def days_since_delivery(self, order: dict) -> Optional[int]:
        if not order.get("delivered_at"):
            return None
        return (self.today - date.fromisoformat(order["delivered_at"])).days

    def tracking(self, customer_id: str, order_id: str) -> list:
        order = self.get_order(customer_id, order_id)
        if not order or not order.get("tracking_no"):
            return []
        return copy.deepcopy(self.logistics.get(order["tracking_no"], []))

    def idle_days(self, customer_id: str, order_id: str) -> Optional[int]:
        """Days since the parcel's last tracking update (None when there is no tracking)."""
        events = self.tracking(customer_id, order_id)
        if not events:
            return None
        last = datetime.strptime(events[-1]["time"], "%Y-%m-%d %H:%M").date()
        return (self.today - last).days

    # -- refund --

    def check_refund(self, customer_id: str, order_id: str, reason_type: str) -> Check:
        order = self.get_order(customer_id, order_id)
        if order is None:
            return False, f"未找到订单 {order_id}（只能处理本人订单）"
        status = order["status"]
        if status == "待发货":
            return True, "订单未发货，可直接取消并全额退款"
        if status == "已发货":
            return False, "包裹正在运输中，暂不能直接退款；可拒收（拒收后自动退款）或签收后再申请退货"
        if status != "已签收":
            return False, f"订单当前状态为「{status}」，已在售后处理中，不能重复申请"
        days = self.days_since_delivery(order)
        if reason_type == "quality":
            if days <= QUALITY_DAYS:
                return True, f"签收 {days} 天，质量问题 {QUALITY_DAYS} 天内可退"
            return False, (f"签收已 {days} 天，超过 {QUALITY_DAYS} 天质量问题退换期限，"
                           "需走保修流程（人工登记检测）")
        if order["item"].get("custom"):
            return False, "定制商品不支持七天无理由退货（质量问题除外）"
        if days <= NO_REASON_DAYS:
            return True, f"签收 {days} 天，符合 {NO_REASON_DAYS} 天无理由退货"
        return False, f"签收已 {days} 天，超过 {NO_REASON_DAYS} 天无理由退货期限"

    def do_refund(self, customer_id: str, order_id: str, reason: str) -> str:
        order = self.get_order(customer_id, order_id)
        if order["status"] == "待发货":
            order["status"] = "已取消（已退款）"
            return f"订单 {order['order_id']} 已取消，¥{order['amount']} 将在 1-3 个工作日原路退回"
        order["status"] = "退货中"
        return (f"订单 {order['order_id']} 退货申请已提交，请在 7 天内寄回商品，"
                f"仓库验货后退款 ¥{order['amount']}")

    # -- exchange --

    def check_exchange(self, customer_id: str, order_id: str, new_size: str) -> Check:
        order = self.get_order(customer_id, order_id)
        if order is None:
            return False, f"未找到订单 {order_id}（只能处理本人订单）"
        item = order["item"]
        if order["status"] != "已签收":
            return False, f"订单当前状态为「{order['status']}」，签收后才能换货"
        if item.get("custom"):
            return False, "定制商品不支持换货"
        days = self.days_since_delivery(order)
        if days > EXCHANGE_DAYS:
            return False, f"签收已 {days} 天，超过 {EXCHANGE_DAYS} 天换货期限"
        sizes = self.stock.get(item["sku"])
        if not sizes:
            return False, "该商品没有尺码可换"
        new_size = str(new_size).strip().upper()
        if new_size == str(item.get("size", "")).upper():
            return False, "新尺码和原尺码相同"
        if new_size not in sizes:
            return False, f"该商品没有 {new_size} 码，可选：{'/'.join(sizes)}"
        if sizes[new_size] <= 0:
            return False, f"{new_size} 码暂无库存"
        return True, f"可将 {item.get('size')} 码换为 {new_size} 码（库存 {sizes[new_size]}）"

    def do_exchange(self, customer_id: str, order_id: str, new_size: str) -> str:
        order = self.get_order(customer_id, order_id)
        new_size = str(new_size).strip().upper()
        self.stock[order["item"]["sku"]][new_size] -= 1
        old = order["item"]["size"]
        order["status"] = "换货中"
        order["exchange_to"] = new_size
        return (f"订单 {order['order_id']} 换货已登记：{old} 码 → {new_size} 码，"
                "请寄回原商品，仓库签收后 48 小时内发出新商品")

    # -- address --

    def check_address_change(self, customer_id: str, order_id: str) -> Check:
        order = self.get_order(customer_id, order_id)
        if order is None:
            return False, f"未找到订单 {order_id}（只能处理本人订单）"
        if order["status"] != "待发货":
            return False, f"订单「{order['status']}」，已发货后无法修改地址，可联系快递改派或签收后处理"
        return True, "订单待发货，可以修改地址"

    def do_address_change(self, customer_id: str, order_id: str, new_address: str) -> str:
        order = self.get_order(customer_id, order_id)
        old = order["address"]
        order["address"] = new_address
        return f"订单 {order['order_id']} 收货地址已由「{old}」改为「{new_address}」"
