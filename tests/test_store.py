from datetime import date

import pytest

from cs_agent.store import Store


@pytest.fixture
def store():
    return Store(today=date(2026, 10, 13))


def test_customer_can_only_see_own_orders(store):
    assert store.get_order("C001", "A1001")["item"]["name"] == "轻跑运动鞋"
    assert store.get_order("C002", "A1001") is None
    assert [o["order_id"] for o in store.list_orders("C001")] == ["A1002", "A1001"]


def test_days_since_delivery(store):
    assert store.days_since_delivery(store.get_order("C001", "A1001")) == 4
    assert store.days_since_delivery(store.get_order("C001", "A1002")) is None


def test_refund_within_seven_days_no_reason_ok(store):
    ok, msg = store.check_refund("C001", "A1001", "no_reason")
    assert ok, msg


def test_refund_on_day_seven_ok_day_eight_rejected():
    s = Store(today=date(2026, 10, 16))  # A1001 delivered 10-09 -> day 7
    assert s.check_refund("C001", "A1001", "no_reason")[0]
    s = Store(today=date(2026, 10, 17))
    ok, msg = s.check_refund("C001", "A1001", "no_reason")
    assert not ok and "7" in msg


def test_refund_beyond_window_rejected(store):
    ok, msg = store.check_refund("C002", "A2001", "no_reason")
    assert not ok and "7" in msg


def test_quality_refund_beyond_fifteen_days_rejected(store):
    ok, msg = store.check_refund("C002", "A2001", "quality")
    assert not ok and "保修" in msg


def test_custom_item_no_reason_refund_rejected(store):
    ok, msg = store.check_refund("C003", "A3001", "no_reason")
    assert not ok and "定制" in msg


def test_unshipped_order_refund_ok(store):
    assert store.check_refund("C002", "A2002", "no_reason")[0]


def test_in_transit_refund_rejected(store):
    ok, msg = store.check_refund("C001", "A1002", "no_reason")
    assert not ok and "运输" in msg


def test_unknown_order_rejected(store):
    ok, msg = store.check_refund("C001", "A9999", "no_reason")
    assert not ok and "未找到" in msg


def test_do_refund_changes_status(store):
    store.do_refund("C002", "A2002", "不想要了")
    assert store.get_order("C002", "A2002")["status"] == "已取消（已退款）"
    assert not store.check_refund("C002", "A2002", "no_reason")[0]


def test_exchange_rules(store):
    assert store.check_exchange("C005", "A5001", "31")[0]
    ok, msg = store.check_exchange("C005", "A5001", "32")
    assert not ok and "库存" in msg
    ok, msg = store.check_exchange("C005", "A5001", "30")
    assert not ok
    ok, msg = store.check_exchange("C003", "A3001", "L")
    assert not ok and "定制" in msg
    ok, msg = store.check_exchange("C001", "A1002", "L")
    assert not ok  # not delivered yet


def test_do_exchange_consumes_stock(store):
    before = store.stock["JEANS-STRAIGHT"]["31"]
    store.do_exchange("C005", "A5001", "31")
    assert store.stock["JEANS-STRAIGHT"]["31"] == before - 1
    assert store.get_order("C005", "A5001")["status"] == "换货中"


def test_address_change_only_before_shipping(store):
    assert store.check_address_change("C002", "A2002")[0]
    ok, msg = store.check_address_change("C001", "A1002")
    assert not ok and "已发货" in msg
    store.do_address_change("C002", "A2002", "北京市海淀区中关村大街1号")
    assert store.get_order("C002", "A2002")["address"] == "北京市海淀区中关村大街1号"


def test_tracking_events(store):
    events = store.tracking("C001", "A1002")
    assert events[-1]["desc"].startswith("到达广州")
    assert store.tracking("C002", "A2002") == []


def test_idle_days_since_last_tracking_update(store):
    assert store.idle_days("C003", "A3002") == 4
    assert store.idle_days("C001", "A1002") == 0
    assert store.idle_days("C002", "A2002") is None  # not shipped, no tracking


def test_stores_are_isolated():
    a, b = Store(), Store()
    a.do_address_change("C002", "A2002", "X")
    assert b.get_order("C002", "A2002")["address"] != "X"


# -- fresh orders for the held-out set (C007-C010) --

def test_holdout_orders_follow_the_same_rules(store):
    assert store.check_refund("C009", "B9001", "no_reason")[0]           # delivered exactly 7 days ago
    assert not Store(today=date(2026, 10, 14)).check_refund("C009", "B9001", "no_reason")[0]
    assert store.check_refund("C009", "B9002", "no_reason")[0]           # custom, but not shipped yet
    ok, msg = store.check_refund("C008", "B8001", "no_reason")
    assert not ok and "售后处理中" in msg                                  # return already in progress
    assert not store.check_refund("C007", "B7002", "no_reason")[0]       # 13 days: past no-reason window
    assert store.check_refund("C007", "B7002", "quality")[0]             # ...but within 15-day quality window
    assert store.idle_days("C010", "B0002") == 5                         # stalled parcel
    assert store.idle_days("C008", "B8002") == 0                         # fresh parcel
    assert not store.check_exchange("C009", "B9001", "43")[0]            # out of stock
    assert store.check_exchange("C009", "B9001", "44")[0]
    assert store.get_order("C007", "B0001") is None                      # still can't see others' orders
