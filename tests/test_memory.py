from cs_agent.config import DATA_DIR
from cs_agent.memory import MemoryStore


def test_seed_is_copied_on_first_load(tmp_path):
    mem = MemoryStore(tmp_path, DATA_DIR / "memory_seed")
    data = mem.load("C001")
    assert data["preferences"]["鞋码"] == "37"
    assert (tmp_path / "C001.json").exists()


def test_unknown_customer_starts_empty(tmp_path):
    mem = MemoryStore(tmp_path, DATA_DIR / "memory_seed")
    assert mem.load("C006") == {"preferences": {}, "after_sales": []}
    assert mem.render("C006") == ""


def test_preferences_overwrite_and_persist(tmp_path):
    mem = MemoryStore(tmp_path, DATA_DIR / "memory_seed")
    mem.remember_preference("C001", "鞋码", "38")
    again = MemoryStore(tmp_path, DATA_DIR / "memory_seed")
    assert again.load("C001")["preferences"]["鞋码"] == "38"


def test_after_sales_history_appends(tmp_path):
    mem = MemoryStore(tmp_path, None)
    mem.add_after_sales("C006", {"date": "2026-10-13", "order_id": "A6002", "type": "退款", "detail": "x"})
    mem.add_after_sales("C006", {"date": "2026-10-13", "order_id": "A6001", "type": "换货", "detail": "y"})
    assert [r["order_id"] for r in mem.load("C006")["after_sales"]] == ["A6002", "A6001"]


def test_render_mentions_preferences_and_history(tmp_path):
    text = MemoryStore(tmp_path, DATA_DIR / "memory_seed").render("C001")
    assert "鞋码: 37" in text and "A0907" in text
