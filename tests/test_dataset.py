import re
from collections import Counter

from cs_agent.judge import INTENTS
from cs_agent.store import Store
from cs_agent.tools import TOOL_SCHEMAS
from eval.metrics import DATASET_PATH, load_dataset

CATEGORIES = {"standard", "trap", "trap_human", "explicit_human", "ambiguous"}


def test_dataset_shape():
    cases = load_dataset(DATASET_PATH)
    assert 20 <= len(cases) <= 30
    assert len({c["id"] for c in cases}) == len(cases)
    counts = Counter(c["gold_route"] for c in cases)
    assert counts["agent"] >= 10 and counts["human"] >= 10


def test_labels_are_valid():
    store = Store()
    for c in load_dataset(DATASET_PATH):
        assert c["category"] in CATEGORIES, c["id"]
        assert c["gold_route"] in ("agent", "human"), c["id"]
        assert c["gold_intents"] and set(c["gold_intents"]) <= set(INTENTS), c["id"]
        assert set(c["expect_tools"]) | set(c["forbid_success"]) <= set(TOOL_SCHEMAS), c["id"]
        assert c["customer_id"] in store.customers, c["id"]
        if c["gold_route"] == "human":
            assert not c["expect_tools"], c["id"]


def test_history_alternates_and_ends_with_agent():
    for c in load_dataset(DATASET_PATH):
        roles = [role for role, _ in c["history"]]
        assert roles == ["customer", "agent"] * (len(roles) // 2), c["id"]


def test_referenced_orders_belong_to_the_customer():
    store = Store()
    for c in load_dataset(DATASET_PATH):
        text = c["message"] + " ".join(t for _, t in c["history"])
        for oid in re.findall(r"A\d{4}", text):
            assert store.get_order(c["customer_id"], oid), (c["id"], oid)


def test_trap_labels_match_business_rules():
    store = Store()
    assert not store.check_refund("C002", "A2001", "no_reason")[0]      # T15
    assert not store.check_address_change("C001", "A1002")[0]           # T16
    assert not store.check_refund("C003", "A3001", "no_reason")[0]      # T17
    assert store.get_order("C004", "A4001")["amount"] > 2000            # T18
    assert store.check_exchange("C001", "A1001", "38")[0]               # T03
    assert store.check_exchange("C001", "A1001", "36")[0]               # T12
    assert store.check_refund("C006", "A6002", "no_reason")[0]          # T09 (v2)
    assert store.idle_days("C003", "A3002") >= 3                        # T14 can register_urge


def test_t09_customer_has_no_open_dispute_in_memory():
    # v2: C002's seeded unresolved refund made the agent escalate T09 for an unrelated reason
    from cs_agent.config import DATA_DIR
    case = {c["id"]: c for c in load_dataset(DATASET_PATH)}["T09"]
    assert not (DATA_DIR / "memory_seed" / f"{case['customer_id']}.json").exists()
