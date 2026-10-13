"""Everything one customer conversation owns. Nothing here is global, so
many sessions can run side by side (the evaluation does exactly that)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from .config import DATA_DIR
from .memory import MemoryStore
from .permission import ConfirmFn, PermissionGate
from .rag import PolicyIndex
from .store import Store


@lru_cache(maxsize=1)
def policy_index() -> PolicyIndex:
    return PolicyIndex.from_file(DATA_DIR / "policies.md")


@dataclass
class Session:
    customer_id: str
    store: Store
    memory: MemoryStore
    permission: PermissionGate
    policies: PolicyIndex
    tool_log: List[dict] = field(default_factory=list)
    handoff: Optional[dict] = None


def new_session(customer_id: str, memory_root: Path, confirm_fn: ConfirmFn,
                today: Optional[date] = None) -> Session:
    store = Store(today=today)
    if customer_id not in store.customers:
        raise ValueError(f"unknown customer {customer_id}")
    return Session(
        customer_id=customer_id,
        store=store,
        memory=MemoryStore(memory_root, DATA_DIR / "memory_seed"),
        permission=PermissionGate(store, confirm_fn),
        policies=policy_index(),
    )
