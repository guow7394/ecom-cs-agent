"""s09 memory, narrowed to what a support desk needs to remember about a
returning customer: stable preferences (sizes) and past after-sales cases.

One JSON file per customer. Preferences are written by the agent through the
remember_preference tool; after-sales records are written by code whenever a
refund / exchange / address change succeeds, so history can't be invented.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Optional

RECENT_CASES = 5


class MemoryStore:
    def __init__(self, root: Path, seed_dir: Optional[Path] = None):
        self.root = Path(root)
        self.seed_dir = seed_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, customer_id: str) -> Path:
        return self.root / f"{customer_id}.json"

    def load(self, customer_id: str) -> dict:
        path = self._path(customer_id)
        if not path.exists() and self.seed_dir and (self.seed_dir / path.name).exists():
            shutil.copy(self.seed_dir / path.name, path)
        if not path.exists():
            return {"preferences": {}, "after_sales": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("preferences", {})
        data.setdefault("after_sales", [])
        return data

    def _save(self, customer_id: str, data: dict) -> None:
        self._path(customer_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember_preference(self, customer_id: str, category: str, value: str) -> str:
        with self._lock:
            data = self.load(customer_id)
            data["preferences"][category.strip()] = value.strip()
            self._save(customer_id, data)
        return f"已记住：{category} = {value}"

    def add_after_sales(self, customer_id: str, record: dict) -> None:
        with self._lock:
            data = self.load(customer_id)
            data["after_sales"].append(record)
            self._save(customer_id, data)

    def render(self, customer_id: str) -> str:
        data = self.load(customer_id)
        lines = []
        if data["preferences"]:
            lines.append("偏好：" + "；".join(f"{k}: {v}" for k, v in data["preferences"].items()))
        for r in data["after_sales"][-RECENT_CASES:]:
            lines.append(f"历史售后：{r['date']} {r['order_id']} {r['type']} — {r['detail']}")
        return "\n".join(lines)
