"""The generative model behind the agent.

AnthropicLLM talks to any Anthropic-compatible endpoint (DeepSeek here),
exactly like learn-claude-code: base URL + MODEL_ID from .env.
MockLLM is a keyword-driven stand-in so the whole pipeline and the tests
run without an API key. Its answers are NOT meant to be good.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: List[Any]
    stop_reason: str = "end_turn"


def block_get(block: Any, key: str, default=None):
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(block_get(b, "text", "") or "" for b in content
                   if block_get(b, "type") == "text").strip()


class AnthropicLLM:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 8192, **client_kwargs):
        # Thinking-mode models (deepseek-v4 by default) spend output tokens on reasoning
        # before answering; a tight budget truncates exactly the hard turns.
        from anthropic import Anthropic
        client_kwargs.setdefault("max_retries", 3)
        client_kwargs.setdefault("timeout", 90)
        self.client = Anthropic(base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"), **client_kwargs)
        self.model = model or os.environ["MODEL_ID"]
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=None):
        kwargs = dict(model=self.model, system=system, messages=messages,
                      max_tokens=max_tokens or self.max_tokens, temperature=self.temperature)
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

_ORDER_RE = re.compile(r"[A-Z]\d{4}")
_SIZE_RE = re.compile(r"(\d{2}|XS|S|M|L|XL)\s*码")
_ADDRESS_RE = re.compile(r"(?:改成|改到|改为)[:：]?\s*(.+)$")

# (keywords, preferred tools in order); first rule whose keywords match wins.
_RULES = [
    ("人工|经理|投诉|12315|律师|曝光|过敏|赔偿|说法", ["transfer_to_human"]),
    ("地址", ["change_address"]),
    ("换", ["after_sales_specialist", "create_exchange"]),
    ("退|不想要|不要了", ["after_sales_specialist", "apply_refund"]),
    ("物流|快递|到哪|不动|发货", ["query_logistics"]),
    ("运费|保修|定制|政策|发票|多久|规则|补差价", ["search_policy"]),
    ("订单|买了|买的", ["query_order"]),
]
_NEEDS_ORDER = {"query_logistics", "change_address", "apply_refund", "create_exchange"}


def _all_user_text(messages) -> str:
    return "\n".join(m["content"] for m in messages
                     if m["role"] == "user" and isinstance(m["content"], str))


class MockLLM:
    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=None):
        names = [t["name"] for t in (tools or [])]
        if "report_judgment" in names:
            return self._judge(messages)
        if "report_quality" in names:
            return LLMResponse([Block("tool_use", id="q1", name="report_quality",
                                      input={"relevance": 0.8, "accuracy": 0.8, "completeness": 0.7,
                                             "helpfulness": 0.7, "comment": "mock"})], "tool_use")
        if not tools:  # summariser
            return LLMResponse([Block("text", text="（模拟摘要）" + text_of(messages[-1]["content"])[-200:])])
        last = messages[-1]["content"]
        if isinstance(last, list):  # tool results came back -> answer with them
            result = str(block_get(last[0], "content", ""))
            return LLMResponse([Block("text", text=f"（模拟回复）{result[:160]}")])
        return self._act(last, _all_user_text(messages), names)

    def _judge(self, messages):
        from .judge import MockJudge
        state = json.loads(messages[-1]["content"])
        j = MockJudge().judge(state)
        verbal = round(min(1.0, j.confidence + 0.2), 2)  # LLMs tend to over-report
        return LLMResponse([Block("tool_use", id="j1", name="report_judgment",
                                  input={"intent": j.intent, "confidence": verbal,
                                         "needs_human": j.needs_human, "reason": "mock"})], "tool_use")

    def _act(self, text: str, context: str, names: List[str]):
        for keywords, preferred in _RULES:
            if not re.search(keywords, text):
                continue
            tool = next((t for t in preferred if t in names), None)
            if tool is None:
                continue
            args = self._args(tool, text, context)
            if args is None:  # no order id anywhere -> look the orders up first
                tool, args = "query_order", {}
            return LLMResponse([Block("tool_use", id=f"m{uuid.uuid4().hex[:8]}",
                                      name=tool, input=args)], "tool_use")
        return LLMResponse([Block("text", text="（模拟回复）您好，我是售后客服，请问订单有什么问题？")])

    @staticmethod
    def _args(tool: str, text: str, context: str):
        found = _ORDER_RE.findall(text.upper()) or _ORDER_RE.findall(context.upper())
        if tool in _NEEDS_ORDER and not found:
            return None
        oid = found[-1] if found else ""
        if tool == "transfer_to_human":
            return {"reason": "顾客诉求需人工处理", "summary": text[:60]}
        if tool == "after_sales_specialist":
            return {"task": f"顾客说：{text}"}
        if tool == "search_policy":
            return {"query": text}
        if tool == "query_order":
            return {"order_id": oid} if oid else {}
        if tool == "query_logistics":
            return {"order_id": oid}
        if tool == "change_address":
            m = _ADDRESS_RE.search(text)
            return {"order_id": oid, "new_address": m.group(1).strip() if m else "（未提供）"}
        if tool == "apply_refund":
            quality = re.search("质量|坏|破损|没声音|划痕", text)
            return {"order_id": oid, "reason_type": "quality" if quality else "no_reason",
                    "reason": text[:40]}
        m = _SIZE_RE.search(text)
        return {"order_id": oid, "new_size": m.group(1) if m else "M"}
