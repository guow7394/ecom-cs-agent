"""The judgment layer: one structured decision per customer turn.

    customer message
        |
        v
    judge(state) -> Judgment{intent, confidence, probabilities, needs_human}  <- Jev / LLM / Mock
        |
        v
    gate(judgment, policy) -> GateDecision{route: agent | human}  <- plain code
        | human: confidence < threshold, or the intent itself needs a person,
        |        or needs_human >= threshold (clear intent, but not a standard case)
        | agent: hand the turn to the LLM agent, with the intent as a hint

Two questions are asked about the same state. Jev evaluates them in parallel
in one call, so the second question adds no latency:

* intent       Choice over INTENTS -> which request is this, and how sure?
* needs_human  Noul -> does handling it need a person's judgment? Intent
               confidence alone misses "clear intent, hard case" turns,
               e.g. a confident "return" for a washed, faded item.

Three interchangeable backends make the experiment possible:

* JevJudge  TypeSafe Jev, a discriminative "System One" model. One Choice
            question returns calibrated probabilities over our intents and
            confidence = (p_max - 1/n) / (1 - 1/n).
* LLMJudge  The same question asked to DeepSeek, which reports its own
            confidence through a forced tool call (the "verbalised" baseline).
* MockJudge Keyword scores -> softmax -> the same confidence formula. Offline
            only; its numbers mean nothing.

The decision to hand off is made by code (gate), never by the model, so
the threshold is an explicit, tunable risk dial.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import httpx

from .llm import block_get, text_of

# Order matters for Jev (it can lean toward the first option); keep it stable
# so thresholds tuned on one run transfer to the next.
INTENTS: Dict[str, str] = {
    "order_status": "Asks about the order itself: which orders they have, its status or amount, "
                    "whether it has shipped yet. 查订单 / 订单状态 / 发货了吗 / 买了什么",
    "logistics_query": "Asks about a parcel that is already shipped: where it is, when it arrives, "
                       "tracking not updating. 物流 / 快递到哪了 / 什么时候到 / 好几天不动",
    "return_refund": "Wants to return an item, get money back, or cancel an order. "
                     "退货 / 退款 / 不想要了 / 取消订单",
    "exchange": "Wants to swap the item for another size or color, or a replacement. "
                "换货 / 换尺码 / 换大一码 / 换颜色",
    "address_change": "Wants to change the delivery address of an order. 改地址 / 地址填错了",
    "policy_question": "Asks a general question about after-sales rules without asking to act on "
                       "an order: return window, shipping insurance, warranty, custom items, invoices. "
                       "运费险 / 保修多久 / 定制的能退吗 / 规则",
    "request_human": "Explicitly asks to talk to a human agent or a manager. "
                     "转人工 / 人工客服 / 找经理 / 不想跟机器人说",
    "complaint_escalation": "Escalating complaint: demands compensation, reports injury or allergy, "
                            "threatens 12315, lawyers or media exposure, accuses the shop of cheating. "
                            "投诉 / 赔偿 / 过敏 / 12315 / 律师 / 曝光 / 骗子",
    "out_of_scope": "Not an after-sales request: chit-chat, weather, pre-sale product recommendations. "
                    "闲聊 / 推荐商品 / 天气",
}
INTENT_ZH = {
    "order_status": "查订单", "logistics_query": "查物流", "return_refund": "退货退款",
    "exchange": "换货", "address_change": "改地址", "policy_question": "政策咨询",
    "request_human": "要求人工", "complaint_escalation": "投诉升级", "out_of_scope": "非售后",
}
HUMAN_INTENTS = frozenset({"request_human", "complaint_escalation"})
SENSITIVE_INTENTS = frozenset({"return_refund", "exchange", "address_change"})

JEV_INSTRUCTIONS = ("What is the customer's primary after-sales intent in `latest_customer_message`? "
                    "Use `recent_conversation` only as context. Messages are in Chinese.")
# Derived from the clauses in data/policies.md that reserve a case for human staff.
NEEDS_HUMAN_INSTRUCTIONS = ("Does handling `latest_customer_message` need a human agent's judgment, "
                            "rather than one standard after-sales procedure? "
                            "Use `recent_conversation` only as context. Messages are in Chinese.")
NEEDS_HUMAN_CRITERIA = {
    "true": ("Needs a human: the message mixes several different requests, or the customer has not "
             "decided what they want and leaves it to the shop; or it is a case the shop rules reserve "
             "for human review: quality problems past the 15-day return window (warranty inspection), "
             "damage that needs photo review, promises made outside the standard rules, refunds that "
             "never arrived, invoice changes, or compensation."),
    "false": ("One clear, standard request the bot can handle with the rules: check an order or a "
              "parcel, return or exchange an item, change an address, or ask about a policy. "
              "Stays false even when the rules may end up refusing the request."),
}
STATE_TURNS = 6


class JudgeError(RuntimeError):
    pass


@dataclass
class Judgment:
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    backend: str
    latency_ms: float = 0.0
    reason: str = ""
    needs_human: Optional[float] = None   # P(needs a person's judgment), from the second question
    sources: Optional[dict] = None        # per-leg scores (three-way fusion backend only)
    degraded: Optional[str] = None        # set when a backup judge answered because the primary failed
    raw: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw")
        return d


def choice_confidence(probabilities: Union[Mapping[str, float], Sequence[float]]) -> float:
    values = list(probabilities.values()) if isinstance(probabilities, Mapping) else list(probabilities)
    n = len(values)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (max(values) - 1 / n) / (1 - 1 / n)))


def build_state(transcript: List[Tuple[str, str]], latest: str) -> dict:
    """Only what the intent decision needs: Jev gets less accurate as state grows."""
    return {
        "recent_conversation": [{"from": role, "text": text}
                                for role, text in transcript[-STATE_TURNS:]],
        "latest_customer_message": latest,
    }


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------

RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class JevJudge:
    BASE_URL = "https://api.typesafe.ai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 30.0, max_retries: int = 4,
                 transport: Optional[httpx.BaseTransport] = None,
                 sleep: Callable[[float], None] = time.sleep):
        key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("TYPESAFE_API_KEY 未设置，无法使用 Jev 判断层")
        self.model = model or os.getenv("JEV_MODEL", "jev-latest")
        self.max_retries = max_retries
        self.sleep = sleep
        self.client = httpx.Client(base_url=base_url or os.getenv("TYPESAFE_BASE_URL", self.BASE_URL),
                                   headers={"Authorization": f"Bearer {key}"},
                                   timeout=timeout, transport=transport)

    def request_body(self, state: dict) -> dict:
        return {"model": self.model, "state": state,
                "questions": {"intent": {"type": "choice", "instructions": JEV_INSTRUCTIONS,
                                         "criteria": dict(INTENTS)},
                              "needs_human": {"type": "noul", "instructions": NEEDS_HUMAN_INSTRUCTIONS,
                                              "criteria": dict(NEEDS_HUMAN_CRITERIA)}}}

    def _post(self, body: dict) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post("/v1/systemone", json=body)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise JudgeError(f"Jev 网络错误：{e}") from e
                self.sleep(0.5 * 2 ** attempt)
                continue
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = response.headers.get("retry-after")
                self.sleep(float(retry_after) if retry_after else 0.5 * 2 ** attempt)
                continue
            return response
        raise AssertionError("unreachable")

    def judge(self, state: dict) -> Judgment:
        started = time.perf_counter()
        response = self._post(self.request_body(state))
        if response.status_code >= 400:
            raise JudgeError(f"Jev HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            answer = data["answers"]["intent"]
            intent = answer["choice"]
            probabilities = {k: float(v) for k, v in answer["probabilities"].items()}
            confidence = float(answer["confidence"])
            needs_human = float(data["answers"]["needs_human"]["noul"])
        except (KeyError, TypeError, ValueError) as e:
            raise JudgeError(f"Jev 响应格式异常：{e!r}") from e
        if intent not in INTENTS:
            raise JudgeError(f"Jev 返回未知意图：{intent}")
        return Judgment(intent, confidence, probabilities, f"jev:{data.get('model', self.model)}",
                        (time.perf_counter() - started) * 1000, needs_human=needs_human, raw=data)


# ---------------------------------------------------------------------------
# LLM self-reported confidence
# ---------------------------------------------------------------------------

LLM_JUDGE_SYSTEM = (
    "你是电商售后客服系统的意图判断模块。输入是 JSON：latest_customer_message 是要判断的消息，"
    "recent_conversation 仅作上下文。从下列意图中选出最主要的一个，并给出你对该判断的置信度"
    "（0~1：1 表示完全确定，0.5 左右表示拿不准，越不确定越低）。必须调用 report_judgment。\n意图：\n"
    + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    + "\n另外给出 needs_human：处理这条消息需要人工判断的概率（0~1）。判断标准：\n"
    + f"- 接近 1：{NEEDS_HUMAN_CRITERIA['true']}\n- 接近 0：{NEEDS_HUMAN_CRITERIA['false']}"
)
REPORT_TOOL = {
    "name": "report_judgment",
    "description": "报告意图判断结果",
    "input_schema": {"type": "object", "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_human": {"type": "number", "minimum": 0, "maximum": 1,
                        "description": "处理这条消息需要人工判断的概率"},
        "reason": {"type": "string", "description": "一句话理由"},
    }, "required": ["intent", "confidence", "needs_human", "reason"]},
}
_JSON_RE = re.compile(r"\{.*\}", re.S)
# deepseek-v4 thinks before calling report_judgment; 400 tokens cut off ambiguous turns.
LLM_JUDGE_MAX_TOKENS = 4096


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    def judge(self, state: dict) -> Judgment:
        started = time.perf_counter()
        request = dict(system=LLM_JUDGE_SYSTEM, tools=[REPORT_TOOL], max_tokens=LLM_JUDGE_MAX_TOKENS,
                       messages=[{"role": "user", "content": json.dumps(state, ensure_ascii=False)}])
        try:
            response = self.llm.create(tool_choice={"type": "tool", "name": "report_judgment"}, **request)
        except Exception:
            # Some compatible endpoints reject a forced tool (e.g. with thinking on);
            # the system prompt still tells the model to call report_judgment.
            response = self.llm.create(**request)
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise JudgeError(f"LLM 判断层输出被 max_tokens={LLM_JUDGE_MAX_TOKENS} 截断（思考过长）")
        data = next((block_get(b, "input") for b in response.content
                     if block_get(b, "type") == "tool_use"
                     and block_get(b, "name") == "report_judgment"), None)
        if data is None:  # some compatible endpoints ignore tool_choice; accept JSON in text
            match = _JSON_RE.search(text_of(response.content))
            try:
                data = json.loads(match.group(0)) if match else None
            except json.JSONDecodeError:
                data = None
        if not isinstance(data, dict) or data.get("intent") not in INTENTS:
            raise JudgeError(f"LLM 判断层输出无法解析：{data!r}")
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence"))))
            needs_human = max(0.0, min(1.0, float(data.get("needs_human"))))
        except (TypeError, ValueError) as e:
            raise JudgeError(f"LLM 判断层数值无效：{data!r}") from e
        return Judgment(data["intent"], confidence, {data["intent"]: confidence}, "llm",
                        (time.perf_counter() - started) * 1000, reason=str(data.get("reason", "")),
                        needs_human=needs_human)


# ---------------------------------------------------------------------------
# Mock (offline)
# ---------------------------------------------------------------------------

_LEXICON = {
    "order_status": ["订单", "买了", "买的", "状态", "发货了吗"],
    "logistics_query": ["物流", "快递", "到哪", "什么时候到", "不动", "派送"],
    "return_refund": ["退", "退款", "不想要", "取消"],
    "exchange": ["换", "大一码", "小一号", "尺码"],
    "address_change": ["地址", "改到", "改成"],
    "policy_question": ["运费险", "保修", "政策", "规则", "多久", "能退吗", "发票", "补差价"],
    "request_human": ["人工", "经理", "机器人"],
    "complaint_escalation": ["投诉", "12315", "律师", "曝光", "过敏", "赔", "骗子", "说法"],
    "out_of_scope": ["天气", "推荐", "你好"],
}
_NEEDS_HUMAN_WORDS = ["还有", "可能", "看着办", "怎么办", "补偿", "照片", "保价", "发票", "维修", "先别"]
MOCK_SHARPNESS = 2.0
MOCK_CONTEXT_WEIGHT = 0.3


class MockJudge:
    def judge(self, state: dict) -> Judgment:
        latest = state.get("latest_customer_message", "")
        context = " ".join(t["text"] for t in state.get("recent_conversation", [])
                           if t.get("from") == "customer")
        scores = {intent: sum(latest.count(w) for w in words)
                  + MOCK_CONTEXT_WEIGHT * sum(context.count(w) for w in words)
                  for intent, words in _LEXICON.items()}
        exps = {k: math.exp(MOCK_SHARPNESS * v) for k, v in scores.items()}
        total = sum(exps.values())
        probabilities = {k: exps[k] / total for k in INTENTS}
        intent = max(probabilities, key=probabilities.get)
        needs_human = min(0.95, 0.1 + 0.25 * sum(latest.count(w) for w in _NEEDS_HUMAN_WORDS))
        return Judgment(intent, choice_confidence(probabilities), probabilities, "mock",
                        needs_human=needs_human)


# ---------------------------------------------------------------------------
# Gate: plain code decides, using the judgment
# ---------------------------------------------------------------------------

@dataclass
class GatePolicy:
    threshold: float = 0.5
    sensitive_margin: float = 0.1   # "thresholds scale with risk" (Jev docs)
    max_threshold: float = 0.95
    needs_human_threshold: Optional[float] = 0.5   # None = intent-only gate (the v1 ablation)

    def effective_threshold(self, intent: str) -> float:
        if intent in SENSITIVE_INTENTS:
            return min(self.threshold + self.sensitive_margin, self.max_threshold)
        return self.threshold


@dataclass
class GateDecision:
    route: str       # "agent" | "human"
    reason: str      # "" | "low_confidence" | "human_intent" | "needs_judgment" | "judge_error"
    threshold: float


def gate(judgment: Optional[Judgment], policy: GatePolicy) -> GateDecision:
    if judgment is None:  # judge failed: fail safe
        return GateDecision("human", "judge_error", policy.threshold)
    threshold = policy.effective_threshold(judgment.intent)
    if judgment.confidence < threshold:
        return GateDecision("human", "low_confidence", threshold)
    if judgment.intent in HUMAN_INTENTS:
        return GateDecision("human", "human_intent", threshold)
    if (policy.needs_human_threshold is not None and judgment.needs_human is not None
            and judgment.needs_human >= policy.needs_human_threshold):
        return GateDecision("human", "needs_judgment", threshold)
    return GateDecision("agent", "", threshold)
