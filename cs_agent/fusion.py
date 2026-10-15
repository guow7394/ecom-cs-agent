"""Three-way intent fusion, ported from EchoMind (core/intent_recognizer.py).

    LLM self-report (LLMJudge)        ─┐ x 0.7
    char n-gram embedding vs examples ─┼ x 0.2  → add weighted scores per intent → best intent
    keyword patterns                  ─┘ x 0.1    fused score = its total (a heuristic, NOT a probability)

It plugs in as a fourth judgment backend so the experiment can compare
"hand-weighted fusion" with Jev's calibrated probabilities on the same gate.

Adapted to this project: our 9 intents have no coarse/fine hierarchy, so
EchoMind's group vote reduces to a vote per intent; the LLM leg reuses
LLMJudge so it sees the same intent and needs_human definitions as the
other arms; needs_human comes from that LLM leg (embedding and patterns
cannot answer it). As in EchoMind, a failed LLM leg falls back to the
embedding vote, then the pattern vote.
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .judge import INTENTS, JudgeError, Judgment

WEIGHTS = {"llm": 0.7, "embedding": 0.2, "pattern": 0.1}
EMBED_DIMS = 256

# Example phrases per intent, written for this project and kept apart from the test set.
TEMPLATES: Dict[str, List[str]] = {
    "order_status": ["我的订单现在是什么状态", "订单付款成功了没", "看看我下过哪些单"],
    "logistics_query": ["包裹现在到哪儿了", "物流信息一直没更新", "大概几天能送到"],
    "return_refund": ["申请退货退款", "这个东西我不要了能退吗", "取消订单把钱退给我"],
    "exchange": ["尺码不合适想换", "能不能换个颜色", "换成别的号码"],
    "address_change": ["收货地址填错了", "修改一下收件地址", "改寄到另一个地方"],
    "policy_question": ["七天无理由退货怎么算", "运费险怎么理赔", "数码产品保修多长时间"],
    "request_human": ["转接人工客服", "我想和真人说话", "让你们主管来处理"],
    "complaint_escalation": ["我要投诉", "必须赔偿我的损失", "我要去消费者协会投诉你们"],
    "out_of_scope": ["今天天气怎么样", "给我推荐一款手机", "讲个笑话"],
}
PATTERNS: Dict[str, List[str]] = {
    "request_human": ["转人工", "人工客服", "找人工", "真人", "经理", "主管"],
    "complaint_escalation": ["投诉", "赔偿", "12315", "律师", "曝光", "骗子", "过敏"],
    "logistics_query": ["物流", "快递", "到哪", "派送", "配送", "不动"],
    "order_status": ["订单状态", "发货了吗", "买了哪些", "什么状态"],
    "return_refund": ["退款", "退货", "退了", "不想要"],
    "exchange": ["换货", "换大", "换小", "换成", "尺码", "码数"],
    "address_change": ["地址", "收件人"],
    "policy_question": ["运费险", "保修", "规则", "政策", "能退吗", "多久"],
    "out_of_scope": ["天气", "推荐", "笑话"],
}
assert set(TEMPLATES) == set(PATTERNS) == set(INTENTS)

Vote = Tuple[Optional[str], float]


def local_embedding(text: str, dims: int = EMBED_DIMS) -> List[float]:
    """Signed hash of character 1-3 grams (EchoMind's offline fallback embedding)."""
    normalized = text.lower().strip()
    tokens = {normalized[i:i + n] for n in (1, 2, 3) for i in range(len(normalized) - n + 1)} or {normalized}
    vec = [0.0] * dims
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        vec[int.from_bytes(digest[:4], "big") % dims] += 1.0 if digest[4] % 2 == 0 else -1.0
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = sum(x * x for x in a) ** 0.5, sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


_TEMPLATE_VECTORS = {intent: [local_embedding(t) for t in texts] for intent, texts in TEMPLATES.items()}


def embedding_vote(text: str) -> Vote:
    query = local_embedding(text)
    best, best_score = None, 0.0
    for intent, vectors in _TEMPLATE_VECTORS.items():
        score = max(cosine(query, v) for v in vectors)
        if score > best_score:
            best, best_score = intent, score
    return best, min(1.0, best_score)


def pattern_vote(text: str) -> Vote:
    """One keyword hit gives 0.5; each extra hit in the same intent adds 0.25 (EchoMind's rule)."""
    lowered = text.lower()
    best, best_score = None, 0.0
    for intent, keywords in PATTERNS.items():
        hits = sum(1 for k in keywords if k in lowered)
        score = min(1.0, 0.5 + 0.25 * (hits - 1)) if hits else 0.0
        if score > best_score:
            best, best_score = intent, score
    return best, best_score


def fuse(llm: Optional[Judgment], emb: Vote, pat: Vote) -> Tuple[str, float, Dict[str, float]]:
    if llm is None:  # LLM leg failed: embedding, then pattern, no weighted vote
        for intent, score in (emb, pat):
            if intent and score > 0:
                return intent, score, {intent: score}
        raise JudgeError("三路融合：LLM 失败，向量和关键词也没有结果")
    scores: Dict[str, float] = defaultdict(float)
    for (intent, score), weight in (((llm.intent, llm.confidence), WEIGHTS["llm"]),
                                    (emb, WEIGHTS["embedding"]), (pat, WEIGHTS["pattern"])):
        if intent:
            scores[intent] += weight * score
    best = max(scores, key=scores.get)
    return best, scores[best], dict(scores)


class FusionJudge:
    def __init__(self, llm_judge):
        self.llm_judge = llm_judge

    def judge(self, state: dict) -> Judgment:
        started = time.perf_counter()
        text = state["latest_customer_message"]
        llm, llm_source = None, None
        try:
            llm = self.llm_judge.judge(state)
            llm_source = {"intent": llm.intent, "confidence": llm.confidence}
        except JudgeError as e:
            llm_source = {"error": str(e)}
        emb, pat = embedding_vote(text), pattern_vote(text)
        intent, score, scores = fuse(llm, emb, pat)
        return Judgment(
            intent, round(min(1.0, score), 4), {k: round(v, 4) for k, v in scores.items()}, "fusion",
            (time.perf_counter() - started) * 1000, reason=llm.reason if llm else "",
            needs_human=llm.needs_human if llm else None,
            sources={"llm": llm_source,
                     "embedding": {"intent": emb[0], "confidence": round(emb[1], 4)},
                     "pattern": {"intent": pat[0], "confidence": pat[1]},
                     "fused": {k: round(v, 4) for k, v in scores.items()}})
