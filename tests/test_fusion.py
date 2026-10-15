"""Three-way intent fusion ported from EchoMind (LLM 0.7 + embedding 0.2 + pattern 0.1)."""
import pytest

from cs_agent.agent import build_judge
from cs_agent.fusion import (FusionJudge, TEMPLATES, cosine, embedding_vote, fuse, local_embedding,
                             pattern_vote)
from cs_agent.judge import INTENTS, Judgment, JudgeError, build_state
from cs_agent.llm import MockLLM
from eval.metrics import DATASET_PATH, load_dataset


def J(intent, confidence, needs_human=0.2):
    return Judgment(intent, confidence, {intent: confidence}, "llm", needs_human=needs_human)


def test_templates_cover_every_intent_and_do_not_copy_the_test_set():
    assert set(TEMPLATES) == set(INTENTS)
    messages = {c["message"] for c in load_dataset(DATASET_PATH)}
    assert not messages & {t for ts in TEMPLATES.values() for t in ts}


def test_local_embedding_is_deterministic():
    a, b = local_embedding("快递到哪了"), local_embedding("快递到哪了")
    assert a == b and len(a) == 256
    assert cosine(a, b) == pytest.approx(1.0)
    assert cosine(a, [0.0] * 256) == 0.0


def test_embedding_vote_finds_the_nearest_template():
    intent, score = embedding_vote("包裹现在到哪儿了呀")
    assert intent == "logistics_query" and 0 < score <= 1


@pytest.mark.parametrize("text,expected", [
    ("转人工", ("request_human", 0.5)),
    ("转人工，叫经理来", ("request_human", 0.75)),           # two keywords hit
    ("转人工，我要找人工客服", ("request_human", 1.0)),      # three hits, capped at 1.0
    ("嗯……", (None, 0.0)),
])
def test_pattern_vote(text, expected):
    assert pattern_vote(text) == expected


def test_fuse_adds_weighted_scores_per_intent():
    intent, score, scores = fuse(J("exchange", 0.9), ("exchange", 0.6), (None, 0.0))
    assert intent == "exchange" and score == pytest.approx(0.7 * 0.9 + 0.2 * 0.6)
    assert scores == {"exchange": pytest.approx(0.75)}


def test_fuse_two_weak_legs_can_outvote_nothing_but_not_a_confident_llm():
    intent, score, _ = fuse(J("return_refund", 0.5), ("logistics_query", 0.9), ("logistics_query", 0.5))
    assert intent == "return_refund" and score == pytest.approx(0.35)   # 0.35 vs 0.18 + 0.05


def test_fuse_falls_back_when_the_llm_leg_fails():
    assert fuse(None, ("exchange", 0.6), ("exchange", 0.5))[:2] == ("exchange", 0.6)
    assert fuse(None, (None, 0.0), ("request_human", 0.5))[:2] == ("request_human", 0.5)
    with pytest.raises(JudgeError):
        fuse(None, (None, 0.0), (None, 0.0))


class FixedLLMJudge:
    def __init__(self, judgment=None, error=None):
        self.judgment, self.error = judgment, error

    def judge(self, state):
        if self.error:
            raise JudgeError(self.error)
        return self.judgment


def test_fusion_judge_returns_a_fused_judgment_with_sources():
    judge = FusionJudge(FixedLLMJudge(J("logistics_query", 0.9, needs_human=0.1)))
    j = judge.judge(build_state([], "我的包裹到哪了"))
    assert j.backend == "fusion" and j.intent == "logistics_query"
    assert 0.63 <= j.confidence <= 1.0
    assert j.needs_human == 0.1                       # only the LLM leg answers the second question
    assert set(j.sources) == {"llm", "embedding", "pattern", "fused"}
    assert j.sources["llm"] == {"intent": "logistics_query", "confidence": 0.9}


def test_fusion_judge_records_an_llm_failure_and_falls_back():
    j = FusionJudge(FixedLLMJudge(error="timeout")).judge(build_state([], "转人工"))
    assert j.intent == "request_human" and j.needs_human is None
    assert j.sources["llm"] == {"error": "timeout"}


def test_fusion_backend_is_wired_into_the_factory():
    judge = build_judge("fusion", MockLLM(), mock=True)
    assert judge.judge(build_state([], "快递到哪了")).backend == "fusion"


def test_fusion_is_an_experiment_only_backend():
    from cs_agent.agent import JUDGE_BACKENDS, PRODUCT_JUDGES
    assert "fusion" in JUDGE_BACKENDS and "fusion" not in PRODUCT_JUDGES
