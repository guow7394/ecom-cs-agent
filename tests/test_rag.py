import pytest

from cs_agent.config import DATA_DIR
from cs_agent.llm import Block, LLMResponse
from cs_agent.rag import PolicyIndex, PolicySearcher


def index():
    return PolicyIndex.from_file(DATA_DIR / "policies.md")


def test_sections_are_split_by_heading():
    titles = [s.title for s in index().sections]
    assert "七天无理由退货" in titles and "人工客服" in titles
    assert all(s.body for s in index().sections)


def test_shipping_insurance_query():
    hits = index().search("退货运费谁出？有运费险吗", k=3)
    assert hits[0].title == "运费与运费险"


def test_custom_item_query():
    assert index().search("定制的东西能退吗")[0].title == "定制商品"


def test_warranty_query():
    assert index().search("保修期是多久")[0].title == "保修政策"


def test_price_protection_query():
    assert index().search("降价了能补差价吗")[0].title == "价格保护与补差价"


def test_empty_query_returns_nothing():
    assert index().search("   ") == []
    assert index().search("！？") == []


# -- query rewrite + rerank (ported from EchoMind tooling/tool_manager.py) --


class Scripted:
    def __init__(self, *texts, stop="end_turn"):
        self.texts, self.stop, self.calls = list(texts), stop, []

    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=None):
        self.calls.append({"system": system, "content": messages[-1]["content"]})
        return LLMResponse([Block("text", text=self.texts.pop(0))], self.stop)


def test_searcher_without_llm_is_plain_search():
    sections, meta = PolicySearcher(index()).search("运费险", k=3)
    assert [s.title for s in sections] == [s.title for s in index().search("运费险", k=3)]
    assert meta == {"queries": ["运费险"], "recalled": len(index().search("运费险", k=5)), "reranked": False}


def test_rewrite_recall_dedupe_and_rerank():
    llm = Scripted('["运费险怎么理赔", "七天无理由退货运费", "退货运费谁承担"]', "[1, 0, 2]")
    sections, meta = PolicySearcher(index(), llm).search("退货运费谁出", k=3)
    assert meta["queries"][0] == "退货运费谁出" and len(meta["queries"]) == 4
    assert meta["recalled"] > 3 and meta["reranked"] is True
    recalled_order = llm.calls[1]["content"]
    assert len(sections) == 3 and len({s.title for s in sections}) == 3
    assert "0." in recalled_order and "1." in recalled_order       # rerank saw numbered candidates
    assert len(llm.calls) == 2


@pytest.mark.parametrize("rewrite,rerank", [
    ("不是 JSON", "[0, 1, 2]"),           # rewrite fails -> original query only
    ('["运费险"]', "乱写"),                # rerank fails -> recall order kept
    ('["运费险"]', "[99, 0, 0]"),          # bad / repeated indices are ignored
])
def test_failures_fall_back_without_breaking_search(rewrite, rerank):
    sections, meta = PolicySearcher(index(), Scripted(rewrite, rerank)).search("退货运费谁出", k=3)
    assert len(sections) == 3 and len({s.title for s in sections}) == 3


def test_truncated_llm_output_counts_as_failure():
    llm = Scripted('["运费险"]', "[1, 0]", stop="max_tokens")
    sections, meta = PolicySearcher(index(), llm).search("退货运费谁出", k=3)
    assert meta["queries"] == ["退货运费谁出"] and meta["reranked"] is False and sections


def test_search_policy_tool_reports_the_retrieval(make_session):
    from cs_agent.tools import ToolBox
    s = make_session()
    s.policy_searcher = PolicySearcher(s.policies, Scripted('["运费险怎么理赔"]', "[0, 1, 2, 3, 4]"))
    out = ToolBox(s).run("search_policy", {"query": "运费险"}).content
    assert out.startswith("【检索】") and "运费与运费险" in out


def test_agent_turns_rewrite_on_unless_disabled(make_session, monkeypatch):
    from cs_agent.agent import CustomerServiceAgent
    from cs_agent.llm import MockLLM
    monkeypatch.delenv("RAG_REWRITE", raising=False)
    assert CustomerServiceAgent(MockLLM(), make_session()).session.policy_searcher is not None
    monkeypatch.setenv("RAG_REWRITE", "0")
    assert CustomerServiceAgent(MockLLM(), make_session()).session.policy_searcher is None


def test_rewrite_and_rerank_leave_room_for_thinking():
    # measured on deepseek-v4-flash: rerank alone reasons for 1.5k-2k+ tokens; 2048 truncated 2 of 3 calls
    from cs_agent.rag import LLM_MAX_TOKENS
    assert LLM_MAX_TOKENS >= 4096


class Exploding:
    def create(self, **kwargs):
        raise TimeoutError("DeepSeek 超时")


def test_llm_failures_are_reported_not_swallowed(make_session):
    sections, meta = PolicySearcher(index(), Exploding()).search("退货运费谁出", k=3)
    assert sections and meta["queries"] == ["退货运费谁出"]
    assert meta["errors"] == ["改写失败：TimeoutError", "重排失败：TimeoutError"]
    from cs_agent.tools import ToolBox
    s = make_session()
    s.policy_searcher = PolicySearcher(s.policies, Exploding())
    header = ToolBox(s).run("search_policy", {"query": "退货运费谁出"}).content.splitlines()[0]
    assert "改写失败：TimeoutError" in header
