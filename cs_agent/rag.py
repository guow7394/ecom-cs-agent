"""search_policy backend: split the policy doc by `##` and rank sections by
character-bigram overlap. Chinese has no spaces, so bigrams are a cheap,
dependency-free stand-in for tokenisation; a vector store would slot in here.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .llm import text_of

_KEEP = re.compile(r"[一-鿿A-Za-z0-9]+")
# Function words that would otherwise match every section.
_STOP = {"可以", "什么", "怎么", "多久", "能不", "不能", "是不", "吗", "的", "了", "我", "你们"}
TITLE_WEIGHT = 3.0


def bigrams(text: str) -> Set[str]:
    grams = set()
    for run in _KEEP.findall(text.lower()):
        if len(run) == 1:
            grams.add(run)
        grams.update(run[i:i + 2] for i in range(len(run) - 1))
    return grams - _STOP


@dataclass
class Section:
    title: str
    body: str

    def render(self) -> str:
        return f"【{self.title}】{self.body}"


class PolicyIndex:
    def __init__(self, sections: List[Section]):
        self.sections = sections
        self._grams = [(bigrams(s.title), bigrams(s.body)) for s in sections]

    @classmethod
    def from_file(cls, path: Path) -> "PolicyIndex":
        sections, title, lines = [], None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if title:
                    sections.append(Section(title, "\n".join(lines).strip()))
                title, lines = line[3:].strip(), []
            elif title:
                lines.append(line)
        if title:
            sections.append(Section(title, "\n".join(lines).strip()))
        return cls(sections)

    def search(self, query: str, k: int = 3) -> List[Section]:
        q = bigrams(query)
        if not q:
            return []
        scored = []
        for section, (t_grams, b_grams) in zip(self.sections, self._grams):
            body_hits = len(q & b_grams) / math.sqrt(len(b_grams) or 1)
            score = TITLE_WEIGHT * len(q & t_grams) + body_hits
            if score > 0:
                scored.append((score, section))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [section for _, section in scored[:k]]


# ---------------------------------------------------------------------------
# Query rewrite + rerank, ported from EchoMind (tooling/tool_manager.py)
#
#   query ─► LLM rewrites it into up to 3 angles ─► recall each (k >= 5)
#         ─► merge, dedupe by section ─► LLM reranks if more than k ─► top k
#
# Every LLM step is optional: a failed or truncated call falls back to the
# plain keyword search, so retrieval never gets worse than without it.
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = ("把顾客的问题改写成 3 个角度不同、便于在电商售后政策文档里检索的查询。"
                  "只返回 JSON 字符串数组，例如 [\"运费险理赔\", \"退货运费承担\", \"七天无理由\"]。")
RERANK_SYSTEM = ("根据顾客的问题，把下面编号的政策片段按相关性从高到低排序。"
                 "只返回片段编号组成的 JSON 数组，例如 [2, 0, 1]。")
LLM_MAX_TOKENS = 4096
# Rewriting and ranking are formatting jobs: with thinking on, a rerank took 24-27s and
# ~4k tokens (often truncated); with it off, 0.7s and 27 tokens with the same top result.
NO_THINKING = {"type": "disabled"}
_ARRAY_RE = re.compile(r"\[.*?\]", re.S)


class PolicySearcher:
    def __init__(self, index: PolicyIndex, llm=None, rewrites: int = 3):
        self.index = index
        self.llm = llm
        self.rewrites = rewrites

    def search(self, query: str, k: int = 3) -> Tuple[List[Section], dict]:
        errors: List[str] = []
        queries = [query] + [q for q in self._rewrite(query, errors) if q != query][:self.rewrites]
        merged, seen = [], set()
        for q in queries:
            for section in self.index.search(q, k=max(k, 5)):
                if section.title not in seen:
                    seen.add(section.title)
                    merged.append(section)
        ranked, reranked = merged, False
        if len(merged) > k and self.llm is not None:
            order = self._rerank(query, merged, errors)
            if order:
                ranked, reranked = [merged[i] for i in order], True
        meta = {"queries": queries, "recalled": len(merged), "reranked": reranked}
        if errors:  # fallbacks stay visible: a silent fallback looks like a model choice
            meta["errors"] = errors
        return ranked[:k], meta

    def _ask(self, system: str, content: str, step: str, errors: List[str]) -> Optional[list]:
        try:
            response = self.llm.create(system=system, messages=[{"role": "user", "content": content}],
                                       max_tokens=LLM_MAX_TOKENS, thinking=NO_THINKING)
        except Exception as e:
            errors.append(f"{step}失败：{type(e).__name__}")
            return None
        if getattr(response, "stop_reason", None) == "max_tokens":
            errors.append(f"{step}失败：输出被截断")
            return None
        match = _ARRAY_RE.search(text_of(response.content))
        try:
            value = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            value = None
        if not isinstance(value, list):
            errors.append(f"{step}失败：没有返回 JSON 数组")
            return None
        return value

    def _rewrite(self, query: str, errors: List[str]) -> List[str]:
        if self.llm is None:
            return []
        value = self._ask(REWRITE_SYSTEM, query, "改写", errors) or []
        return list(dict.fromkeys(q.strip() for q in value if isinstance(q, str) and q.strip()))

    def _rerank(self, query: str, sections: List[Section], errors: List[str]) -> List[int]:
        listing = "\n".join(f"{i}. 【{s.title}】{s.body[:120]}" for i, s in enumerate(sections))
        value = self._ask(RERANK_SYSTEM, f"顾客的问题：{query}\n\n{listing}", "重排", errors)
        if not value:
            return []
        order = list(dict.fromkeys(i for i in value if isinstance(i, int) and not isinstance(i, bool)
                                   and 0 <= i < len(sections)))
        # anything the model left out keeps its recall order at the end
        return order + [i for i in range(len(sections)) if i not in order] if order else []
