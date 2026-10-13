"""search_policy backend: split the policy doc by `##` and rank sections by
character-bigram overlap. Chinese has no spaces, so bigrams are a cheap,
dependency-free stand-in for tokenisation; a vector store would slot in here.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

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
