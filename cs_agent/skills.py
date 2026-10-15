"""s07 skill loading, combined with EchoMind's keyword injection.

    data/skills/<name>/SKILL.md      frontmatter: name, description, keywords, agents
            |
            v
    SkillManager
      catalog        every skill's name + description goes into the system prompt   (s07 layer 1)
      auto-inject    skills whose keywords appear in the customer's text, for this
                     agent role, are injected in full, within a character budget      (EchoMind)
      load_skill     the model can still pull any other skill on demand              (s07 layer 2)

Skills describe *how* to handle a case (steps, what to say, where the line is);
the rules themselves stay in code (store.py / permission.py), so a skill can
never make the bot do something the rules forbid.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from .config import DATA_DIR

SKILLS_DIR = DATA_DIR / "skills"
MAX_PROMPT_CHARS = 3000


@dataclass
class Skill:
    name: str
    description: str
    body: str
    keywords: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)   # empty = every agent role

    def hits(self, text: str, agent: str) -> List[str]:
        if self.agents and agent not in self.agents:
            return []
        lowered = text.lower()
        return [k for k in self.keywords if k.lower() in lowered]


def _split_list(value: str) -> List[str]:
    return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]


def parse_skill(text: str) -> Skill:
    if not text.startswith("---"):
        raise ValueError("缺少 --- frontmatter")
    _, meta, body = text.split("---", 2)
    fields = {}
    for line in meta.strip().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    if not fields.get("name"):
        raise ValueError("frontmatter 缺少 name")
    return Skill(name=fields["name"], description=fields.get("description", ""), body=body.strip(),
                 keywords=_split_list(fields.get("keywords", "")),
                 agents=[a.lower() for a in _split_list(fields.get("agents", ""))])


class SkillManager:
    def __init__(self, root: Path = SKILLS_DIR, max_chars: int = MAX_PROMPT_CHARS):
        self.root = Path(root)
        self.max_chars = max_chars
        self.skills: List[Skill] = []
        self.errors: List[str] = []
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> List[Skill]:
        skills, errors = [], []
        for path in sorted(self.root.glob("*/SKILL.md")):
            try:
                skills.append(parse_skill(path.read_text(encoding="utf-8")))
            except ValueError as e:
                errors.append(f"{path.parent.name}: {e}")
        with self._lock:
            self.skills, self.errors = skills, errors
        return skills

    def get(self, name: str) -> Optional[Skill]:
        return next((s for s in self.skills if s.name == name), None)

    def select(self, text: str, agent: str) -> List[Tuple[Skill, List[str]]]:
        return [(s, hits) for s in self.skills for hits in [s.hits(text, agent)] if hits]

    def prompt_for(self, text: str, agent: str) -> str:
        if not self.skills:
            return ""
        lines = ["业务规范目录（需要时用 load_skill 加载全文）："]
        lines += [f"- {s.name}：{s.description}" for s in self.skills]
        budget, blocks = self.max_chars, []
        for skill, _ in self.select(text, agent):
            block = f"### {skill.name}\n{skill.body}"
            if len(block) > budget:
                block = block[:max(0, budget)].rstrip() + "…"
            blocks.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        if blocks:
            lines += ["", "[已按本轮内容自动加载的业务规范]"] + blocks
        return "\n".join(lines)


@lru_cache(maxsize=1)
def default_skills() -> SkillManager:
    return SkillManager()
