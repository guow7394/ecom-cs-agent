"""Skills: s07 catalog + load_skill, plus EchoMind-style keyword auto-injection."""
from cs_agent.agent import CustomerServiceAgent
from cs_agent.llm import MockLLM
from cs_agent.skills import SkillManager
from cs_agent.tools import BASE_TOOL_NAMES, SPECIALIST_TOOL_NAMES, ToolBox

SKILL = """---
name: {name}
description: {desc}
keywords: {keywords}
agents: {agents}
---
# {name}
{body}
"""


def write_skill(root, folder, name="测试规范", desc="测试用", keywords="退款,退货", agents="main", body="正文"):
    (root / folder).mkdir(parents=True)
    (root / folder / "SKILL.md").write_text(
        SKILL.format(name=name, desc=desc, keywords=keywords, agents=agents, body=body), encoding="utf-8")


def test_bundled_skills_load():
    names = {s.name for s in SkillManager().skills}
    assert {"退货退款处理规范", "换货处理规范", "物流查询与催件规范", "投诉安抚规范"} <= names
    assert SkillManager().errors == []


def test_selection_by_role_and_keywords():
    manager = SkillManager()
    picked = {s.name: hits for s, hits in manager.select("那双鞋挤脚，想换大一码", "main")}
    assert set(picked["换货处理规范"]) >= {"换", "大一码"}
    assert "物流查询与催件规范" not in picked
    assert not [s for s, _ in manager.select("快递到哪了", "specialist")]   # logistics norm is front-desk only


def test_prompt_contains_catalog_and_matched_bodies_within_budget(tmp_path):
    write_skill(tmp_path, "a", name="甲", keywords="退款", body="甲" * 50)
    write_skill(tmp_path, "b", name="乙", keywords="换货", body="乙" * 50)
    manager = SkillManager(tmp_path, max_chars=30)
    prompt = manager.prompt_for("我要退款", "main")
    assert "- 甲：测试用" in prompt and "- 乙：测试用" in prompt      # catalog lists every skill
    assert "### 甲" in prompt and "### 乙" not in prompt              # only the matched one in full
    assert "甲" * 31 not in prompt and "…" in prompt                   # body cut to the budget
    assert manager.prompt_for("你好", "main").count("###") == 0


def test_files_without_frontmatter_are_reported_not_loaded(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
    manager = SkillManager(tmp_path)
    assert manager.skills == [] and "broken" in manager.errors[0]


def test_reload_picks_up_new_files(tmp_path):
    manager = SkillManager(tmp_path)
    assert manager.skills == []
    write_skill(tmp_path, "new")
    assert [s.name for s in manager.reload()] == ["测试规范"]


def test_load_skill_tool(make_session):
    assert "load_skill" in BASE_TOOL_NAMES and "load_skill" in SPECIALIST_TOOL_NAMES
    box = ToolBox(make_session())
    found = box.run("load_skill", {"name": "投诉安抚规范"})
    assert found.ok and "投诉安抚规范" in found.content
    missing = box.run("load_skill", {"name": "不存在"})
    assert not missing.ok and "退货退款处理规范" in missing.content      # lists what exists


class SpyLLM(MockLLM):
    def __init__(self):
        self.systems = []

    def create(self, system, messages, tools=None, tool_choice=None, max_tokens=None, thinking=None):
        self.systems.append(system)
        return super().create(system, messages, tools, tool_choice, max_tokens)


def test_matched_skills_reach_both_agents_and_are_reported(make_session):
    llm, events = SpyLLM(), []
    s = make_session("C005")
    s.on_event = lambda t, d: events.append((t, d))
    CustomerServiceAgent(llm, s).handle("A5001 牛仔裤小了，想换31码")
    front_desk, specialist = llm.systems[0], llm.systems[1]
    assert "### 换货处理规范" in front_desk and "### 换货处理规范" in specialist
    assert ("skills", {"by": "main", "names": ["换货处理规范"]}) in events
