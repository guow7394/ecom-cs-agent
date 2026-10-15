"""The customer-service agent: judgment layer in front of an s01 agent loop.

    handle(text)
      ├─ judge(state)  ──► gate(judgment, policy)
      │                      ├─ human ──► hand_off(): ticket + canned reply, LLM never runs
      │                      └─ agent ──► respond(text, judgment)
      └─ (no judge = baseline) ─────────► respond(text)

    respond(text, judgment)
      s08 compact history → system prompt (+ s09 memory, + judge hint)
      → s01 agent_loop with s02 tools, s03 gated actions, s06 specialist

handle() is what production uses. The evaluation calls judge + respond
directly so one judge call can be scored at many thresholds.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .compact import Compactor, llm_summarizer
from .config import PROJECT_ROOT, default_needs_human_threshold, default_threshold, load_env
from .judge import (INTENT_ZH, GateDecision, GatePolicy, JevJudge, Judgment, LLMJudge,
                    MockJudge, build_state, gate)
from .llm import AnthropicLLM, MockLLM
from .loop import agent_loop
from .permission import ConfirmFn
from .session import Session, new_session
from .subagent import SPECIALIST_TOOL, run_specialist
from .tools import BASE_TOOL_NAMES, ToolBox

SYSTEM_TEMPLATE = """你是「小安」，电商平台的售后客服助手。
顾客ID：{customer_id}（{customer_name}，{level}）；今天是 {today}。
{memory}
工作规则：
1. 涉及订单、物流、政策的事实必须先调用工具查询，不要编造。顾客没说订单号时，先 query_order 列出订单再判断是哪一单。
2. 退货退款、换货交给 after_sales_specialist；改地址用 change_address。系统会自动请顾客确认这些操作，你不需要先问"确认吗"。
3. 以下情况调用 transfer_to_human：顾客明确要求人工；投诉、索赔、人身伤害或过敏、法律或媒体威胁；问题超出政策或工具能力（需人工审核、检测、补偿、保价、发票改抬头等）；工具结果提示需要人工；你无法确定顾客想要什么。
4. 顾客透露尺码等长期偏好时，用 remember_preference 记下来。
5. 与售后无关的问题，礼貌说明你只负责售后。
6. 用简洁、礼貌的中文回复，不超过 120 字。"""

JUDGE_HINT = "\n[判断层] 本轮顾客意图：{intent}（{zh}），置信度 {confidence:.2f}。仅供参考，以对话内容为准。"
HANDOFF_REPLY = "您的问题需要人工客服进一步处理，已为您转接（工单号 {ticket}），专员会尽快联系您，请稍候。"
MAX_ROUNDS = 8
FAILURE_REASONS = {"loop_limit": "工具调用轮数超限", "truncated": "模型输出被 max_tokens 截断"}


@dataclass
class TurnResult:
    reply: str
    route: str                      # "agent" | "human"
    reason: str                     # why it ended up there ("" when the agent handled it)
    judgment: Optional[Judgment] = None
    gate: Optional[GateDecision] = None
    tools: List[dict] = field(default_factory=list)
    error: str = ""


class CustomerServiceAgent:
    def __init__(self, llm, session: Session, judge=None, policy: Optional[GatePolicy] = None,
                 compactor: Optional[Compactor] = None, max_rounds: int = MAX_ROUNDS):
        self.llm = llm
        self.session = session
        self.judge = judge
        self.policy = policy or GatePolicy()
        self.compactor = compactor
        self.max_rounds = max_rounds
        self.messages: list = []                       # what the LLM sees
        self.transcript: List[Tuple[str, str]] = []    # plain text, what the judge sees

    def preload_history(self, pairs: List[Tuple[str, str]]) -> None:
        for role, text in pairs:
            expected = "customer" if len(self.transcript) % 2 == 0 else "agent"
            if role != expected:
                raise ValueError(f"history must alternate customer/agent, got {role} at {len(self.transcript)}")
            self.transcript.append((role, text))
            self.messages.append({"role": "user" if role == "customer" else "assistant", "content": text})

    # -- production entry point --

    def handle(self, text: str) -> TurnResult:
        if self.judge is None:
            return self.respond(text)
        judgment, error = None, ""
        try:
            judgment = self.judge.judge(build_state(self.transcript, text))
        except Exception as e:  # fail safe: an unavailable judge means a human looks at it
            error = f"{type(e).__name__}: {e}"
        if judgment is None:
            self.session.emit("judgment", error=error)
        else:
            self.session.emit("judgment", intent=judgment.intent, intent_zh=INTENT_ZH[judgment.intent],
                              confidence=judgment.confidence, needs_human=judgment.needs_human,
                              latency_ms=round(judgment.latency_ms), backend=judgment.backend)
        decision = gate(judgment, self.policy)
        self.session.emit("gate", route=decision.route, reason=decision.reason, threshold=decision.threshold,
                          needs_human_threshold=self.policy.needs_human_threshold)
        if decision.route == "human":
            return self.hand_off(text, judgment, decision, error)
        return self.respond(text, judgment, decision)

    def hand_off(self, text: str, judgment: Optional[Judgment], decision: GateDecision,
                 error: str = "") -> TurnResult:
        ticket = f"H{uuid.uuid4().hex[:6].upper()}"
        self.session.handoff = {
            "by": "gate", "ticket_id": ticket, "reason": decision.reason,
            "intent": judgment.intent if judgment else None,
            "confidence": judgment.confidence if judgment else None,
            "recent": self.transcript[-6:] + [("customer", text)],
        }
        reply = HANDOFF_REPLY.format(ticket=ticket)
        self._record(text, reply)
        self.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return TurnResult(reply, "human", decision.reason, judgment, decision, [], error)

    # -- the LLM agent --

    def system_prompt(self, judgment: Optional[Judgment], text: str = "") -> str:
        s = self.session
        customer = s.store.customers[s.customer_id]
        memory = s.memory.render(s.customer_id)
        prompt = SYSTEM_TEMPLATE.format(
            customer_id=s.customer_id, customer_name=customer["name"], level=customer["level"],
            today=s.store.today.isoformat(),
            memory=f"顾客记忆：\n{memory}" if memory else "顾客记忆：（无）")
        if s.skills is not None:
            prompt += "\n\n" + s.skills.prompt_for(text, "main")
            matched = [skill.name for skill, _ in s.skills.select(text, "main")]
            if matched:
                s.emit("skills", by="main", names=matched)
        if judgment is not None:
            prompt += JUDGE_HINT.format(intent=judgment.intent, zh=INTENT_ZH[judgment.intent],
                                        confidence=judgment.confidence)
        return prompt

    def respond(self, text: str, judgment: Optional[Judgment] = None,
                decision: Optional[GateDecision] = None) -> TurnResult:
        s = self.session
        first_tool = len(s.tool_log)
        handoff_before = s.handoff
        self.messages.append({"role": "user", "content": text})
        if self.compactor:
            self.messages = self.compactor.maybe_compact(self.messages)

        box = ToolBox(s)

        def dispatch(name: str, args: dict) -> str:
            if name == "after_sales_specialist":
                s.emit("tool_start", name=name, by="main", input=args)
                s.tool_log.append({"name": name, "input": args, "ok": True, "by": "main", "output": ""})
                conclusion = run_specialist(self.llm, s, args.get("task", ""))
                s.emit("tool_end", name=name, by="main", ok=True, output=conclusion[:300])
                return conclusion
            return box.run(name, args).content

        outcome = agent_loop(self.llm, self.system_prompt(judgment, text), self.messages,
                             box.schemas(BASE_TOOL_NAMES) + [SPECIALIST_TOOL], dispatch, self.max_rounds)
        reply, route, reason = outcome.text, "agent", ""
        # Fail safe: a turn the agent could not finish goes to a human, and says why.
        failure = "loop_limit" if outcome.exhausted else "truncated" if outcome.truncated else ""
        if failure and s.handoff is handoff_before:
            s.handoff = {"by": failure, "ticket_id": f"H{uuid.uuid4().hex[:6].upper()}",
                         "reason": FAILURE_REASONS[failure]}
        if s.handoff is not handoff_before:
            route = "human"
            reason = "agent_transfer" if s.handoff["by"] == "agent" else s.handoff["by"]
            reply = reply or HANDOFF_REPLY.format(ticket=s.handoff["ticket_id"])
        reply = reply or "抱歉，我需要再确认一下，请稍等。"
        self._record(text, reply)
        return TurnResult(reply, route, reason, judgment, decision, s.tool_log[first_tool:])

    def _record(self, text: str, reply: str) -> None:
        self.transcript += [("customer", text), ("agent", reply)]


# ---------------------------------------------------------------------------
# Factories shared by chat.py and the evaluation
# ---------------------------------------------------------------------------

JUDGE_BACKENDS = ("none", "jev", "llm", "fusion")


def build_llm(mock: bool = False):
    return MockLLM() if mock else AnthropicLLM()


def build_judge(backend: str, llm, mock: bool = False):
    if backend == "none":
        return None
    if backend == "jev":
        return MockJudge() if mock else JevJudge()
    if backend == "llm":
        return LLMJudge(llm)  # with mock=True, llm is MockLLM, which answers report_judgment
    if backend == "fusion":
        from .fusion import FusionJudge  # EchoMind-style LLM + embedding + pattern vote
        return FusionJudge(LLMJudge(llm))
    raise ValueError(f"unknown judge backend {backend!r}; choose from {JUDGE_BACKENDS}")


def build_agent(customer_id: str, judge_backend: str = "jev", mock: bool = False,
                threshold: Optional[float] = None, confirm_fn: Optional[ConfirmFn] = None,
                memory_root: Path = PROJECT_ROOT / ".memory",
                transcript_dir: Optional[Path] = PROJECT_ROOT / ".transcripts") -> CustomerServiceAgent:
    load_env()
    llm = build_llm(mock)
    session = new_session(customer_id, memory_root, confirm_fn or (lambda _: True))
    return CustomerServiceAgent(
        llm, session, judge=build_judge(judge_backend, llm, mock),
        policy=GatePolicy(threshold=default_threshold() if threshold is None else threshold,
                          needs_human_threshold=default_needs_human_threshold()),
        compactor=Compactor(llm_summarizer(llm), transcript_dir=transcript_dir))
