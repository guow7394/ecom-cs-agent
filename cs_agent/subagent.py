"""s06: the after-sales specialist subagent.

The front-desk agent hands refund / exchange work to a specialist that runs
its own agent_loop with a fresh message list and a different tool set. Only
the specialist's final conclusion comes back to the front desk, so long
policy lookups never bloat the customer conversation.

    Front desk                       Specialist
    +---------------------+          +----------------------+
    | messages=[...]      |  task    | messages=[task]      |
    | after_sales_specialist -------> | refund / exchange    |
    | tool_result         | <------- | final conclusion     |
    +---------------------+          +----------------------+
"""
from __future__ import annotations

from .loop import agent_loop
from .session import Session
from .tools import SPECIALIST_TOOL_NAMES, ToolBox

SPECIALIST_TOOL = {
    "name": "after_sales_specialist",
    "description": "把退货退款、换货任务交给售后专员子 Agent 处理。写清楚顾客诉求、订单号（如有）、原因、想换的尺码。返回专员的处理结论。",
    "input_schema": {"type": "object",
                     "properties": {"task": {"type": "string", "description": "交给专员的任务描述"}},
                     "required": ["task"]},
}

SPECIALIST_SYSTEM = """你是电商平台的售后专员子 Agent，只处理退货退款和换货。
顾客ID：{customer_id}；今天是 {today}。
{memory}
流程：
1. 先用 query_order 核实订单（没有订单号就先列出顾客订单再判断是哪一单）；需要时用 search_policy 查政策。
2. 符合条件就直接调用 apply_refund / create_exchange。系统会校验规则并向顾客确认，你不需要再征求同意。
3. 工具返回失败时，不要换参数绕过规则，如实说明原因。如果原因属于需要人工（大额退款、保修检测、超出政策），在结论里写明「建议转人工」。
4. 最后用 2-3 句话给前台客服一个结论：做了什么、结果如何、顾客下一步要做什么。"""

SPECIALIST_MAX_ROUNDS = 6


def run_specialist(llm, session: Session, task: str) -> str:
    box = ToolBox(session, by="specialist")
    system = SPECIALIST_SYSTEM.format(
        customer_id=session.customer_id, today=session.store.today.isoformat(),
        memory=session.memory.render(session.customer_id) or "（无历史记忆）")
    messages = [{"role": "user", "content": task}]
    outcome = agent_loop(llm, system, messages, box.schemas(SPECIALIST_TOOL_NAMES),
                         lambda name, args: box.run(name, args).content, SPECIALIST_MAX_ROUNDS)
    if outcome.exhausted or outcome.truncated:
        return "售后专员处理超时，没有得出结论，建议转人工。"
    return outcome.text or "（售后专员没有返回结论）"
