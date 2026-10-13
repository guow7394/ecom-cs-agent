"""s01: the agent loop, unchanged in spirit from learn-claude-code.

    while True:
        response = LLM(messages, tools)
        if response contains no tool_use:
            break
        execute tools
        append results

The only addition is a round limit, because a support bot that loops
forever is worse than one that hands the customer to a human.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .llm import block_get, text_of

Dispatch = Callable[[str, dict], str]


@dataclass
class LoopOutcome:
    text: str
    exhausted: bool
    rounds: int
    truncated: bool = False   # final answer cut off by max_tokens (no tool call, no full reply)


def agent_loop(llm, system: str, messages: list, tools: List[dict],
               dispatch: Dispatch, max_rounds: int = 8) -> LoopOutcome:
    for round_no in range(1, max_rounds + 1):
        response = llm.create(system=system, messages=messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if block_get(b, "type") == "tool_use"]
        if not tool_calls:
            truncated = getattr(response, "stop_reason", None) == "max_tokens"
            return LoopOutcome("" if truncated else text_of(response.content), False, round_no, truncated)

        results = []
        for block in tool_calls:
            output = dispatch(block_get(block, "name"), block_get(block, "input") or {})
            results.append({"type": "tool_result", "tool_use_id": block_get(block, "id"),
                            "content": output})
        messages.append({"role": "user", "content": results})

    # Close the turn so the next user message keeps user/assistant alternation.
    messages.append({"role": "assistant", "content": "（工具调用轮数超限，已停止）"})
    return LoopOutcome("", True, max_rounds)
