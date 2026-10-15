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
import time
from typing import Callable, List, Optional

from .llm import block_get, text_of

Dispatch = Callable[[str, dict], str]
OnRound = Callable[[dict], None]


@dataclass
class LoopOutcome:
    text: str
    exhausted: bool
    rounds: int
    truncated: bool = False   # final answer cut off by max_tokens (no tool call, no full reply)


def agent_loop(llm, system: str, messages: list, tools: List[dict],
               dispatch: Dispatch, max_rounds: int = 8, on_round: Optional[OnRound] = None) -> LoopOutcome:
    for round_no in range(1, max_rounds + 1):
        started = time.perf_counter()
        response = llm.create(system=system, messages=messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if block_get(b, "type") == "tool_use"]
        if on_round:  # for traces: how long each LLM round took and what it asked for
            usage = getattr(response, "usage", None)
            on_round({"round": round_no, "ms": round((time.perf_counter() - started) * 1000),
                      "stop_reason": getattr(response, "stop_reason", None),
                      "tools": [block_get(b, "name") for b in tool_calls],
                      "input_tokens": getattr(usage, "input_tokens", None),
                      "output_tokens": getattr(usage, "output_tokens", None)})
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
