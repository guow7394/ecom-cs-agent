"""s08: context compaction for long support conversations.

When the message list grows past max_chars, everything before a safe cut
point is summarised into one message. The cut is always at a customer's
plain-text turn, so a tool_use is never separated from its tool_result.
The full transcript is saved to disk first (same idea as upstream's
.transcripts/), so nothing is lost for audit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from .llm import block_get, text_of

SUMMARY_SYSTEM = ("你负责压缩客服对话历史。用要点总结：顾客诉求、涉及的订单号、"
                  "已执行的操作及结果、尚未解决的问题。不超过 200 字。")


def render_messages(messages: list) -> str:
    lines = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(("顾客：" if m["role"] == "user" else "客服：") + content)
            continue
        for b in content:
            kind = block_get(b, "type")
            if kind == "text":
                lines.append("客服：" + block_get(b, "text", ""))
            elif kind == "tool_use":
                lines.append(f"[调用 {block_get(b, 'name')}] "
                             + json.dumps(block_get(b, "input") or {}, ensure_ascii=False))
            elif kind == "tool_result":
                lines.append("[结果] " + str(block_get(b, "content", ""))[:300])
    return "\n".join(lines)


def llm_summarizer(llm) -> Callable[[str], str]:
    def summarize(text: str) -> str:
        response = llm.create(system=SUMMARY_SYSTEM,
                              messages=[{"role": "user", "content": text}], max_tokens=600)
        return text_of(response.content)
    return summarize


class Compactor:
    def __init__(self, summarize: Callable[[str], str], max_chars: int = 12000,
                 keep_last: int = 8, transcript_dir: Optional[Path] = None):
        self.summarize = summarize
        self.max_chars = max_chars
        self.keep_last = keep_last
        self.transcript_dir = transcript_dir

    @staticmethod
    def size(messages: list) -> int:
        return len(json.dumps(messages, ensure_ascii=False, default=str))

    def _cut_index(self, messages: list) -> int:
        limit = len(messages) - self.keep_last
        cuts = [i for i, m in enumerate(messages)
                if 0 < i <= limit and m["role"] == "user" and isinstance(m["content"], str)]
        return cuts[-1] if cuts else 0

    def maybe_compact(self, messages: list) -> list:
        if self.size(messages) <= self.max_chars:
            return messages
        cut = self._cut_index(messages)
        if cut == 0:
            return messages
        if self.transcript_dir:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            path = self.transcript_dir / f"transcript_{int(time.time() * 1000)}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
        summary = self.summarize(render_messages(messages[:cut]))
        return [{"role": "user", "content": f"[早前对话摘要]\n{summary}"},
                {"role": "assistant", "content": "好的，我已了解之前的对话内容。"}] + messages[cut:]
