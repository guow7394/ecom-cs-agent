#!/usr/bin/env python3
"""Interactive after-sales chat.

    python3 chat.py --customer C001                 # Jev 判断层（默认）
    python3 chat.py --customer C001 --judge none    # 无判断层（baseline）
    python3 chat.py --customer C001 --judge llm     # DeepSeek 自报置信度
    python3 chat.py --customer C001 --judge fusion  # 三路加权融合（参考 EchoMind）
    python3 chat.py --mock                          # 不需要任何 API key 的离线演示

Customers: C001-C006 (see cs_agent/data/customers.json).
"""
from __future__ import annotations

import argparse

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit (same as learn-claude-code)
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from cs_agent.agent import JUDGE_BACKENDS, build_agent
from cs_agent.judge import INTENT_ZH

YELLOW, CYAN, GREY, RED, GREEN, RESET = "\033[33m", "\033[36m", "\033[90m", "\033[31m", "\033[32m", "\033[0m"


def confirm(text: str) -> bool:
    print(f"\n{YELLOW}[需要您确认] {text}{RESET}")
    try:
        answer = input("   确认请输入 y，取消直接回车：")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes", "是", "确认")


def show_turn(result) -> None:
    j, g = result.judgment, result.gate
    if j is not None:
        arrow = f"{GREEN}→ Agent{RESET}" if g.route == "agent" else f"{RED}→ 转人工（{g.reason}）{RESET}"
        print(f"{GREY}[判断层 {j.backend}] 意图={j.intent}（{INTENT_ZH[j.intent]}） "
              f"置信度={j.confidence:.2f}（阈值 {g.threshold:.2f}） 需人工={j.needs_human:.2f} {RESET}{arrow}")
    elif result.reason == "judge_error":
        print(f"{RED}[判断层出错，按规则转人工] {result.error}{RESET}")
    for t in result.tools:
        mark = "✓" if t["ok"] else "✗"
        preview = t["output"][:80].replace("\n", " ")
        print(f"{GREY}  [{t['by']}] {t['name']} {mark} {preview}{RESET}")
    print(result.reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="电商售后客服 Agent")
    parser.add_argument("--customer", default="C001")
    parser.add_argument("--judge", choices=JUDGE_BACKENDS, default="jev")
    parser.add_argument("--threshold", type=float, help="判断层阈值，默认读 HANDOFF_THRESHOLD")
    parser.add_argument("--mock", action="store_true", help="离线模式：MockLLM + MockJudge")
    args = parser.parse_args()

    agent = build_agent(args.customer, args.judge, args.mock, args.threshold, confirm)
    name = agent.session.store.customers[args.customer]["name"]
    mode = "MOCK 离线" if args.mock else "在线"
    print(f"售后客服小安（{mode}，判断层={args.judge}，阈值={agent.policy.threshold}）· 顾客 {args.customer} {name}")
    print("输入问题回车发送，输入 q 退出。\n")

    while True:
        try:
            query = input(f"\001{CYAN}\002顾客 >> \001{RESET}\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        result = agent.handle(query)
        show_turn(result)
        print()
        if result.route == "human":
            h = agent.session.handoff
            print(f"{YELLOW}=== 已转人工：工单 {h['ticket_id']}，原因 {h.get('reason')}。对话结束 ==={RESET}")
            break


if __name__ == "__main__":
    main()
