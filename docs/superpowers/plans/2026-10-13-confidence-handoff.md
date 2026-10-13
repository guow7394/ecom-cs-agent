# 置信度驱动转人工 · 售后客服 Agent 实现计划

> **For agentic workers:** 本计划在同一会话内联执行（executing-plans 方式）。代码直接写入源文件，本文档定义任务边界、接口和测试。

**Goal:** 基于 learn-claude-code 的 s01/s02/s03/s06/s08/s09 做一个电商售后客服 Agent，加入 Jev 判断层（意图+置信度门控转人工），并用 30 条对话三组对比评测。

**Architecture:** 纯 Python 包 `cs_agent/`。判断层与 Agent 解耦：`judge.py` 只产出 `Judgment`，`gate()` 是纯函数，`CustomerServiceAgent.handle()` = judge → gate → `respond()`。评测直接调 `judge` + `respond`，各阈值的结果离线计算。

**Tech Stack:** Python ≥3.9、anthropic SDK（DeepSeek Anthropic 兼容接口）、httpx（直连 Jev HTTP API）、python-dotenv、pytest。

## Global Constraints

- Python 3.9 兼容：每个模块 `from __future__ import annotations`，不用 `match`、`zip(strict=)`、运行时 `X | Y`。
- 不新增依赖（anthropic 自带 httpx）。
- 没有任何 API key 时也能跑通全部测试（Mock 后端）。
- 默认 `CS_TODAY=2026-10-13`；判断层默认阈值 τ=0.5，敏感意图加 0.1。
- 判断层出错时一律转人工（fail-safe）。
- Mock 模式的报告必须醒目标注"仅验证流程，数字无实验意义"。

## 文件结构

| 文件 | 职责 | 对应 learn-claude-code |
|---|---|---|
| `cs_agent/config.py` | 加载 .env、路径、今天日期 | — |
| `cs_agent/store.py` | 订单/物流/库存数据 + 业务规则硬校验 | s02 工具后端 |
| `cs_agent/rag.py` | 政策文档分段 + bigram 检索 | s02（RAG 查政策） |
| `cs_agent/memory.py` | 顾客记忆（偏好 + 售后历史） | s09 |
| `cs_agent/permission.py` | 三道闸：硬规则 → 业务规则 → 顾客确认 | s03 |
| `cs_agent/session.py` | Session 数据类（顾客、store、memory、闸、日志、交接） | — |
| `cs_agent/tools.py` | 工具 schema + ToolBox 分发 | s02 |
| `cs_agent/llm.py` | AnthropicLLM（DeepSeek）+ MockLLM + Block | s01 |
| `cs_agent/loop.py` | `agent_loop`：while True 工具循环 | s01 |
| `cs_agent/subagent.py` | 售后专员子 Agent | s06 |
| `cs_agent/compact.py` | 历史压缩 | s08 |
| `cs_agent/judge.py` | 意图定义、Jev/LLM/Mock 判断、gate | **创新点** |
| `cs_agent/agent.py` | CustomerServiceAgent：judge→gate→respond | 主循环 |
| `chat.py` | 交互 CLI | — |
| `eval/dataset.jsonl` | 30 条标注对话 | — |
| `eval/metrics.py` / `eval/report.py` / `eval/run_eval.py` | 指标、报告、实验驱动 | — |

## Tasks

### Task 1: 脚手架 + 数据 + store（业务规则）
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `cs_agent/__init__.py`, `cs_agent/config.py`, `cs_agent/data/{orders,logistics,stock,customers}.json`, `cs_agent/data/policies.md`, `cs_agent/data/memory_seed/*.json`, `cs_agent/store.py`
- Test: `tests/test_store.py`
- Produces: `Store(data_dir=DATA_DIR, today=None)`，以及 `.get_order(cid, oid)->dict|None`、`.list_orders(cid)`、`.days_since_delivery(order)->int|None`、`.check_refund(cid, oid, reason_type)->(bool,str)`、`.do_refund(...)`、`.check_exchange(cid, oid, new_size)`、`.do_exchange(...)`、`.check_address_change(cid, oid)`、`.do_address_change(...)`、`.tracking(oid)`
- 测试要点：7 天无理由边界、定制品拒绝、质量问题 15 天、待发货可退、运输中不可退、已发货不可改地址、不能查别人的订单、换货无库存拒绝。

### Task 2: RAG 查政策
- Create: `cs_agent/rag.py`；Test: `tests/test_rag.py`
- Produces: `PolicyIndex.from_file(path)`、`.search(query, k=3)->list[Section]`（Section 含 title 和 body）
- 测试要点："运费险"→运费险段排第一；"定制"→定制商品段；空查询返回空。

### Task 3: 记忆 s09
- Create: `cs_agent/memory.py`；Test: `tests/test_memory.py`
- Produces: `MemoryStore(root, seed_dir)`、`.load(cid)`、`.remember_preference(cid, category, value)`、`.add_after_sales(cid, record)`、`.render(cid)->str`
- 测试要点：种子复制、偏好覆盖写、售后记录追加并持久化、空顾客 render 为 ""。

### Task 4: 权限三道闸 s03 + Session + ToolBox s02
- Create: `cs_agent/permission.py`, `cs_agent/session.py`, `cs_agent/tools.py`；Test: `tests/test_permission.py`, `tests/test_tools.py`
- Produces: `PermissionGate(store, confirm_fn)`、`.check(cid, tool, args)->(bool,str)`；`Session(customer_id, store, memory, permission)`；`ToolResult(ok, content)`；`ToolBox(session, by="main")`、`.schemas(names)`、`.run(name, args)->ToolResult`；`BASE_TOOL_NAMES`、`SPECIALIST_TOOL_NAMES`
- 测试要点：大额退款被硬规则拦截且不询问顾客；业务规则不过不询问顾客；顾客拒绝则不执行；成功后写入售后记忆；所有调用都记入 `session.tool_log`（含 ok 和 by）；transfer_to_human 设置 `session.handoff`。

### Task 5: LLM 封装 + agent_loop s01 + 子 Agent s06 + 压缩 s08
- Create: `cs_agent/llm.py`, `cs_agent/loop.py`, `cs_agent/subagent.py`, `cs_agent/compact.py`；Test: `tests/test_compact.py`, `tests/test_loop.py`
- Produces: `Block`、`LLMResponse`、`AnthropicLLM(model, base_url, temperature=0)`、`MockLLM()`，以上都有 `.create(system, messages, tools=None, tool_choice=None, max_tokens=None)`；`agent_loop(llm, system, messages, tools, dispatch, max_rounds)->LoopOutcome(text, exhausted)`；`run_specialist(llm, session, task)->str`；`Compactor(summarize, max_chars, keep_last)`、`.maybe_compact(messages)->list`
- 测试要点：压缩后首条为 user 摘要、保留段以 user 文本开头、tool_use/tool_result 不被拆开、未超限时原样返回；循环在无 tool_use 时停止、超过轮数时 exhausted=True。

### Task 6: 判断层（创新点）
- Create: `cs_agent/judge.py`；Test: `tests/test_judge.py`
- Produces: `INTENTS`（有序 dict：name→英文描述+中文关键词）、`INTENT_ZH`、`HUMAN_INTENTS`、`SENSITIVE_INTENTS`、`Judgment`、`choice_confidence(probs)`、`build_state(transcript, latest)`、`JevJudge(api_key, model, transport=None)`、`LLMJudge(llm)`、`MockJudge()`，三者都有 `.judge(state)->Judgment`；`GatePolicy(threshold=0.5, sensitive_margin=0.1)`、`.effective_threshold(intent)`；`GateDecision(route, reason, threshold)`；`gate(judgment|None, policy)`
- 测试要点：置信度公式与 Jev 文档例子一致；Jev 请求体形状正确（httpx.MockTransport 断言）、能解析响应、429 后重试成功；LLMJudge 能解析 tool_use，也能退化为从文本里抓 JSON；gate 处理低置信、人工意图、敏感意图阈值、judgment=None 转人工。

### Task 7: CustomerServiceAgent + CLI
- Create: `cs_agent/agent.py`, `chat.py`；Test: `tests/test_agent.py`
- Produces: `TurnResult(reply, route, reason, judgment, gate, tools)`、`CustomerServiceAgent(llm, session, judge=None, policy=GatePolicy(), compactor=None)`、`.preload_history(pairs)`、`.handle(text)->TurnResult`、`.respond(text, judgment=None)->TurnResult`、`build_agent(customer_id, judge_backend, mock, threshold, confirm_fn, memory_root)`
- 测试要点（Mock）：低置信 → 转人工且不调 LLM；"转人工" → human_intent；查物流 → 调用 query_logistics、route=agent；judge 抛异常 → judge_error 转人工；respond 会把判断层提示注入 system。

### Task 8: 数据集 + 指标 + 报告 + 实验驱动
- Create: `eval/__init__.py`, `eval/dataset.jsonl`, `eval/metrics.py`, `eval/report.py`, `eval/run_eval.py`；Test: `tests/test_dataset.py`, `tests/test_metrics.py`
- Produces: `load_dataset(path)`、`case_correct(case, route, tools)->(route_ok, e2e_ok)`、`materialize(record, case, tau, policy_margin)->row`、`summarize(rows)->dict`、`calibration(rows)`、`render_report(...)->str`；CLI 参数 `--arms --thresholds --tau --workers --mock --limit --cases --out`
- 测试要点：数据集共 30 条、意图和工具名合法、引用订单属于该顾客、history 角色交替；指标在构造样例上数值正确；`python -m eval.run_eval --mock` 能端到端产出报告。

### Task 9: README + 全量验证
- Create: `README.md`
- 运行：`pytest -q` 全绿；`python -m eval.run_eval --mock --workers 4` 产出报告；`printf ... | python chat.py --mock` 冒烟。
