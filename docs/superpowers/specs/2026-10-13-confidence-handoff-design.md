# 电商售后客服 Agent：置信度驱动转人工 — 设计文档

日期：2026-10-13 状态：已确认（三组对比实验）

## 1. 目标

以 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的 s01/s02/s03/s06/s08/s09 为骨架，做一个电商售后客服 Agent。
创新点：**每轮先用 TypeSafe Jev 做一次结构化判断**（意图 Choice + 0~1 置信度），由代码门控：置信度低于阈值或意图属于"必须人工"类，直接转人工；否则交给 LLM Agent 处理。
用 30 条带标注的测试对话，对比三组：

| 组 | 判断层 | 说明 |
|---|---|---|
| `baseline` | 无 | Agent 自己决定是否调用 `transfer_to_human` |
| `jev` | Jev（判别式模型，返回校准概率） | 本文创新点 |
| `llm` | DeepSeek 自报置信度（强制 tool_use 输出 JSON） | 对照：证明提升来自"校准"而非"有门控" |

三组共用同一个 DeepSeek Agent、同一套工具、同一个 system prompt。差别只在 Agent 前面有没有判断层。

## 2. 每轮数据流

```
顾客消息
  └─► 判断层 judge(state) → Judgment{intent, confidence, probabilities}
        └─► gate(judgment, τ)
              ├─ human：confidence < τ_eff，或 intent ∈ {request_human, complaint_escalation}
              │        → 生成交接工单（意图、置信度、最近对话），回复"已转人工"
              └─ agent：把 "[判断层] 意图=X 置信度=Y" 注入 system，进入主 Agent 循环
                       ├ s02 工具：query_order / query_logistics / search_policy(RAG) /
                       │           remember_preference / transfer_to_human
                       ├ s06 after_sales_specialist 子 Agent（退货退款/换货）
                       ├ s03 三道闸：硬规则（大额>2000 拒绝自动执行）→ 业务规则（代码校验时效、
                       │           定制品、是否已发货）→ 顾客确认（y/N）
                       ├ s08 历史超长时把早期对话总结成摘要
                       └ s09 每位顾客一个 JSON 记忆（尺码偏好 + 售后历史）
```

`τ_eff = τ`；敏感意图（return_refund / exchange / address_change）`τ_eff = min(τ + 0.1, 0.95)`（Jev 文档建议"阈值随风险升高"）。

## 3. 意图集合（9 类）

| intent | 含义 | 门控默认去向 |
|---|---|---|
| order_status | 订单本身：有哪些订单、状态、金额、是否发货 | agent |
| logistics_query | 已发出包裹：到哪了、何时到、物流不动 | agent |
| return_refund | 退货 / 退款 / 取消订单 | agent（敏感） |
| exchange | 换尺码 / 颜色 / 换新 | agent（敏感） |
| address_change | 改收货地址 | agent（敏感） |
| policy_question | 咨询售后规则（无理由、运费险、保修、定制品） | agent |
| request_human | 明确要求人工 / 找经理 | human |
| complaint_escalation | 投诉升级：索赔、人身伤害、12315/律师/曝光 | human |
| out_of_scope | 闲聊、售前推荐等与售后无关 | agent（礼貌引导） |

## 4. 判断层（`cs_agent/judge.py`）

- `Judgment` 数据类：`intent, confidence, probabilities, backend, latency_ms, raw`。
- `build_state(...)`：最近 6 条对话（纯文本）+ 最新顾客消息。不放订单数据，因为 Jev 文档指出无关信息越多越不准，而意图判断用不到订单。
- `JevJudge`：`POST https://api.typesafe.ai/v1/systemone`，一道 Choice 题 `intent`。criteria 用英文描述 + 中文关键词（Jev 以英文为主要训练语言）。用 httpx 直连，对 429/529 指数退避重试。不用官方 SDK，因为它要求 Python ≥ 3.10。
- `LLMJudge`：DeepSeek，`tool_choice` 强制调用 `report_judgment{intent(enum), confidence, reason}`，temperature=0，同样的 state 与意图定义。
- `MockJudge`：关键词打分 → softmax → 概率，置信度按 Jev 公式 `(p_max − 1/n)/(1 − 1/n)` 计算。用于离线测试，数字无实验意义。
- `gate(judgment, GatePolicy) -> GateDecision{route, reason, threshold}`：纯函数。

## 5. 业务数据与规则（`cs_agent/data/`, `cs_agent/store.py`）

- 固定"今天"= `CS_TODAY`（默认 2026-10-13），保证实验可复现。
- 6 位顾客、12 个订单、物流轨迹、库存；`policies.md` 按 `##` 切段供 RAG 检索（中文 bigram 重叠打分，无额外依赖）。
- 代码硬校验（LLM 无法绕过）：
  - 退款：待发货 → 取消并退款；已签收 ≤7 天 → 无理由（定制品除外）；质量问题 ≤15 天；其余拒绝。
  - 换货：已签收 ≤15 天、非定制、目标尺码有库存。
  - 改地址：仅待发货。
  - 只能操作当前顾客自己的订单。
  - 金额 > 2000 的退款不允许自动执行（硬规则闸）。

## 6. 测试集（`eval/dataset.jsonl`，30 条）

| 类别 | 条数 | gold_route |
|---|---|---|
| standard 标准可自动处理（含多轮、依赖记忆） | 14 | agent |
| trap 敏感操作陷阱（超时效退款、已发货改地址、定制品退货） | 3 | agent（必须拒绝执行） |
| trap_human 大额退款（硬规则强制人工） | 1 | human |
| explicit_human 明确需人工（要求人工、投诉、过敏索赔、12315） | 6 | human |
| ambiguous 模糊/超政策/多意图混杂 | 6 | human |

字段：`id, category, customer_id, history[[role,text]], message, gold_intents[], gold_route, expect_tools[], forbid_success[], note`。

## 7. 指标（`eval/metrics.py`）

- 路由准确率：预测去向（agent/human）是否等于 gold_route。
- 端到端正确率：路由正确，并且当 gold=agent 时 `expect_tools` 全部被调用、`forbid_success` 中的工具没有执行成功。
- 意图准确率：仅判断层组有此指标，判断为预测意图 ∈ gold_intents。
- 转人工率、漏转率（gold=human 却被机器人处理，这是危险错误）、误转率（gold=agent 却转了人工）。
- 覆盖率与自动处理正确率（选择性预测）。
- 阈值扫描：τ ∈ {0.3…0.8}。判断层只调一次，各阈值离线计算。被门控挡掉的样本不需要跑 Agent。
- 校准表：按置信度分桶统计意图准确率。

输出：`eval/results/<时间戳>/{results.jsonl, summary.json, report.md}`。Mock 模式的报告页头会醒目标注"仅验证流程"。

## 8. 错误处理

- Jev / DeepSeek 调用失败时**判断层失败即转人工**（fail-safe），并记录 `reason=judge_error`。
- Agent 循环最多 8 轮工具调用，超出则转人工。
- 工具异常以字符串形式返回给模型，不中断对话。

## 9. 测试

pytest + 无网络假后端：门控逻辑、置信度公式、Jev/LLM 响应解析（httpx.MockTransport）、业务规则、确认闸、压缩边界（不拆开 tool_use/tool_result）、记忆持久化、Agent 端到端（Mock）、指标计算、数据集合法性。

## 10. 不做（YAGNI）

Web UI、真实数据库、多语言、向量数据库、多次重复采样统计方差（temperature=0，留作扩展）。

## 11. v2 变更（第一次真实实验之后）

- **修复：** DeepSeek v4 默认开启思考模式，并且拒绝强制 tool_choice。原先判断层 `max_tokens=400`、Agent `max_tokens=2048`，模糊样本会被截断。截断原先被当作成功或「判断层出错转人工」，抬高了准确率。现在预算分别是 4096 和 8192，截断被显式记为故障并 fail-safe 转人工，报告单列「系统故障数」。
- **第二道判断题 needs_human（Noul）：** 和意图 Choice 在同一次 Jev 调用里并行判断。gate 新增规则：needs_human ≥ 0.5 → 转人工（reason=needs_judgment）。LLM 对照组用完全相同的定义输出 needs_human。
- **消融：** 报告对每个判断层后端给出「仅意图」和「意图+需人工」两列，由同一批判断结果离线重放。
- **新工具 register_urge：** 物流 72 小时无更新时登记催件（代码校验时间）。
- **数据集 v2：** 修订 T09（换成无历史纠纷的顾客）、T14（要求登记催件）、T16（政策写明不需要转人工）。
- **局限：** needs_human 的判断标准是看过 v1 结果后加的，这份数据集对它已不是严格的留出集，需要新样本复核。
