# 电商售后客服 Agent · 置信度驱动转人工

以 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的 Agent 骨架为基础，做了一个电商售后客服 Agent。

**创新点：** 每轮对话先用 [TypeSafe Jev](https://docs.typesafe.ai/introduction) 做一次结构化判断，同一次调用并行回答两道题：「意图 + 0~1 置信度」和「是否需要人工判断（0~1 概率）」。由代码门控：置信度低于阈值、意图本身需要人工、或需要人工判断时，直接转人工，LLM 根本不会被调用。项目还附带 30 条标注对话，三组对比「无判断层 / Jev 判断层 / LLM 自报置信度判断层」的准确率。

```
顾客消息
  └─► 判断层 judge(state) ──► Judgment{intent, confidence, needs_human}      ← Jev / DeepSeek / Mock
        └─► gate()（纯代码）
              ├─ 置信度 < τ（退款/换货/改地址用 τ+0.1），或意图 ∈ {要求人工, 投诉升级}，或 needs_human ≥ 0.5
              │      └─► 转人工：生成工单（意图、置信度、最近对话），LLM 不运行
              └─ 否则 ─► 主 Agent 循环（DeepSeek），system 中注入"[判断层] 意图=…, 置信度=…"
                         ├ s02 工具：查订单 / 查物流 / 登记催件 / 查政策(RAG) / 记偏好 / transfer_to_human
                         ├ s06 售后专员子 Agent：退货退款、换货
                         ├ s03 三道闸：硬规则 → 业务规则 → 顾客确认
                         ├ s08 长对话压缩
                         └ s09 顾客记忆（尺码偏好 + 售后历史）
```

## 和 learn-claude-code 的对应关系

| learn-claude-code | 在这里变成 | 文件 |
|---|---|---|
| s01 主循环 | 不变：`while True` 调 LLM → 执行工具 → 回填结果；加了 8 轮上限，超限转人工 | `cs_agent/loop.py` |
| s02 工具调用 | bash/读写文件 换成 查订单、查物流、登记催件、查售后政策（RAG，中文 bigram 检索） | `cs_agent/tools.py` `rag.py` `store.py` |
| s03 权限 | 退款/换货/改地址三道闸：①硬规则（退款 > ¥2000 不自动办）②业务规则（7 天无理由、15 天质量问题、定制品、已发货不能改地址，代码校验，LLM 绕不过）③**顾客**确认 y/N | `cs_agent/permission.py` |
| s06 子 Agent | 前台 Agent 负责接待，退换货交给「售后专员」子 Agent（独立 messages，只回传结论） | `cs_agent/subagent.py` |
| s08 上下文压缩 | 超长后把早期对话总结成摘要，切点保证不拆开 tool_use/tool_result，原文存 `.transcripts/` | `cs_agent/compact.py` |
| s09 记忆 | 每位顾客一个 JSON：尺码偏好（Agent 用工具写）+ 售后历史（操作成功后由代码写，不会被编造） | `cs_agent/memory.py` |
| — | **判断层（创新点）** | `cs_agent/judge.py` |

## 判断层怎么做

Jev 是 TypeSafe AI 在 2026-09 发布的判别式 "System One" 模型。它不生成文本，你给它 `state` 和类型化的问题，它返回答案、每个选项的概率和**校准过的**置信度。我们每轮在同一次调用里问两道题，Jev 并行判断，不增加延迟：

```json
POST https://api.typesafe.ai/v1/systemone
{
  "model": "jev-1.13.0",
  "state": {"recent_conversation": [...最近 6 条...], "latest_customer_message": "那双鞋挤脚，想换大一码"},
  "questions": {"intent": {"type": "choice",
    "instructions": "What is the customer's primary after-sales intent in `latest_customer_message`? ...",
    "criteria": {"order_status": "...", "logistics_query": "...", "return_refund": "...", "exchange": "...",
                 "address_change": "...", "policy_question": "...", "request_human": "...",
                 "complaint_escalation": "...", "out_of_scope": "..."}},
    "needs_human": {"type": "noul",
      "instructions": "Does handling `latest_customer_message` need a human agent's judgment ...?",
      "criteria": {"true": "several requests mixed / customer undecided / cases the rules reserve for humans ...",
                   "false": "one clear standard request ... stays false even when the rules may refuse it"}}}
}
→ {"answers": {"intent": {"choice": "exchange", "confidence": 0.86, "probabilities": {...}},
               "needs_human": {"noul": 0.08}}}
```

- **置信度定义（Jev 官方）：** `confidence = (p_max − 1/n) / (1 − 1/n)`。概率平均分布时为 0，全部压在一个选项上时为 1。
- **为什么要第二道题：** v1 实验发现，意图置信度衡量的是「属于哪个意图」有多确定，而不是「机器人能不能办」。例如顾客说「洗了一次掉色，你说怎么办」，意图（退换）很清楚，但是否算质量问题需要人判断。`needs_human` 的判断标准取自政策文档里「需人工处理」的条款。
- **门控在代码里，不在模型里**（`gate()` 是纯函数）。Jev 文档建议「阈值随风险升高」，所以敏感意图（退款/换货/改地址）的阈值自动 +0.1。
- **判断层出错（网络、限流、格式异常）一律转人工**，即 fail-safe。
- 有三个可替换的后端：`JevJudge`（HTTP 直连，429/529 指数退避重试）、`LLMJudge`（DeepSeek 用 tool_use 报告自己的置信度和 needs_human，定义与 Jev 完全相同，作为对照组）、`MockJudge`（离线）。
- 没有用 Jev 的官方 Python SDK，因为它要求 Python ≥ 3.10。这里直接走 HTTP，所以 3.9 也能跑。

## 快速开始

```bash
python3 -m pip install -r requirements.txt
```

```bash
cp .env.example .env
```

在 `.env` 中填入 DeepSeek key（`ANTHROPIC_API_KEY`）和 TypeSafe key（`TYPESAFE_API_KEY`），然后运行：

```bash
python3 chat.py --customer C001
```

其他用法：

```bash
python3 chat.py --customer C002 --judge none
```

```bash
python3 chat.py --customer C004 --judge llm --threshold 0.6
```

不需要任何 key 的离线演示（关键词规则，回答很蠢，只用来看流程）：

```bash
python3 chat.py --mock
```

内置 10 位顾客（C001–C010）和 20 个订单（C007–C010 是给留出测试集准备的新订单），数据在 `cs_agent/data/`。"今天"固定为 2026-10-13，可用 `CS_TODAY` 修改。可以试试这些输入：
- C001：`连衣裙到哪了` · `那双运动鞋挤脚，想换大一码`（会用记忆里的 37 码）· `A1002 地址改到杭州`（已发货，会被规则拒绝）
- C004：`A4001 电脑不想要了退款`（¥6999，硬规则要求人工）· `洗面奶用了过敏`（投诉，转人工）

## 网页演示

在浏览器里以顾客身份聊天，每一轮都能实时看到判断层在做什么：

```bash
python3 -m web.server --mock
```

不需要任何 key（关键词规则，回答没有意义，只用来看流程）。配好 `.env` 后去掉 `--mock`，就是真实的 Jev + DeepSeek。然后打开 http://127.0.0.1:8000 。

- **左栏：** 选顾客、判断层（Jev / LLM 自报 / 无判断层）、置信度阈值；下面是该顾客的订单和记忆，还有几条示例句，点一下就发送。
- **对话流：** 每轮按真实发生的顺序出现：判断层卡片（意图、置信度条、需人工条，带阈值竖线和耗时）→ 门控去向 → 工具调用逐条出现（前台 / 专员，✓ / ✗）→ 退款等敏感操作弹出确认条 → 回复（带本轮总耗时）。
- **对比演示：** 同一句话切换判断层分别发一次，能直观看到差别：直接转人工的对话 0.3 秒左右就有结果，交给大模型的对话要等十几秒。
- **实现：**
  - 给 Agent 加了事件钩子（learn-claude-code s04 的思路），每一步都会通过 `session.emit()` 报告出来。
  - 每轮对话在后台线程里跑，事件经队列用 SSE 流式推给浏览器。
  - 敏感操作需要确认时，后台线程暂停等待，直到网页调用 `/confirm` 接口；2 分钟没人回应就按取消处理。
  - 代码在 `web/turns.py` 和 `web/server.py`。前端是不需要构建的原生 HTML/JS，所有文字都用 `textContent` 写入，不会被当成 HTML 解析。
- 服务只监听本机 127.0.0.1，API key 不会离开你的电脑。

## 跑对比实验

```bash
python3 -m eval.run_eval
```

- 共 90 个任务（30 条 × 3 组），默认 4 个并发。判断层组每条只调一次判断层，阈值 τ ∈ {0.3…0.8} 全部离线计算。被门控挡下的样本不跑 Agent，结果和线上 `handle()` 完全一致。
- 报告里有 5 列：无判断层、Jev·仅意图、Jev·意图+需人工、LLM·仅意图、LLM·意图+需人工。「仅意图」和「意图+需人工」来自同一批判断结果，只是离线开关 needs_human 检查，所以是干净的消融对比。
- 结果写到 `eval/results/<时间戳>/`：`report.md`（报告）、`summary.json`（全部指标）、`results.jsonl`（每条原始记录，含回复和工具调用）。
- 常用参数：`--arms baseline,jev` 只跑部分组；`--cases T01,T19` 只跑部分样本；`--tau 0.6` 改主阈值；`--margin 0` 关闭敏感意图加严；`--needs-human 0.6` 改 needs_human 阈值；`--mock` 离线跑通流程（**数字没有意义**）。

**指标**（`eval/metrics.py`）：

| 指标 | 含义 |
|---|---|
| 路由准确率 | 机器人处理 / 转人工的去向是否和标注一致 |
| 端到端正确率 | 去向正确，并且：该调的工具都调了（`expect_tools`），违规操作没有真正执行（`forbid_success`；被权限闸拦下的尝试不算错） |
| 意图准确率 | 预测意图 ∈ 标注的可接受意图（多意图样本可接受多个） |
| 漏转率 ⚠ | 该转人工却被机器人处理，属于危险错误 |
| 误转率 | 机器人能处理却转了人工，浪费人力 |
| 自动处理正确率 | 机器人自己处理的那部分里做对的比例（选择性预测：覆盖率换准确率） |
| AUROC | 不依赖阈值：判断信号（意图置信度、needs_human）区分「该转人工」和「机器人能办」的能力。0.5 = 随机，1 = 完美 |
| 系统故障数 ⚠ | 判断层出错、输出截断、轮数超限。这些按 fail-safe 转人工，可能被算成「转对了」，所以单独列出，不为 0 时要谨慎解读 |
| 校准表 / ECE | 按置信度分桶的意图准确率。用来检验「置信度高 ⇒ 更准」是否成立 |

**测试集**（`eval/dataset.jsonl`，30 条，其中 4 条是多轮对话）：

| 类别 | 条数 | 应去向 | 例子 |
|---|---|---|---|
| 标准可自动处理 | 14 | 机器人 | 查物流、7 天无理由退货、换尺码（含依赖记忆/上文的）、改地址、政策咨询、闲聊 |
| 敏感操作陷阱 | 3 | 机器人（必须拒绝执行） | 签收 23 天要无理由退、已发货改地址、定制品退货 |
| 大额退款 | 1 | 人工 | ¥6999 退款，硬规则 |
| 明确需人工 | 6 | 人工 | 「转人工」「找经理」、过敏索赔、12315/微博曝光、律师 |
| 模糊/超政策/多意图 | 6 | 人工 | 超过 15 天的质量问题、直播间保价、运输损坏要补偿、一句话里好几个诉求 |

**数据集 v2 修订**（第一次真实实验后，对所有组影响相同，原因也写在各条的 `note` 里）：
- T09：原顾客 C002 的记忆里有一笔未答复的旧退款，Agent 办完退款后会主动转人工跟进旧问题（行为合理），干扰了本条要测的「未发货订单退款」，改用没有历史纠纷的 C006。
- T14：新增 `register_urge`（登记催件）工具，政策说物流异常可直接登记催件，本条现在要求调用它。
- T16：政策写明「已发货改地址请顾客自行联系快递改派，不需要转人工」，原来的表述有歧义。

## 留出测试集

原来的 30 条已经被三组系统全部做对，而且 needs_human 的标准是看过它们之后写的。要真正检验判断层，需要一份系统没见过的新测试集。

- 编写指南：[eval/holdout/编写指南.md](eval/holdout/编写指南.md)，包含写作纪律、订单表、配比、标注规则和自查清单。
- 模板：[eval/holdout/template.csv](eval/holdout/template.csv)，可直接用 Excel 或 Numbers 填写。
- 工具：`python3 -m eval.dataset_tools`
  - `convert`：校验并转换
  - `blind`：生成给第二位标注者的盲标副本
  - `agree`：计算两人标注一致性（Cohen's kappa）
- 运行：`python3 -m eval.run_eval --dataset eval/holdout/dataset.jsonl --out eval/results/holdout-v1`

## 测试

```bash
python3 -m pytest -q
```

共 171 个测试，全部离线，不需要 key。覆盖业务规则边界、权限三道闸、RAG、记忆、压缩切点、置信度公式（对照 Jev 文档的例子）、Jev 请求格式与重试、门控、Agent 端到端，以及用真实 anthropic SDK 加假 HTTP 服务器验证的请求格式、指标计算和数据集合法性。

## 目录

```
chat.py                 交互 CLI
web/
  server.py             FastAPI 接口：配置、顾客、会话、SSE 消息流、确认
  turns.py              每个会话的后台线程、事件队列、确认等待
  static/               前端页面（index.html / style.css / app.js）
cs_agent/
  judge.py              判断层：意图定义、Jev / LLM / Mock 后端、gate()
  agent.py              CustomerServiceAgent：handle() = judge → gate → respond()
  loop.py               s01 agent_loop
  tools.py              s02 工具定义与分发
  rag.py                s02 政策检索
  store.py              订单/物流/库存 + 业务规则
  permission.py         s03 三道闸
  subagent.py           s06 售后专员
  compact.py            s08 压缩
  memory.py             s09 顾客记忆
  llm.py                DeepSeek（Anthropic 兼容接口）+ MockLLM
  data/                 订单、物流、库存、政策文档、记忆种子
eval/
  dataset.jsonl         30 条标注对话
  run_eval.py           实验驱动
  metrics.py report.py  指标与报告
docs/superpowers/       设计文档与实现计划
tests/
```

## 注意事项

- **Jev 以英文为主要训练语言**，官方文档注明中文准确率较低。意图描述已写成「英文 + 中文关键词」来缓解。这一点本身也是值得在报告里讨论的发现。
- Jev 文档提到它**可能偏向排在前面的选项**。意图顺序固定在 `INTENTS` 中，换顺序后需要重新标定阈值。`jev-latest` 是会移动的别名，做实验请固定 `JEV_MODEL=jev-1.13.0`。
- **样本量 30**，1 条约等于 3.3 个百分点。不要在同一个测试集上挑最优阈值再报告这个阈值下的成绩（会高估）。报告的主结果用事先设定的 τ=0.5，阈值扫描只作为权衡曲线展示。
- DeepSeek 的 Anthropic 兼容接口支持 tools、tool_choice 和 temperature。如果接口拒绝强制 tool_choice，`LLMJudge` 会自动退回不强制的模式重试。
