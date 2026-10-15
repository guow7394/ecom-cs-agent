# 网页演示前端 — 设计文档

日期：2026-10-15 状态：待审

## 1. 目标与范围

**用途：** 本地演示和录屏（面试共享屏幕、README 里的 GIF）。API key 只留在本机，不部署上线。

**内容：** 客服对话 + 判断层实时可视化。

**成功标准：**
1. 一条命令启动：`python3 -m web.server`（可加 `--mock`），浏览器打开 `http://127.0.0.1:8000`。
2. 能选顾客、选判断层（Jev / LLM 自报 / 无）、调置信度阈值，然后以顾客身份聊天。
3. 每轮按真实发生顺序实时出现：判断层结果（含耗时）→ 门控去向 → 工具调用逐条出现 → 敏感操作弹出确认 → 回复（含本轮总耗时）。
4. 没有 key 也能用 Mock 模式演示流程，页面醒目标出 MOCK。
5. 原有 142 个测试不受影响。

**不做（YAGNI）：** 三组同屏对比、实验结果看板、部署上线、登录、会话持久化、多人同时使用。

**假设：** 只有本机一个人用。会话存在内存里，服务重启即清空。每个新会话的订单和顾客记忆都重置为初始数据。

## 2. 架构

```
浏览器  web/static/{index.html, style.css, app.js}
   │  GET  /api/config                     运行模式、可用的判断层、默认阈值
   │  GET  /api/customers                  顾客、订单、记忆（左栏）
   │  POST /api/sessions                   新建会话
   │  POST /api/sessions/{id}/messages     发一句话 → SSE 流式返回本轮事件
   │  POST /api/sessions/{id}/confirm      回答敏感操作的确认请求
   ▼
web/server.py    FastAPI：路由、静态文件、启动入口；只监听 127.0.0.1
web/turns.py     WebSession：每个会话的 Agent、事件队列、后台线程、确认等待
   ▼
cs_agent（现有）  通过事件钩子报告每一步
```

### 2.1 事件钩子（对现有代码的改动）

借鉴 learn-claude-code s04 hooks 的思路：Agent 每做一步就报告一次。不传回调时，行为和现在完全一样。

- `cs_agent/session.py`：`Session` 增加字段 `on_event: Callable[[str, dict], None]`，默认是空函数；增加方法 `emit(type, **data)`，内部调用 `on_event(type, data)`。
- `cs_agent/tools.py`：`ToolBox.run` 在执行前发 `tool_start`（name、by、input），执行后发 `tool_end`（name、by、ok、output 前 300 字）。主 Agent 调用 `after_sales_specialist` 时同样发这两个事件。
- `cs_agent/agent.py`：`handle()` 拿到判断结果后发 `judgment`，门控之后发 `gate`。判断层出错时，`judgment` 事件带 `error` 字段。

## 3. API

**GET /api/config** → `{"mock": bool, "judges": {"jev": bool, "llm": bool, "none": bool}, "default_threshold": 0.5, "needs_human_threshold": 0.5}`
- `judges` 表示哪些选项可用。非 Mock 模式下，`jev` 需要 TYPESAFE_API_KEY，`llm` 和 `none` 需要 ANTHROPIC_API_KEY。前端把不可用的选项置灰。

**GET /api/customers** → 列表，每项为 `{id, name, level, memory, orders: [{order_id, item, size, amount, status, note}]}`。`note` 由代码生成，例如"已签收 · 第 4 天""运输中""物流 5 天未更新""定制商品"。

**POST /api/sessions** `{customer_id, judge, threshold}` → `{"session_id": "..."}`
- 校验：顾客必须存在；judge ∈ {jev, llm, none} 且当前可用；threshold ∈ [0.3, 0.8]。不合法返回 400。

**POST /api/sessions/{id}/messages** `{text}` → `text/event-stream`
- 404：会话不存在；409：上一轮还没跑完，或会话已转人工结束；400：text 为空。
- 每个事件是一行 `data: <json>`，后跟一个空行：

| type | 字段 | 何时发出 |
|---|---|---|
| `judgment` | intent, intent_zh, confidence, needs_human, latency_ms, backend；出错时为 error | 判断层返回后（无判断层时不发） |
| `gate` | route, reason, threshold, needs_human_threshold | 门控决定后（无判断层时不发） |
| `tool_start` | name, by, input | 每个工具开始前 |
| `tool_end` | name, by, ok, output | 每个工具结束后 |
| `confirm_request` | request_id, text | 敏感操作需要顾客确认 |
| `confirm_resolved` | request_id, approved, timed_out | 确认有了结果 |
| `reply` | text, route, reason, ticket_id | 本轮回复 |
| `error` | message | 本轮出现未处理的异常 |
| `done` | elapsed_ms, ended | 本轮结束；ended=true 表示会话已转人工，不能再继续聊 |

**POST /api/sessions/{id}/confirm** `{request_id, approve}` → 204。没有对应的待确认请求时返回 404。

## 4. 并发与确认（`web/turns.py`）

`WebSession` 负责一个会话：
- 持有 `CustomerServiceAgent`、一把锁（同一时间只跑一轮）、`ended` 标志、待确认请求表。
- `start_turn(text)` 返回一个事件迭代器。这一轮在后台守护线程里跑，事件放进 `queue.Queue`，迭代器逐个取出，直到 `done`。
- 确认函数（作为 `confirm_fn` 传给 Agent）：生成 request_id，发 `confirm_request`，然后在 `threading.Event` 上最多等待 `CONFIRM_TIMEOUT=120` 秒；拿到结果后发 `confirm_resolved` 并返回。超时视为拒绝。
- `resolve(request_id, approve)`：设置对应的 Event，未找到时返回 False。
- 本轮结束后，如果 `session.handoff` 不为空，就把 `ended` 设为 True。
- 浏览器中途断开：后台线程会继续跑完；还在等的确认会超时，按拒绝处理。
- 每个会话的记忆目录是一个临时目录，从 `memory_seed` 复制而来，保证每个新会话都是初始数据。

会话表 `SessionRegistry` 是一个内存字典 `session_id → WebSession`。

## 5. 前端（`web/static/`，原生 HTML/CSS/JS，无构建）

**布局（按评审过的草图）：**
- **左栏：** 标题；顾客下拉框；判断层下拉框；阈值滑块 0.3–0.8（步长 0.1）；"新会话"按钮；当前顾客的订单列表和记忆；"试试这些"示例句（通用句子，比如"我的快递到哪了""想换大一码""不想要了，给我退款""运费险怎么赔""转人工！"），点击即发送。
- **右侧对话流：** 顾客消息气泡靠右。之后依次是：
  - 判断层卡片：去向标签和耗时；意图标签；置信度条和需人工条，各带一根阈值竖线。
  - 工具步骤列表：前台或专员、工具名、✓/✗/进行中、输出摘要。
  - 确认条：带"取消"和"确认"按钮，处理后变为结果文字。
  - 客服回复气泡：附本轮总耗时。
- **转人工：** 卡片标签变为红色"转人工"并写明原因；出现工单横幅；输入框锁定，提示开新会话。
- **其他：** 顶部在 Mock 模式下显示"MOCK 模式"标签。修改顾客、判断层或阈值时，自动新建会话并清空对话。

**实现要点：**
- 用 `fetch` 加 `ReadableStream` 解析 SSE（因为接口是 POST，不能用 EventSource）。
- 所有来自用户和模型的文字都用 `textContent` 写入页面，不用 `innerHTML`，防止注入。
- 一轮进行中时，输入框和示例句处于等待状态。
- 支持浅色和深色模式（`prefers-color-scheme`），颜色用 CSS 变量定义。

## 6. 出错处理

- **判断层出错：** 后端按 fail-safe 转人工，`judgment` 事件带 error，卡片显示原因。
- **大模型或其他异常：** 发 `error` 事件，对话里显示红色错误条；然后发 `done`（ended=false），会话可以继续用。
- **确认超时：** 按拒绝处理，`confirm_resolved.timed_out=true`，确认条显示"已超时取消"。
- **网络断开或页面刷新：** 当前会话丢失，开新会话即可。
- **请求不合法：** 返回 400/404/409，前端在输入框下方显示一行提示。

## 7. 测试

- **钩子单元测试**（`tests/test_hooks.py`）：
  - Session 默认的 emit 不报错。
  - ToolBox 对成功和失败的工具都会发出成对的 tool_start/tool_end。
  - Agent 在有判断层时发出 judgment 和 gate，在判断层出错时发出带 error 的 judgment。
- **WebSession 测试**（`tests/test_web_turns.py`，Mock 模式，不走 HTTP）：
  - 事件顺序是 judgment → gate → tool_* → reply → done。
  - 确认流程：用另一个线程调用 resolve，这一轮能继续，并发出 confirm_resolved。
  - 超时时按拒绝处理（测试里把超时改成 0.1 秒）。
  - 转人工后 ended=true，再发消息被拒绝。
  - 上一轮没结束时再发会被拒绝。
- **HTTP 测试**（`tests/test_web_api.py`，FastAPI TestClient，Mock 模式）：
  - config 和 customers 接口正常。
  - 新建会话的参数校验。
  - 发消息后的 SSE 事件解析正确。
  - 404、409、400 三种错误码。
  - confirm 接口在没有待确认请求时返回 404。
- **手动验证：** 在内置浏览器里用 Mock 模式点一遍主流程并截图：查物流、换货确认、转人工、切换判断层。
- 原有 142 个测试必须全部通过。

## 8. 依赖

新增 `fastapi`、`uvicorn`，写入 `requirements.txt`，安装命令为 `python3 -m pip install fastapi uvicorn`。代码保持兼容 Python 3.9。
