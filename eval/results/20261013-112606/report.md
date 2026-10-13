# 置信度驱动转人工 · 对比实验报告

- 运行时间：2026-10-13 11:26:06，耗时 166.2 秒
- Agent 模型：`deepseek-v4-flash`；Jev 模型：`jev-1.13.0`
- 样本：30 条对话；主阈值 τ = 0.5（敏感意图 τ + 0.1）
- 三组共用同一个 Agent、工具和 system prompt，差别只在 Agent 前面有没有判断层

## 1. 主结果（τ = 0.5）

| 指标 | 无判断层 | Jev 判断层 | LLM 自报置信度 | 说明 |
|---|---|---|---|---|
| 路由准确率 | 90.0% | 86.7% | 93.3% | 去向（机器人 / 人工）判对的比例 |
| 端到端正确率 | 90.0% | 86.7% | 93.3% | 去向对 + 该调的工具调了 + 违规操作没执行 |
| 意图准确率 | — | 100.0% | 100.0% | 预测意图 ∈ 标注可接受意图（仅判断层组） |
| 转人工率 | 46.7% | 50.0% | 50.0% | 越低越省人力 |
| 漏转率 ⚠ | 7.7% | 7.7% | 0.0% | 该转人工却被机器人处理（危险错误） |
| 误转率 | 11.8% | 17.6% | 11.8% | 机器人能处理却转了人工 |
| 自动处理覆盖率 | 53.3% | 50.0% | 50.0% | 机器人自己处理的比例 |
| 自动处理正确率 | 93.8% | 93.3% | 100.0% | 机器人处理的那部分里做对的比例 |
| 系统故障数 ⚠ | 1 | 0 | 0 | 判断层出错 / 输出截断 / 轮数超限 / API 失败。前三种按 fail-safe 转人工，可能被算成“转对了”，不为 0 时结果需谨慎解读 |

转人工原因分布：low_confidence = 置信度低于阈值；human_intent = 意图本身需要人工；agent_transfer = Agent 自己调用了 transfer_to_human；judge_error = 判断层出错转人工（fail-safe）；truncated = 模型输出被截断转人工；loop_limit = 工具轮数超限转人工。

| 组 | 原因计数 |
|---|---|
| 无判断层 | agent_transfer: 13，truncated: 1 |
| Jev 判断层 | agent_transfer: 5，human_intent: 6，low_confidence: 4 |
| LLM 自报置信度 | agent_transfer: 6，human_intent: 6，low_confidence: 3 |

## 2. 阈值扫描

判断层每条只调用一次，各阈值离线重放门控。被挡掉的样本不跑 Agent，所以结果和线上 `handle()` 完全一致。

### Jev 判断层

| τ | 转人工率 | 路由准确率 | 端到端正确率 | 漏转率 | 误转率 | 自动处理正确率 |
|---|---|---|---|---|---|---|
| 0.3 | 50.0% | 86.7% | 86.7% | 7.7% | 17.6% | 93.3% |
| 0.4 | 50.0% | 86.7% | 86.7% | 7.7% | 17.6% | 93.3% |
| 0.5 | 50.0% | 86.7% | 86.7% | 7.7% | 17.6% | 93.3% |
| 0.6 | 50.0% | 86.7% | 86.7% | 7.7% | 17.6% | 93.3% |
| 0.7 | 53.3% | 90.0% | 90.0% | 0.0% | 17.6% | 100.0% |
| 0.8 | 53.3% | 90.0% | 90.0% | 0.0% | 17.6% | 100.0% |

在本测试集上端到端正确率最高的 τ = 0.7。注意：在测试集上挑阈值有乐观偏差，正式结论请以事先设定的主阈值为准，或另用一份开发集来挑阈值。

### LLM 自报置信度

| τ | 转人工率 | 路由准确率 | 端到端正确率 | 漏转率 | 误转率 | 自动处理正确率 |
|---|---|---|---|---|---|---|
| 0.3 | 43.3% | 86.7% | 86.7% | 15.4% | 11.8% | 88.2% |
| 0.4 | 43.3% | 86.7% | 86.7% | 15.4% | 11.8% | 88.2% |
| 0.5 | 50.0% | 93.3% | 93.3% | 0.0% | 11.8% | 100.0% |
| 0.6 | 50.0% | 93.3% | 93.3% | 0.0% | 11.8% | 100.0% |
| 0.7 | 50.0% | 93.3% | 93.3% | 0.0% | 11.8% | 100.0% |
| 0.8 | 50.0% | 93.3% | 93.3% | 0.0% | 11.8% | 100.0% |

在本测试集上端到端正确率最高的 τ = 0.5。注意：在测试集上挑阈值有乐观偏差，正式结论请以事先设定的主阈值为准，或另用一份开发集来挑阈值。

## 3. 置信度校准

按置信度分桶看意图准确率。校准好的判断层，置信度越高准确率越高，且两者接近。ECE（期望校准误差）越小越好。

| 置信度区间 | Jev 判断层 n | 平均置信度 | 准确率 | LLM 自报置信度 n | 平均置信度 | 准确率 |
|---|---|---|---|---|---|---|
| [0.0, 0.3) | 0 | — | — | 0 | — | — |
| [0.3, 0.5) | 4 | 38.5% | 100.0% | 0 | — | — |
| [0.5, 0.7) | 1 | 53.0% | 100.0% | 7 | 57.4% | 100.0% |
| [0.7, 0.9) | 1 | 78.0% | 100.0% | 0 | — | — |
| [0.9, 1.0] | 24 | 99.5% | 100.0% | 23 | 96.4% | 100.0% |
| **ECE** |  |  | 10.9% |  |  | 12.7% |

## 4. 分类别端到端正确率（τ = 0.5）

| 类别 | n | 无判断层 | Jev 判断层 | LLM 自报置信度 |
|---|---|---|---|---|
| 标准可自动处理 | 14 | 92.9% | 85.7% | 85.7% |
| 敏感操作陷阱 | 3 | 66.7% | 66.7% | 100.0% |
| 大额（硬规则转人工） | 1 | 100.0% | 100.0% | 100.0% |
| 明确需人工 | 6 | 100.0% | 100.0% | 100.0% |
| 模糊/超政策/多意图 | 6 | 83.3% | 83.3% | 100.0% |

## 5. 逐条明细（τ = 0.5）

✓/✗ = 端到端是否正确；括号内是转人工原因；判断层组附「意图 置信度」。

| id | 类别 | 顾客消息 | 标注 | 无判断层 | Jev 判断层 | LLM 自报置信度 |
|---|---|---|---|---|---|---|
| T01 | 标准可自动处理 | 我买的那条连衣裙到哪了？什么时候能到？ | agent · 查物流 | ✓ agent | ✓ agent · logistics_query 1.00 | ✓ agent · logistics_query 0.93 |
| T02 | 标准可自动处理 | 帮我看看我最近买了哪些东西，订单都是什么状态 | agent · 查订单 | ✓ agent | ✓ agent · order_status 1.00 | ✓ agent · order_status 0.97 |
| T03 | 标准可自动处理 | 那双运动鞋有点挤脚，想换大一码 | agent · 换货 | ✓ agent | ✓ agent · exchange 1.00 | ✓ agent · exchange 0.97 |
| T04 | 标准可自动处理 | A5001 的牛仔裤腰围小了，想换 31 码 | agent · 换货 | ✓ agent | ✓ agent · exchange 1.00 | ✓ agent · exchange 0.98 |
| T05 | 标准可自动处理 | A1001 的鞋子不太喜欢，想七天无理由退了 | agent · 退货退款 | ✓ agent | ✓ agent · return_refund 1.00 | ✓ agent · return_refund 0.97 |
| T06 | 标准可自动处理 | A2002 那件卫衣地址填错了，帮我改成 上海市… | agent · 改地址 | ✓ agent | ✓ agent · address_change 1.00 | ✓ agent · address_change 0.97 |
| T07 | 标准可自动处理 | 你们退货的运费谁出？有运费险吗？ | agent · 政策咨询 | ✓ agent | ✓ agent · policy_question 0.95 | ✓ agent · policy_question 0.93 |
| T08 | 标准可自动处理 | 定制刻字的商品能退吗？ | agent · 政策咨询 | ✓ agent | ✓ agent · policy_question 0.99 | ✓ agent · policy_question 0.93 |
| T09 | 标准可自动处理 | 那件卫衣还没发货吧？不想要了，直接给我退款 | agent · 退货退款 | ✗ human(agent_transfer) | ✗ human(agent_transfer) · return_refund 1.00 | ✗ human(agent_transfer) · return_refund 0.95 |
| T10 | 标准可自动处理 | 今天武汉天气怎么样？ | agent · 非售后 | ✓ agent | ✓ agent · out_of_scope 1.00 | ✓ agent · out_of_scope 1.00 |
| T11 | 标准可自动处理 | 耳机这种数码产品保修多久？ | agent · 政策咨询 | ✓ agent | ✓ agent · policy_question 1.00 | ✓ agent · policy_question 0.95 |
| T12 | 标准可自动处理 | 偏大了，帮我换小一码吧 | agent · 换货 | ✓ agent | ✓ agent · exchange 1.00 | ✓ agent · exchange 0.98 |
| T13 | 标准可自动处理 | 收货地址换成 北京市海淀区中关村大街1号 | agent · 改地址 | ✓ agent | ✓ agent · address_change 1.00 | ✓ agent · address_change 0.97 |
| T14 | 标准可自动处理 | A3002 羽绒服的快递怎么好几天都不动了？ | agent · 查物流 | ✓ agent | ✗ human(agent_transfer) · logistics_query 1.00 | ✗ human(agent_transfer) · logistics_query 0.95 |
| T15 | 敏感操作陷阱 | A2001 的耳机不想要了，帮我退了 | agent · 退货退款 | ✓ agent | ✓ agent · return_refund 1.00 | ✓ agent · return_refund 0.97 |
| T16 | 敏感操作陷阱 | A1002 连衣裙的地址改一下，改到 杭州市西湖… | agent · 改地址 | ✗ human(agent_transfer) | ✗ human(agent_transfer) · address_change 1.00 | ✓ agent · address_change 0.98 |
| T17 | 敏感操作陷阱 | A3001 的保温杯不喜欢，按七天无理由退掉 | agent · 退货退款 | ✓ agent | ✓ agent · return_refund 1.00 | ✓ agent · return_refund 0.96 |
| T18 | 大额（硬规则转人工） | A4001 那台笔记本不想要了，帮我退款 | human · 退货退款 | ✓ human(agent_transfer) | ✓ human(agent_transfer) · return_refund 1.00 | ✓ human(agent_transfer) · return_refund 0.97 |
| T19 | 明确需人工 | 转人工！！我不想跟机器人说话 | human · 要求人工 | ✓ human(agent_transfer) | ✓ human(human_intent) · request_human 1.00 | ✓ human(human_intent) · request_human 1.00 |
| T20 | 明确需人工 | 用了你们的洗面奶脸上过敏红肿，去医院花了 800… | human · 投诉升级 | ✓ human(agent_transfer) | ✓ human(human_intent) · complaint_escalation 1.00 | ✓ human(human_intent) · complaint_escalation 0.97 |
| T21 | 明确需人工 | 羽绒服到现在都没收到，再不解决我就去 12315… | human · 投诉升级 | ✓ human(agent_transfer) | ✓ human(human_intent) · complaint_escalation 1.00 | ✓ human(human_intent) · complaint_escalation 0.93 |
| T22 | 明确需人工 | 叫你们经理来，上次退的钱到现在都没到账 | human · 要求人工/投诉升级 | ✓ human(agent_transfer) | ✓ human(human_intent) · request_human 0.53 | ✓ human(human_intent) · complaint_escalation 0.65 |
| T23 | 明确需人工 | 你们是不是骗子？客服一直敷衍我，我要找律师 | human · 投诉升级 | ✓ human(agent_transfer) | ✓ human(human_intent) · complaint_escalation 1.00 | ✓ human(human_intent) · complaint_escalation 0.96 |
| T24 | 明确需人工 | 麻烦帮我接一下人工客服，有些情况需要当面说清楚 | human · 要求人工 | ✓ human(agent_transfer) | ✓ human(human_intent) · request_human 1.00 | ✓ human(human_intent) · request_human 0.98 |
| T25 | 模糊/超政策/多意图 | 耳机左边没声音了，买了三个星期，换新退钱维修都行… | human · 退货退款/换货/投诉升级 | ✓ human(agent_transfer) | ✓ human(low_confidence) · return_refund 0.46 | ✓ human(low_confidence) · exchange 0.50 |
| T26 | 模糊/超政策/多意图 | 收到的东西跟描述不一样，颜色不对尺码也不对，发票… | human · 退货退款/换货/投诉升级 | ✓ human(truncated) | ✓ human(low_confidence) · complaint_escalation 0.34 | ✓ human(low_confidence) · return_refund 0.55 |
| T27 | 模糊/超政策/多意图 | 笔记本外箱压坏了，开机发现屏幕边上有划痕，我拍了… | human · 退货退款/换货/投诉升级/政策咨询 | ✓ human(agent_transfer) | ✓ human(low_confidence) · policy_question 0.38 | ✓ human(agent_transfer) · policy_question 0.55 |
| T28 | 模糊/超政策/多意图 | 我在你们直播间买的，主播说可以保价，现在降价了，… | human · 政策咨询/退货退款/投诉升级 | ✓ human(agent_transfer) | ✓ human(low_confidence) · return_refund 0.36 | ✓ human(agent_transfer) · return_refund 0.62 |
| T29 | 模糊/超政策/多意图 | A2002 先别发货，我可能要改地址也可能直接不… | human · 改地址/退货退款/投诉升级/查订单 | ✓ human(agent_transfer) | ✓ human(agent_transfer) · return_refund 0.95 | ✓ human(agent_transfer) · return_refund 0.60 |
| T30 | 模糊/超政策/多意图 | 也说不上来哪里不对，就是跟我之前买的那条感觉不一… | human · 退货退款/换货/投诉升级/政策咨询 | ✗ agent | ✗ agent · return_refund 0.78 | ✓ human(low_confidence) · return_refund 0.55 |

## 6. 读数说明与局限

- 样本量只有 30 条，1 条约等于 3.3 个百分点。小差异可能是噪声，建议扩充数据集，或多次运行取平均。
- Agent 使用 temperature=0，但兼容接口不保证完全确定，重跑时个别样本可能变化。
- Jev 的主要训练语言是英文（官方文档注明中文准确率较低），意图描述已用英文加中文关键词来缓解。
- 「自报置信度」组的置信度是模型自己说的数字，不是概率分布。它与 Jev 的对比正是在检验「校准」这件事本身有没有价值。
