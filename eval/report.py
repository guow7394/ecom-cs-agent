"""Render the experiment results as a Markdown report (Chinese)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

from cs_agent.judge import INTENT_ZH

ARM_LABEL = {"baseline": "无判断层", "jev": "Jev·仅意图", "jev+nh": "Jev·意图+需人工",
             "llm": "LLM·仅意图", "llm+nh": "LLM·意图+需人工",
             "fusion": "三路融合·仅意图", "fusion+nh": "三路融合·意图+需人工"}
BACKEND_LABEL = {"jev": "Jev", "llm": "LLM 自报", "fusion": "三路融合"}
METRICS = [
    ("route_acc", "路由准确率", "去向（机器人 / 人工）判对的比例"),
    ("e2e_acc", "端到端正确率", "去向对 + 该调的工具调了 + 违规操作没执行"),
    ("intent_acc", "意图准确率", "预测意图 ∈ 标注可接受意图（仅判断层组）"),
    ("handoff_rate", "转人工率", "越低越省人力"),
    ("missed_handoff", "漏转率 ⚠", "该转人工却被机器人处理（危险错误）"),
    ("false_handoff", "误转率", "机器人能处理却转了人工"),
    ("coverage", "自动处理覆盖率", "机器人自己处理的比例"),
    ("auto_acc", "自动处理正确率", "机器人处理的那部分里做对的比例"),
]
SWEEP_METRICS = ["handoff_rate", "route_acc", "e2e_acc", "missed_handoff", "false_handoff", "auto_acc"]
CATEGORY_LABEL = OrderedDict([("standard", "标准可自动处理"), ("trap", "敏感操作陷阱"),
                              ("trap_human", "大额（硬规则转人工）"), ("explicit_human", "明确需人工"),
                              ("ambiguous", "模糊/超政策/多意图"), ("adversarial", "对抗/诱导")])
REASONS_NOTE = ("转人工原因：low_confidence = 意图置信度低于阈值；human_intent = 意图本身需要人工；"
                "needs_judgment = 第二道题判断需要人工；agent_transfer = Agent 自己调用了 transfer_to_human；"
                "judge_error / truncated / loop_limit = 系统故障后按 fail-safe 转人工。")


def pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def _cell(row: dict) -> str:
    mark = "✓" if row["e2e_ok"] else "✗"
    route = row["route"] if not row["reason"] else f"{row['route']}({row['reason']})"
    if row["intent"] is None:
        return f"{mark} {route}"
    nh = f" / 需人工 {row['needs_human']:.2f}" if row["needs_human"] is not None else ""
    return f"{mark} {route} · {row['intent']} {row['confidence']:.2f}{nh}"


def _short(text: str, n: int = 24) -> str:
    return text[:n] + ("…" if len(text) > n else "")


def best_tau(sweep: Dict[str, dict]) -> str:
    # max end-to-end accuracy, ties broken by fewer handoffs
    return max(sweep, key=lambda t: (sweep[t]["e2e_acc"] or 0, -(sweep[t]["handoff_rate"] or 0)))


def render_report(meta: dict, summary: Dict[str, Dict[str, dict]], calib: Dict[str, tuple],
                  sep: Dict[str, dict], rows_main: Dict[str, List[dict]], cases: List[dict]) -> str:
    variants = meta["variants"]
    backends = [b for b in ("jev", "llm", "fusion") if b in variants]
    full = [v for v in variants if v == "baseline" or v.endswith("+nh")]
    tau = meta["tau_key"]
    out = ["# 置信度驱动转人工 · 对比实验报告", ""]
    if meta["mock"]:
        out += ["> ⚠️ **MOCK 模式**：Agent 和判断层都是关键词规则，**这些数字只用来验证流程能跑通，没有任何实验意义。**",
                "> 配好 `.env` 后去掉 `--mock` 重跑，才是真实结果。", ""]
    out += [f"- 运行时间：{meta['started_at']}，耗时 {meta['elapsed_s']} 秒",
            f"- Agent 模型：`{meta['model_id']}`；Jev 模型：`{meta['jev_model']}`",
            f"- 数据集：`{meta['dataset']}`" + ("（留出测试集）" if meta["holdout"] else ""),
            f"- 样本：{meta['n_cases']} 条对话；主阈值 τ = {meta['tau']}（敏感意图 τ + {meta['margin']}）；"
            f"needs_human 阈值 = {meta['needs_human_threshold']}",
            "- 所有组共用同一个 Agent、工具和 system prompt，差别只在 Agent 前面的判断层。"
            "「仅意图」和「意图+需人工」由同一批判断结果离线重放，是干净的消融对比",
            ""]

    out += [f"## 1. 主结果（τ = {meta['tau']}）", ""]
    header = ["指标"] + [ARM_LABEL[v] for v in variants] + ["说明"]
    body = [[label] + [pct(summary[v][tau][key]) for v in variants] + [note] for key, label, note in METRICS]
    body.append(["系统故障数 ⚠"] + [str(summary[v][tau]["failures"]) for v in variants]
                + ["判断层出错 / 输出截断 / 轮数超限 / API 失败。前三种按 fail-safe 转人工，可能被算成“转对了”，不为 0 时需谨慎解读"])
    out += _table(header, body) + ["", REASONS_NOTE, ""]
    out += _table(["组", "转人工原因计数"],
                  [[ARM_LABEL[v], "，".join(f"{k}: {n}" for k, n in sorted(summary[v][tau]["reasons"].items())) or "—"]
                   for v in variants]) + [""]

    if backends:
        out += ["## 2. 第二道题（needs_human）改变了哪些样本", "",
                "同一批判断结果，只关掉/打开 needs_human 检查，列出去向发生变化的样本。", ""]
        for b in backends:
            before = {r["case_id"]: r for r in rows_main[b]}
            after = {r["case_id"]: r for r in rows_main[b + "+nh"]}
            changed = [c for c in cases if before[c["id"]]["route"] != after[c["id"]]["route"]]
            out += [f"### {BACKEND_LABEL[b]}：{len(changed)} 条变化", ""]
            if changed:
                out += _table(["id", "顾客消息", "标注去向", "needs_human", "仅意图", "意图+需人工"],
                              [[c["id"], _short(c["message"]), c["gold_route"], num(after[c["id"]]["needs_human"]),
                                _cell(before[c["id"]]), _cell(after[c["id"]])] for c in changed])
            out += [""]

        out += ["## 3. 判断信号的区分能力（AUROC，不依赖阈值）", "",
                "只看意图本身不强制转人工的样本：这些样本的去向完全取决于判断信号。"
                "AUROC 0.5 = 随机猜，1.0 = 完美区分「该转人工」与「机器人能办」。样本少，差异需谨慎解读。", ""]
        out += _table(["判断层", "样本（应人工 / 能办）", "意图置信度 AUROC", "needs_human AUROC",
                       "needs_human 均值（应人工）", "needs_human 均值（能办）"],
                      [[BACKEND_LABEL[b], f"{sep[b]['n_human']} / {sep[b]['n_agent']}",
                        num(sep[b]["auroc_confidence"]), num(sep[b]["auroc_needs_human"]),
                        num(sep[b]["mean_needs_human_human"]), num(sep[b]["mean_needs_human_agent"])]
                       for b in backends]) + [""]

        out += ["## 4. 阈值扫描（意图置信度阈值 τ）", "",
                "判断层每条只调用一次，各阈值离线重放门控。被挡掉的样本不跑 Agent，结果和线上 `handle()` 一致。", ""]
        for v in [v for v in variants if v != "baseline"]:
            sweep = summary[v]
            out += [f"### {ARM_LABEL[v]}", ""]
            out += _table(["τ", "转人工率", "路由准确率", "端到端正确率", "漏转率", "误转率", "自动处理正确率"],
                          [[t] + [pct(sweep[t][m]) for m in SWEEP_METRICS] for t in sweep])
            out += ["", f"本测试集上端到端正确率最高的 τ = {best_tau(sweep)}（在测试集上挑阈值有乐观偏差，仅供参考）。", ""]

        out += ["## 5. 意图置信度校准", "",
                "按置信度分桶看意图准确率。注意多意图样本标了多个可接受意图，意图准确率普遍很高，这张表区分度有限。", ""]
        header = ["置信度区间"]
        for b in backends:
            header += [f"{BACKEND_LABEL[b]} n", "平均置信度", "准确率"]
        rows = []
        for i, bin_ in enumerate(calib[backends[0]][0]):
            cells = [f"[{bin_['lo']:.1f}, {bin_['hi']:.1f}{']' if bin_['hi'] == 1.0 else ')'}"]
            for b in backends:
                bb = calib[b][0][i]
                cells += [str(bb["n"]), pct(bb["mean_confidence"]), pct(bb["accuracy"])]
            rows.append(cells)
        rows.append(["**ECE**"] + sum([["", "", pct(calib[b][1])] for b in backends], []))
        out += _table(header, rows) + [""]

    out += [f"## 6. 分类别端到端正确率（τ = {meta['tau']}）", ""]
    rows = []
    for cat, label in CATEGORY_LABEL.items():
        members = {v: [r for r in rows_main[v] if r["category"] == cat] for v in variants}
        n = len(members[variants[0]])
        if n:
            rows.append([label, str(n)] + [pct(sum(r["e2e_ok"] for r in members[v]) / n) for v in variants])
    out += _table(["类别", "n"] + [ARM_LABEL[v] for v in variants], rows) + [""]

    out += [f"## 7. 逐条明细（τ = {meta['tau']}，完整系统）", "",
            "✓/✗ = 端到端是否正确；括号内是转人工原因；判断层组附「意图 置信度 / 需人工概率」。", ""]
    by_variant = {v: {r["case_id"]: r for r in rows_main[v]} for v in full}
    rows = []
    for c in cases:
        gold = f"{c['gold_route']} · {'/'.join(INTENT_ZH[i] for i in c['gold_intents'])}"
        rows.append([c["id"], CATEGORY_LABEL[c["category"]], _short(c["message"]), gold]
                    + [_cell(by_variant[v][c["id"]]) for v in full])
    out += _table(["id", "类别", "顾客消息", "标注"] + [ARM_LABEL[v] for v in full], rows) + [""]

    out += ["## 8. 读数说明与局限", "",
            f"- 样本量只有 {meta['n_cases']} 条，1 条约等于 {100 / max(meta['n_cases'], 1):.1f} 个百分点。小差异可能是噪声，建议扩充数据集，或多次运行取平均。",
            *(["- 这是留出测试集：系统（包括 needs_human 的判断标准）是在没看过这些样本的情况下设计的。"
               "请确认样本在跑实验前已冻结，跑完后只修正明显笔误并记录在 note 里。"] if meta["holdout"] else
              ["- needs_human 的判断标准是在看过 v1 实验结果之后加的（内容取自政策文档里「需人工」的条款，没有照抄测试句子）。"
               "尽管如此，这份数据集对它来说已经不是严格的留出测试集，结论需要用留出测试集复核（见 eval/holdout/编写指南.md）。",
               "- 数据集 v2 修订了 T09、T14、T16（原因见 dataset.jsonl 的 note），对所有组影响相同。"]),
            "- Agent 使用 temperature=0，但思考模式和兼容接口不保证完全确定，重跑时个别样本可能变化。",
            "- Jev 的主要训练语言是英文（官方文档注明中文准确率较低），意图描述已用英文加中文关键词来缓解。",
            "- 「LLM 自报」的置信度是模型自己说的数字，不是概率分布。它与 Jev 的对比检验的正是「校准」本身有没有价值。",
            ""]
    return "\n".join(out)
