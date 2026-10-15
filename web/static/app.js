"use strict";

// One turn on screen, in the order things really happen:
//   customer bubble → judgment card (+ gate) → tool steps → confirm bar → reply
// Events arrive over SSE from POST /api/sessions/{id}/messages (see web/turns.py).
// Every string from the customer or the model is set with textContent: it is never parsed as HTML.

const EXAMPLES = ["我的快递到哪了", "想换大一码", "不想要了，给我退款", "运费险怎么赔", "转人工！"];
const REASONS = {
  low_confidence: "意图置信度低于阈值",
  human_intent: "意图本身需要人工",
  needs_judgment: "需要人工判断",
  judge_error: "判断层出错",
  agent_transfer: "Agent 主动转人工",
  truncated: "模型输出被截断",
  loop_limit: "工具调用轮数超限",
};
const WHO = { main: "前台", specialist: "专员" };

const $ = (id) => document.getElementById(id);
const state = { config: null, customers: [], sessionId: null, busy: false, ended: false };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function add(node) {
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}

function showNotice(text) {
  $("notice").textContent = text || "";
  $("notice").hidden = !text;
}

function shorten(text, n) {
  const flat = String(text).replace(/\s+/g, " ").trim();
  return flat.length > n ? flat.slice(0, n) + "…" : flat;
}

function reasonText(reason) {
  return REASONS[reason] || reason || "";
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(detail);
  }
  return res;
}

// ---------------------------------------------------------------- setup

async function init() {
  state.config = await (await api("/api/config")).json();
  state.customers = await (await api("/api/customers")).json();

  $("mock-badge").hidden = !state.config.mock;
  for (const option of $("judge").options) option.disabled = !state.config.judges[option.value];
  const usable = [...$("judge").options].find((o) => !o.disabled);
  if (usable) $("judge").value = usable.value;
  $("threshold").value = state.config.default_threshold;

  for (const c of state.customers) $("customer").appendChild(new Option(`${c.id} ${c.name}`, c.id));
  for (const text of EXAMPLES) {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => send(text));
    $("examples").appendChild(chip);
  }

  $("customer").addEventListener("change", newSession);
  $("judge").addEventListener("change", newSession);
  $("threshold").addEventListener("input", showThreshold);
  $("threshold").addEventListener("change", newSession);
  $("new-session").addEventListener("click", newSession);
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    send($("input").value);
  });

  renderCustomer();
  showThreshold();
  if (!usable) {
    showNotice("没有可用的判断层：请在 .env 里配置 API key，或用 python3 -m web.server --mock 启动");
    setBusy(false);
    return;
  }
  await newSession();
}

function showThreshold() {
  $("threshold-value").textContent = Number($("threshold").value).toFixed(1);
}

function renderCustomer() {
  const customer = state.customers.find((c) => c.id === $("customer").value);
  if (!customer) return;
  $("orders").replaceChildren(...customer.orders.map((o) => {
    const size = o.size ? ` ${o.size} 码` : "";
    const li = el("li", null, `${o.order_id} ${o.item}${size} ¥${o.amount}`);
    li.appendChild(el("span", "sub", o.note));
    return li;
  }));
  $("memory").textContent = customer.memory || "（无）";
}

async function newSession() {
  if (state.busy) return;
  renderCustomer();
  showThreshold();
  showNotice("");
  const body = {
    customer_id: $("customer").value,
    judge: $("judge").value,
    threshold: Number($("threshold").value),
  };
  try {
    const res = await api("/api/sessions", { method: "POST", body: JSON.stringify(body) });
    state.sessionId = (await res.json()).session_id;
    state.ended = false;
    const who = $("customer").selectedOptions[0].text;
    const judge = $("judge").selectedOptions[0].text;
    $("messages").replaceChildren(el("p", "empty", `新会话 · ${who} · ${judge}\n以顾客身份说点什么吧`));
  } catch (err) {
    state.sessionId = null;
    showNotice(`开会话失败：${err.message}`);
  }
  setBusy(false);
}

function setBusy(busy) {
  state.busy = busy;
  const locked = busy || state.ended || !state.sessionId;
  $("input").disabled = locked;
  $("composer").querySelector("button").disabled = locked;
  for (const chip of $("examples").children) chip.disabled = locked;
  for (const id of ["customer", "judge", "threshold", "new-session"]) $(id).disabled = busy;
  $("input").placeholder = state.ended ? "会话已转人工，点「新会话」重新开始" : "以顾客身份输入，回车发送";
  if (!locked) $("input").focus();
}

// ---------------------------------------------------------------- one turn

async function send(text) {
  text = (text || "").trim();
  if (!text || state.busy || state.ended || !state.sessionId) return;
  $("input").value = "";
  showNotice("");
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();
  add(el("div", "bubble customer", text));

  const turn = new Turn();
  setBusy(true);
  try {
    const res = await api(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    await readEvents(res, (event) => turn.handle(event));
  } catch (err) {
    turn.error(`请求失败：${err.message}`);
  } finally {
    turn.finish();
    setBusy(false);
  }
}

// POST can't use EventSource, so parse the "data: {...}\n\n" stream by hand.
async function readEvents(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  }
}

function meter(label, value, kind) {
  const row = el("div", "meter");
  const track = el("div", "track");
  const fill = el("div", `fill ${kind}`);
  track.appendChild(fill);
  row.append(el("span", "label", label), track, el("span", "value", value.toFixed(2)));
  requestAnimationFrame(() => { fill.style.width = `${Math.round(value * 100)}%`; });
  return {
    row,
    mark(threshold, crossed) {
      const mark = el("div", "mark");
      mark.style.left = `${threshold * 100}%`;
      mark.title = `阈值 ${threshold.toFixed(2)}`;
      track.appendChild(mark);
      if (crossed) fill.classList.add("over");
    },
  };
}

class Turn {
  constructor() {
    this.open = [];          // running tool steps, innermost last (the specialist runs inside the front desk)
    this.steps = null;
    this.confirms = new Map();
    this.meta = null;
    this.typing = add(el("div", "typing", "思考中…"));
  }

  handle(event) {
    const handler = this[`on_${event.type}`];
    if (handler) handler.call(this, event);
    if (this.typing) add(this.typing);   // keep "thinking" at the bottom while the turn runs
  }

  on_judgment(e) {
    const card = el("div", "block card");
    const head = el("div", "card-head");
    const title = e.error
      ? "判断层 · 出错"
      : `判断层 · ${e.backend} · ${(e.latency_ms / 1000).toFixed(2)}s`;
    this.routeTag = el("span", "tag tag-wait", "判断中");
    head.append(el("span", null, title), this.routeTag);
    card.appendChild(head);
    if (e.degraded) {
      const why = e.degraded === "熔断中" ? "熔断中，冷却后自动重试" : e.degraded;
      card.appendChild(el("div", "degraded", `Jev 暂时不可用（${why}），本轮由备用判断层回答`));
    }
    if (e.error) {
      card.appendChild(el("div", "muted", e.error));
    } else {
      const intent = el("div", null, "意图 ");
      intent.appendChild(el("span", "tag tag-intent", `${e.intent} · ${e.intent_zh}`));
      card.appendChild(intent);
      this.confidence = meter("置信度", e.confidence, "conf");
      card.appendChild(this.confidence.row);
      if (typeof e.needs_human === "number") {
        this.needsHuman = meter("需人工", e.needs_human, "nh");
        card.appendChild(this.needsHuman.row);
      }
    }
    add(card);
  }

  on_gate(e) {
    if (this.confidence) this.confidence.mark(e.threshold, e.reason === "low_confidence");
    if (this.needsHuman && typeof e.needs_human_threshold === "number") {
      this.needsHuman.mark(e.needs_human_threshold, e.reason === "needs_judgment");
    }
    if (!this.routeTag) return;
    const human = e.route === "human";
    this.routeTag.className = `tag ${human ? "tag-bad" : "tag-ok"}`;
    this.routeTag.textContent = human ? `转人工 · ${reasonText(e.reason)}` : "交给 Agent";
  }

  on_skills(e) {
    if (!this.steps) this.steps = add(el("div", "block steps"));
    const row = el("div", "step skill");
    row.append(el("span", "status", "§ "), el("span", "who", `${WHO[e.by] || e.by} · `),
      el("span", null, `加载规范：${e.names.join("、")}`));
    this.steps.appendChild(row);
  }

  on_tool_start(e) {
    if (!this.steps) this.steps = add(el("div", "block steps"));
    const row = el("div", "step");
    row.append(el("span", "status", "⋯ "), el("span", "who", `${WHO[e.by] || e.by} · `),
      el("span", null, e.name), el("span", "out", ""));
    this.steps.appendChild(row);
    this.open.push({ key: `${e.by}:${e.name}`, row });
  }

  on_tool_end(e) {
    const index = this.open.map((o) => o.key).lastIndexOf(`${e.by}:${e.name}`);
    if (index < 0) return;
    const { row } = this.open.splice(index, 1)[0];
    row.classList.add(e.ok ? "ok" : "fail");
    row.querySelector(".status").textContent = e.ok ? "✓ " : "✗ ";
    row.querySelector(".out").textContent = e.output ? ` · ${shorten(e.output, 70)}` : "";
  }

  on_confirm_request(e) {
    const bar = el("div", "block confirm");
    bar.appendChild(el("span", null, `需要您确认：${e.text}`));
    const actions = el("div", "actions");
    const no = el("button", "btn", "取消");
    const yes = el("button", "btn btn-primary", "确认");
    no.type = "button";
    yes.type = "button";
    const answer = async (approve) => {
      no.disabled = true;
      yes.disabled = true;
      try {
        await api(`/api/sessions/${state.sessionId}/confirm`, {
          method: "POST",
          body: JSON.stringify({ request_id: e.request_id, approve }),
        });
      } catch (err) {
        showNotice(`确认失败：${err.message}`);
      }
    };
    no.addEventListener("click", () => answer(false));
    yes.addEventListener("click", () => answer(true));
    actions.append(no, yes);
    bar.appendChild(actions);
    this.confirms.set(e.request_id, bar);
    add(bar);
  }

  on_confirm_resolved(e) {
    const bar = this.confirms.get(e.request_id);
    if (!bar) return;
    const text = e.timed_out ? "确认超时，已取消" : e.approved ? "✓ 顾客已确认" : "顾客已取消";
    bar.replaceChildren(el("span", null, text));
  }

  on_reply(e) {
    const bubble = el("div", "bubble agent", e.text);
    this.meta = el("span", "meta", "");
    bubble.appendChild(this.meta);
    add(bubble);
    if (e.route === "human") {
      add(el("div", "block banner", `已转人工 · 工单 ${e.ticket_id || "—"} · ${reasonText(e.reason)}`));
    }
  }

  on_error(e) {
    this.error(`这一轮出错了：${e.message}`);
  }

  on_done(e) {
    if (this.meta) this.meta.textContent = `本轮 ${(e.elapsed_ms / 1000).toFixed(1)}s`;
    state.ended = Boolean(e.ended);
  }

  error(text) {
    add(el("div", "block error", text));
  }

  finish() {
    if (this.typing) this.typing.remove();
    this.typing = null;
  }
}

init().catch((err) => showNotice(`页面初始化失败：${err.message}`));
