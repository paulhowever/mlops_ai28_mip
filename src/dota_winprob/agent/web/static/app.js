"use strict";

const ACTIONS_EVERY = 4000;
const INCIDENTS_EVERY = 6000;
const INFO_EVERY = 30000;
const HISTORY_TURNS = 8;
const HISTORY_TURN = 1400;
const IMAGE_TYPES = ["image/png", "image/jpeg"];
const EXAMPLES = [
  "Что сейчас с очередью и хватает ли воркеров?",
  "Почему задачи попали в DLQ?",
  "Верни задачи из DLQ в работу",
  "Какова вероятность победы Radiant в матче 566?",
  "Насколько можно верить модели?",
];
const ACTION_STATUS = {
  proposed: ["ждёт решения", "warn"],
  executing: ["выполняется", "warn"],
  done: ["выполнено", "ok"],
  partial: ["выполнено частично", "bad"],
  failed: ["не выполнено", "bad"],
  rejected: ["отклонено", ""],
  expired: ["срок истёк", ""],
};
const ANALYSIS_STATUS = {
  queued: "разбор в очереди",
  running: "идёт разбор",
  done: "разобран",
  facts_only: "только факты",
  failed: "разбор не удался",
  skipped: "разбор пропущен",
};
const STEP_BAD = new Set(["unavailable", "timeout", "malformed", "failed", "error", "invalid_input"]);

const ui = Object.fromEntries(
  [
    "model", "state", "log", "welcome", "examples", "form", "preview", "thumb", "imageName",
    "dropImage", "question", "file", "reset", "hint", "send", "actions", "incidents",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  info: null,
  history: [],
  image: null,
  busy: false,
  deciding: new Set(),
  drawn: { actions: "", incidents: "" },
  opened: new Set(),
};

function el(tag, className, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children) {
    if (child === null || child === undefined || child === "") continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function button(label, className, handler) {
  const node = el("button", className, label);
  node.type = "button";
  node.addEventListener("click", handler);
  return node;
}

function badge(label, tone) {
  return el("span", `badge ${tone || ""}`.trim(), label);
}

function clock(value) {
  const moment = new Date(value);
  return Number.isNaN(moment.getTime()) ? "" : moment.toLocaleTimeString("ru-RU");
}

function seconds(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} с` : `${Math.round(ms)} мс`;
}

function traceLink(traceId) {
  if (!/^[0-9a-f]{32}$/.test(traceId || "")) return null;
  const label = el("span", "mono muted", `trace_id ${traceId}`);
  if (!state.info || !/^https?:\/\//.test(state.info.grafana_url)) return label;
  const panes = {
    a: {
      datasource: "tempo",
      queries: [
        { refId: "A", datasource: { type: "tempo", uid: "tempo" }, queryType: "traceql", query: traceId },
      ],
      range: { from: "now-6h", to: "now" },
    },
  };
  const link = el("a", "mono", `трейс ${traceId.slice(0, 8)}… в Grafana`);
  const query = new URLSearchParams({ schemaVersion: "1", orgId: "1", panes: JSON.stringify(panes) });
  link.href = `${state.info.grafana_url}/explore?${query}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function details(key, title, ...children) {
  const node = el("details", "", el("summary", "", title), ...children);
  if (state.opened.has(key)) node.open = true;
  node.addEventListener("toggle", () => {
    if (node.open) state.opened.add(key);
    else state.opened.delete(key);
  });
  return node;
}

function pretty(value) {
  return el("div", "mono muted text", JSON.stringify(value, null, 1));
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { cache: "no-store", credentials: "same-origin", ...options });
  } catch {
    throw { detail: "Агент не отвечает. Запущен ли он: make agent-up?", reason: "network" };
  }
  const body = await response.json().catch(() => null);
  if (response.ok) return body;
  const invalid = Array.isArray(body?.detail) ? body.detail.map((item) => item.msg).join("; ") : null;
  throw {
    status: response.status,
    reason: body?.reason || "error",
    detail: invalid || body?.detail || `Ошибка ${response.status}`,
    retryAfter: Number(response.headers.get("Retry-After")) || null,
    traceId: body?.trace_id,
  };
}

function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function show(node) {
  ui.welcome.hidden = true;
  ui.log.append(node);
  node.scrollIntoView({ block: "nearest" });
  return node;
}

function stepList(key, steps) {
  if (!steps.length) return null;
  const total = steps.reduce((sum, step) => sum + step.duration_ms, 0);
  const list = el("ol", "steps");
  for (const step of steps) {
    const tone = STEP_BAD.has(step.status) ? "bad" : step.status === "ok" ? "ok" : "warn";
    const item = el(
      "li", "",
      el("div", "row", el("strong", "mono", step.tool), badge(step.status, tone), el("span", "muted small", seconds(step.duration_ms))),
      el("div", "text small", step.summary),
    );
    if (Object.keys(step.arguments || {}).length) item.append(pretty(step.arguments));
    list.append(item);
  }
  return details(key, `Шаги агента: ${steps.length}, ${seconds(total)}`, list);
}

function prediction(item) {
  return el(
    "div", "prediction",
    el("strong", "", `${(item.win_probability * 100).toFixed(2)}%`),
    el("span", "", "вероятность победы Radiant"),
    el("span", "muted small mono", `${item.tool} · ${item.task_id} · модель v${item.model_version ?? "?"}`),
  );
}

function answerNode(body) {
  const node = el("article", "message", el("div", "text", body.answer));
  if (body.status === "step_limit") node.append(el("div", "notice", "Ответ неполный: закончился бюджет шагов."));
  for (const warning of body.warnings) node.append(el("div", "notice", warning));
  for (const item of body.predictions) node.append(prediction(item));
  if (body.actions.length) {
    node.append(el("div", "notice", "Создано предложение. Решение за вами: в блоке «Действия»."));
  }
  node.append(stepList(`steps-${body.trace_id}`, body.steps) || "");
  const usage = `${body.model}, запросов ${body.usage.llm_requests}, токенов ${body.usage.total_tokens}`;
  node.append(el("div", "row small", el("span", "muted", usage), traceLink(body.trace_id)));
  return node;
}

function errorNode(error) {
  const node = el("article", "message error", el("div", "text", error.detail));
  if (error.reason === "no_key") {
    node.append(el("div", "muted small", "Инциденты, факты и подтверждение действий работают без языковой модели."));
  } else if (error.retryAfter) {
    const left = el("div", "muted small");
    node.append(left);
    const until = Date.now() + error.retryAfter * 1000;
    const tick = () => {
      const rest = Math.ceil((until - Date.now()) / 1000);
      left.textContent = rest > 0 ? `Повторите через ${rest} с` : "Можно повторить";
      if (rest > 0 && node.isConnected) setTimeout(tick, 1000);
    };
    tick();
  }
  const trace = traceLink(error.traceId);
  if (trace) node.append(el("div", "small", trace));
  return node;
}

function setBusy(busy) {
  state.busy = busy;
  ui.send.disabled = busy;
  ui.send.textContent = busy ? "Думает" : "Спросить";
}

async function ask(text) {
  const question = text.trim();
  if (state.busy) return;
  if (!question) {
    hint("Напишите вопрос.");
    return;
  }
  const limit = state.info?.max_question || 2000;
  if (question.length > limit) {
    hint(`Вопрос длиннее ${limit} символов.`);
    return;
  }
  hint("");
  const image = state.image;
  const mine = el("article", "message user", el("div", "text", question));
  if (image) {
    const picture = el("img");
    picture.alt = "Картинка к вопросу";
    picture.src = image.url;
    mine.append(picture);
  }
  show(mine);
  const waiting = show(el("article", "message", el("div", "muted waiting-dots", "Агент проверяет данные")));
  const started = Date.now();
  const timer = setInterval(() => {
    waiting.firstChild.textContent = `Агент проверяет данные, ${Math.round((Date.now() - started) / 1000)} с`;
  }, 1000);
  setBusy(true);
  try {
    const body = { question, history: state.history };
    if (image) body.image = { data: image.base64 };
    const answer = await post("/agent/ask", body);
    waiting.replaceWith(answerNode(answer));
    remember(question, answer.answer);
    ui.question.value = "";
    dropImage(false);
    if (answer.actions.length) refreshActions();
  } catch (error) {
    waiting.replaceWith(errorNode(error));
  } finally {
    clearInterval(timer);
    setBusy(false);
    ui.log.lastElementChild?.scrollIntoView({ block: "nearest" });
    ui.question.focus();
  }
}

function remember(question, answer) {
  state.history.push(
    { role: "user", content: question.slice(0, HISTORY_TURN) },
    { role: "assistant", content: (answer || "…").slice(0, HISTORY_TURN) },
  );
  state.history = state.history.slice(-HISTORY_TURNS);
}

function hint(text) {
  ui.hint.textContent = text;
}

function readImage(file) {
  if (!file) return;
  if (!IMAGE_TYPES.includes(file.type)) {
    hint("Нужна картинка PNG или JPEG.");
    return;
  }
  const limit = state.info?.max_image_bytes || 4 * 1024 * 1024;
  if (file.size > limit) {
    hint(`Картинка больше ${Math.round(limit / 1024 / 1024)} МБ.`);
    return;
  }
  const reader = new FileReader();
  reader.onerror = () => hint("Не удалось прочитать картинку.");
  reader.onload = () => {
    dropImage();
    const base64 = String(reader.result).split(",", 2)[1] || "";
    state.image = { base64, url: URL.createObjectURL(file) };
    ui.thumb.src = state.image.url;
    ui.imageName.textContent = `${file.name || "из буфера"}, ${Math.ceil(file.size / 1024)} КБ`;
    ui.preview.hidden = false;
    hint("");
  };
  reader.readAsDataURL(file);
}

function dropImage(revoke = true) {
  if (state.image && revoke) URL.revokeObjectURL(state.image.url);
  state.image = null;
  ui.preview.hidden = true;
  ui.thumb.removeAttribute("src");
  ui.file.value = "";
}

function figures(entries) {
  const list = el("dl", "figures");
  for (const [name, value] of entries) list.append(el("div", "", el("dt", "", name), el("dd", "", value)));
  return list;
}

function resultNode(result) {
  if (!result) return null;
  const node = el("div", "stack small");
  node.append(el("div", "", `Возвращено задач: ${result.returned.length}`));
  if (result.missing) node.append(el("div", "muted", `Уже не было в DLQ: ${result.missing}`));
  if (result.uncertain.length) {
    node.append(el("div", "notice", `Без подтверждения брокера: ${result.uncertain.join(", ")}. Повтор безопасен.`));
  }
  if (result.stopped) node.append(el("div", "notice bad", `Остановлено: ${result.stopped}`));
  if (result.dlq_left !== null) node.append(el("div", "muted", `Осталось в DLQ: ${result.dlq_left}`));
  return node;
}

function actionNode(action) {
  const [label, tone] = ACTION_STATUS[action.status] || [action.status, ""];
  const waiting = action.status === "proposed";
  const plan = action.plan;
  const node = el(
    "article", waiting ? "item waiting" : "item",
    el("div", "row", el("h3", "", "Вернуть задачи из DLQ"), badge(label, tone)),
  );
  if (waiting) node.append(el("div", "muted small", `Действует до ${clock(action.expires_at)}`));
  node.append(figures([
    ["в DLQ", plan.depth], ["просмотрено", plan.scanned], ["валидных", plan.valid], ["мусора", plan.garbage],
  ]));
  const list = el("ul", "list small");
  for (const task of plan.tasks) {
    list.append(el("li", "", el("span", "mono", task.task_id), el("span", "muted", ` ${task.source}, ${task.reason}, попытка ${task.attempts ?? "?"}`)));
  }
  node.append(details(`plan-${action.id}`, `Вернутся в ${plan.target}: ${plan.tasks.length}`, list));
  for (const note of plan.notes) node.append(el("div", "muted small", note));
  if (waiting) {
    const locked = state.deciding.has(action.id);
    const confirm = button("Подтвердить", "primary", () => decide(action.id, "confirm"));
    const reject = button("Отклонить", "", () => decide(action.id, "reject"));
    confirm.disabled = reject.disabled = locked;
    node.append(el("div", "row", confirm, reject));
  }
  node.append(resultNode(action.result) || "");
  const trace = traceLink(action.execution_trace_id || action.trace_id);
  if (trace) node.append(el("div", "small", trace));
  return node;
}

async function decide(id, decision) {
  if (state.deciding.has(id)) return;
  state.deciding.add(id);
  drawActions(state.lastActions || []);
  try {
    await post(`/agent/actions/${encodeURIComponent(id)}/${decision}`, {});
  } catch (error) {
    show(errorNode(error));
  } finally {
    state.deciding.delete(id);
    state.drawn.actions = "";
    refreshActions();
  }
}

function incidentNode(incident) {
  const analysis = incident.analysis;
  const firing = incident.status === "firing";
  const node = el(
    "article", "item",
    el("div", "row", badge(firing ? "горит" : "закрыт", firing ? "bad" : "ok"), badge(ANALYSIS_STATUS[analysis.status] || analysis.status, "")),
    el("h3", "", incident.title),
    el("div", "muted small", `с ${clock(incident.opened_at)}${incident.resolved_at ? `, закрыт в ${clock(incident.resolved_at)}` : ""}${incident.repeats ? `, повторов ${incident.repeats}` : ""}`),
    incident.summary ? el("div", "text small", incident.summary) : null,
  );
  if (analysis.text) node.append(details(`text-${incident.id}`, "Разбор агента", el("div", "text small", analysis.text)));
  if (analysis.reason) node.append(el("div", "muted small", analysis.reason));
  for (const warning of analysis.warnings) node.append(el("div", "notice small", warning));
  if (analysis.facts.length) {
    const list = el("ol", "steps");
    for (const fact of analysis.facts) {
      const tone = STEP_BAD.has(fact.status) ? "bad" : "ok";
      list.append(el(
        "li", "",
        el("div", "row", el("strong", "mono", fact.tool), badge(fact.status, tone)),
        el("div", "text small", fact.summary),
        details(`fact-${incident.id}-${fact.tool}`, "данные", pretty(fact.data)),
      ));
    }
    node.append(details(`facts-${incident.id}`, `Факты: ${analysis.facts.length}`, list));
  }
  node.append(stepList(`incident-steps-${incident.id}`, analysis.steps) || "");
  const trace = traceLink(analysis.trace_id);
  if (trace) node.append(el("div", "small", trace));
  return node;
}

function draw(name, target, items, build, empty) {
  const snapshot = JSON.stringify([items, [...state.deciding], state.info?.grafana_url]);
  if (state.drawn[name] === snapshot) return;
  state.drawn[name] = snapshot;
  target.replaceChildren(...(items.length ? items.map(build) : [el("p", "muted small", empty)]));
}

function drawActions(items) {
  state.lastActions = items;
  draw("actions", ui.actions, items, actionNode, "Предложений нет. Попросите агента вернуть задачи из DLQ.");
}

async function refreshActions() {
  try {
    drawActions(await request("/agent/actions"));
  } catch (error) {
    failed("actions", ui.actions, error);
  }
}

async function refreshIncidents() {
  try {
    const items = await request("/agent/incidents");
    draw("incidents", ui.incidents, items, incidentNode, "Инцидентов не было.");
  } catch (error) {
    failed("incidents", ui.incidents, error);
  }
}

function failed(name, target, error) {
  state.drawn[name] = "";
  target.replaceChildren(el("p", "notice bad small", error.detail));
}

async function refreshInfo() {
  try {
    state.info = await request("/agent/info");
    ui.model.textContent = `модель ${state.info.model}`;
    ui.state.className = state.info.llm_configured ? "badge ok" : "badge warn";
    ui.state.textContent = state.info.llm_configured ? "готов отвечать" : "нет ключа модели, вопросы недоступны";
  } catch {
    ui.state.className = "badge bad";
    ui.state.textContent = "агент не отвечает";
  }
}

function poll(refresh, every) {
  refresh();
  setInterval(() => {
    if (!document.hidden) refresh();
  }, every);
}

for (const example of EXAMPLES) {
  ui.examples.append(button(example, "", () => ask(example)));
}
ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(ui.question.value);
});
ui.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    ask(ui.question.value);
  }
});
document.addEventListener("paste", (event) => {
  const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith("image/"));
  if (file) {
    event.preventDefault();
    readImage(file);
  }
});
ui.form.addEventListener("dragover", (event) => {
  event.preventDefault();
  ui.form.classList.add("drag");
});
ui.form.addEventListener("dragleave", () => ui.form.classList.remove("drag"));
ui.form.addEventListener("drop", (event) => {
  event.preventDefault();
  ui.form.classList.remove("drag");
  readImage(event.dataTransfer?.files?.[0]);
});
ui.file.addEventListener("change", () => readImage(ui.file.files[0]));
ui.dropImage.addEventListener("click", () => dropImage());
ui.reset.addEventListener("click", () => {
  state.history = [];
  ui.log.replaceChildren(ui.welcome);
  ui.welcome.hidden = false;
  hint("");
});

poll(refreshInfo, INFO_EVERY);
poll(refreshActions, ACTIONS_EVERY);
poll(refreshIncidents, INCIDENTS_EVERY);
