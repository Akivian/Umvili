const COLORS = {
  independent_q_learning: "#569cd6",
  qmix: "#ce9178",
  rule_based: "#4ec9b0",
  sugar: "#dcdcaa",
  spice: "#c586c0",
  hazard: "#f44747",
};

const LABELS = {
  independent_q_learning: "IQL",
  qmix: "QMIX",
  rule_based: "Rule",
};

const state = {
  hash: "",
  paused: false,
  sugar: {},
  loss: {},
  lastMap: null,
  lastAgents: [],
};

const $ = (id) => document.getElementById(id);
const form = $("tools");

function notice(text) {
  $("notice").textContent = text || "";
}

function readForm() {
  const data = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox") data[element.name] = element.checked;
    else if (element.type === "number") data[element.name] = element.value === "" ? null : Number(element.value);
    else data[element.name] = element.value;
  }
  return data;
}

function writeForm(controls) {
  for (const [name, value] of Object.entries(controls)) {
    const element = form.elements[name];
    if (!element) continue;
    if (element.type === "checkbox") element.checked = Boolean(value);
    else element.value = value;
  }
}

function formatParams(count) {
  if (count >= 1e6) return (count / 1e6).toFixed(2) + " M 参数";
  if (count >= 1e3) return (count / 1e3).toFixed(1) + " K 参数";
  return count + " 参数";
}

async function preview() {
  const response = await fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readForm()),
  });
  const payload = await response.json();
  if (!response.ok) {
    notice(payload.detail || "参数无效");
    return;
  }
  notice("");
  $("draftHash").textContent = payload.protocol_hash;
  const capacity = $("capacity");
  const families = payload.families === 0 ? "无学习器" : payload.families + " 套网络";
  capacity.textContent = formatParams(payload.params) + " · 约 " + payload.vram_mb + " MB · " + families;
  capacity.classList.toggle("warn", payload.warn);
  $("applyHint").textContent = payload.hash_changes
    ? "协议会变，曲线将清空"
    : "同一协议下，换算法会留下另一条曲线";
}

let previewTimer = 0;
function schedulePreview() {
  window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(() => {
    preview().catch((error) => notice(String(error)));
  }, 200);
}

async function loadMeta() {
  const response = await fetch("/api/meta");
  const meta = await response.json();
  const mode = form.elements.algorithm_mode;
  mode.innerHTML = "";
  for (const item of meta.modes) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    mode.appendChild(option);
  }
  for (const [name, limit] of Object.entries(meta.limits)) {
    const element = form.elements[name];
    if (!element) continue;
    element.min = limit.min;
    element.max = limit.max;
    if (name === "learning_rate") element.step = "0.0001";
  }
  const presets = $("presets");
  for (const [name, values] of Object.entries(meta.presets)) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name === "tiny" ? "Tiny" : name === "default" ? "Default" : "Wide";
    button.addEventListener("click", () => {
      writeForm(values);
      schedulePreview();
    });
    presets.appendChild(button);
  }
  writeForm(meta.defaults);
  const stateResponse = await fetch("/api/state");
  const snapshot = await stateResponse.json();
  if (snapshot.controls) writeForm(snapshot.controls);
  if (snapshot.hash) {
    state.hash = snapshot.hash;
    $("appliedHash").textContent = snapshot.hash;
  }
  setSpeed(snapshot.speed || 1);
  await preview();
}

function setSpeed(speed) {
  for (const button of $("speeds").querySelectorAll("button")) {
    button.classList.toggle("active", Number(button.dataset.speed) === Number(speed));
  }
}

async function control(action, extra) {
  const response = await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...extra }),
  });
  const payload = await response.json();
  if (!response.ok) notice(payload.detail || "操作失败");
  return payload;
}

function replaceSeries(payload) {
  state.sugar = payload.sugar_per_step || {};
  state.loss = payload.loss || {};
  drawChart($("sugarChart"), state.sugar, "sugar / step");
  drawChart($("lossChart"), state.loss, "loss");
}

function appendPoints(points) {
  if (!points || !points.length) return;
  for (const point of points) {
    addPoints(state.sugar, point.step, point.sugar);
    addPoints(state.loss, point.step, point.loss);
  }
  drawChart($("sugarChart"), state.sugar, "sugar / step");
  drawChart($("lossChart"), state.loss, "loss");
}

function addPoints(bucket, step, values) {
  for (const [name, value] of Object.entries(values || {})) {
    const series = bucket[name] || (bucket[name] = []);
    const last = series[series.length - 1];
    if (last && last.step === step) last.value = value;
    else series.push({ step, value });
  }
}

function applyFrame(frame) {
  if (!frame) return;
  if (frame.hash) {
    state.hash = frame.hash;
    $("appliedHash").textContent = frame.hash;
  }
  state.paused = frame.state === "paused";
  $("pause").textContent = state.paused ? "继续" : "暂停";
  setSpeed(frame.speed || 1);
  const rates = Object.entries(frame.sugar || {})
    .map(([name, value]) => (LABELS[name] || name) + " " + Number(value).toFixed(2))
    .join("  ");
  const epsilon = frame.epsilon ? "  ε " + Number(frame.epsilon).toFixed(2) : "";
  $("status").textContent = [
    frame.error ? frame.error : frame.state,
    "step " + (frame.step || 0),
    "alive " + (frame.alive || 0),
    rates,
    epsilon,
    frame.device || "",
  ].filter(Boolean).join("   ");
  if (frame.map) {
    state.lastMap = frame.map;
    state.lastAgents = frame.agents || [];
    drawMap();
  }
}

function drawMap() {
  const canvas = $("map");
  const map = state.lastMap;
  if (!map) return;
  const view = prepare(canvas);
  const ctx = view.ctx;
  const bytes = Uint8Array.from(atob(map.rgb), (char) => char.charCodeAt(0));
  const offscreen = document.createElement("canvas");
  offscreen.width = map.width;
  offscreen.height = map.height;
  const image = offscreen.getContext("2d").createImageData(map.width, map.height);
  for (let i = 0, o = 0; i < bytes.length; i += 3, o += 4) {
    const sugar = bytes[i] / 255;
    const spice = bytes[i + 1] / 255;
    const hazard = bytes[i + 2] / 255;
    let red = 30;
    let green = 30;
    let blue = 30;
    red += (220 - 30) * sugar;
    green += (220 - 30) * sugar;
    blue += (170 - 30) * sugar;
    red += (197 - red) * spice;
    green += (134 - green) * spice;
    blue += (192 - blue) * spice;
    red += (244 - red) * hazard;
    green += (71 - green) * hazard;
    blue += (71 - blue) * hazard;
    image.data[o] = red;
    image.data[o + 1] = green;
    image.data[o + 2] = blue;
    image.data[o + 3] = 255;
  }
  offscreen.getContext("2d").putImageData(image, 0, 0);
  const scale = Math.max(1, Math.floor(Math.min(view.width, view.height) / map.width));
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#1e1e1e";
  ctx.fillRect(0, 0, view.width, view.height);
  ctx.drawImage(offscreen, 0, 0, map.width * scale, map.height * scale);
  for (const agent of state.lastAgents) {
    ctx.beginPath();
    ctx.fillStyle = COLORS[agent.type] || "#d4d4d4";
    ctx.arc((agent.x + 0.5) * scale, (agent.y + 0.5) * scale, Math.max(2.2, scale * 0.34), 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawChart(canvas, bucket, unit) {
  const view = prepare(canvas);
  const ctx = view.ctx;
  const width = view.width;
  const height = view.height;
  ctx.clearRect(0, 0, width, height);
  const names = Object.keys(bucket);
  const pad = { left: 48, right: 12, top: 16, bottom: 22 };
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.fillStyle = "#858585";
  ctx.textAlign = "left";
  let legendX = pad.left;
  names.forEach((name) => {
    const label = LABELS[name] || name;
    ctx.fillStyle = COLORS[name] || "#d4d4d4";
    ctx.fillRect(legendX, 4, 8, 8);
    ctx.fillText(label, legendX + 12, 12);
    legendX += ctx.measureText(label).width + 28;
  });
  if (!names.length) {
    ctx.fillStyle = "#858585";
    ctx.fillText("等待第一步", pad.left, height / 2);
    return;
  }
  let minStep = Infinity;
  let maxStep = -Infinity;
  let maxValue = 0;
  for (const rows of Object.values(bucket)) {
    for (const point of rows) {
      minStep = Math.min(minStep, point.step);
      maxStep = Math.max(maxStep, point.step);
      maxValue = Math.max(maxValue, point.value);
    }
  }
  if (!Number.isFinite(minStep)) return;
  if (maxStep === minStep) maxStep += 1;
  if (maxValue <= 0) maxValue = 1;
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.strokeStyle = "#3c3c3c";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();
  ctx.fillStyle = "#858585";
  ctx.fillText(maxValue.toFixed(2), 4, pad.top + 10);
  ctx.fillText("0", 4, pad.top + plotH);
  for (const [name, rows] of Object.entries(bucket)) {
    if (!rows.length) continue;
    ctx.beginPath();
    ctx.strokeStyle = COLORS[name] || "#d4d4d4";
    ctx.lineWidth = 1.6;
    rows.forEach((point, index) => {
      const x = pad.left + ((point.step - minStep) / (maxStep - minStep)) * plotW;
      const y = pad.top + plotH - (point.value / maxValue) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#858585";
  ctx.fillText(unit, pad.left, height - 4);
}

function prepare(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const pixelWidth = Math.floor(width * ratio);
  const pixelHeight = Math.floor(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(protocol + "://" + location.host + "/ws");
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.kind === "hello") {
      if (message.hash) {
        state.hash = message.hash;
        $("appliedHash").textContent = message.hash;
      }
      if (message.replace) replaceSeries(message.replace);
      applyFrame(message.frame);
      return;
    }
    if (message.replace) replaceSeries(message.replace);
    else {
      if (message.hash && state.hash && message.hash !== state.hash) {
        state.sugar = {};
        state.loss = {};
      }
      appendPoints(message.points);
    }
    applyFrame(message);
  };
  socket.onclose = () => {
    $("status").textContent = "连接中断，正在重试";
    window.setTimeout(connect, 1500);
  };
}

function buildLegend() {
  const items = [
    ["sugar", "糖"],
    ["spice", "香料"],
    ["hazard", "危险"],
    ["independent_q_learning", "IQL"],
    ["qmix", "QMIX"],
    ["rule_based", "Rule"],
  ];
  $("legend").innerHTML = items
    .map(([key, label]) => `<span><i style="background:${COLORS[key]}"></i>${label}</span>`)
    .join("");
}

$("pause").addEventListener("click", () => {
  control(state.paused ? "resume" : "pause").catch((error) => notice(String(error)));
});
$("reset").addEventListener("click", () => {
  control("reset").catch((error) => notice(String(error)));
});
$("speeds").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  control("speed", { speed: Number(button.dataset.speed) })
    .then(() => setSpeed(button.dataset.speed))
    .catch((error) => notice(String(error)));
});
$("apply").addEventListener("click", async () => {
  $("apply").disabled = true;
  try {
    const response = await fetch("/api/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const payload = await response.json();
    if (!response.ok) notice(payload.detail || "应用失败");
    else {
      notice("");
      $("appliedHash").textContent = payload.protocol_hash;
      state.hash = payload.protocol_hash;
      await preview();
    }
  } catch (error) {
    notice(String(error));
  } finally {
    $("apply").disabled = false;
  }
});
$("export").addEventListener("click", () => {
  window.location = "/api/export";
});
form.addEventListener("input", schedulePreview);
form.addEventListener("change", schedulePreview);
window.addEventListener("resize", () => {
  drawMap();
  drawChart($("sugarChart"), state.sugar, "sugar / step");
  drawChart($("lossChart"), state.loss, "loss");
});

buildLegend();
const observer = new ResizeObserver(() => {
  drawMap();
  drawChart($("sugarChart"), state.sugar, "sugar / step");
  drawChart($("lossChart"), state.loss, "loss");
});
observer.observe($("map"));
observer.observe($("sugarChart"));
observer.observe($("lossChart"));
loadMeta().catch((error) => notice(String(error)));
connect();
