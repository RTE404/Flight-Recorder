const API = "/api";
let cy = null;
let currentTraceId = null;
let ws = null;
let diffMode = null; // {a, b, changedKeys} when active

const COL_W = 120;
const LANE_H = 90;

const ROLE_COLOR = {
  recorded: "#8a93a6",
  reused: "#2bb3a3",
  mutated: "#e0a52c",
  live: "#e0603f",
};

const TYPE_GLYPH = {
  llm_call: "llm",
  tool_call: "tool",
  clock: "now",
  random: "rnd",
  agent_msg: "msg",
};

async function api(path, opts) {
  const resp = await fetch(API + path, opts);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${path}: ${body}`);
  }
  return resp.json();
}

async function loadTraces() {
  const traces = await api("/traces");
  const list = document.getElementById("trace-list");
  list.innerHTML = "";
  const selA = document.getElementById("diff-a");
  const selB = document.getElementById("diff-b");
  selA.innerHTML = "";
  selB.innerHTML = "";
  for (const t of traces) {
    const li = document.createElement("li");
    li.textContent = t.parent_trace_id
      ? `${t.trace_id} (fork of ${t.parent_trace_id})`
      : t.trace_id;
    const status = document.createElement("div");
    status.className = "status";
    status.textContent = `${t.status} — ${t.task}`;
    li.appendChild(status);
    if (t.trace_id === currentTraceId) li.classList.add("selected");
    li.addEventListener("click", () => openTrace(t.trace_id));
    list.appendChild(li);

    for (const sel of [selA, selB]) {
      const opt = document.createElement("option");
      opt.value = t.trace_id;
      opt.textContent = t.trace_id;
      sel.appendChild(opt);
    }
  }
}

function initCytoscape() {
  cy = cytoscape({
    container: document.getElementById("cy"),
    layout: { name: "preset" },
    style: [
      { selector: "node", style: {
          "background-color": (ele) => ROLE_COLOR[ele.data("role")] || ROLE_COLOR.recorded,
          "label": (ele) => TYPE_GLYPH[ele.data("event_type")] || ele.data("event_type"),
          "font-size": 9, "color": "#0b0c0e", "text-valign": "center", "text-halign": "center",
          "width": 34, "height": 34, "shape": "round-rectangle",
          "border-width": (ele) => (ele.data("role") === "mutated" ? 3 : 1),
          "border-color": (ele) => (ele.hasClass("changed") ? "#ff3b3b" : "#0b0c0e"),
      }},
      { selector: "node.dimmed", style: { "opacity": 0.35 } },
      { selector: "edge[kind='message']", style: {
          "width": 2.5, "line-color": "#cfd6e4", "target-arrow-color": "#cfd6e4",
          "target-arrow-shape": "triangle", "curve-style": "bezier",
      }},
      { selector: "edge[kind='sequence']", style: {
          "width": 1, "line-color": "#4a4f5a", "target-arrow-color": "#4a4f5a",
          "target-arrow-shape": "triangle", "curve-style": "bezier",
      }},
    ],
  });
  cy.on("tap", "node", (evt) => showDetail(evt.target.data()));
}

function renderGraph(graph) {
  if (!cy) initCytoscape();
  cy.elements().remove();

  const elements = [];
  for (const n of graph.nodes) {
    elements.push({
      group: "nodes",
      data: { id: n.event_id, ...n },
      position: { x: n.column * COL_W + 80, y: n.lane * LANE_H + 40 },
    });
  }
  for (const e of graph.edges) {
    elements.push({
      group: "edges",
      data: { id: `${e.from}->${e.to}-${e.kind}`, source: e.from, target: e.to, kind: e.kind },
    });
  }
  cy.add(elements);
  cy.layout({ name: "preset" }).run();
  cy.fit(undefined, 60);

  renderLaneLabels(graph.trace.agents);
  if (diffMode) applyDiffOverlay();
}

function renderLaneLabels(agents) {
  let container = document.getElementById("lane-labels");
  if (!container) {
    container = document.createElement("div");
    container.id = "lane-labels";
    document.getElementById("cy").appendChild(container);
  }
  container.innerHTML = agents.map((a, i) =>
    `<div style="position:absolute;left:8px;top:${i * LANE_H + 24}px;font-size:11px;opacity:0.6">${a}</div>`
  ).join("");
}

function showDetail(n) {
  const panel = document.getElementById("detail-panel");
  panel.classList.remove("hidden");
  document.getElementById("detail-body").innerHTML = `
    <dl>
      <dt>agent</dt><dd>${n.agent_id}</dd>
      <dt>event</dt><dd>${n.event_type} #${n.seq}</dd>
      <dt>role</dt><dd>${n.role}</dd>
      <dt>vector_clock</dt><dd>${JSON.stringify(n.vector_clock)}</dd>
      <dt>causal_rank</dt><dd>${n.causal_rank}</dd>
      <dt>boundary_hash</dt><dd>${n.boundary_hash}</dd>
      <dt>request</dt><dd>${n.request_preview}</dd>
      <dt>response</dt><dd>${n.response_preview}</dd>
    </dl>`;
  document.getElementById("fork-form").dataset.eventId = n.event_id;
}

async function doFork(atEventId, mutationText) {
  let mutation;
  try {
    mutation = JSON.parse(mutationText);
  } catch (e) {
    alert("Mutation must be valid JSON: " + e.message);
    return;
  }
  const result = await api(`/traces/${currentTraceId}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ at_event_id: atEventId, mutation }),
  });
  await loadTraces();
  await openTrace(result.child_trace_id);
}

async function doDiff(a, b) {
  const overlay = await api(`/diff/${a}/${b}`);
  diffMode = { a, b, changedKeys: new Set(overlay.changed_keys.map((k) => k.join("|"))) };
  applyDiffOverlay();
}

function applyDiffOverlay() {
  if (!cy || !diffMode) return;
  cy.nodes().forEach((node) => {
    const d = node.data();
    const key = `${d.agent_id}|${d.event_type}|${d.seq}`;
    if (diffMode.changedKeys.has(key)) {
      node.addClass("changed");
      node.removeClass("dimmed");
    } else {
      node.removeClass("changed");
      node.addClass("dimmed");
    }
  });
}

function clearDiff() {
  diffMode = null;
  if (cy) cy.nodes().removeClass("changed").removeClass("dimmed");
}

async function openTrace(id) {
  currentTraceId = id;
  clearDiff();
  const graph = await api(`/traces/${id}`);
  renderGraph(graph);
  await loadTraces();

  if (ws) ws.close();
  ws = new WebSocket(`ws://${location.host}/ws/traces/${id}`);
  ws.onmessage = (evt) => renderGraph(JSON.parse(evt.data));
}

document.getElementById("new-run-form").addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const input = document.getElementById("new-run-task");
  const task = input.value.trim();
  if (!task) return;
  const { trace_id } = await api("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  });
  input.value = "";
  await loadTraces();
  await openTrace(trace_id);
});

document.getElementById("diff-go").addEventListener("click", () => {
  const a = document.getElementById("diff-a").value;
  const b = document.getElementById("diff-b").value;
  if (a && b) doDiff(a, b);
});

document.getElementById("diff-clear").addEventListener("click", clearDiff);

document.getElementById("detail-close").addEventListener("click", () => {
  document.getElementById("detail-panel").classList.add("hidden");
});

document.getElementById("fork-form").addEventListener("submit", (evt) => {
  evt.preventDefault();
  const atEventId = evt.target.dataset.eventId;
  const mutationText = document.getElementById("fork-mutation").value;
  doFork(atEventId, mutationText);
});

loadTraces();
