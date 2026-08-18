# Version 3 — Flight Recorder: Web DAG Viewer

## Purpose of this document

A complete build prompt for a web UI that renders a recorded trace as a **swimlane DAG**
(one lane per agent, events placed left-to-right by causal rank, edges for message sends
and intra-agent sequence), colors fork state (reused / mutated / live), overlays diffs,
supports forking from the UI, and updates **live** over a websocket while a run is in
progress.

Stack: **FastAPI** backend + **Cytoscape.js** frontend, served as a local dev tool.

Implement exactly to the contracts below. The recorded event store from V2 is **trusted
and treated as read-only**: the only change to verified V2 code is a single `PRAGMA`
line (Section 3). Everything the viewer needs — causal edges, fork roles, diff overlay —
is computed **read-side** from data already in the store. No schema change, no change to
the interceptor, fork, replay, or diff logic.

---

## 1. What already exists (V2, verified — do not modify)

Events are stored with `agent_id`, `event_type`, `seq`, `boundary_hash`, `request_json`,
`response_json`, `vector_clock` (canonical JSON of `{agent_id: int}`), `causal_rank`
(`sum(vector_clock.values())`), `logical_clock`, `wall_clock`, and `event_id`.
`store.get_events(trace_id)` returns events in canonical causal order
(`ORDER BY causal_rank, agent_id, event_type, seq`), deterministic and independent of
thread timing. `store.get_event(event_id)` looks up globally by id. Traces carry
`parent_trace_id`, `branch_point_event`, `mutation`, `task`, `status`, `created_at`.

`flightrec.clock` already provides `happens_before(a: dict, b: dict) -> bool` and
`vc_rank`. The viewer reuses these; it does not reimplement causality.

---

## 2. Architecture

```
flightrec/
  web/
    __init__.py
    server.py     # FastAPI app, endpoints, websocket, uvicorn entry
    graph.py      # build_graph(), edges, fork roles, diff overlay — pure read-side
    schemas.py    # pydantic response/request models
  static/
    index.html
    app.js
    style.css
tests/
  test_web_graph.py   # graph builder, edges, fork roles (offline, fake LLM)
  test_web_api.py     # FastAPI TestClient endpoint tests (offline)
```

New dependencies (add to `pyproject.toml`): `fastapi`, `uvicorn[standard]`, and (dev /
test) `httpx` for `TestClient`. Add a `[project.optional-dependencies].web` group or fold
into main deps.

New CLI command `flightrec serve` (Typer) runs uvicorn against `FLIGHTREC_DB` so the UI
views the same traces the CLI writes. Bind to `127.0.0.1` by default (local tool, no
auth — state this in `--help`).

Design principles:
- Backend endpoints are **synchronous `def`** handlers (FastAPI runs them in a threadpool,
  so blocking SQLite never stalls the event loop). The websocket is `async` and does DB
  reads via `starlette.concurrency.run_in_threadpool`.
- The server holds **one** `Store` instance, shared across requests. `Store` is already
  thread-safe (internal lock + `check_same_thread=False`), so concurrent reads serialize
  safely.
- The graph is derived state. Nodes are events; edges and fork roles are computed, never
  stored.

---

## 3. The one change to verified code

`flightrec/store.py`, in `Store.__init__` (or `init_schema`), enable WAL so a reader
process (the server) sees a writer process's (the CLI `run`) committed events without lock
contention — required for live viewing during an active run:

```python
self.conn.execute("PRAGMA journal_mode=WAL")
```

This is additive and safe; it does not alter any recorded data or ordering. It is the
**only** modification to V2 code.

---

## 4. Backend

### 4.1 `flightrec/web/graph.py` (pure, read-side)

All functions take a `Store` and read committed data only.

**Agent lane order** — deterministic top-to-bottom flow:

```python
def _agent_order(events):
    # first causal appearance; events are already in causal order
    order = []
    for e in events:
        if e.agent_id not in order:
            order.append(e.agent_id)
    return order   # e.g. ["planner", "worker_a", "worker_b", "synthesizer"]
```

**Column packing** — dense-rank distinct `causal_rank` values so concurrent events (equal
rank) share a column and there are no visual gaps, while causal order is preserved:

```python
def _columns(events):
    ranks = sorted({e.causal_rank for e in events})
    return {r: i for i, r in enumerate(ranks)}   # causal_rank -> column index
```

**Sequence edges** — consecutive events within an agent (causal order):

```python
def _sequence_edges(events, agents):
    edges = []
    for a in agents:
        lane = [e for e in events if e.agent_id == a]   # already causal-ordered
        for x, y in zip(lane, lane[1:]):
            edges.append({"from": x.event_id, "to": y.event_id, "kind": "sequence"})
    return edges
```

**Message edges** — computed from `agent_msg` events and vector clocks, no persisted edge
table. The send is the `agent_msg` event (attributed to `from`); the receive is the
recipient's earliest event whose knowledge of the sender reaches the send's own component:

```python
def _message_edges(events):
    by_agent = {}
    for e in events:
        by_agent.setdefault(e.agent_id, []).append(e)   # causal order preserved
    edges = []
    for e in events:
        if e.event_type != "agent_msg":
            continue
        req = json.loads(e.request_json)                 # {"from","to","payload"}
        sender, recipient = e.agent_id, req["to"]
        send_component = json.loads(e.vector_clock).get(sender, 0)
        target = next((r for r in by_agent.get(recipient, [])
                       if json.loads(r.vector_clock).get(sender, 0) >= send_component),
                      None)
        if target is not None:
            edges.append({"from": e.event_id, "to": target.event_id, "kind": "message"})
    return edges
```

This is correct because a recipient merges the sender's full vector at its next tick after
delivery, so the recipient's `vector_clock[sender]` reaches the send's component at exactly
the receiving event. Sends from one agent strictly increase that component, so `>=`
selects the event that received *this* send.

**Fork roles** — derived from the branch vector clock, not from any runtime taint state:

```python
def _fork_context(store, trace):
    if not (trace.parent_trace_id and trace.branch_point_event):
        return None
    be = store.get_event(trace.branch_point_event)       # lives in the parent trace
    return {"key": (be.agent_id, be.event_type, be.seq),
            "vec": json.loads(be.vector_clock)}

def _role(event, ctx):
    if ctx is None:
        return "recorded"
    if (event.agent_id, event.event_type, event.seq) == ctx["key"]:
        return "mutated"
    if happens_before(ctx["vec"], json.loads(event.vector_clock)):
        return "live"       # in the branch's causal future → rerun
    return "reused"         # branch's past or concurrent → replayed/copied
```

`build_graph(store, trace_id) -> dict` assembles: `trace` metadata + `agents`, `nodes`
(one per event with `lane`, `column`, `role`, truncated `request_preview` /
`response_preview` at ~200 chars, plus `vector_clock`, `causal_rank`, `boundary_hash`,
`event_type`, `seq`, `wall_clock`), and `edges` (sequence + message). Return the exact
shape in Section 6.

`diff_overlay(store, trace_a, trace_b) -> dict` reuses `flightrec.diff.diff` and adds a
`changed_keys` list (`[agent_id, event_type, seq]` triples) so the frontend can recolor
changed nodes:

```python
def diff_overlay(store, a, b):
    ea = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(a)}
    eb = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(b)}
    changed = [list(k) for k in (set(ea) | set(eb))
               if k not in ea or k not in eb
               or (ea[k].boundary_hash, ea[k].response_json)
                  != (eb[k].boundary_hash, eb[k].response_json)]
    rep = diff(store, a, b)
    return {"branch_event": list(rep.branch_event) if rep.branch_event else None,
            "changed_by_agent": rep.changed_by_agent,
            "final_a": rep.final_a, "final_b": rep.final_b,
            "changed_keys": changed}
```

### 4.2 `flightrec/web/server.py` (FastAPI)

One module-level `Store` on `FLIGHTREC_DB`. Serve `static/` at `/`.

**A run/fork serialization lock.** The interceptor uses a process-global `_active`
context, so two concurrent record/fork operations in the same process would clobber it.
Guard every operation that runs the agent (`/api/run`, `/api/fork`) with a single
module-level `threading.Lock` so they run one at a time. Read endpoints are unaffected.

Endpoints:

- `GET /api/traces` → `list[TraceSummary]` from `store.list_traces()`.
- `GET /api/traces/{trace_id}` → `GraphResponse` from `build_graph`. `404` if unknown.
- `POST /api/traces/{trace_id}/fork` → body `ForkRequest {at_event_id, mutation}` →
  under the run-lock call `flightrec.fork.fork(store, trace_id, at_event_id, mutation)` →
  `ForkResponse {child_trace_id}`. **Note in the docstring:** a fork reruns the branch's
  causal future *live*, so it needs a real model/API key unless a fake LLM is installed
  (tests install one). `4xx` on bad `at_event_id` / malformed mutation.
- `GET /api/diff/{trace_a}/{trace_b}` → `DiffResponse` from `diff_overlay`.
- `POST /api/run` (optional but recommended) → body `RunRequest {task}` → under the
  run-lock, launch `flightrec.cli.record_run(store, task)` in a background thread and
  return the new `trace_id` immediately so the client can open the live websocket. Needs a
  real key (or fake in tests).
- `WS /ws/traces/{trace_id}` → poll-and-push live graph:

```python
@app.websocket("/ws/traces/{trace_id}")
async def ws_trace(websocket, trace_id):
    await websocket.accept()
    last_sig = None
    try:
        while True:
            graph = await run_in_threadpool(build_graph, store, trace_id)
            sig = (len(graph["nodes"]), len(graph["edges"]), graph["trace"]["status"])
            if sig != last_sig:
                await websocket.send_json(graph)
                last_sig = sig
            if graph["trace"]["status"] in ("complete", "failed"):
                await asyncio.sleep(1.0)     # one more push margin, then keep idle-polling
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
```

The client **replaces** its graph on each push (nodes re-sorted by `column`/`lane`), never
appends — during a live run a late-written event can carry a lower causal rank than an
already-seen one, so incremental append would misplace it.

### 4.3 `flightrec/web/schemas.py`

Pydantic models mirroring Section 6 exactly: `TraceSummary`, `TraceMeta`, `Node`, `Edge`,
`GraphResponse`, `DiffResponse`, `ForkRequest`, `ForkResponse`, `RunRequest`,
`RunResponse`. Use these as endpoint response models so the OpenAPI schema is generated and
the JSON contract is enforced.

---

## 5. Frontend (`static/`)

**Library:** Cytoscape.js with `layout: { name: 'preset' }` — we supply exact positions
(`x = column * COL_W`, `y = lane * LANE_H`), and Cytoscape gives pan/zoom/selection/hover
for free. Load Cytoscape from a CDN in `index.html`.

**Layout constants:** `COL_W ≈ 120`, `LANE_H ≈ 90`. Draw a faint horizontal band and a
left-margin label per agent lane.

**Node styling by `role` (data attribute):**
- `recorded` → neutral slate (root traces).
- `reused` → teal (fork: replayed, copied unchanged).
- `mutated` → amber, thicker border (the branch event).
- `live` → coral (fork: rerun).
- Shape/label: small rounded rectangle or circle labeled with a short `event_type` glyph
  (`llm`, `tool`, `now`, `uuid`/`rand`, `msg`). Full detail on click.

**Edge styling by `kind`:**
- `message` → solid, arrowhead, slightly heavier.
- `sequence` → thin, light gray, arrowhead.

**Diff overlay:** when two traces are compared, nodes whose key is in `changed_keys` get a
distinct ring/outline (e.g. red) layered over their role color; unchanged nodes dim
slightly.

**Interactions (minimum):**
1. Left sidebar lists traces (`GET /api/traces`), showing id, status, and `parent → child`
   lineage. Selecting a trace loads its graph and opens the live websocket.
2. Click a node → detail panel: `agent_id`, `event_type`, `seq`, `vector_clock`,
   `causal_rank`, `boundary_hash`, and request/response previews.
3. From a node's detail panel, a **Fork here** action prompts for mutation JSON, POSTs to
   `/api/traces/{id}/fork`, and on success selects the returned child trace.
4. A **Diff** mode: pick two traces → `GET /api/diff/{a}/{b}` → recolor.
5. `New run` input (if `/api/run` is built): submit a task, receive a `trace_id`, open its
   live websocket, and watch nodes/edges appear as the pipeline executes.

`app.js` structure: `loadTraces()`, `openTrace(id)` (fetch graph, `renderGraph`, open
`ws`), `renderGraph(graph)` (map nodes/edges to Cytoscape elements with preset positions
and `data.role`/`data.kind`, apply the stylesheet), `showDetail(node)`, `doFork(...)`,
`doDiff(a, b)`. Keep it dependency-light (vanilla JS + Cytoscape; no build step).

---

## 6. API JSON contract (exact shapes)

```jsonc
// GET /api/traces
[{ "trace_id": "tr_abc", "task": "...", "status": "complete",
   "parent_trace_id": null, "branch_point_event": null, "created_at": 1710000000.0 }]

// GET /api/traces/{id}   →  GraphResponse
{
  "trace": {
    "trace_id": "tr_abc", "task": "...", "status": "complete",
    "parent_trace_id": null, "branch_point_event": null, "mutation": null,
    "agents": ["planner", "worker_a", "worker_b", "synthesizer"]
  },
  "nodes": [{
    "event_id": "…", "agent_id": "worker_a", "event_type": "tool_call", "seq": 0,
    "lane": 1, "column": 4, "causal_rank": 6, "logical_clock": 4,
    "vector_clock": {"planner": 2, "worker_a": 4},
    "boundary_hash": "…", "wall_clock": 1710000000.5,
    "request_preview": "…(≤200 chars)…", "response_preview": "…(≤200 chars)…",
    "role": "recorded"        // recorded | reused | mutated | live
  }],
  "edges": [
    {"from": "<event_id>", "to": "<event_id>", "kind": "sequence"},
    {"from": "<event_id>", "to": "<event_id>", "kind": "message"}
  ]
}

// POST /api/traces/{id}/fork   body:
{ "at_event_id": "…", "mutation": {"results": ["MUTATED"]} }
//   →
{ "child_trace_id": "tr_def" }

// GET /api/diff/{a}/{b}   →  DiffResponse
{ "branch_event": ["worker_a", "tool_call", 0],
  "changed_by_agent": {"worker_a": 3, "synthesizer": 1},
  "final_a": "…", "final_b": "…",
  "changed_keys": [["worker_a","tool_call",0], ["worker_a","llm_call",0], ...] }

// POST /api/run   body { "task": "…" }  →  { "trace_id": "tr_ghi" }
```

---

## 7. Tests (offline, fake LLM, temp DB)

### `tests/test_web_graph.py`
- Record a trace with the fake LLM; `build_graph`:
  - `agents == ["planner", "worker_a", "worker_b", "synthesizer"]` (or the actual order).
  - node count equals event count; every node has a valid `lane`, `column`, and `role ==
    "recorded"`.
  - sequence edges exist within every agent; message edges connect `planner → each worker`
    and `each worker → synthesizer`.
- Fork at one worker's `tool_call`, then `build_graph(child)`:
  - the branch node's `role == "mutated"`;
  - the **other** worker's nodes are all `role == "reused"`;
  - that worker's suffix (`llm_call`) and the synthesizer node are `role == "live"`.
- `diff_overlay(parent, child)`: `branch_event == [worker, "tool_call", 0]`,
  `changed_keys` non-empty and excludes the untouched worker's keys.

### `tests/test_web_api.py` (FastAPI `TestClient`, fake LLM installed)
- `GET /api/traces` returns the recorded trace.
- `GET /api/traces/{id}` returns a `GraphResponse` matching the recorded event count.
- `POST /api/traces/{id}/fork` (fake LLM → offline suffix) returns a `child_trace_id`, and
  a subsequent `GET /api/traces/{child}` shows the mutated/live/reused roles.
- `GET /api/diff/{parent}/{child}` returns the correct branch and non-empty `changed_keys`.
- `404` for an unknown trace id.

The websocket can be smoke-tested with `TestClient.websocket_connect`: connect to a
completed trace and assert one graph frame is received with the expected node count.

---

## 8. Gotchas (call these out; they are the real traps)

1. **Cross-process SQLite.** The server reads the DB the CLI writes. WAL (Section 3) is
   required; without it a writer's exclusive lock stalls the reader during a live run.
2. **Global interceptor is not reentrant.** `/api/run` and `/api/fork` both drive the
   process-global `_active` interceptor. Serialize them with one server-level lock — never
   run two agent executions concurrently in the one server process.
3. **Live graph is replace, not append.** A late-written event can have a lower causal rank
   than one already shown; the client rebuilds the graph on every websocket frame.
4. **Fork/run make real calls.** The branch's causal future reruns live; document that
   these endpoints need a model/API key, and keep tests offline via the fake LLM.
5. **Don't block the event loop.** Endpoints are sync `def` (threadpool); the async
   websocket wraps DB reads in `run_in_threadpool`.
6. **Fork role comes from vector clocks, not taint.** There is no persisted taint state;
   `happens_before(branch_vec, event_vec)` is the source of truth for `live` vs `reused`.
7. **Message-edge target uses `>=` on the sender's component**, and relies on the
   recipient merging the sender's full vector at its next tick — correct for the
   single-send-per-pair pipeline; if an agent ever sends to the same recipient more than
   once, the strictly-increasing sender component still disambiguates.
8. **Local tool, no auth.** Bind `127.0.0.1`; say so. Do not expose publicly without adding
   auth.

---

## 9. Definition of done

- `pip install -e ".[dev,web]"` succeeds; `flightrec serve` starts on `127.0.0.1:8000`.
- `python -m pytest -v` green: all existing tests plus `test_web_graph.py` and
  `test_web_api.py`.
- Opening the UI on a recorded trace shows the swimlane DAG: planner at top, the two
  workers as parallel lanes at the same columns (visibly concurrent), synthesizer joining
  after both; sequence and message edges rendered distinctly.
- Forking a worker's `tool_call` from the UI produces a child whose DAG shows that worker's
  `tool_call` amber (mutated), its suffix and the synthesizer coral (live), and the **other
  worker's entire lane teal (reused)** — visually confirming the causal fork reused the
  concurrent worker untouched.
- Diffing parent vs child rings exactly the changed nodes.
- Starting a run from the UI (or running `flightrec run` in a terminal) streams nodes and
  edges into an open trace view live.

---

## 10. Documentation

Add a "Web viewer (V3)" section to `README.md`: `flightrec serve`, what the lane colors
mean (recorded / reused / mutated / live), the live-view note, and the local-only caveat.
Note that the viewer is pure read-side over the trusted V2 store (WAL pragma aside), so it
cannot affect recording, replay, or fork correctness.
