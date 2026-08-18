# Flight Recorder V3 — Web DAG Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI + Cytoscape.js web UI that renders a recorded Flight Recorder trace as a swimlane DAG (one lane per agent, events placed by causal rank), colors nodes by fork role (recorded/reused/mutated/live), overlays diffs between two traces, supports forking from the UI, and streams live updates over a websocket while a run is in progress.

**Architecture:** The V2 event store is trusted and read-only. A pure `graph.py` module computes lanes, columns, sequence/message edges, and fork roles entirely from data already in the store (`vector_clock`, `causal_rank`, `agent_msg` payloads) — no schema change, no touch to the interceptor/fork/replay/diff logic. `server.py` (FastAPI) exposes that computation plus thin run/fork actions serialized behind one lock. `static/` (vanilla JS + Cytoscape.js from a CDN, no build step) renders the graph with `layout: 'preset'` using server-supplied `(column, lane)` coordinates.

**Tech Stack:** FastAPI, `uvicorn[standard]`, Cytoscape.js (CDN), Python 3.11, pytest + `httpx` (for `TestClient`). Builds on the existing V2 codebase (Pydantic 2, Typer, SQLite).

**Source spec:** `flightrec_v3_web_dag_viewer_build_prompt.md` (repo root) — authoritative; this plan operationalizes it task-by-task.

## Global Constraints

(Copied verbatim from the spec — every task's changes must satisfy all of these.)

- The V2 event store is **trusted and read-only**. The **only** modification to verified V2 code is one line in `flightrec/store.py` enabling WAL mode. No schema change, no change to the interceptor, fork, replay, or diff logic.
- Everything the viewer needs (causal edges, fork roles, diff overlay) is computed **read-side**, never stored.
- Backend endpoints are **synchronous `def`** handlers (FastAPI runs them in a threadpool). The websocket is `async` and wraps DB reads in `starlette.concurrency.run_in_threadpool`.
- The server holds **one** `Store` instance, shared across requests.
- A single module-level `threading.Lock` serializes every operation that drives the process-global interceptor (`/api/run`, `/api/fork`) — the interceptor's `_active` context is not reentrant, so two concurrent record/fork operations in one process would clobber it. Read endpoints are unaffected.
- Fork roles come from `happens_before(branch_vec, event_vec)` against the branch event's vector clock — **not** from any runtime taint state (there is none at read time; taint only exists during a live fork execution).
- The live graph view **replaces** its whole node/edge set on every websocket push, never appends — a late-written event can carry a lower causal rank than one already shown.
- Bind `127.0.0.1` by default; this is a local, unauthenticated dev tool — state that in `--help` and in the README.
- All new tests run fully offline (fake LLM fixture, temp DB) — `fork`/`run` endpoints make real LLM calls outside tests.

---

## Task 1: WAL pragma + web/dev dependencies

**Files:**
- Modify: `flightrec/store.py:48-53` (the `Store.__init__`/`init_schema` area)
- Modify: `pyproject.toml`
- Test: `tests/test_store.py` (must pass unchanged — do not edit)

**Interfaces:**
- Produces: SQLite connections opened by `Store` now run in WAL journal mode. New installable extras: `web` (`fastapi`, `uvicorn[standard]`) and an updated `dev` extra that adds `httpx` (needed for FastAPI's `TestClient` in later tasks).

- [ ] **Step 1: Add the WAL pragma**

In `flightrec/store.py`, inside `init_schema`, add the pragma right after the connection is confirmed usable (before or after `executescript` — WAL is a durable per-file setting, so once is enough per process, but setting it every `init_schema` call is harmless and simplest):

```python
    def init_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(_SCHEMA)
            self.conn.commit()
            # CREATE TABLE IF NOT EXISTS leaves a pre-existing V1 `events` table
            # (no vector_clock/causal_rank columns) untouched, so a stale V1 db
            # would otherwise fail later with a raw sqlite "no such column" error
            # on the first append. Detect the mismatch here and fail clearly.
            cols = {row["name"] for row in
                    self.conn.execute("PRAGMA table_info(events)").fetchall()}
            missing = {"vector_clock", "causal_rank"} - cols
            if cols and missing:
                raise RuntimeError(
                    f"incompatible V1 database (events table missing {sorted(missing)}). "
                    "Delete the old flightrec.db before running V2, or set FLIGHTREC_DB "
                    "to a fresh path."
                )
```

(Only the `self.conn.execute("PRAGMA journal_mode=WAL")` line is new — everything else in this block already exists; keep it exactly as-is.)

- [ ] **Step 2: Run the store tests**

Run: `python -m pytest tests/test_store.py tests/test_concurrency.py -v`
Expected: PASS, all tests, unchanged files. WAL mode is additive/safe — it changes SQLite's on-disk journal format, not any query result or ordering, so this must not affect any existing assertion.

- [ ] **Step 3: Update `pyproject.toml`**

Replace the `[project.optional-dependencies]` block with:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]
web = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
```

And add a package-data declaration so `flightrec/static/*` ships with the installed package (it's not a Python package itself, so `packages.find` alone won't pick it up):

```toml
[tool.setuptools.package-data]
flightrec = ["static/*"]
```

- [ ] **Step 4: Verify the package still installs**

Run: `python -m pip install -e ".[dev,web]"`
Expected: succeeds, installs `fastapi`, `uvicorn`, `httpx` alongside the existing deps.

- [ ] **Step 5: Commit**

```bash
git add flightrec/store.py pyproject.toml
git commit -m "feat: enable WAL mode for cross-process reads; add web/dev extras"
```

---

## Task 2: `flightrec/web/graph.py` — pure read-side graph builder

**Files:**
- Create: `flightrec/web/__init__.py`
- Create: `flightrec/web/graph.py`
- Test: `tests/test_web_graph.py`

**Interfaces:**
- Consumes: `Store.get_events(trace_id) -> list[Event]`, `Store.get_event(event_id) -> Event | None`, `Store.get_trace(trace_id) -> Trace | None` (all existing, unchanged), `flightrec.clock.happens_before(a: dict, b: dict) -> bool` (existing), `flightrec.diff.diff(store, trace_a, trace_b) -> DiffReport` (existing).
- Produces: `build_graph(store, trace_id) -> dict` (raises `ValueError` if the trace doesn't exist) and `diff_overlay(store, trace_a, trace_b) -> dict`, both returning the exact shapes below — consumed by `server.py` in Task 4.

- [ ] **Step 1: Create the package init**

`flightrec/web/__init__.py`:

```python
"""Read-side web viewer for Flight Recorder traces (FastAPI + Cytoscape.js)."""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_web_graph.py`:

```python
import os
import pytest
from flightrec.store import Store
from flightrec import cli
from flightrec.fork import fork
from flightrec.web.graph import build_graph, diff_overlay


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "g.db"))
    tid = cli.record_run(store, "compare X and Y")
    return store, tid


def _first_tool_event(store, trace_id):
    for e in store.get_events(trace_id):
        if e.event_type == "tool_call":
            return e
    raise AssertionError("no tool_call event")


def test_build_graph_structure(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    g = build_graph(store, tid)

    assert g["trace"]["trace_id"] == tid
    assert g["trace"]["agents"] == ["planner", "worker_a", "worker_b", "synthesizer"]

    events = store.get_events(tid)
    assert len(g["nodes"]) == len(events)
    for n in g["nodes"]:
        assert n["role"] == "recorded"
        assert isinstance(n["lane"], int) and n["lane"] >= 0
        assert isinstance(n["column"], int) and n["column"] >= 0

    kinds = {e["kind"] for e in g["edges"]}
    assert kinds == {"sequence", "message"}

    msg_froms = {e["from"] for e in g["edges"] if e["kind"] == "message"}
    planner_sends = [e for e in events if e.agent_id == "planner" and e.event_type == "agent_msg"]
    worker_sends = [e for e in events if e.agent_id in ("worker_a", "worker_b") and e.event_type == "agent_msg"]
    assert len(planner_sends) == 2
    assert len(worker_sends) == 2
    for send in planner_sends + worker_sends:
        assert send.event_id in msg_froms


def test_build_graph_unknown_trace_raises(tmp_path):
    store = Store(os.path.join(tmp_path, "empty.db"))
    with pytest.raises(ValueError):
        build_graph(store, "tr_nope")


def test_build_graph_fork_roles(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)  # worker_a's tool_call (always ranks first)
    child = fork(store, parent, branch.event_id, {"query": "q", "results": ["MUTATED"]})

    g = build_graph(store, child)
    by_key = {(n["agent_id"], n["event_type"], n["seq"]): n for n in g["nodes"]}

    mutated = by_key[(branch.agent_id, branch.event_type, branch.seq)]
    assert mutated["role"] == "mutated"

    other = "worker_b" if branch.agent_id == "worker_a" else "worker_a"
    other_nodes = [n for n in g["nodes"] if n["agent_id"] == other]
    assert other_nodes
    assert all(n["role"] == "reused" for n in other_nodes)

    mutated_agent_llm = next(n for n in g["nodes"]
                             if n["agent_id"] == branch.agent_id and n["event_type"] == "llm_call")
    assert mutated_agent_llm["role"] == "live"
    synth_node = next(n for n in g["nodes"] if n["agent_id"] == "synthesizer")
    assert synth_node["role"] == "live"


def test_diff_overlay(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    child = fork(store, parent, branch.event_id, {"query": "q", "results": ["MUT"]})

    overlay = diff_overlay(store, parent, child)
    assert overlay["branch_event"] == [branch.agent_id, branch.event_type, branch.seq]
    assert overlay["changed_keys"]
    other = "worker_b" if branch.agent_id == "worker_a" else "worker_a"
    assert all(k[0] != other for k in overlay["changed_keys"])
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_web_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flightrec.web'`.

- [ ] **Step 4: Implement `flightrec/web/graph.py`**

```python
"""Pure, read-side graph construction for the web DAG viewer. Never mutates the store."""
from __future__ import annotations

import json
from typing import Optional

from .. import diff as diff_mod
from ..clock import happens_before
from ..store import Store

_PREVIEW_LEN = 200


def _preview(s: str) -> str:
    return s if len(s) <= _PREVIEW_LEN else s[:_PREVIEW_LEN - 3] + "..."


def _agent_order(events):
    # first causal appearance; events are already in causal order
    order = []
    for e in events:
        if e.agent_id not in order:
            order.append(e.agent_id)
    return order


def _columns(events):
    ranks = sorted({e.causal_rank for e in events})
    return {r: i for i, r in enumerate(ranks)}   # causal_rank -> column index


def _sequence_edges(events, agents):
    edges = []
    for a in agents:
        lane = [e for e in events if e.agent_id == a]   # already causal-ordered
        for x, y in zip(lane, lane[1:]):
            edges.append({"from": x.event_id, "to": y.event_id, "kind": "sequence"})
    return edges


def _message_edges(events):
    by_agent: dict[str, list] = {}
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


def _fork_context(store: Store, trace) -> Optional[dict]:
    if not (trace.parent_trace_id and trace.branch_point_event):
        return None
    be = store.get_event(trace.branch_point_event)       # lives in the parent trace
    return {"key": (be.agent_id, be.event_type, be.seq), "vec": json.loads(be.vector_clock)}


def _role(event, ctx: Optional[dict]) -> str:
    if ctx is None:
        return "recorded"
    if (event.agent_id, event.event_type, event.seq) == ctx["key"]:
        return "mutated"
    if happens_before(ctx["vec"], json.loads(event.vector_clock)):
        return "live"       # in the branch's causal future -> rerun
    return "reused"         # branch's past or concurrent -> replayed/copied


def build_graph(store: Store, trace_id: str) -> dict:
    trace = store.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"no such trace: {trace_id}")

    events = store.get_events(trace_id)
    agents = _agent_order(events)
    lane_of = {a: i for i, a in enumerate(agents)}
    cols = _columns(events)
    ctx = _fork_context(store, trace)

    nodes = [{
        "event_id": e.event_id,
        "agent_id": e.agent_id,
        "event_type": e.event_type,
        "seq": e.seq,
        "lane": lane_of[e.agent_id],
        "column": cols[e.causal_rank],
        "causal_rank": e.causal_rank,
        "logical_clock": e.logical_clock,
        "vector_clock": json.loads(e.vector_clock),
        "boundary_hash": e.boundary_hash,
        "wall_clock": e.wall_clock,
        "request_preview": _preview(e.request_json),
        "response_preview": _preview(e.response_json),
        "role": _role(e, ctx),
    } for e in events]

    edges = _sequence_edges(events, agents) + _message_edges(events)

    return {
        "trace": {
            "trace_id": trace.trace_id,
            "task": trace.task,
            "status": trace.status,
            "parent_trace_id": trace.parent_trace_id,
            "branch_point_event": trace.branch_point_event,
            "mutation": trace.mutation,
            "agents": agents,
        },
        "nodes": nodes,
        "edges": edges,
    }


def diff_overlay(store: Store, trace_a: str, trace_b: str) -> dict:
    ea = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_a)}
    eb = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_b)}
    changed = [list(k) for k in (set(ea) | set(eb))
               if k not in ea or k not in eb
               or (ea[k].boundary_hash, ea[k].response_json)
                  != (eb[k].boundary_hash, eb[k].response_json)]
    rep = diff_mod.diff(store, trace_a, trace_b)
    return {
        "branch_event": list(rep.branch_event) if rep.branch_event else None,
        "changed_by_agent": rep.changed_by_agent,
        "final_a": rep.final_a,
        "final_b": rep.final_b,
        "changed_keys": changed,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_web_graph.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 6: Run the full existing suite to confirm no regression**

Run: `python -m pytest -v`
Expected: PASS, all tests (44 pre-existing + 4 new = 48).

- [ ] **Step 7: Commit**

```bash
git add flightrec/web/__init__.py flightrec/web/graph.py tests/test_web_graph.py
git commit -m "feat: pure read-side graph builder (lanes, columns, edges, fork roles)"
```

---

## Task 3: `flightrec/web/schemas.py` — API contract as Pydantic models

**Files:**
- Create: `flightrec/web/schemas.py`

**Interfaces:**
- Produces: `TraceSummary`, `TraceMeta`, `Node`, `Edge`, `GraphResponse`, `DiffResponse`, `ForkRequest`, `ForkResponse`, `RunRequest`, `RunResponse` — all Pydantic `BaseModel`s, consumed as FastAPI `response_model=`/body-parameter types in Task 4.

- [ ] **Step 1: Implement `flightrec/web/schemas.py`**

```python
"""Pydantic request/response models mirroring the V3 API JSON contract exactly."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    trace_id: str
    task: str
    status: str
    parent_trace_id: Optional[str] = None
    branch_point_event: Optional[str] = None
    created_at: float


class TraceMeta(BaseModel):
    trace_id: str
    task: str
    status: str
    parent_trace_id: Optional[str] = None
    branch_point_event: Optional[str] = None
    mutation: Optional[str] = None
    agents: list[str]


class Node(BaseModel):
    event_id: str
    agent_id: str
    event_type: str
    seq: int
    lane: int
    column: int
    causal_rank: int
    logical_clock: int
    vector_clock: dict[str, int]
    boundary_hash: str
    wall_clock: float
    request_preview: str
    response_preview: str
    role: str


class Edge(BaseModel):
    model_config = {"populate_by_name": True}

    from_: str = Field(..., alias="from")
    to: str
    kind: str


class GraphResponse(BaseModel):
    trace: TraceMeta
    nodes: list[Node]
    edges: list[Edge]


class DiffResponse(BaseModel):
    branch_event: Optional[list] = None
    changed_by_agent: dict[str, int]
    final_a: str
    final_b: str
    changed_keys: list[list]


class ForkRequest(BaseModel):
    at_event_id: str
    mutation: dict


class ForkResponse(BaseModel):
    child_trace_id: str


class RunRequest(BaseModel):
    task: str


class RunResponse(BaseModel):
    trace_id: str
```

Note on `Edge.from_`: `from` is a reserved word in Python, so the field is named `from_` with `alias="from"` — this makes the model both accept and (by FastAPI's default `response_model_by_alias=True`) re-emit the JSON key `"from"`, matching `graph.py`'s plain dicts (`{"from": ..., "to": ..., "kind": ...}`) and the Section 6 contract exactly.

- [ ] **Step 2: Sanity-check the alias round-trips**

Run:
```bash
python -c "
from flightrec.web.schemas import Edge
e = Edge(**{'from': 'a', 'to': 'b', 'kind': 'sequence'})
print(e.model_dump(by_alias=True))
"
```
Expected output: `{'from': 'a', 'to': 'b', 'kind': 'sequence'}`

- [ ] **Step 3: Commit**

```bash
git add flightrec/web/schemas.py
git commit -m "feat: pydantic schemas for the web viewer API contract"
```

---

## Task 4: `flightrec/web/server.py` — FastAPI app, endpoints, websocket

**Files:**
- Create: `flightrec/web/server.py`
- Test: `tests/test_web_api.py`

**Interfaces:**
- Consumes: `graph.build_graph`, `graph.diff_overlay` (Task 2); all `schemas.*` classes (Task 3); existing `Store`, `flightrec.fork.fork(store, trace_id, at_event_id, mutation) -> str`, `flightrec.interceptor.record_into(store, trace_id)` (context manager), `flightrec.agent.reference_agent.run_agent(task) -> dict`, `flightrec.models.Trace`.
- Produces: a module-level FastAPI `app` object and a module-level `store: Store` — both imported by `flightrec/cli.py`'s new `serve` command in Task 5. Route paths and shapes exactly as in spec Section 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_api.py`:

```python
import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(tmp_path, monkeypatch, fake_llm):
    db_path = os.path.join(tmp_path, "api.db")
    monkeypatch.setenv("FLIGHTREC_DB", db_path)
    # server.py builds its module-level Store at import time from FLIGHTREC_DB, so
    # reload it fresh (after the env var is set) for every test to get an isolated db.
    import flightrec.web.server as server_mod
    importlib.reload(server_mod)
    from flightrec import cli
    client = TestClient(server_mod.app)
    return client, server_mod.store, cli


def test_list_and_get_trace(tmp_path, api_client):
    client, store, cli = api_client
    tid = cli.record_run(store, "compare X and Y")

    resp = client.get("/api/traces")
    assert resp.status_code == 200
    assert tid in {t["trace_id"] for t in resp.json()}

    resp = client.get(f"/api/traces/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace"]["trace_id"] == tid
    assert len(body["nodes"]) == len(store.get_events(tid))


def test_get_trace_404(api_client):
    client, store, cli = api_client
    resp = client.get("/api/traces/tr_nope")
    assert resp.status_code == 404


def test_fork_and_diff_endpoints(api_client):
    client, store, cli = api_client
    tid = cli.record_run(store, "compare X and Y")
    branch = next(e for e in store.get_events(tid) if e.event_type == "tool_call")

    resp = client.post(f"/api/traces/{tid}/fork",
                       json={"at_event_id": branch.event_id,
                             "mutation": {"query": "q", "results": ["MUT"]}})
    assert resp.status_code == 200
    child_id = resp.json()["child_trace_id"]

    resp = client.get(f"/api/traces/{child_id}")
    assert resp.status_code == 200
    roles = {n["role"] for n in resp.json()["nodes"]}
    assert {"mutated", "live", "reused"} <= roles

    resp = client.get(f"/api/diff/{tid}/{child_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["branch_event"] == [branch.agent_id, branch.event_type, branch.seq]
    assert body["changed_keys"]


def test_run_endpoint_returns_trace_id_and_completes(api_client):
    client, store, cli = api_client
    resp = client.post("/api/run", json={"task": "compare X and Y"})
    assert resp.status_code == 200
    tid = resp.json()["trace_id"]
    assert store.get_trace(tid) is not None

    # Poll briefly for the background thread to finish (fake LLM makes this fast).
    import time
    for _ in range(50):
        if store.get_trace(tid).status in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert store.get_trace(tid).status == "complete"


def test_websocket_smoke(api_client):
    client, store, cli = api_client
    tid = cli.record_run(store, "compare X and Y")
    with client.websocket_connect(f"/ws/traces/{tid}") as ws:
        frame = ws.receive_json()
        assert frame["trace"]["trace_id"] == tid
        assert len(frame["nodes"]) == len(store.get_events(tid))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flightrec.web.server'`.

- [ ] **Step 3: Implement `flightrec/web/server.py`**

```python
"""FastAPI server for the V3 web DAG viewer. Pure read-side over the trusted V2 store,
plus thin run/fork actions serialized behind one lock. Local dev tool - no auth."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import interceptor as itc
from ..agent.reference_agent import run_agent
from ..fork import fork as fork_agent
from ..models import Trace
from ..store import Store
from . import graph as graph_mod
from .schemas import (DiffResponse, ForkRequest, ForkResponse, GraphResponse,
                      RunRequest, RunResponse, TraceSummary)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

app = FastAPI(title="Flight Recorder viewer")
store = Store(os.environ.get("FLIGHTREC_DB", "flightrec.db"))

# The interceptor's process-global `_active` context is not reentrant: two
# concurrent record/fork executions in this process would clobber each other.
# Every operation that drives the agent (run, fork) is serialized behind this lock.
_run_lock = threading.Lock()


@app.get("/api/traces", response_model=list[TraceSummary])
def list_traces():
    return list(store.list_traces())


@app.get("/api/traces/{trace_id}", response_model=GraphResponse)
def get_trace_graph(trace_id: str):
    try:
        return graph_mod.build_graph(store, trace_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"no such trace: {trace_id}")


@app.post("/api/traces/{trace_id}/fork", response_model=ForkResponse)
def fork_trace(trace_id: str, body: ForkRequest):
    """Reruns the branch's causal future LIVE - needs a real model/API key unless a
    fake LLM is installed (tests install one)."""
    with _run_lock:
        try:
            child_id = fork_agent(store, trace_id, body.at_event_id, body.mutation)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return ForkResponse(child_trace_id=child_id)


@app.get("/api/diff/{trace_a}/{trace_b}", response_model=DiffResponse)
def get_diff(trace_a: str, trace_b: str):
    return graph_mod.diff_overlay(store, trace_a, trace_b)


def _run_in_background(trace_id: str, task: str) -> None:
    with _run_lock:
        try:
            with itc.record_into(store, trace_id):
                run_agent(task)
            store.set_status(trace_id, "complete")
        except Exception:
            store.set_status(trace_id, "failed")


@app.post("/api/run", response_model=RunResponse)
def start_run(body: RunRequest):
    """Launches the agent live in a background thread and returns the trace_id
    immediately so the client can open the live websocket. Needs a real model/API
    key unless a fake LLM is installed (tests install one).

    The trace row is created synchronously (mirroring flightrec.cli.record_run's id
    format) so the id is available before the agent execution - which is what runs
    in the background thread - completes.
    """
    trace_id = "tr_" + uuid.uuid4().hex[:12]
    store.create_trace(Trace(trace_id=trace_id, task=body.task, status="recording",
                             created_at=time.time()))
    threading.Thread(target=_run_in_background, args=(trace_id, body.task),
                     daemon=True).start()
    return RunResponse(trace_id=trace_id)


@app.websocket("/ws/traces/{trace_id}")
async def ws_trace(websocket: WebSocket, trace_id: str):
    await websocket.accept()
    last_sig = None
    try:
        while True:
            try:
                graph = await run_in_threadpool(graph_mod.build_graph, store, trace_id)
            except ValueError:
                await websocket.close(code=4404)
                return
            sig = (len(graph["nodes"]), len(graph["edges"]), graph["trace"]["status"])
            if sig != last_sig:
                await websocket.send_json(graph)
                last_sig = sig
            if graph["trace"]["status"] in ("complete", "failed"):
                await asyncio.sleep(1.0)     # one more push margin, then keep idle-polling
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
```

Note: `app.mount("/", ...)` must stay the **last** line that touches route registration — FastAPI/Starlette matches routes in registration order, so every `@app.get`/`@app.post`/`@app.websocket` above it is matched first; the mount only serves paths that don't match an API route.

- [ ] **Step 4: Run to verify tests pass**

Run: `python -m pytest tests/test_web_api.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `python -m pytest -v`
Expected: PASS, all tests (48 pre-existing + 5 new = 53).

- [ ] **Step 6: Commit**

```bash
git add flightrec/web/server.py tests/test_web_api.py
git commit -m "feat: FastAPI server (traces/graph/fork/diff/run endpoints + live websocket)"
```

---

## Task 5: `flightrec serve` CLI command

**Files:**
- Modify: `flightrec/cli.py`

**Interfaces:**
- Produces: `flightrec serve [--host 127.0.0.1] [--port 8000]` — launches uvicorn against `flightrec.web.server:app`, which reads `FLIGHTREC_DB` at import time (same env var the rest of the CLI uses).

- [ ] **Step 1: Add the command**

In `flightrec/cli.py`, add after the existing `diff` command (before `if __name__ == "__main__":`):

```python
@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Launch the local web DAG viewer at http://host:port. No auth - local tool only,
    do not bind to a public interface."""
    import uvicorn
    uvicorn.run("flightrec.web.server:app", host=host, port=port)
```

- [ ] **Step 2: Verify it's wired up**

Run: `python -c "from flightrec.cli import app; print([c.name for c in app.registered_commands])"`
Expected: output includes `serve` alongside `run`, `ls`, `show`, `replay`, `fork`, `diff`.

- [ ] **Step 3: Run the existing CLI test to confirm no regression**

Run: `python -m pytest tests/test_cli_run_show.py -v`
Expected: PASS, unchanged file.

- [ ] **Step 4: Commit**

```bash
git add flightrec/cli.py
git commit -m "feat: add 'flightrec serve' command to launch the web viewer"
```

---

## Task 6: Frontend (`flightrec/static/`)

**Files:**
- Create: `flightrec/static/index.html`
- Create: `flightrec/static/style.css`
- Create: `flightrec/static/app.js`

**Interfaces:**
- Consumes: the API contract from Task 4 exactly — `GET /api/traces`, `GET /api/traces/{id}`, `POST /api/traces/{id}/fork`, `GET /api/diff/{a}/{b}`, `POST /api/run`, `WS /ws/traces/{id}`.
- No automated test — this task's spec (Section 7) only requires backend tests; verification here is the manual checklist in Task 7's Definition of Done. Keep this dependency-light: vanilla JS + Cytoscape.js from a CDN, no build step.

- [ ] **Step 1: Create `flightrec/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Flight Recorder — DAG Viewer</title>
  <link rel="stylesheet" href="/style.css">
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h1>Flight Recorder</h1>
      <form id="new-run-form">
        <input id="new-run-task" type="text" placeholder="New run: describe a task..." />
        <button type="submit">Run</button>
      </form>
      <div id="diff-picker">
        <select id="diff-a"></select>
        <span>vs</span>
        <select id="diff-b"></select>
        <button id="diff-go" type="button">Diff</button>
        <button id="diff-clear" type="button">Clear</button>
      </div>
      <ul id="trace-list"></ul>
    </aside>
    <main>
      <div id="cy"></div>
    </main>
    <aside id="detail-panel" class="hidden">
      <button id="detail-close" type="button">&times;</button>
      <div id="detail-body"></div>
      <form id="fork-form">
        <h3>Fork here</h3>
        <textarea id="fork-mutation" placeholder='{"results": ["..."]}'></textarea>
        <button type="submit">Fork</button>
      </form>
    </aside>
  </div>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `flightrec/static/style.css`**

```css
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: #14161a; color: #e6e6e6; }
#app { display: flex; height: 100vh; }
#sidebar { width: 260px; padding: 12px; background: #1b1e24; overflow-y: auto; border-right: 1px solid #2a2e37; }
#sidebar h1 { font-size: 16px; margin: 0 0 12px; }
#new-run-form, #diff-picker { display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }
#new-run-form input { flex: 1; min-width: 0; }
#trace-list { list-style: none; padding: 0; margin: 0; }
#trace-list li { padding: 6px 8px; border-radius: 4px; cursor: pointer; font-size: 13px; }
#trace-list li:hover { background: #262a33; }
#trace-list li.selected { background: #2f3a4f; }
#trace-list .status { font-size: 11px; opacity: 0.7; }
main { flex: 1; position: relative; }
#cy { position: absolute; inset: 0; }
#lane-labels { position: absolute; left: 0; top: 0; pointer-events: none; }
#detail-panel { width: 320px; padding: 12px; background: #1b1e24; overflow-y: auto; border-left: 1px solid #2a2e37; }
#detail-panel.hidden { display: none; }
#detail-body dt { font-size: 11px; opacity: 0.6; margin-top: 8px; }
#detail-body dd { margin: 0; font-size: 13px; word-break: break-word; }
#fork-form textarea { width: 100%; height: 60px; margin: 6px 0; }
button { cursor: pointer; }
```

- [ ] **Step 3: Create `flightrec/static/app.js`**

```javascript
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
```

- [ ] **Step 4: Byte-check the files are well-formed**

Run: `python -c "import json,sys; print('ok')"` is not applicable to JS; instead just confirm no syntax errors by loading it in Task 7's manual browser check. There is no automated frontend test in this plan (per spec Section 7, only backend tests are required) — do not invent one.

- [ ] **Step 5: Commit**

```bash
git add flightrec/static/index.html flightrec/static/style.css flightrec/static/app.js
git commit -m "feat: Cytoscape.js swimlane DAG frontend (traces, detail panel, fork, diff, live run)"
```

---

## Task 7: Full regression + manual Definition-of-Done verification

**Files:** none (verification only).

- [ ] **Step 1: Clean install**

Run: `python -m pip install -e ".[dev,web]"`
Expected: succeeds.

- [ ] **Step 2: Full automated suite**

Run: `python -m pytest -v`
Expected: all tests pass (53: the pre-existing 44 + 9 new across Tasks 2 and 4).

- [ ] **Step 3: Launch the server**

```bash
rm -f flightrec.db   # start from a clean V3 db (WAL mode requires a fresh init on first use)
export GROQ_API_KEY=...   # or FLIGHTREC_MODEL + OPENAI_API_KEY - needed for real run/fork
flightrec serve
```

Expected: starts on `http://127.0.0.1:8000` with no errors.

- [ ] **Step 4: Record a trace via the existing CLI, then view it**

In a second terminal:
```bash
flightrec run "Compare Postgres and SQLite for a small app"
```
Open `http://127.0.0.1:8000` in a browser. Confirm:
- The trace appears in the sidebar and clicking it renders a swimlane DAG.
- `planner` is a lane at the top; `worker_a`/`worker_b` occupy parallel lanes at overlapping columns (visibly concurrent, not offset by column); `synthesizer` is a lane whose node(s) appear after both workers'.
- Sequence edges (thin gray) connect consecutive events within a lane; message edges (heavier, arrowed) connect `planner → worker_a`, `planner → worker_b`, `worker_a → synthesizer`, `worker_b → synthesizer`.
- Clicking a node opens the detail panel with `agent_id`, `event_type`, `seq`, `vector_clock`, `causal_rank`, `boundary_hash`, and request/response previews.

- [ ] **Step 5: Fork from the UI**

Click a `worker_a` `tool_call` node → enter a mutation JSON (e.g. `{"results": ["mutated"]}`) in the Fork panel → submit. Confirm:
- A child trace appears and is opened automatically.
- In the child's DAG: the clicked node is amber (`mutated`); that worker's later nodes and the synthesizer are coral (`live`); the **other worker's entire lane is teal (`reused`)** — visually confirming the causal fork left the concurrent worker untouched.

- [ ] **Step 6: Diff parent vs child**

Pick the parent and child trace ids in the diff selectors → click Diff. Confirm the changed nodes (mutated worker + synthesizer) get a red ring/outline and other nodes dim.

- [ ] **Step 7: Live run from the UI**

In the "New run" box, submit a task. Confirm a new trace opens immediately, its websocket connects, and nodes/edges appear progressively as the pipeline executes (rather than all at once at the end).

- [ ] **Step 8: Report results**

No commit needed for this task (verification only) — report the outcome of Steps 3-7 back in conversation so the V3 Definition of Done (spec Section 9) can be confirmed satisfied end-to-end.

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a "Web viewer (V3)" section**

Insert after the "Concurrency & determinism (V2)" section, before "## Tests":

```markdown
## Web viewer (V3)

```
flightrec serve                 # http://127.0.0.1:8000, local only, no auth
```

Renders a recorded trace as a swimlane DAG: one lane per agent, events placed
left-to-right by causal rank. Node color is the event's **fork role**:

- **slate** (`recorded`) — a root trace, no fork context.
- **teal** (`reused`) — fork: past-of or concurrent-with the branch, replayed/copied unchanged.
- **amber** (`mutated`) — the forked event itself.
- **coral** (`live`) — fork: in the branch's causal future, rerun live.

Click a node for its full detail (vector clock, causal rank, request/response), fork
directly from that panel, or pick two traces to diff (changed nodes get a red ring).
Opening a trace also opens a live websocket: while a run is in progress, nodes and
edges stream in as the pipeline executes; each push **replaces** the whole graph
(a late-written event can carry a lower causal rank than one already shown, so
incremental append would misplace it).

The viewer is pure read-side over the trusted V2 store (aside from enabling SQLite's
WAL journal mode, so the server can read a trace while `flightrec run`/`fork` is still
writing it) — it cannot affect recording, replay, or fork correctness. It is a local
dev tool with no authentication: `flightrec serve` binds `127.0.0.1` by default: do
not expose it on a public interface without adding auth.
```

- [ ] **Step 2: Update the Install section**

Add after the existing V2 migration note:

```markdown
V3's web viewer needs the `web` extra: `python -m pip install -e ".[dev,web]"`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the V3 web DAG viewer (flightrec serve)"
```

---

## Self-review notes (from writing this plan)

- **Spec coverage:** Section 2 (architecture/deps) → Task 1 + file layout across Tasks 2-6. Section 3 (WAL, the one V2 change) → Task 1. Section 4.1 (`graph.py`) → Task 2. Section 4.2 (`server.py`, endpoints, websocket, run-lock) → Task 4. Section 4.3 (`schemas.py`) → Task 3. Section 5 (frontend) → Task 6. Section 6 (exact JSON contract) → Tasks 2-4 (dict shapes) and Task 3 (Pydantic mirror). Section 7 (tests) → Tasks 2 and 4. Section 8 (gotchas) → addressed inline (WAL in Task 1, run-lock in Task 4, replace-not-append in Task 6's `renderGraph`, sync `def` handlers + `run_in_threadpool` in Task 4, fork-role-from-vector-clock in Task 2, local-only bind in Tasks 5 and 8). Section 9 (Definition of Done) → Task 7. Section 10 (docs) → Task 8.
- **Resolved spec ambiguity:** Section 4.2 says `/api/run` should "launch `flightrec.cli.record_run(store, task)` in a background thread and return the new trace_id immediately." Taken fully literally this is impossible: `record_run` doesn't expose its generated `trace_id` until the whole (possibly slow, real-LLM) run completes. Task 4 resolves this by creating the trace row synchronously in the endpoint (same `"tr_" + uuid.uuid4().hex[:12]` id format already used identically in `cli.py` and `fork.py`) and running only the interceptor/agent-execution part — the actual heavy lifting — in the background thread. This satisfies the spec's intent (the agent execution is backgrounded; the id is available immediately for opening the websocket) without touching `cli.py` or `record_run` itself, honoring the "only change to verified V2 code is the WAL line" constraint.
- **Placeholder scan:** no TBDs; every step has literal, runnable code or an exact command with an expected result.
- **Type consistency:** `build_graph(store, trace_id) -> dict` and `diff_overlay(store, trace_a, trace_b) -> dict` (Task 2) are called with the same signature in Task 4's endpoints and Task 2's own tests. The `Edge`/`Node`/`GraphResponse`/etc. field names in `schemas.py` (Task 3) match `graph.py`'s dict keys exactly (verified field-by-field against Section 6). `flightrec.fork.fork(store, trace_id, at_event_id, mutation) -> str` (existing, unchanged) is used identically in Task 4's `/fork` endpoint and Task 2's tests.
