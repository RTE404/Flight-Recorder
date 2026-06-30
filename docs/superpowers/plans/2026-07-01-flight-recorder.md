# Flight Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MVP record / replay / time-travel debugger for a sequential multi-agent
LLM system, where a replayed run reproduces the original byte-for-byte and a fork mutates
one recorded value then re-runs the suffix live.

**Architecture:** Agents touch non-determinism only through five boundary functions
(`llm`, `tool_call`, `now`, `new_uuid`, `rand`). A process-global interceptor records every
boundary crossing into an append-only SQLite event store in RECORD mode, and returns the
recorded value in REPLAY mode without performing the real operation. Replay re-runs the
agent under the interceptor and asserts the produced event sequence equals the recording.
Fork copies the recorded prefix, substitutes a mutated value at the branch event, then flips
the interceptor to RECORD so the suffix runs live into a new child trace.

**Tech Stack:** Python 3.11, LiteLLM, stdlib `sqlite3`, Pydantic v2, Typer, pytest.

## Global Constraints

- Python 3.11+ (verified available as `python` → 3.11.7).
- All non-determinism MUST go through `flightrec/boundaries.py`. No direct `datetime.now`,
  `time.time`, `uuid.uuid4`, or `random.*` anywhere in `flightrec/agent/` or `boundaries`'
  agent-facing paths.
- Canonical JSON everywhere:
  `canonical(obj) = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
- `boundary_hash = sha256(canonical(request_json)).hexdigest()`.
- `events` rows are strictly append-only: never UPDATE or DELETE an `events` row. (The
  `traces.status` column is the one allowed mutation, via `Store.set_status`.)
- `wall_clock` is display-only — never used for matching or control flow. Ordering uses
  `seq` and `logical_clock` only; read-back order is insertion order (sqlite rowid).
- Replay must perform zero real LLM/tool calls and cost zero tokens.
- Default model `groq/llama-3.1-8b-instant`, overridable via env `FLIGHTREC_MODEL`. Provider
  key read from env by LiteLLM (Groq free tier: `GROQ_API_KEY`). DB path env `FLIGHTREC_DB`,
  default `flightrec.db`.
- Replay match key: `(agent_id, event_type, seq)`. Determinism/divergence comparison tuple:
  `(agent_id, event_type, seq, boundary_hash, response_json)`.

---

## File Structure

- `pyproject.toml` — package metadata, deps, `flightrec` console script.
- `flightrec/__init__.py` — package marker, version.
- `flightrec/models.py` — `canonical`, `sha256_hex`, Pydantic `Event` and `Trace`.
- `flightrec/store.py` — `Store`: schema, create/read traces, append/read events.
- `flightrec/clock.py` — `LamportClock`.
- `flightrec/interceptor.py` — exceptions, `Interceptor`, `current`, `record_into`,
  `replay_from`; the `cross` method (record/replay/fork-branch logic) and network guard.
- `flightrec/boundaries.py` — `llm`, `tool_call`, `now`, `new_uuid`, `rand`.
- `flightrec/agent/__init__.py` — package marker.
- `flightrec/agent/tools.py` — pure `search(query, seed)` and `run_tool` dispatch.
- `flightrec/agent/reference_agent.py` — planner → worker_a/worker_b → synthesizer.
- `flightrec/replay.py` — `replay(store, trace_id)` + determinism assertion.
- `flightrec/fork.py` — `fork(store, trace_id, at_event_id, mutation)`.
- `flightrec/diff.py` — `diff(store, trace_a, trace_b)` report.
- `flightrec/cli.py` — Typer app: run, ls, show, replay, fork, diff.
- `tests/__init__.py`, `tests/conftest.py` — shared fixtures.
- `tests/test_record_replay.py`, `tests/test_fork.py` — acceptance tests.

---

### Task 1: Project scaffold + models (canonical JSON, hashing, Pydantic)

**Files:**
- Create: `pyproject.toml`, `flightrec/__init__.py`, `flightrec/agent/__init__.py`,
  `tests/__init__.py`, `flightrec/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `canonical(obj: Any) -> str`
  - `sha256_hex(s: str) -> str`
  - `EventType` = `Literal["llm_call","tool_call","clock","random","agent_msg"]`
  - `class Event(BaseModel)`: `event_id: str`, `trace_id: str`, `seq: int`,
    `logical_clock: int`, `wall_clock: float`, `agent_id: str`, `event_type: str`,
    `request_json: str`, `response_json: str`, `boundary_hash: str`
  - `class Trace(BaseModel)`: `trace_id: str`, `parent_trace_id: Optional[str]`,
    `branch_point_event: Optional[str]`, `mutation: Optional[str]`, `task: str`,
    `status: str`, `created_at: float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
import json
from flightrec.models import canonical, sha256_hex, Event, Trace


def test_canonical_is_sorted_and_compact():
    assert canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    # key order in input does not change output
    assert canonical({"a": 2, "b": 1}) == canonical({"b": 1, "a": 2})


def test_canonical_handles_nested_and_unicode():
    assert canonical({"x": ["é", 1, True, None]}) == '{"x":["é",1,true,null]}'


def test_sha256_hex_stable():
    h = sha256_hex(canonical({"a": 1}))
    assert h == sha256_hex(canonical({"a": 1}))
    assert len(h) == 64


def test_event_roundtrips_through_dict():
    e = Event(
        event_id="e1", trace_id="t1", seq=0, logical_clock=1, wall_clock=123.0,
        agent_id="planner", event_type="llm_call",
        request_json='{"a":1}', response_json='{"b":2}', boundary_hash="abc",
    )
    assert Event(**e.model_dump()) == e


def test_trace_optional_fields_default_none():
    t = Trace(trace_id="t1", task="hi", status="recording", created_at=1.0)
    assert t.parent_trace_id is None
    assert t.branch_point_event is None
    assert t.mutation is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec'`.

- [ ] **Step 3: Write scaffold + implementation**

Create `pyproject.toml`:

```toml
[project]
name = "flightrec"
version = "0.1.0"
description = "Record / replay / time-travel debugger for multi-agent systems"
requires-python = ">=3.11"
dependencies = [
    "litellm>=1.40",
    "pydantic>=2.6",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
flightrec = "flightrec.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["flightrec*"]
```

Create `flightrec/__init__.py`:

```python
__version__ = "0.1.0"
```

Create empty `flightrec/agent/__init__.py` and `tests/__init__.py` (zero bytes).

Create `flightrec/models.py`:

```python
"""Pydantic models plus canonical JSON + hashing helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel

EventType = Literal["llm_call", "tool_call", "clock", "random", "agent_msg"]


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, unicode preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Event(BaseModel):
    event_id: str
    trace_id: str
    seq: int
    logical_clock: int
    wall_clock: float
    agent_id: str
    event_type: str
    request_json: str
    response_json: str
    boundary_hash: str


class Trace(BaseModel):
    trace_id: str
    parent_trace_id: Optional[str] = None
    branch_point_event: Optional[str] = None
    mutation: Optional[str] = None
    task: str
    status: str
    created_at: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_models.py -v`
Expected: 5 passed. (The editable install also makes `litellm`/`pydantic`/`typer` available
for later tasks.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml flightrec/__init__.py flightrec/agent/__init__.py tests/__init__.py flightrec/models.py tests/test_models.py
git commit -m "feat: project scaffold + Event/Trace models with canonical JSON"
```

---

### Task 2: SQLite event store

**Files:**
- Create: `flightrec/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Event`, `Trace` from `flightrec.models`.
- Produces `class Store`:
  - `Store(path: str)` — opens connection, calls `init_schema()`.
  - `init_schema() -> None`
  - `create_trace(trace: Trace) -> None`
  - `set_status(trace_id: str, status: str) -> None`
  - `get_trace(trace_id: str) -> Optional[Trace]`
  - `list_traces() -> list[Trace]` (ordered by `created_at`)
  - `append_event(event: Event) -> None`
  - `get_events(trace_id: str) -> list[Event]` (insertion order)
  - `get_event(event_id: str) -> Optional[Event]`
  - `close() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
import os
from flightrec.models import Event, Trace
from flightrec.store import Store


def _store(tmp_path):
    return Store(os.path.join(tmp_path, "t.db"))


def _event(trace_id, seq, lc, etype="llm_call", agent="planner", eid=None):
    return Event(
        event_id=eid or f"{trace_id}-{etype}-{seq}", trace_id=trace_id, seq=seq,
        logical_clock=lc, wall_clock=0.0, agent_id=agent, event_type=etype,
        request_json='{"r":1}', response_json='{"v":2}', boundary_hash="h",
    )


def test_trace_roundtrip(tmp_path):
    s = _store(tmp_path)
    t = Trace(trace_id="t1", task="q", status="recording", created_at=1.0)
    s.create_trace(t)
    assert s.get_trace("t1") == t
    assert s.get_trace("missing") is None


def test_set_status(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    s.set_status("t1", "complete")
    assert s.get_trace("t1").status == "complete"


def test_events_returned_in_insertion_order(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    s.append_event(_event("t1", 0, 1, eid="b"))
    s.append_event(_event("t1", 1, 2, eid="a"))
    got = [e.event_id for e in s.get_events("t1")]
    assert got == ["b", "a"]  # insertion order, NOT event_id order


def test_list_traces_orders_by_created_at(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t2", task="q", status="complete", created_at=2.0))
    s.create_trace(Trace(trace_id="t1", task="q", status="complete", created_at=1.0))
    assert [t.trace_id for t in s.list_traces()] == ["t1", "t2"]


def test_get_event(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    s.append_event(_event("t1", 0, 1, eid="x"))
    assert s.get_event("x").event_id == "x"
    assert s.get_event("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.store'`.

- [ ] **Step 3: Write implementation**

Create `flightrec/store.py`:

```python
"""Append-only SQLite event store."""
from __future__ import annotations

import sqlite3
from typing import Optional

from .models import Event, Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id            TEXT PRIMARY KEY,
    parent_trace_id     TEXT,
    branch_point_event  TEXT,
    mutation            TEXT,
    task                TEXT,
    status              TEXT,
    created_at          REAL
);
CREATE TABLE IF NOT EXISTS events (
    rowid_pk        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT UNIQUE NOT NULL,
    trace_id        TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    logical_clock   INTEGER NOT NULL,
    wall_clock      REAL NOT NULL,
    agent_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    response_json   TEXT NOT NULL,
    boundary_hash   TEXT NOT NULL,
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);
"""

_EVENT_COLS = (
    "event_id, trace_id, seq, logical_clock, wall_clock, agent_id, "
    "event_type, request_json, response_json, boundary_hash"
)
_TRACE_COLS = (
    "trace_id, parent_trace_id, branch_point_event, mutation, task, status, created_at"
)


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def create_trace(self, trace: Trace) -> None:
        self.conn.execute(
            f"INSERT INTO traces ({_TRACE_COLS}) VALUES (?,?,?,?,?,?,?)",
            (trace.trace_id, trace.parent_trace_id, trace.branch_point_event,
             trace.mutation, trace.task, trace.status, trace.created_at),
        )
        self.conn.commit()

    def set_status(self, trace_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE traces SET status = ? WHERE trace_id = ?", (status, trace_id)
        )
        self.conn.commit()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        row = self.conn.execute(
            f"SELECT {_TRACE_COLS} FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return Trace(**dict(row)) if row else None

    def list_traces(self) -> list[Trace]:
        rows = self.conn.execute(
            f"SELECT {_TRACE_COLS} FROM traces ORDER BY created_at, trace_id"
        ).fetchall()
        return [Trace(**dict(r)) for r in rows]

    def append_event(self, event: Event) -> None:
        self.conn.execute(
            f"INSERT INTO events ({_EVENT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (event.event_id, event.trace_id, event.seq, event.logical_clock,
             event.wall_clock, event.agent_id, event.event_type, event.request_json,
             event.response_json, event.boundary_hash),
        )
        self.conn.commit()

    def get_events(self, trace_id: str) -> list[Event]:
        rows = self.conn.execute(
            f"SELECT {_EVENT_COLS} FROM events WHERE trace_id = ? ORDER BY rowid_pk",
            (trace_id,),
        ).fetchall()
        return [Event(**dict(r)) for r in rows]

    def get_event(self, event_id: str) -> Optional[Event]:
        row = self.conn.execute(
            f"SELECT {_EVENT_COLS} FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return Event(**dict(row)) if row else None

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/store.py tests/test_store.py
git commit -m "feat: append-only SQLite event store"
```

---

### Task 3: Lamport clock + interceptor core (record/replay/fork-branch logic)

This is the heart of the engine. It is tested with a synthetic `live_fn` (no real LLM), so
no API key is needed.

**Files:**
- Create: `flightrec/clock.py`, `flightrec/interceptor.py`
- Test: `tests/test_clock.py`, `tests/test_interceptor.py`

**Interfaces:**
- Consumes: `Store`, `Event`, `Trace`, `canonical`, `sha256_hex`.
- Produces `flightrec/clock.py`:
  - `class LamportClock`: attr `value: int` (starts 0); `tick() -> int` (pre-increment,
    return new value); `update(other: int) -> int` (`value = max(value, other) + 1`).
- Produces `flightrec/interceptor.py`:
  - Exceptions `ReplayViolation(Exception)`, `ReplayDrift(Exception)`,
    `NoActiveInterceptor(Exception)`.
  - `RECORD = "record"`, `REPLAY = "replay"`.
  - `class Interceptor` with method
    `cross(agent_id: str, event_type: str, request_obj: Any, live_fn: Callable[[], Any]) -> Any`
    and `guard_real_call() -> None`. Attribute `produced: list[tuple]` of
    `(agent_id, event_type, seq, boundary_hash, response_json)`.
  - `current() -> Interceptor` (raises `NoActiveInterceptor` if none active).
  - context managers `record_into(store, trace_id)` and
    `replay_from(store, trace_id, *, branch_key=None, mutation=None, write_trace_id=None)`.

**Design notes (implementers: read before coding):**
- `cross` reserves `seq = next_seq(agent_id, event_type)` at entry (before `live_fn`), so
  nested calls (none in this MVP, but future-proof) get later seqs.
- `phase` is `RECORD` or `REPLAY`. `record_into` sets phase=RECORD; `replay_from` sets
  phase=REPLAY.
- RECORD phase: call `live_fn()`, lamport `tick()`, write event to `write_trace_id`, append
  to `produced`, return value.
- REPLAY phase: look up recorded event by `(agent_id, event_type, seq)` from `read_trace_id`
  (raise `ReplayDrift` if absent); verify `boundary_hash` of the *current* request equals the
  recorded hash (raise `ReplayDrift` on mismatch); lamport `tick()`.
  - If `branch_key` is set and equals `(agent_id, event_type, seq)`: this is the fork branch.
    Compute the mutated value (`_apply_mutation(json.loads(recorded.response_json), mutation)`),
    write a NEW event (with the mutated `response_json`) to `write_trace_id`, append the
    mutated tuple to `produced`, **flip phase to RECORD**, return mutated value.
  - Else: append the recorded tuple to `produced`, return `json.loads(recorded.response_json)`
    (no write — pure replay writes nothing; fork prefix is pre-copied by `fork.py`).
- `guard_real_call()` raises `ReplayViolation` if `phase == REPLAY`. Boundaries call it
  inside `live_fn` right before any real external op; in normal replay `live_fn` is never
  invoked, so this only fires on an unclamped boundary or a tool that ignores the contract.
- `_apply_mutation(recorded_value, mutation)`: if both `dict`, return `{**recorded, **mutation}`;
  else return `mutation`.
- `_write_event` builds an `Event` with `event_id = uuid4().hex` (event ids are NOT part of
  any comparison), `wall_clock = real time.time()` (display only). **These two are the only
  sanctioned direct uses of `uuid`/`time` in the codebase — they are infrastructure, not
  agent logic, and never affect matching.**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clock.py`:

```python
from flightrec.clock import LamportClock


def test_tick_increments_and_returns():
    c = LamportClock()
    assert c.tick() == 1
    assert c.tick() == 2
    assert c.value == 2


def test_update_takes_max_plus_one():
    c = LamportClock()
    c.tick()  # value = 1
    assert c.update(5) == 6
    assert c.update(2) == 7
```

Create `tests/test_interceptor.py`:

```python
import os
import pytest
from flightrec.models import Trace
from flightrec.store import Store
from flightrec import interceptor as itc


def _store(tmp_path):
    return Store(os.path.join(tmp_path, "t.db"))


def _new_trace(store, trace_id):
    store.create_trace(Trace(trace_id=trace_id, task="q", status="recording", created_at=1.0))


def test_current_raises_when_inactive():
    with pytest.raises(itc.NoActiveInterceptor):
        itc.current()


def test_record_then_replay_returns_recorded_without_live(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        v = itc.current().cross("planner", "clock", {"op": "now"}, lambda: 111.0)
    assert v == 111.0

    # Replay: live_fn raises -> must NOT be called.
    def boom():
        raise AssertionError("live_fn called during replay")

    with itc.replay_from(store, "t1"):
        v2 = itc.current().cross("planner", "clock", {"op": "now"}, boom)
        produced = list(itc.current().produced)
    assert v2 == 111.0
    assert produced[0][:3] == ("planner", "clock", 0)


def test_replay_drift_on_request_mismatch(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        itc.current().cross("planner", "llm_call", {"prompt": "A"}, lambda: {"content": "x"})
    with itc.replay_from(store, "t1"):
        with pytest.raises(itc.ReplayDrift):
            itc.current().cross("planner", "llm_call", {"prompt": "B"}, lambda: {"content": "x"})


def test_guard_real_call_blocks_in_replay(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        itc.current().cross("planner", "clock", {"op": "now"}, lambda: 1.0)
    with itc.replay_from(store, "t1"):
        with pytest.raises(itc.ReplayViolation):
            itc.current().guard_real_call()


def test_seq_increments_per_agent_and_type(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        itc.current().cross("worker_a", "random", {"op": "uuid"}, lambda: "u0")
        itc.current().cross("worker_a", "random", {"op": "rand"}, lambda: 0.5)
        itc.current().cross("worker_a", "clock", {"op": "now"}, lambda: 9.0)
    evs = store.get_events("t1")
    by = [(e.agent_id, e.event_type, e.seq) for e in evs]
    assert by == [("worker_a", "random", 0), ("worker_a", "random", 1), ("worker_a", "clock", 0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clock.py tests/test_interceptor.py -v`
Expected: FAIL with `ModuleNotFoundError` for `flightrec.clock` / `flightrec.interceptor`.

- [ ] **Step 3: Write implementations**

Create `flightrec/clock.py`:

```python
"""Lamport logical clock (concurrency-ready bookkeeping)."""
from __future__ import annotations


class LamportClock:
    def __init__(self) -> None:
        self.value = 0

    def tick(self) -> int:
        self.value += 1
        return self.value

    def update(self, other: int) -> int:
        self.value = max(self.value, other) + 1
        return self.value
```

Create `flightrec/interceptor.py`:

```python
"""Process-global RECORD/REPLAY interceptor: counters, Lamport clock, network guard."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Optional

from .clock import LamportClock
from .models import Event, canonical, sha256_hex
from .store import Store

RECORD = "record"
REPLAY = "replay"


class NoActiveInterceptor(Exception):
    pass


class ReplayViolation(Exception):
    """A real external call was attempted during replay (unclamped boundary)."""


class ReplayDrift(Exception):
    """Replay diverged from the recording (missing event or request hash mismatch)."""


def _apply_mutation(recorded_value: Any, mutation: Any) -> Any:
    if isinstance(recorded_value, dict) and isinstance(mutation, dict):
        return {**recorded_value, **mutation}
    return mutation


class Interceptor:
    def __init__(self, store: Store, *, phase: str, read_trace_id: Optional[str],
                 write_trace_id: Optional[str], branch_key: Optional[tuple] = None,
                 mutation: Any = None):
        self.store = store
        self.phase = phase
        self.read_trace_id = read_trace_id
        self.write_trace_id = write_trace_id
        self.branch_key = branch_key
        self.mutation = mutation
        self.lamport = LamportClock()
        self._counters: dict[tuple[str, str], int] = {}
        self.produced: list[tuple] = []
        self._recorded: dict[tuple[str, str, int], Event] = {}
        if read_trace_id is not None:
            for e in store.get_events(read_trace_id):
                self._recorded[(e.agent_id, e.event_type, e.seq)] = e

    def next_seq(self, agent_id: str, event_type: str) -> int:
        key = (agent_id, event_type)
        seq = self._counters.get(key, 0)
        self._counters[key] = seq + 1
        return seq

    def guard_real_call(self) -> None:
        if self.phase == REPLAY:
            raise ReplayViolation(
                "Attempted a real external call during REPLAY — unclamped boundary."
            )

    def _write_event(self, agent_id, event_type, seq, request_json, response_json,
                     boundary_hash, logical_clock) -> None:
        if self.write_trace_id is None:
            return
        self.store.append_event(Event(
            event_id=uuid.uuid4().hex, trace_id=self.write_trace_id, seq=seq,
            logical_clock=logical_clock, wall_clock=time.time(), agent_id=agent_id,
            event_type=event_type, request_json=request_json, response_json=response_json,
            boundary_hash=boundary_hash,
        ))

    def cross(self, agent_id: str, event_type: str, request_obj: Any,
              live_fn: Callable[[], Any]) -> Any:
        seq = self.next_seq(agent_id, event_type)
        request_json = canonical(request_obj)
        boundary_hash = sha256_hex(request_json)

        if self.phase == RECORD:
            value = live_fn()
            lc = self.lamport.tick()
            response_json = canonical(value)
            self._write_event(agent_id, event_type, seq, request_json, response_json,
                              boundary_hash, lc)
            self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
            return value

        # REPLAY phase
        rec = self._recorded.get((agent_id, event_type, seq))
        if rec is None:
            raise ReplayDrift(
                f"No recorded event for {(agent_id, event_type, seq)} — extra boundary "
                f"crossing during replay."
            )
        if rec.boundary_hash != boundary_hash:
            raise ReplayDrift(
                f"Request drift at {(agent_id, event_type, seq)}: recorded hash "
                f"{rec.boundary_hash[:12]} != live {boundary_hash[:12]}"
            )
        self.lamport.update(rec.logical_clock)

        if self.branch_key is not None and (agent_id, event_type, seq) == self.branch_key:
            value = _apply_mutation(json.loads(rec.response_json), self.mutation)
            response_json = canonical(value)
            self._write_event(agent_id, event_type, seq, request_json, response_json,
                              boundary_hash, rec.logical_clock)
            self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
            self.phase = RECORD  # suffix runs live
            return value

        self.produced.append((agent_id, event_type, seq, boundary_hash, rec.response_json))
        return json.loads(rec.response_json)


_active: Optional[Interceptor] = None


def current() -> Interceptor:
    if _active is None:
        raise NoActiveInterceptor("No active interceptor — use record_into/replay_from.")
    return _active


@contextmanager
def record_into(store: Store, trace_id: str):
    global _active
    prev = _active
    _active = Interceptor(store, phase=RECORD, read_trace_id=None, write_trace_id=trace_id)
    try:
        yield _active
    finally:
        _active = prev


@contextmanager
def replay_from(store: Store, trace_id: str, *, branch_key: Optional[tuple] = None,
                mutation: Any = None, write_trace_id: Optional[str] = None):
    global _active
    prev = _active
    _active = Interceptor(store, phase=REPLAY, read_trace_id=trace_id,
                          write_trace_id=write_trace_id, branch_key=branch_key,
                          mutation=mutation)
    try:
        yield _active
    finally:
        _active = prev
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clock.py tests/test_interceptor.py -v`
Expected: all passed (2 + 5).

- [ ] **Step 5: Commit**

```bash
git add flightrec/clock.py flightrec/interceptor.py tests/test_clock.py tests/test_interceptor.py
git commit -m "feat: Lamport clock + RECORD/REPLAY interceptor with network guard"
```

---

### Task 4: Boundary primitives

**Files:**
- Create: `flightrec/boundaries.py`
- Test: `tests/test_boundaries.py`

**Interfaces:**
- Consumes: `interceptor.current`, `interceptor.guard_real_call`, `agent.tools.run_tool`
  (forward import inside `live_fn`).
- Produces:
  - `DEFAULT_MODEL = "groq/llama-3.1-8b-instant"`
  - `llm(messages: list, *, agent_id: str, **kwargs) -> dict` (returns
    `{"role": "assistant", "content": str}`)
  - `tool_call(name: str, args: dict, *, agent_id: str) -> Any`
  - `now(*, agent_id: str) -> float`
  - `new_uuid(*, agent_id: str) -> str`
  - `rand(*, agent_id: str) -> float`
  - `agent_msg(from_agent: str, to_agent: str, payload: Any) -> Any`

**Design notes:**
- Each boundary builds a `request_obj`, defines a `live_fn` that first calls
  `current().guard_real_call()` then performs the real op, and returns
  `current().cross(agent_id, event_type, request_obj, live_fn)`.
- `llm` request includes `{"model": model, "messages": messages, **kwargs}` so the prompt
  (including any stamped request id / timestamp) is part of `boundary_hash`.
- `llm` normalizes the LiteLLM response to `{"role": "assistant", "content": <str>}`.
- `agent_msg` uses `event_type="agent_msg"`, `agent_id=from_agent`, request/response =
  `{"from": from_agent, "to": to_agent, "payload": payload}`; deterministic `live_fn` returns
  the payload.

- [ ] **Step 1: Write the failing test**

Create `tests/test_boundaries.py`:

```python
import os
import pytest
from flightrec.models import Trace
from flightrec.store import Store
from flightrec import interceptor as itc
from flightrec import boundaries as b


def _ctx(tmp_path):
    store = Store(os.path.join(tmp_path, "t.db"))
    store.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    return store


def test_record_now_uuid_rand_then_replay(tmp_path, monkeypatch):
    store = _ctx(tmp_path)
    monkeypatch.setattr("time.time", lambda: 42.0)
    monkeypatch.setattr("uuid.uuid4", lambda: __import__("uuid").UUID(int=7))
    monkeypatch.setattr("random.random", lambda: 0.25)
    with itc.record_into(store, "t1"):
        n = b.now(agent_id="worker_a")
        u = b.new_uuid(agent_id="worker_a")
        r = b.rand(agent_id="worker_a")
    assert (n, r) == (42.0, 0.25)
    assert u == str(__import__("uuid").UUID(int=7))

    # Replay: break the real primitives; they must never be called.
    monkeypatch.setattr("time.time", lambda: 0.0)
    monkeypatch.setattr("random.random", lambda: 0.0)
    with itc.replay_from(store, "t1"):
        assert b.now(agent_id="worker_a") == 42.0
        assert b.new_uuid(agent_id="worker_a") == str(__import__("uuid").UUID(int=7))
        assert b.rand(agent_id="worker_a") == 0.25


def test_llm_records_and_replays_without_network(tmp_path, monkeypatch):
    store = _ctx(tmp_path)

    class _Msg:
        content = "hello"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr("litellm.completion", fake_completion)
    with itc.record_into(store, "t1"):
        out = b.llm([{"role": "user", "content": "hi"}], agent_id="planner")
    assert out == {"role": "assistant", "content": "hello"}
    assert calls["n"] == 1

    # Replay must not call litellm again.
    def boom(**kwargs):
        raise AssertionError("litellm called during replay")

    monkeypatch.setattr("litellm.completion", boom)
    with itc.replay_from(store, "t1"):
        out2 = b.llm([{"role": "user", "content": "hi"}], agent_id="planner")
    assert out2 == {"role": "assistant", "content": "hello"}
    assert calls["n"] == 1


def test_tool_call_records_and_replays(tmp_path, monkeypatch):
    store = _ctx(tmp_path)
    monkeypatch.setattr("flightrec.agent.tools.run_tool",
                        lambda name, args: {"echo": args})
    with itc.record_into(store, "t1"):
        out = b.tool_call("search", {"query": "x"}, agent_id="worker_a")
    assert out == {"echo": {"query": "x"}}

    monkeypatch.setattr("flightrec.agent.tools.run_tool",
                        lambda name, args: (_ for _ in ()).throw(AssertionError("ran in replay")))
    with itc.replay_from(store, "t1"):
        assert b.tool_call("search", {"query": "x"}, agent_id="worker_a") == {"echo": {"query": "x"}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_boundaries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.boundaries'`.

- [ ] **Step 3: Write implementation**

Create `flightrec/boundaries.py`:

```python
"""The only sanctioned non-deterministic primitives. Agent code MUST use these."""
from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any

from .interceptor import current

DEFAULT_MODEL = "groq/llama-3.1-8b-instant"


def _model() -> str:
    return os.environ.get("FLIGHTREC_MODEL", DEFAULT_MODEL)


def llm(messages: list, *, agent_id: str, **kwargs) -> dict:
    model = _model()
    request = {"model": model, "messages": messages, **kwargs}

    def live():
        current().guard_real_call()
        import litellm
        resp = litellm.completion(model=model, messages=messages, **kwargs)
        content = resp.choices[0].message.content or ""
        return {"role": "assistant", "content": content}

    return current().cross(agent_id, "llm_call", request, live)


def tool_call(name: str, args: dict, *, agent_id: str) -> Any:
    request = {"name": name, "args": args}

    def live():
        current().guard_real_call()
        from .agent import tools
        return tools.run_tool(name, args)

    return current().cross(agent_id, "tool_call", request, live)


def now(*, agent_id: str) -> float:
    def live():
        current().guard_real_call()
        return time.time()

    return current().cross(agent_id, "clock", {"op": "now"}, live)


def new_uuid(*, agent_id: str) -> str:
    def live():
        current().guard_real_call()
        return str(uuid.uuid4())

    return current().cross(agent_id, "random", {"op": "uuid"}, live)


def rand(*, agent_id: str) -> float:
    def live():
        current().guard_real_call()
        return random.random()

    return current().cross(agent_id, "random", {"op": "rand"}, live)


def agent_msg(from_agent: str, to_agent: str, payload: Any) -> Any:
    request = {"from": from_agent, "to": to_agent, "payload": payload}
    return current().cross(from_agent, "agent_msg", request, lambda: payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_boundaries.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/boundaries.py tests/test_boundaries.py
git commit -m "feat: boundary primitives (llm, tool_call, now, new_uuid, rand, agent_msg)"
```

---

### Task 5: Mock tool + reference agent

**Files:**
- Create: `flightrec/agent/tools.py`, `flightrec/agent/reference_agent.py`
- Test: `tests/test_reference_agent.py`

**Interfaces:**
- Consumes: all of `flightrec.boundaries`.
- Produces `flightrec/agent/tools.py`:
  - `search(query: str, seed: float) -> dict` — pure, deterministic given `(query, seed)`;
    returns `{"query": query, "results": [str, str, str]}`.
  - `run_tool(name: str, args: dict) -> Any` — dispatch; `search` reads `args["query"]` and
    `args["seed"]`.
- Produces `flightrec/agent/reference_agent.py`:
  - `run_agent(task: str) -> dict` — executes planner → worker_a → worker_b → synthesizer
    under whatever interceptor is active; returns
    `{"task": task, "plan": dict, "answers": {"worker_a": str, "worker_b": str}, "final": str}`.

**Design notes (the nondeterminism reconciliation):**
- The tool is a **pure function of `(query, seed)`** with NO internal boundary calls. Each
  worker draws its `seed` via `boundaries.rand()` (a recorded `random` event) and passes it
  into `tool_call`. This is how "tool nondeterminism flows through boundaries" while keeping
  `tool_call` a clean short-circuiting black box — nested boundary events inside a
  short-circuited tool would otherwise be orphaned in replay and break byte-identical replay.
- Each worker also draws `req_id = boundaries.new_uuid()` and `ts = boundaries.now()` and
  **stamps both into its extract-LLM prompt**, so an unclamped clock/uuid surfaces as a
  `boundary_hash` mismatch (ReplayDrift) on that `llm_call`.
- `agent_msg` recorded planner→worker_a, planner→worker_b, worker_a→synthesizer,
  worker_b→synthesizer.
- Planner prompt instructs the model to reply with a JSON object
  `{"sub_questions": ["...", "..."]}`. Parse defensively: if JSON parse fails, fall back to
  two derived sub-questions so the pipeline never crashes (the recorded `llm_call` response
  is fixed at replay anyway).

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference_agent.py` (record with a fake LLM — no API key needed — then
assert structure and replay equality):

```python
import json
import os
import pytest
from flightrec.models import Trace
from flightrec.store import Store
from flightrec import interceptor as itc
from flightrec.agent.tools import search, run_tool
from flightrec.agent.reference_agent import run_agent


def test_search_is_pure_given_seed():
    a = search("dogs", 0.5)
    b = search("dogs", 0.5)
    c = search("dogs", 0.6)
    assert a == b
    assert a != c
    assert a["query"] == "dogs" and len(a["results"]) == 3


def test_run_tool_dispatch():
    out = run_tool("search", {"query": "x", "seed": 0.1})
    assert out["query"] == "x"
    with pytest.raises(ValueError):
        run_tool("nope", {})


def _fake_llm(monkeypatch):
    class _Msg:
        def __init__(self, c): self.content = c

    class _Choice:
        def __init__(self, c): self.message = _Msg(c)

    class _Resp:
        def __init__(self, c): self.choices = [_Choice(c)]

    def completion(model, messages, **kwargs):
        # Planner asked for JSON sub_questions; everyone else gets a canned answer.
        last = messages[-1]["content"]
        if "sub_questions" in last:
            return _Resp(json.dumps({"sub_questions": ["q-a", "q-b"]}))
        return _Resp("answer:" + str(len(last)))

    monkeypatch.setattr("litellm.completion", completion)


def test_agent_records_all_four_boundary_types(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    store = Store(os.path.join(tmp_path, "t.db"))
    store.create_trace(Trace(trace_id="t1", task="Q", status="recording", created_at=1.0))
    with itc.record_into(store, "t1"):
        out = run_agent("Q")
    types = {e.event_type for e in store.get_events("t1")}
    assert {"llm_call", "tool_call", "clock", "random", "agent_msg"} <= types
    assert out["final"]
    assert set(out["answers"]) == {"worker_a", "worker_b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reference_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.agent.tools'`.

- [ ] **Step 3: Write implementations**

Create `flightrec/agent/tools.py`:

```python
"""Mock tools. Pure functions; all randomness is supplied by the caller via a recorded seed."""
from __future__ import annotations

import random
from typing import Any


def search(query: str, seed: float) -> dict:
    """Deterministic given (query, seed). Different seeds yield different results, so a
    live run (random seed) never reproduces without record/replay."""
    rng = random.Random(f"{query}|{seed}")
    pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    picks = rng.sample(pool, 3)
    results = [f"{query}: {p} ({rng.randint(1000, 9999)})" for p in picks]
    return {"query": query, "results": results}


def run_tool(name: str, args: dict) -> Any:
    if name == "search":
        return search(args["query"], args["seed"])
    raise ValueError(f"unknown tool: {name}")
```

Create `flightrec/agent/reference_agent.py`:

```python
"""Sequential reference pipeline: planner -> worker_a / worker_b -> synthesizer."""
from __future__ import annotations

import json

from .. import boundaries as b

PLANNER = "planner"
SYNTH = "synthesizer"


def _plan(task: str) -> dict:
    prompt = (
        "You are a planner. Break the task into exactly two sub-questions. "
        'Reply ONLY with JSON: {"sub_questions": ["...", "..."]}.\n\nTask: ' + task
    )
    resp = b.llm([{"role": "user", "content": prompt}], agent_id=PLANNER)
    try:
        data = json.loads(resp["content"])
        subs = list(data["sub_questions"])[:2]
        if len(subs) != 2:
            raise ValueError
    except Exception:
        subs = [f"What is essential about: {task}?", f"What are the risks of: {task}?"]
    return {"sub_questions": subs}


def _work(agent_id: str, sub_question: str) -> str:
    req_id = b.new_uuid(agent_id=agent_id)
    ts = b.now(agent_id=agent_id)
    seed = b.rand(agent_id=agent_id)
    results = b.tool_call("search", {"query": sub_question, "seed": seed}, agent_id=agent_id)
    prompt = (
        f"request_id={req_id} ts={ts}\n"
        f"Using these search results, answer the question in one sentence.\n"
        f"Question: {sub_question}\nResults: {json.dumps(results['results'])}"
    )
    resp = b.llm([{"role": "user", "content": prompt}], agent_id=agent_id)
    return resp["content"]


def run_agent(task: str) -> dict:
    plan = _plan(task)
    sub_a, sub_b = plan["sub_questions"]

    b.agent_msg(PLANNER, "worker_a", {"sub_question": sub_a})
    ans_a = _work("worker_a", sub_a)
    b.agent_msg("worker_a", SYNTH, {"answer": ans_a})

    b.agent_msg(PLANNER, "worker_b", {"sub_question": sub_b})
    ans_b = _work("worker_b", sub_b)
    b.agent_msg("worker_b", SYNTH, {"answer": ans_b})

    prompt = (
        "Combine these two answers into a final response.\n"
        f"A: {ans_a}\nB: {ans_b}"
    )
    final = b.llm([{"role": "user", "content": prompt}], agent_id=SYNTH)["content"]
    return {"task": task, "plan": plan, "answers": {"worker_a": ans_a, "worker_b": ans_b},
            "final": final}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reference_agent.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/agent/tools.py flightrec/agent/reference_agent.py tests/test_reference_agent.py
git commit -m "feat: pure mock search tool + sequential reference agent"
```

---

### Task 6: CLI `run` + `ls` + `show` (live recording)

**Files:**
- Create: `flightrec/cli.py`
- Test: `tests/test_cli_run_show.py`

**Interfaces:**
- Consumes: `Store`, `Trace`, `interceptor.record_into`, `reference_agent.run_agent`.
- Produces:
  - `_db() -> Store` (helper reading `FLIGHTREC_DB`, default `flightrec.db`).
  - `_new_trace_id() -> str`.
  - `record_run(store: Store, task: str) -> str` — creates a `recording` trace, runs the
    agent under `record_into`, sets status `complete` (or `failed` on exception), returns
    `trace_id`. **This function is the testable core; the Typer command is a thin wrapper.**
  - Typer `app` with commands `run`, `ls`, `show`.

**Design notes:**
- `_new_trace_id` / `created_at` may use `uuid`/`time` directly — CLI infrastructure, not
  agent logic, never part of matching.
- `show` prints, per event in order: `seq logical_clock agent_id event_type event_id` then a
  truncated response.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_run_show.py`:

```python
import json
import os
import pytest
from flightrec import cli
from flightrec.store import Store


def _fake_llm(monkeypatch):
    class _R:
        def __init__(self, c):
            self.choices = [type("C", (), {"message": type("M", (), {"content": c})()})()]

    def completion(model, messages, **kwargs):
        if "sub_questions" in messages[-1]["content"]:
            return _R(json.dumps({"sub_questions": ["a", "b"]}))
        return _R("ok")

    monkeypatch.setattr("litellm.completion", completion)


def test_record_run_creates_complete_trace(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    db = os.path.join(tmp_path, "f.db")
    store = Store(db)
    tid = cli.record_run(store, "What is X?")
    t = store.get_trace(tid)
    assert t.status == "complete"
    assert t.task == "What is X?"
    assert len(store.get_events(tid)) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_run_show.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.cli'` (or `AttributeError`).

- [ ] **Step 3: Write implementation**

Create `flightrec/cli.py`:

```python
"""Typer CLI for Flight Recorder."""
from __future__ import annotations

import os
import time
import uuid

import typer

from .models import Trace
from .store import Store
from . import interceptor as itc
from .agent.reference_agent import run_agent

app = typer.Typer(add_completion=False, help="Record / replay / time-travel debugger.")


def _db() -> Store:
    return Store(os.environ.get("FLIGHTREC_DB", "flightrec.db"))


def _new_trace_id() -> str:
    return "tr_" + uuid.uuid4().hex[:12]


def record_run(store: Store, task: str) -> str:
    trace_id = _new_trace_id()
    store.create_trace(Trace(trace_id=trace_id, task=task, status="recording",
                             created_at=time.time()))
    try:
        with itc.record_into(store, trace_id):
            run_agent(task)
        store.set_status(trace_id, "complete")
    except Exception:
        store.set_status(trace_id, "failed")
        raise
    return trace_id


@app.command()
def run(task: str):
    """Run the reference agent live, record everything, print the trace id."""
    store = _db()
    trace_id = record_run(store, task)
    typer.echo(trace_id)


@app.command()
def ls():
    """List traces."""
    store = _db()
    for t in store.list_traces():
        parent = t.parent_trace_id or "-"
        typer.echo(f"{t.trace_id}  parent={parent}  {t.status:9}  {t.task[:50]}")


@app.command()
def show(trace_id: str):
    """Print the event log for a trace."""
    store = _db()
    if store.get_trace(trace_id) is None:
        typer.echo(f"no such trace: {trace_id}", err=True)
        raise typer.Exit(1)
    for e in store.get_events(trace_id):
        resp = e.response_json if len(e.response_json) <= 70 else e.response_json[:67] + "..."
        typer.echo(f"seq={e.seq} lc={e.logical_clock} {e.agent_id:11} "
                   f"{e.event_type:9} {e.event_id}  {resp}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_run_show.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/cli.py tests/test_cli_run_show.py
git commit -m "feat: CLI run/ls/show with record_run core"
```

---

### Task 7: Replay engine + determinism assertion + drift acceptance test

**Files:**
- Create: `flightrec/replay.py`, `tests/conftest.py`
- Modify: `flightrec/cli.py` (add `replay` command)
- Test: `tests/test_record_replay.py`

**Interfaces:**
- Consumes: `Store`, `interceptor.replay_from`, `reference_agent.run_agent`,
  `store.get_events`.
- Produces:
  - `class DeterminismError(Exception)`.
  - `replay(store: Store, trace_id: str) -> list[tuple]` — re-runs the agent under
    `replay_from(store, trace_id)`, then asserts the interceptor's `produced` list equals the
    recorded sequence compared on `(agent_id, event_type, seq, boundary_hash, response_json)`.
    Raises `DeterminismError` with the first diverging pair. Returns the produced list.
  - `recorded_tuples(store, trace_id) -> list[tuple]` helper.
- Modify `cli.py`: add `replay` command printing a green check or the drift error.

**Design notes:**
- `replay` reads recorded events first (for the expected list), then runs under
  `replay_from`. After the run, compare element-by-element. The agent's control flow during
  replay is driven entirely by recorded values, so the produced list must equal recorded.
- The drift test deliberately bypasses a boundary (uses real `time.time` stamped into the
  worker prompt). Because the worker stamps `ts` into the extract-LLM prompt, the `llm_call`
  request differs at replay → `ReplayDrift` is raised from inside `cross` *before* the
  determinism comparison. The test accepts either `ReplayDrift` or `DeterminismError` as
  "fails loudly" (both are subclasses-free distinct exceptions; assert it raises one of them).

- [ ] **Step 1: Write `conftest.py` + the failing acceptance tests**

Create `tests/conftest.py`:

```python
import json
import pytest


@pytest.fixture
def fake_llm(monkeypatch):
    """Deterministic fake LiteLLM so recording needs no API key."""
    def _install():
        class _R:
            def __init__(self, c):
                self.choices = [type("C", (), {"message": type("M", (), {"content": c})()})()]

        def completion(model, messages, **kwargs):
            if "sub_questions" in messages[-1]["content"]:
                return _R(json.dumps({"sub_questions": ["qa", "qb"]}))
            return _R("answer-" + str(len(messages[-1]["content"])))

        monkeypatch.setattr("litellm.completion", completion)
    _install()
    return monkeypatch
```

Create `tests/test_record_replay.py`:

```python
import os
import pytest
from flightrec.store import Store
from flightrec import cli
from flightrec import interceptor as itc
from flightrec.replay import replay, recorded_tuples, DeterminismError
from flightrec.agent.reference_agent import run_agent


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "f.db"))
    tid = cli.record_run(store, "Explain caching")
    return store, tid


def test_recording_covers_all_four_boundary_types(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    types = {e.event_type for e in store.get_events(tid)}
    assert {"llm_call", "tool_call", "clock", "random"} <= types


def test_replay_is_byte_identical(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    produced = replay(store, tid)
    assert produced == recorded_tuples(store, tid)


def test_replay_makes_zero_real_calls(tmp_path, fake_llm, monkeypatch):
    store, tid = _record(tmp_path, fake_llm)

    def boom_llm(**kwargs):
        raise AssertionError("litellm called during replay")

    monkeypatch.setattr("litellm.completion", boom_llm)
    monkeypatch.setattr("flightrec.agent.tools.run_tool",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tool in replay")))
    produced = replay(store, tid)  # must not raise
    assert produced == recorded_tuples(store, tid)


def test_unclamped_clock_makes_replay_fail_loudly(tmp_path, fake_llm, monkeypatch):
    """Bypass the clock boundary in the worker -> ts in the LLM prompt drifts -> raise."""
    store = Store(os.path.join(tmp_path, "f.db"))
    import flightrec.agent.reference_agent as ra
    import time as _t
    # Patch the worker to stamp REAL time instead of the recorded boundary.
    real_now = {"v": 1000.0}

    def patched_work(agent_id, sub_question):
        import json
        from flightrec import boundaries as b
        req_id = b.new_uuid(agent_id=agent_id)
        real_now["v"] += 1.0
        ts = real_now["v"]  # UNCLAMPED on purpose
        seed = b.rand(agent_id=agent_id)
        results = b.tool_call("search", {"query": sub_question, "seed": seed}, agent_id=agent_id)
        prompt = (f"request_id={req_id} ts={ts}\nUsing these search results, answer.\n"
                  f"Question: {sub_question}\nResults: {json.dumps(results['results'])}")
        resp = b.llm([{"role": "user", "content": prompt}], agent_id=agent_id)
        return resp["content"]

    monkeypatch.setattr(ra, "_work", patched_work)
    tid = cli.record_run(store, "drift demo")
    with pytest.raises((itc.ReplayDrift, DeterminismError)):
        replay(store, tid)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_record_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.replay'`.

- [ ] **Step 3: Write implementation**

Create `flightrec/replay.py`:

```python
"""Faithful replay + determinism assertion."""
from __future__ import annotations

from .store import Store
from . import interceptor as itc
from .agent.reference_agent import run_agent


class DeterminismError(Exception):
    pass


def recorded_tuples(store: Store, trace_id: str) -> list[tuple]:
    return [
        (e.agent_id, e.event_type, e.seq, e.boundary_hash, e.response_json)
        for e in store.get_events(trace_id)
    ]


def replay(store: Store, trace_id: str) -> list[tuple]:
    trace = store.get_trace(trace_id)
    if trace is None:
        raise DeterminismError(f"no such trace: {trace_id}")
    expected = recorded_tuples(store, trace_id)
    with itc.replay_from(store, trace_id) as inter:
        run_agent(trace.task)
        produced = list(inter.produced)
    if produced != expected:
        n = min(len(produced), len(expected))
        for i in range(n):
            if produced[i] != expected[i]:
                raise DeterminismError(
                    f"replay drift at index {i}:\n  recorded={expected[i]}\n  replayed={produced[i]}"
                )
        raise DeterminismError(
            f"length mismatch: recorded {len(expected)} events, replayed {len(produced)}"
        )
    return produced
```

Modify `flightrec/cli.py`: add the import and command (insert after the `show` command):

```python
from .replay import replay as _replay, DeterminismError


@app.command()
def replay(trace_id: str):
    """Faithful replay + determinism assertion."""
    store = _db()
    try:
        produced = _replay(store, trace_id)
    except (DeterminismError, itc.ReplayDrift, itc.ReplayViolation) as exc:
        typer.echo(f"✗ DRIFT: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"✓ replay deterministic: {len(produced)} events reproduced, 0 real calls")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_record_replay.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/replay.py flightrec/cli.py tests/conftest.py tests/test_record_replay.py
git commit -m "feat: replay engine with determinism assertion + drift detection"
```

---

### Task 8: Fork (branch manager)

**Files:**
- Create: `flightrec/fork.py`
- Modify: `flightrec/cli.py` (add `fork` command)
- Test: `tests/test_fork.py` (fork portion)

**Interfaces:**
- Consumes: `Store`, `Event`, `Trace`, `interceptor.replay_from`, `reference_agent.run_agent`,
  `models.canonical`.
- Produces:
  - `fork(store: Store, trace_id: str, at_event_id: str, mutation: Any) -> str` — returns new
    child `trace_id`.
- Modify `cli.py`: add `fork` command with `--at` and `--set` (JSON string) options.

**Design notes — the fork algorithm:**
1. Load parent trace (raise if missing) and the branch event via `store.get_event`
   (raise if missing / wrong trace).
2. `branch_key = (branch.agent_id, branch.event_type, branch.seq)`.
3. Create child `Trace(parent_trace_id=trace_id, branch_point_event=at_event_id,
   mutation=canonical(mutation), task=parent.task, status="recording", created_at=time.time())`.
4. **Copy the prefix**: every parent event with `logical_clock < branch.logical_clock`,
   re-inserted into the child with a fresh `event_id` (`uuid4().hex`) and all other fields
   (seq, logical_clock, wall_clock, agent_id, event_type, request_json, response_json,
   boundary_hash) preserved. The branch event itself is NOT copied — it is written fresh
   (mutated) during re-execution.
5. Re-run the agent under
   `replay_from(store, trace_id, branch_key=branch_key, mutation=mutation, write_trace_id=child_id)`.
   The interceptor replays the prefix (no writes), writes the mutated branch event to the
   child and flips to RECORD, then records the live suffix into the child.
6. Set child status `complete` (or `failed`), return child id.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fork.py`:

```python
import json
import os
import pytest
from flightrec.store import Store
from flightrec import cli
from flightrec.fork import fork
from flightrec.replay import recorded_tuples


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "f.db"))
    tid = cli.record_run(store, "compare X and Y")
    return store, tid


def _first_tool_event(store, trace_id):
    for e in store.get_events(trace_id):
        if e.event_type == "tool_call":
            return e
    raise AssertionError("no tool_call event")


def test_fork_shares_prefix_and_diverges_at_branch(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    new_results = {"query": json.loads(branch.response_json)["query"],
                   "results": ["MUTATED-1", "MUTATED-2", "MUTATED-3"]}
    child = fork(store, parent, branch.event_id, new_results)

    p = store.get_events(parent)
    c = store.get_events(child)
    # Identical prefix up to (not including) the branch event.
    branch_idx = next(i for i, e in enumerate(p) if e.event_id == branch.event_id)
    for i in range(branch_idx):
        assert (p[i].agent_id, p[i].event_type, p[i].seq, p[i].boundary_hash,
                p[i].response_json) == (c[i].agent_id, c[i].event_type, c[i].seq,
                                        c[i].boundary_hash, c[i].response_json)
    # Branch event: same request, mutated response.
    assert c[branch_idx].boundary_hash == p[branch_idx].boundary_hash
    assert json.loads(c[branch_idx].response_json)["results"] == ["MUTATED-1", "MUTATED-2", "MUTATED-3"]


def test_fork_suffix_is_live_and_recorded(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    child = fork(store, parent, branch.event_id, {"results": ["Z"], "query": "q"})
    t = store.get_trace(child)
    assert t.parent_trace_id == parent
    assert t.branch_point_event == branch.event_id
    assert t.status == "complete"
    # Suffix exists: child has events after the branch index.
    c = store.get_events(child)
    branch_idx = next(i for i, e in enumerate(c)
                      if (e.agent_id, e.event_type, e.seq) ==
                      (branch.agent_id, branch.event_type, branch.seq))
    assert branch_idx < len(c) - 1  # at least one live suffix event
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fork.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.fork'`.

- [ ] **Step 3: Write implementation**

Create `flightrec/fork.py`:

```python
"""Fork a recorded trace: copy prefix, mutate the branch event, run the suffix live."""
from __future__ import annotations

import time
import uuid
from typing import Any

from .models import Event, Trace, canonical
from .store import Store
from . import interceptor as itc
from .agent.reference_agent import run_agent


def _new_child_id() -> str:
    return "tr_" + uuid.uuid4().hex[:12]


def fork(store: Store, trace_id: str, at_event_id: str, mutation: Any) -> str:
    parent = store.get_trace(trace_id)
    if parent is None:
        raise ValueError(f"no such trace: {trace_id}")
    branch = store.get_event(at_event_id)
    if branch is None or branch.trace_id != trace_id:
        raise ValueError(f"event {at_event_id} not found in trace {trace_id}")

    branch_key = (branch.agent_id, branch.event_type, branch.seq)
    child_id = _new_child_id()
    store.create_trace(Trace(
        trace_id=child_id, parent_trace_id=trace_id, branch_point_event=at_event_id,
        mutation=canonical(mutation), task=parent.task, status="recording",
        created_at=time.time(),
    ))

    # Copy the faithful prefix (strictly before the branch event).
    for e in store.get_events(trace_id):
        if e.logical_clock < branch.logical_clock:
            store.append_event(Event(
                event_id=uuid.uuid4().hex, trace_id=child_id, seq=e.seq,
                logical_clock=e.logical_clock, wall_clock=e.wall_clock, agent_id=e.agent_id,
                event_type=e.event_type, request_json=e.request_json,
                response_json=e.response_json, boundary_hash=e.boundary_hash,
            ))

    try:
        with itc.replay_from(store, trace_id, branch_key=branch_key, mutation=mutation,
                             write_trace_id=child_id):
            run_agent(parent.task)
        store.set_status(child_id, "complete")
    except Exception:
        store.set_status(child_id, "failed")
        raise
    return child_id
```

Modify `flightrec/cli.py`: add import and command (after `replay`):

```python
import json as _json
from .fork import fork as _fork


@app.command()
def fork(trace_id: str,
         at: str = typer.Option(..., "--at", help="event_id to fork at"),
         set: str = typer.Option(..., "--set", help="JSON mutation for the branch event")):
    """Fork a trace: mutate one recorded value at --at, run the suffix live."""
    store = _db()
    mutation = _json.loads(set)
    child = _fork(store, trace_id, at, mutation)
    typer.echo(child)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fork.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/fork.py flightrec/cli.py tests/test_fork.py
git commit -m "feat: fork with prefix copy, branch mutation, live suffix"
```

---

### Task 9: Diff + CLI diff + fork-diff acceptance test

**Files:**
- Create: `flightrec/diff.py`
- Modify: `flightrec/cli.py` (add `diff` command)
- Modify: `tests/test_fork.py` (add diff assertion)

**Interfaces:**
- Consumes: `Store`, `store.get_events`.
- Produces:
  - `class DiffReport`: fields `branch_index: Optional[int]`, `branch_event: Optional[tuple]`
    (the `(agent_id, event_type, seq)` of first divergence), `changed_by_agent: dict[str,int]`,
    `final_a: str`, `final_b: str`.
  - `diff(store: Store, trace_a: str, trace_b: str) -> DiffReport`.
  - `format_report(report: DiffReport) -> str`.
- Modify `cli.py`: add `diff` command printing `format_report`.

**Design notes:**
- Compare event lists of A and B positionally on the full tuple
  `(agent_id, event_type, seq, boundary_hash, response_json)`. First differing index =
  `branch_index`; `branch_event` = that event's `(agent_id, event_type, seq)` from A (or B if
  A is shorter). Count, per agent, events from `branch_index` onward that differ (or exist in
  only one list).
- `final_a`/`final_b`: the `response_json` content of the last `llm_call` by `synthesizer`
  in each trace (fallback to the last event's `response_json` if absent).

- [ ] **Step 1: Write the failing test (extend test_fork.py)**

Append to `tests/test_fork.py`:

```python
from flightrec.diff import diff, format_report


def test_diff_reports_branch_and_changes(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    child = fork(store, parent, branch.event_id, {"results": ["MUT"], "query": "q"})
    report = diff(store, parent, child)
    # Branch detected exactly at the mutated tool_call.
    assert report.branch_event == (branch.agent_id, branch.event_type, branch.seq)
    # Non-empty downstream changes, attributed to agents.
    assert sum(report.changed_by_agent.values()) > 0
    text = format_report(report)
    assert "branch" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fork.py::test_diff_reports_branch_and_changes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flightrec.diff'`.

- [ ] **Step 3: Write implementation**

Create `flightrec/diff.py`:

```python
"""Textual diff between two traces."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .models import Event
from .store import Store


@dataclass
class DiffReport:
    trace_a: str
    trace_b: str
    branch_index: Optional[int]
    branch_event: Optional[tuple]
    changed_by_agent: dict[str, int] = field(default_factory=dict)
    final_a: str = ""
    final_b: str = ""


def _tuple(e: Event) -> tuple:
    return (e.agent_id, e.event_type, e.seq, e.boundary_hash, e.response_json)


def _final(events: list[Event]) -> str:
    for e in reversed(events):
        if e.agent_id == "synthesizer" and e.event_type == "llm_call":
            try:
                return json.loads(e.response_json).get("content", e.response_json)
            except Exception:
                return e.response_json
    return events[-1].response_json if events else ""


def diff(store: Store, trace_a: str, trace_b: str) -> DiffReport:
    a = store.get_events(trace_a)
    b = store.get_events(trace_b)
    branch_index: Optional[int] = None
    branch_event: Optional[tuple] = None
    n = min(len(a), len(b))
    for i in range(n):
        if _tuple(a[i]) != _tuple(b[i]):
            branch_index = i
            branch_event = (a[i].agent_id, a[i].event_type, a[i].seq)
            break
    if branch_index is None and len(a) != len(b):
        branch_index = n
        src = a[n] if len(a) > len(b) else b[n]
        branch_event = (src.agent_id, src.event_type, src.seq)

    changed: dict[str, int] = {}
    if branch_index is not None:
        for i in range(branch_index, max(len(a), len(b))):
            ea = a[i] if i < len(a) else None
            eb = b[i] if i < len(b) else None
            if ea is None or eb is None or _tuple(ea) != _tuple(eb):
                agent = (ea or eb).agent_id
                changed[agent] = changed.get(agent, 0) + 1

    return DiffReport(trace_a=trace_a, trace_b=trace_b, branch_index=branch_index,
                      branch_event=branch_event, changed_by_agent=changed,
                      final_a=_final(a), final_b=_final(b))


def format_report(r: DiffReport) -> str:
    lines = [f"diff {r.trace_a} -> {r.trace_b}"]
    if r.branch_index is None:
        lines.append("no divergence: traces are identical")
    else:
        lines.append(f"branch point: index {r.branch_index} at {r.branch_event}")
        total = sum(r.changed_by_agent.values())
        lines.append(f"changed downstream events: {total}")
        for agent, count in sorted(r.changed_by_agent.items()):
            lines.append(f"  {agent}: {count}")
    lines.append(f"final (A): {r.final_a[:120]}")
    lines.append(f"final (B): {r.final_b[:120]}")
    return "\n".join(lines)
```

Modify `flightrec/cli.py`: add import and command (after `fork`):

```python
from .diff import diff as _diff, format_report


@app.command()
def diff(trace_a: str, trace_b: str):
    """Show the divergence report between two traces."""
    store = _db()
    typer.echo(format_report(_diff(store, trace_a, trace_b)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fork.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add flightrec/diff.py flightrec/cli.py tests/test_fork.py
git commit -m "feat: trace diff report + CLI diff"
```

---

### Task 10: Full suite green + README walkthrough

**Files:**
- Create: `README.md`
- Test: entire suite.

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest -v`
Expected: ALL tests pass (models, store, clock, interceptor, boundaries, reference_agent,
cli_run_show, record_replay, fork). No test requires an API key (all use `fake_llm`).

- [ ] **Step 2: Write the README**

Create `README.md` with: one-paragraph pitch; the four boundaries; install
(`pip install -e ".[dev]"`); set `GROQ_API_KEY` for live `run`/`fork`; and a copy-paste
walkthrough:

```markdown
# Flight Recorder

Record everything a multi-agent system does, replay it deterministically with zero new
LLM/tool calls, fork a run at any event (mutate one value, re-run the rest live), and diff
the timelines.

## Why
An agent leaks non-determinism through four boundaries: LLM calls, tool calls, the clock,
and randomness. Record every value crossing those boundaries; replay by feeding the recorded
values back. The agent's own logic runs for real — only the boundaries are scripted. Replay
is free and offline; any real call during replay raises `ReplayViolation`.

## Install
    python -m pip install -e ".[dev]"
    export GROQ_API_KEY=...        # Groq free tier; or set FLIGHTREC_MODEL + that provider's key

## Walkthrough
    flightrec run "Compare Postgres and SQLite for a small app"   # -> tr_abc123
    flightrec show tr_abc123                                       # event log
    flightrec replay tr_abc123                                     # ✓ deterministic, 0 real calls
    # pick a tool_call event_id from `show`, then fork with a mutated result:
    flightrec fork tr_abc123 --at <event_id> \
        --set '{"results": ["sqlite is faster for this workload"]}'   # -> tr_def456
    flightrec diff tr_abc123 tr_def456                            # branch point + changes

## Tests
    python -m pytest -v        # all offline; live run/fork need GROQ_API_KEY
```

- [ ] **Step 3: Verify a live smoke run (optional, needs key)**

If `GROQ_API_KEY` is set:
Run: `python -m flightrec.cli run "Compare Postgres and SQLite"` then
`python -m flightrec.cli replay <printed_id>`
Expected: `run` prints a trace id; `replay` prints `✓ replay deterministic: N events
reproduced, 0 real calls`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with record/replay/fork walkthrough"
```

---

## Self-Review

**1. Spec coverage:**
- Reference multi-agent system (planner→2 workers→synthesizer), sequential, raw LiteLLM —
  Task 5. ✓
- Interceptor RECORD/REPLAY — Task 3. ✓
- All four boundary types clamped (+ agent_msg) — Tasks 3–4. ✓
- SQLite event-sourced append-only store — Task 2 (events append-only; `set_status` is the
  one allowed traces mutation, documented in Global Constraints). ✓
- Faithful replay with determinism assertion — Task 7. ✓
- Network guard raising on real call in replay — Task 3 (`guard_real_call`), exercised
  Tasks 4 & 7. ✓
- Lamport clock per event, bump-on-message — Tasks 3 (`tick`/`update`) & 4 (`agent_msg`). ✓
- Fork: replay prefix → mutate at N → live suffix → child trace — Task 8. ✓
- Textual diff — Task 9. ✓
- CLI for all operations — Tasks 6–9. ✓
- Content hash per event for drift detection, match by counter — Tasks 1–3
  (`boundary_hash`, `seq`). ✓
- Acceptance tests §11 — Tasks 7 (test_record_replay) & 8–9 (test_fork). ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"add error handling"-style placeholders; every code
step contains complete code. ✓

**3. Type consistency:** `cross(agent_id, event_type, request_obj, live_fn)`, `next_seq`,
`record_into(store, trace_id)`, `replay_from(store, trace_id, *, branch_key, mutation,
write_trace_id)`, `replay(store, trace_id)`, `recorded_tuples(store, trace_id)`,
`fork(store, trace_id, at_event_id, mutation)`, `diff(store, trace_a, trace_b)` — names and
signatures match across tasks. Comparison tuple `(agent_id, event_type, seq, boundary_hash,
response_json)` is identical in interceptor.produced, replay, and diff. ✓

**Resolved design decision (documented for the implementer):** the mock tool is a *pure*
function seeded by a worker-level `boundaries.rand()` draw, rather than calling boundaries
*inside* the tool body. This is a deliberate refinement of spec §7's "non-determinism inside
the tool goes through boundaries": it keeps nondeterminism flowing through recorded boundary
events while keeping `tool_call` a clean short-circuiting black box, because nested boundary
events inside a short-circuited tool would be orphaned during replay and break byte-identical
reproduction (the MVP's single success criterion).
