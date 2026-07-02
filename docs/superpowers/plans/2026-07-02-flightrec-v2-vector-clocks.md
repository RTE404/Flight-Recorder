# Flight Recorder V2 — Vector Clocks & Real Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two reference-agent workers run concurrently on real OS threads (`threading.Thread`), replacing the single global Lamport-ordered interceptor with a per-agent vector-clock causality model, so replay/fork/diff are correct under genuine concurrency instead of relying on accidental single-threaded total ordering.

**Architecture:** Every agent gets its own `VectorClock` component inside one process-global, lock-protected `Interceptor`; `agent_msg` is the only synchronization point and carries/merges vectors through a per-recipient mailbox. `Interceptor.cross()` computes `(seq, vector, causal_rank)` under a short-held lock, releases the lock for the actual blocking `live_fn()` call (network/tool), then re-acquires it to persist the event — so the two worker threads' LLM/tool calls genuinely overlap. SQLite stores a `vector_clock` + `causal_rank` per event and `get_events` returns a deterministic `(causal_rank, agent_id, event_type, seq)` order that is a valid linear extension of happens-before, making replay/fork/diff order-independent of thread scheduling.

**Tech Stack:** Python 3.11, stdlib `threading`, `sqlite3` (`check_same_thread=False`), Pydantic 2, pytest. No new dependencies.

**Source spec:** `flightrec_v2_build_prompt.md` (repo root) — treat it as authoritative; this plan operationalizes it task-by-task. Delete it (or leave it — it's untracked/gitignored-adjacent scratch) once the work lands; it is not meant to live in the repo long-term.

## Global Constraints

(Copied verbatim from `flightrec_v2_build_prompt.md` section 3 — every task's changes must satisfy all of these.)

- The boundary API in `boundaries.py` keeps its exact signatures. Every boundary takes an explicit `agent_id` (or `from_agent`); the interceptor never needs thread-local "who am I" state.
- `flightrec.interceptor` keeps `record_into`, `replay_from`, `current`, `NoActiveInterceptor`, `ReplayViolation`, `ReplayDrift`, and `Interceptor.cross(agent_id, event_type, request_obj, live_fn)` with the same signatures. `Interceptor.produced` remains a list of 5-tuples `(agent_id, event_type, seq, boundary_hash, response_json)`.
- `Interceptor.next_seq(agent_id, event_type)` keeps per-`(agent, event_type)` semantics.
- `guard_real_call()` raises `ReplayViolation` **iff** the interceptor's phase is `REPLAY` (never in `RECORD` or `FORK`).
- `reference_agent.run_agent(task)` returns `{"final": <str>, "answers": {"worker_a": ..., "worker_b": ...}}` and `_work(agent_id, sub_question) -> str` keeps that exact signature (a test monkeypatches `_work`).
- The store stays append-only / event-sourced.
- `Event` and `Trace` stay constructible from their V1 fields alone (new fields must have defaults), so `Event(**e.model_dump()) == e` still holds.
- The lock inside `Interceptor` must never be held while `live_fn()` runs.
- SQLite connection opened with `check_same_thread=False`; every `Store` method that touches the connection is serialized by one `threading.Lock`.
- `get_events` orders by `(causal_rank, agent_id, event_type, seq)` — deterministic, independent of thread timing or `rowid`.

## Decisions already made with the user (do not re-litigate)

- **Workflow:** do this on an isolated feature branch/worktree (matches how V1's "flight-recorder" branch was merged). Use `superpowers:using-git-worktrees` at execution time.
- **Overlap proof:** add an automated wall-clock overlap test (Task 10), in addition to the manual smoke test (Task 13).
- **Live smoke test:** the user has a `GROQ_API_KEY` (or will set `FLIGHTREC_MODEL`/`OPENAI_API_KEY`) and wants Task 13 actually executed, not skipped.
- **DB migration:** no migration code — `flightrec.db` is already gitignored/untracked; just delete the local file before the first v2 run. No task needed for this beyond a one-line reminder in Task 13.

## Important note on test "redness" mid-plan

This is one tightly-coupled subsystem (not several independent ones), so a few existing tests will **transiently fail between tasks** because `store.py`'s new event ordering only becomes fully correct once `interceptor.py`, `reference_agent.py`, `replay.py`, `fork.py`, and `diff.py` all land together. Each task below states exactly which test files must be green *at that point* and which ones are expected to still be red (and which later task fixes them). Do not attempt to fix a test flagged "expected red — fixed in Task N" out of order.

---

### Task 1: `Event` model gains `vector_clock` and `causal_rank`

**Files:**
- Modify: `flightrec/models.py:22-32` (the `Event` class)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Event.vector_clock: str = "{}"` (canonical JSON of `{agent_id: int}`), `Event.causal_rank: int = 0` (`sum(vector_clock.values())`). Both have defaults so all existing `Event(...)` call sites keep working unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
def test_event_new_fields_default_and_roundtrip():
    e = Event(
        event_id="e1", trace_id="t1", seq=0, logical_clock=1, wall_clock=123.0,
        agent_id="planner", event_type="llm_call",
        request_json='{"a":1}', response_json='{"b":2}', boundary_hash="abc",
    )
    assert e.vector_clock == "{}"
    assert e.causal_rank == 0

    e2 = Event(**{**e.model_dump(), "vector_clock": '{"planner":2}', "causal_rank": 2})
    assert Event(**e2.model_dump()) == e2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py::test_event_new_fields_default_and_roundtrip -v`
Expected: FAIL with `TypeError` or `AttributeError` — `vector_clock`/`causal_rank` don't exist on `Event` yet.

- [ ] **Step 3: Add the fields**

In `flightrec/models.py`, change the `Event` class to:

```python
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
    vector_clock: str = "{}"
    causal_rank: int = 0
```

`Trace` is unchanged. `canonical`/`sha256_hex` are unchanged.

- [ ] **Step 4: Run full model test file**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (all tests, including the pre-existing `test_event_roundtrips_through_dict`).

- [ ] **Step 5: Commit**

```bash
git add flightrec/models.py tests/test_models.py
git commit -m "feat: add vector_clock and causal_rank fields to Event"
```

---

### Task 2: `VectorClock` + causality helpers in `clock.py`

**Files:**
- Modify: `flightrec/clock.py` (keep `LamportClock` untouched, append new code)
- Create: `tests/test_vector_clock.py`

**Interfaces:**
- Produces: `VectorClock(agent_id, initial=None)` with `.tick() -> dict`, `.merge(other: dict) -> None`, `.snapshot() -> dict`; module functions `vc_rank(v: dict) -> int`, `happens_before(a: dict, b: dict) -> bool`, `concurrent(a: dict, b: dict) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vector_clock.py`:

```python
from flightrec.clock import VectorClock, vc_rank, happens_before, concurrent


def test_tick_increments_own_component():
    vc = VectorClock("a")
    assert vc.tick() == {"a": 1}
    assert vc.tick() == {"a": 2}


def test_merge_takes_elementwise_max():
    vc = VectorClock("a", {"a": 1})
    vc.merge({"a": 0, "b": 3})
    assert vc.snapshot() == {"a": 1, "b": 3}
    vc.merge({"a": 5, "b": 1})
    assert vc.snapshot() == {"a": 5, "b": 3}


def test_merge_is_commutative_and_associative():
    vc1 = VectorClock("a")
    vc1.merge({"b": 2})
    vc1.merge({"c": 5})
    vc2 = VectorClock("a")
    vc2.merge({"c": 5})
    vc2.merge({"b": 2})
    assert vc1.snapshot() == vc2.snapshot()


def test_vc_rank_sums_components():
    assert vc_rank({"a": 2, "b": 3}) == 5
    assert vc_rank({}) == 0


def test_happens_before_true_when_strictly_dominated():
    assert happens_before({"a": 1}, {"a": 1, "b": 1})
    assert not happens_before({"a": 1, "b": 1}, {"a": 1})


def test_happens_before_false_for_equal_vectors():
    assert not happens_before({"a": 1}, {"a": 1})


def test_concurrent_true_for_incomparable_vectors():
    assert concurrent({"a": 1}, {"b": 1})
    assert not concurrent({"a": 1}, {"a": 1, "b": 1})
    assert not concurrent({"a": 1}, {"a": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vector_clock.py -v`
Expected: FAIL — `ImportError: cannot import name 'VectorClock'`.

- [ ] **Step 3: Implement in `flightrec/clock.py`**

Append (do not touch the existing `LamportClock` class):

```python
class VectorClock:
    def __init__(self, agent_id: str, initial: dict | None = None):
        self.agent_id = agent_id
        self.v: dict[str, int] = dict(initial or {})

    def tick(self) -> dict:
        self.v[self.agent_id] = self.v.get(self.agent_id, 0) + 1
        return dict(self.v)

    def merge(self, other: dict) -> None:
        for k, val in other.items():
            self.v[k] = max(self.v.get(k, 0), val)

    def snapshot(self) -> dict:
        return dict(self.v)


def vc_rank(v: dict) -> int:
    return sum(v.values())


def happens_before(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    le = all(a.get(k, 0) <= b.get(k, 0) for k in keys)
    return le and a != b


def concurrent(a: dict, b: dict) -> bool:
    return a != b and not happens_before(a, b) and not happens_before(b, a)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_vector_clock.py tests/test_clock.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add flightrec/clock.py tests/test_vector_clock.py
git commit -m "feat: add VectorClock and happens-before/concurrent helpers"
```

---

### Task 3: `store.py` — thread safety, schema, causal ordering

**Files:**
- Modify: `flightrec/store.py` (full rewrite of the class body)
- Test: `tests/test_store.py` (must pass unchanged — do not edit this file)

**Interfaces:**
- Consumes: `Event.vector_clock`, `Event.causal_rank` from Task 1.
- Produces: `Store(path)` now safe to share across threads; `get_events(trace_id)` returns events ordered by `(causal_rank, agent_id, event_type, seq)` instead of `rowid_pk`.

- [ ] **Step 1: Rewrite `flightrec/store.py`**

```python
"""Append-only SQLite event store."""
from __future__ import annotations

import sqlite3
import threading
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
    vector_clock    TEXT NOT NULL DEFAULT '{}',
    causal_rank     INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);
"""

_EVENT_COLS = (
    "event_id, trace_id, seq, logical_clock, wall_clock, agent_id, "
    "event_type, request_json, response_json, boundary_hash, "
    "vector_clock, causal_rank"
)
_TRACE_COLS = (
    "trace_id, parent_trace_id, branch_point_event, mutation, task, status, created_at"
)


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def create_trace(self, trace: Trace) -> None:
        with self._lock:
            self.conn.execute(
                f"INSERT INTO traces ({_TRACE_COLS}) VALUES (?,?,?,?,?,?,?)",
                (trace.trace_id, trace.parent_trace_id, trace.branch_point_event,
                 trace.mutation, trace.task, trace.status, trace.created_at),
            )
            self.conn.commit()

    def set_status(self, trace_id: str, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE traces SET status = ? WHERE trace_id = ?", (status, trace_id)
            )
            self.conn.commit()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_TRACE_COLS} FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return Trace(**dict(row)) if row else None

    def list_traces(self) -> list[Trace]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_TRACE_COLS} FROM traces ORDER BY created_at, trace_id"
            ).fetchall()
        return [Trace(**dict(r)) for r in rows]

    def append_event(self, event: Event) -> None:
        with self._lock:
            self.conn.execute(
                f"INSERT INTO events ({_EVENT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (event.event_id, event.trace_id, event.seq, event.logical_clock,
                 event.wall_clock, event.agent_id, event.event_type, event.request_json,
                 event.response_json, event.boundary_hash, event.vector_clock,
                 event.causal_rank),
            )
            self.conn.commit()

    def get_events(self, trace_id: str) -> list[Event]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_EVENT_COLS} FROM events WHERE trace_id = ? "
                f"ORDER BY causal_rank, agent_id, event_type, seq",
                (trace_id,),
            ).fetchall()
        return [Event(**dict(r)) for r in rows]

    def get_event(self, event_id: str) -> Optional[Event]:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_EVENT_COLS} FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return Event(**dict(row)) if row else None

    def close(self) -> None:
        with self._lock:
            self.conn.close()
```

- [ ] **Step 2: Run store tests**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS, all 5 tests, **unchanged file**. In particular `test_events_returned_in_insertion_order` still yields `["b", "a"]`: both events default to `causal_rank=0` and `agent_id="planner"`/`event_type="llm_call"`, so the tiebreak falls through to `seq` (`0` then `1`), preserving insertion order for that specific fixture.

- [ ] **Step 3: Sanity-check expected new redness**

Run: `python -m pytest tests/test_record_replay.py tests/test_fork.py -v`
Expected: Some of these **now fail** because `get_events` reorders by `(0, agent_id, event_type, seq)` (all `causal_rank` are still `0` — `interceptor.py` hasn't been updated yet) while `replay.py`/`fork.py` still compare positionally. This is expected — fixed in Tasks 6–8. Do not attempt a fix here.

- [ ] **Step 4: Commit**

```bash
git add flightrec/store.py
git commit -m "feat: thread-safe Store with vector_clock/causal_rank columns and causal ordering"
```

---

### Task 4: `interceptor.py` core rewrite — RECORD / REPLAY / FORK phases

**Files:**
- Modify: `flightrec/interceptor.py` (full rewrite)
- Test: `tests/test_interceptor.py`, `tests/test_boundaries.py` (must pass unchanged)

**Interfaces:**
- Consumes: `VectorClock`, `vc_rank`, `happens_before` from Task 2; `Event` with new fields from Task 1; causally-ordered `Store.get_events` from Task 3.
- Produces: `Interceptor.cross(agent_id, event_type, request_obj, live_fn)` (unchanged signature), `Interceptor.next_seq(agent_id, event_type)`, `Interceptor.guard_real_call()`, `Interceptor.produced` (list of 5-tuples), `Interceptor._vectors_by_key: dict[(agent,type,seq), dict]` (consumed by `replay.py` in Task 6), context managers `record_into(store, trace_id)`, `replay_from(store, trace_id, *, branch_key=None, mutation=None, write_trace_id=None)`, and the **new** `fork_from(store, read_trace_id, *, write_trace_id, branch_key, branch_vec, mutation)` (consumed by `fork.py` in Task 7).

- [ ] **Step 1: Rewrite `flightrec/interceptor.py`**

```python
"""Process-global RECORD/REPLAY/FORK interceptor: per-agent vector clocks, mailbox,
thread-safe lock, network guard."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from .clock import VectorClock, happens_before, vc_rank
from .models import Event, canonical, sha256_hex
from .store import Store

RECORD = "record"
REPLAY = "replay"
FORK = "fork"

LIVE = "live"
MUTATE = "mutate"


class NoActiveInterceptor(Exception):
    pass


class ReplayViolation(Exception):
    """A real external call was attempted during replay (unclamped boundary)."""


class ReplayDrift(Exception):
    """Replay diverged from the recording (missing event, request/causal mismatch)."""


def _apply_mutation(recorded_value: Any, mutation: Any) -> Any:
    if isinstance(recorded_value, dict) and isinstance(mutation, dict):
        return {**recorded_value, **mutation}
    return mutation


class Interceptor:
    def __init__(self, store: Store, *, phase: str, read_trace_id: Optional[str],
                 write_trace_id: Optional[str], branch_key: Optional[tuple] = None,
                 branch_vec: Optional[dict] = None, mutation: Any = None):
        self.store = store
        self.phase = phase
        self.read_trace_id = read_trace_id
        self.write_trace_id = write_trace_id
        self.branch_key = branch_key
        self._branch_vec = branch_vec or {}
        self.mutation = mutation

        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], int] = {}
        self._vectors: dict[str, VectorClock] = {}
        self._mailbox: dict[str, list[dict]] = {}
        self.produced: list[tuple] = []
        self._vectors_by_key: dict[tuple[str, str, int], dict] = {}
        self._tainted: set[str] = set()

        self._recorded: dict[tuple[str, str, int], Event] = {}
        if read_trace_id is not None:
            for e in store.get_events(read_trace_id):
                self._recorded[(e.agent_id, e.event_type, e.seq)] = e

    # -- sequence numbers -----------------------------------------------
    def next_seq(self, agent_id: str, event_type: str) -> int:
        # Public entry point acquires the lock itself. cross() already holds
        # self._lock when it needs a seq, so it calls _next_seq_locked directly
        # (threading.Lock is not reentrant — calling next_seq() from inside a
        # `with self._lock:` block would deadlock).
        with self._lock:
            return self._next_seq_locked(agent_id, event_type)

    def _next_seq_locked(self, agent_id: str, event_type: str) -> int:
        key = (agent_id, event_type)
        seq = self._counters.get(key, 0)
        self._counters[key] = seq + 1
        return seq

    # -- vector clock bookkeeping (must run under self._lock) ------------
    def _tick(self, agent: str) -> dict:
        vc = self._vectors.setdefault(agent, VectorClock(agent))
        for delivered in self._mailbox.pop(agent, []):
            vc.merge(delivered)
        return vc.tick()

    def _deliver(self, event_type: str, request_obj: Any, sender_vec: dict,
                 sender_agent: str) -> None:
        if event_type != "agent_msg":
            return
        to_agent = request_obj["to"]
        self._mailbox.setdefault(to_agent, []).append(dict(sender_vec))
        if self.phase == FORK and sender_agent in self._tainted:
            self._tainted.add(to_agent)

    # -- replay/fork mode decision (must run under self._lock) -----------
    def _decide_mode(self, agent: str, key: tuple) -> str:
        if self.phase == RECORD:
            return LIVE
        if self.phase == REPLAY:
            return REPLAY
        # FORK
        if key == self.branch_key:
            return MUTATE
        if agent in self._tainted:
            return LIVE
        rec = self._recorded.get(key)
        if rec is not None and happens_before(self._branch_vec, json.loads(rec.vector_clock)):
            self._tainted.add(agent)
            return LIVE
        return REPLAY

    def guard_real_call(self) -> None:
        if self.phase == REPLAY:
            raise ReplayViolation(
                "Attempted a real external call during REPLAY — unclamped boundary."
            )

    def _write(self, agent_id, event_type, seq, request_json, response_json,
               boundary_hash, vec: dict, rank: int) -> None:
        if self.write_trace_id is None:
            return
        self.store.append_event(Event(
            event_id=os.urandom(16).hex(), trace_id=self.write_trace_id, seq=seq,
            logical_clock=vec.get(agent_id, 0), wall_clock=time.time(), agent_id=agent_id,
            event_type=event_type, request_json=request_json, response_json=response_json,
            boundary_hash=boundary_hash, vector_clock=canonical(vec), causal_rank=rank,
        ))

    def cross(self, agent_id: str, event_type: str, request_obj: Any,
              live_fn: Callable[[], Any]) -> Any:
        request_json = canonical(request_obj)
        boundary_hash = sha256_hex(request_json)

        with self._lock:
            seq = self._next_seq_locked(agent_id, event_type)
            vec = self._tick(agent_id)
            key = (agent_id, event_type, seq)
            rank = vc_rank(vec)
            self._vectors_by_key[key] = vec
            mode = self._decide_mode(agent_id, key)

        if mode == LIVE:
            value = live_fn()  # network/tool call happens OUTSIDE the lock
            response_json = canonical(value)
            with self._lock:
                self._write(agent_id, event_type, seq, request_json, response_json,
                            boundary_hash, vec, rank)
                self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
                self._deliver(event_type, request_obj, vec, agent_id)
            return value

        rec = self._recorded.get(key)
        if rec is None:
            raise ReplayDrift(f"no recorded event for {key} — extra boundary crossing")
        if rec.boundary_hash != boundary_hash:
            raise ReplayDrift(
                f"request drift at {key}: {rec.boundary_hash[:12]} != {boundary_hash[:12]}"
            )
        if canonical(vec) != rec.vector_clock:
            raise ReplayDrift(f"causal drift at {key}: vector clock differs from recording")

        if mode == MUTATE:
            value = _apply_mutation(json.loads(rec.response_json), self.mutation)
            response_json = canonical(value)
            with self._lock:
                self._tainted.add(agent_id)
                self._write(agent_id, event_type, seq, request_json, response_json,
                            boundary_hash, vec, rank)
                self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
                self._deliver(event_type, request_obj, vec, agent_id)
            return value

        # mode == REPLAY
        with self._lock:
            if self.write_trace_id is not None:
                self._write(agent_id, event_type, seq, request_json, rec.response_json,
                            boundary_hash, vec, rank)
            self.produced.append((agent_id, event_type, seq, boundary_hash, rec.response_json))
            self._deliver(event_type, request_obj, vec, agent_id)
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


@contextmanager
def fork_from(store: Store, read_trace_id: str, *, write_trace_id: str,
             branch_key: tuple, branch_vec: dict, mutation: Any):
    global _active
    prev = _active
    _active = Interceptor(store, phase=FORK, read_trace_id=read_trace_id,
                          write_trace_id=write_trace_id, branch_key=branch_key,
                          branch_vec=branch_vec, mutation=mutation)
    try:
        yield _active
    finally:
        _active = prev
```

Note `replay_from` deliberately still accepts `branch_key`/`mutation` keywords for API back-compat (a test calls `replay_from(store, "t1")` with no branch args) even though real forking now goes through `fork_from` — in `REPLAY` phase `_decide_mode` never looks at `branch_key`, so passing it is a harmless no-op.

- [ ] **Step 2: Run interceptor + boundaries tests**

Run: `python -m pytest tests/test_interceptor.py tests/test_boundaries.py -v`
Expected: PASS, all tests, **both files unchanged**. These are single-threaded so the added vector-clock check never fires (record and replay recompute identical vectors).

- [ ] **Step 3: Re-run reference agent + models/clock/store tests**

Run: `python -m pytest tests/test_reference_agent.py tests/test_models.py tests/test_clock.py tests/test_vector_clock.py tests/test_store.py -v`
Expected: PASS (reference_agent.py is still V1/sequential at this point, but it works fine against the new interceptor since vector-clock bookkeeping is agnostic to whether callers are threaded).

- [ ] **Step 4: Sanity-check expected new redness**

Run: `python -m pytest tests/test_record_replay.py tests/test_fork.py -v`
Expected: still failing (replay.py/fork.py haven't been updated to the causal-order comparison / `fork_from` yet — fixed in Tasks 6–8).

- [ ] **Step 5: Commit**

```bash
git add flightrec/interceptor.py
git commit -m "feat: rewrite Interceptor with per-agent vector clocks, mailbox, FORK phase"
```

---

### Task 5: `reference_agent.py` — real threading for the two workers

**Files:**
- Modify: `flightrec/agent/reference_agent.py` (full rewrite)
- Test: `tests/test_reference_agent.py` (must pass unchanged)

**Interfaces:**
- Consumes: `boundaries.llm/tool_call/now/new_uuid/rand/agent_msg` (unchanged), `Interceptor.cross` from Task 4.
- Produces: `run_agent(task) -> {"final": str, "answers": {"worker_a": str, "worker_b": str}}`, `_work(agent_id: str, sub_question: str) -> str` (exact signature preserved — a test monkeypatches this), module constant `WORKER_IDS = ["worker_a", "worker_b"]`.

- [ ] **Step 1: Rewrite `flightrec/agent/reference_agent.py`**

```python
"""Concurrent reference pipeline: planner -> worker_a / worker_b (parallel) -> synthesizer."""
from __future__ import annotations

import json
import threading

from .. import boundaries as b

PLANNER = "planner"
SYNTH = "synthesizer"
WORKER_IDS = ["worker_a", "worker_b"]


def _plan(task: str) -> list[str]:
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
    return subs


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


def _synthesize(task: str, answers: dict) -> str:
    prompt = (
        "Combine these two answers into a final response.\n"
        f"A: {answers['worker_a']}\nB: {answers['worker_b']}"
    )
    return b.llm([{"role": "user", "content": prompt}], agent_id=SYNTH)["content"]


def run_agent(task: str) -> dict:
    sub_questions = _plan(task)

    # Planner -> worker handoffs happen in the main thread, in fixed order, so
    # planner's own events are totally ordered and never race with each other.
    assignments = {wid: b.agent_msg(PLANNER, wid, sq)
                   for wid, sq in zip(WORKER_IDS, sub_questions)}

    answers: dict[str, str] = {}
    errors: dict[str, BaseException] = {}
    lock = threading.Lock()

    def worker_entry(wid: str) -> None:
        try:
            ans = _work(wid, assignments[wid])
            b.agent_msg(wid, SYNTH, ans)  # real join edge -> taints synth on fork
            with lock:
                answers[wid] = ans
        except BaseException as e:  # threads don't propagate exceptions through join()
            with lock:
                errors[wid] = e

    threads = [threading.Thread(target=worker_entry, args=(wid,)) for wid in WORKER_IDS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:  # re-raise deterministically, in WORKER_IDS order
        for wid in WORKER_IDS:
            if wid in errors:
                raise errors[wid]

    final = _synthesize(task, answers)
    return {"final": final, "answers": answers}
```

Two points this preserves on purpose:
1. **Thread exception propagation** — exceptions raised inside a `Thread` do not propagate through `join()`; `worker_entry` captures them and the main thread re-raises in `WORKER_IDS` order after both joins.
2. **Planner sends before spawn** — doing both `agent_msg(planner, wid, ...)` calls in the main thread before `t.start()` establishes the happens-before edge via `Thread.start()`'s synchronization, so each worker's first `_tick()` deterministically merges the planner's vector before the worker ticks its own.

- [ ] **Step 2: Run reference agent tests**

Run: `python -m pytest tests/test_reference_agent.py -v`
Expected: PASS, all tests, **file unchanged**. In particular `test_agent_records_all_four_boundary_types` still sees `agent_msg` among the recorded types and `out["answers"]` still has both `worker_a`/`worker_b` keys.

- [ ] **Step 3: Sanity-check expected new redness**

Run: `python -m pytest tests/test_record_replay.py tests/test_fork.py -v`
Expected: still failing (replay.py/fork.py not yet updated — fixed in Tasks 6–8).

- [ ] **Step 4: Commit**

```bash
git add flightrec/agent/reference_agent.py
git commit -m "feat: run worker_a/worker_b concurrently on real threads"
```

---

### Task 6: `replay.py` — causal-order-aware determinism check

**Files:**
- Modify: `flightrec/replay.py` (full rewrite)
- Test: `tests/test_record_replay.py` (must pass unchanged)

**Interfaces:**
- Consumes: `Interceptor.produced`, `Interceptor._vectors_by_key` from Task 4; `vc_rank` from Task 2; causally-ordered `store.get_events` from Task 3.
- Produces: `recorded_tuples(store, trace_id) -> list[tuple]` (unchanged shape), `replay(store, trace_id) -> list[tuple]`, `DeterminismError` (unchanged).

- [ ] **Step 1: Rewrite `flightrec/replay.py`**

```python
"""Faithful replay + determinism assertion (order-independent across interleavings)."""
from __future__ import annotations

from .store import Store
from . import interceptor as itc
from .clock import vc_rank
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
        vbk = dict(inter._vectors_by_key)

    # Re-order into the same canonical causal order used by get_events, so a
    # correct concurrent run yields produced == expected regardless of thread
    # scheduling during this particular replay.
    produced.sort(key=lambda t: (vc_rank(vbk[(t[0], t[1], t[2])]), t[0], t[1], t[2]))

    if produced != expected:
        by_key_p = {(t[0], t[1], t[2]): t for t in produced}
        by_key_e = {(t[0], t[1], t[2]): t for t in expected}
        for key in sorted(set(by_key_e) | set(by_key_p)):
            if by_key_e.get(key) != by_key_p.get(key):
                raise DeterminismError(
                    f"replay drift at {key}:\n"
                    f"  recorded={by_key_e.get(key)}\n"
                    f"  replayed={by_key_p.get(key)}"
                )
        raise DeterminismError(
            f"length mismatch: recorded {len(expected)} events, replayed {len(produced)}"
        )
    return produced
```

- [ ] **Step 2: Run record/replay tests**

Run: `python -m pytest tests/test_record_replay.py -v`
Expected: PASS, all 4 tests, **file unchanged** — including `test_unclamped_clock_makes_replay_fail_loudly`, which now proves the drift is raised inside a worker *thread* and still surfaces through `replay()` thanks to Task 5's exception capture/re-raise.

- [ ] **Step 3: Sanity-check expected new redness**

Run: `python -m pytest tests/test_fork.py -v`
Expected: still failing (`fork.py` not yet rewritten — fixed in Tasks 7–8).

- [ ] **Step 4: Commit**

```bash
git add flightrec/replay.py
git commit -m "feat: causal-order-aware replay comparison"
```

---

### Task 7: `fork.py` — causal fork via `fork_from` (no prefix-copy loop)

**Files:**
- Modify: `flightrec/fork.py` (full rewrite)
- Test: `tests/test_fork.py::test_fork_shares_prefix_and_diverges_at_branch`, `tests/test_fork.py::test_fork_suffix_is_live_and_recorded`

**Interfaces:**
- Consumes: `itc.fork_from` from Task 4.
- Produces: `fork(store, trace_id, at_event_id, mutation) -> child_trace_id` (unchanged signature).

- [ ] **Step 1: Rewrite `flightrec/fork.py`**

```python
"""Causal fork: rerun only the branch's causal future; reuse past/concurrent events."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .models import Trace, canonical
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
    branch_vec = json.loads(branch.vector_clock)
    child_id = _new_child_id()
    store.create_trace(Trace(
        trace_id=child_id, parent_trace_id=trace_id, branch_point_event=at_event_id,
        mutation=canonical(mutation), task=parent.task, status="recording",
        created_at=time.time(),
    ))
    try:
        with itc.fork_from(store, trace_id, write_trace_id=child_id,
                           branch_key=branch_key, branch_vec=branch_vec, mutation=mutation):
            run_agent(parent.task)
        store.set_status(child_id, "complete")
    except Exception:
        store.set_status(child_id, "failed")
        raise
    return child_id
```

There is no more manual "copy prefix" loop — `fork_from`'s per-crossing `_decide_mode` handles past/concurrent (REPLAY, copied into child), the branch (MUTATE), and the causal future (LIVE) all inside `run_agent`'s single re-execution.

- [ ] **Step 2: Run the two non-diff fork tests**

Run: `python -m pytest tests/test_fork.py::test_fork_shares_prefix_and_diverges_at_branch tests/test_fork.py::test_fork_suffix_is_live_and_recorded -v`
Expected: PASS, both tests, **file unchanged**.

- [ ] **Step 3: Sanity-check expected new redness**

Run: `python -m pytest tests/test_fork.py -v`
Expected: `test_diff_reports_branch_and_changes` still fails — `diff.py` hasn't been updated yet (fixed in Task 8).

- [ ] **Step 4: Commit**

```bash
git add flightrec/fork.py
git commit -m "feat: causal fork via fork_from, drop manual prefix-copy loop"
```

---

### Task 8: `diff.py` — key-based alignment and rank-based branch detection

**Files:**
- Modify: `flightrec/diff.py` (full rewrite)
- Test: `tests/test_fork.py::test_diff_reports_branch_and_changes`

**Interfaces:**
- Consumes: `Event.causal_rank` from Task 1, causally-ordered `store.get_events` from Task 3.
- Produces: `DiffReport` (same fields; `branch_index` now always `None`, kept for back-compat), `diff(store, trace_a, trace_b) -> DiffReport`, `format_report(report) -> str` (must still contain the word "branch").

- [ ] **Step 1: Rewrite `flightrec/diff.py`**

```python
"""Causally-keyed diff between two traces."""
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


def _final(events: list[Event]) -> str:
    for e in reversed(events):
        if e.agent_id == "synthesizer" and e.event_type == "llm_call":
            try:
                return json.loads(e.response_json).get("content", e.response_json)
            except Exception:
                return e.response_json
    return events[-1].response_json if events else ""


def diff(store: Store, trace_a: str, trace_b: str) -> DiffReport:
    a = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_a)}
    b = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_b)}
    keys = set(a) | set(b)

    changed_keys = []
    for k in keys:
        ea, eb = a.get(k), b.get(k)
        if ea is None or eb is None or (
            (ea.boundary_hash, ea.response_json) != (eb.boundary_hash, eb.response_json)
        ):
            changed_keys.append(k)

    def rank_of(k):
        e = a.get(k) or b.get(k)
        return (e.causal_rank, k)

    branch = min(changed_keys, key=rank_of) if changed_keys else None

    changed_by_agent: dict[str, int] = {}
    for (agent, _t, _s) in changed_keys:
        changed_by_agent[agent] = changed_by_agent.get(agent, 0) + 1

    return DiffReport(
        trace_a=trace_a, trace_b=trace_b, branch_index=None,
        branch_event=branch, changed_by_agent=changed_by_agent,
        final_a=_final(store.get_events(trace_a)),
        final_b=_final(store.get_events(trace_b)),
    )


def format_report(r: DiffReport) -> str:
    lines = [f"diff {r.trace_a} -> {r.trace_b}"]
    if r.branch_event is None:
        lines.append("no divergence: traces are identical")
    else:
        lines.append(f"branch point: {r.branch_event}")
        total = sum(r.changed_by_agent.values())
        lines.append(f"changed downstream events: {total}")
        for agent, count in sorted(r.changed_by_agent.items()):
            lines.append(f"  {agent}: {count}")
    lines.append(f"final (A): {r.final_a[:120]}")
    lines.append(f"final (B): {r.final_b[:120]}")
    return "\n".join(lines)
```

- [ ] **Step 2: Run full fork test file**

Run: `python -m pytest tests/test_fork.py -v`
Expected: PASS, all 3 tests, **file unchanged**.

- [ ] **Step 3: Commit**

```bash
git add flightrec/diff.py
git commit -m "feat: key-based diff alignment with causal-rank branch detection"
```

---

### Task 9: `cli.py` — show vector clock / rank; no behavioral change elsewhere

**Files:**
- Modify: `flightrec/cli.py:61-72` (the `show` command only)
- Test: `tests/test_cli_run_show.py` (must pass unchanged)

**Interfaces:**
- Produces: `flightrec show <trace_id> [--vector]` prints `lc=` and `rank=`; with `--vector` also prints `vc=<vector_clock json>`. `run`, `ls`, `replay`, `fork`, `diff` commands unchanged (they already inherit new semantics from the modules above).

- [ ] **Step 1: Update the `show` command**

Replace lines 61-72 of `flightrec/cli.py`:

```python
@app.command()
def show(trace_id: str,
        vector: bool = typer.Option(False, "--vector", help="also print vector_clock")):
    """Print the event log for a trace."""
    store = _db()
    if store.get_trace(trace_id) is None:
        typer.echo(f"no such trace: {trace_id}", err=True)
        raise typer.Exit(1)
    for e in store.get_events(trace_id):
        resp = e.response_json if len(e.response_json) <= 70 else e.response_json[:67] + "..."
        line = (f"seq={e.seq} lc={e.logical_clock} rank={e.causal_rank} {e.agent_id:11} "
               f"{e.event_type:9} {e.event_id}  {resp}")
        if vector:
            line += f"  vc={e.vector_clock}"
        typer.echo(line)
```

- [ ] **Step 2: Run CLI test**

Run: `python -m pytest tests/test_cli_run_show.py -v`
Expected: PASS, unchanged file (it only exercises `record_run`, not `show`'s text format).

- [ ] **Step 3: Manual smoke-check of the new flag**

```bash
python -c "
from flightrec.store import Store
from flightrec import cli
store = Store(':memory:')
class F:
    def __call__(self, model, messages, **kw):
        import json
        class M: content = json.dumps({'sub_questions':['a','b']}) if 'sub_questions' in messages[-1]['content'] else 'ok'
        class C: message = M()
        class R: choices=[C()]
        return R()
import litellm; litellm.completion = F()
tid = cli.record_run(store, 'smoke test')
print(tid)
"
```
Then run `flightrec show <printed-trace-id> --vector` against the same DB path (or just eyeball that the code imports cleanly via `python -c "import flightrec.cli"`). This step is a low-stakes formatting check, not a correctness gate — the pytest run in Step 2 is the real gate.

- [ ] **Step 4: Commit**

```bash
git add flightrec/cli.py
git commit -m "feat: show prints causal_rank and optional vector_clock"
```

---

### Task 10: New concurrency integration tests (`tests/test_concurrency.py`)

**Files:**
- Create: `tests/test_concurrency.py`

**Interfaces:**
- Consumes: `cli.record_run`, `replay.replay`, `fork.fork`, `clock.happens_before`/`concurrent`, `interceptor.ReplayDrift`, `replay.DeterminismError` — all from Tasks 1–8, already implemented and green.

- [ ] **Step 1: Write the file**

```python
import json
import os
import time

import pytest

from flightrec.store import Store
from flightrec import cli
from flightrec import interceptor as itc
from flightrec.clock import happens_before, concurrent
from flightrec.replay import replay, DeterminismError
from flightrec.fork import fork


def _fake_completion(model, messages, **kwargs):
    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    last = messages[-1]["content"]
    if "sub_questions" in last:
        return _Resp(json.dumps({"sub_questions": ["qa", "qb"]}))
    return _Resp("answer-" + str(len(last)))


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr("litellm.completion", _fake_completion)
    return monkeypatch


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "c.db"))
    tid = cli.record_run(store, "compare X and Y")
    return store, tid


def test_concurrent_replay_is_byte_identical_across_repeats(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    first = replay(store, tid)
    second = replay(store, tid)
    assert first == second


def test_recorded_vectors_are_causally_consistent(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    by_key = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(tid)}

    msg_a_vec = json.loads(by_key[("worker_a", "agent_msg", 0)].vector_clock)
    msg_b_vec = json.loads(by_key[("worker_b", "agent_msg", 0)].vector_clock)
    synth_vec = json.loads(by_key[("synthesizer", "llm_call", 0)].vector_clock)
    tool_a_vec = json.loads(by_key[("worker_a", "tool_call", 0)].vector_clock)
    tool_b_vec = json.loads(by_key[("worker_b", "tool_call", 0)].vector_clock)

    assert happens_before(msg_a_vec, synth_vec)
    assert happens_before(msg_b_vec, synth_vec)
    assert concurrent(tool_a_vec, tool_b_vec)


def test_causal_fork_reuses_concurrent_recording(tmp_path, fake_llm, monkeypatch):
    store, tid = _record(tmp_path, fake_llm)
    events = store.get_events(tid)
    branch = next(e for e in events if e.event_type == "tool_call" and e.agent_id == "worker_a")

    def guarded_completion(model, messages, **kwargs):
        last = messages[-1]["content"]
        if "qb" in last:
            raise AssertionError("worker_b's llm_call must not run live during this fork")
        return _fake_completion(model, messages, **kwargs)

    def guarded_run_tool(name, args):
        if args.get("query") == "qb":
            raise AssertionError("worker_b's tool_call must not run live during this fork")
        return {"query": args["query"], "results": ["r1", "r2", "r3"]}

    monkeypatch.setattr("litellm.completion", guarded_completion)
    monkeypatch.setattr("flightrec.agent.tools.run_tool", guarded_run_tool)

    child = fork(store, tid, branch.event_id, {"query": "qa", "results": ["MUTATED"]})

    parent_by_key = {(e.agent_id, e.event_type, e.seq): e for e in events}
    child_by_key = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(child)}
    for key, pe in parent_by_key.items():
        if pe.agent_id == "worker_b":
            ce = child_by_key[key]
            assert (ce.boundary_hash, ce.response_json) == (pe.boundary_hash, pe.response_json)


def test_thread_exception_from_one_worker_propagates_through_replay(tmp_path, fake_llm, monkeypatch):
    store, tid = _record(tmp_path, fake_llm)

    import flightrec.agent.reference_agent as ra
    real_work = ra._work
    real_now = {"v": 1000.0}

    def patched_work(agent_id, sub_question):
        if agent_id != "worker_b":
            return real_work(agent_id, sub_question)
        from flightrec import boundaries as b
        req_id = b.new_uuid(agent_id=agent_id)
        real_now["v"] += 1.0
        ts = real_now["v"]  # UNCLAMPED on purpose -> drift only for worker_b
        seed = b.rand(agent_id=agent_id)
        results = b.tool_call("search", {"query": sub_question, "seed": seed}, agent_id=agent_id)
        prompt = (f"request_id={req_id} ts={ts}\nUsing these search results, answer.\n"
                  f"Question: {sub_question}\nResults: {json.dumps(results['results'])}")
        resp = b.llm([{"role": "user", "content": prompt}], agent_id=agent_id)
        return resp["content"]

    monkeypatch.setattr(ra, "_work", patched_work)
    with pytest.raises((itc.ReplayDrift, DeterminismError)):
        replay(store, tid)


def test_worker_legs_run_concurrently(tmp_path, monkeypatch):
    """Prove the two worker legs overlap on real threads (not just interleave stepwise)."""
    DELAY = 0.2

    def slow_completion(model, messages, **kwargs):
        time.sleep(DELAY)
        last = messages[-1]["content"]
        if "sub_questions" in last:
            return _fake_completion(model, messages, **kwargs)
        return _fake_completion(model, messages, **kwargs)

    def slow_run_tool(name, args):
        time.sleep(DELAY)
        return {"query": args["query"], "results": ["r1", "r2", "r3"]}

    monkeypatch.setattr("litellm.completion", slow_completion)
    monkeypatch.setattr("flightrec.agent.tools.run_tool", slow_run_tool)

    store = Store(os.path.join(tmp_path, "timing.db"))
    start = time.time()
    cli.record_run(store, "timing check")
    elapsed = time.time() - start

    # 6 blocking calls total (1 planner llm + 2x(tool+llm) per worker + 1 synth llm).
    # Fully sequential (V1) would take ~6*DELAY. Concurrent workers only add one
    # worker's tool+llm once, so the critical path is ~4*DELAY. Assert well below
    # the sequential bound to prove genuine overlap, with slack for scheduling jitter.
    assert elapsed < 5 * DELAY, f"workers do not appear to run concurrently: {elapsed:.2f}s"
```

- [ ] **Step 2: Run the new file**

Run: `python -m pytest tests/test_concurrency.py -v`
Expected: PASS, all 5 tests. If `test_worker_legs_run_concurrently` is flaky on a slow/loaded CI box, bump `DELAY` up (e.g. to `0.3`) rather than loosening the assertion multiplier — the ratio (5x vs the true 4x/6x split) is what matters.

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test: vector-clock causality, causal fork reuse, thread-exception propagation, and real overlap"
```

---

### Task 11: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the entire suite**

Run: `python -m pytest -v`
Expected: **All tests pass** — every file from Tasks 1–10, with no exceptions this time (this is the point where all prior "expected red" notes must have resolved). If anything is still red here, it is a genuine bug — root-cause it with `superpowers:systematic-debugging` rather than patching around it.

- [ ] **Step 2: Confirm the package still installs cleanly**

Run: `python -m pip install -e ".[dev]"`
Expected: succeeds with no errors (no new dependencies were introduced, so this should be a no-op reinstall).

- [ ] **Step 3: Commit (only if Step 1/2 required any fix)**

If any fix was needed to get to fully green:
```bash
git add -A
git commit -m "fix: resolve regressions found in full v2 test suite pass"
```
If nothing needed fixing, skip this step — there's nothing to commit.

---

### Task 12: `README.md` documentation updates

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the "How it works" table row for `interceptor.py`**

Change the row:
```
| `flightrec/interceptor.py` | Process-global RECORD/REPLAY context, per-`(agent,event_type)` counters, Lamport clock, network guard. |
```
to:
```
| `flightrec/interceptor.py` | Process-global RECORD/REPLAY/FORK interceptor: per-agent vector clocks, a per-recipient message mailbox, a thread-safe lock, and per-crossing mode decisions (live / replay / mutate). |
```

- [ ] **Step 2: Add a "Concurrency & determinism (V2)" section**

Insert after the "How it works" section, before "## Tests":

```markdown
## Concurrency & determinism (V2)

`worker_a` and `worker_b` run on real OS threads (`threading.Thread`); LiteLLM calls are
blocking HTTP so the GIL is released during them and the two worker legs genuinely overlap.

V1's determinism guarantee ("the whole event sequence is byte-identical") no longer applies,
because there is no single global order once two threads race. V2's guarantee instead is:

> For every agent, replay reproduces that agent's own sequence of boundary crossings
> byte-for-byte (`boundary_hash` and `response_json` per `(agent_id, event_type, seq)`),
> **and** the recorded happens-before partial order is reproduced exactly (recomputed vector
> clock equals recorded vector clock for every event). The interleaving of concurrent events
> is explicitly allowed to differ.

Each agent is a "process" with its own vector-clock component; `agent_msg` is the only
send/receive synchronization point, carrying and merging vectors through a mailbox. Events
are stored with a `vector_clock` (canonical JSON of `{agent_id: int}`) and a `causal_rank`
(sum of its components, used only for a deterministic display/storage order — a valid linear
extension of happens-before). `get_events` returns events ordered by
`(causal_rank, agent_id, event_type, seq)`, independent of thread scheduling or SQLite `rowid`.

**Causal fork:** because events carry real vector clocks, `fork` reruns only the events in the
branch point's causal future and reuses the recorded values of events that are past or
concurrent with the branch. Forking one worker's `tool_call` does not rerun the other worker —
the untouched worker's events are replayed and copied into the child byte-identically, with
zero real LLM/tool calls.
```

- [ ] **Step 3: Note the schema change**

In the "Install" section, add a line after the `FLIGHTREC_DB` sentence:

```markdown
V2 adds `vector_clock`/`causal_rank` columns to the events table. If you have an existing
`flightrec.db` from V1, delete it before your first V2 run (it's gitignored, so this is safe).
```

- [ ] **Step 4: Update the "Tests" section closing paragraph**

Replace the final paragraph:
```
The suite runs fully offline via a fake LiteLLM fixture. It proves: recordings cover all
four boundary types, replay is byte-identical, replay makes zero real calls, an unclamped
clock makes replay fail loudly, forks share an identical prefix and diverge exactly at the
branch point with a live suffix, and diff reports the correct branch point.
```
with:
```
The suite runs fully offline via a fake LiteLLM fixture. It proves: recordings cover all
boundary types (including `agent_msg`), replay is byte-identical per-agent and reproduces
the recorded happens-before order, replay makes zero real calls, an unclamped clock makes
replay fail loudly even when the drift originates inside a worker thread, forks reuse the
concurrent worker's recording untouched while rerunning only the mutated worker's causal
future, diff reports the correct branch point by causal rank, and the two worker legs
provably overlap in wall-clock time.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document V2 vector-clock determinism model and causal fork"
```

---

### Task 13: Live smoke test (manual, against a real LLM)

**Files:** none (manual verification against a real provider).

- [ ] **Step 1: Delete any stale V1 database**

```bash
rm -f flightrec.db
```

- [ ] **Step 2: Set credentials**

Set `GROQ_API_KEY` (or `FLIGHTREC_MODEL` + `OPENAI_API_KEY`) in the shell before running the CLI — the user has confirmed a key is available for this step.

- [ ] **Step 3: Run the walkthrough**

```bash
flightrec run "Compare Postgres and SQLite for a small app"
# -> tr_abc123
flightrec show tr_abc123 --vector
flightrec replay tr_abc123
# -> OK replay deterministic: N events reproduced, 0 real calls
```

- [ ] **Step 4: Fork at a worker's `tool_call` and diff**

Pick a `worker_a` (or `worker_b`) `tool_call` event id from `show`'s output, then:

```bash
flightrec fork tr_abc123 --at <event_id> --set '{"results": ["a mutated search result"]}'
# -> tr_def456
flightrec diff tr_abc123 tr_def456
```

Confirm in the diff output: the mutated worker and the synthesizer show non-zero changed
events; the *other* worker shows **zero** changed events (fully replayed, not rerun).

- [ ] **Step 5: Confirm real overlap by wall-clock feel**

Time `flightrec run "..."` once; it should feel roughly as slow as *one* worker's
`tool_call + llm` round trip, not the sum of both workers' round trips. (Task 10's automated
test already proves this mechanically with fake sleeping stubs — this step just confirms it
holds against the real provider too.)

- [ ] **Step 6: Report results back**

No commit needed for this task — report the trace ids and diff output back in conversation
so we can confirm the V2 definition of done (spec section 7) is fully satisfied.

---

## Self-review notes (from writing this plan)

- **Spec coverage:** every file in spec section 4 (`models.py`, `clock.py`, `store.py`,
  `interceptor.py`, `boundaries.py` [no-op, confirmed unchanged], `reference_agent.py`,
  `replay.py`, `fork.py`, `diff.py`, `cli.py`) has a task. Section 5's checklist items
  (lock discipline, SQLite thread-safety, thread exceptions, deterministic message arrival,
  no global phase flip, deterministic ordering) are all satisfied by Tasks 3–5's code and
  exercised by Task 10's tests. Section 6's "new tests to add" are Tasks 2 and 10 (5 spec
  items + 1 user-approved overlap test). Section 7's definition of done is Tasks 11 and 13.
  Section 8's doc updates are Task 12.
- **Resolved spec ambiguity:** the spec's `cross()` pseudocode calls `self._next_seq(...)`
  (private) while the hard invariants require a public `next_seq(agent_id, event_type)`
  method. Since `threading.Lock` isn't reentrant and `cross()` already holds `self._lock`
  when it needs a sequence number, Task 4 keeps `next_seq()` as the public, lock-acquiring
  entry point and adds a lock-free `_next_seq_locked()` for internal use inside `cross()` —
  preserving the public API exactly while avoiding a self-deadlock.
- **Placeholder scan:** no TBDs — every step has literal, complete code or an exact command
  with an expected result.
- **Type consistency:** `Interceptor.cross(agent_id, event_type, request_obj, live_fn)`,
  `next_seq`, `produced` (5-tuples), `_vectors_by_key` (keyed by `(agent_id, event_type, seq)`
  everywhere it's read in Tasks 6–8), `fork_from(store, read_trace_id, *, write_trace_id,
  branch_key, branch_vec, mutation)` are used identically across Tasks 4, 6, 7, 10.
