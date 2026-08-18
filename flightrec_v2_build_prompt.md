# Version 2 — Flight Recorder: Real Parallelism with Vector Clocks

## Purpose of this document

This is a complete build prompt for upgrading Flight Recorder from a strictly
sequential reference agent to one where the two workers run **concurrently on real OS
threads**, with a **vector-clock** causality model that makes deterministic replay,
fork, and diff correct under concurrency.

Implement exactly to the contracts below. Every design decision here has been checked
against the existing test suite; the "Test compatibility" section states precisely which
tests must still pass unchanged and which must be added or updated. Do not deviate from
the contracts without updating both the tests and this document.

---

## 1. What V1 is (starting point)

The reference agent is a sequential pipeline `planner → worker_a / worker_b →
synthesizer` built on six sanctioned boundaries (`llm`, `tool_call`, `now`, `new_uuid`,
`rand`, `agent_msg`) in `flightrec/boundaries.py`. A single process-global `Interceptor`
records every boundary crossing into an append-only SQLite store; replay feeds recorded
values back and asserts the produced event sequence is byte-identical to the recording.

V1 works only because execution is single-threaded, which makes three orderings
coincide:

1. the order `live_fn` fired (execution order),
2. the order rows landed in SQLite (`rowid`),
3. the order `replay()` compares against (positional list equality).

Under concurrency these separate. V2 fixes that.

---

## 2. Goal and the distributed-systems model

**Concurrency model:** OS threads (`threading.Thread`). LiteLLM calls are blocking HTTP,
so the GIL is released during them and the two worker legs genuinely overlap.

**Causality model:** vector clocks. Each agent is a "process". Every boundary crossing is
a local event that increments the agent's own component. `agent_msg` is the send/receive
synchronization point that carries and merges vectors.

**Correct notion of determinism:** V2 stops claiming a single global total order.
The determinism guarantee becomes:

> For every agent, replay reproduces that agent's own sequence of boundary crossings
> byte-for-byte (`boundary_hash` and `response_json` per `(agent_id, event_type, seq)`),
> **and** the recorded happens-before partial order is reproduced exactly (recomputed
> vector clock equals recorded vector clock for every event). The interleaving of
> concurrent events is explicitly allowed to differ.

This is deterministic replay of a **message-passing (actor) system**: agents share no
mutable memory and communicate only through `agent_msg`, so determinism is a per-agent
property and cross-agent interleaving is free.

**Bonus payoff — causal fork.** Because events carry real vector clocks, `fork` reruns
only the events in the branch point's *causal future* and reuses the recorded values of
events that are *concurrent* with the branch. Forking `worker_a`'s `tool_call` must
**not** rerun `worker_b`.

---

## 3. Hard invariants (do not break)

- The boundary API in `boundaries.py` keeps its exact signatures. Every boundary takes an
  explicit `agent_id` (or `from_agent`), so the interceptor never needs thread-local
  "who am I" state — the caller always states the agent.
- `flightrec.interceptor` keeps `record_into`, `replay_from`, `current`,
  `NoActiveInterceptor`, `ReplayViolation`, `ReplayDrift`, and
  `Interceptor.cross(agent_id, event_type, request_obj, live_fn)` with the same
  signatures. `Interceptor.produced` remains a list of 5-tuples
  `(agent_id, event_type, seq, boundary_hash, response_json)`.
- `Interceptor.next_seq(agent_id, event_type)` keeps per-`(agent, event_type)` semantics.
- `guard_real_call()` raises `ReplayViolation` **iff** the interceptor's phase is
  `REPLAY`. (It must not raise in `RECORD` or `FORK`.)
- `reference_agent.run_agent(task)` returns `{"final": <str>, "answers": {"worker_a":
  ..., "worker_b": ...}}` and `_work(agent_id, sub_question) -> str` keeps that exact
  signature (a test monkeypatches `_work`).
- The store stays append-only / event-sourced.
- `Event` and `Trace` stay constructible from their V1 fields alone (new fields must have
  defaults), so `Event(**e.model_dump()) == e` still holds.

---

## 4. File-by-file change specification

### 4.1 `flightrec/models.py`

Add two fields to `Event`, both with defaults so existing constructors keep working:

```python
class Event(BaseModel):
    event_id: str
    trace_id: str
    seq: int
    logical_clock: int            # V2: the agent's OWN vector component at this event
    wall_clock: float
    agent_id: str
    event_type: str
    request_json: str
    response_json: str
    boundary_hash: str
    vector_clock: str = "{}"      # NEW: canonical JSON of {agent_id: int}
    causal_rank: int = 0          # NEW: sum(vector_clock.values()); used only for ordering
```

`canonical` and `sha256_hex` are unchanged. `Trace` is unchanged.

Rationale: `logical_clock` stays as a per-agent scalar for display; `vector_clock` is the
real causality; `causal_rank` is a denormalized integer that gives a deterministic,
happens-before-consistent total order for storage/display (`a → b ⇒ rank(a) < rank(b)`
because element-wise `<` implies a strictly smaller sum).

### 4.2 `flightrec/clock.py`

Keep `LamportClock` exactly as is (existing `test_clock.py` covers it). Add a vector clock
plus pure helpers:

```python
class VectorClock:
    def __init__(self, agent_id: str, initial: dict | None = None):
        self.agent_id = agent_id
        self.v: dict[str, int] = dict(initial or {})

    def tick(self) -> dict:                     # local event
        self.v[self.agent_id] = self.v.get(self.agent_id, 0) + 1
        return dict(self.v)

    def merge(self, other: dict) -> None:       # receive: element-wise max
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

Note: `merge` is commutative and associative (max), so an agent that merges several
delivered vectors gets the same result regardless of delivery order. This is what makes
recomputed vectors interleaving-independent.

### 4.3 `flightrec/store.py`

- Open the connection with `sqlite3.connect(path, check_same_thread=False)` because worker
  threads write concurrently.
- Add a `threading.Lock` and wrap **every** method that touches the connection
  (`create_trace`, `set_status`, `get_trace`, `list_traces`, `append_event`,
  `get_events`, `get_event`) in `with self._lock:`.
- Extend the `events` schema and `_EVENT_COLS` with the two new columns:
  `vector_clock TEXT NOT NULL DEFAULT '{}'` and `causal_rank INTEGER NOT NULL DEFAULT 0`.
- `append_event` inserts `event.vector_clock` and `event.causal_rank`.
- **Change `get_events` ordering** from `ORDER BY rowid_pk` to:

  ```sql
  ORDER BY causal_rank, agent_id, event_type, seq
  ```

  `(agent_id, event_type, seq)` is globally unique within a trace, so this is a total
  order and a valid linear extension of happens-before. It is fully deterministic because
  it depends only on recorded vectors and keys — never on thread timing or `rowid`.

Migration note: the new columns change the schema. Delete any existing `flightrec.db` (or
add a migration) before first V2 run. Tests use fresh temp DBs, so they are unaffected.

### 4.4 `flightrec/interceptor.py` (core rewrite)

Keep the module-global single `_active` interceptor — all agent threads share it, which is
what we want (one coordinator). Thread safety comes from an internal lock, **not** from
per-thread interceptors.

Phases: `RECORD`, `REPLAY`, `FORK`.

Interceptor internal state (all guarded by `self._lock: threading.Lock`):
- `self._counters: dict[(agent, event_type), int]` — per-agent-per-type seq.
- `self._vectors: dict[agent, VectorClock]` — one vector clock per agent.
- `self._mailbox: dict[agent, list[dict]]` — vectors delivered by messages, consumed at
  the recipient's next tick.
- `self.produced: list[tuple]` — 5-tuples, in append order (API compat).
- `self._vectors_by_key: dict[(agent, event_type, seq), dict]` — the recomputed vector
  for each crossing (used by replay's causal assertion and by sorting).
- `self._recorded: dict[(agent, event_type, seq), Event]` — built once at init from the
  read trace (read-only afterward, safe to read without the lock).
- FORK-only: `self.branch_key: tuple`, `self._branch_vec: dict`, `self.mutation`,
  `self._tainted: set[str]`.

**The lock must never be held while `live_fn()` runs** (that is the slow network call —
holding the lock there would re-serialize the work we parallelized). Compute
seq + vector under the lock, release, run `live_fn`, then re-acquire the lock to write the
event and append to `produced`.

Helper (must run under the lock):

```python
def _tick(self, agent: str) -> dict:
    vc = self._vectors.setdefault(agent, VectorClock(agent))
    for delivered in self._mailbox.pop(agent, []):   # consume inbox
        vc.merge(delivered)
    return vc.tick()                                 # increment own, return snapshot

def _deliver(self, event_type, request_obj, sender_vec, sender_agent) -> None:
    if event_type != "agent_msg":
        return
    to_agent = request_obj["to"]
    self._mailbox.setdefault(to_agent, []).append(dict(sender_vec))
    if self.phase == FORK and sender_agent in self._tainted:
        self._tainted.add(to_agent)                  # taint spreads through messages
```

`cross` contract:

```python
def cross(self, agent_id, event_type, request_obj, live_fn):
    request_json = canonical(request_obj)
    boundary_hash = sha256_hex(request_json)

    with self._lock:
        seq  = self._next_seq(agent_id, event_type)
        vec  = self._tick(agent_id)
        key  = (agent_id, event_type, seq)
        rank = vc_rank(vec)
        self._vectors_by_key[key] = vec
        mode = self._decide_mode(agent_id, key)      # LIVE | REPLAY | MUTATE

    if mode == LIVE:
        value = live_fn()                            # network happens OUTSIDE the lock
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
        raise ReplayDrift(f"request drift at {key}: {rec.boundary_hash[:12]} != {boundary_hash[:12]}")
    if canonical(vec) != rec.vector_clock:
        raise ReplayDrift(f"causal drift at {key}: vector clock differs from recording")

    if mode == MUTATE:
        value = _apply_mutation(json.loads(rec.response_json), self.mutation)
        response_json = canonical(value)
        with self._lock:
            self._tainted.add(agent_id)              # agent goes live from its next event
            self._write(agent_id, event_type, seq, request_json, response_json,
                        boundary_hash, vec, rank)
            self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
            self._deliver(event_type, request_obj, vec, agent_id)
        return value

    # mode == REPLAY
    with self._lock:
        if self.write_trace_id is not None:          # FORK copies past/concurrent into child
            self._write(agent_id, event_type, seq, request_json, rec.response_json,
                        boundary_hash, vec, rank)
        self.produced.append((agent_id, event_type, seq, boundary_hash, rec.response_json))
        self._deliver(event_type, request_obj, vec, agent_id)
    return json.loads(rec.response_json)
```

`_decide_mode` (called under the lock):

```python
def _decide_mode(self, agent, key):
    if self.phase == RECORD:
        return LIVE
    if self.phase == REPLAY:
        return REPLAY
    # FORK:
    if key == self.branch_key:
        return MUTATE
    if agent in self._tainted:
        return LIVE
    rec = self._recorded.get(key)
    if rec is not None and happens_before(self._branch_vec, json.loads(rec.vector_clock)):
        self._tainted.add(agent)                     # safety: in branch's future
        return LIVE
    return REPLAY                                    # past or concurrent → reuse recording
```

`guard_real_call`: unchanged — raise `ReplayViolation` iff `self.phase == REPLAY`. Note
that in `LIVE`/`MUTATE`/`REPLAY` handling above, `live_fn` is invoked **only** in `LIVE`
mode, so a correct replay never reaches `guard_real_call` at all; it fires only if agent
code bypasses a boundary during pure replay.

`_write`: builds an `Event` with `event_id = os.urandom(16).hex()`,
`logical_clock = vec[agent_id]`, `wall_clock = time.time()`, `vector_clock = canonical(vec)`,
`causal_rank = rank`, and `trace_id = self.write_trace_id`; only writes if
`write_trace_id is not None`.

Context managers:

```python
@contextmanager
def record_into(store, trace_id): ...                    # phase=RECORD, read=None,  write=trace_id
@contextmanager
def replay_from(store, trace_id, *, branch_key=None, mutation=None, write_trace_id=None): ...
                                                          # phase=REPLAY, read=trace_id, write=write_trace_id
@contextmanager
def fork_from(store, read_trace_id, *, write_trace_id, branch_key, branch_vec, mutation): ...
                                                          # phase=FORK
```

Keep `replay_from`'s existing keyword parameters (a test calls
`replay_from(store, "t1")`), but FORK now uses the dedicated `fork_from`. Each context
manager saves/restores the previous `_active`.

### 4.5 `flightrec/boundaries.py`

No behavioral change. Confirm `agent_msg` still builds
`request = {"from": from_agent, "to": to_agent, "payload": payload}` and calls
`current().cross(from_agent, "agent_msg", request, live)` — the interceptor reads
`request["to"]` for delivery. All other boundaries unchanged.

### 4.6 `flightrec/agent/reference_agent.py` (threading)

- `_plan(task)` unchanged (planner `llm` call returning two `sub_questions`; the prompt
  must contain the substring `sub_questions` so the fake LLMs branch correctly).
- `_work(agent_id, sub_question) -> str` keeps its exact body and signature: `new_uuid`,
  `now`, `rand`, `tool_call("search", {"query": ..., "seed": ...})`, then `llm`. The
  worker→synthesizer message must live **outside** `_work` (see below) so the monkeypatch
  in `test_unclamped_clock_makes_replay_fail_loudly` still composes.
- `run_agent(task)`:

```python
def run_agent(task):
    sub_questions = _plan(task)

    # planner → worker handoffs happen in the main thread, in fixed order,
    # so planner's events are totally ordered and never race.
    assignments = {wid: b.agent_msg("planner", wid, sq)
                   for wid, sq in zip(WORKER_IDS, sub_questions)}

    answers, errors, lock = {}, {}, threading.Lock()

    def worker_entry(wid):
        try:
            ans = _work(wid, assignments[wid])
            b.agent_msg(wid, "synthesizer", ans)     # real join edge → taints synth on fork
            with lock:
                answers[wid] = ans
        except BaseException as e:                   # capture; threads don't propagate
            with lock:
                errors[wid] = e

    threads = [threading.Thread(target=worker_entry, args=(wid,)) for wid in WORKER_IDS]
    for t in threads: t.start()
    for t in threads: t.join()

    if errors:                                       # re-raise deterministically
        for wid in WORKER_IDS:
            if wid in errors:
                raise errors[wid]

    final = _synthesize(task, answers)
    return {"final": final, "answers": answers}
```

Two things here are load-bearing:

1. **Thread exception propagation.** Exceptions raised inside a `Thread` do **not**
   propagate through `join()`. `ReplayDrift`, `ReplayViolation`, and test `AssertionError`s
   raised in a worker must be captured per worker and re-raised in the main thread
   (in `WORKER_IDS` order, for determinism) after join. Without this, `replay()` never sees
   the drift and the drift test fails.
2. **Planner sends before spawn.** Doing the `planner → worker` `agent_msg` in the main
   thread before `t.start()` keeps all planner-attributed events sequential and establishes
   the happens-before edge via `Thread.start()`'s synchronization, so the worker's first
   `_tick` deterministically merges the planner's vector.

`_synthesize` (synthesizer `llm` call) is unchanged; it reads `answers["worker_a"]` and
`answers["worker_b"]` by fixed key, so its input order is deterministic (never
arrival-order). Preserve this discipline.

### 4.7 `flightrec/replay.py`

```python
def recorded_tuples(store, trace_id):
    # get_events is now causal-ordered, so this list is deterministic.
    return [(e.agent_id, e.event_type, e.seq, e.boundary_hash, e.response_json)
            for e in store.get_events(trace_id)]

def replay(store, trace_id):
    trace = store.get_trace(trace_id)
    if trace is None:
        raise DeterminismError(f"no such trace: {trace_id}")
    expected = recorded_tuples(store, trace_id)
    with itc.replay_from(store, trace_id) as inter:
        run_agent(trace.task)
        produced = list(inter.produced)
        vbk = dict(inter._vectors_by_key)
    # Re-order produced into canonical causal order so it aligns with `expected`.
    produced.sort(key=lambda t: (vc_rank(vbk[(t[0], t[1], t[2])]), t[0], t[1], t[2]))
    if produced != expected:
        # find first divergence for a precise message, then raise DeterminismError
        ...
    return produced
```

The per-crossing `boundary_hash` and vector-clock checks inside `cross` already catch
request drift and causal drift loudly. The final list comparison catches missing/extra
events and value drift. Because concurrent events are sorted by the same deterministic key
on both sides, a correct concurrent run yields `produced == expected`.

### 4.8 `flightrec/fork.py` (causal fork)

No prefix-copy step. A single `fork_from` re-execution writes the entire child trace:
past and concurrent events are replayed from the recording and copied into the child;
the branch event is mutated; the causal future is rerun live.

```python
def fork(store, trace_id, at_event_id, mutation):
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

Keep the signature `fork(store, trace_id, at_event_id, mutation)`.

Worked example (branch = `worker_a`'s `tool_call`): `planner` and `worker_a`'s
`uuid/now/rand` are the branch's causal past → replayed and copied. The `tool_call` is
mutated and taints `worker_a`. `worker_a`'s `llm` and its `worker_a → synthesizer`
message are in the future → live. That message taints `synthesizer`, so the synthesizer
`llm` is live. `worker_b` is concurrent with the branch (`happens_before(branch_vec,
worker_b_vec)` is false) and is never tainted → fully replayed and copied. **`worker_b`
makes zero real calls on fork.**

### 4.9 `flightrec/diff.py`

Align by key rather than by position (positional alignment is fragile once traces can
diverge in length):

```python
def diff(store, trace_a, trace_b):
    a = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_a)}
    b = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_b)}
    keys = set(a) | set(b)

    changed_keys = []
    for k in keys:
        ea, eb = a.get(k), b.get(k)
        if ea is None or eb is None or (ea.boundary_hash, ea.response_json) != (eb.boundary_hash, eb.response_json):
            changed_keys.append(k)

    # branch point = differing event with the smallest causal rank (deterministic tiebreak on key)
    def rank_of(k):
        e = a.get(k) or b.get(k)
        return (e.causal_rank, k)
    branch = min(changed_keys, key=rank_of) if changed_keys else None

    changed_by_agent = {}
    for (agent, _t, _s) in changed_keys:
        changed_by_agent[agent] = changed_by_agent.get(agent, 0) + 1

    return DiffReport(trace_a=trace_a, trace_b=trace_b,
                      branch_index=None,                     # kept for back-compat; may be None
                      branch_event=(branch[0], branch[1], branch[2]) if branch else None,
                      changed_by_agent=changed_by_agent,
                      final_a=_final(store.get_events(trace_a)),
                      final_b=_final(store.get_events(trace_b)))
```

`_final` and `format_report` keep their V1 behavior; `format_report` must still contain
the word "branch". `_final` still returns the last `synthesizer` `llm_call` content.

### 4.10 `flightrec/cli.py`

- `show`: iterate `store.get_events` (now causal order) and additionally print the
  agent's own clock and rank, e.g. append `lc={e.logical_clock} rank={e.causal_rank}`.
  Add an optional `--vector` flag to also print `e.vector_clock`.
- `replay`, `fork`, `diff`, `run`, `ls`, `record_run`: behavior unchanged; they inherit the
  new semantics from the modules above. `replay` still catches
  `(DeterminismError, ReplayDrift, ReplayViolation)` and exits non-zero on drift.

---

## 5. Concurrency correctness checklist (call these out; they are the usual traps)

1. **Lock discipline.** Bookkeeping (counters, vectors, mailbox, `produced`, writes) is
   under `self._lock`. `live_fn()` runs **outside** the lock. Never hold the lock across a
   network call.
2. **SQLite across threads.** Connection opened with `check_same_thread=False` and every
   store method serialized by a `threading.Lock`.
3. **Thread exceptions.** Worker threads capture exceptions; the joiner re-raises them in
   `WORKER_IDS` order. Verified necessary by the drift and zero-real-calls tests.
4. **Deterministic message arrival.** Agents consume inputs by fixed key
   (`answers["worker_a"]`, `answers["worker_b"]`), never by wall-clock arrival. Vector
   merges are max-based, so delivery order does not affect any recomputed vector.
5. **No global phase flip.** V1 flipped `self.phase = RECORD` at the branch — a global
   data race under threads. V2 decides replay-vs-live **per crossing** from causality
   (`_decide_mode` + taint set), never by mutating a shared phase mid-run.
6. **Deterministic ordering.** `get_events` orders by `(causal_rank, agent_id,
   event_type, seq)` — a function of recorded data only, independent of thread timing or
   `rowid`.

---

## 6. Test compatibility

### Existing tests that must still pass unchanged

- `test_models.py` — new `Event` fields have defaults; `model_dump` round-trip holds.
- `test_store.py` — new columns have SQL defaults; `test_events_returned_in_insertion_order`
  still yields `["b","a"]` because both events have `causal_rank=0` and tiebreak
  `(planner, llm_call, seq 0) < (…, seq 1)`.
- `test_clock.py` — `LamportClock` is untouched.
- `test_boundaries.py`, `test_interceptor.py` — single-threaded; `cross`, `next_seq`,
  `produced` 5-tuples, `guard_real_call`, `ReplayDrift` on request mismatch all preserved.
  (The added vector-clock check never fires here because record and replay recompute the
  same vectors.)
- `test_reference_agent.py` — `run_agent` still records all five event types (now including
  `agent_msg` on both the planner→worker and worker→synthesizer edges) and returns
  `answers` for both workers. `search` purity and `run_tool` dispatch unchanged.
- `test_cli_run_show.py` — `record_run` still completes and writes events.
- `test_record_replay.py`:
  - `test_recording_covers_all_four_boundary_types` — subset check, unaffected.
  - `test_replay_is_byte_identical` — `replay()` returns produced sorted into causal order,
    equal to `recorded_tuples` (also causal order).
  - `test_replay_makes_zero_real_calls` — replay makes no live calls; monkeypatched
    `litellm`/`run_tool` never fire.
  - `test_unclamped_clock_makes_replay_fail_loudly` — patched `_work` bakes real time into
    the `llm` prompt; on replay the `llm` request hash drifts → `ReplayDrift` raised in the
    worker thread → captured and re-raised → `pytest.raises((ReplayDrift, DeterminismError))`
    satisfied. (The patched `_work` keeps signature `(agent_id, sub_question)`; the
    worker→synth message stays outside `_work`, so it still composes.)
- `test_fork.py`:
  - `test_fork_shares_prefix_and_diverges_at_branch` — parent and child share an identical
    canonical causal ordering (concurrent and past events have identical vectors and keys
    in both), so the positional prefix comparison over `get_events` holds; the branch event
    keeps its request (same `boundary_hash`) with the mutated `results`.
  - `test_fork_suffix_is_live_and_recorded` — child has live suffix events after the branch;
    `parent_trace_id`, `branch_point_event`, and `status` set as expected.
  - `test_diff_reports_branch_and_changes` — `diff` reports the mutated `tool_call` as the
    branch (smallest causal rank among differing keys) and non-zero downstream changes;
    `format_report` contains "branch".

### New tests to add (`tests/test_vector_clock.py`, `tests/test_concurrency.py`)

1. `VectorClock` unit tests: `tick` increments own component; `merge` takes element-wise
   max; `happens_before` / `concurrent` classify `{a:1}` vs `{a:1,b:1}` (before) and
   `{a:1}` vs `{b:1}` (concurrent).
2. Concurrent record → replay is byte-identical across repeated replays (run `replay`
   twice; assert equal each time), proving interleaving-independence.
3. Recorded events carry causally-consistent vectors: for the recorded
   `worker_a → synthesizer` and `worker_b → synthesizer` messages and the synthesizer
   `llm`, assert `happens_before(msg_vec, synth_vec)`, and assert the two workers'
   `tool_call` events are pairwise `concurrent`.
4. **Causal fork reuses concurrent recording:** fork at one worker's `tool_call` with
   `litellm.completion` and `run_tool` monkeypatched to raise if called for the *other*
   worker's agent; assert the fork succeeds and the other worker's child events are
   byte-identical to the parent's (i.e., it was replayed, not rerun).
5. Thread-exception propagation: force a `ReplayDrift` inside one worker during replay and
   assert it surfaces from `replay()`.

---

## 7. Definition of done

- `python -m pip install -e ".[dev]"` succeeds.
- `python -m pytest -v` is green: all existing tests plus the new vector-clock,
  concurrency, and causal-fork tests.
- A live smoke run (`GROQ_API_KEY` set, or `FLIGHTREC_MODEL`/`OPENAI_API_KEY`):
  `flightrec run … → show → replay (0 real calls, byte-identical) → fork at a worker
  tool_call → diff`. In the diff, the mutated worker and the synthesizer show downstream
  changes; the *other* worker shows **zero** changes and made **zero** real calls during
  the fork.
- Wall-clock: a run's two worker legs overlap (a run is roughly as slow as one worker's
  `tool_call + llm`, not two).

---

## 8. Documentation updates

Update `README.md`:
- "How it works" table: `interceptor.py` now maintains per-agent vector clocks, a message
  mailbox, a thread-safe lock, and three phases (record / replay / fork).
- State the V2 determinism guarantee (per-agent byte-identity + reproduced happens-before
  partial order; concurrent interleaving free).
- Describe causal fork: reruns only the branch's causal future, reuses concurrent events.
- Note the schema change (new `vector_clock`, `causal_rank` columns) and that
  `get_events` returns canonical causal order.
