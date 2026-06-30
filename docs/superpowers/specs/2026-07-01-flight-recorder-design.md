# Flight Recorder — MVP Design

**Date:** 2026-07-01
**Status:** Approved (ready for implementation plan)

## 1. Purpose & success criterion

Flight Recorder records everything a sequential multi-agent AI system does, replays a
run *deterministically* with zero new LLM/tool calls, lets the user **fork** a run at any
recorded event (mutate one value, re-run the suffix live), and **diffs** two timelines.

The MVP succeeds or fails on one property: **a replayed run reproduces the original run
byte-for-byte.** Everything else serves that. Replay must be free and offline; any real
external call during replay is a bug.

## 2. Core mechanism

An agent is a deterministic program that leaks non-determinism through four boundaries:
LLM calls, tool calls, the clock, and randomness. Record every value crossing those
boundaries *into* the agent; replay by intercepting the same boundaries and returning the
recorded values. The agent's own logic runs for real; only the boundaries are scripted.

**Time-travel / fork** = replay the recorded prefix faithfully up to event N, substitute a
different value at N, then switch the interceptor to *live* mode so everything after N runs
for real and is recorded as a new child branch.

## 3. Tech stack

- Python 3.11+ (verified: 3.11.7 available as `python`).
- **LiteLLM** — single provider-agnostic LLM boundary.
- **SQLite** via stdlib `sqlite3` — event-sourced, append-only. No ORM.
- **Pydantic v2** — event/trace models.
- **Typer** — CLI.
- **pytest** — acceptance tests.

No web framework, no async/concurrency, sequential agents only.

### LLM provider decision

Recording always hits a **real** provider (decision: "real key required"). The target is
**Groq's free tier** — `GROQ_API_KEY` in env, no cost. Model is env-driven: default
`groq/llama-3.1-8b-instant` (a current free Groq model), overridable via `FLIGHTREC_MODEL`.
LiteLLM auto-reads the provider key (`GROQ_API_KEY`, `OPENAI_API_KEY`, …) from env, so
"read provider/key from env" is satisfied without hardcoding.

### Test strategy decision

- Acceptance tests that need a *recording* require a real key in env and **skip if absent**.
- **Replay tests are fully mocked**: litellm and the real tool are monkeypatched to raise
  if called, proving replay makes zero real calls.

## 4. Repository structure

```
flightrec/
  __init__.py
  models.py          # Pydantic Event / Trace models + canonical JSON helper
  store.py           # SQLite event store (schema, append, read, trace tree)
  clock.py           # Lamport logical clock
  interceptor.py     # global RECORD/REPLAY context, mode switch, per-agent counters, network guard
  boundaries.py      # the ONLY sanctioned non-deterministic primitives
  replay.py          # replay engine + determinism assertion
  fork.py            # branch manager
  diff.py            # trace diff + report
  cli.py             # Typer entrypoint
  agent/
    __init__.py
    reference_agent.py   # planner -> 2 workers -> synthesizer
    tools.py             # mock tools, all routed through boundaries
tests/
  test_record_replay.py
  test_fork.py
pyproject.toml
README.md
docs/superpowers/specs/2026-07-01-flight-recorder-design.md
```

## 5. Data model

SQLite schema is **append-only** — never UPDATE or DELETE a row.

```sql
CREATE TABLE traces (
    trace_id            TEXT PRIMARY KEY,
    parent_trace_id     TEXT,            -- NULL for root recordings
    branch_point_event  TEXT,            -- event_id in parent where this fork diverged
    mutation            TEXT,            -- JSON: what was changed at the fork
    task                TEXT,            -- the input task/question
    status              TEXT,            -- 'recording' | 'complete' | 'failed'
    created_at          REAL
);

CREATE TABLE events (
    event_id        TEXT PRIMARY KEY,
    trace_id        TEXT NOT NULL,
    seq             INTEGER NOT NULL,    -- per-(agent,event_type) monotonic counter, the REPLAY MATCH KEY
    logical_clock   INTEGER NOT NULL,    -- Lamport clock (global causal order)
    wall_clock      REAL NOT NULL,       -- DISPLAY ONLY — never branch logic on this
    agent_id        TEXT NOT NULL,       -- 'planner' | 'worker_a' | 'worker_b' | 'synthesizer'
    event_type      TEXT NOT NULL,       -- 'llm_call' | 'tool_call' | 'clock' | 'random' | 'agent_msg'
    request_json    TEXT NOT NULL,       -- canonical JSON (sorted keys) of the request/input
    response_json   TEXT NOT NULL,       -- canonical JSON of the value returned across the boundary
    boundary_hash   TEXT NOT NULL,       -- sha256 of canonical request_json (for drift detection)
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);
```

Pydantic models mirror these. All JSON serialized canonically:
`canonical(obj) = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
`boundary_hash = sha256(canonical(request_json))`.

**Replay match key:** `(agent_id, event_type, seq)`. During faithful replay, the Nth
boundary crossing of a given type by a given agent maps to the recorded event with that
`seq`. `boundary_hash` is checked on the incoming request; mismatch = drift = raise.

## 6. The boundary contract (`boundaries.py`)

The only sanctioned ways to touch non-determinism. Agent and tool code MUST use these and
never call stdlib equivalents directly.

```python
def llm(messages, *, agent_id, **kwargs) -> dict
def tool_call(name, args, *, agent_id)
def now(*, agent_id) -> float
def new_uuid(*, agent_id) -> str
def rand(*, agent_id) -> float
```

Each, on entry:
1. asks the active interceptor for the next `seq` for `(agent_id, event_type)`;
2. **RECORD:** performs the real operation, writes an event row, returns the value;
3. **REPLAY:** looks up the recorded event by `(agent_id, event_type, seq)`, verifies
   `boundary_hash` matches the current request (raise `ReplayDrift` on mismatch), returns
   recorded `response_json` **without performing any real operation**;
4. bumps the Lamport clock.

**Network guard:** in REPLAY, the code path that would call LiteLLM / a real tool raises
`ReplayViolation` if ever reached. In normal replay the recorded-lookup path returns first,
so the guard only fires on an unclamped boundary. Treat any `ReplayViolation` as a clamping
bug, not a test bug.

**Interceptor:** module-level singleton holding `(mode, trace_id, store, counters, lamport)`.
Context managers `record_into(trace_id)` and `replay_from(trace_id, until_seq=None,
mutation=None)`.

## 7. Lamport clock

One Lamport counter in the interceptor, bumped on every event and on every `agent_msg`
(handoff). Stored per-event as `logical_clock`. Per-agent "last seen" tracking is included
so the bump-on-receive rule is already concurrency-ready, even though execution is linear
now. `wall_clock` is display-only; ordering uses `seq` and `logical_clock` only.

## 8. Reference agent (`agent/reference_agent.py`)

Sequential pipeline so boundaries are obvious:

1. **Planner** — one `llm()`: task → JSON plan with two sub-questions (one per worker).
2. **Worker A / Worker B** — each: `now()` + `new_uuid()` stamped into the prompt as a
   request id, one `tool_call('search', {...})`, then one `llm()` to extract an answer.
3. **Synthesizer** — one `llm()` combining both workers' answers into a final response.

`agent/tools.py` mock `search` is intentionally non-deterministic (results seeded by
`boundaries.now()` / `boundaries.rand()`), so without record/replay the run never
reproduces — making the determinism test meaningful while staying recordable.

`agent_msg` event recorded at each handoff (planner→workers, workers→synthesizer). Every
step tagged with the correct `agent_id`.

## 9. Replay, fork, diff

**`replay.py` — `replay(trace_id) -> new_event_list`:** re-run the agent under
`replay_from(trace_id)`, feeding recorded values at every boundary. Then **assert** the
freshly produced event sequence equals the recorded one, compared on
`(agent_id, event_type, seq, boundary_hash, response_json)`. First divergence raised with a
clear before/after diff. This assertion is the MVP's proof of correctness and a CLI command.

**`fork.py` — `fork(trace_id, at_event_id, mutation) -> new_trace_id`:**
1. create a child trace (`parent_trace_id`, `branch_point_event`, `mutation`);
2. replay faithfully for every boundary crossing ordered before the branch event;
3. at the branch event return the **mutated** value instead of the recorded one;
4. flip the interceptor to RECORD for everything after — suffix runs live, recorded into
   the child trace;
5. return the new `trace_id`.

**Prefix handling (decision: COPY):** prefix events up to and including the mutated branch
event are **copied** into the child trace, so the child is fully self-contained
(append-only, simplest `show`/`diff` queries). Live suffix events are appended after.

**Mutation format:** JSON merged into / replacing the recorded `response_json` of the
branch event (e.g. `--set '{"results": [...]}'` for a tool_call). Stored verbatim in
`traces.mutation`. The suffix is genuinely live: control flow may diverge and make
different calls than the original — expected and correct.

**`diff.py` — `diff(trace_a, trace_b) -> report`:** align the two event lists (identical up
to the branch point by construction), find the first divergence, count changed downstream
events grouped by agent, and print: branch point, number/agents of changed events, and
old-vs-new final output. Plain text.

## 10. CLI (`cli.py`, Typer)

```
flightrec run "<task>"                         # run agent live, record, print trace_id
flightrec ls                                   # list traces (id, parent, status, task)
flightrec show <trace_id>                       # print the event log for a trace
flightrec replay <trace_id>                     # faithful replay + determinism assertion
flightrec fork <trace_id> --at <event_id> --set '<json>'   # fork with a mutation, run suffix live
flightrec diff <trace_a> <trace_b>              # divergence report
```

## 11. Build order

1. `models.py` + `store.py` — schema, append, read back, canonical JSON + hashing; round-trip tests.
2. `clock.py` + `interceptor.py` — global context, counters, mode switch, network guard.
3. `boundaries.py` — four primitives wired to the interceptor.
4. `agent/tools.py` + `agent/reference_agent.py` — running live, fully recordable.
5. `flightrec run` + `flightrec show` — confirm a real run produces a complete trace.
6. `replay.py` + `flightrec replay` — make the determinism assertion pass; guard never fires.
7. `fork.py` + `flightrec fork`.
8. `diff.py` + `flightrec diff`.
9. Tests green; README walkthrough (record → replay → fork).

## 12. Acceptance criteria

`tests/test_record_replay.py`:
- Recording produces a trace whose events cover all four boundary types.
- `replay(trace_id)` produces an event sequence byte-identical to the recording.
- Replay makes zero real LLM/tool calls (monkeypatch to raise if called).
- A deliberately unclamped `datetime.now()` in a fixture makes replay fail loudly (proves
  drift detection).

`tests/test_fork.py`:
- Forking at a tool-call event with a mutated result yields a new trace sharing an
  identical prefix and diverging exactly at the branch point.
- The forked suffix contains live (newly recorded) events.
- `diff` reports the correct branch point and a non-empty set of changed downstream events.

## 13. Critical gotchas

- All non-determinism goes through `boundaries.py`. One direct `datetime.now()`/`uuid4()`/
  `random` call silently breaks replay. Guard + assertion exist to catch this.
- `wall_clock` is display-only; never branch matching/control flow on it.
- Canonical JSON everywhere so `boundary_hash` is reproducible.
- Append-only store; forks create new traces, never mutate recorded events.
- Lamport clock per-agent, bump-on-message now, so adding concurrency later is a drop-in.
- Replay must be free and offline.
