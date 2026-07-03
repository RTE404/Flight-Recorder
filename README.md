# Flight Recorder

Record everything a multi-agent system does, replay it deterministically with **zero new
LLM/tool calls**, fork a run at any event (mutate one value, re-run the rest live), and diff
the two timelines.

## Why

An agent is a deterministic program that leaks non-determinism through four boundaries: LLM
calls, tool calls, the clock, and randomness. Record every value that crosses those
boundaries; replay by feeding the recorded values back in. The agent's own logic runs for
real — only the boundaries are scripted. Replay is free and offline; any real external call
during replay raises `ReplayViolation`, which is how unclamped boundaries get caught.

The reference system is a **planner → worker_a / worker_b → synthesizer** pipeline where the
two workers run concurrently on real OS threads (see "Concurrency & determinism (V2)" below),
running on raw LiteLLM tool-calling. The mock `search` tool is intentionally non-deterministic
(its result depends on a random seed drawn per run), so without record/replay a run would
never reproduce. That makes the determinism guarantee meaningful: `flightrec replay`
re-derives each agent's event sequence and asserts it is byte-identical to the recording.

## What's new in V2

V1 was a strictly **sequential** pipeline: it worked only because a single thread made three
orderings coincide (execution order, SQLite `rowid`, and the list `replay` compared against).
V2 makes the two workers run on real OS threads and replaces that accidental total order with
an explicit causality model. The concrete changes:

- **Real parallelism.** `worker_a` and `worker_b` run on `threading.Thread`s and genuinely
  overlap — a run is about as slow as *one* worker's `tool_call + llm`, not two. The
  planner→worker handoffs happen in the main thread before spawn; the joiner captures each
  worker's exceptions and re-raises them in a fixed order so a `ReplayDrift` raised *inside* a
  thread still surfaces from `replay()`.
- **Vector clocks instead of a global counter.** Each agent is a "process" with its own
  vector-clock component; `agent_msg` is the sole send/receive sync point, merging vectors via
  a mailbox (element-wise max, so delivery order can't change a recomputed vector). The old
  Lamport scalar is retained only for display.
- **A new, honest determinism guarantee** — per-agent byte-identity **plus** a reproduced
  happens-before partial order, with concurrent interleaving explicitly free (stated in full
  under "Concurrency & determinism (V2)" below).
- **Causal fork.** Forking one worker's event reruns only that event's *causal future* live
  and reuses (replays) everything in its past or concurrent with it. Forking `worker_a`'s
  `tool_call` makes **zero** real calls for `worker_b`.
- **Thread-safe store.** SQLite opens with `check_same_thread=False` and every store method is
  serialized by a lock. `get_events` now returns a deterministic causal order
  (`ORDER BY causal_rank, agent_id, event_type, seq`) that never depends on thread timing.
- **Schema change.** Events gain two columns, `vector_clock` and `causal_rank` (see the
  migration note in Install).
- **Interceptor rewrite.** One process-global, lock-protected interceptor with three phases
  (record / replay / fork) that decides live-vs-replay-vs-mutate **per crossing** from
  causality + a taint set — replacing V1's mid-run `phase = RECORD` flip (a data race under
  threads). The lock is never held across the network call, so parallelism is preserved.
- **CLI.** `flightrec show` prints each event's `rank=` and takes an optional `--vector` flag
  to also print the full `vector_clock`.
- **Tests.** New `tests/test_vector_clock.py` and `tests/test_concurrency.py` cover the clock
  algebra, interleaving-independent replay, causal-fork reuse, thread-exception propagation,
  and real wall-clock overlap (44 tests total, all offline).

## What's new in V3

V3 adds a local web UI for browsing recorded traces — no new record/replay/fork semantics,
purely a read-side view over the V2 store. The concrete changes:

- **`flightrec serve` command.** Launches a FastAPI app (`flightrec/web/server.py`) at
  `http://127.0.0.1:8000`, local only, no authentication.
- **Swimlane DAG viewer.** One lane per agent, events placed left-to-right by causal rank,
  colored by fork role (recorded / reused / mutated / live). Click a node for its full detail
  (vector clock, causal rank, request/response); fork or diff two traces from the UI (see
  "Web viewer (V3)" below).
- **Live websocket updates.** While a run is in progress, `/ws/traces/{trace_id}` streams the
  graph as it grows; each push replaces the whole node/edge set rather than appending, since a
  late-written event can carry a lower causal rank than one already shown.
- **Read-side graph computation.** `flightrec/web/graph.py` derives causal edges, fork roles,
  and the diff overlay entirely at read time from `happens_before` — nothing new is persisted.
- **One V2 store change.** SQLite now opens in WAL journal mode (`flightrec/store.py`), so the
  server can read a trace while `flightrec run`/`fork` is still writing it. No schema change,
  no change to the interceptor, fork, replay, or diff logic.
- **Frontend.** Static HTML/CSS/JS (`flightrec/static/`) using Cytoscape.js from a CDN, no
  build step.
- **Tests.** New `tests/test_web_graph.py` and `tests/test_web_api.py` cover graph
  construction, fork-role assignment, and every HTTP/websocket endpoint — fully offline via the
  fake LLM fixture (54 tests total). The `dev` extra now pulls in the `web` extra
  (`flightrec[web]`) so `pip install -e ".[dev]"` alone is enough to run the full suite.

## Install

```
python -m pip install -e ".[dev]"
export GROQ_API_KEY=...        # Groq free tier
# or point at another provider: export FLIGHTREC_MODEL=gpt-4o-mini ; export OPENAI_API_KEY=...
```

`FLIGHTREC_MODEL` overrides the default model (`groq/llama-3.1-8b-instant`). `FLIGHTREC_DB`
overrides the SQLite path (default `flightrec.db`).

V2 adds `vector_clock`/`causal_rank` columns to the events table. If you have an existing
`flightrec.db` from V1, delete it before your first V2 run (it's gitignored, so this is safe).

V3's web viewer needs the `web` extra: `python -m pip install -e ".[dev,web]"`.

## Walkthrough

```
flightrec run "Compare Postgres and SQLite for a small app"   # -> tr_abc123
flightrec show tr_abc123                                       # the event log
flightrec replay tr_abc123                                     # ✓ deterministic, 0 real calls

# pick a tool_call event_id from `show`, then fork with a mutated result:
flightrec fork tr_abc123 --at <event_id> \
    --set '{"results": ["sqlite is faster for this workload"]}'   # -> tr_def456

flightrec diff tr_abc123 tr_def456                            # branch point + changed events
```

`replay` re-runs the agent against the recorded boundary values and asserts the produced
events equal the recording on `(agent_id, event_type, seq, boundary_hash, response_json)`,
per agent. `fork` re-runs the recording as a causal fork: it replays events that are in the
branch's past or concurrent with it, mutates the branch event, and re-runs the branch's
causal future live so the suffix genuinely diverges (see "Concurrency & determinism (V2)").

## How it works

| Piece | Responsibility |
|-------|----------------|
| `flightrec/boundaries.py` | The only sanctioned non-determinism: `llm`, `tool_call`, `now`, `new_uuid`, `rand`, `agent_msg`. |
| `flightrec/interceptor.py` | Process-global RECORD/REPLAY/FORK interceptor: per-agent vector clocks, a per-recipient message mailbox, a thread-safe lock, and per-crossing mode decisions (live / replay / mutate). |
| `flightrec/store.py` | Append-only SQLite event store (event-sourced). |
| `flightrec/replay.py` | Faithful replay + the byte-identical determinism assertion. |
| `flightrec/fork.py` | Causal fork: replay past/concurrent events, mutate at the branch, re-run the causal future live into a child trace. |
| `flightrec/diff.py` | Align two traces by `(agent_id, event_type, seq)`, report changed events per agent and the branch point. |
| `flightrec/agent/` | The reference agent and the pure mock `search` tool. |

The replay match key is `(agent_id, event_type, seq)`; a per-event `boundary_hash` of the
canonical request detects drift. `wall_clock` is stored for display only and never used for
matching. Each event also records a per-agent `vector_clock` so replay/fork/diff stay correct
under real concurrency (see below).

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

## Tests

```
python -m pytest -v        # all offline; live `run`/`fork` need GROQ_API_KEY
```

The suite runs fully offline via a fake LiteLLM fixture. It proves: recordings cover all
boundary types (including `agent_msg`), replay is byte-identical per-agent and reproduces
the recorded happens-before order, replay makes zero real calls, an unclamped clock makes
replay fail loudly even when the drift originates inside a worker thread, forks reuse the
concurrent worker's recording untouched while rerunning only the mutated worker's causal
future, diff reports the correct branch point by causal rank, and the two worker legs
provably overlap in wall-clock time.
