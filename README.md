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

The reference system is a sequential pipeline — **planner → worker_a / worker_b →
synthesizer** — running on raw LiteLLM tool-calling. The mock `search` tool is intentionally
non-deterministic (its result depends on a random seed drawn per run), so without
record/replay a run would never reproduce. That makes the determinism guarantee meaningful:
`flightrec replay` re-derives the entire event sequence and asserts it is byte-identical to
the recording.

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
event sequence equals the recording on `(agent_id, event_type, seq, boundary_hash,
response_json)`. `fork` copies the recorded prefix, substitutes the mutated value at the
branch event, then flips to live recording so the suffix genuinely re-runs and may diverge.

## How it works

| Piece | Responsibility |
|-------|----------------|
| `flightrec/boundaries.py` | The only sanctioned non-determinism: `llm`, `tool_call`, `now`, `new_uuid`, `rand`, `agent_msg`. |
| `flightrec/interceptor.py` | Process-global RECORD/REPLAY/FORK interceptor: per-agent vector clocks, a per-recipient message mailbox, a thread-safe lock, and per-crossing mode decisions (live / replay / mutate). |
| `flightrec/store.py` | Append-only SQLite event store (event-sourced). |
| `flightrec/replay.py` | Faithful replay + the byte-identical determinism assertion. |
| `flightrec/fork.py` | Copy prefix → mutate at branch → run suffix live into a child trace. |
| `flightrec/diff.py` | Align two traces, find the first divergence, report changed events per agent. |
| `flightrec/agent/` | The reference agent and the pure mock `search` tool. |

The replay match key is `(agent_id, event_type, seq)`; a per-event `boundary_hash` of the
canonical request detects drift. `wall_clock` is stored for display only and never used for
matching. The Lamport clock is recorded per event so adding real concurrency later is a
drop-in.

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
