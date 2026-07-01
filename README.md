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
| `flightrec/interceptor.py` | Process-global RECORD/REPLAY context, per-`(agent,event_type)` counters, Lamport clock, network guard. |
| `flightrec/store.py` | Append-only SQLite event store (event-sourced). |
| `flightrec/replay.py` | Faithful replay + the byte-identical determinism assertion. |
| `flightrec/fork.py` | Copy prefix → mutate at branch → run suffix live into a child trace. |
| `flightrec/diff.py` | Align two traces, find the first divergence, report changed events per agent. |
| `flightrec/agent/` | The reference agent and the pure mock `search` tool. |

The replay match key is `(agent_id, event_type, seq)`; a per-event `boundary_hash` of the
canonical request detects drift. `wall_clock` is stored for display only and never used for
matching. The Lamport clock is recorded per event so adding real concurrency later is a
drop-in.

## Tests

```
python -m pytest -v        # all offline; live `run`/`fork` need GROQ_API_KEY
```

The suite runs fully offline via a fake LiteLLM fixture. It proves: recordings cover all
four boundary types, replay is byte-identical, replay makes zero real calls, an unclamped
clock makes replay fail loudly, forks share an identical prefix and diverge exactly at the
branch point with a live suffix, and diff reports the correct branch point.
