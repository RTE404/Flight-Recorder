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
