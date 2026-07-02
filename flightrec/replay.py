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
