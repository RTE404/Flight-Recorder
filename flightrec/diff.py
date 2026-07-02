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
