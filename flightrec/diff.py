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
