"""Pure, read-side graph construction for the web DAG viewer. Never mutates the store."""
from __future__ import annotations

import json
from typing import Optional

from .. import diff as diff_mod
from ..clock import happens_before
from ..store import Store

_PREVIEW_LEN = 200


def _preview(s: str) -> str:
    return s if len(s) <= _PREVIEW_LEN else s[:_PREVIEW_LEN - 3] + "..."


def _agent_order(events):
    # first causal appearance; events are already in causal order
    order = []
    for e in events:
        if e.agent_id not in order:
            order.append(e.agent_id)
    return order


def _columns(events):
    ranks = sorted({e.causal_rank for e in events})
    return {r: i for i, r in enumerate(ranks)}   # causal_rank -> column index


def _sequence_edges(events, agents):
    edges = []
    for a in agents:
        lane = [e for e in events if e.agent_id == a]   # already causal-ordered
        for x, y in zip(lane, lane[1:]):
            edges.append({"from": x.event_id, "to": y.event_id, "kind": "sequence"})
    return edges


def _message_edges(events):
    by_agent: dict[str, list] = {}
    for e in events:
        by_agent.setdefault(e.agent_id, []).append(e)   # causal order preserved
    edges = []
    for e in events:
        if e.event_type != "agent_msg":
            continue
        req = json.loads(e.request_json)                 # {"from","to","payload"}
        sender, recipient = e.agent_id, req["to"]
        send_component = json.loads(e.vector_clock).get(sender, 0)
        target = next((r for r in by_agent.get(recipient, [])
                       if json.loads(r.vector_clock).get(sender, 0) >= send_component),
                      None)
        if target is not None:
            edges.append({"from": e.event_id, "to": target.event_id, "kind": "message"})
    return edges


def _fork_context(store: Store, trace) -> Optional[dict]:
    if not (trace.parent_trace_id and trace.branch_point_event):
        return None
    be = store.get_event(trace.branch_point_event)       # lives in the parent trace
    return {"key": (be.agent_id, be.event_type, be.seq), "vec": json.loads(be.vector_clock)}


def _role(event, ctx: Optional[dict]) -> str:
    if ctx is None:
        return "recorded"
    if (event.agent_id, event.event_type, event.seq) == ctx["key"]:
        return "mutated"
    if happens_before(ctx["vec"], json.loads(event.vector_clock)):
        return "live"       # in the branch's causal future -> rerun
    return "reused"         # branch's past or concurrent -> replayed/copied


def build_graph(store: Store, trace_id: str) -> dict:
    trace = store.get_trace(trace_id)
    if trace is None:
        raise ValueError(f"no such trace: {trace_id}")

    events = store.get_events(trace_id)
    agents = _agent_order(events)
    lane_of = {a: i for i, a in enumerate(agents)}
    cols = _columns(events)
    ctx = _fork_context(store, trace)

    nodes = [{
        "event_id": e.event_id,
        "agent_id": e.agent_id,
        "event_type": e.event_type,
        "seq": e.seq,
        "lane": lane_of[e.agent_id],
        "column": cols[e.causal_rank],
        "causal_rank": e.causal_rank,
        "logical_clock": e.logical_clock,
        "vector_clock": json.loads(e.vector_clock),
        "boundary_hash": e.boundary_hash,
        "wall_clock": e.wall_clock,
        "request_preview": _preview(e.request_json),
        "response_preview": _preview(e.response_json),
        "role": _role(e, ctx),
    } for e in events]

    edges = _sequence_edges(events, agents) + _message_edges(events)

    return {
        "trace": {
            "trace_id": trace.trace_id,
            "task": trace.task,
            "status": trace.status,
            "parent_trace_id": trace.parent_trace_id,
            "branch_point_event": trace.branch_point_event,
            "mutation": trace.mutation,
            "agents": agents,
        },
        "nodes": nodes,
        "edges": edges,
    }


def diff_overlay(store: Store, trace_a: str, trace_b: str) -> dict:
    ea = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_a)}
    eb = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(trace_b)}
    changed = [list(k) for k in (set(ea) | set(eb))
               if k not in ea or k not in eb
               or (ea[k].boundary_hash, ea[k].response_json)
                  != (eb[k].boundary_hash, eb[k].response_json)]
    rep = diff_mod.diff(store, trace_a, trace_b)
    return {
        "branch_event": list(rep.branch_event) if rep.branch_event else None,
        "changed_by_agent": rep.changed_by_agent,
        "final_a": rep.final_a,
        "final_b": rep.final_b,
        "changed_keys": changed,
    }
