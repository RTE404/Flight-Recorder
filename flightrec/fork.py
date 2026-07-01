"""Fork a recorded trace: copy prefix, mutate the branch event, run the suffix live."""
from __future__ import annotations

import time
import uuid
from typing import Any

from .models import Event, Trace, canonical
from .store import Store
from . import interceptor as itc
from .agent.reference_agent import run_agent


def _new_child_id() -> str:
    return "tr_" + uuid.uuid4().hex[:12]


def fork(store: Store, trace_id: str, at_event_id: str, mutation: Any) -> str:
    parent = store.get_trace(trace_id)
    if parent is None:
        raise ValueError(f"no such trace: {trace_id}")
    branch = store.get_event(at_event_id)
    if branch is None or branch.trace_id != trace_id:
        raise ValueError(f"event {at_event_id} not found in trace {trace_id}")

    branch_key = (branch.agent_id, branch.event_type, branch.seq)
    child_id = _new_child_id()
    store.create_trace(Trace(
        trace_id=child_id, parent_trace_id=trace_id, branch_point_event=at_event_id,
        mutation=canonical(mutation), task=parent.task, status="recording",
        created_at=time.time(),
    ))

    # Copy the faithful prefix (strictly before the branch event).
    for e in store.get_events(trace_id):
        if e.logical_clock < branch.logical_clock:
            store.append_event(Event(
                event_id=uuid.uuid4().hex, trace_id=child_id, seq=e.seq,
                logical_clock=e.logical_clock, wall_clock=e.wall_clock, agent_id=e.agent_id,
                event_type=e.event_type, request_json=e.request_json,
                response_json=e.response_json, boundary_hash=e.boundary_hash,
            ))

    try:
        with itc.replay_from(store, trace_id, branch_key=branch_key, mutation=mutation,
                             write_trace_id=child_id):
            run_agent(parent.task)
        store.set_status(child_id, "complete")
    except Exception:
        store.set_status(child_id, "failed")
        raise
    return child_id
