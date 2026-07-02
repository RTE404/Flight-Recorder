"""Causal fork: rerun only the branch's causal future; reuse past/concurrent events."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .models import Trace, canonical
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
    branch_vec = json.loads(branch.vector_clock)
    child_id = _new_child_id()
    store.create_trace(Trace(
        trace_id=child_id, parent_trace_id=trace_id, branch_point_event=at_event_id,
        mutation=canonical(mutation), task=parent.task, status="recording",
        created_at=time.time(),
    ))
    try:
        with itc.fork_from(store, trace_id, write_trace_id=child_id,
                           branch_key=branch_key, branch_vec=branch_vec, mutation=mutation):
            run_agent(parent.task)
        store.set_status(child_id, "complete")
    except Exception:
        store.set_status(child_id, "failed")
        raise
    return child_id
