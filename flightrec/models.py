"""Pydantic models plus canonical JSON + hashing helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel

EventType = Literal["llm_call", "tool_call", "clock", "random", "agent_msg"]


def canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, unicode preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Event(BaseModel):
    event_id: str
    trace_id: str
    seq: int
    logical_clock: int
    wall_clock: float
    agent_id: str
    event_type: str
    request_json: str
    response_json: str
    boundary_hash: str
    vector_clock: str = "{}"
    causal_rank: int = 0


class Trace(BaseModel):
    trace_id: str
    parent_trace_id: Optional[str] = None
    branch_point_event: Optional[str] = None
    mutation: Optional[str] = None
    task: str
    status: str
    created_at: float
