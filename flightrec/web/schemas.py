"""Pydantic request/response models mirroring the V3 API JSON contract exactly."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    trace_id: str
    task: str
    status: str
    parent_trace_id: Optional[str] = None
    branch_point_event: Optional[str] = None
    created_at: float


class TraceMeta(BaseModel):
    trace_id: str
    task: str
    status: str
    parent_trace_id: Optional[str] = None
    branch_point_event: Optional[str] = None
    mutation: Optional[str] = None
    agents: list[str]


class Node(BaseModel):
    event_id: str
    agent_id: str
    event_type: str
    seq: int
    lane: int
    column: int
    causal_rank: int
    logical_clock: int
    vector_clock: dict[str, int]
    boundary_hash: str
    wall_clock: float
    request_preview: str
    response_preview: str
    role: str


class Edge(BaseModel):
    model_config = {"populate_by_name": True}

    from_: str = Field(..., alias="from")
    to: str
    kind: str


class GraphResponse(BaseModel):
    trace: TraceMeta
    nodes: list[Node]
    edges: list[Edge]


class DiffResponse(BaseModel):
    branch_event: Optional[list] = None
    changed_by_agent: dict[str, int]
    final_a: str
    final_b: str
    changed_keys: list[list]


class ForkRequest(BaseModel):
    at_event_id: str
    mutation: dict


class ForkResponse(BaseModel):
    child_trace_id: str


class RunRequest(BaseModel):
    task: str


class RunResponse(BaseModel):
    trace_id: str
