"""Append-only SQLite event store."""
from __future__ import annotations

import sqlite3
import threading
from typing import Optional

from .models import Event, Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id            TEXT PRIMARY KEY,
    parent_trace_id     TEXT,
    branch_point_event  TEXT,
    mutation            TEXT,
    task                TEXT,
    status              TEXT,
    created_at          REAL
);
CREATE TABLE IF NOT EXISTS events (
    rowid_pk        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT UNIQUE NOT NULL,
    trace_id        TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    logical_clock   INTEGER NOT NULL,
    wall_clock      REAL NOT NULL,
    agent_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    response_json   TEXT NOT NULL,
    boundary_hash   TEXT NOT NULL,
    vector_clock    TEXT NOT NULL DEFAULT '{}',
    causal_rank     INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
);
"""

_EVENT_COLS = (
    "event_id, trace_id, seq, logical_clock, wall_clock, agent_id, "
    "event_type, request_json, response_json, boundary_hash, "
    "vector_clock, causal_rank"
)
_TRACE_COLS = (
    "trace_id, parent_trace_id, branch_point_event, mutation, task, status, created_at"
)


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def create_trace(self, trace: Trace) -> None:
        with self._lock:
            self.conn.execute(
                f"INSERT INTO traces ({_TRACE_COLS}) VALUES (?,?,?,?,?,?,?)",
                (trace.trace_id, trace.parent_trace_id, trace.branch_point_event,
                 trace.mutation, trace.task, trace.status, trace.created_at),
            )
            self.conn.commit()

    def set_status(self, trace_id: str, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE traces SET status = ? WHERE trace_id = ?", (status, trace_id)
            )
            self.conn.commit()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_TRACE_COLS} FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return Trace(**dict(row)) if row else None

    def list_traces(self) -> list[Trace]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_TRACE_COLS} FROM traces ORDER BY created_at, trace_id"
            ).fetchall()
        return [Trace(**dict(r)) for r in rows]

    def append_event(self, event: Event) -> None:
        with self._lock:
            self.conn.execute(
                f"INSERT INTO events ({_EVENT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (event.event_id, event.trace_id, event.seq, event.logical_clock,
                 event.wall_clock, event.agent_id, event.event_type, event.request_json,
                 event.response_json, event.boundary_hash, event.vector_clock,
                 event.causal_rank),
            )
            self.conn.commit()

    def get_events(self, trace_id: str) -> list[Event]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_EVENT_COLS} FROM events WHERE trace_id = ? "
                f"ORDER BY causal_rank, agent_id, event_type, seq",
                (trace_id,),
            ).fetchall()
        return [Event(**dict(r)) for r in rows]

    def get_event(self, event_id: str) -> Optional[Event]:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_EVENT_COLS} FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return Event(**dict(row)) if row else None

    def close(self) -> None:
        with self._lock:
            self.conn.close()
