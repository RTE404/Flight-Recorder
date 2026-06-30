"""Process-global RECORD/REPLAY interceptor: counters, Lamport clock, network guard."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from .clock import LamportClock
from .models import Event, canonical, sha256_hex
from .store import Store

RECORD = "record"
REPLAY = "replay"


class NoActiveInterceptor(Exception):
    pass


class ReplayViolation(Exception):
    """A real external call was attempted during replay (unclamped boundary)."""


class ReplayDrift(Exception):
    """Replay diverged from the recording (missing event or request hash mismatch)."""


def _apply_mutation(recorded_value: Any, mutation: Any) -> Any:
    if isinstance(recorded_value, dict) and isinstance(mutation, dict):
        return {**recorded_value, **mutation}
    return mutation


class Interceptor:
    def __init__(self, store: Store, *, phase: str, read_trace_id: Optional[str],
                 write_trace_id: Optional[str], branch_key: Optional[tuple] = None,
                 mutation: Any = None):
        self.store = store
        self.phase = phase
        self.read_trace_id = read_trace_id
        self.write_trace_id = write_trace_id
        self.branch_key = branch_key
        self.mutation = mutation
        self.lamport = LamportClock()
        self._counters: dict[tuple[str, str], int] = {}
        self.produced: list[tuple] = []
        self._recorded: dict[tuple[str, str, int], Event] = {}
        if read_trace_id is not None:
            for e in store.get_events(read_trace_id):
                self._recorded[(e.agent_id, e.event_type, e.seq)] = e

    def next_seq(self, agent_id: str, event_type: str) -> int:
        key = (agent_id, event_type)
        seq = self._counters.get(key, 0)
        self._counters[key] = seq + 1
        return seq

    def guard_real_call(self) -> None:
        if self.phase == REPLAY:
            raise ReplayViolation(
                "Attempted a real external call during REPLAY — unclamped boundary."
            )

    def _write_event(self, agent_id, event_type, seq, request_json, response_json,
                     boundary_hash, logical_clock) -> None:
        if self.write_trace_id is None:
            return
        self.store.append_event(Event(
            event_id=os.urandom(16).hex(), trace_id=self.write_trace_id, seq=seq,
            logical_clock=logical_clock, wall_clock=time.time(), agent_id=agent_id,
            event_type=event_type, request_json=request_json, response_json=response_json,
            boundary_hash=boundary_hash,
        ))

    def cross(self, agent_id: str, event_type: str, request_obj: Any,
              live_fn: Callable[[], Any]) -> Any:
        seq = self.next_seq(agent_id, event_type)
        request_json = canonical(request_obj)
        boundary_hash = sha256_hex(request_json)

        if self.phase == RECORD:
            value = live_fn()
            lc = self.lamport.tick()
            response_json = canonical(value)
            self._write_event(agent_id, event_type, seq, request_json, response_json,
                              boundary_hash, lc)
            self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
            return value

        # REPLAY phase
        rec = self._recorded.get((agent_id, event_type, seq))
        if rec is None:
            raise ReplayDrift(
                f"No recorded event for {(agent_id, event_type, seq)} — extra boundary "
                f"crossing during replay."
            )
        if rec.boundary_hash != boundary_hash:
            raise ReplayDrift(
                f"Request drift at {(agent_id, event_type, seq)}: recorded hash "
                f"{rec.boundary_hash[:12]} != live {boundary_hash[:12]}"
            )
        self.lamport.update(rec.logical_clock)

        if self.branch_key is not None and (agent_id, event_type, seq) == self.branch_key:
            value = _apply_mutation(json.loads(rec.response_json), self.mutation)
            response_json = canonical(value)
            self._write_event(agent_id, event_type, seq, request_json, response_json,
                              boundary_hash, rec.logical_clock)
            self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
            self.phase = RECORD  # suffix runs live
            return value

        self.produced.append((agent_id, event_type, seq, boundary_hash, rec.response_json))
        return json.loads(rec.response_json)


_active: Optional[Interceptor] = None


def current() -> Interceptor:
    if _active is None:
        raise NoActiveInterceptor("No active interceptor — use record_into/replay_from.")
    return _active


@contextmanager
def record_into(store: Store, trace_id: str):
    global _active
    prev = _active
    _active = Interceptor(store, phase=RECORD, read_trace_id=None, write_trace_id=trace_id)
    try:
        yield _active
    finally:
        _active = prev


@contextmanager
def replay_from(store: Store, trace_id: str, *, branch_key: Optional[tuple] = None,
                mutation: Any = None, write_trace_id: Optional[str] = None):
    global _active
    prev = _active
    _active = Interceptor(store, phase=REPLAY, read_trace_id=trace_id,
                          write_trace_id=write_trace_id, branch_key=branch_key,
                          mutation=mutation)
    try:
        yield _active
    finally:
        _active = prev
