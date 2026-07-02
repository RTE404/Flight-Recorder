"""Process-global RECORD/REPLAY/FORK interceptor: per-agent vector clocks, mailbox,
thread-safe lock, network guard."""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from .clock import VectorClock, happens_before, vc_rank
from .models import Event, canonical, sha256_hex
from .store import Store

RECORD = "record"
REPLAY = "replay"
FORK = "fork"

LIVE = "live"
MUTATE = "mutate"


class NoActiveInterceptor(Exception):
    pass


class ReplayViolation(Exception):
    """A real external call was attempted during replay (unclamped boundary)."""


class ReplayDrift(Exception):
    """Replay diverged from the recording (missing event, request/causal mismatch)."""


def _apply_mutation(recorded_value: Any, mutation: Any) -> Any:
    if isinstance(recorded_value, dict) and isinstance(mutation, dict):
        return {**recorded_value, **mutation}
    return mutation


class Interceptor:
    def __init__(self, store: Store, *, phase: str, read_trace_id: Optional[str],
                 write_trace_id: Optional[str], branch_key: Optional[tuple] = None,
                 branch_vec: Optional[dict] = None, mutation: Any = None):
        self.store = store
        self.phase = phase
        self.read_trace_id = read_trace_id
        self.write_trace_id = write_trace_id
        self.branch_key = branch_key
        self._branch_vec = branch_vec or {}
        self.mutation = mutation

        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], int] = {}
        self._vectors: dict[str, VectorClock] = {}
        self._mailbox: dict[str, list[dict]] = {}
        self.produced: list[tuple] = []
        self._vectors_by_key: dict[tuple[str, str, int], dict] = {}
        self._tainted: set[str] = set()

        self._recorded: dict[tuple[str, str, int], Event] = {}
        if read_trace_id is not None:
            for e in store.get_events(read_trace_id):
                self._recorded[(e.agent_id, e.event_type, e.seq)] = e

    # -- sequence numbers -----------------------------------------------
    def next_seq(self, agent_id: str, event_type: str) -> int:
        # Public entry point acquires the lock itself. cross() already holds
        # self._lock when it needs a seq, so it calls _next_seq_locked directly
        # (threading.Lock is not reentrant — calling next_seq() from inside a
        # `with self._lock:` block would deadlock).
        with self._lock:
            return self._next_seq_locked(agent_id, event_type)

    def _next_seq_locked(self, agent_id: str, event_type: str) -> int:
        key = (agent_id, event_type)
        seq = self._counters.get(key, 0)
        self._counters[key] = seq + 1
        return seq

    # -- vector clock bookkeeping (must run under self._lock) ------------
    def _tick(self, agent: str) -> dict:
        vc = self._vectors.setdefault(agent, VectorClock(agent))
        for delivered in self._mailbox.pop(agent, []):
            vc.merge(delivered)
        return vc.tick()

    def _deliver(self, event_type: str, request_obj: Any, sender_vec: dict,
                 sender_agent: str) -> None:
        if event_type != "agent_msg":
            return
        to_agent = request_obj["to"]
        self._mailbox.setdefault(to_agent, []).append(dict(sender_vec))
        if self.phase == FORK and sender_agent in self._tainted:
            self._tainted.add(to_agent)

    # -- replay/fork mode decision (must run under self._lock) -----------
    def _decide_mode(self, agent: str, key: tuple) -> str:
        if self.phase == RECORD:
            return LIVE
        if self.phase == REPLAY:
            return REPLAY
        # FORK
        if key == self.branch_key:
            return MUTATE
        if agent in self._tainted:
            return LIVE
        rec = self._recorded.get(key)
        if rec is not None and happens_before(self._branch_vec, json.loads(rec.vector_clock)):
            self._tainted.add(agent)
            return LIVE
        return REPLAY

    def guard_real_call(self) -> None:
        if self.phase == REPLAY:
            raise ReplayViolation(
                "Attempted a real external call during REPLAY — unclamped boundary."
            )

    def _write(self, agent_id, event_type, seq, request_json, response_json,
               boundary_hash, vec: dict, rank: int) -> None:
        if self.write_trace_id is None:
            return
        self.store.append_event(Event(
            event_id=os.urandom(16).hex(), trace_id=self.write_trace_id, seq=seq,
            logical_clock=vec.get(agent_id, 0), wall_clock=time.time(), agent_id=agent_id,
            event_type=event_type, request_json=request_json, response_json=response_json,
            boundary_hash=boundary_hash, vector_clock=canonical(vec), causal_rank=rank,
        ))

    def cross(self, agent_id: str, event_type: str, request_obj: Any,
              live_fn: Callable[[], Any]) -> Any:
        request_json = canonical(request_obj)
        boundary_hash = sha256_hex(request_json)

        with self._lock:
            seq = self._next_seq_locked(agent_id, event_type)
            vec = self._tick(agent_id)
            key = (agent_id, event_type, seq)
            rank = vc_rank(vec)
            self._vectors_by_key[key] = vec
            mode = self._decide_mode(agent_id, key)

        if mode == LIVE:
            value = live_fn()  # network/tool call happens OUTSIDE the lock
            response_json = canonical(value)
            with self._lock:
                self._write(agent_id, event_type, seq, request_json, response_json,
                            boundary_hash, vec, rank)
                self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
                self._deliver(event_type, request_obj, vec, agent_id)
            return value

        rec = self._recorded.get(key)
        if rec is None:
            raise ReplayDrift(f"no recorded event for {key} — extra boundary crossing")
        if rec.boundary_hash != boundary_hash:
            raise ReplayDrift(
                f"request drift at {key}: {rec.boundary_hash[:12]} != {boundary_hash[:12]}"
            )
        if canonical(vec) != rec.vector_clock:
            raise ReplayDrift(f"causal drift at {key}: vector clock differs from recording")

        if mode == MUTATE:
            value = _apply_mutation(json.loads(rec.response_json), self.mutation)
            response_json = canonical(value)
            with self._lock:
                self._tainted.add(agent_id)
                self._write(agent_id, event_type, seq, request_json, response_json,
                            boundary_hash, vec, rank)
                self.produced.append((agent_id, event_type, seq, boundary_hash, response_json))
                self._deliver(event_type, request_obj, vec, agent_id)
            return value

        # mode == REPLAY
        with self._lock:
            if self.write_trace_id is not None:
                self._write(agent_id, event_type, seq, request_json, rec.response_json,
                            boundary_hash, vec, rank)
            self.produced.append((agent_id, event_type, seq, boundary_hash, rec.response_json))
            self._deliver(event_type, request_obj, vec, agent_id)
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


@contextmanager
def fork_from(store: Store, read_trace_id: str, *, write_trace_id: str,
             branch_key: tuple, branch_vec: dict, mutation: Any):
    global _active
    prev = _active
    _active = Interceptor(store, phase=FORK, read_trace_id=read_trace_id,
                          write_trace_id=write_trace_id, branch_key=branch_key,
                          branch_vec=branch_vec, mutation=mutation)
    try:
        yield _active
    finally:
        _active = prev
