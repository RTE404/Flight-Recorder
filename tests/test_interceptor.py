import os
import pytest
from flightrec.models import Trace
from flightrec.store import Store
from flightrec import interceptor as itc


def _store(tmp_path):
    return Store(os.path.join(tmp_path, "t.db"))


def _new_trace(store, trace_id):
    store.create_trace(Trace(trace_id=trace_id, task="q", status="recording", created_at=1.0))


def test_current_raises_when_inactive():
    with pytest.raises(itc.NoActiveInterceptor):
        itc.current()


def test_record_then_replay_returns_recorded_without_live(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        v = itc.current().cross("planner", "clock", {"op": "now"}, lambda: 111.0)
    assert v == 111.0

    # Replay: live_fn raises -> must NOT be called.
    def boom():
        raise AssertionError("live_fn called during replay")

    with itc.replay_from(store, "t1"):
        v2 = itc.current().cross("planner", "clock", {"op": "now"}, boom)
        produced = list(itc.current().produced)
    assert v2 == 111.0
    assert produced[0][:3] == ("planner", "clock", 0)


def test_replay_drift_on_request_mismatch(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        itc.current().cross("planner", "llm_call", {"prompt": "A"}, lambda: {"content": "x"})
    with itc.replay_from(store, "t1"):
        with pytest.raises(itc.ReplayDrift):
            itc.current().cross("planner", "llm_call", {"prompt": "B"}, lambda: {"content": "x"})


def test_guard_real_call_blocks_in_replay(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        itc.current().cross("planner", "clock", {"op": "now"}, lambda: 1.0)
    with itc.replay_from(store, "t1"):
        with pytest.raises(itc.ReplayViolation):
            itc.current().guard_real_call()


def test_seq_increments_per_agent_and_type(tmp_path):
    store = _store(tmp_path)
    _new_trace(store, "t1")
    with itc.record_into(store, "t1"):
        itc.current().cross("worker_a", "random", {"op": "uuid"}, lambda: "u0")
        itc.current().cross("worker_a", "random", {"op": "rand"}, lambda: 0.5)
        itc.current().cross("worker_a", "clock", {"op": "now"}, lambda: 9.0)
    evs = store.get_events("t1")
    by = [(e.agent_id, e.event_type, e.seq) for e in evs]
    assert by == [("worker_a", "random", 0), ("worker_a", "random", 1), ("worker_a", "clock", 0)]
