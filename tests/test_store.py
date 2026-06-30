import os
from flightrec.models import Event, Trace
from flightrec.store import Store


def _store(tmp_path):
    return Store(os.path.join(tmp_path, "t.db"))


def _event(trace_id, seq, lc, etype="llm_call", agent="planner", eid=None):
    return Event(
        event_id=eid or f"{trace_id}-{etype}-{seq}", trace_id=trace_id, seq=seq,
        logical_clock=lc, wall_clock=0.0, agent_id=agent, event_type=etype,
        request_json='{"r":1}', response_json='{"v":2}', boundary_hash="h",
    )


def test_trace_roundtrip(tmp_path):
    s = _store(tmp_path)
    t = Trace(trace_id="t1", task="q", status="recording", created_at=1.0)
    s.create_trace(t)
    assert s.get_trace("t1") == t
    assert s.get_trace("missing") is None


def test_set_status(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    s.set_status("t1", "complete")
    assert s.get_trace("t1").status == "complete"


def test_events_returned_in_insertion_order(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    s.append_event(_event("t1", 0, 1, eid="b"))
    s.append_event(_event("t1", 1, 2, eid="a"))
    got = [e.event_id for e in s.get_events("t1")]
    assert got == ["b", "a"]  # insertion order, NOT event_id order


def test_list_traces_orders_by_created_at(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t2", task="q", status="complete", created_at=2.0))
    s.create_trace(Trace(trace_id="t1", task="q", status="complete", created_at=1.0))
    assert [t.trace_id for t in s.list_traces()] == ["t1", "t2"]


def test_get_event(tmp_path):
    s = _store(tmp_path)
    s.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    s.append_event(_event("t1", 0, 1, eid="x"))
    assert s.get_event("x").event_id == "x"
    assert s.get_event("nope") is None
