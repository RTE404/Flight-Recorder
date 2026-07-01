from flightrec.models import canonical, sha256_hex, Event, Trace


def test_canonical_is_sorted_and_compact():
    assert canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    # key order in input does not change output
    assert canonical({"a": 2, "b": 1}) == canonical({"b": 1, "a": 2})


def test_canonical_handles_nested_and_unicode():
    assert canonical({"x": ["é", 1, True, None]}) == '{"x":["é",1,true,null]}'


def test_sha256_hex_stable():
    h = sha256_hex(canonical({"a": 1}))
    assert h == sha256_hex(canonical({"a": 1}))
    assert len(h) == 64


def test_event_roundtrips_through_dict():
    e = Event(
        event_id="e1", trace_id="t1", seq=0, logical_clock=1, wall_clock=123.0,
        agent_id="planner", event_type="llm_call",
        request_json='{"a":1}', response_json='{"b":2}', boundary_hash="abc",
    )
    assert Event(**e.model_dump()) == e


def test_trace_optional_fields_default_none():
    t = Trace(trace_id="t1", task="hi", status="recording", created_at=1.0)
    assert t.parent_trace_id is None
    assert t.branch_point_event is None
    assert t.mutation is None
