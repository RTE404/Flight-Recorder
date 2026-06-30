import json
import os
import pytest
from flightrec.models import Trace
from flightrec.store import Store
from flightrec import interceptor as itc
from flightrec.agent.tools import search, run_tool
from flightrec.agent.reference_agent import run_agent


def test_search_is_pure_given_seed():
    a = search("dogs", 0.5)
    b = search("dogs", 0.5)
    c = search("dogs", 0.6)
    assert a == b
    assert a != c
    assert a["query"] == "dogs" and len(a["results"]) == 3


def test_run_tool_dispatch():
    out = run_tool("search", {"query": "x", "seed": 0.1})
    assert out["query"] == "x"
    with pytest.raises(ValueError):
        run_tool("nope", {})


def _fake_llm(monkeypatch):
    class _Msg:
        def __init__(self, c): self.content = c

    class _Choice:
        def __init__(self, c): self.message = _Msg(c)

    class _Resp:
        def __init__(self, c): self.choices = [_Choice(c)]

    def completion(model, messages, **kwargs):
        # Planner asked for JSON sub_questions; everyone else gets a canned answer.
        last = messages[-1]["content"]
        if "sub_questions" in last:
            return _Resp(json.dumps({"sub_questions": ["q-a", "q-b"]}))
        return _Resp("answer:" + str(len(last)))

    monkeypatch.setattr("litellm.completion", completion)


def test_agent_records_all_four_boundary_types(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    store = Store(os.path.join(tmp_path, "t.db"))
    store.create_trace(Trace(trace_id="t1", task="Q", status="recording", created_at=1.0))
    with itc.record_into(store, "t1"):
        out = run_agent("Q")
    types = {e.event_type for e in store.get_events("t1")}
    assert {"llm_call", "tool_call", "clock", "random", "agent_msg"} <= types
    assert out["final"]
    assert set(out["answers"]) == {"worker_a", "worker_b"}
