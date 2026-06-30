import os
import pytest
from flightrec.models import Trace
from flightrec.store import Store
from flightrec import interceptor as itc
from flightrec import boundaries as b


def _ctx(tmp_path):
    store = Store(os.path.join(tmp_path, "t.db"))
    store.create_trace(Trace(trace_id="t1", task="q", status="recording", created_at=1.0))
    return store


def test_record_now_uuid_rand_then_replay(tmp_path, monkeypatch):
    store = _ctx(tmp_path)
    monkeypatch.setattr("time.time", lambda: 42.0)
    monkeypatch.setattr("uuid.uuid4", lambda: __import__("uuid").UUID(int=7))
    monkeypatch.setattr("random.random", lambda: 0.25)
    with itc.record_into(store, "t1"):
        n = b.now(agent_id="worker_a")
        u = b.new_uuid(agent_id="worker_a")
        r = b.rand(agent_id="worker_a")
    assert (n, r) == (42.0, 0.25)
    assert u == str(__import__("uuid").UUID(int=7))

    # Replay: break the real primitives; they must never be called.
    monkeypatch.setattr("time.time", lambda: 0.0)
    monkeypatch.setattr("random.random", lambda: 0.0)
    with itc.replay_from(store, "t1"):
        assert b.now(agent_id="worker_a") == 42.0
        assert b.new_uuid(agent_id="worker_a") == str(__import__("uuid").UUID(int=7))
        assert b.rand(agent_id="worker_a") == 0.25


def test_llm_records_and_replays_without_network(tmp_path, monkeypatch):
    store = _ctx(tmp_path)

    class _Msg:
        content = "hello"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr("litellm.completion", fake_completion)
    with itc.record_into(store, "t1"):
        out = b.llm([{"role": "user", "content": "hi"}], agent_id="planner")
    assert out == {"role": "assistant", "content": "hello"}
    assert calls["n"] == 1

    # Replay must not call litellm again.
    def boom(**kwargs):
        raise AssertionError("litellm called during replay")

    monkeypatch.setattr("litellm.completion", boom)
    with itc.replay_from(store, "t1"):
        out2 = b.llm([{"role": "user", "content": "hi"}], agent_id="planner")
    assert out2 == {"role": "assistant", "content": "hello"}
    assert calls["n"] == 1


def test_tool_call_records_and_replays(tmp_path, monkeypatch):
    store = _ctx(tmp_path)
    monkeypatch.setattr("flightrec.agent.tools.run_tool",
                        lambda name, args: {"echo": args})
    with itc.record_into(store, "t1"):
        out = b.tool_call("search", {"query": "x"}, agent_id="worker_a")
    assert out == {"echo": {"query": "x"}}

    monkeypatch.setattr("flightrec.agent.tools.run_tool",
                        lambda name, args: (_ for _ in ()).throw(AssertionError("ran in replay")))
    with itc.replay_from(store, "t1"):
        assert b.tool_call("search", {"query": "x"}, agent_id="worker_a") == {"echo": {"query": "x"}}
