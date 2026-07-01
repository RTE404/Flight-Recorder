import os
import pytest
from flightrec.store import Store
from flightrec import cli
from flightrec import interceptor as itc
from flightrec.replay import replay, recorded_tuples, DeterminismError


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "f.db"))
    tid = cli.record_run(store, "Explain caching")
    return store, tid


def test_recording_covers_all_four_boundary_types(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    types = {e.event_type for e in store.get_events(tid)}
    assert {"llm_call", "tool_call", "clock", "random"} <= types


def test_replay_is_byte_identical(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    produced = replay(store, tid)
    assert produced == recorded_tuples(store, tid)


def test_replay_makes_zero_real_calls(tmp_path, fake_llm, monkeypatch):
    store, tid = _record(tmp_path, fake_llm)

    def boom_llm(**kwargs):
        raise AssertionError("litellm called during replay")

    monkeypatch.setattr("litellm.completion", boom_llm)
    monkeypatch.setattr("flightrec.agent.tools.run_tool",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tool in replay")))
    produced = replay(store, tid)  # must not raise
    assert produced == recorded_tuples(store, tid)


def test_unclamped_clock_makes_replay_fail_loudly(tmp_path, fake_llm, monkeypatch):
    """Bypass the clock boundary in the worker -> ts in the LLM prompt drifts -> raise."""
    store = Store(os.path.join(tmp_path, "f.db"))
    import flightrec.agent.reference_agent as ra
    # Patch the worker to stamp REAL time instead of the recorded boundary.
    real_now = {"v": 1000.0}

    def patched_work(agent_id, sub_question):
        import json
        from flightrec import boundaries as b
        req_id = b.new_uuid(agent_id=agent_id)
        real_now["v"] += 1.0
        ts = real_now["v"]  # UNCLAMPED on purpose
        seed = b.rand(agent_id=agent_id)
        results = b.tool_call("search", {"query": sub_question, "seed": seed}, agent_id=agent_id)
        prompt = (f"request_id={req_id} ts={ts}\nUsing these search results, answer.\n"
                  f"Question: {sub_question}\nResults: {json.dumps(results['results'])}")
        resp = b.llm([{"role": "user", "content": prompt}], agent_id=agent_id)
        return resp["content"]

    monkeypatch.setattr(ra, "_work", patched_work)
    tid = cli.record_run(store, "drift demo")
    with pytest.raises((itc.ReplayDrift, DeterminismError)):
        replay(store, tid)
