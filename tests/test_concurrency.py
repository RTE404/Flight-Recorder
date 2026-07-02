import json
import os
import time

import pytest

from flightrec.store import Store
from flightrec import cli
from flightrec import interceptor as itc
from flightrec.clock import happens_before, concurrent
from flightrec.replay import replay, DeterminismError
from flightrec.fork import fork


def _fake_completion(model, messages, **kwargs):
    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    last = messages[-1]["content"]
    if "sub_questions" in last:
        return _Resp(json.dumps({"sub_questions": ["qa", "qb"]}))
    return _Resp("answer-" + str(len(last)))


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr("litellm.completion", _fake_completion)
    return monkeypatch


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "c.db"))
    tid = cli.record_run(store, "compare X and Y")
    return store, tid


def test_concurrent_replay_is_byte_identical_across_repeats(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    first = replay(store, tid)
    second = replay(store, tid)
    assert first == second


def test_recorded_vectors_are_causally_consistent(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    by_key = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(tid)}

    msg_a_vec = json.loads(by_key[("worker_a", "agent_msg", 0)].vector_clock)
    msg_b_vec = json.loads(by_key[("worker_b", "agent_msg", 0)].vector_clock)
    synth_vec = json.loads(by_key[("synthesizer", "llm_call", 0)].vector_clock)
    tool_a_vec = json.loads(by_key[("worker_a", "tool_call", 0)].vector_clock)
    tool_b_vec = json.loads(by_key[("worker_b", "tool_call", 0)].vector_clock)

    assert happens_before(msg_a_vec, synth_vec)
    assert happens_before(msg_b_vec, synth_vec)
    assert concurrent(tool_a_vec, tool_b_vec)


def test_causal_fork_reuses_concurrent_recording(tmp_path, fake_llm, monkeypatch):
    store, tid = _record(tmp_path, fake_llm)
    events = store.get_events(tid)
    branch = next(e for e in events if e.event_type == "tool_call" and e.agent_id == "worker_a")

    def guarded_completion(model, messages, **kwargs):
        last = messages[-1]["content"]
        if "qb" in last:
            raise AssertionError("worker_b's llm_call must not run live during this fork")
        return _fake_completion(model, messages, **kwargs)

    def guarded_run_tool(name, args):
        if args.get("query") == "qb":
            raise AssertionError("worker_b's tool_call must not run live during this fork")
        return {"query": args["query"], "results": ["r1", "r2", "r3"]}

    monkeypatch.setattr("litellm.completion", guarded_completion)
    monkeypatch.setattr("flightrec.agent.tools.run_tool", guarded_run_tool)

    child = fork(store, tid, branch.event_id, {"query": "qa", "results": ["MUTATED"]})

    parent_by_key = {(e.agent_id, e.event_type, e.seq): e for e in events}
    child_by_key = {(e.agent_id, e.event_type, e.seq): e for e in store.get_events(child)}
    for key, pe in parent_by_key.items():
        if pe.agent_id == "worker_b":
            ce = child_by_key[key]
            assert (ce.boundary_hash, ce.response_json) == (pe.boundary_hash, pe.response_json)


def test_thread_exception_from_one_worker_propagates_through_replay(tmp_path, fake_llm, monkeypatch):
    store, tid = _record(tmp_path, fake_llm)

    import flightrec.agent.reference_agent as ra
    real_work = ra._work
    real_now = {"v": 1000.0}

    def patched_work(agent_id, sub_question):
        if agent_id != "worker_b":
            return real_work(agent_id, sub_question)
        from flightrec import boundaries as b
        req_id = b.new_uuid(agent_id=agent_id)
        real_now["v"] += 1.0
        ts = real_now["v"]  # UNCLAMPED on purpose -> drift only for worker_b
        seed = b.rand(agent_id=agent_id)
        results = b.tool_call("search", {"query": sub_question, "seed": seed}, agent_id=agent_id)
        prompt = (f"request_id={req_id} ts={ts}\nUsing these search results, answer.\n"
                  f"Question: {sub_question}\nResults: {json.dumps(results['results'])}")
        resp = b.llm([{"role": "user", "content": prompt}], agent_id=agent_id)
        return resp["content"]

    monkeypatch.setattr(ra, "_work", patched_work)
    with pytest.raises((itc.ReplayDrift, DeterminismError)):
        replay(store, tid)


def test_worker_legs_run_concurrently(tmp_path, monkeypatch):
    """Prove the two worker legs overlap on real threads (not just interleave stepwise)."""
    DELAY = 0.2

    def slow_completion(model, messages, **kwargs):
        time.sleep(DELAY)
        last = messages[-1]["content"]
        if "sub_questions" in last:
            return _fake_completion(model, messages, **kwargs)
        return _fake_completion(model, messages, **kwargs)

    def slow_run_tool(name, args):
        time.sleep(DELAY)
        return {"query": args["query"], "results": ["r1", "r2", "r3"]}

    monkeypatch.setattr("litellm.completion", slow_completion)
    monkeypatch.setattr("flightrec.agent.tools.run_tool", slow_run_tool)

    store = Store(os.path.join(tmp_path, "timing.db"))
    start = time.time()
    cli.record_run(store, "timing check")
    elapsed = time.time() - start

    # 6 blocking calls total (1 planner llm + 2x(tool+llm) per worker + 1 synth llm).
    # Fully sequential (V1) would take ~6*DELAY. Concurrent workers only add one
    # worker's tool+llm once, so the critical path is ~4*DELAY. Assert well below
    # the sequential bound to prove genuine overlap, with slack for scheduling jitter.
    assert elapsed < 5 * DELAY, f"workers do not appear to run concurrently: {elapsed:.2f}s"
