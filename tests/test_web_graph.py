import os
import pytest
from flightrec.store import Store
from flightrec import cli
from flightrec.fork import fork
from flightrec.web.graph import build_graph, diff_overlay


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "g.db"))
    tid = cli.record_run(store, "compare X and Y")
    return store, tid


def _first_tool_event(store, trace_id):
    for e in store.get_events(trace_id):
        if e.event_type == "tool_call":
            return e
    raise AssertionError("no tool_call event")


def test_build_graph_structure(tmp_path, fake_llm):
    store, tid = _record(tmp_path, fake_llm)
    g = build_graph(store, tid)

    assert g["trace"]["trace_id"] == tid
    assert g["trace"]["agents"] == ["planner", "worker_a", "worker_b", "synthesizer"]

    events = store.get_events(tid)
    assert len(g["nodes"]) == len(events)
    for n in g["nodes"]:
        assert n["role"] == "recorded"
        assert isinstance(n["lane"], int) and n["lane"] >= 0
        assert isinstance(n["column"], int) and n["column"] >= 0

    kinds = {e["kind"] for e in g["edges"]}
    assert kinds == {"sequence", "message"}

    msg_froms = {e["from"] for e in g["edges"] if e["kind"] == "message"}
    planner_sends = [e for e in events if e.agent_id == "planner" and e.event_type == "agent_msg"]
    worker_sends = [e for e in events if e.agent_id in ("worker_a", "worker_b") and e.event_type == "agent_msg"]
    assert len(planner_sends) == 2
    assert len(worker_sends) == 2
    for send in planner_sends + worker_sends:
        assert send.event_id in msg_froms


def test_build_graph_unknown_trace_raises(tmp_path):
    store = Store(os.path.join(tmp_path, "empty.db"))
    with pytest.raises(ValueError):
        build_graph(store, "tr_nope")


def test_build_graph_fork_roles(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)  # worker_a's tool_call (always ranks first)
    child = fork(store, parent, branch.event_id, {"query": "q", "results": ["MUTATED"]})

    g = build_graph(store, child)
    by_key = {(n["agent_id"], n["event_type"], n["seq"]): n for n in g["nodes"]}

    mutated = by_key[(branch.agent_id, branch.event_type, branch.seq)]
    assert mutated["role"] == "mutated"

    other = "worker_b" if branch.agent_id == "worker_a" else "worker_a"
    other_nodes = [n for n in g["nodes"] if n["agent_id"] == other]
    assert other_nodes
    assert all(n["role"] == "reused" for n in other_nodes)

    mutated_agent_llm = next(n for n in g["nodes"]
                             if n["agent_id"] == branch.agent_id and n["event_type"] == "llm_call")
    assert mutated_agent_llm["role"] == "live"
    synth_node = next(n for n in g["nodes"] if n["agent_id"] == "synthesizer")
    assert synth_node["role"] == "live"


def test_diff_overlay(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    child = fork(store, parent, branch.event_id, {"query": "q", "results": ["MUT"]})

    overlay = diff_overlay(store, parent, child)
    assert overlay["branch_event"] == [branch.agent_id, branch.event_type, branch.seq]
    assert overlay["changed_keys"]
    other = "worker_b" if branch.agent_id == "worker_a" else "worker_a"
    assert all(k[0] != other for k in overlay["changed_keys"])
