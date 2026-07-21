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


def test_message_edges_target_correct_events(tmp_path, fake_llm):
    """Verify that agent_msg edges land on the semantically correct recipient event.

    This strengthens the test coverage: the previous test only checked that send events
    appear in some message edge "from" field, but didn't validate that the "to" side
    lands on the correct target. A buggy implementation could match sends to wrong events.

    Expected message edges in the reference agent pipeline:
    - planner->worker_a should land on worker_a's FIRST event (random event)
    - planner->worker_b should land on worker_b's FIRST event (random event)
    - worker_a->synthesizer should land on synthesizer's llm_call event
    - worker_b->synthesizer should land on synthesizer's llm_call event
    """
    store, tid = _record(tmp_path, fake_llm)
    g = build_graph(store, tid)

    # Index nodes by (agent_id, event_type, seq) for precise matching
    by_key = {(n["agent_id"], n["event_type"], n["seq"]): n for n in g["nodes"]}

    # Index nodes by event_id for edge traversal
    by_event_id = {n["event_id"]: n for n in g["nodes"]}

    # Get all message edges, indexed by from_id for easy lookup
    msg_edges = {e["from"]: e for e in g["edges"] if e["kind"] == "message"}

    # Get all agent_msg send events from the store
    events = store.get_events(tid)
    sends = {e.event_id: e for e in events if e.event_type == "agent_msg"}

    # Find the first event of each worker (should be random event)
    worker_a_first = by_key[("worker_a", "random", 0)]
    worker_b_first = by_key[("worker_b", "random", 0)]
    synthesizer_event = by_key[("synthesizer", "llm_call", 0)]

    # Validate each message edge
    for send_eid, send_event in sends.items():
        assert send_eid in msg_edges, f"agent_msg {send_eid} not in any message edge"
        edge = msg_edges[send_eid]
        target = by_event_id[edge["to"]]

        if send_event.agent_id == "planner":
            # Planner sends should target the recipient's first event
            if send_event.seq == 0:
                # planner's first agent_msg goes to worker_a
                assert target["agent_id"] == "worker_a"
                assert target["event_id"] == worker_a_first["event_id"], \
                    f"planner->worker_a should target worker_a's first event ({worker_a_first['event_id'][:8]}), " \
                    f"but got {target['event_id'][:8]}"
            elif send_event.seq == 1:
                # planner's second agent_msg goes to worker_b
                assert target["agent_id"] == "worker_b"
                assert target["event_id"] == worker_b_first["event_id"], \
                    f"planner->worker_b should target worker_b's first event ({worker_b_first['event_id'][:8]}), " \
                    f"but got {target['event_id'][:8]}"
        elif send_event.agent_id in ("worker_a", "worker_b"):
            # Worker sends should target the synthesizer's llm_call event
            assert target["agent_id"] == "synthesizer"
            assert target["event_type"] == "llm_call"
            assert target["event_id"] == synthesizer_event["event_id"], \
                f"{send_event.agent_id}->synthesizer should target synthesizer's llm_call " \
                f"({synthesizer_event['event_id'][:8]}), but got {target['event_id'][:8]}"


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
