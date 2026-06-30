import json
import os
import pytest
from flightrec.store import Store
from flightrec import cli
from flightrec.fork import fork
from flightrec.replay import recorded_tuples
from flightrec.diff import diff, format_report


def _record(tmp_path, fake_llm):
    store = Store(os.path.join(tmp_path, "f.db"))
    tid = cli.record_run(store, "compare X and Y")
    return store, tid


def _first_tool_event(store, trace_id):
    for e in store.get_events(trace_id):
        if e.event_type == "tool_call":
            return e
    raise AssertionError("no tool_call event")


def test_fork_shares_prefix_and_diverges_at_branch(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    new_results = {"query": json.loads(branch.response_json)["query"],
                   "results": ["MUTATED-1", "MUTATED-2", "MUTATED-3"]}
    child = fork(store, parent, branch.event_id, new_results)

    p = store.get_events(parent)
    c = store.get_events(child)
    # Identical prefix up to (not including) the branch event.
    branch_idx = next(i for i, e in enumerate(p) if e.event_id == branch.event_id)
    for i in range(branch_idx):
        assert (p[i].agent_id, p[i].event_type, p[i].seq, p[i].boundary_hash,
                p[i].response_json) == (c[i].agent_id, c[i].event_type, c[i].seq,
                                        c[i].boundary_hash, c[i].response_json)
    # Branch event: same request, mutated response.
    assert c[branch_idx].boundary_hash == p[branch_idx].boundary_hash
    assert json.loads(c[branch_idx].response_json)["results"] == ["MUTATED-1", "MUTATED-2", "MUTATED-3"]


def test_fork_suffix_is_live_and_recorded(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    child = fork(store, parent, branch.event_id, {"results": ["Z"], "query": "q"})
    t = store.get_trace(child)
    assert t.parent_trace_id == parent
    assert t.branch_point_event == branch.event_id
    assert t.status == "complete"
    # Suffix exists: child has events after the branch index.
    c = store.get_events(child)
    branch_idx = next(i for i, e in enumerate(c)
                      if (e.agent_id, e.event_type, e.seq) ==
                      (branch.agent_id, branch.event_type, branch.seq))
    assert branch_idx < len(c) - 1  # at least one live suffix event


def test_diff_reports_branch_and_changes(tmp_path, fake_llm):
    store, parent = _record(tmp_path, fake_llm)
    branch = _first_tool_event(store, parent)
    child = fork(store, parent, branch.event_id, {"results": ["MUT"], "query": "q"})
    report = diff(store, parent, child)
    # Branch detected exactly at the mutated tool_call.
    assert report.branch_event == (branch.agent_id, branch.event_type, branch.seq)
    # Non-empty downstream changes, attributed to agents.
    assert sum(report.changed_by_agent.values()) > 0
    text = format_report(report)
    assert "branch" in text.lower()
