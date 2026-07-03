import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(tmp_path, monkeypatch, fake_llm):
    db_path = os.path.join(tmp_path, "api.db")
    monkeypatch.setenv("FLIGHTREC_DB", db_path)
    # server.py builds its module-level Store at import time from FLIGHTREC_DB, so
    # reload it fresh (after the env var is set) for every test to get an isolated db.
    import flightrec.web.server as server_mod
    importlib.reload(server_mod)
    from flightrec import cli
    client = TestClient(server_mod.app)
    return client, server_mod.store, cli


def test_list_and_get_trace(tmp_path, api_client):
    client, store, cli = api_client
    tid = cli.record_run(store, "compare X and Y")

    resp = client.get("/api/traces")
    assert resp.status_code == 200
    assert tid in {t["trace_id"] for t in resp.json()}

    resp = client.get(f"/api/traces/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace"]["trace_id"] == tid
    assert len(body["nodes"]) == len(store.get_events(tid))


def test_get_trace_404(api_client):
    client, store, cli = api_client
    resp = client.get("/api/traces/tr_nope")
    assert resp.status_code == 404


def test_fork_and_diff_endpoints(api_client):
    client, store, cli = api_client
    tid = cli.record_run(store, "compare X and Y")
    branch = next(e for e in store.get_events(tid) if e.event_type == "tool_call")

    resp = client.post(f"/api/traces/{tid}/fork",
                       json={"at_event_id": branch.event_id,
                             "mutation": {"query": "q", "results": ["MUT"]}})
    assert resp.status_code == 200
    child_id = resp.json()["child_trace_id"]

    resp = client.get(f"/api/traces/{child_id}")
    assert resp.status_code == 200
    roles = {n["role"] for n in resp.json()["nodes"]}
    assert {"mutated", "live", "reused"} <= roles

    resp = client.get(f"/api/diff/{tid}/{child_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["branch_event"] == [branch.agent_id, branch.event_type, branch.seq]
    assert body["changed_keys"]


def test_run_endpoint_returns_trace_id_and_completes(api_client):
    client, store, cli = api_client
    resp = client.post("/api/run", json={"task": "compare X and Y"})
    assert resp.status_code == 200
    tid = resp.json()["trace_id"]
    assert store.get_trace(tid) is not None

    # Poll briefly for the background thread to finish (fake LLM makes this fast).
    import time
    for _ in range(50):
        if store.get_trace(tid).status in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert store.get_trace(tid).status == "complete"


def test_websocket_smoke(api_client):
    client, store, cli = api_client
    tid = cli.record_run(store, "compare X and Y")
    with client.websocket_connect(f"/ws/traces/{tid}") as ws:
        frame = ws.receive_json()
        assert frame["trace"]["trace_id"] == tid
        assert len(frame["nodes"]) == len(store.get_events(tid))
