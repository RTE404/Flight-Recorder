"""FastAPI server for the V3 web DAG viewer. Pure read-side over the trusted V2 store,
plus thin run/fork actions serialized behind one lock. Local dev tool - no auth."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import interceptor as itc
from ..agent.reference_agent import run_agent
from ..fork import fork as fork_agent
from ..models import Trace
from ..store import Store
from . import graph as graph_mod
from .schemas import (DiffResponse, ForkRequest, ForkResponse, GraphResponse,
                      RunRequest, RunResponse, TraceSummary)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

app = FastAPI(title="Flight Recorder viewer")
store = Store(os.environ.get("FLIGHTREC_DB", "flightrec.db"))

# The interceptor's process-global `_active` context is not reentrant: two
# concurrent record/fork executions in this process would clobber each other.
# Every operation that drives the agent (run, fork) is serialized behind this lock.
_run_lock = threading.Lock()


@app.get("/api/traces", response_model=list[TraceSummary])
def list_traces():
    return list(store.list_traces())


@app.get("/api/traces/{trace_id}", response_model=GraphResponse)
def get_trace_graph(trace_id: str):
    try:
        return graph_mod.build_graph(store, trace_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"no such trace: {trace_id}")


@app.post("/api/traces/{trace_id}/fork", response_model=ForkResponse)
def fork_trace(trace_id: str, body: ForkRequest):
    """Reruns the branch's causal future LIVE - needs a real model/API key unless a
    fake LLM is installed (tests install one)."""
    with _run_lock:
        try:
            child_id = fork_agent(store, trace_id, body.at_event_id, body.mutation)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return ForkResponse(child_trace_id=child_id)


@app.get("/api/diff/{trace_a}/{trace_b}", response_model=DiffResponse)
def get_diff(trace_a: str, trace_b: str):
    return graph_mod.diff_overlay(store, trace_a, trace_b)


def _run_in_background(trace_id: str, task: str) -> None:
    with _run_lock:
        try:
            with itc.record_into(store, trace_id):
                run_agent(task)
            store.set_status(trace_id, "complete")
        except Exception:
            store.set_status(trace_id, "failed")


@app.post("/api/run", response_model=RunResponse)
def start_run(body: RunRequest):
    """Launches the agent live in a background thread and returns the trace_id
    immediately so the client can open the live websocket. Needs a real model/API
    key unless a fake LLM is installed (tests install one).

    The trace row is created synchronously (mirroring flightrec.cli.record_run's id
    format) so the id is available before the agent execution - which is what runs
    in the background thread - completes.
    """
    trace_id = "tr_" + uuid.uuid4().hex[:12]
    store.create_trace(Trace(trace_id=trace_id, task=body.task, status="recording",
                             created_at=time.time()))
    threading.Thread(target=_run_in_background, args=(trace_id, body.task),
                     daemon=True).start()
    return RunResponse(trace_id=trace_id)


@app.websocket("/ws/traces/{trace_id}")
async def ws_trace(websocket: WebSocket, trace_id: str):
    await websocket.accept()
    last_sig = None
    try:
        while True:
            try:
                graph = await run_in_threadpool(graph_mod.build_graph, store, trace_id)
            except ValueError:
                await websocket.close(code=4404)
                return
            sig = (len(graph["nodes"]), len(graph["edges"]), graph["trace"]["status"])
            if sig != last_sig:
                await websocket.send_json(graph)
                last_sig = sig
            if graph["trace"]["status"] in ("complete", "failed"):
                await asyncio.sleep(1.0)     # one more push margin, then keep idle-polling
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
