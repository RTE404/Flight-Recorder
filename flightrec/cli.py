"""Typer CLI for Flight Recorder."""
from __future__ import annotations

import json as _json
import os
import time
import uuid

import typer

from .models import Trace
from .store import Store
from . import interceptor as itc
from .agent.reference_agent import run_agent
from .replay import replay as _replay, DeterminismError
from .fork import fork as _fork

app = typer.Typer(add_completion=False, help="Record / replay / time-travel debugger.")


def _db() -> Store:
    return Store(os.environ.get("FLIGHTREC_DB", "flightrec.db"))


def _new_trace_id() -> str:
    return "tr_" + uuid.uuid4().hex[:12]


def record_run(store: Store, task: str) -> str:
    trace_id = _new_trace_id()
    store.create_trace(Trace(trace_id=trace_id, task=task, status="recording",
                             created_at=time.time()))
    try:
        with itc.record_into(store, trace_id):
            run_agent(task)
        store.set_status(trace_id, "complete")
    except Exception:
        store.set_status(trace_id, "failed")
        raise
    return trace_id


@app.command()
def run(task: str):
    """Run the reference agent live, record everything, print the trace id."""
    store = _db()
    trace_id = record_run(store, task)
    typer.echo(trace_id)


@app.command()
def ls():
    """List traces."""
    store = _db()
    for t in store.list_traces():
        parent = t.parent_trace_id or "-"
        typer.echo(f"{t.trace_id}  parent={parent}  {t.status:9}  {t.task[:50]}")


@app.command()
def show(trace_id: str):
    """Print the event log for a trace."""
    store = _db()
    if store.get_trace(trace_id) is None:
        typer.echo(f"no such trace: {trace_id}", err=True)
        raise typer.Exit(1)
    for e in store.get_events(trace_id):
        resp = e.response_json if len(e.response_json) <= 70 else e.response_json[:67] + "..."
        typer.echo(f"seq={e.seq} lc={e.logical_clock} {e.agent_id:11} "
                   f"{e.event_type:9} {e.event_id}  {resp}")


@app.command()
def replay(trace_id: str):
    """Faithful replay + determinism assertion."""
    store = _db()
    try:
        produced = _replay(store, trace_id)
    except (DeterminismError, itc.ReplayDrift, itc.ReplayViolation) as exc:
        typer.echo(f"✗ DRIFT: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"✓ replay deterministic: {len(produced)} events reproduced, 0 real calls")


@app.command()
def fork(trace_id: str,
         at: str = typer.Option(..., "--at", help="event_id to fork at"),
         set: str = typer.Option(..., "--set", help="JSON mutation for the branch event")):
    """Fork a trace: mutate one recorded value at --at, run the suffix live."""
    store = _db()
    mutation = _json.loads(set)
    child = _fork(store, trace_id, at, mutation)
    typer.echo(child)


if __name__ == "__main__":
    app()
