# Flight Recorder Six-Issue Hardening Design

**Date:** 2026-08-18  
**Status:** Approved for implementation planning  
**Target branch:** `master` only

## 1. Purpose

Harden the current V3 repository without changing Flight Recorder's core product model:
record nondeterministic boundaries, replay them offline, fork at one event, and compare the
resulting causal timelines. The work addresses six verified problems in the current checkout:

1. The local Python environment is inconsistent and not reproducible.
2. The default database uses the V1 schema and is rejected by V2/V3.
3. The active interceptor is process-global and non-reentrant.
4. Boundary discipline is documented but not validated.
5. The V3 viewer is unsafe outside a trusted localhost environment.
6. Repository state and automated verification are incomplete.

The implementation stays appropriate for a local developer tool. It does not turn Flight
Recorder into a multi-tenant service or a general Python sandbox.

## 2. Success criteria

The hardening is complete when all of the following are true:

- A clean Python 3.11 environment can be created from committed dependency metadata and
  produces a passing `pip check` and full test run without an in-memory LiteLLM substitute.
- Opening a V1 database creates a recoverable backup and migrates all existing traces in
  place; reopening it is idempotent and all migrated traces can be shown and replayed.
- Two top-level recordings can execute concurrently in one process without sharing active
  interceptor state, while each run's worker threads still share that run's interceptor.
- Owned agent orchestration is rejected before execution if it imports or directly calls a
  known nondeterministic API outside `flightrec.boundaries`.
- The V3 server rejects non-loopback clients and unsafe websocket origins, renders all trace
  data as text rather than HTML, and loads Cytoscape from a committed local asset.
- `master` is the only branch/worktree used for development, local runtime artifacts remain
  ignored, CI runs the same locked installation and test commands, and the final tree is clean.

## 3. Dependency and environment design

### 3.1 Supported development runtime

Python 3.11 is the reproducible development and CI baseline. The package may retain a wider
runtime declaration only where the locked dependency set is verified to support it; otherwise
`requires-python` will match the tested range.

### 3.2 Dependency policy

- Bound direct dependencies to compatible major/minor ranges instead of open-ended minimums.
- Explicitly declare compatibility-critical transitive interfaces such as OpenAI and HTTPX
  when LiteLLM imports them directly and their versions determine whether Flight Recorder can
  start.
- Generate a committed development lock from a clean Python 3.11 environment.
- Provide a Windows bootstrap script that creates or refreshes `.venv`, installs the lock,
  installs Flight Recorder editable without re-resolving dependencies, runs `pip check`, and
  leaves the exact verification commands documented in the README.
- CI uses Windows and Python 3.11 so the committed lock is tested in the environment it targets.

The lock is the reproducibility authority. `pyproject.toml` remains the package metadata and
human-readable compatibility policy.

## 4. V1 database migration design

### 4.1 Safety rules

The existing instruction to delete a V1 database is removed. When `Store` detects the exact
known V1 shape (only `vector_clock` and `causal_rank` are absent), it will:

1. Create a SQLite-consistent backup next to the source database using SQLite's backup API.
2. Start a transaction and re-check the schema after acquiring the write lock.
3. Add the V2 columns.
4. Reconstruct vector clocks and causal ranks for every event.
5. Commit the migration and mark the schema version.

Any unexpected schema shape fails with a clear error and leaves the source untouched. In-memory
and temporary test databases do not create filesystem backups.

### 4.2 Causality reconstruction

Events are processed per trace in original `rowid_pk` order, which is the only authoritative V1
execution order. Reconstruction applies the same rules as the live V2 interceptor:

- Each agent owns one vector-clock component.
- Before an event ticks, it merges every vector waiting in its mailbox.
- Every event increments the current agent's component.
- An `agent_msg` event delivers its resulting vector to the request's `to` agent.
- `causal_rank` is the sum of the reconstructed vector.
- `logical_clock` is normalized to the agent's own component for V2 display semantics.

This preserves real message causality while keeping independent worker events concurrent even
when V1 happened to execute those workers sequentially. Failed traces with no events migrate
without special handling.

### 4.3 Migration tests

Tests create an actual V1-schema fixture, migrate it, verify backup creation, validate expected
happens-before/concurrent relationships, reopen it, and confirm that the second open performs no
additional migration. A forced migration failure proves the source remains readable.

## 5. Reentrant interceptor design

Replace the module-level `_active` value with a `ContextVar` containing the active `Interceptor`.
Context managers use token-based set/reset operations, making nesting deterministic and isolating
top-level executions in separate threads or async tasks.

Python threads do not automatically inherit a caller's context. `run_agent` therefore creates a
fresh `copy_context()` for each worker thread and runs that worker entry point inside its copied
context. Both workers receive the same interceptor object for their run, but concurrent top-level
runs receive different objects.

After this change, the V3 server does not need a process-wide run lock solely to protect active
interceptor state. Store-level SQLite serialization remains unchanged.

Tests cover nested contexts, two concurrent top-level recordings, correct worker propagation,
and exception cleanup that restores the previous context.

## 6. Boundary-discipline validation

Flight Recorder cannot safely claim to sandbox arbitrary Python code. Instead, it will enforce a
clear and testable contract for agent orchestration owned by this repository.

A new source validator parses an agent module's AST before execution and rejects:

- imports of direct nondeterministic providers such as `time`, `random`, `uuid`, `litellm`,
  `requests`, `httpx`, `socket`, or `subprocess`; and
- direct calls to known nondeterministic APIs when reached through aliases.

`flightrec.boundaries` and tool implementations invoked behind `tool_call` are trusted boundary
providers and are outside this scan. The reference agent is validated from every run, replay, and
fork entry path. A dedicated exception reports the file, line, and prohibited symbol.

Tests include direct imports, aliased imports, allowed deterministic modules, and the existing
reference agent. Documentation will state the remaining limit honestly: dynamic code and external
agent packages require their own validation or stronger process isolation.

## 7. V3 local-security design

### 7.1 Network boundary

- `flightrec serve` accepts loopback hosts only.
- HTTP requests from non-loopback clients are rejected.
- Websocket connections require both a loopback client and an allowed loopback origin.
- The FastAPI app is created through a factory so tests can use an explicit test mode without
  weakening production defaults.

The viewer remains authentication-free because the application itself enforces the local-only
boundary. Remote or multi-user hosting remains explicitly unsupported.

### 7.2 Browser rendering

- Replace trace-derived `innerHTML` construction with DOM nodes and `textContent`.
- Add regression tests or static assertions proving trace request/response content is not passed
  to an HTML sink.
- Vendor the pinned Cytoscape browser asset and its license under `flightrec/static/vendor/`.
- Load the local asset from `index.html`; the viewer must function without internet access.

### 7.3 API behavior

Unknown traces produce consistent 404 responses for graph, diff, fork, and websocket paths.
Background-run failures retain the current failed trace status and log the exception locally
instead of silently swallowing it.

## 8. Repository and CI design

- Continue development directly on `master`; do not create another feature branch or worktree.
- Keep `.tokensave/`, `.claude/worktrees/`, SQLite sidecars, virtual environments, generated
  caches, and local databases ignored.
- Add a GitHub Actions workflow that installs the committed lock on Windows/Python 3.11, runs
  `pip check`, and executes the full offline suite.
- Preserve the project reference and V2/V3 implementation documents already committed to master.
- Do not add a software license without an explicit owner decision.

## 9. Implementation sequence

The issues are implemented and verified one at a time:

1. Reproducible dependency environment and CI installation path.
2. Non-destructive V1 database migration.
3. Context-local, thread-propagated interceptor state.
4. Agent boundary-discipline validation.
5. V3 local-only and browser security hardening.
6. Final repository hygiene and CI verification.

Each issue begins with failing tests, receives the smallest implementation that satisfies the
design, and passes its focused tests before the next issue begins. The full suite runs after every
issue and once more before the final push.

## 10. Explicit non-goals

- A syscall-level deterministic sandbox.
- Authentication, accounts, or remote hosting for the V3 viewer.
- A multi-process or distributed interceptor coordinator.
- Generalizing the reference pipeline beyond two workers.
- Changing replay, causal-fork, or diff semantics except where migration compatibility requires it.
- Deleting the existing `flightrec.db` or its historical traces.

