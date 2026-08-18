# Flight Recorder Six-Issue Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current V3 checkout reproducible, migration-safe, reentrant, boundary-validated, localhost-secure, and continuously verified on `master`.

**Architecture:** Keep the existing boundary/interceptor/store model. Add a universal `uv.lock`, an isolated V1-to-V2 migration module, context-local interceptor activation with explicit worker-thread propagation, an AST boundary validator, and a local-only FastAPI security layer. Every subsystem receives focused tests before implementation and a full-suite gate afterward.

**Tech Stack:** Python 3.11, uv, SQLite, ContextVars, AST, FastAPI/Starlette, vanilla JavaScript, Cytoscape.js, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-18-flightrec-six-issue-hardening-design.md`

## Global Constraints

- Work directly on `master`; do not create another branch or worktree.
- Preserve `flightrec.db` and create a recoverable backup before changing its schema.
- Retain the record/replay/fork/diff semantics and the two-worker reference topology.
- Start each behavioral task with a failing test and run the full suite after its focused tests pass.
- V3 remains a localhost-only developer tool with no remote-hosting or authentication scope.
- Do not add a software license without an explicit owner decision.
- Do not commit `.venv`, local databases, SQLite sidecars, caches, `.tokensave`, or `.claude/worktrees`.

---

## File map

**Create**

- `.python-version` — pins the development interpreter to Python 3.11.
- `uv.lock` — exact universal dependency resolution.
- `scripts/bootstrap.ps1` — reproducible Windows environment bootstrap and integrity check.
- `.github/workflows/ci.yml` — locked Windows/Python 3.11 verification.
- `flightrec/migrations.py` — V1 detection, backup, vector reconstruction, and schema upgrade.
- `tests/test_migrations.py` — real V1 database fixtures and migration safety tests.
- `flightrec/audit.py` — AST boundary-discipline validator.
- `tests/test_boundary_audit.py` — validator acceptance/rejection tests.
- `flightrec/web/security.py` — loopback and origin predicates plus HTTP middleware.
- `tests/test_web_security.py` — network, DOM-sink, local asset, and error-path tests.
- `flightrec/static/vendor/cytoscape.min.js` — pinned local viewer dependency.
- `flightrec/static/vendor/LICENSE-cytoscape.txt` — upstream license for the vendored asset.

**Modify**

- `pyproject.toml` — bounded dependencies, package data, and tested Python policy.
- `README.md` — bootstrap, migration, enforcement scope, and local-security contract.
- `.gitignore` — verify generated artifacts remain excluded.
- `flightrec/store.py` — call the migration module before enabling WAL.
- `flightrec/interceptor.py` — replace `_active` with `ContextVar` token handling.
- `flightrec/agent/reference_agent.py` — validate source and propagate context to worker threads.
- `flightrec/web/server.py` — app factory, local-only enforcement, consistent errors, logging.
- `flightrec/web/graph.py` — reject unknown traces in diff overlays.
- `flightrec/cli.py` — validate `serve` bind hosts.
- `flightrec/static/index.html` — load vendored Cytoscape.
- `flightrec/static/app.js` — build trace-derived UI with `textContent`.
- `tests/test_interceptor.py`, `tests/test_concurrency.py`, `tests/test_web_api.py` — focused regressions.

---

### Task 1: Reproducible Python environment and locked CI

**Files:**
- Create: `.python-version`
- Create: `uv.lock`
- Create: `scripts/bootstrap.ps1`
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_dependency_contract.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: `uv sync --locked --extra dev` as the canonical installation command.
- Produces: `scripts/bootstrap.ps1` as the Windows one-command bootstrap.
- Produces: a normal `uv run pytest -q` path that imports real LiteLLM successfully.

- [ ] **Step 1: Add failing dependency-contract tests**

Create `tests/test_dependency_contract.py` with assertions that:

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_python_and_dependency_contract_is_bounded():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11"
    dependencies = "\n".join(project["dependencies"])
    for package in ("litellm", "openai", "httpx", "pydantic", "typer"):
        line = next(x for x in project["dependencies"] if x.startswith(package))
        assert "<" in line
    assert project["requires-python"] == ">=3.11,<3.14"


def test_locked_bootstrap_and_ci_use_the_same_install_path():
    assert (ROOT / "uv.lock").is_file()
    bootstrap = (ROOT / "scripts" / "bootstrap.ps1").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for text in (bootstrap, workflow):
        assert "uv sync --locked --extra dev" in text
        assert "uv run pytest -q" in text
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m pytest tests/test_dependency_contract.py -v`  
Expected: FAIL because `.python-version`, `uv.lock`, bootstrap, CI, and bounded declarations are absent.

- [ ] **Step 3: Bound package metadata and generate the lock**

Set:

```toml
requires-python = ">=3.11,<3.14"
dependencies = [
    "litellm>=1.89,<1.90",
    "openai>=2.20,<3",
    "httpx>=0.28,<1",
    "pydantic>=2.10,<3",
    "typer>=0.12,<1",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9", "flightrec[web]"]
web = ["fastapi>=0.110,<1", "uvicorn[standard]>=0.29,<1"]
```

Write `.python-version` as `3.11`, then run:

```powershell
uv lock --python 3.11
uv sync --locked --extra dev
uv pip check
uv run pytest -q
```

- [ ] **Step 4: Add bootstrap and CI**

`scripts/bootstrap.ps1` must stop on errors and run:

```powershell
$ErrorActionPreference = "Stop"
uv sync --locked --extra dev
uv pip check
uv run pytest -q
```

`.github/workflows/ci.yml` uses `windows-latest`, `actions/setup-python` with `3.11`, installs
the exact uv version used to generate the lock, then runs the same three commands.

- [ ] **Step 5: Verify Issue 1**

Run:

```powershell
uv lock --check
uv sync --locked --extra dev
uv pip check
uv run pytest -q
```

Expected: lock unchanged, dependency check passes, full suite passes without module substitution.

- [ ] **Step 6: Commit Issue 1**

```powershell
git add .python-version uv.lock scripts/bootstrap.ps1 .github/workflows/ci.yml tests/test_dependency_contract.py pyproject.toml README.md
git commit -m "build: lock reproducible development environment"
```

---

### Task 2: Non-destructive V1 database migration

**Files:**
- Create: `flightrec/migrations.py`
- Create: `tests/test_migrations.py`
- Modify: `flightrec/store.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `ensure_current_schema(conn: sqlite3.Connection, path: str) -> Optional[Path]`.
- Produces: `Store.last_backup_path: Optional[Path]`.
- Consumes: `VectorClock`, `vc_rank`, and canonical JSON helpers.

- [ ] **Step 1: Write a real V1 fixture and failing migration tests**

The fixture creates `traces` and the eleven-column V1 `events` table directly with sqlite3,
then inserts a planner send, independent worker events, both worker sends, and a synthesizer event.
Tests assert:

```python
store = Store(str(db_path))
assert store.last_backup_path is not None
assert store.last_backup_path.exists()
events = store.get_events("t1")
assert all(json.loads(e.vector_clock) for e in events)
assert happens_before(planner_send_vec, worker_a_vec)
assert concurrent(worker_a_vec, worker_b_vec)
assert happens_before(worker_a_send_vec, synth_vec)
```

A second open asserts `last_backup_path is None`. A monkeypatched reconstruction failure asserts
the original V1 columns remain and the backup is readable.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_migrations.py -v`  
Expected: FAIL with the current incompatible-V1 `RuntimeError`.

- [ ] **Step 3: Implement migration primitives**

In `flightrec/migrations.py`, define:

```python
SCHEMA_VERSION = 2
V2_COLUMNS = {"vector_clock", "causal_rank"}


def ensure_current_schema(conn: sqlite3.Connection, path: str) -> Optional[Path]:
    """Migrate the exact V1 event schema transactionally and return its backup path."""
```

Use `conn.backup()` for file databases. Under `BEGIN IMMEDIATE`, re-read columns, add both columns,
process each trace by `rowid_pk`, merge mailboxes before `tick()`, deliver `agent_msg` vectors after
the tick, update `logical_clock`, `vector_clock`, and `causal_rank`, set `PRAGMA user_version=2`,
and commit. Roll back on every exception.

- [ ] **Step 4: Integrate Store and remove destructive guidance**

`Store.init_schema()` creates new tables, calls `ensure_current_schema`, stores the returned path,
then enables WAL. README explains automatic backup/migration and removes “delete the DB.”

- [ ] **Step 5: Verify Issue 2 and migrate the real database**

Run:

```powershell
uv run pytest tests/test_migrations.py tests/test_store.py tests/test_record_replay.py -v
uv run pytest -q
$env:FLIGHTREC_DB="flightrec.db"
uv run flightrec ls
```

Expected: tests pass; the real database lists its existing traces; a timestamped V1 backup exists;
`PRAGMA table_info(events)` contains both V2 columns.

- [ ] **Step 6: Commit Issue 2**

```powershell
git add flightrec/migrations.py flightrec/store.py tests/test_migrations.py README.md
git commit -m "feat: migrate V1 databases without data loss"
```

---

### Task 3: Context-local, thread-propagated interceptor

**Files:**
- Modify: `flightrec/interceptor.py`
- Modify: `flightrec/agent/reference_agent.py`
- Modify: `flightrec/web/server.py`
- Modify: `tests/test_interceptor.py`
- Modify: `tests/test_concurrency.py`

**Interfaces:**
- Produces: `_active: ContextVar[Optional[Interceptor]]`.
- Preserves: `current()`, `record_into`, `replay_from`, and `fork_from` public signatures.
- Produces: worker threads entered through one fresh `copy_context()` per worker.

- [ ] **Step 1: Add failing isolation tests**

Add a nested-context test that verifies the outer interceptor is restored. Add a concurrent-run
test using two top-level threads and two trace IDs; synchronize their entry with a barrier and
assert every trace contains only its own task's boundary hashes/responses.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_interceptor.py tests/test_concurrency.py -v`  
Expected: the concurrent top-level run test fails because one global `_active` is overwritten.

- [ ] **Step 3: Implement ContextVar activation**

Replace assignment/restoration with:

```python
_active: ContextVar[Optional[Interceptor]] = ContextVar("flightrec_active", default=None)


def current() -> Interceptor:
    active = _active.get()
    if active is None:
        raise NoActiveInterceptor(...)
    return active


token = _active.set(interceptor)
try:
    yield interceptor
finally:
    _active.reset(token)
```

- [ ] **Step 4: Propagate context into worker threads**

Create one `copy_context()` per worker before starting it and use
`threading.Thread(target=context.run, args=(worker_entry, wid))`. Remove the V3 run lock whose
only purpose was protecting `_active`; Store locking continues to serialize SQLite methods.

- [ ] **Step 5: Verify and commit Issue 3**

Run:

```powershell
uv run pytest tests/test_interceptor.py tests/test_concurrency.py tests/test_web_api.py -v
uv run pytest -q
git add flightrec/interceptor.py flightrec/agent/reference_agent.py flightrec/web/server.py tests/test_interceptor.py tests/test_concurrency.py
git commit -m "fix: isolate interceptor state per execution context"
```

---

### Task 4: Enforced boundary-discipline contract

**Files:**
- Create: `flightrec/audit.py`
- Create: `tests/test_boundary_audit.py`
- Modify: `flightrec/agent/reference_agent.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `BoundaryDisciplineError`.
- Produces: `assert_boundary_discipline(path: str | Path) -> None`.
- Consumes: reference-agent `__file__` at the start of `run_agent`.

- [ ] **Step 1: Add failing AST-validator tests**

Create temporary modules and assert:

```python
assert_boundary_discipline(allowed_module)
with pytest.raises(BoundaryDisciplineError, match=r"time.*line 1"):
    assert_boundary_discipline(module_with_import_time)
with pytest.raises(BoundaryDisciplineError, match="httpx"):
    assert_boundary_discipline(module_with_aliased_httpx)
assert_boundary_discipline(Path(reference_agent.__file__))
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_boundary_audit.py -v`  
Expected: collection fails because `flightrec.audit` does not exist.

- [ ] **Step 3: Implement the validator**

Parse source with `ast.parse`. Reject `Import` and `ImportFrom` roots in:

```python
FORBIDDEN_MODULES = {
    "time", "random", "uuid", "secrets", "litellm",
    "requests", "httpx", "socket", "subprocess",
}
```

Report `path`, `node.lineno`, and the original imported symbol. Cache successful scans by resolved
path, `st_mtime_ns`, and file size so edited source is revalidated.

- [ ] **Step 4: Enforce at the agent entry point and document limits**

Call `assert_boundary_discipline(__file__)` before `_plan(task)`. README states that validation
covers owned orchestration source, tools run behind `tool_call` are trusted providers, and dynamic
or external code needs process isolation for stronger guarantees.

- [ ] **Step 5: Verify and commit Issue 4**

Run:

```powershell
uv run pytest tests/test_boundary_audit.py tests/test_reference_agent.py tests/test_record_replay.py -v
uv run pytest -q
git add flightrec/audit.py flightrec/agent/reference_agent.py tests/test_boundary_audit.py README.md
git commit -m "feat: validate agent boundary discipline"
```

---

### Task 5: Local-only, injection-safe, offline V3 viewer

**Files:**
- Create: `flightrec/web/security.py`
- Create: `tests/test_web_security.py`
- Create: `flightrec/static/vendor/cytoscape.min.js`
- Create: `flightrec/static/vendor/LICENSE-cytoscape.txt`
- Modify: `flightrec/web/server.py`
- Modify: `flightrec/web/graph.py`
- Modify: `flightrec/cli.py`
- Modify: `flightrec/static/index.html`
- Modify: `flightrec/static/app.js`
- Modify: `pyproject.toml`
- Modify: `tests/test_web_api.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `is_loopback_host(host: str) -> bool`.
- Produces: `is_allowed_origin(origin: Optional[str]) -> bool`.
- Produces: `LocalOnlyMiddleware`.
- Produces: `create_app(store: Optional[Store] = None, *, testing: bool = False) -> FastAPI`.

- [ ] **Step 1: Add failing security and error-contract tests**

Tests assert:

```python
assert is_loopback_host("127.0.0.1")
assert is_loopback_host("::1")
assert not is_loopback_host("0.0.0.0")
assert client.get("/api/traces", client=("203.0.113.5", 1234)).status_code == 403
assert client.get("/api/diff/missing/a").status_code == 404
```

Static-source tests assert `index.html` contains `/vendor/cytoscape.min.js` and no `unpkg.com`,
and `app.js` contains no assignment of trace-derived values to `innerHTML`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_web_security.py tests/test_web_api.py -v`  
Expected: FAIL because the security module/app factory/local asset do not exist and diff returns 200.

- [ ] **Step 3: Implement the network boundary and app factory**

Use `ipaddress.ip_address(host).is_loopback` with explicit `localhost` handling. Middleware rejects
non-loopback clients with 403 unless `testing=True`. Websockets close with 4403 unless client and
parsed origin hostname are loopback. `flightrec serve` raises `typer.BadParameter` for non-loopback
bind hosts. Move Store ownership to `app.state.store`; retain `app = create_app()` for uvicorn.

- [ ] **Step 4: Normalize error and background behavior**

`diff_overlay` raises `ValueError` if either trace is absent; the endpoint maps it to 404.
Background-run exception handling calls `logger.exception` before marking the trace failed.

- [ ] **Step 5: Remove HTML sinks and vendor Cytoscape**

Replace `innerHTML` construction with `replaceChildren`, `createElement`, and `textContent`.
Download Cytoscape 3.28.1 and its upstream license from the tagged official repository, place them
under `flightrec/static/vendor/`, load `/vendor/cytoscape.min.js`, and add `static/vendor/*` to
setuptools package data.

- [ ] **Step 6: Verify and commit Issue 5**

Run:

```powershell
uv run pytest tests/test_web_security.py tests/test_web_api.py tests/test_web_graph.py -v
uv run pytest -q
git add flightrec/web flightrec/cli.py flightrec/static pyproject.toml tests/test_web_security.py tests/test_web_api.py README.md uv.lock
git commit -m "fix: harden the local V3 viewer"
```

---

### Task 6: Final repository and release verification

**Files:**
- Modify: `.gitignore` only if verification exposes another generated path.
- Modify: `README.md` for the final canonical setup/run/check instructions.

**Interfaces:**
- Consumes: every command and artifact produced by Tasks 1-5.
- Produces: one clean, pushed `master` with no additional branches/worktrees.

- [ ] **Step 1: Run repository hygiene assertions**

Run:

```powershell
git worktree list
git branch -a
git status --short --branch
git check-ignore .venv .tokensave .claude/worktrees/example.db flightrec.db-shm flightrec.db-wal
```

Expected: one worktree, only `master` plus `origin/master`, and every local artifact is ignored.

- [ ] **Step 2: Run the complete verification matrix**

Run:

```powershell
uv lock --check
uv sync --locked --extra dev
uv pip check
uv run pytest -v
uv run flightrec --help
uv run flightrec ls
git diff --check
```

Expected: all commands exit zero; the migrated historical traces are listed.

- [ ] **Step 3: Inspect the final diff and commit documentation cleanup**

If README or ignore changes remain:

```powershell
git add README.md .gitignore
git commit -m "docs: finalize hardened development workflow"
```

- [ ] **Step 4: Push and verify remote state**

```powershell
git push origin master
git fetch --prune origin
git status --short --branch
git branch -a
```

Expected: local `master` equals `origin/master`; no other branches or worktrees exist.

