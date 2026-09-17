"""Experiment database — one SQLite file that outlives any single run.

Modelled on CHIA's :class:`chia.database.sqlite_node.SQLiteNode` (see also
`chia/examples/titan/db_node.py` and `chia/examples/gem5_align/alignment_db.py`):
same connection PRAGMAs, same ``list[dict]`` row shape, same
``spawn_query_tool()`` handle for letting the optimizer agent read its own
experiment history over MCP.

Why not literally *be* a ``SQLiteNode``
--------------------------------------
``SQLiteNode`` is a ``ColocatedNode``: every member is a ``@ChiaFunction``
dispatched through Ray, and construction requires an initialized Ray context
plus a placement group / node pin. This store is written from three places
that must keep working without Ray:

  * ``python loop/db.py --summary`` (a plain CLI on the driver's laptop),
  * ``loop.py``'s write callsites, which must still land on disk when a run
    dies mid-iteration (a Ray dispatch is a strictly worse failure mode than
    a local ``commit()``), and
  * the outer loop's ``sample_parent()``, a pure function over the file.

The file also lives on the driver's own disk, so the colocation guarantee
``SQLiteNode`` exists to provide buys nothing here. We therefore keep stdlib
``sqlite3`` for reads/writes but mirror ``SQLiteNode``'s conventions, and use
the *real* CHIA ``SQLiteQueryTool`` for the agent-facing tool — that is the
piece that actually has to interoperate. If the store ever moves to a remote
host, swapping the private ``connect()`` helper for a ``SQLiteNode`` handle is
a contained change: the public API below is already SQLiteNode-shaped.

Tables
------
    design_points — one row per hardware design point (the outer loop's unit)
    runs          — one row per `run_inner()` invocation
    iters         — one row per measurement, iteration 0 being the baseline

Every write is its own connection + commit: cheap at this scale, and it means
a `kill -9` mid-run loses at most the iteration in flight. Schema changes are
applied idempotently by :func:`migrate` on every ``connect()``, so old
``out/loop/*.db`` files pick up new columns in place.

    python loop/db.py --summary        # best result per (config, kernel)
    python loop/db.py --runs           # every run, newest first
    python loop/db.py --iters RUN_ID   # one run's iterations
    python loop/db.py --design-points  # every design point
    python loop/db.py --sql "SELECT ..."   # ad-hoc read-only query
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterable, Sequence

try:
    from constants import DB_PATH as _DEFAULT_DB_PATH
except ImportError:  # invoked as `python loop/db.py` from elsewhere
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from constants import DB_PATH as _DEFAULT_DB_PATH

DB_PATH = Path(os.environ.get("AETHER_DB", _DEFAULT_DB_PATH))

#: WAL is SQLiteNode's default, but it is *unsafe on network filesystems* and
#: ``out/`` may well be one. Off by default; ``AETHER_DB_WAL=1`` to opt in.
_WAL = os.environ.get("AETHER_DB_WAL", "") not in ("", "0", "false", "no")
_BUSY_TIMEOUT_S = float(os.environ.get("AETHER_DB_BUSY_TIMEOUT", "30"))

# --- status vocabulary (iters.status) --------------------------------------
ST_OK = "ok"
ST_BUILD_FAILED = "build_failed"
ST_SELFCHECK_FAILED = "selfcheck_failed"
ST_SIM_FAILED = "sim_failed"
ST_RETRY_EXHAUSTED = "retry_exhausted"

#: Statuses that disqualify an iteration from being sampled as a parent.
DEAD_STATUSES = (ST_BUILD_FAILED, ST_SELFCHECK_FAILED, ST_SIM_FAILED,
                 ST_RETRY_EXHAUSTED)

#: `nodes.Measurement.kind` -> `iters.status`.
_KIND_TO_STATUS = {
    "ok": ST_OK,
    "baseline": ST_OK,
    "build_failure": ST_BUILD_FAILED,
    "incorrect": ST_SELFCHECK_FAILED,
    "sim_failure": ST_SIM_FAILED,
    "retry_exhausted": ST_RETRY_EXHAUSTED,
}


def status_from_kind(kind: str | None, passed: bool) -> str:
    """Map a `Measurement.kind` onto the `iters.status` vocabulary."""
    st = _KIND_TO_STATUS.get((kind or "").strip())
    if st is not None:
        return st
    return ST_OK if passed else ST_SIM_FAILED


SCHEMA = """
CREATE TABLE IF NOT EXISTS design_points (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    params_json     TEXT NOT NULL DEFAULT '{}',
    config          TEXT NOT NULL,
    area_proxy_mm2  REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    parent_id       INTEGER REFERENCES design_points(id),
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    config          TEXT NOT NULL,
    kernel          TEXT NOT NULL,
    started         TEXT NOT NULL,
    llm_model       TEXT,
    finished        TEXT,
    iters_planned   INTEGER,
    baseline_cycles INTEGER,
    best_cycles     INTEGER,
    cost_usd        REAL DEFAULT 0,
    in_tokens       INTEGER DEFAULT 0,
    out_tokens      INTEGER DEFAULT 0,
    out_dir         TEXT,
    note            TEXT,
    design_point_id INTEGER REFERENCES design_points(id)
);

CREATE TABLE IF NOT EXISTS iters (
    run_id      TEXT NOT NULL,
    iter        INTEGER NOT NULL,
    kind        TEXT,
    mcycle      INTEGER,
    minstret    INTEGER,
    passed      INTEGER,
    cost_usd    REAL DEFAULT 0,
    in_tokens   INTEGER DEFAULT 0,
    out_tokens  INTEGER DEFAULT 0,
    cache_read_tokens     INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    num_turns   INTEGER,
    kernel_path TEXT,
    note        TEXT,
    iter_id         TEXT,
    design_point_id INTEGER REFERENCES design_points(id),
    parent_id       TEXT,
    diff            TEXT,
    fitness         INTEGER,
    status          TEXT,
    PRIMARY KEY (run_id, iter)
);

"""

#: Indexes live apart from the tables: several cover columns that `migrate()`
#: only ALTERs in afterwards, so they must be created after that pass.
INDEXES = """
CREATE INDEX IF NOT EXISTS iters_run ON iters(run_id);
CREATE INDEX IF NOT EXISTS runs_ck ON runs(config, kernel);
CREATE UNIQUE INDEX IF NOT EXISTS iters_iid ON iters(iter_id);
CREATE INDEX IF NOT EXISTS iters_dp ON iters(design_point_id, fitness);
CREATE INDEX IF NOT EXISTS runs_dp ON runs(design_point_id);
"""

#: Columns added after the original two-table schema shipped. `migrate()`
#: adds any that a pre-existing file is missing (SQLiteNode.add_column_if_missing).
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "runs": [
        ("design_point_id", "INTEGER"),
    ],
    "iters": [
        ("iter_id", "TEXT"),
        ("design_point_id", "INTEGER"),
        ("parent_id", "TEXT"),
        ("diff", "TEXT"),
        ("fitness", "INTEGER"),
        ("status", "TEXT"),
    ],
}


# ---------------------------------------------------------------------------
# Connection / migration
# ---------------------------------------------------------------------------

def iter_id(run_id: str, iter: int) -> str:
    """Stable surrogate key for one iteration row (``iters.parent_id`` points
    at one of these). SQLite cannot bolt an AUTOINCREMENT column onto an
    existing table, so the id is derived from the natural key instead."""
    return f"{run_id}#{iter}"


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def migrate(con: sqlite3.Connection) -> list[str]:
    """Bring an existing file up to the current schema. Idempotent; returns
    the list of changes applied (empty on an already-current file)."""
    applied: list[str] = []
    con.executescript(SCHEMA)
    for table, cols in _ADDED_COLUMNS.items():
        have = _table_columns(con, table)
        if not have:            # table did not exist; executescript made it
            continue
        for name, decl in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                applied.append(f"{table}.{name}")

    # Backfill derived columns for rows written by the pre-migration loop.
    n = con.execute(
        "UPDATE iters SET iter_id = run_id || '#' || iter "
        "WHERE iter_id IS NULL").rowcount
    if n:
        applied.append(f"iters.iter_id backfilled ({n})")
    n = con.execute(
        "UPDATE iters SET status = CASE"
        "  WHEN passed = 1 THEN 'ok'"
        "  WHEN kind = 'build_failure' THEN 'build_failed'"
        "  WHEN kind = 'incorrect' THEN 'selfcheck_failed'"
        "  ELSE 'sim_failed' END "
        "WHERE status IS NULL").rowcount
    if n:
        applied.append(f"iters.status backfilled ({n})")
    n = con.execute(
        "UPDATE iters SET fitness = mcycle "
        "WHERE fitness IS NULL AND status = 'ok' AND mcycle IS NOT NULL").rowcount
    if n:
        applied.append(f"iters.fitness backfilled ({n})")

    con.executescript(INDEXES)      # after the ALTERs: some index new columns
    con.commit()
    return applied


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the store with SQLiteNode's PRAGMA defaults and migrate it."""
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=_BUSY_TIMEOUT_S)
    con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
    con.execute("PRAGMA foreign_keys=ON")
    if _WAL:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    migrate(con)
    return con


@contextmanager
def _tx(path=None):
    """A connection that commits on clean exit, rolls back on error, and —
    unlike a bare ``sqlite3.Connection`` context manager — always closes."""
    con = connect(path)
    try:
        with con:
            yield con
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rows(cur: Iterable[sqlite3.Row]) -> list[dict]:
    """SQLiteNode's row shape: plain ``list[dict]``."""
    return [dict(r) for r in cur]


# ---------------------------------------------------------------------------
# Generic read helpers (SQLiteNode member names, local implementation)
# ---------------------------------------------------------------------------

def query(sql: str, params: Sequence | dict = (), *, limit: int | None = None,
          path=None) -> list[dict]:
    with _tx(path) as con:
        cur = con.execute(sql, params)
        rows = cur.fetchmany(limit) if limit else cur.fetchall()
    return _rows(rows)


def query_one(sql: str, params: Sequence | dict = (), path=None) -> dict | None:
    rows = query(sql, params, limit=1, path=path)
    return rows[0] if rows else None


def query_value(sql: str, params: Sequence | dict = (), *, default: Any = None,
                path=None) -> Any:
    row = query_one(sql, params, path=path)
    if row is None:
        return default
    return next(iter(row.values()))


def schema(path=None) -> str:
    """CREATE statements from the live DB (what the agent's tool reports)."""
    rows = query(
        "SELECT sql FROM sqlite_master WHERE type IN ('table','index') "
        "AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type, name",
        path=path)
    return ";\n\n".join(r["sql"] for r in rows) + (";" if rows else "")


# ---------------------------------------------------------------------------
# Design points (the outer loop's unit of work)
# ---------------------------------------------------------------------------

def _params_dict(params: Any) -> dict:
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    to_dict = getattr(params, "to_dict", None)   # hwconfig.HwPoint
    if callable(to_dict):
        return dict(to_dict())
    if hasattr(params, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(params)
    raise TypeError(f"cannot serialize design-point params of type {type(params)}")


def add_design_point(name: str, params: Any = None, config: str = "", *,
                     area_proxy_mm2: float | None = None,
                     status: str = "pending", parent_id: int | None = None,
                     path=None) -> int:
    """Insert (or update in place, by unique *name*) one design point and
    return its integer id.

    *params* may be a dict or anything with ``.to_dict()`` — e.g.
    ``hwconfig.HwPoint``. *config* defaults to ``params["config"]`` when the
    caller leaves it blank.
    """
    p = _params_dict(params)
    config = config or str(p.get("config") or "")
    if not config:
        raise ValueError("design point needs a config (arg or params['config'])")
    with _tx(path) as con:
        con.execute(
            "INSERT INTO design_points"
            " (name, params_json, config, area_proxy_mm2, status, parent_id,"
            "  created_at) VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET"
            "  params_json=excluded.params_json, config=excluded.config,"
            "  area_proxy_mm2=COALESCE(excluded.area_proxy_mm2, area_proxy_mm2),"
            "  status=excluded.status,"
            "  parent_id=COALESCE(excluded.parent_id, parent_id)",
            (name, json.dumps(p, sort_keys=True), config, area_proxy_mm2,
             status, parent_id, _now()))
        return int(con.execute("SELECT id FROM design_points WHERE name=?",
                               (name,)).fetchone()[0])


def update_design_point(design_point_id: int, *, path=None, **fields) -> None:
    """Patch any of area_proxy_mm2 / status / params_json / parent_id / config."""
    allowed = {"area_proxy_mm2", "status", "params_json", "parent_id", "config",
               "name"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"unknown design_points column(s): {sorted(bad)}")
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with _tx(path) as con:
        con.execute(f"UPDATE design_points SET {sets} WHERE id=?",
                    (*fields.values(), design_point_id))


def get_design_point(ref: int | str, path=None) -> dict | None:
    """Look one up by id (int) or name (str)."""
    col = "id" if isinstance(ref, int) else "name"
    return query_one(f"SELECT * FROM design_points WHERE {col}=?", (ref,),
                     path=path)


def design_points(path=None, limit: int = 200) -> list[dict]:
    return query("SELECT * FROM design_points ORDER BY id LIMIT ?", (limit,),
                 path=path)


# ---------------------------------------------------------------------------
# Writes (called from loop.py)
# ---------------------------------------------------------------------------

def start_run(run_id: str, config: str, kernel: str, llm_model: str,
              iters_planned: int, out_dir: str, design_point_id: int | None = None,
              path=None) -> None:
    with _tx(path) as con:
        con.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, config, kernel, started, llm_model, iters_planned,"
            " out_dir, design_point_id) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, config, kernel, _now(), llm_model, iters_planned, out_dir,
             design_point_id))


def record_iter(run_id: str, iter: int, kind: str, mcycle: int | None,
                minstret: int | None, passed: bool, kernel_path: str = "",
                note: str = "", cost: dict | None = None,
                design_point_id: int | None = None,
                parent_id: str | None = None, diff: str | None = None,
                status: str | None = None, path=None) -> str:
    """One measurement; returns the row's :func:`iter_id`.

    *cost* is the LLM usage that produced it (empty for the baseline, which no
    LLM turn preceded). *parent_id* is the ``iter_id`` this kernel was derived
    from, *diff* the unified diff against that parent (or a path to it), and
    *fitness* is derived: the measured cycle count when the self-check passed,
    NULL otherwise, so a failed attempt can never win a ``MIN(fitness)``.
    """
    c = cost or {}
    iid = iter_id(run_id, iter)
    st = status or status_from_kind(kind, passed)
    fitness = mcycle if (passed and st == ST_OK) else None
    with _tx(path) as con:
        if design_point_id is None:
            row = con.execute("SELECT design_point_id FROM runs WHERE run_id=?",
                              (run_id,)).fetchone()
            design_point_id = row["design_point_id"] if row else None
        con.execute(
            "INSERT OR REPLACE INTO iters (run_id, iter, kind, mcycle, minstret,"
            " passed, cost_usd, in_tokens, out_tokens, cache_read_tokens,"
            " cache_creation_tokens, num_turns, kernel_path, note,"
            " iter_id, design_point_id, parent_id, diff, fitness, status)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, iter, kind, mcycle, minstret, int(bool(passed)),
             c.get("cost_usd", 0.0), c.get("input_tokens", 0),
             c.get("output_tokens", 0), c.get("cache_read_input_tokens", 0),
             c.get("cache_creation_input_tokens", 0), c.get("num_turns"),
             kernel_path, note, iid, design_point_id, parent_id, diff,
             fitness, st))
    return iid


def finish_run(run_id: str, baseline_cycles: int | None, best_cycles: int | None,
               note: str = "", path=None) -> None:
    """Close a run out and roll the per-iteration cost up into its row."""
    with _tx(path) as con:
        tot = con.execute(
            "SELECT COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(in_tokens),0) i,"
            " COALESCE(SUM(out_tokens),0) o FROM iters WHERE run_id=?",
            (run_id,)).fetchone()
        con.execute(
            "UPDATE runs SET finished=?, baseline_cycles=?, best_cycles=?,"
            " cost_usd=?, in_tokens=?, out_tokens=?, note=? WHERE run_id=?",
            (_now(), baseline_cycles, best_cycles, tot["c"], tot["i"], tot["o"],
             note, run_id))


def make_diff(old: bytes | str | None, new: bytes | str | None, *,
              from_name: str = "parent", to_name: str = "child",
              max_chars: int = 20000) -> str | None:
    """Unified diff between two kernel sources, for ``iters.diff``.

    Returns None when either side is missing. Long diffs are truncated to
    *max_chars* with a marker — the DB is an index, not an artifact store; the
    full sources are on disk at ``iters.kernel_path``.
    """
    if old is None or new is None:
        return None
    dec = lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else b
    text = "".join(difflib.unified_diff(
        dec(old).splitlines(keepends=True), dec(new).splitlines(keepends=True),
        fromfile=from_name, tofile=to_name))
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[diff truncated]...\n"
    return text


# ---------------------------------------------------------------------------
# Parent selection for the outer loop
# ---------------------------------------------------------------------------

def sample_parent(design_point_id: int, k: int = 3, *, kernel: str | None = None,
                  rng: random.Random | None = None, path=None) -> list[dict]:
    """Candidate parent kernels for a new attempt on *design_point_id*.

    Returns the top-*k* viable iterations by fitness (fewest cycles first)
    plus one uniformly random viable iteration from outside that top-k — the
    exploit/explore mix a regenerative outer loop wants. Rows whose status is
    in :data:`DEAD_STATUSES`, or that never produced a fitness, are excluded,
    as are rows with no kernel on disk.

    Pure function of the DB: no Ray, no side effects. Deterministic apart from
    the one random pick, which takes *rng* if you need it reproducible.
    Restrict to one benchmark with ``kernel=``. The list is ordered best-first
    with the random pick last; it may be shorter than ``k + 1`` (or empty).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    placeholders = ",".join("?" * len(DEAD_STATUSES))
    sql = (
        "SELECT i.*, r.kernel AS kernel, r.config AS config"
        " FROM iters i JOIN runs r USING (run_id)"
        " WHERE i.design_point_id IS ? AND i.fitness IS NOT NULL"
        f" AND (i.status IS NULL OR i.status NOT IN ({placeholders}))"
        " AND i.kernel_path IS NOT NULL AND i.kernel_path != ''")
    params: list = [design_point_id, *DEAD_STATUSES]
    if kernel is not None:
        sql += " AND r.kernel = ?"
        params.append(kernel)
    sql += " ORDER BY i.fitness ASC, i.iter ASC"
    rows = query(sql, params, path=path)

    top = rows[:k]
    rest = rows[k:]
    if rest:
        top.append((rng or random).choice(rest))
    return top


def best_iter(design_point_id: int, kernel: str | None = None,
              path=None) -> dict | None:
    """The single best passing iteration for a design point (None if none)."""
    rows = sample_parent(design_point_id, 1, kernel=kernel, path=path)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# LLM-facing read-only SQL tool (CHIA's SQLiteQueryTool)
# ---------------------------------------------------------------------------

#: One-line blurb for prompts; see `llm.py`.
QUERY_TOOL_BLURB = (
    "You also have `{name}_query` (read-only SQL) and `{name}_schema` over the "
    "experiment database, which holds every measurement ever taken on this and "
    "other hardware design points (tables: design_points, runs, iters — "
    "`fitness` is cycles, lower is better)."
)


def spawn_query_tool(name: str = "aether_db", path=None, *,
                     read_write: bool = False, task_options: dict | None = None,
                     row_limit: int = 100, **tool_kwargs):
    """CHIA's ``SQLiteNode.spawn_query_tool`` pattern, minus the node.

    Returns a live :class:`chia.database.sqlite_node.SQLiteQueryTool` MCP tool
    exposing ``<name>_query`` (read-only SQL; the connection is opened
    ``mode=ro`` so LLM-authored writes raise) and ``<name>_schema``. Requires
    an initialized Ray context — the tool is an MCP server actor, exactly like
    ``BashTool``. Call ``.stop()`` when done (``llm.py`` does this at exit).

    The DB file must be reachable from wherever the actor lands; pass
    ``task_options={"resources": {...}}`` (or a NodeAffinity scheduling
    strategy) to pin it if the cluster is ever multi-node.
    """
    from chia.database.sqlite_node import SQLiteQueryTool

    p = Path(path or DB_PATH)
    connect(p).close()      # make sure the file + schema exist before mode=ro
    return SQLiteQueryTool(name, db_path=str(p), task_options=task_options,
                           read_write=read_write, row_limit=row_limit,
                           **tool_kwargs)


# ---------------------------------------------------------------------------
# Reads / CLI
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(no rows)"
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows))
         for i, h in enumerate(headers)]
    line = "  ".join("-" * x for x in w)
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)), line]
    out += ["  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(out)


def summary(path=None) -> str:
    """Best result per (config, kernel) across every run in the store."""
    with _tx(path) as con:
        rows = con.execute("""
            SELECT r.config, r.kernel,
                   COUNT(DISTINCT r.run_id)                       AS runs,
                   MAX(CASE WHEN i.iter=0 THEN i.mcycle END)      AS baseline,
                   MIN(CASE WHEN i.passed=1 THEN i.mcycle END)    AS best,
                   COALESCE(SUM(i.cost_usd),0)                    AS cost,
                   COALESCE(SUM(i.in_tokens),0)                   AS in_tok,
                   COALESCE(SUM(i.out_tokens),0)                  AS out_tok
            FROM runs r JOIN iters i USING (run_id)
            GROUP BY r.config, r.kernel
            ORDER BY r.kernel, r.config
        """).fetchall()
    body = []
    for r in rows:
        base, best = r["baseline"], r["best"]
        sp = f"{base / best:.3f}x" if base and best else "n/a"
        body.append([r["config"], r["kernel"], r["runs"], base or "n/a",
                     best or "n/a", sp, f"${r['cost']:.2f}",
                     r["in_tok"], r["out_tok"]])
    return _table(["config", "kernel", "runs", "baseline", "best", "speedup",
                   "cost", "in_tok", "out_tok"], body)


def runs_table(path=None, limit: int = 30) -> str:
    with _tx(path) as con:
        rows = con.execute(
            "SELECT * FROM runs ORDER BY started DESC LIMIT ?", (limit,)).fetchall()
    body = [[r["run_id"], r["kernel"], r["config"], r["design_point_id"] or "-",
             r["llm_model"] or "-",
             r["baseline_cycles"] or "-", r["best_cycles"] or "-",
             f"${r['cost_usd'] or 0:.4f}", r["in_tokens"] or 0,
             r["out_tokens"] or 0, "yes" if r["finished"] else "no"]
            for r in rows]
    return _table(["run_id", "kernel", "config", "dp", "model", "baseline",
                   "best", "cost", "in_tok", "out_tok", "done"], body)


def iters_table(run_id: str, path=None) -> str:
    with _tx(path) as con:
        rows = con.execute(
            "SELECT * FROM iters WHERE run_id=? ORDER BY iter", (run_id,)).fetchall()
    body = [[r["iter"], r["kind"], r["status"] or "-", r["mcycle"] or "-",
             r["minstret"] or "-", r["fitness"] or "-",
             r["parent_id"] or "-", f"${r['cost_usd'] or 0:.4f}",
             r["in_tokens"] or 0, r["out_tokens"] or 0,
             r["cache_read_tokens"] or 0, r["num_turns"] or "-"]
            for r in rows]
    return _table(["iter", "kind", "status", "mcycle", "minstret", "fitness",
                   "parent", "cost", "in_tok", "out_tok", "cache_rd", "turns"],
                  body)


def design_points_table(path=None) -> str:
    rows = design_points(path)
    body = [[r["id"], r["name"], r["config"], r["status"],
             r["area_proxy_mm2"] if r["area_proxy_mm2"] is not None else "-",
             r["parent_id"] if r["parent_id"] is not None else "-",
             r["created_at"]] for r in rows]
    return _table(["id", "name", "config", "status", "area_mm2", "parent",
                   "created"], body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true",
                    help="best result per (config, kernel) across all runs")
    ap.add_argument("--runs", action="store_true", help="list runs, newest first")
    ap.add_argument("--iters", metavar="RUN_ID", help="one run's iterations")
    ap.add_argument("--design-points", action="store_true",
                    help="list hardware design points")
    ap.add_argument("--sql", metavar="SELECT", help="ad-hoc read-only query")
    ap.add_argument("--schema", action="store_true", help="dump the live schema")
    ap.add_argument("--migrate", action="store_true",
                    help="apply pending schema migrations and report them")
    ap.add_argument("--db", default=None, help=f"database path (default {DB_PATH})")
    args = ap.parse_args()

    path = args.db
    if args.migrate:
        con = connect(path)          # connect() migrates; report what it did
        print("\n".join(migrate(con)) or "(schema already current)")
        con.close()
    elif args.schema:
        print(schema(path))
    elif args.sql:
        rows = query(args.sql, path=path)
        if not rows:
            print("(no rows)")
        else:
            heads = list(rows[0])
            print(_table(heads, [[r[h] for h in heads] for r in rows]))
    elif args.iters:
        print(iters_table(args.iters, path))
    elif args.runs:
        print(runs_table(path))
    elif args.design_points:
        print(design_points_table(path))
    else:  # --summary is the default view
        print(f"# {Path(path or DB_PATH)}\n")
        print(summary(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
