"""The lineage database: every variant, its score, and what it came from.

Open by default, like the reveng benchmark's store, because the deliverable is
not a predictor -- it is the claim that a fixed-depth score misranks parts of
its own Pareto front, and a claim like that is worth exactly as much as the
data behind it.

Two things are recorded that a plain evolutionary log would not bother with:

* ``depth_scores`` per variant, so the rank correlation between VFS and
  end-to-end IPC can be recomputed at any depth after the fact without
  re-running a simulation.  The counters are measured once; the depth is
  arithmetic.
* the *parent* of every variant, so a design that wins can be traced back to
  the edit that made it win.  A search that cannot say which change helped has
  produced a number, not a finding.
"""

from __future__ import annotations

import json

from chia.base.ChiaFunction import get
from chia.database.sqlite_node import SQLiteNode

SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    arm          TEXT NOT NULL,
    seed_predictor TEXT NOT NULL,
    generations  INTEGER NOT NULL,
    inner_traces INTEGER NOT NULL,
    depth_sweep  TEXT NOT NULL,
    notes        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS variants (
    sweep_id     INTEGER NOT NULL,
    variant_id   TEXT NOT NULL,
    generation   INTEGER NOT NULL,
    parent_id    TEXT,
    struct_name  TEXT NOT NULL,
    template_args TEXT DEFAULT '',
    rationale    TEXT,
    source       TEXT NOT NULL,
    failure      TEXT NOT NULL,
    archived     INTEGER NOT NULL,
    cell         TEXT,
    repair_rounds INTEGER DEFAULT 0,
    -- Kendall tau between this design's VFS-at-depth and gem5's measured
    -- IPC-at-depth.  NULL until Tier 2 has run: it is the per-variant form of
    -- the question the whole loop asks, so it belongs next to the variant
    -- rather than buried in a tier row.
    tau_vfs_vs_ipc REAL,
    PRIMARY KEY (sweep_id, variant_id)
);

-- One row per (variant, tier).  Tier 0 fills vfs/epi/latency/mpki; tier 1
-- fills mpki/ipc/rob; tier 2 fills ipc and per_config_json.
CREATE TABLE IF NOT EXISTS tier_results (
    sweep_id     INTEGER NOT NULL,
    variant_id   TEXT NOT NULL,
    tier         INTEGER NOT NULL,
    built        INTEGER NOT NULL,
    ran          INTEGER NOT NULL,
    vfs          REAL,
    epi          REAL,
    p1_latency   INTEGER,
    p2_latency   INTEGER,
    mpki         REAL,
    ipc          REAL,
    cpi          REAL,
    rob_at_mispredict REAL,
    n_traces     INTEGER,
    build_s      REAL,
    per_config_json TEXT,
    diagnostics  TEXT,
    PRIMARY KEY (sweep_id, variant_id, tier)
);

-- The depth sweep, one row per (variant, depth).  This is the table the
-- headline result is computed from.
CREATE TABLE IF NOT EXISTS depth_scores (
    sweep_id     INTEGER NOT NULL,
    variant_id   TEXT NOT NULL,
    depth        INTEGER NOT NULL,
    vfs          REAL NOT NULL,
    cpi          REAL NOT NULL,
    PRIMARY KEY (sweep_id, variant_id, depth)
);

-- Archive state after each generation, so coverage over time is recoverable.
CREATE TABLE IF NOT EXISTS archive_snapshots (
    sweep_id     INTEGER NOT NULL,
    generation   INTEGER NOT NULL,
    cells        INTEGER NOT NULL,
    coverage     REAL NOT NULL,
    best_vfs     REAL,
    best_variant TEXT,
    archive_json TEXT NOT NULL,
    PRIMARY KEY (sweep_id, generation)
);
"""


class LineageDB:
    """Thin wrapper over :class:`SQLiteNode` with this loop's schema."""

    def __init__(self, db_path: str):
        self.node = SQLiteNode(db_path, pin_to_current_node=True)
        self.sweep_id: int | None = None

    def __enter__(self):
        self.node.__enter__()
        get(self.node.init_schema.chia_remote(SCHEMA))
        return self

    def __exit__(self, *exc):
        return self.node.__exit__(*exc)

    # -- writes --------------------------------------------------------------

    def start_sweep(self, started_at: str, arm: str, seed_predictor: str,
                    generations: int, inner_traces: int, depth_sweep,
                    notes: str = "") -> int:
        get(self.node.execute.chia_remote(
            "INSERT INTO sweeps (started_at, arm, seed_predictor, generations, "
            "inner_traces, depth_sweep, notes) VALUES (?,?,?,?,?,?,?)",
            (started_at, arm, seed_predictor, generations, inner_traces,
             json.dumps(list(depth_sweep)), notes)))
        self.sweep_id = int(get(self.node.query_value.chia_remote(
            "SELECT MAX(sweep_id) FROM sweeps")))
        return self.sweep_id

    def record_evaluation(self, evaluation, repair_rounds: int = 0) -> None:
        """One variant and every tier it reached.

        Called for failures too.  A variant that never compiled is the most
        informative row in the table when the question is whether the agent
        understands HARCOM, and dropping it would make the build-failure rate
        unrecoverable.
        """
        v = evaluation.variant
        sid = self.sweep_id
        get(self.node.execute.chia_remote(
            "INSERT OR REPLACE INTO variants (sweep_id, variant_id, generation, "
            "parent_id, struct_name, template_args, rationale, source, failure, "
            "archived, cell, repair_rounds, tau_vfs_vs_ipc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, v.variant_id, v.generation, v.parent_id, v.struct_name,
             v.template_args, v.rationale, v.harcom_source, evaluation.failure.value,
             int(evaluation.archived),
             json.dumps(list(evaluation.cell)) if evaluation.cell else None,
             repair_rounds, evaluation.tau_vfs_vs_ipc)))

        rows = []
        for tier, out in evaluation.tiers.items():
            m = out.metrics
            rows.append((
                sid, v.variant_id, int(tier), int(out.built), int(out.ran),
                m.vfs if m else None,
                m.epi if m else None,
                m.p1_latency if m else None,
                m.p2_latency if m else None,
                out.mpki if out.mpki is not None else (m.mpki if m else None),
                out.ipc if out.ipc is not None else (m.ipc if m else None),
                m.cpi if m else None,
                out.rob_occupancy_at_mispredict,
                m.n_traces if m else None,
                out.build_duration_s,
                json.dumps(out.per_config_ipc) if out.per_config_ipc else None,
                (out.build_diagnostics or out.run_diagnostics)[:8000],
            ))
        if rows:
            get(self.node.executemany.chia_remote(
                "INSERT OR REPLACE INTO tier_results (sweep_id, variant_id, tier, "
                "built, ran, vfs, epi, p1_latency, p2_latency, mpki, ipc, cpi, "
                "rob_at_mispredict, n_traces, build_s, per_config_json, "
                "diagnostics) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows))

        t0 = evaluation.tier0
        if t0 and t0.depth_scores:
            get(self.node.executemany.chia_remote(
                "INSERT OR REPLACE INTO depth_scores (sweep_id, variant_id, "
                "depth, vfs, cpi) VALUES (?,?,?,?,?)",
                [(sid, v.variant_id, d, s.vfs, s.cpi)
                 for d, s in sorted(t0.depth_scores.items())]))

    def snapshot_archive(self, generation: int, archive) -> None:
        best = archive.best()
        get(self.node.execute.chia_remote(
            "INSERT OR REPLACE INTO archive_snapshots (sweep_id, generation, "
            "cells, coverage, best_vfs, best_variant, archive_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (self.sweep_id, generation, len(archive), archive.coverage(),
             best.vfs if best else None,
             best.variant_id if best else None,
             archive.to_json())))

    # -- reads ---------------------------------------------------------------

    def depth_rank_table(self) -> list[dict]:
        """(depth, variant_id, vfs) rows for this sweep -- the headline result.

        Ordered so that reading down a depth gives that depth's ranking
        directly; the rank correlation between two depths is then a zip.
        """
        return get(self.node.query.chia_remote(
            "SELECT depth, variant_id, vfs FROM depth_scores WHERE sweep_id = ? "
            "ORDER BY depth, vfs DESC", (self.sweep_id,)))

    def archived_variants(self) -> list[dict]:
        """Every variant that took a cell, with its Tier-0 numbers."""
        return get(self.node.query.chia_remote(
            "SELECT v.variant_id, v.generation, v.parent_id, v.cell, "
            "       t.vfs, t.epi, t.p1_latency, t.p2_latency, t.mpki "
            "FROM variants v JOIN tier_results t "
            "  ON t.sweep_id = v.sweep_id AND t.variant_id = v.variant_id "
            "WHERE v.sweep_id = ? AND v.archived = 1 AND t.tier = 0 "
            "ORDER BY t.vfs DESC", (self.sweep_id,)))
