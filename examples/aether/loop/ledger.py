"""ledger.py — reproducible per-round time/cost/result ledger for the inner loop.

Joins three read-only sources, never writes to any of them:

  * ``out/loop/rounds.json``   — which run_ids belong to which named round
    (hand-maintained; round3+ is appended by whoever runs that round).
  * ``loop/aether.db``         — ``runs`` + ``iters`` tables (see ``db.py``).
  * ``loop/kernels.py``        — ``KERNELS[name].roofline_cycles`` (read-only
    import; this script never touches ``loop.py`` / ``constants.py``).

and writes two reproducible reports:

  * ``out/loop/ledger.md``    — human-readable tables (a-d, see module CLI help)
  * ``out/loop/ledger.json``  — the same data as structured JSON

Known data gaps (also written at the top of ledger.md):

  1. ``runs.cost_usd`` / ``runs.baseline_cycles`` / ``runs.best_cycles`` are
     only rolled up by ``finish_run()``. A run that died mid-flight (session
     limit, crash) has ``finished IS NULL`` and those columns stuck at their
     insert-time defaults (0 / NULL) even though iterations — and cost — did
     happen. This script therefore ALWAYS recomputes cost and best-cycles
     from ``iters`` directly and ignores the ``runs`` rollup columns.
  2. ``iters`` has no per-iteration timestamp column. Wall-clock time per
     iteration is inferred from the on-disk kernel file's mtime
     (``iters.kernel_path``), relative to the previous iteration's mtime (or
     the run's ``started`` time for the first iteration). This is an
     approximation: it is the time the file was last written, not the exact
     moment the simulator finished, and it silently breaks if the run
     directory is ever touched/rsynced after the fact (mtimes would then all
     collapse to the copy time). Iterations whose kernel file is missing get
     ``duration_s: null``.
  3. There is no distinct "LLM summary" column; the appendix's per-iteration
     note is ``iters.note``, which is loop.py's own status string (e.g.
     "passed, cycles=132653, instret=None"), not a model-authored summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

try:
    # Read-only import: only KERNELS[name].roofline_cycles is used below.
    # kernels.py is owned by another agent and may be mid-edit / momentarily
    # broken; degrade to "roofline unknown" rather than crash the ledger.
    from kernels import KERNELS  # noqa: E402
    _KERNELS_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001
    KERNELS = {}
    _KERNELS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUNDS_FILE = REPO_ROOT / "out" / "loop" / "rounds.json"
DEFAULT_MD_OUT = REPO_ROOT / "out" / "loop" / "ledger.md"
DEFAULT_JSON_OUT = REPO_ROOT / "out" / "loop" / "ledger.json"

DATA_GAPS_NOTE = """\
> **Data sources & limitations**
> - Every run's cost / best-cycles is recomputed from `iters` (`SUM(cost_usd)`,
>   `MIN(fitness)`), not read from `runs.cost_usd` / `runs.best_cycles` —
>   those columns are only rolled up by `finish_run()`, so a run that died
>   mid-flight (session limit, crash) would otherwise show `$0.00`.
> - `iters` has no per-iteration timestamp. Per-iteration duration is inferred
>   from the on-disk kernel file's mtime (`iters.kernel_path`), relative to
>   the previous iteration's mtime (or the run's `started` time for the first
>   iteration) — an approximation, and it breaks if the run directory is ever
>   touched/copied after the fact.
> - The appendix "note" column is `iters.note` (loop.py's own status string),
>   not a model-authored summary — the DB does not store one.
> - Regenerate with: `python loop/ledger.py --rounds-file out/loop/rounds.json`
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _mtime(path: str | None) -> datetime | None:
    if not path:
        return None
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _fmt_money(x: float | None) -> str:
    return "?" if x is None else f"${x:,.4f}"


def _fmt_cycles(x) -> str:
    return "?" if x is None else f"{int(x):,}"


def _fmt_ratio(best, roof) -> str:
    if not best or not roof:
        return "n/a"
    return f"{best / roof:.3f}x"


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------

def load_rounds(rounds_file: Path) -> list[dict]:
    data = json.loads(Path(rounds_file).read_text())
    return data["rounds"]


def build_run(run_entry: dict) -> dict:
    """One rounds.json run entry -> a fully joined record (db + on-disk mtimes
    + kernels.py roofline), independent of everything else in the round."""
    run_id = run_entry["run_id"]
    kernel = run_entry["kernel"]
    run_row = db.query_one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    iter_rows = db.query("SELECT * FROM iters WHERE run_id=? ORDER BY iter", (run_id,))

    started = _parse_ts(run_row["started"]) if run_row else None
    finished = _parse_ts(run_row["finished"]) if run_row else None
    llm_model = run_row["llm_model"] if run_row else None

    iters: list[dict] = []
    prev_ts = started
    total_cost = 0.0
    ok_count = 0
    fail_count = 0
    last_ts = started
    for it in iter_rows:
        ts = _mtime(it.get("kernel_path"))
        dur = (ts - prev_ts).total_seconds() if (ts and prev_ts) else None
        if ts:
            prev_ts = ts
            last_ts = ts
        cost = it.get("cost_usd") or 0.0
        total_cost += cost
        status = it.get("status")
        if status == db.ST_OK:
            ok_count += 1
        elif it["iter"] >= 0:  # don't count the carried-over seed as a "failure"
            fail_count += 1
        iters.append({
            "run_id": run_id,
            "iter": it["iter"],
            "kind": it.get("kind"),
            "status": status,
            "mcycle": it.get("mcycle"),
            "cost_usd": cost,
            "completed_at": ts.isoformat() if ts else None,
            "duration_s": dur,
            "duration_source": "kernel_file_mtime" if ts else None,
            "kernel_path": it.get("kernel_path"),
            "note": it.get("note"),
        })

    seed_row = next((i for i in iter_rows if i["iter"] == -1), None)
    baseline_row = next((i for i in iter_rows if i["iter"] == 0), None)
    seed_cycles = seed_row["mcycle"] if seed_row else None
    baseline_cycles = baseline_row["mcycle"] if baseline_row else (
        run_row["baseline_cycles"] if run_row else None)
    ok_fitness = [i["fitness"] for i in iter_rows if i.get("fitness") is not None]
    best_cycles = min(ok_fitness) if ok_fitness else None

    kdef = KERNELS.get(kernel)
    roofline = kdef.roofline_cycles if kdef else None

    end_ts = finished or last_ts
    wall_s = (end_ts - started).total_seconds() if (started and end_ts) else None
    ongoing = finished is None

    completed_iters = sum(1 for i in iter_rows if i["iter"] >= 1)

    return {
        "run_id": run_id,
        "kernel": kernel,
        "planned_iters": run_entry.get("planned_iters", run_row["iters_planned"] if run_row else None),
        "completed_iters": completed_iters,
        "seed_cycles": seed_cycles,
        "baseline_cycles": baseline_cycles,
        "best_cycles": best_cycles,
        "roofline_cycles": roofline,
        "best_over_roofline": _fmt_ratio(best_cycles, roofline),
        "cost_usd": round(total_cost, 6),
        "started": started.isoformat() if started else None,
        "finished": finished.isoformat() if finished else None,
        "ongoing_or_aborted": ongoing,
        "wall_clock_s": wall_s,
        "llm_model": llm_model,
        "ok_iters": ok_count,
        "fail_iters": fail_count,
        "note": run_entry.get("note", "") or (run_row["note"] if run_row else "") or "",
        "in_db": run_row is not None,
        "iters": iters,
    }


def build_round(round_entry: dict) -> dict:
    runs = [build_run(r) for r in round_entry["runs"]]
    starts = [ _parse_ts(r["started"]) for r in runs if r["started"] ]
    ends = []
    for r in runs:
        if r["finished"]:
            ends.append(_parse_ts(r["finished"]))
        elif r["iters"]:
            last = [i for i in r["iters"] if i["completed_at"]]
            if last:
                ends.append(_parse_ts(last[-1]["completed_at"]))
    round_start = min(starts) if starts else None
    round_end = max(ends) if ends else None
    wall_s = (round_end - round_start).total_seconds() if (round_start and round_end) else None

    models = sorted({r["llm_model"] for r in runs if r["llm_model"]})
    total_iters = sum(len(r["iters"]) for r in runs)
    ok_iters = sum(r["ok_iters"] for r in runs)
    fail_iters = sum(r["fail_iters"] for r in runs)
    total_cost = round(sum(r["cost_usd"] for r in runs), 6)

    return {
        "name": round_entry["name"],
        "round_note": round_entry.get("note", ""),
        "start": round_start.isoformat() if round_start else None,
        "end": round_end.isoformat() if round_end else None,
        "wall_clock_s": wall_s,
        "num_runs": len(runs),
        "total_iters": total_iters,
        "ok_iters": ok_iters,
        "fail_iters": fail_iters,
        "total_cost_usd": total_cost,
        "llm_models": models,
        "runs": runs,
    }


def build_kernel_history(rounds: list[dict]) -> list[dict]:
    """One row per kernel: baseline, then best-cycles/cumulative-cost after
    each round the kernel appears in, in round order."""
    kernels = {}
    for rnd in rounds:
        for r in rnd["runs"]:
            k = kernels.setdefault(r["kernel"], {
                "kernel": r["kernel"],
                "baseline_cycles": None,
                "roofline_cycles": r["roofline_cycles"],
                "points": [],  # (round_name, best_cycles, cumulative_cost)
                "_cum_cost": 0.0,
            })
            if k["baseline_cycles"] is None and r["baseline_cycles"] is not None:
                k["baseline_cycles"] = r["baseline_cycles"]
            k["_cum_cost"] += r["cost_usd"]
            k["points"].append({
                "round": rnd["name"],
                "best_cycles": r["best_cycles"],
                "cumulative_cost_usd": round(k["_cum_cost"], 6),
            })
    out = []
    for k in kernels.values():
        k.pop("_cum_cost")
        out.append(k)
    return sorted(out, key=lambda k: k["kernel"])


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(rounds: list[dict], kernel_history: list[dict]) -> str:
    lines = ["# Inner-loop ledger", "", DATA_GAPS_NOTE, ""]

    # (a) per-round summary table
    lines.append("## a. Round summary")
    headers = ["round", "start", "end", "wall-clock", "runs", "iters",
               "ok", "fail", "cost", "model(s)"]
    rows = []
    for rnd in rounds:
        rows.append([
            rnd["name"], rnd["start"] or "?", rnd["end"] or "?",
            _fmt_dur(rnd["wall_clock_s"]), rnd["num_runs"], rnd["total_iters"],
            rnd["ok_iters"], rnd["fail_iters"], _fmt_money(rnd["total_cost_usd"]),
            ", ".join(rnd["llm_models"]) or "?",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # (b) per-round per-run table
    lines.append("## b. Per-run detail")
    for rnd in rounds:
        lines.append(f"### {rnd['name']}" + (f" — {rnd['round_note']}" if rnd["round_note"] else ""))
        headers = ["kernel", "run_id", "iters (done/planned)", "seed", "baseline",
                   "best", "roofline", "best/roofline", "cost", "wall-clock", "note"]
        rows = []
        for r in rnd["runs"]:
            rows.append([
                r["kernel"], r["run_id"],
                f"{r['completed_iters']}/{r['planned_iters']}",
                _fmt_cycles(r["seed_cycles"]), _fmt_cycles(r["baseline_cycles"]),
                _fmt_cycles(r["best_cycles"]), _fmt_cycles(r["roofline_cycles"]),
                r["best_over_roofline"], _fmt_money(r["cost_usd"]),
                _fmt_dur(r["wall_clock_s"]) + (" (ongoing/aborted)" if r["ongoing_or_aborted"] else ""),
                r["note"].replace("|", "\\|")[:80],
            ])
        lines.append(_md_table(headers, rows))
        lines.append("")

    # (c) per-kernel history
    lines.append("## c. Per-kernel history (baseline -> round-by-round best)")
    headers = ["kernel", "baseline", "roofline"] + [rnd["name"] for rnd in rounds]
    rows = []
    for k in kernel_history:
        cell_by_round = {p["round"]: p for p in k["points"]}
        row = [k["kernel"], _fmt_cycles(k["baseline_cycles"]), _fmt_cycles(k["roofline_cycles"])]
        for rnd in rounds:
            p = cell_by_round.get(rnd["name"])
            if p is None:
                row.append("-")
            else:
                row.append(f"{_fmt_cycles(p['best_cycles'])} (cum {_fmt_money(p['cumulative_cost_usd'])})")
        rows.append(row)
    lines.append(_md_table(headers, rows))
    lines.append("")

    # (d) appendix: one row per iteration
    lines.append("## d. Appendix — every iteration")
    headers = ["run_id", "iter", "status", "cycles", "cost", "duration", "note"]
    rows = []
    for rnd in rounds:
        for r in rnd["runs"]:
            for it in r["iters"]:
                rows.append([
                    it["run_id"], it["iter"], it["status"] or it["kind"] or "?",
                    _fmt_cycles(it["mcycle"]), _fmt_money(it["cost_usd"]),
                    _fmt_dur(it["duration_s"]) if it["duration_s"] is not None else "?",
                    (it["note"] or "").replace("|", "\\|")[:100],
                ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds-file", default=str(DEFAULT_ROUNDS_FILE),
                    help="JSON describing which run_ids belong to which round")
    ap.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    ap.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    ap.add_argument("--db", default=None, help="override AETHER_DB path")
    args = ap.parse_args(argv)

    if args.db:
        db.DB_PATH = Path(args.db)

    rounds_raw = load_rounds(Path(args.rounds_file))
    rounds = [build_round(r) for r in rounds_raw]
    kernel_history = build_kernel_history(rounds)

    md = render_markdown(rounds, kernel_history)
    Path(args.md_out).write_text(md)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rounds_file": str(args.rounds_file),
        "rounds": rounds,
        "kernel_history": kernel_history,
    }
    Path(args.json_out).write_text(json.dumps(payload, indent=2))

    missing = [r["run_id"] for rnd in rounds for r in rnd["runs"] if not r["in_db"]]
    if missing:
        print(f"WARNING: {len(missing)} run_id(s) in rounds.json not found in db: {missing}",
              file=sys.stderr)
    if _KERNELS_IMPORT_ERROR:
        print(f"WARNING: loop/kernels.py failed to import ({_KERNELS_IMPORT_ERROR}); "
              f"roofline_cycles is unavailable for this run of ledger.py. "
              f"Re-run once kernels.py is stable again.", file=sys.stderr)

    print(f"wrote {args.md_out}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
