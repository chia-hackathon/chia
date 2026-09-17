"""journal.py — per-ITERATION drill-down for the inner loop (companion to
``ledger.py``, which only summarizes per-ROUND / per-RUN).

Reuses ``ledger.py``'s already-solved plumbing rather than reimplementing it:

  * ``ledger.load_rounds`` for reading ``out/loop/rounds.json``.
  * ``ledger._parse_ts`` / ``ledger._mtime`` for the same timestamp/duration
    conventions ledger.py documents (see its own "Known data gaps" #2).
  * ``db.py`` for the ``runs`` / ``iters`` tables (read-only).
  * ``kernels.KERNELS[name].roofline_cycles`` (read-only import, exactly like
    ledger.py — degrades to "roofline unknown" if kernels.py is mid-edit).

What this script adds on top of ledger.py: one row per (run_id, iter) — not
just one row per run — with a per-iteration "new best at the time" flag and
an "agent note" pulled from the on-disk ``agent_NN.txt`` / ``kernel_NN.h``
files, which ledger.py's appendix does not attempt.

Ordering convention (see task docstring below for why): ``iters`` has no
per-iteration timestamp column (ledger.py's own gap #2). Rather than lean on
kernel-file mtimes (fragile: collapses if a run directory is ever
touched/copied after the fact — see ledger.py's docstring), this script
sorts rows by ``(round start time from rounds.json, run's `runs.started`,
iter number)``. That is sufecient for correct ordering *within* a round and
*within* a run (iteration numbers are strictly increasing by construction)
without depending on filesystem metadata that can silently go stale.

Known data gaps (also written at the top of iterations.md):

  1. Same as ledger.py gap #1: ``runs.cost_usd`` / ``best_cycles`` are only
     rolled up by ``finish_run()``, so a run that died mid-flight has stale
     rollup columns. This script (like ledger.py) reads ``iters.cost_usd``
     and ``iters.fitness`` directly per row, never the ``runs`` rollup.
  2. No per-iteration timestamp (see "Ordering convention" above) — rows are
     ordered by round/run start time + iter number, not by a true wall-clock
     timestamp. This is coarser than ledger.py's per-iteration mtime-based
     duration estimate, but robust to stale mtimes; this script does not
     report per-iteration duration at all for that reason.
  3. "Agent note" is best-effort text-mined from ``agent_NN.txt`` (preferred)
     or the ``kernel_NN.h`` header comment (fallback) via a HYPOTHESIS-marker
     regex; it is not a structured DB column. Missing/unparseable files leave
     the field blank rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import ledger  # noqa: E402  (reuse load_rounds / _parse_ts / _mtime)

try:
    from kernels import KERNELS  # noqa: E402
    _KERNELS_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001
    KERNELS = {}
    _KERNELS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUNDS_FILE = REPO_ROOT / "out" / "loop" / "rounds.json"
DEFAULT_MD_OUT = REPO_ROOT / "out" / "loop" / "iterations.md"
DEFAULT_JSON_OUT = REPO_ROOT / "out" / "loop" / "iterations.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "out" / "loop"

DATA_GAPS_NOTE = """\
> **Data sources & limitations**
> - Companion to `ledger.py` (which reports per-round/per-run only): one row
>   per `(run_id, iter)`, joining `out/loop/rounds.json`, `loop/aether.db`
>   (`runs`+`iters`), and `loop/kernels.py` (`roofline_cycles`, read-only).
> - Cost and cycle counts are always read from `iters` directly (`cost_usd`,
>   `fitness`/`mcycle`), never from `runs`'s rollup columns — a run that died
>   mid-flight never got its `finish_run()` rollup (same convention as
>   `ledger.py`).
> - `iters` has no per-iteration timestamp. Rows are ordered by
>   `(round start time, run's runs.started, iter number)` rather than by
>   on-disk kernel-file mtime (ledger.py's approach) — coarser, but immune to
>   mtimes going stale if a run directory is ever touched/copied after the
>   fact. No per-iteration duration is reported here for that reason.
> - The "agent note" column is best-effort text mined from `agent_NN.txt`
>   (preferred) or the `kernel_NN.h` header comment (fallback) via a
>   HYPOTHESIS-marker regex — not a structured DB column. Blank means neither
>   file existed or neither contained a recognizable HYPOTHESIS marker.
> - Regenerate with: `python loop/journal.py --rounds-file out/loop/rounds.json`
"""

HYP_RE = re.compile(
    r"\*{0,2}HYPOTHESIS:?\*{0,2}\s*(.*?)"
    r"(?=\*{0,2}(?:CHANGE|EXPECTED):?\*{0,2}|\n\s*\n|\Z)",
    re.S | re.I,
)


def _clean_fragment(frag: str) -> str:
    """Strip leading `//` comment markers / bullets, collapse whitespace."""
    lines = []
    for line in frag.splitlines():
        line = re.sub(r"^\s*//\s?", "", line)
        line = re.sub(r"^\s*\*\s?", "", line)
        lines.append(line.strip())
    return " ".join(l for l in lines if l)


def _truncate(s: str, n: int = 200) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    cut = s.rfind(" ", 0, n)
    if cut < n * 0.6:  # no good boundary nearby; hard cut
        cut = n
    return s[:cut].rstrip() + "..."


def extract_hypothesis(text: str) -> str | None:
    m = HYP_RE.search(text)
    if not m:
        return None
    frag = _clean_fragment(m.group(1))
    return frag or None


def agent_note_for(run_id: str, it: int, runs_dir: Path) -> tuple[str | None, str | None]:
    """Returns (note_text, source) where source is 'agent_txt', 'kernel_header',
    or None. *it* must be >= 1 (agent/kernel files are 1-indexed optimization
    attempts; the seed (-1) and baseline (0) iterations have none)."""
    if it < 1:
        return None, None
    run_dir = runs_dir / run_id
    agent_path = run_dir / f"agent_{it:02d}.txt"
    if agent_path.exists():
        try:
            text = agent_path.read_text(errors="replace")
        except OSError:
            text = ""
        if text.strip():
            return _truncate(text), "agent_txt"

    kernel_path = run_dir / f"kernel_{it:02d}.h"
    if kernel_path.exists():
        try:
            text = kernel_path.read_text(errors="replace")
        except OSError:
            text = ""
        # Only look at the header comment block (first ~60 lines) — the rest
        # of the file is code, not the agent's note.
        header = "\n".join(text.splitlines()[:60])
        hyp = extract_hypothesis(header)
        if hyp:
            return _truncate(hyp), "kernel_header"

    return None, None


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------

def build_run_rows(run_entry: dict, round_name: str, round_start_iso: str | None,
                   runs_dir: Path) -> tuple[list[dict], bool]:
    """Returns (rows, in_db) for one rounds.json run entry."""
    run_id = run_entry["run_id"]
    kernel = run_entry["kernel"]
    run_row = db.query_one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    iter_rows = db.query("SELECT * FROM iters WHERE run_id=? ORDER BY iter", (run_id,))

    run_started_iso = run_row["started"] if run_row else None
    seed_row = next((i for i in iter_rows if i["iter"] == -1), None)
    baseline_row = next((i for i in iter_rows if i["iter"] == 0), None)
    baseline_cycles = baseline_row["mcycle"] if baseline_row else (
        run_row["baseline_cycles"] if run_row else None)
    if baseline_cycles is None and seed_row:
        baseline_cycles = seed_row["mcycle"]

    kdef = KERNELS.get(kernel)
    roofline = kdef.roofline_cycles if kdef else None

    rows: list[dict] = []
    best_so_far: int | None = None
    for it in iter_rows:
        cycles = it.get("mcycle")
        fitness = it.get("fitness")
        status = it.get("status")
        new_best = False
        if fitness is not None:
            if best_so_far is None or fitness < best_so_far:
                new_best = True
                best_so_far = fitness

        ratio_baseline = (cycles / baseline_cycles
                          if (cycles is not None and baseline_cycles) else None)
        ratio_roofline = (cycles / roofline
                          if (cycles is not None and roofline) else None)

        note, note_source = agent_note_for(run_id, it["iter"], runs_dir)

        rows.append({
            "round": round_name,
            "kernel": kernel,
            "run_id": run_id,
            "iter": it["iter"],
            "status": status,
            "cycles": cycles,
            "ratio_vs_baseline": ratio_baseline,
            "ratio_vs_roofline": ratio_roofline,
            "new_best": new_best,
            "cost_usd": it.get("cost_usd") or 0.0,
            "agent_note": note,
            "agent_note_source": note_source,
            "_round_start": round_start_iso,
            "_run_started": run_started_iso,
        })

    return rows, run_row is not None


def build_round(round_entry: dict, runs_dir: Path) -> dict:
    round_name = round_entry["name"]
    round_start_iso = round_entry.get("start")
    all_rows: list[dict] = []
    missing_run_ids: list[str] = []
    for r in round_entry["runs"]:
        rows, in_db = build_run_rows(r, round_name, round_start_iso, runs_dir)
        if not in_db:
            missing_run_ids.append(r["run_id"])
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (
        r["_round_start"] or "",
        r["_run_started"] or "",
        r["iter"],
    ))

    n = len(all_rows)
    ok_n = sum(1 for r in all_rows if r["status"] == db.ST_OK)
    total_cost = round(sum(r["cost_usd"] for r in all_rows), 6)

    return {
        "name": round_name,
        "round_note": round_entry.get("note", ""),
        "rows": all_rows,
        "missing_run_ids": missing_run_ids,
        "summary": {
            "iterations": n,
            "ok": ok_n,
            "pass_rate": round(ok_n / n, 4) if n else None,
            "total_cost_usd": total_cost,
        },
    }


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def _fmt_cycles(x) -> str:
    return "?" if x is None else f"{int(x):,}"


def _fmt_money(x) -> str:
    return "?" if x is None else f"${x:,.4f}"


def _fmt_ratio(x) -> str:
    return "n/a" if x is None else f"{x:.3f}x"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(rounds: list[dict], overall: dict) -> str:
    lines = ["# Inner-loop iteration journal", "", DATA_GAPS_NOTE, ""]

    headers = ["kernel", "run_id", "iter", "status", "cycles", "vs baseline",
               "vs roofline", "new best?", "cost", "agent note"]
    for rnd in rounds:
        lines.append(f"## {rnd['name']}" + (f" — {rnd['round_note']}" if rnd["round_note"] else ""))
        if rnd["missing_run_ids"]:
            lines.append(f"> **WARNING:** run_id(s) not found in db: {rnd['missing_run_ids']}")
            lines.append("")
        rows = []
        for r in rnd["rows"]:
            rows.append([
                r["kernel"], r["run_id"], r["iter"], r["status"] or "?",
                _fmt_cycles(r["cycles"]), _fmt_ratio(r["ratio_vs_baseline"]),
                _fmt_ratio(r["ratio_vs_roofline"]), "★" if r["new_best"] else "",
                _fmt_money(r["cost_usd"]),
                (r["agent_note"] or "").replace("|", "\\|").replace("\n", " "),
            ])
        lines.append(_md_table(headers, rows))
        s = rnd["summary"]
        pr = f"{s['ok']}/{s['iterations']}" + (f" ({s['pass_rate']:.1%})" if s["pass_rate"] is not None else "")
        lines.append("")
        lines.append(f"**{rnd['name']} summary** — iterations: {s['iterations']}, "
                     f"pass rate: {pr}, total cost: {_fmt_money(s['total_cost_usd'])}")
        lines.append("")

    lines.append("## Overall summary")
    lines.append(f"- total iterations: {overall['iterations']}")
    lines.append(f"- overall pass rate: {overall['ok']}/{overall['iterations']}"
                 + (f" ({overall['pass_rate']:.1%})" if overall["pass_rate"] is not None else ""))
    lines.append(f"- total cost across all rounds: {_fmt_money(overall['total_cost_usd'])}")
    if overall.get("ledger_total_cost_usd") is not None:
        match = "MATCHES" if overall["cost_reconciles"] else "DOES NOT MATCH"
        lines.append(f"- cross-check vs ledger.json's sum of per-round `total_cost_usd`: "
                     f"{_fmt_money(overall['ledger_total_cost_usd'])} — {match}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds-file", default=str(DEFAULT_ROUNDS_FILE))
    ap.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    ap.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR),
                    help="directory containing out/loop/<run_id>/ subdirs")
    ap.add_argument("--db", default=None, help="override AETHER_DB path")
    ap.add_argument("--ledger-json", default=str(REPO_ROOT / "out" / "loop" / "ledger.json"),
                    help="path to ledger.json for the cost cross-check (optional)")
    args = ap.parse_args(argv)

    if args.db:
        db.DB_PATH = Path(args.db)

    runs_dir = Path(args.runs_dir)
    rounds_raw = ledger.load_rounds(Path(args.rounds_file))
    rounds = [build_round(r, runs_dir) for r in rounds_raw]

    total_iters = sum(r["summary"]["iterations"] for r in rounds)
    total_ok = sum(r["summary"]["ok"] for r in rounds)
    total_cost = round(sum(r["summary"]["total_cost_usd"] for r in rounds), 6)
    overall = {
        "iterations": total_iters,
        "ok": total_ok,
        "pass_rate": round(total_ok / total_iters, 4) if total_iters else None,
        "total_cost_usd": total_cost,
    }

    ledger_total = None
    try:
        ledger_payload = json.loads(Path(args.ledger_json).read_text())
        ledger_total = round(sum(r["total_cost_usd"] for r in ledger_payload["rounds"]), 6)
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    if ledger_total is not None:
        overall["ledger_total_cost_usd"] = ledger_total
        overall["cost_reconciles"] = abs(ledger_total - total_cost) < 0.01

    md = render_markdown(rounds, overall)
    Path(args.md_out).write_text(md)

    # Drop the internal sort keys before writing JSON.
    json_rounds = []
    for rnd in rounds:
        rnd = dict(rnd)
        rnd["rows"] = [{k: v for k, v in row.items() if not k.startswith("_")}
                       for row in rnd["rows"]]
        json_rounds.append(rnd)

    payload = {
        "rounds_file": str(args.rounds_file),
        "rounds": json_rounds,
        "overall": overall,
    }
    Path(args.json_out).write_text(json.dumps(payload, indent=2))

    all_missing = [rid for rnd in rounds for rid in rnd["missing_run_ids"]]
    if all_missing:
        print(f"WARNING: {len(all_missing)} run_id(s) in rounds.json not found in db: {all_missing}",
              file=sys.stderr)
    if _KERNELS_IMPORT_ERROR:
        print(f"WARNING: loop/kernels.py failed to import ({_KERNELS_IMPORT_ERROR}); "
              f"roofline ratios are unavailable for this run.", file=sys.stderr)

    print(f"wrote {args.md_out}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
