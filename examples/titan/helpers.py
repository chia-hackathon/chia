"""Result persistence, verdict classification, and debug-feedback shaping.

The interesting part of this module is :func:`classify_run`.  In
riscv_extensions the verdict came from Spike lockstep, so a directed test only
had to execute without trapping.  Titan's Stage 1 tests carry their own judge
and print it, so classification is a matter of reading what the program said --
and, importantly, of telling three outcomes apart rather than two:

    pass          the IME path and the RVV reference path agree
    skip          the DUT legally declined the requested tile geometry
    mismatch      they disagree, at a named row and column
    bad_geometry  the DUT answered with a geometry WARL does not permit

Collapsing "skip" into "fail" would be the single most expensive mistake
available here.  The set of supported lambda values is implementation-defined:
the architecture requires only that it be nonempty, and expects software to
discover it by writing a value and reading back what stuck.  A loop that
counted every unsupported geometry as a failure would send the agent chasing
a bug that does not exist, for as many iterations as it had patience for.

The mistake that actually happened is the mirror image, and cost five
iterations: *every* geometry disagreement was called a skip, so a DUT that
answered LAMBDA=4 to a request for LAMBDA=2 was granted amnesty fifteen times
per run and told in writing that it was "not counted against you".  The spec
(v0.9.0, "Writing vtype.lambda") is narrow about what a legal answer is:

    the implementation shall select the largest supported nonzero lambda
    value that is less than or equal to the requested value; if no supported
    value is less than or equal to the request, it shall select the smallest
    supported nonzero lambda value.

So selecting a *larger* lambda is legal in exactly one case -- when the DUT
supports nothing at or below the request -- and that is a property of the run
as a whole, not of one program.  :func:`classify_run` therefore calls every
round-up ``bad_geometry``, and :func:`reconcile_geometry` afterwards downgrades
the ones the whole-run evidence excuses.  Run it over the outcomes before
reporting them; ``_run_directed`` does.
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from constants import (AGENT_LOG_DIR, BUILD_ERROR_MAX_CHARS,
                       CHIPYARD_PATH,
                       BUILD_ERROR_MAX_FRAMES, COMMIT_LOG_TAIL_LINES,
                       DUMP_TAIL_LINES, MAX_DIFF_LINES, MAX_DUMP_ROWS,
                       MAX_EVIDENCE_CHARS, MAX_EVIDENCE_TESTS,
                       MAX_OUTPUT_CHARS, NOTES_MAX_CHARS)

#: Emitted by every ime_tests.py program.  See its verdict contract.
_PASS_RE = re.compile(r"^TITAN PASS\b(?P<geom>.*)$", re.M)
_SKIP_RE = re.compile(r"^TITAN SKIP lambda=(?P<got>-?\d+)\s+"
                      r"\(requested (?P<want>\d+)\)(?P<geom>.*)$", re.M)
_FAIL_RE = re.compile(r"^TITAN FAIL row=(?P<row>\d+) col=(?P<col>\d+)"
                      r"(?P<geom>.*)$", re.M)

#: The geometry every verdict line carries (TileGeometry.describe()).
_GEOM_RE = re.compile(r"VLEN=(?P<vlen>\d+) SEW=(?P<sew>\d+) "
                      r"LAMBDA=(?P<lam>\d+)")

#: The evidence a failing program prints ahead of its verdict.
_DIFF_RE = re.compile(r"^TITAN DIFF .*$", re.M)
_CDUMP_RE = re.compile(r"^TITAN (?:CDUMP|CREF|CLAYOUT) .*$", re.M)


def _permissible(vlen: Optional[int], sew: Optional[int]) -> Optional[list]:
    """The architecturally permissible nonzero LAMBDA set, or None.

    Imported lazily and defensively: helpers is also imported by tooling that
    has no reason to pull in the spec JSON, and a classifier that raises is
    strictly worse than one that falls back to the ordering test alone.
    """
    if vlen is None or sew is None:
        return None
    try:
        import rvv_ref
        return rvv_ref.permissible_lambdas(vlen, sew)
    except Exception:
        return None


@dataclass
class Outcome:
    """What one directed program did.

    ``kind`` is one of ``pass``, ``skip``, ``mismatch``, ``bad_geometry``,
    ``trap``, ``timeout``, ``silent``.  ``skip`` is not a failure and must not
    be counted as one; ``bad_geometry`` is one, and must not be mistaken for
    a skip just because the program exited down the same path.
    """

    kind: str
    detail: str = ""
    row: Optional[int] = None
    col: Optional[int] = None
    #: Set on every outcome whose verdict line carried a geometry.
    vlen: Optional[int] = None
    sew: Optional[int] = None
    requested_lambda: Optional[int] = None
    #: Set on skip / bad_geometry: the lambda the DUT actually left in vtype.
    selected_lambda: Optional[int] = None
    #: Verbatim TITAN DIFF / TITAN CDUMP lines the failing program printed.
    diffs: Tuple[str, ...] = ()
    dump: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.kind == "pass"

    @property
    def failed(self) -> bool:
        """Skips and passes are both "not a failure"; everything else is."""
        return self.kind not in ("pass", "skip")


def classify_run(run) -> Outcome:
    """Read a simulator RunResult and say what the program decided.

    The printed marker wins over the exit status.  The exit status is a
    redundant second channel, and how a non-zero return from ``main`` under
    htif_nano surfaces in the harness log is a detail nobody has confirmed on
    real hardware yet -- so it is used only when the program printed nothing,
    which itself means something went wrong before the verdict.
    """
    text = (run.log or "") + "\n" + (getattr(run, "out", "") or "")

    fail = _FAIL_RE.search(text)
    if fail:
        row, col = int(fail.group("row")), int(fail.group("col"))
        vlen, sew, lam = _geometry(fail.group("geom"))
        return Outcome("mismatch", f"C[{row},{col}] differs between the IME "
                                   f"path and the RVV reference"
                                   f"{fail.group('geom')}", row, col,
                       vlen=vlen, sew=sew, requested_lambda=lam,
                       diffs=tuple(_DIFF_RE.findall(text))[:MAX_DIFF_LINES],
                       dump=tuple(_CDUMP_RE.findall(text))[:2 * MAX_DUMP_ROWS])
    skip = _SKIP_RE.search(text)
    if skip:
        got, want = int(skip.group("got")), int(skip.group("want"))
        vlen, sew, _ = _geometry(skip.group("geom"))
        allowed = _permissible(vlen, sew)
        common = (f"the DUT selected lambda={got} for a requested "
                  f"lambda={want}{skip.group('geom')}")
        if got <= 0:
            return Outcome("bad_geometry",
                           f"{common}. Zero means vtype.lambda holds no "
                           f"selected lambda at all: the write left the unit "
                           f"unconfigured for matrix work rather than "
                           f"selecting a supported geometry.",
                           vlen=vlen, sew=sew, requested_lambda=want,
                           selected_lambda=got)
        if allowed is not None and got not in allowed:
            return Outcome("bad_geometry",
                           f"{common}. lambda={got} is not architecturally "
                           f"permissible at this VLEN and SEW (permissible: "
                           f"{allowed}) -- EMUL_C = VLEN/(SEW*lambda**2) must "
                           f"be an integer in {{1,2,4,8,16}}, so no "
                           f"implementation may select it.",
                           vlen=vlen, sew=sew, requested_lambda=want,
                           selected_lambda=got)
        if got > want:
            # WARL clamps down.  Rounding up is legal only if the DUT
            # supports nothing at or below the request, which one program
            # cannot show; reconcile_geometry settles that across the run.
            return Outcome("bad_geometry",
                           f"{common}. WARL selects the largest supported "
                           f"lambda <= the request, so answering with a "
                           f"larger one is legal only if this DUT supports "
                           f"no lambda at or below {want} for this "
                           f"(VLEN, SEW).",
                           vlen=vlen, sew=sew, requested_lambda=want,
                           selected_lambda=got)
        return Outcome("skip", f"{common}; the DUT clamped down to a "
                               f"supported geometry, which WARL allows",
                       vlen=vlen, sew=sew, requested_lambda=want,
                       selected_lambda=got)
    passed = _PASS_RE.search(text)
    if passed:
        vlen, sew, lam = _geometry(passed.group("geom"))
        return Outcome("pass", vlen=vlen, sew=sew, requested_lambda=lam)

    # Nothing printed.  Either the program never reached its verdict or the
    # simulator did not finish.
    rc = getattr(run, "returncode", None)
    if rc is None or (isinstance(rc, int) and rc < 0):
        return Outcome("timeout", "simulator did not terminate")
    if not getattr(run, "success", False):
        return Outcome("trap", f"simulator exited {rc} with no verdict line -- "
                               f"most likely an illegal-instruction trap "
                               f"before the program printed")
    return Outcome("silent", "simulator exited cleanly but printed no TITAN "
                             "verdict line; the test did not run to completion")


def _geometry(text: str):
    """(VLEN, SEW, LAMBDA) out of a verdict line's geometry suffix."""
    m = _GEOM_RE.search(text or "")
    if not m:
        return None, None, None
    return int(m.group("vlen")), int(m.group("sew")), int(m.group("lam"))


def reconcile_geometry(
        outcomes: Sequence[Tuple[str, Outcome]]) -> List[Tuple[str, Outcome]]:
    """Excuse the round-ups the spec's one exception actually covers.

    A DUT may answer a request with a *larger* lambda only when it supports
    nothing at or below what was asked.  Whether that holds is visible only
    across the whole run: every lambda the DUT retained (a program that got
    past its read-back check) and every lambda it selected is, by
    construction, a lambda it supports.  If any of those is <= the request,
    the round-up had a legal alternative and is a WARL violation; if none is,
    the DUT has one supported geometry at this (VLEN, SEW) and the round-up
    stands as an ordinary skip.

    Deliberately conservative in the direction that costs iterations rather
    than the direction that hides bugs: with no evidence either way the
    outcome is downgraded to skip.
    """
    supported: Dict[Tuple[int, int], set] = {}
    for _, o in outcomes:
        if o.vlen is None or o.sew is None:
            continue
        key = (o.vlen, o.sew)
        # A retained lambda: the program only ran because vtype kept it.
        if o.kind in ("pass", "mismatch") and o.requested_lambda:
            supported.setdefault(key, set()).add(o.requested_lambda)
        # A selected lambda is supported by definition, whoever asked.
        if o.selected_lambda:
            supported.setdefault(key, set()).add(o.selected_lambda)

    out: List[Tuple[str, Outcome]] = []
    for name, o in outcomes:
        if (o.kind == "bad_geometry" and o.selected_lambda
                and o.requested_lambda
                and o.selected_lambda > o.requested_lambda):
            key = (o.vlen, o.sew)
            lower = sorted(l for l in supported.get(key, ())
                           if l <= o.requested_lambda)
            if not lower:
                o = replace(o, kind="skip", detail=(
                    f"the DUT selected lambda={o.selected_lambda} for a "
                    f"requested lambda={o.requested_lambda}; nothing this run "
                    f"shows it supporting any lambda at or below the request "
                    f"at VLEN={o.vlen} SEW={o.sew}, so rounding up is the "
                    f"fallback WARL permits"))
            else:
                o = replace(o, detail=(
                    o.detail + f" It does: this run shows lambda in {lower} "
                    f"working at VLEN={o.vlen} SEW={o.sew}, so the round-up "
                    f"had a legal alternative and is a WARL violation."))
        out.append((name, o))
    return out


def geometry_support(
        outcomes: Sequence[Tuple[str, Outcome]]) -> List[str]:
    """One line per (VLEN, SEW): which lambdas the DUT actually supports.

    A run of 27 programs in which 15 report SKIP is not a failing run, but it
    is not a tested one either, and the totals alone hide that: they say
    "12 mismatch, 15 skip" as though the same amount of work had been done.
    A DUT that supports one lambda per SEW declines every other geometry
    *legally*, and every legal decline is a geometry the loop is not
    exercising -- so the fact is stated where the agent reads it rather than
    left to be inferred from a count.
    """
    seen: Dict[Tuple[int, int], Dict[str, set]] = {}
    for _, o in outcomes:
        if o.vlen is None or o.sew is None:
            continue
        rec = seen.setdefault((o.vlen, o.sew),
                              {"works": set(), "declined": set()})
        if o.kind in ("pass", "mismatch") and o.requested_lambda:
            rec["works"].add(o.requested_lambda)
        if o.selected_lambda:
            rec["works"].add(o.selected_lambda)
        if o.kind in ("skip", "bad_geometry") and o.requested_lambda:
            rec["declined"].add(o.requested_lambda)

    lines = []
    for (vlen, sew), rec in sorted(seen.items()):
        allowed = _permissible(vlen, sew)
        declined = sorted(rec["declined"] - rec["works"])
        note = f"  VLEN={vlen} SEW={sew}: supports lambda in " \
               f"{sorted(rec['works'])}"
        if allowed is not None:
            note += f" of the {allowed} this VLEN and SEW permit"
        if declined:
            note += f"; declined {declined}"
        lines.append(note)
    return lines


def summarize(outcomes: Sequence[Tuple[str, Outcome]]) -> Dict[str, object]:
    """Counts plus the failing names, for the loop's status file and log."""
    by_kind: Dict[str, List[str]] = {}
    for name, outcome in outcomes:
        by_kind.setdefault(outcome.kind, []).append(name)
    return {
        "total": len(outcomes),
        "counts": {kind: len(names) for kind, names in sorted(by_kind.items())},
        "failing": [name for name, o in outcomes if o.failed],
    }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

class Dumper:
    """Write every node result to disk, timestamped at write time.

    Timestamping per write rather than per run keeps files from concurrent
    pipelines sharing one out/ directory in chronological order.
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.out_dir / f"{stamp}_{name}"

    def text(self, name: str, content: str) -> None:
        self._path(name).write_text(content, encoding="utf-8")

    def bytes(self, name: str, content: bytes) -> None:
        self._path(name).write_bytes(content)

    def json(self, name: str, obj: object) -> None:
        self._path(name).write_text(json.dumps(obj, indent=2, default=str),
                                    encoding="utf-8")


def dump_llm(dump: Dumper, name: str, cli) -> None:
    """Persist an LLM turn: verdict, usage if the backend reports it, output."""
    parts = [f"success={getattr(cli, 'success', None)}"]
    usage = getattr(cli, "usage", None)
    if usage:
        parts.append("usage=" + json.dumps(usage, default=str))
    parts.append(str(getattr(cli, "result", "")))
    stream = getattr(cli, "stream_result", None)
    if stream:
        parts.append("## Stream transcript\n" + str(stream))
    dump.text(f"{name}.md", "\n\n".join(parts))


# ---------------------------------------------------------------------------
# debug feedback
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int, keep: str = "tail") -> str:
    if len(text) <= max_chars:
        return text
    elided = f"\n[... {len(text) - max_chars} characters elided ...]\n"
    return (elided + text[-max_chars:]) if keep == "tail" \
        else (text[:max_chars] + elided)


def _tail_lines(text: str, n: int) -> str:
    return "\n".join((text or "").splitlines()[-n:])


#: Lines that identify a Chisel/Scala failure.  sbt prefixes real errors with
#: `[error]`; firrtl/CIRCT and the Scala runtime announce themselves with an
#: exception class name; `at ...` lines are the frames under it.
_ERR_RE = re.compile(r"^\s*(\[error\]|\[warn\]\s*\[error\]|.*Exception[:\s]"
                     r"|.*\bError\b.*:|Caused by:)")
_FRAME_RE = re.compile(r"^\s+at\s+\S")


def _extract_build_errors(text: str) -> str:
    """The compiler's own diagnosis, and nothing else.

    An elaboration failure used to reach the agent as up to 100,000 characters
    of stderr on the theory that the answer was somewhere in it.  It always
    was -- in the `[error]` lines, which are about forty of the four thousand.
    Everything else is sbt resolving dependencies, and every character of it
    was replayed on every subsequent turn by ``resume_session``.

    Keeps, in order: every error-ish line, and the first
    ``BUILD_ERROR_MAX_FRAMES`` stack frames that follow one.  Falls back to
    the tail if the output contains nothing that looks like an error at all,
    because a build that failed with no error line is itself a fact worth
    seeing.
    """
    kept: List[str] = []
    frames = 0
    for line in (text or "").splitlines():
        if _ERR_RE.match(line):
            kept.append(line.rstrip())
            frames = 0
        elif _FRAME_RE.match(line) and kept:
            if frames < BUILD_ERROR_MAX_FRAMES:
                kept.append(line.rstrip())
                frames += 1
    if not kept:
        return _truncate(text or "", BUILD_ERROR_MAX_CHARS)
    return _truncate("\n".join(kept), BUILD_ERROR_MAX_CHARS)


def format_build_failure(artifact, attempt: int,
                         log_path: Optional[str] = None) -> str:
    """Elaboration failed: the error lines, and where the whole log is.

    A pointer, not a dump.  The complete stdout and stderr are written into
    the tree the agent edits (see ``AGENT_LOG_DIR``), so anything this
    extraction drops is one ``grep`` away rather than gone.
    """
    stdout = getattr(artifact, "stdout", "") or ""
    stderr = getattr(artifact, "stderr", "") or ""
    errors = _extract_build_errors(stderr + "\n" + stdout)
    out = [f"# Attempt {attempt}: the Chisel elaboration failed\n",
           "## the compiler's error lines\n```\n" + errors + "\n```\n"]
    out.append(_where_to_look(log_path, "build.stdout.txt / build.stderr.txt"))
    return "\n".join(out)


def _where_to_look(log_path: Optional[str], what: str = "") -> str:
    """The pointer that replaces the pasted logs.

    Every iteration's artifacts are copied into the chipyard container, in
    the tree the agent already has a shell in, so "read the log" is a thing
    it can actually do.  Saying so in the message matters as much as copying
    the files: an agent that is not told the directory exists will not look.
    """
    path = log_path or AGENT_LOG_DIR
    detail = f" ({what})" if what else ""
    return (f"## the full logs are on disk, read them yourself\n"
            f"`{path}`{detail} -- on the machine your bash tool runs on. "
            f"It holds the complete build output and the complete simulator "
            f"log for every directed program, not a tail. Grep it before you "
            f"guess; nothing in this message is a summary you cannot check.\n")


def _evidence(outcome: Outcome) -> str:
    """The values a failing program printed, not just where it failed.

    `TITAN FAIL row=3 col=5` localises a fault; it does not characterise one.
    Zeros where the reference has data mean the accumulator never got written;
    the right values in the wrong places mean the tile index is wrong; one bad
    row out of eight means one register of the C group is. All three look
    identical from the coordinate alone, which is why the tile is dumped.
    """
    if not outcome.diffs and not outcome.dump:
        return ""
    parts = []
    if outcome.diffs:
        parts.append("### first differing elements (exp = RVV reference, "
                     "got = IME path)\n```\n"
                     + "\n".join(outcome.diffs[:MAX_DIFF_LINES]) + "\n```\n")
    if outcome.dump:
        parts.append("### the C tile, physical M x M, CDUMP = IME path, "
                     "CREF = reference\n```\n"
                     + "\n".join(outcome.dump[:2 * MAX_DUMP_ROWS])
                     + "\n```\n")
    return "\n".join(parts)


def _one_line(name: str, o: Outcome) -> str:
    """One failing test, one line: what it ran and what went wrong.

    Everything an agent needs to see the *pattern* across twelve failures --
    which is the diagnosis -- and nothing it needs to scroll past to see it.
    """
    geom = ""
    if o.vlen is not None:
        geom = f"VLEN={o.vlen} SEW={o.sew} LAMBDA={o.requested_lambda}"
    if o.kind == "bad_geometry":
        detail = (f"requested lambda={o.requested_lambda}, DUT selected "
                  f"lambda={o.selected_lambda}")
    elif o.row is not None:
        detail = f"first differs at C[{o.row},{o.col}]"
        # The first DIFF line carries the values; the coordinate alone was
        # what ten iterations of r4/r5 had, and it was not enough.
        if o.diffs:
            first = o.diffs[0].replace("TITAN DIFF ", "").strip()
            detail += f"  {first}"
    else:
        detail = (o.detail or "").split(".")[0][:80]
    return f"  {name:32s} {o.kind:12s} {geom}  {detail}"


def format_directed_failure(attempt: int,
                            outcomes: Sequence[Tuple[str, Outcome]],
                            logs: Dict[str, str],
                            log_path: Optional[str] = None) -> str:
    """A directed test disagreed with the RVV reference.

    Shrunk, in r5, from "everything we have" to "the shape of the failure,
    the values behind the first few, and where to read the rest".  The old
    message pasted a 300-line simulator tail per failure and up to 100kB of
    build output; with ``resume_session`` on, all of it was replayed every
    turn (145M cached tokens over five calls) and none of it contained a
    single wrong *value*.  What replaces it is bounded three ways -- one line
    per failure, an evidence block capped by both MAX_EVIDENCE_TESTS and
    MAX_EVIDENCE_CHARS, and a path -- so the message stays roughly constant
    however badly the run goes.
    """
    failing = [(name, o) for name, o in outcomes if o.failed]
    counts: Dict[str, int] = {}
    for _, o in outcomes:
        counts[o.kind] = counts.get(o.kind, 0) + 1
    totals = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    lines = [f"# Attempt {attempt}: {len(failing)} of {len(outcomes)} "
             f"directed tests failed\n",
             f"totals: {totals}\n"]

    if counts.get("skip"):
        lines.append(
            "`skip` is a legal decline (WARL clamped lambda DOWN to a value "
            "the DUT supports) and is not counted against you -- but a "
            "declined geometry is an untested one, so widening the supported "
            "set is part of the work.")
    if counts.get("bad_geometry"):
        lines.append(
            "`bad_geometry` is a failure: vtype.lambda answered with a value "
            "WARL does not permit (larger than the request while smaller "
            "supported values exist, or one no implementation may select at "
            "that VLEN/SEW). Look at the vsetvl lambda selection before you "
            "look at any datapath.")

    lines.append("\n## every failing test, one line each\n```")
    lines += [_one_line(n, o) for n, o in failing]
    lines.append("```\n")

    support = geometry_support(outcomes)
    if support:
        lines.append("## lambda the DUT actually supports, per (VLEN, SEW)\n"
                     "```\n" + "\n".join(support) + "\n```\n")

    # Evidence: values, not coordinates -- but only until the budget is
    # spent.  Zeros where the reference has data mean the accumulator never
    # got written; right values in the wrong places mean the tile index is
    # wrong; one bad row out of eight means one register of the C group is.
    # All three are invisible in a coordinate and obvious in a dump.
    budget = MAX_EVIDENCE_CHARS
    shown = 0
    for name, outcome in failing[:MAX_EVIDENCE_TESTS]:
        block = _evidence(outcome)
        if not block:
            continue
        if shown and len(block) > budget:
            break
        lines.append(f"## evidence: {name}\n\n"
                     f"{outcome.kind}: {outcome.detail}\n")
        lines.append(block)
        if COMMIT_LOG_TAIL_LINES:
            lines.append("### simulator log tail\n```\n"
                         + _tail_lines(logs.get(name, ""),
                                       COMMIT_LOG_TAIL_LINES) + "\n```\n")
        budget -= len(block)
        shown += 1
    remaining = len(failing) - shown
    if shown == 0:
        lines.append("(No TITAN DIFF / CDUMP evidence in these logs. If the "
                     "programs are not printing it, that is a harness "
                     "problem, not yours -- read the logs at the path below "
                     "and say so.)\n")
    elif remaining > 0:
        lines.append(f"[{remaining} further failure(s) not quoted. Their full "
                     f"logs, evidence included, are at the path below; the "
                     f"one-line table above is complete.]\n")

    lines.append(_where_to_look(log_path, "one sim_<test>.log per program"))
    lines.append("Remember: the reference path is plain RVV 1.0 and is not "
                 "under test. If the two disagree, the IME path is wrong.\n")
    return "\n".join(lines)


def diff_stat(diff: str) -> str:
    """``git diff --stat``, computed from a diff we already have in hand.

    A fresh session opens on a tree it did not edit, so it has to be told
    what is already in it.  Deriving this from ``collect_diff``'s output
    rather than dispatching a second remote ``git diff --stat`` keeps the
    chipyard resource free and cannot disagree with the diff the run
    archived.
    """
    files: List[Tuple[str, int, int]] = []
    path, add, rem = None, 0, 0
    for line in (diff or "").splitlines():
        if line.startswith("diff --git "):
            if path:
                files.append((path, add, rem))
            parts = line.split()
            path, add, rem = parts[-1][2:], 0, 0
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+"):
            add += 1
        elif line.startswith("-"):
            rem += 1
    if path:
        files.append((path, add, rem))
    if not files:
        return "  (the working tree is clean: nothing has been edited yet)"
    out = [f"  {p:70s} +{a:<5d} -{r}" for p, a, r in files]
    out.append(f"  {len(files)} file(s), +{sum(a for _, a, _ in files)} "
               f"-{sum(r for _, _, r in files)}")
    return "\n".join(out)


def format_regression_block(best_passed: int, best_attempt: int,
                            now_passed: int, total: int,
                            best_path: str) -> str:
    """Open the turn with the fact that the tree used to be better.

    A fresh session cannot know that iteration 3 scored 26/27 and iteration 4
    scores 0/27: it sees one status file and a diffstat, and it will happily
    spend the turn debugging instrumentation somebody left in the tree.  So
    the numbers, and the diff that produced the better ones, go first.

    Deliberately not an auto-revert.  A regression is sometimes a half-landed
    change worth finishing, and the loop is in no position to tell which --
    but reverting has to be one command, not an archaeology exercise.
    """
    return (
        f"# !! REGRESSION !!\n\n"
        f"Your tree scored **{now_passed}/{total}** on the run just "
        f"completed. The best this run has scored is **{best_passed}/{total}**"
        f", at iteration {best_attempt} -- "
        f"{best_passed - now_passed} test(s) better than what you have now.\n\n"
        f"The diff that scored {best_passed}/{total} is saved at "
        f"`{best_path}`, readable from your bash tool. Before anything else, "
        f"compare your tree against it:\n\n"
        f"```\ncd {CHIPYARD_PATH} && git diff | diff - {best_path}\n```\n\n"
        f"and account for every difference. Diagnostic printfs, assertion "
        f"scaffolding and half-applied edits are the usual cause: the loop "
        f"builds and grades the tree **exactly as you leave it**, so anything "
        f"you left in for your own benefit was graded. If you cannot explain "
        f"why the current tree should be better, restore the saved one: "
        f"`cd {CHIPYARD_PATH} && git checkout -- . && git apply "
        f"{best_path}` -- then work forward from there.\n\n"
    )


def format_iteration_message(attempt: int, max_iters: int, body: str,
                             log_path: str, diffstat: str,
                             notes: str) -> str:
    """The whole of what one iteration hands the agent.

    Sessions are fresh by default from r5 on (``TITAN_RESUME_SESSION=0``), so
    this message is the agent's entire memory of the run.  That is the
    trade: replaying a 145M-token transcript bought continuity nobody was
    reading, and the same continuity fits in a notes file, a diffstat and a
    directory of logs -- all three of which the agent can re-read at will,
    which a transcript it has to be re-sent cannot claim.
    """
    return (
        f"# Iteration {attempt} of {max_iters}\n\n"
        f"You are starting a **fresh session**: you have no memory of "
        f"previous iterations. Everything you know about this run is below, "
        f"in your notes, and in the files on disk. Read `read_knowledge` and "
        f"`read_status` before you edit anything, and write what you learn "
        f"and what you tried back into the notes before this turn ends -- "
        f"the next iteration is a different session that will have only "
        f"them.\n\n"
        f"## what is already in the working tree\n```\n{diffstat}\n```\n\n"
        f"## your notes so far (`read_knowledge`)\n```\n"
        f"{_truncate(notes or '(empty -- nothing recorded yet)', NOTES_MAX_CHARS)}\n"
        f"```\n\n"
        f"{body}\n"
        f"## logs for this run\n"
        f"`{log_path}` on the machine your bash tool runs on: one directory "
        f"per iteration, each holding the build output, the full simulator "
        f"log of every directed program, the status file and the summary "
        f"json.\n")


#: Budget for the evidence block in a regression message.  The message opens
#: an iteration and competes with the diffstat, the notes and the directed
#: results for the agent's attention; three failing tests quoted properly beat
#: eight quoted in fragments, and 841 quoted at all is not a message.
REGRESSION_EVIDENCE_CHARS = 3000


def format_regression_failure(attempt: int,
                              failing: Sequence[Tuple[str, str, str]],
                              log_path: str = "") -> str:
    """riscv-vector-tests regressed: baseline RVV behaviour was broken.

    ``failing`` is ``(name, reason, log)``; the log is cospike's abort window.
    r8 shipped only the names, so an agent told "841 tests fail" had no way to
    see that all 841 died at the same instruction -- which is exactly the
    shape that distinguishes a broken harness from a broken design, and the
    one thing the message must never hide.
    """
    lines = [f"# Attempt {attempt}: {len(failing)} RVV regression test(s) "
             f"now fail\n",
             "These are Saturn's own riscv-vector-tests, judged by lockstep "
             "cosimulation against Spike. They passed before your changes. "
             "Whatever you did to add the matrix path has broken plain RVV "
             "behaviour -- that is a regression, not a trade-off, and it must "
             "be fixed rather than worked around.\n"]
    reasons = [r for _, r, _ in failing]
    if len(set(reasons)) == 1 and len(failing) > 1:
        lines.append(f"**Every one of them fails the same way**, which usually "
                     f"means one shared cause, not {len(failing)} bugs:\n"
                     f"`{reasons[0][:300]}`\n")
    budget = REGRESSION_EVIDENCE_CHARS
    for name, reason, log in failing:
        if budget <= 0:
            break
        body = "\n".join((log or "").splitlines()[:30]) or "(no log captured)"
        block = f"## {name}\n{reason}\n```\n{body[:budget]}\n```\n"
        lines.append(block)
        budget -= len(block)
    if len(failing) > 1:
        lines.append(f"({len(failing)} failing in total; the rest are listed "
                     f"in `failures.txt`.)\n")
    if log_path:
        lines.append(f"Full cospike abort windows for the first failures: "
                     f"`{log_path}` (read them with your bash tool).\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# instruction scope, and the "extend a converged design" opening message
# ---------------------------------------------------------------------------

def instruction_scope() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(already implemented, newly in scope), read from ``constants``.

    Round two adds an instruction to a design that already converged, so
    every prompt that opens on a seeded tree has to name both halves: what is
    in there and must keep passing, and what is not in there yet.  The names
    live in ``constants`` because the directed suite is generated from the
    same list; read with ``getattr`` so this keeps working whichever of the
    plausible constant names the suite's author settles on, and so a round
    three does not need an edit here.
    """
    import constants as _c
    done = (getattr(_c, "IMPLEMENTED_INSNS", None)
            or getattr(_c, "ROUND_ONE_INSNS", ()) or ())
    new = (getattr(_c, "NEW_INSNS", None)
           or getattr(_c, "ROUND_TWO_INSNS", None) or ())
    if not new:
        # Last resort: whatever the full scope holds that the done list does
        # not.  An empty result is legal -- the message just drops the
        # sentence that would have named the new instruction.
        every = getattr(_c, "ALL_INSNS", None) or getattr(_c, "INSNS", ()) or ()
        new = tuple(i for i in every if i not in done)
    return tuple(done), tuple(new)


def _insn_list(names: Sequence[str]) -> str:
    """Instruction names for a prompt, with a truthful fallback.

    ``constants`` may not yet name the new round's instructions when this
    runs (the directed suite and the constant land together, and the loop
    must not break in the window between).  The fallback says what the agent
    can still act on: the failing tests below are the scope.
    """
    return (", ".join(f"`{n}`" for n in names)
            or "the instruction(s) the directed suite has just grown tests "
               "for (read the failing test names below for which)")


def format_model_seed_message(source: str,
                              outcomes: Sequence[Tuple[str, Outcome]],
                              logs: Dict[str, str],
                              done: Sequence[str],
                              new: Sequence[str],
                              log_path: Optional[str] = None) -> str:
    """Stage 0's first turn when the tree was seeded with a working model.

    Pure: everything it needs is an argument, so it can be tested without a
    cluster (and is).  The failure half is ``format_directed_failure``
    verbatim -- same one-line table, same per-test evidence with real values,
    same log path -- because the agent's job here is exactly the debug job,
    only the framing differs: the failures are the new instruction's tests,
    and the passes are a previous round's work that must not regress.
    """
    summary = summarize(outcomes)
    failing = summary["failing"]
    head = [
        "# You are extending a model that already works\n",
        f"A converged Spike model for {_insn_list(done)} is **already "
        f"applied in the working tree** (seeded from `{source}`); it is not "
        f"something you have to write. Read it before you edit anything.\n",
        f"Your job this round is to add {_insn_list(new)} to it **without "
        f"breaking the instructions that already pass**. Extend the code "
        f"that is there -- do not re-derive it, do not restructure it, and "
        f"do not change the behaviour of an instruction that is already "
        f"correct in order to make a new one work.\n",
        f"The directed suite has been run against the seeded model as it "
        f"stands. Its result is below: {summary['counts']}, "
        f"{len(failing)} of {summary['total']} failing.\n",
    ]
    if not failing:
        head.append(
            "Nothing is failing yet, which means the suite does not cover "
            "the new instruction on this tree -- or the geometry it needs is "
            "being declined. Add the instruction from the spec's SAIL "
            "anyway, and use `read_status` to check which geometries are "
            "being skipped.\n")
        return "\n".join(head)
    head.append(
        "**Those failures are the new instruction's tests.** The other "
        "programs pass because the seeded model is right about them; if a "
        "test that passes here starts failing after your edit, you have "
        "broken something that worked.\n")
    return "\n".join(head) + "\n" + format_directed_failure(
        1, outcomes, logs, log_path=log_path)


def format_rtl_seed_scope(done: Sequence[str], new: Sequence[str],
                          failing: bool) -> str:
    """The same framing for the RTL agent's first turn after ``--rtl-diff``.

    Kept separate from the model version because the RTL note is appended to
    the standing implement prompt rather than replacing it, and because the
    RTL agent is also told what a *clean* attempt-0 means.
    """
    if failing:
        return (
            f"Scope: the RTL in the tree already implements {_insn_list(done)}"
            f" and passes their tests. This round adds {_insn_list(new)}. "
            f"**The failures below are the NEW instruction's tests -- the "
            f"others already pass, and regressing one of them is a failure "
            f"of this round even if the new instruction starts working.** "
            f"Extend the design; do not rewrite what is already correct.\n")
    return (
        f"Scope: the RTL in the tree already implements {_insn_list(done)} "
        f"and every directed test currently passes or skips. This round adds "
        f"{_insn_list(new)}; add it without regressing any of the above.\n")
