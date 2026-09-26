#!/usr/bin/env python3
"""Titan: an agentic implementation of the RISC-V IME (Zvvm) on Saturn.

Four stages, and the ordering principle behind them is the only design
decision here that matters:

    0   model        Zvvm in Spike, from the spec's SAIL   judge: the S1 programs
    S1  directed     paired IME / RVV-1.0 programs         judge: rvv_ref.py
    S2  regression   Saturn's own riscv-vector-tests       judge: ships with Saturn
    S3  stress       randomised tile geometries            judge: the stage 0 model

(Stage 0 was called stage M before 2026-09-25; older logs use that name.)

**Every judge predates the defendant it judges.** rvv_ref.py is written outside
the loop (Claude Code sessions directed by the authors) before any RTL exists; riscv-vector-tests shipped with Saturn and passed
before anyone touched it; and the Spike model runs *first*, so its author
cannot have seen the RTL it will later judge -- that independence is a
property of the ordering, not of a rule in a prompt that an agent might work
around.  None of the three judges is the implementer's own work, which is the
whole reason a pass from them means anything.

Note what stage 0 is judged by: the same paired programs stage 1 uses.  Spike's
RVV 1.0 support is upstream and mature, so in a program running on Spike the
reference path is trustworthy and the IME path is the new thing -- a
disagreement convicts the model.  The same programs therefore judge two
different defendants, and seed the randomised third stage.

This is the structural departure from riscv_extensions, whose three stages all
rested on Spike.  It could: Zb*, Zk* and Zicond are ratified and upstream Spike
implements them.  Zvvm is a draft Spike has never heard of, so the model is a
deliverable of this loop rather than a dependency of it, and S1 is built to
need no Spike at all rather than to wait for one.

    chia job submit -- python examples/titan/titan_loop.py
"""
from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import hostload
import json
import math
import os
import tarfile
import tempfile
import time
import zlib
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from chia.base.ChiaFunction import get
from chia.base.tools.BashTool import BashTool
from chia.trace.profiler import get_profiler, start_collector

import db_node
import helpers
import ime_stress
import ime_tests
import llm as llm_mod
import nodes
import tools as titan_tools
from constants import (AGENT_LOG_DIR, AGENT_LOG_REL, AGENT_RUNS_PER_ITER,
                       CHIPYARD_PATH, COSIM_CONFIG, DEBUG_MAX_ITERS,
                       SYNTH_CONFIG as DIRECTED_CONFIG,
                       MAX_ITERS, MODEL_MAX_ITERS, MODEL_MAX_SKIP_FRACTION,
                       REGRESSION_ALERT_DELTA, REGRESSION_BASELINE_PATH,
                       REGRESSION_CONFIRM_MAX, REGRESSION_CONFIRM_REPS,
                       REGRESSION_FLAKY_REPEAT_ALERT,
                       REGRESSION_MAX_INFLIGHT,
                       REGRESSION_SAMPLE, RVV_AGENT_MAX_TESTS,
                       RUNTIME_ENV, S2_COSIM_SCALA, S2_COSIM_SCALA_REL,
                       SIMLOG_TAIL_LINES, SPEC_DIR,
                       SPIKE_SRC_REL,
                       STRESS_TEST_CASES_PER_GEOM,
                       STRESS_TEST_HOURS, TITAN_LOG_ROOT, VLEN)

EXTENSION = "ime"

#: Key under which the S2 suite is staged in the database.
#:
#: Saturn ships riscv-vector-tests as a submodule with a build-tests.sh already
#: configured for TEST_MODE=cosim at VLEN 128 and 256, so the S2 gate costs
#: nothing to *design*.  It does have to be built once and staged here before
#: the loop runs; that is a setup step, deliberately outside the loop, because
#: a regression suite the loop builds for itself is a regression suite the loop
#: could get wrong.
RVV_REGRESSION_KEY = "rvv"

#: Scratch on the worker nodes.  ``tempfile.gettempdir()`` honours $TMPDIR,
#: which cluster.yaml points at /share1/.../titan_scratch/node_tmp on every
#: node (the root filesystem is small and shared with everyone else).
SIM_WORK_DIR = os.path.join(tempfile.gettempdir(), "titan-sim")
BUILD_WORK_DIR = os.path.join(tempfile.gettempdir(), "titan-build")

#: PPA synthesis is written (`synth_node.run_shuttle_tile_synthesis`) but not
#: wired into this loop yet, and `SYNTH_CONFIG` is imported only so the two
#: config names stay together in one place.  riscv_extensions dispatches a
#: baseline synth at the start of a run and a comparison synth after
#: convergence, then logs the delta; Titan will want the same, but it belongs
#: after Stage 3 produces a design worth measuring.  Left explicit rather than
#: silently dead so the next reader knows it is a gap, not an oversight.
#:
#: Note `synth_runtime_env()` in constants.py: the synthesis path needs
#: `sky130_vlsi` shipped to the worker, which the default RUNTIME_ENV does not
#: do.  Wiring synth without it fails on the worker with an ImportError.
PPA_SYNTH_WIRED = False


#: How many LLM turns may come back ``success=False`` in a row before the run
#: gives up.  A CLI that fails instantly -- a bad flag, a missing credential,
#: a model name the backend does not serve -- does not raise: it returns a
#: result object with ``success=False`` and an empty transcript, and the loop
#: cheerfully builds and judges an untouched tree for every remaining
#: iteration.  Two in a row is not a flake; it is a broken configuration, and
#: continuing costs a build per iteration for nothing.
LLM_MAX_CONSECUTIVE_FAILURES = 2

#: No-progress stop (r26).  Stage 0 of r26_0926_0144 ran 25 attempts ($458)
#: whose results were identical from attempt 2 on -- 795 pass / 1 skip /
#: 2 bad_geometry, same two failing names -- because the agent had concluded
#: (correctly) that the judge was wrong and stopped editing.  Iterating
#: cannot fix that, so K consecutive graded attempts with the same counts and
#: the same failing set end the stage.  0 disables it.
NO_PROGRESS_STOP = int(os.environ.get("TITAN_NO_PROGRESS_STOP", "3"))

#: Set by the first stage that stalls; copied into the run summary as
#: ``result["stalled"]``.  Reset at the top of :func:`run`.
_STALLED: Dict[str, object] = {}


class _StallTracker:
    """Counts consecutive graded attempts with an identical outcome.

    An outcome is ``(counts, failing set)`` -- plus, for S2, the confirmed
    regression failures -- so a changed count *or* a different failing name
    is progress.  Attempts that were never graded (LLM failure) are not
    observed at all; a build failure is observed as its own outcome, so it
    breaks a streak of graded results rather than extending it.
    """

    def __init__(self, stage: str, limit: Optional[int] = None):
        self.stage = stage
        self.limit = NO_PROGRESS_STOP if limit is None else limit
        self.key = None
        self.streak = 0
        self.first = 0

    def observe(self, attempt: int, counts: Dict[str, int],
                failing: Sequence[str], extra: Sequence[str] = ()
                ) -> Optional[Dict[str, object]]:
        key = (tuple(sorted((k, int(v)) for k, v in counts.items())),
               tuple(sorted(failing)), tuple(sorted(extra)))
        if key == self.key:
            self.streak += 1
        else:
            self.key, self.streak, self.first = key, 1, attempt
        if self.limit and self.limit > 0 and self.streak >= self.limit:
            return {"stage": self.stage, "attempts": [self.first, attempt],
                    "consecutive": self.streak, "limit": self.limit,
                    "counts": dict(key[0]), "failing": list(key[1]),
                    "s2_failing": list(key[2]),
                    "reason": (f"{self.streak} consecutive attempts "
                               f"({self.first}..{attempt}) produced the "
                               f"identical outcome; stopping "
                               f"(TITAN_NO_PROGRESS_STOP={self.limit})")}
        return None


def _stall(info: Dict[str, object]) -> None:
    """Record the first stall of the run and emit it to the trace."""
    if not _STALLED:
        _STALLED.update(info)
    _event("stage_stalled", **{k: v for k, v in info.items()
                               if isinstance(v, (int, float, str, bool))})
    print(f"[stall] {info['stage']}: {info['reason']}", flush=True)


def _llm_call_failed(cli) -> bool:
    """True when the backend reported the turn itself failed.

    ``helpers.dump_llm`` writes this same field at the top of every
    transcript, which is how the failed runs were identified after the fact.
    ``None`` (a backend that reports nothing) is not a failure.
    """
    return getattr(cli, "success", None) is False


def _event(event: str, **kw) -> None:
    get_profiler().log_event(event, extension=EXTENSION, **kw)


def _tar_dir(path: str) -> bytes:
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(path, arcname=".")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# stage 1 + 2: build, run the directed suite, run the regression suite
# ---------------------------------------------------------------------------

def _build_directed(dump: helpers.Dumper, vlen: int, full_sweep: bool
                    ) -> Tuple[List[Tuple[str, bytes]], Dict[str, object]]:
    """Assemble the directed programs and return (tests, geometry by name).

    Two tiers.  Per-iteration runs one program per (SEW, LAMBDA, LMUL) at full
    VL -- 92, covering every tile geometry and register allocation. The gate
    sweeps partial N as well, which is 608 programs and well over a million
    lines of assembly: worth paying once when the cheap tier is already
    green, not worth paying sixty times.  Partial N is where the C tile tail
    policy lives, so the gate cannot skip it.
    """
    suite = ime_tests.directed_suite(vlen, full_vl_only=not full_sweep)
    refs = {name: nodes.build_ime_test.chia_remote(asm, name, BUILD_WORK_DIR,
                                                   extension=EXTENSION)
            for name, asm, _ in suite}
    tests, geometries, unbuildable = [], {}, []
    for name, asm, geom in suite:
        elf = get(refs[name])
        geometries[name] = geom
        if elf:
            tests.append((name, elf))
        else:
            unbuildable.append(name)
            dump.text(f"unbuildable_{name}.S", asm)
    if unbuildable:
        # The generator produced something the assembler rejected.  That is
        # our bug, not the DUT's, and it must not be reported to the agent as
        # a failing test.
        dump.json("unbuildable.json", unbuildable)
        raise RuntimeError(
            f"{len(unbuildable)} directed programs failed to assemble "
            f"(sources dumped). This is a generator bug: fix ime_tests.py "
            f"before running the loop. First: {unbuildable[0]}")
    return tests, geometries


def _dump_sim_logs(dump: helpers.Dumper, stem: str, outcomes, logs: Dict[str, str],
                   tail_lines: int = SIMLOG_TAIL_LINES) -> None:
    """Keep the tail of every simulator log on disk.  The loop feeds these to
    the agent but otherwise dropped them, which made "27/27 trap" impossible
    to diagnose after the fact.

    The tail has to be long enough to hold a failing program's evidence block
    as well as its verdict: the verdict is printed last precisely so that a
    tail which catches it catches the ~26 lines of TITAN DIFF / CDUMP / CREF
    that explain it, and the old 60-line window left no room for anything the
    simulator said afterwards."""
    kinds = {name: o.kind for name, o in outcomes}
    chunks = []
    for name, log in logs.items():
        lines = (log or "").splitlines()
        body = "\n".join(lines[-tail_lines:])
        chunks.append(f"===== {name}  [{kinds.get(name, '?')}]  "
                      f"({len(lines)} lines, last {min(len(lines), tail_lines)})\n{body}\n")
    dump.text(f"{stem}.simlogs.txt", "\n".join(chunks))


#: Head of the build log kept for the agent's copy.  The whole thing is
#: 400kB of sbt resolving dependencies; this is enough to hold any real
#: failure and small enough to ship as a Ray object every iteration.
AGENT_BUILD_TAIL_LINES = 4000

#: Per-actor cache so ``run_directed(rebuild=False)`` can re-run the tests on
#: the design it just built.  Lives in the tool server actor's process, which
#: is the same process for every call in a turn.
_AGENT_ARTIFACTS: Dict[str, object] = {}

#: The same, for the S2 cosim build ``run_rvv`` needs.  Kept apart from
#: ``_AGENT_ARTIFACTS`` because the two are different elaborations of the
#: same tree (``DIRECTED_CONFIG`` vs ``COSIM_CONFIG``) and one cannot stand
#: in for the other: the cosim harness is what makes lockstep possible.
_AGENT_COSIM_ARTIFACTS: Dict[str, object] = {}

#: The STOCK ``libriscv``, built once at the top of ``run()`` from the
#: pristine riscv-isa-sim checkout -- before Stage 0 patches it -- and passed
#: as ``build_saturn``'s ``golden_model`` for every S2 cosim build.
#:
#: Why it has to exist.  ``nodes.build_spike`` installs what it builds into
#: ``$RISCV/lib/libriscv.so``, and nothing ever puts the stock library back:
#: ``reset_chipyard`` resets git-tracked *sources*, and an installed library
#: is a build artifact, not a tracked file.  So from Stage 0 onwards the
#: cosim simulator S2 elaborates links the agent's *IME model* -- whose
#: ``vectorUnit_t`` carries four extra ``reg_t`` members and a rewritten
#: ``set_vl`` -- while the cospike bridge next to it is compiled against the
#: stock headers in ``$RISCV/include/riscv`` (``build_spike`` does not update
#: those).  S2 was therefore grading the RTL against a Spike that is neither
#: stock nor ABI-compatible with its own bridge.  (This was first blamed for
#: r10's 22 ``.vf`` failures; later measurements -- titan_runs/s2_judge*/ --
#: showed those were real: 9/10 against clean stock Spike, 0/10 on a pristine
#: tree, i.e. an RTL stall bug.  The contamination was a judge bug all the same.)
#:
#: Passing the artifact makes ``ChiselBuildNode`` restage its bytes into
#: ``$RISCV/lib`` before make (and carry them in the simulator's
#: ``runtime_libs``), so S2 and the agent's ``run_rvv`` cosimulate against
#: stock Spike -- what nodes.py's docstrings and prompts/implement.md have
#: always claimed -- while Stage 3 keeps passing the model artifact and keeps
#: the model.  ``clean_sim=True`` is already set, so the relink is free.
#:
#: Set in ``run()`` (this process, for ``_run_s2``) and handed to the agent's
#: tool actor through ``cfg["stock_spike"]`` -- a different process, where a
#: module global set here would not be visible.
_STOCK_SPIKE = None


def _publish_logs(run_id: str, leaf: str, pg_opts, *,
                  build_stdout: str = "", build_stderr: str = "",
                  logs: Optional[Dict[str, str]] = None,
                  status_path: Optional[str] = None,
                  summary: Optional[Dict[str, object]] = None,
                  message: Optional[str] = None,
                  extra_files: Optional[Dict[str, str]] = None) -> str:
    """Copy this iteration's artifacts into the tree the agent can read.

    The reason the feedback message could shrink.  Everything the loop knows
    is on the chipyard node, in full, one directory per iteration -- so the
    message can be a pointer instead of a paste, and an agent that wants the
    2,000-line simulator log of one test can have it without the other 26
    being pasted alongside.

    Dispatched with ``pg_opts`` like every other chipyard node: the placement
    group holds that node's whole ``chipyard`` resource, and a task scheduled
    outside it would sit PENDING_NODE_ASSIGNMENT for the rest of the run.
    """
    files: Dict[str, str] = {}
    if build_stdout:
        files["build.stdout.txt"] = "\n".join(
            build_stdout.splitlines()[-AGENT_BUILD_TAIL_LINES:])
    if build_stderr:
        files["build.stderr.txt"] = "\n".join(
            build_stderr.splitlines()[-AGENT_BUILD_TAIL_LINES:])
    for name, log in (logs or {}).items():
        files[f"sim_{name}.log"] = log or ""      # in full, not a tail
    if status_path and os.path.exists(status_path):
        with open(status_path, encoding="utf-8") as fh:
            files["status.md"] = fh.read()
    if summary is not None:
        files["directed.json"] = json.dumps(summary, indent=2, default=str)
    if message:
        files["feedback.md"] = message
    # Anything the caller wants published verbatim -- best_rtl.diff, which is
    # not an artifact of one iteration but of the run's high-water mark.
    files.update(extra_files or {})
    target = os.path.normpath(os.path.join(AGENT_LOG_DIR, run_id, leaf))
    try:
        get(nodes.write_files.options(**pg_opts).chia_remote(
            target, files, exclude_rel=AGENT_LOG_REL, extension=EXTENSION))
    except Exception as exc:                                # noqa: BLE001
        # A failed copy must not fail the run: the loop still has the same
        # artifacts on the head, the agent just cannot read them this turn.
        _event("agent_logs_failed", leaf=leaf, error=str(exc)[:200])
        return target
    return target


def agent_run_directed(cfg: Dict[str, object], tests: str, rebuild: bool,
                       seq: int) -> str:
    """What ``RunDirectedTool`` actually does.  Runs inside the tool actor.

    Deliberately the *same* build and the same ``_run_directed`` the loop
    calls between iterations -- a tool that tested something adjacent would
    be worse than no tool, because a clean result from it would not predict a
    clean iteration.

    The one thing that had to be got right is where the build is dispatched.
    ``build_saturn`` needs ``chipyard`` and there is exactly one; the loop's
    placement group has reserved it for the whole run.  So this dispatches
    with the loop's ``pg_opts``, into the same bundle.  That is safe because
    the agent's ``BashTool`` does *not* hold the resource while the turn is in
    progress: ``ChiaTool``'s server is ``_ToolServerActor``, declared
    ``@ray.remote(num_cpus=0)``, and ``pg_opts`` is a scheduling strategy and
    nothing else -- so the actor is placed in the bundle but reserves none of
    its ``chipyard``.  Dispatching *outside* the group is the deadlock (see
    ``_run_on_spike``: r2 sat on 27 pending tasks for four hours that way).
    """
    pg_opts = cfg["pg_opts"]
    run_id, tag = cfg["run_id"], cfg["tag"]
    vlen, status_path = cfg["vlen"], cfg["status_path"]
    dump = helpers.Dumper(cfg["out_dir"])

    suite, geometries = _build_directed(dump, vlen, full_sweep=False)
    by_name = dict(suite)
    if tests.strip() == "all":
        selected = list(suite)
        which = "all"
    elif tests.strip() in ("", "failing"):
        wanted = _read_failing(cfg["failing_path"])
        selected = [(n, e) for n, e in suite if n in wanted] or list(suite)
        which = "failing" if wanted else "all (nothing has failed yet)"
    else:
        names = [t.strip() for t in tests.split(",") if t.strip()]
        unknown = [n for n in names if n not in by_name]
        if unknown:
            return (f"No such directed test(s): {', '.join(unknown)}. Names "
                    f"are as they appear in the status file, e.g. "
                    f"{suite[0][0]}.")
        selected = [(n, by_name[n]) for n in names]
        which = ", ".join(names)

    artifact = _AGENT_ARTIFACTS.get(tag)
    if rebuild or artifact is None:
        artifact = get(nodes.build_saturn.options(**pg_opts)
                       .chia_remote(DIRECTED_CONFIG, extension=EXTENSION))
        _AGENT_ARTIFACTS[tag] = artifact
    leaf = f"agent_{seq}"
    if not artifact.success:
        path = _publish_logs(run_id, leaf, pg_opts,
                             build_stdout=getattr(artifact, "stdout", "") or "",
                             build_stderr=getattr(artifact, "stderr", "") or "")
        titan_tools.record_agent_run(cfg["events_path"], {
            "event": "agent_directed", "tag": tag, "seq": seq,
            "tests": which, "rebuild": bool(rebuild), "build": "failed"})
        return helpers.format_build_failure(artifact, seq, log_path=path)

    outcomes, logs = _run_directed(artifact, selected)
    titan_tools.write_status(status_path, outcomes, geometries)
    summary = helpers.summarize(outcomes)
    # Only the tests that actually ran can change state: a subset run must
    # not shrink the failing set to its own failures.
    ran = {n for n, _ in selected}
    prev_failing = _read_failing(cfg["failing_path"])
    _write_failing(cfg["failing_path"],
                   (prev_failing - ran) | set(summary["failing"]))
    path = _publish_logs(run_id, leaf, pg_opts,
                         build_stdout=getattr(artifact, "stdout", "") or "",
                         logs=logs, status_path=status_path, summary=summary)
    titan_tools.record_agent_run(cfg["events_path"], {
        "event": "agent_directed", "tag": tag, "seq": seq, "tests": which,
        "rebuild": bool(rebuild), "build": "ok", **summary["counts"]})

    if not summary["failing"]:
        return (f"All {len(selected)} directed test(s) run ({which}) are "
                f"clean: {summary['counts']}. If you ran a subset, run "
                f"`run_directed_start('all')` before you call finish -- a subset "
                f"passing is not the gate.\n\nLogs: {path}")
    body = helpers.format_directed_failure(seq, outcomes, logs, log_path=path)
    return f"(run_directed: {which}, rebuild={bool(rebuild)})\n\n{body}"


def agent_run_rvv(cfg: Dict[str, object], tests: str, rebuild: bool,
                  seq: int) -> str:
    """What ``run_rvv_start`` actually does.  Runs inside the tool actor.

    The S2 counterpart of :func:`agent_run_directed`, and deliberately the
    same build (``COSIM_CONFIG``, written into the tree by
    ``_write_s2_cosim_config`` and taken back out immediately) and the same
    ``_run_regression`` the loop's own gate runs -- so a clean answer here
    predicts a clean S2 rather than resembling one.

    The difference from the gate: no sampling.  The gate samples because it
    has to cover the suite in bounded time; the agent has *named* what it
    wants (or inherited the failing set, which is already small), and
    sampling a named selection would silently drop tests the agent asked
    for.  Baseline subtraction stays on: a test riscv-vector-tests already
    failed on a pristine tree is not evidence about anybody's change.
    """
    pg_opts = cfg["pg_opts"]
    run_id, tag = cfg["run_id"], cfg["tag"]
    rvv_failing_path = cfg.get("rvv_failing_path")

    choice = (tests or "").strip()
    if choice == "all":
        return ("`all` is 841 cosimulations, roughly nine hours -- it cannot "
                "run inside a turn. Use 'failing' (the tests the last "
                "regression run failed, listed in the status file), 'sample' "
                "(the loop's own sample, ~35 minutes), or name up to "
                f"{RVV_AGENT_MAX_TESTS} tests.")
    sample = False
    if choice in ("", "failing"):
        wanted = sorted(_read_failing(rvv_failing_path or ""))
        if not wanted:
            return ("No RVV regression test is currently marked failing, so "
                    "there is nothing for 'failing' to run. Either S2 is "
                    "clean or it has not run yet -- check the status file. "
                    "If you want coverage anyway, name tests explicitly, or "
                    "use 'sample' (~35 minutes).")
        select, which = wanted, f"failing ({len(wanted)})"
    elif choice == "sample":
        select, sample, which = None, True, "sample"
    else:
        names = [t.strip() for t in choice.split(",") if t.strip()]
        if len(names) > RVV_AGENT_MAX_TESTS:
            return (f"{len(names)} tests is too many for one call (cap "
                    f"{RVV_AGENT_MAX_TESTS}); a cosim run is ~15s per test. "
                    f"Narrow it, or use 'failing'.")
        select, which = names, ", ".join(names)

    artifact = _AGENT_COSIM_ARTIFACTS.get(tag)
    if rebuild or artifact is None:
        _write_s2_cosim_config(pg_opts)
        try:
            # The same stock golden model _run_s2 links -- see _STOCK_SPIKE.
            # It arrives through cfg because this runs in the tool actor's
            # process, not the driver's.
            artifact = get(nodes.build_saturn.options(**pg_opts)
                           .chia_remote(COSIM_CONFIG, cfg.get("stock_spike"),
                                        extension=EXTENSION))
        finally:
            # Out of the tree again before anything else elaborates it --
            # the loop's own build would otherwise inherit the agent's
            # harness file.  Same contract as _run_s2.
            _write_s2_cosim_config(pg_opts, enable=False)
        _AGENT_COSIM_ARTIFACTS[tag] = artifact
    leaf = f"agent_rvv_{seq}"
    if not artifact.success:
        path = _publish_logs(run_id, leaf, pg_opts,
                             build_stdout=getattr(artifact, "stdout", "") or "",
                             build_stderr=getattr(artifact, "stderr", "") or "")
        titan_tools.record_agent_run(cfg["events_path"], {
            "event": "agent_rvv", "tag": tag, "seq": seq, "tests": which,
            "rebuild": bool(rebuild), "build": "failed"})
        return helpers.format_build_failure(artifact, seq, log_path=path)

    failing, ran = _run_regression(artifact, select=select, sample=sample)
    _merge_failing(rvv_failing_path, ran, [n for n, _, _ in failing])
    path = _publish_logs(
        run_id, leaf, pg_opts,
        build_stdout=getattr(artifact, "stdout", "") or "",
        logs={n: log for n, _, log in failing if log},
        extra_files={"failures.txt": "\n".join(f"{n}: {r}"
                                               for n, r, _ in failing)
                     or "none\n",
                     "ran.txt": "\n".join(ran) + "\n"})
    titan_tools.record_agent_run(cfg["events_path"], {
        "event": "agent_rvv", "tag": tag, "seq": seq, "tests": which,
        "rebuild": bool(rebuild), "build": "ok", "ran": len(ran),
        "failing": len(failing)})

    if not ran:
        return (f"(run_rvv: {which}) No test matched. Names are as they "
                f"appear in the status file's failing-RVV list; a test the "
                f"baseline already fails is excluded and cannot be run.")
    if not failing:
        return (f"(run_rvv: {which}, rebuild={bool(rebuild)})\n\n"
                f"{len(ran)}/{len(ran)} pass. A subset passing is not the "
                f"S2 gate -- the loop runs its own sample after your turn.\n"
                f"\nLogs: {path}")
    body = helpers.format_regression_failure(seq, failing, log_path=path)
    return (f"(run_rvv: {which}, rebuild={bool(rebuild)}, "
            f"{len(failing)}/{len(ran)} failing)\n\n{body}")


def _read_failing(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as fh:
            return set(json.load(fh))
    except (OSError, ValueError):
        return set()


def _write_failing(path: str, names) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(list(names), fh)
    except OSError:
        pass


def _merge_failing(path: Optional[str], ran, failing) -> None:
    """Same merge rule as the directed failing set: only what ran changes.

    A ``run_rvv_start('vmv...')`` that passes must clear exactly that test,
    not the 200 it never touched -- otherwise ``'failing'`` would collapse to
    "whatever the last partial run happened to break" and the agent would
    lose the set the loop actually grades.
    """
    if not path:
        return
    ran = set(ran)
    _write_failing(path, (_read_failing(path) - ran) | set(failing))


def _dump_regression_logs(dump: helpers.Dumper, stem: str,
                          failing: Sequence[Tuple[str, str, str]]) -> None:
    """The S2 equivalent of ``_dump_sim_logs``.

    r8's S2 wrote a JSON list of 841 names and nothing else, so "why did
    every test fail?" was unanswerable without re-running the whole gate by
    hand.  This keeps the head of the first few failing logs -- the head, not
    the tail, because a cospike abort prints its divergence and then dumps
    spike's architectural state, so the interesting part is at the top -- plus
    a histogram of the last line of every log, which is what tells 841
    identical harness deaths apart from 841 real divergences.
    """
    chunks = [f"{len(failing)} failing regression test(s)\n"]
    for name, reason, log in list(failing)[:5]:
        lines = (log or "").splitlines()
        head = "\n".join(lines[:40]) or "(no log captured)"
        chunks.append(f"===== {name}\n{reason}\n({len(lines)} lines, "
                      f"first {min(len(lines), 40)})\n{head}\n")
    hist: Dict[str, int] = {}
    for _, reason, log in failing:
        lines = [ln for ln in (log or "").splitlines() if ln.strip()]
        key = lines[-1].strip() if lines else f"(no log) {reason}"
        hist[key] = hist.get(key, 0) + 1
    chunks.append("===== histogram of the last line of each log\n")
    for key, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        chunks.append(f"{n:5d}  {key[:200]}")
    dump.text(f"{stem}.simlogs.txt", "\n".join(chunks))


def _run_directed(artifact, tests: Sequence[Tuple[str, bytes]]
                  ) -> Tuple[List[Tuple[str, helpers.Outcome]], Dict[str, str]]:
    refs = {name: nodes.verilator_run_remote.chia_remote(
        artifact, elf, name, SIM_WORK_DIR, extension=EXTENSION)
        for name, elf in tests}
    outcomes, logs = [], {}
    for name, _ in tests:
        run = get(refs[name])
        outcomes.append((name, helpers.classify_run(run)))
        logs[name] = run.log or ""
    # Whether a DUT that answered a lambda request with a larger value broke
    # the WARL rule or used its one legal fallback depends on what it did
    # with the *other* geometries, so it can only be settled here, with the
    # whole run in hand.
    return helpers.reconcile_geometry(outcomes), logs


#: What the loop leaves in place of its S2 config when it is not building it.
#:
#: Not deletion, and not the config itself.  The file is git-ignored so that
#: it never enters a probe diff, which also means ``reset_chipyard``'s
#: ``git clean -fd`` (no ``-x``) leaves it alone -- and sbt compiles every
#: source in the tree for *any* config, so a leftover file naming
#: ``DIRECTED_CONFIG`` makes every build fail the moment the tree does not
#: contain that config: a fresh tree, the preflight, the baseline PPA build,
#: iteration 0 before the agent has written anything.  A comment cannot.
S2_COSIM_STUB = ("// titan_loop: S2 cosim harness goes here during Stage 2.\n"
                 "// Deliberately empty otherwise -- see constants.S2_COSIM_SCALA.\n")


def _write_s2_cosim_config(pg_opts, enable: bool = True) -> None:
    """Put the loop's own S2 cosim config into the tree, or take it back out.

    Generated from ``DIRECTED_CONFIG`` so S2 grades exactly the design the
    directed suite just passed, and pinned to ``retireWidth = 1`` because
    Shuttle's multi-lane DebugROB drops/reorders trace entries through the
    ``debug_rob`` DPI -- see ``constants.S2_COSIM_SCALA`` for the evidence.
    Written with ``exclude_rel`` so it is git-ignored: it must never appear
    in ``collect_diff`` (the diff is what reseeds a run) and it has to
    survive ``reset_chipyard``'s ``git clean -fd``.

    Call with ``enable=False`` once the cosim build is done: see
    ``S2_COSIM_STUB``.
    """
    body = (S2_COSIM_SCALA % {"cosim": COSIM_CONFIG, "base": DIRECTED_CONFIG}
            if enable else S2_COSIM_STUB)
    get(nodes.write_files.options(**pg_opts).chia_remote(
        CHIPYARD_PATH, {S2_COSIM_SCALA_REL: body},
        exclude_rel=S2_COSIM_SCALA_REL, extension=EXTENSION))


#: How many failing regression tests get their log kept in full.  841 logs of
#: a few hundred kB each would be a 200MB Ray object and an unreadable
#: directory; the first handful is what a diagnosis actually needs.
REGRESSION_LOGS_KEPT = 12

#: 迴歸跑到一半時每幾支重新評估一次主機負載。Gate 的 837 支要跑一小時以上，
#: 主機在這段時間裡會變；ps 一次約 20ms，這個頻率的成本可以忽略。
HOSTLOAD_RECHECK_EVERY = int(
    os.environ.get("TITAN_HOSTLOAD_RECHECK_EVERY", "50"))

#: 接續（--rtl-diff）時，如果 attempt 0 的 directed 全過、S2 抽樣也乾淨，
#: 就不再叫 LLM，直接進 Gate。r22 的教訓：樹已經收斂，agent 的第一個 turn
#: 卻「順手」改了 MatrixFPMultiplyPipe，讓 cosim 在每支 RVV 測試都死在第 0
#: 條指令；它最後自己 revert 了，但那一整個 turn 是純成本加純風險。Gate 失敗
#: 時仍走原本的 gate-debug 流程叫 LLM，所以這條捷徑不會跳過任何判官。
#: 設 TITAN_RESUME_FAST_PATH=0 可關閉。
RESUME_FAST_PATH = os.environ.get("TITAN_RESUME_FAST_PATH", "1") != "0"


def _run_regression(artifact, select: Optional[Sequence[str]] = None,
                    sample: bool = True
                    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """S2: Saturn's riscv-vector-tests must still pass.

    Returns ``(name, reason, log)`` for every failure -- the log being
    cospike's abort window, which until r9 was decompressed nowhere and
    thrown away.  That is how r8 got 841/841 failing with nothing on disk
    but a list of names.

    Judged by lockstep cosimulation, not by reading the log.  Saturn's
    build-tests.sh builds the suite with ``TEST_MODE=cosim``, which compiles
    ``-DCOSIM_TEST_CASE`` and **removes the self-verification** -- the tests
    are designed to be judged by a co-simulator and never print a verdict.
    Grepping for "*** PASSED ***" (the TEST_MODE=self contract) would find
    nothing and pass everything.

    Cosim needs no IME model here: these are pure RVV 1.0 tests, and the
    caller links the stock libriscv (``_STOCK_SPIKE``) into the simulator.
    That is not free of charge, and the comment that used to sit here said
    the opposite: the golden model does NOT come from the verilator_run
    image.  ``ChiselBuildNode`` bundles the chipyard container's own
    ``$RISCV/lib/libriscv.so`` into the artifact's ``runtime_libs``, and the
    run worker puts that on ``LD_LIBRARY_PATH`` -- so whatever Stage 0 last
    installed is exactly what this gate would cosimulate against.  See
    ``_STOCK_SPIKE``.

    ``select`` narrows the suite to named tests (the agent's ``run_rvv``);
    ``sample=False`` runs every selected test rather than
    ``REGRESSION_SAMPLE``'s stride.  Returns ``(failures, ran)`` -- the
    names that actually ran, which is what lets a partial run update the
    failing set without shrinking it to its own failures.  Baseline
    subtraction applies either way: a test that was already broken before
    the run is not this run's problem, whoever asked for it.
    """
    # fetch_tests' single parameter is the DB key, which doubles as the
    # profiler's extension tag -- it is not the extension= tagging kwarg the
    # other nodes carry.
    suite = get(db_node.fetch_tests.chia_remote(RVV_REGRESSION_KEY))
    if not suite:
        # Loud, not silent.  A gate that quietly passes because its suite is
        # missing is worse than no gate: it certifies nothing while looking
        # like it certified something.
        raise RuntimeError(
            f"the RVV regression suite is not staged under "
            f"'{RVV_REGRESSION_KEY}'. Run stage_rvv_tests.py first.")
    # Known-bad first, so they are neither dispatched nor waited on.
    try:
        with open(REGRESSION_BASELINE_PATH, encoding="utf-8") as fh:
            known_bad = set(json.load(fh))
    except (OSError, ValueError):
        known_bad = set()
        _event("regression_baseline_missing", path=REGRESSION_BASELINE_PATH)
    suite = [(n, e) for n, e in suite if n not in known_bad]
    if select is not None:
        wanted = set(select)
        suite = [(n, e) for n, e in suite if n in wanted]
    if sample and REGRESSION_SAMPLE and 0 < REGRESSION_SAMPLE < len(suite):
        stride = len(suite) / float(REGRESSION_SAMPLE)
        suite = [suite[int(i * stride)] for i in range(REGRESSION_SAMPLE)]
    def _submit(nm, el):
        return nodes.cosim_run.chia_remote(
            artifact, el, nm, 0, SIM_WORK_DIR, extension=EXTENSION)

    # 降載旋鈕：REGRESSION_MAX_INFLIGHT > 0 時改成滑動視窗送出，一次只讓 N 支在飛。
    # 0 = 交給 hostload 依當下餘裕決定（它有自己的上限），不再是整個 suite 一次送出。
    # 總工作量不變、覆蓋不變，只有牆鐘變長 —— 見 constants 裡的說明。
    #
    # 視窗由主機負載決定而非寫死：這台機器與 AETHER 叢集及約 40 位使用者共用，
    # 而 cluster.yaml 的配額是按獨佔主機寫的。hostload 只量「別人」的用量
    # （排除自己的 uid，否則我們自己的 sim 會把自己擋住），而且永遠會放行 ——
    # 它是節流閥不是開關，主機上有掛了數週的長期佔用，等待條件可能永不成立。
    window, why = hostload.advised_window(REGRESSION_MAX_INFLIGHT)
    print(f"[hostload] 派工 {len(suite)} 支：{why}", flush=True)
    queued = list(suite)
    if window > 0:
        refs = {nm: _submit(nm, el) for nm, el in queued[:window]}
        queued = queued[window:]
    else:
        refs = {nm: _submit(nm, el) for nm, el in queued}
        queued = []
    failing: List[Tuple[str, str, str]] = []
    done = 0
    for name, _ in suite:
        res = get(refs.pop(name))
        done += 1
        # 中途重新評估：Gate 的 837 支要跑一小時以上，主機在這段時間裡會變。
        # ps 一次約 20ms，所以每 50 支看一次的成本可以忽略。
        if queued and done % HOSTLOAD_RECHECK_EVERY == 0:
            new_window, new_why = hostload.advised_window(
                REGRESSION_MAX_INFLIGHT)
            if new_window != window:
                print(f"[hostload] {done}/{len(suite)} 已完成，視窗 "
                      f"{window} -> {new_window}：{new_why}", flush=True)
                window = new_window
        # 補到視窗大小為止（視窗變大就多送幾支，變小就這輪不送，靠自然消耗縮小）
        while queued and len(refs) < window:
            nxt_nm, nxt_el = queued.pop(0)
            refs[nxt_nm] = _submit(nxt_nm, nxt_el)
        if res.match:
            continue
        if res.crashed:
            reason = (f"the simulator died (exit {res.returncode}) after "
                      f"{res.matched} committed instructions -- no verdict")
        elif res.first_divergence:
            reason = (f"diverged after {res.matched} committed instructions: "
                      f"{res.first_divergence.get('line', '')} "
                      f"(spike {res.first_divergence.get('spike', '?')}, "
                      f"DUT {res.first_divergence.get('dut', '?')})")
        elif not res.completed:
            reason = (f"stopped after {res.matched} committed instructions "
                      f"without reaching the end of the program (timeout or "
                      f"hang), exit {res.returncode}")
        else:
            reason = (f"no divergence reported but the run did not count as "
                      f"a match: matched={res.matched} "
                      f"completed={res.completed} exit={res.returncode}")
        log = ""
        if len(failing) < REGRESSION_LOGS_KEPT and res.failing_trace_gz:
            try:
                log = gzip.decompress(res.failing_trace_gz).decode(
                    "utf-8", "replace")
            except (OSError, EOFError, zlib.error):     # noqa: BLE001
                log = ""
        failing.append((name, reason, log))
    return failing, [n for n, _ in suite]



#: Every confirmation pass this process ran, oldest first, so ``run()`` can
#: put the single-run-vs-confirmed numbers in the run summary without
#: threading a return value through four call sites.
_CONFIRM_LOG: List[Dict[str, object]] = []

#: How many repeatedly-cleared names chance may be expected to produce
#: before the alert is allowed to fire.  Below one name: if the background
#: would already put a name at this repeat count, the repeat says nothing.
_REPEAT_EXPECTED_MAX = 0.5

#: test name -> the confirmation passes that cleared it as a bridge flip.
#:
#: Across-pass memory is the whole point.  Within one pass an intermittent RTL
#: bug and a trace-bridge flip are the same observation, and unanimity throws
#: both away.  Across passes they need not be: r6 measured 82-87% of the
#: failing set reshuffling run to run, so an intermittent bug should keep
#: naming itself while flake keeps naming somebody else.
_CLEARED_HISTORY: Dict[str, List[str]] = {}

#: test name -> [reps it failed, reps it ran] pooled over every confirmation
#: pass, counting the confirmation reps only (never the first run, which is
#: what selected the test in the first place -- conditioning on it would put
#: a fail in every name's numerator for free).
_CLEARED_OBS: Dict[str, List[int]] = {}


def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p).  Exact; n is two digits at most."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    return sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
               for i in range(k, n + 1))


def _repeat_suspects() -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Which cleared tests look less like flake than the run's own flake does.

    This is the half-step past "record it and hope someone looks".  Unanimity
    cannot tell an intermittent real bug from a trace-bridge flip inside one
    pass, so the question is what separates them across the whole run.

    Counting how many passes cleared a name -- the obvious answer -- does not
    work.  With ~250 of 837 tests cleared per pass the per-test clear rate is
    ~0.30, so chance alone puts 837*0.30**3 = 23 names at three repeats, and
    a measurement of that (below) showed the naive counter naming ~26 pure
    flakes and missing the planted bug.

    What does work is data the confirmation already produced and used to
    throw away: HOW MANY reps each cleared test failed before it passed.
    Early exit makes that a run-length -- a flake flipping at p=0.31 fails
    1/(1-0.31) = 1.4 reps before passing, a test that genuinely misbehaves
    70% of the time fails 3.3 -- and those pool across passes into a real
    sample.  The null is estimated from the run itself (p_hat over every
    cleared test's pooled reps), so the alarm rate is calibrated to whatever
    the bridge is doing today rather than to r6's 0.31.

    A name fires when ``N * P(X >= fails | reps, p_hat)`` is below
    ``_REPEAT_EXPECTED_MAX`` -- fewer than half a name expected by chance
    across the whole family -- with ``REGRESSION_FLAKY_REPEAT_ALERT`` passes
    as a floor.

    Signal only.  Nothing re-judges on this: automatic re-judging would hand
    back the unpassable gate this whole mechanism exists to remove.

    Power is limited and the limit is the point.  The statistic separates a
    bug from flake only when the bug's per-run rate is well above the
    background and several passes have accumulated; a bug that fails 40% of
    runs against a 31% background will not surface, and an empty list means
    "nothing exceeded chance", never "no intermittent bug".

    Returns ``(suspects, context)``.  ``context`` is published either way,
    because "checked, nothing exceeded chance" and "never looked" must not
    read the same in the logs.
    """
    fails = sum(v[0] for v in _CLEARED_OBS.values())
    reps = sum(v[1] for v in _CLEARED_OBS.values())
    names = len(_CLEARED_OBS)
    passes = len(_CONFIRM_LOG)
    p_hat = (fails / reps) if reps else 0.0
    floor = max(1, REGRESSION_FLAKY_REPEAT_ALERT)
    context: Dict[str, object] = {
        "passes": passes, "distinct_names_cleared": names,
        "pooled_fail_reps": fails, "pooled_reps": reps,
        "background_fail_rate": round(p_hat, 4),
        "floor_passes": floor, "expected_max": _REPEAT_EXPECTED_MAX,
    }
    if names < 2 or reps < 2 or p_hat <= 0.0 or p_hat >= 1.0:
        return [], context
    suspects = []
    for name, (f, r) in _CLEARED_OBS.items():
        cleared_in = _CLEARED_HISTORY.get(name, [])
        if len(cleared_in) < floor:
            continue
        expected = names * _binom_tail(f, r, p_hat)
        if expected >= _REPEAT_EXPECTED_MAX:
            continue
        suspects.append({"test": name, "passes": len(cleared_in),
                         "cleared_in": cleared_in,
                         "failed_reps": f, "total_reps": r,
                         "expected_by_chance": round(expected, 4)})
    suspects.sort(key=lambda d: (d["expected_by_chance"], -d["failed_reps"]))
    context["suspects"] = len(suspects)
    return suspects, context


def _confirm_failures(artifact, failing: Sequence[Tuple[str, str, str]],
                      label: str, eligible: int = 0
                      ) -> Tuple[List[Tuple[str, str, str]], Dict[str, object]]:
    """Re-run only the failing tests and keep the ones that fail every time.

    The single most expensive mistake in this loop was treating one cosim run
    as a verdict.  r6 measured what that costs: on a byte-identical binary the
    full suite failed 281/248/249 tests on three consecutive runs, 82% of the
    failing set reshuffled between runs, and the tests that failed all three
    times were exactly the count independent per-run coin flips predict.  The
    trace bridge is also load-dependent -- ``machine_vdivu_vx-0`` diverges at
    instruction 55255 every time inside the 837-way harness and passes every
    time when run alone -- so this cannot be fixed by "run it somewhere else".

    So: rep 1 is the caller's run, which has already happened.  Each further
    rep re-runs *only* what is still failing, on the SAME build, and a test
    that passes any rep is dropped.  Unanimity, not majority, because with
    p~=0.31 per-run flips a majority of 3 would still confirm ~20% of pure
    flakes -- 50 false failures on a gate-sized set -- while unanimity over
    ``REGRESSION_CONFIRM_REPS`` leaves 0.31**(n-1) each.  The survivor set
    shrinks by ~3x a rep, so the extra cost is a fraction of one suite pass,
    not (n-1) passes.

    Two assumptions, both UNVERIFIED, written down so they are not later
    mistaken for measurements:

    1. *The bridge only manufactures spurious divergences; it never hides a
       real one.*  Clearing a test because it passed one rep is only sound if
       a pass cannot be an artefact.  Nothing has tested that.  If the bridge
       can mask a genuine mismatch, this function converts a real failure into
       a clean S2 -- and so did the single-run code before it, which never
       re-checked a pass either.
    2. *Flips are independent between reps.*  The arithmetic behind
       ``REGRESSION_CONFIRM_REPS`` (0.31**(n-1)) assumes that.  r6's three
       full-suite runs are consistent with independence -- the triple-failure
       count matched 837*0.31**3 -- but three samples cannot establish it.
       If flips are correlated (per test, per node, per binary), the residual
       false-confirmation rate is WORSE than the arithmetic predicts, and the
       visible symptom would be a gate that keeps failing on the same names:
       see ``_repeat_suspects``, which would then be firing on flake rather
       than on an intermittent bug.

    Returns ``(confirmed, stats)``.  ``confirmed`` is a subset of ``failing``
    in the same ``(name, reason, log)`` shape, with the reason and log taken
    from the most recent rep that produced one.  ``stats`` is what gets
    published: nothing here is allowed to disappear quietly.
    """
    names = [n for n, _, _ in failing]
    reps = max(1, REGRESSION_CONFIRM_REPS)
    stats: Dict[str, object] = {
        "label": label, "reps": reps, "reported": len(names),
        "confirmed": len(names), "cleared": [], "per_rep": [len(names)],
        "extra_runs": 0, "skipped": None,
    }
    if not names or reps == 1:
        stats["skipped"] = "disabled" if reps == 1 else "no failures"
        _CONFIRM_LOG.append(stats)
        return list(failing), stats
    if REGRESSION_CONFIRM_MAX and len(names) > REGRESSION_CONFIRM_MAX:
        # Not flake: a tree this broken is broken, and 500 x 6 cosims would
        # tell the agent something it can already see.
        stats["skipped"] = f"more than {REGRESSION_CONFIRM_MAX} failures"
        _event("regression_confirm_skipped", label=label,
               reported=len(names), cap=REGRESSION_CONFIRM_MAX)
        _CONFIRM_LOG.append(stats)
        return list(failing), stats

    _event("section_start", name=f"confirm:{label}")
    t0 = time.time()
    entry = {n: (n, r, log) for n, r, log in failing}
    cleared: Dict[str, int] = {}
    survivors = list(names)
    # (fails, reps) per name over the confirmation reps of THIS pass -- the
    # run-length that _repeat_suspects pools across passes.  The first run is
    # excluded on purpose: every name here failed it by construction.
    obs: Dict[str, List[int]] = {n: [0, 0] for n in names}
    for rep in range(2, reps + 1):
        if not survivors:
            break
        again, ran = _run_regression(artifact, select=survivors, sample=False)
        stats["extra_runs"] = int(stats["extra_runs"]) + len(ran)
        failed_this_rep = {n for n, _, _ in again}
        for n in ran:
            if n in obs:
                obs[n][1] += 1
                if n in failed_this_rep:
                    obs[n][0] += 1
        for n, r, log in again:
            # Keep the freshest evidence; an empty log this rep must not
            # overwrite a full one from an earlier rep.
            entry[n] = (n, r, log or entry[n][2])
        failed_now = {n for n, _, _ in again}
        ran_set = set(ran)
        # A name that did not run at all is kept, not cleared: silence is
        # not a pass.  (It should not happen -- baseline subtraction is
        # deterministic -- but "cleared because we never asked" is exactly
        # the failure mode this function exists to prevent.)
        still = [n for n in survivors if n in failed_now or n not in ran_set]
        for n in survivors:
            if n not in still:
                cleared[n] = rep
        survivors = still
        stats["per_rep"].append(len(survivors))

    confirmed = [entry[n] for n in names if n in survivors]
    stats["confirmed"] = len(confirmed)
    stats["cleared"] = sorted(cleared)
    stats["seconds"] = round(time.time() - t0, 1)
    for name in cleared:
        _CLEARED_HISTORY.setdefault(name, []).append(label)
        pooled = _CLEARED_OBS.setdefault(name, [0, 0])
        pooled[0] += obs[name][0]
        pooled[1] += obs[name][1]
    stats["eligible"] = eligible or len(names)
    suspects, context = _repeat_suspects()
    stats["repeat_suspects"] = suspects
    stats["repeat_context"] = context
    if suspects:
        _event("regression_flaky_repeat", label=label,
               count=len(suspects), threshold=context.get("threshold"),
               background=context.get("background_clear_rate"),
               tests=[d["test"] for d in suspects][:20])
    _event("regression_confirm", label=label, reported=len(names),
           confirmed=len(confirmed), reps=reps,
           extra_runs=stats["extra_runs"], seconds=stats["seconds"])
    _event("section_end", name=f"confirm:{label}",
           reported=len(names), confirmed=len(confirmed))
    _CONFIRM_LOG.append(stats)
    return confirmed, stats


def _confirm_note(stats: Dict[str, object]) -> str:
    """One paragraph, for the agent and for whoever reads the logs.

    The point is that the single-run number stays visible.  Reporting only
    the confirmed count would make this look like a suite that simply got
    less flaky, which is the one reading of it that is false.
    """
    reported, confirmed = stats["reported"], stats["confirmed"]
    if stats.get("skipped") == "no failures":
        return ""
    if stats.get("skipped"):
        return (f"(regression verdict: CONFIRMATION WAS SKIPPED -- "
                f"{stats['skipped']}. The {reported} failure(s) below are "
                f"SINGLE-RUN numbers: nothing here was re-run and none of it "
                f"is confirmed. At the measured ~30% per-run flip rate a "
                f"fraction of them are trace-bridge artefacts, but this pass "
                f"cannot tell you which.)\n\n" + _repeat_block(stats))
    cleared = reported - confirmed
    head = (f"(regression verdict: {reported} test(s) failed the first run; "
            f"re-running just those on the same binary, up to "
            f"{stats['reps']} times, left {confirmed} that failed EVERY "
            f"run. {cleared} passed on a re-run and are counted as trace-"
            f"bridge flips, not as your bug -- the cospike/DebugROB bridge "
            f"is nondeterministic and load-dependent, ~30% of tests flip per "
            f"run.")
    if confirmed:
        head += (" Only the confirmed ones are listed below and only they "
                 "are in the failing set.)")
    else:
        head += " Nothing survived, so S2 counts as clean.)"
    return head + "\n\n" + _repeat_block(stats)


def _repeat_block(stats: Dict[str, object]) -> str:
    """The loud half: cleared tests that do not behave like this run's flake.

    Unanimity's blind spot is an intermittent *real* bug, which it discards
    with the same shrug as a bridge flip.  What separates them is how long
    they keep failing before they let go, pooled over the run; see
    ``_repeat_suspects``.  This prints that and asks a human to re-run them
    standalone.  It re-judges nothing: doing so automatically would put back
    the unpassable gate.
    """
    suspects = stats.get("repeat_suspects") or []
    ctx = stats.get("repeat_context") or {}
    if not suspects:
        return ""
    lines = ["!! REPEATEDLY 'FLAKY' TESTS -- possible intermittent real bug",
             f"   {len(suspects)} test(s) were cleared as trace-bridge flips "
             f"but did not behave",
             f"   like this run's flake does. Background, measured on this "
             f"run's own",
             f"   {ctx.get('distinct_names_cleared')} cleared test(s): a "
             f"cleared test fails "
             f"{ctx.get('background_fail_rate')} of its re-runs",
             f"   ({ctx.get('pooled_fail_reps')}/{ctx.get('pooled_reps')} "
             f"reps over {ctx.get('passes')} confirmation pass(es)). These "
             f"kept failing far",
             "   longer than that, often enough that chance would produce "
             "fewer than",
             f"   {_REPEAT_EXPECTED_MAX} such name(s) across the whole suite. "
             f"They are NOT counted as failures and",
             "   nothing has been re-judged -- but re-run them on their own, "
             "outside the",
             "   parallel harness (the bridge is load-dependent), before the "
             "design is",
             "   called correct:"]
    for item in suspects[:20]:
        lines.append(f"   - {item['test']}: failed {item['failed_reps']} of "
                     f"{item['total_reps']} re-runs across "
                     f"{item['passes']} pass(es) "
                     f"({', '.join(item['cleared_in'][:6])}); "
                     f"expected by chance {item['expected_by_chance']}")
    if len(suspects) > 20:
        lines.append(f"   ... and {len(suspects) - 20} more.")
    return "\n".join(lines) + "\n\n"


#: How many times the gate may hand the design back to the agent and then
#: re-check it before giving up.
#:
#: Before r17 there was no re-check at all: the gate ran the directed sweep
#: and then the full RVV regression, and on a failure handed the agent one
#: debug ``_iterate``.  ``_iterate`` converges on *its own* criteria -- the
#: directed suite plus the sampled regression -- so in r16_0918_0129 a gate
#: that had failed 12 of 837 full-suite tests was "fixed" by a directed pass
#: of 92/92 and the run walked straight into Stage 3 with the regression
#: never re-run.  The gate now re-runs ``_gate`` in full after every debug
#: re-entry, so whatever it failed on is what it re-checks, and only a clean
#: directed sweep *and* a clean full suite pass it.  Each re-entry keeps its
#: own ``DEBUG_MAX_ITERS`` iteration cap; this bounds how many re-entries the
#: gate will pay for.
GATE_DEBUG_ROUNDS = int(os.environ.get("TITAN_GATE_DEBUG_ROUNDS", "3"))


#: The cosim simulator the most recent successful ``_run_s2`` built, together
#: with the digest of the tree it was elaborated from.
#:
#: ``_iterate`` returns to ``run()`` *immediately* after a clean S2, and the
#: first thing ``run()`` then does is call ``_gate`` -- whose regression half
#: needs exactly the simulator that S2 just built, from a tree nothing has
#: touched in between.  Rebuilding ``COSIM_CONFIG`` there would spend ~2.5
#: minutes producing a bit-identical binary.  Keyed on ``_tree_digest`` rather
#: than on an iteration counter so a stale artifact can never be handed to a
#: design that has moved on: if the digests disagree, the gate builds its own.
_LAST_COSIM: Dict[str, object] = {}


def _tree_digest(pg_opts) -> str:
    """Digest of the tree's current diff -- the identity of the design.

    One ``git diff`` on the chipyard node (the same call every iteration
    already makes before its build), which is what makes the ``_LAST_COSIM``
    reuse check cheap enough to be unconditional.
    """
    diff = get(nodes.collect_diff.options(**pg_opts)
               .chia_remote(CHIPYARD_PATH, extension=EXTENSION))
    return _digest(diff or "")


def _run_s2(dump: helpers.Dumper, label: str, attempt: int, pg_opts,
           run_id: str, rvv_failing_path: Optional[str] = None
           ) -> Tuple[Optional[List[Tuple[str, str, str]]],
                      Optional[str]]:
    """S2: build the cosim harness, then judge it against the RVV regression
    sample.  Shared by ``_iterate`` (``attempt`` >= 1) and ``_rtl_resume``
    (``attempt`` 0, ``label`` ``"impl_resume"``) so both dump, publish and
    grade S2 identically -- only ``label``/``attempt`` vary the filenames and
    the ``iter{attempt}`` publish leaf.

    Returns ``(failures, message)``. On a clean sample: ``([], None)``. On a
    cosim build failure: ``(None, message)`` -- the sentinel ``None`` marks
    that the failures are not regression failures at all, they never ran.
    On regression failures: ``(failures, message)`` with the ``(name,
    reason, log)`` list ``_run_regression`` returned.  ``message`` is never
    raised on; the caller decides what a build failure means for it.
    """
    _write_s2_cosim_config(pg_opts)
    try:
        # _STOCK_SPIKE, not None: None means "link whatever libriscv is
        # installed", which from Stage 0 onwards is the agent's IME model.
        cosim_artifact = get(nodes.build_saturn.options(**pg_opts)
                             .chia_remote(COSIM_CONFIG, _STOCK_SPIKE,
                                          extension=EXTENSION))
    finally:
        # Out of the tree again before anything else elaborates it.
        _write_s2_cosim_config(pg_opts, enable=False)
    dump.text(f"{label}_cosim_build_attempt{attempt}.stdout.txt",
              getattr(cosim_artifact, "stdout", "") or "")
    if not cosim_artifact.success:
        _event("build_failure", attempt=attempt, config=COSIM_CONFIG)
        message = helpers.format_build_failure(
            cosim_artifact, attempt + 1,
            log_path=_publish_logs(
                run_id, f"iter{attempt}_cosim", pg_opts,
                build_stdout=getattr(cosim_artifact, "stdout", "") or "",
                build_stderr=getattr(cosim_artifact, "stderr", "") or ""))
        return None, message

    # Hand the build to the final gate (see _LAST_COSIM).  Recorded before
    # the regression runs, because the gate wants the *build*, not a verdict.
    _LAST_COSIM.clear()
    _LAST_COSIM.update(artifact=cosim_artifact, tree=_tree_digest(pg_opts),
                       label=label, attempt=attempt)

    # NO LONGER A SINGLE-RUN VERDICT.  It was, and the comment here used to
    # justify it with a 12% flip rate; r6 measured the real one at ~30% per
    # test per run, which makes a clean single-run S2 unreachable -- and
    # since run() returns before the gate unless S2 is clean, it meant the
    # gate and Stage 3 never executed.  The first run still runs once (that
    # is the cost this was protecting); only its failures are re-run, which
    # is cheap because they are few and get fewer every rep.
    reported, ran = _run_regression(cosim_artifact)
    regression_failures, confirm = _confirm_failures(
        cosim_artifact, reported, f"{label}_attempt{attempt}",
        eligible=len(ran))
    dump.json(f"{label}_regression_attempt{attempt}_confirm.json", confirm)
    # Before the branch: a clean run has to clear the set too, or the agent's
    # `run_rvv_start("failing")` would keep re-running tests that now pass.
    # The *confirmed* set is what the agent inherits -- handing it 45 names
    # of which 43 are bridge flips is how r6 burned iterations.
    _merge_failing(rvv_failing_path, ran,
                   [n for n, _, _ in regression_failures])
    if regression_failures:
        stem = f"{label}_regression_attempt{attempt}"
        dump.json(f"{stem}.json",
                  [{"test": n, "reason": r} for n, r, _ in regression_failures])
        _dump_regression_logs(dump, stem, regression_failures)
        # The agent cannot see TITAN_LOG_ROOT, so the logs it is told to
        # read have to be copied into its container like the directed
        # ones.  Only the handful kept in full are worth publishing.
        regress_path = _publish_logs(
            run_id, f"iter{attempt}/regress", pg_opts,
            logs={n: log for n, _, log in regression_failures if log},
            extra_files={"failures.txt": "\n".join(
                f"{n}: {r}" for n, r, _ in regression_failures)})
        _event("regression_failure", attempt=attempt,
               count=len(regression_failures), reported=len(reported))
        message = _confirm_note(confirm) + helpers.format_regression_failure(
            attempt + 1, regression_failures, log_path=regress_path)
        return regression_failures, message

    if reported:
        # Clean only after confirmation.  Said out loud, because "S2 clean"
        # now means something weaker than it used to and the trace should
        # show which one it meant.
        _event("regression_all_flaky", attempt=attempt,
               reported=len(reported))
    _event("directed_pass", attempt=attempt)
    return [], None


def _orient(attempt: int, max_iters: int, run_id: str, label: str,
            knowledge_path: Optional[str], body: str, pg_opts,
            regression: str = "") -> str:
    """Wrap one iteration's evidence in everything a fresh session needs.

    With ``TITAN_RESUME_SESSION=0`` the agent has no transcript, so this
    message is its whole world: which iteration it is, what is already in the
    working tree (a diffstat, derived from the diff the loop collects anyway),
    what it wrote in its own notes, the result, and the directory of logs it
    can open for itself.  That is roughly 6kB where the replayed transcript
    was 29M cached tokens a call.
    """
    notes = ""
    if knowledge_path and os.path.exists(knowledge_path):
        try:
            with open(knowledge_path, encoding="utf-8") as fh:
                notes = fh.read()
        except OSError:
            notes = ""
    try:
        diff = get(nodes.collect_diff.options(**pg_opts)
                   .chia_remote(CHIPYARD_PATH, extension=EXTENSION))
    except Exception:                                       # noqa: BLE001
        diff = ""
    # RTL half only.  The tree also holds the Stage 0 Spike model, and the
    # whole reason that model can judge this agent's RTL in Stage 3 is that
    # its author never saw the RTL -- pasting the model's diffstat into the
    # RTL agent's prompt would not break that direction, but it hands one
    # agent a summary of the other's independent derivation for no reason.
    diff = _split_diff(diff)[1]
    # The regression block goes *first*, ahead of the iteration header: an
    # agent that has just made things much worse has one job before it reads
    # anything else, and burying that under the diffstat is how r8 spent
    # iterations debugging a tree that had already been better.
    return regression + helpers.format_iteration_message(
        attempt, max_iters, body,
        os.path.join(AGENT_LOG_DIR, run_id), helpers.diff_stat(diff), notes)


def _iterate(llm, tools_list, dump: helpers.Dumper, status_path: str,
             finish_tool, *, label: str, vlen: int, max_iters: int,
             first_message: Optional[str], pg_opts,
             first_note: Optional[str] = None,
             run_id: str = "run", knowledge_path: Optional[str] = None,
             run_tool=None, events_path: Optional[str] = None,
             failing_path: Optional[str] = None,
             rvv_failing_path: Optional[str] = None) -> Tuple[bool, object]:
    """Edit -> build -> S1 -> S2, until both are green or we run out of turns.

    S2 is inside this loop rather than after it: a directed pass that broke
    plain RVV is not a pass, and re-entering with regression feedback is the
    same motion as re-entering with a directed failure.
    """
    artifact = None
    message = first_message
    llm_failures = 0
    # The run's high-water mark, and the block that tells the agent it has
    # fallen off it.  Nothing here reverts the tree -- the agent decides that
    # -- but the diff that scored best is one `git apply` away, published
    # where its bash tool can read it.
    best = {"passed": -1, "attempt": 0}
    regression = ""
    stall = _StallTracker(label)
    for attempt in range(1, max_iters + 1):
        _event("section_start", name=f"{label}:iter{attempt}")
        # No guard on `message`: the implement stage enters with none, and a
        # guard here would skip the agent's first turn entirely -- building
        # and testing the untouched tree, then handing the agent a debug
        # prompt about failures it never had a chance to cause.  The ternary
        # is what chooses the prompt; _model_stage has the same shape.
        finish_tool.reset()
        if run_tool is not None:
            # The budget is per iteration, not per run.
            run_tool.reset_budget()
        # `first_note` rides on the implement prompt rather than replacing
        # it: after a --rtl-diff resume the task is still "implement Zvvm",
        # but the tree is not empty and the agent has to be told so before it
        # starts writing code it has already written.
        # A fresh session (the default from r5) remembers nothing, so the
        # orientation -- which iteration this is, what is already in the
        # tree, what the notes say, where the logs are -- has to ride on
        # every message rather than on the first one only.
        if attempt == 1 and first_message is None:
            cli = llm_mod.implement(
                llm, tools_list,
                _orient(attempt, max_iters, run_id, label,
                        knowledge_path, first_note or "", pg_opts,
                        regression))
        else:
            cli = llm_mod.debug(
                llm, tools_list,
                _orient(attempt, max_iters, run_id, label,
                        knowledge_path, message or "", pg_opts,
                        regression))
        helpers.dump_llm(dump, f"{label}_llm_attempt{attempt}", cli)

        # The turn is over, but a run_directed job it started may not be.
        # In `-p` mode the session ends the moment the model stops calling
        # tools, and the CLI's 120s MCP backgrounding makes that easy to do
        # by accident (r8 iteration 3).  A live job holds the placement
        # group's single `chipyard`, so the loop cannot build on top of it:
        # wait it out first, and mark its record so the trace shows the
        # agent never read the answer.
        orphans = run_tool.drain() if run_tool is not None else []
        orphan_seqs = {j.get("seq") for j in orphans}
        if orphans:
            _event("agent_run_orphaned", attempt=attempt,
                   jobs=[j.get("job_id") for j in orphans],
                   state=orphans[0].get("state"))
        seen: set = set()
        if events_path:
            for rec in titan_tools.drain_agent_runs(events_path):
                name = rec.pop("event", "agent_directed")
                if rec.get("seq") in orphan_seqs:
                    rec["orphaned"] = True
                    seen.add(rec.get("seq"))
                _event(name, attempt=attempt, **rec)
        for job in orphans:
            if job.get("seq") in seen:
                continue
            _event("agent_rvv" if job.get("kind") == "rvv"
                   else "agent_directed",
                   attempt=attempt, seq=job.get("seq"),
                   tests=job.get("tests"), orphaned=True,
                   build=job.get("state"))

        # A turn that failed outright did not edit anything, so building and
        # judging the tree again would grade the previous attempt twice and
        # pay for it.  One is a flake; LLM_MAX_CONSECUTIVE_FAILURES in a row
        # is a configuration the run cannot fix by iterating.
        if _llm_call_failed(cli):
            llm_failures += 1
            _event("llm_failed", stage=label, attempt=attempt,
                   consecutive=llm_failures,
                   limit=LLM_MAX_CONSECUTIVE_FAILURES,
                   result=str(getattr(cli, "result", ""))[:400])
            if llm_failures >= LLM_MAX_CONSECUTIVE_FAILURES:
                _event("run_abort", reason="llm_failed", stage=label,
                       attempt=attempt)
                return False, artifact
            continue
        llm_failures = 0

        # Snapshot before building, every iteration.  nodes.collect_diff
        # covers the root repo and every submodule in CHIPYARD_DIFF_SUBMODULES
        # -- including generators/rocket-chip, which is not optional: the
        # vtype CSR is defined there, so a diff without it cannot rebuild the
        # design it claims to describe.
        diff = get(nodes.collect_diff.options(**pg_opts)
                   .chia_remote(CHIPYARD_PATH, extension=EXTENSION))
        if diff:
            dump.text(f"{label}_diff_attempt{attempt}.diff", diff)

        # Directed tests run on the design WITHOUT the cosim harness.  The
        # cosim config links the image's stock libriscv, which has never
        # heard of Zvvm: the first vtype write with a lambda field diverges
        # from that Spike, cospike aborts, and every program classifies as
        # "trap" before it can print a verdict (r3_0907_1742 burned two S1
        # iterations, 27/27 trap each, exactly this way).  The self-checking
        # programs need no golden model -- that is the point of S1.
        artifact = get(nodes.build_saturn.options(**pg_opts)
                       .chia_remote(DIRECTED_CONFIG, extension=EXTENSION))
        dump.text(f"{label}_build_attempt{attempt}.stdout.txt",
                  getattr(artifact, "stdout", "") or "")
        if not artifact.success:
            _event("build_failure", attempt=attempt)
            stall.observe(attempt, {"build_failure": 1}, ())
            path = _publish_logs(
                run_id, f"iter{attempt}", pg_opts,
                build_stdout=getattr(artifact, "stdout", "") or "",
                build_stderr=getattr(artifact, "stderr", "") or "")
            message = helpers.format_build_failure(artifact, attempt + 1,
                                                   log_path=path)
            continue

        tests, geometries = _build_directed(dump, vlen, full_sweep=False)
        outcomes, logs = _run_directed(artifact, tests)
        _dump_sim_logs(dump, f"{label}_directed_attempt{attempt}", outcomes, logs)
        titan_tools.write_status(status_path, outcomes, geometries)
        summary = helpers.summarize(outcomes)
        dump.json(f"{label}_directed_attempt{attempt}.json", summary)
        _event("directed_iter", attempt=attempt, **summary["counts"])

        # `diff` is the tree exactly as this iteration graded it, snapshotted
        # a few lines above and before the build -- so the diff filed as
        # "best" is the one that produced the score, not a later edit of it.
        passed = int(summary["counts"].get("pass", 0))
        total = int(summary.get("total") or len(tests))
        if passed > best["passed"]:
            best.update(passed=passed, attempt=attempt)
            best_rtl = _split_diff(diff)[1]
            dump.text(f"{label}_best_rtl.diff", best_rtl)
            with open(os.path.join(dump.out_dir, "best_rtl.diff"), "w",
                      encoding="utf-8") as fh:
                fh.write(best_rtl)
            _publish_logs(run_id, ".", pg_opts,
                          extra_files={"best_rtl.diff": best_rtl})
            _event("best_result", attempt=attempt, passed=passed, total=total)
            regression = ""
        elif passed <= best["passed"] - REGRESSION_ALERT_DELTA:
            best_path = os.path.join(AGENT_LOG_DIR, run_id, "best_rtl.diff")
            _event("regression", attempt=attempt, passed=passed, total=total,
                   best=best["passed"], best_attempt=best["attempt"],
                   delta=best["passed"] - passed)
            regression = helpers.format_regression_block(
                best["passed"], best["attempt"], passed, total, best_path)
        else:
            regression = ""

        if failing_path:
            _write_failing(failing_path, summary["failing"])
        path = _publish_logs(
            run_id, f"iter{attempt}", pg_opts,
            build_stdout=getattr(artifact, "stdout", "") or "",
            logs=logs, status_path=status_path, summary=summary)

        if summary["failing"]:
            stalled = stall.observe(attempt, summary["counts"],
                                    summary["failing"])
            if stalled:
                _stall(stalled)
                return False, artifact
            message = helpers.format_directed_failure(attempt + 1, outcomes,
                                                      logs, log_path=path)
            continue

        # S2 needs the cosim harness (the RVV suite is judged by lockstep
        # against stock libriscv, which is fine: those tests never touch
        # Zvvm).  Built only once directed passes, so most iterations pay
        # for one elaboration, not two.  The harness config is the loop's,
        # written into the tree here -- see _write_s2_cosim_config and
        # _run_s2 (shared with _rtl_resume's attempt-0 pre-run).
        s2_fail, frag = _run_s2(dump, label, attempt, pg_opts, run_id,
                                rvv_failing_path=rvv_failing_path)
        if frag is not None:
            # A cosim build failure (s2_fail None) is its own outcome.
            stalled = stall.observe(
                attempt, summary["counts"], summary["failing"],
                ["<cosim build failure>"] if s2_fail is None
                else [n for n, _, _ in s2_fail])
            if stalled:
                _stall(stalled)
                return False, artifact
            message = frag
            continue

        return True, artifact
    return False, artifact


def _cosim_is_current(cache: Dict[str, object], tree: str):
    """The reuse rule, alone and side-effect free so it can be tested.

    Returns the cached cosim artifact when it was built from ``tree``, else
    ``None``.  A missing artifact, a missing or empty cached digest, and a
    digest that disagrees all mean "build your own" -- the failure mode worth
    designing against is grading a design with a simulator elaborated from a
    different one, so anything short of an exact match is a rebuild.
    """
    artifact = cache.get("artifact")
    cached_tree = cache.get("tree")
    if artifact is None or not cached_tree or not tree:
        return None
    return artifact if cached_tree == tree else None


def _gate_cosim(dump: helpers.Dumper, pg_opts) -> Tuple[object, bool]:
    """The cosim simulator the gate regression runs on.

    Reuses the one ``_run_s2`` built for the iteration that just converged
    when the tree has not moved since (the ordinary case: ``_iterate``
    returns straight into the gate), and builds one otherwise.  Returns
    ``(artifact, reused)``; ``artifact`` may be an unsuccessful build, which
    the caller reports the way ``_run_s2`` reports its own.
    """
    tree = _tree_digest(pg_opts)
    cached = _cosim_is_current(_LAST_COSIM, tree)
    if cached is not None:
        return cached, True
    _write_s2_cosim_config(pg_opts)
    try:
        cosim_artifact = get(nodes.build_saturn.options(**pg_opts)
                             .chia_remote(COSIM_CONFIG, _STOCK_SPIKE,
                                          extension=EXTENSION))
    finally:
        _write_s2_cosim_config(pg_opts, enable=False)
    dump.text("gate_cosim_build.stdout.txt",
              getattr(cosim_artifact, "stdout", "") or "")
    return cosim_artifact, False


def _gate_regression(dump: helpers.Dumper, pg_opts, run_id: str,
                     rvv_failing_path: Optional[str] = None
                     ) -> Tuple[bool, Optional[str]]:
    """S2 at the gate: the WHOLE riscv-vector-tests suite, no sampling.

    Why this is here.  Per-iteration S2 grades ``REGRESSION_SAMPLE`` tests --
    a deterministic stride, cheap enough to pay every iteration -- and
    ``constants.REGRESSION_SAMPLE``'s own comment has always said "the final
    gate should still see the whole suite".  It did not: until this, ``_gate``
    ran the directed sweep and nothing else, so a design could converge with
    691 of the 841 regression tests never once executed against it.

    ``sample=False`` is the whole point.  Baseline subtraction stays on, so
    the tests in ``REGRESSION_BASELINE_PATH`` -- the ones that already fail on
    a pristine Saturn against stock Spike -- are never dispatched and cannot
    fail the gate; anything outside that list that fails, does.

    Returns ``(ok, message)``.  ``message`` is the agent-facing feedback for a
    failure (a build failure or a real regression), ``None`` on success, and
    is shaped by the same ``helpers.format_regression_failure`` a
    per-iteration S2 failure uses -- so the debug re-entry reads identically
    whichever gate produced it.
    """
    _event("section_start", name="gate_regression")
    t0 = time.time()
    cosim_artifact, reused = _gate_cosim(dump, pg_opts)
    if not cosim_artifact.success:
        _event("build_failure", attempt="gate", config=COSIM_CONFIG)
        message = helpers.format_build_failure(
            cosim_artifact, "gate",
            log_path=_publish_logs(
                run_id, "gate_cosim", pg_opts,
                build_stdout=getattr(cosim_artifact, "stdout", "") or "",
                build_stderr=getattr(cosim_artifact, "stderr", "") or ""))
        _event("section_end", name="gate_regression", build_failure=True,
               seconds=round(time.time() - t0, 1))
        return False, message

    # COVERAGE IS UNCHANGED: still every test in the suite, still unsampled,
    # still baseline-subtracted.  What changed is the verdict.  Every test
    # here used to run exactly once, and the cospike/DebugROB trace bridge
    # flips ~30% of tests per run on an unchanged binary (r6: 281/248/249 of
    # 837 over three runs, 82% of the set reshuffling), so a single-run gate
    # over 837 tests reports ~250 failures no matter how correct the design
    # is -- it could never pass and it was never reached.  So the failures,
    # and only the failures, are re-run to unanimity; see _confirm_failures.
    # This does not relax what the gate tests, only how many observations it
    # needs before it calls one of them a bug.
    reported, ran = _run_regression(cosim_artifact, sample=False)
    failing, confirm = _confirm_failures(cosim_artifact, reported, "gate",
                                         eligible=len(ran))
    wall = round(time.time() - t0, 1)
    # Same merge rule as _run_s2: only what ran changes, so the agent's
    # rvv failing set is the gate's verdict and not a partial view of it.
    _merge_failing(rvv_failing_path, ran, [n for n, _, _ in failing])
    dump.json("gate_regression.json",
              {"ran": len(ran), "failing": len(failing),
               "reported_single_run": len(reported), "confirm": confirm,
               "reused_build": reused, "seconds": wall,
               "baseline_path": REGRESSION_BASELINE_PATH,
               "tests": [{"test": n, "reason": r} for n, r, _ in failing],
               "cleared_as_flaky": confirm.get("cleared", []),
               "repeat_suspects": confirm.get("repeat_suspects", []),
               "repeat_context": confirm.get("repeat_context", {}),
               "field_notes": {
                   "reported_single_run":
                       "failures of the FIRST run of the full suite -- the "
                       "number the old single-run gate would have reported.",
                   "failing":
                       "of those, the ones that failed every one of "
                       f"{REGRESSION_CONFIRM_REPS} reps of the same binary. "
                       "This is the gate's verdict.",
                   "cleared_as_flaky":
                       "reported by the first run and passed at least one "
                       "re-run, so counted as a cospike/DebugROB trace-bridge "
                       "flip. ASSUMPTION, unverified: the bridge only "
                       "manufactures spurious divergences and never hides a "
                       "real one -- a passing rep is therefore taken as "
                       "proof the test is not broken. Nothing has tested "
                       "that assumption; if it is false, a real failure can "
                       "be cleared here (the single-run gate never "
                       "re-checked a pass either, so this is not a new "
                       "exposure, but it is not a measured one).",
                   "repeat_suspects":
                       "cleared-as-flaky tests that kept failing their "
                       "re-runs longer than this run's OWN flake does: the "
                       "cut is where chance would produce fewer than "
                       f"{_REPEAT_EXPECTED_MAX} such names, with a floor of "
                       f"{max(1, REGRESSION_FLAKY_REPEAT_ALERT)} passes. A "
                       "plain repeat count does not work (at a ~0.30 clear "
                       "rate chance alone puts ~23 of 837 names at three "
                       "repeats), so the statistic is reps-to-first-pass "
                       "pooled over passes. These may be intermittent REAL "
                       "bugs that unanimity discarded. Signal only: nothing "
                       "was re-judged. Re-run them standalone, outside the "
                       "parallel harness. An empty list means 'checked, "
                       "nothing exceeded chance' -- NOT that an intermittent "
                       "bug is excluded: the test has power only when the "
                       "bug's per-run rate is well above the background and "
                       "several passes have accumulated.",
                   "repeat_context":
                       "the null this was judged against, measured on this "
                       "run: confirmation passes so far, distinct names ever "
                       "cleared, and the pooled fraction of re-runs a "
                       "cleared test fails.",
                   "independence":
                       "the confirmation arithmetic assumes flips are "
                       "independent between reps. r6's three full-suite runs "
                       "are consistent with that but cannot establish it; if "
                       "flips are correlated the residual false-confirmation "
                       "rate is worse than 0.31**(reps-1) per test."}})
    _event("gate_regression", ran=len(ran), failing=len(failing),
           reported=len(reported), reused_build=reused, seconds=wall)
    if not failing:
        _event("section_end", name="gate_regression", ran=len(ran), failing=0,
               seconds=wall)
        return True, None

    _dump_regression_logs(dump, "gate_regression", failing)
    # Published exactly the way the per-iteration S2 publishes: its own leaf
    # under the run, so the gate's evidence lands beside the iterations' and
    # neither `best_rtl.diff` (published at the run root) nor any earlier
    # iteration's logs are disturbed.
    regress_path = _publish_logs(
        run_id, "gate/regress", pg_opts,
        logs={n: log for n, _, log in failing if log},
        extra_files={"failures.txt": "\n".join(
            f"{n}: {r}" for n, r, _ in failing)})
    message = _confirm_note(confirm) + helpers.format_regression_failure(
        "gate (full suite)", failing, log_path=regress_path)
    _event("section_end", name="gate_regression", ran=len(ran),
           failing=len(failing), seconds=wall)
    return False, message


def _gate(artifact, dump: helpers.Dumper, status_path: str, vlen: int,
          pg_opts=None, run_id: str = "run",
          rvv_failing_path: Optional[str] = None
          ) -> Tuple[bool, Optional[str]]:
    """The final gate: the full directed sweep, then the full RVV regression.

    Two halves, in this order because the cheap one localises the fault: the
    directed sweep with partial N included (``full_sweep=True``), and then --
    only if that passes -- every riscv-vector-test, unsampled.  Returns
    ``(ok, message)``; ``message`` is ``None`` when the directed half is what
    failed (the caller already has better words for that case) and carries the
    regression feedback when the S2 half failed.

    ``pg_opts=None`` skips the regression half; it exists for callers with no
    placement group (tests, tools) and is never what ``run()`` passes.
    """
    _event("section_start", name="gate")
    tests, geometries = _build_directed(dump, vlen, full_sweep=True)
    outcomes, _ = _run_directed(artifact, tests)
    titan_tools.write_status(status_path, outcomes, geometries)
    summary = helpers.summarize(outcomes)
    dump.json("gate.json", summary)
    _event("section_end", name="gate", **summary["counts"])
    if summary["failing"]:
        return False, None
    if pg_opts is None:
        return True, None
    return _gate_regression(dump, pg_opts, run_id, rvv_failing_path)


# ---------------------------------------------------------------------------
# stage 0: the Spike model
# ---------------------------------------------------------------------------

def _run_on_spike(tests: Sequence[Tuple[str, bytes]], vlen: int, pg_opts
                  ) -> Tuple[List[Tuple[str, helpers.Outcome]], Dict[str, str]]:
    """Run the directed programs on Spike alone -- no DUT, none exists yet.

    Dispatched *inside* the chipyard placement group: the group reserves the
    build node's whole ``chipyard`` resource, so a ``spike_run`` scheduled
    outside it can never be placed (r2_0907_1327 sat on 27 pending tasks for
    four hours).  The bundle carries 4 CPUs, so two runs overlap
    (chipyard 0.4 each).
    """
    refs = {name: nodes.spike_run.options(**pg_opts)
                       .chia_remote(elf, name, SIM_WORK_DIR,
                                    vlen=vlen, extension=EXTENSION)
            for name, elf in tests}
    outcomes, logs = [], {}
    for name, _ in tests:
        result = get(refs[name])
        outcomes.append((name, helpers.classify_run(result)))
        logs[name] = result.log or ""
    # Same whole-run reconciliation as the RTL path: a model that answers a
    # lambda request with a larger value has the same case to answer.
    return helpers.reconcile_geometry(outcomes), logs


def _model_stage(dump: helpers.Dumper, status_path: str, tool_list,
                 finish_tool, vlen: int, max_iters: int, pg_opts,
                 first_note: Optional[str] = None):
    """Stage 0: an agent implements Zvvm in Spike, judged by the S1 programs.

    Runs before any RTL exists.  That is not a scheduling convenience -- it is
    what makes the model an independent judge in Stage 3.  An agent cannot
    copy an implementation that has not been written yet, so the prompt's rule
    against reading the RTL is backed by the fact that there is nothing there
    to read.

    Returns (converged, SpikeBuildArtifact).  The artifact is the golden model
    Stage 3 links into the simulator.

    ``first_note`` is how ``--model-seed`` gets in: the tree already holds a
    converged model for the previous round's instructions, and the note --
    built by ``_seed_model`` from a real run of the directed suite against it
    -- rides on the implement prompt rather than replacing it.  Everything
    after attempt 1 is the ordinary loop, and the convergence rule (all pass,
    skip rate under MODEL_MAX_SKIP_FRACTION, inside max_iters) is unchanged:
    a seeded run is not held to a weaker standard than a cold one.
    """
    _event("section_start", name="model")
    llm = llm_mod.make_model_llm()
    tests, geometries = _build_directed(dump, vlen, full_sweep=False)
    message = None
    artifact = None
    llm_failures = 0
    stall = _StallTracker("model")

    for attempt in range(1, max_iters + 1):
        finish_tool.reset()
        cli = (llm_mod.implement_model(llm, tool_list, first_note or "")
               if message is None
               else llm_mod.debug_model(llm, tool_list, message))
        helpers.dump_llm(dump, f"model_llm_attempt{attempt}", cli)

        # Same guard as _iterate: a turn that failed outright edited nothing,
        # and rebuilding Spike to re-judge the same tree costs a build an
        # iteration for no new information.
        if _llm_call_failed(cli):
            llm_failures += 1
            _event("llm_failed", stage="model", attempt=attempt,
                   consecutive=llm_failures,
                   limit=LLM_MAX_CONSECUTIVE_FAILURES,
                   result=str(getattr(cli, "result", ""))[:400])
            if llm_failures >= LLM_MAX_CONSECUTIVE_FAILURES:
                _event("run_abort", reason="llm_failed", stage="model",
                       attempt=attempt)
                return False, artifact
            continue
        llm_failures = 0

        artifact = get(nodes.build_spike.options(**pg_opts)
                       .chia_remote(extension=EXTENSION))
        dump.text(f"model_build_attempt{attempt}.stdout.txt",
                  getattr(artifact, "stdout", "") or "")
        if not artifact.success:
            _event("model_build_failure", attempt=attempt)
            stall.observe(attempt, {"build_failure": 1}, ())
            message = helpers.format_build_failure(artifact, attempt + 1)
            continue

        outcomes, logs = _run_on_spike(tests, vlen, pg_opts)
        _dump_sim_logs(dump, f"model_attempt{attempt}", outcomes, logs)
        titan_tools.write_status(status_path, outcomes, geometries)
        summary = helpers.summarize(outcomes)
        dump.json(f"model_attempt{attempt}.json", summary)
        _event("model_iter", attempt=attempt, **summary["counts"])

        skipped = sum(1 for _, o in outcomes if o.kind == "skip")
        converging = (not summary["failing"] and not (
            outcomes and skipped / len(outcomes) > MODEL_MAX_SKIP_FRACTION))
        if not converging:
            stalled = stall.observe(attempt, summary["counts"],
                                    summary["failing"])
            if stalled:
                _stall(stalled)
                return False, artifact

        if summary["failing"]:
            message = helpers.format_directed_failure(attempt + 1, outcomes,
                                                      logs)
            continue

        # Everything passed -- but a model that clamps LAMBDA everywhere
        # passes by declining to be tested, and would then agree with any RTL
        # in Stage 3 for the same empty reason.
        if outcomes and skipped / len(outcomes) > MODEL_MAX_SKIP_FRACTION:
            _event("model_skip_rate", attempt=attempt, skipped=skipped)
            message = (
                f"# Attempt {attempt + 1}: your model declines too much\n\n"
                f"{skipped} of {len(outcomes)} programs reported SKIP, which "
                f"means your model clamped `lambda` away from what they asked "
                f"for. Nothing failed, but nothing much was tested either: a "
                f"model that supports one geometry would agree with any "
                f"implementation, for no reason.\n\n"
                f"Read the status file for which geometries you are "
                f"declining, then re-read the spec's rules on which LAMBDA "
                f"values are architecturally permissible for a given "
                f"(VLEN, SEW). Support the ones you can compute correctly.\n")
            continue

        _event("model_pass", attempt=attempt, digest=artifact.digest[:12],
               skipped=skipped)
        return True, artifact

    return False, artifact


# ---------------------------------------------------------------------------
# stage 3
# ---------------------------------------------------------------------------

def _stress(artifact, pool_dir: str, deadline: float, dump: helpers.Dumper
            ) -> Tuple[Optional[Tuple[str, str]], int]:
    """Run the pool under lockstep cosimulation until something diverges.

    This is where the two independent derivations of the specification meet.
    The RTL and the Spike model were written by agents that never saw each
    other's work, from the same text; a divergence here means at least one of
    them misread it, and which one is a question the spec answers.  That is a
    strictly stronger signal than either agreeing with itself.

    The programs also still carry their own self-check, but cosim is what this
    stage is for -- and CosimResult reports the divergence, not the program's
    stdout, so the lockstep verdict is the one read here.
    """
    state = get(db_node.pool_state.chia_remote(pool_dir, extension=EXTENSION))
    pending = [n for n, (status, _) in state.items() if status == "pending"]
    done = 0
    for name in pending:
        if time.time() > deadline:
            break
        elf = get(db_node.pool_elf.chia_remote(pool_dir, name,
                                               extension=EXTENSION))
        res = get(nodes.cosim_run.chia_remote(artifact, elf, name, 0,
                                              SIM_WORK_DIR,
                                              extension=EXTENSION))
        done += 1
        if not res.match:
            get(db_node.pool_mark.chia_remote(pool_dir, name, "failed",
                                              extension=EXTENSION))
            if res.failing_trace_gz:
                dump.bytes(f"stress_divergence_{name}.trace.gz",
                           res.failing_trace_gz)
            dump.json(f"stress_divergence_{name}.json",
                      {"first_divergence": res.first_divergence,
                       "matched": res.matched, "crashed": res.crashed})
            _event("stress_divergence", name=name, matched=res.matched)
            detail = (f"RTL and Spike model diverged after {res.matched} "
                      f"committed instructions: {res.first_divergence}")
            return (name, detail), done
        get(db_node.pool_mark.chia_remote(pool_dir, name, "passed",
                                          extension=EXTENSION))
    return None, done


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

#: The one path prefix that separates "the model" from "the RTL" in a
#: collect_diff output.  Everything under it is Spike; everything else is
#: the design.  The two resume flags split on exactly this line, in opposite
#: directions, so applying both reproduces the original diff exactly and
#: neither can touch a file the other also touches.
_MODEL_PATH_PREFIX = SPIKE_SRC_REL.strip("/")


def _split_diff(diff: str) -> Tuple[str, str]:
    """Split a chipyard-rooted `collect_diff` output into (model, rtl).

    `model` is every ``diff --git`` section under riscv-isa-sim; `rtl` is
    everything else -- saturn, rocket-chip, shuttle and the chipyard root.
    Split per file section rather than per hunk: a unified diff's hunks
    belong to the file header above them, and a header without its hunks (or
    hunks without their header) is not a diff `git apply` will take.
    """
    model: List[str] = []
    rtl: List[str] = []
    current: Optional[List[str]] = None
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            # "diff --git a/<path> b/<path>": classify on the a-side path,
            # which is present even for a file the agent created (git emits
            # a/<path> for /dev/null adds too).
            parts = line.split()
            path = parts[2][2:] if len(parts) > 3 and parts[2].startswith("a/") \
                else (parts[-1][2:] if parts[-1].startswith("b/") else "")
            current = model if path.startswith(_MODEL_PATH_PREFIX) else rtl
        if current is None:
            # Preamble before the first file header (there is none in a
            # collect_diff output, but a hand-edited diff may have one).
            continue
        current.append(line)
    return "".join(model), "".join(rtl)


def _diff_files(diff: str) -> List[str]:
    """The b-side paths a diff touches, for logging."""
    out = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            out.append(line.split()[-1][2:])
    return out


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _rtl_resume(dump: helpers.Dumper, status_path: str, vlen: int,
                diff_path: str, pg_opts, run_id: str = "run",
                failing_path: Optional[str] = None,
                rvv_failing_path: Optional[str] = None
                ) -> Tuple[str, str, object]:
    """Resume S1 from a previous run's RTL edits.

    Applies the non-riscv-isa-sim part of *diff_path* to the freshly reset
    tree, then builds and runs the directed suite on it *before* the first
    LLM call.  That pre-run is the point: without it the agent opens on a
    tree full of code it did not write and no idea what that code scores, so
    its first turn is spent rediscovering the state r4 already paid five
    iterations for.  It counts as attempt 0 and dumps like any other attempt.

    Returns (note for the first prompt, digest of the applied diff,
    clean artifact).  The third element is the directed build when attempt 0
    passed directed AND the S2 sample came back clean (cosim built, nothing
    confirmed failing) -- the evidence ``run()`` needs to go straight to the
    gate -- and ``None`` otherwise.
    """
    with open(diff_path) as f:
        rtl = _split_diff(f.read())[1]
    if not rtl.strip():
        raise RuntimeError(
            f"--rtl-diff {diff_path}: no non-riscv-isa-sim hunks in this "
            f"diff -- there is nothing to resume from.")
    digest = _digest(rtl)
    dump.text("rtl_resume.diff", rtl)
    err = get(nodes.apply_diff.options(**pg_opts)
              .chia_remote(rtl, extension=EXTENSION))
    if err:
        raise RuntimeError(f"--rtl-diff {diff_path}: git apply failed:\n{err}")
    _event("rtl_resume", source=diff_path, digest=digest,
           files=len(_diff_files(rtl)))

    # Attempt 0: the same build-and-judge every S1 iteration does, so the
    # feedback the agent opens on is real rather than remembered.
    artifact = get(nodes.build_saturn.options(**pg_opts)
                   .chia_remote(DIRECTED_CONFIG, extension=EXTENSION))
    dump.text("impl_resume_build_attempt0.stdout.txt",
              getattr(artifact, "stdout", "") or "")
    done, new = helpers.instruction_scope()
    head = ("A previous run's RTL changes are already applied in the working "
            f"tree (see `git diff` in {CHIPYARD_PATH}, and in its "
            "generators/* submodules). You are continuing that work, not "
            "starting from an empty tree: read those changes before you edit "
            "anything, and do not re-derive what is already there.\n\n"
            "Their last directed result was:\n")
    if not artifact.success:
        _event("rtl_resume_build_failure")
        return head + helpers.format_build_failure(
            artifact, 1, log_path=_publish_logs(
                run_id, "iter0", pg_opts,
                build_stdout=getattr(artifact, "stdout", "") or "",
                build_stderr=getattr(artifact, "stderr", "") or "")), digest, None

    tests, geometries = _build_directed(dump, vlen, full_sweep=False)
    outcomes, logs = _run_directed(artifact, tests)
    _dump_sim_logs(dump, "impl_resume_directed_attempt0", outcomes, logs)
    titan_tools.write_status(status_path, outcomes, geometries)
    summary = helpers.summarize(outcomes)
    dump.json("impl_resume_directed_attempt0.json", summary)
    _event("rtl_resume_directed", attempt=0, **summary["counts"])
    if failing_path:
        _write_failing(failing_path, summary["failing"])
    path = _publish_logs(run_id, "iter0", pg_opts,
                         build_stdout=getattr(artifact, "stdout", "") or "",
                         logs=logs, status_path=status_path, summary=summary)
    if summary["failing"]:
        # A failing attempt 0 is the normal way a round-two run starts, not
        # an anomaly: the suite has grown tests for an instruction the tree
        # does not implement yet.  Say which is which before the evidence,
        # or the agent reads a regression into a design that never claimed
        # to cover these geometries.  S2 is deliberately NOT run here: the
        # RVV regression cannot be interpreted until directed is clean, and
        # an elaboration of COSIM_CONFIG costs more than the first turn
        # would gain from it.
        body = helpers.format_directed_failure(1, outcomes, logs,
                                               log_path=path)
        scope = helpers.format_rtl_seed_scope(done, new, failing=True)
        return scope + "\n" + head + body, digest, None

    head = helpers.format_rtl_seed_scope(done, new, failing=False) + "\n" + head
    body = (f"every directed test passed or skipped: {summary['counts']}. "
            f"Confirm that yourself before changing anything.\n\n"
            f"Their last S2 regression result was:\n")
    # Directed is clean -- do what _iterate does the moment it sees the
    # same thing, rather than making the agent's first turn "nothing
    # failing" and leaving the S2 regression feedback a full iteration
    # away.  Same helper, same dump labels (impl_resume_ prefix via
    # label="impl_resume", attempt=0), same publish paths.
    regression_failures, frag = _run_s2(dump, "impl_resume", 0, pg_opts,
                                        run_id,
                                        rvv_failing_path=rvv_failing_path)
    _event("rtl_resume_regression", attempt=0,
           cosim_build_ok=regression_failures is not None,
           failing=len(regression_failures or []))
    body += frag if frag is not None else "regression sample clean.\n"
    clean = artifact if regression_failures == [] else None
    return head + body, digest, clean


def _reseed_model(dump: helpers.Dumper, status_path: str, vlen: int,
                  diff_path: str, pg_opts):
    """Stage 0 without the agent: reapply a converged model from an earlier
    run (the riscv-isa-sim part of that run's collect_diff output), rebuild
    Spike, and re-judge it with the same directed programs the model stage
    uses.  It must pass outright -- a reseeded model that fails is a
    configuration error, not something to iterate on."""
    # Only the riscv-isa-sim half.  A run's collect_diff output holds the
    # model *and* the RTL; feeding the whole thing here would seed the RTL
    # stage behind the agent's back, and would collide with --rtl-diff when
    # both flags name the same file.  The split is the same one --rtl-diff
    # uses, in the other direction.
    with open(diff_path) as f:
        diff = _split_diff(f.read())[0]
    if not diff.strip():
        raise RuntimeError(
            f"--model-diff {diff_path}: no {_MODEL_PATH_PREFIX} hunks in "
            f"this diff -- there is no model in it to reseed.")
    err = get(nodes.apply_diff.options(**pg_opts)
              .chia_remote(diff, extension=EXTENSION))
    if err:
        raise RuntimeError(f"--model-diff {diff_path}: git apply failed:\n{err}")
    artifact = get(nodes.build_spike.options(**pg_opts)
                   .chia_remote(extension=EXTENSION))
    dump.text("model_reseed_build.stdout.txt",
              getattr(artifact, "stdout", "") or "")
    if not artifact.success:
        raise RuntimeError(f"--model-diff {diff_path}: Spike build failed")
    tests, geometries = _build_directed(dump, vlen, full_sweep=False)
    outcomes, logs = _run_on_spike(tests, vlen, pg_opts)
    _dump_sim_logs(dump, "model_reseed", outcomes, logs)
    titan_tools.write_status(status_path, outcomes, geometries)
    summary = helpers.summarize(outcomes)
    dump.json("model_reseed.json", summary)
    _event("model_reseed", source=diff_path, **summary["counts"])
    skipped = sum(1 for _, o in outcomes if o.kind == "skip")
    if summary["failing"] or (
            outcomes and skipped / len(outcomes) > MODEL_MAX_SKIP_FRACTION):
        raise RuntimeError(
            f"--model-diff {diff_path}: reseeded model does not pass the "
            f"directed programs: {summary['counts']}")
    return artifact


def _seed_model(dump: helpers.Dumper, status_path: str, vlen: int,
                diff_path: str, pg_opts) -> str:
    """Seed Stage 0 from a converged model and then *run the stage anyway*.

    The difference from ``_reseed_model`` is the whole point of the flag.
    ``--model-diff`` says "this model is finished, reuse it and skip the
    agent", and enforces that by raising if the directed suite disagrees.
    ``--model-seed`` says "this model is finished *for the instructions it was
    written for*, and the suite has since grown a new one" -- so the suite is
    expected to fail, the failures are the agent's task, and raising on them
    would make a round-two run impossible to start.

    Applies the riscv-isa-sim half of *diff_path*, rebuilds Spike, runs the
    directed programs on it (attempt 0, dumped like any other attempt) and
    returns the note that opens the model agent's first turn.  The caller
    hands that note to ``_model_stage``, which then behaves exactly as it does
    on a cold tree.
    """
    with open(diff_path) as f:
        diff = _split_diff(f.read())[0]
    if not diff.strip():
        raise RuntimeError(
            f"--model-seed {diff_path}: no {_MODEL_PATH_PREFIX} hunks in "
            f"this diff -- there is no model in it to seed from.")
    digest = _digest(diff)
    dump.text("model_seed.diff", diff)
    err = get(nodes.apply_diff.options(**pg_opts)
              .chia_remote(diff, extension=EXTENSION))
    if err:
        raise RuntimeError(f"--model-seed {diff_path}: git apply failed:\n{err}")
    artifact = get(nodes.build_spike.options(**pg_opts)
                   .chia_remote(extension=EXTENSION))
    dump.text("model_seed_build.stdout.txt",
              getattr(artifact, "stdout", "") or "")
    if not artifact.success:
        # A seeded model that does not *compile* is a configuration error in
        # the same way a bad diff path is: there is nothing for the agent to
        # extend.  Failing tests are the agent's job; a failing build is not.
        _event("model_seed_build_failure", source=diff_path, digest=digest)
        raise RuntimeError(
            f"--model-seed {diff_path}: Spike build failed on the seeded "
            f"tree:\n" + (getattr(artifact, "stderr", "") or "")[-2000:])

    tests, geometries = _build_directed(dump, vlen, full_sweep=False)
    outcomes, logs = _run_on_spike(tests, vlen, pg_opts)
    _dump_sim_logs(dump, "model_seed", outcomes, logs)
    titan_tools.write_status(status_path, outcomes, geometries)
    summary = helpers.summarize(outcomes)
    dump.json("model_seed.json", summary)
    _event("model_seed", source=diff_path, digest=digest,
           files=len(_diff_files(diff)), total=summary["total"],
           failing=len(summary["failing"]), **summary["counts"])

    done, new = helpers.instruction_scope()
    note = helpers.format_model_seed_message(diff_path, outcomes, logs,
                                             done, new)
    dump.text("model_seed_note.md", note)
    return note


def run(run_id: str, vlen: int = VLEN, stress: bool = True,
        model: bool = True, model_diff: Optional[str] = None,
        model_seed: Optional[str] = None,
        rtl_diff: Optional[str] = None,
        rtl_diff_note: Optional[str] = None) -> Dict[str, object]:
    out_dir = os.path.join(TITAN_LOG_ROOT, run_id)
    dump = helpers.Dumper(out_dir)
    work_root = os.path.join(out_dir, "work")
    os.makedirs(work_root, exist_ok=True)
    # CPU 4, not 1: build_* need one each, and _run_on_spike runs two
    # spike_run (num_cpus=1, chipyard 0.4) side by side inside the bundle.
    pg = placement_group([{"CPU": 4, "chipyard": 1}], strategy="STRICT_PACK")
    ray.get(pg.ready())
    pg_opts = {"scheduling_strategy": PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=0)}
    head_local = {"resources": {"head_local": 0.1}}

    # Two agents, two sets of scratch files.  The status file differs because
    # they are testing different things; the knowledge file differs because a
    # shared notebook would leak one derivation into the other, in whichever
    # direction happened to run second -- and the value of Stage 3 is that
    # neither derivation informed the other.
    def _tools(tag: str, with_run_directed: bool = False,
               stock_spike=None):
        status_path = os.path.join(work_root, f"status_{tag}.md")
        knowledge_path = os.path.join(work_root, f"knowledge_{tag}.md")
        events_path = os.path.join(work_root, f"agent_runs_{tag}.jsonl")
        failing_path = os.path.join(work_root, f"failing_{tag}.json")
        rvv_failing_path = os.path.join(work_root, f"rvv_failing_{tag}.json")
        finish = titan_tools.FinishTool(
            f"titan_finish_{tag}_{run_id}",
            os.path.join(work_root, f"finished_{tag}"),
            task_options=head_local)
        tools_list = [
            BashTool(name=f"titan_edit_{tag}_{run_id}", work_dir=CHIPYARD_PATH,
                     timeout_seconds=300, task_options=pg_opts),
            titan_tools.SpecTool(f"titan_spec_{tag}_{run_id}", str(SPEC_DIR),
                                 "specs/ime", task_options=head_local),
            titan_tools.StatusTool(f"titan_status_{tag}_{run_id}", status_path,
                                   rvv_failing_path=rvv_failing_path,
                                   task_options=head_local),
            titan_tools.KnowledgeTool(
                f"titan_know_{tag}_{run_id}", knowledge_path,
                task_options=head_local),
            finish,
        ]
        run_tool = None
        if with_run_directed:
            # Placed on the head like the other bookkeeping tools; the *work*
            # it dispatches goes into the loop's placement group, which is
            # the part that matters (see agent_run_directed).  The BashTool
            # holds no `chipyard` while the turn runs -- _ToolServerActor is
            # num_cpus=0 with a scheduling strategy and no resources -- so a
            # build dispatched from here has the bundle's chipyard to itself.
            cfg = {"pg_opts": pg_opts, "run_id": run_id, "tag": tag,
                   "vlen": vlen, "status_path": status_path,
                   "out_dir": out_dir, "events_path": events_path,
                   "failing_path": failing_path,
                   "rvv_failing_path": rvv_failing_path,
                   "stock_spike": stock_spike}
            run_tool = titan_tools.RunDirectedTool(
                f"titan_run_{tag}_{run_id}",
                functools.partial(agent_run_directed, cfg),
                os.path.join(work_root, f"agent_budget_{tag}"),
                max_runs=AGENT_RUNS_PER_ITER,
                rvv_runner=functools.partial(agent_run_rvv, cfg),
                task_options=head_local)
            tools_list.insert(4, run_tool)
        return (tools_list, status_path, finish, knowledge_path, run_tool,
                events_path, failing_path, rvv_failing_path)

    result: Dict[str, object] = {"run_id": run_id, "vlen": vlen,
                                 "converged": False}
    all_tools: List[object] = []
    _STALLED.clear()
    try:
        _event("run_start", run_id=run_id, vlen=vlen)
        # Reset once, at the top.  Not between stages: the model stage's edits
        # to riscv-isa-sim live in this same tree, and a reset before the RTL
        # stage would throw away the golden model the run just built.
        get(nodes.reset_chipyard.options(**pg_opts)
            .chia_remote(CHIPYARD_PATH, extension=EXTENSION))

        # The stock golden model for S2, built HERE -- after the reset, before
        # Stage 0 writes a line of Zvvm -- because this is the only moment in
        # a run when the riscv-isa-sim checkout is guaranteed pristine.  See
        # _STOCK_SPIKE for what goes wrong without it.  One spike build
        # (~1 min) per run; S2 itself pays nothing, the relink was already
        # unconditional (build_saturn passes clean_sim=True).
        global _STOCK_SPIKE
        _STOCK_SPIKE = get(nodes.build_spike.options(**pg_opts)
                           .chia_remote(extension=EXTENSION))
        if not _STOCK_SPIKE.success:
            raise RuntimeError(
                "could not build stock libriscv from the pristine spike "
                "checkout; S2 would silently cosimulate against whatever "
                "libriscv is installed:\n"
                + (getattr(_STOCK_SPIKE, "stderr", "") or "")[-2000:])
        _event("stock_spike", digest=_STOCK_SPIKE.digest[:12])
        result["stock_spike_digest"] = _STOCK_SPIKE.digest[:12]

        # Built after it, so the agent's tool actor gets the artifact by value
        # (cfg is serialised into that actor at construction time).
        model_tools, model_status, model_finish, model_know, _, _, _, _ = \
            _tools("model")
        (rtl_tools, status_path, finish, knowledge_path, run_tool,
         events_path, failing_path, rvv_failing_path) = _tools(
            "rtl", with_run_directed=True, stock_spike=_STOCK_SPIKE)
        all_tools = model_tools + rtl_tools

        # Stage 0, first, so that "the model's author never saw the RTL" is
        # true by construction rather than by instruction.
        spike_artifact = None
        model_note = None
        if model and model_seed and model_diff:
            raise RuntimeError(
                "--model-diff and --model-seed are the same half of the same "
                "diff read two ways: --model-diff reuses it and skips the "
                "agent, --model-seed hands it to the agent to extend. Pick "
                "one.")
        if model and model_seed:
            # Apply, build, judge -- then fall through into the ordinary
            # stage with the result as the agent's opening evidence.
            model_note = _seed_model(dump, model_status, vlen, model_seed,
                                     pg_opts)
            result["model_source"] = model_seed
            result["model_seeded"] = True
        if model and model_diff:
            spike_artifact = _reseed_model(dump, model_status, vlen,
                                           model_diff, pg_opts)
            result["model"] = True
            result["model_source"] = model_diff
            result["model_digest"] = spike_artifact.digest[:12]
        elif model:
            ok, spike_artifact = _model_stage(
                dump, model_status, model_tools, model_finish, vlen,
                MODEL_MAX_ITERS, pg_opts, first_note=model_note)
            result["model"] = ok
            if not ok:
                return result
            result["model_digest"] = spike_artifact.digest[:12]

        # Stage S1 resume, after the model and before the agent: the RTL
        # half of a previous run's diff, plus a real build-and-judge of it so
        # the first prompt carries true feedback rather than a claim.
        first_note = None
        resume_clean = None
        if rtl_diff:
            first_note, rtl_digest, resume_clean = _rtl_resume(dump, status_path, vlen,
                                                 rtl_diff, pg_opts, run_id,
                                                 failing_path=failing_path,
                                                 rvv_failing_path=rvv_failing_path)
            result["rtl_source"] = rtl_diff
            result["rtl_digest"] = rtl_digest
        if rtl_diff_note:
            # An override for when the pre-run is not wanted (or not
            # affordable): whatever this file says is what the agent is told.
            with open(rtl_diff_note) as fh:
                extra = fh.read()
            first_note = f"{first_note}\n\n{extra}" if first_note else extra

        llm = llm_mod.make_llm(f"titan_{run_id}")
        if resume_clean is not None and RESUME_FAST_PATH:
            # Attempt 0 already is the S1+S2 verdict ``_iterate`` would have
            # to reproduce: same build, same directed suite, same sampled
            # regression with the same unanimity confirmation.  Nothing is
            # left for an agent turn to do except change a converged tree.
            _event("rtl_resume_fast_path", digest=result.get("rtl_digest"))
            print("[resume] attempt 0 directed + S2 sample clean -> "
                  "skipping the agent turn, going straight to the gate",
                  flush=True)
            ok, artifact = True, resume_clean
        else:
            ok, artifact = _iterate(llm, rtl_tools, dump, status_path, finish,
                                    label="impl", vlen=vlen, max_iters=MAX_ITERS,
                                    first_message=None, pg_opts=pg_opts,
                                    first_note=first_note, run_id=run_id,
                                    knowledge_path=knowledge_path,
                                    run_tool=run_tool, events_path=events_path,
                                    failing_path=failing_path,
                                    rvv_failing_path=rvv_failing_path)
        result["s1_s2"] = ok
        result["regression_confirm"] = _confirm_summary()
        if not ok:
            return result

        gate_ok, gate_message = _gate(artifact, dump, status_path, vlen,
                                      pg_opts=pg_opts, run_id=run_id,
                                      rvv_failing_path=rvv_failing_path)
        result["gate"] = gate_ok
        result["regression_confirm"] = _confirm_summary()
        gate_rounds = 0
        while not gate_ok:
            # Two ways to get here.  Without a message: the cheap tier passed
            # and the full sweep did not, so the fault lives in partial-N
            # behaviour -- the C tile tail policy, almost certainly.  With
            # one: the directed sweep passed and the *full* RVV suite did
            # not, and the message is the same regression feedback a
            # per-iteration S2 failure produces.  Re-enter with whichever it
            # is as the evidence.
            if gate_rounds >= GATE_DEBUG_ROUNDS:
                _event("gate_debug_exhausted", rounds=gate_rounds)
                result["gate"] = False
                result["gate_failure"] = (
                    "gate", f"still failing after {gate_rounds} debug "
                    f"re-entries (cap GATE_DEBUG_ROUNDS={GATE_DEBUG_ROUNDS})")
                return result
            gate_rounds += 1
            _event("gate_debug_start", round=gate_rounds,
                   half="regression" if gate_message else "directed")
            ok, artifact = _iterate(
                llm, rtl_tools, dump, status_path, finish,
                label=f"gate{gate_rounds}" if gate_rounds > 1 else "gate",
                vlen=vlen, max_iters=DEBUG_MAX_ITERS,
                first_message=gate_message or (
                              "The full directed sweep, which includes "
                              "partial-N geometries, is failing where the "
                              "full-VL subset passes. Read the status file: "
                              "the fault is in behaviour that only appears "
                              "when N < N_max."),
                pg_opts=pg_opts, run_id=run_id,
                knowledge_path=knowledge_path, run_tool=run_tool,
                events_path=events_path, failing_path=failing_path,
                rvv_failing_path=rvv_failing_path)
            result["gate"] = ok
            if not ok:
                return result
            # The whole point of r17.  ``_iterate`` converged on the directed
            # suite and the *sampled* regression; neither is what the gate
            # failed on.  Re-run ``_gate`` itself -- the full directed sweep
            # and then, unsampled, the whole riscv-vector-tests suite -- so
            # the half that failed is the half that has to come back clean.
            # ``_gate_regression`` reuses the cosim build ``_run_s2`` left in
            # ``_LAST_COSIM`` when the tree has not moved since, and emits its
            # ``gate_regression`` event on every pass, so each re-check shows
            # up in the trace.
            gate_ok, gate_message = _gate(artifact, dump, status_path, vlen,
                                          pg_opts=pg_opts, run_id=run_id,
                                          rvv_failing_path=rvv_failing_path)
            result["gate"] = gate_ok
            result["regression_confirm"] = _confirm_summary()

        if stress:
            if spike_artifact is None:
                raise RuntimeError(
                    "Stage 3 needs the Spike model as its golden model; run "
                    "without --no-model, or pass --no-stress.")
            # Relink the simulator against the model this run produced. The
            # image's own libriscv has never heard of Zvvm, so cosimulating
            # against it would compare the RTL to a Spike that traps on the
            # first matrix instruction.
            _write_s2_cosim_config(pg_opts)
            try:
                artifact = get(nodes.build_saturn.options(**pg_opts)
                               .chia_remote(COSIM_CONFIG, spike_artifact,
                                            extension=EXTENSION))
            finally:
                _write_s2_cosim_config(pg_opts, enable=False)
            if not artifact.success:
                result["stress_failure"] = ("build", "relink against the "
                                            "golden model failed")
                return result
            pool_dir = db_node.pool_path(run_id)
            added = ime_stress.fill_pool(pool_dir, vlen,
                                         STRESS_TEST_CASES_PER_GEOM)
            _event("stress_pool_filled", count=added)
            deadline = time.time() + STRESS_TEST_HOURS * 3600
            failure, done = _stress(artifact, pool_dir, deadline, dump)
            result["stress_done"] = done
            if failure:
                result["stress_failure"] = failure
                return result

        result["converged"] = True
        return result
    finally:
        if _STALLED:
            result["stalled"] = dict(_STALLED)
        _event("run_end", **{k: v for k, v in result.items()
                             if isinstance(v, (int, float, str, bool))})
        for tool in all_tools:
            try:
                tool.stop()
            except Exception:  # pragma: no cover - teardown must not mask
                pass
        remove_placement_group(pg)
        dump.json("summary.json", result)
        _archive(run_id, out_dir, result)


def _archive(run_id: str, out_dir: str, result: Dict[str, object]) -> None:
    """Persist the run to the database node and free the Stage 3 scratch pool.

    Two separate reasons, both easy to forget until they bite.

    The pool lives under ``DB_ROOT/tmp/<run_id>/`` and nothing else ever
    deletes it, so without ``pool_finalize`` every run leaks a directory of
    ELFs onto the database node forever.

    The logs live under ``TITAN_LOG_ROOT`` on whichever machine ran the
    driver, which is a ``/tmp`` path in a container.  A run that is not
    archived is a run whose diffs, prompts and failure logs vanish with the
    container -- and those are the only record of why an agent did what it
    did.

    Teardown must never mask the real outcome, so every step here swallows
    its own failure and says so rather than raising.
    """
    try:
        sweep_n, sweep_path = get(
            db_node.claim_sweep.chia_remote(EXTENSION))
        get(db_node.archive_dir.chia_remote(
            sweep_path, run_id, _tar_dir(out_dir), extension=EXTENSION))
        get(db_node.write_text.chia_remote(
            sweep_path, "summary.md", _render_summary(run_id, result),
            extension=EXTENSION))
        print(f"archived to sweep {sweep_n}: {sweep_path}")
    except Exception as exc:  # noqa: BLE001 - teardown, not control flow
        print(f"WARNING: archiving to the database node failed: {exc}\n"
              f"         the run's logs are still at {out_dir} on this host.")
    try:
        get(db_node.pool_finalize.chia_remote(db_node.pool_path(run_id),
                                              extension=EXTENSION))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not remove the stress pool: {exc}")


def _confirm_summary() -> str:
    """The run-level one-liner: how much of S2/gate's failure reporting was
    the trace bridge rather than the design.  Kept in the result dict and in
    the archived summary so a run cannot be read as "S2 was clean" when what
    happened was "S2 reported 47 failures and none of them reproduced"."""
    if not _CONFIRM_LOG:
        return "not run"
    reported = sum(int(c["reported"]) for c in _CONFIRM_LOG)
    confirmed = sum(int(c["confirmed"]) for c in _CONFIRM_LOG)
    extra = sum(int(c.get("extra_runs", 0)) for c in _CONFIRM_LOG)
    line = (f"{len(_CONFIRM_LOG)} pass(es): {reported} single-run "
            f"failure(s) reported, {confirmed} survived "
            f"{REGRESSION_CONFIRM_REPS}-rep unanimity "
            f"({extra} extra cosims)")
    suspects, ctx = _repeat_suspects()
    if suspects:
        # In the archived summary, not only in the trace: a name cleared this
        # often may be an intermittent real bug that unanimity discarded, and
        # whoever reads the run needs to see that without opening the JSON.
        names = ", ".join(f"{d['test']} ({d['failed_reps']}/"
                          f"{d['total_reps']} re-runs failed)"
                          for d in suspects[:6])
        line += (f" -- WARNING: {len(suspects)} test(s) cleared as flaky "
                 f"kept failing far longer than this run's own flake "
                 f"({ctx.get('background_fail_rate')} of re-runs) "
                 f"({names}{', ...' if len(suspects) > 6 else ''}); "
                 f"possible intermittent real bug, re-run them standalone")
    elif ctx.get("distinct_names_cleared", 0) >= 2:
        # Said explicitly: no alert here means "checked, nothing exceeded
        # chance", not "never looked" -- and it is weak evidence either way.
        line += (f" -- no cleared test exceeded this run's own flake profile "
                 f"({ctx.get('background_fail_rate')} of re-runs failed, "
                 f"{ctx.get('distinct_names_cleared')} names over "
                 f"{ctx.get('passes')} pass(es)); weak evidence, not a clean "
                 f"bill of health")
    return line


def _render_summary(run_id: str, result: Dict[str, object]) -> str:
    lines = [f"# Titan run {run_id}", ""]
    for key in ("vlen", "model", "model_digest", "rtl_source", "rtl_digest",
                "s1_s2", "regression_confirm", "gate", "gate_failure",
                "stress_done",
                "stress_failure", "stalled",
                "converged"):
        if key in result:
            lines.append(f"- **{key}**: {result[key]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--vlen", type=int, default=VLEN)
    parser.add_argument("--no-stress", action="store_true")
    parser.add_argument("--no-model", action="store_true",
                        help="skip stage 0 (then stage 3 cannot run)")
    parser.add_argument("--model-diff", metavar="DIFF",
                        help="reuse a converged Stage 0 model: the "
                             "riscv-isa-sim hunks of an earlier run's "
                             "*_impl_diff_attemptN.diff. Skips the model "
                             "agent; the model is rebuilt and re-judged.")
    parser.add_argument("--model-seed", metavar="DIFF",
                        help="seed stage 0 from an earlier run's converged "
                             "model (the riscv-isa-sim hunks of that run's "
                             "diff) and then RUN the model agent on top of "
                             "it: applied, rebuilt and judged before the "
                             "first LLM turn, with the per-test failures "
                             "carried into the agent's opening message as "
                             "the new instruction's work. Unlike "
                             "--model-diff, failures are the point rather "
                             "than an error. Mutually exclusive with it.")
    parser.add_argument("--rtl-diff", metavar="DIFF",
                        help="resume stage S1 from an earlier run's RTL "
                             "work: the NON-riscv-isa-sim hunks of that "
                             "run's *_impl_diff_attemptN.diff. Applied "
                             "before the first LLM turn, then built and "
                             "judged once (attempt 0) so the agent's first "
                             "prompt carries that tree's real directed "
                             "result. Composes with --model-diff, which "
                             "takes the other half of the same file.")
    parser.add_argument("--rtl-diff-note", metavar="FILE",
                        help="extra text appended to the first S1 prompt "
                             "(on top of what --rtl-diff's pre-run found).")
    args = parser.parse_args()

    ray.init(address="auto", runtime_env=RUNTIME_ENV)
    start_collector(log_dir=os.path.join(TITAN_LOG_ROOT, args.run_id, "trace"))
    result = run(args.run_id, vlen=args.vlen,
                 stress=not args.no_stress, model=not args.no_model,
                 model_diff=args.model_diff, model_seed=args.model_seed,
                 rtl_diff=args.rtl_diff,
                 rtl_diff_note=args.rtl_diff_note)
    print(f"converged={result['converged']}  {result}")
    return 0 if result["converged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
