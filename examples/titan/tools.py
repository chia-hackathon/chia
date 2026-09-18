"""MCP tools the implementing agent can call.

Four tools, mirroring riscv_extensions: read the spec, read the current test
status, keep a private notebook, declare done.  The one that changes shape is
:class:`StatusTool`.

riscv_extensions tracked status per instruction, which was the right
granularity there: an instruction either worked or it did not.  For IME an
instruction is not a unit of correctness -- ``vmmacc.vv`` can be right at
LAMBDA=4 and wrong at LAMBDA=1 because the tile geometry, not the opcode,
is what the datapath has to get right.  So status is reported per
(instruction x tile geometry), and the agent can see that, say, everything
with EMUL_C=1 passes and everything with EMUL_C=8 fails -- which points
straight at accumulator group addressing rather than at the multiplier.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# The submodule, not the package: chia/base/tools/ has no __init__.py, so
# `from chia.base.tools import ChiaTool` binds the *module* and every
# `class X(ChiaTool)` below raises TypeError at import time -- before
# ray.init(), before anything reaches a worker.  Every other example in
# the repo spells it this way.
from chia.base.tools.ChiaTool import ChiaTool

import helpers
from constants import (AGENT_RUNS_PER_ITER, RUN_DIRECTED_DRAIN_TIMEOUT_S,
                       RUN_DIRECTED_WAIT_CAP, RVV_AGENT_MAX_REPS,
                       RVV_STATUS_NAMES)


class SpecTool(ChiaTool):
    """Points the agent at the Zvvm spec and its machine-readable encodings."""

    def __init__(self, name: str, spec_dir: str, spec_rel: str,
                 task_options=None) -> None:
        super().__init__(name, task_options=task_options)
        self.spec_dir = spec_dir
        self.spec_rel = spec_rel
        self.mcp.add_tool(self.read_spec, name=f"{name}_read_spec")
        super().__post_init__()

    def read_spec(self) -> str:
        """List the IME specification documents available to you.

        Open them with your own file-reading tool. The asciidoc text is
        authoritative for semantics; instructions.json carries the encodings
        and is generated from that same text, so the two cannot disagree.
        """
        if not os.path.isdir(self.spec_dir):
            return f"No spec directory at {self.spec_rel}."
        entries = []
        for fname in sorted(os.listdir(self.spec_dir)):
            if fname.endswith((".adoc", ".md", ".txt", ".json")):
                entries.append(os.path.join(self.spec_rel, fname))
        if not entries:
            return f"No spec documents under {self.spec_rel}."
        return ("Specification documents (read them; do not rely on memory of "
                "the Zvvm draft):\n" + "\n".join(f"  {e}" for e in entries))


class StatusTool(ChiaTool):
    """Reads the directed-test status file, keyed by instruction x geometry."""

    def __init__(self, name: str, status_path: str,
                 rvv_failing_path: Optional[str] = None,
                 task_options=None) -> None:
        super().__init__(name, task_options=task_options)
        self.status_path = status_path
        # Appended at *read* time rather than baked in by write_status: the
        # set moves whenever an agent run_rvv job finishes, which is after
        # the last write_status of the turn, and a stale list is worse than
        # none -- "failing" is what run_rvv_start('failing') will select.
        self.rvv_failing_path = rvv_failing_path
        self.mcp.add_tool(self.read_status, name=f"{name}_read_status")
        super().__post_init__()

    def read_status(self) -> str:
        """Results of the most recent directed-test run, per tile geometry,
        plus the RVV regression tests that are currently failing.

        Read this at the start of every turn. The pattern across geometries
        is usually more informative than any single failure: a fault confined
        to one EMUL_C, one LAMBDA, or one SEW tells you which part of the
        datapath is wrong.
        """
        if not os.path.exists(self.status_path):
            body = "No directed tests have run yet."
        else:
            with open(self.status_path, encoding="utf-8") as fh:
                body = fh.read()
        return body.rstrip("\n") + "\n" + self._rvv_section()

    def _rvv_section(self) -> str:
        if not self.rvv_failing_path:
            return ""
        try:
            with open(self.rvv_failing_path, encoding="utf-8") as fh:
                names = sorted(json.load(fh))
        except (OSError, ValueError):
            return ""
        lines = ["", "## Failing RVV regression tests (Stage 2)"]
        if not names:
            lines.append("  none -- the last regression run was clean.")
            return "\n".join(lines) + "\n"
        lines.append(f"  {len(names)} failing; these are what "
                     f"`run_rvv_start(\"failing\")` will run"
                     + (f" (first {RVV_STATUS_NAMES} shown)"
                        if len(names) > RVV_STATUS_NAMES else "") + ":")
        for name in names[:RVV_STATUS_NAMES]:
            lines.append(f"  {name}")
        if len(names) > RVV_STATUS_NAMES:
            lines.append(f"  ... and {len(names) - RVV_STATUS_NAMES} more "
                         f"(full list: the regress/failures.txt in this "
                         f"iteration's log directory).")
        return "\n".join(lines) + "\n"


def write_status(status_path: str,
                 outcomes: Sequence[Tuple[str, object]],
                 geometries: Dict[str, object]) -> None:
    """Render the status file StatusTool serves.

    Grouped three ways -- by outcome, by EMUL_C and by LAMBDA -- because the
    grouping *is* the diagnosis. Each grouping is cheap to produce and any
    one of them may be the one that makes the fault obvious.
    """
    lines: List[str] = ["# Directed test status", ""]

    counts: Dict[str, int] = defaultdict(int)
    for _, outcome in outcomes:
        counts[outcome.kind] += 1
    lines.append("## Totals")
    for kind in ("pass", "skip", "mismatch", "bad_geometry", "trap",
                 "timeout", "silent"):
        if counts.get(kind):
            lines.append(f"  {kind:13s} {counts[kind]}")
    lines.append("")
    lines.append("`skip` means the DUT declined that tile geometry *legally*: "
                 "it clamped the requested lambda DOWN to a supported value "
                 "(or, supporting nothing that small, up to the smallest "
                 "value it has). Those are not counted against you.")
    lines.append("")
    lines.append("`bad_geometry` means it answered with a lambda WARL does "
                 "not permit -- larger than the request while smaller "
                 "supported values exist, or a value no implementation may "
                 "select at this VLEN and SEW. Those geometries were never "
                 "tested, and they ARE counted against you.")
    lines.append("")

    def _group(title: str, key) -> None:
        lines.append(f"## By {title}")
        buckets: Dict[object, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for name, outcome in outcomes:
            geom = geometries.get(name)
            if geom is None:
                continue
            buckets[key(geom)][outcome.kind] += 1
        for value in sorted(buckets):
            tally = ", ".join(f"{k}={v}" for k, v in sorted(
                buckets[value].items()))
            lines.append(f"  {title}={value}: {tally}")
        lines.append("")

    support = helpers.geometry_support(outcomes)
    if support:
        lines.append("## LAMBDA the DUT actually supports")
        lines += support
        lines.append("")
        lines.append("Every declined lambda is a tile geometry this run did "
                     "not test. Legal, but untested is not tested.")
        lines.append("")

    _group("EMUL_C", lambda g: g.emul_c)
    _group("LAMBDA", lambda g: g.lam)
    _group("SEW", lambda g: g.sew)
    _group("LMUL", lambda g: g.lmul)

    # `failed` rather than a second copy of the rule: helpers.Outcome is the
    # one place that decides what counts as a failure, and a duplicate of
    # that tuple here is exactly how `bad_geometry` would end up failing the
    # gate but reading as clean in the file the agent is told to trust.
    failing = [(n, o) for n, o in outcomes if o.failed]
    if failing:
        lines.append("## Failing tests")
        for name, outcome in failing:
            geom = geometries.get(name)
            where = f" at C[{outcome.row},{outcome.col}]" \
                if outcome.row is not None else ""
            if outcome.kind == "bad_geometry":
                where = (f": requested lambda={outcome.requested_lambda}, "
                         f"DUT selected lambda={outcome.selected_lambda}")
            lines.append(f"  {name}: {outcome.kind}{where}")
            if geom is not None:
                lines.append(f"      {geom.describe()}")
    else:
        lines.append("## Failing tests\n  none")

    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --- run_rvv repetition ----------------------------------------------------
#: A failing test block in ``helpers.format_regression_failure`` is "## name".
_RVV_FAIL_RE = re.compile(r"^## (\S+)\s*$", re.M)
#: "(run_rvv: ..., 3/12 failing)" and the clean "12/12 pass".
_RVV_FAILING_HEAD_RE = re.compile(r"(\d+)/(\d+) failing")
_RVV_PASS_HEAD_RE = re.compile(r"(\d+)/(\d+) pass\b")


def _rvv_rep_verdict(body: str) -> Tuple[Optional[int], set]:
    """``(tests_run, failing_names)`` for one ``run_rvv`` result body.

    ``None`` for the count means the body was not a verdict at all -- a build
    failure or a harness error -- and that rep must not be counted as a pass.
    """
    fails = set(_RVV_FAIL_RE.findall(body))
    m = _RVV_FAILING_HEAD_RE.search(body)
    if m:
        return int(m.group(2)), fails
    m = _RVV_PASS_HEAD_RE.search(body)
    if m:
        return int(m.group(2)), fails
    return None, fails


def _rvv_reps_summary(bodies: Sequence[str]) -> str:
    """Fold N repeats of the same selection into one majority verdict.

    Why this exists: the cospike/DebugROB DPI trace bridge is nondeterministic
    run to run, and the *same* binary on the *same* test flips about 12% of
    the time (``titan_runs/nondet/``).  One run is not a verdict.  Each test
    that failed in at least one rep is reported as ``k/n passed`` so the agent
    can judge by majority instead of chasing a flip.
    """
    n = len(bodies)
    per_rep = [_rvv_rep_verdict(b) for b in bodies]
    valid = [i for i, (ran, _) in enumerate(per_rep) if ran is not None]
    names = sorted({name for _, fails in per_rep for name in fails})
    lines = [f"(run_rvv: {n} reps of the same build -- a single cosim run is "
             f"not a verdict; the trace bridge flips ~12% of runs, see "
             f"titan_runs/nondet/)\n"]
    if not valid:
        lines.append("No rep produced a verdict (build or harness failure). "
                     "The last rep said:\n")
        lines.append(bodies[-1] if bodies else "(no output)")
        return "\n".join(lines)
    if len(valid) < n:
        lines.append(f"{n - len(valid)} of {n} reps produced no verdict "
                     f"(build or harness failure) and are counted as "
                     f"failures below.\n")
    ran = per_rep[valid[0]][0] or 0
    if not names:
        lines.append(f"All {ran} test(s) passed {len(valid)}/{n} reps.\n")
    else:
        lines.append("Per-test verdicts (k/n = reps passed):\n")
        for name in names:
            passed = sum(1 for i in valid if name not in per_rep[i][1])
            call = ("FAILING" if passed * 2 < n else
                    "flaky -- majority pass" if passed < n else "pass")
            lines.append(f"- {name}: {passed}/{n} passed ({call})")
        clean = ran - len(names)
        if clean > 0:
            lines.append(f"\nThe other {clean} test(s) in this selection "
                         f"passed {n}/{n} reps.")
        lines.append("\nAct on the majority. A test that passed once is not "
                     "cleared; a test that failed once is not broken.")
    lines.append("\n--- last rep, in full ---\n")
    lines.append(bodies[valid[-1]])
    return "\n".join(lines)


class RunDirectedTool(ChiaTool):
    """Lets the agent build and test its own change, mid-turn -- by polling.

    The single most expensive property of the loop up to r5 was that a change
    could only be evaluated by ending the turn: ten iterations, one experiment
    each, and an agent that had to write a whole hypothesis' worth of Chisel
    before learning whether the first line of it elaborated.  This tool closes
    that gap -- ``change -> run_directed_start -> run_directed_wait ->
    change`` -- and the iteration boundary goes back to being what it was
    meant to be, a checkpoint rather than the only feedback channel.

    Why it is a *pair* of calls rather than the single blocking one it was
    through r8.  A Saturn elaboration is ~2.5 minutes and the simulations come
    after it, so the old ``run_directed`` never returned in less than three;
    the ``claude`` CLI moves any MCP call still running after 120 seconds into
    a background task and hands control back to the model with a "you'll be
    notified" note.  In ``-p`` mode a model that has nothing left to do ends
    its turn, and the session -- notification and all -- is over.  That is
    exactly what happened in r8 iteration 3: the agent installed a printf
    instrumented ``MatrixMultiplyPipe``, started a run, said "Waiting for the
    diagnostic run to complete", and stopped.  The loop then graded the
    instrumented tree: 26/27 passing became 27 mismatch.

    So no call this tool serves may approach 120 seconds.  ``start`` returns
    at once (the work goes to a worker thread in the tool server actor, which
    matters because FastMCP calls a *synchronous* tool function on its event
    loop -- a blocking one stops the whole server), and ``wait`` blocks for at
    most :data:`RUN_DIRECTED_WAIT_CAP` seconds before returning "still
    running", which is a result the model has to act on rather than a silence
    it can drift away from.

    One job at a time, because ``chipyard`` is one node and the loop's
    placement group holds all of it.

    Budgeted, and the budget lives in a *file* rather than on this object:
    the method runs inside the MCP server actor, on its own copy of the tool,
    so a counter in memory here would never be visible to the driver that has
    to reset it each iteration.  The job table is a file for the same reason,
    and for one more: :meth:`drain`, which the driver calls after the turn,
    has to see a job the actor started.  Both files are on the head, which is
    where this tool is placed (``head_local``) and where the driver runs.

    ``runner`` is supplied by the loop and does the real work; it is kept out
    of this module so the tool has no opinion about how a build is dispatched
    -- in particular, about which placement group it must go into, which is
    the one detail that will deadlock the run if it is wrong.
    """

    #: The two kinds of job this tool dispatches.  They share everything
    #: that has to be shared -- the one-at-a-time rule, the per-iteration
    #: budget, the job table, the worker thread, :meth:`drain` -- because
    #: they contend for the same single ``chipyard`` node, and differ only
    #: in which runner does the work and what the model is told to call.
    _KINDS = {
        "directed": {"prefix": "rd", "label": "run_directed",
                     "start": "run_directed_start",
                     "wait": "run_directed_wait"},
        "rvv": {"prefix": "rv", "label": "run_rvv",
                "start": "run_rvv_start", "wait": "run_rvv_wait"},
    }

    def __init__(self, name: str, runner: Callable, budget_path: str,
                 max_runs: int = AGENT_RUNS_PER_ITER,
                 rvv_runner: Optional[Callable] = None,
                 task_options=None) -> None:
        super().__init__(name, task_options=task_options)
        self.runner = runner
        self.rvv_runner = rvv_runner
        self.budget_path = budget_path
        self.jobs_path = budget_path + ".jobs.json"
        self.max_runs = max_runs
        self._pool = None
        self.mcp.add_tool(self.run_directed_start,
                          name=f"{name}_run_directed_start")
        self.mcp.add_tool(self.run_directed_wait,
                          name=f"{name}_run_directed_wait")
        self.mcp.add_tool(self.run_directed_status,
                          name=f"{name}_run_directed_status")
        if rvv_runner is not None:
            self.mcp.add_tool(self.run_rvv_start, name=f"{name}_run_rvv_start")
            self.mcp.add_tool(self.run_rvv_wait, name=f"{name}_run_rvv_wait")
        super().__post_init__()

    # A ThreadPoolExecutor is not picklable and this object is re-pickled
    # every turn (it rides along in the tool list handed to llm.prompt).
    def __getstate__(self):
        state = super().__getstate__()
        state["_pool"] = None
        return state

    # --- model-facing -------------------------------------------------
    def run_directed_start(self, tests: str = "failing",
                           rebuild: bool = True) -> str:
        """Start a build-and-test of your current tree. Returns immediately.

        This is the same build and the same programs the loop runs between
        iterations, so a clean result here is a clean iteration.

        It does NOT wait: it hands you a job id, and you call
        `run_directed_wait` (repeatedly, if need be) until that returns the
        result. A build plus one simulation per test takes minutes; a single
        tool call that took minutes would be moved to the background and your
        turn would end on top of a half-tested tree.

        tests: "failing" (the tests that failed last time -- the default, and
            the right one while you are iterating), "all" (every directed
            program; run this before you call finish), or a comma-separated
            list of test names as they appear in the status file.
        rebuild: leave True unless you have changed nothing since your last
            call and only want the tests re-run.

        One job runs at a time. Costs a full elaboration plus a simulation per
        test, so starts are limited to a few per turn -- you are told how many
        remain.
        """
        return self._start("directed", tests, rebuild)

    def run_rvv_start(self, tests: str = "failing",
                      rebuild: bool = True, reps: int = 1) -> str:
        """Start an RVV regression run of your current tree. Returns at once.

        This is Stage 2: Saturn's own riscv-vector-tests, judged by lockstep
        cosimulation against stock Spike on the cosim config -- the same gate
        the loop runs after your turn, so a clean result here is a clean S2.
        Use it whenever the feedback you were given is an RVV regression:
        without it, one hypothesis costs one whole iteration.

        It does NOT wait. Call `run_rvv_wait` (repeatedly) for the result.

        tests: "failing" (the RVV tests that failed the last regression run --
            the default, and the one to use while you are iterating; the
            status file lists their names), "sample" (the loop's own
            deterministic sample of the suite -- thorough, but ~35 minutes,
            so do not start one unless you mean it), or a comma-separated
            list of riscv-vector-tests names (at most 40). "all" is not
            available: the full 841 cannot be run inside a turn.
        rebuild: leave True unless you have changed nothing since your last
            call and only want the tests re-run.
        reps: how many times to run the selection, on the *same* build
            (default 1, max 5). A single cosim run is not a verdict: the
            cospike trace bridge flips about 12% of runs on an unchanged
            binary, so pass reps=3 for a failing test and believe the
            majority. The result then reports each test as "k/n passed".
            Costs one start but reps times the test time.

        A cosim build plus a handful of tests is about four minutes. Shares
        one budget and one chipyard node with run_directed_start, so only one
        job of either kind runs at a time.
        """
        try:
            reps = int(reps)
        except (TypeError, ValueError):
            return "reps must be an integer between 1 and %d." % RVV_AGENT_MAX_REPS
        if reps < 1 or reps > RVV_AGENT_MAX_REPS:
            return (f"reps={reps} is out of range: 1 to "
                    f"{RVV_AGENT_MAX_REPS}. Three is the useful value -- it "
                    f"is enough for a majority and the cost is linear.")
        return self._start("rvv", tests, rebuild, reps)

    async def run_rvv_wait(self, job_id: str = "",
                           max_wait_s: int = 90) -> str:
        """Wait for a run_rvv job and return its result.

        Blocks at most `max_wait_s` seconds (capped at
        RUN_DIRECTED_WAIT_CAP); if the job is not done it says so and you
        call it again. Waits are free and do not count against your budget.

        job_id: the id run_rvv_start gave you. Omit it for the most recent
            job of either kind.

        On completion returns either "N/N pass" or the divergence report:
        one block per failing test with cospike's abort window, and the path
        to the full logs. If the job was started with reps > 1 it returns the
        per-test "k/n passed" majority verdict first, then the last rep in
        full.
        """
        return await self._wait("rvv", job_id, max_wait_s)

    async def run_directed_wait(self, job_id: str = "",
                                max_wait_s: int = 90) -> str:
        """Wait for a run_directed job and return its result.

        Blocks at most `max_wait_s` seconds (capped at
        RUN_DIRECTED_WAIT_CAP). If the job is not finished by then it says so
        and tells you how long it has been running -- call this again. A whole
        build-and-test is several of these calls; that is expected and costs
        nothing against your budget.

        job_id: the id run_directed_start gave you. Omit it for the most
            recent job.

        On completion returns the same summary the loop would have given you:
        totals, one line per failing test, the values behind the first few,
        and the path to the full logs.
        """
        return await self._wait("directed", job_id, max_wait_s)

    def run_directed_status(self) -> str:
        """List this turn's runs (both kinds) and what is left of the budget."""
        jobs = self._read_jobs()
        used = self.runs_used()
        lines = [f"runs started this turn: {used}/{self.max_runs} "
                 f"(one budget, shared by run_directed_start and "
                 f"run_rvv_start)"]
        if not jobs:
            lines.append("No jobs started this turn.")
        for job in sorted(jobs.values(), key=lambda j: j.get("seq", 0)):
            state = job.get("state", "?")
            when = (f"{self._elapsed(job):.0f}s elapsed"
                    if state == "running" else
                    f"took {(job.get('finished') or 0) - job['started']:.0f}s")
            lines.append(f"  {job['job_id']}: {self._label(job)} {state} "
                         f"(tests={job.get('tests')}, "
                         f"rebuild={job.get('rebuild')}, {when})")
        return "\n".join(lines)

    # --- shared start/wait --------------------------------------------
    def _label(self, job: dict) -> str:
        return self._KINDS.get(job.get("kind") or "directed",
                               self._KINDS["directed"])["label"]

    def _start(self, kind: str, tests: str, rebuild: bool,
               reps: int = 1) -> str:
        names = self._KINDS[kind]
        running = self._running_job()
        if running is not None:
            other = self._KINDS.get(running.get("kind") or "directed",
                                    self._KINDS["directed"])
            return (f"A {other['label']} job is already running: "
                    f"{running['job_id']} ({running['tests']}, started "
                    f"{self._elapsed(running):.0f}s ago). There is one "
                    f"chipyard node, so there is one job of either kind. "
                    f"Call {other['wait']}('{running['job_id']}') until it "
                    f"returns a result.")
        used = self.runs_used()
        if used >= self.max_runs:
            return (f"The run budget for this turn is spent "
                    f"({used}/{self.max_runs}); it is one pool shared by "
                    f"run_directed_start and run_rvv_start. Make your "
                    f"remaining edits on the evidence you already have and "
                    f"call finish; the loop will build and test what you "
                    f"leave behind.")
        seq = used + 1
        self._bump(used)
        job_id = f"{names['prefix']}{seq}"
        jobs = self._read_jobs()
        reps = max(1, min(int(reps or 1), RVV_AGENT_MAX_REPS))
        jobs[job_id] = {"job_id": job_id, "seq": seq, "kind": kind,
                        "tests": tests, "rebuild": bool(rebuild),
                        "reps": reps,
                        "state": "running", "started": time.time(),
                        "finished": None, "result": None}
        self._write_jobs(jobs)
        self._submit(job_id, kind, tests, bool(rebuild), seq, reps)
        return (f"Started job {job_id} ({names['label']}, tests={tests}, "
                f"rebuild={bool(rebuild)}"
                + (f", reps={reps}" if reps > 1 else "")
                + f"); {seq}/{self.max_runs} starts used "
                f"this turn (shared budget).\nCall {names['wait']}"
                f"('{job_id}') now, and keep calling it until it returns a "
                f"result. Do not end your turn while it is unfinished: the "
                f"loop grades the tree exactly as you leave it.")

    async def _wait(self, kind: str, job_id: str, max_wait_s: int) -> str:
        names = self._KINDS[kind]
        job = self._find_job(job_id)
        if job is None:
            return (f"No such {names['label']} job"
                    + (f": {job_id}." if job_id else "; none has been "
                       "started this turn.")
                    + f" Call {names['start']} first.")
        job_id = job["job_id"]
        wait_name = self._KINDS.get(job.get("kind") or "directed",
                                    self._KINDS["directed"])["wait"]
        budget = min(max(1, int(max_wait_s or 1)), RUN_DIRECTED_WAIT_CAP)
        deadline = time.time() + budget
        while True:
            job = self._find_job(job_id) or job
            if job.get("state") != "running":
                return self._render(job)
            if time.time() >= deadline:
                break
            # asyncio, not time.sleep: FastMCP runs an async tool on its
            # event loop, and a synchronous sleep here would freeze the
            # whole tool server for the duration.
            await asyncio.sleep(min(2.0, max(0.1, deadline - time.time())))
        return (f"{job_id} is still running (elapsed "
                f"{self._elapsed(job):.0f}s) -- call "
                f"{wait_name}('{job_id}') again. Do not end your turn "
                f"while it is unfinished.")

    # --- worker side --------------------------------------------------
    def _submit(self, job_id: str, kind: str, tests: str, rebuild: bool,
                seq: int, reps: int = 1) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="run_directed")
        self._pool.submit(self._work, job_id, kind, tests, rebuild, seq, reps)

    def _work(self, job_id: str, kind: str, tests: str, rebuild: bool,
              seq: int, reps: int = 1) -> None:
        label = self._KINDS[kind]["label"]
        try:
            runner = self.rvv_runner if kind == "rvv" else self.runner
            if runner is None:
                raise RuntimeError(f"no {label} runner is configured")
            reps = max(1, min(int(reps or 1), RVV_AGENT_MAX_REPS))
            if kind == "rvv" and reps > 1:
                bodies = []
                for rep in range(reps):
                    # Only the first rep may rebuild: the whole point is to
                    # re-run the SAME binary, which is where the flips live.
                    bodies.append(runner(tests, rebuild and rep == 0, seq))
                body = _rvv_reps_summary(bodies)
            else:
                body = runner(tests, rebuild, seq)
        except Exception as exc:                            # noqa: BLE001
            # Never let a dispatch failure look like a test failure: the
            # agent would go and debug its RTL over a Ray problem.
            body = (f"{label} could not complete: {type(exc).__name__}: "
                    f"{exc}\nThis is a harness failure, not a result. Your "
                    f"edits are untouched; note it and continue.")
        jobs = self._read_jobs()
        job = jobs.get(job_id) or {"job_id": job_id, "seq": seq, "kind": kind,
                                   "started": time.time()}
        job.update({"state": "done", "finished": time.time(), "result": body})
        jobs[job_id] = job
        self._write_jobs(jobs)

    def _render(self, job: dict) -> str:
        took = (job.get("finished") or time.time()) - job.get("started", 0)
        return (f"{job.get('result') or '(no output)'}\n\n"
                f"[{job['job_id']}: {self._label(job)} {job.get('seq')}/"
                f"{self.max_runs} runs for this turn, {took:.0f}s]")

    # --- job table ----------------------------------------------------
    def _read_jobs(self) -> Dict[str, dict]:
        try:
            with open(self.jobs_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_jobs(self, jobs: Dict[str, dict]) -> None:
        try:
            os.makedirs(os.path.dirname(self.jobs_path) or ".", exist_ok=True)
            tmp = self.jobs_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(jobs, fh, default=str)
            os.replace(tmp, self.jobs_path)
        except OSError:
            pass

    @staticmethod
    def _elapsed(job: dict) -> float:
        return max(0.0, time.time() - float(job.get("started") or 0))

    def _running_job(self) -> Optional[dict]:
        for job in sorted(self._read_jobs().values(),
                          key=lambda j: j.get("seq", 0), reverse=True):
            if job.get("state") == "running":
                return job
        return None

    def _find_job(self, job_id: str) -> Optional[dict]:
        jobs = self._read_jobs()
        if job_id:
            return jobs.get(job_id.strip())
        if not jobs:
            return None
        return max(jobs.values(), key=lambda j: j.get("seq", 0))

    # --- driver-side, not exposed to the model ------------------------
    def runs_used(self) -> int:
        try:
            with open(self.budget_path) as fh:
                return int(fh.read().strip() or 0)
        except (OSError, ValueError):
            return 0

    def _bump(self, used: int) -> None:
        try:
            os.makedirs(os.path.dirname(self.budget_path), exist_ok=True)
            with open(self.budget_path, "w") as fh:
                fh.write(str(used + 1))
        except OSError:
            pass

    def drain(self, timeout_s: int = RUN_DIRECTED_DRAIN_TIMEOUT_S,
              poll_s: float = 5.0) -> List[dict]:
        """Wait out any job the agent abandoned; return what was in flight.

        Called by the loop as soon as the LLM turn returns.  Not a
        cancellation: the job holds the placement group's single ``chipyard``
        bundle, and the loop's own build dispatched on top of a live one
        double-books that node.  So it is waited out, and the loop's record of
        the turn says the agent never read the answer (``orphaned``).
        """
        running = self._running_job()
        if running is None:
            return []
        orphans = [running]
        deadline = time.time() + max(0, timeout_s)
        while time.time() < deadline:
            job = self._find_job(running["job_id"])
            if job is None or job.get("state") != "running":
                orphans = [job or running]
                break
            time.sleep(poll_s)
        else:
            orphans[0] = dict(running, state="timeout")
        return orphans

    def reset_budget(self) -> None:
        """Called by the loop at the top of every iteration.

        Drains first: clearing the table under a live job would lose the
        loop's only handle on a build that is still holding ``chipyard``.
        """
        self.drain()
        self._write_jobs({})
        self._bump(-1)


def record_agent_run(events_path: str, payload: dict) -> None:
    """Append one ``agent_directed`` record for the driver to re-emit.

    The tool's work happens in an actor, where ``get_profiler()`` is a
    different profiler from the driver's; a trace event logged there would
    not reach the run's trace. So it is written to a file the driver drains
    after the turn, and logged from the driver as well -- cost_report and the
    monitoring see the agent's builds either way.
    """
    try:
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


def drain_agent_runs(events_path: str) -> List[dict]:
    """Read and clear the records ``record_agent_run`` left."""
    if not os.path.exists(events_path):
        return []
    out = []
    try:
        with open(events_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        os.remove(events_path)
    except OSError:
        pass
    return out


class KnowledgeTool(ChiaTool):
    """A notebook that survives across iterations of the loop."""

    def __init__(self, name: str, knowledge_path: str,
                 task_options=None) -> None:
        super().__init__(name, task_options=task_options)
        self.knowledge_path = knowledge_path
        self.mcp.add_tool(self.read_knowledge, name=f"{name}_read_knowledge")
        self.mcp.add_tool(self.append_knowledge,
                          name=f"{name}_append_knowledge")
        super().__post_init__()

    def read_knowledge(self) -> str:
        """What you have learned about Saturn's internals so far.

        Read this at the start of every turn. Your conversation does not
        survive a rebuild; this file does.
        """
        if not os.path.exists(self.knowledge_path):
            return "Nothing recorded yet."
        with open(self.knowledge_path, encoding="utf-8") as fh:
            return fh.read()

    def append_knowledge(self, note: str) -> str:
        """Record a durable fact about the codebase.

        Worth recording: where a signal is defined, how a bundle propagates,
        what a parameter actually controls. Not worth recording: what you are
        about to try, or a restatement of the spec.
        """
        with open(self.knowledge_path, "a", encoding="utf-8") as fh:
            fh.write("\n" + note.rstrip() + "\n")
        return "Recorded."


class FinishTool(ChiaTool):
    """Lets the agent say it is done; the loop decides whether it is."""

    def __init__(self, name: str, sentinel_path: str,
                 task_options=None) -> None:
        super().__init__(name, task_options=task_options)
        self.sentinel_path = sentinel_path
        self.mcp.add_tool(self.finish, name=f"{name}_finish")
        super().__post_init__()

    def finish(self, summary: str) -> str:
        """Declare your edits complete for this turn.

        The loop then builds and tests. Declaring finished is not a claim
        that the tests pass -- it is a claim that you have stopped editing.
        """
        with open(self.sentinel_path, "w", encoding="utf-8") as fh:
            fh.write(summary)
        return "Recorded. The loop will now build and test."

    # Driver-side, not exposed to the model.
    def was_finished(self) -> bool:
        return os.path.exists(self.sentinel_path)

    def reset(self) -> None:
        if os.path.exists(self.sentinel_path):
            os.remove(self.sentinel_path)
