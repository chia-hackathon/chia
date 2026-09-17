"""AETHER loop — agentic kernel optimization on fixed hardware.

    build simulator ONCE (--config, default GENV256D128ShuttleConfig)
              │
    measure the pristine kernel (--kernel)  ──►  baseline cycles
              │
    ┌─────────▼──────────────────────────────────────────┐
    │  agent edits the ONE editable file (kernel_bash)   │
    │        ▼                                            │
    │  cross-compile against SEALED collateral            │
    │        ▼                                            │
    │  run on the Verilator simulator                     │
    │        ▼                                            │
    │  self-check passed?  cycles?  ──► feedback ─────────┘
    └─────────────────────────────────────────────────────┘

The hardware never changes *within* an inner loop, so the simulator artifact is
built once and reused by value. The agent can only ever affect one file; every
build is reassembled from the loop's own pristine copies of everything else.

Structure
---------
The inner loop is the reusable function :func:`run_inner`: given an
already-built simulator handle it optimizes one kernel for N iterations and
returns an :class:`InnerResult`. It never builds hardware and never re-measures
a baseline it can load from cache, so a future OUTER loop (a search over
hardware parameters — Saturn VLEN/DLEN, Gemmini dims) can do::

    ray.init(address="auto", runtime_env=RUNTIME_ENV)
    start_collector(log_dir=...)
    for cfg in design_points:
        select(kernel_name, cfg)          # context.py globals
        sim = build_sim()                 # cached per config by cache.py
        res = run_inner(sim, kernel_name, iters=10, budget_usd=20)

`main()` below is exactly that, with a single design point.

Which benchmark, which file, which self-check and which prompt text all come
from the registry in `kernels.py` (`--kernel NAME`); the simulator config comes
from `--config NAME`. Every measurement — baseline included — plus its LLM
cost is written to the SQLite store in `db.py` as it happens.

Run (after `chia up ~/aether/cluster/aether-local.yaml -y`):

    chia job submit --working-dir ~/aether/loop -- python loop.py
    python loop.py --kernel vec-softmax --baseline-only
    python loop.py --kernel vec-sgemv --iters 5 --budget-usd 25
    python loop.py --kernel llama-softmax --seed 20260907-043200-f0d7
    python db.py --summary
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ray

from chia.base.ChiaFunction import get
from chia.base.tools.BashTool import BashTool
from chia.trace.profiler import get_collector, start_collector, stop_collector

import cache
import db
# collateral.build_inputs / pristine_kernel are wrapped by nodes.*_for (target dispatch)
from constants import (
    AGENT_WORK_DIR,
    BASH_RESOURCE,
    DEFAULT_KERNEL,
    LLM_MODEL,
    MAX_ITERS,
    OUT_DIR,
    RUNTIME_ENV,
    SIM_CONFIG,
)
from context import CONFIG, K, select
from costs import CostMeter, EMPTY, fmt as fmt_cost
from kernels import kernel_names
import llm as agent
from nodes import (
    Measurement,
    build_inputs_for,
    build_kernel,
    build_simulator,
    measure,
    pristine_kernel_for,
    read_kernel,
    run_kernel,
    stage_workspace,
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("aether.loop")


# ---------------------------------------------------------------------------
# Result of one inner loop
# ---------------------------------------------------------------------------

@dataclass
class InnerResult:
    """Everything the outer loop needs to score one hardware design point."""

    run_id: str
    out_dir: Path
    kernel: str                       # benchmark name (kernels.py registry)
    config: str                       # simulator config the sim was built for

    ok: bool                          # baseline passed and the loop completed
    status: str                       # "ok" | "baseline_failed" | "unavailable"

    baseline_cycles: int | None = None      # mcycle of the pristine kernel
    baseline_instret: int | None = None
    best_cycles: int | None = None          # mcycle of the best passing kernel
    best_instret: int | None = None
    best_kernel_path: Path | None = None    # source of the best kernel
    best_iter: int = 0                      # 0 == baseline was never beaten

    # --seed / --seed-run: the loop still measures the pristine kernel as the
    # baseline (the speedup denominator), but hands the agent a previously
    # optimized kernel to edit from. Both numbers are kept.
    seed: str = ""                          # the seed spec that was requested
    seed_cycles: int | None = None          # mcycle of the seed kernel
    seed_kernel_path: Path | None = None    # where the seed source was written

    iters_planned: int = 0
    iters_run: int = 0
    history: list[dict] = field(default_factory=list)   # == history.json
    cost: dict = field(default_factory=lambda: dict(EMPTY))  # costs.py totals
    note: str = ""                    # e.g. "budget exhausted"

    # -- convenience -------------------------------------------------------

    @property
    def speedup(self) -> float:
        if not self.baseline_cycles or not self.best_cycles:
            return 1.0
        return self.baseline_cycles / self.best_cycles

    @property
    def cost_usd(self) -> float:
        return float(self.cost.get("cost_usd", 0.0))

    def summary(self) -> str:
        if not self.ok:
            return (f"kernel={self.kernel}  config={self.config}  "
                    f"FAILED ({self.status}) {self.note}\n")
        pct = ((1 - self.best_cycles / self.baseline_cycles) * 100
               if self.baseline_cycles and self.best_cycles else 0.0)
        c = self.cost
        seed = (f"seed={self.seed}  seed_cycles={self.seed_cycles}"
                f"  (baseline stays the speedup denominator)\n"
                if self.seed_cycles is not None else "")
        return (
            f"kernel={self.kernel}  config={self.config}  model={LLM_MODEL}\n"
            + seed +
            f"baseline={self.baseline_cycles}  best={self.best_cycles}  "
            f"speedup={self.speedup:.3f}x  ({pct:+.1f}%)\n"
            f"iterations={self.iters_run}/{self.iters_planned}"
            f"{'  ' + self.note if self.note else ''}\n"
            f"cost_usd={c['cost_usd']:.4f}  "
            f"input_tokens={c['input_tokens']}  "
            f"output_tokens={c['output_tokens']}  "
            f"cache_read_tokens={c['cache_read_input_tokens']}  "
            f"cache_creation_tokens={c['cache_creation_input_tokens']}  "
            f"llm_turns={c['num_turns']}\n"
        )


# ---------------------------------------------------------------------------
# Hardware: built once per design point, reused by value
# ---------------------------------------------------------------------------

# Process-local memo on top of the on-disk cache, so an outer loop that visits
# the same design point twice pays neither the build nor the unpickle.
_SIM_MEMO: dict[str, object] = {}
_BASELINE_MEMO: dict[str, Measurement] = {}


def build_sim(*, no_cache: bool = False, sim_key: str | None = None):
    """Build (or load) the simulator for the currently selected config.

    Idempotent and cheap to call repeatedly: memory memo -> `out/loop/_cache`
    -> real chisel build (~2 min). The outer loop calls this once per hardware
    design point and hands the result to :func:`run_inner`.
    """
    skey = sim_key or cache.sim_key()
    t0 = time.time()
    if not no_cache:
        sim = _SIM_MEMO.get(skey)
        if sim is not None:
            logger.info("Simulator %s reused from memory (sim-%s)", CONFIG(), skey)
            return sim
        sim = cache.load_simulator(skey)
        if sim is not None:
            logger.info("Simulator %s loaded from cache in %.1fs (sim-%s.pkl)",
                        CONFIG(), time.time() - t0, skey)
            _SIM_MEMO[skey] = sim
            return sim
    sim = build_simulator()
    cache.save_simulator(skey, sim)
    _SIM_MEMO[skey] = sim
    logger.info("Simulator %s built in %.0fs", CONFIG(), time.time() - t0)
    return sim


def _attempt(sim, kernel: bytes) -> tuple[Measurement, bytes | None, str, str]:
    """Compile + run one kernel. Returns (measurement, elf, build_stderr, sim_log)."""
    build = build_kernel(kernel)
    elf_name = K().elf_name
    if not build.success or elf_name not in build.files:
        return (measure(build, None), None, build.stderr or build.stdout, "")
    elf = build.files[elf_name]
    run = run_kernel(sim, elf)
    return (measure(build, run), elf, "", f"{run.log or ''}\n{run.out or ''}")


def measure_baseline(sim, kernel: bytes, *, no_cache: bool = False,
                     sim_key: str | None = None
                     ) -> tuple[Measurement, str, str]:
    """The pristine kernel's measurement on *sim*, cached by (sim, kernel).

    Returns (measurement, build_stderr, sim_log); the two logs are empty when
    the measurement came from a cache. Never re-runs the simulator if a passing
    baseline for this (simulator, kernel bytes) pair is already known.
    """
    bkey = cache.baseline_key(sim_key or cache.sim_key(), kernel, K().name)
    t0 = time.time()
    if not no_cache:
        m = _BASELINE_MEMO.get(bkey)
        if m is not None:
            logger.info("Baseline reused from memory (baseline-%s)", bkey)
            return m, "", ""
        m = cache.load_baseline(bkey)
        if m is not None:
            logger.info("Baseline loaded from cache in %.1fs (baseline-%s.pkl)",
                        time.time() - t0, bkey)
            _BASELINE_MEMO[bkey] = m
            return m, "", ""
    m, _, berr, slog = _attempt(sim, kernel)
    if m.ok:
        cache.save_baseline(bkey, m)
        _BASELINE_MEMO[bkey] = m
    return m, berr, slog


# ---------------------------------------------------------------------------
# Seeding: start the agent from an earlier run's best kernel
# ---------------------------------------------------------------------------

#: ``--seed best`` means "the best passing kernel in the DB for this
#: (kernel, config)", as opposed to ``--seed <RUN_ID>``.
SEED_BEST = "best"

#: Iteration number used for the seed's own measurement row. Iteration 0 stays
#: the pristine baseline so speedups remain comparable across rounds; the seed
#: is recorded *beside* it at -1 rather than displacing it. No schema change:
#: ``iters`` is keyed on (run_id, iter) and accepts a negative iter.
SEED_ITER = -1


def _seed_row(spec: str, kernel_name: str, config: str) -> dict | None:
    """The DB row of the iteration a seed spec points at (None if unknown).

    Only real optimization iterations are eligible (``iter > 0``), so a
    baseline row — or another run's seed row — can never be seeded from.
    """
    if spec == SEED_BEST:
        return db.query_one(
            "SELECT i.*, r.out_dir AS out_dir FROM iters i JOIN runs r USING (run_id)"
            " WHERE r.kernel = ? AND r.config = ? AND i.iter > 0"
            " AND i.fitness IS NOT NULL"
            " ORDER BY i.fitness ASC, i.iter ASC LIMIT 1",
            (kernel_name, config))
    return db.query_one(
        "SELECT i.*, r.out_dir AS out_dir FROM iters i JOIN runs r USING (run_id)"
        " WHERE i.run_id = ? AND i.iter > 0 AND i.fitness IS NOT NULL"
        " ORDER BY i.fitness ASC, i.iter ASC LIMIT 1",
        (spec,))


def resolve_seed(spec: str, kernel_name: str,
                 config: str) -> tuple[bytes | None, str]:
    """Resolve a seed spec to kernel bytes. Returns ``(source, description)``.

    The DB record wins (``iters.kernel_path`` of the best passing iteration);
    ``<out_dir>/kernel_best.*`` is the fallback, so a run whose rows never made
    it into the store can still be seeded from. ``(None, reason)`` when nothing
    could be resolved — the caller then falls back to the pristine kernel.
    """
    row = _seed_row(spec, kernel_name, config)
    tried: list[str] = []

    def _read(p: Path) -> bytes | None:
        tried.append(str(p))
        try:
            return p.read_bytes() if p.is_file() else None
        except OSError:
            return None

    candidates: list[Path] = []
    if row:
        kp = (row.get("kernel_path") or "").strip()
        if kp:
            candidates.append(Path(kp))
        od = (row.get("out_dir") or "").strip()
        if od:
            candidates += sorted(Path(od).glob("kernel_best.*"))
    if spec != SEED_BEST:
        candidates += sorted((Path(OUT_DIR) / spec).glob("kernel_best.*"))

    for c in candidates:
        src = _read(c)
        if src:
            where = str(c)
            if row:
                return src, (f"{where} (run {row['run_id']} iter {row['iter']}, "
                             f"{row['fitness']} cycles)")
            return src, where
    return None, (f"no seed kernel for {spec!r} "
                  f"(tried: {', '.join(tried) or 'nothing'})")


# ---------------------------------------------------------------------------
# The inner loop
# ---------------------------------------------------------------------------

def run_inner(sim, kernel=None, iters: int = MAX_ITERS, *,
              run_id: str | None = None,
              out_dir: Path | str | None = None,
              budget_usd: float | None = None,
              no_cache: bool = False,
              baseline_only: bool = False,
              sim_key: str | None = None,
              design_point_id: int | None = None,
              seed: str | None = None,
              meter: CostMeter | None = None) -> InnerResult:
    """Optimize one kernel on one already-built simulator.

    Parameters
    ----------
    sim
        The simulator handle from :func:`build_sim` / `nodes.build_simulator`
        (a `chia.chipyard.state_def.BuildArtifact`). Never rebuilt here.
    kernel
        Benchmark name from `kernels.py`; ``None`` keeps whatever `context.py`
        currently has selected. The caller is responsible for having called
        ``context.select(kernel, config)`` (or passing the name here) so that
        the selected config matches *sim*.
    iters
        Optimization rounds after the baseline.
    run_id, out_dir
        Identity of this inner run; defaults to a UTC timestamp under
        ``out/loop/<run_id>``.
    budget_usd
        Stop after the iteration in which the accumulated LLM cost (costs.py)
        exceeds this. ``None`` = no limit.
    no_cache
        Ignore the on-disk simulator/baseline cache (remeasure the baseline).
    baseline_only
        Measure the pristine kernel and stop (smoke test); no LLM is used.
    sim_key
        Cache key identifying *sim*; defaults to ``cache.sim_key()`` for the
        currently selected config.
    design_point_id
        Row id from ``db.add_design_point()``; every run/iteration this call
        writes is tagged with it so the outer loop can query per design point
        (``db.sample_parent``). ``None`` for a standalone inner run.
    seed
        Start the agent from an already-optimized kernel instead of the
        pristine one: a previous ``run_id``, or ``"best"`` for the best passing
        iteration in the DB for this (kernel, config). ``None`` (the default)
        keeps the pristine kernel as the starting point. The baseline — and
        therefore the speedup denominator — is the pristine kernel either way;
        the seed's own measurement is reported separately as ``seed_cycles``.
    meter
        Cost meter to bill this run to; one is created if omitted. Requires a
        running profiler collector (``start_collector()``) for non-zero costs.

    Assumes Ray is initialized and (unless ``baseline_only``) that the profiler
    collector is running. Neither is started or stopped here — that is the
    driver's job, so an outer loop keeps one collector across design points.
    """
    k = select(kernel) if kernel else K()
    run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                         + "-" + secrets.token_hex(2))
    out = Path(out_dir) if out_dir else Path(OUT_DIR) / run_id
    out.mkdir(parents=True, exist_ok=True)
    meter = meter if meter is not None else CostMeter()
    iters = 0 if baseline_only else max(0, int(iters))
    skey = sim_key or cache.sim_key()

    res = InnerResult(run_id=run_id, out_dir=out, kernel=k.name, config=CONFIG(),
                      ok=False, status="unavailable", iters_planned=iters)

    if not k.available:
        res.note = k.unavailable_reason or "kernel unavailable"
        logger.error("kernel %s is registered but not runnable: %s",
                     k.name, res.note)
        return res

    logger.info("Run %s  kernel=%s  config=%s -> %s",
                run_id, k.name, CONFIG(), out)
    db.start_run(run_id, CONFIG(), k.name, LLM_MODEL, iters, str(out),
                 design_point_id=design_point_id)

    history = res.history
    total = res.cost

    def record(**kw) -> None:
        history.append(kw)
        (out / "history.json").write_text(json.dumps(history, indent=2))

    # --- baseline (never re-measured when a cached one exists) -------------
    kernel_src = pristine_kernel_for()
    base_m, berr, slog = measure_baseline(sim, kernel_src, no_cache=no_cache,
                                          sim_key=skey)
    if not base_m.ok:
        logger.error("Baseline does not pass — the loop cannot score anything.\n%s",
                     base_m.detail)
        (out / "baseline_failure.txt").write_text(f"{berr}\n{slog}")
        db.record_iter(run_id, 0, base_m.kind, base_m.cycles, base_m.instret,
                       False, note=base_m.detail[:500])
        db.finish_run(run_id, None, None, note="baseline failed")
        res.status, res.note = "baseline_failed", base_m.detail[:500]
        return res

    baseline = base_m.cycles
    ref = k.reference_cycles
    logger.info("Baseline: cycles=%d instret=%s%s", baseline, base_m.instret,
                f" (reference was {ref})" if ref else "")
    record(iter=0, kind="baseline", cycles=baseline, instret=base_m.instret,
           cost=dict(EMPTY))
    suffix = Path(k.kernel_rel).suffix
    kpath = out / f"kernel_baseline{suffix}"
    kpath.write_bytes(kernel_src)
    # The baseline is not attributable to any LLM turn — cost stays zero.
    parent_iid = db.record_iter(run_id, 0, "baseline", baseline, base_m.instret,
                                True, kernel_path=str(kpath),
                                note=base_m.detail[:500])

    res.status = "ok"
    res.ok = True
    res.baseline_cycles = baseline
    res.baseline_instret = base_m.instret
    res.best_cycles = baseline
    res.best_instret = base_m.instret
    res.best_kernel_path = kpath

    if baseline_only:
        logger.info("baseline-only: stopping after the smoke test")
        db.finish_run(run_id, baseline, baseline, note="baseline-only")
        res.note = "baseline-only"
        (out / "summary.txt").write_text(res.summary())
        return res

    # --- optional seed: the agent starts from an already-optimized kernel ---
    # Iteration 0 above stays the pristine baseline, so `baseline`/`speedup`
    # remain comparable with unseeded runs. Only the *starting point* moves.
    seed_note = ""
    start_m = base_m                     # the measurement the agent is shown
    if seed:
        res.seed = seed
        seed_src, seed_where = resolve_seed(seed, k.name, CONFIG())
        if seed_src is None:
            logger.warning("--seed %s: %s — starting from the pristine kernel",
                           seed, seed_where)
            seed_note = f"seed {seed} unresolved"
        elif seed_src == kernel_src:
            logger.info("--seed %s resolves to the pristine kernel; nothing to do",
                        seed)
        else:
            logger.info("Seeding from %s", seed_where)
            spath = out / f"kernel_seed{suffix}"
            spath.write_bytes(seed_src)
            seed_m, _, _, _ = _attempt(sim, seed_src)
            logger.info("Seed: %s", seed_m.detail)
            record(iter=SEED_ITER, kind="seed", cycles=seed_m.cycles,
                   instret=seed_m.instret, detail=seed_m.detail,
                   source=seed_where, cost=dict(EMPTY))
            if not seed_m.ok:
                logger.warning("seed kernel does not pass (%s) — starting from "
                               "the pristine kernel instead", seed_m.kind)
                seed_note = f"seed {seed} failed ({seed_m.kind})"
            else:
                logger.info("seed_cycles=%d  baseline_cycles=%d  "
                            "(speedup denominator stays the baseline)",
                            seed_m.cycles, baseline)
                parent_iid = db.record_iter(
                    run_id, SEED_ITER, "seed", seed_m.cycles, seed_m.instret,
                    True, kernel_path=str(spath),
                    note=f"seed from {seed_where}: {seed_m.detail}"[:500],
                    parent_id=parent_iid,
                    diff=db.make_diff(kernel_src, seed_src,
                                      from_name=parent_iid,
                                      to_name=db.iter_id(run_id, SEED_ITER)))
                seed_note = (f"seeded from {seed} ({seed_where}); "
                             f"seed_cycles={seed_m.cycles} "
                             f"baseline_cycles={baseline}")
                res.seed_cycles, res.seed_kernel_path = seed_m.cycles, spath
                kernel_src, start_m = seed_src, seed_m
                if seed_m.cycles < baseline:
                    # The seed is already better than the pristine kernel, so
                    # it — not the baseline — is the bar the agent must clear.
                    res.best_cycles, res.best_instret = seed_m.cycles, seed_m.instret
                    res.best_kernel_path = out / f"kernel_best{suffix}"
                    res.best_kernel_path.write_bytes(seed_src)

    # --- agent workspace + tool -------------------------------------------
    get(stage_workspace.chia_remote(build_inputs_for(kernel_src), AGENT_WORK_DIR))
    kernel_bash = BashTool(
        name="kernel_bash",
        work_dir=AGENT_WORK_DIR,
        timeout_seconds=300,
        task_options={"resources": {"riscv_build": BASH_RESOURCE}},
    )

    llm = agent.make_llm()
    best, best_kernel = baseline, kernel_src
    if res.seed_cycles is not None and res.seed_cycles < best:
        best = res.seed_cycles          # seeded: the seed is the bar to beat
    feedback = ""

    for i in range(1, iters + 1):
        logger.info("--- iteration %d/%d (best=%d, baseline=%d, spent=$%.4f) ---",
                    i, iters, best, baseline, total["cost_usd"])

        if i == 1:
            reply = agent.first_turn(llm, kernel_bash, start_m, kernel_src)
        else:
            reply = agent.next_turn(llm, kernel_bash, feedback)
        (out / f"agent_{i:02d}.txt").write_text(reply.result or "")

        # Cost of the turn that produced this iteration's kernel.
        cost = meter.poll()
        total.update(CostMeter.add(total, cost))
        logger.info("iteration %d LLM usage: %s", i, fmt_cost(cost))

        kernel_src = get(read_kernel.chia_remote(AGENT_WORK_DIR, k.kernel_rel))
        kpath = out / f"kernel_{i:02d}{suffix}"
        kpath.write_bytes(kernel_src)

        m, _, berr, slog = _attempt(sim, kernel_src)
        (out / f"simlog_{i:02d}.txt").write_text(slog or "")
        logger.info("iteration %d: %s", i, m.detail)
        record(iter=i, kind=m.kind, cycles=m.cycles, instret=m.instret,
               detail=m.detail, cost=cost)
        db.record_iter(run_id, i, m.kind, m.cycles, m.instret, m.ok,
                       kernel_path=str(kpath), note=m.detail[:500], cost=cost,
                       parent_id=parent_iid,
                       diff=db.make_diff(best_kernel, kernel_src,
                                         from_name=parent_iid,
                                         to_name=db.iter_id(run_id, i)))
        res.iters_run = i

        if m.ok and m.cycles < best:
            # The next attempt is built on this kernel, so it is its parent.
            parent_iid = db.iter_id(run_id, i)
            best, best_kernel = m.cycles, kernel_src
            best_path = out / f"kernel_best{suffix}"
            best_path.write_bytes(kernel_src)
            res.best_cycles, res.best_instret = m.cycles, m.instret
            res.best_kernel_path, res.best_iter = best_path, i

        if budget_usd is not None and total["cost_usd"] >= budget_usd:
            res.note = (f"budget exhausted after iteration {i} "
                        f"(${total['cost_usd']:.4f} >= ${budget_usd:.4f})")
            logger.warning("%s — stopping early", res.note)
            break

        feedback = agent.format_feedback(m, best, baseline, berr, slog,
                                         history=history,
                                         best_kernel=best_kernel)

    summary = res.summary()
    logger.info("DONE\n%s", summary)
    (out / "summary.txt").write_text(summary)
    (out / "cost.json").write_text(json.dumps(total, indent=2))
    db.finish_run(run_id, baseline, best,
                  note="; ".join(x for x in (seed_note, res.note) if x))
    return res


# ---------------------------------------------------------------------------
# CLI: one design point = build the sim once, then one inner loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default=DEFAULT_KERNEL,
                    metavar="NAME",
                    help=f"benchmark to optimize (default {DEFAULT_KERNEL}); "
                         f"one of: {', '.join(kernel_names())}")
    ap.add_argument("--config", default=SIM_CONFIG, metavar="NAME",
                    help=f"chipyard simulator config (default {SIM_CONFIG})")
    ap.add_argument("--iters", type=int, default=MAX_ITERS,
                    help="optimization rounds after the baseline")
    ap.add_argument("--baseline-only", action="store_true",
                    help="measure the pristine kernel and stop (smoke test)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the on-disk simulator/baseline cache; "
                         "always rebuild the simulator and remeasure the baseline")
    ap.add_argument("--budget-usd", type=float, default=None, metavar="USD",
                    help="stop once the accumulated LLM cost reaches this")
    ap.add_argument("--seed", default=None, metavar="RUN_ID|best",
                    help="start the agent from an already-optimized kernel "
                         "instead of the pristine one: a previous RUN_ID (its "
                         "best passing iteration, taken from the DB's "
                         "kernel_path and falling back to that run's "
                         "kernel_best.* on disk), or 'best' for the best "
                         "passing iteration in the DB for this kernel+config. "
                         "Iteration 0 is still the pristine baseline, so the "
                         "speedup denominator is unchanged; the seed's own "
                         "measurement is logged and stored as seed_cycles "
                         "(iteration -1 and runs.note). Default: pristine.")
    ap.add_argument("--seed-run", dest="seed", metavar="RUN_ID",
                    help="alias for --seed RUN_ID")
    args = ap.parse_args()

    k = select(args.kernel, args.config)
    if not k.available:
        logger.error("kernel %s is registered but not runnable: %s",
                     k.name, k.unavailable_reason)
        return 2

    ray.init(address="auto", runtime_env=RUNTIME_ENV)

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
              + "-" + secrets.token_hex(2))
    out = Path(OUT_DIR) / run_id
    out.mkdir(parents=True, exist_ok=True)
    # Profiler collector: this is what makes the LLM's cost/token numbers
    # reachable (chia's ClaudeCodeLLM pushes them as call metadata). Started
    # here, once, so run_inner can be called repeatedly under one collector.
    started = get_collector() is None
    if started:
        start_collector(log_dir=str(out / "profiler"))
    try:
        # Hardware first: built (or loaded) exactly once, then reused by value.
        sim = build_sim(no_cache=args.no_cache)
        res = run_inner(sim, args.kernel, args.iters,
                        run_id=run_id, out_dir=out,
                        budget_usd=args.budget_usd,
                        no_cache=args.no_cache,
                        baseline_only=args.baseline_only,
                        seed=args.seed)
    finally:
        if started:
            stop_collector()
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
