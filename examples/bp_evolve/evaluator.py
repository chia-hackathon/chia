"""The evaluator interface: ``build_fn``, ``run_fn``, ``result_mapper_fn``.

These are the three green nodes in the proposal's figure, and the reason the
loop is a CHIA flow rather than a script.  Three simulators, three build
recipes, three resource pools and one cascade is exactly the shape the
interface describes:

* :func:`build_fn` runs the lint gate, then builds the tiers it is asked for.
* :func:`run_fn` fans a tier's traces across the worker pool and decides what
  gets promoted to the next tier.
* :func:`result_mapper_fn` folds the tiers into one fitness plus the cross-tier
  rank comparison, and returns a failure profile the agent can act on.

The division that matters is that **none of them calls an LLM**.  The agent
writes source; these three rule on it.  A design is withdrawn, repaired,
archived or promoted by the rules in :mod:`harcom_lint` and the numbers in
:mod:`vfs`, never by a model's opinion of its own work.

On why ``build_fn`` takes a tier list: the figure draws it feeding all three
builds, and it owns all three.  But a variant that dies at Tier 0 should not
have paid for a gem5 build, so the driver asks for tier 0 first and comes back
for 1 and 2 on promotion.  Same function, called again -- not a different path.
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum

from chia.base.ChiaFunction import get

import constants as C
import gem5_bp
import harcom_lint
from archive import Elite
from vfs import Metrics, TraceCounters, aggregate, kendall_tau, sweep


class Tier(int, Enum):
    CBP_NG = 0
    CHAMPSIM = 1
    GEM5 = 2


class Failure(str, Enum):
    """Why a variant stopped, in the vocabulary the repair prompt uses.

    Ordered roughly by how early it happens.  ``WITHDRAWN`` is the only one
    that is not offered a repair round: it means the design reached around
    HARCOM, and repairing it would teach the agent that the rule bends.
    """
    NONE = "none"
    WITHDRAWN = "withdrawn"                 # a cheat; not repairable
    LINT = "lint"                           # illegal HARCOM, repairable
    BUILD = "build"                         # compiler rejected it
    RUN = "run"                             # crashed or timed out
    NO_IMPROVEMENT = "no_improvement"       # scored, archived nothing
    PORT_MISMATCH = "port_mismatch"         # tiers disagree on MPKI
    RANK_INVERSION = "rank_inversion"       # tiers disagree on ranking


@dataclass
class Variant:
    """A candidate design and its per-tier sources.

    ``champsim_source`` and ``gem5_source`` are filled in only on promotion, by
    the porting agent.  They are the translations the proposal insists are part
    of the work: a HARCOM predictor is called per instruction with a
    block-based two-level timed interface, and both other simulators call a
    predictor once per branch.
    """
    variant_id: str
    struct_name: str
    harcom_source: str
    parent_id: str | None = None
    generation: int = 0
    rationale: str = ""
    template_args: str = ""
    """Contents of the angle brackets in ``-DPREDICTOR=Struct<...>``.

    Empty means every template parameter takes its default.  It lives on the
    variant rather than being passed alongside it because it is *part of the
    design*: two variants sharing a source but differing here are different
    predictors, and recording only the source would make them indistinguishable
    in the lineage table.
    """
    champsim_source: str | None = None
    gem5_source: str | None = None
    gem5_impl: str | None = None
    """gem5's ``.cc``.  ChampSim modules are header-only; a gem5 branch
    predictor is a SimObject and needs a translation unit as well as a
    declaration, so the port produces two files rather than one."""


@dataclass
class TierOutcome:
    """What one tier produced for one variant."""
    tier: Tier
    built: bool = False
    ran: bool = False
    build_diagnostics: str = ""
    run_diagnostics: str = ""
    build_duration_s: float = 0.0
    # Tier 0
    counters: list[TraceCounters] = field(default_factory=list)
    metrics: Metrics | None = None
    depth_scores: dict[int, Metrics] = field(default_factory=dict)
    # Tiers 1 and 2
    mpki: float | None = None
    ipc: float | None = None
    rob_occupancy_at_mispredict: float | None = None
    per_config_ipc: dict[str, float] = field(default_factory=dict)


@dataclass
class Evaluation:
    """Everything known about one variant after the cascade.

    ``failure`` and ``feedback`` are the only fields the agent ever sees.  The
    rest is for the lineage DB and the analysis.
    """
    variant: Variant
    tiers: dict[Tier, TierOutcome] = field(default_factory=dict)
    fitness: float | None = None
    failure: Failure = Failure.NONE
    feedback: str = ""
    archived: bool = False
    cell: tuple[int, int, int] | None = None
    tau_vfs_vs_ipc: float | None = None

    @property
    def tier0(self) -> TierOutcome | None:
        return self.tiers.get(Tier.CBP_NG)

    def to_elite(self) -> Elite | None:
        """The archive entry for this variant, or None if it never scored."""
        t0 = self.tier0
        if t0 is None or t0.metrics is None:
            return None
        m = t0.metrics
        return Elite(
            variant_id=self.variant.variant_id,
            struct_name=self.variant.struct_name,
            source=self.variant.harcom_source,
            parent_id=self.variant.parent_id,
            generation=self.variant.generation,
            template_args=self.variant.template_args,
            vfs=m.vfs, epi=m.epi, p1_latency=m.p1_latency,
            p2_latency=m.p2_latency, mpki=m.mpki, ipc=m.ipc, cpi=m.cpi,
            n_traces=m.n_traces,
            depth_scores={d: s.vfs for d, s in t0.depth_scores.items()},
            notes=self.variant.rationale[:500],
        )


# ---------------------------------------------------------------------------
# build_fn
# ---------------------------------------------------------------------------

def build_fn(
    variant: Variant,
    tiers,
    *,
    cbp_node=None,
    champsim_node=None,
    gem5_node=None,
    cbp_root: str = C.CBP_NG_ROOT,
    champsim_root: str = C.CHAMPSIM_ROOT,
    gem5_root: str = C.GEM5_ROOT,
) -> tuple[dict[Tier, TierOutcome], dict[Tier, object]]:
    """Lint, then build ``variant`` for each tier in ``tiers``.

    Returns the per-tier outcomes and the raw build artifacts (binary bytes for
    Tier 0 and 1, a path artifact for Tier 2), which :func:`run_fn` needs and
    the outcomes deliberately do not carry -- a ChampSim binary is ~5 MB and
    should not end up in the lineage DB.

    The lint gate runs once, before any compiler.  It is cheap and the compile
    is not: at ~7 s per Tier-0 build and four variants a generation, catching a
    branch-on-value textually is most of a minute per generation that buys a
    better repair prompt than the template backtrace would.
    """
    outcomes: dict[Tier, TierOutcome] = {}
    artifacts: dict[Tier, object] = {}

    lint = harcom_lint.prelint(variant.harcom_source,
                               struct_name=variant.struct_name)
    missing = harcom_lint.missing_methods(variant.harcom_source)
    if not lint.ok or missing:
        out = TierOutcome(tier=Tier.CBP_NG)
        parts = []
        if not lint.ok:
            parts.append(lint.render())
        if missing:
            parts.append(
                "unimplemented pure virtuals from cbp.hpp's `predictor`: "
                + ", ".join(missing)
                + "\n  fix: every one must be defined, or the struct is "
                  "abstract and cbp.cpp cannot instantiate it.")
        out.build_diagnostics = "\n".join(parts)
        outcomes[Tier.CBP_NG] = out
        return outcomes, artifacts

    for tier in tiers:
        if tier is Tier.CBP_NG:
            res = get(cbp_node.build_cbp.chia_remote(
                cbp_root, variant.harcom_source, variant.struct_name,
                variant.variant_id, template_args=variant.template_args,
                timeout_s=C.CBP_BUILD_TIMEOUT_S))
            outcomes[tier] = TierOutcome(
                tier=tier, built=res.success,
                build_diagnostics=res.diagnostics,
                build_duration_s=res.build_duration_s)
            if res.success:
                artifacts[tier] = res.binary

        elif tier is Tier.CHAMPSIM:
            if not variant.champsim_source:
                outcomes[tier] = TierOutcome(
                    tier=tier,
                    build_diagnostics="no ChampSim translation was produced")
                continue
            res = get(champsim_node.build_champsim.chia_remote(
                champsim_root, variant.champsim_source, "evolved_bp",
                module_type="branch", timeout_s=C.CHAMPSIM_BUILD_TIMEOUT_S,
                incremental=True))
            outcomes[tier] = TierOutcome(
                tier=tier, built=res.success,
                build_diagnostics=res.build_diagnostics,
                build_duration_s=res.build_duration_s)
            if res.success:
                artifacts[tier] = res.binary

        elif tier is Tier.GEM5:
            if not (variant.gem5_source and variant.gem5_impl):
                outcomes[tier] = TierOutcome(
                    tier=tier,
                    build_diagnostics="no gem5 translation was produced")
                continue
            # A gem5 branch predictor is a SimObject, so it has to be in the
            # source tree, declared to the Python object system and listed in
            # the SConscript before scons will look at it.  That is three
            # edits on the worker, and they have to happen before the build,
            # not as part of it.
            inst = get(gem5_bp.install_gem5_predictor.chia_remote(
                gem5_root, variant.gem5_source, variant.gem5_impl,
                C.GEM5_CONFIG.read_text()))
            if not inst.success:
                outcomes[tier] = TierOutcome(
                    tier=tier, build_diagnostics=inst.message)
                continue
            # Built twice on failure, deliberately. Registering a new
            # SimObject makes scons regenerate the params headers, and under
            # -j that regeneration races its own consumers: the first pass can
            # die on "params/SomethingUnrelated.hh: No such file or directory"
            # for a header that has nothing to do with the predictor, and the
            # second pass then succeeds with no other change. Retrying once is
            # cheaper and more honest than pinning the build to -j1.
            res = get(gem5_node.build_gem5.chia_remote(
                gem5_root, isa=C.GEM5_ISA, variant=C.GEM5_VARIANT,
                timeout_s=C.GEM5_BUILD_TIMEOUT_S))
            if not res.success:
                res = get(gem5_node.build_gem5.chia_remote(
                    gem5_root, isa=C.GEM5_ISA, variant=C.GEM5_VARIANT,
                    timeout_s=C.GEM5_BUILD_TIMEOUT_S))
            outcomes[tier] = TierOutcome(
                tier=tier, built=res.success,
                build_diagnostics=res.stderr_tail,
                build_duration_s=res.build_duration_s)
            if res.success:
                artifacts[tier] = res.binary_path

    return outcomes, artifacts


# ---------------------------------------------------------------------------
# run_fn
# ---------------------------------------------------------------------------

def run_fn_tier0(
    variant: Variant,
    binary: bytes,
    traces: list[str],
    *,
    cbp_node,
    depths=C.DEPTH_SWEEP,
) -> TierOutcome:
    """Score ``binary`` on every trace, then aggregate at every depth.

    The traces are dispatched as one batch of refs and collected together, so
    the pool runs them concurrently; the depth sweep afterwards costs no
    simulation at all, because depth enters the score only through CPI.  That
    asymmetry is what makes the whole question affordable: re-scoring 168
    traces at eight depths is eight arithmetic passes over counters that were
    measured once.

    A single failed trace fails the variant.  Aggregating over the survivors
    instead would let a design that crashes on its hardest traces outscore one
    that survives them.
    """
    out = TierOutcome(tier=Tier.CBP_NG, built=True)

    # Longest trace first.  Ray hands out queued tasks in submission order, and
    # what ends a generation is not the total work but the last trace to finish
    # -- so a 130M trace that starts late runs on one slot while fifteen sit
    # idle.  Submitting the long ones first lets the short ones fill in behind
    # them, which is the standard longest-processing-time rule for minimising
    # makespan.  Traces of unknown length sort last, at 0; they fall back to the
    # flat timeout anyway, so there is nothing better to say about them.
    #
    # The results are permuted back into the caller's order before anything
    # reads them.  Nothing downstream depends on trace order -- VFS is an
    # unweighted mean and the self-check matches counters by name -- but the
    # per-trace counters are the audit trail for the aggregate, and a list that
    # reorders itself by which trace happened to be longest is one that cannot
    # be diffed against a previous run.
    order = sorted(range(len(traces)),
                   key=lambda i: -(_trace_length(traces[i]) or 0))
    refs = [
        cbp_node.run_cbp.chia_remote(
            binary, traces[i],
            warmup_instructions=C.WARMUP_INSTRUCTIONS,
            simulation_instructions=C.SIM_INSTRUCTIONS,
            timeout_s=cbp_run_timeout_for(traces[i]))
        for i in order
    ]
    dispatched = get(refs)
    results = [None] * len(traces)
    for i, r in zip(order, dispatched):
        results[i] = r

    failed = [r for r in results if not r.success]
    if failed:
        first = failed[0]
        out.run_diagnostics = (
            f"{len(failed)}/{len(results)} traces failed; first was "
            f"{first.trace_name} (rc={first.returncode}"
            f"{', timed out' if first.timed_out else ''}): "
            f"{first.stderr_tail[:600]}")
        return out

    out.ran = True
    out.counters = [r.counters for r in results]
    out.depth_scores = sweep(out.counters, depths)
    out.metrics = out.depth_scores.get(C.SHIPPED_DEPTH) or aggregate(
        out.counters, C.SHIPPED_DEPTH)
    out.mpki = out.metrics.mpki
    return out


def _conditional_mpki(result) -> float | None:
    """Mispredictions per kilo-instruction over the branch types Tier 0 sees.

    ChampSim's headline ``branch_mpki`` charges the predictor for every branch
    type, including the indirect targets and returns that the BTB and the
    return-address stack get wrong.  A HARCOM predictor is never asked about
    any of those -- it answers "taken or not" for conditional branches -- so
    the totals are not comparable and their difference is dominated by
    structures the design does not contain.  On the seed and the converted gcc
    trace, total MPKI is 8.28 against a Tier-0 5.62, while the conditional rate
    is 5.36: one of those two comparisons is about branch prediction.
    """
    counts = getattr(result, "branch_mispredictions", None) or {}
    instructions = getattr(result, "instructions", 0) or 0
    if not counts or instructions <= 0:
        return None
    wanted = [counts.get(t, 0) for t in C.TIER1_MPKI_BRANCH_TYPES]
    if not any(t in counts for t in C.TIER1_MPKI_BRANCH_TYPES):
        # The stat names changed under us; a silent zero here would read as a
        # perfect predictor.
        return None
    return sum(wanted) * 1000.0 / instructions


_TRACE_LENGTHS: dict | None = None


def _trace_lengths() -> dict:
    """basename -> instruction count, loaded once.

    Missing or unreadable is not fatal: every caller falls back to the global
    window and says so. A fidelity check that silently used the wrong number
    would be worse than one that admits it does not know.
    """
    global _TRACE_LENGTHS
    if _TRACE_LENGTHS is None:
        try:
            with open(C.TRACE_LENGTHS_FILE) as fh:
                _TRACE_LENGTHS = json.load(fh)
        except Exception:
            _TRACE_LENGTHS = {}
    return _TRACE_LENGTHS


def _trace_length(trace: str) -> int | None:
    """Measured instruction count for a CBP or ChampSim trace path.

    trace_lengths.json is keyed by the CBP name (``X_trace.gz``); the ChampSim
    conversion of the same trace is ``X_trace.champsimtrace.gz``
    (tools/cbp2champsim.cpp, convert_all.sh).  Looking the ChampSim name up
    as-is missed every time, so Tier 1 always fell back to the global window.
    """
    name = os.path.basename(trace)
    if name.endswith(".champsimtrace.gz"):
        name = name[: -len(".champsimtrace.gz")] + ".gz"
    return _trace_lengths().get(name)


def tier1_window_for(trace: str,
                     warmup: int = C.WARMUP_INSTRUCTIONS) -> tuple[int, bool]:
    """How many instructions Tier 1 should simulate on this trace.

    Returns ``(instructions, exact)``.  ``exact`` is True when the answer came
    from a measured trace length, which is the only case in which Tier 1 runs
    to the end of the trace and stops there.

    This has to be per-trace.  ChampSim reopens a trace it runs off the end of,
    so the window is the only thing that stops it, and the official set runs
    from 13.0M to 130.0M instructions -- a tenfold spread.  One global number
    either wraps the short traces (measuring a predictor that has seen the
    program several times) or truncates the long ones (throwing away 90% of the
    workload).  Neither failure announces itself in the MPKI.
    """
    n = _trace_length(trace)
    if not n:
        return C.TIER1_SIM_INSTRUCTIONS, False
    return max(1, n - warmup - C.TIER1_TAIL_MARGIN), True


def cbp_run_timeout_for(trace: str) -> int:
    """How long to allow one Tier-0 HARCOM run, sized from the trace itself.

    Tier 0 runs every trace to its end, so the time a run needs is set by the
    trace, and the official set spans a factor of ten in length and a further
    2.5 in rate -- ``compress_44_trace`` sustains 124 kIPS with a core to
    itself where ``int_32_trace`` does 317, because HARCOM's work is per
    prediction and branch density differs.  Contended across sixteen slots the
    slow end drops under 67 kIPS.

    A flat 1800s therefore killed every trace past roughly 110M instructions --
    six of the 168 -- on every variant.  Because one failed trace fails the
    variant (deliberately: see run_fn_tier0), that meant every variant failed
    and the archive never grew past the seed, while each generation still cost
    a full hour of compute to learn nothing.

    Falls back to the flat floor when the length is unknown, which is the same
    thing tier1_window_for does and for the same reason: guessing a number here
    would be worse than admitting the set is not measured.
    """
    n = _trace_length(trace)
    if not n:
        return C.CBP_RUN_TIMEOUT_S
    return max(C.CBP_RUN_TIMEOUT_S,
               C.CBP_RUN_TIMEOUT_SLACK_S + int(n / (C.CBP_RUN_FLOOR_KIPS * 1000)))


def run_fn_tier1(
    variant: Variant,
    binary: bytes,
    traces: list[str],
    *,
    champsim_node,
) -> TierOutcome:
    """Run the ChampSim port and read back MPKI and ROB occupancy.

    ``avg_rob_occupancy_at_mispredict`` is the number this tier exists for: it
    says how much work each mispredict actually squashed, which is precisely
    what CBP-NG's fixed nine-stage penalty assumes is constant.
    """
    out = TierOutcome(tier=Tier.CHAMPSIM, built=True)
    # Not C.SIM_INSTRUCTIONS, and not one shared number either: ChampSim
    # reopens a trace it runs off the end of, so the window is per-trace and
    # comes from the measured length.  See tier1_window_for.
    windows = [tier1_window_for(t) for t in traces]
    inexact = [t for t, (_, exact) in zip(traces, windows) if not exact]
    if inexact:
        out.run_diagnostics = (
            f"{len(inexact)}/{len(traces)} traces have no measured length, so "
            f"Tier 1 fell back to the global {C.TIER1_SIM_INSTRUCTIONS:,}-"
            f"instruction window and may have wrapped them; first is "
            f"{os.path.basename(inexact[0])}. Rebuild trace_lengths.json.")
    refs = [
        champsim_node.run_champsim.chia_remote(
            binary, t,
            warmup_instructions=C.WARMUP_INSTRUCTIONS,
            simulation_instructions=w,
            timeout_s=C.CHAMPSIM_RUN_TIMEOUT_S)
        for t, (w, _) in zip(traces, windows)
    ]
    results = get(refs)
    ok = [r for r in results if r.success]
    if not ok:
        first = results[0] if results else None
        out.run_diagnostics = (
            "every ChampSim run failed"
            + (f"; first rc={first.returncode}: {first.stdout_tail[-400:]}"
               if first else ""))
        return out

    out.ran = True
    mpkis = [m for m in (_conditional_mpki(r) for r in ok) if m is not None]
    robs = [r.avg_rob_occupancy_at_mispredict for r in ok
            if r.avg_rob_occupancy_at_mispredict is not None]
    out.mpki = statistics.fmean(mpkis) if mpkis else None
    out.ipc = statistics.fmean([r.ipc for r in ok])
    out.rob_occupancy_at_mispredict = statistics.fmean(robs) if robs else None

    notes = []
    if len(ok) < len(results):
        notes.append(f"{len(results) - len(ok)}/{len(results)} traces failed")
    # A wrapped trace is not a failure ChampSim reports; it is a line in the
    # log.  Say so loudly, because everything downstream that compares this
    # MPKI to Tier 0's is measuring the window rather than the predictor.
    looped = [r for r in ok
              if C.TIER1_LOOP_MARKER in (getattr(r, "stdout_tail", "") or "")]
    if looped:
        notes.append(
            f"{len(looped)}/{len(ok)} runs wrapped the trace "
            f"(BPE_TIER1_SIM_INSTRUCTIONS={C.TIER1_SIM_INSTRUCTIONS} exceeds "
            f"the trace); Tier-1 MPKI is not comparable to Tier-0 MPKI")
    if notes:
        out.run_diagnostics = "; ".join(notes)
    return out


def run_fn_tier2(
    variant: Variant,
    gem5_bin: str,
    workloads: list[str],
    depths,
    *,
    gem5_node,
    outdir_root: str,
    config_script: str,
    workload_dir: str = C.GEM5_WORKLOAD_DIR,
    max_insts: int = C.GEM5_MAX_INSTS,
) -> TierOutcome:
    """Sweep the pipeline depth CBP-NG holds fixed, on a real O3 core.

    Every run in the sweep is the same predictor on the same program; only the
    distance from a prediction to its resolution changes.  That is the one
    variable CBP-NG's score cannot express, and the only reason this tier costs
    what it does.

    The label of each config is the depth as a decimal integer, because
    :func:`result_mapper_fn` parses it back out and pairs it with the Tier-0
    score at the same depth.  A label that does not parse is skipped there
    rather than guessed at, so keep it a number.

    Tier 2 executes binaries rather than replaying traces, so it changes the
    workload as well as the cost model.  It is reported as corroboration, never
    as the same measurement.
    """
    out = TierOutcome(tier=Tier.GEM5, built=True)
    refs = {}
    for depth in depths:
        for wl in workloads:
            key = f"{depth}/{wl}"
            refs[key] = gem5_node.run_gem5.chia_remote(
                gem5_bin,
                config_script=config_script,
                outdir=f"{outdir_root}/{variant.variant_id}/d{depth}_{wl}",
                workload_name=wl,
                config_args=[
                    "--binary", f"{workload_dir}/{wl}",
                    "--depth", str(depth),
                    "--max-insts", str(max_insts),
                ],
                stats_keys={
                    "branch_mispredicts": [
                        "system.cpu.branchPred.condIncorrect",
                        "system.cpu.branchPred.mispredicted",
                    ],
                    "branch_lookups": [
                        "system.cpu.branchPred.condPredicted",
                        "system.cpu.branchPred.lookups",
                    ],
                },
                timeout_s=C.GEM5_RUN_TIMEOUT_S)
    results = {k: get(v) for k, v in refs.items()}

    per_config: dict[str, list[float]] = {}
    failures = []
    for key, r in results.items():
        if r.status != "ok" or not r.num_cycles or not r.sim_insts:
            failures.append(f"{key}: {getattr(r, 'status', 'no status')}")
            continue
        per_config.setdefault(key.split("/", 1)[0], []).append(
            r.sim_insts / r.num_cycles)

    if not per_config:
        out.run_diagnostics = (
            "no gem5 run produced a parseable stats.txt; "
            + "; ".join(failures[:4]))
        return out

    out.ran = True
    # Mean over workloads at each depth.  A depth where some workloads failed
    # is still reported, but the count is in the diagnostics: an IPC averaged
    # over a different set of programs at each depth is not a depth sweep.
    out.per_config_ipc = {k: statistics.fmean(v) for k, v in per_config.items()}
    out.ipc = statistics.fmean(out.per_config_ipc.values())
    if failures:
        out.run_diagnostics = (
            f"{len(failures)}/{len(results)} gem5 runs failed: "
            + "; ".join(failures[:4]))
    return out


# ---------------------------------------------------------------------------
# promotion gates
# ---------------------------------------------------------------------------

def promote_to_tier1(evaluations: list[Evaluation], archive,
                     max_promoted: int = C.TIER1_MAX_PROMOTED) -> list[Evaluation]:
    """The Pareto front, capped.

    Front rather than top-N on VFS, for the reason :meth:`Archive.pareto_front`
    gives: the designs most likely to be misranked by a fixed-depth score are
    exactly the ones that trade latency for accuracy, and those are never the
    VFS winner at nine stages.
    """
    scored = [e for e in evaluations
              if e.tier0 and e.tier0.metrics is not None]
    if not scored:
        return []
    front_ids = {e.variant_id for e in archive.pareto_front()}
    front = [e for e in scored if e.variant.variant_id in front_ids]
    if not front:
        front = sorted(scored, key=lambda e: -e.tier0.metrics.vfs)[:1]
    return sorted(front, key=lambda e: -e.tier0.metrics.vfs)[:max_promoted]


def promote_to_tier2(evaluations: list[Evaluation],
                     max_promoted: int = C.TIER2_MAX_PROMOTED) -> list[Evaluation]:
    """Only designs whose port was verified reach gem5.

    A gem5 sweep is the most expensive thing the loop does, and a mistranslated
    predictor would look exactly like the cost-model disagreement the sweep is
    for.  So :func:`check_port_fidelity` must have passed first -- promotion is
    not a ranking question here, it is a trust question.
    """
    eligible = [e for e in evaluations
                if e.failure is not Failure.PORT_MISMATCH
                and e.tiers.get(Tier.CHAMPSIM)
                and e.tiers[Tier.CHAMPSIM].ran]
    return sorted(eligible,
                  key=lambda e: -(e.tier0.metrics.vfs if e.tier0 and e.tier0.metrics else 0)
                  )[:max_promoted]


def port_selfcheck_mpki(template_args: str, traces: list[str],
                        *, warmup: int = C.WARMUP_INSTRUCTIONS,
                        sim: int = C.SIM_INSTRUCTIONS,
                        jobs: int = C.SELFCHECK_JOBS,
                        timeout_s: int = C.SELFCHECK_TIMEOUT_S,
                        ) -> tuple[float | None, str]:
    """Conditional MPKI of the *ported algorithm*, with no simulator involved.

    ``tools/port_selfcheck.py`` renders ports/tage_core.h.in at these template
    arguments and drives it over the CBP-NG traces exactly as the ChampSim
    adapter does -- predict every branch, train the conditional ones, advance
    the history on all of them -- for the same instruction window Tier 0
    measures.  ``traces`` must be the set Tier 0 scored on, and the number that
    comes back is the unweighted mean of the per-trace rates, because that is
    what :func:`vfs.aggregate` computes.

    This exists because the obvious way to check a port (compare Tier-1 MPKI to
    Tier-0 MPKI) compares two harnesses as well as two predictors, and the
    harnesses turn out to dominate.  Here the trace, the window and the warmup
    are all Tier 0's, so what is left is the algorithm.  Measured across four
    designs spanning 6.2 to 14.4 Tier-0 MPKI, the residual is 2-6% and tight
    (ratios 0.945 to 0.979) -- that is the per-branch-versus-per-block history
    advance, and it is genuinely a constant.

    Runs on the driver, not a worker: it needs a C++ compiler, zlib and the
    traces, which is why ``--host-trace-dir`` exists.  Returns ``(None, why)``
    when it cannot run; a fidelity check that cannot run must not pass.
    """
    tool = C.TOOLS_DIR / "port_selfcheck.py"
    if not tool.exists():
        return None, f"{tool} is missing"
    if not template_args:
        return None, "no template arguments; this is not an offline-arm design"
    if not traces:
        return None, "no traces given"
    missing = [t for t in traces if not pathlib.Path(t).exists()]
    if missing:
        return None, (f"{len(missing)}/{len(traces)} traces are not on the "
                      f"driver, starting with {missing[0]}")
    # --sim matters as soon as the traces are longer than the window. Tier 0
    # stops after ``warmup + sim`` instructions; without passing the same cap
    # here the self-check would read each trace to its end and compare a
    # 150M-instruction measurement against Tier 0's 41M one. Same window or the
    # gate is measuring trace length.
    # --jobs because this runs over Tier 0's whole trace set, and that set is
    # now the 126 inner traces at full length. The self-check itself is cheap --
    # 6.7 MIPS per core, so ~10 core-minutes for the set -- but it sits between
    # a generation and the next one, and the per-trace runs share nothing but
    # the compile, so there is no reason to pay for it serially.
    cmd = [sys.executable, str(tool), "--trace", *traces,
           "--params", template_args, "--warmup", str(warmup),
           "--sim", str(sim), "--jobs", str(jobs)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, f"port self-check timed out after {timeout_s}s"
    if res.returncode != 0:
        return None, f"port self-check failed: {res.stderr.strip()[-400:]}"
    for line in res.stdout.splitlines():
        key, _, value = line.partition(",")
        if key == "conditional_mpki":
            return float(value), (f"port self-check over {len(traces)} trace"
                                  f"{'' if len(traces) == 1 else 's'}")
    return None, "port self-check printed no conditional_mpki"


def check_port_fidelity(evaluation: Evaluation,
                        tolerance: float = C.MPKI_AGREEMENT_TOLERANCE,
                        baseline_ratio: float | None = None,
                        port_mpki: float | None = None,
                        port_mpki_note: str = "") -> tuple[bool, str]:
    """Do Tier 0 and Tier 1 agree on how often the design mispredicts?

    This is the check that makes every later cross-tier claim mean anything.
    Tiers 0 and 1 are driven from the same branch stream -- the ChampSim trace
    is converted from the CBP-NG one by ``tools/cbp2champsim.cpp``, and the
    conversion is checked against the instruction and branch totals cbp-ng
    reports for itself -- so a gap in MPKI is about the predictor, not about
    which program was run.

    What a gap is *not* necessarily about is the translation.  CBP-NG predicts
    a whole cache line per cycle and advances its global history once per
    prediction block; ChampSim asks once per branch and offers no block hook.
    Two correct ports of the same predictor, fed histories that advance at
    different rates, compute different indices.

    There are two ways to measure that, and they do not agree.

    ``port_mpki`` is the good one: :func:`port_selfcheck_mpki` runs the ported
    algorithm over Tier 0's own traces and window, so the only difference left
    is the interface.  It comes out at 2-6% and one-sided, which is why this
    path is held to ``C.PORT_SELFCHECK_TOLERANCE`` rather than the tighter
    ``tolerance`` the Tier-1 comparison uses.

    ``baseline_ratio`` is the legacy one, kept only for ports that have no
    template arguments to re-render: compare Tier-1 MPKI to Tier-0 MPKI and
    calibrate away the difference.  Do not trust it.  Tier 1 and Tier 0 do not
    measure the same instructions -- ChampSim reopens a trace it runs off the
    end of -- and the resulting ratio moved between 0.48 and 0.82 across three
    designs in one sweep.  A calibration constant that depends on the design
    being calibrated is not a calibration constant.  C.TIER1_SIM_INSTRUCTIONS
    now sizes the window to the trace, which should bring this path back into
    agreement with the self-check; until a sweep confirms that, prefer
    ``port_mpki``.
    """
    t0, t1 = evaluation.tier0, evaluation.tiers.get(Tier.CHAMPSIM)
    if not (t0 and t0.mpki is not None):
        return False, "Tier-0 MPKI is missing"
    if t0.mpki <= 0:
        return False, f"Tier-0 MPKI is {t0.mpki}, so no ratio is meaningful"

    if port_mpki is not None:
        ratio = port_mpki / t0.mpki
        rel = abs(ratio - 1.0)
        if rel > C.PORT_SELFCHECK_TOLERANCE:
            return False, (
                f"Tier-0 MPKI {t0.mpki:.3f} vs ported algorithm "
                f"{port_mpki:.3f} on the same traces and window "
                f"({port_mpki_note}; ratio {ratio:.3f}, {rel:.1%} off, "
                f"tolerance {C.PORT_SELFCHECK_TOLERANCE:.0%}). No simulator "
                f"is involved in "
                f"either number, so this is the translation. Check the "
                f"provider and alternate selection, whether the global history "
                f"advances on every branch, and the allocation policy.")
        return True, (f"ported algorithm reproduces Tier-0 MPKI to "
                      f"{rel:.1%} ({port_mpki:.3f} vs {t0.mpki:.3f}, "
                      f"{port_mpki_note})")

    if not (t1 and t1.mpki is not None):
        return False, "MPKI missing from one of the two tiers"

    ratio = t1.mpki / t0.mpki
    target = baseline_ratio if baseline_ratio else 1.0
    rel = abs(ratio - target) / target
    against = (f"the calibrated port ratio {target:.3f}" if baseline_ratio
               else "Tier 0 outright")

    if rel > tolerance:
        return False, (
            f"Tier-0 MPKI {t0.mpki:.3f} vs Tier-1 conditional MPKI "
            f"{t1.mpki:.3f} (ratio {ratio:.3f}, {rel:.1%} from {against}, "
            f"tolerance {tolerance:.0%}). Same predictor, same branch stream "
            f"-- this is the ChampSim translation, not the cost model. Check "
            f"the block-based vs per-branch call sites, whether the global "
            f"history is advanced on every branch, and whether the second "
            f"prediction level is being applied at all.")
    return True, (f"Tier-1/Tier-0 MPKI ratio {ratio:.3f}, within "
                  f"{rel:.1%} of {against}")


# ---------------------------------------------------------------------------
# result_mapper_fn
# ---------------------------------------------------------------------------

def result_mapper_fn(evaluation: Evaluation, archive, parent: Elite | None = None,
                     *, port_baseline_ratio: float | None = None,
                     port_mpki: float | None = None,
                     port_mpki_note: str = "") -> Evaluation:
    """Fold the tiers into one fitness, a rank comparison and a failure profile.

    Fitness is VFS at the shipped depth, and only that.  It is tempting to fold
    the depth sweep in -- to score a design by how well it holds up at sixteen
    stages -- but that would answer the loop's own question in the fitness
    function.  The search optimises the score CBP-NG actually ships; whether
    that score was worth optimising is the *finding*, measured separately by
    the rank correlations recorded here.
    """
    t0 = evaluation.tier0

    if t0 is None or (not t0.built and t0.build_diagnostics):
        lint = harcom_lint.prelint(evaluation.variant.harcom_source,
                                   struct_name=evaluation.variant.struct_name)
        if lint.withdrawn:
            evaluation.failure = Failure.WITHDRAWN
            evaluation.feedback = (
                "This design was withdrawn, not rejected. It reaches around "
                "HARCOM's cost model, so any score it produced would not "
                "describe hardware.\n\n" + lint.render())
            return evaluation
        evaluation.failure = Failure.LINT if not lint.ok else Failure.BUILD
        evaluation.feedback = (t0.build_diagnostics if t0 else "no build was attempted")
        return evaluation

    if not t0.ran:
        evaluation.failure = Failure.RUN
        evaluation.feedback = t0.run_diagnostics or "the Tier-0 run produced no counters"
        return evaluation

    m = t0.metrics
    evaluation.fitness = m.vfs

    # -- cross-tier ranking -------------------------------------------------
    t1 = evaluation.tiers.get(Tier.CHAMPSIM)
    t2 = evaluation.tiers.get(Tier.GEM5)
    notes = []

    if t1 and t1.ran:
        ok, why = check_port_fidelity(
            evaluation, baseline_ratio=port_baseline_ratio,
            port_mpki=port_mpki, port_mpki_note=port_mpki_note)
        if not ok:
            evaluation.failure = Failure.PORT_MISMATCH
            evaluation.feedback = why
            return evaluation
        notes.append(f"Tier-1 port verified: {why}.")
        if t1.rob_occupancy_at_mispredict is not None:
            notes.append(
                f"ChampSim reports {t1.rob_occupancy_at_mispredict:.1f} "
                f"instructions in the ROB at a mispredict; CBP-NG charges a "
                f"flat {C.SHIPPED_DEPTH}-stage penalty regardless.")

    if t2 and t2.ran and len(t2.per_config_ipc) >= 2:
        # Correlate VFS-at-depth against measured IPC-at-depth, over the depths
        # both tiers have.  Config labels must parse as integers for this to
        # mean anything; a label that does not is skipped rather than guessed.
        paired = []
        for label, ipc in t2.per_config_ipc.items():
            digits = "".join(ch for ch in label if ch.isdigit())
            if not digits:
                continue
            depth = int(digits)
            if depth in t0.depth_scores:
                paired.append((t0.depth_scores[depth].vfs, ipc))
        if len(paired) >= 2:
            evaluation.tau_vfs_vs_ipc = kendall_tau(
                [p[0] for p in paired], [p[1] for p in paired])
            notes.append(
                f"VFS vs gem5 IPC across {len(paired)} depths: "
                f"tau={evaluation.tau_vfs_vs_ipc:+.2f}.")
            if evaluation.tau_vfs_vs_ipc < 0.5:
                evaluation.failure = Failure.RANK_INVERSION

    # -- archive ------------------------------------------------------------
    elite = evaluation.to_elite()
    accepted, cell = archive.add(elite)
    evaluation.archived = accepted
    evaluation.cell = cell

    if not accepted and evaluation.failure is Failure.NONE:
        evaluation.failure = Failure.NO_IMPROVEMENT

    evaluation.feedback = _feedback(evaluation, archive, parent, notes)
    return evaluation


def _feedback(evaluation: Evaluation, archive, parent: Elite | None,
              notes: list[str]) -> str:
    """The failure profile: what the design did, and where it has room.

    Written for the agent, so it names the trade rather than the number.  It
    deliberately does not suggest an edit -- proposing the next design is the
    agent's job, and a mapper that proposed one would be an LLM in the
    framework, which is the thing this module does not do.
    """
    t0 = evaluation.tier0
    m = t0.metrics
    lines = [
        f"VFS {m.vfs:.4f} at the shipped depth of {m.depth} stages, over "
        f"{m.n_traces} traces.",
        f"  energy/instruction {m.epi:.1f}, prediction latencies P1={m.p1_latency} "
        f"P2={m.p2_latency} cycles, MPKI {m.mpki:.3f}, throughput {m.ipc:.3f} "
        f"predicted instructions/cycle.",
    ]

    if parent is not None:
        d = m.vfs - parent.vfs
        lines.append(
            f"  against parent {parent.variant_id}: VFS {d:+.4f}, "
            f"energy {m.epi - parent.epi:+.1f}, MPKI {m.mpki - parent.mpki:+.3f}.")

    if t0.depth_scores:
        depths = sorted(t0.depth_scores)
        lines.append("  same counters re-scored at other depths: " + ", ".join(
            f"{d}:{t0.depth_scores[d].vfs:.3f}" for d in depths))

    cell = evaluation.cell
    if evaluation.archived:
        lines.append(
            f"  ARCHIVED in cell {cell} -- best design in that region so far. "
            f"Archive holds {len(archive)} cells ({archive.coverage():.0%} "
            f"of the grid).")
    else:
        occupant = archive.cells.get(cell)
        lines.append(
            f"  NOT ARCHIVED: cell {cell} is already held by "
            f"{occupant.variant_id} at VFS {occupant.vfs:.4f}."
            if occupant else f"  NOT ARCHIVED (cell {cell}).")
        lines.append(
            "  To take a cell, either beat that design on VFS at the same "
            "energy and latency, or move to an empty region -- a different "
            "energy band or a different P1/P2 latency pair is worth as much "
            "to the archive as a higher score.")

    if evaluation.failure is Failure.RANK_INVERSION:
        lines.append(
            "  WARNING: this design's VFS ranking and its measured gem5 IPC "
            "ranking disagree across the depth sweep. That is a finding, not a "
            "defect -- it is recorded, and the design is kept.")

    lines.extend("  " + n for n in notes)
    return "\n".join(lines)
