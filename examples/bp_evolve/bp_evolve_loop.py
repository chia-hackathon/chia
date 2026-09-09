#!/usr/bin/env python3
"""Driver: evolve HARCOM branch predictors, then re-rank them on two more
complete models of the core.

    chia job submit -- python examples/bp_evolve/bp_evolve_loop.py --selftest
    chia job submit -- python examples/bp_evolve/bp_evolve_loop.py --arm offline --generations 3
    chia job submit -- python examples/bp_evolve/bp_evolve_loop.py

The loop is the one in the proposal's figure.  A parent is drawn from the elite
archive; the agent edits its HARCOM source behind a lint gate; every variant is
scored on CBP-NG; the Pareto front is promoted to ChampSim; only the elites
reach gem5; and the mapper compares the three rankings and returns a failure
profile the agent can act on.

Two things about the shape are worth stating, because they are choices rather
than conveniences.

**Fitness is VFS at the shipped depth, and nothing else.** Folding the depth
sweep into fitness would answer the loop's own question inside the fitness
function.  The search optimises the score CBP-NG actually ships; whether that
score was worth optimising is measured separately, from the depth table the
same counters give for free.

**Held-out traces never enter the inner loop.** The subset the agent's feedback
is computed from and the set a final claim is scored on are disjoint, chosen
once from a seeded shuffle.  A design that wins on the traces its own feedback
came from has not been shown to generalise, and that is the difference between
a result and an overfit.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chia.base.ChiaFunction import get

import agents
import constants as C
from archive import Archive, Elite
from cbp_ng import CbpNgNode, find_traces
from db import LineageDB
from evaluator import (
    Evaluation, Failure, Tier, Variant,
    build_fn, port_selfcheck_mpki, promote_to_tier1, promote_to_tier2,
    result_mapper_fn, run_fn_tier0, run_fn_tier1, run_fn_tier2,
)
import gem5_bp
from vfs import kendall_tau


# ---------------------------------------------------------------------------
# Trace split
# ---------------------------------------------------------------------------

def split_traces(all_traces: list[str], *, inner: int, held_out_fraction: float,
                 seed: int) -> tuple[list[str], list[str], list[str]]:
    """(inner, full-pass, held-out), disjoint by construction.

    The shuffle is seeded so the split is a property of the trace *set*, not of
    when the sweep happened: two sweeps over the same traces compare, and a
    sweep that is resumed does not silently move a held-out trace into the
    inner loop.
    """
    traces = list(all_traces)
    if len(traces) < 2:
        raise ValueError(
            f"a held-out split needs at least 2 traces; found {len(traces)}. "
            f"Point --trace-dir at the CBP-NG trace set.")
    random.Random(seed).shuffle(traces)
    # Hold out a quarter, but never so many that the inner loop is left empty:
    # a sweep with no inner traces would run, score nothing, and report an
    # empty archive as though the search had failed.
    n_held = min(max(1, int(len(traces) * held_out_fraction)), len(traces) - 1)
    held_out = traces[:n_held]
    rest = traces[n_held:]
    return rest[:inner], rest, held_out


# ---------------------------------------------------------------------------
# One variant through the cascade
# ---------------------------------------------------------------------------

def _evaluate_core(variant: Variant, *, cbp_node, traces, archive,
                   parent: Elite | None, llm=None,
                   cbp_root: str = C.CBP_NG_ROOT,
                   max_repair_rounds: int = C.MAX_REPAIR_ROUNDS,
                   ) -> tuple[Evaluation, Variant, int, bool]:
    """Lint, build, repair and run Tier 0.  Returns (ev, variant, rounds, scored).

    This is the expensive half of evaluating a variant and it is deliberately
    free of archive mutation, which is what makes it safe to run several at
    once.  ``result_mapper_fn`` is called here only on a build failure, and on
    that path it returns at the lint/build branch -- well before the
    ``archive.add`` at the end of it.  ``scored`` says whether the evaluation
    that comes back is already final (a withdrawal or an unrepairable build) or
    still needs scoring against the archive by the caller.

    A withdrawn design is never repaired: reaching around HARCOM is not a
    mistake to fix, and offering a round would say otherwise.
    """
    repair_rounds = 0

    while True:
        outcomes, artifacts = build_fn(
            variant, [Tier.CBP_NG], cbp_node=cbp_node, cbp_root=cbp_root)
        ev = Evaluation(variant=variant, tiers=outcomes)

        if Tier.CBP_NG in artifacts:
            break

        ev = result_mapper_fn(ev, archive, parent)
        if ev.failure is Failure.WITHDRAWN:
            return ev, variant, repair_rounds, True
        if llm is None or repair_rounds >= max_repair_rounds:
            return ev, variant, repair_rounds, True

        repair_rounds += 1
        try:
            fixed = agents.repair(
                llm, source=variant.harcom_source,
                struct_name=variant.struct_name,
                diagnostics=ev.feedback, round_no=repair_rounds)
        except agents.AgentError as e:
            ev.feedback = f"{ev.feedback}\n\n(repair abandoned: {e})"
            return ev, variant, repair_rounds, True
        variant = Variant(
            variant_id=variant.variant_id, struct_name=fixed["struct_name"],
            harcom_source=fixed["source"], parent_id=variant.parent_id,
            generation=variant.generation,
            rationale=f"{variant.rationale} | repair {repair_rounds}: "
                      f"{fixed['rationale']}")

    ev.tiers[Tier.CBP_NG] = run_fn_tier0(
        variant, artifacts[Tier.CBP_NG], traces, cbp_node=cbp_node)
    ev.variant = variant
    return ev, variant, repair_rounds, False


def evaluate_variant(variant: Variant, *, cbp_node, traces, archive,
                     parent: Elite | None, llm=None,
                     cbp_root: str = C.CBP_NG_ROOT,
                     max_repair_rounds: int = C.MAX_REPAIR_ROUNDS,
                     ) -> tuple[Evaluation, int]:
    """Build, run and score one variant.  The sequential path.

    Kept for the seed, the self-test and --score-held-out, all of which
    evaluate exactly one design and want the archive updated on return.
    """
    ev, _variant, rounds, scored = _evaluate_core(
        variant, cbp_node=cbp_node, traces=traces, archive=archive,
        parent=parent, llm=llm, cbp_root=cbp_root,
        max_repair_rounds=max_repair_rounds)
    if scored:
        return ev, rounds
    return result_mapper_fn(ev, archive, parent), rounds


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def init_local_ray(cbp_slots: int) -> None:
    """Start a single-process Ray with the custom resources the nodes ask for.

    For the smoke test only.  ``CbpNgNode`` tasks request ``{"cbp_ng": 1.0}``,
    and a bare ``ray.init()`` declares no such resource -- the tasks then queue
    forever against an autoscaler that cannot satisfy them, which looks like a
    hang rather than a misconfiguration.  A real sweep gets the resource from
    ``cluster.yaml``; this exists so ``--arm offline --local`` runs on a laptop
    without one.
    """
    import ray
    if ray.is_initialized():
        return
    ray.init(resources={"cbp_ng": float(cbp_slots)},
             ignore_reinit_error=True,
             logging_level="ERROR")



def init_cluster_ray() -> None:
    """Attach to the cluster ``chia up`` started, and ship this example to it.

    ``address="auto"`` rather than a default ``ray.init()``: the latter starts a
    brand-new single-process Ray that declares none of the tier resources, so
    every task queues forever against a cluster that is running fine next to it.

    The runtime env is what makes the worker able to import this example at
    all. The tier functions live in ``cbp_ng.py`` and ``evaluator.py``, which
    exist only in this directory; without shipping it, a Tier-0 task fails on
    the worker with ``ModuleNotFoundError: constants``.
    """
    import ray
    if ray.is_initialized():
        return
    ray.init(address="auto", runtime_env=C.RUNTIME_ENV)

def resolve_traces(worker_dir: str, host_dir: str | None) -> list[str]:
    """List traces here, hand the workers their own path to the same files.

    Under docker the same directory has two names -- a bind mount is
    ``~/bp_evolve_data/cbp_traces`` outside the container and
    ``/home/ray/cbp-ng/traces`` inside it -- and the loop needs both: it
    enumerates the set on the driver and simulates it on the worker. Passing
    one path for both jobs fails with "no traces under ..." for a directory
    that is full of traces on the machine that matters.

    With no ``host_dir`` the two are the same, which is the case when the
    driver runs inside the same filesystem as the workers.
    """
    if not host_dir:
        return find_traces(worker_dir)
    return [os.path.join(worker_dir, os.path.basename(t))
            for t in find_traces(host_dir)]


def run_sweep(args) -> int:
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.local:
        init_local_ray(args.local_cbp_slots)
    else:
        init_cluster_ray()
    rng = random.Random(args.seed)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Two different paths to the same checkout. The seed is read here, on the
    # driver; --cbp-root is where the *workers* find cbp-ng, and under docker
    # those differ (a bind mount lands at /home/ray/cbp-ng inside the container
    # and somewhere else outside it). Conflating them fails at the first line
    # of the sweep with a missing seed, which is a confusing way to learn that
    # a path was for the other machine.
    seed_root = args.seed_root or args.cbp_root
    seed_path = os.path.join(seed_root, C.SEED_SOURCE)
    if not os.path.exists(seed_path):
        print(f"seed predictor not found: {seed_path}", file=sys.stderr)
        return 2
    seed_source = open(seed_path).read()

    all_traces = resolve_traces(args.trace_dir, args.host_trace_dir)
    if not all_traces:
        print(f"no traces under {args.trace_dir}", file=sys.stderr)
        return 2
    inner, full_pass, held_out = split_traces(
        all_traces, inner=args.inner_traces,
        held_out_fraction=C.HELD_OUT_FRACTION, seed=C.TRACE_SUBSET_SEED)
    # `full_pass` is the whole non-held-out pool -- the set an elite would be
    # re-scored on if elites earned a full pass, which today they do not (see
    # README, "Not yet run"). At --inner-traces 126 it is the same list as
    # `inner`, so the line says so rather than reporting two counts that look
    # like two different trace sets.
    extra = (" (the inner set is the whole pool)" if len(inner) == len(full_pass)
             else f", {len(full_pass)} in the full-pass pool")
    print(f"{len(all_traces)} traces: {len(inner)} inner, "
          f"{len(held_out)} held out{extra}")

    # Resuming reloads the archive and nothing else.  The archive IS the search
    # state of MAP-Elites -- parents are drawn from it and from nothing else --
    # so a reloaded one continues the same search.  What does not come back is
    # the per-generation scaffolding: `last_feedback`, `offline_args` and the
    # promotion bookkeeping all restart empty, so the first resumed generation
    # mutates its parents from template defaults rather than from where the
    # previous generation had walked them to.  That is a real discontinuity in
    # the trajectory, not a cosmetic one, and it is why the resumed run reports
    # its provenance in the sweep notes rather than pretending to be one run.
    if args.resume_archive:
        archive = Archive.from_json(open(args.resume_archive).read())
        resume_from = max((e.generation for e in archive.elites), default=0) + 1
        print(f"resuming from {args.resume_archive}: {len(archive)} cells, "
              f"best {archive.best().variant_id} VFS {archive.best().vfs:.4f}; "
              f"continuing at generation {resume_from}")
    else:
        archive = Archive(C.EPI_BIN_EDGES, C.P1_LATENCY_BINS, C.P2_LATENCY_BINS)
        resume_from = 1
    llm = None if args.arm == "offline" else agents.make_llm("bp_evolve_design")

    with CbpNgNode(require_colocated=False) as cbp_node, \
            LineageDB(str(C.OUT_DIR / "bp_evolve.db")) as db:

        db.start_sweep(started, args.arm, C.SEED_PREDICTOR, args.generations,
                       len(inner), C.DEPTH_SWEEP, notes=args.notes)

        # -- generation 0: the seed itself ---------------------------------
        #
        # Skipped on resume: the seed's own cell came back with the archive, and
        # re-running it would spend a full generation of Tier 0 to re-derive a
        # number we already hold.
        #
        # Scored under exactly the same path as every variant.  A baseline
        # measured a different way is not a baseline; it is a second
        # experiment.
        seed_variant = Variant(
            variant_id="gen000_seed",
            struct_name=f"seed_{C.SEED_PREDICTOR}",
            harcom_source=agents.rename_struct(
                seed_source, C.SEED_PREDICTOR, f"seed_{C.SEED_PREDICTOR}"),
            generation=0,
            rationale=f"unmodified {C.SEED_SOURCE}")
        if not args.resume_archive:
            ev, rounds = evaluate_variant(
                seed_variant, cbp_node=cbp_node, traces=inner, archive=archive,
                parent=None, llm=None, cbp_root=args.cbp_root)
            db.record_evaluation(ev, rounds)
            if ev.fitness is None:
                print("the seed predictor did not score; nothing downstream can "
                      "mean anything until it does:", file=sys.stderr)
                print(ev.feedback, file=sys.stderr)
                return 3
            print(f"[gen 0] seed VFS {ev.fitness:.4f}  {ev.feedback.splitlines()[1].strip()}")
            db.snapshot_archive(0, archive)

        last_feedback: dict[str, str] = {}
        offline_args: dict[str, dict] = {}
        # Survives across generations: everything the port-fidelity gate needs.
        # `host_trace_by_name` is how the self-check gets from a counter line
        # (which names the worker's path) to a file it can open on the driver.
        promo_state: dict = {
            "host_trace_by_name": {
                _trace_key(t): t
                for t in find_traces(args.host_trace_dir or args.trace_dir)},
        }

        # -- generations ---------------------------------------------------
        for gen in range(resume_from, args.generations + 1):
            evaluations = []

            # A generation runs in three phases, and the split is what lets the
            # variants overlap.
            #
            #   propose   sequential -- draws from `rng`, so order fixes the run
            #   build+run concurrent -- the expensive part, and archive-free
            #   score     sequential -- archive.add() in proposal order
            #
            # The middle phase is safe to overlap because _evaluate_core never
            # reaches archive.add: it touches the archive only through
            # result_mapper_fn on a build failure, which returns at the
            # lint/build branch.  Scoring stays on this thread and in proposal
            # order, so which cells fill and which elites survive does not
            # depend on who finished first -- the same seed still gives the same
            # archive.
            #
            # The cost is that all K variants of a generation are proposed from
            # the archive as it stood at the start of it, so they cannot build
            # on each other within the generation.  That is ordinary batched
            # MAP-Elites.
            #
            # What it buys depends on how the trace count compares to the slot
            # count, and it is worth being exact about that rather than saying
            # "K times faster".  At len(inner) < cbp_ng slots the old loop left
            # most of the machine idle and this is close to a K-fold cut.  At
            # len(inner) = 126 on 16 slots one variant already saturates them,
            # and what overlapping removes is the K-1 stragglers: a variant is
            # not done until its longest trace is, and that trace is 130M
            # instructions -- around ten minutes during which fifteen slots had
            # nothing to do.  Concurrent variants fill them, and the builds
            # (~8 s each) stop being serialised behind the scoring as well.
            proposals = []
            for k in range(args.variants_per_generation):
                parent = archive.select_parent(rng)
                if parent is None:
                    break
                vid = f"gen{gen:03d}_{k}"
                try:
                    proposal = _propose(
                        args.arm, llm, parent, archive, rng,
                        feedback=last_feedback.get(parent.variant_id, ""),
                        generation=gen, vid=vid,
                        offline_args=offline_args.get(parent.variant_id))
                except agents.AgentError as e:
                    print(f"[gen {gen}] {vid}: agent failed: {e}")
                    continue

                variant = Variant(
                    variant_id=vid, struct_name=proposal["struct_name"],
                    harcom_source=proposal["source"], parent_id=parent.variant_id,
                    generation=gen, rationale=proposal["rationale"],
                    template_args=proposal.get("template_args", ""))
                if args.arm == "offline":
                    offline_args[vid] = proposal["args"]
                proposals.append((vid, variant, parent))

            cores: list = []
            if proposals:
                with futures.ThreadPoolExecutor(
                        max_workers=len(proposals),
                        thread_name_prefix=f"gen{gen:03d}") as pool:
                    futs = [
                        pool.submit(
                            _evaluate_core, variant, cbp_node=cbp_node,
                            traces=inner, archive=archive, parent=parent,
                            llm=llm, cbp_root=args.cbp_root)
                        for _vid, variant, parent in proposals]
                    for (vid, _v, _p), fut in zip(proposals, futs):
                        try:
                            cores.append(fut.result())
                        except Exception as e:   # one variant must not sink the
                            # generation; the others already ran.
                            print(f"[gen {gen}] {vid}: evaluation raised: "
                                  f"{type(e).__name__}: {e}")
                            cores.append(None)

            for (vid, _variant, parent), core in zip(proposals, cores):
                if core is None:
                    continue
                ev, _v, rounds, scored = core
                if not scored:
                    ev = result_mapper_fn(ev, archive, parent)
                db.record_evaluation(ev, rounds)
                evaluations.append(ev)
                last_feedback[vid] = ev.feedback

                flag = ("ARCHIVED" if ev.archived else ev.failure.value)
                score = f"{ev.fitness:.4f}" if ev.fitness is not None else "  --  "
                print(f"[gen {gen}] {vid}: VFS {score}  {flag}"
                      + (f"  (repairs: {rounds})" if rounds else ""))

            db.snapshot_archive(gen, archive)
            best = archive.best()
            print(f"[gen {gen}] archive: {len(archive)} cells "
                  f"({archive.coverage():.0%}), best {best.variant_id} "
                  f"VFS {best.vfs:.4f}")

            if args.tier1 and evaluations:
                _promote(evaluations, archive, db, gen, llm, args, promo_state)

        # -- the headline result -------------------------------------------
        _report(archive, db, held_out, args)

    return 0


def _propose(arm, llm, parent, archive, rng, *, feedback, generation, vid,
             offline_args):
    """One proposal from whichever arm is active."""
    if arm == "offline":
        p = agents.offline_design(parent.source, parent.struct_name, rng,
                                  parent_args=offline_args
                                  or _parse_template_args(parent.template_args))
        p["source"] = agents.rename_struct(parent.source, parent.struct_name, vid)
        p["struct_name"] = vid
        return p
    return agents.design(
        llm, parent_source=parent.source,
        parent_summary=parent.summary(),
        archive_summary=_archive_summary(archive),
        feedback=feedback, generation=generation)


def _parse_template_args(text: str) -> dict | None:
    """Turn ``"6,8,11,12,11,100,14,6"`` back into the offline arm's parameter dict.

    The archive stores the rendered argument list, not the dict, because that
    is what the build actually used; reconstructing it here keeps one
    representation authoritative instead of two that can drift.
    """
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != len(agents._TAGE_ORDER):
        return None
    try:
        return {k: int(v) for k, v in zip(agents._TAGE_ORDER, parts)}
    except ValueError:
        return None


def _archive_summary(archive: Archive, limit: int = 12) -> str:
    """What the archive looks like, for the design prompt.

    Cells and their costs, not sources: the agent is choosing a region to aim
    at, and a dozen full predictor headers would crowd out the design it is
    supposed to be writing.
    """
    if not len(archive):
        return "(empty)"
    lines = [f"{len(archive)} cells occupied ({archive.coverage():.0%} of the grid)."]
    for e in archive.elites[:limit]:
        lines.append(
            f"  cell {archive.descriptor(e)}: {e.variant_id} VFS {e.vfs:.4f}, "
            f"EPI {e.epi:.0f}, P1 {e.p1_latency}, P2 {e.p2_latency}, "
            f"MPKI {e.mpki:.3f}")
    if len(archive) > limit:
        lines.append(f"  ... and {len(archive) - limit} more")
    return "\n".join(lines)


def _promote(evaluations, archive, db, gen, llm, args, state):
    """Tier 1, then Tier 2 behind the port-fidelity gate.

    Kept in its own function because it is the part that needs a ChampSim pool
    and a gem5 pool: a sweep with ``--no-tier1`` is a complete Tier-0 search and
    is what runs when only CBP-NG is provisioned.

    ``state`` carries what the port-fidelity gate needs across generations:
    the host paths of the CBP-NG traces, one self-check result per distinct
    design (see :func:`_port_selfcheck`), and -- only for designs the
    self-check cannot handle -- the legacy Tier-1/Tier-0 ratio.
    """
    from chia.simulators.champsim import ChampSimNode

    front = promote_to_tier1(evaluations, archive)
    if not front:
        return
    print(f"[gen {gen}] promoting {len(front)} designs to Tier 1")

    champsim_traces = resolve_traces(args.champsim_trace_dir,
                                     args.host_champsim_trace_dir)
    if not champsim_traces:
        print(f"[gen {gen}] no ChampSim traces under {args.champsim_trace_dir}; "
              f"Tier 1 skipped. Convert them with tools/convert_traces.py.")
        return

    verified = []
    with ChampSimNode(require_colocated=True) as cs_node:
        for ev in front:
            try:
                ported = _port(llm, ev, "champsim")
            except (agents.AgentError, ValueError) as e:
                print(f"[gen {gen}] {ev.variant.variant_id}: port failed: {e}")
                continue
            ev.variant.champsim_source = ported["source"]

            outcomes, artifacts = build_fn(
                ev.variant, [Tier.CHAMPSIM], champsim_node=cs_node,
                champsim_root=args.champsim_root)
            ev.tiers[Tier.CHAMPSIM] = outcomes[Tier.CHAMPSIM]
            if Tier.CHAMPSIM not in artifacts:
                print(f"[gen {gen}] {ev.variant.variant_id}: Tier-1 build "
                      f"failed: {outcomes[Tier.CHAMPSIM].build_diagnostics[:200]}")
                db.record_evaluation(ev)
                continue

            ev.tiers[Tier.CHAMPSIM] = run_fn_tier1(
                ev.variant, artifacts[Tier.CHAMPSIM],
                champsim_traces[:args.tier1_traces], champsim_node=cs_node)
            port_mpki, port_note = _port_selfcheck(ev, state)
            if port_mpki is None:
                _port_baseline(ev, state)
            ev = result_mapper_fn(
                ev, archive, None,
                port_baseline_ratio=state.get("port_baseline_ratio"),
                port_mpki=port_mpki, port_mpki_note=port_note)
            db.record_evaluation(ev)
            ok = ev.failure is not Failure.PORT_MISMATCH
            if ok:
                verified.append(ev)
            print(f"[gen {gen}] {ev.variant.variant_id}: Tier 1 "
                  f"{'ok' if ok else 'PORT MISMATCH'} -- "
                  f"{ev.feedback.splitlines()[-1].strip()}")

    if not args.tier2:
        return
    elites = promote_to_tier2(verified)
    if not elites:
        print(f"[gen {gen}] nothing survived the port check; Tier 2 skipped")
        return
    _run_tier2(elites, archive, db, gen, llm, args, state)


def _port(llm, ev, target):
    """The port for one design, from whichever arm is active.

    The offline arm renders a template and can only port the TAGE family; the
    LLM arm can port anything and is checked by exactly the same gate.
    """
    if llm is None:
        return agents.offline_port(ev.variant.template_args, target=target)
    return agents.port(
        llm, harcom_source=ev.variant.harcom_source,
        struct_name=ev.variant.struct_name, target=target,
        tier0_mpki=ev.tier0.metrics.mpki)


def _trace_key(path: str) -> str:
    """The name a counter line and a trace file on the driver agree on.

    ``CbpNgNode.run_cbp`` labels each run with the trace basename minus its
    final extension, so ``web_114_trace.gz`` comes back from CBP-NG as
    ``web_114_trace``.  Both sides of the self-check's lookup go through here,
    because keying the map on one convention and reading it with the other is
    precisely the bug this replaces: the map held ``web_114_trace.gz``, every
    counter asked for ``web_114_trace``, no lookup ever hit, and the port gate
    fell back to the weak Tier-1/Tier-0 ratio on every promotion while
    reporting only that the driver had no traces.

    It strips a known compression suffix rather than calling ``splitext``,
    which makes it idempotent -- and it has to be, because one side is applied
    to a filename and the other to a label that has already been stripped.
    ``splitext`` would take ``gmsh-5.4132_0_trace`` down to ``gmsh-5``.
    """
    base = os.path.basename(path)
    for suffix in (".gz", ".xz", ".bz2", ".zst"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _tier0_host_traces(ev, state) -> list[str]:
    """The traces Tier 0 actually scored this design on, named as the driver
    sees them.

    Tier 0's MPKI is the unweighted mean over its trace set, so the self-check
    has to run over that same set or the two numbers are not the same kind of
    number.  ``ev.tier0.counters`` records it exactly -- one line per trace,
    in the order they were dispatched -- and it is preferred over the sweep's
    ``inner`` list because elites are re-scored on the full pass.

    Trace names are matched through :func:`_trace_key`: the counters carry the
    label CBP-NG was given (``web_114_trace``) and the self-check needs the
    driver's path (``~/bp_evolve_data/cbp_traces/web_114_trace.gz``).
    """
    by_name = state.get("host_trace_by_name") or {}
    counters = (ev.tier0.counters if ev.tier0 else None) or []
    paths = [by_name.get(_trace_key(c.name)) for c in counters]
    return [p for p in paths if p]


def _port_selfcheck(ev, state):
    """Conditional MPKI of the ported algorithm, no simulator, cached.

    This is the port-fidelity evidence that is actually about the port.  It
    renders the same ports/tage_core.h.in the Tier-1 and Tier-2 sources are
    built from and drives it over Tier 0's own traces and window, so the only
    thing left between it and Tier-0 MPKI is the interface -- CBP-NG advances
    its history once per prediction block, the port once per branch.  That gap
    measures 2-6%.

    Cached on the template arguments, which are the whole design: two variants
    that render the same core predict identically, and the check costs a
    compile.  Designs with no template arguments (the LLM arm writes HARCOM
    directly) get ``(None, why)``, and the caller falls back to the ratio.
    """
    key = getattr(ev.variant, "template_args", "")
    if not key:
        return None, "no template arguments; the self-check has nothing to render"
    cache = state.setdefault("port_selfcheck", {})
    if key not in cache:
        traces = _tier0_host_traces(ev, state)
        if not traces:
            cache[key] = (None, "no CBP-NG traces on the driver; pass "
                                "--host-trace-dir so the self-check can run")
        else:
            cache[key] = port_selfcheck_mpki(key, traces)
        mpki, note = cache[key]
        if mpki is None:
            print(f"      port self-check unavailable ({note}); falling back "
                  f"to the Tier-1/Tier-0 ratio, which is weaker")
        else:
            t0 = ev.tier0.metrics.mpki if ev.tier0 and ev.tier0.metrics else None
            print(f"      port self-check: {mpki:.3f} conditional MPKI"
                  + (f" against Tier 0's {t0:.3f} (ratio {mpki / t0:.3f})"
                     if t0 else "")
                  + f" -- {note}")
    return cache[key]


def _port_baseline(ev, state):
    """Calibrate what the *interface* costs, once, from the first port.

    Only reached when :func:`_port_selfcheck` cannot run.  Read its warning
    first: this ratio mixes the port with the two harnesses, and it moved
    between 0.48 and 0.82 across three designs in a single sweep.

    CBP-NG predicts a cache line per cycle and advances its history once per
    prediction block; ChampSim asks per branch. Two correct ports of the same
    predictor therefore disagree on MPKI by a roughly constant factor, and
    holding every port to a ratio of exactly 1.0 would reject correct
    translations for a difference that is the interface's, not theirs.

    The first port to run establishes the ratio, and it is the *offline* port
    whenever the offline arm is active -- that one is correct by construction,
    which is the only reason it can be used as a reference. Later ports are
    then measured against it, which is a test of the translation and nothing
    else.
    """
    if "port_baseline_ratio" in state:
        return
    t0, t1 = ev.tier0, ev.tiers.get(Tier.CHAMPSIM)
    if not (t0 and t0.mpki and t1 and t1.ran and t1.mpki):
        return
    ratio = t1.mpki / t0.mpki
    state["port_baseline_ratio"] = ratio
    print(f"      port calibration: Tier-1/Tier-0 conditional MPKI ratio "
          f"{ratio:.3f} ({t1.mpki:.3f} / {t0.mpki:.3f}) from "
          f"{ev.variant.variant_id}; later ports are checked against this, "
          f"not against 1.0")


def _run_tier2(elites, archive, db, gen, llm, args, state):
    """The gem5 depth sweep, on the designs whose port was verified.

    This is the most expensive thing the loop does and the only tier that can
    answer its question directly: same predictor, same program, four different
    distances from a prediction to its resolution.
    """
    from chia.simulators.gem5 import Gem5Node

    print(f"[gen {gen}] {len(elites)} designs to Tier 2 "
          f"(gem5 depths {', '.join(str(d) for d in C.GEM5_DEPTHS)})")
    with Gem5Node(require_colocated=True) as g5_node:
        for ev in elites:
            try:
                ported = _port(llm, ev, "gem5")
            except (agents.AgentError, ValueError) as e:
                print(f"[gen {gen}] {ev.variant.variant_id}: gem5 port failed: {e}")
                continue
            ev.variant.gem5_source = ported["source"]
            ev.variant.gem5_impl = ported.get("impl")

            outcomes, artifacts = build_fn(
                ev.variant, [Tier.GEM5], gem5_node=g5_node,
                gem5_root=args.gem5_root)
            ev.tiers[Tier.GEM5] = outcomes[Tier.GEM5]
            if Tier.GEM5 not in artifacts:
                print(f"[gen {gen}] {ev.variant.variant_id}: gem5 build failed: "
                      f"{outcomes[Tier.GEM5].build_diagnostics[:300]}")
                db.record_evaluation(ev)
                continue

            ev.tiers[Tier.GEM5] = run_fn_tier2(
                ev.variant, artifacts[Tier.GEM5],
                list(C.GEM5_WORKLOADS), C.GEM5_DEPTHS,
                gem5_node=g5_node,
                outdir_root=args.gem5_outdir,
                config_script=gem5_bp.gem5_config_path(args.gem5_root),
                workload_dir=args.gem5_workload_dir)
            # Cached from Tier 1 -- the mapper re-runs the port check on
            # every call, and it must see the same evidence both times.
            port_mpki, port_note = _port_selfcheck(ev, state)
            ev = result_mapper_fn(
                ev, archive, None,
                port_baseline_ratio=state.get("port_baseline_ratio"),
                port_mpki=port_mpki, port_mpki_note=port_note)
            db.record_evaluation(ev)
            t2 = ev.tiers[Tier.GEM5]
            if t2.ran:
                sweep_txt = "  ".join(
                    f"d{d}:{ipc:.3f}" for d, ipc in sorted(
                        t2.per_config_ipc.items(), key=lambda kv: int(kv[0])))
                print(f"[gen {gen}] {ev.variant.variant_id}: gem5 IPC  {sweep_txt}"
                      + (f"   tau(VFS,IPC)={ev.tau_vfs_vs_ipc:+.2f}"
                         if ev.tau_vfs_vs_ipc is not None else ""))
            else:
                print(f"[gen {gen}] {ev.variant.variant_id}: gem5 run produced "
                      f"nothing -- {t2.run_diagnostics[:200]}")


def _report(archive, db, held_out, args):
    """The result the proposal actually promises.

    Not "the evolved predictor wins" -- that may or may not happen.  The
    deliverable is the rank correlation between the shipped score and the same
    designs re-scored at other depths, over the front the search produced.  It
    is computed here from the archive, and it stands either way.
    """
    elites = archive.elites
    print("\n" + "=" * 72)
    print(f"archive: {len(elites)} elites, {archive.coverage():.0%} coverage, "
          f"{archive.rejected} variants rejected by an occupant")

    depths = sorted({d for e in elites for d in e.depth_scores})
    if len(elites) < 2 or len(depths) < 2:
        print("too few scored designs for a rank comparison")
        return

    base = [e.depth_scores[C.SHIPPED_DEPTH] for e in elites
            if C.SHIPPED_DEPTH in e.depth_scores]
    print(f"\nVFS of the evolved front, re-scored at each depth "
          f"({len(elites)} designs):")
    header = "  depth  " + "".join(f"{e.variant_id[-10:]:>12}" for e in elites[:6])
    print(header)
    for d in depths:
        row = "".join(f"{e.depth_scores.get(d, float('nan')):12.4f}"
                      for e in elites[:6])
        print(f"  {d:>5}  {row}")

    print(f"\nKendall tau against the shipped depth ({C.SHIPPED_DEPTH} stages):")
    for d in depths:
        vec = [e.depth_scores[d] for e in elites if d in e.depth_scores]
        if len(vec) == len(base) and len(vec) >= 2:
            print(f"  depth {d:>3}: tau = {kendall_tau(base, vec):+.3f}")

    out = C.OUT_DIR / "archive.json"
    out.write_text(archive.to_json())
    print(f"\narchive written to {out}")
    print(f"{len(held_out)} held-out traces were never used for feedback; "
          f"score the front on them with --score-held-out")


# ---------------------------------------------------------------------------
# Held-out scoring
# ---------------------------------------------------------------------------

def score_held_out(args) -> int:
    """Re-score an archive's front on traces no feedback ever came from.

    This is the check that separates a real improvement from an overfit, and it
    is a separate command rather than the tail of a sweep for a reason: the
    inner loop scores on 24 traces because 168 is too expensive to do per
    variant, so held-out scoring is a *full pass*, run once, over designs that
    have already earned it. Folding it into the sweep would either make every
    generation cost a full pass or quietly score the front on a subset, and
    the second is the failure mode this whole function exists to detect.

    Nothing here consults the archive's recorded VFS. Each elite is rebuilt
    from its own source and template arguments and re-run, so a discrepancy
    between the two numbers is visible rather than assumed away.
    """
    if args.local:
        init_local_ray(args.local_cbp_slots)
    else:
        init_cluster_ray()

    archive_path = args.archive or (C.OUT_DIR / "archive.json")
    if not os.path.exists(archive_path):
        print(f"no archive at {archive_path}; run a sweep first", file=sys.stderr)
        return 2
    archive = Archive.from_json(open(archive_path).read())

    all_traces = resolve_traces(args.trace_dir, args.host_trace_dir)
    _, _, held_out = split_traces(
        all_traces, inner=args.inner_traces,
        held_out_fraction=C.HELD_OUT_FRACTION, seed=C.TRACE_SUBSET_SEED)
    if not held_out:
        print("the split left no held-out traces", file=sys.stderr)
        return 2

    elites = archive.elites
    print(f"scoring {len(elites)} elites on {len(held_out)} held-out traces\n")
    print(f"  {'variant':<16}{'inner VFS':>11}{'held-out':>11}{'delta':>9}  "
          f"{'EPI':>8}{'MPKI':>8}")

    rows = []
    with CbpNgNode(require_colocated=False) as cbp_node:

        def rescore(e):
            variant = Variant(
                variant_id=f"heldout_{e.variant_id}", struct_name=e.struct_name,
                harcom_source=e.source, template_args=e.template_args,
                generation=e.generation, rationale="held-out re-score")
            outcomes, artifacts = build_fn(
                variant, [Tier.CBP_NG], cbp_node=cbp_node, cbp_root=args.cbp_root)
            if Tier.CBP_NG not in artifacts:
                return e, None, ("build failed: "
                                 f"{outcomes[Tier.CBP_NG].build_diagnostics[:120]}")
            out = run_fn_tier0(variant, artifacts[Tier.CBP_NG], held_out,
                               cbp_node=cbp_node)
            if not out.ran:
                return e, None, f"run failed: {out.run_diagnostics[:120]}"
            return e, out, ""

        # Concurrent for the same reason the generation loop is, and with less
        # to be careful about: nothing here writes to the archive, it is read
        # from a file and never mutated. Each elite's build goes to its own
        # predictors/evolved_<variant_id>.hpp, so the shared checkout is safe.
        # 42 traces on 16 slots leaves the last few minutes of each elite's run
        # using one core; overlapping the elites fills them.
        with futures.ThreadPoolExecutor(max_workers=max(1, len(elites))) as pool:
            done = list(pool.map(rescore, elites))

    for e, out, why in done:
        if out is None:
            print(f"  {e.variant_id:<16}  {why}")
            continue
        m = out.metrics
        print(f"  {e.variant_id:<16}{e.vfs:11.4f}{m.vfs:11.4f}"
              f"{m.vfs - e.vfs:+9.4f}  {m.epi:8.0f}{m.mpki:8.3f}")
        rows.append((e, out))

    if len(rows) < 2:
        print("\ntoo few designs survived for a rank comparison")
        return 0 if rows else 1

    # The question is not whether the scores match -- they are different trace
    # sets, so they will not -- but whether the *ordering* survives. An
    # ordering that does not is a search that fitted its own subset.
    inner_rank = [e.vfs for e, _ in rows]
    held_rank = [o.metrics.vfs for _, o in rows]
    tau = kendall_tau(inner_rank, held_rank)
    print(f"\nKendall tau, inner-loop ranking vs held-out ranking: {tau:+.3f}")
    if tau < 0.6:
        print("  The front does not survive being re-scored on unseen traces. "
              "That is an overfit to the inner subset, not a result.")

    print(f"\nHeld-out VFS re-scored at each depth:")
    depths = sorted(C.DEPTH_SWEEP)
    print("  depth  " + "".join(f"{e.variant_id[-10:]:>12}" for e, _ in rows[:6]))
    for d in depths:
        print(f"  {d:>5}  " + "".join(
            f"{o.depth_scores[d].vfs:12.4f}" for _, o in rows[:6]))
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest(args) -> int:
    """Prove the pieces work before spending a sweep on them.

    Checks, in order: the scoring reimplementation against the organisers'
    published numbers, the lint gate against every shipped predictor, and one
    real build-and-run of the seed.  Each is a thing that has silently broken
    at least once in development.
    """
    import subprocess
    ok = True

    print("== vfs.py against the published results ==")
    csv = os.path.join(args.cbp_root, "docs",
                       "example_and_reference_predictor_results.csv")
    if os.path.exists(csv):
        rc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "vfs.py"),
             "--selftest", csv]).returncode
        ok &= (rc == 0)
    else:
        print(f"  SKIP: {csv} not found")

    print("\n== lint gate against the shipped predictors ==")
    import harcom_lint
    pdir = os.path.join(args.cbp_root, "predictors")
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".hpp") or fn == "common.hpp":
            continue
        name = fn[:-4]
        src = open(os.path.join(pdir, fn)).read()
        res = harcom_lint.prelint(src, struct_name=name)
        missing = harcom_lint.missing_methods(src)
        # always_taken/never_taken are trivial and may not define every method.
        bad = res.findings or (missing and name not in
                               ("always_taken", "never_taken", "tutorial"))
        if bad:
            ok = False
            print(f"  FAIL {name}: {[f.rule for f in res.findings]} missing={missing}")
        else:
            print(f"  ok   {name}")

    print("\n== build and run the seed ==")
    seed = os.path.join(args.cbp_root, C.SEED_SOURCE)
    trace = args.selftest_trace or os.path.join(args.cbp_root, "gcc_test_trace.gz")
    if not (os.path.exists(seed) and os.path.exists(trace)):
        print(f"  SKIP: need {seed} and {trace}")
        return 0 if ok else 1

    src = agents.rename_struct(open(seed).read(), C.SEED_PREDICTOR, "selftest_bp")
    b = CbpNgNode.build_cbp(args.cbp_root, src, "selftest_bp", "selftest")
    print(f"  build success={b.success} in {b.build_duration_s:.1f}s")
    if not b.success:
        print(b.diagnostics[:2000])
        return 1
    r = CbpNgNode.run_cbp(b.binary, trace, trace_name="selftest",
                          simulation_instructions=args.selftest_instructions)
    print(f"  run   success={r.success} in {r.wall_s:.1f}s")
    if not r.success:
        print(r.stderr_tail[:1000])
        return 1

    from vfs import sweep as depth_sweep
    scores = depth_sweep([r.counters], C.DEPTH_SWEEP)
    m = scores[C.SHIPPED_DEPTH]
    print(f"  VFS {m.vfs:.4f} at depth {C.SHIPPED_DEPTH}, EPI {m.epi:.0f}, "
          f"P1 {m.p1_latency}, P2 {m.p2_latency}, MPKI {m.mpki:.3f}")
    print("  depth sweep: " + ", ".join(
        f"{d}:{s.vfs:.3f}" for d, s in sorted(scores.items())))

    # The port-fidelity gate, on the one design whose Tier-0 number we just
    # measured.  This is the cheapest place the claim "Tiers 1 and 2 run the
    # same algorithm" can be checked at all, and it needs no simulator, so
    # there is no excuse for not checking it on every selftest.
    print("\n== port fidelity: the ported algorithm against Tier 0 ==")
    port_mpki, note = port_selfcheck_mpki(agents.seed_template_args(), [trace])
    if port_mpki is None:
        print(f"  SKIP: {note}")
    else:
        ratio = port_mpki / m.mpki
        within = abs(ratio - 1.0) <= C.PORT_SELFCHECK_TOLERANCE
        ok &= within
        print(f"  Tier 0 {m.mpki:.3f} vs ported {port_mpki:.3f} conditional "
              f"MPKI  ratio {ratio:.3f}  ({note})")
        print(f"  {'ok' if within else 'FAIL'}: the gap is the per-block vs "
              f"per-branch history advance; tolerance is "
              f"{C.PORT_SELFCHECK_TOLERANCE:.0%}")

    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="check scoring, lint gate and one real build/run, then exit")
    p.add_argument("--arm", choices=["main", "offline"], default="main",
                   help="'offline' replaces the design agent with deterministic "
                        "template-parameter mutation -- the harness control")
    p.add_argument("--generations", type=int, default=C.GENERATIONS)
    p.add_argument("--variants-per-generation", type=int,
                   default=C.VARIANTS_PER_GENERATION)
    p.add_argument("--inner-traces", type=int, default=C.INNER_TRACES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cbp-root", default=C.CBP_NG_ROOT,
                   help="where the WORKERS find cbp-ng (inside the container)")
    p.add_argument("--seed-root", default=None,
                   help="where THIS process finds cbp-ng, to read the seed "
                        "predictor from (default: --cbp-root)")
    p.add_argument("--trace-dir", default=C.CBP_TRACE_DIR,
                   help="where the WORKERS find the CBP-NG traces")
    p.add_argument("--host-trace-dir", default=None,
                   help="where THIS process can list them, if that differs "
                        "(a bind mount has two names)")
    p.add_argument("--champsim-root", default=C.CHAMPSIM_ROOT)
    p.add_argument("--champsim-trace-dir", default=C.CHAMPSIM_TRACE_DIR)
    p.add_argument("--host-champsim-trace-dir", default=None)
    p.add_argument("--tier1-traces", type=int, default=8)
    p.add_argument("--gem5-root", default=C.GEM5_ROOT)
    p.add_argument("--gem5-workload-dir", default=C.GEM5_WORKLOAD_DIR)
    p.add_argument("--gem5-outdir", default=C.GEM5_OUTDIR_ROOT,
                   help="where gem5 writes stats.txt, ON THE WORKER")
    p.add_argument("--no-tier1", dest="tier1", action="store_false",
                   help="Tier-0 search only; no ChampSim pool needed")
    p.add_argument("--no-tier2", dest="tier2", action="store_false")
    p.add_argument("--selftest-trace", default=None)
    p.add_argument("--selftest-instructions", type=int, default=40_000_000)
    p.add_argument("--local", action="store_true",
                   help="start a single-process Ray declaring the cbp_ng "
                        "resource; for the smoke test, not for a sweep")
    p.add_argument("--local-cbp-slots", type=int, default=4,
                   help="concurrent Tier-0 tasks under --local")
    p.add_argument("--score-held-out", action="store_true",
                   help="re-score an existing archive's front on the traces no "
                        "feedback ever came from, and report whether the "
                        "ordering survives")
    p.add_argument("--archive", default=None,
                   help="archive.json to score (default: <out>/archive.json)")
    p.add_argument("--resume-archive", default=None,
                   help="archive.json from an interrupted sweep: reload it, skip "
                        "generation 0, and continue at the generation after the "
                        "newest elite in it")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest(args)
    if args.score_held_out:
        return score_held_out(args)
    try:
        return run_sweep(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
