"""Paths, budgets and knobs for the branch-predictor evolution loop.

Everything the loop can be told without editing code lives here. The numbers
that matter for cost -- ``INNER_TRACES``, ``FULL_PASS_TRACES``, the tier
promotion fractions -- are the ones the proposal's cost table was computed
from; changing them changes what a generation costs, so they are named rather
than buried.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
PORTS_DIR = HERE / "ports"
TOOLS_DIR = HERE / "tools"
GEM5_CONFIG = HERE / "gem5" / "depth_sweep.py"
OUT_DIR = Path(os.environ.get("BPE_OUT_DIR", str(HERE / "out")))

# What gets shipped to every worker.  The example's own modules travel as the
# working directory and this repo's chia as py_modules, so a task imports THIS
# chia and THIS cbp_ng.py whatever each worker image happens to have baked in.
# Without it the Tier-0 tasks fail on the worker with an import error for
# `constants`, which reads like a missing dependency and is really a missing
# runtime env.
_REPO_ROOT = HERE.parents[1]
RUNTIME_ENV = {
    "working_dir": str(HERE),
    "py_modules": [str(_REPO_ROOT / "chia")],
    "excludes": ["out/", "out*/", "__pycache__", "**/__pycache__", "**/*.pyc",
                 "tools/cbp2champsim", "gem5/workloads/bsearch",
                 "gem5/workloads/sort", "gem5/workloads/dijkstra"],
}

# -- checkouts on the worker filesystem --------------------------------------

CBP_NG_ROOT = os.environ.get("BPE_CBP_NG_ROOT", "/home/ray/cbp-ng")
CHAMPSIM_ROOT = os.environ.get("BPE_CHAMPSIM_ROOT", "/home/ray/champsim")
GEM5_ROOT = os.environ.get("BPE_GEM5_ROOT", "/home/ray/gem5")

# CBP-NG traces (Tier 0) and the ChampSim traces converted from them (Tier 1).
CBP_TRACE_DIR = os.environ.get("BPE_CBP_TRACE_DIR", "/home/ray/cbp-ng/traces")
CHAMPSIM_TRACE_DIR = os.environ.get(
    "BPE_CHAMPSIM_TRACE_DIR", "/home/ray/champsim/traces")
GEM5_WORKLOAD_DIR = os.environ.get("BPE_GEM5_WORKLOAD_DIR", "/home/ray/workloads")

# -- the seed ----------------------------------------------------------------

SEED_PREDICTOR = os.environ.get("BPE_SEED_PREDICTOR", "tage")
SEED_SOURCE = f"predictors/{SEED_PREDICTOR}.hpp"

# -- trace budget ------------------------------------------------------------
#
# The 168 CBP-NG traces hold 4,989,128,151 instructions between them -- median
# 24.3M, shortest 13.0M, longest 130.0M.  HELD_OUT_FRACTION takes a seeded
# quarter (42 traces, 1,108,195,562 instructions) out of reach of feedback
# entirely: a design that wins on traces the agent's feedback was computed from
# has not been shown to generalise.  That leaves 126 for the inner loop,
# 3,880,932,589 instructions per variant.
#
# Throughput, measured on this machine (16 cores) over whole passes rather than
# extrapolated from a sample -- an earlier figure here came from timing the
# first 16 traces alphabetically and was 50% optimistic, which is the same
# sampling mistake in miniature that the 15-trace subset used to make.
#
# HARCOM's rate is per *prediction*, so it varies 2.5x across the set by branch
# density, not by length:
#
#   int_32_trace       317 kIPS solo
#   502-gcc-all_16112  228 kIPS solo
#   compress_44_trace  124 kIPS solo, under 67 contended sixteen ways
#
# What to plan with is the aggregate over a full 126-trace pass:
#
#   2.09 MIPS  one variant alone   (3.88G instructions in 31 minutes)
#   1.96 MIPS  a generation of 3   (11.6G of Tier 0 in 99 minutes; 112 with
#              its Tier-1 and Tier-2 promotion, which is the figure to plan on)
#
# Sixteen cores buy about seven, not sixteen: HARCOM is memory-bound and the
# slots contend.  Ten generations is therefore ~19 hours, not the ~9 the first
# estimate here suggested.
#
# 24 is the *default*, not what the full sweep runs; launch_full.sh passes
# --inner-traces 126, the whole non-held-out pool.  The smaller default is for
# --selftest and for trying the loop out, where a 5.75-core-hour variant would
# turn a smoke test into an afternoon.

INNER_TRACES = int(os.environ.get("BPE_INNER_TRACES", "24"))
FULL_PASS_TRACES = int(os.environ.get("BPE_FULL_PASS_TRACES", "168"))
HELD_OUT_FRACTION = float(os.environ.get("BPE_HELD_OUT_FRACTION", "0.25"))
TRACE_SUBSET_SEED = int(os.environ.get("BPE_TRACE_SUBSET_SEED", "20260826"))

WARMUP_INSTRUCTIONS = int(os.environ.get("BPE_WARMUP_INSTRUCTIONS", "1000000"))

# Tier 0's measurement window. The default is deliberately larger than the
# longest trace in the official set (130.0M), which makes it mean "run to the
# end of the trace": cbp.hpp's loop is
#     while (!warmed_up || ninstr < measurement_instructions)
# and it leaves that loop by throwing out_of_instructions at EOF. A finite
# window smaller than the trace would be a real cap, and on this set 153 of the
# 168 traces are shorter than 41M -- so a 40M window silently measured "the
# whole trace" for most of them and a genuine 40M for fifteen, which is two
# different experiments reported as one number.
SIM_INSTRUCTIONS = int(os.environ.get("BPE_SIM_INSTRUCTIONS", "2000000000"))

# Tier 1's measurement window, separately settable, and it matters more than it
# looks.  ChampSim does not stop at the end of a trace -- it reopens it and
# keeps going ("*** Reached end of trace") -- so asking for more instructions
# than the trace holds measures a predictor that has seen the same code several
# times over.  With the 3.0M-instruction converted gcc trace and the 40M default
# the trace runs thirteen times, and Tier 1's conditional MPKI comes out 20-50%
# below Tier 0's for no reason connected to the predictor.  Worse, the size of
# the gap depends on how fast the design converges, so it is not a constant that
# can be calibrated away.
#
# This is now only the FALLBACK, used for a trace whose length is not in
# TRACE_LENGTHS_FILE. The window is per-trace: one global number cannot be
# right for a set whose traces run from 13.0M to 130.0M instructions, and
# picking a single value means either wrapping the short traces or truncating
# the long ones. See ``tier1_window_for`` in evaluator.py.
TIER1_SIM_INSTRUCTIONS = int(
    os.environ.get("BPE_TIER1_SIM_INSTRUCTIONS", "2000000"))

# basename -> instruction count, so Tier 1 can be told exactly how far to run
# each trace. Built by tools/measure_traces.py.
TRACE_LENGTHS_FILE = os.environ.get(
    "BPE_TRACE_LENGTHS",
    str(Path.home() / "bp_evolve_data" / "trace_lengths.json"))

# Instructions of headroom left at the end of a trace for Tier 1. ChampSim
# fetches past the instruction it is retiring to keep the pipeline full, so a
# window set to exactly (length - warmup) can still read off the end and wrap.
# Losing this many instructions off a >=13M trace costs under 1% of the
# measurement and makes the wrap impossible.
TIER1_TAIL_MARGIN = int(os.environ.get("BPE_TIER1_TAIL_MARGIN", "100000"))

# What ChampSim prints when it wraps a trace.  Seeing it means the window above
# is too large for the trace, and any Tier-0/Tier-1 MPKI comparison is void.
TIER1_LOOP_MARKER = "Reached end of trace"

# -- promotion ---------------------------------------------------------------
#
# Tier 0 scores everything; Tier 1 takes the Pareto front; Tier 2 takes elites.
# Caps are absolute, not fractions, because a generation that happens to
# produce a wide front should not silently cost ten times the previous one.

TIER1_MAX_PROMOTED = int(os.environ.get("BPE_TIER1_MAX_PROMOTED", "6"))
TIER2_MAX_PROMOTED = int(os.environ.get("BPE_TIER2_MAX_PROMOTED", "2"))

# A promoted design must reproduce its Tier-0 branch behaviour on the Tier-1
# branch stream before any cross-tier claim is made.  Two ports of the same
# predictor should agree on MPKI to within this relative tolerance; beyond it
# the disagreement is the translation, not the cost model.
MPKI_AGREEMENT_TOLERANCE = float(
    os.environ.get("BPE_MPKI_AGREEMENT_TOLERANCE", "0.05"))

# The same question asked of the port alone, by tools/port_selfcheck.py: same
# algorithm, same traces, same window, no simulator on either side.  What is
# left is one structural difference -- CBP-NG advances its global history once
# per prediction block, the port once per branch -- and it is one-sided: the
# port comes out slightly *better*, by 2.1% to 5.5% over designs from 6.2 to
# 14.4 Tier-0 MPKI (ratios 0.979, 0.962, 0.946, 0.945).
#
# So this is looser than MPKI_AGREEMENT_TOLERANCE above and must be: 5% would
# reject the correct ports at the bottom of that range.  It is not slack for a
# translation error -- a real one moves MPKI by far more than 10% -- it is the
# width of a known, measured, structural gap.
PORT_SELFCHECK_TOLERANCE = float(
    os.environ.get("BPE_PORT_SELFCHECK_TOLERANCE", "0.10"))

# The Tier-1 MPKI to compare against is the *conditional*-branch rate, not the
# all-branch rate ChampSim prints first.  A HARCOM predictor is only asked for
# conditional directions; ChampSim's total also charges it for the BTB and the
# return-address stack, which no Tier-0 number contains.  Comparing totals
# would make every port look ~50% wrong on this trace set and none of it would
# be the predictor.
TIER1_MPKI_BRANCH_TYPES = tuple(
    t for t in os.environ.get(
        "BPE_TIER1_MPKI_BRANCH_TYPES", "BRANCH_CONDITIONAL").split(",") if t)

# -- the depth sweep ---------------------------------------------------------
#
# CBP-NG pins prediction-to-execution distance at nine stages.  These are the
# depths the mapper re-scores every design at.  Nine is the shipped value and
# must stay in the list -- it is the baseline every rank correlation is against.

SHIPPED_DEPTH = 9
DEPTH_SWEEP = tuple(int(d) for d in os.environ.get(
    "BPE_DEPTH_SWEEP", "3,6,9,12,16,20,25,30").split(","))

# -- MAP-Elites archive ------------------------------------------------------
#
# Bins are set by what the simulator reports, not by what the agent claims:
# energy per instruction and the two prediction latencies.  CBP-NG imposes no
# storage budget, so those three are the whole cost model -- binning on them is
# what stops the search collapsing into "make TAGE bigger".

EPI_BIN_EDGES = tuple(float(x) for x in os.environ.get(
    "BPE_EPI_BIN_EDGES", "100,250,500,900,1500").split(","))
P1_LATENCY_BINS = (0, 1, 2, 3)          # cycles; >=3 collapses into the last bin
P2_LATENCY_BINS = (0, 1, 2, 3, 4, 6)

# -- generations -------------------------------------------------------------

GENERATIONS = int(os.environ.get("BPE_GENERATIONS", "20"))
VARIANTS_PER_GENERATION = int(os.environ.get("BPE_VARIANTS_PER_GENERATION", "4"))
MAX_REPAIR_ROUNDS = int(os.environ.get("BPE_MAX_REPAIR_ROUNDS", "3"))

# -- timeouts ----------------------------------------------------------------

CBP_BUILD_TIMEOUT_S = int(os.environ.get("BPE_CBP_BUILD_TIMEOUT_S", "300"))

# The floor for a Tier-0 run, and the only value used for a trace whose length
# is unknown.  See cbp_run_timeout_for(): a trace with a measured length gets a
# timeout sized from it instead, because a single number cannot serve a set
# that runs from 13M to 130M instructions at rates that themselves vary 2.5x.
CBP_RUN_TIMEOUT_S = int(os.environ.get("BPE_CBP_RUN_TIMEOUT_S", "1800"))

# Measured on this machine, per HARCOM slot with all 16 busy:
#
#   int_32_trace         317 kIPS solo
#   502-gcc-all_16112    228 kIPS solo
#   compress_44_trace    124 kIPS solo, and under 67 contended
#
# The spread is branch density, not instruction count -- HARCOM's work is per
# prediction.  40 kIPS is well under the worst of them, which is the point: the
# cost of a timeout that is too generous is one slot idling on a genuinely hung
# run, and the cost of one that is too tight is every variant failing on the
# six traces past 110M and an archive that never grows past the seed.  That is
# not hypothetical; it is what the flat 1800s did.
CBP_RUN_FLOOR_KIPS = int(os.environ.get("BPE_CBP_RUN_FLOOR_KIPS", "40"))

# Added on top, for process start, binary write and opening a gzip trace.
CBP_RUN_TIMEOUT_SLACK_S = int(os.environ.get("BPE_CBP_RUN_TIMEOUT_SLACK_S", "600"))
CHAMPSIM_BUILD_TIMEOUT_S = int(os.environ.get("BPE_CHAMPSIM_BUILD_TIMEOUT_S", "900"))
CHAMPSIM_RUN_TIMEOUT_S = int(os.environ.get("BPE_CHAMPSIM_RUN_TIMEOUT_S", "1800"))
GEM5_BUILD_TIMEOUT_S = int(os.environ.get("BPE_GEM5_BUILD_TIMEOUT_S", "5400"))
GEM5_RUN_TIMEOUT_S = int(os.environ.get("BPE_GEM5_RUN_TIMEOUT_S", "3600"))

# -- LLM ---------------------------------------------------------------------

LLM_BACKEND = os.environ.get("BPE_LLM_BACKEND", "claude").lower()
LLM_MODEL = os.environ.get("BPE_LLM_MODEL", "claude-opus-4-6")
LLM_TIMEOUT_SECONDS = int(os.environ.get("BPE_LLM_TIMEOUT_SECONDS", "1800"))
LLM_EXTRA_CLI_ARGS = ["--effort", "max"]
LLM_RESOURCE = float(os.environ.get("BPE_LLM_RESOURCE", "1.0"))
CLAUDE_PROJECTS_DIR = os.environ.get(
    "BPE_CLAUDE_PROJECTS_DIR", "/home/ray/.claude/projects/-home-ray-llm-env")
GEMINI_MODEL = os.environ.get("BPE_GEMINI_MODEL", "gemini-3.6-flash")
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
VERTEX_MAX_TOKENS = int(os.environ.get("BPE_VERTEX_MAX_TOKENS", "65536"))

LLM_SYSTEM_MESSAGE = (
    "You are a microarchitect writing branch predictors in HARCOM, a C++20 "
    "hardware-complexity library. HARCOM's reg/val/arr/ram types are opaque: "
    "you cannot read their values, so you cannot branch on them. Every "
    "data-dependent choice is a select() (a mux) or an execute_if(). Every "
    "compiler warning is an error. Reply with exactly the JSON object asked "
    "for, and nothing else."
)

MAX_OUTPUT_CHARS = int(os.environ.get("BPE_MAX_OUTPUT_CHARS", "200000"))


# -- Tier 2: gem5 -------------------------------------------------------------
#
# The depth sweep is the point of this tier, so it is spelled out rather than
# left to a config default.  gem5's O3 model splits the distance from fetch to
# execute across four inter-stage delays; the labels below name the *total*,
# which is what result_mapper_fn correlates VFS against.  DEPTH 9 is the one
# CBP-NG assumes, and it must stay in the list for the same reason
# SHIPPED_DEPTH must: every correlation is against it.
#
# The delays are distributed as evenly as the four stages allow, because
# nothing in CBP-NG's model says *where* the pipeline is long -- only how far
# it is from a prediction to its resolution.

GEM5_ISA = os.environ.get("BPE_GEM5_ISA", "X86")
GEM5_VARIANT = os.environ.get("BPE_GEM5_VARIANT", "opt")
GEM5_DEPTHS = tuple(int(d) for d in os.environ.get(
    "BPE_GEM5_DEPTHS", "6,9,16,25").split(","))
GEM5_MAX_INSTS = int(os.environ.get("BPE_GEM5_MAX_INSTS", "20000000"))

# Where gem5 writes stats.txt, on the WORKER. It cannot default to OUT_DIR: the
# archive and the lineage DB live on the driver, and run_gem5 creates this
# directory inside the container, where the driver's home does not exist and is
# not writable. The failure is a bare PermissionError on a path that looks
# perfectly good from the machine you typed it on.
GEM5_OUTDIR_ROOT = os.environ.get("BPE_GEM5_OUTDIR_ROOT", "/home/ray/gem5_out")

# SE-mode binaries, statically linked.  gem5 executes programs rather than
# replaying traces, so Tier 2 changes the workload as well as the cost model --
# it is corroboration, never the same measurement.
GEM5_WORKLOADS = tuple(
    w for w in os.environ.get("BPE_GEM5_WORKLOADS", "bsearch,sort,dijkstra").split(",")
    if w)


# The port self-check runs on the driver, not a worker, and it runs over the
# same trace set Tier 0 scored -- which is now 126 traces at full length. Those
# per-trace runs are independent, and nothing else is on the machine while a
# design is being promoted (Tier 0 has finished, Tier 1's ChampSim runs are
# already collected), so the driver may have all the cores.
SELFCHECK_JOBS = int(os.environ.get("BPE_SELFCHECK_JOBS", "16"))

# Very generous against a measured ~10 core-minutes for the whole inner set:
# the self-check runs at 6.7 MIPS per core, some 36x HARCOM, because it drives
# the rendered core directly with no library around it.  Ten minutes of work at
# --jobs 16 is well under a minute of wall time; an hour is there for a loaded
# machine and for trace sets larger than this one.
#
# The gate returns (None, why) on timeout and the caller falls back to the much
# weaker Tier-1/Tier-0 ratio, so a tight timeout here quietly downgrades the
# check rather than failing loudly -- which is exactly how it failed before:
# see _trace_key in bp_evolve_loop.py.
SELFCHECK_TIMEOUT_S = int(os.environ.get("BPE_SELFCHECK_TIMEOUT_S", "3600"))
