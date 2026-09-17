"""Configuration for the Titan loop.

Packaging follows the memcpy example rather than riscv_extensions: flat
top-level modules plus a declared ``RUNTIME_ENV``, not ``sys.path.insert``
with package-style imports.  Two reasons, and the second is the load-bearing
one:

  * Flat modules make every ``@ChiaFunction`` deserializable on a worker by
    name, with no ``register_pickle_by_value``.
  * ``py_modules`` ships the head's *current* ``chia`` package and shadows
    whatever is baked into the worker image.  Titan uses nodes newer than the
    published images, and this removes an entire class of "the chia on the
    worker is too old" failure.

One consequence to keep in mind: ``working_dir`` puts this directory on
sys.path on every worker.  A module here named after a stdlib module would
shadow it -- which is why the encoding tables live in ``ime_encodings.py``
and not ``encodings.py``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = EXAMPLE_DIR.resolve().parents[1]

OUT_DIR = EXAMPLE_DIR / "out"
PROMPTS_DIR = EXAMPLE_DIR / "prompts"
SPEC_DIR = EXAMPLE_DIR / "specs" / "ime"

RUNTIME_ENV = {
    "working_dir": str(EXAMPLE_DIR),
    "py_modules": [str(_REPO_ROOT / "chia")],
    "excludes": ["out/", "__pycache__", ".mypy_cache"],
}

# --- container paths -------------------------------------------------------
CHIPYARD_PATH = os.environ.get("TITAN_CHIPYARD_PATH", "/home/ray/chipyard")
CHIPYARD_SRC_PATH = os.environ.get(
    "TITAN_CHIPYARD_SRC_PATH", os.path.join(CHIPYARD_PATH, "generators"))

#: Saturn, not BOOM.  riscv_extensions targets `generators/boom`; every
#: submodule path in this example moves accordingly.
SATURN_REPO_REL = "generators/saturn"
SATURN_SRC_REL = "generators/saturn/src/main/scala"

#: Diff capture.  Every repo a loop agent may edit has to be snapshotted each
#: iteration or a failing run cannot be reproduced.  rocket-chip is on this
#: list because the vtype CSR is defined there rather than in Saturn, and
#: riscv-isa-sim because the Stage M agent edits Spike.
CHIPYARD_DIFF_SUBMODULES = [
    SATURN_REPO_REL,
    "generators/rocket-chip",
    "generators/shuttle",
    # The Stage 2 model agent edits Spike in this same tree.  It is reset once
    # at the top of a run and never between stages -- resetting before the RTL
    # stage would throw away the model the run just built.
    "toolchains/riscv-tools/riscv-isa-sim",
]

# --- build targets ---------------------------------------------------------
#: The cosim harness config the agent is responsible for creating.  Saturn
#: ships Rocket and Shuttle integrations only -- there is no BOOM path -- so
#: the MegaBOOM configs riscv_extensions uses do not transfer.  Shuttle is
#: the better host: in-order, so a divergence is far easier to localise.
#: r9: the loop now OWNS this config instead of asking the agent for it.
#: See ``S2_COSIM_SCALA`` below for why (a Shuttle DebugROB DPI hazard that
#: made every S2 test fail at the first bootrom instruction).
COSIM_CONFIG = os.environ.get("TITAN_COSIM_CONFIG", "TitanS2CosimConfig")

#: Where the loop drops that config in the chipyard tree.  Ignored via
#: ``.git/info/exclude`` (write_files' ``exclude_rel``), so it never rides
#: along in ``collect_diff`` and survives ``reset_chipyard``'s ``git clean -fd``
#: (no ``-x``).
#:
#: It lives under ``generators/chipyard`` -- which is the chipyard repo
#: itself, not a submodule -- and NOT next to SaturnConfigs.scala, because
#: ``collect_diff`` runs ``git add -N .`` separately inside each submodule and
#: a submodule has its own exclude file: a generated source dropped into
#: ``generators/saturn`` would be ignored by the superproject and staged by
#: the Saturn pass anyway, putting the loop's own harness into the diff that
#: reseeds the run.
S2_COSIM_SCALA_REL = os.environ.get(
    "TITAN_S2_COSIM_SCALA_REL",
    "generators/chipyard/src/main/scala/config/TitanS2Cosim.scala")

#: The S2 harness, as source, generated from SYNTH_CONFIG.
#:
#: Three things this encodes, all of them learned the hard way in r8/r9:
#:
#: 1. **retireWidth 1.**  Shuttle's DebugROB support calls the ``debug_rob``
#:    DPI once per retire lane -- ``retireWidth`` separate ``popTrace``
#:    instances reading one shared C++ deque, in an order Verilator does not
#:    define.  With the default 2-wide core the two lanes routinely pop the
#:    pair out of order, so cospike sees the second instruction first and
#:    aborts with ``PC mismatch spike 10000 != DUT 10004`` -- at the *first
#:    bootrom instruction*, before any test code runs.  That is why r8's S2
#:    reported 841/841 failing in 2.5 minutes.  It reproduces on a pristine
#:    tree with Saturn's own ``GENV256D128ShuttleCosimConfig``, so it is a
#:    harness bug, not anything an agent did.  Rocket cosim is unaffected
#:    (retireWidth is 1 there, one popTrace).
#: 2. **It extends SYNTH_CONFIG.**  S2 must judge the *same* design the
#:    directed suite just passed.  A hand-written parallel cosim config is a
#:    second place for the design to be specified, and r8 proved the agent
#:    will let the two drift.
#: 3. **The loop writes it, not the agent.**  A gate whose harness the graded
#:    party maintains is not a gate.
#:
#: The cost of (1) is real and worth stating: S2 grades the vector unit on a
#: 1-wide host, so a bug that only appears when two instructions commit in the
#: same cycle is outside this gate.  S1 (directed) still runs on the 2-wide
#: DIRECTED_CONFIG, so the design proper is exercised dual-issue; only the
#: lockstep RVV regression is narrowed.  The principled fix is upstream --
#: Shuttle should drain its DebugROB through one popTrace with an ordered
#: multi-pop DPI, or pass a real ``has_wb`` the way RocketCore does -- and if
#: that lands, drop the WithShuttleRetireWidth line and nothing else changes.
S2_COSIM_SCALA = """// GENERATED BY titan_loop -- do not edit, do not commit.
package chipyard

import org.chipsalliance.cde.config.{Config}

class %(cosim)s extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new shuttle.common.WithShuttleDebugROB ++
  new shuttle.common.WithShuttleRetireWidth(1) ++
  new %(base)s)
"""

#: The same design without the cosim harness, so PPA numbers do not include
#: debug plumbing.  Anything that changes the design itself must go in the
#: shared base config, not in the cosim fragment, or the two diverge.
SYNTH_CONFIG = os.environ.get(
    "TITAN_SYNTH_CONFIG", "TitanV256D128ShuttleConfig")

#: An unmodified Saturn config that already exists in this checkout.  The
#: preflight builds it to prove the environment works before any agent has
#: written anything -- a failure there is an environment problem, not a
#: design problem, and telling those apart on day one is worth a lot.
BASELINE_CONFIG = os.environ.get("TITAN_BASELINE_CONFIG",
                                 "GENV256D128ShuttleConfig")

CONFIG_PACKAGE = os.environ.get("TITAN_CONFIG_PACKAGE", "chipyard")
BUILD_MAKE_JOBS = int(os.environ.get("TITAN_BUILD_MAKE_JOBS", "16"))
BUILD_TIMEOUT_S = int(os.environ.get("TITAN_BUILD_TIMEOUT_S", "3600"))
VERILATOR_THREADS = int(os.environ.get("TITAN_VERILATOR_THREADS", "8"))
SIM_ZERO_INIT_DEFINES = "+define+RANDOM=0"

# --- DUT geometry ----------------------------------------------------------
#: Saturn takes vLen/dLen as elaboration parameters, so the Stage 3 design
#: sweep gets its knobs for free.  These are the round-one defaults; the
#: directed suite derives every legal tile geometry from VLEN alone.
VLEN = int(os.environ.get("TITAN_VLEN", "256"))
DLEN = int(os.environ.get("TITAN_DLEN", "128"))

#: Round one implements the integer non-widening subset: the tile load/store
#: pair and the same-width multiply-accumulate.
ROUND_ONE_INSNS = ("vmmacc.vv", "vmtl.v", "vmts.v")

#: Round two adds the quad-widening integer multiply-accumulate -- Int8 x Int8
#: accumulated into Int32 (Zvvi8i32mm), which is the shape llama.cpp's INT8
#: GEMM actually needs; vmmacc.vv can only accumulate at the input width and
#: so wraps a K-length dot product into 8 bits.  Its A/B tiles ride the same
#: vmtl.v / vmts.v from round one, because vtype.SEW is the *accumulator*
#: width for the whole family and the tile load/stores move SEW-wide storage
#: elements without looking inside the four Int8 values packed in each.
#: vwmmacc.vv (W=2), v8wmmacc.vv (W=8), the transposing tile load/stores and
#: the floating-point group follow in later rounds.  The signed/unsigned
#: variants are not separate mnemonics: vtype.altfmt_A / altfmt_B select them,
#: and only the plain signed form is generated.
ROUND_TWO_INSNS = ("vqmmacc.vv",)

#: Round three closes out the integer half of the family and adds the
#: transposing tile transfers:
#:
#:   vwmmacc.vv  (W=2)  -- Int8 -> Int16 at SEW=16, Int16 -> Int32 at SEW=32
#:   v8wmmacc.vv (W=8)  -- Int8 -> Int64 at SEW=64
#:   vmttl.v / vmtts.v  -- Zvvmttls, the transposing tile load/store pair
#:
#: The two multiply-accumulates are the round-two datapath at a different
#: packing depth: vtype.SEW is still the accumulator width, the A/B tiles
#: still ride vmtl.v, and only funct6 and the number of logical elements per
#: storage element change (spec Zvvmm table: W in {1,2,4,8}, A/B width SEW/W,
#: C width SEW).  The transposing pair differs from vmtl.v / vmts.v in one
#: line of the Sail body -- `mem_off = (i % linesize) * LD + (i / linesize)`
#: instead of `(i / linesize) * LD + (i % linesize)` -- and in two encoding
#: bits; the register-side tile_reg_idx mapping is identical.  It is what
#: lets an INT8 GEMM consume a row-major B (or a column-major A) without a
#: software repack, which is the dominant non-MAC cost in a tiled kernel.
#:
#: Deliberately *not* in round three: the whole Zvvfmm floating-point family
#: and the microscaled integer-input forms (vfwimmacc.vv / vfqimmacc.vv /
#: vf8wimmacc.vv).  They need an FP datapath, altfmt format decoding, the
#: G/psm/rnd accumulation-and-rounding model and fflags -- none of which the
#: stated INT8 edge-LLM goal needs, and all of which land in the shared
#: int/fp issue path where round one's bugs clustered.  See
#: titan_runs/round3_design.md.
ROUND_THREE_INSNS = ("vwmmacc.vv", "v8wmmacc.vv", "vmttl.v", "vmtts.v")

#: Every instruction the generators know how to emit and judge.  The order is
#: round order, so an index into this is an implementation milestone.
ALL_INSNS = ROUND_ONE_INSNS + ROUND_TWO_INSNS + ROUND_THREE_INSNS

#: What a seeded tree already implements, and what this round adds.
#: ``helpers.instruction_scope`` reads these two names first and only falls
#: back to (ROUND_ONE_INSNS, ROUND_TWO_INSNS) when they are absent -- the
#: hook it left open so that a new round needs no edit in helpers.py.  Set
#: them and every prompt names the right halves: rounds one and two are the
#: regression surface, round three is the work.
IMPLEMENTED_INSNS = ROUND_ONE_INSNS + ROUND_TWO_INSNS
NEW_INSNS = ROUND_THREE_INSNS


def _scope_insns() -> tuple:
    """Which instructions this run's directed suite and prompts cover.

    ``TITAN_INSNS`` selects the scope.  It accepts a round name -- ``one``,
    ``two``, ``three``, ``all`` -- or an explicit comma-separated mnemonic
    list, and defaults to ``all`` (rounds one + two + three).  Unknown
    mnemonics raise here rather than silently generating an empty suite six
    minutes into a loop iteration.

    This is the single knob the rest of the loop reads: ime_tests'
    ``directed_suite`` derives its tiers from :data:`INSNS`, and
    ``helpers.implemented_and_new`` splits ROUND_ONE_INSNS from
    ROUND_TWO_INSNS for the prompt text.  To re-run a round-one-only
    regression, ``TITAN_INSNS=one``.

    A round name selects *only* that round's instructions, so
    ``TITAN_INSNS=three`` emits the round-three tiers alone -- useful for
    bisecting a new-instruction regression without recompiling the 244
    programs rounds one and two already cover.
    """
    raw = os.environ.get("TITAN_INSNS", "all").strip()
    named = {"one": ROUND_ONE_INSNS, "1": ROUND_ONE_INSNS,
             "two": ROUND_TWO_INSNS, "2": ROUND_TWO_INSNS,
             "three": ROUND_THREE_INSNS, "3": ROUND_THREE_INSNS,
             "all": ALL_INSNS, "": ALL_INSNS}
    if raw.lower() in named:
        return named[raw.lower()]
    chosen = tuple(tok.strip() for tok in raw.split(",") if tok.strip())
    unknown = [name for name in chosen if name not in ALL_INSNS]
    if unknown:
        raise ValueError(
            f"TITAN_INSNS={raw!r}: unknown mnemonic(s) {unknown}; "
            f"expected a round name (one/two/three/all) or a subset of "
            f"{ALL_INSNS}")
    return chosen


#: The active instruction scope for this run.  See :func:`_scope_insns`.
INSNS = _scope_insns()

#: The build harness assembles with riscv64-unknown-elf-gcc, whose baseline
#: -march is rv64imafd_zicsr_zifencei -- no vector at all.  So the tests must
#: add `_v` for the RVV reference path.  They must NOT ask for zvvm: GCC has
#: no Zvvm support and `-menable-experimental-extensions` is a clang flag it
#: would reject outright.  The three IME instructions go in as `.insn` words
#: built by ime_encodings from the spec, which needs nothing from the
#: assembler and cannot drift from the spec without the encoding self-test
#: saying so.
#: Note the position of `v`: a RISC-V ISA string puts single-letter
#: extensions before multi-letter ones, so it is rv64imafd**v**_zicsr...,
#: not rv64imafd_zicsr_zifencei_v -- gcc rejects the latter outright with
#: "unexpected ISA string at end".
MARCH_IME = os.environ.get("TITAN_MARCH", "rv64imafdv_zicsr_zifencei")
IME_CFLAGS = [f"-march={MARCH_IME}"]

#: The clang route, for when LLVM >= 23.1 is in the chia-riscv-cross image and
#: we want real mnemonics in the disassembly instead of raw words.  An
#: optimisation for debuggability, not a prerequisite -- and note that
#: assembling successfully would only prove the encoding matches LLVM's 0.1
#: draft, never that the v0.9.0 semantics are right.
MARCH_IME_CLANG = "rv64gcv_zvvmm0p1_zvvmtls0p1"
IME_CFLAGS_CLANG = ["-menable-experimental-extensions",
                    f"-march={MARCH_IME_CLANG}"]

#: Spike's source, which the Stage 2 model agent edits.  Upstream
#: riscv-isa-sim has never heard of Zvvm, so this model does not exist until
#: the loop produces it -- which is why Stage 1 is designed to need no Spike
#: at all rather than to wait for one.
SPIKE_SRC_REL = os.environ.get("TITAN_SPIKE_SRC_REL",
                               "toolchains/riscv-tools/riscv-isa-sim")
SPIKE_SRC_PATH = os.path.join(CHIPYARD_PATH, SPIKE_SRC_REL)

#: --isa for a standalone Spike run.  Baseline RVV only: the model agent
#: decides whether its extension needs an ISA-string opt-in, and if it adds
#: one this has to name it or every matrix instruction traps as illegal.
#: Deliberately not pre-populated with a guess at the string it will pick.
SPIKE_ISA = os.environ.get("TITAN_SPIKE_ISA", "rv64gcv")

#: A model that clamps LAMBDA everywhere passes every test by declining every
#: one of them.  Above this fraction of skips the model stage treats the run
#: as a failure and says so.
MODEL_MAX_SKIP_FRACTION = float(
    os.environ.get("TITAN_MODEL_MAX_SKIP_FRACTION", "0.5"))

#: Base-ISA / RVV regression: Saturn already ships riscv-vector-tests with a
#: build-tests.sh preconfigured for TEST_MODE=cosim at VLEN 128 and 256.  The
#: S2 gate is therefore free -- it exists before the implementation does.
RVV_REGRESSION_DIR = os.environ.get(
    "TITAN_RVV_TESTS_DIR",
    os.path.join(CHIPYARD_PATH, "generators", "saturn", "riscv-vector-tests"))

# --- loop control ----------------------------------------------------------
TITAN_LOG_ROOT = os.environ.get(
    "TITAN_LOG_ROOT", os.path.join(tempfile.gettempdir(), "titan"))
MAX_ITERS = int(os.environ.get("TITAN_MAX_ITERS", "60"))
#: Stage M is a transcription task against a formal semantics, not a
#: microarchitecture search, so it should converge in far fewer turns
#: than the RTL stage -- and if it does not, that is a signal about the
#: spec or the prompt rather than a reason to keep paying.
MODEL_MAX_ITERS = int(os.environ.get("TITAN_MODEL_MAX_ITERS", "25"))
DEBUG_MAX_ITERS = int(os.environ.get("TITAN_DEBUG_MAX_ITERS", "5"))
SIM_TIMEOUT_CYCLES = int(os.environ.get("TITAN_SIM_TIMEOUT_CYCLES", "10000000"))

#: Stage 3 randomised sweep.  riscv-dv cannot reach vector instructions at
#: all, so there is no generator node here: ime_stress.py enumerates legal
#: tile geometries and emits paired programs on the head.  That also removes
#: the `gen` worker (and its Xcelium licence) from the cluster.
STRESS_TEST_HOURS = float(os.environ.get("TITAN_STRESS_TEST_HOURS", "24"))
STRESS_TEST_CASES_PER_GEOM = int(
    os.environ.get("TITAN_STRESS_CASES_PER_GEOM", "64"))
STRESS_TEST_MAX_CYCLES = int(
    os.environ.get("TITAN_STRESS_MAX_CYCLES", "10000000"))
COSIM_VRUN = float(os.environ.get("TITAN_COSIM_VRUN", "0.9"))

#: How many of the 841 riscv-vector-tests S2 runs per iteration.  0 = all.
#:
#: Measured on this cluster (r9): a cosim of one test is 5-20s, and the
#: cluster sustains 8 concurrent cosims (16 ``verilator_run`` slots, but
#: ``num_cpus=VERILATOR_THREADS=8`` against 96 CPUs is the binding
#: constraint), so the full suite is roughly an hour of wall clock *per
#: iteration that reaches S2*.  r8 never noticed because all 841 died in
#: 0.6s each.  Set this to e.g. 150 to sample the suite between iterations;
#: the sample is a deterministic stride over the sorted test list, so it
#: spans every instruction family rather than stopping at the v-a-* tests,
#: and it is the same sample every iteration, so a pass and a later failure
#: are comparable.  The final gate should still see the whole suite.
REGRESSION_SAMPLE = int(os.environ.get("TITAN_REGRESSION_SAMPLE", "0"))

#: JSON list of riscv-vector-tests that ALREADY fail on a pristine Saturn, so
#: S2 does not bill them to the agent.
#:
#: build-tests.sh prunes the suites Saturn does not implement, which is why
#: this was assumed empty -- but r9 measured it and it is not: e.g.
#: ``machine_vfcvt_f_x_v-0`` diverges from Spike at the same instruction on a
#: pristine tree (an FP rounding difference, 1 ulp), and would otherwise be
#: reported as a regression the agent caused, every iteration, forever.
#: Produce it once per Saturn pin by running the suite on an unmodified tree
#: and saving the failing names here.  Missing file = no subtraction, and a
#: ``regression_baseline_missing`` event so the gap is visible rather than
#: silent.
REGRESSION_BASELINE_PATH = os.environ.get(
    "TITAN_REGRESSION_BASELINE",
    os.path.join(str(EXAMPLE_DIR), "rvv_baseline_failures.json"))
STRESS_TEST_VRUN_FRACTION = float(
    os.environ.get("TITAN_STRESS_VRUN_FRACTION", "0.5"))

# --- LLM -------------------------------------------------------------------
LLM_MODEL = os.environ.get("TITAN_LLM_MODEL", "claude-opus-4-7")
LLM_EXTRA_ARGS = ["--effort", "max"]
LLM_TIMEOUT_SECONDS = int(os.environ.get("TITAN_LLM_TIMEOUT_SECONDS", "1800"))

# --- debug feedback shaping ------------------------------------------------
#: Deliberately larger than memcpy's 50-line windows.  A vector unit's
#: relevant state -- vtype, vl, the tile geometry in force, which lane
#: diverged -- does not fit in a 50-line commit-log tail.
DUMP_TAIL_LINES = int(os.environ.get("TITAN_DUMP_TAIL_LINES", "200"))
#: Raw simulator-log tail quoted per failing test in the feedback message.
#: Zero by default, and the zero is the point.  Ten S1 iterations across r4
#: and r5 pasted a 300-line tail for each of eight failures on top of a
#: 100,000-character build log, and the agent still could not see a single
#: wrong *value* -- the tails are Verilator's "Assertion failed" boilerplate
#: repeated eight times.  The evidence block (TITAN DIFF / CDUMP / CREF)
#: carries what the tail was standing in for, and the full logs are written
#: into the tree the agent can read (AGENT_LOG_DIR).  Set this above zero
#: only to debug the harness itself.
COMMIT_LOG_TAIL_LINES = int(os.environ.get("TITAN_COMMIT_LOG_TAIL_LINES", "0"))

#: Ceiling on any single quoted blob.  Kept for the paths that still quote
#: raw output; the build-failure path no longer uses it (see below).
MAX_OUTPUT_CHARS = int(os.environ.get("TITAN_MAX_OUTPUT_CHARS", "100000"))

#: A failed elaboration reaches the agent as *extracted error lines*, not as
#: stderr.  100,000 characters of sbt chatter is not evidence, it is a haystack
#: with the needle already in it: what identifies a Chisel failure is the
#: `[error]` lines and the first few stack frames under them.
BUILD_ERROR_MAX_CHARS = int(os.environ.get("TITAN_BUILD_ERROR_MAX_CHARS",
                                           "4000"))
BUILD_ERROR_MAX_FRAMES = int(os.environ.get("TITAN_BUILD_ERROR_FRAMES", "15"))

#: Total budget for the evidence block.  MAX_EVIDENCE_TESTS caps how many
#: failures may carry evidence; this caps how much they may carry between
#: them, because a 16x16 tile at SEW=8 is ~70 characters a row and six full
#: dumps would be 7kB of prompt on their own.  Tests are added whole until the
#: budget is spent; the rest are a verdict line plus a pointer to the log the
#: agent can open itself.
MAX_EVIDENCE_CHARS = int(os.environ.get("TITAN_MAX_EVIDENCE_CHARS", "4000"))

#: How much of the agent's own notes file is quoted back to it. Its notes are
#: its memory now that sessions are fresh, so this is generous -- but it is
#: the one part of the message the agent controls the size of, and an
#: unbounded quote would undo the shrinking everything else just paid for.
#: It can always read the whole file with ``read_knowledge``.
NOTES_MAX_CHARS = int(os.environ.get("TITAN_NOTES_MAX_CHARS", "8000"))

# --- artifacts the agent can read for itself -------------------------------
#: Every iteration the loop copies its artifacts (build stdout tail, all 27
#: per-test simulator logs *in full*, the status file, the directed summary)
#: into the chipyard tree, on the chipyard node, where the agent's BashTool
#: can open them.  Until r5 they were written to TITAN_LOG_ROOT on the head,
#: which the agent's container cannot see -- so the loop had to paste, and
#: pasting is what made an iteration cost $23.
#:
#: Inside the git tree deliberately: it is the directory the agent already
#: works in, so `ls titan_logs/` needs no explanation.  ``write_files`` adds
#: the directory to ``.git/info/exclude`` on first write, which keeps it out
#: of ``collect_diff`` (that does ``git add -N .``, so an untracked tree WOULD
#: otherwise land in every probe diff) and safe from ``reset_chipyard``
#: (``git clean -fd``, without ``-x``, leaves ignored paths alone).
AGENT_LOG_REL = os.environ.get("TITAN_AGENT_LOG_REL", "titan_logs")
AGENT_LOG_DIR = os.environ.get("TITAN_AGENT_LOG_DIR",
                               os.path.join(CHIPYARD_PATH, AGENT_LOG_REL))

#: How many runs per iteration the agent may start (waits are free and
#: uncounted).  ONE pool, shared by ``run_directed_start`` and
#: ``run_rvv_start``: both take the single ``chipyard`` node for an
#: elaboration, so a separate budget for each would just be a way of
#: spending twice as much cluster.  Four was enough for change -> test ->
#: fix -> confirm on the directed suite alone; with S2 debugging in the same
#: pool it is six, because the elaboration is ~2.5 minutes either way and an
#: agent that has to choose between confirming S1 and diagnosing S2 will do
#: neither.
AGENT_RUNS_PER_ITER = int(os.environ.get("TITAN_AGENT_RUNS_PER_ITER", "6"))

#: Most riscv-vector-tests one ``run_rvv_start`` call may name explicitly.
#: A cosim run is ~15s per test, so 40 is ~10 minutes on top of the build --
#: about the most that fits in a turn without the model losing the thread.
RVV_AGENT_MAX_TESTS = int(os.environ.get("TITAN_RVV_AGENT_MAX_TESTS", "40"))

#: How many failing RVV test names ``read_status`` prints.  Same cap: the
#: point of the list is that the agent knows what "failing" selects, and a
#: 400-name paste does not tell it that any better than 40 plus a count.
RVV_STATUS_NAMES = int(os.environ.get("TITAN_RVV_STATUS_NAMES", "40"))

#: The longest a single ``run_directed_wait`` MCP call may block.  The
#: ``claude`` CLI moves any MCP tool call still running after 120s into a
#: background task and *returns* to the model; in ``-p`` mode the model then
#: has nothing to do, ends its turn, and the session dies with the job still
#: running (r8 iteration 3: an instrumented tree was graded, 26/27 -> 27
#: mismatch).  So no call this tool serves is allowed to approach 120s: the
#: build runs in a worker thread and the model polls for it.
RUN_DIRECTED_WAIT_CAP = int(os.environ.get("TITAN_RUN_DIRECTED_WAIT_CAP",
                                           "100"))

#: How long the driver waits, after the LLM turn returns, for a job the agent
#: abandoned.  It must be waited out rather than killed: the build holds the
#: placement group's single ``chipyard`` resource, and the loop's own build
#: dispatched alongside it would double-book that node.
RUN_DIRECTED_DRAIN_TIMEOUT_S = int(
    os.environ.get("TITAN_RUN_DIRECTED_DRAIN_TIMEOUT_S", "1800"))

#: How far below the best official result this run has seen a new result has
#: to fall before the next iteration's orientation opens with a REGRESSION
#: block.  Three tests: enough to be a broken change rather than noise.
REGRESSION_ALERT_DELTA = int(os.environ.get("TITAN_REGRESSION_DELTA", "3"))

#: Belt-and-braces against the same hazard, from the other end.  The CLI
#: reads ``CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`` (clamped to [0, 2^31-1]) as
#: the threshold before an MCP call is moved to the background; 0 disables
#: backgrounding altogether.  Passed to the ``claude`` subprocess through the
#: LLM task's ``runtime_env``, so it needs no cluster restart.  Empty string
#: (``TITAN_CLI_BACKGROUND_MS=``) turns the override off.
CLI_MCP_AUTO_BACKGROUND_MS = os.environ.get("TITAN_CLI_BACKGROUND_MS",
                                            "1800000")

#: Hard cap on assistant turns within one LLM call.  Zero = do not pass the
#: flag.  Zero is the default because the ``claude`` CLI installed on this
#: cluster has no ``--max-turns`` (it has ``--max-budget-usd`` instead), and
#: passing an unknown flag fails the call outright.  The plumbing is in
#: ``ClaudeCodeLLM(max_turns=...)`` for whenever the CLI grows it back.
MAX_TURNS = int(os.environ.get("TITAN_MAX_TURNS", "0"))

#: Built-in CLI tools the agent may not use.  ``Task`` spawns sub-agents: in
#: r5 the agent used it freely, which put edits and token spend outside every
#: accounting the loop keeps, and gave the sub-agents none of the context the
#: system prompt carries.
DISALLOWED_TOOLS = [t for t in os.environ.get("TITAN_DISALLOWED_TOOLS",
                                              "Task").split(",") if t]

#: How much evidence a failing directed program prints, and how much of it
#: reaches the agent.  Both sides read these: ime_tests bakes the caps into
#: the emitted assembly, helpers.format_directed_failure quotes at most the
#: same amount back.  A failing tile prints MAX_DIFF_LINES differing elements
#: and MAX_DUMP_ROWS rows of each C buffer; the feedback message reproduces
#: that in full for the first MAX_EVIDENCE_TESTS failures and gives the rest
#: a verdict line only, because twenty-seven full tile dumps in one prompt
#: bury the pattern they were meant to expose.
MAX_DIFF_LINES = int(os.environ.get("TITAN_MAX_DIFF_LINES", "8"))
MAX_DUMP_ROWS = int(os.environ.get("TITAN_MAX_DUMP_ROWS", "8"))
MAX_EVIDENCE_TESTS = int(os.environ.get("TITAN_MAX_EVIDENCE_TESTS", "6"))

#: Tail kept per test in the .simlogs.txt dump.  Was 60, which predates the
#: evidence block: 8 diff lines plus 2 x 8 dump rows plus the verdict is ~26
#: lines of the tail on its own, and anything the simulator says after the
#: program returns would have pushed the verdict out of a 60-line window.
SIMLOG_TAIL_LINES = int(os.environ.get("TITAN_SIMLOG_TAIL_LINES", "200"))

# --- synthesis -------------------------------------------------------------
SKY130_COL_PATH = os.environ.get("TITAN_SKY130_COL_PATH", "#FILL")
CACTI_PATH = os.environ.get("TITAN_CACTI_PATH", "#FILL")
SYNTH_OBJ_ROOT = os.environ.get(
    "TITAN_SYNTH_OBJ_ROOT", os.path.join(tempfile.gettempdir(), "titan-synth"))
SYNTH_TIMEOUT_S = int(os.environ.get("TITAN_SYNTH_TIMEOUT_S", "86400"))

#: What Hammer synthesises.  riscv_extensions uses BoomTile; the equivalent
#: unit of interest here is the Shuttle tile that contains the vector unit.
SYNTH_VLSI_TOP = os.environ.get("TITAN_SYNTH_VLSI_TOP", "ShuttleTile")


def synth_runtime_env() -> dict:
    """RUNTIME_ENV extended with the sky130_vlsi package.

    ``synth_node`` imports ``sky130_vlsi``, which lives in a sibling example
    rather than in ``chia``, so the flat ``working_dir`` packaging does not
    reach it -- a synthesis dispatch would fail on the worker with an
    ImportError and nothing else.  Only the Stage 3 PPA path needs this, so
    it is a separate entry point rather than a permanent widening of
    RUNTIME_ENV.

    ``examples/common`` is deliberately absent: it is BOOM-only
    (``build_megaboom``, ``boom_tile_syn``), and shipping it would invite
    someone to import from it.
    """
    examples = _REPO_ROOT / "examples"
    env = dict(RUNTIME_ENV)
    env["py_modules"] = list(RUNTIME_ENV["py_modules"]) + [
        str(examples / "sky130_vlsi"),
    ]
    return env

# --- database --------------------------------------------------------------
DB_ROOT_ENV = "TITAN_DB_ROOT"


def _default_db_root() -> str:
    """First writable candidate, resolved on whichever node asks.

    riscv_extensions hardcodes ``/scratch/vext-db``; on this cluster
    ``/scratch`` does not exist at all, and the failure surfaces only at the
    very end of a long build, as a PermissionError from ``makedirs``.  Each
    node resolves this itself, so a worker with different storage still lands
    somewhere it can write.  Set ``TITAN_DB_ROOT`` to override -- and do set
    it if the database node's storage is not shared with the head, because
    then "writable" is not the same as "the right place".
    """
    import getpass
    candidates = [
        "/scratch",
        f"/share1/saves/{getpass.getuser().removesuffix('_l')}",
        os.path.expanduser("~"),
    ]
    for base in candidates:
        if os.path.isdir(base) and os.access(base, os.W_OK):
            return os.path.join(base, "titan-db")
    return os.path.join(os.path.expanduser("~"), "titan-db")


DB_ROOT_DEFAULT = _default_db_root()
