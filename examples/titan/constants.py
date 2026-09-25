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
#: riscv-isa-sim because the Stage 0 agent edits Spike.
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
#: 8 stays, measured (``titan_runs/nondet/``, 2026-09-18).  Median per cosim
#: test: 47-66 s at 8 threads vs 115-174 s at 1 thread (2.3-2.6x slower), and
#: 1 thread is also *flakier* -- 9/32 runs failed there against 8/64 at 8
#: threads, with the failure mode changing to the DebugROB ``popTrace`` "PC
#: mismatch spike 10004 != DUT 10000" at bootrom instruction 2.  Thread count
#: is not the source of the run-to-run flips; one thread is worse on both
#: axes.
VERILATOR_THREADS = int(os.environ.get("TITAN_VERILATOR_THREADS", "8"))

#: Pins every ``RANDOMIZE_*`` init word to constant zero.  firrtl2 emits the
#: ``RANDOM`` macro ``ifndef``-guarded, so this command-line define wins and
#: ``$random`` never fires: simulator randomisation is *inert*, and two builds
#: of one tree are byte-identical (``titan_runs/nondet/``).  Register-init
#: randomisation therefore cannot make a logically inert RTL change flip a
#: test -- that story (``titan_runs/dummy_fu/``) is disproved.
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

#: Round four opens the floating-point half of the family with exactly one
#: instruction:
#:
#:   vfmmacc.vv (W=1) at SEW = 32 and SEW = 64 -- binary32 x binary32 ->
#:   binary32 and binary64 x binary64 -> binary64.
#:
#: Why one and not seven.  A floating-point result is *not* determined by
#: the instruction alone: spec 1536-1600 partitions the K_eff products into
#: groups of an implementation-defined size G, reduces each group to a
#: partial sum under an implementation-defined psm, and rounds that partial
#: sum under an implementation-defined rnd, and an implementation must
#: disclose its (SEW, W, LAMBDA) -> (G, psm, rnd) table (spec 1652-1656).
#: Titan discloses G=1, psm=0, rnd=frm, which spec 1771 names as the tuple
#: that makes a Zvvm implementation match the analogous Zvtm instruction
#: "For input element widths of 32 bits or greater".  At W=1 that collapses
#: to one scalar multiply and one scalar add per k, in increasing k --
#: which is the only arrangement the *test program* can recompute on the DUT
#: with baseline rv64imafd instructions, so the harness keeps its
#: differential shape and its exact bit-for-bit comparison.
#:
#: The other six are deferred, each for a concrete missing capability:
#:
#:   vfwmmacc.vv / vfqmmacc.vv / vf8wmmacc.vv -- at W>1 a group is a
#:     sub-dot-product of W products that psm=0 requires to be summed
#:     *exactly* before a single rounding (spec 1546-1553, 1566).  No
#:     sequence of rv64imafd operations reproduces round_frm(C + p0 + p1)
#:     with one rounding, so the reference would have to become a
#:     precomputed image and the programs would stop being differential.
#:     They additionally need the vtype.altfmt_A / altfmt_B input-format
#:     decode at every SEW below 64.
#:   vfmmacc.vv at SEW 8 and 16 -- OFP8 (E4M3/E5M2), binary16 and bfloat16
#:     inputs, selected by altfmt_A / altfmt_B (spec 1129-1142).  The
#:     baseline -march has no scalar arithmetic at any of those widths, so
#:     the on-DUT reference path cannot be written.
#:   vfwimmacc.vv / vfqimmacc.vv / vf8wimmacc.vv -- the vm=0 microscaled
#:     integer-input forms.  They need the E8M0 paired block scales in v0
#:     (spec 2116-2180), the bs block-size field, the NaN-scale early-exit
#:     rule (spec 1812-1826) and a v0 that no VectorAlloc currently
#:     reserves.  Their arithmetic is fully deterministic (spec 1643-1648),
#:     so they are the natural round five.
#:
#: See titan_runs/round4_design.md for the full survey and the citations.
ROUND_FOUR_INSNS = ("vfmmacc.vv",)

#: Round six opens the microscaled half of the family with the three
#: integer-input, floating-point-accumulate forms:
#:
#:   vfwimmacc.vv  (W=2)  -- MXINT8 -> binary16 / bfloat16 at SEW=16
#:   vfqimmacc.vv  (W=4)  -- MXINT4 -> binary16 / bfloat16 at SEW=16,
#:                           MXINT8 -> binary32 at SEW=32
#:   vf8wimmacc.vv (W=8)  -- MXINT4 -> binary32 at SEW=32,
#:                           MXINT8 -> binary64 at SEW=64
#:
#: Why these three before the floating-point widening forms, when
#: round4_design.md called the latter "the natural round five".
#:
#: 1. They are the only members of the family whose arithmetic is *fully
#:    determined by the instruction*.  Sail ``int_scaled_gemm`` (5373-5410)
#:    never calls get_fp_grouping / get_fp_psm / get_fp_rnd and has no G
#:    legality check; spec 1283-1287 and 1645-1652 state it in prose.  So
#:    round six needs no implementation disclosure, no psm=1 SAIL fragment,
#:    and none of the "define every g_len in 1..G" obligation that spec
#:    1667-1671 attaches to a disclosed reduction.
#: 2. With C = +0.0, one block and both E8M0 scales at 2**0, the
#:    architectural result is literally ``int_to_fp(dot)`` -- an exact
#:    integer dot product and one ``fcvt``.  That keeps the directed
#:    programs differential and bit-exact *even where the accumulator is
#:    binary16 or bfloat16*, which baseline rv64imafd cannot round to.  It is
#:    the reason the narrow accumulator is not a blocker here and is one in
#:    round eight.
#: 3. This is where the microscaling front-end gets built and judged: the
#:    paired E8M0 scales folded into v0 at row stride R = LAMBDA*SEW/pw
#:    (spec 2129-2170), the vtype.bs block-size field (1160-1176), the
#:    legality rules SEW*LAMBDA >= 16 and, at bs=1, W*LMUL <= SEW (Sail
#:    5151-5158), and the NaN-scale early exit (1959-1975, Sail 5390-5392).
#:    Round eight's floating-point vm=0 path reuses all of it, so building it
#:    against a deterministic arithmetic is strictly cheaper than building it
#:    against a disclosed one.
#:
#: Note what these three are *not*: separate encodings.  funct6 0x39/0x3a/0x3b
#: on OPIVV are vwmmacc.vv / vqmmacc.vv / v8wmmacc.vv at vm=1 and these three
#: at vm=0 (Sail 5963-5975, 6196-6205, 6082-6091).  Round six therefore puts
#: rounds one to three's integer MACs back on the regression surface with a
#: new reason to fail: a decoder that mis-routes vm.
#:
#: Deliberately not in round six: the whole Zvvfmm floating-point widening
#: group.  See titan_runs/round6_design.md for the split, and note that
#: round4_design.md's claim that W>1 is unreachable has been retracted --
#: a disclosed psm=1 sequential reduction reaches it.  That is round seven.
#:
#: RETRACTED IN ROUND SEVEN -- the clause above claiming that round seven
#: needs "a disclosed psm=1 sequential reduction" is wrong.  It is left in
#: place rather than deleted so that the retraction is legible to whoever
#: reads the round-six note next; what follows is the evidence, by line, so
#: that a future reader can tell which version to believe without re-deriving
#: the argument.
#:
#: 1. Spec 1565 defines psm=0 as "the partial sum `S` is formed using exact
#:    computation: the contributing products and sums are computed in
#:    sufficiently precise internal form, without rounding to the C
#:    accumulator format, until the next rounding step".  Nothing in that
#:    sentence is conditioned on W.  A group of G=1 sub-dot-products holds
#:    G*W = W products (spec 1558-1562), and psm=0 sums all W of them
#:    exactly.
#: 2. Spec 1596-1599: "For a given (`SEW`, `W`, `lambda`) entry, the
#:    disclosed (`G`, `psm`, `rnd`) tuple applies uniformly to every legal
#:    floating-point input-format combination, C accumulator format, and
#:    unscaled or microscaled operation having that geometry."  So the one
#:    tuple Titan already discloses covers every round-seven cell; there is
#:    no per-format obligation to disclose a second one.
#: 3. Spec 1656-1666 attaches the SAIL-fragment obligation -- and the
#:    "define behavior for every actual group length" obligation -- to
#:    "each entry with `psm=1`".  At psm=0 neither applies.
#: 4. The update is therefore fully determined by the architecture alone:
#:    with rnd=frm (spec 1584), C <- round_frm(C + round_frm(exact sum of
#:    the W products)), per spec 1591-1593.  That is computable exactly in
#:    the reference model with ``Fraction``, so no implementation-specific
#:    fragment is needed to judge it.
#:
#: What round4_design.md actually got right is narrower than its own claim:
#: W>1 is unreachable *by a differential program built from baseline
#: rv64imafd*, because an exact W-product partial sum is not a scalar
#: fmul/fadd pair.  That is a statement about how the test program checks
#: itself, not about what the architecture determines, and round seven
#: answers it with embedded golden bytes rather than with a psm choice.
#:
#: And note what this is not: keeping psm=0 is not a coverage reduction.
#: `psm` is a parameter of the *implementation disclosure* (spec 1594-1601),
#: not a test dimension -- we disclose psm=0 and judge against psm=0, which
#: is a complete architectural description.  Disclosing psm=1 instead would
#: not test more; it would oblige us to publish a SAIL fragment defining a
#: reduction we do not implement.  This is a different kind of act from
#: skipping a geometry or loosening a tolerance, and should not be read as
#: one.
ROUND_SIX_INSNS = ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv")

#: Round seven closes the family with the three widening floating-point
#: multiply-accumulates -- the last three of the spec's fifteen:
#:
#:   vfwmmacc.vv   (W=2)  -- OFP4->OFP8, OFP8->FP16/BF16, FP16/BF16->FP32,
#:                           FP32->FP64
#:   vfqmmacc.vv   (W=4)  -- OFP4->FP16/BF16, OFP8->FP32, FP16/BF16->FP64
#:   vf8wmmacc.vv  (W=8)  -- OFP4->FP32, OFP8->FP64
#:
#: The legal cells are a *table*, not a rule: tbl-fp-encoding-map at spec
#: 7288-7370, with the row-selection rules at 7270-7286.  rvv_ref.FP_CELLS
#: transcribes it and is the single place this harness decides whether an
#: encoding is architecture or is reserved.
#:
#: Three things make this round unlike round six, in increasing cost:
#:
#: 1. A and B are *floating-point* elements, not two's-complement integers.
#:    Round six's inputs were MXINT4/MXINT8, whose semantics spec 2334-2342
#:    states in full; round seven's are OFP8 (E4M3/E5M2) and OFP4 (E2M1),
#:    whose bit-level semantics the IME spec does not restate.  Spec
#:    1908-1913 cites OCP normatively instead, and Sail 4831-4838 leaves
#:    ``fp_is_NaN`` / ``fp_defaultNaN`` / ``fp_zero`` as undefined helpers
#:    deferring to that definition.  See rvv_ref.OCP_PENDING_FORMATS: those
#:    rows are declared and deliberately left unpopulated until the OCP
#:    documents are on disk.
#: 2. OFP8 appears as an *accumulator* format for the first time
#:    (vfwmmacc.vv at SEW=8, spec 1098-1099), so the round-four assumption
#:    that a C format is an IEEE binary{32,64} chosen by width no longer
#:    holds anywhere in this round.
#: 3. Mixed input formats are new surface.  At EEW=4, altfmt_A=1 is
#:    reserved (spec 1442-1443), so OFP4 never mixes; at EEW=8 and EEW=16
#:    altfmt_A and altfmt_B are independent and all four combinations are
#:    legal (spec 1446-1456, 1458-1476).  Mixed cells get their own test
#:    tier rather than sharing one with the same-format cells: if they
#:    shared, a failure could not be attributed between the mixing logic and
#:    the per-format decode.
#: Note which two, and why it is not three.  ``vf8wmmacc.vv`` (W=8) is
#: absent: every one of its encoding-map cells takes an OFP input format --
#: (W=8, SEW=32) is OFP4 -> binary32 and (W=8, SEW=64) is OFP8 -> binary64
#: (spec 7362-7370) -- so with the OFP extensions declared unsupported it has
#: no implementable cell at all.  It is not a partial implementation that was
#: skipped; the instruction is entirely OFP-dependent and belongs to whatever
#: round obtains the OCP documents.
#:
#: This was not obvious from the mnemonic list and was found by deriving the
#: set rather than writing it down: rvv_ref.fpw_resolved_cells() reports
#: which cells have defined formats, and ime_stress's
#: check_pool_covers_every_instruction refused a hand-written third mnemonic
#: that no geometry could back.  The derivation is kept for that reason.
ROUND_SEVEN_INSNS = ("vfwmmacc.vv", "vfqmmacc.vv")

#: The declared implementation scope for the Zvvm floating-point family.
#:
#: Titan implements the five widening floating-point extensions whose element
#: formats the IME specification defines on its own, and declares the OFP
#: extensions unsupported.  This is a scope decision, and it is a legal one:
#:
#:   * Spec 810-844 splits the Zvvm family into independent extensions, one
#:     per (input type, accumulator type) pair, each separately named and
#:     separately required.  Spec 805-806 notes that the microscaling-related
#:     extensions are listed elsewhere again.
#:   * No clause in the specification requires an implementation to support
#:     any OFP extension.  The only "shall support at least one" requirement
#:     in the document is about the LAMBDA configuration domain, not about
#:     element formats.
#:
#: So an implementation that provides the five below and none of the
#: Zvvofp* extensions is a conforming subset, and a program that uses an
#: unimplemented one takes an illegal-instruction exception, which is the
#: architecturally defined outcome.
#:
#: Supported:
#:
#:   Zvvfp16fp32mm, Zvvbf16fp32mm   vfwmmacc.vv  W=2, SEW=32
#:                                  binary16 / bfloat16 -> binary32
#:   Zvvfp32fp64mm                  vfwmmacc.vv  W=2, SEW=64
#:                                  binary32 -> binary64
#:   Zvvfp16fp64mm, Zvvbf16fp64mm   vfqmmacc.vv  W=4, SEW=64
#:                                  binary16 / bfloat16 -> binary64
#:
#: Mixed-format operation (altfmt_A != altfmt_B) is supported wherever the
#: encoding map allows it for these cells, which per spec 1480-1490 requires
#: both of the paired extensions to be present.
#:
#: Not supported, and why:
#:
#:   Zvvofp4ofp8mm, Zvvofp8fp16mm, Zvvofp8bf16mm, Zvvofp4fp16mm,
#:   Zvvofp4bf16mm, Zvvofp8fp32mm, Zvvofp4fp32mm, Zvvofp8fp64mm,
#:   and every Zvvx*/Zvvxn* microscaled counterpart.
#:
#: These use the OFP8 (E4M3, E5M2) and OFP4 (E2M1) element formats.  The IME
#: specification does not define those formats: it cites the OCP
#: Microscaling Formats (MX) v1.0 specification normatively at lines
#: 1908-1913 and leaves the corresponding SAIL helpers (fp_is_NaN,
#: fp_defaultNaN, fp_zero, at 4831-4838) undefined, deferring to that
#: document.  Spec 1400-1410 states only the significand widths.
#:
#: Those documents are not available to this project.  We therefore decline
#: to implement the OFP extensions rather than reconstruct their semantics
#: from secondary sources or by analogy with IEEE 754.  The reconstruction
#: would not be sound: E4M3 encodes no infinity, so its overflow behaviour
#: cannot be inferred from an IEEE rounding path, and its NaN set is not the
#: IEEE "exponent all ones with nonzero significand" predicate.  Published
#: secondary descriptions also disagree with each other in ways that are
#: invisible without the specification to arbitrate -- one widely used
#: reference implementation reports E4M3 as having five significand bits
#: because of an internal convention -- so adopting one would be choosing a
#: definition rather than implementing a standard.
#:
#: An incorrectly reconstructed format would not fail loudly.  It would
#: produce a judge that is confidently wrong, and every disagreement with the
#: hardware would be attributed to the hardware.  Declaring the extensions
#: unsupported is the honest outcome of not having the specification; it is
#: not a reduction in the coverage of what we do claim to implement.
#:
#: The machinery to judge the OFP cells is nonetheless built and tested:
#: rvv_ref.FP_FORMAT_TABLE carries their rows unpopulated,
#: rvv_ref.OCP_PENDING_FORMATS names what each needs, the reference GEMM and
#: operand packer are format-agnostic and the sub-byte path is exercised
#: against a synthetic descriptor.  If the documents are obtained, the work
#: is to fill in three table rows.
ROUND_SEVEN_SUPPORTED = (
    "Zvvfp16fp32mm", "Zvvbf16fp32mm",
    "Zvvfp32fp64mm",
    "Zvvfp16fp64mm", "Zvvbf16fp64mm",
)

#: The extensions this implementation declares it does not provide.  Kept as
#: data so that a test can assert no program is generated for them.
ROUND_SEVEN_UNSUPPORTED = (
    "Zvvofp4ofp8mm", "Zvvofp8fp16mm", "Zvvofp8bf16mm", "Zvvofp4fp16mm",
    "Zvvofp4bf16mm", "Zvvofp8fp32mm", "Zvvofp4fp32mm", "Zvvofp8fp64mm",
)

#: Round eight: the OFP8 (E4M3 / E5M2) input cells of the widening family.
#:
#: What changed: the OCP 8-bit Floating Point Specification (OFP8) Revision
#: 1.0 is now on disk (titan/ocp-spec/*.pdf; text for the agents at
#: specs/ime/ocp-ofp8-v1.0.txt, gitignored like the adoc).  It defines E4M3
#: and E5M2 completely (OFP8 p.11-15); rvv_ref.FP_FORMAT_TABLE now carries
#: both rows, with every OCP citation and Titan's disclosures for what OCP /
#: the IME adoc leave open in rvv_ref.OFP8_DISCLOSURE.  OCP Microscaling
#: Formats (MX) v1.0, which defines E2M1 (OFP4), is still NOT on disk.
#:
#: Which cells that makes live is derived, not listed: a cell is live when
#: every format in every one of its encoding-map rows (spec 7288-7370) is
#: resolved -- rvv_ref.fpw_resolved_cells().  From the adoc's table:
#:
#:   (W=2, SEW=16)  vfwmmacc.vv   E4M3/E5M2 x E4M3/E5M2 -> binary16 / bfloat16
#:                                Zvvofp8fp16mm, Zvvofp8bf16mm    LIVE
#:   (W=4, SEW=32)  vfqmmacc.vv   E4M3/E5M2 x E4M3/E5M2 -> binary32
#:                                Zvvofp8fp32mm                   LIVE
#:   (W=8, SEW=64)  vf8wmmacc.vv  E4M3/E5M2 x E4M3/E5M2 -> binary64
#:                                Zvvofp8fp64mm                   LIVE
#:   (W=2, SEW=8)   vfwmmacc.vv   E2M1 x E2M1 -> E4M3 / E5M2
#:                                Zvvofp4ofp8mm        needs E2M1 (MX)
#:   (W=4, SEW=16)  vfqmmacc.vv   E2M1 x E2M1 -> binary16 / bfloat16
#:                                Zvvofp4fp16mm, Zvvofp4bf16mm   needs E2M1
#:   (W=8, SEW=32)  vf8wmmacc.vv  E2M1 x E2M1 -> binary32
#:                                Zvvofp4fp32mm        needs E2M1 (MX)
#:
#: So the live cells take OFP8 as INPUTS only; every accumulator is IEEE.
#: No round-eight program rounds to OFP8.  The adoc's two OFP8-accumulator
#: extensions are Zvvofp4ofp8mm (needs E2M1) and Zvvofp8mm -- vfmmacc.vv at
#: SEW=8, W=1 -- which is outside the widening family and sits with
#: Zvvfp16mm / Zvvbf16mm, the W=1 narrow cells round four deferred (no
#: baseline scalar arithmetic at 8/16 bits) and no round has built since.
#: Both stay unsupported.  The OFP8 rounding rule (RNE; overflow
#: non-saturating, disclosed) is nevertheless encoded and negative-controlled
#: in rvv_ref so that whichever round builds them inherits it.
#:
#: All four altfmt_A / altfmt_B combinations, mixed E4M3 x E5M2 included,
#: are covered by the one OFP8 extension per output format (spec 1445-1451),
#: so every live cell is judged on its mixed rows too.
#:
#: vf8wmmacc.vv enters ALL_INSNS here: (W=8, SEW=64) is its first cell with
#: a generator behind it, so ime_stress's check_pool_covers_every_instruction
#: can be satisfied honestly.  vfwmmacc.vv / vfqmmacc.vv were already there;
#: round eight adds cells to them, not mnemonics.
#:
#: Still unsupported, and why:
#:   Zvvofp4ofp8mm, Zvvofp4fp16mm, Zvvofp4bf16mm, Zvvofp4fp32mm
#:       E2M1 is defined by OCP MX v1.0, which is not on disk; rvv_ref keeps
#:       raising OCPSpecUnavailable for it and these cells stay out of every
#:       pool.  A program that uses one takes an illegal-instruction trap.
#:   Zvvofp8mm
#:       vfmmacc.vv at SEW=8 (W=1, OFP8 accumulator); see above.
#:   Zvvxofp8* / Zvvxnofp8* (and all MX FP counterparts)
#:       vm=0 microscaled FP operation is not built for any cell yet.
ROUND_EIGHT_INSNS = ("vf8wmmacc.vv",)

ROUND_EIGHT_SUPPORTED = (
    "Zvvofp8fp16mm", "Zvvofp8bf16mm",
    "Zvvofp8fp32mm",
    "Zvvofp8fp64mm",
)

ROUND_EIGHT_UNSUPPORTED = (
    "Zvvofp4ofp8mm", "Zvvofp4fp16mm", "Zvvofp4bf16mm", "Zvvofp4fp32mm",
    "Zvvofp8mm",
)

#: Every instruction the generators know how to emit and judge.  The order is
#: round order, so an index into this is an implementation milestone.
#:
#: ROUND_SEVEN_INSNS joined this tuple when its generators landed.  It was
#: deliberately held out while they did not exist: ime_stress's
#: ``check_pool_covers_every_instruction`` reads this tuple as "geometries
#: exist for these", so listing a mnemonic with no pool behind it would have
#: made a coverage check pass vacuously -- a judge lying about its own
#: reach.  The three mnemonics are here now because fpw_directed_tiers emits
#: programs across the five supported extensions; the OFP cells are
#: absent from that pool by construction (rvv_ref.fpw_resolved_cells derives
#: it from which formats are defined), not by omission.
ALL_INSNS = (ROUND_ONE_INSNS + ROUND_TWO_INSNS + ROUND_THREE_INSNS
             + ROUND_FOUR_INSNS + ROUND_SIX_INSNS + ROUND_SEVEN_INSNS
             + ROUND_EIGHT_INSNS)

#: Round five adds no instruction.  It adds a *check* over instructions that
#: are already in ALL_INSNS, so it is named as a tier token rather than as a
#: mnemonic -- and deliberately kept out of ALL_INSNS, which ime_stress's
#: ``check_pool_covers_every_instruction`` reads as "every mnemonic a
#: geometry can select" and helpers.instruction_scope reads as "what the
#: model agent has to implement".
#:
#: ``clayout`` is the C-tile register-layout tier.  Every pre-round-five
#: directed program writes the C tile with ``vmts.v`` and reads it back with
#: ``vmts.v``, so a register-side C index that is wrong *in the same way* on
#: the transfer and on the multiply-accumulate produces a correct memory
#: image and passes.  That is not hypothetical: Titan's RTL indexes C with a
#: linear ``i*M + j`` where the spec routes ``i*N_max + j`` through
#: ``tile_reg_idx``, and at LAMBDA=1 those two differ by exactly a transpose.
#: The clayout tier reads the C register group back with an ordinary
#: architectural ``vse<SEW>.v`` instead, so the register-side layout is
#: observed rather than cancelled.  See titan_runs/round5_design.md.
ROUND_FIVE_TIERS = ("clayout",)


#: The full selectable scope: every instruction, plus every check tier.
#: ``TITAN_INSNS`` validates against this, not against ALL_INSNS.
ALL_SCOPE = ALL_INSNS + ROUND_FIVE_TIERS

#: What a seeded tree already implements, and what this round adds.
#: ``helpers.instruction_scope`` reads these two names first and only falls
#: back to (ROUND_ONE_INSNS, ROUND_TWO_INSNS) when they are absent -- the
#: hook it left open so that a new round needs no edit in helpers.py.  Set
#: them and every prompt names the right halves: rounds one and two are the
#: regression surface, round three is the work.
IMPLEMENTED_INSNS = (ROUND_ONE_INSNS + ROUND_TWO_INSNS + ROUND_THREE_INSNS
                     + ROUND_FOUR_INSNS + ROUND_SIX_INSNS + ROUND_SEVEN_INSNS)
# Round 8 (2026-09-24) adds OFP8 (E4M3/E5M2) input *cells*: the new mnemonic
# vf8wmmacc plus new cells of vfwmmacc/vfqmmacc, whose IEEE cells are round 7
# and must keep passing.  Hence the overlap with IMPLEMENTED_INSNS.
NEW_INSNS = ROUND_EIGHT_INSNS + ROUND_SEVEN_INSNS


def _scope_insns() -> tuple:
    """Which instructions this run's directed suite and prompts cover.

    ``TITAN_INSNS`` selects the scope.  It accepts a round name -- ``one``,
    ``two``, ``three``, ``four``, ``five``, ``all`` -- or an explicit
    comma-separated list of mnemonics and/or check tiers, and defaults to
    ``all`` (rounds one to five).  Unknown names raise here rather than
    silently generating an empty suite six minutes into a loop iteration.

    Round five's token is not a mnemonic: ``clayout`` names a *tier* over
    instructions that earlier rounds already cover (see
    :data:`ROUND_FIVE_TIERS`).  It is selected and excluded exactly like the
    others -- ``TITAN_INSNS=five`` runs the C-layout tier alone,
    ``TITAN_INSNS=one`` excludes it -- which is the whole reason it goes
    through this knob rather than through a second environment variable.

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
             "four": ROUND_FOUR_INSNS, "4": ROUND_FOUR_INSNS,
             "five": ROUND_FIVE_TIERS, "5": ROUND_FIVE_TIERS,
             "six": ROUND_SIX_INSNS, "6": ROUND_SIX_INSNS,
             "seven": ROUND_SEVEN_INSNS, "7": ROUND_SEVEN_INSNS,
             "eight": ROUND_EIGHT_INSNS, "8": ROUND_EIGHT_INSNS,
             "all": ALL_SCOPE, "": ALL_SCOPE}
    if raw.lower() in named:
        return named[raw.lower()]
    chosen = tuple(tok.strip() for tok in raw.split(",") if tok.strip())
    unknown = [name for name in chosen if name not in ALL_SCOPE]
    if unknown:
        raise ValueError(
            f"TITAN_INSNS={raw!r}: unknown mnemonic(s) {unknown}; "
            f"expected a round name (one/two/three/four/five/six/all) or a "
            f"subset of {ALL_SCOPE}")
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
    "TITAN_LOG_ROOT", str(OUT_DIR / "logs"))
MAX_ITERS = int(os.environ.get("TITAN_MAX_ITERS", "60"))
#: Stage 0 is a transcription task against a formal semantics, not a
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
#: How many times a *failing* regression test is re-run before the loop calls
#: it a real failure.  1 = the old single-run verdict.
#:
#: Why this exists (r6, measured, not guessed).  The cospike/DebugROB DPI
#: trace bridge is nondeterministic AND load-dependent: Shuttle retires two
#: instructions a cycle and the bridge does not order them the way Spike
#: does.  On a byte-identical binary the full 837-test suite failed
#: 281/248/249 tests on three consecutive runs -- ~30% -- with an 82%
#: reshuffle of *which* tests failed, and the 24 tests that failed all three
#: times are exactly the 837*0.31^3 = 24.9 a per-run coin flip predicts.  A
#: single-run S2 verdict is therefore not "unlikely to be clean", it is
#: unreachable -- and before this the gate ran only when S2 came back clean,
#: so in r6 the gate never ran at all.
#:
#: The rule is unanimity with early exit: only the tests still failing are
#: re-run, and a test that passes ANY rep is dropped as a bridge flip.  Under
#: the measured p=0.31 per-run flip rate a pure flake survives n reps with
#: probability 0.31^(n-1), so 7 leaves 0.0009 per test -- about 0.2 expected
#: false failures on a 250-failure gate -- while the shrinking survivor set
#: (250 -> ~78 -> ~24 -> ~8 -> ~2) costs ~370 extra cosims on top of the 837,
#: not 7*837.  That is ~45% of one suite pass, i.e. roughly +25 minutes on an
#: hour-long gate; simulated over five seeds in
#: titan_runs/out/work/test_confirm.py.
#:
#: What it does NOT catch: a genuinely intermittent RTL bug looks exactly
#: like a bridge flip and is cleared the same way.  Every clearing is
#: recorded (``regression_confirm`` events, ``*_confirm.json``, and the
#: agent-facing message) so it is visible rather than silent.
REGRESSION_CONFIRM_REPS = int(
    os.environ.get("TITAN_REGRESSION_CONFIRM_REPS", "7"))

#: Above this many reported failures, confirmation is skipped and every
#: reported failure is taken at face value.  A tree that breaks half the
#: suite is broken, not flaky, and re-running it proves nothing the agent
#: does not already know.
REGRESSION_CONFIRM_MAX = int(
    os.environ.get("TITAN_REGRESSION_CONFIRM_MAX", "500"))

#: 迴歸派工時一次最多讓幾支測試在飛。0 = 舊行為：整個 suite（837 支）一次送出，
#: 併發完全由叢集的 ``verilator_run`` 資源決定。
#:
#: 為什麼需要這個旋鈕：`cluster.yaml` 的 CPU 配額是按「獨佔主機」寫的
#: （verilator 4 節點 × 16 CPU = 64 核，全部節點加總正好等於主機的 96 執行緒），
#: 但主機是與 AETHER 叢集及另外約 40 位使用者共用的。2026-09-23 把 cosim
#: ``num_workers`` 從 4 降到 2，峰值降到 32 核；設定檔只在叢集重啟後生效，
#: 這個旋鈕則隨時可調，是不重啟就能降載的那一條路。
#:
#: 注意這不改變總工作量，只改變同時在飛的數量——牆鐘會變長，覆蓋不變。
REGRESSION_MAX_INFLIGHT = int(
    os.environ.get("TITAN_REGRESSION_MAX_INFLIGHT", "0"))

#: FLOOR on how many separate confirmation passes must clear a test as
#: "flaky" before the loop will say out loud that it might not be flaky.
#: A floor, not the rule: the rule is in ``_repeat_suspects``, which only
#: fires when the repeat count exceeds what the run's own measured background
#: clear rate produces by chance.  A bare count does not work -- at ~250 of
#: 837 tests cleared per pass the per-test rate is ~0.30, so chance alone
#: puts 837*0.30**3 = 23 names at three repeats, and an alert that names 23
#: flakes every run hides the one that matters.
#:
#: This is the only handle there is on the hole unanimity leaves.  A genuinely
#: intermittent RTL bug and a trace-bridge flip are indistinguishable in one
#: pass, and unanimity discards both.  But they differ ACROSS passes: r6
#: measured 82-87% of the failing set reshuffling between runs, so pure flake
#: hits a different random subset every time, so a name that comes back far
#: more often than the background rate explains is the one residual signature
#: of the class of bug this mechanism throws away.  The statistic is not the
#: repeat count but reps-to-first-pass pooled over passes (see
#: ``_repeat_suspects``); the null is measured on the run itself.
#:
#: Its power, simulated at a 0.31 background over 837 tests with one planted
#: intermittent bug, 6 seeds each (titan_runs/out/work/test_confirm.py --
#: "caught" = confirmed as a real failure OR flagged as a suspect):
#:
#:     bug fails ...% of runs   8 passes   16 passes
#:     50%                      0/6        0/6
#:     70%                      2/6        4/6
#:     80%                      3/6        6/6
#:     90%                      5/6        5/6
#:
#: So: a bug that fails half its runs is invisible to both halves, and even a
#: 70% one needs a long run.  Silence here is weak evidence, not a clean bill
#: of health.
#:
#: Signal only.  Nothing in the loop re-judges on this count, because
#: automatic re-judging would put back exactly the unpassability the
#: confirmation was built to remove.  It is printed for a human to chase with
#: an independent, unloaded re-run.
REGRESSION_FLAKY_REPEAT_ALERT = int(
    os.environ.get("TITAN_REGRESSION_FLAKY_REPEAT", "3"))

STRESS_TEST_VRUN_FRACTION = float(
    os.environ.get("TITAN_STRESS_VRUN_FRACTION", "0.5"))

# --- LLM -------------------------------------------------------------------
# 2026-09-23 改用 Opus 5.5（round 7 起）。需要 llm 容器內 Claude Code >= 2.1.280；
# image 內建的是 2.1.252，容器重建後要再 `docker exec -u root <llm 容器> npm i -g
# @anthropic-ai/claude-code@latest`，否則 API 回 400 unsupported model。
LLM_MODEL = os.environ.get("TITAN_LLM_MODEL", "claude-opus-5-5")
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

#: Most repeats one ``run_rvv_start`` call may ask for (``reps``).
#: A single cosim run is not a verdict: the cospike/DebugROB DPI trace bridge
#: is nondeterministic run to run, and the same binary on the same test flips
#: ~12% of runs (``titan_runs/nondet/``: 8/64 at 8 threads, 9/32 at 1).
#: ``reps`` re-runs the selection on the *same* build and reports ``k/n
#: passed`` per test so the agent can judge by majority.  5 is the ceiling
#: because the cost is linear and a turn has to end.
RVV_AGENT_MAX_REPS = int(os.environ.get("TITAN_RVV_AGENT_MAX_REPS", "5"))

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
    "TITAN_SYNTH_OBJ_ROOT", str(OUT_DIR / "synth"))
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


#: Stage 3 的測試程式在 riscv_build 容器裡組譯，所以工作目錄必須是那個容器
#: 掛載得到的路徑。專案的 ``out/`` 沒有掛進去 —— r22 的 S3 就是在這裡以
#: PermissionError 當掉的。``titan_scratch`` 是每個 node type 都有掛的共用
#: scratch（cluster.yaml 的 TMPDIR 也指向它），而且在 /share1 上。
STRESS_WORK_DIR = os.environ.get(
    "TITAN_STRESS_WORK_DIR",
    f"/share1/saves/{__import__('getpass').getuser().removesuffix('_l')}"
    "/titan_scratch/node_tmp/titan-stress")
