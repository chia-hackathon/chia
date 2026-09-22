#!/usr/bin/env python3
"""Directed S1 tests: paired IME / RVV programs over the same random data.

Each generated program computes C <- A x B^T + C twice on one machine -- once
through the IME instructions under test, once through a plain RVV 1.0
sequence -- into two separate buffers, then compares them element by element
and reports the first differing row in its exit code.  No Spike, no host-side
expected values, no golden file: the program carries its own judge.

Replaces riscv_extensions/isa_tests.py, whose comment says outright that its
tests need no self-check because "Spike is the judge, so the test only has to
execute the instruction".  Titan has no Spike in Stage 1, and cospike could
not help anyway -- its comparison surface is the PC and the integer register
writeback, which cannot see a vector register.

Toolchain: the build harness Makefile assembles with riscv64-unknown-elf-gcc,
not clang, so the Zvvm mnemonics and clang's -menable-experimental-extensions
are both unavailable.  The three IME instructions are therefore emitted as
`.insn` directives carrying words built by ime_encodings from the spec.  This
needs nothing of the toolchain beyond baseline RVV, and it is immune to a
toolchain that claims Zvvm support but tracks a different draft.

Verdict contract (read by helpers.classify_run).  The program both prints a
marker and returns an exit status, and the marker is the primary signal:

    TITAN PASS                    -> pass, exit 0
    TITAN SKIP lambda=<n> (requested <r>) imm=<e>
                                  -> the DUT selected a different lambda.
                                     Only a *clamp down* (selected < requested
                                     and architecturally permissible) is a
                                     legal skip; a DUT that selects a LARGER
                                     lambda than requested has violated the
                                     WARL rule and helpers.classify_run calls
                                     that `bad_geometry`, which is a failure.
                                     Exit 1 either way.
    TITAN FAIL row=<i> col=<j>    -> the two paths disagree.  Exit 2 + i.

On a FAIL the program first prints the evidence, so the agent debugging the
datapath sees values rather than a bare coordinate:

    TITAN DIFF r=<i> c=<j> exp=0x<ref> got=0x<ime>   (first 8 differences)
    TITAN CDUMP r=<i>: 0x.. 0x.. ...                 (first 8 rows of c_ime)
    TITAN CREF  r=<i>: 0x.. 0x.. ...                 (the same rows of c_rvv)

A and B are deliberately not dumped: they are already in the program text as
.byte/.half/.word data, and at LMUL=8 dumping them would dominate the log.
Hex is printed with %x (and a %08x%08x pair at SEW=64) rather than %llx --
newlib-nano's printf is built without long-long support, so %llx would print
garbage on exactly the widest elements that matter most.

Printing rather than relying on the exit status alone is deliberate: how a
non-zero return from main under htif_nano surfaces in the simulator log is a
harness detail nobody here has verified, whereas every other test in this
repo is judged by grepping stdout (memcpy prints "MEMCPY Num Correct: N").
The exit status is kept as a redundant second channel.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import ime_encodings as ime
import rvv_ref
#: Evidence caps.  A failing tile prints at most this many differing elements
#: and this many dumped rows per buffer -- enough to see a pattern (a whole
#: bad row, a bad register in the C group) without burying the verdict line
#: under a 256-element dump.  They live in constants.py because
#: helpers.format_directed_failure quotes back what this bakes in, and the
#: two caps drifting apart would silently truncate the evidence mid-tile.
from constants import MAX_DIFF_LINES, MAX_DUMP_ROWS
from rvv_ref import Matrix, TileGeometry

#: Exit codes.  Anything >= MISMATCH_BASE localises to a C tile row.
EXIT_PASS = 0
EXIT_UNSUPPORTED_GEOMETRY = 1
EXIT_MISMATCH_BASE = 2

#: mstatus.VS = Initial.  htif_nano's crt0 runs main in M-mode and does not
#: enable vector state, so a program that skips this traps on the first
#: vector instruction and looks exactly like an unimplemented-instruction bug.
MSTATUS_VS_INITIAL = 0x600

_SEW_SUFFIX = {8: "b", 16: "h", 32: "w", 64: "d"}
#: Zero-extending loads, for printing a value as hex.  The compare path uses
#: the sign-extending forms; both paths are loaded the same way there, so the
#: comparison is unaffected, but a printed value must not be sign-extended.
_SEW_LOADU = {8: "lbu", 16: "lhu", 32: "lwu", 64: "ld"}
_VSEW_FIELD = {8: 0, 16: 1, 32: 2, 64: 3}
_VLMUL_FIELD = {1: 0, 2: 1, 4: 2, 8: 3}
_DATA_DIRECTIVE = {8: ".byte", 16: ".half", 32: ".word", 64: ".dword"}

#: mstatus.FS = Initial.  A floating-point program has to turn the F/D unit
#: on for the same reason every program turns the vector unit on: the harness
#: starts in M-mode with both extension state fields Off, and an flw or an
#: fmul.s would take an illegal-instruction trap.  FS is mstatus[14:13], one
#: field below VS at mstatus[10:9].
MSTATUS_FS_INITIAL = 0x2000

#: Scalar floating-point load / store / multiply / add suffix per SEW.  Round
#: four's two accumulator widths are exactly the two the baseline -march
#: supplies scalar arithmetic for: constants.MARCH_IME is
#: rv64imafdv_zicsr_zifencei, so `f` gives binary32 and `d` gives binary64
#: and neither needs an extension the build harness does not already ask
#: for.  There is no `h`, which is the mechanical reason round four stops at
#: SEW >= 32 (see rvv_ref.FP_FORMATS and round4_design.md).
_FP_SUFFIX = {32: "s", 64: "d"}
_FP_LOAD = {32: "flw", 64: "fld"}
_FP_STORE = {32: "fsw", 64: "fsd"}

#: Scalar registers the emitted IME instructions name.  The `.insn` words bake
#: register numbers in, so these are not free choices -- the assembly must use
#: exactly these ABI names.
#: What one line of a C-tile hex dump is, per tile-transfer variant.  The
#: buffers carry the layout the tile store wrote, so a transposing program
#: dumps physical C *columns*.  Round one's programs print "r=", unchanged.
_C_LINE_LABEL = {"op": "r", "t": "c"}

RS1_ADDR = 10   # a0 -- tile base address
RS2_LD = 11     # a1 -- leading dimension


@dataclass(frozen=True)
class VectorAlloc:
    """Which architectural registers hold A, B and C.

    C sits at the top of the file because its base must be a multiple of
    EMUL_C, and 32 - EMUL_C always is.  A and B go below it, LMUL-aligned by
    construction.
    """

    a: int
    b: int
    c: int

    @staticmethod
    def allocate(geom: TileGeometry,
                 reserve_v0: bool = False) -> "VectorAlloc":
        """Place A, B and C.

        *reserve_v0* keeps ``v0`` free for the round-six microscaled forms,
        whose paired E8M0 block scales live there (spec 2129-2170) and which
        therefore cannot have the A tile sitting on top of them.  It defaults
        to False and, at False, returns exactly what every round one to five
        caller got before -- those tiers' emitted programs are required to
        stay byte-identical, so this is an added option and not a changed
        default.

        At True the A tile starts at the first LMUL-aligned register above
        v0, which is v[LMUL]: LMUL is a power of two, so v[LMUL] is
        LMUL-aligned by construction, and it is the lowest such register that
        is not v0.  That costs one LMUL-sized hole and is why the fit check
        below asks for 3*LMUL rather than 2*LMUL.

        A geometry that does not fit raises, and ime_stress filters on that
        exception.  It deliberately does not fall back to a different
        geometry: quietly testing a shape other than the one asked for is how
        a family ends up with no coverage at all where it matters.
        """
        c_base = 32 - geom.emul_c
        groups = 3 if reserve_v0 else 2
        if groups * geom.lmul > c_base:
            held = " (v0 held for the E8M0 block scales)" if reserve_v0 else ""
            raise ValueError(
                f"{geom.describe()}: A, B ({groups - 1} x LMUL={geom.lmul})"
                f"{held} and C (EMUL_C={geom.emul_c}) do not fit in 32 "
                f"vector registers")
        base = geom.lmul if reserve_v0 else 0
        return VectorAlloc(a=base, b=base + geom.lmul, c=c_base)


def vtype_value(geom: TileGeometry, *, lmul: int, xlen: int = 64,
                vta: int = 0, vma: int = 0, altfmt_a: int = 0,
                altfmt_b: int = 0, bs: int = 0, altfmt: int = 0) -> int:
    """Assemble a full vtype word, IME fields included.

    vsetvli/vsetivli cannot reach the IME fields -- they live above the
    vtypei immediate -- so every configuration here goes through the register
    form, `vsetvl`, which writes vtype wholesale from rs2.

    ``altfmt`` is the *base* Zvfbfa output-format field, not one of the three
    IME fields next to it, and it is the one round six needs: at SEW=16 it is
    all that separates a binary16 accumulator from a bfloat16 one (spec
    1092-1106).  It is keyed by an absolute bit position rather than by an
    offset below XLEN -- see ime_encodings.VTYPE_BASE_FIELDS -- so it is
    assembled in its own loop.  It defaults to 0, and 0 ORs in nothing, so
    every round one to five caller gets exactly the word it got before.
    """
    value = (_VLMUL_FIELD[lmul] | (_VSEW_FIELD[geom.sew] << 3)
             | (vta << 6) | (vma << 7))
    for name, field in (("lambda", ime.lambda_imm(geom.lam)), ("bs", bs),
                        ("altfmt_A", altfmt_a), ("altfmt_B", altfmt_b)):
        offset, width = ime.VTYPE_IME_FIELDS[name]
        assert field < (1 << width)
        value |= field << (xlen - offset)
    for name, field in (("altfmt", altfmt),):
        lsb, width = ime.VTYPE_BASE_FIELDS[name]
        assert field < (1 << width)
        value |= field << lsb
    return value


def _c_off(geom: TileGeometry, i: int, j: int) -> int:
    """Element offset of C[i,j] in the in-memory c_init / c_ime / c_rvv image.

    Row-major (``i*M + j``) for an order-preserving program, column-major
    (``j*M + i``) for a transposing one.  The three buffers always share a
    layout, so the element-wise comparison is a plain walk either way; what
    changes is which (i, j) a given offset denotes, and both the RVV
    reference path and the compare go through this function so they cannot
    disagree.

    This is :func:`rvv_ref.c_memory_offset` / ``c_memory_offset_t`` at
    LD = M, restated in the generator because the generator is what must
    stay byte-identical; the self-test asserts the two agree.
    """
    return i * geom.m + j if geom.tload == "op" else j * geom.m + i


def _matrix_data(label: str, mat: Matrix, sew: int) -> List[str]:
    directive = _DATA_DIRECTIVE[sew]
    mask = (1 << sew) - 1
    lines = [f"{label}:"]
    for row in mat:
        lines.append(f"    {directive} " + ", ".join(f"0x{v & mask:x}"
                                                     for v in row))
    return lines


def _check_lambda_retained(geom: TileGeometry, xlen: int = 64) -> List[str]:
    """Bail out cleanly if the DUT clamped our requested lambda.

    A nonzero lambda written through vsetvl is a *request*: the spec's WARL
    rule says an implementation that does not support it selects the largest
    supported value <= the request instead.  So a silently different geometry
    is a legal outcome, not a bug, and a test that ignores it would compare
    two paths that were configured differently and blame the RTL.  The
    architecture's own discovery mechanism is exactly this: write, read back,
    see what stuck.
    """
    offset, width = ime.VTYPE_IME_FIELDS["lambda"]
    # The conditional branch hops two instructions; the long jump to the
    # epilogue is `j`, which reaches +-1MB.  A conditional branch straight to
    # the epilogue would exceed its +-4KB range once the unrolled RVV path is
    # inlined between them -- and would do so only for the larger geometries,
    # so it would look like a flaky assembler rather than a range error.
    return [
        "    csrr  t2, vtype",
        f"    srli  t3, t2, {xlen - offset}",
        f"    andi  t3, t3, {(1 << width) - 1}",
        f"    li    t4, {ime.lambda_imm(geom.lam)}",
        "    beq   t3, t4, 1f",
        "    mv    a1, t3             # report the lambda the DUT chose",
        "    j     .Lskip",
        "1:",
    ]


def _configure(geom: TileGeometry, *, lmul: int, vl: int,
               comment: str) -> List[str]:
    return [
        f"    # {comment}",
        f"    li    t0, {vl}",
        f"    li    t1, 0x{vtype_value(geom, lmul=lmul):x}",
        "    vsetvl x0, t0, t1",
    ]


def _ime_path(geom: TileGeometry, alloc: VectorAlloc, *,
              emit_store: bool = True) -> List[str]:
    """vsetvl -> load C -> load A, B -> multiply-accumulate -> store C.

    The widening (W=4) case reuses this sequence unchanged apart from the
    multiply-accumulate opcode, and that is not a shortcut -- it is what the
    spec says.  vtype.SEW is the *accumulator* width for the whole family,
    the tile load/stores move SEW-wide *storage* elements and never look
    inside a packed widening element ("For widening input tiles ... the
    instruction does not transpose the packed logical elements within
    them"), and M / N_max / EMUL_C / VL are all functions of SEW alone.  So
    the C transfer config, the A/B transfer config, the compute VL and the
    leading dimensions are bit-identical to round one; only the four Int8
    values packed into each 32-bit A/B storage element, and funct6, differ.
    W=2 (vwmmacc.vv) and W=8 (v8wmmacc.vv) ride the identical sequence.

    The transposing variant (``geom.tload == "t"``) is the same five
    instructions with vmttl.v / vmtts.v substituted for vmtl.v / vmts.v and
    a column-major memory image in place of the row-major one.  Per the
    Sail, the register side of a transposing transfer is character-for-
    character the order-preserving one -- only ``mem_off`` swaps
    ``i / linesize`` and ``i % linesize`` -- so the registers the
    multiply-accumulate sees are bit-identical and the arithmetic half of
    the program does not change at all.  LD becomes M for A, B *and* C (the
    column stride of an M x K_eff or M x M tile), which is also the rs2 = x0
    default the spec gives the transposing pair.

    ``emit_store=False`` stops after the multiply-accumulate, leaving the
    result in the C register group instead of writing it out with the tile
    store.  That is what round five's C-layout probe wants: the whole point
    of that tier is to look at the register group with an *architectural*
    access, and a vmts.v that mis-indexes C in the same way the
    multiply-accumulate does would hide exactly the bug being hunted.  The
    default is True, so every round-one-to-four program is unchanged.
    """
    lam_imm = 0  # 0 as an instruction immediate means "use vtype.lambda"
    out: List[str] = ["", "    # ---- IME path ----"]

    # The C accumulator moves under its own configuration: LMUL = EMUL_C and
    # VL = M * N_max, which makes linesize = LAMBDA * EMUL_C = M, one line per
    # physical C row.  The compute VL is emphatically not the right VL here --
    # the active columns of a partial block are not a contiguous 1D segment of
    # the physical tile.
    out += _configure(geom, lmul=geom.lmul_c, vl=geom.vl_c_full,
                      comment=f"C tile transfer config (LMUL=EMUL_C="
                              f"{geom.emul_c}, VL={geom.vl_c_full})")
    out += _check_lambda_retained(geom)
    out += [
        "    la    a0, c_init",
        f"    li    a1, {geom.m}          # LD = M: "
        f"{'row' if geom.tload == 'op' else 'column'}-major M x M block",
        f"    {ime.insn(geom.load_mnemonic, vd=alloc.c, rs1=RS1_ADDR, rs2=RS2_LD, vm=1, **{'lambda': lam_imm})}"
        f"    # {geom.load_mnemonic} v{alloc.c}, (a0), a1",
    ]

    out += _configure(geom, lmul=geom.lmul, vl=geom.lmul * geom.elems_per_reg,
                      comment=f"A/B config (LMUL={geom.lmul}, full VL)")
    # Per the Sail (vmtl.v: mem_off = (i/linesize)*LD + (i%linesize), with i
    # the sequential tile element index), the memory image of an A/B tile is
    # plain row-major with stride LD at every LMUL -- the LMUL>1 packing is
    # entirely on the register side, via tile_reg_idx.  LD = linesize = K_eff
    # here (W = 1), so the tile buffers are byte-identical to the row-major
    # copies the RVV path reads; they are emitted separately only so that the
    # layout has exactly one authority (rvv_ref.tile_layout_buffer).
    # A transposing load reads the column-major image instead, with
    # LD = M (the column stride, and the rs2=x0 default for vmttl.v).
    ab_ld = geom.linesize if geom.tload == "op" else geom.m
    for label, base in (("mat_a_tile", alloc.a), ("mat_b_tile", alloc.b)):
        out += [
            f"    la    a0, {label}",
            f"    li    a1, {ab_ld}",
            f"    {ime.insn(geom.load_mnemonic, vd=base, rs1=RS1_ADDR, rs2=RS2_LD, vm=1, **{'lambda': lam_imm})}"
            f"    # {geom.load_mnemonic} v{base}, (a0), a1",
        ]

    out += _configure(geom, lmul=geom.lmul, vl=geom.vl,
                      comment=f"compute config (VL={geom.vl} -> N={geom.n})")
    out += [
        f"    {ime.insn(geom.mnemonic, vd=alloc.c, vs1=alloc.a, vs2=alloc.b)}"
        f"    # {geom.mnemonic} v{alloc.c}, v{alloc.a}, v{alloc.b}",
    ]

    if not emit_store:
        return out

    out += _configure(geom, lmul=geom.lmul_c, vl=geom.vl_c_full,
                      comment="back to the C tile config to store")
    out += [
        "    la    a0, c_ime",
        f"    li    a1, {geom.m}",
        f"    {ime.insn(geom.store_mnemonic, vs3=alloc.c, rs1=RS1_ADDR, rs2=RS2_LD, vm=1, **{'lambda': lam_imm})}"
        f"    # {geom.store_mnemonic} v{alloc.c}, (a0), a1",
    ]
    return out


def _rvv_path(geom: TileGeometry) -> List[str]:
    """The same GEMM in RVV 1.0, fully unrolled.

    Unrolled on purpose: straight-line code has no loop bounds or induction
    variables to get wrong, and this is the judge.  M x N is at most 256
    dot products, so the size is affordable.

    Each dot product is vmul.vv followed by vredsum.vs at SEW, which is
    exactly modulo-2**SEW arithmetic -- the accumulation rule the spec states
    for the integer multiply-accumulate.
    """
    if geom.kind == "fp":
        return _fp_ref_path(geom)
    if geom.w != 1:
        return _rvv_path_widening(geom)
    sfx = _SEW_SUFFIX[geom.sew]
    esz = geom.sew // 8
    out: List[str] = ["", "    # ---- RVV 1.0 reference path ----",
                      f"    li    t0, {geom.k_eff}",
                      f"    vsetvli t1, t0, e{geom.sew}, m1, ta, ma",
                      "    la    a2, mat_a",
                      "    la    a3, mat_b",
                      "    la    a4, c_init",
                      "    la    a5, c_rvv"]

    for i in range(geom.m):
        for j in range(geom.n_max):
            c_off = _c_off(geom, i, j) * esz
            if j >= geom.n:
                # Tail column: not part of the computation.  vta=0 leaves it
                # undisturbed on the IME side, so copy it across unchanged
                # and the comparison stays exact over the whole tile.
                out += [
                    f"    li    t0, {c_off}",
                    "    add   t2, a4, t0",
                    "    add   t3, a5, t0",
                    f"    l{sfx}    t4, 0(t2)",
                    f"    s{sfx}    t4, 0(t3)",
                ]
                continue
            out += [
                f"    li    t0, {i * geom.k_eff * esz}",
                "    add   t2, a2, t0",
                f"    li    t0, {j * geom.k_eff * esz}",
                "    add   t3, a3, t0",
                f"    vle{geom.sew}.v v1, (t2)",
                f"    vle{geom.sew}.v v2, (t3)",
                "    vmul.vv v3, v1, v2",
                "    vmv.s.x v5, x0",
                "    vredsum.vs v4, v3, v5",
                "    vmv.x.s t5, v4",
                f"    li    t0, {c_off}",
                "    add   t2, a4, t0",
                "    add   t3, a5, t0",
                f"    l{sfx}    t4, 0(t2)",
                "    add   t4, t4, t5",
                f"    s{sfx}    t4, 0(t3)",
            ]
    return out


def _rvv_path_widening(geom: TileGeometry) -> List[str]:
    """The W=4 GEMM in RVV 1.0: Int8 x Int8 accumulated at SEW.

    No widening multiply is used, and none is needed.  The generator emits a
    second copy of A and B (`mat_a`, `mat_b`) whose narrow elements are
    already *sign-extended* to SEW-wide words -- see :func:`emit_test` -- so
    an ordinary SEW-wide ``vmul.vv`` reproduces the exact product: each
    input fits in EEW_A = SEW/4 bits, so the true product needs at most
    2*(SEW/4) <= SEW bits and cannot wrap.  ``vredsum.vs`` then sums modulo
    2**SEW, which is the spec's accumulation rule ("all intermediate results
    are reduced modulo 2^SEW").  Reading the elements back as unsigned in
    the model is harmless for the same reason multiplication mod 2**SEW is:
    the low SEW bits of a product do not depend on how the operands were
    signed -- what the sign extension buys is that the *value* being
    multiplied is the sign-extended one, which at W>1 is material.

    Unlike round one's path this cannot always do a K row in one vle: the
    logical row is K_eff = LAMBDA*4*LMUL elements long but each element is
    read as a SEW-wide word, so at K_eff > VLEN/SEW it is split into
    equal chunks of VLMAX and the partial sums are added in a scalar
    register.  The chunk count is a power of two and both are compile-time
    constants, so the code stays straight-line with no induction variable.
    """
    sfx = _SEW_SUFFIX[geom.sew]
    esz = geom.sew // 8
    chunk = min(geom.elems_per_reg, geom.k_eff)   # VLMAX at e{SEW}, m1
    assert geom.k_eff % chunk == 0, geom.describe()
    nchunks = geom.k_eff // chunk

    out: List[str] = ["", "    # ---- RVV 1.0 reference path (W=4) ----",
                      f"    li    t0, {chunk}",
                      f"    vsetvli t1, t0, e{geom.sew}, m1, ta, ma",
                      "    la    a2, mat_a",
                      "    la    a3, mat_b",
                      "    la    a4, c_init",
                      "    la    a5, c_rvv"]

    for i in range(geom.m):
        for j in range(geom.n_max):
            c_off = _c_off(geom, i, j) * esz
            if j >= geom.n:
                # C tile tail column: vta=0 leaves it undisturbed on the IME
                # side, so copy it across and keep the compare exact.
                out += [
                    f"    li    t0, {c_off}",
                    "    add   t2, a4, t0",
                    "    add   t3, a5, t0",
                    f"    l{sfx}    t4, 0(t2)",
                    f"    s{sfx}    t4, 0(t3)",
                ]
                continue
            out += ["    li    t6, 0              # running K partial sum"]
            for c in range(nchunks):
                out += [
                    f"    li    t0, {(i * geom.k_eff + c * chunk) * esz}",
                    "    add   t2, a2, t0",
                    f"    li    t0, {(j * geom.k_eff + c * chunk) * esz}",
                    "    add   t3, a3, t0",
                    f"    vle{geom.sew}.v v1, (t2)",
                    f"    vle{geom.sew}.v v2, (t3)",
                    "    vmul.vv v3, v1, v2",
                    "    vmv.s.x v5, x0",
                    "    vredsum.vs v4, v3, v5",
                    "    vmv.x.s t5, v4",
                    "    add   t6, t6, t5",
                ]
            out += [
                f"    li    t0, {c_off}",
                "    add   t2, a4, t0",
                "    add   t3, a5, t0",
                f"    l{sfx}    t4, 0(t2)",
                "    add   t4, t4, t6",
                f"    s{sfx}    t4, 0(t3)",
            ]
    return out


def _fp_ref_path(geom: TileGeometry) -> List[str]:
    """The same GEMM in scalar rv64f / rv64d, fully unrolled.

    Round four's reference is *not* an RVV reference, and that is the point.
    A floating-point matrix multiply-accumulate is not bit-determined by the
    instruction alone: spec 1536-1600 lets an implementation choose a
    grouping factor G, a partial-sum mode psm and a partial-sum rounding rnd,
    and requires it to disclose the (SEW, W, LAMBDA) -> (G, psm, rnd) table
    (1652-1656).  Titan discloses::

        G = 1, psm = 0, rnd = frm       for every (SEW, W=1, LAMBDA)

    which spec 1771 names as the tuple that makes a Zvvm implementation
    match the analogous Zvtm instruction "For input element widths of 32
    bits or greater ... so that each group contains a single product and the
    partial sum `S` is that product rounded according to `frm`."

    Under that disclosure the Sail (5243-5268, with G=1 and W=1) collapses to

        acc = fp_add(acc, fp_round_to_frm(fp_mul_exact(A[i,k], B[j,k])))

    for k = 0, 1, ... K_eff-1 in strictly increasing order -- two rounding
    points per k, which is a scalar ``fmul`` followed by a scalar ``fadd``.
    So the reference is three instructions per term and the comparison stays
    what it has been since round one: an *exact* bitwise compare of the two
    C images, with no tolerance and no FP-aware printf.  A fused
    multiply-add would be the rnd=xct disclosure and is a different answer;
    rvv_ref.check_round_four_fp_gemm carries a witness that distinguishes
    them, so this is not a distinction without a difference.

    Why not the vector unit: RVV 1.0 has no reduction with the ordering and
    rounding this needs (vfredosum.vs rounds once per element with no
    intermediate product rounding, which is the rnd=xct answer), and a
    per-element vfmul.vv + vfadd chain would be the same instruction count
    with a second vtype to get wrong.  The scalar F/D unit is also a more
    independent witness: it is not the pipeline under test.

    Rounding mode: frm is set to RNE once, in the prologue, rather than
    trusted from reset -- it is the only rounding-mode source the extension
    has (Sail 5695 calls the ordinary ``get_fp_rounding_mode()``; there is no
    matrix rounding CSR), so it governs both paths and both must see the
    same value.

    fflags are deliberately not compared.  Both paths accrue into the same
    architectural register, so a difference between them is unobservable
    from inside one program, and the reference path raises inexact on almost
    every term.  Spec 1806-1822 makes the *temporal* order of flag raises
    implementation-defined anyway; what it pins is the final bitwise OR,
    which needs a program that runs one path at a time.  Out of scope for
    round four, and named as such in round4_design.md.
    """
    sfx = _SEW_SUFFIX[geom.sew]
    fsfx = _FP_SUFFIX[geom.sew]
    fl, fs = _FP_LOAD[geom.sew], _FP_STORE[geom.sew]
    esz = geom.sew // 8
    out: List[str] = [
        "", f"    # ---- scalar rv64{fsfx} reference path "
            f"(G=1, psm=0, rnd=frm) ----",
        "    csrwi frm, 0             # RNE, for both paths",
        "    la    a2, mat_a",
        "    la    a3, mat_b",
        "    la    a4, c_init",
        "    la    a5, c_rvv"]

    for i in range(geom.m):
        for j in range(geom.n_max):
            c_off = _c_off(geom, i, j) * esz
            if j >= geom.n:
                # C tile tail column: vta=0 leaves it undisturbed on the IME
                # side, so copy the bits across.  An integer copy, not an
                # FP one: a floating-point load/store pair would canonicalise
                # a signalling NaN on some implementations, and the compare
                # is over raw bits.
                out += [
                    f"    li    t0, {c_off}",
                    "    add   t2, a4, t0",
                    "    add   t3, a5, t0",
                    f"    l{sfx}    t4, 0(t2)",
                    f"    s{sfx}    t4, 0(t3)",
                ]
                continue
            out += [
                f"    li    t0, {c_off}",
                "    add   t2, a4, t0",
                "    add   t3, a5, t0",
                f"    {fl}   ft0, 0(t2)        # acc = C[{i},{j}]",
            ]
            for k in range(geom.k_eff):
                out += [
                    f"    li    t0, {(i * geom.k_eff + k) * esz}",
                    "    add   t4, a2, t0",
                    f"    {fl}   ft1, 0(t4)",
                    f"    li    t0, {(j * geom.k_eff + k) * esz}",
                    "    add   t4, a3, t0",
                    f"    {fl}   ft2, 0(t4)",
                    f"    fmul.{fsfx} ft1, ft1, ft2   # S = round_frm(A*B)",
                    f"    fadd.{fsfx} ft0, ft0, ft1   # acc = round_frm(acc+S)",
                ]
            out += [f"    {fs}   ft0, 0(t3)"]
    return out


def _compare(geom: TileGeometry) -> List[str]:
    """Element-wise compare, exit code carrying the first differing row.

    An exact compare rather than a checksum: at M x M <= 256 elements it costs
    nothing, it cannot alias, and the row index is the localisation the plan
    asks a tile-level checksum to provide.
    """
    sfx = _SEW_SUFFIX[geom.sew]
    esz = geom.sew // 8
    out: List[str] = ["", "    # ---- compare ----",
                      "    la    a4, c_ime",
                      "    la    a5, c_rvv"]
    # On a mismatch, stage the row/col for printf and long-jump to the shared
    # epilogue.  The conditional branch only ever hops three instructions.
    for i in range(geom.m):
        for j in range(geom.n_max):
            off = _c_off(geom, i, j) * esz
            out += [
                f"    li    t0, {off}",
                "    add   t2, a4, t0",
                "    add   t3, a5, t0",
                f"    l{sfx}    t4, 0(t2)",
                f"    l{sfx}    t5, 0(t3)",
                "    beq   t4, t5, 1f",
                f"    li    a1, {i}",
                f"    li    a2, {j}",
                f"    li    s1, {EXIT_MISMATCH_BASE + i}",
                "    j     .Lfail",
                "1:",
            ]
    out += _epilogue(geom)
    return out


def _hex_fmt(sew: int) -> str:
    """printf conversion for one SEW-wide value.

    newlib-nano's printf has no long-long support, so a 64-bit element is
    printed as two zero-padded 32-bit halves rather than with %llx.
    """
    return "0x%08x%08x" if sew == 64 else "0x%x"


def _hex_args(sew: int, src: str, dst: Sequence[str]) -> List[str]:
    """Stage a zero-extended value in *src* into printf arg register(s)."""
    if sew == 64:
        return [f"    srli  {dst[0]}, {src}, 32",
                f"    slli  {dst[1]}, {src}, 32",
                f"    srli  {dst[1]}, {dst[1]}, 32"]
    return [f"    mv    {dst[0]}, {src}"]


def _dump_diffs(geom: TileGeometry) -> List[str]:
    """Print the first MAX_DIFF_LINES differing C elements, expected first.

    A single `TITAN FAIL row= col=` coordinate tells the agent where the
    datapath broke but not how, and "how" is what separates a transposed
    accumulator from an off-by-one register index: 0x00 where 0x2a was
    expected is a dead lane, 0x2a in the wrong place is an addressing bug.
    So the failing tile prints its own evidence, bounded, before the verdict.

    Runs as a loop rather than unrolled -- the compare path above is unrolled
    because it is the judge and must have no induction variable to get wrong,
    but this path only reports, and M x M unrolled printf call sites would
    double the size of every program in the suite.
    """
    loadu = _SEW_LOADU[geom.sew]
    esz = geom.sew // 8
    exp_dst = ("a3", "a4") if geom.sew == 64 else ("a3",)
    got_dst = ("a5", "a6") if geom.sew == 64 else ("a4",)
    return [
        "",
        "    # ---- evidence: first differing elements ----",
        "    li    s4, 0              # row",
        "    li    s5, 0              # column",
        "    li    s6, 0              # differences printed so far",
        "    li    s7, 0              # byte offset into the C tile",
        ".Ldump_diff:",
        f"    li    t0, {geom.m * geom.m * esz}",
        "    bge   s7, t0, .Ldump_diff_done",
        "    la    t1, c_ime",
        "    add   t1, t1, s7",
        "    la    t2, c_rvv",
        "    add   t2, t2, s7",
        f"    {loadu}  t3, 0(t1)          # got: the IME path",
        f"    {loadu}  t4, 0(t2)          # exp: the RVV reference",
        "    beq   t3, t4, .Ldump_diff_next",
        f"    li    t0, {MAX_DIFF_LINES}",
        "    bge   s6, t0, .Ldump_diff_done",
        # s4 counts the outer index of the linear walk and s5 the inner one.
        # In a row-major image that is (row, col); in the column-major image
        # a transposing program uses it is (col, row), so the two swap.
        # Printing them the other way round would report the transpose of
        # the failing coordinate -- a diagnosis that sends the agent after a
        # transposed accumulator that isn't there.
        f"    mv    a1, s{4 if geom.tload == 'op' else 5}",
        f"    mv    a2, s{5 if geom.tload == 'op' else 4}",
    ] + _hex_args(geom.sew, "t4", exp_dst)       + _hex_args(geom.sew, "t3", got_dst) + [
        "    la    a0, .Lfmt_diff",
        "    call  printf",
        "    addi  s6, s6, 1",
        ".Ldump_diff_next:",
        f"    addi  s7, s7, {esz}",
        "    addi  s5, s5, 1",
        f"    li    t0, {geom.m}",
        "    blt   s5, t0, .Ldump_diff",
        "    li    s5, 0",
        "    addi  s4, s4, 1",
        "    j     .Ldump_diff",
        ".Ldump_diff_done:",
    ]


def _dump_tile(geom: TileGeometry, buf: str, fmt: str, tag: str) -> List[str]:
    """Print the first MAX_DUMP_ROWS lines of *buf* as hex.

    The *physical* M x M tile, not the N active columns: the whole point of
    dumping it is that a wrong tail column or a wrong C register in the group
    is invisible in the active window.

    A "line" is a physical C row in an order-preserving program and a
    physical C *column* in a transposing one, because the buffers carry the
    layout the tile store wrote.  The format string says which (see the
    ``.Lfmt_cdump`` / ``.Lfmt_cref`` definitions in :func:`emit_test`).
    """
    loadu = _SEW_LOADU[geom.sew]
    esz = geom.sew // 8
    rows = min(geom.m, MAX_DUMP_ROWS)
    el_dst = ("a1", "a2") if geom.sew == 64 else ("a1",)
    return [
        "",
        f"    # ---- evidence: {tag} ----",
        "    li    s4, 0              # row",
        "    li    s7, 0              # byte offset",
        f".Ldump_{tag}_row:",
        f"    li    t0, {rows}",
        f"    bge   s4, t0, .Ldump_{tag}_done",
        f"    la    a0, {fmt}",
        "    mv    a1, s4",
        "    call  printf",
        "    li    s5, 0",
        f".Ldump_{tag}_col:",
        f"    li    t0, {geom.m}",
        f"    bge   s5, t0, .Ldump_{tag}_eol",
        f"    la    t1, {buf}",
        "    add   t1, t1, s7",
        f"    {loadu}  t3, 0(t1)",
    ] + _hex_args(geom.sew, "t3", el_dst) + [
        "    la    a0, .Lfmt_elem",
        "    call  printf",
        "    addi  s5, s5, 1",
        f"    addi  s7, s7, {esz}",
        f"    j     .Ldump_{tag}_col",
        f".Ldump_{tag}_eol:",
        "    la    a0, .Lfmt_nl",
        "    call  printf",
        "    addi  s4, s4, 1",
        f"    j     .Ldump_{tag}_row",
        f".Ldump_{tag}_done:",
    ]


def _epilogue(geom: TileGeometry) -> List[str]:
    """Print a grep-able verdict, then return it as the exit status too.

    s1 carries the exit status across the printf call because printf clobbers
    a0-a7; it is callee-saved, so the prologue spills it.  s2..s7 are spilled
    for the same reason: the failure path prints element values in a loop and
    has to survive its own printf calls.

    The verdict line is printed *last*, after the evidence, so that a log tail
    that catches the verdict also catches the evidence that explains it.
    """
    out = [
        "",
        "    # ---- verdict ----",
        "    la    a0, .Lfmt_pass",
        "    call  printf",
        f"    li    s1, {EXIT_PASS}",
        "    j     .Lret",
        "",
        ".Lskip:",
        # a1 arrives holding the *encoded* lambda[2:0] field the DUT left in
        # vtype.  Report the lambda it denotes as well as the raw field: the
        # two are easy to confuse (lambda=4 encodes as 0b011) and a feedback
        # message that confuses them sends the agent after the wrong bug.
        "    mv    a2, a1             # raw vtype.lambda[2:0] field",
        "    li    a1, 0              # decoded: 0 means no selected lambda",
        "    beqz  a2, 1f",
        "    addi  t0, a2, -1",
        "    li    t1, 1",
        "    sll   a1, t1, t0         # lambda = 1 << (imm - 1)",
        "1:",
        "    la    a0, .Lfmt_skip",
        "    call  printf",
        f"    li    s1, {EXIT_UNSUPPORTED_GEOMETRY}",
        "    j     .Lret",
        "",
        ".Lfail:",
        "    mv    s2, a1             # first differing row",
        "    mv    s3, a2             # first differing column",
    ]
    out += _dump_diffs(geom)
    out += _dump_tile(geom, "c_ime", ".Lfmt_cdump", "cdump")
    out += _dump_tile(geom, "c_rvv", ".Lfmt_cref", "cref")
    out += [
        "",
        "    # ---- verdict, last so a log tail keeps it ----",
        "    la    a0, .Lfmt_fail",
        "    mv    a1, s2",
        "    mv    a2, s3",
        "    call  printf",
        "",
        ".Lret:",
        "    mv    a0, s1",
        "    ld    s7, 0(sp)",
        "    ld    s6, 8(sp)",
        "    ld    s5, 16(sp)",
        "    ld    s4, 24(sp)",
        "    ld    s3, 32(sp)",
        "    ld    s2, 40(sp)",
        "    ld    s1, 48(sp)",
        "    ld    ra, 56(sp)",
        "    addi  sp, sp, 64",
        "    ret",
    ]
    return out


def emit_test(geom: TileGeometry, case: Tuple[Matrix, Matrix, Matrix],
              name: str = "ime_directed") -> str:
    """Assemble one self-checking paired program for *geom* and *case*.

    The harness links htif_nano.specs, which supplies _start and tohost and
    calls main -- so this must define main and must not define either of
    those.  Return value of main becomes the exit status.
    """
    geom.validate()
    if geom.emul_c == 16:
        raise ValueError(
            f"{geom.describe()}: EMUL_C=16 has no single-instruction C tile "
            f"transfer (LMUL=16 is not a legal vtype), so the accumulator "
            f"must be moved as two m8 halves via the m16 pair/unpair ops. "
            f"Out of scope for round one.")
    alloc = VectorAlloc.allocate(geom)
    a, b, c = case

    head = [
        f"# {name}: {geom.describe()}",
        "#",
        "# Generated by ime_tests.py from rvv_ref.py -- do not edit by hand,",
        "# and do not edit rvv_ref.py: it is the judge, not the defendant.",
        "#",
        f"# IME instructions are emitted as .insn (encodings from Zvvm "
        f"v{ime.SPEC_VERSION}):",
    ]
    if geom.kind == "int" and geom.w == 1 and geom.tload == "op":
        used = ime.ROUND_ONE          # round one's exact three, in its order
    else:
        used = (geom.load_mnemonic, geom.mnemonic, geom.store_mnemonic)
    for mnemonic in used:
        head.append(f"#   {mnemonic}")
    head += [
        "",
        "    .text",
        "    .balign 4",
        "    .globl main",
        "main:",
        # 64 bytes: ra, plus s1..s7.  s1 carries the exit status past printf;
        # s2..s7 carry the failure-evidence loop state past its printf calls.
        "    addi  sp, sp, -64",
        "    sd    ra, 56(sp)",
        "    sd    s1, 48(sp)         # carries the exit status past printf",
        "    sd    s2, 40(sp)",
        "    sd    s3, 32(sp)",
        "    sd    s4, 24(sp)",
        "    sd    s5, 16(sp)",
        "    sd    s6, 8(sp)",
        "    sd    s7, 0(sp)",
        f"    li    t0, {MSTATUS_VS_INITIAL}",
        "    csrs  mstatus, t0        # enable vector state",
    ]
    if geom.kind == "fp":
        # The reference path runs on the scalar F/D unit, so mstatus.FS has
        # to leave Off as well.  Emitted only for a floating-point geometry,
        # which is what keeps every round-one-to-three program byte-identical.
        head += [
            f"    li    t0, {MSTATUS_FS_INITIAL}",
            "    csrs  mstatus, t0        # enable floating-point state",
        ]

    body = _ime_path(geom, alloc) + _rvv_path(geom) + _compare(geom)

    data = [
        "", "    .data", "    .balign 8",
        # Grep-able verdict markers.  helpers.classify_run matches these; the
        # geometry is embedded so a failing line in a 200-program run says
        # which configuration broke without cross-referencing the test name.
        f'.Lfmt_pass:  .asciz "TITAN PASS {geom.describe()}\\n"',
        f'.Lfmt_skip:  .asciz "TITAN SKIP lambda=%d (requested {geom.lam}) '
        f'imm=%d {geom.describe()}\\n"',
        f'.Lfmt_fail:  .asciz "TITAN FAIL row=%d col=%d {geom.describe()}\\n"',
        f'.Lfmt_diff:  .asciz "TITAN DIFF r=%d c=%d '
        f'exp={_hex_fmt(geom.sew)} got={_hex_fmt(geom.sew)}\\n"',
        f'.Lfmt_cdump: .asciz "TITAN CDUMP {_C_LINE_LABEL[geom.tload]}=%d:"',
        f'.Lfmt_cref:  .asciz "TITAN CREF {_C_LINE_LABEL[geom.tload]}=%d:"',
        f'.Lfmt_elem:  .asciz " {_hex_fmt(geom.sew)}"',
        '.Lfmt_nl:    .asciz "\\n"',
        "    .balign 8",
    ]
    # Row-major copies feed the RVV path (a vle needs contiguous K rows);
    # tile-layout copies feed the IME path.  Per the Sail these agree at
    # every LMUL; emitting both keeps rvv_ref the single authority on the
    # memory layout, so a future divergence shows up here rather than
    # silently.
    # mat_a / mat_b feed the RVV path and are always emitted at SEW: at W=1
    # that is the element width, and at W>1 it sign-extends each narrow input
    # into a SEW-wide word so a plain vmul.vv reproduces the exact product.
    # _matrix_data masks to SEW bits, so a negative Int8 becomes its correct
    # 32-bit two's-complement image.
    data += _matrix_data("mat_a", a, geom.sew)
    data += _matrix_data("mat_b", b, geom.sew)
    # c_init / c_ime / c_rvv all carry the layout the tile transfer uses:
    # row-major for vmtl.v / vmts.v, column-major for vmttl.v / vmtts.v.
    # Emitting the transpose of C here is what makes the transposing program
    # a real test: an implementation that ignored the transpose would load
    # C^T, accumulate into it and store a tile that disagrees with the RVV
    # path in every off-diagonal element.
    data += _matrix_data("c_init",
                         c if geom.tload == "op"
                         else [[c[i][j] for i in range(geom.m)]
                               for j in range(geom.n_max)],
                         geom.sew)
    # The tile copies feed the IME path and are emitted at the *logical*
    # input width, packed: one line is K_eff narrow elements, which is
    # exactly the byte image a vmtl.v at the SEW-wide storage width reads
    # (storage element s carries logical k in [W*s, W*s+W), least-significant
    # byte first, so the packing is just little-endian memory order).  At
    # W=1 ab_linesize == linesize and eew_ab == sew, so round one's data is
    # byte-for-byte unchanged.
    for label, mat in (("mat_a_tile", a), ("mat_b_tile", b)):
        if geom.tload == "op":
            width = geom.ab_linesize
            buf = rvv_ref.tile_layout_buffer(mat, width, geom)
        else:
            # Transposing: the K_eff x M column-major image of the same
            # M x K_eff tile, LD = M.  tload='t' is W=1 only, so
            # ab_linesize == linesize and eew_ab == sew here.
            width = geom.m
            buf = rvv_ref.tile_layout_buffer_t(mat, width, geom)
        chunked = [buf[i:i + width] for i in range(0, len(buf), width)]
        data += ["    .balign 8"] + _matrix_data(label, chunked, geom.eew_ab)
    data += ["    .balign 8", "c_ime:",
             f"    .zero {geom.m * geom.m * geom.sew // 8}",
             "    .balign 8", "c_rvv:",
             f"    .zero {geom.m * geom.m * geom.sew // 8}"]

    return "\n".join(head + body + data) + "\n"


# ---------------------------------------------------------------------------
# round five: the C-tile register-layout probe
# ---------------------------------------------------------------------------

def _clayout_readback(geom: TileGeometry, alloc: VectorAlloc) -> List[str]:
    """Copy the whole C register group out with an ordinary vse<SEW>.v.

    This is the entire point of the tier.  Every pre-round-five program
    writes the C tile with ``vmts.v`` and reads it back with ``vmts.v``, so
    a register-side C index that is wrong in the same way on the transfer
    and on the multiply-accumulate produces a *correct memory image* and
    passes.  An architectural unit-stride store has no tile semantics at
    all -- element p of the group goes to byte p*SEW/8, by definition -- so
    what lands in ``c_raw`` is the register group itself.

    The configuration is LMUL = EMUL_C at VL = EMUL_C * (VLEN/SEW), i.e.
    VLMAX, which is exactly M * N_max elements: ``mat_C_idx`` is a bijection
    onto the group, so the tile and the group are the same set of elements.
    Note this is a plain ``vsetvli`` -- no LAMBDA, nothing IME about it.
    """
    total = geom.m * geom.n_max
    assert total == geom.emul_c * geom.elems_per_reg
    return [
        "",
        "    # ---- architectural read-back of the C register group ----",
        f"    # LMUL=EMUL_C={geom.emul_c}, VL=VLMAX={total}: v{alloc.c}"
        f"..v{alloc.c + geom.emul_c - 1} verbatim, no tile indexing",
        f"    li    t0, {total}",
        f"    vsetvli t1, t0, e{geom.sew}, m{geom.emul_c}, ta, ma",
        "    la    a2, c_raw",
        f"    vse{geom.sew}.v v{alloc.c}, (a2)",
    ]


def _clayout_compare(geom: TileGeometry) -> List[str]:
    """Element-wise compare of the read-back group against the spec image.

    Unrolled, and walked in (i, j) order rather than in flat order, so that
    the first mismatch reports the C tile coordinate the agent reasons in --
    and so that, exactly as in :func:`_compare`, the judge has no induction
    variable to get wrong.  ``c_element_index`` is the Sail ``mat_C_idx``:
    it is the thing under test, and it appears here only as an address.
    """
    sfx = _SEW_SUFFIX[geom.sew]
    esz = geom.sew // 8
    out: List[str] = ["", "    # ---- compare against the spec's C register "
                          "image (rvv_ref.c_element_index) ----",
                      "    la    a4, c_raw",
                      "    la    a5, c_exp"]
    for i in range(geom.m):
        for j in range(geom.n_max):
            off = rvv_ref.c_element_index(i, j, geom) * esz
            out += [
                f"    li    t0, {off}          # C[{i},{j}] -> group element "
                f"{off // esz}",
                "    add   t2, a4, t0",
                "    add   t3, a5, t0",
                f"    l{sfx}    t4, 0(t2)",
                f"    l{sfx}    t5, 0(t3)",
                "    beq   t4, t5, 1f",
                f"    li    a1, {i}",
                f"    li    a2, {j}",
                f"    li    s1, {EXIT_MISMATCH_BASE + i}",
                "    j     .Lfail",
                "1:",
            ]
    return out


def _clayout_dump_diffs(geom: TileGeometry) -> List[str]:
    """Print the first MAX_DIFF_LINES differing elements, in the usual form.

    Same ``TITAN DIFF r= c= exp= got=`` line the pair tiers print, so
    helpers.format_directed_failure quotes this tier's evidence back to the
    agent with no change at all.  The walk is over the flat register group,
    which is the order the failure actually has structure in (a whole bad
    register shows up as a run), and the (i, j) the line reports is looked up
    from the ``c_idx_r`` / ``c_idx_c`` tables rather than computed -- the
    inverse of ``mat_C_idx`` is not something to derive in assembly.
    """
    loadu = _SEW_LOADU[geom.sew]
    esz = geom.sew // 8
    exp_dst = ("a3", "a4") if geom.sew == 64 else ("a3",)
    got_dst = ("a5", "a6") if geom.sew == 64 else ("a4",)
    return [
        "",
        "    # ---- evidence: first differing elements ----",
        "    li    s4, 0              # flat C register-group element index",
        "    li    s6, 0              # differences printed so far",
        "    li    s7, 0              # byte offset into the group",
        ".Ldump_diff:",
        f"    li    t0, {geom.m * geom.n_max}",
        "    bge   s4, t0, .Ldump_diff_done",
        "    la    t1, c_raw",
        "    add   t1, t1, s7",
        "    la    t2, c_exp",
        "    add   t2, t2, s7",
        f"    {loadu}  t3, 0(t1)          # got: the C register group",
        f"    {loadu}  t4, 0(t2)          # exp: the spec's mat_C_idx image",
        "    beq   t3, t4, .Ldump_diff_next",
        f"    li    t0, {MAX_DIFF_LINES}",
        "    bge   s6, t0, .Ldump_diff_done",
        "    la    t1, c_idx_r",
        "    add   t1, t1, s4",
        "    lbu   a1, 0(t1)",
        "    la    t1, c_idx_c",
        "    add   t1, t1, s4",
        "    lbu   a2, 0(t1)",
    ] + _hex_args(geom.sew, "t4", exp_dst) + _hex_args(geom.sew, "t3", got_dst) + [
        "    la    a0, .Lfmt_diff",
        "    call  printf",
        "    addi  s6, s6, 1",
        ".Ldump_diff_next:",
        "    addi  s4, s4, 1",
        f"    addi  s7, s7, {esz}",
        "    j     .Ldump_diff",
        ".Ldump_diff_done:",
    ]


def _clayout_verdict(geom: TileGeometry) -> List[str]:
    """Name the wrong layout when the whole group is the expected transpose.

    56 scattered `TITAN DIFF` lines say "the C tile is wrong somewhere".
    One line saying `TITAN CLAYOUT transposed` says which bug it is, and the
    difference matters because the two are debugged in completely different
    places: a transposed group is a register-index bug in the sequencer's
    ``mat_C_idx``, not a datapath, accumulator or tile-transfer fault.

    ``c_xpose`` is the register image of C_ref transposed.  The scan is
    exhaustive -- *every* element must match, not merely the ones that
    differed -- because "is a transpose" is a claim about the whole tile and
    a partial match would be a worse diagnosis than none.  When the scan
    fails the tier still says something useful: the layout is wrong but is
    not the known transpose, so the agent knows not to go looking for it.
    """
    loadu = _SEW_LOADU[geom.sew]
    esz = geom.sew // 8
    return [
        "",
        "    # ---- evidence: is the group exactly the expected transpose? ----",
        "    li    s4, 0",
        "    li    s7, 0",
        "    li    s6, 1              # assume transposed until proven not",
        ".Lclayout_scan:",
        f"    li    t0, {geom.m * geom.n_max}",
        "    bge   s4, t0, .Lclayout_done",
        "    la    t1, c_raw",
        "    add   t1, t1, s7",
        "    la    t2, c_xpose",
        "    add   t2, t2, s7",
        f"    {loadu}  t3, 0(t1)",
        f"    {loadu}  t4, 0(t2)",
        "    beq   t3, t4, .Lclayout_next",
        "    li    s6, 0",
        "    j     .Lclayout_done",
        ".Lclayout_next:",
        "    addi  s4, s4, 1",
        f"    addi  s7, s7, {esz}",
        "    j     .Lclayout_scan",
        ".Lclayout_done:",
        "    beqz  s6, 1f",
        "    la    a0, .Lfmt_clayout_t",
        "    call  printf",
        "    j     2f",
        "1:",
        "    la    a0, .Lfmt_clayout_o",
        "    call  printf",
        "2:",
    ]


def _clayout_epilogue(geom: TileGeometry) -> List[str]:
    """The same verdict contract as :func:`_epilogue`, plus the CLAYOUT line."""
    out = [
        "",
        "    # ---- verdict ----",
        "    la    a0, .Lfmt_pass",
        "    call  printf",
        f"    li    s1, {EXIT_PASS}",
        "    j     .Lret",
        "",
        ".Lskip:",
        "    mv    a2, a1             # raw vtype.lambda[2:0] field",
        "    li    a1, 0              # decoded: 0 means no selected lambda",
        "    beqz  a2, 1f",
        "    addi  t0, a2, -1",
        "    li    t1, 1",
        "    sll   a1, t1, t0         # lambda = 1 << (imm - 1)",
        "1:",
        "    la    a0, .Lfmt_skip",
        "    call  printf",
        f"    li    s1, {EXIT_UNSUPPORTED_GEOMETRY}",
        "    j     .Lret",
        "",
        ".Lfail:",
        "    mv    s2, a1             # first differing row",
        "    mv    s3, a2             # first differing column",
    ]
    out += _clayout_dump_diffs(geom)
    out += _dump_tile(geom, "c_raw", ".Lfmt_cdump", "cdump")
    out += _dump_tile(geom, "c_exp", ".Lfmt_cref", "cref")
    out += _clayout_verdict(geom)
    out += [
        "",
        "    # ---- verdict, last so a log tail keeps it ----",
        "    la    a0, .Lfmt_fail",
        "    mv    a1, s2",
        "    mv    a2, s3",
        "    call  printf",
        "",
        ".Lret:",
        "    mv    a0, s1",
        "    ld    s7, 0(sp)",
        "    ld    s6, 8(sp)",
        "    ld    s5, 16(sp)",
        "    ld    s4, 24(sp)",
        "    ld    s3, 32(sp)",
        "    ld    s2, 40(sp)",
        "    ld    s1, 48(sp)",
        "    ld    ra, 56(sp)",
        "    addi  sp, sp, 64",
        "    ret",
    ]
    return out


def emit_clayout_test(geom: TileGeometry, name: str = "ime_clayout") -> str:
    """One C-tile register-layout probe program for *geom*.

    Structurally this is :func:`emit_test` with the last two stages replaced.
    A, B and the initial C tile still arrive through ``vmtl.v`` exactly as
    they do in the pair tiers -- deliberately, because the A/B tile path is
    already covered and reusing it keeps the implementation's own A/B element
    placement consistent with what its multiply-accumulate then reads, so the
    only thing left under test is where the C elements *land*.  What changes
    is the exit: instead of a ``vmts.v`` back to memory and a comparison
    against a second on-DUT computation, the C register group is copied out
    with an architectural ``vse<SEW>.v`` and compared against the register
    image the spec's ``mat_C_idx`` prescribes.

    That makes this program non-differential: its expected values are
    precomputed by rvv_ref rather than recomputed on the DUT.  That is not a
    regression in rigour, it is the only way to see the register layout at
    all -- a differential pair cancels any C permutation that the
    implementation applies to both halves, which is precisely how a
    transposed C tile survived four rounds of green.  The operands are
    correspondingly not random: :func:`rvv_ref.clayout_case` picks them so
    that every one of the M*N_max results is distinct and the linear-index
    hypothesis lands on an exact transpose.
    """
    geom.validate()
    if not rvv_ref.clayout_capable(geom):
        raise ValueError(
            f"{geom.describe()}: not a C-layout-probe geometry "
            f"(see rvv_ref.clayout_capable)")
    alloc = VectorAlloc.allocate(geom)
    a, b, c_init = rvv_ref.clayout_case(geom)
    c_ref = rvv_ref.reference_gemm(a, b, c_init, geom)
    transposed = [[c_ref[j][i] for j in range(geom.n_max)]
                  for i in range(geom.m)]
    exp = rvv_ref.clayout_reg_image(c_ref, geom)
    xpose = rvv_ref.clayout_reg_image(transposed, geom)
    assert exp != xpose, geom.describe()   # clayout_capable guarantees M >= 2
    assert geom.m <= 256, "the c_idx_* tables are .byte"

    head = [
        f"# {name}: {geom.describe()}",
        "#",
        "# Generated by ime_tests.py from rvv_ref.py -- do not edit by hand,",
        "# and do not edit rvv_ref.py: it is the judge, not the defendant.",
        "#",
        "# Round five: the C tile is read back with an architectural",
        f"# vse{geom.sew}.v of the whole EMUL_C={geom.emul_c} register group,"
        f" not with",
        f"# {geom.store_mnemonic}, so the register-side layout "
        f"(Sail mat_C_idx) is observed",
        "# rather than cancelled by a matching permutation on both sides.",
        "#",
        f"# IME instructions are emitted as .insn (encodings from Zvvm "
        f"v{ime.SPEC_VERSION}):",
    ]
    for mnemonic in (geom.load_mnemonic, geom.mnemonic):
        head.append(f"#   {mnemonic}")
    head += [
        "",
        "    .text",
        "    .balign 4",
        "    .globl main",
        "main:",
        "    addi  sp, sp, -64",
        "    sd    ra, 56(sp)",
        "    sd    s1, 48(sp)         # carries the exit status past printf",
        "    sd    s2, 40(sp)",
        "    sd    s3, 32(sp)",
        "    sd    s4, 24(sp)",
        "    sd    s5, 16(sp)",
        "    sd    s6, 8(sp)",
        "    sd    s7, 0(sp)",
        f"    li    t0, {MSTATUS_VS_INITIAL}",
        "    csrs  mstatus, t0        # enable vector state",
    ]

    body = (_ime_path(geom, alloc, emit_store=False)
            + _clayout_readback(geom, alloc)
            + _clayout_compare(geom)
            + _clayout_epilogue(geom))

    data = [
        "", "    .data", "    .balign 8",
        f'.Lfmt_pass:  .asciz "TITAN PASS {geom.describe()}\\n"',
        f'.Lfmt_skip:  .asciz "TITAN SKIP lambda=%d (requested {geom.lam}) '
        f'imm=%d {geom.describe()}\\n"',
        f'.Lfmt_fail:  .asciz "TITAN FAIL row=%d col=%d {geom.describe()}\\n"',
        f'.Lfmt_diff:  .asciz "TITAN DIFF r=%d c=%d '
        f'exp={_hex_fmt(geom.sew)} got={_hex_fmt(geom.sew)}\\n"',
        # A "line" of this dump is M consecutive *flat register-group*
        # elements, not a C row -- which is the whole point, so the label is
        # neither "r" nor "c".
        f'.Lfmt_cdump: .asciz "TITAN CDUMP f=%d:"',
        f'.Lfmt_cref:  .asciz "TITAN CREF f=%d:"',
        f'.Lfmt_clayout_t: .asciz "TITAN CLAYOUT transposed '
        f'{geom.describe()}\\n"',
        f'.Lfmt_clayout_o: .asciz "TITAN CLAYOUT other '
        f'{geom.describe()}\\n"',
        f'.Lfmt_elem:  .asciz " {_hex_fmt(geom.sew)}"',
        '.Lfmt_nl:    .asciz "\\n"',
        "    .balign 8",
    ]
    data += _matrix_data("c_init", c_init, geom.sew)
    for label, mat in (("mat_a_tile", a), ("mat_b_tile", b)):
        width = geom.ab_linesize
        buf = rvv_ref.tile_layout_buffer(mat, width, geom)
        chunked = [buf[i:i + width] for i in range(0, len(buf), width)]
        data += ["    .balign 8"] + _matrix_data(label, chunked, geom.eew_ab)
    # The two register images, chunked M elements to a line so that a diff of
    # the generated assembly is readable: line n of c_exp is exactly what
    # `TITAN CREF f=n:` prints.
    for label, image in (("c_exp", exp), ("c_xpose", xpose)):
        rows = [image[i:i + geom.m] for i in range(0, len(image), geom.m)]
        data += ["    .balign 8"] + _matrix_data(label, rows, geom.sew)
    # flat register-group element index -> the (row, column) of the C tile
    # element the spec puts there.  The inverse of mat_C_idx, tabulated.
    inv_r = [0] * (geom.m * geom.n_max)
    inv_c = [0] * (geom.m * geom.n_max)
    for i in range(geom.m):
        for j in range(geom.n_max):
            flat = rvv_ref.c_element_index(i, j, geom)
            inv_r[flat], inv_c[flat] = i, j
    for label, table in (("c_idx_r", inv_r), ("c_idx_c", inv_c)):
        rows = [table[i:i + geom.m] for i in range(0, len(table), geom.m)]
        data += ["    .balign 8"] + _matrix_data(label, rows, 8)
    data += ["    .balign 8", "c_raw:",
             f"    .zero {geom.m * geom.n_max * geom.sew // 8}"]

    return "\n".join(head + body + data) + "\n"



# ---------------------------------------------------------------------------
# round six: the MX integer-input, FP-accumulate family
# ---------------------------------------------------------------------------
#
# vfwimmacc.vv / vfqimmacc.vv / vf8wimmacc.vv, all at vm=0.  rvv_ref owns the
# architecture (see its "round six" section, transcribed from Sail
# `int_scaled_gemm`, spec 5373-5410); everything here is program *shape*.
#
# Four tiers, each a separate verdict and each with its own reason to exist:
#
#   ime_mx_   exactness.  C = +0.0, one microscaling block, both E8M0 scale
#             bytes 0x7F (= 2**0).  Under those conditions Sail 5396-5399
#             collapses to `acc = int_to_fp(dot)` with no floating-point
#             rounding anywhere, so the on-DUT reference is an exact integer
#             dot product and one integer-to-float conversion -- which is
#             what makes the tier work at binary16 and bfloat16, where the
#             baseline -march has no scalar arithmetic at all.
#   ime_mxs_  scale.  eA, eB = 127 +/- small over several blocks, still
#             exact.  This is the tier that drives the block loop and the
#             R-strided v0 layout.
#   ime_mxn_  NaN.  0xFF planted in one scale byte, and the spec 2021-2024
#             case where both encoded scales are finite but the converted
#             pair is +0 x +inf.
#   ime_mxl_  legality.  The negative cases that must take an
#             illegal-instruction trap -- and, on the same three funct6
#             values, the vm=1 words that must *not*.
#
# Nothing in this section is reachable from a round one to five geometry:
# every entry point requires `geom.kind == "mx"`, which
# TileGeometry.validate only accepts for the five (W, SEW) cells of
# rvv_ref.MX_CELLS.  The earlier tiers' programs are byte-identical.

#: What goes in a v0 scale byte that the architecture must never read.
#: 0xFF is the E8M0 NaN encoding (spec 1993), so an implementation that
#: reads one of these positions produces the default NaN instead of a
#: number and the tier goes red on the spot.  The positions are:
#:
#:   * ``s >= S_blocks`` -- padding within a row of the scale array.  Spec
#:     2222-2224: "Scale elements at block indices at or beyond `S_blocks`
#:     are ignored."
#:   * the ``scale_B`` byte of a pair at ``m >= N`` -- spec 2176-2181 says
#:     those fields are ignored and do not affect `fflags`.
#:
#: Filling them with an inert 0x7F would test nothing; filling them with
#: 0xFF is the cheapest way to turn "ignored" from a claim into a check.
MX_POISON_SCALE = rvv_ref.MX_E8M0_NAN


def mx_scale_image(geom: TileGeometry,
                   scales_a: Sequence[Sequence[int]],
                   scales_b: Sequence[Sequence[int]]) -> List[int]:
    """The ``v0`` paired-scale register image, as VLEN/16 16-bit elements.

    Spec 2129-2170 and Sail ``read_block_scales`` (5097-5122).  ``sw = 8``
    and the pair width ``pw = 2*sw = 16``, so v0 is read as 16-bit elements:
    the low byte [7:0] is ``scale_A`` -- one *row* of the A tile -- and the
    high byte [15:8] is ``scale_B`` -- one *column* of B^T.  Pair element
    ``p = m*R + s`` with the row stride ``R = LAMBDA * SEW / pw``
    (:func:`rvv_ref.mx_scale_stride`, spec 2139), and the Sail reads the A
    scale at ``i*R + s`` and the B scale at ``j*R + s`` -- the same index
    function of a row/column number and a block number, out of the *same*
    register.  There are not two scale registers.

    ``M * R == VLEN / pw`` exactly (spec 2192-2197), so the image is one
    whole vector register with nothing left over; that identity is asserted
    here rather than assumed, because if it ever fails the padding rule
    below is silently indexing off the end of the array.

    *scales_a* and *scales_b* are both indexed ``[m][s]`` over the full
    ``M x R`` array, padding included -- the caller decides what goes in the
    padding, and :data:`MX_POISON_SCALE` is what the tiers put there.
    """
    r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
    pairs = geom.vlen // rvv_ref.MX_PAIR_WIDTH
    assert geom.m * r == pairs, (geom.describe(), r, pairs)
    image = [0] * pairs
    for m in range(geom.m):
        for s in range(r):
            a, b = scales_a[m][s], scales_b[m][s]
            assert 0 <= a <= 0xFF and 0 <= b <= 0xFF, (m, s, a, b)
            image[rvv_ref.mx_pair_index(m, s, r)] = a | (b << 8)
    return image


def mx_describe(geom: TileGeometry, altfmt: int, bs: int) -> str:
    """The verdict-line geometry string, plus the two vtype bits it needs.

    ``TileGeometry.describe`` cannot carry these: ``altfmt`` (the C
    accumulator format) and ``bs`` (the microscaling block size) are vtype
    fields, not tile geometry, and two round-six programs over the *same*
    geometry differ in nothing else -- at (W=2, SEW=16) and (W=4, SEW=16)
    altfmt picks binary16 or bfloat16 (spec 1092-1106) and the two are the
    same 16 bits of storage.  Without this clause the two programs would
    print identical verdict lines and a failure could not be attributed.

    The clause goes at the very end, after ``describe``'s own, so that
    helpers._GEOM_RE still sees ``VLEN=.. SEW=.. LAMBDA=..`` as three
    adjacent fields.
    """
    _ewidth, fmt = rvv_ref.mx_legal_cell(geom.w, geom.sew, altfmt)
    return (f"{geom.describe()} MXC={fmt} altfmt={altfmt} bs={bs} "
            f"BLK={rvv_ref.mx_block_size(bs)}")


def _pow2_bits(exponent: int, width: int, fmt: str):
    """Bit pattern of an exactly representable ``2**exponent``, else None.

    A normal power of two only: biased exponent in ``[1, emax-1]``, so no
    subnormal, no infinity, no rounding.  That is deliberately narrower than
    what E8M0 can encode -- :func:`rvv_ref.mx_decode_scale` will happily
    return +0 or +inf for an out-of-range scale (spec 2008-2019), and those
    cases belong to the NaN tier, not to the two exact ones.
    """
    _ebits, prec, bias, emax = rvv_ref.fp_fields(width, fmt)
    biased = exponent + bias
    if not 1 <= biased <= emax - 1:
        return None
    return biased << (prec - 1)


@dataclass(frozen=True)
class MxPlan:
    """Everything one round-six directed program is built from.

    Deliberately not a "case" tuple like rounds one to four use: a
    microscaled program needs the two scale arrays as well as A, B and C,
    and it needs the two vtype bits (``altfmt``, ``bs``) that the geometry
    does not carry.  Bundling them means :func:`mx_element_plan` -- the one
    place that decides what the DUT recomputes and what it is told -- takes
    a single argument and cannot be handed a half-matched set.

    ``scales_a`` and ``scales_b`` are both ``M x R``: the *whole* v0 array
    including the padding columns, indexed ``[m][s]``.
    ``rvv_ref.int_scaled_gemm_reference`` reads only ``s < S_blocks`` of
    them, which is the padding rule (spec 2222-2224) expressed by the
    reference simply declining to look.
    """

    geom: TileGeometry
    altfmt: int
    bs: int
    a: Matrix
    b: Matrix
    c: Matrix
    scales_a: List[List[int]]
    scales_b: List[List[int]]
    tier: str            # "mx" / "mxs" / "mxn" -- names the prefix and intent
    tag: str             # what this particular program is for, in one phrase

    @property
    def fmt(self) -> str:
        return rvv_ref.mx_legal_cell(self.geom.w, self.geom.sew,
                                     self.altfmt)[1]

    @property
    def blocks(self) -> int:
        return rvv_ref.mx_block_count(self.geom.k_eff,
                                      rvv_ref.mx_block_size(self.bs))

    def describe(self) -> str:
        return mx_describe(self.geom, self.altfmt, self.bs)


#: What the reference path does for one C tile element.
#:
#:   "compute"  -- recompute it on the DUT: an exact integer dot product
#:                 over the whole K interval, one integer-to-float
#:                 conversion, and one exponent addition for the common
#:                 block scale.  Differential, which is the point.
#:   "literal"  -- the architectural answer is not something a baseline
#:                 rv64i sequence can recompute (a default NaN, an infinity,
#:                 a signed zero out of a +0 x +inf pair), so the program
#:                 carries rvv_ref's answer as a constant.  Round five's
#:                 C-layout tier is the precedent: a non-differential
#:                 expectation is the only way to observe some things at
#:                 all, and it is honest as long as it is *named*.
#:   "copy"     -- a C tile tail column, j >= N.  vta=0 leaves it
#:                 undisturbed, so the reference copies c_init across and
#:                 the comparison stays exact over the whole physical tile.
MX_MODE_COMPUTE, MX_MODE_LITERAL, MX_MODE_COPY = 0, 1, 2


def mx_element_plan(plan: MxPlan):
    """``(reference tile, {(i, j): (mode, integer, exponent, literal)})``.

    The reference tile is :func:`rvv_ref.int_scaled_gemm_reference`, full
    stop -- this function never computes an expected value of its own.  What
    it decides is only *how the program will arrive at that value*, and it
    proves the choice: an element is marked "compute" only after the closed
    form the emitted assembly implements has been checked, here, to equal
    the reference bit for bit.  A judge that disagreed with rvv_ref would
    therefore fail to generate rather than emit a wrong expectation.

    The closed form is::

        T = sum over blocks s of int_block_dot(i, j, block s)
        result = int_to_fp(T, EEW_C, fmt_C)  +  E << (prec - 1)

    and it is valid exactly when

      * C[i][j] is +0.0 -- Sail 5395 seeds `acc` from C, and a nonzero seed
        would need a floating-point add the reference path cannot do;
      * every block's paired scale is a *normal power of two* 2**E with the
        same E (so `fp_mul(blk_scale, fp_sum)` is an exponent addition and
        never rounds, and the sum over blocks may be reassociated); and
      * every partial sum is exactly representable, which is what
        :func:`rvv_ref.mx_exact_operand_bound` and the per-K clamp in
        :func:`_mx_operand_bound` buy.

    Anything else falls through to "literal".  The fall-through is *not* a
    quiet fix: each tier asserts how many of each mode it expects, so a
    bound that stops being exact turns into a failing self-test rather than
    into a tier that silently stopped being differential.
    """
    geom, bs, altfmt = plan.geom, plan.bs, plan.altfmt
    width = geom.sew
    _ewidth, fmt = rvv_ref.mx_legal_cell(geom.w, width, altfmt)
    _ebits, prec, _bias, _emax = rvv_ref.fp_fields(width, fmt)
    block_size = rvv_ref.mx_block_size(bs)
    blocks = plan.blocks
    ref = rvv_ref.int_scaled_gemm_reference(
        plan.a, plan.b, plan.c, plan.scales_a, plan.scales_b, geom,
        bs=bs, altfmt=altfmt)

    out = {}
    for i in range(geom.m):
        for j in range(geom.n_max):
            want = ref[i][j]
            if j >= geom.n:
                assert want == plan.c[i][j], (i, j)   # vta=0, spec 1811-1813
                out[(i, j)] = (MX_MODE_COPY, 0, 0, 0)
                continue
            exps = set()
            for s in range(blocks):
                ea, eb = plan.scales_a[i][s], plan.scales_b[j][s]
                exponent = (ea - rvv_ref.MX_E8M0_BIAS) \
                    + (eb - rvv_ref.MX_E8M0_BIAS)
                pattern = _pow2_bits(exponent, width, fmt)
                blk, is_nan = rvv_ref.mx_block_scale(ea, eb, width, fmt)
                if is_nan or pattern is None or blk != pattern:
                    exps = None
                    break
                exps.add(exponent)
            if exps is None or len(exps) != 1 or plan.c[i][j] != 0:
                out[(i, j)] = (MX_MODE_LITERAL, 0, 0, want)
                continue
            exponent = exps.pop()
            total = sum(
                rvv_ref.mx_int_block_dot(
                    plan.a, plan.b, i, j,
                    *rvv_ref.mx_block_interval(s, block_size, geom.k_eff))
                for s in range(blocks))
            if not _mx_dut_representable(geom, plan.a, plan.b, i, j,
                                         total, prec):
                out[(i, j)] = (MX_MODE_LITERAL, 0, 0, want)
                continue
            bits = rvv_ref.mx_int_to_fp(total, width, fmt)
            if total:
                bits += exponent << (prec - 1)
            if bits != want:
                out[(i, j)] = (MX_MODE_LITERAL, 0, 0, want)
                continue
            out[(i, j)] = (MX_MODE_COMPUTE, total,
                           exponent << (prec - 1), 0)
    return ref, out


def _mx_dut_representable(geom: TileGeometry, a: Matrix, b: Matrix,
                          i: int, j: int, total: int, prec: int) -> bool:
    """Can the emitted reference path actually compute this element?

    Two conditions, and neither is implied by "the closed form equals
    rvv_ref".  That distinction is the one a negative control caught: with a
    single block, ``int_to_fp(dot)`` *is* the architectural answer whether or
    not the dot fits the significand, so comparing the closed form against
    the reference says nothing about whether the DUT-side assembly can
    reproduce it.  What the assembly needs is:

      * ``|T| <= 2**prec``, because :func:`_mx_convert_path` converts by
        splicing bits rather than by rounding.  Above that bound the
        conversion is a real rounding and the splice is simply wrong; and
      * every per-chunk partial sum inside ``[-2**(SEW-1), 2**(SEW-1))``,
        because ``vredsum.vs`` reduces modulo 2**SEW and ``vmv.x.s``
        sign-extends what survives.  The chunking is the one
        :func:`_mx_dot_path` emits, so this checks the sums the program will
        actually form, not an idealised whole-row sum.

    Elements that fail either test fall back to a literal, and each tier
    asserts that none of its active elements did -- so a bound that stopped
    being exact is a failing self-test, not a silently weakened tier.
    """
    if abs(total) > (1 << prec):
        return False
    chunk = min(geom.elems_per_reg, geom.k_eff)
    limit = 1 << (geom.sew - 1)
    for start in range(0, geom.k_eff, chunk):
        partial = sum(a[i][k] * b[j][k]
                      for k in range(start, min(start + chunk, geom.k_eff)))
        if not -limit <= partial < limit:
            return False
    return True


def _mx_operand_bound(geom: TileGeometry, altfmt: int, bs: int) -> int:
    """Largest |A|, |B| that keeps every *prefix* of the K sum exact.

    :func:`rvv_ref.mx_exact_operand_bound` bounds one block; the reference
    path adds the blocks up in one integer register and then converts once,
    so what has to stay inside the significand is the running total over the
    whole K interval, which is up to ``S_blocks`` times larger.  Clamping to
    both is the honest thing: the tier's exactness claim is about the value
    the program actually computes.

    Both bounds are at least 2 for every one of the seven cells (the binding
    one is bfloat16, 8 significand bits), so no cell degenerates to an
    all-zero tile -- which would pass anything.
    """
    ewidth, fmt = rvv_ref.mx_legal_cell(geom.w, geom.sew, altfmt)
    _ebits, prec, _bias, _emax = rvv_ref.fp_fields(geom.sew, fmt)
    native = (1 << (ewidth - 1)) - 1
    bound = 0
    while bound + 1 <= native \
            and geom.k_eff * (bound + 1) ** 2 <= (1 << prec):
        bound += 1
    bound = min(bound, rvv_ref.mx_exact_operand_bound(
        geom.w, geom.sew, altfmt, bs))
    if bound < 1:
        raise ValueError(
            f"{mx_describe(geom, altfmt, bs)}: no nonzero operand bound "
            f"keeps the whole K interval exact")
    return bound


def _mx_configure(geom: TileGeometry, *, lmul: int, vl: int, altfmt: int,
                  bs: int, comment: str) -> List[str]:
    """:func:`_configure` with the two round-six vtype bits set.

    Separate from ``_configure`` rather than a pair of new keyword arguments
    on it, because ``_configure`` emits the literal text of every round one
    to five program and that text must not move.
    """
    return [
        f"    # {comment}",
        f"    li    t0, {vl}",
        f"    li    t1, 0x{vtype_value(geom, lmul=lmul, altfmt=altfmt, bs=bs):x}",
        "    vsetvl x0, t0, t1",
    ]


def _mx_ime_path(geom: TileGeometry, alloc: VectorAlloc, *, altfmt: int,
                 bs: int) -> List[str]:
    """The round-one tile sequence, plus v0, minus the A tile's old home.

    Structurally :func:`_ime_path`: configure for the C transfer, load C,
    configure for A/B, load both tiles, configure for compute, one
    multiply-accumulate, configure back, store C.  Two things are new.

    First, ``v0`` carries the paired E8M0 block scales and is loaded with an
    ordinary ``vle16.v`` at LMUL=1, VL=VLEN/16 -- an architectural load of
    the whole register, with no tile semantics, because the scale array is
    not a tile: it is one register of 16-bit pairs (spec 2192-2197).  That
    is why ``VectorAlloc.allocate`` is called with ``reserve_v0=True`` here
    and why the A tile starts at v[LMUL] instead of v0.

    Second, the compute configuration sets ``vtype.bs`` and ``vtype.altfmt``.
    Neither is in the instruction encoding: ``bs`` is ``vtype[XLEN-5]``
    (spec 1160-1176) and ``altfmt`` is the base Zvfbfa field.  They are set
    on *every* vsetvl in the program, not just the compute one, because
    `vsetvli`/`vsetivli` cannot reach them and a program that set them once
    and then reconfigured for the C transfer would be relying on retention
    rules (spec 888-889) that are not what is under test here.

    The multiply-accumulate is emitted at vm=0 -- ime_encodings hardwires it,
    so there is no operand to get wrong -- and at vm=1 the same funct6 would
    be the round-one-to-three integer form.  ``ime_mxl_`` is the tier that
    holds the decoder to that.
    """
    lam_imm = 0  # 0 as an instruction immediate means "use vtype.lambda"
    out: List[str] = ["", "    # ---- IME path (microscaled, vm=0) ----"]
    out += _mx_configure(geom, lmul=geom.lmul_c, vl=geom.vl_c_full,
                         altfmt=altfmt, bs=bs,
                         comment=f"C tile transfer config (LMUL=EMUL_C="
                                 f"{geom.emul_c}, VL={geom.vl_c_full})")
    out += _check_lambda_retained(geom)
    out += [
        "    la    a0, c_init",
        f"    li    a1, {geom.m}          # LD = M: row-major M x M block",
        f"    {ime.insn(geom.load_mnemonic, vd=alloc.c, rs1=RS1_ADDR, rs2=RS2_LD, vm=1, **{'lambda': lam_imm})}"
        f"    # {geom.load_mnemonic} v{alloc.c}, (a0), a1",
    ]
    out += _mx_configure(geom, lmul=geom.lmul,
                         vl=geom.lmul * geom.elems_per_reg,
                         altfmt=altfmt, bs=bs,
                         comment=f"A/B config (LMUL={geom.lmul}, full VL)")
    for label, base in (("mat_a_tile", alloc.a), ("mat_b_tile", alloc.b)):
        out += [
            f"    la    a0, {label}",
            f"    li    a1, {geom.linesize}",
            f"    {ime.insn(geom.load_mnemonic, vd=base, rs1=RS1_ADDR, rs2=RS2_LD, vm=1, **{'lambda': lam_imm})}"
            f"    # {geom.load_mnemonic} v{base}, (a0), a1",
        ]
    r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
    pairs = geom.vlen // rvv_ref.MX_PAIR_WIDTH
    out += [
        "",
        f"    # the paired E8M0 block scales into v0: {pairs} x 16-bit",
        f"    # elements, pair p = m*R + s with R = LAMBDA*SEW/pw = {r}",
        f"    # (spec 2129-2170, Sail 5097-5122).  A plain vle16.v at LMUL=1:",
        "    # the scale array is one register of pairs, not a tile.",
        f"    li    t0, {pairs}",
        "    vsetvli t1, t0, e16, m1, ta, ma",
        "    la    a0, v0_scales",
        "    vle16.v v0, (a0)",
    ]
    out += _mx_configure(geom, lmul=geom.lmul, vl=geom.vl,
                         altfmt=altfmt, bs=bs,
                         comment=f"compute config (VL={geom.vl} -> "
                                 f"N={geom.n}, bs={bs}, altfmt={altfmt})")
    out += [
        f"    {ime.insn(geom.mnemonic, vd=alloc.c, vs1=alloc.a, vs2=alloc.b)}"
        f"    # {geom.mnemonic} v{alloc.c}, v{alloc.a}, v{alloc.b}, v0"
        f"  (vm=0)",
    ]
    out += _mx_configure(geom, lmul=geom.lmul_c, vl=geom.vl_c_full,
                         altfmt=altfmt, bs=bs,
                         comment="back to the C tile config to store")
    out += [
        "    la    a0, c_ime",
        f"    li    a1, {geom.m}",
        f"    {ime.insn(geom.store_mnemonic, vs3=alloc.c, rs1=RS1_ADDR, rs2=RS2_LD, vm=1, **{'lambda': lam_imm})}"
        f"    # {geom.store_mnemonic} v{alloc.c}, (a0), a1",
    ]
    return out


def _mx_dot_path(geom: TileGeometry, elements) -> List[str]:
    """Phase one of the reference: the exact integer dot products.

    One ``vmul.vv`` + ``vredsum.vs`` per K chunk at vtype.SEW, summed into a
    scalar register and spilled to ``ref_int`` as a 64-bit value.  This is
    round two's widening reference path (:func:`_rvv_path_widening`) with
    the accumulate-into-C step removed, and it is reused rather than rewritten
    for the reason that path gives: ``mat_a`` / ``mat_b`` carry each narrow
    input already sign-extended to a SEW-wide word, so an ordinary SEW-wide
    ``vmul.vv`` reproduces the exact product.

    The modular arithmetic of ``vredsum.vs`` is not a problem *and is not
    being relied on to wrap*: the operand bound
    (:func:`_mx_operand_bound`) keeps |T| <= 2**prec < 2**(SEW-1), so every
    chunk partial and the total are exact signed values and ``vmv.x.s``
    sign-extends them faithfully.  Sail ``int_block_dot`` (5128-5147)
    returns an unbounded mathematical integer with no reduction at all, and
    this tier stays inside the range where the two agree -- deliberately, so
    that the conversion afterwards has nothing to round.

    Note what is *absent*: no LMUL step loop and no block-and-step
    intersection.  ``int_scaled_gemm`` has neither (spec 2030-2060 belongs
    to ``fp_scaled_gemm``), and because every block of these tiers shares one
    exponent the per-block dots may be added up as one sum over the whole K
    interval.  :func:`mx_element_plan` proves that per element against
    rvv_ref before this is allowed to run.
    """
    esz = geom.sew // 8
    chunk = min(geom.elems_per_reg, geom.k_eff)
    assert geom.k_eff % chunk == 0, geom.describe()
    nchunks = geom.k_eff // chunk
    out: List[str] = [
        "",
        "    # ---- reference path, phase 1: exact integer dot products ----",
        f"    li    t0, {chunk}",
        f"    vsetvli t1, t0, e{geom.sew}, m1, ta, ma",
        "    la    a2, mat_a",
        "    la    a3, mat_b",
        "    la    a4, ref_int",
    ]
    for i in range(geom.m):
        for j in range(geom.n_max):
            if elements[(i, j)][0] != MX_MODE_COMPUTE:
                continue
            out += [f"    li    t6, 0              # T for C[{i},{j}]"]
            for c in range(nchunks):
                out += [
                    f"    li    t0, {(i * geom.k_eff + c * chunk) * esz}",
                    "    add   t2, a2, t0",
                    f"    li    t0, {(j * geom.k_eff + c * chunk) * esz}",
                    "    add   t3, a3, t0",
                    f"    vle{geom.sew}.v v1, (t2)",
                    f"    vle{geom.sew}.v v2, (t3)",
                    "    vmul.vv v3, v1, v2",
                    "    vmv.s.x v5, x0",
                    "    vredsum.vs v4, v3, v5",
                    "    vmv.x.s t5, v4",
                    "    add   t6, t6, t5",
                ]
            out += [
                f"    li    t0, {_c_off(geom, i, j) * 8}",
                "    add   t2, a4, t0",
                "    sd    t6, 0(t2)",
            ]
    return out


def _mx_convert_path(geom: TileGeometry, width: int, fmt: str) -> List[str]:
    """Phase two: int_to_fp, the block-scale exponent, and the tail copy.

    A loop, not an unrolled body, and the distinction from :func:`_compare`
    (which is unrolled precisely because it is the judge) is that this loop
    does the *same* thing to every element: it reads three tables the
    generator emitted and writes one result.  There is no per-element
    control flow to get wrong, and 256 inlined copies of a 20-instruction
    conversion would triple the size of every program in the tier.

    The conversion itself is a bit-splice, not a rounding routine, and that
    is the whole reason the exactness tier exists.  ``|T| <= 2**prec`` makes
    ``int_to_fp`` exact, so:

        sign     = T < 0
        e        = floor(log2(|T|))
        fraction = (|T| << (prec-1-e)) mod 2**(prec-1)
        bits     = sign << (width-1) | (e + bias) << (prec-1) | fraction

    with no round bit, no sticky bit and no tie rule anywhere -- which is
    what lets the tier run at binary16 and bfloat16, where rv64imafd has no
    scalar arithmetic and therefore no ``fcvt`` to call.  ``e == prec``
    (i.e. |T| == 2**prec exactly) is the one case where the shift would go
    negative; it is clamped to zero, and the mask then removes the hidden
    bit for free.

    The exponent of the common block scale is added afterwards, as an
    integer addition into the exponent field.  That is exact because
    :func:`mx_element_plan` has already established that every block's
    paired scale is the *normal* power of two ``2**E``, so
    ``fp_mul(2**E, x)`` is an exponent shift and cannot round or overflow.
    ``T == 0`` skips it: ``int_to_fp(0)`` is +0.0 and ``2**E * (+0.0)`` is
    +0.0, whose exponent field must stay zero.
    """
    _ebits, prec, bias, _emax = rvv_ref.fp_fields(width, fmt)
    sfx = _SEW_SUFFIX[width]
    esz = width // 8
    shift = {2: 1, 4: 2, 8: 3}[esz]
    frac_shift = 64 - (prec - 1)
    total = geom.m * geom.n_max
    return [
        "",
        f"    # ---- reference path, phase 2: int_to_fp to {fmt} ----",
        "    li    s4, 0              # flat C element index",
        ".Lmxcvt:",
        f"    li    t0, {total}",
        "    bge   s4, t0, .Lmxcvt_done",
        "    slli  s5, s4, 3          # 8-byte stride of the side tables",
        "    la    t1, ref_mode",
        "    add   t1, t1, s4",
        "    lbu   t2, 0(t1)",
        f"    li    t0, {MX_MODE_LITERAL}",
        "    beq   t2, t0, .Lmxcvt_lit",
        f"    li    t0, {MX_MODE_COPY}",
        "    beq   t2, t0, .Lmxcvt_copy",
        "",
        "    # int_to_fp(T): exact, so a bit-splice with no rounding",
        "    la    t1, ref_int",
        "    add   t1, t1, s5",
        "    ld    a6, 0(t1)          # T, the exact integer dot product",
        "    li    a7, 0",
        "    beqz  a6, .Lmxcvt_store  # T = 0 -> +0.0, exponent stays clear",
        "    mv    t1, a6",
        "    li    t0, 0",
        "    bge   a6, x0, .Lmxcvt_mag",
        "    sub   t1, x0, a6         # |T|",
        "    li    t0, 1",
        ".Lmxcvt_mag:",
        f"    slli  t0, t0, {width - 1}    # sign bit in place",
        "    li    t2, 0              # e",
        "    mv    t3, t1",
        ".Lmxcvt_msb:",
        "    li    t4, 1",
        "    beq   t3, t4, .Lmxcvt_norm",
        "    srli  t3, t3, 1",
        "    addi  t2, t2, 1",
        "    j     .Lmxcvt_msb",
        ".Lmxcvt_norm:",
        f"    li    t4, {prec - 1}",
        "    sub   t4, t4, t2         # prec-1-e",
        "    bge   t4, x0, .Lmxcvt_shift",
        "    li    t4, 0              # |T| = 2**prec: hidden bit only",
        ".Lmxcvt_shift:",
        "    sll   t5, t1, t4",
        f"    slli  t5, t5, {frac_shift}",
        f"    srli  t5, t5, {frac_shift}   # fraction, hidden bit dropped",
        f"    addi  t2, t2, {bias}",
        f"    slli  t2, t2, {prec - 1}",
        "    add   a7, t0, t2",
        "    add   a7, a7, t5",
        "",
        "    # + the common block-scale exponent E (exact: 2**E is normal)",
        "    la    t1, ref_bump",
        "    add   t1, t1, s5",
        "    ld    t0, 0(t1)",
        "    add   a7, a7, t0",
        "    j     .Lmxcvt_store",
        "",
        ".Lmxcvt_lit:",
        "    # rvv_ref's answer, carried as a constant: a default NaN, an",
        "    # infinity or a signed zero that no rv64i sequence recomputes.",
        "    la    t1, ref_lit",
        "    add   t1, t1, s5",
        "    ld    a7, 0(t1)",
        "    j     .Lmxcvt_store",
        "",
        ".Lmxcvt_copy:",
        "    # C tile tail column, j >= N: vta=0 leaves it undisturbed.",
        "    la    t1, c_init",
        f"    slli  t0, s4, {shift}",
        "    add   t1, t1, t0",
        f"    l{sfx}    a7, 0(t1)",
        "",
        ".Lmxcvt_store:",
        "    la    t1, c_rvv",
        f"    slli  t0, s4, {shift}",
        "    add   t1, t1, t0",
        f"    s{sfx}    a7, 0(t1)",
        "    addi  s4, s4, 1",
        "    j     .Lmxcvt",
        ".Lmxcvt_done:",
    ]


def _dword_table(label: str, values: Sequence[int], per_line: int) -> List[str]:
    """A ``.dword`` table, *per_line* entries to a line, two's complement.

    Separate from :func:`_matrix_data` because these tables are 64-bit
    regardless of SEW: ``ref_int`` holds an exact integer dot product and
    ``ref_bump`` an exponent-field delta that is negative whenever the
    combined block scale is below 1.0, and both are read with ``ld``.
    """
    mask = (1 << 64) - 1
    lines = [f"{label}:"]
    for start in range(0, len(values), per_line):
        row = values[start:start + per_line]
        lines.append("    .dword " + ", ".join(f"0x{v & mask:x}" for v in row))
    return lines


def _mx_tile_data(label: str, mat: Matrix, geom: TileGeometry) -> List[str]:
    """The A or B tile in tile-load memory layout, at the logical width.

    Identical to what :func:`emit_test` emits for a widening geometry, with
    one addition: at ``EEW_A = 4`` (MXINT4, the W=4/SEW=16 and W=8/SEW=32
    cells) there is no 4-bit assembler directive, so each line is packed two
    elements to a byte with the even index in the low nibble -- spec
    1206-1219, via :func:`rvv_ref.mx_pack_int4`, which is the same authority
    the reference model reads the packing from.

    The line length works out either way: a tile line is ``ab_linesize``
    logical elements = ``linesize`` storage elements of SEW bits, so
    ``K_eff/2`` bytes at EEW_A=4 is exactly ``linesize * SEW/8``.  That
    identity is asserted rather than trusted, because if it were false the
    lines would overlap and every element but the first row would be wrong
    in a way that looks like a tile-addressing bug in the DUT.
    """
    width = geom.ab_linesize
    buf = rvv_ref.tile_layout_buffer(mat, width, geom)
    chunked = [buf[i:i + width] for i in range(0, len(buf), width)]
    if geom.eew_ab == 4:
        packed = [rvv_ref.mx_pack_int4(line) for line in chunked]
        assert len(packed[0]) == geom.linesize * geom.sew // 8, geom.describe()
        return _matrix_data(label, packed, 8)
    return _matrix_data(label, chunked, geom.eew_ab)


def emit_mx_test(plan: MxPlan, name: str = "ime_mx") -> str:
    """One microscaled differential program for *plan*.

    The same shape every pair program has had since round one -- compute the
    tile twice on the DUT, into ``c_ime`` and ``c_rvv``, and compare the two
    memory images element by element with no tolerance -- with the reference
    half replaced by the two phases :func:`_mx_dot_path` and
    :func:`_mx_convert_path` describe.  The comparison, the failure evidence
    and the verdict contract are :func:`_compare` and :func:`_epilogue`,
    unchanged and uncopied.
    """
    geom = plan.geom
    geom.validate()
    if geom.kind != "mx":
        raise ValueError(f"{geom.describe()}: emit_mx_test needs kind='mx'")
    if geom.emul_c == 16:
        raise ValueError(
            f"{geom.describe()}: EMUL_C=16 has no single-instruction C tile "
            f"transfer (LMUL=16 is not a legal vtype)")
    rvv_ref.mx_check_legality(geom.w, geom.lmul, geom.sew, geom.lam, plan.bs)
    width = geom.sew
    _ewidth, fmt = rvv_ref.mx_legal_cell(geom.w, width, plan.altfmt)
    alloc = VectorAlloc.allocate(geom, reserve_v0=True)
    ref, elements = mx_element_plan(plan)
    desc = plan.describe()

    head = [
        f"# {name}: {desc}",
        "#",
        "# Generated by ime_tests.py from rvv_ref.py -- do not edit by hand,",
        "# and do not edit rvv_ref.py: it is the judge, not the defendant.",
        "#",
        f"# Round six, tier {plan.tier}: {plan.tag}.",
        f"# Blocks: S_blocks={plan.blocks} at block_size="
        f"{rvv_ref.mx_block_size(plan.bs)}; scale row stride R="
        f"{rvv_ref.mx_scale_stride(geom.sew, geom.lam)}.",
        f"# The paired E8M0 block scales are in v0 (spec 2129-2170), so the",
        f"# A tile starts at v{alloc.a} rather than v0.",
        "#",
        f"# IME instructions are emitted as .insn (encodings from Zvvm "
        f"v{ime.SPEC_VERSION}):",
    ]
    for mnemonic in (geom.load_mnemonic, geom.mnemonic, geom.store_mnemonic):
        head.append(f"#   {mnemonic}")
    head += [
        "",
        "    .text",
        "    .balign 4",
        "    .globl main",
        "main:",
        "    addi  sp, sp, -64",
        "    sd    ra, 56(sp)",
        "    sd    s1, 48(sp)         # carries the exit status past printf",
        "    sd    s2, 40(sp)",
        "    sd    s3, 32(sp)",
        "    sd    s4, 24(sp)",
        "    sd    s5, 16(sp)",
        "    sd    s6, 8(sp)",
        "    sd    s7, 0(sp)",
        f"    li    t0, {MSTATUS_VS_INITIAL}",
        "    csrs  mstatus, t0        # enable vector state",
    ]
    # No mstatus.FS: the reference path is integer throughout.  That is the
    # point of the exactness construction, not an oversight -- see
    # _mx_convert_path.

    body = (_mx_ime_path(geom, alloc, altfmt=plan.altfmt, bs=plan.bs)
            + _mx_dot_path(geom, elements)
            + _mx_convert_path(geom, width, fmt)
            + _compare(geom))

    data = [
        "", "    .data", "    .balign 8",
        f'.Lfmt_pass:  .asciz "TITAN PASS {desc}\\n"',
        f'.Lfmt_skip:  .asciz "TITAN SKIP lambda=%d (requested {geom.lam}) '
        f'imm=%d {desc}\\n"',
        f'.Lfmt_fail:  .asciz "TITAN FAIL row=%d col=%d {desc}\\n"',
        f'.Lfmt_diff:  .asciz "TITAN DIFF r=%d c=%d '
        f'exp={_hex_fmt(width)} got={_hex_fmt(width)}\\n"',
        f'.Lfmt_cdump: .asciz "TITAN CDUMP r=%d:"',
        f'.Lfmt_cref:  .asciz "TITAN CREF r=%d:"',
        f'.Lfmt_elem:  .asciz " {_hex_fmt(width)}"',
        '.Lfmt_nl:    .asciz "\\n"',
        "    .balign 8",
    ]
    # Row-major sign-extended copies for the reference path's vmul.vv, and
    # tile-layout copies for the tile loads -- the same two images every
    # widening program carries, from the same rvv_ref authority.
    data += _matrix_data("mat_a", plan.a, width)
    data += _matrix_data("mat_b", plan.b, width)
    data += _matrix_data("c_init", plan.c, width)
    data += ["    .balign 8"] + _mx_tile_data("mat_a_tile", plan.a, geom)
    data += ["    .balign 8"] + _mx_tile_data("mat_b_tile", plan.b, geom)
    image = mx_scale_image(geom, plan.scales_a, plan.scales_b)
    r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
    data += [
        "    .balign 8",
        f"# v0: pair p = m*R + s, R={r}; low byte scale_A, high byte scale_B",
        f"# (spec 2129-2170).  0x{MX_POISON_SCALE:02x} is the E8M0 NaN code "
        f"and marks every",
        "# position the architecture must never read -- padding at "
        "s >= S_blocks",
        "# (spec 2222-2224) and the scale_B field of a pair at m >= N "
        "(spec 2176-2181).",
    ]
    data += _matrix_data("v0_scales",
                         [image[i:i + r] for i in range(0, len(image), r)]
                         if r > 1 else [image], 16)
    # The three side tables the conversion loop reads, one entry per flat C
    # tile element in c_rvv memory order.
    modes, bumps, lits = [], [], []
    for off in range(geom.m * geom.n_max):
        modes.append(0)
        bumps.append(0)
        lits.append(0)
    for i in range(geom.m):
        for j in range(geom.n_max):
            mode, _total, bump, lit = elements[(i, j)]
            off = _c_off(geom, i, j)
            modes[off], bumps[off], lits[off] = mode, bump, lit
    data += ["    .balign 8"]
    data += _matrix_data("ref_mode",
                         [modes[i:i + geom.m]
                          for i in range(0, len(modes), geom.m)], 8)
    data += ["    .balign 8"] + _dword_table("ref_bump", bumps, geom.m)
    data += ["    .balign 8"] + _dword_table("ref_lit", lits, geom.m)
    data += ["    .balign 8", "ref_int:",
             f"    .zero {geom.m * geom.n_max * 8}",
             "    .balign 8", "c_ime:",
             f"    .zero {geom.m * geom.m * width // 8}",
             "    .balign 8", "c_rvv:",
             f"    .zero {geom.m * geom.m * width // 8}"]

    del ref
    return "\n".join(head + body + data) + "\n"


def _mx_emittable(geom: TileGeometry) -> bool:
    """Can a round-six program be built for this geometry at all?

    Two reasons it might not be, both of which ime_stress filters on too:
    EMUL_C=16 has no single-instruction C tile transfer, and
    ``VectorAlloc.allocate(reserve_v0=True)`` needs ``3*LMUL <= 32-EMUL_C``
    because v0 is spoken for.  Neither is quietly swapped for a geometry
    that does fit -- that would be testing something other than what was
    asked for.
    """
    if geom.emul_c == 16:
        return False
    try:
        VectorAlloc.allocate(geom, reserve_v0=True)
    except ValueError:
        return False
    return True


def _mx_scale_arrays(geom: TileGeometry, blocks: int, byte_a, byte_b):
    """``(scales_a, scales_b)``, each the full M x R array with padding.

    *byte_a* and *byte_b* are called as ``f(m, s)`` for the ``s <
    S_blocks`` positions of each row.  Everything else -- the padding
    columns, and the ``scale_B`` field of every pair at ``m >= N`` -- gets
    :data:`MX_POISON_SCALE`.
    """
    r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
    assert blocks <= r, (geom.describe(), blocks, r)
    sa = [[MX_POISON_SCALE] * r for _ in range(geom.m)]
    sb = [[MX_POISON_SCALE] * r for _ in range(geom.m)]
    for m in range(geom.m):
        for s in range(blocks):
            sa[m][s] = byte_a(m, s)
            if m < geom.n:
                sb[m][s] = byte_b(m, s)
    for row in sa + sb:
        assert all(0 <= v <= 0xFF for v in row), row
    return sa, sb


def _mx_operands(geom: TileGeometry, altfmt: int, bs: int,
                 rng: random.Random):
    """Random A and B inside the exactness bound, and a C that is +0.0.

    C is +0.0 in the active window because the closed form the reference
    path implements starts from zero (see :func:`mx_element_plan`), and
    deliberately *not* zero in the tail columns: those are the ones vta=0
    must leave undisturbed, and a tail of zeros against a tile of zeros
    would confirm nothing.
    """
    bound = _mx_operand_bound(geom, altfmt, bs)
    a = [[rng.randint(-bound, bound) for _ in range(geom.k_eff)]
         for _ in range(geom.m)]
    b = [[rng.randint(-bound, bound) for _ in range(geom.k_eff)]
         for _ in range(geom.n_max)]
    mask = (1 << geom.sew) - 1
    c = [[0 if j < geom.n else rng.randint(1, mask) for j in range(geom.n_max)]
         for _ in range(geom.m)]
    return a, b, c


def _mx_geometries(vlen: int, blocks_wanted: str):
    """``(geom, bs)`` pairs whose S_blocks is 1 ("one") or >= 2 ("many").

    bs is enumerated inside the geometry loop rather than outside it because
    it is a vtype bit, not a tile shape: the same geometry can be a
    single-block program at bs=0 and a two-block one at bs=1, and both are
    worth having.  ``mx_check_legality`` decides whether bs=1 is even legal
    here (Sail 5156: ``W*LMUL <= SEW``), so the illegal combinations are
    filtered by the architecture rather than by a hand-written list.
    """
    for geom in rvv_ref.mx_legal_configs(vlen, full_vl_only=True):
        if not _mx_emittable(geom):
            continue
        for bs in (0, 1):
            try:
                rvv_ref.mx_check_legality(geom.w, geom.lmul, geom.sew,
                                          geom.lam, bs)
            except ValueError:
                continue
            blocks = rvv_ref.mx_block_count(geom.k_eff,
                                            rvv_ref.mx_block_size(bs))
            if (blocks == 1) == (blocks_wanted == "one"):
                yield geom, bs


def mx_exact_plans(vlen: int, seed: int = 0):
    """The ``ime_mx_`` tier: one microscaling block, both scales 2**0.

    With ``K_eff <= block_size`` there is exactly one block, and with both
    E8M0 bytes at 0x7F the paired scale is 1.0 exactly, so Sail 5396-5399
    reduces to ``acc = fp_add(+0.0, fp_mul(1.0, int_to_fp(dot)))`` =
    ``int_to_fp(dot)``.  No floating-point rounding survives anywhere in the
    architectural computation, which is the only reason a tier can cover the
    binary16 and bfloat16 accumulator cells at all: the DUT has no scalar
    arithmetic at those widths, and none is needed.

    All seven cells of the encoding map are reached, because `altfmt` is
    enumerated from :func:`rvv_ref.mx_altfmts` -- the table, not a list here.
    """
    rng = random.Random(seed)
    partial_seen = set()
    for geom, bs in _mx_geometries(vlen, "one"):
        shapes = [geom]
        # One partial-N program per (W, SEW, bs).  Two things exist only at
        # N < N_max and are otherwise dead: the C tile tail columns, which
        # vta=0 must leave undisturbed, and the scale_B fields at
        # m >= N, which spec 2176-2181 says "are ignored and do not affect
        # the result or fflags" -- and which these programs fill with the
        # E8M0 NaN code, so "ignored" is a check rather than a claim.
        key = (geom.w, geom.sew, bs)
        if key not in partial_seen and geom.n_max > 1:
            half = TileGeometry(vlen, geom.sew, geom.lam, geom.lmul,
                                (geom.n_max // 2) * geom.lam * geom.lmul,
                                geom.w, "op", "mx")
            try:
                half.validate()
            except ValueError:
                half = None
            if half is not None and _mx_emittable(half):
                partial_seen.add(key)
                shapes.append(half)
        for shape in shapes:
            for altfmt in rvv_ref.mx_altfmts(shape.w, shape.sew):
                a, b, c = _mx_operands(shape, altfmt, bs, rng)
                sa, sb = _mx_scale_arrays(
                    shape, 1,
                    lambda m, s: rvv_ref.MX_E8M0_BIAS,
                    lambda m, s: rvv_ref.MX_E8M0_BIAS)
                yield MxPlan(shape, altfmt, bs, a, b, c, sa, sb, "mx",
                             f"one block, both E8M0 scales 0x7f = 2**0"
                             f"{'' if shape.n == shape.n_max else '; N < N_max, so the C tail and the inactive scale_B fields are live'}")


def mx_scale_plans(vlen: int, seed: int = 100):
    """The ``ime_mxs_`` tier: several blocks, scales 127 +/- small.

    ``eA[i][s] = 127 + dA(i) + g(s)`` and ``eB[j][s] = 127 + dB(j) - g(s)``.
    The ``g(s)`` term is the point: both scale arrays vary along *both* axes,
    so an implementation that indexed v0 as ``s*R + m`` instead of
    ``m*R + s`` reads a different byte and gets a different answer.  It
    cancels in the combined exponent ``E = dA(i) + dB(j)``, which is what
    keeps every block of one output element on the same exponent and
    therefore keeps the whole accumulation exact -- the only way this tier
    can compare bit-for-bit against an integer reference.

    The windows are chosen for the narrowest accumulator, binary16 (5
    exponent bits): ``dA + g`` and ``dB - g`` stay within [-14, 15] so each
    decoded scale is a normal power of two, and ``E`` stays within [-2, 2]
    so ``2**E * T`` with ``|T| <= 2**11`` is normal too.  Wider accumulators
    have strictly more room; the tier does not widen the window for them
    because the same numbers then exercise every cell identically and a
    failure that appears only at binary16 is a range bug, not a layout one.
    """
    rng = random.Random(seed)
    for geom, bs in _mx_geometries(vlen, "many"):
        blocks = rvv_ref.mx_block_count(geom.k_eff,
                                        rvv_ref.mx_block_size(bs))
        mid = (blocks - 1) // 2
        for altfmt in rvv_ref.mx_altfmts(geom.w, geom.sew):
            a, b, c = _mx_operands(geom, altfmt, bs, rng)
            sa, sb = _mx_scale_arrays(
                geom, blocks,
                lambda m, s: rvv_ref.MX_E8M0_BIAS + (m % 3) - 1 + (s - mid),
                lambda m, s: rvv_ref.MX_E8M0_BIAS + (m % 3) - 1 - (s - mid))
            yield MxPlan(geom, altfmt, bs, a, b, c, sa, sb, "mxs",
                         f"{blocks} blocks, eA/eB = 127 +/- small with a "
                         f"per-block term that cancels in E")


def mx_nan_plans(vlen: int, seed: int = 200):
    """The ``ime_mxn_`` tier: NaN block scales and the early exit.

    Three plantings, each testing a different sentence of the spec:

      * ``scale_A[i][0] = 0xFF``.  The first block's scale is the E8M0 NaN
        encoding (spec 1993), so ``read_block_scales`` reports NaN, Sail
        5390-5392 breaks out of the block loop and 5403-5406 writes
        ``fp_defaultNaN``.  The whole of row i goes NaN; every other row is
        untouched, which is the half of the claim that an implementation
        which simply NaN-ed the tile would fail.
      * ``scale_B[j][S_blocks-1] = 0xFF`` at a multi-block geometry.  Same
        outcome, but the poisoned block is not the first one -- an
        implementation that only checks block 0 passes the case above and
        fails this one.  Column j goes NaN.
      * a binary16 accumulator with ``eA = 0x00`` and ``eB = 0xFE``.  Both
        are *finite* E8M0 codes (2**-127 and 2**127), but neither survives
        conversion to binary16: one underflows to +0 and the other overflows
        to +inf, and +0 x +inf is the default NaN with invalid raised --
        spec 2021-2024, "even though both encoded E8M0 scales are finite".
        A model that tests the two bytes for 0xFF instead of testing the
        *product* calls this one wrong.

    Everything the poison touches beyond the targeted element -- the +0 rows
    and +-inf columns the third planting produces -- comes back from
    :func:`rvv_ref.int_scaled_gemm_reference` as a literal, because no
    baseline integer sequence recomputes an infinity.  That is named in the
    program text rather than hidden: see MX_MODE_LITERAL.
    """
    rng = random.Random(seed)
    seen = set()
    for want in ("one", "many"):
        for geom, bs in _mx_geometries(vlen, want):
            for altfmt in rvv_ref.mx_altfmts(geom.w, geom.sew):
                key = (geom.w, geom.sew, altfmt, want)
                if key in seen:
                    continue
                seen.add(key)
                blocks = rvv_ref.mx_block_count(
                    geom.k_eff, rvv_ref.mx_block_size(bs))
                fmt = rvv_ref.mx_legal_cell(geom.w, geom.sew, altfmt)[1]
                ip, jp = geom.m // 2, geom.n // 2
                variants = [
                    ("a0", f"scale_A[{ip}][0] = 0xff: row {ip} is the "
                           f"default NaN, the rest of the tile is not",
                     {("a", ip, 0): rvv_ref.MX_E8M0_NAN}),
                ]
                if blocks >= 2:
                    variants.append(
                        ("bl", f"scale_B[{jp}][{blocks - 1}] = 0xff: the "
                               f"poisoned block is not block 0",
                         {("b", jp, blocks - 1): rvv_ref.MX_E8M0_NAN}))
                if fmt == "binary16":
                    variants.append(
                        ("pm", f"eA=0x00, eB=0xfe at [{ip}][0] / [{jp}][0]: "
                               f"+0 x +inf in binary16, both codes finite",
                         {("a", ip, 0): 0x00, ("b", jp, 0): 0xFE}))
                for suffix, tag, poison in variants:
                    a, b, c = _mx_operands(geom, altfmt, bs, rng)

                    def byte_a(m, s, _p=poison):
                        return _p.get(("a", m, s), rvv_ref.MX_E8M0_BIAS)

                    def byte_b(m, s, _p=poison):
                        return _p.get(("b", m, s), rvv_ref.MX_E8M0_BIAS)

                    sa, sb = _mx_scale_arrays(geom, blocks, byte_a, byte_b)
                    yield suffix, MxPlan(geom, altfmt, bs, a, b, c, sa, sb,
                                         "mxn", tag)


# ---------------------------------------------------------------------------
# round six: the legality tier
# ---------------------------------------------------------------------------
#
# Every other tier in this file asks "is the answer right?".  This one asks
# "does the instruction exist?", which needs a different program shape: the
# outcome under test is an illegal-instruction *trap*, so the program
# installs its own M-mode handler, runs each case, and checks whether the
# trap fired.
#
# The reason it is worth a tier of its own is the last group of cases.  The
# three round-six funct6 values are 0x39, 0x3a and 0x3b, and at vm=1 those
# same encodings are vwmmacc.vv, vqmmacc.vv and v8wmmacc.vv -- instructions
# rounds one to three already implement and the loop already passes.  A
# decoder that routes on funct6 and forgets vm therefore breaks three green
# instructions the moment the MX forms are added, and it breaks them
# silently: the integer programs would start taking illegal-instruction
# traps, which surfaces as a simulation that dies rather than as a
# mismatch.  The vm=1 cases below are what catches that on the first
# iteration instead of the tenth.

#: Vector registers the legality tier names.  Fixed rather than allocated,
#: because a case whose (W, SEW) cell is reserved has no legal geometry to
#: allocate from -- and picking these three by hand is safe for every case
#: the tier emits: v8 and v16 are 8-register aligned so they are legal
#: vs1/vs2 groups at any LMUL <= 8, and v24 is 8-register aligned so it is a
#: legal vd group at any EMUL_C <= 8.  The one exception is called out where
#: it arises (SEW=8 with LAMBDA=1 gives EMUL_C = VLEN/8 = 32, which is not a
#: legal C group anywhere).
MXL_VD, MXL_VS1, MXL_VS2 = 24, 8, 16

#: The integer multiply-accumulate that each round-six funct6 decodes to at
#: vm=1.  Not a lookup table of its own: it is asserted against
#: ime_encodings in :func:`mxl_cases`, which re-encodes both words and
#: requires them to differ in exactly one bit.
MXL_VM1_PEER = {"vfwimmacc.vv": "vwmmacc.vv",
                "vfqimmacc.vv": "vqmmacc.vv",
                "vf8wimmacc.vv": "v8wmmacc.vv"}

#: The canonical legal cell each round-six mnemonic is exercised at, as
#: (SEW, LAMBDA, LMUL).  LAMBDA is the smallest value that keeps EMUL_C at
#: or below 8 so that :data:`MXL_VD` is a legal C group; LMUL is 1 except
#: where a case needs it larger, and those cases carry their own.
MXL_CELL = {2: (16, 2, 1), 4: (32, 1, 1), 8: (64, 1, 1)}


@dataclass(frozen=True)
class MxlCase:
    """One (vtype, instruction word) probe and the outcome it must have."""

    tag: str
    geom: TileGeometry
    vtype: int
    word: int
    mnemonic: str
    vm: int
    expect_trap: int
    vl: int


def _mxl_is_illegal(w: int, sew: int, lam: int, lmul: int, *, bs: int,
                    altfmt: int, altfmt_a: int, altfmt_b: int) -> bool:
    """Does the *vm=0* MX form raise Illegal_Instruction here?

    Derived from rvv_ref, never hand-written, so that the tier and the
    reference model cannot disagree about which cases are negative:

      * ``altfmt_A`` or ``altfmt_B`` = 1 -- Sail 6045-6046 (vfwimmacc),
        6162-6163 (vf8wimmacc), 6271-6272 (vfqimmacc).  MXINT is signed by
        definition (spec 2335-2342), so "unsigned" has no meaning here and
        the encoding is reserved rather than ignored;
      * the (W, SEW, altfmt) cell -- :func:`rvv_ref.mx_legal_cell`, which is
        tbl-intmx-encoding-map (spec 7469-7540) and the per-instruction SEW
        guards read back the other way round;
      * ``check_microscaling_legality`` -- :func:`rvv_ref.mx_check_legality`,
        Sail 5151-5158.
    """
    if altfmt_a or altfmt_b:
        return True
    try:
        rvv_ref.mx_legal_cell(w, sew, altfmt)
    except ValueError:
        return True
    try:
        rvv_ref.mx_check_legality(w, lmul, sew, lam, bs)
    except ValueError:
        return True
    return False


def mxl_cases(vlen: int, mnemonic: str) -> List[MxlCase]:
    """Every legality probe for one round-six mnemonic, in emission order.

    The case list is *derived*: `expect_trap` comes from
    :func:`_mxl_is_illegal` for a vm=0 word and is unconditionally false for
    a vm=1 word, whose legality is that of the already-implemented integer
    form.  Nothing here hard-codes an outcome, so sabotaging the legality
    rules in rvv_ref changes what the tier demands -- which is what the
    negative control in :func:`check_mxl_emission` relies on.
    """
    w = {"vfwimmacc.vv": 2, "vfqimmacc.vv": 4, "vf8wimmacc.vv": 8}[mnemonic]
    peer = MXL_VM1_PEER[mnemonic]
    base_sew, base_lam, base_lmul = MXL_CELL[w]
    cases: List[MxlCase] = []

    def add(tag, sew, lam, lmul, *, bs=0, altfmt=0, altfmt_a=0, altfmt_b=0,
            vm=0):
        geom = TileGeometry(vlen, sew, lam, lmul, lam * lmul * (vlen // sew)
                            // lam, w, "op", "int")
        vtype = vtype_value(geom, lmul=lmul, bs=bs, altfmt=altfmt,
                            altfmt_a=altfmt_a, altfmt_b=altfmt_b)
        name = mnemonic if vm == 0 else peer
        word = ime.encode(name, vd=MXL_VD, vs1=MXL_VS1, vs2=MXL_VS2)
        back, ops = ime.decode(word)
        assert back == name and ops["vd"] == MXL_VD, (name, back)
        trap = (0 if vm else
                int(_mxl_is_illegal(w, sew, lam, lmul, bs=bs, altfmt=altfmt,
                                    altfmt_a=altfmt_a, altfmt_b=altfmt_b)))
        cases.append(MxlCase(tag, geom, vtype, word, name, vm, trap,
                             geom.elems_per_reg * lmul))

    # 1. The positive control.  Without it a decoder that raised
    #    Illegal_Instruction on all three funct6 values would pass every
    #    negative case below and the tier would certify nothing.
    add(f"{mnemonic} at its canonical legal cell must NOT trap",
        base_sew, base_lam, base_lmul)

    # 2. Reserved (W, SEW) cells -- the SEW guard of this instruction's Sail,
    #    read out of tbl-intmx-encoding-map.  LAMBDA is 2 at SEW=8 so that
    #    EMUL_C stays at 8 and the C register group is a legal one: the
    #    reserved *cell* is then the only thing wrong with the encoding.
    for sew in (8, 16, 32, 64):
        if (w, sew) in rvv_ref.MX_CELLS:
            continue
        add(f"SEW={sew} is a reserved cell for {mnemonic} (spec 7469-7540)",
            sew, 2 if sew == 8 else 1, 1)

    # 3. Reserved altfmt at a legal (W, SEW) cell -- the C accumulator
    #    format table, spec 1092-1106: altfmt=1 is reserved at SEW 32 and
    #    64, and legal (bfloat16) at SEW=16.
    for sew in sorted(s for (ww, s) in rvv_ref.MX_CELLS if ww == w):
        if 1 in rvv_ref.mx_altfmts(w, sew):
            continue
        add(f"altfmt=1 is reserved at SEW={sew} (spec 1092-1106)",
            sew, 2 if sew == 16 else 1, 1, altfmt=1)

    # 4. altfmt_A / altfmt_B = 1.  MXINT inputs are signed unconditionally.
    for field in ("altfmt_A", "altfmt_B"):
        add(f"{field}=1 is reserved: MXINT inputs are signed "
            f"(spec 2335-2342)",
            base_sew, base_lam, base_lmul,
            **{"altfmt_a" if field == "altfmt_A" else "altfmt_b": 1})

    # 5. check_microscaling_legality, Sail 5151-5158.
    #
    #    5a. bs=1 with W*LMUL > SEW.  Reachable for W=4 and W=8; at W=2 the
    #        only legal cell is SEW=16 and LMUL <= 8, so W*LMUL <= 16 = SEW
    #        always and the rule cannot be violated.  The tier says so here
    #        rather than silently emitting nothing.
    for sew in sorted(s for (ww, s) in rvv_ref.MX_CELLS if ww == w):
        lam = 2 if sew == 16 else 1
        lmul = next((l for l in (2, 4, 8) if w * l > sew), None)
        if lmul is None:
            continue
        probe = TileGeometry(vlen, sew, lam, lmul, lam * lmul, w, "op", "mx")
        try:
            probe.validate()
        except ValueError:
            continue
        add(f"bs=1 with W*LMUL={w * lmul} > SEW={sew} (Sail 5156)",
            sew, lam, lmul, bs=1)
        break

    #    5b. EEW_C * LAMBDA < pw, i.e. SEW*LAMBDA < 16.  At every (W, SEW)
    #        cell this family defines, SEW is 16 or more and LAMBDA is at
    #        least 1, so SEW*LAMBDA >= 16 and the rule is *unreachable in
    #        isolation*: the only configuration that violates it is SEW=8,
    #        which is a reserved cell for all three mnemonics anyway, and
    #        where at VLEN=256 EMUL_C = VLEN/SEW = 32 is not a legal C
    #        group either.  The case is emitted because the rule is real
    #        and an implementation must trap; it is documented as
    #        multiply-illegal because a green result here does not prove
    #        that *this* check is the one that fired.
    add("SEW*LAMBDA = 8 < pw = 16 (Sail 5155); also a reserved cell and "
        "EMUL_C=32, so this only asserts that something traps",
        8, 1, 1)

    # 6. The vm=1 regression.  Same funct6, same vtype as case 4 -- which is
    #    illegal for the MX form precisely because altfmt_A=1 -- but at vm=1
    #    the encoding is the integer multiply-accumulate, for which
    #    altfmt_A=1 means "read A as unsigned" and is perfectly legal (spec
    #    1145-1155).  So the two words must behave differently, and the only
    #    bit between them is vm.
    add(f"vm=1 on the same funct6 is {peer}, which altfmt_A=1 does not make "
        f"illegal (spec 1145-1155)",
        base_sew, base_lam, base_lmul, altfmt_a=1, vm=1)
    add(f"vm=1 on the same funct6 is {peer}, which bs is not even defined "
        f"for (spec 1160-1176: bs is ignored at vm=1)",
        base_sew, base_lam, base_lmul, bs=1, vm=1)

    # The claim the previous two cases rest on, checked here rather than
    # assumed: the MX word and its integer peer differ in exactly the vm bit.
    mx_word = ime.encode(mnemonic, vd=MXL_VD, vs1=MXL_VS1, vs2=MXL_VS2)
    int_word = ime.encode(peer, vd=MXL_VD, vs1=MXL_VS1, vs2=MXL_VS2)
    assert mx_word ^ int_word == 1 << 25, (mnemonic, peer,
                                           hex(mx_word), hex(int_word))
    assert any(c.expect_trap for c in cases), mnemonic
    assert any(not c.expect_trap for c in cases), mnemonic
    return cases


def emit_mxl_test(vlen: int, mnemonic: str, name: str = "ime_mxl") -> str:
    """The legality probe program for one round-six mnemonic.

    Non-differential by necessity: what it observes is whether an
    instruction raised Illegal_Instruction, which no amount of recomputing a
    tile can reveal.  The program installs its own M-mode trap handler,
    which advances ``mepc`` past the faulting instruction and sets a flag,
    runs each case, and compares the flag against what
    :func:`_mxl_is_illegal` derived from rvv_ref.

    The handler is installed and removed around the case list, and ``mtvec``
    is restored before any ``printf`` -- the harness (htif_nano) has its own
    handler and the verdict has to be printed through it, not through this
    one.

    Every case runs at the LAMBDA its cell needs, and each is preceded by the
    usual read-back check: a DUT that clamps LAMBDA down is configured
    differently from what the case assumed, so the program reports
    ``TITAN SKIP`` rather than judging a geometry it did not get.
    """
    cases = mxl_cases(vlen, mnemonic)
    w = {"vfwimmacc.vv": 2, "vfqimmacc.vv": 4, "vf8wimmacc.vv": 8}[mnemonic]
    base_sew, base_lam, base_lmul = MXL_CELL[w]
    shown = TileGeometry(vlen, base_sew, base_lam, base_lmul,
                         base_lam * base_lmul * (vlen // base_sew)
                         // base_lam, w, "op", "mx")
    shown.validate()
    desc = mx_describe(shown, 0, 0)

    head = [
        f"# {name}: legality probes for {mnemonic} at VLEN={vlen}",
        "#",
        "# Generated by ime_tests.py from rvv_ref.py -- do not edit by hand,",
        "# and do not edit rvv_ref.py: it is the judge, not the defendant.",
        "#",
        "# Each case configures vtype with vsetvl, executes one .insn word,",
        "# and checks whether an illegal-instruction trap fired.  A failing",
        "# case reports its index as the TITAN FAIL row:",
        "#",
    ]
    for index, case in enumerate(cases):
        head.append(f"#   {index:2d}  {'trap' if case.expect_trap else 'run '}"
                    f"  vm={case.vm}  {case.mnemonic}  "
                    f"vtype=0x{case.vtype:x}  {case.tag}")
    head += [
        "#",
        f"# Registers are fixed at vd=v{MXL_VD}, vs1=v{MXL_VS1}, "
        f"vs2=v{MXL_VS2}: a reserved cell",
        "# has no legal geometry to allocate from.  See MXL_VD.",
        "",
        "    .text",
        "    .balign 4",
        "    .globl main",
        "main:",
        "    addi  sp, sp, -80",
        "    sd    ra, 72(sp)",
        "    sd    s1, 64(sp)         # carries the exit status past printf",
        "    sd    s2, 56(sp)",
        "    sd    s3, 48(sp)",
        "    sd    s4, 40(sp)",
        "    sd    s5, 32(sp)",
        "    sd    s6, 24(sp)",
        "    sd    s7, 16(sp)",
        "    sd    s8, 8(sp)          # the harness's mtvec",
        "    sd    s9, 0(sp)          # the trap flag",
        f"    li    t0, {MSTATUS_VS_INITIAL}",
        "    csrs  mstatus, t0        # enable vector state",
        "",
        "    # install our own M-mode trap handler",
        "    csrr  s8, mtvec",
        "    la    t0, .Ltrap",
        "    csrw  mtvec, t0",
    ]

    body: List[str] = []
    for index, case in enumerate(cases):
        body += [
            "",
            f"    # ---- case {index}: "
            f"{'must trap' if case.expect_trap else 'must run'} -- "
            f"{case.tag} ----",
            "    li    s9, 0              # trap flag",
            f"    li    t0, {case.vl}",
            f"    li    t1, 0x{case.vtype:x}",
            "    vsetvl x0, t0, t1",
        ]
        # Report WHICH case skipped.  _check_lambda_retained loads the
        # DUT's lambda into a1 and jumps to .Lskip, whose format string
        # names only the program's nominal geometry -- so a skip from case 3
        # and a skip from case 0 print the same line.  In r19 that cost the
        # Stage M agent an iteration: the verdict said "lambda=0 (requested
        # 1)" against case 0's header while the real culprit was a reserved
        # -altfmt case further down the list.  a2 carries the index so the
        # two are distinguishable.
        body.append(f"    li    a3, {index}       # case index, for TITAN SKIP")
        body += _check_lambda_retained(case.geom)
        body += [
            f"    .insn 4, {case.word:#010x}"
            f"    # {case.mnemonic} (vm={case.vm})",
            f"    li    t0, {case.expect_trap}",
            "    beq   s9, t0, 1f",
            f"    li    a1, {index}",
            f"    li    a2, {case.expect_trap}",
            f"    li    s1, {EXIT_MISMATCH_BASE + index}",
            "    j     .Lfail",
            "1:",
        ]

    tail = [
        "",
        "    # ---- verdict ----",
        "    csrw  mtvec, s8          # hand the harness its handler back",
        "    la    a0, .Lfmt_pass",
        "    call  printf",
        f"    li    s1, {EXIT_PASS}",
        "    j     .Lret",
        "",
        ".Lskip:",
        "    csrw  mtvec, s8",
        "    mv    a2, a1             # raw vtype.lambda[2:0] field",
        "    li    a1, 0              # decoded: 0 means no selected lambda",
        "    beqz  a2, 1f",
        "    addi  t0, a2, -1",
        "    li    t1, 1",
        "    sll   a1, t1, t0         # lambda = 1 << (imm - 1)",
        "1:",
        "    la    a0, .Lfmt_skip",
        "    call  printf",
        f"    li    s1, {EXIT_UNSUPPORTED_GEOMETRY}",
        "    j     .Lret",
        "",
        ".Lfail:",
        "    csrw  mtvec, s8          # before printf: the harness traps too",
        "    mv    s2, a1             # case index",
        "    mv    s3, a2             # the outcome the case required",
        "",
        "    # ---- evidence: which case, what was required, what happened",
        "    la    a0, .Lfmt_diff",
        "    mv    a1, s2",
        "    mv    a2, s3",
        "    mv    a3, s3",
        "    mv    a4, s9",
        "    call  printf",
        "",
        "    # ---- verdict, last so a log tail keeps it ----",
        "    la    a0, .Lfmt_fail",
        "    mv    a1, s2",
        "    mv    a2, s3",
        "    call  printf",
        "",
        ".Lret:",
        "    mv    a0, s1",
        "    ld    s9, 0(sp)",
        "    ld    s8, 8(sp)",
        "    ld    s7, 16(sp)",
        "    ld    s6, 24(sp)",
        "    ld    s5, 32(sp)",
        "    ld    s4, 40(sp)",
        "    ld    s3, 48(sp)",
        "    ld    s2, 56(sp)",
        "    ld    s1, 64(sp)",
        "    ld    ra, 72(sp)",
        "    addi  sp, sp, 80",
        "    ret",
        "",
        "    # The handler.  Placed after the return so control cannot fall",
        "    # into it, and 4-byte aligned because mtvec's low bits are the",
        "    # mode field.  Every probed instruction is 4 bytes (no",
        "    # compressed encoding exists for .insn 4), so mepc+4 resumes at",
        "    # the instruction after the faulting one.",
        "    .balign 4",
        ".Ltrap:",
        "    csrr  t0, mepc",
        "    addi  t0, t0, 4",
        "    csrw  mepc, t0",
        "    li    s9, 1",
        "    mret",
    ]

    data = [
        "", "    .data", "    .balign 8",
        f'.Lfmt_pass:  .asciz "TITAN PASS {desc}\\n"',
        f'.Lfmt_skip:  .asciz "TITAN SKIP lambda=%d (requested {shown.lam}) '
        f'imm=%d case=%d {desc}\\n"',
        f'.Lfmt_fail:  .asciz "TITAN FAIL row=%d col=%d {desc}\\n"',
        '.Lfmt_diff:  .asciz "TITAN DIFF r=%d c=%d exp=0x%x got=0x%x\\n"',
    ]
    return "\n".join(head + body + tail + data) + "\n"



def mx_legal_bs(geom: TileGeometry) -> Tuple[int, ...]:
    """The block-size selectors legal at this geometry, in order.

    bs=0 is always legal; bs=1 adds ``W*LMUL <= SEW`` (Sail 5156).  The
    ``S_blocks <= R`` filter is belt-and-braces: the architecture already
    guarantees it -- the scale array has R columns per row and S_blocks
    blocks to name -- and :func:`_mx_scale_arrays` asserts it, so a geometry
    that violated it would be a discovery about the spec rather than a case
    to drop.  It is filtered here too so that a stress sampler cannot turn
    that discovery into a crashed pool build.
    """
    out = []
    r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
    for bs in (0, 1):
        try:
            rvv_ref.mx_check_legality(geom.w, geom.lmul, geom.sew,
                                      geom.lam, bs)
        except ValueError:
            continue
        if rvv_ref.mx_block_count(geom.k_eff,
                                  rvv_ref.mx_block_size(bs)) <= r:
            out.append(bs)
    return tuple(out)


def mx_random_plan(geom: TileGeometry, rng: random.Random,
                   bs: int = None, altfmt: int = None) -> MxPlan:
    """A randomised round-six plan for *geom* -- the stress-pool entry point.

    The scale construction is the ime_mxs_ one
    (``eA[i][s] = 127 + dA(i) + g(s)``, ``eB[j][s] = 127 + dB(j) - g(s)``),
    which degenerates gracefully to the ime_mx_ one when there is a single
    block: ``g(0) = 0``, so the two scales are 127 +/- a small per-row term
    and the combined exponent still varies across the tile.  Using one
    construction for both means a stress program is never a *weaker* test
    than a directed one at the same geometry -- it is the same arithmetic
    with different numbers, which is what a stress pool is for.

    ``bs`` and ``altfmt`` default to a random legal choice, so one pass of
    the stress mix covers both block sizes and both accumulator formats at
    the cells that have two.
    """
    if bs is None:
        bs = rng.choice(mx_legal_bs(geom))
    if altfmt is None:
        altfmt = rng.choice(rvv_ref.mx_altfmts(geom.w, geom.sew))
    blocks = rvv_ref.mx_block_count(geom.k_eff, rvv_ref.mx_block_size(bs))
    mid = (blocks - 1) // 2
    a, b, c = _mx_operands(geom, altfmt, bs, rng)
    sa, sb = _mx_scale_arrays(
        geom, blocks,
        lambda m, s: rvv_ref.MX_E8M0_BIAS + (m % 3) - 1 + (s - mid),
        lambda m, s: rvv_ref.MX_E8M0_BIAS + (m % 3) - 1 - (s - mid))
    return MxPlan(geom, altfmt, bs, a, b, c, sa, sb, "mx",
                  f"randomised, {blocks} block(s) at block_size="
                  f"{rvv_ref.mx_block_size(bs)}")


_MX_DIRECTIVE_WIDTH = {".byte": 8, ".half": 16, ".word": 32, ".dword": 64}


def mx_parse_data(asm: str):
    """``{label: [raw integers]}`` for every data label in an emitted program.

    The point of reading a program's own ``.data`` back rather than trusting
    the values that went into it is that the self-tests can then re-derive
    the whole architectural answer *from the emitted text* -- the operands
    the DUT will actually load, the v0 image it will actually read -- and
    compare that against rvv_ref.  A generator bug that corrupts the image
    on the way out is invisible to any check that only inspects its own
    inputs.
    """
    out, label = {}, None
    body = asm.split("    .data\n", 1)
    if len(body) != 2:
        return out
    for line in body[1].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":"):
            label = stripped[:-1]
            out.setdefault(label, [])
            continue
        head, _, rest = stripped.partition(" ")
        if head in _MX_DIRECTIVE_WIDTH and label is not None:
            out[label].extend(int(tok, 0) for tok in rest.split(","))
        elif head == ".zero" and label is not None:
            out[label].extend([0] * int(rest, 0))
        elif head in (".balign", ".globl", ".text"):
            label = None
    return out


def _sext(value: int, width: int) -> int:
    return value - (1 << width) if value >> (width - 1) else value


def mx_reconstruct(asm: str, plan: MxPlan, *, pair_index=None, stride=None):
    """Re-derive the whole architectural answer from an emitted program.

    Reads ``mat_a``, ``mat_b``, ``c_init`` and ``v0_scales`` out of the
    program's ``.data``, unpacks the v0 image with the spec's own index
    functions, and runs :func:`rvv_ref.int_scaled_gemm_reference` on the
    result.  Then it checks, element by element, that what the program will
    *do* at run time -- copy c_init, load a literal, or convert
    ``ref_int + ref_bump`` -- lands on that answer.

    *pair_index* and *stride* default to rvv_ref's honest implementations
    and are parameters so that a negative control can sabotage the
    **generator**, re-emit, and then decode with the honest ones.  Patching
    both sides at once would be a control with no teeth: the two errors
    would cancel and the check would stay green, which is the exact failure
    mode round five warned about.
    """
    pair_index = pair_index or rvv_ref.mx_pair_index
    stride = stride or rvv_ref.mx_scale_stride
    geom = plan.geom
    width, fmt = geom.sew, plan.fmt
    _ebits, prec, _bias, _emax = rvv_ref.fp_fields(width, fmt)
    data = mx_parse_data(asm)

    def matrix(label, rows, cols, element_width, signed):
        flat = data[label]
        assert len(flat) == rows * cols, (label, len(flat), rows * cols)
        return [[_sext(v, element_width) if signed else v
                 for v in flat[i * cols:(i + 1) * cols]] for i in range(rows)]

    a = matrix("mat_a", geom.m, geom.k_eff, width, True)
    b = matrix("mat_b", geom.n_max, geom.k_eff, width, True)
    c = matrix("c_init", geom.m, geom.n_max, width, False)

    r = stride(geom.sew, geom.lam)
    image = data["v0_scales"]
    assert len(image) == geom.vlen // rvv_ref.MX_PAIR_WIDTH, len(image)
    scales_a = [[0] * r for _ in range(geom.m)]
    scales_b = [[0] * r for _ in range(geom.m)]
    for m in range(geom.m):
        for s in range(r):
            pair = image[pair_index(m, s, r)]
            scales_a[m][s] = pair & 0xFF
            scales_b[m][s] = (pair >> 8) & 0xFF

    ref = rvv_ref.int_scaled_gemm_reference(a, b, c, scales_a, scales_b,
                                            geom, bs=plan.bs,
                                            altfmt=plan.altfmt)
    block_size = rvv_ref.mx_block_size(plan.bs)
    blocks = rvv_ref.mx_block_count(geom.k_eff, block_size)
    modes, bumps, lits = data["ref_mode"], data["ref_bump"], data["ref_lit"]
    for i in range(geom.m):
        for j in range(geom.n_max):
            off = _c_off(geom, i, j)
            mode = modes[off]
            want = ref[i][j]
            if mode == MX_MODE_COPY:
                assert j >= geom.n, (i, j)
                assert want == c[i][j], (i, j, hex(want), hex(c[i][j]))
            elif mode == MX_MODE_LITERAL:
                assert lits[off] == want, (i, j, hex(lits[off]), hex(want))
            else:
                total = sum(
                    rvv_ref.mx_int_block_dot(
                        a, b, i, j,
                        *rvv_ref.mx_block_interval(s, block_size, geom.k_eff))
                    for s in range(blocks))
                bits = rvv_ref.mx_int_to_fp(total, width, fmt)
                if total:
                    bits = (bits + _sext(bumps[off], 64)) & ((1 << width) - 1)
                assert bits == want, (i, j, hex(bits), hex(want))
    # ... and the tile-layout copies the IME path loads must carry the same
    # matrices as the row-major copies the reference path loads.
    for label, mat in (("mat_a_tile", a), ("mat_b_tile", b)):
        expected = mx_parse_data(
            "\n    .data\n" + "\n".join(_mx_tile_data(label, mat, geom))
            + "\n")[label]
        assert data[label] == expected, label
    return ref, modes


def _mx_name(prefix: str, plan: MxPlan, suffix: str = "") -> str:
    geom = plan.geom
    return (f"{prefix}sew{geom.sew}_lam{geom.lam}_lmul{geom.lmul}"
            f"_w{geom.w}_n{geom.n}_af{plan.altfmt}_bs{plan.bs}{suffix}")


def mx_directed_tiers(vlen: int, insns: Sequence[str], seed: int = 0
                      ) -> List[Tuple[str, str, TileGeometry]]:
    """Round six's four tiers, in order, for the mnemonics in *insns*.

    Returned as a list rather than generated inline in
    :func:`directed_suite` so that the self-tests can ask for the tiers
    without asking for the 638 programs that precede them.  The order --
    ime_mx_, ime_mxs_, ime_mxn_, ime_mxl_ -- is fixed for the same
    append-don't-interleave reason every earlier round fixed its own: a tree
    that passed round five must still see exactly the programs it saw,
    in exactly the order, before any of these.
    """
    out: List[Tuple[str, str, TileGeometry]] = []
    for plan in mx_exact_plans(vlen, seed):
        if plan.geom.mnemonic not in insns:
            continue
        name = _mx_name("ime_mx_", plan)
        out.append((name, emit_mx_test(plan, name), plan.geom))
    for plan in mx_scale_plans(vlen, seed + 100):
        if plan.geom.mnemonic not in insns:
            continue
        name = _mx_name("ime_mxs_", plan)
        out.append((name, emit_mx_test(plan, name), plan.geom))
    for suffix, plan in mx_nan_plans(vlen, seed + 200):
        if plan.geom.mnemonic not in insns:
            continue
        name = _mx_name("ime_mxn_", plan, f"_{suffix}")
        out.append((name, emit_mx_test(plan, name), plan.geom))
    for mnemonic in ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv"):
        if mnemonic not in insns:
            continue
        w = {"vfwimmacc.vv": 2, "vfqimmacc.vv": 4,
             "vf8wimmacc.vv": 8}[mnemonic]
        sew, lam, lmul = MXL_CELL[w]
        geom = TileGeometry(vlen, sew, lam, lmul,
                            lmul * (vlen // sew), w, "op", "mx")
        name = f"ime_mxl_{mnemonic.split('.')[0]}"
        out.append((name, emit_mxl_test(vlen, mnemonic, name), geom))
    return out


def directed_suite(vlen: int, seed: int = 0,
                   sews: Sequence[int] = (8, 16, 32, 64),
                   lmuls: Sequence[int] = (1, 2, 4, 8),
                   full_vl_only: bool = True,
                   insns: Sequence[str] = None,
                   widening_sews: Sequence[int] = None,
                   widening_sews_by_w: dict = None,
                   ) -> List[Tuple[str, str, TileGeometry]]:
    """(name, asm, geometry) for the round-one-reachable legal geometries.

    Two tiers, because the suite is fully unrolled and the whole partial-N
    sweep is roughly half a million lines of assembly -- 200 compiles per
    iteration would dominate a 60-iteration loop.

    ``full_vl_only=True`` (the default, and what runs every iteration) keeps
    one program per (SEW, LAMBDA, LMUL) at N = N_max: about 30 programs, which
    still exercises every tile geometry and every register allocation.

    ``full_vl_only=False`` sweeps N as well and is the S1 *gate*: the three
    instructions must agree with the RVV reference across every legal
    lambda / SEW / LMUL / VL combination the DUT supports.  Partial N is where
    the C tile tail policy lives, so the gate cannot skip it -- but it only
    has to run when the cheap tier is already green.

    Geometries the DUT does not support exit 1 and count as skipped, not
    failed: the supported lambda set is implementation-defined and the
    architecture expects software to discover it by writing and reading back.

    **Rounds.**  ``insns`` selects which instructions the suite covers and
    defaults to ``constants.INSNS`` (round one + round two unless
    ``TITAN_INSNS`` says otherwise).  The W=1 tier is emitted first and its
    programs -- names and text alike -- are byte-for-byte what round one
    produced, so a tree that passed round one still passes exactly those
    programs.  The W=4 tier is appended, with an ``ime_q_`` name prefix so a
    widening geometry can never collide with the W=1 geometry it shares
    (SEW, LAMBDA, LMUL, N) with.

    Round three appends three more tiers, in this order and always after the
    round-one and round-two tiers, for the same byte-identity reason:

      ``ime_w_``   vwmmacc.vv  (W=2, SEW 16 and 32)
      ``ime_8w_``  v8wmmacc.vv (W=8, SEW 64)
      ``ime_t_``   the transposing tile transfers, W=1, every SEW

    Round four appends one more, last:

      ``ime_f_``   vfmmacc.vv, W=1, SEW 32 and 64 (binary32 / binary64)

    whose programs carry a *scalar* rv64f / rv64d reference instead of the
    RVV one -- see :func:`_fp_ref_path` for why, and for the implementation
    disclosure the exact comparison rests on.

    Round five appends two more, last of all, selected by the ``clayout``
    tier token rather than by a mnemonic:

      ``ime_cl_``   vmmacc.vv,  W=1, every SEW -- C read back with vse<SEW>.v
      ``ime_clq_``  vqmmacc.vv, W=4, SEW 32

    These do not use ``vmts.v`` at all.  See :func:`emit_clayout_test`.

    The transposing tier reuses vmmacc.vv for the arithmetic: what is under
    test is vmttl.v / vmtts.v, and pairing them with a multiply-accumulate
    that is already green isolates a tile-layout failure from an arithmetic
    one.  It is generated for the whole SEW / LAMBDA / LMUL / N sweep rather
    than a sample, because tile layout is where round one's bugs clustered
    and because partial N is the only place vmtts.v meets the C tile tail
    policy at all.
    """
    if insns is None:
        import constants
        insns = constants.INSNS
    if widening_sews is None:
        widening_sews = rvv_ref.WIDENING_SEWS
    if widening_sews_by_w is None:
        widening_sews_by_w = rvv_ref.WIDENING_SEWS_BY_W

    rng = random.Random(seed)
    out = []
    tiers = []
    if "vmmacc.vv" in insns:
        tiers.append(("ime_", sews, (1,), ("op",), ("int",)))
    if "vqmmacc.vv" in insns:
        # SEW is the accumulator width, so the widening tier is enumerated
        # over the *C* widths that have a modelled input width: SEW=32 is
        # Zvvi8i32mm (Int8 -> Int32), which is what llama.cpp needs.
        tiers.append(("ime_q_", [w for w in widening_sews if w in sews], (4,),
                      ("op",), ("int",)))
    for mnemonic, prefix, w in (("vwmmacc.vv", "ime_w_", 2),
                                ("v8wmmacc.vv", "ime_8w_", 8)):
        if mnemonic in insns:
            tiers.append((prefix,
                          [x for x in widening_sews_by_w[w] if x in sews],
                          (w,), ("op",), ("int",)))
    if "vmttl.v" in insns or "vmtts.v" in insns:
        tiers.append(("ime_t_", sews, (1,), ("t",), ("int",)))
    if "vfmmacc.vv" in insns:
        # Round four.  Appended last, after every integer tier, for the same
        # byte-identity reason the round-three tiers were: a tree that passed
        # round three must still see exactly the programs it saw, in order.
        # The SEW list is intersected with rvv_ref.FP_FORMATS rather than
        # taken from `sews`, because the floating-point tier's legal
        # accumulator widths are a property of the format table (spec
        # 1490-1493: altfmt is ignored, hence reserved-free, only at
        # binary32 and binary64), not of the caller's sweep.
        tiers.append(("ime_f_",
                      [x for x in sews if x in rvv_ref.FP_FORMATS],
                      (1,), ("op",), ("fp",)))

    # Round five.  Appended after every tier above, for the same
    # byte-identity reason each earlier round was appended after its
    # predecessors -- and note it is keyed on a *tier* token, not a
    # mnemonic: it re-tests instructions rounds one and two already cover,
    # with a program shape that can see what those programs cannot.  Always
    # at full VL: see rvv_ref.clayout_capable for why a partial-N tile has
    # no all-distinct construction, and note the consequence -- the gate and
    # the per-iteration suite carry the *same* clayout programs.
    if "clayout" in insns:
        tiers.append(("ime_cl_", sews, (1,), ("op",), ("int",)))
        tiers.append(("ime_clq_", [w for w in widening_sews if w in sews],
                      (4,), ("op",), ("int",)))

    for prefix, tier_sews, ws, tloads, kinds in tiers:
        clayout = prefix.startswith("ime_cl")
        for geom in rvv_ref.ime_legal_configs(
                vlen, sews=tier_sews, lmuls=lmuls,
                full_vl_only=True if clayout else full_vl_only,
                ws=ws, tloads=tloads, kinds=kinds,
                checks=("clayout",) if clayout else ("pair",)):
            if geom.emul_c == 16:
                continue
            if clayout and not rvv_ref.clayout_capable(geom):
                continue
            try:
                VectorAlloc.allocate(geom)
            except ValueError:
                continue
            name = (f"{prefix}sew{geom.sew}_lam{geom.lam}_lmul{geom.lmul}"
                    f"_n{geom.n}")
            asm = (emit_clayout_test(geom, name) if clayout
                   else emit_test(geom, rvv_ref.random_case(geom, rng), name))
            out.append((name, asm, geom))

    # Round six, appended after every tier above and never interleaved into
    # one, for the same byte-identity reason each earlier round was appended
    # after its predecessors.  The four microscaled tiers do not go through
    # the (prefix, sews, ws, tloads, kinds) table because they are not a
    # sweep over it: each one picks its geometries by a property the table
    # cannot express -- how many microscaling blocks the K interval has --
    # and each carries its own operand construction.  See mx_directed_tiers.
    round_six = [m for m in ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv")
                 if m in insns]
    if round_six:
        out += mx_directed_tiers(vlen, round_six, seed)
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def check_vtype_fields() -> None:
    """The IME fields must land where the spec puts them, below vill."""
    geom = TileGeometry(256, 32, 2, 1, 8)
    geom.validate()
    value = vtype_value(geom, lmul=1, altfmt_a=1, altfmt_b=1, bs=1)
    assert value & 0x7 == 0, "vlmul field for LMUL=1 is 0"
    assert (value >> 3) & 0x7 == 2, "vsew field for SEW=32 is 2"
    assert (value >> 60) & 0x7 == ime.lambda_imm(2), "lambda at vtype[62:60]"
    assert (value >> 59) & 1 == 1, "bs at vtype[59]"
    assert (value >> 58) & 1 == 1, "altfmt_A at vtype[58]"
    assert (value >> 57) & 1 == 1, "altfmt_B at vtype[57]"
    assert (value >> 63) & 1 == 0, "vill must stay clear"


def check_allocation_disjoint() -> None:
    """A, B and C register groups must not overlap, at any legal geometry."""
    for vlen in (64, 128, 256, 512, 1024):
        for geom in rvv_ref.ime_legal_configs(vlen, full_vl_only=True):
            try:
                alloc = VectorAlloc.allocate(geom)
            except ValueError:
                continue
            a = set(range(alloc.a, alloc.a + geom.lmul))
            b = set(range(alloc.b, alloc.b + geom.lmul))
            c = set(range(alloc.c, alloc.c + geom.emul_c))
            assert not (a & b) and not (a & c) and not (b & c), geom.describe()
            assert max(c) < 32 and alloc.c % geom.emul_c == 0, geom.describe()
            assert alloc.a % geom.lmul == 0 and alloc.b % geom.lmul == 0


def check_emitted_structure() -> None:
    """Every emitted program must obey the harness contract."""
    rng = random.Random(3)
    for vlen in (128, 256):
        widening = list(rvv_ref.ime_legal_configs(
            vlen, sews=rvv_ref.WIDENING_SEWS, lmuls=(1, 2), ws=(4,)))
        for w in (2, 8):
            widening += list(rvv_ref.ime_legal_configs(
                vlen, sews=rvv_ref.WIDENING_SEWS_BY_W[w], lmuls=(1, 2),
                ws=(w,)))
        transposing = list(rvv_ref.ime_legal_configs(
            vlen, sews=(8, 32), lmuls=(1, 2), tloads=("t",)))
        for geom in list(rvv_ref.ime_legal_configs(
                vlen, sews=(8, 32), lmuls=(1, 2))) + widening + transposing:
            if geom.emul_c == 16:
                continue
            try:
                VectorAlloc.allocate(geom)
            except ValueError:
                continue
            asm = emit_test(geom, rvv_ref.random_case(geom, rng))
            assert ".globl main" in asm and "\nmain:" in asm
            for forbidden in ("_start", "tohost", "riscv_test.h"):
                assert forbidden not in asm, f"{forbidden} must not appear"
            # C load, A load, B load, vmmacc, C store.
            assert asm.count(".insn 4,") == 5, (
                f"{geom.describe()}: {asm.count('.insn 4,')} .insn, want 5")
            # No branch in the *unrolled* code may target a named label:
            # those hops span the inlined RVV path and would exceed a
            # conditional branch's +-4KB range at the larger geometries.
            # The failure-evidence block is exempt -- it is a compact loop,
            # under 100 instructions end to end, so its branches to
            # `.Ldump_*` are always in range.
            for line in asm.splitlines():
                stripped = line.strip()
                if stripped.startswith(("beq", "bne", "blt", "bge")):
                    assert stripped.endswith("1f") or ".Ldump" in stripped, \
                        stripped

            # The evidence a failing tile prints must actually be emitted.
            for marker in ("TITAN DIFF", "TITAN CDUMP", "TITAN CREF"):
                assert marker in asm, f"{geom.describe()}: no {marker}"

            # Every data buffer must be big enough for a full-VL tile load,
            # which reads M rows regardless of N.  The tile buffers hold the
            # same element count, re-chunked into lines of the width the
            # tile transfer's LD implies: ab_linesize for an
            # order-preserving image, M (the column stride) for a
            # transposing one.
            tile_width = (geom.ab_linesize if geom.tload == "op" else geom.m)
            for label, rows, width in (
                    ("mat_a", geom.m, geom.sew),
                    ("mat_b", geom.n_max, geom.sew),
                    ("c_init", geom.m, geom.sew),
                    ("mat_a_tile",
                     geom.m * geom.k_eff // tile_width, geom.eew_ab),
                    ("mat_b_tile",
                     geom.n_max * geom.k_eff // tile_width,
                     geom.eew_ab)):
                block = asm.split(f"{label}:")[1].split(":")[0]
                emitted = [ln for ln in block.splitlines()
                           if ln.strip().startswith(_DATA_DIRECTIVE[width])]
                assert len(emitted) == rows, (
                    f"{geom.describe()}: {label} has {len(emitted)} rows, "
                    f"want {rows}")


def check_insn_words_decode() -> None:
    """Every .insn word an emitted program contains must decode back."""
    rng = random.Random(5)
    geom = TileGeometry(256, 32, 2, 1, 8)
    alloc = VectorAlloc.allocate(geom)
    asm = emit_test(geom, rvv_ref.random_case(geom, rng))
    words = [int(line.split(",")[1].split()[0], 16)
             for line in asm.splitlines() if ".insn 4," in line]
    names = [ime.decode(w)[0] for w in words]
    assert names == ["vmtl.v", "vmtl.v", "vmtl.v", "vmmacc.vv", "vmts.v"], names
    _, macc = ime.decode(words[3])
    assert macc == {"vd": alloc.c, "vs1": alloc.a, "vs2": alloc.b}, macc

    # The widening program has the same five words with vqmmacc.vv in the
    # multiply slot -- the tile transfers do not change (they move SEW-wide
    # storage elements either way), which is the whole reason round two is
    # cheap on the software side.
    qgeom = TileGeometry(256, 32, 2, 2, 2 * 2 * 4, 4)
    qalloc = VectorAlloc.allocate(qgeom)
    qasm = emit_test(qgeom, rvv_ref.random_case(qgeom, rng))
    qwords = [int(line.split(",")[1].split()[0], 16)
              for line in qasm.splitlines() if ".insn 4," in line]
    qnames = [ime.decode(w)[0] for w in qwords]
    assert qnames == ["vmtl.v", "vmtl.v", "vmtl.v", "vqmmacc.vv", "vmts.v"], \
        qnames
    _, qmacc = ime.decode(qwords[3])
    assert qmacc == {"vd": qalloc.c, "vs1": qalloc.a, "vs2": qalloc.b}, qmacc

    # Round three: W=2 and W=8 put their own funct6 in the multiply slot and
    # change nothing else; the transposing program swaps all three tile
    # transfers and leaves the multiply alone.
    for probe, want in (
            (TileGeometry(256, 32, 2, 1, 2 * 1 * 4, 2),
             ["vmtl.v", "vmtl.v", "vmtl.v", "vwmmacc.vv", "vmts.v"]),
            (TileGeometry(256, 64, 1, 1, 1 * 1 * 4, 8),
             ["vmtl.v", "vmtl.v", "vmtl.v", "v8wmmacc.vv", "vmts.v"]),
            (TileGeometry(256, 32, 2, 2, 2 * 2 * 4, 1, "t"),
             ["vmttl.v", "vmttl.v", "vmttl.v", "vmmacc.vv", "vmtts.v"])):
        probe.validate()
        palloc = VectorAlloc.allocate(probe)
        pasm = emit_test(probe, rvv_ref.random_case(probe, rng))
        pwords = [int(line.split(",")[1].split()[0], 16)
                  for line in pasm.splitlines() if ".insn 4," in line]
        assert [ime.decode(w)[0] for w in pwords] == want, probe.describe()
        _, pmacc = ime.decode(pwords[3])
        assert pmacc == {"vd": palloc.c, "vs1": palloc.a, "vs2": palloc.b}, \
            probe.describe()


def check_widening_verdict_geometry() -> None:
    """helpers._GEOM_RE must still recover VLEN/SEW/LAMBDA from a W=4 line.

    The classifier scrapes three adjacent fields out of the verdict string.
    A widening geometry adds a clause to that string, and if it landed in
    the middle of the run the classifier would quietly lose the geometry of
    every widening skip -- and a lost geometry is how round one granted a
    bad lambda fifteen iterations of amnesty.
    """
    import helpers

    class _Run:
        def __init__(self, log): self.log, self.success, self.returncode = log, True, 0

    probes = list(rvv_ref.ime_legal_configs(256, sews=rvv_ref.WIDENING_SEWS,
                                            ws=(4,), full_vl_only=True))
    for w in (2, 8):
        probes += list(rvv_ref.ime_legal_configs(
            256, sews=rvv_ref.WIDENING_SEWS_BY_W[w], ws=(w,),
            full_vl_only=True))
    # The transposing clause is appended after EMUL_C, past everything the
    # classifier scrapes; check it does not disturb the scrape either.
    probes += list(rvv_ref.ime_legal_configs(256, full_vl_only=True,
                                             tloads=("t",)))
    for geom in probes:
        text = f"TITAN SKIP lambda=1 (requested {geom.lam}) imm=1 " \
               f"{geom.describe()}"
        out = helpers.classify_run(_Run(text))
        assert (out.vlen, out.sew, out.requested_lambda) == \
            (geom.vlen, geom.sew, geom.lam), (geom.describe(), out)
        assert helpers.classify_run(
            _Run(f"TITAN PASS {geom.describe()}")).kind == "pass"


def check_widening_emission() -> None:
    """The W=4 programs must be the round-one shape with four bytes packed.

    Three things could go wrong and be invisible in a green suite: the tile
    buffers could be emitted at the wrong width, the sign-extended RVV copies
    could be truncated, and the C buffers could be sized at the input width
    rather than the accumulator width.  Each is checked against the numbers
    the geometry implies, not against the generator.
    """
    rng = random.Random(13)
    probes = []
    for w, tier_sews in sorted(rvv_ref.WIDENING_SEWS_BY_W.items()):
        probes += list(rvv_ref.ime_legal_configs(256, sews=tier_sews, ws=(w,),
                                                 full_vl_only=True))
    for geom in probes:
        if not _allocatable_here(geom):
            continue
        case = rvv_ref.random_case(geom, rng)
        a, b, c = case
        asm = emit_test(geom, case)

        # Five .insn words, and the multiply is the widening one.
        assert asm.count(".insn 4,") == 5, geom.describe()
        assert f"# {geom.mnemonic} " in asm, geom.describe()
        assert "# vmmacc.vv " not in asm, geom.describe()

        # The A/B tile images are emitted at the *logical* input width
        # EEW_A = SEW/W -- .byte at W=4/SEW=32 and W=8/SEW=64, .half at
        # W=2/SEW=32 -- K_eff per line, M (or N_max) lines.
        directive = _DATA_DIRECTIVE[geom.eew_ab]
        mask = (1 << geom.eew_ab) - 1
        for label, mat in (("mat_a_tile", a), ("mat_b_tile", b)):
            block = asm.split(f"{label}:")[1].split(":")[0]
            rows = [ln for ln in block.splitlines()
                    if ln.strip().startswith(directive)]
            assert len(rows) == len(mat), (geom.describe(), label, len(rows))
            for line, want in zip(rows, mat):
                got = [int(tok, 0) for tok in
                       line.split(directive)[1].split(",")]
                assert got == [v & mask for v in want], \
                    (geom.describe(), label)

        # The RVV copies are .word: each narrow input sign-extended to SEW.
        for label, mat in (("mat_a", a), ("mat_b", b)):
            block = asm.split(f"\n{label}:")[1].split(":")[0]
            rows = [ln for ln in block.splitlines()
                    if ln.strip().startswith(_DATA_DIRECTIVE[geom.sew])]
            assert len(rows) == len(mat), (geom.describe(), label)
            got = [int(tok, 0) for tok in
                   rows[0].split(_DATA_DIRECTIVE[geom.sew])[1].split(",")]
            assert got == [v & ((1 << geom.sew) - 1) for v in mat[0]], \
                (geom.describe(), label)

        # C buffers are sized at the accumulator width, M x M.
        want_bytes = geom.m * geom.m * geom.sew // 8
        assert f".zero {want_bytes}" in asm, (geom.describe(), want_bytes)

        # The reference path must cover K_eff exactly, in VLMAX-sized chunks.
        chunk = min(geom.elems_per_reg, geom.k_eff)
        assert f"vsetvli t1, t0, e{geom.sew}, m1, ta, ma" in asm
        per_dot = geom.k_eff // chunk
        assert asm.count("    vmul.vv v3, v1, v2") == \
            geom.m * geom.n * per_dot, geom.describe()


def check_transposing_emission() -> None:
    """The vmttl.v / vmtts.v programs must carry a genuinely transposed image.

    The failure mode this exists to catch is a transposing program that is
    green because it is secretly an order-preserving one: same registers,
    same arithmetic, and a memory image that was never transposed.  So the
    tile and C buffers are checked against the transpose of the matrices
    the case actually holds, element by element, and the leading dimensions
    are checked to be M (the column stride, and the rs2 = x0 default the
    spec gives the transposing pair).
    """
    rng = random.Random(23)
    for geom in rvv_ref.ime_legal_configs(256, full_vl_only=True,
                                          tloads=("t",)):
        if not _allocatable_here(geom):
            continue
        case = rvv_ref.random_case(geom, rng)
        a, b, c = case
        asm = emit_test(geom, case)
        assert geom.w == 1 and geom.eew_ab == geom.sew, geom.describe()

        # Five .insn words: three vmttl.v, the multiply, one vmtts.v.
        assert asm.count(".insn 4,") == 5, geom.describe()
        names = [ime.decode(int(line.split(",")[1].split()[0], 16))[0]
                 for line in asm.splitlines() if ".insn 4," in line]
        assert names == ["vmttl.v"] * 3 + ["vmmacc.vv", "vmtts.v"], \
            (geom.describe(), names)

        # Every LD written into a1 is M: the C tile is M x M and the A/B
        # tiles are K_eff x M in the column-major image.
        ime_path = asm.split("# ---- IME path ----")[1].split("# ----")[0]
        lds = {int(ln.split(",")[1].split()[0])
               for ln in ime_path.splitlines()
               if ln.strip().startswith("li    a1,")}
        assert lds == {geom.m}, (geom.describe(), lds)

        # The A/B tile images: K_eff lines of M elements, line k holding
        # column k of the tile.
        directive = _DATA_DIRECTIVE[geom.sew]
        mask = (1 << geom.sew) - 1
        for label, mat in (("mat_a_tile", a), ("mat_b_tile", b)):
            block = asm.split(f"{label}:")[1].split(":")[0]
            rows = [ln for ln in block.splitlines()
                    if ln.strip().startswith(directive)]
            assert len(rows) == geom.k_eff, (geom.describe(), label,
                                             len(rows))
            for k, line in enumerate(rows):
                got = [int(tok, 0) for tok in
                       line.split(directive)[1].split(",")]
                assert got == [mat[r][k] & mask for r in range(geom.m)], \
                    (geom.describe(), label, k)

        # c_init is the transpose of C, and the row-major RVV copies of A
        # and B are *not* transposed -- the reference path reads contiguous
        # K rows and must keep doing so.
        block = asm.split("\nc_init:")[1].split(":")[0]
        rows = [ln for ln in block.splitlines()
                if ln.strip().startswith(directive)]
        assert len(rows) == geom.n_max, (geom.describe(), len(rows))
        for j, line in enumerate(rows):
            got = [int(tok, 0) for tok in line.split(directive)[1].split(",")]
            assert got == [c[i][j] & mask for i in range(geom.m)], \
                (geom.describe(), j)
        block = asm.split("\nmat_a:")[1].split(":")[0]
        rows = [ln for ln in block.splitlines()
                if ln.strip().startswith(directive)]
        got = [int(tok, 0) for tok in rows[0].split(directive)[1].split(",")]
        assert got == [v & mask for v in a[0]], geom.describe()

        # The generator's C offset must agree with rvv_ref's Sail-derived
        # one at LD = M, in both directions.
        for i in range(geom.m):
            for j in range(geom.n_max):
                assert _c_off(geom, i, j) == \
                    rvv_ref.c_memory_offset_t(i, j, geom.m, geom), \
                    (geom.describe(), i, j)
        op = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul, geom.vl)
        for i in range(op.m):
            for j in range(op.n_max):
                assert _c_off(op, i, j) == \
                    rvv_ref.c_memory_offset(i, j, op.m, op), \
                    (op.describe(), i, j)


def _allocatable_here(geom: TileGeometry) -> bool:
    try:
        VectorAlloc.allocate(geom)
    except ValueError:
        return False
    return geom.emul_c != 16


def check_round_four_emission() -> None:
    """A floating-point program must be the round-one program plus FP.

    The claim round four rests on is that vfmmacc.vv is the *same program*
    as vmmacc.vv with a different funct6 and a different reference path: the
    tile transfers, the vtype sequence, the C tile tail policy and the exact
    bitwise comparison are all unchanged (spec 1500, "The K-dimension,
    tile-dimension formulas, EMUL_C, and instruction-to-widening-factor
    mapping are the same as for the integer family").  This asserts that
    shape, and asserts the three things that are genuinely new.
    """
    import helpers

    for geom in rvv_ref.ime_legal_configs(256, full_vl_only=True,
                                          kinds=("fp",)):
        if geom.emul_c == 16 or not _allocatable_here(geom):
            continue
        asm = emit_test(geom, rvv_ref.random_case(geom, random.Random(4)),
                        "fp_probe")
        peer = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul,
                            geom.vl, 1, "op", "int")
        int_asm = emit_test(peer, rvv_ref.random_case(peer, random.Random(4)),
                            "int_probe")

        # 1. Same five IME instructions, one of which is now vfmmacc.vv.
        assert asm.count(".insn 4,") == 5, geom.describe()
        assert f"# {geom.mnemonic} v" in asm
        assert geom.mnemonic == "vfmmacc.vv"
        for mnemonic in ("vmtl.v", "vmts.v"):
            assert f"# {mnemonic} v" in asm, mnemonic
        words = re.findall(r"\.insn 4, (0x[0-9a-f]+)", asm)
        assert len(words) == 5
        assert ime.decode(int(words[3], 0))[0] == "vfmmacc.vv", words
        #    ... and the other four are byte-identical to the integer
        #    program's, because only the arithmetic opcode moved.
        int_words = re.findall(r"\.insn 4, (0x[0-9a-f]+)", int_asm)
        assert words[:3] + words[4:] == int_words[:3] + int_words[4:], (
            geom.describe())

        # 2. The reference is the scalar F/D unit, not RVV, and it sets frm.
        fsfx = _FP_SUFFIX[geom.sew]
        assert "    csrwi frm, 0" in asm
        assert f"    fmul.{fsfx} ft1, ft1, ft2" in asm
        assert f"    fadd.{fsfx} ft0, ft0, ft1" in asm
        assert _FP_LOAD[geom.sew] in asm and _FP_STORE[geom.sew] in asm
        assert "vmul.vv" not in asm and "vredsum.vs" not in asm, (
            "the floating-point tier must not use the integer RVV path")
        #    One fmul and one fadd per (active element, k): the accumulation
        #    is not reassociated and not fused.
        terms = geom.m * geom.n * geom.k_eff
        assert asm.count(f"fmul.{fsfx}") == terms, geom.describe()
        assert asm.count(f"fadd.{fsfx}") == terms, geom.describe()
        assert "fmadd" not in asm and "fmacc" not in asm, (
            "a fused multiply-add is the rnd=xct disclosure, not rnd=frm")

        # 3. Both extension state fields are enabled.
        assert f"li    t0, {MSTATUS_VS_INITIAL}" in asm
        assert f"li    t0, {MSTATUS_FS_INITIAL}" in asm
        assert f"li    t0, {MSTATUS_FS_INITIAL}" not in int_asm

        # The comparison is untouched: an exact integer compare over the raw
        # SEW-wide bit patterns, with no tolerance and no FP printf.  That is
        # what makes a NaN result testable -- spec 1880-1886 canonicalises
        # every materialised NaN, so a NaN compares equal to a NaN and
        # differs from everything else, exactly like any other value.
        assert asm.count("beq   t4, t5, 1f") == geom.m * geom.n_max
        assert "%f" not in asm and "%e" not in asm and "%g" not in asm
        assert "%llx" not in asm

        # The verdict line keeps helpers._GEOM_RE's three adjacent fields
        # and gains the FP clause at the end.
        assert f"TITAN PASS {geom.describe()}" in asm
        assert geom.describe().endswith(f" FP=binary{geom.sew}")
        match = helpers._GEOM_RE.search(f"TITAN PASS {geom.describe()}")
        assert match and int(match.group("sew")) == geom.sew
        assert int(match.group("vlen")) == geom.vlen
        assert int(match.group("lam")) == geom.lam

    # The data image is the FP bit patterns, emitted at the accumulator
    # width -- W=1, so EEW_A == SEW and A, B and C all use one directive.
    geom = TileGeometry(256, 32, 2, 1, 8, 1, "op", "fp")
    a = [[0x3F800000] * geom.k_eff for _ in range(geom.m)]
    b = [[0xBF800000] * geom.k_eff for _ in range(geom.n_max)]
    c = [[0x00000000] * geom.n_max for _ in range(geom.m)]
    asm = emit_test(geom, (a, b, c), "fp_data")
    assert "    .word 0x3f800000" in asm.replace("0X", "0x").lower()
    want = rvv_ref.fp_gemm_reference(a, b, c, geom)
    #   -1.0 accumulated K_eff times, each a separately rounded product.
    for row in want:
        for value in row:
            assert value == rvv_ref.fp_round(
                1, rvv_ref.Fraction(geom.k_eff), geom.sew), hex(value)


def check_clayout_emission() -> None:
    """Round five's programs must actually do what the tier claims.

    Five things, each of which the tier would be worthless without:

      * no tile *store* anywhere in the program -- the read-back is an
        architectural vse<SEW>.v of the whole C group.  A clayout program
        that still contained a vmts.v would be a pair program with extra
        steps;
      * the read-back is configured at LMUL = EMUL_C and VL = VLMAX, so it
        copies out the entire group and not the first register of it;
      * the expected image in .data is rvv_ref's, element for element, and
        the compare walks it at the spec's mat_C_idx offsets;
      * the transposed image really is the transpose, and really differs
        from the expected one (otherwise `TITAN CLAYOUT transposed` could
        fire on a correct tile); and
      * the tier spans EMUL_C and W rather than one corner of the space --
        at VLEN=256 it must reach EMUL_C 2 and 8, and both vmmacc.vv and
        vqmmacc.vv.
    """
    seen_emul, seen_w = set(), set()
    for geom in rvv_ref.clayout_geometries(256):
        if not _allocatable_here(geom):
            continue
        seen_emul.add(geom.emul_c)
        seen_w.add(geom.w)
        alloc = VectorAlloc.allocate(geom)
        asm = emit_clayout_test(geom)

        # The tile store appears exactly once, in the header prose that says
        # it is *not* used; never as an instruction.
        body_lines = [ln for ln in asm.splitlines()
                      if ln.strip() and not ln.lstrip().startswith("#")]
        assert not any(geom.store_mnemonic in ln for ln in body_lines), \
            geom.describe()
        assert ime.insn(geom.store_mnemonic, vs3=alloc.c, rs1=RS1_ADDR,
                        rs2=RS2_LD, vm=1,
                        **{"lambda": 0}).split(", ")[1] not in asm, \
            geom.describe()
        assert f"# {geom.load_mnemonic} v{alloc.c}, (a0), a1" in asm, \
            geom.describe()
        assert f"# {geom.mnemonic} v{alloc.c}" in asm, geom.describe()

        total = geom.m * geom.n_max
        assert (f"    li    t0, {total}\n"
                f"    vsetvli t1, t0, e{geom.sew}, m{geom.emul_c}, ta, ma\n"
                f"    la    a2, c_raw\n"
                f"    vse{geom.sew}.v v{alloc.c}, (a2)") in asm, \
            geom.describe()

        a, b, c0 = rvv_ref.clayout_case(geom)
        ref = rvv_ref.reference_gemm(a, b, c0, geom)
        exp = rvv_ref.clayout_reg_image(ref, geom)
        xpose = rvv_ref.clayout_reg_image(
            [[ref[j][i] for j in range(geom.n_max)] for i in range(geom.m)],
            geom)
        assert exp != xpose, geom.describe()
        mask = (1 << geom.sew) - 1
        for label, image in (("c_exp", exp), ("c_xpose", xpose)):
            body = asm.split(f"{label}:\n", 1)[1]
            body = body.split("    .balign", 1)[0]
            values = [int(tok, 0) for line in body.splitlines()
                      for tok in line.split(None, 1)[1].split(",")]
            assert values == [v & mask for v in image], (label,
                                                         geom.describe())
        # ... and the transpose really is the transpose of the tile, not of
        # the flat image, which is a different permutation whenever
        # EMUL_C > 1.
        for i in range(geom.m):
            for j in range(geom.n_max):
                flat = rvv_ref.c_element_index(i, j, geom)
                assert exp[flat] == ref[i][j]
                assert xpose[flat] == ref[j][i]
                esz = geom.sew // 8
                assert (f"    li    t0, {flat * esz}          "
                        f"# C[{i},{j}] -> group element {flat}") in asm, (
                    geom.describe(), i, j)

        fmts = dict(re.findall(r'^\.Lfmt_(\w+):\s+\.asciz "(.*)"$', asm,
                               re.M))
        assert set(fmts) == {"pass", "skip", "fail", "diff", "cdump", "cref",
                             "clayout_t", "clayout_o", "elem",
                             "nl"}, sorted(fmts)
        assert fmts["clayout_t"].startswith("TITAN CLAYOUT transposed ")
        assert fmts["clayout_o"].startswith("TITAN CLAYOUT other ")
        assert "%" not in fmts["clayout_t"] + fmts["clayout_o"]

    assert {2, 8} <= seen_emul, sorted(seen_emul)
    assert seen_w == {1, 4}, sorted(seen_w)


def check_clayout_verdict_contract() -> None:
    """helpers.classify_run must read a clayout program's verdicts too.

    The tier prints the *same* markers as the pair tiers -- that is the
    point, so the loop's feedback machinery needs no change to quote its
    evidence back.  The one new line, `TITAN CLAYOUT`, is deliberately not
    a verdict: it is evidence, and it is asserted here to be inert to the
    classifier so that adding it cannot reclassify a run.
    """
    import helpers

    class _Run:
        def __init__(self, log):
            self.log, self.success, self.returncode = log, True, 0

    geom = TileGeometry(256, 32, 1, 8, 64, 1, "op", "int", "clayout")
    asm = emit_clayout_test(geom)
    fmts = dict(re.findall(r'^\.Lfmt_(\w+):\s+\.asciz "(.*)"$', asm, re.M))

    def expand(kind, *args):
        text = fmts[kind].replace("\\n", "\n")
        for value in args:
            text = re.sub(r"%d", str(value), text, count=1)
        return text

    assert helpers.classify_run(_Run(expand("pass"))).kind == "pass"
    failed = helpers.classify_run(_Run(expand("fail", 3, 2)))
    assert failed.kind == "mismatch" and failed.failed, failed
    assert (failed.row, failed.col) == (3, 2), failed
    assert failed.vlen == 256 and failed.sew == 32, failed

    # The evidence lines the loop quotes back verbatim.
    diff = expand("diff", 3, 2).replace("0x%x", "0x2a", 1) \
                               .replace("0x%x", "0x2b", 1)
    out = helpers.classify_run(_Run(diff + expand("fail", 3, 2)))
    assert out.diffs and out.diffs[0].startswith("TITAN DIFF r=3 c=2"), out

    # `TITAN CLAYOUT` must not read as a verdict on its own.
    alone = helpers.classify_run(_Run(expand("clayout_t")))
    assert alone.failed and alone.kind != "pass", alone


def check_verdict_contract() -> None:
    """helpers.classify_run must read what these programs actually print.

    The format strings live here and the regexes live there, so nothing but a
    test keeps them in step.  Rather than restate the markers, this pulls the
    real `.asciz` strings out of an emitted program and expands them the way
    printf would -- so editing a format string without editing the parser
    fails here rather than silently reclassifying every result at runtime.
    """
    import helpers

    class _Run:
        def __init__(self, log): self.log, self.success, self.returncode = log, True, 0

    geom = TileGeometry(256, 32, 2, 1, 8)
    asm = emit_test(geom, rvv_ref.random_case(geom, random.Random(0)))
    fmts = dict(re.findall(r'^\.Lfmt_(\w+):\s+\.asciz "(.*)"$', asm, re.M))
    assert set(fmts) == {"pass", "skip", "fail", "diff", "cdump", "cref",
                         "elem", "nl"}, sorted(fmts)

    def expand(kind: str, *args: int) -> str:
        text = fmts[kind].replace("\\n", "\n")
        for value in args:
            text = re.sub(r"%d", str(value), text, count=1)
        return text

    assert helpers.classify_run(_Run(expand("pass"))).kind == "pass"

    # geom requests LAMBDA=2 at VLEN=256, SEW=32, where {1, 2} are the
    # architecturally permissible values.  A DUT that clamps down to 1 has
    # done what WARL allows; one that answers 4 has not, and calling that a
    # skip is what let five iterations pass with 15 tests untested.
    skipped = helpers.classify_run(_Run(expand("skip", 1, 1)))
    assert skipped.kind == "skip" and not skipped.failed, skipped
    assert (skipped.requested_lambda, skipped.selected_lambda) == (2, 1)

    clamped_up = helpers.classify_run(_Run(expand("skip", 4, 3)))
    assert clamped_up.kind == "bad_geometry" and clamped_up.failed, clamped_up
    assert (clamped_up.requested_lambda,
            clamped_up.selected_lambda) == (2, 4), clamped_up

    # ... unless the DUT supports nothing at or below the request, which is
    # the one case the spec lets it round up.  That is a property of the run
    # as a whole, not of one program, so helpers.reconcile_geometry decides
    # it once every outcome is in.
    alone = helpers.reconcile_geometry([("up", clamped_up)])
    assert alone[0][1].kind == "skip", alone
    with_evidence = helpers.reconcile_geometry(
        [("up", clamped_up),
         ("low", helpers.classify_run(_Run(expand("pass"))))])
    assert with_evidence[0][1].kind == "bad_geometry", with_evidence

    failed = helpers.classify_run(_Run(expand("fail", 3, 2)))
    assert failed.kind == "mismatch" and failed.failed, failed
    assert (failed.row, failed.col) == (3, 2), failed

    # A program that printed nothing must never read as a pass.
    for run in (_Run(""), _Run("some unrelated simulator chatter")):
        assert helpers.classify_run(run).failed



# ---------------------------------------------------------------------------
# round six: self-tests
# ---------------------------------------------------------------------------
#
# Every one of these carries at least one **negative control**: the model is
# sabotaged in the one way the tier exists to catch, and the tier is
# required to go red.  Round five's lesson, stated in its design note and
# repeated here because it is the whole reason this section is longer than
# the generator it tests: a tier that cannot fail is not a tier.
#
# The controls all sabotage the *generator* and then judge with the honest
# model.  Patching both sides would let the two errors cancel -- which is
# precisely how a transposed C tile survived four rounds of green.


class _sabotage:
    """Temporarily replace attributes, for a negative control."""

    def __init__(self, module, **attrs):
        self.module, self.attrs, self.saved = module, attrs, {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.saved[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(self.module, name, value)
        return False


def _must_fail(what: str, fn) -> None:
    """Assert *fn* raises -- the negative-control assertion itself."""
    try:
        fn()
    except (AssertionError, ValueError, KeyError, IndexError):
        return
    raise AssertionError(f"negative control has no teeth: {what} passed")


def check_mx_scale_image() -> None:
    """The v0 paired-scale image must be the spec's, not a plausible one.

    Three claims, and the third is the one with the history: the low byte is
    scale_A and the high byte scale_B (spec 2161-2170); every pair position
    is accounted for, because M*R == VLEN/pw exactly (spec 2192-2197); and
    the pair index is ``m*R + s``, row stride outermost.

    Negative control: swap the index to ``s*R + m``.  The image must change
    at every geometry whose R is greater than 1 -- which is every geometry
    with more than one block, i.e. exactly the ones the ime_mxs_ tier is
    built from.
    """
    swapped_somewhere = False
    for geom, _bs in _mx_geometries(256, "one"):
        r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
        sa = [[(m * 7 + s + 1) & 0x7F for s in range(r)]
              for m in range(geom.m)]
        sb = [[(m * 5 + s * 3 + 2) & 0x7F for s in range(r)]
              for m in range(geom.m)]
        image = mx_scale_image(geom, sa, sb)
        assert len(image) == geom.vlen // rvv_ref.MX_PAIR_WIDTH
        for m in range(geom.m):
            for s in range(r):
                pair = image[rvv_ref.mx_pair_index(m, s, r)]
                assert pair & 0xFF == sa[m][s], (geom.describe(), m, s)
                assert (pair >> 8) & 0xFF == sb[m][s], (geom.describe(), m, s)

        def swapped(m, s, rr):
            return s * rr + m

        if r > 1:
            # The swapped index either lands on a different byte or walks
            # off the end of the array entirely (M=2, R=8 at SEW=64 puts
            # s*R+m as high as 57 in a 16-element register).  Both are the
            # control firing; only "same image" is the control being inert.
            with _sabotage(rvv_ref, mx_pair_index=swapped):
                try:
                    other = mx_scale_image(geom, sa, sb)
                except IndexError:
                    other = None
            assert other != image, geom.describe()
            swapped_somewhere = True
    assert swapped_somewhere, "no R > 1 geometry: the layout control is inert"

    # The padding rule, as data rather than as prose: every position the
    # architecture must not read carries the E8M0 NaN code.
    geom, bs = next(g for g in _mx_geometries(256, "one")
                    if rvv_ref.mx_scale_stride(g[0].sew, g[0].lam) > 1)
    r = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
    sa, sb = _mx_scale_arrays(geom, 1, lambda m, s: rvv_ref.MX_E8M0_BIAS,
                              lambda m, s: rvv_ref.MX_E8M0_BIAS)
    assert all(row[s] == MX_POISON_SCALE for row in sa for s in range(1, r))
    assert all(row[s] == MX_POISON_SCALE for row in sb for s in range(1, r))
    # ... and the reference model really does ignore them: poisoning the
    # padding must not change the answer.
    a, b, c = _mx_operands(geom, 0, bs, random.Random(1))
    clean_a = [[rvv_ref.MX_E8M0_BIAS] * r for _ in range(geom.m)]
    clean_b = [[rvv_ref.MX_E8M0_BIAS] * r for _ in range(geom.m)]
    poisoned = rvv_ref.int_scaled_gemm_reference(a, b, c, sa, sb, geom, bs=bs)
    clean = rvv_ref.int_scaled_gemm_reference(a, b, c, clean_a, clean_b,
                                              geom, bs=bs)
    assert poisoned == clean, geom.describe()


def _check_mx_program(plan: MxPlan, asm: str) -> None:
    """The structural invariants every ime_mx_ / ime_mxs_ / ime_mxn_ has."""
    geom = plan.geom
    alloc = VectorAlloc.allocate(geom, reserve_v0=True)

    # Five IME instructions, exactly as every pair program since round one:
    # load C, load A, load B, multiply-accumulate, store C.
    words = re.findall(r"\.insn 4, (0x[0-9a-f]+)", asm)
    assert len(words) == 5, (plan.describe(), words)
    assert ime.decode(int(words[3], 0))[0] == geom.mnemonic, words
    assert ime.decode(int(words[3], 0))[1]["vd"] == alloc.c

    # v0 holds the scales, so the A tile does not.
    assert alloc.a != 0 and alloc.b != 0, plan.describe()
    assert f"    vle16.v v0, (a0)" in asm, plan.describe()
    assert f"    la    a0, v0_scales" in asm, plan.describe()
    assert f"# {geom.load_mnemonic} v{alloc.a}, (a0), a1" in asm

    # vtype carries bs and altfmt on every configuration.
    for lmul in {geom.lmul, geom.lmul_c}:
        want = vtype_value(geom, lmul=lmul, altfmt=plan.altfmt, bs=plan.bs)
        assert f"    li    t1, 0x{want:x}" in asm, (plan.describe(), lmul)
    if plan.bs:
        offset, _width = ime.VTYPE_IME_FIELDS["bs"]
        assert (vtype_value(geom, lmul=geom.lmul, bs=1) >> (64 - offset)) & 1
    if plan.altfmt:
        lsb, _width = ime.VTYPE_BASE_FIELDS["altfmt"]
        assert (vtype_value(geom, lmul=geom.lmul, altfmt=1) >> lsb) & 1

    # The reference path is integer end to end: no scalar FP unit is even
    # enabled, which is what lets the tier cover binary16 and bfloat16.
    assert f"li    t0, {MSTATUS_FS_INITIAL}" not in asm, plan.describe()
    for banned in ("fmul.", "fadd.", "fmadd", "fcvt", "flw", "fld"):
        assert banned not in asm, (banned, plan.describe())

    # The comparison is the usual exact one over every physical element.
    assert asm.count("beq   t4, t5, 1f") == geom.m * geom.n_max
    assert "%f" not in asm and "%llx" not in asm

    # The verdict line keeps helpers._GEOM_RE's three adjacent fields and
    # names the accumulator format, which describe() cannot.
    import helpers
    line = f"TITAN PASS {plan.describe()}"
    assert line in asm
    match = helpers._GEOM_RE.search(line)
    assert match and int(match.group("sew")) == geom.sew
    assert f"MXC={plan.fmt}" in line and f"bs={plan.bs}" in line


def check_mx_emission() -> None:
    """Round six's exactness tier: every cell, every element differential.

    The claims:

      * all seven (W, SEW, altfmt) cells of tbl-intmx-encoding-map are
        reached -- the tier enumerates `altfmt` from rvv_ref.mx_altfmts, so
        "seven" is the table's number, not one written down here;
      * every program is a single-block program with both scales 2**0;
      * every *active* element is recomputed on the DUT.  Nothing falls back
        to a literal, which is what "exact" means operationally: if the
        operand bound stopped keeping the dot product inside the
        significand, elements would start arriving as constants and this
        would fail;
      * and the whole answer re-derived from the emitted .data agrees with
        rvv_ref.int_scaled_gemm_reference.

    Two negative controls, each sabotaging the generator and judging with
    the honest model:

      * the v0 pair index becomes ``s*R + m`` -- the row-stride mistake
        round6_design.md names as the most likely implementation bug;
      * the operand bound is blown open to the full integer range, so the
        dot product no longer fits the significand and int_to_fp starts
        rounding.  The tier must notice that it has stopped being exact
        rather than quietly emitting rounded constants.
    """
    cells, plans, tails = set(), [], 0
    for plan in mx_exact_plans(256):
        assert plan.blocks == 1, plan.describe()
        assert all(b == rvv_ref.MX_E8M0_BIAS
                   for row in plan.scales_a for b in row[:1])
        asm = emit_mx_test(plan, "mx_probe")
        _check_mx_program(plan, asm)
        _ref, elements = mx_element_plan(plan)
        for i in range(plan.geom.m):
            for j in range(plan.geom.n):
                assert elements[(i, j)][0] == MX_MODE_COMPUTE, \
                    (plan.describe(), i, j)
                assert elements[(i, j)][2] == 0, "2**0 needs no exponent bump"
        if plan.geom.n < plan.geom.n_max:
            # The two things that exist only at partial N.
            tails += 1
            assert any(elements[(i, j)][0] == MX_MODE_COPY
                       for i in range(plan.geom.m)
                       for j in range(plan.geom.n, plan.geom.n_max)), \
                plan.describe()
            assert any(plan.scales_b[m][0] == MX_POISON_SCALE
                       for m in range(plan.geom.n, plan.geom.m)), \
                plan.describe()
        mx_reconstruct(asm, plan)
        cells.add((plan.geom.w, plan.geom.sew, plan.altfmt))
        plans.append(plan)
    assert len(cells) == len(
        [1 for (w, sew), (_e, fmts) in rvv_ref.MX_CELLS.items()
         for _f in fmts]) == 7, sorted(cells)
    assert tails, "the tier never reaches N < N_max: the C tail policy and " \
                  "the inactive scale_B fields are untested"

    # Negative control 1: the v0 row stride.
    def swapped(m, s, r):
        return s * r + m

    victim = next(p for p in plans
                  if rvv_ref.mx_scale_stride(p.geom.sew, p.geom.lam) > 1)
    with _sabotage(rvv_ref, mx_pair_index=swapped):
        bad = emit_mx_test(victim, "mx_probe")
    _must_fail("v0 pair index s*R+m", lambda: mx_reconstruct(bad, victim))

    # Negative control 2: exactness.  bfloat16 has 8 significand bits, so
    # the full Int8 range overflows it by a mile.
    bf16 = next(p for p in plans if p.fmt == "bfloat16")

    def unbounded(geom, altfmt, bs):
        return (1 << (rvv_ref.mx_legal_cell(geom.w, geom.sew,
                                            altfmt)[0] - 1)) - 1

    def rebuild():
        with _sabotage(sys.modules[__name__],
                       _mx_operand_bound=unbounded):
            a, b, c = _mx_operands(bf16.geom, bf16.altfmt, bf16.bs,
                                   random.Random(7))
        loose = MxPlan(bf16.geom, bf16.altfmt, bf16.bs, a, b, c,
                       bf16.scales_a, bf16.scales_b, "mx", "unbounded")
        _ref, elements = mx_element_plan(loose)
        for i in range(loose.geom.m):
            for j in range(loose.geom.n):
                assert elements[(i, j)][0] == MX_MODE_COMPUTE, (i, j)

    _must_fail("operand bound blown open", rebuild)


def check_mxs_emission() -> None:
    """Round six's scale tier: the block loop and the R-strided v0 layout.

    What this tier has that the exactness tier does not is *variation along
    both axes of v0*.  Both scale arrays are a function of the row index and
    of the block index, so reading the pair at ``s*R + m`` instead of
    ``m*R + s`` lands on a different byte and produces a different exponent;
    and both are checked here to actually vary, because a tier whose scale
    array happened to be constant would be an exactness tier with extra
    steps.

    The combined exponent ``E = dA(i) + dB(j)`` is constant across blocks by
    construction, which is what keeps the accumulation exact -- and is
    asserted per element, not assumed, by mx_element_plan.

    Negative controls:

      * the v0 pair index swap again, which here bites at *every* geometry
        rather than only at R > 1;
      * ``read_block_scales`` reduced to reading scale_A and ignoring
        scale_B -- a one-line implementation slip that the exactness tier,
        where both bytes are 0x7F, cannot see at all.
    """
    plans, blocks_seen, bs_seen = [], set(), set()
    for plan in mx_scale_plans(256):
        assert plan.blocks >= 2, plan.describe()
        geom = plan.geom
        # Both arrays vary along both axes.
        rows = {tuple(row[:plan.blocks]) for row in plan.scales_a}
        assert len(rows) > 1, plan.describe()
        assert any(len(set(row[:plan.blocks])) > 1 for row in plan.scales_a)
        assert any(len(set(row[:plan.blocks])) > 1
                   for row in plan.scales_b[:geom.n])
        asm = emit_mx_test(plan, "mxs_probe")
        _check_mx_program(plan, asm)
        _ref, elements = mx_element_plan(plan)
        bumps = set()
        for i in range(geom.m):
            for j in range(geom.n):
                assert elements[(i, j)][0] == MX_MODE_COMPUTE, \
                    (plan.describe(), i, j)
                bumps.add(elements[(i, j)][2])
        assert len(bumps) > 1, (plan.describe(),
                                "every element has the same E: the exponent "
                                "path is not being exercised")
        mx_reconstruct(asm, plan)
        plans.append(plan)
        blocks_seen.add(plan.blocks)
        bs_seen.add(plan.bs)
    assert plans, "the scale tier is empty"
    assert bs_seen == {0, 1}, sorted(bs_seen)
    assert max(blocks_seen) >= 2, sorted(blocks_seen)

    def swapped(m, s, r):
        return s * r + m

    victim = plans[0]
    with _sabotage(rvv_ref, mx_pair_index=swapped):
        bad = emit_mx_test(victim, "mxs_probe")
    _must_fail("v0 pair index s*R+m", lambda: mx_reconstruct(bad, victim))

    honest_scale = rvv_ref.mx_block_scale

    def scale_a_only(scale_a, scale_b, width, fmt):
        return honest_scale(scale_a, rvv_ref.MX_E8M0_BIAS, width, fmt)

    with _sabotage(rvv_ref, mx_block_scale=scale_a_only):
        bad = emit_mx_test(victim, "mxs_probe")
    _must_fail("read_block_scales ignoring scale_B",
               lambda: mx_reconstruct(bad, victim))


def check_mxn_emission() -> None:
    """Round six's NaN tier: the early exit and the finite +0 x +inf pair.

    Three claims:

      * a 0xFF scale byte makes exactly the row (or column) it belongs to
        the default NaN, and leaves the rest of the tile alone.  Both halves
        matter: an implementation that NaN-ed the whole tile would satisfy
        the first and fail the second;
      * the poison works at a block index other than zero, which is what
        separates "checks every block's scale" from "checks block 0"; and
      * a binary16 accumulator with the *finite* codes 0x00 and 0xFE
        produces the default NaN, because the converted pair is +0 x +inf
        (spec 2021-2024).  This is the case a model that tests the encoded
        bytes for 0xFF instead of testing the product gets wrong.

    Negative controls:

      * ``read_block_scales`` never reporting NaN -- the early exit deleted.
        Every planted element must stop being the default NaN;
      * ``S_blocks`` forced to 1 -- an implementation that only looks at
        block 0.  The later-block planting must stop producing a NaN.
    """
    variants, nan_plans = set(), []
    for suffix, plan in mx_nan_plans(256):
        geom = plan.geom
        width, fmt = geom.sew, plan.fmt
        nan = rvv_ref.fp_default_nan(width, fmt)
        asm = emit_mx_test(plan, "mxn_probe")
        _check_mx_program(plan, asm)
        ref, elements = mx_element_plan(plan)
        planted = [(i, j) for i in range(geom.m) for j in range(geom.n)
                   if ref[i][j] == nan]
        assert planted, (suffix, plan.describe())
        clean = [(i, j) for i in range(geom.m) for j in range(geom.n)
                 if elements[(i, j)][0] == MX_MODE_COMPUTE]
        assert clean, (suffix, plan.describe(),
                       "the whole tile went NaN: the poison is not localised")
        for i, j in planted:
            assert elements[(i, j)][0] == MX_MODE_LITERAL, (i, j)
        mx_reconstruct(asm, plan)
        variants.add(suffix)
        nan_plans.append((suffix, plan, nan))
    assert variants == {"a0", "bl", "pm"}, sorted(variants)

    # The spec 2021-2024 case, spelled out: both bytes finite, product NaN.
    suffix, plan, nan = next(t for t in nan_plans if t[0] == "pm")
    assert plan.fmt == "binary16"
    assert 0x00 in {b for row in plan.scales_a for b in row}
    assert 0xFE in {b for row in plan.scales_b for b in row}
    for byte in (0x00, 0xFE):
        _bits, is_nan = rvv_ref.mx_decode_scale(byte, plan.geom.sew, "binary16")
        assert not is_nan, hex(byte)
    assert rvv_ref.mx_block_scale(0x00, 0xFE, plan.geom.sew, "binary16")[1]

    # Negative control 1: 0xFF decoded as an ordinary value.
    #
    # Note what this control is *not*: it is not "delete the early exit".
    # Sail 5390-5392's `break` is unobservable in the result -- a NaN block
    # scale multiplies and adds into the accumulator as a NaN anyway, so the
    # element comes out the default NaN with or without it, and only fflags
    # (spec 1815-1824, out of scope) can tell the two apart.  Saying so here
    # matters, because a control that "passes" for that reason would be
    # exactly the toothless tier this section exists to prevent.
    #
    # What the tier does catch is an implementation that never recognises
    # 0xFF at all -- the plausible RTL slip, since E8M0 has no other special
    # code -- so that is what is sabotaged.
    honest_decode = rvv_ref.mx_decode_scale

    def nan_blind(byte, width, fmt):
        if byte == rvv_ref.MX_E8M0_NAN:
            return rvv_ref.fp_one(width, fmt), False
        return honest_decode(byte, width, fmt)

    def blind_to_the_nan_code():
        for suffix, plan, nan in nan_plans:
            with _sabotage(rvv_ref, mx_decode_scale=nan_blind):
                ref, _elements = mx_element_plan(plan)
            assert any(ref[i][j] == nan for i in range(plan.geom.m)
                       for j in range(plan.geom.n)), (suffix,
                                                      plan.describe())

    _must_fail("0xFF decoded as an ordinary scale", blind_to_the_nan_code)

    # Negative control 3: the paired scale computed by *adding exponents*
    # instead of converting each byte to fmt_C and multiplying there.  That
    # is the shortcut an RTL designer reaches for -- E8M0 is exponent-only,
    # so why not? -- and spec 2008-2024 is the answer: 0x00 and 0xFE are
    # 2**-127 and 2**127, which in binary16 are +0 and +inf, and their
    # product is the default NaN even though the exponents sum to zero.
    honest_scale = rvv_ref.mx_block_scale
    suffix, pm_plan, pm_nan = next(t for t in nan_plans if t[0] == "pm")

    def exponent_add(scale_a, scale_b, width, fmt):
        if rvv_ref.MX_E8M0_NAN in (scale_a, scale_b):
            return honest_scale(scale_a, scale_b, width, fmt)
        bits = _pow2_bits((scale_a - rvv_ref.MX_E8M0_BIAS)
                          + (scale_b - rvv_ref.MX_E8M0_BIAS), width, fmt)
        if bits is None:
            return honest_scale(scale_a, scale_b, width, fmt)
        return bits, False

    def scales_multiplied_in_fmt_c():
        with _sabotage(rvv_ref, mx_block_scale=exponent_add):
            ref, _elements = mx_element_plan(pm_plan)
        assert any(ref[i][j] == pm_nan for i in range(pm_plan.geom.m)
                   for j in range(pm_plan.geom.n)), pm_plan.describe()

    _must_fail("block scale by exponent addition",
               scales_multiplied_in_fmt_c)

    # Negative control 2: only block 0 is inspected.
    suffix, plan, nan = next(t for t in nan_plans if t[0] == "bl")

    def one_block(k_eff, block_size):
        return 1

    def only_block_zero():
        with _sabotage(rvv_ref, mx_block_count=one_block):
            ref, _elements = mx_element_plan(plan)
        assert any(ref[i][j] == nan for i in range(plan.geom.m)
                   for j in range(plan.geom.n)), plan.describe()

    _must_fail("only block 0 inspected", only_block_zero)


def check_mxl_emission() -> None:
    """Round six's legality tier, including the vm=1 regression guard.

    The invariants, each of which is a different way for the tier to be
    worthless:

      * both polarities are present.  A tier of nothing but "must trap"
        cases is passed by a decoder that traps on all three funct6 values,
        which would break rounds one to three;
      * every case whose vm is 1 is required *not* to trap, and its word
        decodes to the already-implemented integer multiply-accumulate --
        the MX word and the integer word differ in exactly bit 25;
      * every reserved (W, SEW) cell of the encoding map is probed, and
        every altfmt_A / altfmt_B = 1 case is required to trap;
      * the program installs and restores a trap handler, and restores
        mtvec on every exit path including the failing one -- the harness
        has its own handler and printf runs through it.

    Negative controls:

      * ``check_microscaling_legality`` neutered.  The bs=1 case must stop
        being a trap case, which the invariant catches;
      * the vm=1 peer table pointed back at the MX mnemonic, so the two
        words no longer differ in the vm bit.  Generation must refuse.
    """
    seen_reserved, polarity = set(), set()
    for mnemonic in ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv"):
        w = {"vfwimmacc.vv": 2, "vfqimmacc.vv": 4,
             "vf8wimmacc.vv": 8}[mnemonic]
        cases = mxl_cases(256, mnemonic)
        asm = emit_mxl_test(256, mnemonic, "mxl_probe")
        assert asm.count(".insn 4,") == len(cases), mnemonic
        for index, case in enumerate(cases):
            polarity.add(case.expect_trap)
            back, ops = ime.decode(case.word)
            assert back == case.mnemonic
            assert ops == {"vd": MXL_VD, "vs1": MXL_VS1, "vs2": MXL_VS2}
            if case.vm:
                assert case.expect_trap == 0, (mnemonic, index)
                assert case.mnemonic == MXL_VM1_PEER[mnemonic]
                assert case.mnemonic in ("vwmmacc.vv", "vqmmacc.vv",
                                         "v8wmmacc.vv")
            else:
                assert case.mnemonic == mnemonic
            if (w, case.geom.sew) not in rvv_ref.MX_CELLS and not case.vm:
                assert case.expect_trap == 1, (mnemonic, index)
                seen_reserved.add((w, case.geom.sew))
            assert f"    .insn 4, {case.word:#010x}" in asm
            assert f"    li    t1, 0x{case.vtype:x}" in asm
        # Exactly one case per mnemonic is the positive control at its own
        # canonical cell, and it comes first so a total-trap decoder fails
        # on case 0 rather than deep in the list.
        assert cases[0].expect_trap == 0 and cases[0].vm == 0, mnemonic

        # The handler, and mtvec discipline.
        assert ".Ltrap:" in asm and "    mret" in asm
        assert "    csrr  s8, mtvec" in asm
        assert asm.count("    csrw  mtvec, s8") == 3, mnemonic
        assert asm.index(".Ltrap:") > asm.index(".Lret:"), mnemonic
        assert "TITAN PASS" in asm and "TITAN FAIL row=%d" in asm

    # Every reserved cell in the table is probed by some program.
    for w in (2, 4, 8):
        for sew in (8, 16, 32, 64):
            if (w, sew) not in rvv_ref.MX_CELLS:
                assert (w, sew) in seen_reserved, (w, sew)
    assert polarity == {0, 1}

    # Negative control 1: neuter check_microscaling_legality.
    def permissive(w, lmul, sew, lam, bs=0):
        return None

    def bs_case_still_traps():
        with _sabotage(rvv_ref, mx_check_legality=permissive):
            cases = mxl_cases(256, "vf8wimmacc.vv")
        bs_cases = [c for c in cases if "W*LMUL" in c.tag]
        assert bs_cases, "no bs=1 case at all"
        assert all(c.expect_trap for c in bs_cases), \
            "bs=1 with W*LMUL > SEW stopped being a trap case"

    _must_fail("check_microscaling_legality neutered", bs_case_still_traps)

    # Negative control 2: point the vm=1 peer back at the MX mnemonic.
    def vm_confused():
        with _sabotage(sys.modules[__name__],
                       MXL_VM1_PEER={"vfwimmacc.vv": "vfwimmacc.vv",
                                     "vfqimmacc.vv": "vfqimmacc.vv",
                                     "vf8wimmacc.vv": "vf8wimmacc.vv"}):
            mxl_cases(256, "vfwimmacc.vv")

    _must_fail("vm=1 routed back to the MX form", vm_confused)


def check_mx_verdict_contract() -> None:
    """helpers.classify_run must read round six's verdicts too.

    Same contract as rounds one to five -- the markers are deliberately
    unchanged, so the loop's feedback machinery needs no round-six branch --
    with one addition that has to be checked rather than hoped for: the
    geometry clause now carries ``MXC=<format>`` after ``describe()``'s own
    fields, and helpers._GEOM_RE scrapes ``VLEN= SEW= LAMBDA=`` as three
    adjacent fields.  Appending would break that if it were inserted.
    """
    import helpers

    class _Run:
        def __init__(self, log):
            self.log, self.success, self.returncode = log, True, 0

    def expand(fmts, kind, *args):
        text = fmts[kind].replace("\\n", "\n")
        for value in args:
            text = re.sub(r"%d", str(value), text, count=1)
        return text

    plan = next(iter(mx_exact_plans(256)))
    asm = emit_mx_test(plan, "mx_probe")
    fmts = dict(re.findall(r'^\.Lfmt_(\w+):\s+\.asciz "(.*)"$', asm, re.M))
    assert set(fmts) == {"pass", "skip", "fail", "diff", "cdump", "cref",
                         "elem", "nl"}, sorted(fmts)
    assert helpers.classify_run(_Run(expand(fmts, "pass"))).kind == "pass"
    failed = helpers.classify_run(_Run(expand(fmts, "fail", 3, 2)))
    assert failed.kind == "mismatch" and failed.failed, failed
    assert (failed.row, failed.col) == (3, 2), failed
    assert failed.vlen == 256 and failed.sew == plan.geom.sew, failed
    skipped = helpers.classify_run(_Run(expand(fmts, "skip", 1, 1)))
    assert skipped.kind in ("skip", "bad_geometry"), skipped
    assert skipped.requested_lambda == plan.geom.lam, skipped

    mxl = emit_mxl_test(256, "vfwimmacc.vv", "mxl_probe")
    fmts = dict(re.findall(r'^\.Lfmt_(\w+):\s+\.asciz "(.*)"$', mxl, re.M))
    assert set(fmts) == {"pass", "skip", "fail", "diff"}, sorted(fmts)
    assert helpers.classify_run(_Run(expand(fmts, "pass"))).kind == "pass"
    failed = helpers.classify_run(_Run(expand(fmts, "fail", 5, 1)))
    assert failed.kind == "mismatch" and failed.failed, failed
    assert failed.row == 5, failed
    for run in (_Run(""), _Run("unrelated simulator chatter")):
        assert helpers.classify_run(run).failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument(
        "--emit", metavar="SEW,LAMBDA,LMUL,N[,W][,op|t][,pair|clayout]",
        help="print one program instead of running self-tests")
    parser.add_argument(
        "--emit-mx", metavar="NAME",
        help="print one round-six program by name (ime_mx_..., ime_mxs_..., "
             "ime_mxn_..., ime_mxl_...), or list them all if NAME is 'list'")
    args = parser.parse_args()

    if args.emit_mx:
        tiers = mx_directed_tiers(
            args.vlen, ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv"))
        if args.emit_mx == "list":
            for name, _asm, geom in tiers:
                print(f"{name}  {geom.describe()}")
            return 0
        for name, asm, _geom in tiers:
            if name == args.emit_mx:
                print(asm)
                return 0
        print(f"no round-six program named {args.emit_mx!r}; "
              f"--emit-mx list shows them all", file=sys.stderr)
        return 1

    if args.emit:
        toks = args.emit.split(",")
        tload = "op"
        if toks[-1] in ("op", "t"):
            tload = toks.pop()
        check = "pair"
        if toks[-1] in ("pair", "clayout"):
            check = toks.pop()
        sew, lam, lmul, n, *rest = (int(x) for x in toks)
        geom = TileGeometry(args.vlen, sew, lam, lmul, n * lam * lmul,
                            rest[0] if rest else 1, tload, "int", check)
        if check == "clayout":
            print(emit_clayout_test(geom))
        else:
            print(emit_test(geom, rvv_ref.random_case(geom, random.Random(0))))
        return 0

    for check in (check_vtype_fields, check_allocation_disjoint,
                  check_insn_words_decode, check_emitted_structure,
                  check_widening_emission, check_transposing_emission,
                  check_widening_verdict_geometry,
                  check_round_four_emission,
                  check_clayout_emission,
                  check_verdict_contract,
                  check_clayout_verdict_contract,
                  check_mx_scale_image,
                  check_mx_emission,
                  check_mxs_emission,
                  check_mxn_emission,
                  check_mxl_emission,
                  check_mx_verdict_contract):
        check()
        print(f"  ok  {check.__name__}")

    for label, full_vl_only in (("per-iteration", True), ("S1 gate", False)):
        suite = directed_suite(args.vlen, full_vl_only=full_vl_only)
        lines = sum(a.count("\n") for _, a, _ in suite)
        tally = {}
        for _, _, g in suite:
            key = (g.check, g.kind, g.w, g.tload)
            tally[key] = tally.get(key, 0) + 1
        breakdown = " + ".join(f"{n} {chk}/{kind} W={w}/{tl}"
                               for (chk, kind, w, tl), n in sorted(tally.items()))
        print(f"\nVLEN={args.vlen} {label}: {len(suite)} programs "
              f"({breakdown}), {lines:,} lines of assembly")
        # Round six's four tiers all land in the same (check, kind, W, tload)
        # buckets as each other, so the line above cannot separate them.
        # They are counted again by name prefix -- and only when there are
        # any, so the output of a round one to five scope is unchanged.
        tiers = {}
        for name, _asm, _g in suite:
            for prefix in ("ime_mx_", "ime_mxs_", "ime_mxn_", "ime_mxl_"):
                if name.startswith(prefix):
                    tiers[prefix] = tiers.get(prefix, 0) + 1
        if tiers:
            print("  round six by tier: "
                  + " + ".join(f"{n} {p}" for p, n in sorted(tiers.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
