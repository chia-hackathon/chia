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
    def allocate(geom: TileGeometry) -> "VectorAlloc":
        c_base = 32 - geom.emul_c
        if 2 * geom.lmul > c_base:
            raise ValueError(
                f"{geom.describe()}: A, B (2 x LMUL={geom.lmul}) and C "
                f"(EMUL_C={geom.emul_c}) do not fit in 32 vector registers")
        return VectorAlloc(a=0, b=geom.lmul, c=c_base)


def vtype_value(geom: TileGeometry, *, lmul: int, xlen: int = 64,
                vta: int = 0, vma: int = 0, altfmt_a: int = 0,
                altfmt_b: int = 0, bs: int = 0) -> int:
    """Assemble a full vtype word, IME fields included.

    vsetvli/vsetivli cannot reach the IME fields -- they live above the
    vtypei immediate -- so every configuration here goes through the register
    form, `vsetvl`, which writes vtype wholesale from rs2.
    """
    value = (_VLMUL_FIELD[lmul] | (_VSEW_FIELD[geom.sew] << 3)
             | (vta << 6) | (vma << 7))
    for name, field in (("lambda", ime.lambda_imm(geom.lam)), ("bs", bs),
                        ("altfmt_A", altfmt_a), ("altfmt_B", altfmt_b)):
        offset, width = ime.VTYPE_IME_FIELDS[name]
        assert field < (1 << width)
        value |= field << (xlen - offset)
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


def _ime_path(geom: TileGeometry, alloc: VectorAlloc) -> List[str]:
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

    for prefix, tier_sews, ws, tloads, kinds in tiers:
        for geom in rvv_ref.ime_legal_configs(vlen, sews=tier_sews,
                                              lmuls=lmuls,
                                              full_vl_only=full_vl_only,
                                              ws=ws, tloads=tloads,
                                              kinds=kinds):
            if geom.emul_c == 16:
                continue
            try:
                VectorAlloc.allocate(geom)
            except ValueError:
                continue
            name = (f"{prefix}sew{geom.sew}_lam{geom.lam}_lmul{geom.lmul}"
                    f"_n{geom.n}")
            out.append((name, emit_test(geom, rvv_ref.random_case(geom, rng),
                                        name), geom))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument("--emit", metavar="SEW,LAMBDA,LMUL,N[,W][,op|t]",
                        help="print one program instead of running self-tests")
    args = parser.parse_args()

    if args.emit:
        toks = args.emit.split(",")
        tload = "op"
        if toks[-1] in ("op", "t"):
            tload = toks.pop()
        sew, lam, lmul, n, *rest = (int(x) for x in toks)
        geom = TileGeometry(args.vlen, sew, lam, lmul, n * lam * lmul,
                            rest[0] if rest else 1, tload)
        print(emit_test(geom, rvv_ref.random_case(geom, random.Random(0))))
        return 0

    for check in (check_vtype_fields, check_allocation_disjoint,
                  check_insn_words_decode, check_emitted_structure,
                  check_widening_emission, check_transposing_emission,
                  check_widening_verdict_geometry,
                  check_round_four_emission,
                  check_verdict_contract):
        check()
        print(f"  ok  {check.__name__}")

    for label, full_vl_only in (("per-iteration", True), ("S1 gate", False)):
        suite = directed_suite(args.vlen, full_vl_only=full_vl_only)
        lines = sum(a.count("\n") for _, a, _ in suite)
        tally = {}
        for _, _, g in suite:
            key = (g.kind, g.w, g.tload)
            tally[key] = tally.get(key, 0) + 1
        breakdown = " + ".join(f"{n} {kind} W={w}/{tl}"
                               for (kind, w, tl), n in sorted(tally.items()))
        print(f"\nVLEN={args.vlen} {label}: {len(suite)} programs "
              f"({breakdown}), {lines:,} lines of assembly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
