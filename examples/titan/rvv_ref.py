#!/usr/bin/env python3
"""The trust root: an RVV-1.0 equivalent of the IME round-one instructions.

Stage 1 of the Titan loop has no Spike model to judge against -- upstream
riscv-isa-sim does not know Zvvm, and even if it did, cospike's comparison
surface is the PC and the integer register writeback, which cannot see a
vector register.  So S1's judge is this file: for every directed test the
loop emits a *pair* of programs from the same random data, one taking the IME
path and one taking a plain RVV 1.0 path, and compares tile checksums.

Two properties make that a real judge rather than a rubber stamp:

  * it is written by a human, before the implementation exists, and
  * the agent is forbidden from editing it (see prompts/system.md).

If this file is wrong, everything downstream is wrong and nothing will say so.
Hence the self-test at the bottom, which derives every geometry rule twice by
independent routes and asserts the two agree.  Run it before trusting a
single directed result:

    python rvv_ref.py

Scope: round one is the integer non-widening subset -- vmmacc.vv, vmtl.v,
vmts.v -- at LMUL in {1,2,4,8}.  Register-side tile layouts follow the
spec's executable Sail appendix (``tile_reg_idx`` / ``mat_A_idx`` /
``mat_B_idx`` / ``mat_C_idx``), which is normative.

Spec: Zvvm v0.9.0 (ARC review candidate, tag 5a2d0f65).  Section names in
comments refer to that text.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

# NOT named `encodings` -- CPython imports the stdlib `encodings` package
# during interpreter startup, so it is already in sys.modules before any
# user code runs and a local encodings.py would be silently ignored.  The
# flat-module packaging in constants.py puts this directory on sys.path on
# every worker, which makes that collision a production hazard, not a
# curiosity.  The build plan lists this file as `encodings.py`; it is
# `ime_encodings.py` for that reason.
import ime_encodings as ime

Matrix = List[List[int]]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileGeometry:
    """Tile shape derived from (VLEN, SEW, LAMBDA, LMUL, VL).

    Spec "Tile dimensions derived from the vector configuration":

        K_eff = LAMBDA * W * LMUL          (W = 1 for the non-widening subset)
        M     = (VLEN / SEW) / LAMBDA      rows of A, edge of the physical C
        N_max = M
        N     = VL / (LAMBDA * LMUL)       active B rows / active C columns
        EMUL_C = (VLEN / SEW) / LAMBDA**2  registers in the C group

    Note EMUL_C is independent of LMUL: LMUL stretches A and B along K only
    and never changes the C tile.

    **Widening (W > 1).**  `sew` is always the *C accumulator* element width
    -- that is what vtype.SEW means for the whole vmmacc family, widening or
    not.  The A/B inputs are `SEW / W` wide and W of them are packed into
    each SEW-wide storage position.  Sail `decode_gemm_geometry`::

        let EEW_C    : int = 2 ^ get_sew_pow();     // accumulator = vtype.SEW
        let EEW_A    : int = EEW_C / W;
        let epr_A    : int = vlen / EEW_A;
        let epr_C    : int = vlen / EEW_C;
        let K_eff    : int = lambda * W * LMUL;
        let M        : int = (LMUL * vlen / EEW_A) / K_eff;
        let N_max    : int = M;
        let EMUL_C   : int = M / lambda;

    Two consequences worth stating, because both are easy to get wrong:

      * `M`, `N_max`, `EMUL_C`, `N` and the permissible LAMBDA set are
        **unchanged** by W.  `M = (LMUL*VLEN/EEW_A)/K_eff` collapses to
        `VLEN/(SEW*LAMBDA)` once EEW_A = SEW/W and K_eff = LAMBDA*W*LMUL are
        substituted, so the C side of the geometry only ever sees SEW.
        In particular `mat_C_idx` keeps using `epr_C = VLEN/SEW` and EMUL_C
        is *not* multiplied by W.
      * Only the A/B side moves: `K_eff` is W times longer, `mat_A_idx` /
        `mat_B_idx` take `row_elems_per_reg = lambda * W` rather than
        `lambda`, and they index at `epr_A = W * epr_C`.

    `w` defaults to 1, so every round-one construction is unchanged.
    """

    vlen: int
    sew: int          # EEW_C: the *C accumulator* element width (= vtype.SEW)
    lam: int          # LAMBDA -- `lambda` is a Python keyword
    lmul: int
    vl: int
    w: int = 1        # widening factor: 1/2/4/8 = vmmacc/vw/vq/v8wmmacc.vv
    tload: str = "op"  # "op" = vmtl.v/vmts.v, "t" = vmttl.v/vmtts.v

    @property
    def elems_per_reg(self) -> int:
        """L = VLEN/SEW, the number of SEW-wide elements in one register.

        This is Sail's `epr_C`: the C accumulator's elements per register,
        and equally the *storage* element count the tile load/stores see
        (they move SEW-wide storage elements and never look inside a packed
        widening element).  :attr:`epr_ab` is the A/B counterpart.
        """
        return self.vlen // self.sew

    @property
    def eew_ab(self) -> int:
        """EEW_A = SEW / W: the logical A/B input element width."""
        return self.sew // self.w

    @property
    def epr_ab(self) -> int:
        """Sail `epr_A` = VLEN / EEW_A, i.e. W * elems_per_reg."""
        return self.vlen // self.eew_ab

    @property
    def ab_row_elems_per_reg(self) -> int:
        """`lambda * W`, the second argument mat_A_idx/mat_B_idx pass on."""
        return self.lam * self.w

    @property
    def ab_linesize(self) -> int:
        """Contiguous run of *logical* A/B elements per tile-memory line.

        LAMBDA * W * LMUL = K_eff.  At W = 1 this is exactly
        :attr:`linesize`, which is why round one never needed the
        distinction: there, one storage element is one logical element.
        """
        return self.lam * self.w * self.lmul

    @property
    def mnemonic(self) -> str:
        """The multiply-accumulate this geometry's W selects."""
        try:
            return {1: "vmmacc.vv", 2: "vwmmacc.vv",
                4: "vqmmacc.vv", 8: "v8wmmacc.vv"}[self.w]
        except KeyError:
            raise ValueError(f"no mnemonic modelled for W={self.w}") from None

    @property
    def load_mnemonic(self) -> str:
        """The tile load this geometry's :attr:`tload` selects."""
        return {"op": "vmtl.v", "t": "vmttl.v"}[self.tload]

    @property
    def store_mnemonic(self) -> str:
        """The tile store this geometry's :attr:`tload` selects."""
        return {"op": "vmts.v", "t": "vmtts.v"}[self.tload]

    @property
    def m(self) -> int:
        return self.elems_per_reg // self.lam

    @property
    def n_max(self) -> int:
        return self.m

    @property
    def k_eff(self) -> int:
        """LAMBDA * W * LMUL -- logical K elements, not storage elements."""
        return self.lam * self.w * self.lmul

    @property
    def n(self) -> int:
        return self.vl // (self.lam * self.lmul)

    @property
    def emul_c(self) -> int:
        return self.elems_per_reg // (self.lam * self.lam)

    @property
    def linesize(self) -> int:
        """Contiguous run length of an order-preserving tile load/store."""
        return self.lam * self.lmul

    @property
    def vl_c_full(self) -> int:
        """VL for loading/storing the *physical* C tile.

        Explicitly not the compute VL.  The spec calls this out because the
        active C columns of a partial block are not a contiguous 1D segment
        of the physical tile, so a C transfer always moves M x M elements and
        masks columns instead.
        """
        return self.m * self.n_max

    @property
    def lmul_c(self) -> int:
        """LMUL to configure for a C tile transfer -- EMUL_C, not LMUL."""
        return self.emul_c

    def validate(self) -> None:
        """Raise ValueError if this configuration is not IME-legal."""
        if self.sew not in (8, 16, 32, 64):
            raise ValueError(f"SEW={self.sew} is not a legal element width")
        if self.w not in (1, 2, 4, 8):
            raise ValueError(
                f"W={self.w}: the Zvvmm family defines W in {{1,2,4,8}} "
                f"(vmmacc.vv, vwmmacc.vv, vqmmacc.vv, v8wmmacc.vv)")
        if self.tload not in ("op", "t"):
            raise ValueError(
                f"tload={self.tload!r}: expected 'op' (vmtl.v / vmts.v) or "
                f"'t' (vmttl.v / vmtts.v)")
        if self.tload == "t" and self.w != 1:
            # Zvvmttls: "The transposing instructions vmttl.v and vmtts.v
            # transpose SEW-bit storage elements.  They do not unpack,
            # transpose, or repack logical elements contained within an
            # SEW-bit storage element ... a transposing tile instruction
            # performs a direct logical matrix transpose only when the
            # logical element width equals SEW."  W>1 packs W logical
            # elements per storage element, so a transposing load of a
            # widening tile is not a logical transpose and the paired-program
            # judge would have to model a half-transposed matrix.  Out of
            # scope: round three transposes only at W=1.
            raise ValueError(
                f"tload='t' with W={self.w}: a transposing tile transfer is "
                f"a logical matrix transpose only when the logical element "
                f"width equals SEW (W=1)")
        if self.w != 1:
            # vqmmacc.vv Exceptions: "SEW = 8 (EEW = SEW / 4 = 2 is
            # reserved)"; the same rule generalises to every W as
            # EEW_A = SEW/W >= 4.  EEW=4 (Int4) is packed two per byte,
            # which no part of this harness emits, so the modelled set is
            # EEW_A in {8,16,32}: W=2 at SEW>=16, W=4 at SEW>=32,
            # W=8 at SEW=64.
            if self.eew_ab < 4:
                raise ValueError(
                    f"W={self.w} with SEW={self.sew} is reserved "
                    f"(EEW = SEW/{self.w} < 4)")
            if self.eew_ab < 8:
                raise ValueError(
                    f"W={self.w} with SEW={self.sew} gives EEW_A="
                    f"{self.eew_ab}; sub-byte (Int4) input tiles are not "
                    f"modelled -- they pack two logical elements per byte")
        if self.lmul not in (1, 2, 4, 8):
            raise ValueError(
                f"LMUL={self.lmul}: Zvvm supports only integer LMUL in "
                f"{{1,2,4,8}}; fractional LMUL raises illegal-instruction")
        if self.lam not in ime.LAMBDA_ENCODING or self.lam == 0:
            raise ValueError(f"LAMBDA={self.lam} is not encodable")
        if self.lam not in permissible_lambdas(self.vlen, self.sew):
            raise ValueError(
                f"LAMBDA={self.lam} is not architecturally permissible for "
                f"VLEN={self.vlen}, SEW={self.sew}; permissible: "
                f"{sorted(permissible_lambdas(self.vlen, self.sew))}")
        if self.vl % (self.lam * self.lmul):
            raise ValueError(
                f"VL={self.vl} must be a multiple of LAMBDA*LMUL="
                f"{self.lam * self.lmul}")
        if not 0 <= self.n <= self.n_max:
            raise ValueError(f"N={self.n} outside [0, {self.n_max}]")
        if self.lmul > self.m:
            raise ValueError(
                f"K_eff={self.k_eff} exceeds one m1 register group "
                f"({self.elems_per_reg} elements); the RVV reference path "
                f"loads each K row with a single m1 vle, so LMUL<=M")

    def describe(self) -> str:
        """One-line geometry summary; embedded in every TITAN verdict line.

        The W = 1 form is byte-for-byte what round one printed -- the
        `W=<n>` clause only appears for a widening geometry, so a tree that
        passed round one keeps producing identical verdict strings.

        The clause goes *after* LAMBDA, not between SEW and LAMBDA where it
        reads more naturally, because helpers._GEOM_RE scrapes the verdict
        line with ``VLEN=(\d+) SEW=(\d+) LAMBDA=(\d+)`` -- three adjacent
        fields.  Splitting that run would silently stop the classifier from
        recovering the geometry of a widening skip or failure, which is
        exactly the information a lambda-clamp diagnosis needs.
        """
        widen = "" if self.w == 1 else f" W={self.w} EEW_AB={self.eew_ab}"
        # The transposing clause goes at the very end, after EMUL_C, for the
        # same reason the W clause goes after LAMBDA: helpers._GEOM_RE scrapes
        # `VLEN=(\d+) SEW=(\d+) LAMBDA=(\d+)` as three adjacent fields, and
        # an "op" geometry must print byte-for-byte what round two printed.
        trans = "" if self.tload == "op" else f" TL={self.tload}"
        return (f"VLEN={self.vlen} SEW={self.sew} LAMBDA={self.lam}{widen} "
                f"LMUL={self.lmul} VL={self.vl} -> M={self.m} N={self.n} "
                f"K_eff={self.k_eff} EMUL_C={self.emul_c}{trans}")


def permissible_lambdas(vlen: int, sew: int, w: int = 1) -> List[int]:
    """Architecturally permissible nonzero LAMBDA for (VLEN, SEW).

    *w* is accepted and deliberately ignored.  Sail `decode_gemm_geometry`
    gates lambda on the **C** element width::

        if not (is_ime_lambda_supported(EEW_C, lambda)) then
          return Illegal_Instruction();

    with `EEW_C = 2 ^ get_sew_pow()` -- vtype.SEW, i.e. the accumulator
    width -- not on `EEW_A = EEW_C / W`.  The legality statement behind it,
    `EMUL_C = VLEN / (SEW * LAMBDA^2) in {1,2,4,8,16}`, is likewise purely
    C-side.  So a widening instruction has exactly the same permissible
    LAMBDA set as the non-widening one at the same vtype.SEW, and the
    parameter exists only so a caller can say which W it meant.

    Derivation A -- the constructive rule from the spec's prose: with
    L = VLEN/SEW, a perfect square L admits {sqrt(L), sqrt(L)/2, sqrt(L)/4}
    (giving EMUL_C in {1,4,16}) and an L that is twice a perfect square
    admits {sqrt(L/2), sqrt(L/2)/2} (giving EMUL_C in {2,8}).

    :func:`_permissible_lambdas_by_emul` derives the same set from the
    independent statement that EMUL_C = L/LAMBDA**2 must be an integer in
    {1,2,4,8,16}.  The self-test asserts the two agree for every (VLEN, SEW);
    a mismatch means one of the two readings is wrong.
    """
    if vlen < sew:
        return []  # outside the IME-legal domain: no nonzero lambda exists
    L = vlen // sew
    if L & (L - 1):
        raise ValueError(f"VLEN/SEW={L} must be a power of two")
    n = L.bit_length() - 1
    if n % 2 == 0:
        root, offsets = 1 << (n // 2), (0, 1, 2)
    else:
        root, offsets = 1 << ((n - 1) // 2), (0, 1)
    return sorted(root >> d for d in offsets if root >> d >= 1)


def _permissible_lambdas_by_emul(vlen: int, sew: int) -> List[int]:
    """Derivation B: brute force over the EMUL_C legality statement."""
    if vlen < sew:
        return []
    L = vlen // sew
    out = []
    lam = 1
    while lam * lam <= L:
        if L % (lam * lam) == 0 and L // (lam * lam) in (1, 2, 4, 8, 16):
            out.append(lam)
        lam *= 2
    return sorted(out)


def ime_legal_configs(vlen: int, sews: Sequence[int] = (8, 16, 32, 64),
                      lmuls: Sequence[int] = (1, 2, 4, 8),
                      full_vl_only: bool = False,
                      ws: Sequence[int] = (1,),
                      tloads: Sequence[str] = ("op",)
                      ) -> Iterator[TileGeometry]:
    """Every IME-legal (SEW, LAMBDA, LMUL, VL) for this VLEN.

    ime_stress.py walks this to build its random mix; the directed suite
    walks it to build the exhaustive S1 gate.  Configurations rejected by
    :meth:`TileGeometry.validate` are skipped, so the caller never has to
    know the legality rules.

    ``ws`` selects the widening factors to enumerate.  It defaults to
    ``(1,)`` -- the round-one set -- so every existing caller sees exactly
    the geometries, in exactly the order, that it saw before.  W is the
    outer-most loop so that appending ``4`` extends the sequence rather than
    interleaving into it.

    ``tloads`` selects the tile-transfer variant -- ``"op"`` for the
    order-preserving vmtl.v / vmts.v pair, ``"t"`` for the transposing
    vmttl.v / vmtts.v pair of round three -- and is outside ``ws`` for the
    same reason.  It defaults to ``("op",)``.
    """
    for tload in tloads:
        for w in ws:
            for sew in sews:
                for lam in permissible_lambdas(vlen, sew, w):
                    for lmul in lmuls:
                        probe = TileGeometry(vlen, sew, lam, lmul,
                                             lam * lmul, w, tload)
                        n_values = ([probe.n_max] if full_vl_only
                                    else range(1, probe.n_max + 1))
                        for n in n_values:
                            geom = TileGeometry(vlen, sew, lam, lmul,
                                                n * lam * lmul, w, tload)
                            try:
                                geom.validate()
                            except ValueError:
                                continue
                            yield geom


# ---------------------------------------------------------------------------
# register-group layout
# ---------------------------------------------------------------------------

def tile_reg_idx(i: int, group_regs: int, row_elems_per_reg: int,
                 elems_per_reg: int) -> int:
    """The spec's Sail ``tile_reg_idx``, transcribed verbatim.

    Appendix (executable Sail), "Map sequential tile element index i to the
    flat register-group element index"::

        function tile_reg_idx(i : int, group_regs : int,
                              row_elems_per_reg : int,
                              elems_per_reg : int) -> int = {
          let linesize     : int = row_elems_per_reg * group_regs;
          let line         : int = i / linesize;
          let elem_in_line : int = i % linesize;
          let regoff       : int = elem_in_line / row_elems_per_reg;
          let elementoff   : int =
              line * row_elems_per_reg + elem_in_line % row_elems_per_reg;
          regoff * elems_per_reg + elementoff
        }

    This is the single source of truth for every register-side index in this
    file.  The Sail appendix is normative: where the prose and the Sail can
    be read differently, the Sail wins.
    """
    linesize = row_elems_per_reg * group_regs
    line, elem_in_line = divmod(i, linesize)
    regoff, rem = divmod(elem_in_line, row_elems_per_reg)
    elementoff = line * row_elems_per_reg + rem
    return regoff * elems_per_reg + elementoff


def ab_sequential_index(r: int, k: int, geom: TileGeometry) -> int:
    """The sequential tile element index of A_tile[r,k] (or B_tile[r,k]).

    Sail::

        function mat_A_idx(i : int, k : int, K_eff : int, ...) -> int =
          tile_reg_idx(i * K_eff + k, LMUL, lambda * W, epr)
        function mat_B_idx(k : int, j : int, K_eff : int, ...) -> int =
          tile_reg_idx(j * K_eff + k, LMUL, lambda * W, epr)

    i.e. both operands present the same ``row * K_eff + col`` sequential
    index to :func:`tile_reg_idx`; B differs only in that its "row" is the
    output column j (B_tile is N x K_eff, its transposed view B^T is
    K_eff x N).  So callers that hold B as an N x K_eff matrix use this
    function unchanged with ``r = j``.
    """
    return r * geom.k_eff + k


def ab_element_index(r: int, k: int, geom: TileGeometry) -> int:
    """Flat register-group element index of A_tile[r,k] (or B_tile[r,k]).

    ``mat_A_idx(r, k)`` / ``mat_B_idx(k, r)`` of the Sail appendix::

        tile_reg_idx(i * K_eff + k, LMUL, lambda * W, epr_A)

    The index is in units of *logical* EEW_A elements, and the group is
    walked at ``epr_A = VLEN / EEW_A``.  At W = 1, ``lambda * W`` is
    ``lambda`` and ``epr_A`` is ``elems_per_reg``, so this is bit-identical
    to the round-one expression.
    """
    return tile_reg_idx(ab_sequential_index(r, k, geom),
                        geom.lmul, geom.ab_row_elems_per_reg, geom.epr_ab)


def c_sequential_index(i: int, j: int, geom: TileGeometry) -> int:
    """Sequential tile element index of C_tile[i,j].

    The physical C tile is M x N_max; the physical row stride is N_max even
    when only N columns are active.  That is what makes the inactive columns
    "C tile tail elements" rather than ordinary vector tail elements.
    """
    return i * geom.n_max + j


def c_element_index(i: int, j: int, geom: TileGeometry) -> int:
    """Flat register-group element index of C_tile[i,j].

    Sail::

        function mat_C_idx(i : int, j : int, N_max : int,
                           EMUL_C : int, lambda : int, epr : int) -> int =
          tile_reg_idx(i * N_max + j, EMUL_C, lambda, epr)

    Note the group multiplier is EMUL_C, not LMUL: the C group's size comes
    from the tile geometry and is independent of vtype.LMUL.  At EMUL_C = 1
    this collapses to the plain row-major ``i * N_max + j``; at EMUL_C > 1
    it does not.
    """
    return tile_reg_idx(c_sequential_index(i, j, geom),
                        geom.emul_c, geom.lam, geom.elems_per_reg)


def tile_transfer_offset(idx: int, ld: int, geom: TileGeometry,
                         linesize: int = None) -> int:
    """Memory element offset vmtl.v / vmts.v touch for *sequential* index i.

    Sail ``vmtl.v`` / ``vmts.v`` body::

        let flat_idx : int =
          tile_reg_idx(i, LMUL, eff_lambda, elems_per_reg);
        let mem_off : int =
          (i / linesize) * LD + (i % linesize);

    The loop variable ``i`` is the *sequential* index.  Memory is addressed
    with ``i`` directly and only the register side goes through
    ``tile_reg_idx``; the two must not be composed.

    *linesize* defaults to :attr:`TileGeometry.linesize`, the storage-element
    line the load/store instruction itself sees.  A caller working in
    *logical* A/B elements of a widening tile passes
    :attr:`TileGeometry.ab_linesize` instead; the two are equal at W = 1.
    """
    if linesize is None:
        linesize = geom.linesize
    return (idx // linesize) * ld + (idx % linesize)


def tile_transfer_offset_t(idx: int, ld: int, geom: TileGeometry,
                           linesize: int = None) -> int:
    """Memory element offset vmttl.v / vmtts.v touch for sequential index i.

    Sail ``vmttl.v`` body (spec line 6789) and ``vmtts.v`` body (line 6926);
    the order-preserving counterparts are at lines 6493 and 6623::

        let flat_idx : int =
          tile_reg_idx(i, LMUL, eff_lambda, elems_per_reg);

        let mem_off : int =
          (i % linesize) * LD + (i / linesize);

    The *register* side is character-for-character the order-preserving
    ``vmtl.v`` / ``vmts.v`` expression -- the transposing pair differs from
    the order-preserving pair in the memory address expression and in
    nothing else.  ``(i / linesize)`` and ``(i % linesize)`` simply swap
    roles, which is what makes this a transpose of SEW-wide *storage*
    elements.

    That is also why this is worth stating twice: the whole of Zvvmttls,
    in the reference model, is this one line.
    """
    if linesize is None:
        linesize = geom.linesize
    return (idx % linesize) * ld + (idx // linesize)


def transposing_default_ld(geom: TileGeometry) -> int:
    """LD a transposing tile transfer uses when rs2 = x0.

    Spec, "Tile geometry for loads and stores" (line 2640)::

        * For the transposing load/store (`vmttl.v`, `vmtts.v`),
          LD = VLEN / (SEW x LAMBDA) (= `M`, the tile edge)

    and the vmttl.v Sail says the same thing as ``LD = elems_per_reg /
    eff_lambda``.  The order-preserving default is LAMBDA x LMUL
    (:attr:`TileGeometry.linesize`) instead; the two coincide only when
    VLEN/SEW = LAMBDA**2 * LMUL.
    """
    return geom.elems_per_reg // geom.lam


def ab_memory_offset_t(r: int, k: int, ld: int, geom: TileGeometry) -> int:
    """Where A_tile[r,k] must live for a *transposing* vmttl.v to load it.

    Composing :func:`ab_sequential_index` with
    :func:`tile_transfer_offset_t`: with i = r * K_eff + k and
    linesize = K_eff (W = 1), ``i % linesize = k`` and ``i / linesize = r``,
    so

        offset = k * LD + r

    -- the column-major image of the same M x K_eff panel, with LD the
    column stride.  With LD = M this is the exact transpose of the
    row-major buffer :func:`ab_memory_offset` produces, which is the point:
    the register-group contents after vmttl.v from the column-major image
    are identical to those after vmtl.v from the row-major image.

    B is the same formula at r = j.  The natural LD for both is M, because
    both A (M rows) and B (N_max = M rows) present M storage elements per
    column -- and M is also the rs2 = x0 default
    (:func:`transposing_default_ld`).
    """
    return tile_transfer_offset_t(ab_sequential_index(r, k, geom), ld, geom,
                                  linesize=geom.ab_linesize)


def ab_memory_offset(r: int, k: int, ld: int, geom: TileGeometry) -> int:
    """Where A_tile[r,k] must live in memory for vmtl.v to load it.

    Composing :func:`ab_sequential_index` with :func:`tile_transfer_offset`:
    with i = r * K_eff + k and linesize = LAMBDA * LMUL = K_eff (W = 1),
    ``i / linesize = r`` and ``i % linesize = k``, so

        offset = r * LD + k

    -- an ordinary row-major panel, at *every* LMUL.  The LMUL>1 packing is
    entirely on the register side (``tile_reg_idx``); it never reaches
    memory.  B is the same formula with r = j, which is the N x K_eff
    row-major view of B_tile, equivalently the column-major K_eff x N view
    of B^T.

    For W > 1 the same formula holds in *logical* element units: one
    SEW-wide storage element carries W consecutive logical K values in
    increasing-k order (element index ``W*s + k%W`` within the register,
    which is little-endian byte order in memory), so a logical A element at
    (r, k) sits at ``r * LD + k`` narrow elements from the base with LD in
    narrow elements.  The tile load itself still runs at the storage width
    with ``LD_storage = LD / W``, which addresses the identical bytes.
    """
    return tile_transfer_offset(ab_sequential_index(r, k, geom), ld, geom,
                                linesize=geom.ab_linesize)


def tile_layout_buffer(mat: Matrix, ld: int, geom: TileGeometry) -> List[int]:
    """Flatten *mat* into the element order a tile load reads from memory.

    Per the Sail this is plain row-major with stride *ld* at every LMUL.
    The function is kept (rather than inlined at the call sites) because it
    is the one place the memory layout is stated, and because callers pass
    both A (M x K_eff) and B (N_max x K_eff) through it.
    """
    rows = len(mat)
    size = max(ab_memory_offset(r, k, ld, geom)
               for r in range(rows) for k in range(geom.k_eff)) + 1
    buf = [0] * size
    for r in range(rows):
        for k in range(geom.k_eff):
            buf[ab_memory_offset(r, k, ld, geom)] = mat[r][k]
    return buf


def tile_layout_buffer_t(mat: Matrix, ld: int, geom: TileGeometry
                         ) -> List[int]:
    """Flatten *mat* into the element order a *transposing* tile load reads.

    The column-major counterpart of :func:`tile_layout_buffer`.  Same
    matrix, same register-group outcome, transposed memory image.
    """
    rows = len(mat)
    size = max(ab_memory_offset_t(r, k, ld, geom)
               for r in range(rows) for k in range(geom.k_eff)) + 1
    buf = [0] * size
    for r in range(rows):
        for k in range(geom.k_eff):
            buf[ab_memory_offset_t(r, k, ld, geom)] = mat[r][k]
    return buf


def c_memory_offset_t(i: int, j: int, ld: int, geom: TileGeometry) -> int:
    """Where C_tile[i,j] lives for a transposing C-tile vmttl.v / vmtts.v.

    The C transfer runs at LMUL = EMUL_C and VL = M * N_max, so
    linesize = LAMBDA * EMUL_C = M and the sequential index
    s = i * N_max + j gives ``s % linesize = j``, ``s / linesize = i``:

        offset = j * LD + i

    -- column-major C.  With LD = M (also the rs2 = x0 default) this is the
    transpose of what :func:`c_memory_offset` produces.
    """
    c_geom = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul_c,
                          geom.vl_c_full)
    return tile_transfer_offset_t(c_sequential_index(i, j, geom), ld, c_geom)


def c_memory_offset(i: int, j: int, ld: int, geom: TileGeometry) -> int:
    """Where C_tile[i,j] must live in memory for a C-tile vmtl.v / vmts.v.

    A C transfer is configured with LMUL=EMUL_C and VL=M*N_max, which makes
    linesize = LAMBDA*EMUL_C = M: one line per physical C row.  So with
    LD=M the tile is an ordinary row-major M x M block.
    """
    c_geom = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul_c,
                          geom.vl_c_full)
    return tile_transfer_offset(c_sequential_index(i, j, geom), ld, c_geom)


# ---------------------------------------------------------------------------
# golden model
# ---------------------------------------------------------------------------

def _wrap(value: int, sew: int) -> int:
    """Reduce to SEW bits, interpreted as a signed two's-complement value."""
    value &= (1 << sew) - 1
    if value >> (sew - 1):
        value -= 1 << sew
    return value


def reference_gemm(a: Matrix, b: Matrix, c: Matrix,
                   geom: TileGeometry) -> Matrix:
    """C[i,j] += sum_k A[i,k] * B[j,k], wrapping modulo 2**SEW.

    B is the in-register B_tile (N x K_eff), so the transpose is implicit:
    B_tile_T[k,j] = B_tile[j,k].  Only the N active columns are touched; the
    tail columns keep their previous value, which is the vta=0 behaviour the
    tests configure (vta=1 would let the hardware smear all-ones into them).

    Signedness note: altfmt_A / altfmt_B select signed or unsigned inputs,
    but for the non-widening W=1 case they cannot change the result.  The
    low SEW bits of a product are the same whether the operands are read as
    signed or unsigned, and accumulation wraps at 2**SEW.  Round one
    therefore tests that both settings are *accepted*, not that they produce
    different numbers -- expecting a difference would be a test bug.

    That immunity dies at W > 1, and this function is where it dies.  For
    vqmmacc.vv the product of two EEW_A-wide inputs is accumulated at
    SEW = 4*EEW_A, so the sign extension happens *below* the wrap point and
    is visible in the result.  *a* and *b* therefore arrive as signed
    EEW_A-wide Python ints (see :func:`random_case`) and this function reads
    them as-is; the caller is responsible for having sign-extended.  Sail
    `int_block_dot`::

        let a_val : int = if signed_A
                          then signed(read_single_element(g.EEW_A, a_flat, vs1))
                          else unsigned(read_single_element(g.EEW_A, a_flat, vs1));
        ...
        int_sum = int_sum + a_val * b_val

    and `int_gemm` wraps the total once, at EEW_C::

        var acc : int = signed(c_bits);
        acc = acc + int_block_dot(i, j, 0, g.K_eff - 1, ...);
        write_single_element(g.EEW_C, c_flat, vd, to_bits_unsafe(g.EEW_C, acc))

    -- a single modulo-2**SEW reduction of the whole exact sum, which is
    what ``_wrap(acc, geom.sew)`` below does.  Only the plain signed
    vqmmacc.vv (altfmt_A = altfmt_B = 0) is generated here.
    """
    out = [row[:] for row in c]
    for i in range(geom.m):
        for j in range(geom.n):
            acc = out[i][j]
            for k in range(geom.k_eff):
                acc += a[i][k] * b[j][k]
            out[i][j] = _wrap(acc, geom.sew)
    return out


def _reference_gemm_transposed(a: Matrix, b: Matrix, c: Matrix,
                               geom: TileGeometry) -> Matrix:
    """Same result by an independent route: build B_T first, then row-by-col.

    Deliberately not a refactor of :func:`reference_gemm` -- it exists so the
    self-test can disagree with itself.
    """
    bt = [[b[j][k] for j in range(geom.n)] for k in range(geom.k_eff)]
    out = [row[:] for row in c]
    for i in range(geom.m):
        row = a[i]
        for j in range(geom.n):
            column = [bt[k][j] for k in range(geom.k_eff)]
            out[i][j] = _wrap(out[i][j] + sum(x * y for x, y in
                                              zip(row, column)), geom.sew)
    return out


# ---------------------------------------------------------------------------
# random test data
# ---------------------------------------------------------------------------

def random_matrix(rows: int, cols: int, sew: int,
                  rng: random.Random) -> Matrix:
    """Signed SEW-bit values, biased toward the interesting magnitudes.

    A uniform draw over the full range almost never produces a carry chain
    that distinguishes a correct accumulator from a truncating one, so mix in
    extremes and small values.
    """
    lo, hi = -(1 << (sew - 1)), (1 << (sew - 1)) - 1
    pool = (lo, hi, -1, 0, 1, lo // 2, hi // 2)
    out = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(rng.choice(pool) if rng.random() < 0.25
                       else rng.randint(lo, hi))
        out.append(row)
    return out


def random_case(geom: TileGeometry, rng: random.Random
                ) -> Tuple[Matrix, Matrix, Matrix]:
    """(A, B, C) for one directed test, shaped by *geom*.

    B is generated with N_max rows, not N.  A tile load runs at full VL and
    therefore reads M rows' worth of memory whatever N is; sizing the buffer
    to N would walk off the end of the allocation for every partial-N test.
    Rows at or beyond N are read into inactive B rows and never reach the
    arithmetic -- :func:`reference_gemm` ignores them too.

    A and B are drawn at EEW_A = SEW/W (the logical input width) and C at
    SEW (the accumulator width).  At W = 1 the two coincide, so round one's
    data is unchanged.
    """
    return (random_matrix(geom.m, geom.k_eff, geom.eew_ab, rng),
            random_matrix(geom.n_max, geom.k_eff, geom.eew_ab, rng),
            random_matrix(geom.m, geom.n_max, geom.sew, rng))


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------

def tile_checksums(c: Matrix, geom: TileGeometry) -> List[int]:
    """One checksum per physical C row.

    Per-row rather than one number for the whole tile: a single matrix-wide
    checksum tells you "wrong" and nothing else, and the first thing anyone
    asks of a failing matrix unit is which rows moved.  Rows are the finest
    granularity that survives the trip through the scalar domain cheaply --
    a load-and-xor per element, folded per row.

    Tail columns (j >= N) are excluded: with vta=1 the hardware is allowed to
    smear all-ones through them, so including them would make the checksum
    nondeterministic on a *correct* implementation.
    """
    sums = []
    mask = (1 << geom.sew) - 1
    for i in range(geom.m):
        acc = 0
        for j in range(geom.n):
            acc ^= (c[i][j] & mask) + i * 0x9E37 + j  # position-sensitive
            acc &= mask
        sums.append(acc)
    return sums


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

VLENS = (64, 128, 256, 512, 1024)

#: C accumulator widths the W=4 (vqmmacc.vv) tier is generated for.
#: SEW=32 is Zvvi8i32mm -- Int8 x Int8 -> Int32, which is what llama.cpp's
#: INT8 GEMM needs.  SEW=64 (Zvvi16i64mm, Int16 -> Int64) works through the
#: same machinery and is enumerable by passing it explicitly.  SEW=16 is
#: Zvvi4i16mm, whose Int4 inputs pack two per byte; no part of this harness
#: emits that, and TileGeometry.validate rejects it.  SEW=8 is reserved by
#: the spec (EEW = 2).
WIDENING_SEWS = (32,)

#: Round three's widening tiers: C accumulator widths per widening factor.
#: The rule is EEW_A = SEW/W >= 8 -- EEW_A = 4 (Int4) is legal architecture
#: but packs two logical elements per byte, which nothing here emits, and
#: EEW_A = 2 is reserved by the spec.  So:
#:   W=2 -> SEW 16 (Zvvi8i16mm, Int8 -> Int16) and 32 (Int16 -> Int32)
#:   W=4 -> SEW 32 (Zvvi8i32mm, Int8 -> Int32) -- round two, unchanged
#:   W=8 -> SEW 64 (Zvvi8i64mm, Int8 -> Int64); SEW=32 would need Int4
#: SEW=64 at W=2/W=4 works through the same machinery and is enumerable by
#: passing it explicitly; it is left out of the default tiers because it
#: adds programs without adding a distinct EEW_A the suite has not seen.
WIDENING_SEWS_BY_W = {2: (16, 32), 4: WIDENING_SEWS, 8: (64,)}


def check_lambda_derivations() -> None:
    """The two independent readings of the LAMBDA rules must agree."""
    for vlen in (32, 64, 128, 256, 512, 1024, 2048):
        for sew in (8, 16, 32, 64):
            a = permissible_lambdas(vlen, sew)
            b = _permissible_lambdas_by_emul(vlen, sew)
            assert a == b, (vlen, sew, a, b)
            for lam in a:
                geom = TileGeometry(vlen, sew, lam, 1, lam)
                assert geom.emul_c in (1, 2, 4, 8, 16), (vlen, sew, lam)
                assert lam * lam <= geom.elems_per_reg
                assert geom.m >= lam, "M >= LAMBDA is required"

    # Values stated verbatim in the spec text.
    assert permissible_lambdas(512, 32) == [1, 2, 4]
    assert permissible_lambdas(1024, 32) == [2, 4]
    for l_over, expected in ((1, [1]), (2, [1]), (4, [1, 2]), (8, [1, 2])):
        assert permissible_lambdas(32 * l_over, 32) == expected, l_over


def check_layout_bijections() -> None:
    """Both tile layouts must be permutations, not merely mappings."""
    for vlen in VLENS:
        for geom in ime_legal_configs(vlen, full_vl_only=True):
            seen = {ab_element_index(r, k, geom)
                    for r in range(geom.m) for k in range(geom.k_eff)}
            assert len(seen) == geom.m * geom.k_eff, geom.describe()
            assert max(seen) < geom.lmul * geom.elems_per_reg, geom.describe()

            seen_c = {c_element_index(i, j, geom)
                      for i in range(geom.m) for j in range(geom.n_max)}
            assert len(seen_c) == geom.m * geom.n_max, geom.describe()
            assert max(seen_c) < geom.emul_c * geom.elems_per_reg, \
                geom.describe()
            assert geom.m * geom.n_max == geom.emul_c * geom.elems_per_reg


def check_memory_layout() -> None:
    """vmtl.v composed with the tile layout must yield the claimed matrix."""
    for vlen in VLENS:
        for geom in ime_legal_configs(vlen, full_vl_only=True):
            # Row-major at every LMUL -- the Sail addresses memory with the
            # sequential index, not with tile_reg_idx.
            for ld in (geom.k_eff, geom.k_eff + 3, geom.k_eff * 2):
                for r in range(geom.m):
                    for k in range(geom.k_eff):
                        assert ab_memory_offset(r, k, ld, geom) == r * ld + k, (
                            geom.describe(), ld, r, k)

            # A C transfer with LD = M is a plain row-major M x M block.
            for i in range(geom.m):
                for j in range(geom.n_max):
                    assert c_memory_offset(i, j, geom.m, geom) == i * geom.m + j, (
                        geom.describe(), i, j)


def check_sail_worked_examples() -> None:
    """Hand-derived Sail values, computed away from this file's code.

    Every expectation below is worked out directly from the Sail text
    (tile_reg_idx / mat_A_idx / mat_C_idx and the vmtl.v body), not from any
    formula in this module.  If a future refactor reintroduces the old
    flat-index-composed-with-mem_off reading, these fire.
    """
    # VLEN=256 SEW=8 LAMBDA=4 LMUL=2 -> epr=32, M=8, K_eff=8, linesize=8.
    g = TileGeometry(256, 8, 4, 2, 8 * 2)   # N = VL/(lam*lmul) = 2
    g.validate()
    assert (g.elems_per_reg, g.m, g.k_eff, g.linesize) == (32, 8, 8, 8)

    # Memory: row-major with LD.  A[1,0] -> 1*LD + 0; A[0,4] -> 4.
    assert ab_memory_offset(1, 0, g.linesize, g) == 8
    assert ab_memory_offset(0, 4, g.linesize, g) == 4
    assert ab_memory_offset(3, 5, 11, g) == 3 * 11 + 5

    # Register side: mat_A_idx(1,0) = tile_reg_idx(8, LMUL=2, lam=4, epr=32).
    #   linesize=8, line=1, elem_in_line=0, regoff=0, elementoff=1*4+0=4 -> 4
    assert ab_element_index(1, 0, g) == 4
    #   mat_A_idx(0,4) = tile_reg_idx(4, ...): line=0, elem_in_line=4,
    #   regoff=1, elementoff=0*4+0=0 -> 1*32 + 0 = 32
    assert ab_element_index(0, 4, g) == 32
    #   mat_A_idx(3,5) = tile_reg_idx(29,...): line=3, eil=5, regoff=1,
    #   elementoff=3*4+1=13 -> 32+13 = 45
    assert ab_element_index(3, 5, g) == 45

    # EMUL_C > 1 C indices.  VLEN=256 SEW=8 LAMBDA=2: epr=32, M=N_max=16,
    # EMUL_C = 32/4 = 8.  mat_C_idx(i,j) = tile_reg_idx(i*16+j, 8, 2, 32);
    # linesize = 2*8 = 16, so line=i, elem_in_line=j.
    #   regoff = j/2, elementoff = i*2 + j%2  ->  (j/2)*32 + i*2 + j%2
    gc = TileGeometry(256, 8, 2, 1, 2 * 16)
    gc.validate()
    assert (gc.emul_c, gc.m, gc.n_max) == (8, 16, 16)
    for i, j, want in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (0, 2, 32),
                       (5, 7, 3 * 32 + 5 * 2 + 1), (15, 15, 7 * 32 + 31)):
        assert c_element_index(i, j, gc) == want, (i, j, want,
                                                   c_element_index(i, j, gc))

    # VLEN=512 SEW=32 LAMBDA=2: epr=16, M=N_max=8, EMUL_C=4.
    # mat_C_idx(i,j) = tile_reg_idx(i*8+j, 4, 2, 16); linesize=8.
    gc2 = TileGeometry(512, 32, 2, 1, 2 * 8)
    gc2.validate()
    assert gc2.emul_c == 4 and gc2.m == 8
    for i, j, want in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (0, 2, 16),
                       (3, 6, 3 * 16 + 3 * 2), (7, 7, 3 * 16 + 15)):
        assert c_element_index(i, j, gc2) == want, (i, j, want,
                                                    c_element_index(i, j, gc2))

    # EMUL_C = 1 is the degenerate case: plain row-major.
    g1 = TileGeometry(256, 16, 4, 1, 4 * 4)
    g1.validate()
    assert g1.emul_c == 1 and g1.m == 4
    for i in range(g1.m):
        for j in range(g1.n_max):
            assert c_element_index(i, j, g1) == i * g1.n_max + j

    # The register-side map must never be the identity where the Sail says
    # it permutes, and must be a permutation everywhere.
    assert ab_element_index(1, 0, g) != ab_sequential_index(1, 0, g)
    assert c_element_index(0, 2, gc) != c_sequential_index(0, 2, gc)


def check_widening_geometry() -> None:
    """Hand-derived Sail values for W=4 (vqmmacc.vv), VLEN=256, SEW=32.

    Worked out from `decode_gemm_geometry(4)`, `mat_A_idx`, `mat_B_idx` and
    `mat_C_idx` on paper, not from the code above.  The three geometries
    cover LMUL in {1, 2, 4} and both permissible LAMBDA values.

    The fixed points to keep honest are (a) the C side must not notice W at
    all -- epr_C stays VLEN/SEW and EMUL_C stays VLEN/(SEW*LAMBDA^2) -- and
    (b) the A/B side must use `lambda * W` as row_elems_per_reg and walk the
    group at `epr_A = W * epr_C`.  Getting either backwards is the widening
    version of the LMUL>1 layout bug that cost round one days.
    """
    # ---- Q1: VLEN=256 SEW=32 W=4 LAMBDA=2 LMUL=1 -----------------------
    # EEW_A = 8, epr_C = 8, epr_A = 32, M = N_max = 256/(32*2) = 4,
    # EMUL_C = 256/(32*4) = 2, K_eff = 2*4*1 = 8.
    q1 = TileGeometry(256, 32, 2, 1, 2 * 1 * 4, 4)
    q1.validate()
    assert (q1.eew_ab, q1.epr_ab, q1.elems_per_reg) == (8, 32, 8)
    assert (q1.m, q1.n_max, q1.emul_c, q1.k_eff, q1.n) == (4, 4, 2, 8, 4)
    assert permissible_lambdas(256, 32, 4) == [1, 2]
    # mat_A_idx(i,k) = tile_reg_idx(i*8+k, LMUL=1, lambda*W=8, epr=32):
    #   linesize = 8*1 = 8 -> line=i, elem_in_line=k, regoff = k/8 = 0,
    #   elementoff = i*8 + k  ->  i*8 + k.  (LMUL=1 is the identity.)
    for i, k, want in ((0, 0, 0), (1, 0, 8), (0, 5, 5), (3, 7, 31)):
        assert ab_element_index(i, k, q1) == want, (i, k, want)
    # mat_C_idx(i,j) = tile_reg_idx(i*4+j, EMUL_C=2, lambda=2, epr_C=8):
    #   linesize = 2*2 = 4 -> line=i, eil=j, regoff=j/2,
    #   elementoff = i*2 + j%2  ->  (j/2)*8 + i*2 + j%2.
    for i, j, want in ((0, 0, 0), (0, 1, 1), (1, 0, 2), (0, 2, 8),
                       (3, 3, 8 + 6 + 1)):
        assert c_element_index(i, j, q1) == want, (i, j, want)

    # ---- Q2: same, LMUL=2 ----------------------------------------------
    # K_eff = 2*4*2 = 16; M, N_max and EMUL_C are unchanged by LMUL.
    q2 = TileGeometry(256, 32, 2, 2, 2 * 2 * 4, 4)
    q2.validate()
    assert (q2.m, q2.emul_c, q2.k_eff) == (4, 2, 16)
    # mat_A_idx(i,k) = tile_reg_idx(i*16+k, LMUL=2, lambda*W=8, epr=32):
    #   linesize = 8*2 = 16 -> line=i, eil=k, regoff = k/8,
    #   elementoff = i*8 + k%8  ->  (k/8)*32 + i*8 + k%8.
    for i, k, want in ((0, 0, 0), (0, 8, 32), (1, 0, 8),
                       (2, 9, 32 + 16 + 1), (3, 15, 32 + 24 + 7)):
        assert ab_element_index(i, k, q2) == want, (i, k, want)
    assert max(ab_element_index(i, k, q2)
               for i in range(q2.m) for k in range(q2.k_eff)) \
        < q2.lmul * q2.epr_ab
    # The C map is identical to Q1: LMUL does not reach the C tile.
    for i in range(q2.m):
        for j in range(q2.n_max):
            assert c_element_index(i, j, q2) == c_element_index(i, j, q1)

    # ---- Q3: VLEN=256 SEW=32 W=4 LAMBDA=1 LMUL=4 -----------------------
    # M = N_max = 8, EMUL_C = 256/(32*1) = 8, K_eff = 1*4*4 = 16.
    q3 = TileGeometry(256, 32, 1, 4, 1 * 4 * 8, 4)
    q3.validate()
    assert (q3.m, q3.n_max, q3.emul_c, q3.k_eff, q3.n) == (8, 8, 8, 16, 8)
    # mat_A_idx(i,k) = tile_reg_idx(i*16+k, LMUL=4, lambda*W=4, epr=32):
    #   linesize = 4*4 = 16 -> line=i, eil=k, regoff = k/4,
    #   elementoff = i*4 + k%4  ->  (k/4)*32 + i*4 + k%4.
    for i, k, want in ((0, 0, 0), (0, 4, 32), (1, 0, 4),
                       (5, 6, 32 + 20 + 2), (7, 15, 3 * 32 + 28 + 3)):
        assert ab_element_index(i, k, q3) == want, (i, k, want)
    # mat_C_idx(i,j) = tile_reg_idx(i*8+j, EMUL_C=8, lambda=1, epr_C=8):
    #   linesize = 1*8 = 8 -> line=i, eil=j, regoff=j, elementoff=i
    #   ->  j*8 + i.  A transposed-looking C map is what EMUL_C=M gives.
    for i, j, want in ((0, 0, 0), (0, 1, 8), (1, 0, 1), (7, 7, 63)):
        assert c_element_index(i, j, q3) == want, (i, j, want)

    # ---- shared invariants across every widening geometry --------------
    for vlen in VLENS:
        for geom in ime_legal_configs(vlen, sews=(32, 64), ws=(4,)):
            # M / N_max / EMUL_C must match the W=1 geometry at the same SEW.
            base = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul,
                                geom.vl)
            assert (geom.m, geom.n_max, geom.emul_c, geom.n) == \
                (base.m, base.n_max, base.emul_c, base.n), geom.describe()
            assert geom.k_eff == 4 * base.k_eff, geom.describe()
            assert geom.epr_ab == 4 * geom.elems_per_reg, geom.describe()
            # Both maps must still be permutations, inside their groups.
            seen = {ab_element_index(r, k, geom)
                    for r in range(geom.m) for k in range(geom.k_eff)}
            assert len(seen) == geom.m * geom.k_eff, geom.describe()
            assert max(seen) < geom.lmul * geom.epr_ab, geom.describe()
            seen_c = {c_element_index(i, j, geom)
                      for i in range(geom.m) for j in range(geom.n_max)}
            assert len(seen_c) == geom.m * geom.n_max, geom.describe()
            assert max(seen_c) < geom.emul_c * geom.elems_per_reg, \
                geom.describe()
            # Memory stays row-major in *logical* elements at every LMUL.
            for ld in (geom.k_eff, geom.k_eff + 3):
                for r in (0, geom.m - 1):
                    for k in (0, geom.k_eff - 1):
                        assert ab_memory_offset(r, k, ld, geom) == r * ld + k

            # The load-compatibility theorem, and the reason round two needs
            # no new tile load: vmtl.v moves SEW-wide *storage* elements and
            # decodes its own geometry with W=1, so it places storage
            # element s at tile_reg_idx(s, LMUL, lambda, epr_C).  For that
            # to deliver what mat_A_idx then reads at EEW_A, the logical
            # element (r, k) must land at W*flat_storage + k%W -- i.e. the W
            # logical elements of a storage position must sit in increasing
            # k order, which is little-endian byte order in memory.  If this
            # ever fails, a widening A tile needs a repack and the emitted
            # programs are wrong however green they look.
            storage = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul,
                                   geom.vl)
            k_storage = geom.k_eff // geom.w
            for r in range(geom.m):
                for k in range(geom.k_eff):
                    flat_s = tile_reg_idx(r * k_storage + k // geom.w,
                                          geom.lmul, geom.lam,
                                          storage.elems_per_reg)
                    assert ab_element_index(r, k, geom) == \
                        geom.w * flat_s + k % geom.w, \
                        (geom.describe(), r, k)


def check_round_three_widening() -> None:
    """Hand-derived Sail values for W=2 (vwmmacc.vv) and W=8 (v8wmmacc.vv).

    Same method as :func:`check_widening_geometry`: worked from
    `decode_gemm_geometry(W)`, `mat_A_idx` and `mat_C_idx` on paper.  The
    point of doing it again at W=2 and W=8 rather than trusting the W=4
    derivation is that `lambda * W` and `epr_A = VLEN/(SEW/W)` are the two
    places W enters, and both are linear in W -- an off-by-a-factor there is
    invisible at a single W.
    """
    # ---- W2a: VLEN=256 SEW=32 W=2 LAMBDA=2 LMUL=1 ----------------------
    # EEW_A = 16, epr_C = 8, epr_A = 16, M = N_max = 256/(32*2) = 4,
    # EMUL_C = 256/(32*4) = 2, K_eff = 2*2*1 = 4.
    w2a = TileGeometry(256, 32, 2, 1, 2 * 1 * 4, 2)
    w2a.validate()
    assert (w2a.eew_ab, w2a.epr_ab, w2a.elems_per_reg) == (16, 16, 8)
    assert (w2a.m, w2a.n_max, w2a.emul_c, w2a.k_eff, w2a.n) == (4, 4, 2, 4, 4)
    assert w2a.mnemonic == "vwmmacc.vv"
    # mat_A_idx(i,k) = tile_reg_idx(i*4+k, LMUL=1, lambda*W=4, epr=16):
    #   linesize = 4 -> line=i, eil=k, regoff = k/4 = 0,
    #   elementoff = i*4 + k  ->  i*4 + k.
    for i, k, want in ((0, 0, 0), (1, 0, 4), (0, 3, 3), (3, 3, 15)):
        assert ab_element_index(i, k, w2a) == want, (i, k, want)

    # ---- W2b: same, LMUL=2.  K_eff = 2*2*2 = 8. ------------------------
    w2b = TileGeometry(256, 32, 2, 2, 2 * 2 * 4, 2)
    w2b.validate()
    assert (w2b.m, w2b.emul_c, w2b.k_eff) == (4, 2, 8)
    # mat_A_idx(i,k) = tile_reg_idx(i*8+k, LMUL=2, lambda*W=4, epr=16):
    #   linesize = 4*2 = 8 -> line=i, eil=k, regoff = k/4,
    #   elementoff = i*4 + k%4  ->  (k/4)*16 + i*4 + k%4.
    for i, k, want in ((0, 0, 0), (0, 4, 16), (1, 0, 4),
                       (2, 5, 16 + 8 + 1), (3, 7, 16 + 12 + 3)):
        assert ab_element_index(i, k, w2b) == want, (i, k, want)
    # C is untouched by W and by LMUL.
    for i in range(w2b.m):
        for j in range(w2b.n_max):
            assert c_element_index(i, j, w2b) == c_element_index(i, j, w2a)

    # ---- W8a: VLEN=256 SEW=64 W=8 LAMBDA=1 LMUL=1 ----------------------
    # EEW_A = 8, epr_C = 4, epr_A = 32, M = N_max = 256/(64*1) = 4,
    # EMUL_C = 4/1 = 4, K_eff = 1*8*1 = 8.
    w8a = TileGeometry(256, 64, 1, 1, 1 * 1 * 4, 8)
    w8a.validate()
    assert (w8a.eew_ab, w8a.epr_ab, w8a.elems_per_reg) == (8, 32, 4)
    assert (w8a.m, w8a.n_max, w8a.emul_c, w8a.k_eff, w8a.n) == (4, 4, 4, 8, 4)
    assert w8a.mnemonic == "v8wmmacc.vv"
    # mat_A_idx(i,k) = tile_reg_idx(i*8+k, LMUL=1, lambda*W=8, epr=32):
    #   linesize = 8 -> regoff = 0, elementoff = i*8 + k  ->  i*8 + k.
    for i, k, want in ((0, 0, 0), (1, 0, 8), (3, 7, 31)):
        assert ab_element_index(i, k, w8a) == want, (i, k, want)
    # mat_C_idx(i,j) = tile_reg_idx(i*4+j, EMUL_C=4, lambda=1, epr_C=4):
    #   linesize = 4 -> regoff = j, elementoff = i  ->  j*4 + i.
    for i, j, want in ((0, 0, 0), (0, 1, 4), (1, 0, 1), (3, 3, 15)):
        assert c_element_index(i, j, w8a) == want, (i, j, want)

    # ---- W8b: VLEN=256 SEW=64 W=8 LAMBDA=2 LMUL=1 ----------------------
    # M = 4/2 = 2, EMUL_C = 4/4 = 1, K_eff = 2*8*1 = 16.
    w8b = TileGeometry(256, 64, 2, 1, 2 * 1 * 2, 8)
    w8b.validate()
    assert (w8b.m, w8b.emul_c, w8b.k_eff) == (2, 1, 16)
    # tile_reg_idx(i*16+k, 1, 16, 32): linesize=16 -> i*16 + k.
    for i, k, want in ((0, 0, 0), (1, 0, 16), (1, 15, 31)):
        assert ab_element_index(i, k, w8b) == want, (i, k, want)

    # ---- reserved / unmodelled widths ----------------------------------
    for sew, w in ((8, 2), (16, 4), (16, 8), (32, 8)):
        try:
            TileGeometry(256, sew, 1, 1, 1).validate()
        except ValueError:
            continue  # not even legal at W=1 here; nothing to assert
        for lam in permissible_lambdas(256, sew):
            g = TileGeometry(256, sew, lam, 1, lam * (256 // sew // lam), w)
            try:
                g.validate()
            except ValueError:
                continue
            raise AssertionError(
                f"SEW={sew} W={w} gives EEW_A={g.eew_ab} and must not "
                f"validate")

    # ---- shared invariants, now across W in {1,2,4,8} ------------------
    for vlen in VLENS:
        for w, sews in sorted(WIDENING_SEWS_BY_W.items()):
            for geom in ime_legal_configs(vlen, sews=sews, ws=(w,)):
                base = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul,
                                    geom.vl)
                assert (geom.m, geom.n_max, geom.emul_c, geom.n) == \
                    (base.m, base.n_max, base.emul_c, base.n), geom.describe()
                assert geom.k_eff == w * base.k_eff, geom.describe()
                assert geom.epr_ab == w * geom.elems_per_reg, geom.describe()
                seen = {ab_element_index(r, k, geom)
                        for r in range(geom.m) for k in range(geom.k_eff)}
                assert len(seen) == geom.m * geom.k_eff, geom.describe()
                assert max(seen) < geom.lmul * geom.epr_ab, geom.describe()
                # The packing theorem of check_widening_geometry, at every W.
                storage = TileGeometry(geom.vlen, geom.sew, geom.lam,
                                       geom.lmul, geom.vl)
                k_storage = geom.k_eff // w
                for r in range(geom.m):
                    for k in range(geom.k_eff):
                        flat_s = tile_reg_idx(r * k_storage + k // w,
                                              geom.lmul, geom.lam,
                                              storage.elems_per_reg)
                        assert ab_element_index(r, k, geom) == \
                            w * flat_s + k % w, (geom.describe(), r, k)


def check_transposing_layout() -> None:
    """Hand-derived Sail values for vmttl.v / vmtts.v (Zvvmttls).

    The whole of the transposing pair, in the reference model, is one line
    of the Sail body (spec 6789 / 6926)::

        let mem_off : int = (i % linesize) * LD + (i / linesize);

    against the order-preserving (spec 6493 / 6623)::

        let mem_off : int = (i / linesize) * LD + (i % linesize);

    with an identical `flat_idx = tile_reg_idx(i, LMUL, eff_lambda,
    elems_per_reg)` on both.  So the two properties to pin are: the register
    side is *bit-identical* between the two variants, and the memory side is
    an exact transpose at the natural LD.
    """
    # ---- T1: VLEN=256 SEW=32 LAMBDA=2 LMUL=2, W=1 ----------------------
    # epr = 8, M = N_max = 4, K_eff = linesize = 2*2 = 4, EMUL_C = 2.
    t1 = TileGeometry(256, 32, 2, 2, 2 * 2 * 4, 1, "t")
    t1.validate()
    assert (t1.m, t1.k_eff, t1.linesize, t1.emul_c) == (4, 4, 4, 2)
    assert (t1.load_mnemonic, t1.store_mnemonic) == ("vmttl.v", "vmtts.v")
    # rs2 = x0 default: LD = VLEN/(SEW*LAMBDA) = elems_per_reg/lambda = M.
    assert transposing_default_ld(t1) == 4 == t1.m
    # i = r*K_eff + k, linesize = K_eff -> i%linesize = k, i/linesize = r,
    # so mem_off = k*LD + r: column-major.
    for r, k, want in ((0, 0, 0), (1, 0, 1), (0, 1, 4), (3, 3, 15)):
        assert ab_memory_offset_t(r, k, 4, t1) == want, (r, k, want)
        assert ab_memory_offset(r, k, 4, t1) == r * 4 + k, (r, k)
    # A non-natural LD stretches the column stride, nothing else.
    assert ab_memory_offset_t(2, 3, 9, t1) == 3 * 9 + 2
    # C: linesize = lambda*EMUL_C = M = 4, s = i*N_max + j -> mem = j*LD + i.
    for i, j, want in ((0, 0, 0), (0, 1, 4), (1, 0, 1), (3, 3, 15)):
        assert c_memory_offset_t(i, j, 4, t1) == want, (i, j, want)
        assert c_memory_offset(i, j, 4, t1) == i * 4 + j, (i, j)

    # ---- T2: the degenerate square, VLEN=256 SEW=8 LAMBDA=2 LMUL=8 -----
    # Here linesize = 16 = M, so order-preserving and transposing coincide
    # *in shape* but still differ elementwise except on the diagonal.
    t2 = TileGeometry(256, 8, 2, 8, 2 * 8 * 16, 1, "t")
    t2.validate()
    assert t2.k_eff == 16 and t2.m == 16
    assert transposing_default_ld(t2) == t2.m == 16
    diff = sum(1 for r in range(t2.m) for k in range(t2.k_eff)
               if ab_memory_offset_t(r, k, 16, t2)
               != ab_memory_offset(r, k, 16, t2))
    assert diff == t2.m * t2.k_eff - t2.m, diff  # everything but the diagonal

    # ---- invariants over every legal geometry --------------------------
    for vlen in VLENS:
        for geom in ime_legal_configs(vlen, tloads=("t",)):
            op = TileGeometry(geom.vlen, geom.sew, geom.lam, geom.lmul,
                              geom.vl)
            ld = transposing_default_ld(geom)
            assert ld == geom.m, geom.describe()
            # The register side must not move.
            for r in range(geom.m):
                for k in range(geom.k_eff):
                    assert ab_element_index(r, k, geom) == \
                        ab_element_index(r, k, op), geom.describe()
            for i in range(geom.m):
                for j in range(geom.n_max):
                    assert c_element_index(i, j, geom) == \
                        c_element_index(i, j, op), geom.describe()
            # The memory side must be an exact transpose at LD = M, and a
            # bijection onto the same K_eff x M footprint.
            seen = set()
            for r in range(geom.m):
                for k in range(geom.k_eff):
                    off = ab_memory_offset_t(r, k, ld, geom)
                    assert off == k * ld + r, geom.describe()
                    seen.add(off)
            assert len(seen) == geom.m * geom.k_eff, geom.describe()
            assert max(seen) == geom.m * geom.k_eff - 1, geom.describe()
            # And the buffer builder must agree with the offset function.
            mat = [[r * 1000 + k for k in range(geom.k_eff)]
                   for r in range(geom.m)]
            buf = tile_layout_buffer_t(mat, ld, geom)
            assert len(buf) == geom.m * geom.k_eff, geom.describe()
            for r in range(geom.m):
                for k in range(geom.k_eff):
                    assert buf[k * ld + r] == mat[r][k], geom.describe()
            # tload='t' is W=1 only.
            try:
                TileGeometry(geom.vlen, 32, 1, 1, 1, 4, "t").validate()
            except ValueError:
                pass
            else:
                raise AssertionError("tload='t' must reject W>1")


def check_widening_arithmetic() -> None:
    """int8 x int8 -> int32 must sign-extend below the wrap point.

    :func:`check_signedness_is_immaterial` records why round one could
    ignore altfmt_A/altfmt_B.  This is the other half of that claim: at
    W = 4 the signedness is material, so a model that silently read the
    inputs as unsigned would now be caught.
    """
    rng = random.Random(4004)
    geom = TileGeometry(256, 32, 2, 2, 2 * 2 * 4, 4)
    geom.validate()
    a, b, c = random_case(geom, rng)
    for mat in (a, b):
        for row in mat:
            for v in row:
                assert -128 <= v <= 127, "A/B must be drawn at EEW_A=8"
    got = reference_gemm(a, b, c, geom)
    assert got == _reference_gemm_transposed(a, b, c, geom), geom.describe()

    # The full K_eff=16 sum of int8 products cannot overflow int32, so the
    # reference must agree with an unwrapped exact computation.
    for i in range(geom.m):
        for j in range(geom.n):
            exact = c[i][j] + sum(a[i][k] * b[j][k]
                                  for k in range(geom.k_eff))
            assert got[i][j] == _wrap(exact, geom.sew), (i, j)

    # Reading the inputs as unsigned must change the answer -- otherwise the
    # sign extension in this file is not doing anything.
    as_unsigned = lambda mat: [[v & 0xFF for v in row] for row in mat]
    assert reference_gemm(as_unsigned(a), as_unsigned(b), c, geom) != got, \
        "W=4 signedness must be material"

    # Accumulation wraps at SEW=32, not at some wider internal width.
    hot = [[0x7FFF_FFFF] * geom.n_max for _ in range(geom.m)]
    ones_a = [[1] * geom.k_eff for _ in range(geom.m)]
    ones_b = [[1] * geom.k_eff for _ in range(geom.n_max)]
    wrapped = reference_gemm(ones_a, ones_b, hot, geom)
    assert wrapped[0][0] == _wrap(0x7FFF_FFFF + geom.k_eff, 32), wrapped[0][0]
    assert wrapped[0][0] < 0, "the accumulator must wrap into the negatives"


def check_tile_layout_buffer() -> None:
    """The memory image is row-major at every LMUL (Sail vmtl.v)."""
    rng = random.Random(99)
    for vlen in VLENS:
        for geom in ime_legal_configs(vlen, full_vl_only=True):
            mat = random_matrix(geom.m, geom.k_eff, geom.sew, rng)
            buf = tile_layout_buffer(mat, geom.linesize, geom)
            assert len(buf) == geom.m * geom.k_eff, geom.describe()
            flat = [v for row in mat for v in row]
            assert buf == flat, geom.describe()


def check_c_transfer_geometry() -> None:
    """The C transfer config the spec prescribes must be self-consistent."""
    for vlen in VLENS:
        for geom in ime_legal_configs(vlen, full_vl_only=True):
            assert geom.lam * geom.emul_c == geom.n_max, geom.describe()
            assert geom.vl_c_full == geom.m * geom.m
            assert geom.emul_c * geom.elems_per_reg == geom.vl_c_full
            if geom.emul_c == 16:
                # LMUL=16 is not a legal vtype setting, so there is no
                # single-instruction C transfer; the spec routes this through
                # m16 pair/unpair into two m8 halves.  Round one dodges it.
                assert geom.lmul_c not in (1, 2, 4, 8), geom.describe()


def check_golden_model() -> None:
    """Two independent formulations of the GEMM must agree, and wrap."""
    rng = random.Random(20260902)
    for vlen in (128, 256, 512):
        for geom in ime_legal_configs(vlen, sews=(8, 32), lmuls=(1,)):
            a, b, c = random_case(geom, rng)
            got = reference_gemm(a, b, c, geom)
            alt = _reference_gemm_transposed(a, b, c, geom)
            assert got == alt, geom.describe()
            lo, hi = -(1 << (geom.sew - 1)), (1 << (geom.sew - 1)) - 1
            for row in got:
                for v in row:
                    assert lo <= v <= hi, (geom.describe(), v)
            # Tail columns must be untouched.
            for i in range(geom.m):
                for j in range(geom.n, geom.n_max):
                    assert got[i][j] == c[i][j], geom.describe()


def check_signedness_is_immaterial() -> None:
    """For W=1, reading inputs as unsigned cannot change the result.

    This is why round one does not sweep altfmt_A/altfmt_B for *values*.
    If a future widening instruction lands, this assumption dies with it.
    """
    rng = random.Random(7)
    geom = TileGeometry(256, 8, 2, 1, 2 * 16)
    geom.validate()
    a, b, c = random_case(geom, rng)
    unsigned = lambda mat: [[v & ((1 << geom.sew) - 1) for v in row]
                            for row in mat]
    assert reference_gemm(a, b, c, geom) == \
        reference_gemm(unsigned(a), unsigned(b), c, geom)


def check_checksum_sensitivity() -> None:
    """A single-element error must move exactly one row checksum."""
    rng = random.Random(11)
    geom = TileGeometry(256, 32, 2, 1, 2 * 4)
    geom.validate()
    _, _, c = random_case(geom, rng)
    base = tile_checksums(c, geom)
    for i in range(geom.m):
        for j in range(geom.n):
            perturbed = [row[:] for row in c]
            perturbed[i][j] = _wrap(perturbed[i][j] ^ 1, geom.sew)
            got = tile_checksums(perturbed, geom)
            differing = [r for r in range(geom.m) if got[r] != base[r]]
            assert differing == [i], (i, j, differing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=256,
                        help="VLEN to enumerate in the summary")
    args = parser.parse_args()

    for check in (check_lambda_derivations, check_layout_bijections,
                  check_memory_layout, check_sail_worked_examples,
                  check_widening_geometry, check_round_three_widening,
                  check_transposing_layout, check_widening_arithmetic,
                  check_tile_layout_buffer,
                  check_c_transfer_geometry, check_golden_model,
                  check_signedness_is_immaterial, check_checksum_sensitivity):
        check()
        print(f"  ok  {check.__name__}")

    configs = list(ime_legal_configs(args.vlen, full_vl_only=True))
    print(f"\nVLEN={args.vlen}: {len(configs)} full-VL geometries")
    for geom in configs:
        flag = "  <- no single-insn C transfer" if geom.emul_c == 16 else ""
        print(f"  {geom.describe()}{flag}")
    total = sum(1 for _ in ime_legal_configs(args.vlen))
    print(f"\n{total} geometries including partial N "
          f"-- the S1 gate sweeps all of them")

    for w, sews in sorted(WIDENING_SEWS_BY_W.items()):
        widening = list(ime_legal_configs(args.vlen, sews=sews, ws=(w,),
                                          full_vl_only=True))
        mnemonic = TileGeometry(args.vlen, sews[0], 1, 1, 1, w).mnemonic
        print(f"\nVLEN={args.vlen}: {len(widening)} full-VL W={w} "
              f"({mnemonic}) geometries")
        for geom in widening:
            print(f"  {geom.describe()}")

    trans = list(ime_legal_configs(args.vlen, full_vl_only=True,
                                   tloads=("t",)))
    print(f"\nVLEN={args.vlen}: {len(trans)} full-VL transposing "
          f"(vmttl.v / vmtts.v) geometries")
    for geom in trans:
        print(f"  {geom.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
