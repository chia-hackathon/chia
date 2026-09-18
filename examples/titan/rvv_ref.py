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
import struct
from fractions import Fraction
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
    kind: str = "int"  # "int" = Zvvmm, "fp" = Zvvfmm (round four)
    #: Which *check* the directed generator builds for this geometry.  Not a
    #: property of the architecture: the tile shape is identical either way,
    #: and nothing in this module's index arithmetic reads it.  It exists so
    #: that one geometry sequence can carry two kinds of program.
    #:
    #:   "pair"     -- rounds one to four.  Compute the tile twice on the
    #:                 DUT (IME path and RVV/scalar reference path), store
    #:                 both to memory, compare the two memory images.
    #:   "clayout"  -- round five.  Compute it once and read the C register
    #:                 group back with an ordinary architectural
    #:                 ``vse<SEW>.v``, then compare against the *register*
    #:                 image the spec's ``mat_C_idx`` prescribes.
    #:
    #: The distinction is load-bearing because a "pair" program cannot see
    #: the register-side C layout at all: it writes C with ``vmts.v`` and
    #: reads it back with ``vmts.v``, so any C index permutation that the
    #: implementation applies consistently to the transfer and to the
    #: multiply-accumulate cancels out in the memory image.  See
    #: :func:`clayout_capable` and titan_runs/round5_design.md.
    check: str = "pair"

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
        """The multiply-accumulate this geometry's :attr:`kind` and W select.

        The two families are the same table at two funct6 runs: Zvvmm at
        OPIVV 0x38/0x39/0x3a/0x3b and Zvvfmm at OPFVV 0x14/0x15/0x16/0x17,
        both indexed by W in {1,2,4,8} (spec 1249-1253 and 1336-1339).  Round
        four implements only the W=1 floating-point entry; the others are
        listed so that a geometry built with an unimplemented W fails in
        :meth:`validate` with the reason, not here with a KeyError.
        """
        table = {"int": {1: "vmmacc.vv", 2: "vwmmacc.vv",
                         4: "vqmmacc.vv", 8: "v8wmmacc.vv"},
                 "fp": {1: "vfmmacc.vv", 2: "vfwmmacc.vv",
                        4: "vfqmmacc.vv", 8: "vf8wmmacc.vv"}}
        try:
            return table[self.kind][self.w]
        except KeyError:
            raise ValueError(
                f"no mnemonic modelled for kind={self.kind!r} W={self.w}"
            ) from None

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
        if self.kind not in ("int", "fp"):
            raise ValueError(
                f"kind={self.kind!r}: expected 'int' (Zvvmm) or 'fp' "
                f"(Zvvfmm)")
        if self.check not in ("pair", "clayout"):
            raise ValueError(
                f"check={self.check!r}: expected 'pair' (the differential "
                f"programs of rounds one to four) or 'clayout' (round "
                f"five's C-tile register-layout probe)")
        if self.kind == "fp":
            # Round four implements one floating-point instruction,
            # vfmmacc.vv, at the two accumulator widths whose input format
            # needs no vtype.altfmt_A / altfmt_B decode.  Spec 1490-1493:
            # "For IEEE binary32 and IEEE binary64 inputs, `altfmt_A` and
            # `altfmt_B` are ignored (there is only one format per width)";
            # spec 1092-1106 likewise reserves altfmt=1 for the C format at
            # SEW 32 and 64, so the whole (SEW, W, altfmt_A, altfmt_B,
            # altfmt) encoding-map row collapses to a single legal cell.
            # Everything else in the family -- SEW 8/16 inputs, and every
            # W>1 form -- needs either a sub-word format decode or an exact
            # multi-product partial sum.  See titan_runs/round4_design.md.
            if self.w != 1:
                raise ValueError(
                    f"kind='fp' with W={self.w}: round four implements "
                    f"vfmmacc.vv (W=1) only; the widening floating-point "
                    f"forms need an exact W-product partial sum, which no "
                    f"baseline rv64imafd sequence can recompute in one "
                    f"rounding -- see round4_design.md")
            if self.tload != "op":
                raise ValueError(
                    f"kind='fp' with tload={self.tload!r}: the transposing "
                    f"tile pair is tested at kind='int' (round three); "
                    f"pairing it with a new arithmetic instruction would "
                    f"stop isolating a layout failure from an arithmetic "
                    f"one")
            if self.sew not in FP_FORMATS:
                raise ValueError(
                    f"kind='fp' with SEW={self.sew}: round four models the "
                    f"IEEE binary{{32,64}} accumulator widths only "
                    f"(binary16 / bfloat16 / OFP8 / OFP4 need the "
                    f"vtype.altfmt_A / altfmt_B format decode)")
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
        # The floating-point clause goes last, for the same reason the W and
        # TL clauses do not sit between SEW and LAMBDA: helpers._GEOM_RE
        # scrapes `VLEN=(\d+) SEW=(\d+) LAMBDA=(\d+)` as three adjacent
        # fields, and an integer geometry must print byte-for-byte what
        # round three printed.
        fp = "" if self.kind == "int" else f" FP=binary{self.sew}"
        # And the check clause last of all, for the same reason: a "pair"
        # geometry -- every geometry rounds one to four generate -- must
        # print byte-for-byte what round four printed.
        chk = "" if self.check == "pair" else f" CHK={self.check}"
        return (f"VLEN={self.vlen} SEW={self.sew} LAMBDA={self.lam}{widen} "
                f"LMUL={self.lmul} VL={self.vl} -> M={self.m} N={self.n} "
                f"K_eff={self.k_eff} EMUL_C={self.emul_c}{trans}{fp}{chk}")


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
                      tloads: Sequence[str] = ("op",),
                      kinds: Sequence[str] = ("int",),
                      checks: Sequence[str] = ("pair",)
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

    ``kinds`` selects the arithmetic family -- ``"int"`` for Zvvmm, ``"fp"``
    for round four's Zvvfmm -- and is the outer-most loop of all, again so
    that adding it appends rather than interleaves.  It defaults to
    ``("int",)``, so every pre-round-four caller sees exactly the geometries,
    in exactly the order, that it saw before.

    ``checks`` selects the directed *program shape* (see
    :attr:`TileGeometry.check`) and is now the outer-most loop, ahead of
    ``kinds``, for exactly the same append-don't-interleave reason.  It
    defaults to ``("pair",)``.
    """
    for check in checks:
        for kind in kinds:
            for tload in tloads:
                for w in ws:
                    for sew in sews:
                        for lam in permissible_lambdas(vlen, sew, w):
                            for lmul in lmuls:
                                probe = TileGeometry(vlen, sew, lam, lmul,
                                                     lam * lmul, w, tload,
                                                     kind, check)
                                n_values = ([probe.n_max] if full_vl_only
                                            else range(1, probe.n_max + 1))
                                for n in n_values:
                                    geom = TileGeometry(vlen, sew, lam, lmul,
                                                        n * lam * lmul, w,
                                                        tload, kind, check)
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

    A floating-point geometry draws from :func:`random_fp_case` instead: the
    shapes are identical, but a uniformly random SEW-wide bit pattern is
    almost always a NaN-free huge normal whose products overflow, which
    tests nothing.  The dispatch is here rather than at the call sites so
    that ime_tests, ime_stress and sim_check all get it from one place.
    """
    if geom.kind == "fp":
        return random_fp_case(geom, rng)
    return (random_matrix(geom.m, geom.k_eff, geom.eew_ab, rng),
            random_matrix(geom.n_max, geom.k_eff, geom.eew_ab, rng),
            random_matrix(geom.m, geom.n_max, geom.sew, rng))


# ---------------------------------------------------------------------------
# round five: operands for the C-tile register-layout probe
# ---------------------------------------------------------------------------
#
# The clayout tier reads the C register group back with an architectural
# vse<SEW>.v and compares it against the register image the spec prescribes.
# For that comparison to be *evidence* rather than a coincidence it needs two
# properties from the data, and neither is something a random draw can be
# trusted to supply:
#
#   (1) every element of the resulting C tile is distinct, so a value found
#       in the wrong place identifies where it came from; and
#   (2) the observed image under the "linear C index" hypothesis is the exact
#       transpose of the expected one, so the failure has a name.
#
# (2) needs the pre-instruction C tile to be **symmetric**.  Write E(i,j) for
# the spec's ``c_element_index`` and P(i,j) = i*N_max + j for the linear index
# Titan's RTL uses.  At LAMBDA=1, E(i,j) = j*M + i = P(j,i), so an
# implementation that indexes C linearly on both the vmtl.v transfer and the
# multiply-accumulate leaves, in the slot the spec calls (a,b),
#
#     C_init[a][b] + D(b,a)          where D(i,j) = sum_k A[i][k]*B[j][k]
#
# while the spec puts C_init[a][b] + D(a,b) there.  That is C_ref^T -- i.e.
# exactly the transpose -- precisely when C_init[a][b] == C_init[b][a].  With
# an asymmetric C_init the same bug shows up as a scramble that no verdict
# line can name.  So: symmetric C_init, asymmetric D, distinct C_ref.


def clayout_capable(geom: TileGeometry) -> bool:
    """Can this geometry carry an all-distinct C-layout probe?

    The tier's claim is "every C element is distinct", and at SEW=8 with a
    32x32 tile that is arithmetically impossible: 1024 elements cannot take
    1024 distinct values in 8 bits.  Rather than weaken the claim for the
    geometries where it does not fit, the tier declines them and says so.

    The bound is strict, not >=, because the construction also has to keep
    C_ref different from C_init in *every* element (otherwise a multiply-
    accumulate that wrote nothing would pass somewhere).  Its values run
    1 .. M*N_max, so a tile that exactly saturates the residue class -- SEW=8
    with M = 16 -- would wrap its last element onto C_init's 0.  Those
    geometries drop out; SEW=8 keeps its LAMBDA=4 (M=8, EMUL_C=2) half.

    The other bounds are the same kind of statement: the probe is defined at
    full VL only (a partial-N tile leaves N_max - N tail columns holding
    their pre-instruction value, which are then not distinct -- and the C
    tail policy already has exhaustive coverage in the pair tiers), it needs
    M >= 2 for "transposed" to mean anything, and its A/B operands have to
    fit the signed EEW_A range.
    """
    if geom.kind != "int" or geom.tload != "op":
        return False
    if geom.m < 2 or geom.n != geom.n_max or geom.emul_c == 16:
        return False
    # (2*)M*N_max distinct accumulator values must exist at SEW.  The K_eff=1
    # construction below spans 2*M*N_max residues, the general one spans
    # M*N_max.
    span = geom.m * geom.n_max * (1 if geom.k_eff >= 2 else 2)
    if span >= (1 << geom.sew):
        return False
    # The largest A/B operand either construction writes is M.
    if geom.m > (1 << (geom.eew_ab - 1)) - 1:
        return False
    return True


def clayout_case(geom: TileGeometry) -> Tuple[Matrix, Matrix, Matrix]:
    """(A, B, C_init) for the C-layout probe: symmetric C, distinct C_ref.

    Two constructions, chosen by K_eff, both giving a C_ref whose M*N_max
    elements are pairwise distinct modulo 2**SEW:

      K_eff >= 2:  A[i] = (1, i, 0...), B[j] = (j+1, M, 0...), C_init = 0.
                   D(i,j) = 1*(j+1) + i*M = i*M + j + 1, so
                   C_ref[i][j] = i*M + j + 1 -- the M*N_max distinct residues
                   1 .. M*N_max, and C_init is trivially symmetric.

      K_eff == 1:  only LAMBDA=LMUL=W=1 reaches here, so there is one k and
                   D is forced to be rank one -- and a rank-one D cannot be
                   injective on an MxM grid.  The distinctness comes from
                   C_init instead: A[i] = (1,), B[j] = (j+1,),
                   C_init[i][j] = (i+j)*M, giving
                   C_ref[i][j] = (i+j)*M + j + 1.  That is injective (j+1 is
                   recovered modulo M, then i), and C_init is symmetric in
                   (i,j) as required.

    The ``+1`` in B is not cosmetic: with B[j][0] = j the j=0 column would
    get D = 0, and a multiply-accumulate that wrote nothing at all would
    leave that column looking correct.  Offsetting by one makes every one of
    the M*N_max elements change.

    In both cases D(i,j) != D(j,i) for every i != j, which is what makes the
    write coordinate observable at all: a D that were symmetric, or constant,
    would let a read-modify-write at a consistently wrong index return the
    same tile it started with.
    """
    m, nmax, k = geom.m, geom.n_max, geom.k_eff
    a = [[0] * k for _ in range(m)]
    b = [[0] * k for _ in range(nmax)]
    c = [[0] * nmax for _ in range(m)]
    if k >= 2:
        for i in range(m):
            a[i][0], a[i][1] = 1, i
        for j in range(nmax):
            b[j][0], b[j][1] = j + 1, m
    else:
        for i in range(m):
            a[i][0] = 1
        for j in range(nmax):
            b[j][0] = j + 1
        for i in range(m):
            for j in range(nmax):
                c[i][j] = _wrap((i + j) * m, geom.sew)
    return a, b, c


def clayout_reg_image(c: Matrix, geom: TileGeometry) -> List[int]:
    """The MxN_max tile *c* laid out as a flat C register-group image.

    ``out[c_element_index(i, j)] = c[i][j]``.  The group holds exactly
    ``EMUL_C * epr_C = M * N_max`` elements and ``mat_C_idx`` is a bijection
    onto them, so every position is written exactly once.
    """
    out = [0] * (geom.m * geom.n_max)
    for i in range(geom.m):
        for j in range(geom.n_max):
            out[c_element_index(i, j, geom)] = c[i][j]
    return out


# ---------------------------------------------------------------------------
# round four: IEEE-754 floating point (Zvvfmm)
# ---------------------------------------------------------------------------
#
# Round four adds one instruction, vfmmacc.vv (W=1), at SEW in {32, 64} --
# binary32 x binary32 -> binary32 and binary64 x binary64 -> binary64.  See
# titan_runs/round4_design.md for the survey and the batch decision.  What
# this section has to supply is an *exact* IEEE-754 model, because the whole
# harness compares raw bit patterns and a reference that is merely close is
# no reference at all.
#
# Everything here is built on :class:`fractions.Fraction`, so a product or a
# sum is formed exactly as a rational and rounded exactly once, by
# :func:`fp_round`.  There is no reliance on the host's float type for the
# authoritative path; :func:`check_fp_matches_host` cross-checks the Fraction
# model against Python's binary64 arithmetic, which is a genuinely
# independent implementation of the same rounding rule.

#: (exponent bits, significand bits including the hidden bit) per width.
#: Round four models only the two widths whose format needs no altfmt
#: decode -- spec 1490-1493, "For IEEE binary32 and IEEE binary64 inputs,
#: `altfmt_A` and `altfmt_B` are ignored (there is only one format per
#: width)".  binary16/bfloat16/OFP8/OFP4 are deliberately absent: see the
#: design note.
FP_FORMATS = {32: (8, 24), 64: (11, 53)}

#: The Titan implementation disclosure required by spec 1652-1656, as a
#: mapping from (SEW, W, LAMBDA) to (G, psm, rnd).  Round four discloses one
#: tuple for every geometry it implements:
#:
#:     G = 1, psm = 0, rnd = frm
#:
#: which spec 1771 names as the choice that makes a Zvvm implementation
#: match the analogous Zvtm instruction "For input element widths of 32 bits
#: or greater ... so that each group contains a single product and the
#: partial sum `S` is that product rounded according to `frm`."
#:
#: It is also the only tuple whose result the *test program* can recompute
#: on the DUT with baseline rv64imafd instructions: with G=1 and W=1 a group
#: is one sub-dot-product is one product, so Sail fp_gemm collapses to
#:
#:     acc = fp_add(acc, fp_round_to_frm(fp_mul_exact(a, b)))
#:
#: which is a scalar fmul followed by a scalar fadd -- two roundings, not a
#: fused multiply-add.  Using rnd=xct instead would be one FMA per k and
#: equally reproducible, but rnd=frm is the tuple the spec itself names for
#: >=32-bit inputs, so it is the one disclosed here.
FP_DISCLOSURE = {"G": 1, "psm": 0, "rnd": "frm"}

#: The only rounding mode round four generates.  Spec 1495 and Sail 5695:
#: the accumulation rounding mode is the dynamic `frm`, with no matrix-
#: specific rounding CSR anywhere in the extension.  The test programs set
#: frm explicitly with `fsrmi 0` rather than trusting the reset value.
FP_FRM = 0        # RNE, round to nearest, ties to even


def fp_fields(width: int):
    """(exponent bits, significand bits, bias, max biased exponent)."""
    try:
        ebits, prec = FP_FORMATS[width]
    except KeyError:
        raise ValueError(
            f"no IEEE format modelled at {width} bits; round four models "
            f"{sorted(FP_FORMATS)} only (binary16/bfloat16/OFP8/OFP4 need "
            f"the vtype.altfmt_A / altfmt_B decode -- see round4_design.md)"
        ) from None
    return ebits, prec, (1 << (ebits - 1)) - 1, (1 << ebits) - 1


def fp_is_nan(bits: int, width: int) -> bool:
    ebits, prec, _bias, emax = fp_fields(width)
    return (bits >> (prec - 1)) & emax == emax and bits & ((1 << (prec - 1)) - 1)


def fp_is_inf(bits: int, width: int) -> bool:
    ebits, prec, _bias, emax = fp_fields(width)
    return ((bits >> (prec - 1)) & emax == emax
            and not bits & ((1 << (prec - 1)) - 1))


def fp_default_nan(width: int) -> int:
    """Sail ``fp_defaultNaN``: the canonical quiet NaN for *width*.

    Spec 1880-1886: "NaN payloads are not architecturally significant for IME
    floating-point operations ... whenever a shared floating-point helper
    materializes a NaN result in a concrete floating-point format, it shall
    return the default canonical NaN for that format".  So a NaN result is a
    single fixed bit pattern and compares bit-for-bit like any other value --
    which is why the harness needs no NaN-aware comparison.
    """
    ebits, prec, _bias, emax = fp_fields(width)
    return (emax << (prec - 1)) | (1 << (prec - 2))


def fp_unpack(bits: int, width: int):
    """(kind, sign, value) with *value* an exact :class:`Fraction`.

    ``kind`` is ``"nan"``, ``"inf"`` or ``"num"``; ``sign`` is 0 or 1; for
    ``"num"`` the magnitude is exact and zero is representable (sign carries
    the -0.0 / +0.0 distinction, which the bitwise compare cares about).
    """
    ebits, prec, bias, emax = fp_fields(width)
    sign = (bits >> (width - 1)) & 1
    exp = (bits >> (prec - 1)) & emax
    frac = bits & ((1 << (prec - 1)) - 1)
    if exp == emax:
        return ("nan" if frac else "inf"), sign, Fraction(0)
    if exp == 0:                       # zero or subnormal, no hidden bit
        return "num", sign, Fraction(frac, 1 << (bias - 1 + prec - 1))
    sig = frac | (1 << (prec - 1))
    return "num", sign, Fraction(sig, 1 << (prec - 1)) * _pow2(exp - bias)


def _pow2(e: int) -> Fraction:
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def fp_round(sign: int, value: Fraction, width: int) -> int:
    """Round an exact nonnegative rational to *width* bits, RNE.

    This is Sail ``fp_round_to_frm`` at ``frm = RNE``.  It implements the
    single rounding point: overflow goes to infinity (RNE overflows away
    from zero), underflow rounds into the subnormal range with the same
    ties-to-even rule, and the sign is carried through unchanged so that a
    rounded-to-zero negative result is -0.0.

    *value* must be the exact magnitude; the caller supplies the sign.
    """
    assert value >= 0, value
    ebits, prec, bias, emax = fp_fields(width)
    sbit = sign << (width - 1)
    if value == 0:
        return sbit
    # Choose the unbiased exponent e with 2**e <= value < 2**(e+1), then
    # clamp at the subnormal floor so both ranges use one code path.
    e = _floor_log2(value)
    e = max(e, 1 - bias)
    # Scale so the target significand is an integer in [2**(prec-1), 2**prec)
    # for normals, and in [0, 2**(prec-1)) for subnormals.
    scaled = value / _pow2(e - (prec - 1))
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1):
        q += 1
    if q >> prec:                      # carry out of the top: bump exponent
        q >>= 1
        e += 1
    biased = 0 if q >> (prec - 1) == 0 else e + bias
    if biased >= emax:                 # overflow: RNE gives infinity
        return sbit | (emax << (prec - 1))
    return sbit | (biased << (prec - 1)) | (q & ((1 << (prec - 1)) - 1))


def _floor_log2(value: Fraction) -> int:
    n, d = value.numerator, value.denominator
    e = n.bit_length() - d.bit_length()
    # bit_length is off by at most one; correct exactly, no floats involved.
    while _pow2(e) > value:
        e -= 1
    while _pow2(e + 1) <= value:
        e += 1
    return e


def fp_mul(a: int, b: int, width: int) -> int:
    """Sail ``fp_mul``: exact product, one rounding to *width* under frm.

    Special values follow IEEE-754 and spec 1841-1848: an sNaN or qNaN
    operand, or 0 x inf, produces the *canonical* NaN -- never a propagated
    payload.
    """
    ka, sa, va = fp_unpack(a, width)
    kb, sb, vb = fp_unpack(b, width)
    sign = sa ^ sb
    if ka == "nan" or kb == "nan":
        return fp_default_nan(width)
    if ka == "inf" or kb == "inf":
        if (ka == "num" and va == 0) or (kb == "num" and vb == 0):
            return fp_default_nan(width)      # 0 x inf: invalid
        _, prec, _, emax = fp_fields(width)
        return (sign << (width - 1)) | (emax << (prec - 1))
    return fp_round(sign, va * vb, width)


def fp_add(a: int, b: int, width: int) -> int:
    """Sail ``fp_add``: exact sum, one rounding to *width* under frm.

    inf + (-inf) is invalid and yields the canonical NaN (spec 1850-1856).
    The sign of an exact zero result is IEEE's: -0.0 only when both addends
    were -0.0 (RNE never produces -0.0 from a cancelling sum).
    """
    ka, sa, va = fp_unpack(a, width)
    kb, sb, vb = fp_unpack(b, width)
    if ka == "nan" or kb == "nan":
        return fp_default_nan(width)
    _, prec, _, emax = fp_fields(width)
    if ka == "inf" or kb == "inf":
        if ka == "inf" and kb == "inf" and sa != sb:
            return fp_default_nan(width)
        sign = sa if ka == "inf" else sb
        return (sign << (width - 1)) | (emax << (prec - 1))
    total = (-va if sa else va) + (-vb if sb else vb)
    if total == 0:
        # IEEE 754-2019 6.3: x + y with a zero exact sum is +0 in every
        # rounding mode except roundTowardNegative, *unless* both operands
        # were zeros of the same sign, in which case that sign is kept.
        if va == 0 and vb == 0 and sa == sb:
            return sa << (width - 1)
        return 0
    return fp_round(1 if total < 0 else 0, abs(total), width)


def fp_gemm_reference(a: Matrix, b: Matrix, c: Matrix,
                      geom: "TileGeometry", rnd: str = "frm") -> Matrix:
    """The Sail ``fp_gemm`` result, at the Titan disclosure (G=1, psm=0, rnd=frm).

    Sail 5243-5268, with G = get_fp_grouping(...) = 1::

        foreach (j from 0 to (g.N - 1)) {
          foreach (i from 0 to (g.M - 1)) {
            var acc = read_single_element(g.EEW_C, c_flat, vd);
            foreach (step from 0 to (g.LMUL - 1)) {
              foreach (g0 from 0 to (g.lambda - 1) by G) {
                let S = fp_group_sum(i, j, step, g0, G, ...);
                match round_group_sum(S, rnd, ...) {
                  Some(S_bits) => acc = fp_add(acc, S_bits, ...),
                  None()      => acc = fp_add_internal(acc, S, ...)
                }
              }
            };
            write_single_element(g.EEW_C, c_flat, vd, acc)
          }
        }

    With W = 1 and G = 1, ``fp_group_sum`` (Sail 5005-5022) has
    ``k_lo = k_hi = step * lambda + g0``, so the two nested loops enumerate
    k = 0, 1, ... K_eff-1 in strictly increasing order and S is the single
    exact product A[i,k] x B[j,k].  ``rnd = frm`` then rounds S to the
    accumulator format before ``fp_add`` rounds the accumulation -- two
    rounding points per k, which is exactly a scalar ``fmul`` followed by a
    scalar ``fadd``.  That equivalence is what lets the emitted test program
    carry its own reference (see ime_tests._fp_ref_path).

    Columns j >= N are not computed and keep their prior value: vta = 0 in
    these tests, so the C tile tail is undisturbed (spec 1824-1826 also
    excludes them from fflags).

    ``rnd`` selects the disclosed partial-sum rounding.  ``"frm"`` is what
    Titan discloses and what the emitted programs are judged against;
    ``"xct"`` is the other legal choice at G=1 (Sail 5262-5263,
    ``None() => acc = fp_add_internal(acc, S, ...)``) and is a fused
    multiply-add.  The second form exists only so that
    :func:`fp_case_is_rounding_witness` can require every generated case to
    tell them apart -- if it could not, "exact bitwise compare" would be
    decoration and a DUT that fused its multiply-add would be green.
    """
    if rnd not in ("frm", "xct"):
        raise ValueError(
            f"rnd={rnd!r}: at G=1 the disclosed choices this models are "
            f"'frm' (round the product, then round the accumulation) and "
            f"'xct' (one rounding, i.e. a fused multiply-add).  'rto' "
            f"needs a round-to-odd mode rvv_ref does not implement.")
    if geom.kind != "fp":
        raise ValueError(f"{geom.describe()}: not a floating-point geometry")
    if geom.w != 1:
        raise ValueError(
            f"W={geom.w}: round four models the floating-point family at "
            f"W=1 only (vfmmacc.vv); the widening forms need an exact "
            f"multi-product partial sum -- see round4_design.md")
    width = geom.sew
    out = [row[:] for row in c]
    for i in range(geom.m):
        for j in range(geom.n):
            acc = c[i][j]
            for k in range(geom.k_eff):
                if rnd == "frm":
                    acc = fp_add(acc, fp_mul(a[i][k], b[j][k], width), width)
                else:
                    acc = fp_fused_step(acc, a[i][k], b[j][k], width)
            out[i][j] = acc
    return out


def fp_fused_step(acc: int, a: int, b: int, width: int) -> int:
    """``round_frm(acc + exact(a*b))`` -- the rnd=xct step, one rounding.

    Sail ``fp_add_internal`` (5262-5263, helper described at 1866-1868):
    the exact internal product is added to the accumulator and only the
    final result is rounded.  Not the Titan disclosure; used to prove the
    disclosure is observable.
    """
    ka, sa, va = fp_unpack(a, width)
    kb, sb, vb = fp_unpack(b, width)
    kc, sc, vc = fp_unpack(acc, width)
    if "num" not in (ka, kb, kc) or not (ka == kb == kc == "num"):
        # A NaN or an infinity anywhere makes the two forms agree, because
        # neither rounding step is reached; fall back to the ordinary path
        # so the special-value rules stay in one place.
        return fp_add(acc, fp_mul(a, b, width), width)
    total = ((-vc if sc else vc)
             + (-va if sa else va) * (-vb if sb else vb))
    if total == 0:
        return 0
    return fp_round(1 if total < 0 else 0, abs(total), width)


def fp_case_is_rounding_witness(a: Matrix, b: Matrix, c: Matrix,
                                geom: "TileGeometry") -> bool:
    """Does this case distinguish rnd=frm from rnd=xct?

    A floating-point dot product tells the two apart only when some term's
    discarded product tail crosses a rounding boundary of the accumulation
    -- which is a property of the *data*, not of the geometry.  The small
    geometries (M = N = 2 with K_eff = 2 is eight terms in total) draw data
    that fails to often enough that leaving it to chance would mean a few of
    the directed programs silently could not catch a fused multiply-add.
    So :func:`random_fp_case` redraws until this holds, and every emitted
    floating-point program is a witness by construction.
    """
    return (fp_gemm_reference(a, b, c, geom, rnd="frm")
            != fp_gemm_reference(a, b, c, geom, rnd="xct"))


#: Finite, well-scaled bit patterns the directed FP tier draws from.
#: Deliberately no infinities and no NaNs among the *inputs*: once a NaN or
#: an infinity enters an accumulator it absorbs every later term, so the rest
#: of that dot product stops testing anything.  The values that do appear are
#: the ones that exercise the rounding path and are still order-sensitive:
#: +-0.0, the smallest subnormal, a value just under 1 whose square is
#: inexact, and normals several binades apart so that a mis-ordered or
#: mis-rounded accumulation shows up in the low significand bits.
#: NaN, infinity and overflow behaviour is pinned by
#: :func:`check_fp_special_values` in the reference instead, where a NaN
#: cannot mask the rest of the test.
def _fp_pool(width: int) -> List[int]:
    ebits, prec, bias, emax = fp_fields(width)
    pool = [
        0,                                   # +0.0
        1 << (width - 1),                    # -0.0
        1,                                   # smallest positive subnormal
        (1 << (width - 1)) | 1,              # smallest negative subnormal
        (1 << (prec - 1)) - 1,               # largest subnormal
        (bias - 1) << (prec - 1),            # +0.5
        bias << (prec - 1),                  # +1.0
        (1 << (width - 1)) | (bias << (prec - 1)),           # -1.0
        (bias << (prec - 1)) | 1,            # 1.0 + 1ulp: squares inexact
        (bias << (prec - 1)) | ((1 << (prec - 1)) - 1),      # just under 2.0
    ]
    # A spread of normals a few binades apart.  The significands fill the
    # *whole* fraction field, which is the property that matters: a
    # significand with trailing zeros gives an exactly-representable
    # product, and a tier built from those cannot tell rnd=frm from rnd=xct
    # (sim_check.check_fp_rounding_is_load_bearing is the test that says
    # so).  Three fixed patterns rather than random bits, so a program's
    # data is reproducible from its geometry and seed alone.
    mask = (1 << (prec - 1)) - 1
    patterns = (0xAAAAAAAAAAAAAAAA & mask,
                0x5555555555555555 & mask,
                0x9E3779B97F4A7C15 & mask)
    for shift in (-6, -3, -1, 0, 2, 5, 9):
        for sign in (0, 1):
            biased = bias + shift
            for pattern in patterns:
                pool.append((sign << (width - 1)) | (biased << (prec - 1))
                            | pattern)
    return pool


def random_fp_matrix(rows: int, cols: int, width: int,
                     rng: random.Random) -> Matrix:
    """A matrix of FP bit patterns drawn from :func:`_fp_pool`."""
    pool = _fp_pool(width)
    return [[rng.choice(pool) for _ in range(cols)] for _ in range(rows)]


#: How many times :func:`random_fp_case` may redraw before giving up.  Not
#: a tuning knob: every geometry round four generates is a witness within a
#: handful of draws, and a limit that were ever reached would mean a
#: geometry whose programs cannot observe the disclosed rounding, which is a
#: fact worth raising on rather than papering over.
FP_WITNESS_ATTEMPTS = 64


def random_fp_case(geom: "TileGeometry", rng: random.Random):
    """(A, B, C) of bit patterns for one directed floating-point test.

    Same shapes as :func:`random_case`; at W = 1, EEW_A == SEW, so A, B and
    C are all drawn at the accumulator width.

    The draw is rejected and repeated until the case is a rounding witness
    (:func:`fp_case_is_rounding_witness`), so every emitted floating-point
    program provably distinguishes the disclosed rnd=frm from a fused
    multiply-add.  The rejection consumes the caller's rng, so the sequence
    is still reproducible from (geometry, seed) alone; it is applied only to
    floating-point geometries, so no integer program's draw moves.
    """
    for _ in range(FP_WITNESS_ATTEMPTS):
        case = (random_fp_matrix(geom.m, geom.k_eff, geom.sew, rng),
                random_fp_matrix(geom.n_max, geom.k_eff, geom.sew, rng),
                random_fp_matrix(geom.m, geom.n_max, geom.sew, rng))
        if fp_case_is_rounding_witness(*case, geom):
            return case
    raise ValueError(
        f"{geom.describe()}: no rounding witness in "
        f"{FP_WITNESS_ATTEMPTS} draws -- this geometry's programs cannot "
        f"observe the disclosed (G=1, psm=0, rnd=frm), so it does not "
        f"belong in the floating-point tier")


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


def check_fp_round() -> None:
    """Hand-derived IEEE-754 rounding cases for :func:`fp_round`.

    Every constant below is worked out from the format parameters, not read
    back from the host, so this is an independent statement of what the
    rounding is supposed to do.  binary32: 8 exponent bits, bias 127,
    24-bit significand.  binary64: 11 / 1023 / 53.
    """
    # 1.0 = sign 0, biased exponent 127, zero fraction.
    assert fp_round(0, Fraction(1), 32) == 127 << 23
    assert fp_round(1, Fraction(1), 32) == (1 << 31) | (127 << 23)
    assert fp_round(0, Fraction(1), 64) == 1023 << 52
    # Exact zero keeps its sign: -0.0 is a distinct bit pattern and the
    # harness compares bits, so this matters.
    assert fp_round(0, Fraction(0), 32) == 0
    assert fp_round(1, Fraction(0), 32) == 1 << 31

    # Ties to even, at the ulp boundary of 1.0 in binary32 (ulp = 2**-23).
    ulp = Fraction(1, 1 << 23)
    one = 127 << 23
    #   1 + ulp/2 is an exact tie between 1.0 and 1.0+ulp; 1.0 has an even
    #   significand, so it wins.
    assert fp_round(0, 1 + ulp / 2, 32) == one
    #   1 + 3*ulp/2 ties between 1+ulp (odd) and 1+2*ulp (even): 1+2*ulp.
    assert fp_round(0, 1 + 3 * ulp / 2, 32) == one + 2
    #   Just over half an ulp rounds up regardless of parity.
    assert fp_round(0, 1 + ulp * Fraction(2, 3), 32) == one + 1

    # Carry out of the top of the significand bumps the exponent: the value
    # just below 2.0 plus most of an ulp becomes exactly 2.0.
    just_under_two = 2 - ulp
    assert fp_round(0, just_under_two + ulp * Fraction(3, 4), 32) == 128 << 23

    # Subnormals.  The smallest positive binary32 subnormal is 2**-149 and
    # the smallest normal is 2**-126; halfway between 0 and 2**-149 is a tie
    # that must round to even, i.e. to zero.
    tiny = Fraction(1, 1 << 149)
    assert fp_round(0, tiny, 32) == 1
    assert fp_round(0, tiny / 2, 32) == 0                 # tie -> even (zero)
    assert fp_round(1, tiny / 2, 32) == 1 << 31           # ... keeping -0.0
    assert fp_round(0, tiny * Fraction(3, 4), 32) == 1    # over half -> up
    assert fp_round(0, tiny * Fraction(3, 2), 32) == 2    # tie -> even (2)
    #   The largest subnormal plus one more tiny is the smallest normal.
    largest_sub = Fraction((1 << 23) - 1, 1 << 149)
    assert fp_round(0, largest_sub, 32) == (1 << 23) - 1  # exp field 0
    assert fp_round(0, largest_sub + tiny, 32) == 1 << 23  # exp field 1

    # Overflow: RNE rounds away from zero past the largest finite value, so
    # the result is infinity, and the sign is preserved.
    #   largest binary32 finite = (2 - 2**-23) * 2**127
    huge = (2 - ulp) * Fraction(1 << 127)
    assert fp_round(0, huge, 32) == 0x7F7FFFFF
    assert fp_round(0, huge * 2, 32) == 0x7F800000        # +inf
    assert fp_round(1, huge * 2, 32) == 0xFF800000        # -inf
    #   The rounding boundary itself: anything at or above
    #   (2 - 2**-24) * 2**127 rounds to infinity.
    boundary = (2 - ulp / 2) * Fraction(1 << 127)
    assert fp_round(0, boundary, 32) == 0x7F800000
    assert fp_round(0, boundary - tiny, 32) == 0x7F7FFFFF

    # fp_unpack is the exact inverse of fp_round on representable values.
    for width in sorted(FP_FORMATS):
        for bits in _fp_pool(width):
            kind, sign, value = fp_unpack(bits, width)
            assert kind == "num", (width, hex(bits))
            assert fp_round(sign, value, width) == bits, (width, hex(bits))


def check_fp_matches_host() -> None:
    """The Fraction model must agree with the host's binary64 arithmetic.

    An independent implementation of the same rounding rule: CPython's
    ``float`` is IEEE binary64 with round-to-nearest-even, so at SEW=64 the
    comparison is direct.  At SEW=32 the host computes in binary64 and the
    result is then rounded to binary32, which is a *double* rounding -- it
    is innocuous here by Figueroa's theorem (an intermediate format with at
    least 2p+2 bits is enough, and 53 >= 2*24+2 = 50), and this test is the
    empirical half of that argument.

    Random bit patterns rather than the directed pool, so the subnormal and
    overflow boundaries get hit by accident as well as on purpose.
    """
    rng = random.Random(0x4D4D4143)
    for width in sorted(FP_FORMATS):
        pack, unpack = ("<Q", "<d") if width == 64 else ("<I", "<f")
        def to_host(bits: int) -> float:
            return struct.unpack(unpack, struct.pack(pack, bits))[0]
        def from_host(value: float) -> int:
            return struct.unpack(pack, struct.pack(unpack, value))[0]
        for _ in range(20000):
            a = rng.getrandbits(width)
            b = rng.getrandbits(width)
            if fp_is_nan(a, width) or fp_is_nan(b, width):
                continue        # host NaN payloads are not canonicalised
            for op, host in ((fp_mul, lambda x, y: x * y),
                             (fp_add, lambda x, y: x + y)):
                got = op(a, b, width)
                try:
                    want = from_host(host(to_host(a), to_host(b)))
                except (OverflowError, ValueError):   # pragma: no cover
                    continue
                if fp_is_nan(got, width):
                    # The host does not canonicalise; the spec does
                    # (1880-1886), so only agree that it *is* a NaN.
                    assert fp_is_nan(want, width), (width, hex(a), hex(b))
                    assert got == fp_default_nan(width)
                    continue
                assert got == want, (
                    f"{op.__name__} binary{width} {a:#x} {b:#x}: "
                    f"model {got:#x}, host {want:#x}")


def check_fp_special_values() -> None:
    """NaN, infinity, signed zero and subnormal behaviour, pinned explicitly.

    The directed programs draw only finite inputs -- a NaN or an infinity
    entering an accumulator absorbs every later term, so the rest of that
    dot product would stop testing anything.  The behaviour still has to be
    specified, and this is where it is: every case below is a statement
    about what the *reference* does, so a DUT that disagrees is caught by
    sim_check rather than by an unreproducible directed failure.
    """
    for width in sorted(FP_FORMATS):
        _ebits, prec, _bias, emax = fp_fields(width)
        inf = emax << (prec - 1)
        ninf = (1 << (width - 1)) | inf
        nan = fp_default_nan(width)
        one = ((1 << (_ebits - 1)) - 1) << (prec - 1)
        zero, nzero = 0, 1 << (width - 1)

        # Spec 1880-1891: every materialised NaN is the canonical one, both
        # for propagated NaNs and for invalid operations.  A non-canonical
        # NaN input therefore cannot survive into the result.
        noisy = nan | 0x5                       # a different payload
        assert fp_is_nan(noisy, width)
        assert fp_mul(noisy, one, width) == nan
        assert fp_add(noisy, one, width) == nan
        assert fp_mul(one, noisy, width) == nan

        # Invalid operations (spec 1841-1856).
        assert fp_mul(zero, inf, width) == nan          # 0 x inf
        assert fp_mul(nzero, inf, width) == nan
        assert fp_add(inf, ninf, width) == nan          # inf + (-inf)

        # Ordinary infinity arithmetic.
        assert fp_mul(inf, one, width) == inf
        assert fp_mul(ninf, one, width) == ninf
        assert fp_mul(inf, inf, width) == inf
        assert fp_mul(inf, ninf, width) == ninf
        assert fp_add(inf, one, width) == inf
        assert fp_add(inf, inf, width) == inf

        # Signed zero, IEEE 754-2019 6.3: an exact zero sum is +0 under RNE
        # unless both addends are the same-signed zero.
        assert fp_add(zero, zero, width) == zero
        assert fp_add(nzero, nzero, width) == nzero
        assert fp_add(zero, nzero, width) == zero
        assert fp_add(one, one | (1 << (width - 1)), width) == zero
        # ... and a product's zero sign is the xor of the operand signs.
        assert fp_mul(zero, one, width) == zero
        assert fp_mul(nzero, one, width) == nzero
        assert fp_mul(nzero, one | (1 << (width - 1)), width) == zero

        # Subnormals are never flushed: there is no FTZ/DAZ control anywhere
        # in the extension, and fp_mul_exact is defined over "finite
        # operands, including subnormal operands and signed zeros"
        # (spec 1841-1843).
        tiny = 1                                       # smallest subnormal
        assert fp_add(tiny, tiny, width) == 2
        assert fp_mul(tiny, one, width) == tiny
        #   A product that underflows below half the smallest subnormal is a
        #   correctly-signed zero, not an error.
        half = ((1 << (_ebits - 1)) - 2) << (prec - 1)  # 0.5
        assert fp_mul(tiny, half, width) == 0
        assert fp_mul(tiny, half | (1 << (width - 1)), width) == nzero

        # Overflow to infinity, and the largest finite value just below it.
        maxfinite = ((emax - 1) << (prec - 1)) | ((1 << (prec - 1)) - 1)
        assert fp_add(maxfinite, maxfinite, width) == inf
        assert fp_mul(maxfinite, maxfinite, width) == inf


def check_round_four_fp_gemm() -> None:
    """The Sail fp_gemm result at the Titan disclosure, hand-derived.

    Two claims, both of which a plausible implementation gets wrong:

    1.  ``rnd = frm`` means the product is rounded to the accumulator format
        *before* it is accumulated -- two rounding points per k, not the one
        of a fused multiply-add.  The witness below is a dot product whose
        FMA answer and whose mul-then-add answer differ in the last bit.
    2.  The accumulation is in strictly increasing k and is *not*
        reassociable: at G=1 there is one product per group and the
        accumulator is rounded after every one of them.  The witness is a
        three-term sum whose value depends on the order.
    """
    width = 32
    one = 127 << 23                       # 1.0
    ulp = 1                               # 1.0 + 1ulp is `one + 1`

    #   (1 + 2**-23) * (1 + 2**-23) = 1 + 2**-22 + 2**-46.  Rounded to
    #   binary32 that is 1 + 2**-22 (the 2**-46 tail is far below half an
    #   ulp), i.e. `one + 2`.
    prod = fp_mul(one + ulp, one + ulp, width)
    assert prod == one + 2, hex(prod)
    #   Accumulating that rounded product into -1.0 gives exactly 2**-22.
    neg_one = (1 << 31) | one
    assert fp_add(neg_one, prod, width) == (127 - 22) << 23
    #   rnd=frm and rnd=xct are distinguishable, so the disclosure is
    #   load-bearing rather than a formality.  Accumulate the same product
    #   into -(1 + 2**-22): the rounded product cancels it exactly and the
    #   answer is +0.0, while a fused multiply-add keeps the 2**-46 tail and
    #   answers 2**-46.  A DUT that used an FMA here would fail every
    #   directed program that happened to hit this pattern.
    minus_prod = (1 << 31) | (one + 2)          # -(1 + 2**-22)
    assert fp_add(minus_prod, prod, width) == 0         # rnd=frm
    exact_fma = fp_round(0, Fraction(1, 1 << 46), width)
    assert exact_fma == (127 - 46) << 23                # rnd=xct
    assert exact_fma != fp_add(minus_prod, prod, width)
    assert FP_DISCLOSURE == {"G": 1, "psm": 0, "rnd": "frm"}

    #   Order sensitivity: 1.0 + 2**-24 + 2**-24.  Left to right, the first
    #   addition ties to even and drops the bit, so the second one does too
    #   and the answer is 1.0.  Summed exactly first it is 1 + 2**-23.
    small = (127 - 24) << 23
    acc = fp_add(fp_add(one, small, width), small, width)
    assert acc == one, hex(acc)
    assert fp_round(0, Fraction(1) + Fraction(2, 1 << 24), width) == one + 1

    #   Now the same two facts through fp_gemm_reference, on a 1x1 tile of
    #   the smallest legal binary32 geometry.
    geom = TileGeometry(256, 32, 2, 1, 2 * 1 * 4, 1, "op", "fp")
    geom.validate()
    assert geom.mnemonic == "vfmmacc.vv"
    assert geom.k_eff == 2 and geom.m == 4 and geom.eew_ab == 32
    a = [[one + ulp, one] for _ in range(geom.m)]
    b = [[one + ulp, 0] for _ in range(geom.n_max)]
    c = [[minus_prod] * geom.n_max for _ in range(geom.m)]
    out = fp_gemm_reference(a, b, c, geom)
    #   k=0: S = round_frm((1+2**-23)^2) = 1 + 2**-22, which cancels the
    #        accumulator exactly:  acc = round_frm(-(1+2**-22) + S) = +0.0
    #   k=1: S = round_frm(1 * 0) = +0.0, and +0.0 + +0.0 = +0.0
    want = 0
    for i in range(geom.m):
        for j in range(geom.n):
            assert out[i][j] == want, (i, j, hex(out[i][j]), hex(want))

    #   rnd='xct' is the same geometry with one rounding instead of two,
    #   and on this case it answers differently -- which is what makes the
    #   case a witness.
    #   Under rnd=xct the 2**-46 tail of the k=0 product survives the
    #   cancellation and the k=1 term of +0.0 cannot wash it out, so the
    #   same case answers 2**-46 instead of +0.0.
    fused = fp_gemm_reference(a, b, c, geom, rnd="xct")
    assert fused != out
    for i in range(geom.m):
        for j in range(geom.n):
            assert fused[i][j] == (127 - 46) << 23, hex(fused[i][j])
    assert fp_case_is_rounding_witness(a, b, c, geom)
    #   ... and every case the generator hands out is one, by construction.
    rng = random.Random(0x515B)
    for probe in ime_legal_configs(256, full_vl_only=True, kinds=("fp",)):
        for _ in range(4):
            case = random_fp_case(probe, rng)
            assert fp_case_is_rounding_witness(*case, probe), probe.describe()
            #   The drawn values must all be finite: a NaN or an infinity in
            #   an input would absorb the rest of that dot product.
            for mat in case:
                for row in mat:
                    for value in row:
                        assert not fp_is_nan(value, probe.sew)
                        assert not fp_is_inf(value, probe.sew)
            #   ... and so must the results, or the tier would be testing
            #   an absorbing state rather than an accumulation.
            for row in fp_gemm_reference(*case, probe):
                for value in row:
                    assert not fp_is_nan(value, probe.sew)
                    assert not fp_is_inf(value, probe.sew)

    #   Tail columns keep their pre-instruction value (vta=0).
    assert neg_one == (1 << 31) | one and small == (127 - 24) << 23
    partial = TileGeometry(256, 32, 2, 1, 2 * 1 * 2, 1, "op", "fp")
    partial.validate()
    assert partial.n == 2 < partial.n_max
    out = fp_gemm_reference(a, b, c, partial)
    for i in range(partial.m):
        for j in range(partial.n, partial.n_max):
            assert out[i][j] == c[i][j]


def check_round_four_geometry() -> None:
    """A floating-point geometry is the integer geometry at the same vtype.

    Spec 1500: "The K-dimension, tile-dimension formulas, EMUL_C, and
    instruction-to-widening-factor mapping are the same as for the integer
    family"; Sail calls the very same ``decode_gemm_geometry(W)``
    (4884-4912) from ``vfmmacc.vv``'s body at W=1.  So nothing about M,
    N_max, K_eff, EMUL_C or the permissible LAMBDA set may move -- only the
    mnemonic and the arithmetic.
    """
    for vlen in VLENS:
        for sew in sorted(FP_FORMATS):
            for lam in permissible_lambdas(vlen, sew):
                for lmul in (1, 2, 4, 8):
                    vl = lam * lmul
                    i = TileGeometry(vlen, sew, lam, lmul, vl, 1, "op", "int")
                    f = TileGeometry(vlen, sew, lam, lmul, vl, 1, "op", "fp")
                    legal_i = legal_f = True
                    try:
                        i.validate()
                    except ValueError:
                        legal_i = False
                    try:
                        f.validate()
                    except ValueError:
                        legal_f = False
                    assert legal_i == legal_f, (vlen, sew, lam, lmul)
                    if not legal_i:
                        continue
                    for attr in ("m", "n_max", "n", "k_eff", "emul_c",
                                 "linesize", "elems_per_reg", "eew_ab",
                                 "vl_c_full", "lmul_c", "epr_ab",
                                 "ab_linesize", "ab_row_elems_per_reg"):
                        assert getattr(i, attr) == getattr(f, attr), attr
                    assert f.mnemonic == "vfmmacc.vv"
                    assert f.load_mnemonic == "vmtl.v"
                    assert f.store_mnemonic == "vmts.v"
                    # The verdict line must keep the three adjacent fields
                    # helpers._GEOM_RE scrapes, and must differ from the
                    # integer line only by the trailing FP= clause.
                    assert f.describe() == i.describe() + f" FP=binary{sew}"
                    assert (f"VLEN={vlen} SEW={sew} LAMBDA={lam}"
                            in f.describe())

    # The deferred half of the family must refuse to be constructed rather
    # than silently produce a geometry no reference models.
    for bad, why in (
        (TileGeometry(256, 32, 2, 1, 2, 2, "op", "fp"), "W=2"),
        (TileGeometry(256, 16, 4, 1, 4, 1, "op", "fp"), "SEW=16"),
        (TileGeometry(256, 8, 4, 1, 4, 1, "op", "fp"), "SEW=8"),
        (TileGeometry(256, 32, 2, 1, 2, 1, "t", "fp"), "transposing"),
    ):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"round four accepted a {why} FP geometry")

    # ime_legal_configs must append, never interleave: the integer sequence
    # has to be an exact prefix of the int+fp sequence.
    for vlen in (128, 256, 512):
        ints = list(ime_legal_configs(vlen))
        both = list(ime_legal_configs(vlen, kinds=("int", "fp")))
        assert both[:len(ints)] == ints, vlen
        fps = both[len(ints):]
        assert fps == list(ime_legal_configs(vlen, kinds=("fp",)))
        assert all(g.kind == "fp" and g.mnemonic == "vfmmacc.vv"
                   for g in fps)


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


def check_clayout_case() -> None:
    """The C-layout probe's operands must do what the tier claims.

    Four properties, each of which the tier's verdict rests on, checked over
    every clayout-capable geometry at every VLEN this file enumerates:

      * C_init is symmetric -- without which the linear-index hypothesis
        does not produce an exact transpose and `TITAN CLAYOUT transposed`
        could never fire;
      * C_ref's M*N_max elements are pairwise distinct -- so a value in the
        wrong place says where it came from;
      * C_ref differs from C_init in every element -- so a multiply-
        accumulate that wrote nothing at all cannot pass; and
      * the linear-index image really is C_ref transposed, at LAMBDA=1.

    The last one is computed here from the Sail index functions rather than
    asserted from the RTL: it is a statement about ``tile_reg_idx``, which is
    what the directed program compares against.
    """
    checked = 0
    for vlen in VLENS:
        for ws, sews in ((( 1,), (8, 16, 32, 64)),) + tuple(
                ((w,), WIDENING_SEWS_BY_W[w]) for w in (2, 4, 8)):
            for geom in ime_legal_configs(vlen, sews=sews, ws=ws,
                                          full_vl_only=True,
                                          checks=("clayout",)):
                if not clayout_capable(geom):
                    continue
                a, b, c0 = clayout_case(geom)
                assert all(abs(v) <= (1 << (geom.eew_ab - 1)) - 1
                           for row in a + b for v in row), geom.describe()
                for i in range(geom.m):
                    for j in range(geom.n_max):
                        assert c0[i][j] == c0[j][i], geom.describe()
                ref = reference_gemm(a, b, c0, geom)
                flat = [ref[i][j] for i in range(geom.m)
                        for j in range(geom.n_max)]
                assert len(set(flat)) == len(flat), (
                    f"{geom.describe()}: C_ref is not all-distinct")
                assert all(ref[i][j] != c0[i][j] for i in range(geom.m)
                           for j in range(geom.n_max)), (
                    f"{geom.describe()}: a no-op multiply-accumulate would "
                    f"pass this case")
                if geom.lam == 1:
                    # The linear-index hypothesis: read and write C at
                    # i*N_max + j instead of at tile_reg_idx(i*N_max + j).
                    observed = [0] * (geom.m * geom.n_max)
                    for i in range(geom.m):
                        for j in range(geom.n_max):
                            p = c_sequential_index(i, j, geom)
                            observed[p] = _wrap(
                                clayout_reg_image(c0, geom)[p]
                                + sum(a[i][k] * b[j][k]
                                      for k in range(geom.k_eff)), geom.sew)
                    transposed = [[ref[j][i] for j in range(geom.n_max)]
                                  for i in range(geom.m)]
                    assert observed == clayout_reg_image(transposed, geom), (
                        f"{geom.describe()}: the linear-index image is not "
                        f"the transpose of the expected one")
                checked += 1
    assert checked > 20, checked


def clayout_geometries(vlen: int, sews: Sequence[int] = (8, 16, 32, 64),
                        lmuls: Sequence[int] = (1, 2, 4, 8)
                        ) -> List[TileGeometry]:
    """Every clayout-capable geometry, W=1 then W=4, in tier order."""
    out = []
    for ws, tier_sews in ((1, sews), (4, WIDENING_SEWS)):
        out += [g for g in ime_legal_configs(
            vlen, sews=[x for x in tier_sews if x in sews], lmuls=lmuls,
            full_vl_only=True, ws=(ws,), checks=("clayout",))
            if clayout_capable(g)]
    return out


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
                  check_signedness_is_immaterial, check_checksum_sensitivity,
                  check_fp_round, check_fp_matches_host,
                  check_fp_special_values, check_round_four_fp_gemm,
                  check_round_four_geometry, check_clayout_case):
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

    fp = list(ime_legal_configs(args.vlen, full_vl_only=True,
                                kinds=("fp",)))
    print(f"\nVLEN={args.vlen}: {len(fp)} full-VL floating-point "
          f"(vfmmacc.vv) geometries, disclosure "
          f"G={FP_DISCLOSURE['G']} psm={FP_DISCLOSURE['psm']} "
          f"rnd={FP_DISCLOSURE['rnd']}")
    for geom in fp:
        print(f"  {geom.describe()}")

    trans = list(ime_legal_configs(args.vlen, full_vl_only=True,
                                   tloads=("t",)))
    print(f"\nVLEN={args.vlen}: {len(trans)} full-VL transposing "
          f"(vmttl.v / vmtts.v) geometries")
    for geom in trans:
        print(f"  {geom.describe()}")
    clayout = [g for g in clayout_geometries(args.vlen)]
    print(f"\nVLEN={args.vlen}: {len(clayout)} full-VL C-layout probe "
          f"geometries (round five)")
    for geom in clayout:
        print(f"  {geom.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
