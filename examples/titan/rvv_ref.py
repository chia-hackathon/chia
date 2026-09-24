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
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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
                        4: "vfqmmacc.vv", 8: "vf8wmmacc.vv"},
                 # Round six: the MX integer-input / FP-accumulate forms.
                 # They ride the *integer* funct6 run (OPIVV 0x39/0x3a/0x3b)
                 # and are selected by vm=0 -- at vm=1 those same three
                 # encodings are vwmmacc.vv / vqmmacc.vv / v8wmmacc.vv, which
                 # rounds one to three already implement (Sail 5963-5975,
                 # 6196-6205, 6082-6091).  There is no W=1 entry: vmmacc.vv
                 # keeps vm=0 reserved (spec 2333).
                 "mx": {2: "vfwimmacc.vv", 4: "vfqimmacc.vv",
                        8: "vf8wimmacc.vv"},
                 # Round seven: the widening floating-point forms.  Same
                 # three mnemonics as the "fp" row at W>1 -- deliberately,
                 # because they *are* the same encodings (OPFVV 0x15/0x16/
                 # 0x17, spec 1336-1339).  The separate kind exists because
                 # round four's kind='fp' carries a validate() that rejects
                 # W>1, and relaxing that in place would change what an
                 # existing round-four geometry accepts.
                 "fpw": {2: "vfwmmacc.vv", 4: "vfqmmacc.vv",
                         8: "vf8wmmacc.vv"}}
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
        if self.kind not in ("int", "fp", "mx", "fpw"):
            raise ValueError(
                f"kind={self.kind!r}: expected 'int' (Zvvmm), 'fp' "
                f"(Zvvfmm) or 'mx' (round six, the Zvvfmm integer-input "
                f"microscaled forms) or 'fpw' (round seven, the widening "
                f"floating-point forms vfwmmacc.vv / vfqmmacc.vv / "
                f"vf8wmmacc.vv)")
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
        if self.kind == "fpw":
            # Round seven.  As with round six the legal cells are a table
            # (tbl-fp-encoding-map, spec 7288-7370), so validate against it
            # directly rather than against a rule the architecture does not
            # state.  altfmt / altfmt_A / altfmt_B are not TileGeometry
            # fields -- every row of a cell shares this geometry and the
            # generator picks the format triple -- so only the cell's
            # existence is checked here, via the row that every cell has.
            _eew, _rows = FP_CELLS[(self.w, self.sew)] \
                if (self.w, self.sew) in FP_CELLS else (None, None)
            if _rows is None:
                raise ValueError(
                    f"kind='fpw' W={self.w} SEW={self.sew} is a reserved "
                    f"encoding; legal cells are {sorted(FP_CELLS)} "
                    f"(spec 7288-7370)")
            if self.w == 1:
                raise ValueError(
                    "kind='fpw' is the *widening* floating-point family "
                    "(W in {2,4,8}); vfmmacc.vv (W=1) is kind='fp'")
            if self.tload != "op":
                raise ValueError(
                    f"kind='fpw' with tload={self.tload!r}: the transposing "
                    f"pair is tested at kind='int', so that a layout "
                    f"failure stays separable from an arithmetic one")
            # Unscaled (vm=1) needs no microscaling rule; the vm=0 tier calls
            # fpw_check_legality(..., vm=0) per program, because bs and LMUL
            # are program properties rather than geometry ones.
            fpw_check_legality(self.w, self.lmul, self.sew, self.lam, 0, 1)
        if self.kind == "mx":
            # Round six.  The legal (W, SEW) cells are a *table*
            # (tbl-intmx-encoding-map, spec 7469-7540) rather than a rule,
            # so validate against it directly; mx_legal_cell raises with the
            # reserved-encoding reason.  altfmt is not a TileGeometry field
            # -- both rows of a two-row cell share this geometry and the
            # generator picks the accumulator format -- so only altfmt=0 is
            # checked here, which every cell has.
            mx_legal_cell(self.w, self.sew, 0)
            if self.tload != "op":
                raise ValueError(
                    f"kind='mx' with tload={self.tload!r}: the transposing "
                    f"pair is tested at kind='int', so that a layout "
                    f"failure stays separable from an arithmetic one")
            # The microscaling legality rule needs LAMBDA and LMUL, which
            # the encoding-map table does not carry.  bs=0 (block size 32)
            # is the unconditional case; bs=1 adds W*LMUL <= SEW and the
            # generator checks that per program.
            mx_check_legality(self.w, self.lmul, self.sew, self.lam, 0)
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
            if self.eew_ab < 8 and self.kind not in ("mx", "fpw"):
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
        # kind='fpw' accumulators are not binary<SEW> -- vfwmmacc.vv at
        # SEW=8 accumulates in E4M3 or E5M2 -- so the round-seven clause
        # prints the cell rather than a width.  Every pre-round-seven kind
        # keeps its exact previous string, which is what the byte-identical
        # regression over rounds one to six rests on.
        if self.kind == "fpw":
            fp = f" FPW=W{self.w}/SEW{self.sew}"
        else:
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

#: Round six needs a *second* key, because the accumulator format is no
#: longer a function of the storage width: binary16 and bfloat16 are both 16
#: bits wide and differ in where the exponent/significand split falls
#: (spec 1399-1410 -- binary16 p=11, BFloat16 p=8).  vtype.altfmt selects
#: between them (spec 1094-1106), so the Sail carries ``fmt_C`` as a value
#: distinct from ``EEW_C`` and this module now does too.
#:
#: ``fp_fields`` and everything built on it take an optional ``fmt`` naming a
#: row of this table.  When ``fmt`` is None the old width-keyed lookup runs
#: unchanged, so every round one to five call site keeps its exact previous
#: behaviour -- including the ValueError that :func:`fp_fields` raises for an
#: unmodelled width.  That is deliberate: those tiers' generated programs are
#: required to stay byte-identical across rounds.
#:
#: name -> (storage width, exponent bits, significand bits incl. hidden bit)
FP_FORMATS_BY_NAME = {
    "binary16": (16, 5, 11),
    "bfloat16": (16, 8, 8),
    "binary32": (32, 8, 24),
    "binary64": (64, 11, 53),
}

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


#: Round seven's three widening floating-point multiply-accumulates need
#: three element formats that this module has never modelled: OFP8 (E4M3 and
#: E5M2) and OFP4 (E2M1).  They are declared here, and deliberately left
#: *unpopulated*, because the IME specification does not define them.
#:
#: Spec 1908-1913 cites the OCP Microscaling Formats (MX) v1.0 specification
#: normatively instead of restating it; spec 1400-1410 gives only the
#: significand width p (E2M1 p=2, E4M3 p=4, E5M2 p=3); spec 1127-1140 and
#: 1098-1108 name the formats without defining a single bit; and Sail
#: 4831-4838 leaves ``fp_is_NaN``, ``fp_defaultNaN`` and ``fp_zero`` as
#: undefined helpers that defer to that definition.  Contrast E8M0, which
#: spec 1989-2026 restates in full -- which is exactly why round six needed
#: no external document and round seven does.
#:
#: What is undecidable without the documents, and would corrupt this judge
#: silently rather than loudly:
#:
#: * E4M3 has no infinity encoding, so RNE overflow to an E4M3 accumulator
#:   cannot go to infinity the way :func:`fp_round` does at line ~1255.  The
#:   pattern that would be written is an E4M3 NaN.  vfwmmacc.vv at SEW=8 has
#:   E4M3 as a legal *accumulator* format (spec 1098-1099), so this is not a
#:   corner: it is every SEW=8 geometry in the round.
#: * E4M3's NaN set is not "exponent all ones and significand nonzero", so
#:   :func:`fp_is_nan` is wrong for it by construction, and the E4M3
#:   ``fp_defaultNaN`` bit pattern is not derivable from anything here.
#: * E2M1 has neither infinities nor NaNs, so its overflow rule is spec text
#:   rather than an IEEE consequence.
#: * Spec 2010-2012 describes E8M0 scale conversion as yielding "the largest
#:   finite positive value, or +inf", language that presumes an
#:   infinity-capable fmt_C.  Resolving it for an E4M3 or E5M2 accumulator
#:   needs the OFP8 definition.
#:
#: Two documents are required, not one: MX v1.0 defines E2M1, the block
#: structure and E8M0, but for E4M3/E5M2 it refers to the separate OCP 8-bit
#: Floating Point (OFP8) v1.0 specification.  Drop both into
#: titan/ime-spec/ and populate the rows below from them.
#:
#: Secondary sources are explicitly *not* an acceptable substitute.  They
#: disagree with each other in ways that are invisible unless you already
#: know the answer: microsoft/microxcaling reports E4M3 as having five
#: mantissa bits and E2M1 as having three, because it carries an internal
#: ``mbits = m + 2`` convention.  A judge built from that would encode a
#: wrong format and then attribute its own error to the RTL.
#:
#: name -> the reason it is not yet modelled.  Keys, not values, are the
#: contract: :func:`fp_fields` consults this before
#: :data:`FP_FORMATS_BY_NAME` so that a round-seven call site asking for a
#: pending format gets a precise diagnosis instead of a KeyError that reads
#: like a typo.
OCP_PENDING_FORMATS = {
    "e2m1": "OFP4 E2M1: needs OCP Microscaling Formats (MX) v1.0 "
            "(no Inf, no NaN; overflow rule is spec text)",
}

# --- round eight: OFP8 (E4M3 / E5M2) from the OCP OFP8 v1.0 document ------
#
# Round eight obtained ONE of the two documents named above: "OCP 8-bit
# Floating Point Specification (OFP8) Revision 1.0" (approved 2023-06-20;
# PDF at titan/ocp-spec/, text extracted to specs/ime/ocp-ofp8-v1.0.txt for
# the agents).  OCP MX v1.0, which defines E2M1, is still NOT on disk, so
# e2m1 alone stays in OCP_PENDING_FORMATS and every E2M1 cell stays
# unsupported.  Every rule below is quoted from the OFP8 PDF by page
# ("OFP8 p.N"; PDF page numbers, which equal the printed page numbers):
#
#   * OFP8 p.11 (sec. 4.2): "E4M3: ... 1 sign bit, 4 biased exponent bits, 3
#     mantissa bits, and an exponent bias of 7."  "E5M2: ... 1 sign bit, 5
#     biased exponent bits, 2 mantissa bits, and an exponent bias of 15."
#     "max_E4M3 ... 448", "max_E5M2 ... 57,344".
#   * OFP8 p.12 (sec. 5.1): normal v = (-1)^S x 2^(E-bias) x (1 + 2^-m x M);
#     subnormal (E = 0, M > 0) v = (-1)^S x 2^(1-bias) x (0 + 2^-m x M).
#     "The E5M2 format represents infinities and NaNs.  Interpretation of the
#     three mantissa values for NaNs is not defined.  The E4M3 format does
#     not represent infinities and uses only two bit patterns for NaN (a
#     single mantissa-exponent bit pattern but allowing both values of the
#     sign bit) in order to increase emax to 8".
#   * OFP8 p.13 Table 1: bias 7 / 15, emax 8 / 15, emin -6 / -14.  Table 2:
#     E4M3 Inf N/A, NaN S.1111.111, zeros S.0000.000, max normal S.1111.110
#     = +-448, min normal S.0001.000 = 2^-6, max subnormal S.0000.111 =
#     0.875 x 2^-6, min subnormal S.0000.001 = 2^-9.  E5M2 Inf S.11111.00,
#     NaN S.11111.{01,10,11}, max normal S.11110.11 = +-57344, min normal
#     2^-14, max subnormal 0.75 x 2^-14, min subnormal 2^-16.
#   * OFP8 p.14-15 (sec. 5.2.1, Table 3): conversion TO OFP8 requires round
#     to nearest even and BOTH a saturating and a non-saturating mode.  After
#     rounding, a magnitude above max: SAT -> max normal; NONSAT -> E4M3 NaN,
#     E5M2 infinity.  +-Inf source: SAT -> +-max, NONSAT -> E4M3 NaN, E5M2
#     +-Inf.  Below min subnormal -> signed zero.  "Conversion from a NaN
#     produces an implementation-defined OFP8 NaN"; NaN sign bit is left to
#     the implementation.
#
# IME side (integrated-matrix-v0.9.0.adoc):
#
#   * 1133-1140: altfmt_A / altfmt_B = 0 -> E4M3, = 1 -> E5M2 at 8-bit input
#     width, independently per operand; 1445-1451: all four combinations,
#     mixed included, are covered by the one OFP8 extension per output.
#   * 1096-1099: vtype.altfmt x vsew=000 selects an E4M3 / E5M2 *accumulator*
#     -- IME does round TO OFP8, but only in Zvvofp8mm (vfmmacc.vv SEW=8) and
#     Zvvofp4ofp8mm (vfwmmacc.vv SEW=8, E2M1 inputs).  Neither is live in
#     round eight (see constants.ROUND_EIGHT_UNSUPPORTED), so every OFP8
#     value a round-eight program meets is an INPUT; no program rounds to
#     OFP8.  The rounding rule is still encoded, and negative-controlled, so
#     that it is not invented later under time pressure.
#   * 1400-1405: p = 4 (E4M3), 3 (E5M2) -- agrees with OCP (m + hidden bit).
#   * 1872-1878: any NaN materialised in a concrete format is that format's
#     default NaN (fp_defaultNaN), whose OFP8 bit pattern the adoc never
#     states (Sail 4831-4838 leaves the helper undefined).
#   * 1847-1850 (fp_round_to_frm): rounding to fmt_C uses frm and "may raise
#     ... overflow".  The adoc never mentions saturation; OCP requires both
#     modes and does not say which one a consumer uses.
#
# Where the two documents are silent or leave a choice, Titan discloses the
# most literal reading, as data, exactly like FP_DISCLOSURE:
#
#   overflow      "nonsat".  IME rounding is fp_round_to_frm, an IEEE-style
#                 rounding that raises the overflow flag; the only OCP mode
#                 in which overflow behaves like IEEE RNE (E5M2 -> Inf) is
#                 NONSAT, and E4M3 NONSAT gives NaN (OCP Table 3).  Unused by
#                 every live round-eight cell (no OFP8 accumulator).
#   default_nan   E4M3 0x7F (+, S.1111.111 -- its only NaN magnitude);
#                 E5M2 0x7E (+, S.11111.10 -- the RISC-V canonical-NaN shape:
#                 sign 0, exponent all ones, top mantissa bit set).  OCP
#                 leaves both implementation-defined.  Unused by every live
#                 round-eight cell (a NaN result is materialised in fmt_C,
#                 which is IEEE binary16/bfloat16/32/64 there).
#   frm           Only RNE (frm=0) is modelled for an OFP8 *destination*: OCP
#                 requires RNE only ("may implement additional rounding
#                 modes").  Decoding an OFP8 input involves no rounding.
#   snan          OCP p.12: E5M2 NaN mantissa interpretation "is not defined"
#                 and E4M3 has one NaN pattern, so no OFP8 NaN is signalling.
#                 This matters only for fflags, which no program checks.
OFP8_DISCLOSURE = {
    "overflow": "nonsat",
    "default_nan": {"e4m3": 0x7F, "e5m2": 0x7E},
    "frm": "rne-only",
    "snan": "none",
}


class OCPSpecUnavailable(NotImplementedError):
    """Raised where a decision needs a document that is not on disk.

    A distinct type, not a bare NotImplementedError, so that a caller can
    tell "this harness has not implemented X" from "this harness refuses to
    guess X".  The second is a deliberate, recoverable state; the first is a
    bug.  Round seven's generators are expected to let this propagate rather
    than catch it and emit a reduced-coverage suite.
    """


@dataclass(frozen=True)
class FpFormat:
    """A floating-point element format, as data rather than as code.

    Round seven's nine encoding-map cells span seven formats, three of which
    (E4M3, E5M2, E2M1) are not IEEE-shaped and are not yet definable.  The
    generators are written against *this* descriptor rather than against
    ``fp_fields`` so that adding those three is filling in a table row, not
    rewriting a code path -- which is the difference between a second pass
    that is thin and one that rediscovers every decision.

    Every field that an OFP format would make non-IEEE is carried here
    explicitly, even though all three currently-resolved widening cells have
    the IEEE answer for all of them:

    ``has_inf``
        Whether the format encodes infinities at all.  E4M3 and E2M1 do not.
    ``nan_rule``
        How NaNs are encoded.  ``"ieee"`` is exponent-all-ones with a nonzero
        significand; the OFP8 E4M3 rule is a *single* pattern and E2M1 has no
        NaN at all, so this cannot be a predicate hard-coded in one function.
    ``overflow``
        What a rounding that exceeds the format's range produces.  ``"inf"``
        is the IEEE answer under RNE.  A format with no infinity cannot give
        that answer, and which answer it does give is spec text.

    ``None`` in any of those three means "the document that decides this is
    not on disk".  :func:`fp_format` refuses such a descriptor by raising
    :class:`OCPSpecUnavailable`, so a generator cannot reach a half-defined
    format by accident -- it has to be handed one, and it cannot be.
    """
    name: str
    width: int
    ebits: int
    prec: int                      # significand bits, hidden bit included
    has_inf: Optional[bool] = None
    nan_rule: Optional[str] = None
    overflow: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return None not in (self.has_inf, self.nan_rule, self.overflow)

    @property
    def bias(self) -> int:
        return (1 << (self.ebits - 1)) - 1

    @property
    def emax(self) -> int:
        """Largest biased exponent field value (the all-ones pattern)."""
        return (1 << self.ebits) - 1

    @property
    def sub_byte(self) -> bool:
        """True where two elements share a byte (spec 1207-1219).

        No currently-resolved format is sub-byte -- E2M1 is the only one --
        but the packing path is written against this property rather than
        against ``width == 4`` so that the OFP4 cells do not arrive needing a
        new branch.  See :func:`fpw_pack_elements`.
        """
        return self.width < 8


#: The seven element formats round seven's encoding map can name.
#:
#: The four resolved rows restate FP_FORMATS_BY_NAME and are checked against
#: it in :func:`check_round_seven_format_table`, so the two cannot drift.
#: The three unresolved rows carry their width and significand width -- which
#: spec 1400-1410 does state -- and leave every behavioural field None.
#:
#: When the OCP documents land, this table is the edit: three rows gain
#: ``has_inf`` / ``nan_rule`` / ``overflow``, OCP_PENDING_FORMATS empties,
#: and nothing else in the round-seven path changes.  E4M3's row is the one
#: to be careful with: ``has_inf=False`` and an ``overflow`` that is not
#: ``"inf"``.
FP_FORMAT_TABLE = {
    "binary16": FpFormat("binary16", 16, 5, 11, True, "ieee", "inf"),
    "bfloat16": FpFormat("bfloat16", 16, 8, 8, True, "ieee", "inf"),
    "binary32": FpFormat("binary32", 32, 8, 24, True, "ieee", "inf"),
    "binary64": FpFormat("binary64", 64, 11, 53, True, "ieee", "inf"),
    # Round eight: OFP8 from OCP OFP8 v1.0 (see OFP8_DISCLOSURE for every
    # citation).  E4M3: no infinity (OFP8 p.12), NaN only S.1111.111 (Table
    # 2), so S.1111.000-110 are ordinary normals up to 448.  E5M2 is
    # IEEE-shaped (Table 2: Inf S.11111.00, NaN S.11111.{01,10,11}).  The
    # ``overflow`` field is the disclosed non-saturating mode (OCP Table 3
    # NONSAT column): E4M3 -> NaN, E5M2 -> Inf.
    "e4m3": FpFormat("e4m3", 8, 4, 4, False, "ocp_e4m3", "nan"),
    "e5m2": FpFormat("e5m2", 8, 5, 3, True, "ieee", "inf"),
    # E2M1: width and significand from spec 1400-1410 only; everything
    # behavioural is deliberately absent until OCP MX v1.0 is on disk.
    "e2m1": FpFormat("e2m1", 4, 2, 2),
}


def fp_format(name: str) -> FpFormat:
    """The descriptor for *name*, or raise if it is not yet definable."""
    try:
        fmt = FP_FORMAT_TABLE[name]
    except KeyError:
        raise ValueError(
            f"unknown floating-point format {name!r}; the round-seven "
            f"encoding map names {sorted(FP_FORMAT_TABLE)}") from None
    if not fmt.resolved:
        raise OCPSpecUnavailable(
            f"floating-point format {name!r} is declared but not modelled: "
            f"{OCP_PENDING_FORMATS.get(name, 'no definition on disk')}.  The "
            f"IME spec cites OCP normatively (1908-1913) rather than "
            f"restating the encoding; fill in FP_FORMAT_TABLE from the "
            f"documents, do not infer the fields from IEEE 754.")
    return fmt


def fpf_is_nan(bits: int, fmt: FpFormat) -> bool:
    """NaN predicate driven by the descriptor, not by an IEEE assumption.

    ``"ocp_e4m3"`` is OCP OFP8 p.12 / Table 2 (p.13): E4M3's NaN is the
    single exponent-and-mantissa pattern S.1111.111, "allowing both values of
    the sign bit".  S.1111.000 through S.1111.110 are ordinary normals
    (256 .. 448), so the IEEE predicate would misread six finite magnitudes
    per sign as NaN -- the trap round eight's negative controls pin.
    """
    if fmt.nan_rule == "ieee":
        return ((bits >> (fmt.prec - 1)) & fmt.emax == fmt.emax
                and bool(bits & ((1 << (fmt.prec - 1)) - 1)))
    if fmt.nan_rule == "ocp_e4m3":
        mag_mask = (1 << (fmt.width - 1)) - 1
        return bits & mag_mask == mag_mask
    if fmt.nan_rule == "none":
        return False
    raise OCPSpecUnavailable(
        f"NaN rule {fmt.nan_rule!r} for {fmt.name} is not modelled")


def fpf_is_inf(bits: int, fmt: FpFormat) -> bool:
    if not fmt.has_inf:
        return False
    return ((bits >> (fmt.prec - 1)) & fmt.emax == fmt.emax
            and not bits & ((1 << (fmt.prec - 1)) - 1))


def fpf_default_nan(fmt: FpFormat) -> int:
    """Sail ``fp_defaultNaN`` for *fmt* (spec 1872-1878, Sail 4834-4838).

    The four IEEE-table formats delegate to :func:`fp_default_nan`, so their
    patterns are byte-identical to rounds four to seven.  The OFP8 patterns
    are Titan disclosures (:data:`OFP8_DISCLOSURE`): OCP OFP8 p.14 makes the
    NaN produced by a conversion "implementation-defined".
    """
    if fmt.name in FP_FORMATS_BY_NAME:
        return fp_default_nan(fmt.width, fmt.name)
    try:
        return OFP8_DISCLOSURE["default_nan"][fmt.name]
    except KeyError:
        raise OCPSpecUnavailable(
            f"no default NaN is defined for {fmt.name}") from None


def fpf_max_finite(fmt: FpFormat) -> int:
    """Bit pattern of the largest finite positive value of *fmt*.

    IEEE-shaped: biased exponent emax-1, mantissa all ones (E5M2: 0x7B =
    57344, OCP Table 2).  ``"ocp_e4m3"``: exponent all ones, mantissa all
    ones but the NaN pattern, i.e. S.1111.110 = 448 (OCP Table 2).
    """
    mbits = fmt.prec - 1
    if fmt.nan_rule == "ocp_e4m3":
        return (fmt.emax << mbits) | ((1 << mbits) - 2)
    if fmt.has_inf:
        return ((fmt.emax - 1) << mbits) | ((1 << mbits) - 1)
    raise OCPSpecUnavailable(f"max finite of {fmt.name} is not modelled")


def _fpf_unpack_generic(bits: int, fmt: FpFormat):
    """Descriptor-driven decode, for formats outside FP_FORMATS_BY_NAME.

    OCP OFP8 p.12: normal v = (-1)^S 2^(E-bias) (1 + 2^-m M); subnormal
    (E=0, M>0) v = (-1)^S 2^(1-bias) (2^-m M); E=0, M=0 is a signed zero.
    Special encodings come from the descriptor (has_inf, nan_rule), never
    from an assumption that the top binade is reserved: for E4M3 it is not.
    """
    mbits = fmt.prec - 1
    sign = (bits >> (fmt.width - 1)) & 1
    exp = (bits >> mbits) & fmt.emax
    frac = bits & ((1 << mbits) - 1)
    if fpf_is_nan(bits, fmt):
        return "nan", sign, Fraction(0)
    if fpf_is_inf(bits, fmt):
        return "inf", sign, Fraction(0)
    if fmt.nan_rule == "ieee" and exp == fmt.emax:
        raise AssertionError(f"{fmt.name} 0x{bits:x}: unclassified top binade")
    if exp == 0:
        return "num", sign, Fraction(frac, 1 << mbits) * _pow2(1 - fmt.bias)
    return ("num", sign,
            Fraction((1 << mbits) | frac, 1 << mbits) * _pow2(exp - fmt.bias))


def fpf_unpack(bits: int, fmt: FpFormat):
    """``(kind, sign, Fraction)`` for *bits* read as *fmt*.

    Delegates to :func:`fp_unpack` for the four IEEE formats that rounds four
    to seven already exercise, so their decode is byte-for-byte unchanged.
    The OFP8 formats go through :func:`_fpf_unpack_generic`, which reads
    every behavioural fact from the descriptor; it is checked against an
    independent 256-code table built from the OCP formulas in
    :func:`check_round_eight_ofp8_decode`, and against fp_unpack on all
    65536 binary16 codes so the generic path is known IEEE-correct too.
    """
    if not fmt.resolved:
        raise OCPSpecUnavailable(f"{fmt.name} is not modelled")
    if fmt.name in FP_FORMATS_BY_NAME:
        if fmt.nan_rule != "ieee" or not fmt.has_inf:
            raise OCPSpecUnavailable(
                f"{fmt.name} is not IEEE-shaped; its decode is not modelled")
        return fp_unpack(bits, fmt.width, fmt.name)
    return _fpf_unpack_generic(bits, fmt)


def _fpf_round_generic(value: "Fraction", fmt: FpFormat) -> int:
    """RNE into an OFP8 destination, per OCP OFP8 sec. 5.2.1 (p.14).

    "Conversion ... first applies rounding to reduce the mantissa bit count
    to that of the destination OFP8 format.  After that, if the rounded
    magnitude is above the maximum destination magnitude" the overflow rule
    applies (``fmt.overflow``: ``"saturate"`` -> +-max normal, ``"nan"`` ->
    NaN, ``"inf"`` -> +-Inf); if it is "below the minimum subnormal number
    magnitude: generate a corresponding zero" -- which is what RNE into the
    subnormal range already does.  The sign is preserved except for NaN.
    """
    sign = 1 if value < 0 else 0
    mag = -value if sign else value
    sbit = sign << (fmt.width - 1)
    mbits = fmt.prec - 1
    if mag == 0:
        return sbit
    e = max(_floor_log2(mag), 1 - fmt.bias)
    scaled = mag / _pow2(e - mbits)
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1):
        q += 1
    if q >> fmt.prec:
        q >>= 1
        e += 1
    rounded = Fraction(q) * _pow2(e - mbits)
    max_bits = fpf_max_finite(fmt)
    _k, _s, max_mag = _fpf_unpack_generic(max_bits, fmt)
    if rounded > max_mag:
        return _fpf_overflow(sign, fmt)
    if q >> mbits == 0:                        # subnormal (or zero)
        return sbit | q
    return sbit | ((e + fmt.bias) << mbits) | (q & ((1 << mbits) - 1))


def _fpf_overflow(sign: int, fmt: FpFormat) -> int:
    """OCP Table 3 (p.15), the "greater than max OFP8 magnitude" row."""
    sbit = sign << (fmt.width - 1)
    if fmt.overflow == "saturate":
        return sbit | fpf_max_finite(fmt)
    if fmt.overflow == "inf":
        if not fmt.has_inf:
            raise OCPSpecUnavailable(f"{fmt.name} has no infinity")
        return sbit | (fmt.emax << (fmt.prec - 1))
    if fmt.overflow == "nan":
        return fpf_default_nan(fmt)
    raise OCPSpecUnavailable(
        f"{fmt.name} overflow rule {fmt.overflow!r} is not modelled")


def fpf_round(value: "Fraction", fmt: FpFormat, frm: int = 0) -> int:
    """Round an exact value into *fmt*, per the descriptor's overflow rule."""
    if frm != FP_FRM:
        raise ValueError(
            f"frm={frm} is not modelled; round seven generates RNE only "
            f"(FP_FRM), matching spec 1495 and the FP_DISCLOSURE tuple")
    if fmt.name not in FP_FORMATS_BY_NAME:
        # Round eight: OFP8 destinations (OFP8_DISCLOSURE).  No live cell
        # reaches this -- every round-eight accumulator is IEEE -- but the
        # exact-case constructor and the negative controls do.
        if not fmt.resolved:
            raise OCPSpecUnavailable(f"{fmt.name} is not modelled")
        return _fpf_round_generic(value, fmt)
    if fmt.overflow != "inf":
        raise OCPSpecUnavailable(
            f"{fmt.name} overflow rule {fmt.overflow!r} is not modelled; a "
            f"format without infinities cannot use the IEEE overflow path")
    # fp_round takes a nonnegative magnitude plus a sign bit; round seven
    # works in signed Fractions throughout, so the split happens here and
    # nowhere else.  A negative value that rounds to zero must give -0.0,
    # which is why the sign is taken before the magnitude and not from the
    # rounded result.
    sign = 1 if value < 0 else 0
    return fp_round(sign, -value if sign else value, fmt.width, fmt.name)


def fp_fields(width: int, fmt: Optional[str] = None):
    """(exponent bits, significand bits, bias, max biased exponent).

    *fmt* names a row of :data:`FP_FORMATS_BY_NAME` and is what round six
    onwards passes; the storage width must agree with the name's, which is
    checked rather than assumed so that a binary16/bfloat16 mix-up cannot
    pass silently.  ``fmt=None`` keeps the round one to five width-keyed
    lookup byte-for-byte, including the ValueError below.
    """
    if fmt is not None:
        if fmt in OCP_PENDING_FORMATS:
            raise OCPSpecUnavailable(
                f"floating-point format {fmt!r} is declared but not "
                f"modelled: {OCP_PENDING_FORMATS[fmt]}.  The IME spec cites "
                f"OCP normatively (1908-1913) rather than restating the "
                f"encoding, so there is nothing here to derive it from; see "
                f"OCP_PENDING_FORMATS.")
        if fmt in OFP8_DISCLOSURE["default_nan"]:
            # Round eight: OFP8 is modelled by the descriptor helpers
            # (fpf_*) only.  This IEEE-shaped path would misread E4M3's top
            # binade, so it refuses rather than half-working.
            raise ValueError(
                f"{fmt!r} is an OCP OFP8 format; use the FpFormat helpers "
                f"(fpf_unpack / fpf_round / fpf_default_nan), not fp_fields")
        try:
            fwidth, ebits, prec = FP_FORMATS_BY_NAME[fmt]
        except KeyError:
            raise ValueError(
                f"unknown floating-point format {fmt!r}; modelled formats "
                f"are {sorted(FP_FORMATS_BY_NAME)}") from None
        if fwidth != width:
            raise ValueError(
                f"format {fmt!r} is {fwidth} bits wide, called at {width}")
        return ebits, prec, (1 << (ebits - 1)) - 1, (1 << ebits) - 1
    try:
        ebits, prec = FP_FORMATS[width]
    except KeyError:
        raise ValueError(
            f"no IEEE format modelled at {width} bits; round four models "
            f"{sorted(FP_FORMATS)} only (binary16/bfloat16/OFP8/OFP4 need "
            f"the vtype.altfmt_A / altfmt_B decode -- see round4_design.md)"
        ) from None
    return ebits, prec, (1 << (ebits - 1)) - 1, (1 << ebits) - 1


def fp_is_nan(bits: int, width: int, fmt: Optional[str] = None) -> bool:
    ebits, prec, _bias, emax = fp_fields(width, fmt)
    return (bits >> (prec - 1)) & emax == emax and bits & ((1 << (prec - 1)) - 1)


def fp_is_inf(bits: int, width: int, fmt: Optional[str] = None) -> bool:
    ebits, prec, _bias, emax = fp_fields(width, fmt)
    return ((bits >> (prec - 1)) & emax == emax
            and not bits & ((1 << (prec - 1)) - 1))


def fp_default_nan(width: int, fmt: Optional[str] = None) -> int:
    """Sail ``fp_defaultNaN``: the canonical quiet NaN for *width*.

    Spec 1880-1886: "NaN payloads are not architecturally significant for IME
    floating-point operations ... whenever a shared floating-point helper
    materializes a NaN result in a concrete floating-point format, it shall
    return the default canonical NaN for that format".  So a NaN result is a
    single fixed bit pattern and compares bit-for-bit like any other value --
    which is why the harness needs no NaN-aware comparison.
    """
    ebits, prec, _bias, emax = fp_fields(width, fmt)
    return (emax << (prec - 1)) | (1 << (prec - 2))


def fp_unpack(bits: int, width: int, fmt: Optional[str] = None):
    """(kind, sign, value) with *value* an exact :class:`Fraction`.

    ``kind`` is ``"nan"``, ``"inf"`` or ``"num"``; ``sign`` is 0 or 1; for
    ``"num"`` the magnitude is exact and zero is representable (sign carries
    the -0.0 / +0.0 distinction, which the bitwise compare cares about).
    """
    ebits, prec, bias, emax = fp_fields(width, fmt)
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


def fp_round(sign: int, value: Fraction, width: int,
             fmt: Optional[str] = None) -> int:
    """Round an exact nonnegative rational to *width* bits, RNE.

    This is Sail ``fp_round_to_frm`` at ``frm = RNE``.  It implements the
    single rounding point: overflow goes to infinity (RNE overflows away
    from zero), underflow rounds into the subnormal range with the same
    ties-to-even rule, and the sign is carried through unchanged so that a
    rounded-to-zero negative result is -0.0.

    *value* must be the exact magnitude; the caller supplies the sign.
    """
    assert value >= 0, value
    ebits, prec, bias, emax = fp_fields(width, fmt)
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


def fp_mul(a: int, b: int, width: int, fmt: Optional[str] = None) -> int:
    """Sail ``fp_mul``: exact product, one rounding to *width* under frm.

    Special values follow IEEE-754 and spec 1841-1848: an sNaN or qNaN
    operand, or 0 x inf, produces the *canonical* NaN -- never a propagated
    payload.
    """
    ka, sa, va = fp_unpack(a, width, fmt)
    kb, sb, vb = fp_unpack(b, width, fmt)
    sign = sa ^ sb
    if ka == "nan" or kb == "nan":
        return fp_default_nan(width, fmt)
    if ka == "inf" or kb == "inf":
        if (ka == "num" and va == 0) or (kb == "num" and vb == 0):
            return fp_default_nan(width, fmt)  # 0 x inf: invalid
        _, prec, _, emax = fp_fields(width, fmt)
        return (sign << (width - 1)) | (emax << (prec - 1))
    return fp_round(sign, va * vb, width, fmt)


def fp_add(a: int, b: int, width: int, fmt: Optional[str] = None) -> int:
    """Sail ``fp_add``: exact sum, one rounding to *width* under frm.

    inf + (-inf) is invalid and yields the canonical NaN (spec 1850-1856).
    The sign of an exact zero result is IEEE's: -0.0 only when both addends
    were -0.0 (RNE never produces -0.0 from a cancelling sum).
    """
    ka, sa, va = fp_unpack(a, width, fmt)
    kb, sb, vb = fp_unpack(b, width, fmt)
    if ka == "nan" or kb == "nan":
        return fp_default_nan(width, fmt)
    _, prec, _, emax = fp_fields(width, fmt)
    if ka == "inf" or kb == "inf":
        if ka == "inf" and kb == "inf" and sa != sb:
            return fp_default_nan(width, fmt)
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
    return fp_round(1 if total < 0 else 0, abs(total), width, fmt)


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
# round six: the MX integer-input, FP-accumulate family
# ---------------------------------------------------------------------------
#
# vfwimmacc.vv / vfqimmacc.vv / vf8wimmacc.vv.  Sail `int_scaled_gemm`
# (spec 5373-5410) is the whole architecture, and two things about it decide
# the shape of everything below.
#
# 1. It never calls get_fp_grouping / get_fp_psm / get_fp_rnd, and it has no
#    `G` legality check -- contrast `fp_gemm` at 5238-5242.  Spec 1283-1287
#    and 1645-1652 say so in prose: this path "uses the separately specified
#    exact integer block-dot-product path and is not governed by G, psm, or
#    rnd".  So there is no implementation disclosure to model, and no
#    `rnd`/`psm` parameter anywhere in this section.
#
# 2. Its loop nest is j / i / s only -- there is *no* LMUL step loop.  The
#    block-and-step intersection, `subdot_lo/hi` and the shortened-group
#    machinery of spec 2030-2060 belong to `fp_scaled_gemm`, not here;
#    `int_block_dot` is handed the whole block interval.  Copying the
#    group-A intersection logic into this model would be a judge bug that
#    presents as an RTL bug, so it is called out here and asserted in
#    :func:`check_round_six_block_intervals`.

#: Scale format.  Spec 2128: every microscaling extension in v0.9.0 uses
#: E8M0, so sw = 8 and the *pair* width pw = 2 * sw = 16.
MX_SCALE_WIDTH = 8
MX_PAIR_WIDTH = 2 * MX_SCALE_WIDTH

#: E8M0, spec 1990-1993, normatively referencing the OCP Microscaling
#: Formats (MX) v1.0 specification: an 8-bit exponent-only format, bias 127,
#: representing 2**-127 .. 2**127, with 0xFF encoding NaN.
#:
#: Read the value range carefully, because it is what makes this format
#: unlike every IEEE one modelled above: there is no zero encoding, no
#: infinity encoding and no subnormal encoding.  Byte 0x00 is 2**-127 (a
#: perfectly ordinary finite value), byte 0xFE is 2**127, and 0xFF is the
#: *only* non-finite code.  A model that treats 0x00 as zero -- the obvious
#: IEEE reflex -- is wrong, and wrong in a direction that silently produces
#: plausible numbers.
MX_E8M0_BIAS = 127
MX_E8M0_NAN = 0xFF

#: (W, SEW) -> (integer input width, {altfmt: accumulator format name}).
#: The normative table is `tbl-intmx-encoding-map`, spec 7469-7540; the SEW
#: guards are in each instruction's Sail (vfwimmacc 6035-6038, vfqimmacc
#: 6266-6270, vf8wimmacc 6151).  Absent keys are reserved encodings, which is
#: what :func:`mx_legal_cell` reports.
#:
#: altfmt_A and altfmt_B are *not* in this table: spec 2335-2342 makes MXINT
#: signed unconditionally and the Sail raises Illegal_Instruction when either
#: is 1 (6045-6046 / 6162-6163 / 6277-6278).  Only vtype.altfmt, which picks
#: the C accumulator format, is free -- and only where the table gives two
#: rows; at SEW 32 and 64 altfmt=1 is reserved.
MX_CELLS = {
    (2, 16): (8, {0: "binary16", 1: "bfloat16"}),   # vfwimmacc,  MXINT8
    (4, 16): (4, {0: "binary16", 1: "bfloat16"}),   # vfqimmacc,  MXINT4
    (4, 32): (8, {0: "binary32"}),                  # vfqimmacc,  MXINT8
    (8, 32): (4, {0: "binary32"}),                  # vf8wimmacc, MXINT4
    (8, 64): (8, {0: "binary64"}),                  # vf8wimmacc, MXINT8
}


def mx_legal_cell(w: int, sew: int, altfmt: int = 0):
    """``(input width, accumulator format)`` for a legal cell, else raise.

    The single place this module decides whether an (W, SEW, altfmt) triple
    is architecture or is a reserved encoding.
    """
    try:
        ewidth, fmts = MX_CELLS[(w, sew)]
    except KeyError:
        raise ValueError(
            f"W={w} SEW={sew} is a reserved encoding for the MX integer-input "
            f"family; legal (W, SEW) cells are {sorted(MX_CELLS)} "
            f"(spec 7469-7540)") from None
    try:
        return ewidth, fmts[altfmt]
    except KeyError:
        raise ValueError(
            f"W={w} SEW={sew} altfmt={altfmt} is reserved; the C accumulator "
            f"format table offers {sorted(fmts)} here (spec 7469-7540)"
        ) from None


def mx_block_size(bs: int) -> int:
    """Spec 1166-1172 and Sail 5379: bs=0 -> 32 elements, bs=1 -> 16."""
    if bs not in (0, 1):
        raise ValueError(f"bs is one bit, got {bs}")
    return 16 if bs else 32


def mx_check_legality(w: int, lmul: int, sew: int, lam: int,
                      bs: int = 0) -> None:
    """Sail ``check_microscaling_legality`` (5151-5158), verbatim::

        if EEW_C * lambda < pw then return Illegal_Instruction();
        if vtype[bs] == 0b1 & (W * LMUL > EEW_C) then return Illegal_Instruction()

    Raises :class:`ValueError` where the Sail raises Illegal_Instruction, so
    the generator can enumerate the negative cases from the same rule that
    decides the positive ones.
    """
    if sew * lam < MX_PAIR_WIDTH:
        raise ValueError(
            f"SEW*LAMBDA = {sew * lam} < pw = {MX_PAIR_WIDTH}: the paired "
            f"scale elements do not fit in v0 (spec 2343-2347, Sail 5155)")
    if bs == 1 and w * lmul > sew:
        raise ValueError(
            f"bs=1 requires W*LMUL <= SEW, got {w}*{lmul} > {sew} "
            f"(Sail 5156)")


def mx_scale_stride(sew: int, lam: int) -> int:
    """Row stride ``R = LAMBDA * SEW / pw`` (spec 2139, Sail 5381)."""
    r, rem = divmod(lam * sew, MX_PAIR_WIDTH)
    assert rem == 0, (sew, lam)
    return r


def mx_block_count(k_eff: int, block_size: int) -> int:
    """``S_blocks = ceil(K_eff / block_size)`` (spec 2166, Sail 5380)."""
    return -(-k_eff // block_size)


def mx_block_interval(s: int, block_size: int, k_eff: int):
    """``(k_lo, k_hi)`` for block *s*, Sail 5396-5397.

    ``k_hi = min(k_lo + block_size, K_eff) - 1`` -- the final block may be
    short.  Note what is *absent*: no LMUL step is involved.  See the section
    preamble.
    """
    k_lo = s * block_size
    return k_lo, min(k_lo + block_size, k_eff) - 1


def mx_pair_index(m: int, s: int, r: int) -> int:
    """Index of the 16-bit scale-pair element in ``v0`` (spec 2161-2170).

    ``p = m * R + s``.  Sail ``read_block_scales`` (5110-5117) reads the A
    scale at ``i * R + s`` and the B scale at ``j * R + s`` -- the same
    function of a row/column index and a block index, out of the *same*
    register.  Getting the two factors the wrong way round (``s * R + m``) is
    the most likely implementation bug in the whole round, which is why a
    negative control sabotages exactly this.
    """
    return m * r + s


def mx_decode_scale(byte: int, width: int, fmt: str):
    """E8M0 byte -> ``(bits in fmt, is_nan)``.  Spec 2008-2019.

    "Conversion of each non-NaN E8M0 scale value to the accumulator FP format
    uses `frm`.  The converted value and accrued exception flags are those of
    a normal conversion to `fmt_C` under that rounding mode.  Depending on
    `fmt_C`, `frm`, and the scale value, range conversion may produce a
    finite normal or subnormal value, +0, the largest finite positive value,
    or +inf."

    So this is *not* an exponent splice: 2**127 is unrepresentable in
    binary16 and must go through the ordinary rounding path (spec 2013-2016
    gives exactly that example).  :func:`fp_round` at RNE overflows to
    infinity and underflows into the subnormals, which is the frm=RNE column
    of that sentence.
    """
    if not 0 <= byte <= 0xFF:
        raise ValueError(f"E8M0 scale is one byte, got {byte}")
    if byte == MX_E8M0_NAN:
        return fp_default_nan(width, fmt), True
    return fp_round(0, _pow2(byte - MX_E8M0_BIAS), width, fmt), False


def mx_block_scale(scale_a: int, scale_b: int, width: int, fmt: str):
    """Sail ``read_block_scales`` (5097-5122) -> ``(blk_scale, is_nan)``.

    Both bytes are decoded into ``fmt_C`` and multiplied there::

        let blk_scale : bits(EEW_C) = fp_mul(sA, sB, EEW_C, fmt_C, rm);
        (blk_scale, fp_is_NaN(blk_scale, EEW_C, fmt_C))

    The NaN test is on the *product*, not on either byte, and that
    distinction carries real cases: spec 2021-2024 says that if one converted
    scale is +0 and the other +inf the paired multiplication "produces the
    default NaN and raises the invalid-operation flag, even though both
    encoded E8M0 scales are finite".  A model that only checks for 0xFF
    misses those and calls a correct implementation wrong.
    """
    sa, nan_a = mx_decode_scale(scale_a, width, fmt)
    sb, nan_b = mx_decode_scale(scale_b, width, fmt)
    blk = fp_mul(sa, sb, width, fmt)
    del nan_a, nan_b          # subsumed: a NaN operand makes fp_mul NaN
    return blk, fp_is_nan(blk, width, fmt)


def mx_int_block_dot(a: Matrix, b: Matrix, i: int, j: int,
                     k_lo: int, k_hi: int) -> int:
    """Sail ``int_block_dot`` (5128-5147): an exact, unbounded integer.

    Both operands are read *signed* -- the Sail passes ``signed_A`` and
    ``signed_B`` as the literals ``true`` at 5397, independent of
    altfmt_A/altfmt_B (which are forced to 0 for this family anyway).

    There is no modular reduction here and none afterwards: unlike
    ``int_gemm``, whose whole sum is wrapped once at EEW_C, this value feeds
    ``int_to_fp`` as a mathematical integer.  :func:`reference_gemm`'s
    ``_wrap`` has no counterpart in this path.
    """
    return sum(a[i][k] * b[j][k] for k in range(k_lo, k_hi + 1))


def mx_int_to_fp(value: int, width: int, fmt: str) -> int:
    """Sail ``int_to_fp``: one correctly-rounded conversion under frm."""
    return fp_round(1 if value < 0 else 0, Fraction(abs(value)), width, fmt)


def fp_one(width: int, fmt: str) -> int:
    """Sail ``fp_one``: +1.0 in *fmt*."""
    _ebits, prec, bias, _emax = fp_fields(width, fmt)
    return bias << (prec - 1)


def int_scaled_gemm_reference(a: Matrix, b: Matrix, c: Matrix,
                              scales_a: Sequence[Sequence[int]],
                              scales_b: Sequence[Sequence[int]],
                              geom: "TileGeometry", *,
                              bs: int = 0, altfmt: int = 0) -> Matrix:
    """Sail ``int_scaled_gemm`` (5373-5410), transcribed.

    *a* is the M x K_eff A tile and *b* the N x K_eff B tile, both as signed
    Python ints of the cell's integer input width (the caller sign-extends,
    exactly as for :func:`reference_gemm` at W > 1).  *c* is the M x N
    accumulator tile as raw ``fmt_C`` bit patterns.

    *scales_a* is indexed ``[i][s]`` and *scales_b* ``[j][s]``, each an E8M0
    byte.  They are separate arguments here although the architecture folds
    both into one ``v0``: the folding is a *layout* question that
    :func:`mx_pair_index` owns and that ime_tests materialises, and keeping
    it out of the arithmetic means a layout bug and an arithmetic bug cannot
    cancel.

    Returns the new C tile as bit patterns.  Columns j >= N are untouched
    (vta=0, spec 1811-1813).
    """
    width = geom.sew
    _ewidth, fmt = mx_legal_cell(geom.w, geom.sew, altfmt)
    mx_check_legality(geom.w, geom.lmul, geom.sew, geom.lam, bs)
    block_size = mx_block_size(bs)
    blocks = mx_block_count(geom.k_eff, block_size)

    out = [row[:] for row in c]
    for j in range(geom.n):
        for i in range(geom.m):
            acc = out[i][j]
            nan_out = False
            for s in range(blocks):
                blk_scale, is_nan = mx_block_scale(
                    scales_a[i][s], scales_b[j][s], width, fmt)
                if is_nan:
                    # Sail 5392: `break` -- this block's products and every
                    # later block contribute nothing (spec 1815-1824).
                    nan_out = True
                    break
                k_lo, k_hi = mx_block_interval(s, block_size, geom.k_eff)
                dot = mx_int_block_dot(a, b, i, j, k_lo, k_hi)
                fp_sum = mx_int_to_fp(dot, width, fmt)
                acc = fp_add(acc, fp_mul(blk_scale, fp_sum, width, fmt),
                             width, fmt)
            out[i][j] = fp_default_nan(width, fmt) if nan_out else acc
    return out


def mx_exact_dot_bound(w: int, sew: int, altfmt: int = 0,
                       bs: int = 0) -> int:
    """Largest ``|dot|`` a block can hold with the result still exact.

    Every integer with ``|x| <= 2**prec`` is representable in a format of
    ``prec`` significand bits, so ``int_to_fp`` is exact below this bound and
    the whole per-block computation is exact when the scales are 1.0.  The
    directed generator uses this to pick operand magnitudes; see
    :func:`mx_exact_operand_bound`.
    """
    _ewidth, fmt = mx_legal_cell(w, sew, altfmt)
    _ebits, prec, _bias, _emax = fp_fields(sew, fmt)
    del bs
    return 1 << prec


def mx_exact_operand_bound(w: int, sew: int, altfmt: int = 0,
                           bs: int = 0) -> int:
    """Largest ``|a|``, ``|b|`` keeping a whole block's dot product exact.

    ``|dot| <= block_size * bound**2`` must stay within
    :func:`mx_exact_dot_bound`.  Returns the largest such bound, clamped to
    the cell's integer range (Int8: 127, Int4: 7 -- the negative extreme is
    one larger in two's complement but the symmetric bound is what the
    generator wants).

    Four of the seven cells come out unclamped at the full integer range, so
    for those this is not a restriction at all; the BF16 accumulator cells
    are where it bites.  The table is in titan_runs/round6_design.md.
    """
    ewidth, _fmt = mx_legal_cell(w, sew, altfmt)
    block_size = mx_block_size(bs)
    limit = mx_exact_dot_bound(w, sew, altfmt, bs)
    native = (1 << (ewidth - 1)) - 1
    bound = 0
    while (bound + 1) <= native and block_size * (bound + 1) ** 2 <= limit:
        bound += 1
    if bound == 0:
        raise ValueError(
            f"W={w} SEW={sew} altfmt={altfmt} bs={bs}: no nonzero operand "
            f"bound keeps a block exact")
    return bound


#: W -> the SEWs the MX integer-input family is defined at, derived from
#: MX_CELLS so the two cannot drift.  Mirrors WIDENING_SEWS_BY_W.
MX_SEWS_BY_W = {w: tuple(sorted(sew for (ww, sew) in MX_CELLS if ww == w))
                for w in sorted({w for (w, _s) in MX_CELLS})}


def mx_legal_configs(vlen: int, **kwargs) -> Iterator["TileGeometry"]:
    """Every round-six geometry, in W order.

    A thin ordering over :func:`ime_legal_configs` -- the legality rules all
    live in :meth:`TileGeometry.validate`, which now knows ``kind='mx'``, so
    this adds only the per-W SEW restriction that the encoding map imposes.
    W is the outer loop for the same append-don't-interleave reason every
    other tier has.
    """
    for w in sorted(MX_SEWS_BY_W):
        yield from ime_legal_configs(vlen, sews=MX_SEWS_BY_W[w], ws=(w,),
                                     kinds=("mx",), **kwargs)


def mx_altfmts(w: int, sew: int) -> Tuple[int, ...]:
    """The accumulator-format selectors legal at this cell (spec 7469-7540)."""
    _ewidth, fmts = MX_CELLS[(w, sew)]
    return tuple(sorted(fmts))


def mx_pack_int4(values: Sequence[int]) -> List[int]:
    """Pack signed 4-bit elements two per byte, even index in the low nibble.

    Spec 1206-1219.  MXINT4 is "4-bit signed two's-complement elements"
    (spec 2337-2342), defined by this specification rather than by OCP, so
    there are no special values to model -- every one of the 16 codes is an
    ordinary integer in [-8, 7].
    """
    if len(values) % 2:
        raise ValueError("int4 elements are packed in pairs")
    out = []
    for lo, hi in zip(values[0::2], values[1::2]):
        out.append((lo & 0xF) | ((hi & 0xF) << 4))
    return out


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
# --- round seven: the widening floating-point encoding map -----------------
#
# tbl-fp-encoding-map, spec 7288-7370.  This is a *table*, not a rule, and it
# is transcribed row for row rather than generated, because two of its
# regularities are not regular:
#
#   * At (W=2, SEW=16) all eight OFP8 input pairs are legal at BOTH accumulator
#     formats (FP16 and BF16) -- sixteen rows.  At (W=4, SEW=32) and
#     (W=8, SEW=64) the same eight input pairs exist but altfmt=1 is reserved,
#     so only six rows survive (the four same-format rows lose their altfmt=1
#     partner, and the mixed rows appear at altfmt=0 only).
#   * At EEW=4, altfmt_A=1 is reserved (spec 1442-1443), so OFP4 has exactly
#     one input pair and never mixes.  At EEW>=32 altfmt_A/altfmt_B are
#     ignored entirely (spec 1490-1493), which is written here as None rather
#     than as 0 so that a generator cannot accidentally treat "ignored" as
#     "must be zero" and stop exercising the other encoding.
#
# Deriving these from a formula is how a judge acquires a rule the
# architecture does not have.  The table is ~40 lines; the formula would be a
# guess.
#
# Key:   (W, SEW) -> (EEW_AB, rows)
# Row:   (altfmt_A, altfmt_B, altfmt) -> (fmt_A, fmt_B, fmt_C, mx_allowed)
#        altfmt_A / altfmt_B of None means "ignored at this width".
#        mx_allowed is True where the row's MX cells name extensions, i.e.
#        where vm=0 is an encoding at all (spec 7270-7286).
_OFP8 = {0: "e4m3", 1: "e5m2"}
_F16 = {0: "binary16", 1: "bfloat16"}


def _ofp8_rows(fmt_c_by_altfmt, mx):
    """The eight OFP8 input pairs crossed with the legal accumulator formats."""
    rows = {}
    for a in (0, 1):
        for b in (0, 1):
            for altfmt, fmt_c in fmt_c_by_altfmt.items():
                rows[(a, b, altfmt)] = (_OFP8[a], _OFP8[b], fmt_c, mx)
    return rows


def _f16_rows(fmt_c, mx=False):
    """The four FP16/BF16 input pairs; the accumulator is altfmt=0 only."""
    return {(a, b, 0): (_F16[a], _F16[b], fmt_c, mx)
            for a in (0, 1) for b in (0, 1)}


FP_CELLS = {
    # vfwmmacc.vv (W=2), spec 7319-7336.
    (2, 8):  (4, {(0, 0, 0): ("e2m1", "e2m1", "e4m3", True),
                  (0, 0, 1): ("e2m1", "e2m1", "e5m2", True)}),
    (2, 16): (8, _ofp8_rows({0: "binary16", 1: "bfloat16"}, mx=False)),
    (2, 32): (16, _f16_rows("binary32")),
    (2, 64): (32, {(None, None, 0): ("binary32", "binary32",
                                     "binary64", False)}),
    # vfqmmacc.vv (W=4), spec 7340-7356.  SEW=8 (EEW=2) is reserved.
    (4, 16): (4, {(0, 0, 0): ("e2m1", "e2m1", "binary16", True),
                  (0, 0, 1): ("e2m1", "e2m1", "bfloat16", True)}),
    (4, 32): (8, _ofp8_rows({0: "binary32"}, mx=True)),
    (4, 64): (16, _f16_rows("binary64")),
    # vf8wmmacc.vv (W=8), spec 7360-7370.  SEW=8 and 16 are reserved.
    (8, 32): (4, {(0, 0, 0): ("e2m1", "e2m1", "binary32", True)}),
    (8, 64): (8, _ofp8_rows({0: "binary64"}, mx=True)),
}

#: W -> the SEWs the widening FP family is defined at, derived from FP_CELLS
#: so the two cannot drift.  Mirrors MX_SEWS_BY_W and WIDENING_SEWS_BY_W.
FP_SEWS_BY_W = {w: tuple(sorted(sew for (ww, sew) in FP_CELLS if ww == w))
                for w in sorted({w for (w, _s) in FP_CELLS})}


def fpw_legal_row(w, sew, altfmt=0, altfmt_a=0, altfmt_b=0):
    """``(fmt_A, fmt_B, fmt_C, mx_allowed)`` for a legal row, else raise.

    The single place this module decides whether a widening floating-point
    ``(W, SEW, altfmt_A, altfmt_B, altfmt)`` tuple is architecture or is a
    reserved encoding, the way :func:`mx_legal_cell` is for round six.

    Where the width ignores altfmt_A/altfmt_B (spec 1490-1493), *both*
    encodings select the same row, so 0 and 1 are accepted and neither is
    silently normalised away.
    """
    try:
        _eew, rows = FP_CELLS[(w, sew)]
    except KeyError:
        raise ValueError(
            f"W={w} SEW={sew} is a reserved encoding for the widening "
            f"floating-point family; legal (W, SEW) cells are "
            f"{sorted(FP_CELLS)} (spec 7288-7370)") from None
    key = (altfmt_a, altfmt_b, altfmt)
    if key not in rows and (None, None, altfmt) in rows:
        key = (None, None, altfmt)      # altfmt_A / altfmt_B ignored here
    try:
        return rows[key]
    except KeyError:
        raise ValueError(
            f"W={w} SEW={sew} altfmt_A={altfmt_a} altfmt_B={altfmt_b} "
            f"altfmt={altfmt} is reserved; the legal rows at this cell are "
            f"{sorted(rows)} (spec 7288-7370)") from None


def fpw_eew_ab(w: int, sew: int) -> int:
    """Input element width at a legal cell.  Always SEW/W; asserted, not assumed."""
    eew, _rows = FP_CELLS[(w, sew)]
    assert eew == sew // w, (w, sew, eew)
    return eew


def fpw_mixed_rows(w: int, sew: int):
    """The rows whose two input formats differ, as a sorted key list.

    Round seven keeps mixed-format programs in their own tier.  Sharing a
    tier with the same-format rows would make a failure unattributable
    between the mixing logic and the per-format decode -- the two things this
    round adds at once.  This is the selector that tier uses.
    """
    _eew, rows = FP_CELLS[(w, sew)]
    return sorted(k for k, v in rows.items() if v[0] != v[1])


def fpw_check_legality(w: int, lmul: int, sew: int, lam: int,
                       bs: int = 0, vm: int = 1) -> None:
    """Legality for a widening FP multiply-accumulate at this geometry.

    ``vm=1`` (unscaled) adds nothing beyond the encoding-map row, which
    :func:`fpw_legal_row` decides.  ``vm=0`` (microscaled) additionally
    requires the row to have MX cells, and then reuses round six's
    ``check_microscaling_legality`` verbatim -- spec 2343-2347 (SEW*LAMBDA >=
    pw) and Sail 5151-5158 (at bs=1, W*LMUL <= SEW).  The same rule, not a
    parallel copy of it: round seven's vm=0 path is the *same* Sail function
    round six already exercises, and a second transcription is a second place
    to get it wrong.
    """
    if vm not in (0, 1):
        raise ValueError(f"vm is one bit, got {vm}")
    if vm == 0:
        mx_check_legality(w, lmul, sew, lam, bs)


def fpw_pack_elements(values: Sequence[int], fmt: FpFormat) -> List[int]:
    """Pack element bit patterns into bytes, little-endian, for any width.

    Written against ``fmt.sub_byte`` rather than against a width test, and
    built now even though no currently-resolved format reaches the sub-byte
    branch.  The three cells round seven can implement today have EEW_AB of
    16, 32 and 16 -- all byte-or-wider -- so this branch is dead code until
    the OCP documents land.  It is written anyway, because the sub-byte path
    is one of the two main risk surfaces in the six cells that remain, and
    letting the easy three dictate the shape of the packer is exactly how the
    second pass turns into a rewrite.

    The sub-byte order is spec 1207-1219: element 2n in the LOW nibble of
    byte n, element 2n+1 in the HIGH nibble.  Getting it backwards transposes
    every pair of K elements and reads as an arithmetic bug rather than a
    layout one, which is why round six's Machine.vget carries the same rule
    and :func:`check_round_seven_packing` checks the two against each other.
    """
    if fmt.sub_byte:
        if fmt.width != 4:
            raise ValueError(
                f"{fmt.name}: only 4-bit sub-byte packing is modelled "
                f"(spec 1207-1219 defines EEW=4 and nothing narrower)")
        if len(values) % 2:
            raise ValueError("4-bit elements are packed in pairs")
        return [(lo & 0xF) | ((hi & 0xF) << 4)
                for lo, hi in zip(values[0::2], values[1::2])]
    nbytes = fmt.width // 8
    out: List[int] = []
    for v in values:
        out.extend((v & ((1 << fmt.width) - 1)).to_bytes(nbytes, "little"))
    return out


def fpw_unpack_elements(data: Sequence[int], count: int,
                        fmt: FpFormat) -> List[int]:
    """Inverse of :func:`fpw_pack_elements`; round-trips for every width."""
    if fmt.sub_byte:
        out = []
        for n in range((count + 1) // 2):
            out.append(data[n] & 0xF)
            out.append((data[n] >> 4) & 0xF)
        return out[:count]
    nbytes = fmt.width // 8
    return [int.from_bytes(bytes(data[i * nbytes:(i + 1) * nbytes]), "little")
            for i in range(count)]


def _fpw_mul_exact(a_bits: int, fmt_a: FpFormat, b_bits: int,
                   fmt_b: FpFormat):
    """One A x B product, exactly.  ``("nan",)`` / ``("inf", s)`` / ``("num", q)``.

    Spec 1384: "Each A x B element product is defined by the exact
    mathematical product of the corresponding input values, regardless of
    whether `fmt_A` equals `fmt_B`."  No rounding happens here; the product
    is a Fraction with as many significand bits as it needs (spec 1391-1397
    bounds that at p_A + p_B).

    Mixed formats are not a special case in this function, and that is the
    point: the mixed cells differ from the same-format ones only in which
    descriptor each operand is decoded with.
    """
    ka, sa, va = fpf_unpack(a_bits, fmt_a)
    kb, sb, vb = fpf_unpack(b_bits, fmt_b)
    if ka == "nan" or kb == "nan":
        return ("nan",)
    sign = sa ^ sb
    if ka == "inf" or kb == "inf":
        # Spec 1839-1841: zero multiplied by infinity is an invalid
        # operation producing an internal NaN.
        other_zero = (kb == "num" and vb == 0) if ka == "inf" else (va == 0)
        if other_zero:
            return ("nan",)
        return ("inf", sign)
    return ("num", (-va if sa else va) * (-vb if sb else vb))


def _fpw_sum_exact(terms):
    """Exact sum of product terms.  Infinities of opposite sign give NaN."""
    total = Fraction(0)
    inf_sign = None
    for t in terms:
        if t[0] == "nan":
            return ("nan",)
        if t[0] == "inf":
            if inf_sign is not None and inf_sign != t[1]:
                # Spec 1846-1849: adding infinities of opposite signs raises
                # invalid and produces an internal NaN.
                return ("nan",)
            inf_sign = t[1]
        else:
            total += t[1]
    if inf_sign is not None:
        return ("inf", inf_sign)
    return ("num", total)


def _fpw_materialise(value, fmt: FpFormat) -> int:
    """Turn an exact result into *fmt* bits, rounding per frm.

    NaN becomes the format's default NaN (spec 1872-1878: "whenever a shared
    floating-point helper materializes a NaN result in a concrete
    floating-point format, it shall return the default canonical NaN for that
    format").  Infinity is only reachable in a format that has one --
    :func:`fpf_round` refuses the others rather than inventing a pattern.
    """
    if value[0] == "nan":
        return fpf_default_nan(fmt)
    if value[0] == "inf":
        if fmt.name not in FP_FORMATS_BY_NAME:
            # Round eight, OFP8 destination: OCP Table 3 (p.15) "+-Inf" row
            # -- SAT gives +-max, NONSAT gives E5M2 +-Inf / E4M3 NaN, which
            # is exactly the overflow rule the descriptor carries.
            return _fpf_overflow(value[1], fmt)
        if not fmt.has_inf:
            raise OCPSpecUnavailable(
                f"an infinite result in {fmt.name}, which encodes no "
                f"infinity; the OCP document decides what this produces")
        return ((1 << (fmt.width - 1)) if value[1] else 0) | \
               (fmt.emax << (fmt.prec - 1))
    return fpf_round(value[1], fmt, FP_FRM)


def _fpw_add_into(acc_bits: int, addend, fmt: FpFormat) -> int:
    """``C <- round_frm(C + S)`` (spec 1591-1593), with C read back as bits."""
    kind, sign, mag = fpf_unpack(acc_bits, fmt)
    if kind == "nan":
        return fpf_default_nan(fmt)
    acc = ("inf", sign) if kind == "inf" else ("num", -mag if sign else mag)
    return _fpw_materialise(_fpw_sum_exact([acc, addend]), fmt)


def fpw_reference_gemm(geom: "TileGeometry", a, b, c,
                       fmt_a: FpFormat, fmt_b: FpFormat, fmt_c: FpFormat):
    """The architectural result of one widening FP multiply-accumulate.

    Implements the disclosed tuple :data:`FP_DISCLOSURE` -- G=1, psm=0,
    rnd=frm -- for any W, and is therefore format-agnostic by construction:
    the only thing the three descriptors change is how operands decode and
    how results round.

    The reduction, from spec 1552-1593 and 1747-1752:

    * The K dimension splits into ``LAMBDA * LMUL`` groups (the number of
      group updates per output element is ``(LAMBDA * LMUL) / G`` and G=1).
    * Each group is one sub-dot-product, so it holds ``G * W = W`` products,
      at W consecutive logical K indices in increasing order (spec 1679-1684).
    * psm=0: the group's partial sum S is formed exactly, with no rounding to
      fmt_C (spec 1565).
    * rnd=frm: S is then rounded to fmt_C under frm (spec 1584).
    * The group update is ``C <- round_frm(C + S_rounded)`` (spec 1591-1593).

    Note what this is *not*: a fused multiply-add per k, and not an exact dot
    product over all of K.  There are exactly two rounding points per group,
    and a model that collapses them to one agrees with this everywhere the
    partial sums happen to be exact -- which is precisely the region the
    exactly-representable self-check tier lives in, and precisely why that
    tier cannot be the only check.  See :func:`fpw_selfcheck_exact`.
    """
    w, groups = geom.w, geom.lam * geom.lmul
    assert geom.k_eff == groups * w, (geom.k_eff, groups, w)
    # C is the *physical* M x N_max tile, not the active M x N one.  Columns
    # at j >= N are inactive: the instruction does not write them, so they
    # pass through carrying whatever C held on entry.  Returning a narrow
    # M x N result and zero-filling the rest is the mistake this shape
    # prevents -- it produced 177 partial-N failures that looked like an
    # arithmetic bug and were a tail-policy bug.
    assert len(c[0]) == geom.n_max, (len(c[0]), geom.n_max)
    out = [[c[i][j] for j in range(geom.n_max)] for i in range(geom.m)]
    for i in range(geom.m):
        for j in range(geom.n):
            acc = c[i][j]
            for g in range(groups):
                products = [_fpw_mul_exact(a[i][k], fmt_a, b[j][k], fmt_b)
                            for k in range(g * w, (g + 1) * w)]
                s = _fpw_sum_exact(products)
                acc = _fpw_add_into(acc, _fpw_reround(s, fmt_c), fmt_c)
            out[i][j] = acc
    return out


def _fpw_reround(s, fmt_c: FpFormat):
    """rnd=frm: round the partial sum to fmt_C before accumulating it.

    Kept as its own function because it is the single line that distinguishes
    the disclosed tuple from ``rnd=xct``, and the negative control
    :func:`check_round_seven_negative_controls` sabotages exactly it.
    """
    if s[0] != "num":
        return s
    bits = _fpw_materialise(s, fmt_c)
    kind, sign, mag = fpf_unpack(bits, fmt_c)
    if kind == "nan":
        return ("nan",)
    if kind == "inf":
        return ("inf", sign)
    return ("num", -mag if sign else mag)


def fpw_golden_bytes(geom: "TileGeometry", tile, fmt_c: FpFormat) -> List[int]:
    """The C tile as the bytes a program embeds and memcmps against.

    This is the artefact that makes the reference model the sole source of
    truth for a narrow accumulator, and therefore the artefact whose
    correctness the negative controls and the self-check tier exist to
    police.  Row-major, at all LMUL (spec: memory is row-major everywhere),
    over the *physical* M x M tile rather than the active M x N one, because
    that is what a C transfer moves.
    """
    assert len(tile[0]) == geom.n_max, (len(tile[0]), geom.n_max)
    out: List[int] = []
    nbytes = fmt_c.width // 8
    for i in range(geom.m):
        for j in range(geom.n_max):
            out.extend((tile[i][j] & ((1 << fmt_c.width) - 1))
                       .to_bytes(nbytes, "little"))
    return out


def fpw_selfcheck_exact(geom: "TileGeometry", a, b, c,
                        fmt_a: FpFormat, fmt_b: FpFormat,
                        fmt_c: FpFormat) -> bool:
    """True where this case's result is independent of every rounding choice.

    The cross-check tier: a case for which the exact mathematical result is
    representable in fmt_C at every intermediate step is one the *program*
    can verify on the DUT without trusting the reference model, because any
    conforming (G, psm, rnd) tuple produces the same bits.

    It is deliberately a predicate over a case rather than a restriction on
    how cases are generated.  Generating only exactly-representable cases
    would narrow coverage away from rounding, which is the round's new risk
    surface; classifying cases instead means the golden-bytes tier keeps the
    full range and the self-checking tier is a *subset* of the same
    population.  That subset relationship is what makes the two tiers a
    cross-check rather than two disjoint samples that can never disagree --
    see :func:`check_round_seven_tier_overlap`.
    """
    w, groups = geom.w, geom.lam * geom.lmul
    for i in range(geom.m):
        for j in range(geom.n):
            acc = fpf_unpack(c[i][j], fmt_c)
            if acc[0] != "num":
                return False
            running = -acc[2] if acc[1] else acc[2]
            for g in range(groups):
                terms = [_fpw_mul_exact(a[i][k], fmt_a, b[j][k], fmt_b)
                         for k in range(g * w, (g + 1) * w)]
                s = _fpw_sum_exact(terms)
                if s[0] != "num":
                    return False
                # S must be exactly representable, and so must C + S.
                if not _fpw_exact_in(s[1], fmt_c):
                    return False
                running += s[1]
                if not _fpw_exact_in(running, fmt_c):
                    return False
    return True


def _fpw_exact_in(value: "Fraction", fmt: FpFormat) -> bool:
    """Is *value* representable in *fmt* with no rounding at all?"""
    bits = fpf_round(value, fmt, FP_FRM)
    kind, sign, mag = fpf_unpack(bits, fmt)
    if kind != "num":
        return False
    return (-mag if sign else mag) == value


def fpw_legal_configs(vlen: int, **kwargs):
    """Every round-seven geometry, in W order.

    A thin ordering over :func:`ime_legal_configs`, exactly as
    :func:`mx_legal_configs` is for round six: the legality lives in
    :meth:`TileGeometry.validate`, which now knows ``kind='fpw'``, so this
    adds only the per-W SEW restriction the encoding map imposes.  W is the
    outer loop for the same append-don't-interleave reason every other tier
    has.
    """
    for w in sorted(FP_SEWS_BY_W):
        yield from ime_legal_configs(vlen, sews=FP_SEWS_BY_W[w], ws=(w,),
                                     kinds=("fpw",), **kwargs)


def fpw_mx_max_lmul(w: int, sew: int) -> int:
    """Largest LMUL in {1,2,4,8} permitted at bs=1, spec 2285-2320.

    The spec states this as a table; it is computed here from the rule the
    table summarises (W*LMUL <= SEW) and the table is checked against it in
    :func:`check_round_seven_mx_lmul_table`, so a transcription error in
    either one is caught by the other.
    """
    return max(l for l in (1, 2, 4, 8) if w * l <= sew)


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


def _mx_geometries(vlen: int = 256):
    """Every round-six (geometry, altfmt) pair at *vlen*."""
    return [(geom, altfmt)
            for geom in mx_legal_configs(vlen)
            for altfmt in mx_altfmts(geom.w, geom.sew)]


def check_round_six_encoding_map() -> None:
    """MX_CELLS is the Sail's SEW guards, read back the other way round.

    The table at spec 7469-7540 and the per-instruction guards (vfwimmacc
    6035-6038, vfqimmacc 6266-6270, vf8wimmacc 6151) are two statements of
    the same fact, so they can be checked against each other.
    """
    guards = {                       # mnemonic -> the SEWs its Sail accepts
        "vfwimmacc.vv": {16},        # rejects 8, 32, 64
        "vfqimmacc.vv": {16, 32},    # rejects 8 and 64
        "vf8wimmacc.vv": {32, 64},   # rejects < 32
    }
    by_mnemonic: Dict[str, set] = {name: set() for name in guards}
    for (w, sew) in MX_CELLS:
        name = TileGeometry(256, sew, 1, 1, 1, w, kind="mx").mnemonic
        by_mnemonic[name].add(sew)
    assert by_mnemonic == guards, (by_mnemonic, guards)

    # altfmt=1 is reserved wherever the accumulator is 32 or 64 bits wide.
    for (w, sew), (_ew, fmts) in MX_CELLS.items():
        if sew >= 32:
            assert set(fmts) == {0}, (w, sew, fmts)
        else:
            assert set(fmts) == {0, 1}, (w, sew, fmts)

    # Reserved cells raise rather than silently modelling something.
    for w, sew in ((2, 8), (2, 32), (2, 64), (4, 8), (4, 64), (8, 8), (8, 16)):
        try:
            mx_legal_cell(w, sew)
        except ValueError:
            continue
        raise AssertionError(f"W={w} SEW={sew} should be reserved")

    # The integer input width is SEW/W in every cell -- the family's own
    # EEW_A = SEW/W rule (Sail 4885-4918), not an independent table.
    for (w, sew), (ewidth, _fmts) in MX_CELLS.items():
        assert ewidth == sew // w, (w, sew, ewidth)


def check_round_six_e8m0() -> None:
    """E8M0 decode, against the worked cases the spec states in prose."""
    # 0x7F is 2**0 = 1.0 in every accumulator format.
    for width, fmt in ((16, "binary16"), (16, "bfloat16"),
                       (32, "binary32"), (64, "binary64")):
        bits, is_nan = mx_decode_scale(0x7F, width, fmt)
        assert not is_nan and bits == fp_one(width, fmt), (fmt, bits)

    # 0xFF is the one and only NaN code (spec 1993).
    for width, fmt in ((16, "binary16"), (32, "binary32")):
        bits, is_nan = mx_decode_scale(0xFF, width, fmt)
        assert is_nan and bits == fp_default_nan(width, fmt), fmt

    # Every other code is finite -- there is no zero, infinity or subnormal
    # *encoding*.  0x00 is 2**-127, an ordinary value.
    for byte in (0x00, 0x01, 0x7E, 0x80, 0xFE):
        _bits, is_nan = mx_decode_scale(byte, 64, "binary64")
        assert not is_nan, byte
    assert fp_unpack(mx_decode_scale(0x00, 64, "binary64")[0],
                     64, "binary64")[2] == _pow2(-127)
    assert fp_unpack(mx_decode_scale(0xFE, 64, "binary64")[0],
                     64, "binary64")[2] == _pow2(127)

    # Spec 2013-2016, stated as an example in the text: "the E8M0 scale value
    # 2^127 cannot be represented as a finite IEEE binary16 value and,
    # depending on `frm`, converts to either +inf or the largest finite
    # positive binary16 value".  At frm=RNE it is +inf.
    bits, is_nan = mx_decode_scale(0xFE, 16, "binary16")
    assert not is_nan and fp_is_inf(bits, 16, "binary16"), hex(bits)
    # ... and "a sufficiently small positive E8M0 value may similarly convert
    # to either +0 or the minimum positive subnormal value".  2**-127 is far
    # below binary16's 2**-24 minimum subnormal, so RNE gives +0.
    bits, _ = mx_decode_scale(0x00, 16, "binary16")
    assert bits == 0, hex(bits)
    # bfloat16 has binary32's exponent range, so the same two bytes are
    # finite there -- which is exactly why the accumulator format, not the
    # storage width, has to key the decode.
    assert not fp_is_inf(mx_decode_scale(0xFE, 16, "bfloat16")[0],
                         16, "bfloat16")
    assert mx_decode_scale(0x00, 16, "bfloat16")[0] != 0

    # Spec 2021-2024: +0 x +inf from two *finite* encoded scales is the
    # default NaN.  binary16 supplies both ends at once.
    blk, is_nan = mx_block_scale(0x00, 0xFE, 16, "binary16")
    assert is_nan and blk == fp_default_nan(16, "binary16"), hex(blk)
    # And a 0xFF byte alone is enough, in any format.
    for scales in ((0xFF, 0x7F), (0x7F, 0xFF), (0xFF, 0xFF)):
        assert mx_block_scale(*scales, 32, "binary32")[1], scales


def check_round_six_block_intervals() -> None:
    """The block intervals tile [0, K_eff) exactly, and ignore LMUL.

    Sail ``int_scaled_gemm`` has no LMUL step loop; the block/step
    intersection of spec 2030-2060 belongs to ``fp_scaled_gemm``.  Asserting
    the partition here is what stops a future round pasting the group-A
    logic into the group-B judge.
    """
    for geom, _altfmt in _mx_geometries():
        for bs in (0, 1):
            try:
                mx_check_legality(geom.w, geom.lmul, geom.sew, geom.lam, bs)
            except ValueError:
                continue
            block_size = mx_block_size(bs)
            blocks = mx_block_count(geom.k_eff, block_size)
            covered = []
            for s in range(blocks):
                k_lo, k_hi = mx_block_interval(s, block_size, geom.k_eff)
                assert k_lo <= k_hi, (geom.describe(), s)
                covered.extend(range(k_lo, k_hi + 1))
            assert covered == list(range(geom.k_eff)), geom.describe()
            # Only the final block may be short.
            lens = [mx_block_interval(s, block_size, geom.k_eff)[1]
                    - mx_block_interval(s, block_size, geom.k_eff)[0] + 1
                    for s in range(blocks)]
            assert all(n == block_size for n in lens[:-1]), geom.describe()
            assert 1 <= lens[-1] <= block_size, geom.describe()


def check_round_six_scale_layout() -> None:
    """The v0 paired-scale layout closes: M * R == VLEN / pw (spec 2192-2197).

    And the pair index is a bijection from (row, block) onto the positions a
    row actually uses, which is what makes ``i * R + s`` checkable without
    reference to the implementation.
    """
    for geom, _altfmt in _mx_geometries():
        r = mx_scale_stride(geom.sew, geom.lam)
        assert geom.m * r == geom.vlen // MX_PAIR_WIDTH, geom.describe()
        for bs in (0, 1):
            try:
                mx_check_legality(geom.w, geom.lmul, geom.sew, geom.lam, bs)
            except ValueError:
                continue
            blocks = mx_block_count(geom.k_eff, mx_block_size(bs))
            # Spec 2343-2347 / Sail 5155 exist precisely to guarantee this:
            # every block a row needs has a slot inside that row's stride.
            assert blocks <= r, (geom.describe(), bs, blocks, r)
            seen = set()
            for m in range(geom.m):
                for s in range(blocks):
                    p = mx_pair_index(m, s, r)
                    assert p not in seen, (geom.describe(), m, s)
                    seen.add(p)
                    assert 0 <= p < geom.vlen // MX_PAIR_WIDTH


def _mx_exact_case(geom: "TileGeometry", altfmt: int, bs: int,
                   seed: int = 0):
    """An all-scales-1.0, C=+0.0 case whose result is exact by construction."""
    rng = random.Random(seed)
    bound = mx_exact_operand_bound(geom.w, geom.sew, altfmt, bs)
    a = [[rng.randint(-bound, bound) for _ in range(geom.k_eff)]
         for _ in range(geom.m)]
    b = [[rng.randint(-bound, bound) for _ in range(geom.k_eff)]
         for _ in range(geom.n)]
    c = [[0 for _ in range(geom.n)] for _ in range(geom.m)]   # +0.0
    blocks = mx_block_count(geom.k_eff, mx_block_size(bs))
    ones_a = [[0x7F] * blocks for _ in range(geom.m)]
    ones_b = [[0x7F] * blocks for _ in range(geom.n)]
    return a, b, c, ones_a, ones_b


def check_round_six_exactness() -> None:
    """With scales 1.0 and C = +0.0 the result is literally int_to_fp(dot).

    This is the property the ``ime_mx_`` directed tier rests on: the on-DUT
    reference is an integer dot product and one ``fcvt``, with no FP rounding
    to reproduce -- so the tier works even where the accumulator is binary16
    or bfloat16, which baseline rv64imafd cannot round to.
    """
    checked = 0
    for geom, altfmt in _mx_geometries():
        for bs in (0, 1):
            try:
                mx_check_legality(geom.w, geom.lmul, geom.sew, geom.lam, bs)
            except ValueError:
                continue
            a, b, c, sa, sb = _mx_exact_case(geom, altfmt, bs, seed=geom.k_eff)
            out = int_scaled_gemm_reference(a, b, c, sa, sb, geom,
                                            bs=bs, altfmt=altfmt)
            block_size = mx_block_size(bs)
            blocks = mx_block_count(geom.k_eff, block_size)
            _ew, fmt = mx_legal_cell(geom.w, geom.sew, altfmt)
            for i in range(geom.m):
                for j in range(geom.n):
                    # The architectural value, recomputed as a sum of exact
                    # per-block integers -- the shape the DUT program uses.
                    total = 0
                    for s in range(blocks):
                        k_lo, k_hi = mx_block_interval(s, block_size,
                                                       geom.k_eff)
                        total += mx_int_block_dot(a, b, i, j, k_lo, k_hi)
                    got = fp_unpack(out[i][j], geom.sew, fmt)
                    assert got[0] == "num", (geom.describe(), i, j, got)
                    value = -got[2] if got[1] else got[2]
                    assert value == total, (geom.describe(), altfmt, bs,
                                            i, j, value, total)
                    checked += 1
    assert checked > 500, checked


def check_round_six_operand_bounds() -> None:
    """The published exactness bounds are the largest ones that hold.

    titan_runs/round6_design.md tabulates which cells are exact with
    unconstrained operands; that claim is checked here rather than trusted,
    and the bound is checked to be *tight* (one more overflows the
    significand) so a needlessly small bound cannot hide a modelling error.
    """
    unconstrained = {(4, 16, 0), (4, 32, 0), (8, 32, 0), (8, 64, 0)}
    for (w, sew), (ewidth, fmts) in sorted(MX_CELLS.items()):
        for altfmt in sorted(fmts):
            native = (1 << (ewidth - 1)) - 1
            bound = mx_exact_operand_bound(w, sew, altfmt, bs=0)
            limit = mx_exact_dot_bound(w, sew, altfmt, bs=0)
            assert 32 * bound ** 2 <= limit, (w, sew, altfmt)
            if bound < native:
                assert 32 * (bound + 1) ** 2 > limit, (w, sew, altfmt, bound)
            assert ((w, sew, altfmt) in unconstrained) == (bound == native), \
                (w, sew, altfmt, bound, native)
            # bs=1 halves the block, so the bound may only ever grow.
            assert mx_exact_operand_bound(w, sew, altfmt, bs=1) >= bound


def check_round_six_nan_exit() -> None:
    """A NaN block scale zeroes out every later block and yields default NaN.

    Sail 5390-5392 breaks out of the block loop, and 5403-5406 writes
    ``fp_defaultNaN``.  The test plants the NaN in the *first* block and in a
    *later* one, because an implementation that evaluates all blocks and then
    checks would still pass the first case.
    """
    bs = 0
    block_size = mx_block_size(bs)
    # Needs >= 2 blocks (so "break out of the loop early" is observable),
    # and >= 2 rows and columns (so the *unpoisoned* elements witness that
    # the early exit is per output element, not per instruction).
    choice = None
    for cand, altfmt in _mx_geometries():
        if altfmt != 0:
            continue
        if mx_block_count(cand.k_eff, block_size) >= 2 \
                and cand.m >= 2 and cand.n >= 2:
            choice = cand
            break
    assert choice is not None, "no round-six geometry spans two blocks"
    geom = choice
    blocks = mx_block_count(geom.k_eff, block_size)
    _ew, fmt = mx_legal_cell(geom.w, geom.sew, 0)
    a, b, c, sa, sb = _mx_exact_case(geom, 0, bs, seed=7)
    clean = int_scaled_gemm_reference(a, b, c, sa, sb, geom, bs=bs)
    nan_bits = fp_default_nan(geom.sew, fmt)
    for victim in (0, blocks - 1):
        for which in ("a", "b"):
            poisoned_a = [row[:] for row in sa]
            poisoned_b = [row[:] for row in sb]
            (poisoned_a if which == "a" else poisoned_b)[0][victim] = 0xFF
            out = int_scaled_gemm_reference(a, b, c, poisoned_a, poisoned_b,
                                            geom, bs=bs)
            # Row 0 (A scale) or column 0 (B scale) is poisoned, not both.
            hit = [(i, j) for i in range(geom.m) for j in range(geom.n)
                   if (i == 0 if which == "a" else j == 0)]
            for i, j in hit:
                assert out[i][j] == nan_bits, (which, victim, i, j)
            for i in range(geom.m):
                for j in range(geom.n):
                    if (i, j) not in hit:
                        assert out[i][j] == clean[i][j], (which, victim, i, j)


def check_round_six_negative_controls() -> None:
    """Two sabotaged models must be caught.

    Round five's lesson, restated: a tier that cannot fail is not a tier.
    Both sabotages are the specific mistakes this round is most likely to
    make, and each is one line away from the real model.  Each picks its own
    geometry by *searching* for one where the sabotage is observable, and
    asserts that such a geometry exists -- a hard-coded geometry that
    silently stops discriminating is how a negative control rots.
    """
    bs = 0
    block_size = mx_block_size(bs)

    # ---- sabotage 1: the v0 pair index transposed ------------------------
    # `s * R + m` instead of `m * R + s` (spec 2161-2170, Sail 5110-5117).
    # A pure layout error, so it is checked against the index map directly.
    found = None
    for geom, altfmt in _mx_geometries():
        if altfmt:
            continue
        r = mx_scale_stride(geom.sew, geom.lam)
        blocks = mx_block_count(geom.k_eff, block_size)
        honest = {mx_pair_index(m, s, r)
                  for m in range(geom.m) for s in range(blocks)}
        swapped = {mx_pair_index(s, m, r)
                   for m in range(geom.m) for s in range(blocks)}
        if honest != swapped:
            found = (geom, honest, swapped)
            break
    assert found is not None, (
        "no round-six geometry distinguishes m*R+s from s*R+m; the v0 "
        "layout negative control has no teeth")

    # ---- sabotage 2: int_block_dot reduced modulo 2**SEW ------------------
    # The round-one `int_gemm` habit carried into a path whose Sail says
    # "an unbounded mathematical integer (no overflow)" (5128-5130).  It is
    # only observable where a block dot product can exceed the SEW range,
    # which needs enough K and wide enough inputs -- so search for that.
    caught = None
    for geom, altfmt in _mx_geometries():
        if altfmt:
            continue
        ewidth, fmt = mx_legal_cell(geom.w, geom.sew, altfmt)
        blocks = mx_block_count(geom.k_eff, block_size)
        if blocks != 1:
            continue
        peak = (1 << (ewidth - 1)) - 1
        a = [[peak] * geom.k_eff for _ in range(geom.m)]
        b = [[peak] * geom.k_eff for _ in range(geom.n)]
        k_lo, k_hi = mx_block_interval(0, block_size, geom.k_eff)
        exact = mx_int_block_dot(a, b, 0, 0, k_lo, k_hi)
        wrapped = _wrap(exact, geom.sew)
        if exact == wrapped:
            continue
        c = [[0] * geom.n for _ in range(geom.m)]
        ones_a = [[0x7F] * blocks for _ in range(geom.m)]
        ones_b = [[0x7F] * blocks for _ in range(geom.n)]
        out = int_scaled_gemm_reference(a, b, c, ones_a, ones_b, geom,
                                        bs=bs, altfmt=altfmt)
        assert out[0][0] == mx_int_to_fp(exact, geom.sew, fmt), \
            geom.describe()
        if out[0][0] != mx_int_to_fp(wrapped, geom.sew, fmt):
            caught = geom
            break
    assert caught is not None, (
        "no round-six geometry makes the modular-reduction sabotage visible; "
        "the exactness negative control has no teeth")


#: The round-seven cells implementable without the OCP documents: every cell
#: whose three formats are all resolved.  Derived, not listed, so that
#: populating FP_FORMAT_TABLE moves cells into this set with no edit here.
def fpw_resolved_cells():
    """The (W, SEW) cells every one of whose rows uses resolved formats."""
    out = []
    for (w, sew), (_eew, rows) in sorted(FP_CELLS.items()):
        if all(FP_FORMAT_TABLE[f].resolved
               for row in rows.values() for f in row[:3]):
            out.append((w, sew))
    return out


def fpw_rows(w: int, sew: int):
    """Legal ``(altfmt_A, altfmt_B, altfmt) -> (fmt_A, fmt_B, fmt_C)`` rows."""
    _eew, rows = FP_CELLS[(w, sew)]
    return {k: (fp_format(v[0]), fp_format(v[1]), fp_format(v[2]))
            for k, v in sorted(rows.items(), key=lambda kv: str(kv[0]))}


def fpw_operand_pool(fmt: FpFormat, rng: random.Random, n: int) -> List[int]:
    """Element bit patterns biased toward the values that expose rounding.

    Deliberately *not* restricted to exactly-representable magnitudes: the
    golden-bytes tier is the one that has to reach inexact partial sums,
    because inexactness is round seven's new risk surface and a pool that
    avoids it would make the suite agree with a model that rounds in the
    wrong places.  :func:`fpw_selfcheck_exact` classifies afterwards; it does
    not filter beforehand.
    """
    pool = []
    one = fpf_round(Fraction(1), fmt, FP_FRM)
    pool += [0, 1 << (fmt.width - 1), one, one | (1 << (fmt.width - 1))]
    pool.append(one | 1)                       # 1.0 + 1ulp: products inexact
    pool.append(one - 1)                       # just under 1.0
    # Round eight: OFP8 pools also carry the encodings where an IEEE-shaped
    # decoder goes wrong (OFP8_EDGE_OPERANDS).  Appended only for OFP8, so
    # every IEEE pool -- and the rng stream behind it -- is unchanged.
    pool += list(OFP8_EDGE_OPERANDS.get(fmt.name, ()))
    for _ in range(max(0, n - len(pool))):
        # A finite value with a modest exponent: large enough that a W-term
        # partial sum can lose bits, small enough that nothing overflows.
        e = rng.randrange(fmt.bias - 3, fmt.bias + 4)
        frac = rng.randrange(1 << (fmt.prec - 1))
        pool.append((rng.randrange(2) << (fmt.width - 1))
                    | (e << (fmt.prec - 1)) | frac)
    return pool[:max(n, 6)]


#: Round eight.  Finite OFP8 encodings every golden-tier pool carries, each
#: one a place a plausible decoder is wrong (OCP OFP8 Table 2, p.13):
#:
#:   E4M3  0x7E / 0xF9  +448 (max normal, exponent field ALL ONES) and -288:
#:                      an IEEE-shaped decoder calls these NaN.
#:         0x01 / 0x87  min subnormal 2^-9, -max subnormal -0.875*2^-6:
#:                      a flush-to-zero decoder loses them.
#:         0x08         min normal 2^-6, the subnormal/normal boundary.
#:   E5M2  0x7B / 0xF8  +57344 (max normal) and -32768.
#:         0x01 / 0x83  min subnormal 2^-16, -max subnormal -0.75*2^-14.
#:         0x04         min normal 2^-14.
#:
#: NaN and Inf are deliberately NOT in the pools: one per K row would turn
#: most of the tile into NaN and hide the rounding the golden tier exists to
#: reach.  They get their own tier, :func:`fpw_special_case`.
OFP8_EDGE_OPERANDS = {
    "e4m3": (0x7E, 0xF9, 0x01, 0x87, 0x08),
    "e5m2": (0x7B, 0xF8, 0x01, 0x83, 0x04),
}

#: The widening cells, split by the round that made them live.  Generators
#: walk round seven's cells first, with round seven's rng, and round eight's
#: after them with a separate rng -- so every round-seven program stays
#: byte-identical even though (2, 16) and (4, 32) sort in between them.
FPW_ROUND_SEVEN_CELLS = ((2, 32), (2, 64), (4, 64))
FPW_ROUND_EIGHT_CELLS = ((2, 16), (4, 32), (8, 64))

#: The OCP element formats, resolved or not.  "narrow" in the MX sense.
OFP_FORMATS = ("e4m3", "e5m2", "e2m1")


def fpw_case(geom: "TileGeometry", row, rng: random.Random):
    """One ``(A, B, C)`` case for a round-seven geometry and encoding row."""
    fmt_a, fmt_b, fmt_c = row
    pa = fpw_operand_pool(fmt_a, rng, 24)
    pb = fpw_operand_pool(fmt_b, rng, 24)
    pc = fpw_operand_pool(fmt_c, rng, 24)
    a = [[rng.choice(pa) for _ in range(geom.k_eff)] for _ in range(geom.m)]
    b = [[rng.choice(pb) for _ in range(geom.k_eff)] for _ in range(geom.n)]
    # C spans the physical tile width: inactive columns are part of the
    # expected image because the instruction leaves them alone.
    c = [[rng.choice(pc) for _ in range(geom.n_max)] for _ in range(geom.m)]
    return a, b, c


def fpw_exact_case(geom: "TileGeometry", row, rng: random.Random):
    """A case whose result is exact at every rounding point, by construction.

    The self-checking tier cannot be harvested from the random population:
    ``fpw_selfcheck_exact`` requires *every* cell of an M x N tile to be
    exact, and at M=N=8 a random draw essentially never is.  Sampling and
    hoping would leave the cross-check silently empty, which is the failure
    this construction exists to avoid.

    Construction: every A, B and C value is a small integer.  Integer
    products and sums stay integers, and every fmt_C in the family holds
    integers exactly well past the bound used here (binary32 to 2**24,
    binary64 to 2**53), so no rounding point can round.  That makes the
    result independent of (G, psm, rnd) -- which is exactly the property
    that lets the *program* check itself without consulting the reference
    model.

    Note what this does and does not narrow.  It narrows the self-checking
    tier, deliberately: that tier's whole job is to be model-independent, and
    the price of that is a population where rounding cannot occur.  It does
    not narrow the golden-bytes tier, which keeps :func:`fpw_case`'s full
    range including the inexact partial sums that are round seven's new risk
    surface.  The two tiers run the same geometries and the same encoding
    rows, so a reference-model error that changes an exact case is caught
    without the model, and one that changes only inexact cases is at least
    confined to the tier that declares its dependence.
    """
    fmt_a, fmt_b, fmt_c = row
    # Bound the dot product so C + sum(A*B) stays well inside the integers
    # fmt_C represents exactly: |A|,|B| <= 3 gives |product| <= 9, and
    # K_eff of them plus |C| <= 64 is far below 2**24.
    lim, c_hi = 3, 64
    if fmt_c.prec < 24:
        # Round eight: binary16 (integers exact to 2**11) and bfloat16 (to
        # 2**8) accumulators at (W=2, SEW=16).  Shrink the bounds so every
        # partial sum is still an exactly representable integer.  Never
        # taken by a round-seven cell (all binary32/64), so their rng stream
        # is untouched.
        cap = 1 << fmt_c.prec
        if lim * lim * geom.k_eff + c_hi > cap:
            lim = 1
        c_hi = min(c_hi, cap - lim * lim * geom.k_eff)
        assert c_hi >= 0, (geom.describe(), fmt_c.name)
    def ints(fmt, n, hi):
        return [fpf_round(Fraction(rng.randrange(-hi, hi + 1)), fmt, FP_FRM)
                for _ in range(n)]
    a = [ints(fmt_a, geom.k_eff, lim) for _ in range(geom.m)]
    b = [ints(fmt_b, geom.k_eff, lim) for _ in range(geom.n)]
    c = [ints(fmt_c, geom.n_max, c_hi) for _ in range(geom.m)]
    return a, b, c


def fpw_special_case(geom: "TileGeometry", row, rng: random.Random):
    """Round eight's special-value tier: an fpw_case with OFP8 specials planted.

    Golden tier semantics (the reference model is the authority), at fixed,
    documented positions so a failure names the rule it broke:

    * A row 0, k=0: a NaN of fmt_A -- a *non-canonical* one where the format
      has more than one (E5M2 0xFD = -S.11111.01; E4M3 has only 0x7F/0xFF, so
      the negative one) -- must give fmt_C's default NaN (spec 1872-1878).
    * A row 1: +max and -max of fmt_A (E4M3 0x7E/0xFE, E5M2 0x7B/0xFB) at k=0
      and k=1 -- finite, never NaN.
    * A row 2, k=0: +Inf where fmt_A has one (E5M2 0x7C); E4M3 has none, so
      its S.1111.000 = 256 goes there instead -- again finite.
    * A row 3: +Inf at k=0 and -Inf at k=W (two different groups) when fmt_A
      has infinities and there are >= 2 groups; with B columns 0 and W set to
      +1.0 this is Inf + -Inf -> NaN (spec 1843-1845).
    * A row 4: every k the minimum positive subnormal (no flush-to-zero).
    * B row N-1, k=0: a NaN of fmt_B, so a NaN reaching through the *other*
      operand is covered too (one C column, every C row).
    """
    fmt_a, fmt_b, fmt_c = row
    a, b, c = fpw_case(geom, row, rng)
    one_b = fpf_round(Fraction(1), fmt_b, FP_FRM)
    w, groups = geom.w, geom.lam * geom.lmul
    sign_a = 1 << (fmt_a.width - 1)
    nan_a = (fpf_default_nan(fmt_a) | sign_a if fmt_a.nan_rule == "ocp_e4m3"
             else ((fmt_a.emax << (fmt_a.prec - 1)) | 1 | sign_a))
    assert fpf_is_nan(nan_a, fmt_a) and nan_a != fpf_default_nan(fmt_a)
    max_a = fpf_max_finite(fmt_a)
    inf_a = fmt_a.emax << (fmt_a.prec - 1) if fmt_a.has_inf else None
    for j in range(geom.n):                  # known +1.0 at k=0 and k=W
        b[j][0] = one_b
        if w < geom.k_eff:
            b[j][w] = one_b
    if geom.m > 0:
        a[0][0] = nan_a
    if geom.m > 1:
        a[1][0] = max_a
        a[1][1] = max_a | sign_a
    if geom.m > 2:
        a[2][0] = inf_a if inf_a is not None else (fmt_a.emax << (fmt_a.prec - 1))
    if geom.m > 3 and inf_a is not None and groups >= 2:
        a[3][0] = inf_a
        a[3][w] = inf_a | sign_a
    if geom.m > 4:
        a[4] = [1] * geom.k_eff
    if geom.n > 1:
        b[geom.n - 1][0] = fpf_default_nan(fmt_b)
    return a, b, c


def check_round_seven_format_table() -> None:
    """FP_FORMAT_TABLE against FP_FORMATS_BY_NAME, and the pending rows.

    The two tables state the same widths and significands for the four
    resolved formats, so each falsifies the other; and the three pending
    rows must stay behaviourally empty.
    """
    for name, (width, ebits, prec) in FP_FORMATS_BY_NAME.items():
        fmt = FP_FORMAT_TABLE[name]
        assert (fmt.width, fmt.ebits, fmt.prec) == (width, ebits, prec), name
        assert fmt.resolved, name
        assert not fmt.sub_byte, name
    for name in OCP_PENDING_FORMATS:
        fmt = FP_FORMAT_TABLE[name]
        assert not fmt.resolved, name
        assert (fmt.has_inf, fmt.nan_rule, fmt.overflow) == (None, None, None)
        try:
            fp_format(name)
        except OCPSpecUnavailable:
            continue
        raise AssertionError(f"fp_format({name!r}) must refuse")
    # Spec 1400-1410 states p for all seven; check the three pending ones,
    # since that is the only thing about them this harness may assert.
    assert FP_FORMAT_TABLE["e2m1"].prec == 2
    assert FP_FORMAT_TABLE["e4m3"].prec == 4
    assert FP_FORMAT_TABLE["e5m2"].prec == 3
    assert FP_FORMAT_TABLE["e2m1"].sub_byte
    # The cells implementable today are exactly the ones reported: round
    # seven's three plus round eight's three OFP8-input cells (the OCP OFP8
    # document is on disk; OCP MX, which defines E2M1, is not).
    assert fpw_resolved_cells() == sorted(FPW_ROUND_SEVEN_CELLS
                                          + FPW_ROUND_EIGHT_CELLS), \
        fpw_resolved_cells()


def check_round_seven_packing() -> None:
    """The packer round-trips at every width, sub-byte branch included.

    The sub-byte branch is unreachable from any resolved format today, so it
    is exercised here against a synthetic 4-bit descriptor.  That is the
    point: the path is built and tested now, so the OFP4 cells arrive to a
    packer that already works rather than to one that has to grow a branch.
    """
    for name in ("binary16", "binary32", "binary64"):
        fmt = fp_format(name)
        vals = [i * 2654435761 % (1 << fmt.width) for i in range(8)]
        packed = fpw_pack_elements(vals, fmt)
        assert len(packed) == 8 * fmt.width // 8
        assert fpw_unpack_elements(packed, 8, fmt) == vals, name

    nibble = FpFormat("probe4", 4, 2, 2, False, "none", "saturate")
    vals = [i & 0xF for i in range(16)]
    packed = fpw_pack_elements(vals, nibble)
    assert len(packed) == 8, packed
    assert fpw_unpack_elements(packed, 16, nibble) == vals
    # Spec 1207-1219: even index low nibble, odd index high nibble -- the
    # same rule sim_check.Machine.vget carries.  Asserted as bytes so that a
    # swapped implementation cannot pass by round-tripping its own mistake.
    assert packed[0] == 0x10 and packed[1] == 0x32, [hex(x) for x in packed]
    # And an odd count must be refused rather than silently padded.
    try:
        fpw_pack_elements([1, 2, 3], nibble)
    except ValueError:
        pass
    else:
        raise AssertionError("an odd sub-byte element count must raise")


def check_round_seven_gemm() -> None:
    """The exact reduction, against cases whose answer is known by hand.

    Three properties, each of which a plausible-but-wrong implementation
    breaks:

    1. C = +0 and one group of W products of +1.0 gives exactly W.
    2. The reduction is *not* an exact dot product over all of K: with two
       groups it rounds twice, so a case built to lose a bit at the group
       boundary differs from the all-at-once sum.
    3. Mixed formats multiply exactly, so FP16 x BF16 -> FP32 is the same
       number as the rational product.
    """
    geom = TileGeometry(256, 32, 1, 1, 8, 2, kind="fpw")
    fa = fb = fp_format("binary16")
    fc = fp_format("binary32")
    one_a = fpf_round(Fraction(1), fa, FP_FRM)
    zero_c = 0
    a = [[one_a] * geom.k_eff for _ in range(geom.m)]
    b = [[one_a] * geom.k_eff for _ in range(geom.n)]
    c = [[zero_c] * geom.n_max for _ in range(geom.m)]
    out = fpw_reference_gemm(geom, a, b, c, fa, fb, fc)
    want = fpf_round(Fraction(geom.k_eff), fc, FP_FRM)
    assert all(v == want for row in out for v in row), (out[0][0], want)

    # Mixed format: FP16 x BF16 -> FP32, exact product (spec 1418-1421,
    # 11 + 8 = 19 <= 24).
    fb2 = fp_format("bfloat16")
    va = fpf_round(Fraction(3, 2), fa, FP_FRM)
    vb = fpf_round(Fraction(5, 4), fb2, FP_FRM)
    g1 = TileGeometry(256, 32, 1, 1, 8, 2, kind="fpw")
    a1 = [[va] * g1.k_eff for _ in range(g1.m)]
    b1 = [[vb] * g1.k_eff for _ in range(g1.n)]
    c1 = [[0] * g1.n_max for _ in range(g1.m)]
    out1 = fpw_reference_gemm(g1, a1, b1, c1, fa, fb2, fc)
    assert out1[0][0] == fpf_round(Fraction(15, 8) * g1.k_eff, fc, FP_FRM)

    # The self-check predicate agrees with the model on an exact case ...
    assert fpw_selfcheck_exact(g1, a1, b1, c1, fa, fb2, fc)
    # ... and rejects one that is not exactly representable.
    inexact = fpf_round(Fraction(1), fa, FP_FRM) | 1      # 1.0 + 1ulp
    big = fpf_round(Fraction(1 << 20), fc, FP_FRM)
    a2 = [[inexact] * g1.k_eff for _ in range(g1.m)]
    c2 = [[big] * g1.n_max for _ in range(g1.m)]
    assert not fpw_selfcheck_exact(g1, a2, b1, c2, fa, fb2, fc)


def check_round_seven_tier_overlap() -> None:
    """The two tiers must meet, at (geometry, encoding row), not merely exist.

    The golden-bytes tier is only as trustworthy as the reference model that
    produced it, so the self-checking tier exists to catch a model error
    without consulting the model.  That only works if the two tiers overlap:
    if the exactly-representable cases sat at geometries or rows the golden
    tier never visits, a disagreement between them would be unobservable and
    the cross-check would be decorative.

    So three things are asserted, not assumed:

    1. Every resolved cell, every geometry and every encoding row is covered
       by *both* tiers -- the same population, checked two ways.
    2. The self-checking tier really is model-independent there: its cases
       are exact, so the disclosed (G, psm, rnd) tuple and the rnd=xct
       alternative give identical bits.
    3. The golden tier really does reach what the self-checking tier cannot:
       on the same geometry and row, the random population produces inexact
       cases.  Without this the golden tier would be an expensive way to
       re-run the exact one.
    """
    rng = random.Random(20260922)
    cells = fpw_resolved_cells()
    assert cells, "no resolved cell -- this check would be vacuous"
    for (w, sew) in cells:
        geoms = [g for g in fpw_legal_configs(256, full_vl_only=True)
                 if (g.w, g.sew) == (w, sew)]
        assert geoms, (w, sew)
        rows = fpw_rows(w, sew)
        for geom in geoms[:2]:
            golden_rows, selfcheck_rows, inexact_rows = set(), set(), set()
            for key, row in rows.items():
                # --- self-checking tier: constructed exact -----------------
                a, b, c = fpw_exact_case(geom, row, rng)
                assert fpw_selfcheck_exact(geom, a, b, c, *row), (
                    f"W={w} SEW={sew} {key}: the constructed case is not "
                    f"exact, so the self-checking tier is not model-"
                    f"independent here")
                selfcheck_rows.add(key)
                # It must also be a legal member of the golden population --
                # the tiers judge one population, not two.
                exact_out = fpw_reference_gemm(geom, a, b, c, *row)
                assert fpw_golden_bytes(geom, exact_out, row[2])
                golden_rows.add(key)
                # --- golden tier: the full range, including inexact --------
                for _ in range(24):
                    ra, rb, rc = fpw_case(geom, row, rng)
                    fpw_reference_gemm(geom, ra, rb, rc, *row)
                    if not fpw_selfcheck_exact(geom, ra, rb, rc, *row):
                        inexact_rows.add(key)
            assert golden_rows == set(rows), (w, sew, geom.describe())
            assert selfcheck_rows == set(rows), (w, sew, geom.describe())
            assert inexact_rows == set(rows), (
                f"W={w} SEW={sew} {geom.describe()}: rows "
                f"{sorted(set(rows) - inexact_rows)} never produced an "
                f"inexact case, so the golden tier adds nothing over the "
                f"self-checking one there")


def check_round_seven_negative_controls() -> None:
    """Each sabotage must be *proved* detectable at the tier that carries it.

    Round six shipped two controls that could not fail: one broke a
    modulo-2^SEW reduction that was provably the identity on every value the
    programs could produce, and one swapped both A and B nibbles, which only
    reorders a sum.  So each control here names the case it is detected on
    and asserts the detection, rather than asserting that a mutation "looks
    like" a break.

    All three target behaviour that is new in round seven and present in the
    three resolved cells.  The two that are not yet reachable -- E4M3
    overflow and NaN propagation through an OFP format -- are deliberately
    absent rather than stubbed, because a control written against a format
    this harness refuses to define would be a guess wearing a test's clothes.
    """
    rng = random.Random(7)
    fa, fb = fp_format("binary16"), fp_format("bfloat16")
    fc = fp_format("binary32")
    # A *legal* geometry, taken from the enumeration rather than built by
    # hand: a hand-built one silently had N > N_max and every control below
    # indexed past the end of B.  validate() is called for the same reason.
    geom = next(g for g in fpw_legal_configs(256, full_vl_only=True)
                if (g.w, g.sew) == (2, 32) and g.lam * g.lmul > 1
                and g.emul_c != 16)
    geom.validate()
    assert geom.lam * geom.lmul > 1, "need >1 group for the rnd control"

    # --- control 1: rnd=frm collapsed to rnd=xct -------------------------
    # Sabotage: skip the per-group rounding of S.  Detected only where a
    # partial sum is inexact in fmt_C, which is why the pool is not
    # restricted to exactly-representable operands.
    def gemm_xct(a, b, c):
        out = [row[:] for row in c]          # inactive columns pass through
        for i in range(geom.m):
            for j in range(geom.n):
                acc = c[i][j]
                for g in range(geom.lam * geom.lmul):
                    terms = [_fpw_mul_exact(a[i][k], fa, b[j][k], fb)
                             for k in range(g * geom.w, (g + 1) * geom.w)]
                    acc = _fpw_add_into(acc, _fpw_sum_exact(terms), fc)
                out[i][j] = acc
        return out

    caught = 0
    for _ in range(40):
        a, b, c = fpw_case(geom, (fa, fb, fc), rng)
        if fpw_reference_gemm(geom, a, b, c, fa, fb, fc) != gemm_xct(a, b, c):
            caught += 1
    assert caught, (
        "rnd=frm vs rnd=xct is undetectable on this population -- the "
        "control has no teeth and the operand pool is too exact")

    # And the converse, which is what makes the control meaningful rather
    # than merely nonzero: on an exactly-representable case the two agree,
    # so the self-checking tier genuinely cannot distinguish them and the
    # golden tier is the only thing that can.
    one_a = fpf_round(Fraction(1), fa, FP_FRM)
    one_b = fpf_round(Fraction(1), fb, FP_FRM)
    ax = [[one_a] * geom.k_eff for _ in range(geom.m)]
    bx = [[one_b] * geom.k_eff for _ in range(geom.n)]
    cx = [[0] * geom.n_max for _ in range(geom.m)]
    assert fpw_selfcheck_exact(geom, ax, bx, cx, fa, fb, fc)
    assert fpw_reference_gemm(geom, ax, bx, cx, fa, fb, fc) == \
        gemm_xct(ax, bx, cx), (
            "the exact case must NOT distinguish the two tuples; if it does, "
            "fpw_selfcheck_exact is admitting cases it should reject")

    # --- control 2: fmt_A and fmt_B swapped ------------------------------
    # The mixed-format rows are the ones this can touch.  It must be proved
    # detectable, because swapping two *same-format* operands is a no-op and
    # a control that only ever ran on those rows would be the round-six
    # nibble mistake again.
    caught = 0
    for _ in range(40):
        a, b, c = fpw_case(geom, (fa, fb, fc), rng)
        if fpw_reference_gemm(geom, a, b, c, fa, fb, fc) != \
                fpw_reference_gemm(geom, a, b, c, fb, fa, fc):
            caught += 1
    assert caught, "decoding A as B's format is undetectable -- no teeth"
    # Proof that the control is about *mixing* and not about the operands:
    # with fmt_A == fmt_B the same swap is provably the identity, so this
    # control must only ever be scored on mixed rows.
    for _ in range(8):
        a, b, c = fpw_case(geom, (fa, fa, fc), rng)
        assert fpw_reference_gemm(geom, a, b, c, fa, fa, fc) == \
            fpw_reference_gemm(geom, a, b, c, fa, fa, fc)

    # --- control 3: group boundary moved ---------------------------------
    # Sabotage: reduce over one group of K_eff products instead of
    # LAMBDA*LMUL groups of W.  This is the psm/G confusion the disclosure
    # exists to pin down, and it is detectable only because the rounding
    # points move -- which again needs inexact partial sums.
    def gemm_one_group(a, b, c):
        out = [row[:] for row in c]
        for i in range(geom.m):
            for j in range(geom.n):
                terms = [_fpw_mul_exact(a[i][k], fa, b[j][k], fb)
                         for k in range(geom.k_eff)]
                out[i][j] = _fpw_add_into(
                    c[i][j], _fpw_reround(_fpw_sum_exact(terms), fc), fc)
        return out

    caught = 0
    for _ in range(40):
        a, b, c = fpw_case(geom, (fa, fb, fc), rng)
        if fpw_reference_gemm(geom, a, b, c, fa, fb, fc) != \
                gemm_one_group(a, b, c):
            caught += 1
    assert caught, (
        "collapsing LAMBDA*LMUL groups into one is undetectable -- the "
        "disclosed G would then be unobservable and the control has no teeth")


def check_round_seven_nan_controls() -> None:
    """NaN and infinity propagation, and three sabotages of it with teeth.

    Not blocked by the OCP documents: binary16, bfloat16, binary32 and
    binary64 all carry infinities and NaNs, so every rule the specification
    states about special values is reachable in the three cells round seven
    implements.  The rules exercised here are:

    * spec 1839-1841 -- zero times infinity is an invalid operation
      producing an internal NaN;
    * spec 1846-1849 -- adding infinities of opposite signs likewise;
    * spec 1872-1878 -- any NaN a shared helper materialises in a concrete
      format is the *default* NaN for that format, so NaN payloads never
      propagate and a bitwise compare is the right compare.

    Each control below names the case that detects it and asserts the
    detection, and each is checked to be silent on cases where it provably
    cannot matter -- the discipline round six's two dead controls skipped.
    """
    fa, fb = fp_format("binary16"), fp_format("bfloat16")
    fc = fp_format("binary32")
    geom = next(g for g in fpw_legal_configs(256, full_vl_only=True)
                if (g.w, g.sew) == (2, 32) and g.lam * g.lmul > 1
                and g.emul_c != 16)
    geom.validate()
    inf_a = ((1 << fa.ebits) - 1) << (fa.prec - 1)
    nan_a = fp_default_nan(fa.width, fa.name)
    zero_b = 0
    one_a = fpf_round(Fraction(1), fa, FP_FRM)
    one_b = fpf_round(Fraction(1), fb, FP_FRM)
    nan_c = fp_default_nan(fc.width, fc.name)

    def tile(v, rows, cols):
        return [[v] * cols for _ in range(rows)]

    # C tiles span the physical width; A spans K_eff and B spans K_eff too.
    cn = geom.n_max

    # --- the rules themselves -------------------------------------------
    # 0 x inf -> default NaN, and it poisons the whole output element.
    a = tile(inf_a, geom.m, geom.k_eff)
    b = tile(zero_b, geom.n, geom.k_eff)
    c = tile(0, geom.m, cn)
    out = fpw_reference_gemm(geom, a, b, c, fa, fb, fc)
    assert all(v == nan_c for row in out for v in row), hex(out[0][0])

    # A NaN operand propagates as the *default* NaN, payload discarded.
    noisy = nan_a | 0x1                       # a different NaN payload
    assert fpf_is_nan(noisy, fa)
    a2 = tile(noisy, geom.m, geom.k_eff)
    b2 = tile(one_b, geom.n, geom.k_eff)
    out2 = fpw_reference_gemm(geom, a2, b2, c, fa, fb, fc)
    assert all(v == nan_c for row in out2 for v in row), hex(out2[0][0])

    # +inf + -inf -> default NaN (two groups, opposite signs).
    assert geom.lam * geom.lmul >= 2
    a3 = [[inf_a] * geom.k_eff for _ in range(geom.m)]
    b3 = [[one_b] * geom.k_eff for _ in range(geom.n)]
    for row in a3:                            # flip the sign of group 1
        for k in range(geom.w, 2 * geom.w):
            row[k] = inf_a | (1 << (fa.width - 1))
    out3 = fpw_reference_gemm(geom, a3, b3, c, fa, fb, fc)
    assert all(v == nan_c for row in out3 for v in row), hex(out3[0][0])

    # A NaN already in C stays NaN whatever is accumulated into it.
    out4 = fpw_reference_gemm(geom, tile(one_a, geom.m, geom.k_eff), b3,
                              tile(nan_c, geom.m, cn), fa, fb, fc)
    assert all(v == nan_c for row in out4 for v in row)

    # --- control 4: NaN payload propagated instead of canonicalised ------
    # Sabotage: return the operand's NaN bits rather than fp_defaultNaN.
    # Detected on the noisy-payload case above, and provably NOT detectable
    # on a case whose NaN is already canonical -- which is why the control
    # must be scored on the noisy case specifically.
    def gemm_payload(a_, b_, c_):
        out_ = [[0] * geom.n for _ in range(geom.m)]
        for i in range(geom.m):
            for j in range(geom.n):
                poison = next((a_[i][k] for k in range(geom.k_eff)
                               if fpf_is_nan(a_[i][k], fa)), None)
                if poison is not None:
                    # widen the payload into fmt_C the naive way
                    out_[i][j] = nan_c | (poison & 0xF)
                else:
                    out_[i][j] = fpw_reference_gemm(
                        geom, a_, b_, c_, fa, fb, fc)[i][j]
        return out_

    assert fpw_reference_gemm(geom, a2, b2, c, fa, fb, fc) != \
        gemm_payload(a2, b2, c), (
            "payload propagation is undetectable on the noisy-NaN case -- "
            "the control has no teeth")
    # ... and silent where the payload happens to be canonical already,
    # which is exactly why the operand pool must contain a non-canonical NaN.
    a_canon = tile(nan_a, geom.m, geom.k_eff)
    assert fpw_reference_gemm(geom, a_canon, b2, c, fa, fb, fc) == \
        gemm_payload(a_canon, b2, c)

    # --- control 5: 0 x inf treated as 0 ---------------------------------
    # Sabotage: return zero for an infinity-times-zero product instead of
    # raising invalid and producing NaN.  Detected on the first case above.
    def gemm_zero_times_inf(a_, b_, c_):
        out_ = [row[:] for row in c_]
        for i in range(geom.m):
            for j in range(geom.n):
                acc = c_[i][j]
                for g in range(geom.lam * geom.lmul):
                    terms = []
                    for k in range(g * geom.w, (g + 1) * geom.w):
                        t = _fpw_mul_exact(a_[i][k], fa, b_[j][k], fb)
                        ka, _sa, va = fpf_unpack(a_[i][k], fa)
                        kb, _sb, vb = fpf_unpack(b_[j][k], fb)
                        if {ka, kb} == {"inf", "num"} and 0 in (va, vb):
                            t = ("num", Fraction(0))
                        terms.append(t)
                    acc = _fpw_add_into(acc, _fpw_reround(
                        _fpw_sum_exact(terms), fc), fc)
                out_[i][j] = acc
        return out_

    assert fpw_reference_gemm(geom, a, b, c, fa, fb, fc) != \
        gemm_zero_times_inf(a, b, c), (
            "0 x inf -> 0 is undetectable -- the control has no teeth")
    # Silent where no operand is infinite, so it cannot fire spuriously on
    # the ordinary population and mask itself as a general arithmetic break.
    rng = random.Random(3)
    finite, _b2, _c2 = fpw_exact_case(geom, (fa, fb, fc), rng)
    assert fpw_reference_gemm(geom, finite, _b2, _c2, fa, fb, fc) == \
        gemm_zero_times_inf(finite, _b2, _c2)

    # --- control 6: inf - inf treated as inf -----------------------------
    # Sabotage: keep the first infinity's sign instead of producing NaN.
    # Detected on the opposite-signs case; silent on the same-signs one,
    # which is the pair that proves the control is about cancellation and
    # not merely about infinities being present.
    def gemm_inf_keeps_sign(a_, b_, c_):
        out_ = [row[:] for row in c_]
        for i in range(geom.m):
            for j in range(geom.n):
                acc = c_[i][j]
                for g in range(geom.lam * geom.lmul):
                    terms = [_fpw_mul_exact(a_[i][k], fa, b_[j][k], fb)
                             for k in range(g * geom.w, (g + 1) * geom.w)]
                    infs = [t for t in terms if t[0] == "inf"]
                    s = (("inf", infs[0][1]) if infs
                         else _fpw_sum_exact(terms))
                    acc_k, acc_s, _ = fpf_unpack(acc, fc)
                    if acc_k == "inf" and s[0] == "inf":
                        acc = _fpw_materialise(("inf", acc_s), fc)
                    else:
                        acc = _fpw_add_into(acc, _fpw_reround(s, fc), fc)
                out_[i][j] = acc
        return out_

    assert fpw_reference_gemm(geom, a3, b3, c, fa, fb, fc) != \
        gemm_inf_keeps_sign(a3, b3, c), (
            "inf + -inf -> inf is undetectable -- the control has no teeth")
    same = [[inf_a] * geom.k_eff for _ in range(geom.m)]
    assert fpw_reference_gemm(geom, same, b3, c, fa, fb, fc) == \
        gemm_inf_keeps_sign(same, b3, c), (
            "the same-sign case must not distinguish them; if it does, the "
            "control is firing on the presence of infinity rather than on "
            "cancellation")


def check_round_seven_golden_bytes() -> None:
    """The golden image is row-major, full-tile, and sensitive to every cell."""
    rng = random.Random(11)
    fa, fb, fc = (fp_format("binary16"), fp_format("bfloat16"),
                  fp_format("binary32"))
    geom = TileGeometry(256, 32, 2, 1, 8, 2, kind="fpw")
    a, b, c = fpw_case(geom, (fa, fb, fc), rng)
    tile = fpw_reference_gemm(geom, a, b, c, fa, fb, fc)
    img = fpw_golden_bytes(geom, tile, fc)
    assert len(img) == geom.m * geom.n_max * (fc.width // 8)
    # Row-major: element (i, j) sits at (i * N_max + j) * bytes.
    nb = fc.width // 8
    for i in range(geom.m):
        for j in range(geom.n):
            off = (i * geom.n_max + j) * nb
            assert int.from_bytes(bytes(img[off:off + nb]), "little") == \
                tile[i][j], (i, j)
    # Every active cell is load-bearing: perturbing one changes one word.
    for (i, j) in ((0, 0), (geom.m - 1, geom.n - 1)):
        bad = [row[:] for row in tile]
        bad[i][j] ^= 1
        assert fpw_golden_bytes(geom, bad, fc) != img, (i, j)


def check_round_seven_encoding_map() -> None:
    """FP_CELLS against the per-instruction Sail SEW guards and the row rules.

    The same two-sided check round six gets from
    :func:`check_round_six_encoding_map`: the table at spec 7288-7370 and the
    instruction mnemonics are two statements of one fact, so each can
    falsify the other.
    """
    guards = {                       # mnemonic -> the SEWs the map gives it
        "vfwmmacc.vv":  {8, 16, 32, 64},
        "vfqmmacc.vv":  {16, 32, 64},     # SEW=8 is EEW=2, reserved
        "vf8wmmacc.vv": {32, 64},         # SEW=8/16 are EEW=1/2, reserved
    }
    by_mnemonic = {name: set() for name in guards}
    for (w, sew) in FP_CELLS:
        by_mnemonic[TileGeometry(256, sew, 1, 1, 1, w,
                                 kind="fpw").mnemonic].add(sew)
    assert by_mnemonic == guards, (by_mnemonic, guards)

    # EEW_AB is SEW/W in every cell -- the family's own rule (spec 1337-1339),
    # not an independent column.  fpw_eew_ab asserts it; call it everywhere.
    for (w, sew) in FP_CELLS:
        assert fpw_eew_ab(w, sew) == sew // w

    # Spec 1442-1443: at EEW=4, altfmt_A=1 is reserved, so OFP4 has exactly
    # one input pair and mixing is impossible.
    for (w, sew), (eew, rows) in FP_CELLS.items():
        if eew != 4:
            continue
        assert {(a, b) for (a, b, _f) in rows} == {(0, 0)}, (w, sew, rows)
        assert fpw_mixed_rows(w, sew) == [], (w, sew)

    # Spec 1490-1493: at EEW>=32 altfmt_A/altfmt_B are ignored, written as
    # None.  Both encodings must therefore select the same row.
    for (w, sew), (eew, rows) in FP_CELLS.items():
        if eew < 32:
            assert not any(a is None for (a, _b, _f) in rows), (w, sew)
            continue
        assert {(a, b) for (a, b, _f) in rows} == {(None, None)}, (w, sew)
        for a in (0, 1):
            for b in (0, 1):
                assert (fpw_legal_row(w, sew, 0, a, b)
                        == fpw_legal_row(w, sew, 0, 0, 0)), (w, sew, a, b)

    # The asymmetry the table is transcribed rather than generated for:
    # OFP8 inputs carry both accumulator formats only at (W=2, SEW=16).
    # (W=2, SEW=16) is the only OFP8 cell with two accumulator formats, so
    # it has 4 input pairs x 2 altfmt = 8 legal rows (spec 7322-7329).  The
    # other two OFP8 cells keep the same 4 input pairs but altfmt=1 is
    # reserved, so they have 4 (spec 7345-7350, 7365-7370).
    assert len(FP_CELLS[(2, 16)][1]) == 8, len(FP_CELLS[(2, 16)][1])
    assert len(FP_CELLS[(4, 32)][1]) == 4, len(FP_CELLS[(4, 32)][1])
    assert len(FP_CELLS[(8, 64)][1]) == 4, len(FP_CELLS[(8, 64)][1])
    # All four OFP8 input pairs, mixed ones included, exist at every OFP8
    # cell -- spec 1446-1451 says so in prose and the table repeats it.
    for cell in ((2, 16), (4, 32), (8, 64)):
        pairs = {(a, b) for (a, b, _f) in FP_CELLS[cell][1]}
        assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}, (cell, pairs)
        assert len(fpw_mixed_rows(*cell)) == len(FP_CELLS[cell][1]) // 2
    assert {f for (_a, _b, f) in FP_CELLS[(2, 16)][1]} == {0, 1}
    assert {f for (_a, _b, f) in FP_CELLS[(4, 32)][1]} == {0}
    assert {f for (_a, _b, f) in FP_CELLS[(8, 64)][1]} == {0}

    # Reserved cells raise rather than silently modelling something.
    for w, sew in ((2, 128), (4, 8), (8, 8), (8, 16), (1, 32)):
        try:
            fpw_legal_row(w, sew)
        except ValueError:
            continue
        raise AssertionError(f"W={w} SEW={sew} should be reserved")
    # And reserved *rows* inside legal cells: altfmt=1 at a 32/64-bit
    # accumulator (spec 7331-7334, 7346-7348, 7363-7368).
    for w, sew in ((2, 32), (2, 64), (4, 32), (4, 64), (8, 32), (8, 64)):
        try:
            fpw_legal_row(w, sew, altfmt=1)
        except ValueError:
            continue
        raise AssertionError(f"W={w} SEW={sew} altfmt=1 should be reserved")


def check_round_seven_mx_lmul_table() -> None:
    """The bs=1 max-LMUL table (spec 2285-2320) against the rule it summarises.

    The spec states both: the rule ``W * LMUL <= SEW`` (2277-2283, Sail
    5156) and a table of its consequences.  Transcribing one and deriving
    the other means a mistake in either shows up here rather than in a
    silently over-permissive generator.
    """
    stated = {                       # (W, SEW) -> max LMUL, spec 2285-2320
        (2, 8): 4,                   # OFP4 -> OFP8, vfwmmacc
        (4, 16): 4,                  # OFP4 -> FP16/BF16, vfqmmacc
        (8, 32): 4,                  # OFP4 -> FP32, vf8wmmacc
    }
    for (w, sew), want in stated.items():
        assert fpw_mx_max_lmul(w, sew) == want, (w, sew, want)
    # "All other currently defined MX encodings: 8 (no additional
    # block-size-16 restriction)" -- every remaining MX-capable FP cell.
    for (w, sew), (_eew, rows) in FP_CELLS.items():
        if not any(row[3] for row in rows.values()):
            continue
        if (w, sew) in stated:
            continue
        assert fpw_mx_max_lmul(w, sew) == 8, (w, sew)

    # And the rule and the helper must agree at the boundary, both ways.
    for (w, sew) in FP_CELLS:
        cap = fpw_mx_max_lmul(w, sew)
        for lmul in (1, 2, 4, 8):
            legal = w * lmul <= sew
            assert legal == (lmul <= cap), (w, sew, lmul)


def check_round_seven_mx_applicability() -> None:
    """Which rows admit vm=0 at all (spec 2325-2332, 7270-7286).

    Microscaling is defined for OFP4 and OFP8 inputs, and for nothing else in
    this family: the FP16/BF16 and FP32 input rows have empty MX cells, so
    vm=0 there is an illegal encoding rather than an unscaled one.  Getting
    this backwards would make the judge accept a DUT that reads v0 on a
    plain FP16 matmul.
    """
    for (w, sew), (eew, rows) in FP_CELLS.items():
        for key, (fa, fb, _fc, mx) in rows.items():
            narrow = fa in OFP_FORMATS and fb in OFP_FORMATS
            # Necessary, not sufficient: a wide-input row is never MX.
            if not narrow:
                assert not mx, (w, sew, key)
            # And MX is uniform within a cell -- no cell mixes the two.
            assert mx == rows[next(iter(rows))][3], (w, sew, key)
    # The one narrow cell that is nevertheless *not* MX-capable: OFP8 ->
    # FP16/BF16 at (W=2, SEW=16), whose MX columns are "—" (spec 7322-7329).
    assert not any(row[3] for row in FP_CELLS[(2, 16)][1].values())
    # ... and every other narrow cell is.
    for cell in ((2, 8), (4, 16), (4, 32), (8, 32), (8, 64)):
        assert all(row[3] for row in FP_CELLS[cell][1].values()), cell

    # SEW*LAMBDA >= 16 is a hard gate on vm=0 (spec 2343-2347).  SEW=8 with
    # LAMBDA=1 is the case that fails it, and (W=2, SEW=8) is MX-capable, so
    # this is reachable rather than hypothetical.
    try:
        fpw_check_legality(2, 1, 8, 1, bs=0, vm=0)
    except ValueError:
        pass
    else:
        raise AssertionError("SEW=8 LAMBDA=1 must be illegal at vm=0")
    fpw_check_legality(2, 1, 8, 2, bs=0, vm=0)       # SEW*LAMBDA = 16, legal
    # vm=1 is unaffected by either microscaling rule.
    fpw_check_legality(2, 1, 8, 1, bs=0, vm=1)
    fpw_check_legality(8, 8, 32, 1, bs=1, vm=1)      # W*LMUL > SEW, but vm=1


def check_round_seven_ocp_gate() -> None:
    """The formats whose document is missing must refuse, loudly and by name.

    Round seven asserted this for all three OFP formats.  Round eight has the
    OCP OFP8 v1.0 document, so E4M3 and E5M2 moved out; E2M1 (OCP MX v1.0,
    still not on disk) must keep refusing, and every cell that touches it
    must stay unjudgeable.  Filling E2M1 in without the document fails here.
    """
    assert set(OCP_PENDING_FORMATS) == {"e2m1"}, sorted(OCP_PENDING_FORMATS)
    for fmt in OCP_PENDING_FORMATS:
        assert fmt not in FP_FORMATS_BY_NAME, (
            f"{fmt} has been populated but OCP_PENDING_FORMATS still lists "
            f"it -- the two must move together")
        for width in (4, 8):
            try:
                fp_fields(width, fmt)
            except OCPSpecUnavailable:
                break
            except ValueError:
                continue
            raise AssertionError(f"fp_fields({width}, {fmt!r}) must refuse")
        else:
            raise AssertionError(f"{fmt} never raised OCPSpecUnavailable")
        try:
            fp_format(fmt)
        except OCPSpecUnavailable:
            pass
        else:
            raise AssertionError(f"fp_format({fmt!r}) must refuse")
    # OFP8 is resolved, but only through the descriptor helpers: the
    # IEEE-shaped fp_fields path must refuse it rather than misread E4M3.
    for fmt in ("e4m3", "e5m2"):
        assert FP_FORMAT_TABLE[fmt].resolved, fmt
        assert fmt not in FP_FORMATS_BY_NAME, fmt
        try:
            fp_fields(8, fmt)
        except ValueError:
            continue
        raise AssertionError(f"fp_fields(8, {fmt!r}) must refuse")

    # Every cell whose inputs or accumulator is a pending format is
    # unjudgeable.  Derived, so the report's number is not hand-asserted.
    blocked = [(w, sew) for (w, sew), (_e, rows) in FP_CELLS.items()
               if any(f in OCP_PENDING_FORMATS
                      for row in rows.values() for f in row[:3])]
    assert sorted(blocked) == [(2, 8), (4, 16), (8, 32)], blocked
    clear = sorted(set(FP_CELLS) - set(blocked))
    assert clear == sorted(FPW_ROUND_SEVEN_CELLS + FPW_ROUND_EIGHT_CELLS), \
        clear
    for cell in blocked:
        try:
            fpw_rows(*cell)
        except OCPSpecUnavailable:
            continue
        raise AssertionError(f"{cell} resolved without OCP MX")

    # The width-keyed fallback must not become a back door.  fp_fields with
    # fmt=None models binary32/binary64 only, so an E4M3 accumulator cannot
    # be reached by "just pass the width" -- the hole that let round six's
    # first cut return zeros for sub-byte reads is the same shape.
    for width in (4, 8, 16):
        try:
            fp_fields(width)
        except ValueError:
            continue
        raise AssertionError(f"fp_fields({width}) must not resolve")


def check_round_seven_geometry() -> None:
    """The round-seven geometry set is nonempty, disjoint in kind, and stable."""
    geoms = list(fpw_legal_configs(256, full_vl_only=True))
    assert geoms, "round seven enumerates no geometry"
    assert all(g.kind == "fpw" and g.w in (2, 4, 8) for g in geoms)
    # Every enumerated geometry names a real encoding-map cell.
    for g in geoms:
        fpw_legal_row(g.w, g.sew)
    # Sub-byte inputs are reached -- OFP4 at EEW=4 is the round's re-run of
    # the trap that made round six's first cut return zeros for every A/B
    # read.  If no geometry has EEW_AB=4, the nibble path is untested here.
    assert any(g.eew_ab == 4 for g in geoms), \
        "no EEW_AB=4 geometry: the OFP4 sub-byte path would go unexercised"
    # kind='fpw' must not perturb the pre-round-seven enumerations.
    assert not any(g.kind == "fpw"
                   for g in ime_legal_configs(256, kinds=("int", "fp", "mx")))


# --- round eight: OFP8 (E4M3 / E5M2) judges -------------------------------

def _ocp_ofp8_table(name: str):
    """All 256 codes of an OFP8 format, built straight from the OCP formulas.

    Deliberately independent of FpFormat, _fpf_unpack_generic and every
    helper above: the constants are typed in from OCP OFP8 v1.0 -- sec. 4.2
    (p.11: E4M3 4 exponent bits, 3 mantissa bits, bias 7; E5M2 5, 2, 15),
    sec. 5.1 (p.12: the normal and subnormal value formulas, "E4M3 ... does
    not represent infinities and uses only two bit patterns for NaN") and
    Table 2 (p.13: E4M3 NaN S.1111.111; E5M2 Inf S.11111.00, NaN
    S.11111.{01,10,11}).  Returns code -> ("nan"|"inf"|"num", sign, value).
    """
    ebits, mbits, bias = {"e4m3": (4, 3, 7), "e5m2": (5, 2, 15)}[name]
    out = {}
    for code in range(256):
        s = code >> 7
        e = (code >> mbits) & ((1 << ebits) - 1)
        m = code & ((1 << mbits) - 1)
        if name == "e4m3" and e == 15 and m == 7:
            out[code] = ("nan", s, Fraction(0))
        elif name == "e5m2" and e == 31:
            out[code] = ("inf" if m == 0 else "nan", s, Fraction(0))
        elif e == 0:
            out[code] = ("num", s, Fraction(2) ** (1 - bias)
                         * Fraction(m, 2 ** mbits))
        else:
            out[code] = ("num", s, Fraction(2) ** (e - bias)
                         * (1 + Fraction(m, 2 ** mbits)))
    return out


def check_round_eight_ofp8_decode() -> None:
    """Every one of the 512 OFP8 codes against the OCP formulas, and anchors.

    1. fpf_unpack (the path the reference GEMM uses) equals the independent
       table for all 256 codes of each format.
    2. The table itself reproduces OCP Table 1 / Table 2's stated values, so
       a typo in the independent table cannot silently agree with a typo in
       the descriptor.
    3. The generic decoder is IEEE-correct where it should be: on all 65536
       binary16 and bfloat16 codes it equals fp_unpack.
    """
    for name in ("e4m3", "e5m2"):
        fmt = fp_format(name)
        table = _ocp_ofp8_table(name)
        for code in range(256):
            assert fpf_unpack(code, fmt) == table[code], (
                name, hex(code), fpf_unpack(code, fmt), table[code])
            assert fpf_is_nan(code, fmt) == (table[code][0] == "nan")
            assert fpf_is_inf(code, fmt) == (table[code][0] == "inf")
    e4, e5 = _ocp_ofp8_table("e4m3"), _ocp_ofp8_table("e5m2")
    # OCP Table 2, value by value.
    assert e4[0x7E] == ("num", 0, Fraction(448))            # max normal
    assert e4[0xFE] == ("num", 1, Fraction(448))
    assert e4[0x08] == ("num", 0, Fraction(1, 2 ** 6))      # min normal
    assert e4[0x07] == ("num", 0, Fraction(7, 8) / 2 ** 6)  # max subnormal
    assert e4[0x01] == ("num", 0, Fraction(1, 2 ** 9))      # min subnormal
    assert e4[0x78] == ("num", 0, Fraction(256))            # S.1111.000
    assert [c for c in e4 if e4[c][0] == "nan"] == [0x7F, 0xFF]
    assert not [c for c in e4 if e4[c][0] == "inf"]
    assert e5[0x7B] == ("num", 0, Fraction(57344))
    assert e5[0x04] == ("num", 0, Fraction(1, 2 ** 14))
    assert e5[0x03] == ("num", 0, Fraction(3, 4) / 2 ** 14)
    assert e5[0x01] == ("num", 0, Fraction(1, 2 ** 16))
    assert [c for c in e5 if e5[c][0] == "inf"] == [0x7C, 0xFC]
    assert [c for c in e5 if e5[c][0] == "nan"] == [0x7D, 0x7E, 0x7F,
                                                    0xFD, 0xFE, 0xFF]
    assert e4[0x00] == ("num", 0, 0) and e4[0x80] == ("num", 1, 0)
    assert e5[0x00] == ("num", 0, 0) and e5[0x80] == ("num", 1, 0)
    # Table 2 "Dynamic range": 18 and 32 binades (max normal / min subnormal).
    assert Fraction(448) / Fraction(1, 2 ** 9) < 2 ** 18 <= \
        2 * Fraction(448) / Fraction(1, 2 ** 9)
    assert Fraction(57344) / Fraction(1, 2 ** 16) < 2 ** 32
    # Table 1: bias / emax / emin, through the descriptor.
    for name, bias, emax, emin in (("e4m3", 7, 8, -6), ("e5m2", 15, 15, -14)):
        fmt = fp_format(name)
        assert fmt.bias == bias, name
        top = _fpf_unpack_generic(fpf_max_finite(fmt), fmt)[2]
        assert _floor_log2(top) == emax, name
        assert 1 - fmt.bias == emin, name
    # Spec 1400-1405 significand widths agree with OCP's m + hidden bit.
    assert fp_format("e4m3").prec == 3 + 1 and fp_format("e5m2").prec == 2 + 1
    # OFP8_DISCLOSURE's default NaNs are NaNs, and positive.
    for name, pat in OFP8_DISCLOSURE["default_nan"].items():
        assert _ocp_ofp8_table(name)[pat][:2] == ("nan", 0), name
        assert fpf_default_nan(fp_format(name)) == pat
    # 3. The generic path against fp_unpack, exhaustively, for IEEE formats.
    for ieee in ("binary16", "bfloat16"):
        probe = replace(FP_FORMAT_TABLE[ieee], name=f"{ieee}_probe")
        for code in range(1 << 16):
            want = fp_unpack(code, 16, ieee)
            assert _fpf_unpack_generic(code, probe) == want, (ieee, code)


def check_round_eight_ofp8_round() -> None:
    """RNE into OFP8 (OCP sec. 5.2.1, Table 3), both overflow modes.

    No round-eight program rounds to OFP8 (every live accumulator is IEEE),
    so this is the only place the rule is exercised.  It is exercised here
    so the round that makes Zvvofp8mm / Zvvofp4ofp8mm live inherits a tested
    rule instead of inventing one.
    """
    for name in ("e4m3", "e5m2"):
        fmt = fp_format(name)
        sat = replace(fmt, overflow="saturate")
        table = _ocp_ofp8_table(name)
        finite = sorted({(v if s == 0 else -v, c) for c, (k, s, v)
                         in table.items() if k == "num" and v != 0})
        # Every finite code round-trips, in both modes.
        for code, (kind, sign, val) in table.items():
            if kind != "num":
                continue
            v = -val if sign else val
            if val == 0:
                assert fpf_round(v, fmt) == 0     # Fraction has no -0
                continue
            assert fpf_round(v, fmt) == code, (name, hex(code))
            assert fpf_round(v, sat) == code, (name, hex(code))
        # Midpoints between adjacent positive codes round to the even one.
        pos = [(v, c) for v, c in finite if v > 0]
        for (v0, c0), (v1, c1) in zip(pos, pos[1:]):
            mid = (v0 + v1) / 2
            even = c0 if c0 % 2 == 0 else c1
            assert fpf_round(mid, fmt) == even, (name, hex(c0), hex(c1))
            assert fpf_round(-mid, fmt) == even | 0x80
        # Below half the minimum subnormal: signed zero.  At exactly half:
        # tie to even, which is zero.  Just above half: min subnormal.
        tiny = table[0x01][2]
        assert fpf_round(tiny / 2, fmt) == 0x00
        assert fpf_round(-tiny / 2, fmt) == 0x80
        assert fpf_round(tiny / 2 + tiny / 64, fmt) == 0x01
        # Overflow (OCP Table 3, "greater than max OFP8 magnitude").
        mx_code = fpf_max_finite(fmt)
        mx = table[mx_code][2]
        ulp = mx - table[mx_code - 1][2]
        # max + half an ulp is a tie: E4M3 max 0x7E (mantissa 110) is even,
        # so it stays; E5M2 max 0x7B (mantissa 11) is odd, so the tie goes UP
        # to 2^16, above max -> overflow (NONSAT: +Inf).
        assert fpf_round(mx + ulp / 2, fmt) == (
            0x7E if name == "e4m3" else 0x7C), name
        assert fpf_round(mx + ulp / 2, sat) == mx_code, name
        big = mx + ulp                           # rounds above max
        if name == "e4m3":
            assert fpf_round(big, fmt) == 0x7F         # NONSAT -> NaN
            assert fpf_round(-big, fmt) == 0x7F        # default NaN, sign 0
        else:
            assert fpf_round(big, fmt) == 0x7C         # NONSAT -> +Inf
            assert fpf_round(-big, fmt) == 0xFC
        assert fpf_round(big, sat) == mx_code          # SAT -> max normal
        assert fpf_round(-big, sat) == mx_code | 0x80
        # Inf source (Table 3 "+-Inf" row), through the materialiser.
        assert _fpw_materialise(("inf", 0), sat) == mx_code
        assert _fpw_materialise(("inf", 1), sat) == mx_code | 0x80
        assert _fpw_materialise(("inf", 1), fmt) == (
            0x7F if name == "e4m3" else 0xFC)
        assert _fpw_materialise(("nan",), fmt) == \
            OFP8_DISCLOSURE["default_nan"][name]
    # E4M3's 464 = 1.8125 * 2^8 is a tie between 448 (mantissa 110, even)
    # and 480 (111, odd, and above max): it rounds DOWN to 448 in both modes.
    assert fpf_round(Fraction(464), fp_format("e4m3")) == 0x7E
    assert fpf_round(Fraction(480), fp_format("e4m3")) == 0x7F


def _live_geom(cell, need_groups=2, need_m=4):
    return next(g for g in fpw_legal_configs(256, full_vl_only=True)
                if (g.w, g.sew) == cell and g.emul_c != 16
                and g.lam * g.lmul >= need_groups and g.m >= need_m)


def _with_decoder(bad, fn):
    """Run *fn* with _fpf_unpack_generic rebound to *bad* (a sabotage)."""
    g = globals()
    original = g["_fpf_unpack_generic"]
    g["_fpf_unpack_generic"] = bad
    try:
        return fn()
    finally:
        g["_fpf_unpack_generic"] = original


def check_round_eight_negative_controls() -> None:
    """Every OFP8 sabotage must be caught on cases the live tiers generate.

    Round six shipped controls that could not fail, so each one here is run
    against the generators the directed suite actually uses (fpw_case for
    golden, fpw_special_case for special) on a live round-eight geometry, and
    the detection is asserted.  Where a control is provably invisible to a
    tier (the exact tier's small-integer operands), that is asserted too, so
    the report says which tier carries which check.
    """
    rng = random.Random(8008)
    e4, e5 = fp_format("e4m3"), fp_format("e5m2")
    f32 = fp_format("binary32")
    geom = _live_geom((4, 32))
    geom.validate()
    good = _fpf_unpack_generic

    def detects(bad, row, make, tries=12):
        hits = 0
        for _ in range(tries):
            a, b, c = make(geom, row, rng)
            want = fpw_reference_gemm(geom, a, b, c, *row)
            got = _with_decoder(bad, lambda: fpw_reference_gemm(
                geom, a, b, c, *row))
            hits += want != got
        return hits

    def e4m3_only(sabotage):
        def bad(bits, fmt):
            return sabotage(bits, fmt) if fmt.name == "e4m3" else \
                good(bits, fmt)
        return bad

    # 1. E4M3 <-> E5M2 swapped (the altfmt_A/B polarity inverted).
    swap = lambda bits, fmt: good(bits, e5 if fmt.name == "e4m3" else e4)
    for row in ((e4, e4, f32), (e5, e5, f32), (e4, e5, f32)):
        assert detects(swap, row, fpw_case), ("swap", row)
    # 2. E4M3 read IEEE-shaped: exponent field 15 is Inf (M=0) / NaN (M!=0),
    #    so 0x7E and 0xFE (+-448) become NaN and 0x78 (256) becomes Inf.
    ieee_e4 = replace(e4, has_inf=True, nan_rule="ieee", name="e4m3_ieee")
    as_ieee = e4m3_only(lambda bits, fmt: good(bits, ieee_e4))
    assert as_ieee(0x7E, e4)[0] == "nan" and as_ieee(0x78, e4)[0] == "inf"
    assert detects(as_ieee, (e4, e4, f32), fpw_case)
    assert detects(as_ieee, (e4, e4, f32), fpw_special_case)
    # 3. E4M3 0x7F / 0xFF read as +-Inf instead of NaN.
    def seven_f_inf(bits, fmt):
        if bits & 0x7F == 0x7F:
            return "inf", bits >> 7, Fraction(0)
        return good(bits, fmt)
    assert detects(e4m3_only(seven_f_inf), (e4, e4, f32), fpw_special_case)
    # 4. Wrong bias (off by one either way), per format.
    for fmt in (e4, e5):
        for delta in (-1, 1):
            def bad(bits, f, fmt=fmt, delta=delta):
                k, s, v = good(bits, f)
                if f.name == fmt.name and k == "num":
                    v = v * _pow2(-delta)
                return k, s, v
            assert detects(bad, (fmt, fmt, f32), fpw_case), (fmt.name, delta)
    # 5. Subnormal inputs flushed to zero.  Carried by the golden pool's
    #    edge operands; provably invisible to the exact tier (integers).
    def flush(bits, fmt):
        k, s, v = good(bits, fmt)
        if k == "num" and (bits >> (fmt.prec - 1)) & fmt.emax == 0:
            return k, s, Fraction(0)
        return k, s, v
    for fmt in (e4, e5):
        row = (fmt, fmt, f32)
        assert detects(flush, row, fpw_case), fmt.name
        assert detects(flush, row, fpw_special_case), fmt.name
        assert not detects(flush, row, fpw_exact_case), (
            f"{fmt.name}: the exact tier has only integer operands, so a "
            f"subnormal flush cannot be visible there")
    # 6. A/B formats swapped on a mixed row (E4M3 x E5M2 decoded as
    #    E5M2 x E4M3) -- the altfmt_A / altfmt_B wiring crossed.
    caught = 0
    for _ in range(12):
        a, b, c = fpw_case(geom, (e4, e5, f32), rng)
        caught += (fpw_reference_gemm(geom, a, b, c, e4, e5, f32)
                   != fpw_reference_gemm(geom, a, b, c, e5, e4, f32))
    assert caught, "A/B format swap on a mixed row has no teeth"
    # ... and on a same-format row the same swap is the identity, which is
    # why the control is scored on mixed rows only.
    a, b, c = fpw_case(geom, (e4, e4, f32), rng)
    assert fpw_reference_gemm(geom, a, b, c, e4, e4, f32) == \
        fpw_reference_gemm(geom, a, b, c, e4, e4, f32)
    # 7. Saturating vs non-saturating overflow.  No live cell rounds to
    #    OFP8, so this is scored on the reference with an OFP8 accumulator
    #    -- the Zvvofp8mm / Zvvofp4ofp8mm shape -- and recorded as such.
    for fmt in (e4, e5):
        sat = replace(fmt, overflow="saturate")
        big = fpf_max_finite(fmt)
        a = [[big] * geom.k_eff for _ in range(geom.m)]
        b = [[big] * geom.k_eff for _ in range(geom.n)]
        c = [[0] * geom.n_max for _ in range(geom.m)]
        nonsat_out = fpw_reference_gemm(geom, a, b, c, fmt, fmt, fmt)
        sat_out = fpw_reference_gemm(geom, a, b, c, fmt, fmt, sat)
        assert nonsat_out != sat_out, fmt.name
        assert sat_out[0][0] == big, (fmt.name, hex(sat_out[0][0]))
        assert nonsat_out[0][0] == (0x7F if fmt.name == "e4m3" else 0x7C)
    # And no live round-eight row has an OFP8 accumulator, which is why 7
    # cannot be scored on a program.
    for cell in FPW_ROUND_EIGHT_CELLS:
        for row in fpw_rows(*cell).values():
            assert row[2].name not in OFP_FORMATS, (cell, row[2].name)


def check_round_eight_nan_controls() -> None:
    """OFP8 special values through the reduction, and sabotages of them.

    The rules: an OFP8 NaN operand (E4M3 S.1111.111, any E5M2 S.11111.xx
    with xx != 00) yields fmt_C's *default* NaN (spec 1872-1878); E5M2 Inf
    times zero and Inf + -Inf are invalid -> default NaN (spec 1833-1845);
    E4M3 has no Inf, so +-448 stays finite and overflows only in fmt_C.
    """
    e4, e5 = fp_format("e4m3"), fp_format("e5m2")
    f16, f32 = fp_format("binary16"), fp_format("binary32")
    geom = _live_geom((4, 32))
    nan32 = fpf_default_nan(f32)
    one4 = fpf_round(Fraction(1), e4)
    one5 = fpf_round(Fraction(1), e5)
    zero = [[0] * geom.n_max for _ in range(geom.m)]

    def run(av, fa, bv, fb, fc=f32, c=None):
        a = [[av] * geom.k_eff for _ in range(geom.m)]
        b = [[bv] * geom.k_eff for _ in range(geom.n)]
        return fpw_reference_gemm(geom, a, b, c or zero, fa, fb, fc)

    for nan in (0x7F, 0xFF):
        assert run(nan, e4, one5, e5)[0][0] == nan32
    for nan in (0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF):
        assert run(one4, e4, nan, e5)[0][0] == nan32
    assert run(0x7C, e5, 0x00, e4)[0][0] == nan32          # Inf x 0
    assert run(0x7C, e5, one4, e4)[0][0] == 0x7F800000     # +Inf
    # +-448: finite.  In binary32 exactly 448*448*K_eff; in binary16 the
    # first group's product 200704 overflows to +Inf -- not NaN.
    out = run(0x7E, e4, 0x7E, e4)
    assert out[0][0] == fpf_round(Fraction(448 * 448 * geom.k_eff), f32)
    g16 = _live_geom((2, 16))
    a = [[0x7E] * g16.k_eff for _ in range(g16.m)]
    b = [[0x7E] * g16.k_eff for _ in range(g16.n)]
    c = [[0] * g16.n_max for _ in range(g16.m)]
    assert fpw_reference_gemm(g16, a, b, c, e4, e4, f16)[0][0] == 0x7C00
    # Special tier: row 0 (A NaN) and column N-1 (B NaN) are default NaN,
    # row 1 (+-448) and row 4 (subnormals) are not.
    rng = random.Random(88)
    for cell in FPW_ROUND_EIGHT_CELLS:
        sg = _live_geom(cell)
        for row in fpw_rows(*cell).values():
            a, b, c = fpw_special_case(sg, row, rng)
            out = fpw_reference_gemm(sg, a, b, c, *row)
            dn = fpf_default_nan(row[2])
            assert all(out[0][j] == dn for j in range(sg.n)), (cell, row)
            assert all(out[i][sg.n - 1] == dn for i in range(sg.m))
            if row[0].has_inf:                      # E5M2 A: Inf - Inf
                assert all(out[3][j] == dn for j in range(sg.n)), (cell, row)
    # Sabotage: payload propagated (E5M2 0xFD kept as-is, widened).  Must be
    # caught on the special tier, whose A NaN is deliberately non-canonical.
    sg = _live_geom((4, 32))
    a, b, c = fpw_special_case(sg, (e5, e5, f32), rng)
    assert a[0][0] == 0xFD
    good = fpw_reference_gemm(sg, a, b, c, e5, e5, f32)
    bad = [r[:] for r in good]
    bad[0][0] = 0xFF800000 | 0x1                      # "propagated" payload
    assert good != bad and good[0][0] == nan32
    # Sabotage: E5M2 Inf read as NaN.  Caught by row 2 of the special tier.
    orig = _fpf_unpack_generic
    def inf_as_nan(bits, fmt):
        k, s, v = orig(bits, fmt)
        return ("nan", s, v) if k == "inf" else (k, s, v)
    got = _with_decoder(inf_as_nan,
                        lambda: fpw_reference_gemm(sg, a, b, c, e5, e5, f32))
    assert got != good, "E5M2 Inf -> NaN sabotage has no teeth"


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
                  check_round_four_geometry, check_clayout_case,
                  check_round_six_encoding_map, check_round_six_e8m0,
                  check_round_six_block_intervals,
                  check_round_six_scale_layout,
                  check_round_six_exactness,
                  check_round_six_operand_bounds,
                  check_round_six_nan_exit,
                  check_round_six_negative_controls,
                  check_round_seven_format_table,
                  check_round_seven_packing,
                  check_round_seven_gemm,
                  check_round_seven_tier_overlap,
                  check_round_seven_negative_controls,
                  check_round_seven_nan_controls,
                  check_round_seven_golden_bytes,
                  check_round_seven_encoding_map,
                  check_round_seven_mx_lmul_table,
                  check_round_seven_mx_applicability,
                  check_round_seven_ocp_gate,
                  check_round_seven_geometry,
                  check_round_eight_ofp8_decode,
                  check_round_eight_ofp8_round,
                  check_round_eight_negative_controls,
                  check_round_eight_nan_controls):
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
