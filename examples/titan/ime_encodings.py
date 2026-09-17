#!/usr/bin/env python3
"""Zvvm (IME) instruction encodings, derived from the spec rather than typed in.

Nothing here hand-copies an opcode.  Every bit comes from
``specs/ime/instructions.json``, which ``specs/ime/extract_spec.py``
regenerates from the v0.9.0 asciidoc.  When the spec moves, re-run the
extractor and the self-test below tells you immediately whether an encoding
drifted -- which is the whole point.  A silent encoding drift shows up as an
unreproducible RTL bug three weeks later.

Two emission paths:

  ``asm()``     real LLVM mnemonics.  LLVM >= 23.1 assembles Zvvmm / Zvvmtls
                against the *0.1 draft*.  For the three round-one instructions
                the 0.1 and 0.9.0 encodings are bit-identical, so this is safe;
                ``check_llvm_drift()`` re-verifies that claim from the JSON.
  ``insn()``    ``.insn`` directive carrying the raw word.  Toolchain-independent
                escape hatch, and what the objdump consistency test compares
                against.

Assembling successfully does NOT mean the semantics are right: the v0.9.0 text
is 3,211 lines longer than the 0.1 draft LLVM tracks, almost all of it
arithmetic semantics and legality rules.  Semantics come from the spec, always.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_JSON = os.path.join(HERE, "specs", "ime", "instructions.json")

with open(SPEC_JSON, encoding="utf-8") as _fh:
    _SPEC = json.load(_fh)

SPEC_VERSION: str = _SPEC["version"]
INSTRUCTIONS: Dict[str, dict] = _SPEC["instructions"]

#: The three instructions round one implements.  llama.cpp's INT8 GEMM needs
#: exactly these; everything else in the family waits for round two.
ROUND_ONE = ("vmmacc.vv", "vmtl.v", "vmts.v")

#: Round two adds the quad-widening integer multiply-accumulate: Int8 x Int8
#: accumulated into Int32 (Zvvi8i32mm), which is the shape llama.cpp's INT8
#: GEMM actually wants -- round one's vmmacc.vv can only accumulate at the
#: input width.  The family also defines vwmmacc.vv (W=2) and v8wmmacc.vv
#: (W=8) with identical structure, and the signed/unsigned mix is selected by
#: vtype.altfmt_A / altfmt_B rather than by separate mnemonics (see
#: :func:`check_round_two_bits`); only plain signed vqmmacc.vv is in scope.
ROUND_TWO = ("vqmmacc.vv",)

#: Round three completes the integer Zvvmm family -- vwmmacc.vv (W=2) and
#: v8wmmacc.vv (W=8), which are the same datapath as vqmmacc.vv at a
#: different packing depth -- and adds the Zvvmttls transposing tile
#: load/store pair, vmttl.v / vmtts.v.  The transposing pair differs from
#: vmtl.v / vmts.v in exactly one line of the Sail body (the memory offset
#: expression); the register-side `tile_reg_idx` mapping is character-for-
#: character identical, which is what makes it cheap in both RTL and
#: reference model.  The floating-point Zvvfmm family and the microscaled
#: integer-input forms are deliberately *not* here: see
#: titan_runs/round3_design.md.
ROUND_THREE = ("vwmmacc.vv", "v8wmmacc.vv", "vmttl.v", "vmtts.v")

#: Everything the tile load/store and multiply-accumulate emission covers.
IMPLEMENTED = ROUND_ONE + ROUND_TWO + ROUND_THREE

#: ``lambda[2:0]`` in vtype, and the immediate lambda field of a tile
#: load/store.  Spec table "lambda[2:0] (selected lambda) encoding".
#: 0 means "no selected lambda" in vtype, and "use vtype.lambda" as an
#: instruction immediate -- two distinct meanings for the same bits.
LAMBDA_ENCODING = {0: 0b000, 1: 0b001, 2: 0b010, 4: 0b011,
                   8: 0b100, 16: 0b101, 32: 0b110, 64: 0b111}
LAMBDA_DECODING = {v: k for k, v in LAMBDA_ENCODING.items()}

# vtype layout.  The IME fields sit at the high end, immediately below vill:
#   lambda[2:0] @ vtype[XLEN-2:XLEN-4]
#   bs          @ vtype[XLEN-5]
#   altfmt_A    @ vtype[XLEN-6]
#   altfmt_B    @ vtype[XLEN-7]
# They are outside the vtypei immediate of vsetvli/vsetivli, so `vsetvl`
# (the register form) is the only architectural way to write them.
VTYPE_IME_FIELDS = {  # name -> (offset below XLEN, width)
    "lambda": (4, 3),
    "bs": (5, 1),
    "altfmt_A": (6, 1),
    "altfmt_B": (7, 1),
}


class EncodingError(ValueError):
    """Raised for an unknown instruction or an operand that does not fit."""


def _fields(name: str) -> list:
    try:
        return INSTRUCTIONS[name]["fields"]
    except KeyError:
        raise EncodingError(f"unknown instruction {name!r}") from None


def _const(name: str, lsb: int) -> int:
    return next(f["value"] for f in _fields(name)
                if f["kind"] == "const" and f["lsb"] == lsb)


def operands(name: str) -> Tuple[str, ...]:
    """Names of the variable fields of *name*, lsb-first."""
    return tuple(f["name"] for f in _fields(name) if f["kind"] == "field")


def encode(name: str, **ops: int) -> int:
    """Assemble *name* into its 32-bit instruction word.

    Operand names match the spec's wavedrom field names (``vd``, ``vs1``,
    ``vs2``, ``rs1``, ``rs2``, ``vs3``, ``vm``, ``lambda``).  Pass ``lambda``
    as the *encoded* 3-bit value; use :func:`lambda_imm` to convert.
    """
    expected = set(operands(name))
    unknown = set(ops) - expected
    if unknown:
        raise EncodingError(
            f"{name}: unknown operand(s) {sorted(unknown)}; "
            f"expected {sorted(expected)}")

    word = 0
    for f in _fields(name):
        if f["kind"] == "const":
            value = f["value"]
        else:
            if f["name"] not in ops:
                raise EncodingError(f"{name}: missing operand {f['name']!r}")
            value = ops[f["name"]]
        if not 0 <= value < (1 << f["width"]):
            raise EncodingError(
                f"{name}: {f['name'] or f['attr']}={value} does not fit "
                f"{f['width']} bits")
        word |= value << f["lsb"]
    return word


def decode(word: int) -> Tuple[str, Dict[str, int]]:
    """Inverse of :func:`encode`.  Raises if *word* matches no Zvvm encoding."""
    matches = [name for name in INSTRUCTIONS
               if all(((word >> f["lsb"]) & ((1 << f["width"]) - 1)) == f["value"]
                      for f in _fields(name) if f["kind"] == "const")]
    if not matches:
        raise EncodingError(f"no Zvvm instruction matches {word:#010x}")
    if len(matches) > 1:
        raise EncodingError(
            f"{word:#010x} is ambiguous between {matches} -- the encoding "
            f"table is not a partition, which is a spec-extraction bug")
    name = matches[0]
    return name, {f["name"]: (word >> f["lsb"]) & ((1 << f["width"]) - 1)
                  for f in _fields(name) if f["kind"] == "field"}


def lambda_imm(lam: int) -> int:
    """Architectural LAMBDA value -> 3-bit field encoding."""
    try:
        return LAMBDA_ENCODING[lam]
    except KeyError:
        raise EncodingError(
            f"LAMBDA={lam} is not encodable; must be 0 or a power of two <= 64"
        ) from None


def insn(name: str, **ops: int) -> str:
    """``.insn`` directive for *name* -- toolchain-independent emission."""
    return f".insn 4, {encode(name, **ops):#010x}"


def asm(name: str, **ops: int) -> str:
    """LLVM mnemonic form.  Requires -menable-experimental-extensions."""
    if name in ("vmmacc.vv", "vwmmacc.vv", "vqmmacc.vv", "v8wmmacc.vv"):
        # Same operand shape across the whole integer family: vd is the
        # EMUL_C-sized C group, vs1/vs2 the LMUL-sized A and B groups.  The
        # widening forms differ only in funct6 and in how many logical
        # elements vtype.SEW-wide storage carries; SEW is always the
        # *accumulator* width.
        return f"{name} v{{vd}}, v{{vs1}}, v{{vs2}}".format(**ops)
    if name in ("vmtl.v", "vmts.v", "vmttl.v", "vmtts.v"):
        reg = "vd" if name in ("vmtl.v", "vmttl.v") else "vs3"
        tail = ""
        lam = ops.get("lambda", 0)
        if lam:
            tail += f", L{LAMBDA_DECODING[lam]}"
        if not ops.get("vm", 1):
            tail += ", v0.t"
        return f"{name} v{ops[reg]}, (x{ops['rs1']}), x{ops['rs2']}{tail}"
    raise EncodingError(f"no assembly template for {name} yet "
                        f"(implemented: {IMPLEMENTED})")


#: -march for the clang route.  Experimental extensions must carry an explicit
#: version, and LLVM tracks 0.1.  Not what the build harness uses today -- it
#: assembles with riscv64-unknown-elf-gcc, which knows neither the Zvvm
#: mnemonics nor `-menable-experimental-extensions`, so the tests emit
#: :func:`insn` words instead.  See constants.MARCH_IME.
MARCH_ROUND_ONE_CLANG = "rv64gcv_zvvmm0p1_zvvmtls0p1"
CFLAGS_ROUND_ONE_CLANG = ("-menable-experimental-extensions",
                          f"-march={MARCH_ROUND_ONE_CLANG}")


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def check_table() -> None:
    """Every encoding covers 32 bits exactly once and round-trips."""
    for name, insn_def in INSTRUCTIONS.items():
        covered = 0
        for f in insn_def["fields"]:
            mask = ((1 << f["width"]) - 1) << f["lsb"]
            assert not (covered & mask), f"{name}: overlapping fields"
            covered |= mask
        assert covered == 0xFFFFFFFF, f"{name}: does not cover 32 bits"

        ops = {}
        for i, op in enumerate(operands(name)):
            width = next(f["width"] for f in insn_def["fields"]
                         if f["name"] == op)
            # a distinct in-range value per operand, so an operand swap shows
            ops[op] = (i * 3 + 1) % (1 << width)
        word = encode(name, **ops)
        got_name, got_ops = decode(word)
        assert got_name == name, f"{name} decoded as {got_name}"
        assert got_ops == ops, f"{name}: operand round-trip {got_ops} != {ops}"


def check_round_one_bits() -> None:
    """Pin the round-one encodings against the spec text by hand.

    These constants are the only bits in Titan transcribed by a human, and
    they exist solely to catch an extractor that silently produces garbage.
    Values read off the v0.9.0 wavedrom blocks.
    """
    word = encode("vmmacc.vv", vd=4, vs1=8, vs2=12)
    assert decode(word) == ("vmmacc.vv", {"vd": 4, "vs1": 8, "vs2": 12})
    assert word & 0x7F == 0x57, "vmmacc.vv major opcode must be OP-V"
    assert (word >> 12) & 0x7 == 0x0, "vmmacc.vv funct3 must be OPIVV"
    assert (word >> 25) & 0x1 == 1, "vmmacc.vv has vm=1 (vm=0 is reserved)"
    assert (word >> 26) & 0x3F == 0x38, "vmmacc.vv funct6 must be 0x38"

    load = encode("vmtl.v", vd=4, rs1=10, rs2=11, vm=1, **{"lambda": 0})
    assert load & 0x7F == 0x07, "vmtl.v must use the load major opcode"
    store = encode("vmts.v", vs3=4, rs1=10, rs2=11, vm=1, **{"lambda": 0})
    assert store & 0x7F == 0x27, "vmts.v must use the store major opcode"
    for word in (load, store):
        assert (word >> 12) & 0x7 == 0x7, "tile load/store funct3 is 0b111"
        assert (word >> 26) & 0x3 == 0x0, "order-preserving variant"
        assert (word >> 28) & 0x1 == 0x1, "tile ops set the mew bit"


def check_round_two_bits() -> None:
    """Pin vqmmacc.vv against the v0.9.0 wavedrom block, by hand.

    Spec section "vqmmacc.vv", Encoding::

        { bits: 7, name: 0x57, attr: ['OP-V'] },
        { bits: 5, name: 'vd' },
        { bits: 3, name: 0x0,  attr: ['OPIVV'] },
        { bits: 5, name: 'vs1' },
        { bits: 5, name: 'vs2' },
        { bits: 1, name: 1,    attr: ['vm=1'] },
        { bits: 6, name: 0x3a, attr: ['vqmmacc.vv'] }

    So funct6 = 0x3a, vm = 1, funct3 = OPIVV = 0, opcode = OP-V = 0x57 --
    the same shape as vmmacc.vv (funct6 0x38) with a different funct6.  The
    vm bit is *not* a don't-care here: vm=0 on this opcode decodes as
    vfqimmacc.vv, the microscaled integer-input FP-accumulate form, which is
    an entirely different instruction (see :func:`check_mx_split`).
    """
    word = encode("vqmmacc.vv", vd=24, vs1=0, vs2=8)
    assert decode(word) == ("vqmmacc.vv", {"vd": 24, "vs1": 0, "vs2": 8})
    assert word & 0x7F == 0x57, "vqmmacc.vv major opcode must be OP-V"
    assert (word >> 12) & 0x7 == 0x0, "vqmmacc.vv funct3 must be OPIVV"
    assert (word >> 25) & 0x1 == 1, "vqmmacc.vv requires vm=1"
    assert (word >> 26) & 0x3F == 0x3A, "vqmmacc.vv funct6 must be 0x3a"
    # It must not collide with the round-one multiply-accumulate.
    assert word != encode("vmmacc.vv", vd=24, vs1=0, vs2=8)

    # The rest of the integer widening family, reported but not implemented.
    for name, funct6 in (("vwmmacc.vv", 0x39), ("v8wmmacc.vv", 0x3B)):
        other = encode(name, vd=24, vs1=0, vs2=8)
        assert (other >> 26) & 0x3F == funct6, name

    # Signed/unsigned mixes are vtype.altfmt_A / altfmt_B, not mnemonics:
    # there is no vqmmaccu.vv / vqmmaccsu.vv / vqmmaccus.vv encoding to add.
    for spelling in ("vqmmaccu.vv", "vqmmaccsu.vv", "vqmmaccus.vv"):
        assert spelling not in INSTRUCTIONS, (
            f"{spelling} now exists as its own encoding -- the signedness "
            f"model in rvv_ref/ime_tests assumes altfmt selects it")


def check_round_three_bits() -> None:
    """Pin the round-three encodings against the v0.9.0 wavedrom blocks.

    Integer widening family (spec "Zvvmm", encoding table at line 1249, and
    the per-instruction wavedrom blocks at lines 7067 / 5414)::

        | `vmmacc.vv vd, vs1, vs2`   | 1 | SEW   | SEW    funct6 0x38
        | `vwmmacc.vv vd, vs1, vs2`  | 2 | SEW/2 | SEW    funct6 0x39
        | `vqmmacc.vv vd, vs1, vs2`  | 4 | SEW/4 | SEW    funct6 0x3a
        | `v8wmmacc.vv vd, vs1, vs2` | 8 | SEW/8 | SEW    funct6 0x3b

    -- one contiguous funct6 run, OPIVV, vm=1, OP-V.  vm=0 on 0x39/0x3a/0x3b
    is the microscaled integer-input FP-accumulate alias (vfwimmacc.vv,
    vfqimmacc.vv, vf8wimmacc.vv), which :func:`check_mx_split` pins and
    which round three does *not* implement.

    Transposing tile load/store (spec "vmttl.v" line 6674, "vmtts.v" line
    6830).  vmttl.v Encoding::

        { bits: 7, name: 0x07, attr: ['Vector tile (2D) load'] },
        { bits: 5, name: 'vd' },
        { bits: 3, name: 0x7 },
        { bits: 5, name: 'rs1', attr: ['base address'] },
        { bits: 5, name: 'rs2' },
        { bits: 1, name: 'vm' },
        { bits: 2, name: 0x1, attr: ['transposing'] },
        { bits: 1, name: 1 },
        { bits: 3, name: 'lambda' }

    -- byte-for-byte vmtl.v except for the two-bit field at 27:26, which is
    0b00 ("order preserving") on vmtl.v / vmts.v and 0b01 ("transposing")
    on vmttl.v / vmtts.v.  vmtts.v uses major opcode 0x27 (store), as
    vmts.v does.  That single-field difference is the whole of Zvvmttls at
    the encoding level, and is the reason a decoder change is a two-bit
    change rather than a new format.
    """
    for name, funct6 in (("vmmacc.vv", 0x38), ("vwmmacc.vv", 0x39),
                         ("vqmmacc.vv", 0x3A), ("v8wmmacc.vv", 0x3B)):
        word = encode(name, vd=24, vs1=0, vs2=8)
        assert decode(word) == (name, {"vd": 24, "vs1": 0, "vs2": 8}), name
        assert word & 0x7F == 0x57, f"{name} major opcode must be OP-V"
        assert (word >> 12) & 0x7 == 0x0, f"{name} funct3 must be OPIVV"
        assert (word >> 25) & 0x1 == 1, f"{name} requires vm=1"
        assert (word >> 26) & 0x3F == funct6, f"{name} funct6"

    tload = encode("vmttl.v", vd=4, rs1=10, rs2=11, vm=1, **{"lambda": 0})
    assert tload & 0x7F == 0x07, "vmttl.v must use the load major opcode"
    tstore = encode("vmtts.v", vs3=4, rs1=10, rs2=11, vm=1, **{"lambda": 0})
    assert tstore & 0x7F == 0x27, "vmtts.v must use the store major opcode"
    for word in (tload, tstore):
        assert (word >> 12) & 0x7 == 0x7, "tile load/store funct3 is 0b111"
        assert (word >> 26) & 0x3 == 0x1, "transposing variant"
        assert (word >> 28) & 0x1 == 0x1, "tile ops set the mew bit"

    # The transposing pair must differ from the order-preserving pair in
    # exactly the two-bit variant field, and nowhere else.
    for op_name, t_name, reg in (("vmtl.v", "vmttl.v", "vd"),
                                 ("vmts.v", "vmtts.v", "vs3")):
        ops = {reg: 4, "rs1": 10, "rs2": 11, "vm": 1, "lambda": 0}
        delta = encode(op_name, **ops) ^ encode(t_name, **ops)
        assert delta == 0x1 << 26, (
            f"{op_name}/{t_name} differ outside bits 27:26: {delta:#010x}")

    # The immediate lambda field must survive on the transposing pair too --
    # a tile load/store can override vtype.lambda, and round three's
    # generators pass 0 (= "use vtype.lambda") exactly as round one does.
    for lam in (1, 2, 4, 8, 16, 32, 64):
        word = encode("vmttl.v", vd=4, rs1=10, rs2=11, vm=1,
                      **{"lambda": lambda_imm(lam)})
        assert (word >> 29) & 0x7 == lambda_imm(lam), lam
        assert LAMBDA_DECODING[(word >> 29) & 0x7] == lam, lam


def check_llvm_drift() -> None:
    """The round-one three must be encoding-identical to what LLVM assembles.

    LLVM 23.1 implements the 0.1 draft.  Across the family, 12 of 15
    encodings are unchanged in v0.9.0; the three that moved
    (vfwmmacc.vv, vfqmmacc.vv, vf8wmmacc.vv) differ only in the meaning of
    the vm bit, which 0.1 pinned to 1 and v0.9.0 turned into a real field
    selecting unscaled (1) vs microscaled (0).  No round-one instruction is
    affected -- so real mnemonics are safe on day one.

    This checks the structural half of that claim from the JSON: round one
    must not use vm as a variable field.
    """
    fields = {f["attr"] for f in _fields("vmmacc.vv") if f["kind"] == "const"}
    assert "vm=1" in fields, (
        "vmmacc.vv: vm is no longer pinned to 1 -- the MX encoding split now "
        "reaches round one, so revisit the clang route before trusting LLVM")


def check_mx_split() -> None:
    """The vm=0 (microscaled) integer opcodes alias the vm=1 unscaled ones.

    This is the shape the v0.9.0 encoding-map split takes.  If a spec bump
    breaks the aliasing, decode() goes ambiguous and we want to know why.
    """
    pairs = [("vwmmacc.vv", "vfwimmacc.vv"),
             ("vqmmacc.vv", "vfqimmacc.vv"),
             ("v8wmmacc.vv", "vf8wimmacc.vv")]
    for unscaled, scaled in pairs:
        assert _const(unscaled, 26) == _const(scaled, 26), (
            f"{unscaled}/{scaled} no longer share funct6")
        assert (_const(unscaled, 25), _const(scaled, 25)) == (1, 0), (
            f"{unscaled}/{scaled} are no longer split by the vm bit")


def main() -> int:
    for check in (check_table, check_round_one_bits, check_round_two_bits,
                  check_round_three_bits, check_llvm_drift, check_mx_split):
        check()
        print(f"  ok  {check.__name__}")
    print(f"\nZvvm v{SPEC_VERSION}: {len(INSTRUCTIONS)} instructions, "
          f"round one = {', '.join(ROUND_ONE)}, "
          f"round two = {', '.join(ROUND_TWO)}, "
          f"round three = {', '.join(ROUND_THREE)}")
    sample = {"vd": 4, "vs3": 4, "vs1": 8, "vs2": 12, "rs1": 10, "rs2": 11,
              "vm": 1, "lambda": 0}
    for name in IMPLEMENTED:
        ops = {op: sample[op] for op in operands(name)}
        print(f"  {name:12s} {encode(name, **ops):#010x}   {asm(name, **ops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
