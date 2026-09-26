#!/usr/bin/env python3
"""Execute the generated directed programs, without a RISC-V toolchain.

The Day-3 checkpoint of the build plan says: if the reference path does not
run on an unmodified Saturn, stop -- the cost of a wrong understanding of the
tile geometry is a few hours now and a whole 60-iteration run later.  Saturn
needs a cluster.  This is the part of that check that can be done anywhere.

It is a small RV64 interpreter that understands exactly the instructions
``ime_tests.py`` emits, plus a model of the three IME instructions written
straight from the Zvvm v0.9.0 pseudocode.  It loads a generated program,
runs it, and reads the verdict the program prints about itself.

**What a pass here does and does not prove.**  It proves the *emission* is
right: the data buffers are laid out where the tile loads will look for them,
the vtype words carry the fields the spec puts at those bit positions, the
register allocation does not overlap, the branch targets and stack discipline
are sound, and the RVV reference sequence computes what the IME path
computes.  Those are where the bugs have actually been.

It does not independently prove the *reading* of the spec, because the IME
model here and the layout functions in rvv_ref.py share an author.  That
reading is cross-checked separately, by deriving each geometry rule twice by
independent routes in rvv_ref's own self-test.  Real independent confirmation
arrives only when the RTL runs.

    python sim_check.py                 # sweep every round-one geometry
    python sim_check.py --emit 32,2,1,4 # run one, verbosely
"""
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fractions import Fraction

import ime_encodings as ime
import ime_tests
import rvv_ref
from rvv_ref import TileGeometry

MEM_BASE = 0x8000_0000
MEM_SIZE = 1 << 22
STACK_TOP = MEM_BASE + MEM_SIZE - 64

_ABI = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
    **{f"a{i}": 10 + i for i in range(8)},
    **{f"s{i}": 16 + i for i in range(2, 12)},
    **{f"t{i}": 25 + i for i in range(3, 7)},
    **{f"x{i}": i for i in range(32)},
}

#: Floating-point register names, for the round-four reference path.  Only
#: the ABI names the generator actually emits, plus the fN forms.
_FABI = {
    **{f"ft{i}": i for i in range(8)},
    **{f"fs{i}": 8 + i for i in range(2)},
    **{f"fa{i}": 10 + i for i in range(8)},
    **{f"fs{i}": 16 + i for i in range(2, 12)},
    **{f"ft{i}": 25 + i for i in range(3, 9)},
    **{f"f{i}": i for i in range(32)},
}

#: Scalar FP mnemonic suffix -> element width.  ``.s`` is binary32, ``.d``
#: binary64; there is no ``.h`` in the baseline -march, which is why round
#: four's floating-point tier stops at SEW >= 32.
_FP_WIDTH = {"s": 32, "d": 64}


def _sext64(value: int) -> int:
    """Interpret a 64-bit register as signed.

    The register file holds Python ints that may already be negative (an
    ``li`` of a negative immediate stores one directly), so the value is
    masked into 64 bits before the sign test.  Without the mask, Python's
    arithmetic shift on a negative int reports "sign bit set" for every
    negative number and subtracts 2**64 from an already-signed value -- which
    turned every exact-tier expectation into -2**64 and looked like a
    conversion bug in the generator rather than one here.
    """
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >> 63 else value

_DIRECTIVE_WIDTH = {".byte": 1, ".half": 2, ".word": 4, ".dword": 8}


class IllegalInstruction(Exception):
    """The architecture says this instruction traps here.

    Distinct from :class:`SimError`, which means *the model does not know*.
    Conflating the two is how a legality tier ends up reporting "not
    modelled" as "correctly rejected"; keeping them apart is what lets the
    ``ime_mxl_`` programs be judged at all.
    """


class SimError(RuntimeError):
    """The program did something this interpreter refuses to guess about."""


def _sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _zext(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


@dataclass
class Machine:
    vlen: int
    x: List[int] = field(default_factory=lambda: [0] * 32)
    mem: bytearray = field(default_factory=lambda: bytearray(MEM_SIZE))
    v: List[bytearray] = field(default_factory=list)
    vtype: int = 0
    vl: int = 0
    mstatus: int = 0
    #: The scalar floating-point register file, held as raw bit patterns
    #: rather than host floats: NaN payloads, signed zeros and subnormals all
    #: have to survive a load/store round trip unchanged, and the harness
    #: compares bits.  Arithmetic goes through rvv_ref's exact model, so the
    #: host's float type is never in the loop.
    f: List[int] = field(default_factory=lambda: [0] * 32)
    #: fcsr.frm.  The reset value is 0 (RNE) and the emitted programs set it
    #: explicitly with `csrwi frm, 0` anyway; anything else would need a
    #: rounding mode argument threaded through rvv_ref.fp_round, which round
    #: four does not generate.  See rvv_ref.FP_FRM.
    frm: int = 0
    #: M-mode trap state.  The ``ime_mxl_`` legality programs install their
    #: own handler, so the model has to deliver a trap rather than abort.
    mtvec: int = 0
    mepc: int = 0
    mcause: int = 0
    stdout: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.v:
            self.v = [bytearray(self.vlen // 8) for _ in range(32)]
        self.x[2] = STACK_TOP

    # --- scalar memory ---
    def load(self, addr: int, width: int, signed: bool = True) -> int:
        off = addr - MEM_BASE
        if not 0 <= off <= MEM_SIZE - width:
            raise SimError(f"load of {width}B at {addr:#x} is out of range")
        raw = int.from_bytes(self.mem[off:off + width], "little")
        return _sext(raw, width * 8) if signed else raw

    def store(self, addr: int, width: int, value: int) -> None:
        off = addr - MEM_BASE
        if not 0 <= off <= MEM_SIZE - width:
            raise SimError(f"store of {width}B at {addr:#x} is out of range")
        self.mem[off:off + width] = _zext(value, width * 8).to_bytes(width,
                                                                    "little")

    # --- vector register groups, addressed as a flat element sequence ---
    @property
    def elems_per_reg_bytes(self) -> int:
        return self.vlen // 8

    def vget(self, base: int, index: int, eew: int) -> int:
        per_reg = self.vlen // eew
        reg, pos = base + index // per_reg, index % per_reg
        if reg > 31:
            raise SimError(f"vector group v{base} + {index} runs past v31")
        if eew < 8:
            # Sub-byte elements.  Round six is the first family to need
            # them: vfqimmacc.vv at SEW=16 and vf8wimmacc.vv at SEW=32 have
            # EEW_A = SEW/W = 4 (MXINT4).  Spec 1206-1219 fixes the packing
            # -- two per byte, the *even* index in the LOW nibble -- and
            # getting that order backwards silently transposes every pair of
            # K elements, which reads as an arithmetic bug rather than a
            # layout one.  Only eew=4 occurs; anything narrower would need a
            # bit-addressed path and is rejected rather than guessed at.
            if eew != 4:
                raise SimError(f"EEW={eew} is not modelled")
            byte = self.v[reg][pos // 2]
            return (byte >> (4 * (pos % 2))) & 0xF
        width = eew // 8
        return int.from_bytes(self.v[reg][pos * width:(pos + 1) * width],
                              "little")

    def vset(self, base: int, index: int, eew: int, value: int) -> None:
        per_reg = self.vlen // eew
        reg, pos = base + index // per_reg, index % per_reg
        if reg > 31:
            raise SimError(f"vector group v{base} + {index} runs past v31")
        if eew < 8:
            if eew != 4:
                raise SimError(f"EEW={eew} is not modelled")
            shift = 4 * (pos % 2)
            byte = self.v[reg][pos // 2] & ~(0xF << shift) & 0xFF
            self.v[reg][pos // 2] = byte | ((value & 0xF) << shift)
            return
        width = eew // 8
        self.v[reg][pos * width:(pos + 1) * width] = \
            _zext(value, eew).to_bytes(width, "little")

    # --- vtype decoding, IME fields included ---
    @property
    def sew(self) -> int:
        return 8 << ((self.vtype >> 3) & 0x7)

    @property
    def lmul(self) -> int:
        return 1 << (self.vtype & 0x7)

    @property
    def lam(self) -> int:
        offset, width = ime.VTYPE_IME_FIELDS["lambda"]
        code = (self.vtype >> (64 - offset)) & ((1 << width) - 1)
        return ime.LAMBDA_DECODING[code]

    @property
    def bs(self) -> int:
        """vtype.bs -- the microscaling block size selector (spec 1160-1176).

        An IME field, so it is keyed by offset below XLEN like `lambda`, and
        like `lambda` it is unreachable from vsetvli: the emitted programs
        write it through the register form.
        """
        offset, width = ime.VTYPE_IME_FIELDS["bs"]
        return (self.vtype >> (64 - offset)) & ((1 << width) - 1)

    @property
    def altfmt(self) -> int:
        """vtype.altfmt -- the C accumulator format selector.

        NOT an IME field.  Spec 856-861 makes it the *base* altfmt defined by
        Zvfbfa, which is keyed by an absolute lsb rather than by an offset
        below XLEN; conflating the two keyings would silently read the wrong
        bit at XLEN != 64.  See ime_encodings.VTYPE_BASE_FIELDS.
        """
        lsb, width = ime.VTYPE_BASE_FIELDS["altfmt"]
        return (self.vtype >> lsb) & ((1 << width) - 1)

    @property
    def altfmt_ab(self) -> Tuple[int, int]:
        """(vtype.altfmt_A, vtype.altfmt_B), the *input* format selectors."""
        out = []
        for name in ("altfmt_A", "altfmt_B"):
            offset, width = ime.VTYPE_IME_FIELDS[name]
            out.append((self.vtype >> (64 - offset)) & ((1 << width) - 1))
        return out[0], out[1]

    @property
    def vlmax(self) -> int:
        return self.lmul * self.vlen // self.sew


# ---------------------------------------------------------------------------
# assembling the generated source into something executable
# ---------------------------------------------------------------------------

@dataclass
class Program:
    text: List[Tuple[str, List[str]]]        # (mnemonic, operands)
    labels: Dict[str, int]                   # text label -> instruction index
    numeric: Dict[str, List[int]]            # "1" -> every position of `1:`
    data: Dict[str, int]                     # data label -> address
    image: bytes
    data_base: int


def assemble(source: str) -> Program:
    """Turn generated assembly into an instruction list and a data image.

    Only the directives and mnemonics ime_tests.py actually emits are
    handled; anything else raises rather than being silently skipped, so a
    change to the generator cannot quietly stop being simulated.
    """
    in_data = False
    text: List[Tuple[str, List[str]]] = []
    labels: Dict[str, int] = {}
    numeric: Dict[str, List[int]] = {}
    data_labels: Dict[str, int] = {}
    image = bytearray()

    for raw in source.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line == ".text":
            in_data = False
            continue
        if line == ".data":
            in_data = True
            continue

        # A label may or may not be alone on its line.
        m = re.match(r"^([.\w]+):\s*(.*)$", line)
        if m:
            name, line = m.group(1), m.group(2).strip()
            if name.isdigit():
                # Local numeric labels repeat; `1f` means the next one
                # forward, so every position has to be kept.
                numeric.setdefault(name, []).append(len(text))
            elif in_data:
                data_labels[name] = len(image)
            else:
                labels[name] = len(text)
            if not line:
                continue

        if line.startswith("."):
            directive, _, rest = line.partition(" ")
            rest = rest.strip()
            if directive == ".balign":
                if in_data:
                    image.extend(b"\0" * ((-len(image)) % int(rest, 0)))
            elif directive in _DIRECTIVE_WIDTH:
                if not in_data:
                    raise SimError(f"{directive} outside .data: {line}")
                width = _DIRECTIVE_WIDTH[directive]
                for token in rest.split(","):
                    image.extend(_zext(int(token.strip(), 0), width * 8)
                                 .to_bytes(width, "little"))
            elif directive == ".zero":
                image.extend(b"\0" * int(rest, 0))
            elif directive == ".asciz":
                body = rest.strip()[1:-1]  # drop the surrounding quotes
                image.extend(body.encode().decode("unicode_escape").encode()
                             + b"\0")
            elif directive == ".insn":
                text.append((".insn", [t.strip() for t in rest.split(",")]))
            elif directive in (".globl", ".section", ".option"):
                pass
            else:
                raise SimError(f"unhandled directive: {line}")
            continue

        mnemonic, _, rest = line.partition(" ")
        operands = [t.strip() for t in rest.split(",")] if rest.strip() else []
        text.append((mnemonic, operands))

    data_base = MEM_BASE + (1 << 20)  # clear of code and of the stack
    return Program(text, labels, numeric,
                   {k: data_base + v for k, v in data_labels.items()},
                   bytes(image), data_base)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

#: The M-mode CSRs the legality programs touch.  Deliberately a short list:
#: anything else still raises "not modelled" rather than silently reading 0.
_M_CSRS = ("mtvec", "mepc", "mcause")

#: Bytes per instruction, for the one place a program observes a text
#: address: mepc/mtvec in the ime_mxl_ legality tier.  Every IME and RVV
#: instruction the generated programs emit is 4 bytes (no compressed
#: encodings -- the harness assembles with plain rv64imafdv), so the mapping
#: between this model's text index and the architectural address is exact.
_TEXT_STRIDE = 4

_MEM_WIDTH = {"lb": 1, "lh": 2, "lw": 4, "ld": 8,
              "lbu": 1, "lhu": 2, "lwu": 4,
              "sb": 1, "sh": 2, "sw": 4, "sd": 8}

#: The zero-extending loads.  The compare path uses the signed forms; the
#: failure-evidence path uses these, because a printed element must not be
#: sign-extended into sixteen leading f's.
_MEM_UNSIGNED = {"lbu", "lhu", "lwu"}


def _reg(token: str) -> int:
    token = token.strip()
    if token in _ABI:
        return _ABI[token]
    raise SimError(f"not a register: {token!r}")


def _freg(token: str) -> int:
    try:
        return _FABI[token]
    except KeyError:
        raise SimError(f"not a floating-point register: {token!r}") from None


def _vreg(token: str) -> int:
    m = re.fullmatch(r"v(\d+)", token.strip())
    if not m:
        raise SimError(f"not a vector register: {token!r}")
    return int(m.group(1))


def run(program: Program, machine: Machine, limit: int = 20_000_000) -> int:
    """Execute from ``main`` until it returns.  Returns a0."""
    mem_off = program.data_base - MEM_BASE
    machine.mem[mem_off:mem_off + len(program.image)] = program.image

    pc = program.labels.get("main")
    if pc is None:
        raise SimError("no main label")
    depth = 0
    steps = 0

    while True:
        if steps > limit:
            raise SimError("instruction limit exceeded (infinite loop?)")
        steps += 1
        if pc >= len(program.text):
            raise SimError("fell off the end of .text")
        mnemonic, ops = program.text[pc]
        faulting_pc = pc
        pc += 1
        x = machine.x

        if mnemonic == "li":
            x[_reg(ops[0])] = _sext(int(ops[1], 0), 64)
        elif mnemonic == "la":
            if ops[1] in program.data:
                x[_reg(ops[0])] = program.data[ops[1]]
            elif ops[1] in program.labels:
                # A *text* label -- the ime_mxl_ programs take the address of
                # their trap handler.  This model steps a list of decoded
                # instructions rather than bytes, so a text address is the
                # index scaled by 4: the handler does `addi t0, t0, 4` on
                # mepc to step over the faulting instruction, and that has to
                # mean one instruction here too.  _TEXT_STRIDE is the single
                # place that convention is written down.
                x[_reg(ops[0])] = program.labels[ops[1]] * _TEXT_STRIDE
            else:
                raise SimError(f"la: unknown label {ops[1]!r}")
        elif mnemonic == "mv":
            x[_reg(ops[0])] = x[_reg(ops[1])]
        elif mnemonic == "add":
            x[_reg(ops[0])] = _sext(x[_reg(ops[1])] + x[_reg(ops[2])], 64)
        elif mnemonic == "sub":
            # Round six's int_to_fp bit-splice negates an exponent with
            # `sub t1, x0, rs`; the tier is integer-only by construction, so
            # this is the one scalar op it needed that rounds one to five
            # never emitted.
            x[_reg(ops[0])] = _sext(x[_reg(ops[1])] - x[_reg(ops[2])], 64)
        elif mnemonic == "addi":
            x[_reg(ops[0])] = _sext(x[_reg(ops[1])] + int(ops[2], 0), 64)
        elif mnemonic == "srli":
            x[_reg(ops[0])] = _zext(x[_reg(ops[1])], 64) >> int(ops[2], 0)
        elif mnemonic == "slli":
            x[_reg(ops[0])] = _sext(x[_reg(ops[1])] << int(ops[2], 0), 64)
        elif mnemonic == "sll":
            x[_reg(ops[0])] = _sext(
                x[_reg(ops[1])] << (x[_reg(ops[2])] & 63), 64)
        elif mnemonic == "andi":
            x[_reg(ops[0])] = x[_reg(ops[1])] & int(ops[2], 0)
        elif mnemonic in ("beq", "bne", "blt", "bge"):
            a, b = x[_reg(ops[0])], x[_reg(ops[1])]
            taken = {"beq": a == b, "bne": a != b,
                     "blt": a < b, "bge": a >= b}[mnemonic]
            if taken:
                pc = _resolve(program, ops[2], pc)
        elif mnemonic in ("beqz", "bnez"):
            a = x[_reg(ops[0])]
            if (a == 0) if mnemonic == "beqz" else (a != 0):
                pc = _resolve(program, ops[1], pc)
        elif mnemonic == "j":
            pc = _resolve(program, ops[0], pc)
        elif mnemonic == "call":
            if ops[0] != "printf":
                raise SimError(f"call to {ops[0]}: only printf is modelled")
            machine.stdout.append(_printf(machine))
        elif mnemonic == "ret":
            if depth == 0:
                return _sext(x[10], 64)
            depth -= 1
        elif mnemonic == "csrs":
            if ops[0] != "mstatus":
                raise SimError(f"csrs {ops[0]} is not modelled")
            machine.mstatus |= x[_reg(ops[1])]
        elif mnemonic == "csrr":
            if ops[1] == "vtype":
                x[_reg(ops[0])] = machine.vtype
            elif ops[1] in _M_CSRS:
                x[_reg(ops[0])] = getattr(machine, ops[1])
            else:
                raise SimError(f"csrr {ops[1]} is not modelled")
        elif mnemonic == "csrw":
            if ops[0] not in _M_CSRS:
                raise SimError(f"csrw {ops[0]} is not modelled")
            setattr(machine, ops[0], x[_reg(ops[1])])
        elif mnemonic == "mret":
            # The handler advances mepc past the faulting instruction itself
            # (`addi t0, t0, 4`), so mepc already points where execution
            # resumes.  See _TEXT_STRIDE.
            pc = machine.mepc // _TEXT_STRIDE
        elif mnemonic in _MEM_WIDTH:
            width = _MEM_WIDTH[mnemonic]
            m = re.fullmatch(r"(-?\d+)\((\w+)\)", ops[1])
            if not m:
                raise SimError(f"bad memory operand: {ops[1]!r}")
            addr = x[_reg(m.group(2))] + int(m.group(1))
            if mnemonic.startswith("l"):
                x[_reg(ops[0])] = machine.load(
                    addr, width, signed=mnemonic not in _MEM_UNSIGNED)
            else:
                machine.store(addr, width, x[_reg(ops[0])])
        elif mnemonic == "vsetvl":
            _vsetvl(machine, ops)
        elif mnemonic == "vsetvli":
            _vsetvli(machine, ops)
        elif mnemonic.startswith("vle"):
            _vle(machine, mnemonic, ops)
        elif mnemonic.startswith("vse"):
            _vse(machine, mnemonic, ops)
        elif mnemonic == "vmul.vv":
            _vmul(machine, ops)
        elif mnemonic == "vmv.s.x":
            machine.vset(_vreg(ops[0]), 0, machine.sew, x[_reg(ops[1])])
        elif mnemonic == "vmv.x.s":
            x[_reg(ops[0])] = _sext(machine.vget(_vreg(ops[1]), 0,
                                                 machine.sew), machine.sew)
        elif mnemonic == "vredsum.vs":
            _vredsum(machine, ops)
        elif mnemonic == "csrwi":
            # The only CSR the generator writes immediately is frm, and the
            # only value it writes is 0 (RNE).  Anything else would mean the
            # two paths of a program disagree about the rounding mode, which
            # is precisely the bug this refuses to model silently.
            if ops[0] != "frm":
                raise SimError(f"csrwi {ops[0]} is not modelled")
            machine.frm = int(ops[1], 0)
            if machine.frm != rvv_ref.FP_FRM:
                raise SimError(
                    f"csrwi frm, {machine.frm}: rvv_ref's reference model "
                    f"implements round-to-nearest-even only")
        elif mnemonic == "fsrmi":
            # Round seven sets frm explicitly rather than trusting the reset
            # value (spec 1495).  Only RNE is modelled, and anything else is
            # refused rather than silently ignored -- a program that asked
            # for a rounding mode this harness does not implement must not
            # appear to have been judged under it.
            machine.frm = int(ops[0], 0)
            if machine.frm != rvv_ref.FP_FRM:
                raise SimError(
                    f"fsrmi {machine.frm}: rvv_ref's reference model "
                    f"implements round-to-nearest-even only")
        elif mnemonic.startswith("fcvt.") and len(ops) == 2:
            # Integer -> float, the exact tier's one conversion.  Goes
            # through rvv_ref.fp_round so the simulator and the reference
            # model cannot disagree about it.
            _dst, src = mnemonic.split(".")[1:3]
            if src not in ("l", "w"):
                raise SimError(f"{mnemonic} is not modelled")
            try:
                width = _FP_WIDTH[_dst]
            except KeyError:
                raise SimError(f"{mnemonic} is not modelled") from None
            val = _sext64(x[_reg(ops[1])])
            sign = 1 if val < 0 else 0
            machine.f[_freg(ops[0])] = rvv_ref.fp_round(
                sign, Fraction(abs(val)), width)
        elif mnemonic in ("flw", "fld", "fsw", "fsd"):
            # Raw bit moves: no canonicalisation, no host float anywhere, so
            # a signalling NaN or a -0.0 survives the round trip.
            width = 4 if mnemonic[-1] == "w" else 8
            m = re.fullmatch(r"(-?\d+)\((\w+)\)", ops[1])
            if not m:
                raise SimError(f"bad memory operand: {ops[1]!r}")
            addr = x[_reg(m.group(2))] + int(m.group(1))
            if mnemonic[1] == "l":
                machine.f[_freg(ops[0])] = machine.load(addr, width,
                                                        signed=False)
            else:
                machine.store(addr, width, machine.f[_freg(ops[0])])
        elif mnemonic.split(".")[0] in ("fmul", "fadd") and len(ops) == 3:
            # Sail fp_mul / fp_add at frm: one exact operation, one rounding,
            # canonical NaN out (spec 1870-1872, 1880-1886).  rvv_ref is the
            # single authority, so the model here and the reference the
            # directed program is judged against cannot drift apart.
            op, suffix = mnemonic.split(".")
            try:
                width = _FP_WIDTH[suffix]
            except KeyError:
                raise SimError(f"{mnemonic} is not modelled") from None
            fn = rvv_ref.fp_mul if op == "fmul" else rvv_ref.fp_add
            machine.f[_freg(ops[0])] = fn(machine.f[_freg(ops[1])],
                                          machine.f[_freg(ops[2])], width)
        elif mnemonic == ".insn":
            try:
                _ime(machine, int(ops[1], 0))
            except IllegalInstruction:
                # Deliver an M-mode illegal-instruction trap (mcause 2) to
                # the handler the program installed.  Without this the
                # ime_mxl_ legality tier cannot be simulated at all, and an
                # untestable tier is how a judge bug survives.
                machine.mcause = 2
                machine.mepc = faulting_pc * _TEXT_STRIDE
                if not machine.mtvec:
                    raise SimError(
                        "illegal instruction with no handler installed")
                pc = machine.mtvec // _TEXT_STRIDE
        else:
            raise SimError(f"unhandled instruction: {mnemonic} {ops}")

        x[0] = 0


def _resolve(program: Program, target: str, pc: int) -> int:
    if re.fullmatch(r"\d+f", target):  # local numeric label, next one forward
        for index in program.numeric.get(target[:-1], ()):
            if index >= pc:
                return index
        raise SimError(f"no {target[:-1]}: label forward of instruction {pc}")
    if target in program.labels:
        return program.labels[target]
    raise SimError(f"unresolved label {target!r}")


def _printf(machine: Machine) -> str:
    addr = machine.x[10]
    raw = bytearray()
    while True:
        byte = machine.load(addr, 1, signed=False)
        if byte == 0:
            break
        raw.append(byte)
        addr += 1
    text = raw.decode()
    # a1..a7: the failure-evidence formats take up to six arguments (row,
    # column, and a 32-bit hex pair per value at SEW=64).
    args = [machine.x[r] for r in range(11, 18)]
    out, index = [], 0
    i = 0
    while i < len(text):
        # %d, %x and %0Nx -- the conversions the emitted formats use.  No
        # %llx anywhere: newlib-nano is built without long-long support, so
        # a 64-bit element is printed as two zero-padded 32-bit halves.
        m = re.compile(r"%(0(\d+))?([dx])").match(text, i)
        if m:
            value = _zext(args[index], 64) if m.group(3) == "x" \
                else args[index]
            body = format(value, "x") if m.group(3) == "x" else str(value)
            if m.group(2):
                body = body.rjust(int(m.group(2)), "0")
            out.append(body)
            index += 1
            i = m.end()
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _warl_lambda(vlen: int, sew: int, requested: int, current: int) -> int:
    """The lambda a vtype write selects -- spec 1034-1062, full support.

    The reference model supports every architecturally permissible lambda
    (rvv_ref.permissible_lambdas), so a permissible request is retained and
    anything else is clamped exactly as the write rules say.  Before r26 this
    model kept whatever was written, which is how the ime_mxl_ tier could
    demand lambda=1 back at SEW=8/VLEN=256 (EMUL_C=32) and still pass here.
    """
    supported = rvv_ref.permissible_lambdas(vlen, sew)
    if not supported:
        return 0                      # outside the IME-legal domain
    if requested == 0:                # preserve-or-initialize
        return current if current in supported else max(supported)
    if requested in supported:
        return requested
    lower = [l for l in supported if l <= requested]
    return max(lower) if lower else min(supported)


#: The vtype-write lambda selection.  A module global so the negative
#: control in check_mxl_warl_is_load_bearing can swap in a model that keeps
#: the requested value verbatim.
_WARL_LAMBDA = _warl_lambda


def _vsetvl(machine: Machine, ops: List[str]) -> None:
    avl = machine.x[_reg(ops[1])]
    vtype = machine.x[_reg(ops[2])]
    offset, width = ime.VTYPE_IME_FIELDS["lambda"]
    shift, mask = 64 - offset, (1 << width) - 1
    code = (vtype >> shift) & mask
    sew = 8 << ((vtype >> 3) & 0x7)
    lam = _WARL_LAMBDA(machine.vlen, sew, ime.LAMBDA_DECODING[code],
                       machine.lam)
    vtype = (vtype & ~(mask << shift)) | (
        (ime.lambda_imm(lam) if lam else 0) << shift)
    machine.vtype = vtype
    machine.vl = min(avl, machine.vlmax)
    if ops[0] not in ("x0", "zero"):
        machine.x[_reg(ops[0])] = machine.vl


def _vsetvli(machine: Machine, ops: List[str]) -> None:
    avl = machine.x[_reg(ops[1])]
    sew = int(ops[2][1:])
    lmul = int(ops[3][1:])
    vta = 1 if "ta" in ops else 0
    vma = 1 if "ma" in ops else 0
    # vsetvli cannot write the IME fields; they retain their previous values.
    keep = machine.vtype & ~0xFF
    machine.vtype = keep | ({8: 0, 16: 1, 32: 2, 64: 3}[sew] << 3) \
        | {1: 0, 2: 1, 4: 2, 8: 3}[lmul] | (vta << 6) | (vma << 7)
    machine.vl = min(avl, machine.vlmax)
    if ops[0] not in ("x0", "zero"):
        machine.x[_reg(ops[0])] = machine.vl


def _vle(machine: Machine, mnemonic: str, ops: List[str]) -> None:
    eew = int(re.fullmatch(r"vle(\d+)\.v", mnemonic).group(1))
    m = re.fullmatch(r"\((\w+)\)", ops[1])
    if not m:
        raise SimError(f"bad vector load operand: {ops[1]!r}")
    addr = machine.x[_reg(m.group(1))]
    width = eew // 8
    for i in range(machine.vl):
        machine.vset(_vreg(ops[0]), i, eew,
                     machine.load(addr + i * width, width, signed=False))


def _vse(machine: Machine, mnemonic: str, ops: List[str]) -> None:
    """Architectural unit-stride vector store of a whole register group.

    The round-five C-layout probe's read-back path, and the reason it can
    see what the pair tiers cannot: element p of the group goes to byte
    p*EEW/8, with no tile indexing anywhere.  This is the plain RVV 1.0
    definition -- ``vse<EEW>.v vs3, (rs1)`` -- and it spans the LMUL-sized
    group exactly as :meth:`Machine.vget` walks it.
    """
    eew = int(re.fullmatch(r"vse(\d+)\.v", mnemonic).group(1))
    m = re.fullmatch(r"\((\w+)\)", ops[1])
    if not m:
        raise SimError(f"bad vector store operand: {ops[1]!r}")
    addr = machine.x[_reg(m.group(1))]
    width = eew // 8
    for i in range(machine.vl):
        machine.store(addr + i * width, width,
                      machine.vget(_vreg(ops[0]), i, eew))


def _vmul(machine: Machine, ops: List[str]) -> None:
    sew = machine.sew
    vd, vs1, vs2 = (_vreg(o) for o in ops)
    for i in range(machine.vl):
        a = machine.vget(vs1, i, sew)
        b = machine.vget(vs2, i, sew)
        machine.vset(vd, i, sew, a * b)


def _vredsum(machine: Machine, ops: List[str]) -> None:
    sew = machine.sew
    vd, vs2, vs1 = (_vreg(o) for o in ops)
    acc = machine.vget(vs1, 0, sew)
    for i in range(machine.vl):
        acc += machine.vget(vs2, i, sew)
    machine.vset(vd, 0, sew, acc)


# ---------------------------------------------------------------------------
# the three IME instructions, from the v0.9.0 pseudocode
# ---------------------------------------------------------------------------

#: Multiply-accumulate mnemonic -> widening factor W.  The Sail body of each
#: instruction opens with `decode_gemm_geometry(W)` for exactly this W.
_MACC_W = {"vmmacc.vv": 1, "vwmmacc.vv": 2,
           "vqmmacc.vv": 4, "v8wmmacc.vv": 8}

#: Tile transfer mnemonic -> whether it transposes.  The two variants share
#: every other line of their Sail bodies, register side included.
_TILE_LS = {"vmtl.v": False, "vmts.v": False,
            "vmttl.v": True, "vmtts.v": True}

#: Floating-point multiply-accumulate mnemonic -> widening factor W.  Round
#: four implements the W=1 entry only; the widening FP forms are absent
#: rather than present-and-approximate, so a program that reached one would
#: raise SimError instead of quietly scoring itself against a model nobody
#: derived.  See titan_runs/round4_design.md.
_FP_MACC_W = {"vfmmacc.vv": 1}

#: Round seven.  Widening floating-point mnemonic -> W, for the cells whose
#: element formats the IME specification defines.  ``vf8wmmacc.vv`` is
#: absent because every one of its encoding-map cells takes an OFP input
#: format (spec 7362-7370) and those are declared unsupported -- see
#: constants.ROUND_SEVEN_SUPPORTED.  A program that reached an unmodelled
#: cell raises SimError rather than scoring itself against a model nobody
#: derived.
#:
#: Round eight adds ``vf8wmmacc.vv``: its (W=8, SEW=64) cell takes OFP8
#: inputs, which the OCP OFP8 v1.0 document now defines.  Its other cell,
#: (W=8, SEW=32), takes E2M1 and still raises SimError below.
_FPW_MACC_W = {"vfwmmacc.vv": 2, "vfqmmacc.vv": 4, "vf8wmmacc.vv": 8}

#: Round six.  Microscaled integer-input, FP-accumulate mnemonic -> W.
#:
#: These ride the *integer* funct6 run: 0x39/0x3a/0x3b at vm=1 decode as
#: vwmmacc.vv / vqmmacc.vv / v8wmmacc.vv and at vm=0 as these three (Sail
#: 5963-5975, 6196-6205, 6082-6091).  ime_encodings.decode already resolves
#: the vm split, so this table is keyed on the resolved mnemonic and the
#: model never has to look at vm itself -- which is also why a decoder that
#: mis-routed vm would show up here as the *wrong arithmetic*, not as an
#: unknown instruction.
_MX_MACC_W = {"vfwimmacc.vv": 2, "vfqimmacc.vv": 4, "vf8wimmacc.vv": 8}


#: The three Sail index functions the model uses, held as rebindable module
#: globals rather than called through ``rvv_ref.`` directly.  They are the
#: *implementation's* choice of layout, and the negative controls replace
#: them to build a deliberately wrong DUT -- which must stay a property of
#: the model, never of rvv_ref: rvv_ref is what the generated programs are
#: judged against, and sabotaging it would move the reference and the
#: defendant together, which is exactly the failure mode round five exists
#: to close.
_C_INDEX = rvv_ref.c_element_index
_AB_INDEX = rvv_ref.ab_element_index
_TILE_REG_IDX = rvv_ref.tile_reg_idx

#: Round six's two sabotageable pieces, held as globals for the same reason.
#: ``_MX_PAIR_INDEX`` is where the v0 paired-scale layout is decided
#: (spec 2161-2170); ``_MX_BLOCK_DOT`` is where the Sail's "unbounded
#: mathematical integer, no overflow" (5128-5130) is either honoured or
#: quietly turned back into the round-one modular habit.  Those are the two
#: mistakes this round is most likely to make.
_MX_PAIR_INDEX = rvv_ref.mx_pair_index


def _mx_block_dot(machine: Machine, vs1: int, vs2: int, i: int, j: int,
                  k_lo: int, k_hi: int, geom: TileGeometry,
                  eew_ab: int) -> int:
    """Sail ``int_block_dot`` over one microscaling block, exactly.

    Signed regardless of altfmt_A/altfmt_B: Sail 5397 passes the literals
    ``true``/``true``.  No modular reduction anywhere -- that is the whole
    difference from ``int_gemm``.
    """
    total = 0
    for k in range(k_lo, k_hi + 1):
        a = machine.vget(vs1, _AB_INDEX(i, k, geom), eew_ab)
        b = machine.vget(vs2, _AB_INDEX(j, k, geom), eew_ab)
        total += _sext(a, eew_ab) * _sext(b, eew_ab)
    return total


_MX_BLOCK_DOT = _mx_block_dot

#: Round nine: the arithmetic of vfmmacc.vv's narrow cells, rebindable so
#: the round-nine FP controls can substitute a wrong accumulator.
_FPN_GEMM = rvv_ref.fpn_reference_gemm


def _linear_c_index(i: int, j: int, geom: TileGeometry) -> int:
    """C indexed as a plain ``i*N_max + j``, the way Titan's RTL does it.

    ``backend/ExecuteSequencer.scala`` computes ``mat_c_flat = i*M + j``
    (M = N_max for a square tile) where the spec routes ``i*N_max + j``
    through ``tile_reg_idx`` (spec 4810-4812 calling 4777-4786).  At
    LAMBDA=1 the spec's ``mat_C_idx`` evaluates to ``j*M + i``, so the two
    differ by exactly a transpose.
    """
    return rvv_ref.c_sequential_index(i, j, geom)


def _linear_ab_index(r: int, k: int, geom: TileGeometry) -> int:
    """A/B indexed sequentially, i.e. ``tile_reg_idx`` replaced by identity."""
    return rvv_ref.ab_sequential_index(r, k, geom)


def _identity_tile_reg_idx(i: int, group_regs: int, row_elems_per_reg: int,
                           elems_per_reg: int) -> int:
    """``tile_reg_idx`` replaced by the identity, for the gap control."""
    return i


def _geometry(machine: Machine, w: int = 1,
              kind: str = "int") -> TileGeometry:
    """Decode vtype into a tile geometry.

    *w* comes from the *instruction*, not from vtype: vtype.SEW is the C
    accumulator width for the whole vmmacc family, and the widening factor
    is what the opcode selects (Sail: ``decode_gemm_geometry(4)`` inside
    vqmmacc.vv's body, ``decode_gemm_geometry(1)`` inside vmmacc.vv's).  The
    tile load/stores decode their own geometry with W=1, because they move
    SEW-wide storage elements and never look inside a packed element.
    """
    return TileGeometry(machine.vlen, machine.sew, machine.lam,
                        machine.lmul, machine.vl, w, "op", kind)


def _fp_step(acc: int, a: int, b: int, width: int) -> int:
    """One k of the floating-point accumulation, at rnd=frm.

    Split out of :func:`_ime` so that
    :func:`check_fp_rounding_is_load_bearing` can substitute the rnd=xct
    step -- a fused multiply-add -- and show that the directed programs
    actually distinguish the two.  Round-group-sum and accumulation both
    round under frm (Sail 5259-5264 with ``round_group_sum`` returning
    ``Some``), which is two roundings, not one.
    """
    return rvv_ref.fp_add(acc, rvv_ref.fp_mul(a, b, width), width)


def _fp_step_fused(acc: int, a: int, b: int, width: int) -> int:
    """The rnd=xct step: one rounding, over the exact product.

    Sail 5262-5263, ``None() => acc = fp_add_internal(acc, S, ...)``.  Not
    what Titan discloses, and only used by the negative control.
    """
    ka, sa, va = rvv_ref.fp_unpack(a, width)
    kb, sb, vb = rvv_ref.fp_unpack(b, width)
    kc, sc, vc = rvv_ref.fp_unpack(acc, width)
    if "num" != ka or "num" != kb or "num" != kc:
        return _fp_step(acc, a, b, width)      # specials: no difference here
    product = (-va if sa else va) * (-vb if sb else vb)
    total = (-vc if sc else vc) + product
    if total == 0:
        return 0
    return rvv_ref.fp_round(1 if total < 0 else 0, abs(total), width)


def _ime(machine: Machine, word: int) -> None:
    name, fields = ime.decode(word)
    geom = _geometry(machine)
    sew = machine.sew
    width = sew // 8

    if name in _TILE_LS:
        # Sail vmtl.v / vmts.v body, verbatim:
        #     let flat_idx : int =
        #       tile_reg_idx(i, LMUL, eff_lambda, elems_per_reg);
        #     let mem_off : int =
        #       (i / linesize) * LD + (i % linesize);
        # The loop variable i is the *sequential* tile element index: memory
        # uses it directly, the register file uses tile_reg_idx(i).  The two
        # are not composed.
        #
        # vmttl.v / vmtts.v differ in exactly one line -- the two halves of
        # mem_off swap:
        #     let mem_off : int =
        #       (i % linesize) * LD + (i / linesize);
        # -- and in the rs2 = x0 default, which is LAMBDA*LMUL (= linesize)
        # for the order-preserving pair and VLEN/(SEW*LAMBDA) (= M =
        # elems_per_reg/lambda) for the transposing pair.  flat_idx is
        # identical, which is the whole reason Zvvmttls is cheap.
        transposing = _TILE_LS[name]
        linesize = geom.linesize
        base = machine.x[fields["rs1"]]
        default_ld = (geom.elems_per_reg // geom.lam if transposing
                      else linesize)
        ld = machine.x[fields["rs2"]] or default_ld
        reg = fields["vd"] if name in ("vmtl.v", "vmttl.v") else fields["vs3"]
        for i in range(machine.vl):
            mem_off = ((i % linesize) * ld + (i // linesize) if transposing
                       else (i // linesize) * ld + (i % linesize))
            addr = base + width * mem_off
            flat_idx = _TILE_REG_IDX(i, geom.lmul, geom.lam,
                                     geom.elems_per_reg)
            if name in ("vmtl.v", "vmttl.v"):
                machine.vset(reg, flat_idx, sew,
                             machine.load(addr, width, signed=False))
            else:
                machine.store(addr, width, machine.vget(reg, flat_idx, sew))
        return

    if name in _FP_MACC_W and (1, sew) not in rvv_ref.FPN_CELLS:
        # Sail fp_gemm (5243-5268) at the Titan disclosure
        # G=1, psm=0, rnd=frm -- see rvv_ref.FP_DISCLOSURE and
        # rvv_ref.fp_gemm_reference, which is the single authority for the
        # arithmetic so that this model and the reference the directed
        # program is judged against cannot drift apart.
        #
        # With W=1 and G=1, the `step` and `g0` loops of fp_gemm enumerate
        # k = 0 .. K_eff-1 in strictly increasing order and each group is a
        # single product, so the whole body is
        #
        #     acc = fp_add(acc, fp_round_to_frm(fp_mul_exact(a, b)))
        #
        # per k.  The geometry, the flat indices and the tail-column policy
        # are character-for-character the integer ones: decode_gemm_geometry
        # is format-agnostic (spec 1500, Sail 4884-4912), and mat_A_idx /
        # mat_B_idx / mat_C_idx are shared.
        geom = _geometry(machine, _FP_MACC_W[name], kind="fp")
        geom.validate()
        vd, vs1, vs2 = fields["vd"], fields["vs1"], fields["vs2"]
        for i in range(geom.m):
            for j in range(geom.n):
                c_flat = _C_INDEX(i, j, geom)
                acc = machine.vget(vd, c_flat, sew)
                for k in range(geom.k_eff):
                    a = machine.vget(vs1, _AB_INDEX(i, k, geom), sew)
                    b = machine.vget(vs2, _AB_INDEX(j, k, geom), sew)
                    acc = _fp_step(acc, a, b, sew)
                machine.vset(vd, c_flat, sew, acc)
        return

    if name in _FPW_MACC_W or name in _FP_MACC_W:
        # (Round nine: vfmmacc.vv at SEW 8 / 16 -- the W=1 narrow cells in
        # rvv_ref.FPN_CELLS -- arrives here too, as kind='fpw' with W=1, and
        # is judged by rvv_ref.fpn_reference_gemm through _FPN_GEMM.)
        #
        # Round seven.  Sail fp_gemm at the Titan disclosure G=1, psm=0,
        # rnd=frm, with W > 1 -- so unlike the W=1 case above a group is no
        # longer a single product: it is the W products of one
        # sub-dot-product, summed *exactly* (psm=0, spec 1565), rounded once
        # to fmt_C (rnd=frm, spec 1584), and then accumulated
        # (spec 1591-1593).
        #
        # The arithmetic is not reimplemented here.  It is
        # rvv_ref.fpw_reference_gemm, which is also the authority the
        # directed programs' golden bytes come from, so this model and that
        # reference cannot drift apart -- the same discipline the W=1 and
        # microscaled blocks follow.  What this block owns is the *register
        # addressing*: pulling A, B and C out of the vector file through the
        # same _AB_INDEX / _C_INDEX globals the negative controls sabotage.
        w = _FPW_MACC_W.get(name, 1)
        geom = _geometry(machine, w, kind="fpw")
        geom.validate()
        cell = (geom.w, geom.sew)
        if cell not in rvv_ref.fpw_resolved_cells() and \
                cell not in rvv_ref.fpn_resolved_cells():
            raise SimError(
                f"{name} at SEW={geom.sew} is an OFP4 (E2M1) cell; E2M1 is "
                f"defined by OCP MX v1.0, which is not on disk, and this "
                f"implementation declares the cell unsupported")
        altfmt_a, altfmt_b = machine.altfmt_ab
        name_a, name_b, name_c, _mx = rvv_ref.fpw_legal_row(
            geom.w, geom.sew, machine.altfmt, altfmt_a, altfmt_b)
        fmt_a, fmt_b, fmt_c = (rvv_ref.fp_format(name_a),
                               rvv_ref.fp_format(name_b),
                               rvv_ref.fp_format(name_c))
        vd, vs1, vs2 = fields["vd"], fields["vs1"], fields["vs2"]
        eew_ab = geom.eew_ab
        a = [[machine.vget(vs1, _AB_INDEX(i, k, geom), eew_ab)
              for k in range(geom.k_eff)] for i in range(geom.m)]
        b = [[machine.vget(vs2, _AB_INDEX(j, k, geom), eew_ab)
              for k in range(geom.k_eff)] for j in range(geom.n)]
        # C is read across the *physical* tile width: the reference model
        # carries inactive columns (j >= N) through untouched, and reading
        # only the active ones would make it invent them.
        c = [[machine.vget(vd, _C_INDEX(i, j, geom), sew)
              for j in range(geom.n_max)] for i in range(geom.m)]
        gemm = (_FPN_GEMM if cell in rvv_ref.FPN_CELLS
                else rvv_ref.fpw_reference_gemm)
        out = gemm(geom, a, b, c, fmt_a, fmt_b, fmt_c)
        # ... but only the active columns are written back, because the
        # instruction does not write the others.  Writing the pass-through
        # values would be harmless today and would hide a C-index bug that
        # addressed an inactive column.
        for i in range(geom.m):
            for j in range(geom.n):
                machine.vset(vd, _C_INDEX(i, j, geom), sew, out[i][j])
        return

    if name in _MX_MACC_W:
        # Sail int_scaled_gemm (5373-5410).  Two structural facts decide
        # this block, and both are easy to get wrong in the direction of
        # "looks plausible, is not the architecture":
        #
        #  * There is no G / psm / rnd here at all.  int_scaled_gemm never
        #    calls get_fp_grouping / get_fp_psm / get_fp_rnd and has no G
        #    legality check, unlike fp_gemm at 5238-5242; spec 1283-1287 and
        #    1645-1652 state it in prose.  Nothing about the Titan FP
        #    disclosure applies.
        #  * The loop nest is j / i / s -- there is NO LMUL step loop.  The
        #    block/step intersection and shortened groups of spec 2030-2060
        #    belong to fp_scaled_gemm; int_block_dot here is handed the whole
        #    block interval.  Pasting that logic in would be a model bug that
        #    presents as an RTL bug.
        #
        # The arithmetic itself is delegated to rvv_ref so that this model
        # and the reference the generated program is judged against cannot
        # drift -- the same discipline the floating-point branch follows.
        w = _MX_MACC_W[name]
        # Every legality rule for this family lives in TileGeometry.validate
        # and the rvv_ref helpers below, and every one of them corresponds to
        # a place the Sail returns Illegal_Instruction -- reserved (W, SEW)
        # cells, the microscaling constraints, the altfmt rules.  So a
        # rejection here is a *trap*, not a gap in the model, and has to be
        # delivered to the program's handler rather than aborting the run.
        # The ime_mxl_ tier is entirely made of these cases.
        try:
            geom = _geometry(machine, w, kind="mx")
            geom.validate()
        except ValueError:
            raise IllegalInstruction from None
        altfmt = machine.altfmt
        # Architectural legality, not model coverage: each of these is a
        # place the Sail returns Illegal_Instruction, so each raises
        # IllegalInstruction and the run loop delivers a trap.
        try:
            eew_ab, fmt = rvv_ref.mx_legal_cell(w, sew, altfmt)
        except ValueError:
            raise IllegalInstruction from None
        if machine.altfmt_ab != (0, 0):
            # MXINT is signed unconditionally; Sail 6045-6046 / 6162-6163 /
            # 6277-6278 reject altfmt_A or altfmt_B = 1 outright.
            raise IllegalInstruction
        bs = machine.bs
        try:
            rvv_ref.mx_check_legality(w, geom.lmul, sew, geom.lam, bs)
        except ValueError:
            raise IllegalInstruction from None
        block_size = rvv_ref.mx_block_size(bs)
        blocks = rvv_ref.mx_block_count(geom.k_eff, block_size)
        stride = rvv_ref.mx_scale_stride(sew, geom.lam)
        vd, vs1, vs2 = fields["vd"], fields["vs1"], fields["vs2"]
        if 0 in (vd, vs1, vs2) or \
                any(r <= 0 < r + n for r, n in
                    ((vd, geom.emul_c), (vs1, geom.lmul), (vs2, geom.lmul))):
            # Prose Exceptions only -- absent from the Sail -- but a program
            # that overlapped v0 would be reading its own scales as data,
            # which is worth catching loudly rather than simulating.
            raise SimError(
                f"{name}: vd/vs1/vs2 register groups must not overlap v0, "
                f"which holds the paired E8M0 block scales (spec 2129-2170)")

        def scale_pair(m: int, s_idx: int) -> Tuple[int, int]:
            """(scale_A byte, scale_B byte) from v0 at p = m*R + s.

            Spec 2161-2170 and Sail 5110-5117: v0 is read at the *pair*
            width, low byte scale_A and high byte scale_B.  A and B use the
            same function of (row-or-column index, block index) out of the
            same register; only the index they pass differs.
            """
            pair = machine.vget(0, _MX_PAIR_INDEX(m, s_idx, stride),
                                rvv_ref.MX_PAIR_WIDTH)
            return pair & 0xFF, (pair >> 8) & 0xFF

        for j in range(geom.n):
            for i in range(geom.m):
                c_flat = _C_INDEX(i, j, geom)
                acc = machine.vget(vd, c_flat, sew)
                nan_out = False
                for s_idx in range(blocks):
                    a_byte, _ = scale_pair(i, s_idx)
                    _, b_byte = scale_pair(j, s_idx)
                    blk, is_nan = rvv_ref.mx_block_scale(a_byte, b_byte,
                                                         sew, fmt)
                    if is_nan:
                        nan_out = True
                        break          # Sail 5392
                    k_lo, k_hi = rvv_ref.mx_block_interval(
                        s_idx, block_size, geom.k_eff)
                    dot = _MX_BLOCK_DOT(machine, vs1, vs2, i, j,
                                        k_lo, k_hi, geom, eew_ab)
                    fp_sum = rvv_ref.mx_int_to_fp(dot, sew, fmt)
                    acc = rvv_ref.fp_add(
                        acc, rvv_ref.fp_mul(blk, fp_sum, sew, fmt), sew, fmt)
                machine.vset(vd, c_flat, sew,
                             rvv_ref.fp_default_nan(sew, fmt) if nan_out
                             else acc)
        return

    if name not in _MACC_W:
        raise SimError(f"{name} is not modelled "
                       f"(implemented: {ime.IMPLEMENTED})")

    # Sail int_gemm / int_block_dot:
    #
    #   let c_flat : int = mat_C_idx(i, j, g.N_max, g.EMUL_C, g.lambda, g.epr_C);
    #   var acc : int = signed(read_single_element(g.EEW_C, c_flat, vd));
    #   acc = acc + int_block_dot(i, j, 0, g.K_eff - 1, g, ...);
    #   write_single_element(g.EEW_C, c_flat, vd, to_bits_unsafe(g.EEW_C, acc))
    #
    # with int_block_dot reading A and B at EEW_A and sign-extending each,
    # then summing *exactly* -- the single modular reduction happens once, on
    # the way back into the EEW_C-wide C element.  Note the three widths in
    # play: C is read and written at EEW_C = vtype.SEW, A and B are read at
    # EEW_A = SEW/W, and the flat index for each side is produced by its own
    # mat_*_idx.  Only the N active columns participate; vta=0 in these
    # tests, so the tail columns keep their pre-instruction values.
    #
    # Round nine: the per-operand read is signed() or unsigned() by
    # vtype.altfmt_A / altfmt_B (Sail 5137-5142; spec 1145-1154), EEW_A may
    # be 4 (Int4, nibble-packed -- Machine.vget), and SEW may be 64 with
    # W > 1.  The dot product and the C write-back are the rebindable
    # globals _INT_DOT / _INT_WRITEBACK so the round-nine negative controls
    # can sabotage exactly one of them.
    geom = _geometry(machine, _MACC_W[name])
    vd, vs1, vs2 = fields["vd"], fields["vs1"], fields["vs2"]
    ua, ub = machine.altfmt_ab
    for i in range(geom.m):
        for j in range(geom.n):
            c_flat = _C_INDEX(i, j, geom)
            acc = _sext(machine.vget(vd, c_flat, sew), sew)
            acc += _INT_DOT(machine, vs1, vs2, i, j, geom, ua, ub)
            machine.vset(vd, c_flat, sew, _INT_WRITEBACK(acc, sew))


def _int_dot(machine: Machine, vs1: int, vs2: int, i: int, j: int,
             geom: TileGeometry, ua: int, ub: int) -> int:
    """Sail int_block_dot over k = 0 .. K_eff-1, exactly (5127-5145)."""
    eew = geom.eew_ab
    total = 0
    for k in range(geom.k_eff):
        a = machine.vget(vs1, _AB_INDEX(i, k, geom), eew)
        b = machine.vget(vs2, _AB_INDEX(j, k, geom), eew)
        total += (rvv_ref.int_operand(a, eew, ua)
                  * rvv_ref.int_operand(b, eew, ub))
    return total


def _int_writeback(acc: int, sew: int) -> int:
    """to_bits(EEW_C, acc): one reduction modulo 2**SEW (Machine.vset)."""
    return acc


_INT_DOT = _int_dot
_INT_WRITEBACK = _int_writeback


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _fpw_sweep_tier(geom: TileGeometry, seed: int) -> str:
    """Which widening-FP tier the sweep runs for *seed* at *geom*.

    Round seven's cells alternate golden / exact, as they always have, so
    their seeds run what they ran.  Round eight's OFP8 cells cycle golden /
    exact / special.  Exact falls back to golden where no baseline fcvt
    reaches the accumulator (binary16 / bfloat16).
    """
    tier = ("golden", "exact")[seed % 2]
    if (geom.w, geom.sew) in rvv_ref.FPN_CELLS:
        # Round nine: no exact tier exists; alternate golden / special.
        return ("golden", "special")[seed % 2] if geom.k_eff >= 2 \
            else "golden"
    if (geom.w, geom.sew) in rvv_ref.FPW_ROUND_EIGHT_CELLS:
        tier = ("golden", "exact", "special")[seed % 3]
    if tier == "exact" and not ime_tests.fpw_exact_emittable(geom):
        tier = "golden"
    return tier


def simulate(geom: TileGeometry, seed: int = 0) -> Tuple[int, str]:
    """Assemble and run one directed program, whichever shape it is.

    ``geom.check`` selects the generator: "pair" is the differential
    program of rounds one to four, "clayout" is round five's C-tile
    register-layout probe, whose operands are fixed by construction and so
    ignore *seed*.
    """
    if geom.check == "clayout":
        program = assemble(ime_tests.emit_clayout_test(geom))
    elif geom.kind == "fpw":
        # Round seven.  Two tiers per geometry, and the seed selects between
        # them so that a multi-seed sweep covers both: "golden" embeds the
        # reference model's bytes, "exact" recomputes on the DUT from
        # integers.  Both must run, because the whole value of the pair is
        # that they can disagree -- see
        # rvv_ref.check_round_seven_tier_overlap.
        rng = random.Random(seed)
        rows = rvv_ref.fpw_rows(geom.w, geom.sew)
        key = sorted(rows, key=str)[seed % len(rows)]
        tier = _fpw_sweep_tier(geom, seed)
        fpn = (geom.w, geom.sew) in rvv_ref.FPN_CELLS
        case = (rvv_ref.fpw_exact_case(geom, rows[key], rng)
                if tier == "exact"
                else rvv_ref.fpn_special_case(geom, rows[key], rng)
                if tier == "special" and fpn
                else rvv_ref.fpw_special_case(geom, rows[key], rng)
                if tier == "special"
                else rvv_ref.fpw_case(geom, rows[key], rng))
        plan = ime_tests.FpwPlan(geom, key, tier, *case, tag="sweep")
        program = assemble(ime_tests.emit_fpw_test(plan))
    elif geom.kind == "mx":
        # Round six.  Its operands are not a plain random_case: the tier
        # rests on the result being exactly representable, so the plan has
        # to choose bounds, scales and per-element modes together.  See
        # ime_tests.mx_random_plan.
        plan = ime_tests.mx_random_plan(geom, random.Random(seed))
        program = assemble(ime_tests.emit_mx_test(plan))
    else:
        case = rvv_ref.random_case(geom, random.Random(seed))
        program = assemble(ime_tests.emit_test(geom, case))
    machine = Machine(vlen=geom.vlen)
    code = run(program, machine)
    return code, "".join(machine.stdout)


def check_negative_control(geom: TileGeometry) -> None:
    """A deliberately broken IME model must make the program fail.

    Without this, a program that passes proves only that both paths ran; it
    could equally be that the comparison never fires.  Corrupting one C
    element in the model must produce exactly one FAIL naming that element.
    """
    global _ime
    original = _ime

    def sabotage(machine: Machine, word: int) -> None:
        original(machine, word)
        name, fields = ime.decode(word)
        if name in _MACC_W or name in _FP_MACC_W or name in _FPW_MACC_W:
            if name in _FP_MACC_W:
                g = _geometry(machine, _FP_MACC_W[name], kind="fp")
            elif name in _FPW_MACC_W:
                g = _geometry(machine, _FPW_MACC_W[name], kind="fpw")
            else:
                g = _geometry(machine, _MACC_W[name])
            index = rvv_ref.c_element_index(1 % g.m, 0, g)
            machine.vset(fields["vd"], index, machine.sew,
                         machine.vget(fields["vd"], index, machine.sew) ^ 1)

    _ime = sabotage
    try:
        code, output = simulate(geom, seed=5)
    finally:
        _ime = original
    assert "TITAN FAIL" in output, f"sabotage went undetected: {output!r}"
    assert code >= ime_tests.EXIT_MISMATCH_BASE, code
    row = int(re.search(r"row=(\d+)", output).group(1))
    assert row == 1 % geom.m, f"blamed row {row}, corrupted {1 % geom.m}"


def check_transposing_is_load_bearing(vlen: int = 256) -> None:
    """An implementation that ignores the transpose must fail every t program.

    check_negative_control proves the comparison fires; this proves the
    *transposing* programs actually depend on the transpose.  Without it, a
    vmttl.v tier that passes would be consistent with vmttl.v having been
    decoded as vmtl.v -- which is precisely the two-bit decoder mistake the
    encoding makes easy (bits 27:26, `0b00` vs `0b01`).

    Every geometry must fail, not merely one: a transposing program whose
    memory image happens to be symmetric would pass either way, and if such
    a geometry existed it would be dead weight in the tier.
    """
    original = dict(_TILE_LS)
    _TILE_LS.update({"vmttl.v": False, "vmtts.v": False})
    try:
        checked = 0
        for geom in rvv_ref.ime_legal_configs(vlen, full_vl_only=True,
                                              tloads=("t",)):
            if geom.emul_c == 16 or not _allocatable(geom):
                continue
            code, output = simulate(geom, seed=0)
            assert code != ime_tests.EXIT_PASS and "TITAN FAIL" in output, (
                f"{geom.describe()}: a non-transposing vmttl.v went "
                f"undetected")
            checked += 1
        assert checked, "no transposing geometry was checked"
    finally:
        _TILE_LS.clear()
        _TILE_LS.update(original)


def check_fp_rounding_is_load_bearing(vlen: int = 256) -> None:
    """A rnd=xct implementation must fail every floating-point program.

    The floating-point tier's whole claim to be an exact test rests on the
    Titan disclosure (rvv_ref.FP_DISCLOSURE: G=1, psm=0, rnd=frm).  If the
    directed programs could not tell that apart from the other legal
    disclosures, "exact bitwise compare" would be decoration: the DUT could
    fuse the multiply-add -- the single most likely thing an FP MAC array
    does -- and still be green.

    So: substitute the rnd=xct step into the model and require every
    geometry to fail.  Every one, not merely some, because a geometry whose
    programs could not distinguish the two disclosures would be dead weight
    in the tier and should be found now rather than believed later.
    """
    global _fp_step
    original = _fp_step
    _fp_step = _fp_step_fused
    try:
        checked = 0
        for geom in rvv_ref.ime_legal_configs(vlen, full_vl_only=True,
                                              kinds=("fp",)):
            if geom.emul_c == 16 or not _allocatable(geom):
                continue
            code, output = simulate(geom, seed=0)
            assert code != ime_tests.EXIT_PASS and "TITAN FAIL" in output, (
                f"{geom.describe()}: a fused (rnd=xct) multiply-accumulate "
                f"went undetected -- this geometry cannot distinguish the "
                f"disclosed rounding and does not belong in the tier")
            checked += 1
        assert checked, "no floating-point geometry was checked"
    finally:
        _fp_step = original


def check_clayout_catches_transposed_c(vlen: int = 256) -> None:
    """A model that writes C transposed must fail every clayout program.

    The tier exists to catch exactly this, so this is the control that makes
    the tier mean anything: without it, a green clayout run would be
    consistent with the read-back never being compared.

    The sabotage is applied to the *model's* index function, never to
    rvv_ref: the generated program's expected image comes from rvv_ref, and
    moving reference and defendant together is precisely the failure mode
    that let a transposed accumulator survive four rounds of green.

    Every geometry must fail, and every geometry must additionally print
    `TITAN CLAYOUT transposed` -- the named verdict, not just 56 scattered
    diffs.  A geometry that could not produce the named verdict would be
    dead weight in the tier and should be found now.
    """
    global _C_INDEX
    original = _C_INDEX
    _C_INDEX = lambda i, j, geom: original(j, i, geom)   # noqa: E731
    try:
        checked = 0
        for geom in _clayout_geometries(vlen):
            code, output = simulate(geom)
            assert code != ime_tests.EXIT_PASS and "TITAN FAIL" in output, (
                f"{geom.describe()}: a transposed C tile went undetected")
            assert "TITAN CLAYOUT transposed" in output, (
                f"{geom.describe()}: the C tile is transposed but the tier "
                f"did not name it:\n{output}")
            assert "TITAN DIFF " in output, geom.describe()
            checked += 1
        assert checked, "no clayout geometry was checked"
    finally:
        _C_INDEX = original


def check_linear_c_index_is_the_gap(vlen: int = 256) -> None:
    """The round-five gap itself, stated as an executable claim.

    Titan's RTL indexes the tile register groups linearly -- Sail's
    ``tile_reg_idx`` replaced by the identity, and ``mat_C_idx`` by a plain
    ``i*N_max + j`` (backend/ExecuteSequencer.scala around 437 and 445-446
    against spec 4810-4812 and 4777-4786).  Model that, and two things must
    hold at once:

      * **every** pre-round-five directed program still passes.  It writes
        the C tile with vmts.v and reads it back with vmts.v, and the
        permutation cancels; the memory image is correct and the comparison
        is satisfied.  That is the gap, and it is a positive claim, not an
        absence of evidence.
      * **every** clayout program whose geometry the bug can reach fails,
        and at LAMBDA=1 -- where the spec's mat_C_idx evaluates to
        ``j*M + i``, the exact transpose of the linear index -- names the
        layout.  At EMUL_C=1 ``tile_reg_idx`` *is* the identity, so the
        linear index is the spec index and there is nothing to catch: those
        geometries must still pass, which is the tier's no-false-positive
        half and is asserted here rather than assumed.

    If the first assertion ever starts failing, the gap has been closed by
    something else and this control should be re-derived rather than
    deleted; if the second starts failing, the tier has stopped being able
    to see the bug it was built for.
    """
    global _C_INDEX, _AB_INDEX, _TILE_REG_IDX
    saved = (_C_INDEX, _AB_INDEX, _TILE_REG_IDX)
    _C_INDEX, _AB_INDEX, _TILE_REG_IDX = (_linear_c_index, _linear_ab_index,
                                          _identity_tile_reg_idx)
    try:
        blind = 0
        for geom in _every_geometry(vlen):
            if geom.check == "clayout" or geom.emul_c == 16 \
                    or not _allocatable(geom) or geom.n != geom.n_max:
                continue
            code, output = simulate(geom, seed=0)
            assert code == ime_tests.EXIT_PASS and "TITAN PASS" in output, (
                f"{geom.describe()}: a linear C index was caught by a pair "
                f"program -- the round-five premise no longer holds:\n"
                f"{output}")
            blind += 1
        named = caught = unaffected = 0
        for geom in _clayout_geometries(vlen):
            code, output = simulate(geom)
            same = all(rvv_ref.c_element_index(i, j, geom)
                       == rvv_ref.c_sequential_index(i, j, geom)
                       for i in range(geom.m) for j in range(geom.n_max))
            if same:
                # EMUL_C=1: mat_C_idx collapses to i*N_max + j, so the RTL's
                # linear index is the architectural one.  Nothing to catch,
                # and a tier that "caught" it here would be lying.
                assert geom.emul_c == 1, geom.describe()
                assert code == ime_tests.EXIT_PASS, (
                    f"{geom.describe()}: false positive -- the linear index "
                    f"is the spec index at EMUL_C=1:\n{output}")
                unaffected += 1
                continue
            assert code != ime_tests.EXIT_PASS and "TITAN FAIL" in output, (
                f"{geom.describe()}: the clayout tier did not catch the "
                f"linear C index:\n{output}")
            caught += 1
            if geom.lam == 1:
                assert "TITAN CLAYOUT transposed" in output, (
                    f"{geom.describe()}: LAMBDA=1, so the linear index is "
                    f"exactly a transpose, but the tier did not name it:"
                    f"\n{output}")
                named += 1
            else:
                assert "TITAN CLAYOUT other" in output, geom.describe()
        assert blind and caught and named and unaffected, (
            blind, caught, named, unaffected)
    finally:
        _C_INDEX, _AB_INDEX, _TILE_REG_IDX = saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument(
        "--emit",
        metavar="SEW,LAMBDA,LMUL,N[,W][,op|t][,int|fp][,pair|clayout]",
        help="run one geometry and print its output")
    parser.add_argument("--seeds", type=int, default=3,
                        help="random cases per geometry")
    args = parser.parse_args()

    if args.emit:
        toks = args.emit.split(",")
        tload, kind, check = "op", "int", "pair"
        if toks[-1] in ("pair", "clayout"):
            check = toks.pop()
        if toks[-1] in ("int", "fp"):
            kind = toks.pop()
        if toks[-1] in ("op", "t"):
            tload = toks.pop()
        sew, lam, lmul, n, *rest = (int(t) for t in toks)
        geom = TileGeometry(args.vlen, sew, lam, lmul, n * lam * lmul,
                            rest[0] if rest else 1, tload, kind, check)
        code, output = simulate(geom)
        print(f"{geom.describe()}\nexit={code}\n{output}")
        return 0 if code == ime_tests.EXIT_PASS else 1

    geometries = [g for g in _every_geometry(args.vlen)
                  if g.emul_c != 16 and _allocatable(g)]
    failures, ran = [], 0
    for geom in geometries:
        # The clayout probe's operands are fixed by construction, so extra
        # seeds would be the same program run again.
        for seed in range(1 if geom.check == "clayout" else args.seeds):
            code, output = simulate(geom, seed)
            ran += 1
            if code != ime_tests.EXIT_PASS or "TITAN PASS" not in output:
                failures.append((geom.describe(), seed, code, output.strip()))
    for describe, seed, code, output in failures[:10]:
        print(f"  FAIL {describe} seed={seed} exit={code}\n       {output}")

    # Any geometry with more than one C row will do; pick the first legal one
    # rather than hard-coding a shape that is only legal at one VLEN.
    controls = [next(g for g in rvv_ref.ime_legal_configs(args.vlen,
                                                          full_vl_only=True)
                     if g.emul_c != 16 and _allocatable(g) and g.m > 1)]
    # ... and one widening geometry, so the W=4 compare is proven to fire
    # too.  A widening program that "passes" because its comparison never
    # runs would be the most expensive kind of green.
    controls += [g for g in rvv_ref.ime_legal_configs(
        args.vlen, sews=rvv_ref.WIDENING_SEWS, ws=(4,), full_vl_only=True)
        if g.emul_c != 16 and _allocatable(g) and g.m > 1][:1]
    # ... and one of each round-three tier, for the same reason: a W=2, a
    # W=8 and a transposing program must each be shown to be able to fail.
    for w in (2, 8):
        controls += [g for g in rvv_ref.ime_legal_configs(
            args.vlen, sews=rvv_ref.WIDENING_SEWS_BY_W[w], ws=(w,),
            full_vl_only=True)
            if g.emul_c != 16 and _allocatable(g) and g.m > 1][:1]
    controls += [g for g in rvv_ref.ime_legal_configs(
        args.vlen, full_vl_only=True, tloads=("t",))
        if g.emul_c != 16 and _allocatable(g) and g.m > 1][:1]
    # ... and one of each round-four accumulator width.  A floating-point
    # program has a second way to be vacuously green that the integer tiers
    # do not: if both paths agreed on a wrong rounding they would still
    # compare equal, so the control corrupts the *model* and checks the
    # scalar reference disagrees with it.
    for sew in sorted(rvv_ref.FP_FORMATS):
        controls += [g for g in rvv_ref.ime_legal_configs(
            args.vlen, sews=(sew,), full_vl_only=True, kinds=("fp",))
            if g.emul_c != 16 and _allocatable(g) and g.m > 1][:1]
    # ... and one round-five C-layout probe, for the same reason again: its
    # compare is a different compare (a register image against a precomputed
    # one, not two on-DUT computations against each other) and has to be
    # shown to be able to fail.
    controls += [g for g in _clayout_geometries(args.vlen) if g.m > 1][:1]
    # ... and one round-seven geometry per resolved cell, for the same
    # reason again.  This matters more here than anywhere else: a
    # golden-tier program compares against bytes this harness wrote, so if
    # its comparison never fired it would be a program that agrees with
    # itself.  Proving the compare can fail is what stops "the model said
    # so" from being unfalsifiable.
    for cell in rvv_ref.fpw_resolved_cells():
        controls += [g for g in rvv_ref.fpw_legal_configs(args.vlen,
                                                          full_vl_only=True)
                     if (g.w, g.sew) == cell and g.emul_c != 16
                     and _allocatable(g) and g.m > 1][:1]
    # ... and one per round-nine integer cell (Int4, Int16/Int32 -> Int64):
    # a nibble-packed tile or an e64 reference path is a new way for the
    # compare to be silently vacuous.
    for cell in rvv_ref.ROUND_NINE_INT_CELLS:
        controls += [g for g in rvv_ref.int_cell_geometries(args.vlen, cell)
                     if _allocatable(g) and g.emul_c != 16 and g.m > 1][:1]
    for cell in rvv_ref.fpn_resolved_cells():
        controls += [g for g in rvv_ref.fpn_legal_configs(args.vlen,
                                                          full_vl_only=True)
                     if (g.w, g.sew) == cell and g.emul_c != 16
                     and _allocatable(g) and g.m > 1][:1]
    for control in controls:
        check_negative_control(control)
        print(f"  ok  check_negative_control  {control.describe()}")
    check_transposing_is_load_bearing(args.vlen)
    print("  ok  check_transposing_is_load_bearing")
    check_fp_rounding_is_load_bearing(args.vlen)
    print("  ok  check_fp_rounding_is_load_bearing")
    check_clayout_catches_transposed_c(args.vlen)
    print("  ok  check_clayout_catches_transposed_c")
    check_linear_c_index_is_the_gap(args.vlen)
    print("  ok  check_linear_c_index_is_the_gap")
    for check in (check_mx_scale_layout_is_load_bearing,
                  check_mx_wrapping_is_unobservable_here,
                  check_mx_nibble_order_is_load_bearing,
                  check_mxl_tier,
                  check_mxl_legality_is_load_bearing,
                  check_mxl_warl_is_load_bearing,
                  check_round_seven_sub_byte_path,
                  check_round_seven_sweep_is_live,
                  check_round_eight_sweep_is_live,
                  check_round_eight_ofp8_decode_is_load_bearing,
                  check_round_nine_sweep_is_live):
        check(args.vlen)
        print(f"  ok  {check.__name__}")
    tally = check_round_nine_int_controls(args.vlen)
    print("  ok  check_round_nine_int_controls  " + ", ".join(
        f"{bad} {c}/{n}" for bad, (c, n) in tally.items()))
    tally = check_round_nine_fpn_controls(args.vlen)
    print("  ok  check_round_nine_fpn_controls  " + ", ".join(
        f"{bad} rows {c}/{n} ({f} programs failed)"
        for bad, (c, n, f) in tally.items()))

    tally = {}
    for g in geometries:
        key = (g.check, g.kind, g.w, g.tload)
        tally[key] = tally.get(key, 0) + 1
    breakdown = " + ".join(f"{n} {chk}/{kind} W={w}/{tl}"
                           for (chk, kind, w, tl), n in sorted(tally.items()))
    print(f"\nVLEN={args.vlen}: {ran} program executions across "
          f"{len(geometries)} geometries ({breakdown}), "
          f"{len(failures)} failed")
    return 1 if failures else 0


def check_mxl_tier(vlen: int = 256) -> None:
    """The legality programs must pass against the reference model.

    This tier is not geometry-driven -- ``emit_mxl_test`` takes a mnemonic,
    not a ``TileGeometry`` -- so it is invisible to the sweep in
    :func:`main`, and for a while it shipped with no meta-judge at all.
    That mattered: when r19's Stage 0 reported two ``ime_mxl_`` failures
    there was no way to tell a model defect from a judge bug without reading
    the assembly by hand.  Running them here answers that question in one
    command, and is why the model needs CSR and trap support at all.
    """
    for mnemonic in ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv"):
        program = assemble(ime_tests.emit_mxl_test(vlen, mnemonic))
        machine = Machine(vlen=vlen)
        code = run(program, machine)
        output = "".join(machine.stdout)
        assert code == ime_tests.EXIT_PASS and "TITAN PASS" in output, (
            f"ime_mxl_{mnemonic}: the legality tier does not pass against "
            f"the reference model:\n{output.strip()}")


def check_mxl_legality_is_load_bearing(vlen: int = 256) -> None:
    """A model that accepts an illegal configuration must fail the tier.

    Half of each ``ime_mxl_`` program asserts that something traps.  A model
    that never raises would satisfy the other half and could look green if
    the trap cases were mis-wired, so make the model permissive and require
    every program to notice.

    The sabotage targets the *model's* legality gate, not ``rvv_ref``'s
    rules: rvv_ref is what the programs are judged against.
    """
    original = _ime

    def permissive(machine, word):
        try:
            return original(machine, word)
        except IllegalInstruction:
            return None            # silently execute nothing, raise no trap

    globals()["_ime"] = permissive
    try:
        caught = 0
        for mnemonic in ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv"):
            program = assemble(ime_tests.emit_mxl_test(vlen, mnemonic))
            machine = Machine(vlen=vlen)
            code = run(program, machine)
            output = "".join(machine.stdout)
            if code != ime_tests.EXIT_PASS and "TITAN FAIL" in output:
                caught += 1
        assert caught == 3, (
            f"only {caught}/3 legality programs noticed a model that never "
            f"raises illegal-instruction")
    finally:
        globals()["_ime"] = original


def check_mxl_warl_is_load_bearing(vlen: int = 256) -> None:
    """A model that retains an impermissible lambda must fail the tier.

    r26: at VLEN=256 each ime_mxl_ program asks for LAMBDA=1 at SEW=8, which
    no implementation may keep (EMUL_C=32; spec 139-145, 975-988, write
    rules 1050-1054).  Swap the reference model's WARL selection for one that
    keeps the request verbatim -- what the r25 RTL did -- and require every
    program to be classified bad_geometry, with the SKIP line naming SEW=8.
    """
    import helpers

    class _Run:
        def __init__(self, log):
            self.log, self.out, self.success, self.returncode = log, "", \
                True, 0

    global _WARL_LAMBDA
    if 1 in rvv_ref.permissible_lambdas(vlen, 8):
        vlen = 256                    # the probe only exists where L=1 is out
    original = _WARL_LAMBDA
    _WARL_LAMBDA = lambda vlen_, sew, requested, current: (  # noqa: E731
        requested if requested else original(vlen_, sew, requested, current))
    try:
        for mnemonic in ("vfwimmacc.vv", "vfqimmacc.vv", "vf8wimmacc.vv"):
            program = assemble(ime_tests.emit_mxl_test(vlen, mnemonic))
            machine = Machine(vlen=vlen)
            code = run(program, machine)
            output = "".join(machine.stdout)
            verdict = helpers.classify_run(_Run(output))
            assert (code != ime_tests.EXIT_PASS
                    and verdict.kind == "bad_geometry"
                    and (verdict.sew, verdict.requested_lambda,
                         verdict.selected_lambda) == (8, 1, 1)), (
                f"ime_mxl_{mnemonic}: a model that keeps lambda=1 at SEW=8, "
                f"VLEN={vlen} was not convicted: {verdict}\n{output.strip()}")
    finally:
        _WARL_LAMBDA = original


def _mx_pool(vlen: int):
    """The round-six geometries the sweep actually runs."""
    return [g for g in rvv_ref.mx_legal_configs(vlen)
            if g.emul_c != 16 and _allocatable(g)]


def check_mx_scale_layout_is_load_bearing(vlen: int = 256) -> None:
    """A model that transposes the v0 pair index must fail MX programs.

    ``p = m*R + s`` (spec 2161-2170, Sail 5110-5117) against ``s*R + m``.
    This is the single most likely implementation error in round six -- the
    two factors are a row index and a block index out of the same register,
    and nothing about the encoding makes the order obvious.

    Sabotaging the *model* and not ``rvv_ref`` is the point: rvv_ref is what
    the generated programs are judged against, so moving it would move the
    reference and the defendant together.  That is the failure mode round
    five exists to close.
    """
    global _MX_PAIR_INDEX
    original = _MX_PAIR_INDEX
    try:
        _MX_PAIR_INDEX = lambda m, s, r: rvv_ref.mx_pair_index(s, m, r)
        checked = blind = 0
        for geom in _mx_pool(vlen):
            stride = rvv_ref.mx_scale_stride(geom.sew, geom.lam)
            blocks = rvv_ref.mx_block_count(geom.k_eff, rvv_ref.mx_block_size(0))
            if stride == 1 or (geom.m == 1 and blocks == 1):
                # m*R+s and s*R+m coincide; such a geometry cannot see this
                # error at all.  Counted, not silently skipped -- "the
                # control passed" must not be able to mean "the control was
                # never applicable".
                blind += 1
                continue
            try:
                code, output = simulate(geom, seed=0)
            except SimError:
                # Reading off the end of v0 is also a detection: the
                # sabotaged index is out of range for this geometry.
                checked += 1
                continue
            assert code != ime_tests.EXIT_PASS and "TITAN FAIL" in output, (
                f"{geom.describe()}: a transposed v0 paired-scale index went "
                f"undetected")
            checked += 1
        assert checked, (
            f"no round-six geometry could see the v0 index transposition "
            f"({blind} were blind to it)")
    finally:
        _MX_PAIR_INDEX = original


def check_mx_wrapping_is_unobservable_here(vlen: int = 256) -> None:
    """A modulo-2**SEW block dot product provably cannot fail this tier.

    This started life as a negative control -- sabotage ``int_block_dot``
    into the round-one modular habit that Sail 5128-5130 forbids ("returns
    an unbounded mathematical integer (no overflow)") and require the tier
    to notice.  It never fires, and the reason is structural rather than
    accidental, so the check states the structure instead of pretending to
    be a control.

    Every round-six directed program is built so that ``|dot|`` fits the
    accumulator *significand*, which is what makes the result exactly
    representable and the on-DUT integer reference path valid at all.  But
    ``p_C < SEW`` in every one of the seven cells (binary16 11 < 16,
    bfloat16 8 < 16, binary32 24 < 32, binary64 53 < 64), so the exactness
    bound is always far tighter than the wrap point.  A modulo-2**SEW
    reduction is therefore the identity on every value this tier can
    produce.

    Shipping the control anyway would be worse than not having it: it would
    go green for all 192 geometries and look like evidence that the exact
    integer path had been witnessed, when nothing had been witnessed at all.
    That is the mistake this project has already paid for once.

    Where the exactness property *is* witnessed:

      * ``rvv_ref.check_round_six_negative_controls`` sabotages the same line
        in the *reference model* and searches for a geometry where it bites.
        It finds one, because it is free to use peak native operands
        (|a|=|b|=127) rather than the tier's exactness-bounded ones.
      * Stage 3 lockstep against the Spike model, which is not bound by the
        directed tier's operand construction.

    So the assertion here is the arithmetic fact, checked over every cell and
    both block sizes, not a simulation.
    """
    del vlen
    for (w, sew), (_ewidth, fmts) in sorted(rvv_ref.MX_CELLS.items()):
        for altfmt in sorted(fmts):
            for bs in (0, 1):
                bound = rvv_ref.mx_exact_operand_bound(w, sew, altfmt, bs)
                peak = rvv_ref.mx_block_size(bs) * bound * bound
                assert peak < (1 << (sew - 1)), (
                    f"W={w} SEW={sew} altfmt={altfmt} bs={bs}: |dot| can "
                    f"reach {peak} >= 2**{sew - 1}, so a modulo-2**SEW block "
                    f"dot product IS observable here after all -- this check "
                    f"must go back to being a real negative control")


def check_mx_nibble_order_is_load_bearing(vlen: int = 256) -> None:
    """A model that packs the A tile's MXINT4 nibbles backwards must fail.

    Spec 1206-1219 puts the *even* element index in the LOW nibble.

    The sabotage swaps the nibble order on the **A side only**.  Swapping it
    on both sides is the obvious thing to write and is worthless: the block
    dot product is a sum over k, so exchanging a[k] with a[k^1] *and* b[k]
    with b[k^1] merely reorders the terms and the sum is unchanged.  A
    control written that way would go green everywhere and mean nothing.
    One-sided is both the discriminating version and the realistic bug --
    an implementation unpacks A and B through separate paths.
    """
    global _MX_BLOCK_DOT
    original = _MX_BLOCK_DOT

    def swapped_a(machine, vs1, vs2, i, j, k_lo, k_hi, geom, eew_ab):
        total = 0
        for k in range(k_lo, k_hi + 1):
            a = machine.vget(vs1, _AB_INDEX(i, k, geom) ^ 1, eew_ab)
            b = machine.vget(vs2, _AB_INDEX(j, k, geom), eew_ab)
            total += _sext(a, eew_ab) * _sext(b, eew_ab)
        return total

    int4 = [g for g in _mx_pool(vlen) if g.eew_ab == 4 and g.k_eff > 1]
    assert int4, "no multi-K MXINT4 geometry in the round-six pool"
    try:
        _MX_BLOCK_DOT = swapped_a
        caught = blind = 0
        for geom in int4:
            code, output = simulate(geom, seed=0)
            if code != ime_tests.EXIT_PASS and "TITAN FAIL" in output:
                caught += 1
            else:
                blind += 1
        assert caught, (
            f"no MXINT4 geometry noticed a one-sided nibble swap "
            f"({blind} were blind to it)")
    finally:
        _MX_BLOCK_DOT = original


def _every_geometry(vlen: int):
    """Every geometry in scope, tier by tier, in round order.

    W=1 order-preserving, then W=4, then round three's W=2, W=8 and the
    transposing W=1 tier.  Concatenated rather than interleaved so the
    earlier sequences -- and hence the seeds each of those programs gets --
    are exactly what rounds one and two ran.
    """
    yield from rvv_ref.ime_legal_configs(vlen)
    yield from rvv_ref.ime_legal_configs(vlen, sews=rvv_ref.WIDENING_SEWS,
                                         ws=(4,))
    for w in (2, 8):
        yield from rvv_ref.ime_legal_configs(
            vlen, sews=rvv_ref.WIDENING_SEWS_BY_W[w], ws=(w,))
    yield from rvv_ref.ime_legal_configs(vlen, tloads=("t",))
    yield from rvv_ref.ime_legal_configs(vlen, kinds=("fp",))
    # Round five, last: the C-layout probe, full VL only (see
    # rvv_ref.clayout_capable) and in the same W=1-then-W=4 tier order
    # ime_tests.directed_suite emits it in.
    yield from _clayout_geometries(vlen)
    # Round six, appended last for the same append-don't-interleave reason:
    # every earlier tier keeps the position -- and therefore the seeds --
    # it had before.
    yield from rvv_ref.mx_legal_configs(vlen)
    # Round seven's widening floating-point tier, appended last for the
    # same append-don't-interleave reason.  Only the cells whose element
    # formats the IME specification defines are swept: the OFP cells are
    # filtered against rvv_ref.fpw_resolved_cells rather than being
    # silently absent, so that 'declared unsupported' and 'forgotten' are
    # different and distinguishable states.
    _resolved = set(rvv_ref.fpw_resolved_cells())
    yield from (g for g in rvv_ref.fpw_legal_configs(vlen)
                if (g.w, g.sew) in _resolved)
    # Round nine's integer tiers, appended last for the same reason: the new
    # Int4 / Int64 cells over every N, and the signedness rows of all
    # thirteen cells rotated over every N -- the directed gate's set.
    yield from (g for _p, g in ime_tests.int9_geometries(
        vlen, ime_tests.INT9_TOKENS, full_vl_only=False))
    # ... and the W=1 narrow floating-point cells, every N.
    yield from rvv_ref.fpn_legal_configs(vlen)


def check_round_seven_sub_byte_path(vlen: int = 256) -> None:
    """The nibble path, re-proved at every round-seven OFP4 geometry.

    Round six's first cut computed ``width = eew // 8`` in Machine.vget and
    vset, which is 0 at EEW=4, so every MXINT4 A/B read returned 0 and 88
    bfloat16 cells failed with what looked like an arithmetic bug.  Round
    seven brings EEW=4 back -- OFP4 (E2M1) at (W=2,SEW=8), (W=4,SEW=16) and
    (W=8,SEW=32) -- so the same hole would reopen in the same place.

    The path is shared, so this does not re-implement it; it re-exercises it
    over the *round-seven* register-group extents, which are new: an OFP4 A
    tile at LMUL=8 spans more elements per group than any round-six MXINT4
    tile did, and an index that runs past v31 must still be caught.

    This is deliberately format-independent -- it checks packing and
    addressing over raw nibbles, not OFP4 values -- so it runs today and
    keeps running unchanged once the OCP documents land.
    """
    ofp4 = [g for g in rvv_ref.fpw_legal_configs(vlen) if g.eew_ab == 4]
    assert ofp4, "no EEW_AB=4 round-seven geometry to exercise"
    cells = {(g.w, g.sew) for g in ofp4}
    assert cells == {(2, 8), (4, 16), (8, 32)}, cells

    for geom in ofp4:
        m = Machine(vlen)
        per_reg = vlen // 4
        # Every nibble of one m1 group, written then read back, with the
        # even/odd order the spec fixes (1207-1219).  A byte-swapped
        # implementation passes a symmetric pattern, so the value written to
        # index i is a function of i that is *not* symmetric under swapping
        # neighbouring pairs.
        for i in range(per_reg):
            m.vset(8, i, 4, (i * 7 + 3) & 0xF)
        for i in range(per_reg):
            got = m.vget(8, i, 4)
            assert got == (i * 7 + 3) & 0xF, (geom.describe(), i, got)
        # ... and the packing is observable as bytes: element 2n in the low
        # nibble of byte n, element 2n+1 in the high nibble.
        for n in range(per_reg // 2):
            byte = m.v[8][n]
            assert byte & 0xF == (2 * n * 7 + 3) & 0xF, (geom.describe(), n)
            assert byte >> 4 == ((2 * n + 1) * 7 + 3) & 0xF, (
                geom.describe(), n)
        # Reading past the group's last register must raise, not wrap into
        # v0 -- at LMUL=8 an OFP4 K row is the widest extent in the round.
        try:
            m.vget(24, per_reg * 8 + 1, 4)
        except SimError:
            pass
        else:
            raise AssertionError(
                f"{geom.describe()}: a read past v31 must raise")

    # Nothing narrower than a nibble is modelled, and must say so rather
    # than silently returning a truncated value.
    m = Machine(vlen)
    for eew in (1, 2):
        try:
            m.vget(8, 0, eew)
        except SimError:
            continue
        raise AssertionError(f"EEW={eew} must not be modelled")


def check_round_seven_sweep_is_live(vlen: int = 256) -> None:
    """Round seven is in the executed sweep, and only for supported cells.

    This replaces an earlier invariant that asserted the *opposite* -- that
    no kind='fpw' geometry had entered the sweep -- which was right while the
    generators did not exist and became a lie the moment they did.  It is
    restated rather than deleted so that the rule it encoded survives: a
    geometry may be swept only once there are programs behind it, because a
    sweep that counts geometries it cannot run reports coverage it does not
    have.

    The obligation now runs the other way, and both halves are asserted
    because either alone is satisfied by an empty pool:

    * every resolved cell is represented in the sweep, and
    * no unresolved (OFP) cell is, so an extension this implementation
      declares unsupported is never judged.
    """
    # W=1 (round nine's narrow vfmmacc.vv cells, rvv_ref.FPN_CELLS) rides
    # kind='fpw' too but is not part of the widening table; it is checked by
    # check_round_nine_sweep_is_live.
    swept = {(g.w, g.sew) for g in _every_geometry(vlen)
             if g.kind == "fpw" and g.w > 1}
    resolved = set(rvv_ref.fpw_resolved_cells())
    assert resolved, "no resolved cell -- this check would be vacuous"
    assert swept == resolved, (sorted(swept), sorted(resolved))
    unresolved = set(rvv_ref.FP_CELLS) - resolved
    assert unresolved, "no unresolved cell -- the OFP exclusion is untested"
    assert not (swept & unresolved), sorted(swept & unresolved)
    # And an OFP cell must still refuse to resolve, so that a future edit
    # which populates FP_FORMAT_TABLE by guesswork fails here rather than
    # quietly widening the sweep.
    for (w, sew) in sorted(unresolved):
        try:
            rvv_ref.fpw_rows(w, sew)
        except rvv_ref.OCPSpecUnavailable:
            continue
        raise AssertionError(
            f"W={w} SEW={sew} resolved its formats without the OCP "
            f"documents -- FP_FORMAT_TABLE has been populated by guesswork")

def check_round_nine_sweep_is_live(vlen: int = 256) -> None:
    """Round nine's cells and signedness rows are all in the sweep."""
    geoms = [g for g in _every_geometry(vlen)
             if g.emul_c != 16 and _allocatable(g)]
    fpn = {(g.w, g.sew) for g in geoms if g.kind == "fpw" and g.w == 1}
    assert fpn == set(rvv_ref.FPN_CELLS), sorted(fpn)
    assert all(g.mnemonic == "vfmmacc.vv" for g in geoms
               if g.kind == "fpw" and g.w == 1)
    rows = {((g.w, g.sew), tuple(g.altfmt_ab)) for g in geoms
            if g.kind == "int" and g.check == "pair" and g.tload == "op"}
    for cell in rvv_ref.INT_CELLS:
        for signs in rvv_ref.INT_SIGNS:
            assert (cell, signs) in rows, (cell, signs)


def check_round_eight_sweep_is_live(vlen: int = 256) -> None:
    """Round eight's OFP8 cells are swept, vf8wmmacc.vv included, E2M1 not."""
    swept = {(g.w, g.sew) for g in _every_geometry(vlen)
             if g.kind == "fpw" and g.w > 1}
    for cell in rvv_ref.FPW_ROUND_EIGHT_CELLS:
        assert cell in swept, cell
    for cell in ((2, 8), (4, 16), (8, 32)):
        assert cell not in swept, cell
    mnems = {g.mnemonic for g in _every_geometry(vlen) if g.kind == "fpw"}
    assert "vf8wmmacc.vv" in mnems, sorted(mnems)
    # The three tiers are each reachable from the sweep's seed mapping.
    g = next(g for g in _every_geometry(vlen)
             if g.kind == "fpw" and (g.w, g.sew) == (4, 32)
             and g.emul_c != 16 and _allocatable(g))
    tiers = [_fpw_sweep_tier(g, seed) for seed in range(3)]
    assert tiers == ["golden", "exact", "special"], tiers
    for seed in range(3):
        code, out = simulate(g, seed)
        assert code == ime_tests.EXIT_PASS and "TITAN PASS" in out, (seed, out)
    # Round seven's mapping is untouched.
    g7 = next(g for g in _every_geometry(vlen)
              if g.kind == "fpw" and (g.w, g.sew) == (2, 32)
              and g.emul_c != 16 and _allocatable(g))
    assert [_fpw_sweep_tier(g7, s) for s in range(4)] == \
        ["golden", "exact", "golden", "exact"]


def check_round_eight_ofp8_decode_is_load_bearing(vlen: int = 256) -> None:
    """A DUT that decodes OFP8 wrongly must FAIL the programs.

    The rvv_ref controls prove the *reference* distinguishes right from
    wrong.  This proves the *programs* do: each program is built with the
    correct decoder (so its golden bytes are right), then run on a model
    whose OFP8 decode is sabotaged, and must print TITAN FAIL.  Sabotages:
    E4M3 read IEEE-shaped (0x7E = 448 becomes NaN), E4M3 <-> E5M2 swapped,
    and subnormals flushed to zero.
    """
    good = rvv_ref._fpf_unpack_generic
    e4, e5 = rvv_ref.fp_format("e4m3"), rvv_ref.fp_format("e5m2")
    ieee_e4 = rvv_ref.replace(e4, has_inf=True, nan_rule="ieee",
                              name="e4m3_ieee")

    def as_ieee(bits, fmt):
        return good(bits, ieee_e4 if fmt.name == "e4m3" else fmt)

    def swapped(bits, fmt):
        return good(bits, {"e4m3": e5, "e5m2": e4}.get(fmt.name, fmt))

    def flushed(bits, fmt):
        k, sgn, v = good(bits, fmt)
        if k == "num" and (bits >> (fmt.prec - 1)) & fmt.emax == 0:
            v = type(v)(0)
        return k, sgn, v

    rng = random.Random(808)
    tally = []
    for cell in rvv_ref.FPW_ROUND_EIGHT_CELLS:
        geom = next(g for g in rvv_ref.fpw_legal_configs(vlen,
                                                         full_vl_only=True)
                    if (g.w, g.sew) == cell and g.emul_c != 16
                    and _allocatable(g) and g.lam * g.lmul >= 2)
        rows = rvv_ref.fpw_rows(*cell)
        for bad_name, bad in (("ieee-shaped E4M3", as_ieee),
                              ("E4M3<->E5M2 swap", swapped),
                              ("subnormal flush", flushed)):
            caught = total = 0
            for key, row in rows.items():
                if bad is as_ieee and "e4m3" not in (row[0].name,
                                                     row[1].name):
                    continue
                for tier, make in (("golden", rvv_ref.fpw_case),
                                   ("special", rvv_ref.fpw_special_case)):
                    plan = ime_tests.FpwPlan(geom, key, tier,
                                             *make(geom, row, rng),
                                             tag="decode-control")
                    program = assemble(ime_tests.emit_fpw_test(plan))
                    # The same program passes on the correct model, so a
                    # failure below is the sabotage and nothing else.
                    assert run(program, Machine(vlen=vlen)) == \
                        ime_tests.EXIT_PASS, (cell, key, tier)
                    rvv_ref._fpf_unpack_generic = bad
                    try:
                        code = run(program, Machine(vlen=vlen))
                    finally:
                        rvv_ref._fpf_unpack_generic = good
                    caught += code != ime_tests.EXIT_PASS
                    total += 1
            assert caught, f"{cell}: {bad_name} DUT passed every program"
            tally.append((cell, bad_name, caught, total))
    return tally


# ---------------------------------------------------------------------------
# round nine: integer negative controls
# ---------------------------------------------------------------------------

def _r9_int_pool(vlen: int):
    """One full-VL program shape per (cell, signedness row), all 13 x 4.

    The shape is the smallest allocatable one with more than one C row and
    more than one K element, so every control has something to see.
    """
    pool = []
    for cell in sorted(rvv_ref.INT_CELLS):
        for signs in rvv_ref.INT_SIGNS:
            for g in rvv_ref.int_cell_geometries(vlen, cell,
                                                 altfmt_ab=signs):
                if g.emul_c != 16 and _allocatable(g) and g.m > 1 \
                        and g.k_eff > 1:
                    pool.append(g)
                    break
    return pool


def _r9_bad_dot(bad: str):
    """A sabotaged Sail int_block_dot for control *bad*."""
    def dot(machine, vs1, vs2, i, j, geom, ua, ub):
        eew = geom.eew_ab
        sa, sb = ua, ub
        if bad == "ext_swap":
            sa, sb = 1 - ua, 1 - ub           # sext <-> zext, both operands
        elif bad == "b_as_signed":
            sb = 0                            # altfmt_B ignored
        elif bad == "ab_sign_swap":
            sa, sb = ub, ua                   # altfmt_A applied to B
        total = 0
        for k in range(geom.k_eff):
            ia, ib = _AB_INDEX(i, k, geom), _AB_INDEX(j, k, geom)
            if bad == "nibble_order" and eew == 4:
                ia ^= 1                       # A side: high nibble first
            if bad == "int4_as_int8" and eew == 4:
                # The whole byte holding element k, read as an 8-bit value.
                a = rvv_ref.int_operand(machine.vget(vs1, ia // 2, 8), 8, sa)
                b = rvv_ref.int_operand(machine.vget(vs2, ib // 2, 8), 8, sb)
            else:
                a = rvv_ref.int_operand(machine.vget(vs1, ia, eew), eew, sa)
                b = rvv_ref.int_operand(machine.vget(vs2, ib, eew), eew, sb)
            total += a * b
        return total
    return dot


def _acc32_writeback(acc: int, sew: int) -> int:
    """An Int64 accumulator that only keeps 32 bits (sign-extended)."""
    return _sext(acc, 32) if sew == 64 else acc


def check_round_nine_int_controls(vlen: int = 256):
    """Six round-nine mistakes, each run against the emitted programs.

    For every control: every program the mistake *can* affect
    (rvv_ref.r9_control_applies) must FAIL on the sabotaged model, and --
    for the three signedness mistakes at W=1, where the architecture makes
    them invisible -- the program must still PASS, which is the proof that
    the W=1 blindness is structural and not a weak test.  Every program in
    the pool PASSes on the correct model first.  Returns
    {control: (caught, applicable)} for the report.
    """
    global _INT_DOT, _INT_WRITEBACK
    pool = _r9_int_pool(vlen)
    assert {(g.w, g.sew) for g in pool} == set(rvv_ref.INT_CELLS), vlen
    for geom in pool:
        code, out = simulate(geom, seed=0)
        assert code == ime_tests.EXIT_PASS and "TITAN PASS" in out, \
            (geom.describe(), out[-300:])
    tally = {}
    for bad in rvv_ref.R9_CONTROLS:
        caught = applicable = 0
        try:
            if bad == "acc32":
                _INT_WRITEBACK = _acc32_writeback
            else:
                _INT_DOT = _r9_bad_dot(bad)
            for geom in pool:
                code, out = simulate(geom, seed=0)
                failed = code != ime_tests.EXIT_PASS and "TITAN FAIL" in out
                if rvv_ref.r9_control_applies(bad, geom):
                    applicable += 1
                    assert failed, (bad, geom.describe(), out[-200:])
                    caught += 1
                elif bad in ("ext_swap", "b_as_signed", "ab_sign_swap") \
                        and geom.w == 1:
                    assert not failed, (bad, geom.describe())
        finally:
            _INT_DOT, _INT_WRITEBACK = _int_dot, _int_writeback
        assert applicable, bad
        tally[bad] = (caught, applicable)
    return tally


def _r9_fpn_programs(vlen: int):
    """(cell, row, program) per narrow cell x row: one golden, one special."""
    progs = []
    for cell in rvv_ref.fpn_resolved_cells():
        geoms = [g for g in rvv_ref.fpn_legal_configs(vlen, full_vl_only=True)
                 if (g.w, g.sew) == cell and g.emul_c != 16
                 and _allocatable(g)]
        gold = next(g for g in geoms if g.k_eff >= 2 and g.m > 1)
        spec = rvv_ref.fpn_special_geometry(geoms)
        for idx, (key, row) in enumerate(rvv_ref.fpw_rows(*cell).items()):
            for geom, tier in ((gold, "golden"), (spec, "special")):
                rng = random.Random(900 + idx)
                case = (rvv_ref.fpn_special_case(geom, row, rng)
                        if tier == "special"
                        else rvv_ref.fpn_case(geom, row, rng))
                plan = ime_tests.FpwPlan(geom, key, tier, *case, tag="ctl")
                progs.append((cell, row, geom,
                              assemble(ime_tests.emit_fpw_test(plan))))
    return progs


def check_round_nine_fpn_controls(vlen: int = 256):
    """Six narrow-FP mistakes, each run against emitted programs.

    Every program passes on the correct model; for every control and every
    (cell, encoding row) it applies to, at least one of that row's two
    programs (golden, special) must FAIL -- so no row of any cell is blind
    to any of them.  Returns {control: (rows caught, rows applicable,
    programs failed)}.
    """
    global _FPN_GEMM
    progs = _r9_fpn_programs(vlen)
    for cell, row, geom, prog in progs:
        machine = Machine(vlen=vlen)
        code = run(prog, machine)
        out = "".join(machine.stdout)
        assert code == ime_tests.EXIT_PASS and "TITAN PASS" in out, \
            (cell, [f.name for f in row], out[-300:])
    tally = {}
    for bad in rvv_ref.FPN_CONTROLS:
        rows_hit, rows_app, failed = set(), set(), 0
        _FPN_GEMM = (lambda bad_: lambda *args:
                     rvv_ref.fpn_sabotaged_gemm(bad_, *args))(bad)
        try:
            for cell, row, geom, prog in progs:
                if not rvv_ref.fpn_control_applies(bad, geom, row):
                    continue
                rid = (cell, tuple(f.name for f in row))
                rows_app.add(rid)
                machine = Machine(vlen=vlen)
                code = run(prog, machine)
                if code != ime_tests.EXIT_PASS and \
                        "TITAN FAIL" in "".join(machine.stdout):
                    rows_hit.add(rid)
                    failed += 1
        finally:
            _FPN_GEMM = rvv_ref.fpn_reference_gemm
        assert rows_app and rows_hit == rows_app, \
            (bad, sorted(rows_app - rows_hit))
        tally[bad] = (len(rows_hit), len(rows_app), failed)
    return tally


def _clayout_geometries(vlen: int):
    """rvv_ref's clayout tier, minus anything this harness cannot allocate."""
    return [g for g in rvv_ref.clayout_geometries(vlen) if _allocatable(g)]


def _allocatable(geom: TileGeometry) -> bool:
    try:
        ime_tests.VectorAlloc.allocate(geom,
                                       reserve_v0=geom.kind == "mx")
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
