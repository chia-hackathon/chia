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

_DIRECTIVE_WIDTH = {".byte": 1, ".half": 2, ".word": 4, ".dword": 8}


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
        width = eew // 8
        return int.from_bytes(self.v[reg][pos * width:(pos + 1) * width],
                              "little")

    def vset(self, base: int, index: int, eew: int, value: int) -> None:
        per_reg = self.vlen // eew
        reg, pos = base + index // per_reg, index % per_reg
        if reg > 31:
            raise SimError(f"vector group v{base} + {index} runs past v31")
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
        pc += 1
        x = machine.x

        if mnemonic == "li":
            x[_reg(ops[0])] = _sext(int(ops[1], 0), 64)
        elif mnemonic == "la":
            x[_reg(ops[0])] = program.data[ops[1]]
        elif mnemonic == "mv":
            x[_reg(ops[0])] = x[_reg(ops[1])]
        elif mnemonic == "add":
            x[_reg(ops[0])] = _sext(x[_reg(ops[1])] + x[_reg(ops[2])], 64)
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
            if ops[1] != "vtype":
                raise SimError(f"csrr {ops[1]} is not modelled")
            x[_reg(ops[0])] = machine.vtype
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
        elif mnemonic == "vmul.vv":
            _vmul(machine, ops)
        elif mnemonic == "vmv.s.x":
            machine.vset(_vreg(ops[0]), 0, machine.sew, x[_reg(ops[1])])
        elif mnemonic == "vmv.x.s":
            x[_reg(ops[0])] = _sext(machine.vget(_vreg(ops[1]), 0,
                                                 machine.sew), machine.sew)
        elif mnemonic == "vredsum.vs":
            _vredsum(machine, ops)
        elif mnemonic == ".insn":
            _ime(machine, int(ops[1], 0))
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


def _vsetvl(machine: Machine, ops: List[str]) -> None:
    avl = machine.x[_reg(ops[1])]
    machine.vtype = machine.x[_reg(ops[2])]
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


def _geometry(machine: Machine, w: int = 1) -> TileGeometry:
    """Decode vtype into a tile geometry.

    *w* comes from the *instruction*, not from vtype: vtype.SEW is the C
    accumulator width for the whole vmmacc family, and the widening factor
    is what the opcode selects (Sail: ``decode_gemm_geometry(4)`` inside
    vqmmacc.vv's body, ``decode_gemm_geometry(1)`` inside vmmacc.vv's).  The
    tile load/stores decode their own geometry with W=1, because they move
    SEW-wide storage elements and never look inside a packed element.
    """
    return TileGeometry(machine.vlen, machine.sew, machine.lam,
                        machine.lmul, machine.vl, w)


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
            flat_idx = rvv_ref.tile_reg_idx(i, geom.lmul, geom.lam,
                                            geom.elems_per_reg)
            if name in ("vmtl.v", "vmttl.v"):
                machine.vset(reg, flat_idx, sew,
                             machine.load(addr, width, signed=False))
            else:
                machine.store(addr, width, machine.vget(reg, flat_idx, sew))
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
    geom = _geometry(machine, _MACC_W[name])
    eew_ab = geom.eew_ab
    vd, vs1, vs2 = fields["vd"], fields["vs1"], fields["vs2"]
    for i in range(geom.m):
        for j in range(geom.n):
            c_flat = rvv_ref.c_element_index(i, j, geom)
            acc = _sext(machine.vget(vd, c_flat, sew), sew)
            for k in range(geom.k_eff):
                a = machine.vget(vs1, rvv_ref.ab_element_index(i, k, geom),
                                 eew_ab)
                b = machine.vget(vs2, rvv_ref.ab_element_index(j, k, geom),
                                 eew_ab)
                acc += _sext(a, eew_ab) * _sext(b, eew_ab)
            machine.vset(vd, c_flat, sew, acc)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def simulate(geom: TileGeometry, seed: int = 0) -> Tuple[int, str]:
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
        if name in _MACC_W:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument("--emit", metavar="SEW,LAMBDA,LMUL,N[,W][,op|t]",
                        help="run one geometry and print its output")
    parser.add_argument("--seeds", type=int, default=3,
                        help="random cases per geometry")
    args = parser.parse_args()

    if args.emit:
        toks = args.emit.split(",")
        tload = "op"
        if toks[-1] in ("op", "t"):
            tload = toks.pop()
        sew, lam, lmul, n, *rest = (int(t) for t in toks)
        geom = TileGeometry(args.vlen, sew, lam, lmul, n * lam * lmul,
                            rest[0] if rest else 1, tload)
        code, output = simulate(geom)
        print(f"{geom.describe()}\nexit={code}\n{output}")
        return 0 if code == ime_tests.EXIT_PASS else 1

    geometries = [g for g in _every_geometry(args.vlen)
                  if g.emul_c != 16 and _allocatable(g)]
    failures, ran = [], 0
    for geom in geometries:
        for seed in range(args.seeds):
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
    for control in controls:
        check_negative_control(control)
        print(f"  ok  check_negative_control  {control.describe()}")
    check_transposing_is_load_bearing(args.vlen)
    print("  ok  check_transposing_is_load_bearing")

    tally = {}
    for g in geometries:
        tally[(g.w, g.tload)] = tally.get((g.w, g.tload), 0) + 1
    breakdown = " + ".join(f"{n} W={w}/{tl}"
                           for (w, tl), n in sorted(tally.items()))
    print(f"\nVLEN={args.vlen}: {ran} program executions across "
          f"{len(geometries)} geometries ({breakdown}), "
          f"{len(failures)} failed")
    return 1 if failures else 0


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


def _allocatable(geom: TileGeometry) -> bool:
    try:
        ime_tests.VectorAlloc.allocate(geom)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
