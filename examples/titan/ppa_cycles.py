"""Round 12 cycle suite (pure part): deterministic bare-metal cycle probes.

Area bought with cycles must not pass the PPA judge, and until round 12 no
judge looked at cycles at all (the r11 FU takes 8..19 cycles per uop and
nobody ever measured it).  The suite is ~9 programs on the non-cosim
Verilator build (the loop's own DIRECTED_CONFIG artifact -- no DPI trace
bridge, so no cosim nondeterminism), each a *directed* S1 program already
emitted by ``ime_tests`` with one change: ``rdcycle`` around the single IME
multiply-accumulate.  The encoders, data and register allocation are
therefore exactly the directed suite's -- nothing here hand-encodes an
instruction.

Measurement.  The macc is decoupled from the scalar core, so a bare rdcycle
pair would time only its issue.  Before the first rdcycle the program reads
every register the macc consumes (``vmv.x.s`` per tile register, each
followed by an ``add`` that stalls the in-order scalar pipe until the value
has really come back from the vector unit); after the macc it reads the
whole C accumulator group the same way and only then reads rdcycle again.
The delta is "macc issued -> last C register readable", in core cycles.

After printing ``TITAN CYCLES <bench> <delta>`` the program leaves through the
test's own ``.Lret`` exit with status 0: the IME-vs-RVV comparison is not the
point here (S1 does that) and skipping the unrolled RVV reference keeps each
run short.

Compare rule (§3.5): exact numbers against a stored r35 reference; a
benchmark may be faster than r35 (an improvement), never slower.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import ime_tests

#: (bench id, directed test name at VLEN=256, what it exercises).  Names are
#: deterministic (seeded generators); ``check_bench_names`` asserts they exist.
BENCHES: Tuple[Tuple[str, str, str], ...] = (
    ("int8_vqmmacc", "ime_q_sew32_lam1_lmul4_n8",
     "vqmmacc int8 -> int32, LMUL=4"),
    ("fp16_vfwmmacc", "ime_fpw_golden_w2_sew32_l1_m4_000",
     "vfwmmacc binary16 x binary16 -> binary32"),
    ("bf16in_vfwmmacc", "ime_fpw_golden_w2_sew32_l1_m4_110",
     "vfwmmacc bfloat16 x bfloat16 -> binary32"),
    ("ofp4_vf8wmmacc", "ime_fp4_golden_w8_sew32_l1_m1_n8_000",
     "vf8wmmacc OFP4 (E2M1) -> binary32"),
    ("mxfp8_vfwmmacc", "ime_mxf_golden_w2_sew16_bs0_l2_m4_n8_010",
     "microscaled MXFP8 (E4M3 x E5M2) -> binary16"),
    ("mxint8_vfwimmacc", "ime_mx_sew16_lam2_lmul4_w2_n8_af0_bs0",
     "microscaled MXINT8 -> binary16"),
    ("fp32_vfmmacc", "ime_f_sew32_lam1_lmul4_n8",
     "vfmmacc binary32 (SEW=32)"),
    ("fp64_vfmmacc", "ime_f_sew64_lam1_lmul4_n4",
     "vfmmacc binary64 (SEW=64)"),
    ("bf16c_vfwmmacc", "ime_fpw_golden_w2_sew16_l2_m4_001",
     "vfwmmacc E4M3 x E4M3 -> bfloat16 C (bf16-C path)"),
)

CYCLES_RE = re.compile(r"TITAN CYCLES (?P<bench>\w+) (?P<n>-?\d+)")

_SUITE_CACHE: Dict[int, Dict[str, tuple]] = {}


def bench_names() -> List[str]:
    return [b for b, _, _ in BENCHES]


def _suite(vlen: int) -> Dict[str, tuple]:
    """{test name: (asm, geom)} of the directed suite (memoised: ~35 s)."""
    if vlen not in _SUITE_CACHE:
        import constants
        _SUITE_CACHE[vlen] = {
            name: (asm, geom) for name, asm, geom in
            ime_tests.directed_suite(vlen, insns=constants.ALL_SCOPE)}
    return _SUITE_CACHE[vlen]


def check_bench_names(vlen: int = 256) -> List[str]:
    """Names of BENCHES that the directed suite does not contain."""
    suite = _suite(vlen)
    return [t for _, t, _ in BENCHES if t not in suite]


def _regs_to_read(geom, alloc) -> List[int]:
    """Vector registers the macc consumes: A and B tile groups' registers,
    the C group, and v0 (E8M0 scales / A base)."""
    regs = {0}
    regs.update(range(alloc.a, alloc.a + geom.lmul))
    regs.update(range(alloc.b, alloc.b + geom.lmul))
    regs.update(range(alloc.c, alloc.c + geom.emul_c))
    return sorted(regs)


def instrument(asm: str, geom, bench: str) -> str:
    """The directed program ``asm`` with a cycle probe around its macc.

    Raises ValueError when the macc line is not unique or the program has no
    ``.Lret`` exit (the probe would not be sound)."""
    lines = asm.split("\n")
    marker = f"# {geom.mnemonic} v"
    idx = [i for i, ln in enumerate(lines)
           if ln.lstrip().startswith(".insn") and marker in ln]
    if len(idx) != 1:
        raise ValueError(f"{bench}: expected exactly one {geom.mnemonic} "
                         f"line, found {len(idx)}")
    if not any(ln.strip() == ".Lret:" for ln in lines):
        raise ValueError(f"{bench}: program has no .Lret exit")
    reserve = geom.kind in ("mx", "mxf")
    alloc = ime_tests.VectorAlloc.allocate(geom, reserve_v0=reserve)
    regs = _regs_to_read(geom, alloc)
    c_regs = list(range(alloc.c, alloc.c + geom.emul_c))
    pre = ["", f"    # ---- cycle probe ({bench}): drain operands, start ----"]
    for r in regs:
        pre += [f"    vmv.x.s t4, v{r}", "    add   t4, t4, x0"]
    pre += ["    rdcycle t5"]
    post = ["    # ---- cycle probe: wait for the whole C group, stop ----"]
    for r in c_regs:
        post += [f"    vmv.x.s t6, v{r}", "    add   t6, t6, x0"]
    post += ["    rdcycle t3", "    sub   t3, t3, t5",
             "    la    a0, .Lfmt_cyc", "    mv    a1, t3",
             "    call  printf", "    li    s1, 0", "    j     .Lret", ""]
    i = idx[0]
    out = lines[:i] + pre + [lines[i]] + post + lines[i + 1:]
    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    text += (f'    .data\n    .balign 8\n'
             f'.Lfmt_cyc: .asciz "TITAN CYCLES {bench} %d\\n"\n')
    return text


def programs(vlen: int = 256, names: Optional[Sequence[str]] = None
             ) -> List[Tuple[str, str, str]]:
    """[(bench, program name, instrumented asm)] for the suite (or the
    subset ``names`` of bench ids)."""
    suite = _suite(vlen)
    out = []
    for bench, test, _ in BENCHES:
        if names and bench not in names:
            continue
        if test not in suite:
            raise ValueError(f"cycle bench {bench}: directed test {test} "
                             f"does not exist at VLEN={vlen}")
        asm, geom = suite[test]
        out.append((bench, f"cyc_{bench}", instrument(asm, geom, bench)))
    return out


def parse_cycles(log: str) -> Dict[str, int]:
    """{bench: delta} from a simulator log (the last line per bench wins)."""
    res: Dict[str, int] = {}
    for m in CYCLES_RE.finditer(log or ""):
        res[m.group("bench")] = int(m.group("n"))
    return res


def compare(measured: Dict[str, Optional[int]],
            reference: Optional[Dict[str, int]],
            benches: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Pass/fail against the stored reference.

    A benchmark that was not measured (None/missing) fails -- a program that
    crashed or skipped is not "no regression".  Without a reference nothing
    can regress: the result is ``pass`` with ``reference_missing`` set so the
    caller can say so (calibration run)."""
    benches = list(benches or bench_names())
    rows, reasons, ok = [], [], True
    for b in benches:
        got = measured.get(b)
        ref = (reference or {}).get(b)
        row = {"bench": b, "cycles": got, "ref": ref, "delta": None,
               "status": "ok"}
        if got is None:
            row["status"] = "NOT MEASURED"
            ok = False
            reasons.append(f"{b}: no cycle count (program crashed, skipped "
                           f"or timed out)")
        elif ref is None:
            row["status"] = "no reference"
        else:
            row["delta"] = got - ref
            if got > ref:
                row["status"] = "SLOWER"
                ok = False
                reasons.append(f"{b}: {got} cycles > r35 reference {ref} "
                               f"({got - ref:+d})")
            elif got < ref:
                row["status"] = "faster"
        rows.append(row)
    return {"pass": ok, "rows": rows, "reasons": reasons,
            "reference_missing": not reference}


def format_table(res: Dict[str, object]) -> str:
    """The cycle table the agent reads."""
    lines = ["| benchmark | cycles | r35 | delta | status |",
             "|---|---:|---:|---:|---|"]
    for r in res.get("rows") or []:
        lines.append("| {b} | {c} | {ref} | {d} | {s} |".format(
            b=r["bench"], c="-" if r["cycles"] is None else r["cycles"],
            ref="-" if r["ref"] is None else r["ref"],
            d="-" if r["delta"] is None else f"{r['delta']:+d}",
            s=r["status"]))
    if res.get("reference_missing"):
        lines.append("\n(no r35 reference configured: cycles reported only)")
    return "\n".join(lines) + "\n"
