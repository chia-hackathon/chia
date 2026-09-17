"""End-to-end speed projection for Llama-3.2-1B on the AETHER SoC.

The SoC (Saturn RVV VLEN256/DLEN128 + Gemmini 16x16 int8) only runs under
Verilator at ~1.1k cycles/s, so the whole model can never be simulated. This
module instead:

  1. takes the operator inventory from `llama_model.py`,
  2. costs every operator either from a MEASURED single-kernel cycle count
     (extrapolated linearly in MACs / elements, with an optional fixed
     per-call overhead) or, when no measurement exists, from a roofline lower
     bound multiplied by a conservative fudge factor,
  3. reports cycles/token, tokens/s, the per-operator and per-device
     breakdown, and an Amdahl sensitivity table ("if kernel X gets k x
     faster, the end-to-end speedup is ..."),
  4. reports the pure roofline lower bound as the optimisation ceiling.

Usage
-----
    python -m loop.llama_project --scenario decode --S 512
    python -m loop.llama_project --scenario prefill --N 64
    python -m loop.llama_project --scenario decode --S 512 \
        --measured out/llama-profile/measured_cycles.json
    python -m loop.llama_project --scenario decode --S 512 --overlap
    python -m loop.llama_project --self-test
    python -m loop.llama_project --emit-doc out/llama-profile/op_inventory.md

Hardware assumptions (all overridable on the CLI; see `HW` below)
-----------------------------------------------------------------
Gemmini 16x16 int8 systolic array
    16*16 = 256 int8 MACs/cycle at full utilisation. This is the *peak*; a
    GEMV (N=1) cannot fill a 16-wide output dimension, so measured GEMV
    numbers will be far off this - which is exactly why the roofline is only
    ever quoted as a lower bound.
Saturn RVV, VLEN=256 bit, DLEN=128 bit
    DLEN is the per-cycle datapath width, so the sustainable rate is
    128 bit/cycle = 16 int8 lanes -> 16 int8 MACs/cycle, or
    128/32 = 4 fp32 elements/cycle for the norm/softmax/SiLU reductions.
Memory
    16 bytes/cycle of sustained bandwidth into the accelerators. This is a
    guess for an L2/DRAM path on this class of SoC; it matters a lot, because
    at int8 the decode path is weight-streaming bound, not MAC bound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # works both as `python -m loop.llama_project` and `python llama_project.py`
    from loop.llama_model import (MODELS, DEFAULT_MODEL, ModelSpec, Op,
                                  model_ops, totals, tokens_in_scenario)
    from loop import llama_model
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llama_model import (MODELS, DEFAULT_MODEL, ModelSpec, Op,
                             model_ops, totals, tokens_in_scenario)
    import llama_model


# --------------------------------------------------------------------------
# Hardware model
# --------------------------------------------------------------------------
@dataclass
class HW:
    clock_hz: float = 1e9
    gemmini_macs_per_cycle: float = 256.0     # 16x16 int8 array, peak
    saturn_int8_macs_per_cycle: float = 16.0  # DLEN 128 bit / 8 bit
    saturn_fp32_elems_per_cycle: float = 4.0  # DLEN 128 bit / 32 bit
    mem_bytes_per_cycle: float = 16.0         # sustained, estimate

    def peak_macs(self, device: str) -> float:
        return (self.gemmini_macs_per_cycle if device == "gemmini"
                else self.saturn_int8_macs_per_cycle)


# Which unit runs which kernel. Overridable with `--device kernel=unit`.
DEVICE_MAP: dict[str, str] = {
    "int8_gemv": "gemmini",
    "int8_gemm": "gemmini",
    "lm_head_gemv": "gemmini",
    "lm_head_gemm": "gemmini",
    "attn_scores": "gemmini",
    "attn_pv": "gemmini",
    "rmsnorm": "saturn",
    "rope": "saturn",
    "softmax": "saturn",
    "silu_mul": "saturn",
    "add": "saturn",
    "embedding": "saturn",
}

# Kernels whose cost scales with element count, not MAC count.
ELEMENTWISE = {"rmsnorm", "rope", "softmax", "silu_mul", "add", "embedding"}


# --------------------------------------------------------------------------
# Gemmini/Saturn overlap (opt-in, `--overlap`)
# --------------------------------------------------------------------------
# The projection is a pure SEQUENTIAL sum by default: every operator is costed
# on its own and the totals are added. That is deliberately conservative --
# nothing in the sum knows that the Saturn vector work of a decoder layer can
# be issued INSIDE the shadow of the Gemmini weight DMA of the same layer.
#
# `llama-layer-fused-n1` is the one kernel on the board that measures exactly
# that overlap for one decoder-layer slice (1x2048x512 int8 GEMV on Gemmini
# = 1 MiB of cold weights, plus one head's QK^T / softmax / probs@V on
# Saturn, all in ONE timed region):
#
#   strictly sequential baseline                     206,304 cycles
#   best overlapped schedule (round 9 iter 4)        179,033 cycles  (-13.2%)
#   weight stream alone, Saturn serialised out       161,687 cycles
#   Saturn phases alone, serialised                   28,372 cycles
#   => Saturn cost still EXPOSED in the best kernel   17,346 cycles
#   => Saturn cost hidden in the DMA shadow           11,026 cycles  (38.9%)
#
# So the measured, four-rounds-of-tuning overlap efficiency is
# 17,346 / 28,372 = 0.611 of the Saturn cost surviving as wall time.
DEFAULT_OVERLAP_FACTOR = 17_346.0 / 28_372.0   # 0.6114, see above

OVERLAP_NOTE_MD = """\
### Overlap modelling (`--overlap`, off by default)

`--overlap[=F]` multiplies the cycles of every operator costed on **saturn**
by `F` (default `F = 0.611`) and treats the removed cycles as hidden inside
the Gemmini weight-DMA shadow of the same layer. The hidden total is capped by
the Gemmini cycles available to hide under, so the model degenerates to
`gemmini + F x saturn >= max(gemmini, saturn)`.

**Basis.** `F` is not a guess and not a roofline: it is the measured ratio from
the `llama-layer-fused-n1` kernel, which runs one decoder-layer slice (Gemmini
1x2048x512 int8 GEMV over 1 MiB of cold weights + one head's QK^T, softmax and
probs@V on Saturn) in a single timed region:

| what | cycles | source |
|---|---|---|
| strictly sequential baseline | 206,304 | kernel `reference_cycles` |
| best overlapped schedule | 179,033 | run `20260916-195958-36b4` iter 4 |
| weight stream alone (Saturn serialised out) | 161,687 | round 8 iter 6 `PROBE stream=` |
| Saturn phases alone, serialised | 28,372 | same probe (qk+softmax+pv) |
| Saturn cost still exposed | 17,346 | 179,033 - 161,687 |

`F = 17,346 / 28,372 = 0.611`; end to end that one kernel is 13.2% faster than
its sequential baseline.

**Extrapolation limits — read before quoting an overlapped number.**

1. One shape, one layer. The measurement is a *single* 1 MiB GEMV hiding a
   *single* attention head. In the real layer the Saturn/Gemmini ratio per
   GEMV differs, and the projection applies one scalar to all of them.
2. The LM head (about a fifth of decode cycles) streams 4 MiB of weights with
   **no attention work to hide underneath it**; its Saturn partner ops are
   only rmsnorm-sized. Applying `F` there is the weakest part of the model and
   is why the cap above exists.
3. `F` was achieved by a hand-scheduled kernel after four optimisation rounds.
   Nothing in the generic runtime does this automatically, so an overlapped
   projection describes a *hand-tuned* implementation, not today's software.
4. The residual exposure is NOT bytes: DMA-warming the Saturn operands cost
   more than it saved (round 9 iter 1), halving the interleave grain changed
   nothing (iter 2), and the host issue path is not the limiter (iter 3/4
   probes). `F` should therefore be treated as an achieved constant, not as a
   quantity that keeps improving with effort.

The default output is unchanged when `--overlap` is absent.
"""


def roofline_cycles(op: Op, hw: HW, device: str) -> dict[str, float]:
    """Lower bound on cycles for one op: max(compute, memory)."""
    if op.kernel in ELEMENTWISE:
        compute = op.elems / hw.saturn_fp32_elems_per_cycle
    else:
        compute = op.macs / hw.peak_macs(device)
    memory = op.bytes_total / hw.mem_bytes_per_cycle
    return {"compute": compute, "memory": memory, "cycles": max(compute, memory)}


# --------------------------------------------------------------------------
# Measured-cycle database
# --------------------------------------------------------------------------
@dataclass
class Measurement:
    kernel: str
    work_metric: str
    work_measured: float
    cycles: float
    overhead_cycles: float = 0.0
    device: str | None = None
    source: str = ""

    def project(self, op: Op) -> float:
        work = op.elems if self.work_metric == "elems" else op.macs
        if self.work_measured <= 0:
            return self.overhead_cycles * op.calls
        variable = max(self.cycles - self.overhead_cycles, 0.0)
        return op.calls * self.overhead_cycles + variable * work / self.work_measured


@dataclass
class MeasuredDB:
    by_kernel: dict[str, Measurement] = field(default_factory=dict)
    clock_hz: float | None = None
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> "MeasuredDB":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"measured file not found: {p}")
        raw = json.loads(p.read_text())
        db = cls(clock_hz=raw.get("clock_hz"), path=p)
        for e in raw.get("entries", []):
            if e.get("cycles") is None:
                continue  # placeholder, not yet measured
            db.by_kernel[e["kernel"]] = Measurement(
                kernel=e["kernel"],
                work_metric=e.get("work_metric", "macs"),
                work_measured=float(e["work_measured"]),
                cycles=float(e["cycles"]),
                overhead_cycles=float(e.get("overhead_cycles", 0.0)),
                device=e.get("device"),
                source=e.get("source", ""),
            )
        return db


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------
@dataclass
class CostedOp:
    op: Op
    device: str
    cycles: float
    roofline: float
    rf_compute: float
    rf_memory: float
    estimated: bool          # True == no measurement, roofline x fallback
    bound: str               # "compute" or "memory" (of the roofline)
    # Only meaningful with `--overlap`: the sequential cost this op would have
    # had, and how much of it was hidden in the Gemmini DMA shadow.
    cycles_serial: float | None = None
    hidden_cycles: float = 0.0


@dataclass
class Projection:
    spec: ModelSpec
    scenario: str
    S: int
    N: int
    hw: HW
    costed: list[CostedOp]
    tokens: int
    fallback_factor: float
    measured_path: Path | None
    measured_kernels: list[str]
    # None == pure sequential sum (the default). A float is the measured
    # Saturn-exposure factor applied by `--overlap`; see OVERLAP_NOTE_MD.
    overlap_factor: float | None = None

    # ---- aggregates ------------------------------------------------------
    @property
    def total_cycles(self) -> float:
        return sum(c.cycles for c in self.costed)

    @property
    def total_cycles_sequential(self) -> float:
        """Total with no overlap credit (identical to `total_cycles` by default)."""
        return sum(c.cycles_serial if c.cycles_serial is not None else c.cycles
                   for c in self.costed)

    @property
    def total_hidden(self) -> float:
        return sum(c.hidden_cycles for c in self.costed)

    @property
    def total_roofline(self) -> float:
        return sum(c.roofline for c in self.costed)

    @property
    def cycles_per_token(self) -> float:
        return self.total_cycles / self.tokens

    @property
    def roofline_per_token(self) -> float:
        return self.total_roofline / self.tokens

    @property
    def tokens_per_s(self) -> float:
        return self.hw.clock_hz / self.cycles_per_token

    @property
    def roofline_tokens_per_s(self) -> float:
        return self.hw.clock_hz / self.roofline_per_token

    def group(self, key: str) -> dict[str, dict[str, float]]:
        """key in {'op', 'kernel', 'device'} -> aggregated rows."""
        out: dict[str, dict[str, float]] = {}
        for c in self.costed:
            k = {"op": c.op.name, "kernel": c.op.kernel, "device": c.device}[key]
            r = out.setdefault(k, {"cycles": 0.0, "roofline": 0.0, "macs": 0,
                                   "bytes": 0, "calls": 0, "estimated": 0})
            r["cycles"] += c.cycles
            r["roofline"] += c.roofline
            r["macs"] += c.op.macs
            r["bytes"] += c.op.bytes_total
            r["calls"] += c.op.calls
            r["estimated"] += 1 if c.estimated else 0
        return out

    def amdahl(self, key: str = "kernel", speedups=(2, 4, 8, math.inf)) -> list[dict]:
        tot = self.total_cycles
        rows = []
        for name, r in sorted(self.group(key).items(), key=lambda kv: -kv[1]["cycles"]):
            f = r["cycles"] / tot
            row = {"name": name, "fraction": f}
            for k in speedups:
                row[k] = 1.0 / ((1 - f) + (f / k if k != math.inf else 0.0))
            rows.append(row)
        return rows


def _apply_overlap(costed: list[CostedOp], factor: float) -> None:
    """Hide `1 - factor` of the Saturn cost inside the Gemmini DMA shadow.

    Mutates `costed` in place. The amount hidden is capped by the total Gemmini
    cycles available to hide under, which is what makes the model degenerate to
    `max(gemmini, saturn)` rather than going negative on a Saturn-heavy
    scenario. See `OVERLAP_NOTE_MD` for the measured basis of `factor`.
    """
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"overlap factor must be in [0, 1], got {factor}")
    gemmini_cycles = sum(c.cycles for c in costed if c.device == "gemmini")
    sat = [c for c in costed if c.device != "gemmini"]
    want = sum(c.cycles for c in sat) * (1.0 - factor)
    if want <= 0.0:
        return
    scale = min(1.0, gemmini_cycles / want)   # cannot hide more than exists
    for c in sat:
        c.cycles_serial = c.cycles
        c.hidden_cycles = c.cycles * (1.0 - factor) * scale
        c.cycles = c.cycles - c.hidden_cycles


def project(spec: ModelSpec, scenario: str, *, S: int = 512, N: int = 64,
            hw: HW | None = None, measured: MeasuredDB | None = None,
            fallback_factor: float = 4.0,
            device_map: dict[str, str] | None = None,
            gemv_bytes_per_cycle: float | None = None,
            lmhead_bytes_per_cycle: float | None = None,
            overlap_factor: float | None = None) -> Projection:
    """
    `gemv_bytes_per_cycle` / `lmhead_bytes_per_cycle` bypass the measured-tile
    linear (MAC) extrapolation for the decode GEMV (`int8_gemv`) and the LM
    head GEMV (`lm_head_gemv`) respectively, and instead cost them directly as
    `op.macs / rate` (macs == weight bytes at int8, one byte per MAC). This is
    how to apply a COLD-DRAM measured rate instead of the tile's own warm-L2
    rate: see the 2026-09-12 audit in `out/loop/FINAL_REPORT.md` and
    `out/llama-profile/projection_final_v2.md` for where 7.00 / 6.43 / 6.61
    B/cycle come from. `lmhead_bytes_per_cycle` falls back to
    `gemv_bytes_per_cycle` when only the latter is given. Both default to
    None, which reproduces the original measured-DB / roofline-fallback
    behaviour exactly.

    `overlap_factor` defaults to None, which keeps the projection a pure
    sequential sum (the historical behaviour, bit-for-bit). A float f models
    Gemmini/Saturn overlap: every Saturn operator keeps only `f` of its cycles,
    the rest being issued inside the Gemmini weight-DMA shadow, capped so that
    the hidden total never exceeds the Gemmini cycles it hides under. The
    default `DEFAULT_OVERLAP_FACTOR` is the measured exposure ratio of the
    `llama-layer-fused-n1` kernel; see `OVERLAP_NOTE_MD` for the derivation and
    for the extrapolation limits.
    """
    hw = hw or HW()
    measured = measured or MeasuredDB()
    dmap = {**DEVICE_MAP, **(device_map or {})}
    if lmhead_bytes_per_cycle is None:
        lmhead_bytes_per_cycle = gemv_bytes_per_cycle

    ops = model_ops(spec, scenario, S=S, N=N)
    costed: list[CostedOp] = []
    for op in ops:
        device = dmap.get(op.kernel, "saturn")
        cli_dev = (device_map or {}).get(op.kernel)  # explicit --device KERNEL=UNIT
        # With an explicit --device, prefer a device-suffixed entry
        # (e.g. int8_gemv_gemmini); otherwise fall back to the bare kernel name.
        m = measured.by_kernel.get(f"{op.kernel}_{cli_dev}") if cli_dev else None
        if m is None:
            m = measured.by_kernel.get(op.kernel)
        # A measured entry's own device only wins when the CLI did not pin one.
        if m and m.device and not cli_dev:
            device = m.device
        rf = roofline_cycles(op, hw, device)
        override_rate = None
        if op.kernel == "lm_head_gemv" and lmhead_bytes_per_cycle:
            override_rate = lmhead_bytes_per_cycle
        elif op.kernel == "int8_gemv" and gemv_bytes_per_cycle:
            override_rate = gemv_bytes_per_cycle
        if override_rate:
            cycles, est = op.macs / override_rate, False
        elif m is not None:
            cycles, est = m.project(op), False
        else:
            cycles, est = rf["cycles"] * fallback_factor, True
        costed.append(CostedOp(
            op=op, device=device, cycles=cycles, roofline=rf["cycles"],
            rf_compute=rf["compute"], rf_memory=rf["memory"], estimated=est,
            bound="memory" if rf["memory"] >= rf["compute"] else "compute",
        ))

    if overlap_factor is not None:
        _apply_overlap(costed, overlap_factor)

    return Projection(
        spec=spec, scenario=scenario, S=S, N=N, hw=hw, costed=costed,
        overlap_factor=overlap_factor,
        tokens=tokens_in_scenario(scenario, S=S, N=N),
        fallback_factor=fallback_factor,
        measured_path=measured.path,
        measured_kernels=sorted(measured.by_kernel),
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _fmt(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}k"
    return f"{n:.1f}"


def _table(headers, rows, aligns=None) -> str:
    cols = len(headers)
    aligns = aligns or ["l"] + ["r"] * (cols - 1)
    w = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows
         else len(str(headers[i])) for i in range(cols)]
    def line(cells):
        return "  ".join(
            str(c).ljust(w[i]) if aligns[i] == "l" else str(c).rjust(w[i])
            for i, c in enumerate(cells))
    out = [line(headers), "  ".join("-" * x for x in w)]
    out += [line(r) for r in rows]
    return "\n".join(out)


def render(p: Projection, *, show_ops: bool = True, markdown: bool = False) -> str:
    spec, hw = p.spec, p.hw
    tot = p.total_cycles
    L: list[str] = []
    ctx = f"S={p.S}" if p.scenario == "decode" else f"N={p.N}"
    hdr = f"{spec.name} - {p.scenario} ({ctx}) @ {hw.clock_hz/1e9:.2f} GHz"
    L.append(hdr)
    L.append("=" * len(hdr))
    src = (f"measured: {p.measured_path} -> {', '.join(p.measured_kernels) or 'none usable'}"
           if p.measured_path else "measured: none (all operators estimated)")
    L.append(src)
    L.append(f"unmeasured operators costed as roofline x {p.fallback_factor:g} "
             f"and marked 'est'")
    if p.overlap_factor is not None:
        seq = p.total_cycles_sequential
        L.append(f"OVERLAP MODEL ON: saturn operators keep "
                 f"{p.overlap_factor:.3f} of their cycles; "
                 f"{_fmt(p.total_hidden)} cycles ({100*p.total_hidden/seq:.1f}% "
                 f"of the sequential total) hidden in the Gemmini DMA shadow")
        L.append("  basis: llama-layer-fused-n1 measured 17,346 exposed of "
                 "28,372 serialised Saturn cycles (179,033 vs 161,687 stream, "
                 "run 20260916-195958-36b4 iter 4) = -13.2% end to end on that "
                 "kernel.  EXTRAPOLATION IS WEAK: one shape, one layer, and the "
                 "LM head has no attention to hide under it -- see "
                 "OVERLAP_NOTE_MD in llama_project.py.")
    if not spec.verified:
        L.append("WARNING: model spec NOT verified against config.json")
    L.append("")

    t = totals(op.op for op in p.costed)
    L.append(f"per token: {_fmt(t['macs']/p.tokens)} MAC, "
             f"{_fmt(t['bytes_read']/p.tokens)}B read, "
             f"{_fmt(t['bytes_written']/p.tokens)}B written, "
             f"{t['calls']/p.tokens:.0f} kernel calls")
    L.append("")

    if show_ops:
        L.append("--- per-operator breakdown -------------------------------------")
        rows = []
        for name, r in sorted(p.group("op").items(), key=lambda kv: -kv[1]["cycles"]):
            rows.append([
                name + (" *est" if r["estimated"] else ""),
                f"{r['calls']/p.tokens:.0f}",
                _fmt(r["macs"] / p.tokens),
                _fmt(r["bytes"] / p.tokens),
                _fmt(r["cycles"] / p.tokens),
                f"{100*r['cycles']/tot:5.1f}%",
                _fmt(r["roofline"] / p.tokens),
            ])
        L.append(_table(["operator", "calls/tok", "MAC/tok", "B/tok",
                         "cyc/tok", "share", "roofline"], rows))
        L.append("")

    for key, title in (("kernel", "per kernel type"), ("device", "per device")):
        L.append(f"--- {title} " + "-" * (58 - len(title)))
        rows = []
        for name, r in sorted(p.group(key).items(), key=lambda kv: -kv[1]["cycles"]):
            rows.append([name, f"{r['calls']/p.tokens:.0f}",
                         _fmt(r["macs"] / p.tokens),
                         _fmt(r["cycles"] / p.tokens),
                         f"{100*r['cycles']/tot:5.1f}%",
                         _fmt(r["roofline"] / p.tokens)])
        L.append(_table([key, "calls/tok", "MAC/tok", "cyc/tok", "share", "roofline"], rows))
        L.append("")

    L.append("--- end to end -------------------------------------------------")
    L.append(f"projected   : {_fmt(p.cycles_per_token)} cycles/token  "
             f"-> {p.tokens_per_s:9.3f} tokens/s")
    L.append(f"roofline LB : {_fmt(p.roofline_per_token)} cycles/token  "
             f"-> {p.roofline_tokens_per_s:9.3f} tokens/s")
    rf_c = sum(c.rf_compute for c in p.costed) / p.tokens
    rf_m = sum(c.rf_memory for c in p.costed) / p.tokens
    L.append(f"  roofline split: compute-only {_fmt(rf_c)} cyc/tok "
             f"({hw.clock_hz/rf_c:.2f} tok/s), "
             f"memory-only {_fmt(rf_m)} cyc/tok ({hw.clock_hz/rf_m:.2f} tok/s) "
             f"@ {hw.mem_bytes_per_cycle:g} B/cycle")
    nmem = sum(1 for c in p.costed if c.bound == "memory")
    L.append(f"  {nmem}/{len(p.costed)} operators are memory-bound at the roofline")
    L.append(f"headroom    : {p.cycles_per_token/p.roofline_per_token:.1f}x "
             f"between the projection and the roofline")
    L.append("")

    L.append("--- Amdahl sensitivity (speed up ONE kernel type by k x) -------")
    rows = []
    for r in p.amdahl("kernel"):
        rows.append([r["name"], f"{100*r['fraction']:5.1f}%",
                     f"{r[2]:.2f}x", f"{r[4]:.2f}x", f"{r[8]:.2f}x",
                     f"{r[math.inf]:.2f}x"])
    L.append(_table(["kernel", "share", "k=2", "k=4", "k=8", "k=inf"], rows))
    return "\n".join(L)


# --------------------------------------------------------------------------
# op_inventory.md generation
# --------------------------------------------------------------------------
def emit_doc(path: Path, spec: ModelSpec, hw: HW, *, S: int = 512, N: int = 64,
             measured: MeasuredDB | None = None, fallback_factor: float = 4.0) -> None:
    p_params = spec.param_counts()
    dec = project(spec, "decode", S=S, N=N, hw=hw, measured=measured,
                  fallback_factor=fallback_factor)
    pre = project(spec, "prefill", S=S, N=N, hw=hw, measured=measured,
                  fallback_factor=fallback_factor)

    def md_table(headers, rows):
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(out)

    L: list[str] = []
    L.append(f"# {spec.name} operator inventory and roofline projection")
    L.append("")
    L.append("Generated by `loop/llama_project.py --emit-doc`. Pure analysis: "
             "no simulation was run to produce this file.")
    L.append("")

    # -- assumptions
    L.append("## Assumptions")
    L.append("")
    L.append("**Model config** - "
             + ("verified byte-for-byte against the official "
                "`meta-llama/Llama-3.2-1B` `config.json` "
                "(local copy at `/share2/huggingface/hub/models--meta-llama--Llama-3.2-1B`, "
                "cross-checked against the `unsloth/Llama-3.2-1B` mirror on HF)."
                if spec.verified else "**NOT verified** against config.json."))
    L.append("")
    L.append(md_table(["field", "value"], [
        ["hidden_size", spec.hidden], ["intermediate_size", spec.intermediate],
        ["num_hidden_layers", spec.layers], ["num_attention_heads", spec.heads],
        ["num_key_value_heads", f"{spec.kv_heads} (GQA group = {spec.group_size})"],
        ["head_dim", spec.head_dim], ["vocab_size", spec.vocab],
        ["tie_word_embeddings", spec.tied_embeddings],
        ["rms_norm_eps", spec.rms_norm_eps], ["rope_theta", f"{spec.rope_theta:.0f}"],
    ]))
    L.append("")
    L.append("**Parameters**")
    L.append("")
    L.append(md_table(["bucket", "params"], [
        ["per decoder layer", f"{p_params['per_layer_total']:,}"],
        [f"all {spec.layers} layers", f"{p_params['all_layers']:,}"],
        ["embedding / LM head (tied)", f"{p_params['embedding']:,}"],
        ["final norm", f"{p_params['final_norm']:,}"],
        ["**total**", f"**{p_params['total']:,}** (~1.24B)"],
        ["non-embedding", f"{p_params['non_embedding']:,}"],
    ]))
    L.append("")
    L.append("**Numeric pipeline** - int8 weights, int8 activations, int32/fp32 "
             f"accumulators ({spec.w_bytes}/{spec.act_bytes}/{spec.acc_bytes} bytes), "
             f"int8 KV cache ({spec.kv_bytes} byte). GQA K/V cache rows are read "
             f"once per KV head and reused across the group of {spec.group_size} "
             "query heads.")
    L.append("")
    L.append("**Hardware** (all knobs live on `HW` in `loop/llama_project.py`)")
    L.append("")
    L.append(md_table(["knob", "value", "where it comes from"], [
        ["clock", f"{hw.clock_hz/1e9:g} GHz", "assumed; nothing has been taped out"],
        ["Gemmini peak", f"{hw.gemmini_macs_per_cycle:g} int8 MAC/cycle",
         "16x16 systolic array, one MAC per PE per cycle"],
        ["Saturn int8 peak", f"{hw.saturn_int8_macs_per_cycle:g} int8 MAC/cycle",
         "DLEN=128 bit datapath / 8 bit = 16 lanes; VLEN=256 only sets the "
         "register length, DLEN sets throughput"],
        ["Saturn fp32 elementwise", f"{hw.saturn_fp32_elems_per_cycle:g} elem/cycle",
         "DLEN=128 bit / 32 bit"],
        ["memory bandwidth", f"{hw.mem_bytes_per_cycle:g} B/cycle",
         "**estimate**; dominates the decode roofline, so treat every "
         "memory-bound number below as soft"],
    ]))
    L.append("")
    L.append("Roofline per operator = `max(MAC / peak_MAC_per_cycle, "
             "bytes / bytes_per_cycle)`, i.e. a strict lower bound assuming "
             "perfect utilisation. Real kernels will not reach it - a decode "
             "GEMV has N=1 and cannot fill a 16-wide systolic output "
             "dimension - so read it only as the ceiling on optimisation.")
    L.append("")

    # -- inventory
    for proj_, label, ctx in ((dec, "Decode", f"1 new token, KV length S={S}"),
                              (pre, "Prefill", f"N={N} tokens in one pass")):
        L.append(f"## {label} inventory ({ctx})")
        L.append("")
        t = totals(c.op for c in proj_.costed)
        L.append(f"Per token: **{t['macs']/proj_.tokens/1e6:.2f}M MAC**, "
                 f"{t['bytes_read']/proj_.tokens/1e6:.2f}M bytes read, "
                 f"{t['bytes_written']/proj_.tokens/1e6:.2f}M bytes written, "
                 f"{t['calls']/proj_.tokens:.0f} kernel calls.")
        L.append("")
        rows = []
        tot_c = proj_.total_cycles
        for c in proj_.costed:
            o = c.op
            rows.append([
                o.name, o.kernel, f"`{o.shape}`",
                f"{o.calls/proj_.tokens:g}",
                f"{o.macs/proj_.tokens:,.0f}",
                f"{o.bytes_total/proj_.tokens:,.0f}",
                c.device,
                f"{c.roofline/proj_.tokens:,.0f}",
                c.bound,
                f"{100*c.cycles/tot_c:.2f}%" + ("*" if c.estimated else ""),
            ])
        L.append(md_table(["operator", "kernel", "shape", "calls/tok",
                           "MAC/tok", "bytes/tok", "unit", "roofline cyc/tok",
                           "bound", "share of projected"], rows))
        L.append("")
        L.append("`*` = costed from roofline x "
                 f"{proj_.fallback_factor:g} because no measurement exists yet. "
                 "Layer operators are already multiplied by "
                 f"{spec.layers} layers; attention operators by "
                 f"{spec.heads} heads.")
        L.append("")
        L.append(f"### {label}: share by kernel type")
        L.append("")
        rows = []
        for name, r in sorted(proj_.group("kernel").items(),
                              key=lambda kv: -kv[1]["cycles"]):
            rows.append([name, f"{r['calls']/proj_.tokens:g}",
                         f"{r['macs']/proj_.tokens/1e6:.2f}M",
                         f"{100*r['macs']/max(t['macs'],1):.1f}%",
                         _fmt(r['roofline'] / proj_.tokens),
                         f"{100*r['cycles']/tot_c:.1f}%"])
        L.append(md_table(["kernel", "calls/tok", "MAC/tok", "MAC share",
                           "roofline cyc/tok", "projected cycle share"], rows))
        L.append("")
        rf_c = sum(c.rf_compute for c in proj_.costed) / proj_.tokens
        rf_m = sum(c.rf_memory for c in proj_.costed) / proj_.tokens
        L.append(f"**Theoretical best at {hw.clock_hz/1e9:g} GHz**: "
                 f"{proj_.roofline_per_token:,.0f} cycles/token -> "
                 f"**{proj_.roofline_tokens_per_s:.2f} tokens/s**. "
                 f"Split: compute-only {rf_c:,.0f} cyc/tok "
                 f"({hw.clock_hz/rf_c:.2f} tok/s), memory-only {rf_m:,.0f} "
                 f"cyc/tok ({hw.clock_hz/rf_m:.2f} tok/s).")
        L.append("")

    # -- lm head call-out
    lm = next(c for c in dec.costed if c.op.name == "lm_head")
    dec_t = totals(c.op for c in dec.costed)
    L.append("## The LM head dominates decode")
    L.append("")
    L.append(f"`lm_head` is a single {spec.vocab}x{spec.hidden} GEMV per generated "
             f"token: **{lm.op.macs/1e6:.1f}M MAC "
             f"({100*lm.op.macs/dec_t['macs']:.1f}% of all decode MACs)** and "
             f"**{lm.op.bytes_total/1e6:.1f}MB of weight traffic "
             f"({100*lm.op.bytes_total/dec_t['bytes_read']:.1f}% of decode reads)**, "
             f"or {100*lm.roofline/dec.total_roofline:.1f}% of the roofline "
             f"cycle budget. It is the single largest operator in the model "
             f"and it is memory-bound: {lm.rf_memory:,.0f} cycles of weight "
             f"streaming against {lm.rf_compute:,.0f} cycles of MACs. "
             "Because the embedding is tied, that same matrix is also the "
             "embedding table, so it cannot be pruned - but it is the obvious "
             "target for vocabulary pruning, a sparse/top-k head, or keeping "
             "the head resident in scratchpad across tokens.")
    L.append("")
    L.append("## Amdahl sensitivity (decode)")
    L.append("")
    L.append("If exactly one kernel type gets k x faster and nothing else "
             "changes, the end-to-end decode speedup is:")
    L.append("")
    rows = [[r["name"], f"{100*r['fraction']:.1f}%", f"{r[2]:.2f}x",
             f"{r[4]:.2f}x", f"{r[8]:.2f}x", f"{r[math.inf]:.2f}x"]
            for r in dec.amdahl("kernel")]
    L.append(md_table(["kernel", "cycle share", "k=2", "k=4", "k=8", "k=inf"], rows))
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- Nothing here was measured. Fill in "
             "`out/llama-profile/measured_cycles.json` from real Verilator "
             "kernel runs and re-run with `--measured` to replace the "
             "roofline-x-factor estimates.")
    L.append("- Extrapolation is linear in MACs (GEMV/GEMM/attention) or in "
             "elements (norm/RoPE/softmax/SiLU/add), plus an optional fixed "
             "per-call overhead. That is optimistic for GEMV, where a real "
             "kernel is dominated by weight streaming rather than by MACs.")
    L.append("- The memory bandwidth figure is a guess and it is the binding "
             "constraint for decode. If it is wrong, every memory-bound row "
             "moves with it.")
    L.append("- Decode assumes `S` counts the KV entries the new token "
             "attends to (cache length including the current token). Prefill "
             "assumes causal attention and logits for one position only.")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def self_test() -> None:
    llama_model.self_test()
    print("llama_project self-test")
    spec = MODELS[DEFAULT_MODEL]
    hw = HW()

    def check(label, ok, extra=""):
        print(f"  [{'ok ' if ok else 'FAIL'}] {label} {extra}")
        assert ok, label

    dec = project(spec, "decode", S=512, hw=hw)
    # Every op accounted for, shares sum to 1.
    share = sum(r["cycles"] for r in dec.group("kernel").values()) / dec.total_cycles
    check("kernel shares sum to 1", abs(share - 1) < 1e-9, f"({share:.12f})")
    check("device shares sum to 1",
          abs(sum(r["cycles"] for r in dec.group("device").values())
              / dec.total_cycles - 1) < 1e-9)

    # Projection with fallback f must be exactly f x the roofline when nothing
    # is measured.
    check("no measurements -> projection == roofline x fallback",
          abs(dec.total_cycles - dec.fallback_factor * dec.total_roofline) < 1e-6)

    # Roofline is a lower bound per op.
    check("roofline == max(compute, memory) per op",
          all(abs(c.roofline - max(c.rf_compute, c.rf_memory)) < 1e-9
              for c in dec.costed))

    # Amdahl sanity: k=1 is a no-op, k=inf equals 1/(1-f).
    row = dec.amdahl("kernel")[0]
    f = row["fraction"]
    check("Amdahl k=inf == 1/(1-f)", abs(row[math.inf] - 1 / (1 - f)) < 1e-9,
          f"(top kernel {row['name']}, f={f:.3f})")
    check("Amdahl monotone in k", row[2] <= row[4] <= row[8] <= row[math.inf])

    # tokens/s consistency
    check("tokens/s == clock / cycles_per_token",
          abs(dec.tokens_per_s - hw.clock_hz / dec.cycles_per_token) < 1e-6)

    # Attention must scale linearly in S.
    a512 = sum(c.op.macs for c in dec.costed if c.op.kernel in ("attn_scores", "attn_pv"))
    d1024 = project(spec, "decode", S=1024, hw=hw)
    a1024 = sum(c.op.macs for c in d1024.costed
                if c.op.kernel in ("attn_scores", "attn_pv"))
    check("attention MACs linear in S", a1024 == 2 * a512)
    # ...and GEMV MACs must not move with S.
    g512 = sum(c.op.macs for c in dec.costed if c.op.kernel.endswith("gemv"))
    g1024 = sum(c.op.macs for c in d1024.costed if c.op.kernel.endswith("gemv"))
    check("GEMV MACs independent of S", g512 == g1024)

    # Prefill amortises the LM head: MAC/token must be lower than decode.
    pre = project(spec, "prefill", N=64, hw=hw)
    mac_dec = totals(c.op for c in dec.costed)["macs"] / dec.tokens
    mac_pre = totals(c.op for c in pre.costed)["macs"] / pre.tokens
    check("prefill MAC/token < decode MAC/token", mac_pre < mac_dec,
          f"({mac_pre/1e6:.1f}M vs {mac_dec/1e6:.1f}M)")

    # Measured entries must override the estimate and scale linearly.
    m = MeasuredDB(by_kernel={"int8_gemv": Measurement(
        "int8_gemv", "macs", 4_194_304, 100_000.0, overhead_cycles=0.0)})
    pm = project(spec, "decode", S=512, hw=hw, measured=m)
    gemv = [c for c in pm.costed if c.op.kernel == "int8_gemv"]
    check("measured kernels not flagged estimated", not any(c.estimated for c in gemv))
    expect = sum(c.op.macs for c in gemv) / 4_194_304 * 100_000.0
    check("linear MAC extrapolation", abs(sum(c.cycles for c in gemv) - expect) < 1e-3)

    # Per-call overhead is added once per call.
    m2 = MeasuredDB(by_kernel={"int8_gemv": Measurement(
        "int8_gemv", "macs", 4_194_304, 100_000.0, overhead_cycles=1_000.0)})
    pm2 = project(spec, "decode", S=512, hw=hw, measured=m2)
    g2 = [c for c in pm2.costed if c.op.kernel == "int8_gemv"]
    calls = sum(c.op.calls for c in g2)
    expect2 = calls * 1000.0 + 99_000.0 * sum(c.op.macs for c in g2) / 4_194_304
    check("per-call overhead applied", abs(sum(c.cycles for c in g2) - expect2) < 1e-3,
          f"({calls} calls)")

    # The shipped placeholder file must parse and contain no usable numbers yet.
    ph = Path(__file__).resolve().parents[1] / "out" / "llama-profile" / "measured_cycles.json"
    if ph.exists():
        db = MeasuredDB.load(ph)
        check("placeholder measured_cycles.json parses", True,
              f"({len(db.by_kernel)} measured kernels)")

    # ---- overlap model (opt-in; default must be bit-for-bit unchanged) ----
    base = project(spec, "decode", S=512, hw=hw)
    check("overlap off by default", base.overlap_factor is None
          and base.total_hidden == 0.0
          and all(c.cycles_serial is None for c in base.costed))
    check("overlap off -> total == sequential total",
          base.total_cycles == base.total_cycles_sequential)
    ov1 = project(spec, "decode", S=512, hw=hw, overlap_factor=1.0)
    check("overlap factor 1.0 is a no-op",
          abs(ov1.total_cycles - base.total_cycles) < 1e-6
          and ov1.total_hidden == 0.0)
    ov = project(spec, "decode", S=512, hw=hw,
                 overlap_factor=DEFAULT_OVERLAP_FACTOR)
    sat_base = sum(c.cycles for c in base.costed if c.device != "gemmini")
    gem_base = sum(c.cycles for c in base.costed if c.device == "gemmini")
    check("overlap hides (1-f) of the saturn cost",
          abs(ov.total_hidden - sat_base * (1 - DEFAULT_OVERLAP_FACTOR)) < 1e-6,
          f"(saturn {_fmt(sat_base)}, hidden {_fmt(ov.total_hidden)})")
    check("overlap leaves gemmini untouched",
          abs(sum(c.cycles for c in ov.costed if c.device == "gemmini")
              - gem_base) < 1e-6)
    check("overlap sequential total is recoverable",
          abs(ov.total_cycles_sequential - base.total_cycles) < 1e-6)
    check("overlap is strictly faster than sequential",
          ov.total_cycles < base.total_cycles,
          f"({_fmt(base.cycles_per_token)} -> {_fmt(ov.cycles_per_token)} cyc/tok)")
    check("overlap never hides more than the gemmini cycles",
          ov.total_hidden <= gem_base + 1e-6)
    # The cap must bind when there is nothing to hide under.
    capped = [CostedOp(op=c.op, device="saturn", cycles=c.cycles,
                       roofline=c.roofline, rf_compute=c.rf_compute,
                       rf_memory=c.rf_memory, estimated=c.estimated,
                       bound=c.bound) for c in base.costed]
    _apply_overlap(capped, 0.0)
    check("no gemmini work -> nothing hidden",
          sum(c.hidden_cycles for c in capped) == 0.0)
    try:
        _apply_overlap([], 1.5)
        bad = True
    except ValueError:
        bad = False
    check("overlap factor outside [0,1] rejected", not bad)
    check("render with overlap mentions the basis",
          "OVERLAP MODEL ON" in render(ov, show_ops=False)
          and "OVERLAP MODEL ON" not in render(base, show_ops=False))

    print(" all llama_project self-tests passed")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Project Llama-3.2-1B end-to-end speed on the AETHER SoC.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--scenario", default="decode", choices=("decode", "prefill"))
    ap.add_argument("--S", type=int, default=512,
                    help="decode: KV cache length the new token attends to")
    ap.add_argument("--N", type=int, default=64, help="prefill: tokens per pass")
    ap.add_argument("--clock-ghz", type=float, default=1.0)
    ap.add_argument("--measured", default=None,
                    help="path to measured_cycles.json")
    ap.add_argument("--fallback-factor", type=float, default=4.0,
                    help="unmeasured op cost = roofline x this (default 4)")
    ap.add_argument("--mem-bytes-per-cycle", type=float, default=8.0)
    ap.add_argument("--gemv-bytes-per-cycle", type=float, default=None,
                    help="cost int8_gemv (decode GEMV) directly as macs/rate "
                         "B/cycle instead of the measured-tile MAC "
                         "extrapolation -- use this to apply a COLD-DRAM "
                         "measured rate (e.g. 7.00 upper bound / 6.43 lower "
                         "bound, see the 2026-09-12 audit)")
    ap.add_argument("--lmhead-bytes-per-cycle", type=float, default=None,
                    help="same override for lm_head_gemv (default: reuses "
                         "--gemv-bytes-per-cycle if that is set; the measured "
                         "cold lm_head tile rate is 6.61 B/cycle)")
    ap.add_argument("--cold", choices=("upper", "lower"), default=None,
                    help="shorthand for cold-DRAM GEMV rates from the "
                         "2026-09-12 audit (out/loop/20260909-200541-dae1/"
                         "agent_07.txt): 'upper' = 7.00 B/cycle GEMV (the n1 "
                         "tile's own cold rate), 'lower' = 6.43 B/cycle GEMV "
                         "(agent_07's fitted cold-stream floor); both set "
                         "lm_head to 6.61 B/cycle unless "
                         "--gemv-bytes-per-cycle/--lmhead-bytes-per-cycle are "
                         "given explicitly, which take precedence")
    ap.add_argument("--gemmini-macs", type=float, default=256.0)
    ap.add_argument("--saturn-macs", type=float, default=16.0)
    ap.add_argument("--saturn-elems", type=float, default=4.0)
    ap.add_argument("--device", action="append", default=[], metavar="KERNEL=UNIT",
                    help="override the unit for a kernel type, repeatable")
    ap.add_argument("--overlap", nargs="?", type=float,
                    const=DEFAULT_OVERLAP_FACTOR, default=None,
                    metavar="FACTOR",
                    help="model Gemmini/Saturn overlap: saturn operators keep "
                         "only FACTOR of their cycles (default "
                         f"{DEFAULT_OVERLAP_FACTOR:.3f}, the measured exposure "
                         "ratio of llama-layer-fused-n1). OFF by default; see "
                         "OVERLAP_NOTE_MD for basis and limits")
    ap.add_argument("--no-ops", action="store_true", help="skip the per-op table")
    ap.add_argument("--emit-doc", default=None,
                    help="write the markdown inventory to this path and exit")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        self_test()
        return 0

    spec = MODELS[a.model]
    hw = HW(clock_hz=a.clock_ghz * 1e9,
            gemmini_macs_per_cycle=a.gemmini_macs,
            saturn_int8_macs_per_cycle=a.saturn_macs,
            saturn_fp32_elems_per_cycle=a.saturn_elems,
            mem_bytes_per_cycle=a.mem_bytes_per_cycle)
    measured = MeasuredDB.load(a.measured)
    dmap = {}
    for d in a.device:
        k, _, v = d.partition("=")
        dmap[k] = v

    gemv_bpc = a.gemv_bytes_per_cycle
    lmhead_bpc = a.lmhead_bytes_per_cycle
    if a.cold:
        if gemv_bpc is None:
            gemv_bpc = 7.00 if a.cold == "upper" else 6.43
        if lmhead_bpc is None:
            lmhead_bpc = 6.61

    if a.emit_doc:
        emit_doc(Path(a.emit_doc), spec, hw, S=a.S, N=a.N, measured=measured,
                 fallback_factor=a.fallback_factor)
        print(f"wrote {a.emit_doc}")
        return 0

    p = project(spec, a.scenario, S=a.S, N=a.N, hw=hw, measured=measured,
                fallback_factor=a.fallback_factor, device_map=dmap,
                gemv_bytes_per_cycle=gemv_bpc, lmhead_bytes_per_cycle=lmhead_bpc,
                overlap_factor=a.overlap)
    print(render(p, show_ops=not a.no_ops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
