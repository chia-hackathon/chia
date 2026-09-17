"""Batched-decode (N>1) projection for Llama-3.2-1B on the AETHER SoC.

`loop/llama_project.py` has no batch-decode scenario: `--scenario decode`
projects exactly one token (N=1) and `--N` is the *prefill* tokens-per-pass
knob. This module adds the missing batch dimension on top of it, reusing
`llama_project.project()` so that the B=1 column reproduces
`projection_round5.md`'s 159.73M cycles/token / 6.260 tok/s exactly.

Cost model for a decode step with batch size B (B sequences each advancing
one token):

  * weight-streaming GEMVs (`int8_gemv`, `lm_head_gemv`)
        The weight matrix is read from DRAM once and reused by all B
        activation columns, so the cost does NOT scale with B; it scales with
        the measured *tile* cycle count at that N. We use the loop's own
        measured tiles (M=512, K=2048):
            N=1   132,424 cycles   (llama-q8-gemv-gemmini-n1, round2b..6)
            N=16  142,458 cycles   (llama-q8-gemv-gemmini-n16, round5/6)
        and a LINEAR extrapolation in N for any other batch size (N=32 is
        therefore an ESTIMATE, flagged as such by `TileModel.estimated`).
        Every GEMV op's B=1 cost from `llama_project` is simply multiplied by
        tile(B)/tile(1) -- both are linear in MACs off the same 1,048,576-MAC
        tile, so this is exactly "re-cost the op with the N=B kernel".

  * everything else (attention scores/PV, softmax, norms, RoPE, SiLU*mul,
    residual adds, embedding lookup)
        Every sequence has its own KV cache and its own activations, so these
        cost B times the B=1 figure. (No batched measurement exists for these
        kernels; x B is the conservative, no-batching-benefit assumption.)

  * roofline
        Recomputed per op under the same sharing rule: a GEMV moves its
        weights once (op.macs bytes at int8) plus B copies of the activation
        in / accumulator out; every other op is B x its B=1 roofline. Memory
        bandwidth is the corrected Gemmini mbus figure, 8 B/cycle.

Usage
-----
    python -m loop.llama_batch_project --S 512 \
        --measured out/llama-profile/measured_cycles_round5.json \
        --batches 1,16,32
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from loop.llama_project import (HW, MeasuredDB, project, roofline_cycles,
                                    DEVICE_MAP)
    from loop.llama_model import MODELS, DEFAULT_MODEL
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llama_project import HW, MeasuredDB, project, roofline_cycles, DEVICE_MAP
    from llama_model import MODELS, DEFAULT_MODEL


# Measured GEMV tile cycle counts, M=512 K=2048, int8, Gemmini.
# Sources: out/loop/ledger.md (per-kernel history table) and
# out/llama-profile/measured_cycles_round5.json.
TILE_N1 = 132_424.0      # llama-q8-gemv-gemmini-n1,  best since round2b
TILE_N16 = 142_458.0     # llama-q8-gemv-gemmini-n16, best from round5 (round6 found none better)
TILE_MACS = 1_048_576.0  # 512 x 2048
TILE_ROOFLINE = 131_072.0  # 1 MiB of int8 weights / 8 B per cycle

WEIGHT_STREAMING_KERNELS = ("int8_gemv", "lm_head_gemv")

LMHEAD_MACS = 128_256 * 2_048.0  # full lm_head weight bytes (1 B/MAC, int8)


@dataclass
class TileModel:
    """cycles(N) for the M=512/K=2048 GEMV tile, from two measured points."""
    n1: float = TILE_N1
    n16: float = TILE_N16

    @property
    def slope(self) -> float:
        return (self.n16 - self.n1) / 15.0

    def cycles(self, n: int) -> float:
        if n == 1:
            return self.n1
        if n == 16:
            return self.n16
        return self.n1 + self.slope * (n - 1)

    def estimated(self, n: int) -> bool:
        return n not in (1, 16)

    def ratio(self, n: int) -> float:
        return self.cycles(n) / self.n1


def make_tile_model(gemv_bytes_per_cycle: float | None) -> TileModel:
    """Default TileModel(), or one re-based on a COLD-DRAM measured B/cycle
    rate applied directly to the 1 MiB (TILE_MACS-byte) N=1 tile: n1 =
    TILE_MACS / rate. The N=1 -> N=16 ratio measured at the (warm) default
    rate is kept, since no separate cold N=16 measurement exists -- this is
    an approximation, flagged the same way the existing N=32+ linear
    extrapolation already is (`TileModel.estimated`)."""
    if gemv_bytes_per_cycle is None:
        return TileModel()
    n1 = TILE_MACS / gemv_bytes_per_cycle
    n16 = n1 * (TILE_N16 / TILE_N1)
    return TileModel(n1=n1, n16=n16)


def lmhead_tile_cycles(lmhead_bytes_per_cycle: float | None) -> float:
    """Default LMHEAD_TILE_CYCLES (measured 4 MiB tile scaled up), or a
    direct macs/rate costing of the full lm_head GEMV at a COLD-DRAM
    measured B/cycle rate (e.g. 6.61, see out/loop/20260909-200541-dae1/
    agent_07.txt)."""
    if lmhead_bytes_per_cycle is None:
        return LMHEAD_TILE_CYCLES
    return LMHEAD_MACS / lmhead_bytes_per_cycle


@dataclass
class BatchPoint:
    batch: int
    step_cycles: float          # cycles for one decode step of the whole batch
    gemv_cycles: float
    other_cycles: float
    step_roofline: float
    estimated: bool
    tile_cycles: float

    @property
    def cycles_per_token(self) -> float:
        return self.step_cycles / self.batch

    def tok_s_aggregate(self, clock_hz: float) -> float:
        return clock_hz * self.batch / self.step_cycles

    def tok_s_per_seq(self, clock_hz: float) -> float:
        return clock_hz / self.step_cycles

    def roofline_tok_s_aggregate(self, clock_hz: float) -> float:
        return clock_hz * self.batch / self.step_roofline


# The dedicated lm_head tile kernel (M=2048, K=2048, llama-q8-gemv-gemmini-lmhead,
# round4 best 634,507 cycles for 4 MiB of weights = 6.61 B/cycle) projected onto the
# full 128,256 x 2,048 lm_head: 634,507 * (128256/2048).
LMHEAD_TILE_CYCLES = 634_507.0 * (128_256 / 2_048)   # 39,736,001


def batch_point(costed, hw: HW, batch: int, tiles: TileModel,
                lmhead_tile: bool = False,
                lmhead_cycles: float = LMHEAD_TILE_CYCLES) -> BatchPoint:
    ratio = tiles.ratio(batch)
    gemv = other = rf = 0.0
    for c in costed:
        op = c.op
        if op.kernel in WEIGHT_STREAMING_KERNELS:
            base = (lmhead_cycles if (lmhead_tile and op.kernel == "lm_head_gemv")
                    else c.cycles)
            gemv += base * ratio
            k = op.macs / op.elems if op.elems else 0.0
            weight_bytes = op.macs                      # int8 weights, read once
            per_seq_bytes = op.calls * k + op.bytes_written
            mem = (weight_bytes + batch * per_seq_bytes) / hw.mem_bytes_per_cycle
            comp = batch * op.macs / hw.peak_macs(c.device)
            rf += max(mem, comp)
        else:
            other += c.cycles * batch
            rf += c.roofline * batch
    return BatchPoint(batch=batch, step_cycles=gemv + other, gemv_cycles=gemv,
                      other_cycles=other, step_roofline=rf,
                      estimated=tiles.estimated(batch),
                      tile_cycles=tiles.cycles(batch))


def run(S: int, measured_path: str, batches: list[int], model: str = DEFAULT_MODEL,
        mem_bytes_per_cycle: float = 8.0, clock_ghz: float = 1.0,
        lmhead_tile: bool = False,
        gemv_bytes_per_cycle: float | None = None,
        lmhead_bytes_per_cycle: float | None = None,
        cold: str | None = None):
    """`gemv_bytes_per_cycle` / `lmhead_bytes_per_cycle` (or the `cold`
    shorthand, 'upper'=7.00/'lower'=6.43 B/cycle GEMV + 6.61 B/cycle lm_head)
    cost the N=1 (B=1) `int8_gemv` / `lm_head_gemv` ops directly as
    `op.macs / rate` -- bypassing the measured-DB tile-based MAC
    extrapolation, which for lm_head implicitly assumes an unreachable
    ~9.2 B/cycle (see the 2026-09-12 audit in out/loop/FINAL_REPORT.md) --
    and are then grown to B>1 by the SAME warm-tile N1->N16 ratio used by
    default (no separate cold N=16/lm_head measurement exists, so this ratio
    is kept as the least-bad available assumption). Both default to None,
    reproducing the original tile-measurement-based projection exactly."""
    if lmhead_bytes_per_cycle is None:
        lmhead_bytes_per_cycle = gemv_bytes_per_cycle
    if cold:
        if gemv_bytes_per_cycle is None:
            gemv_bytes_per_cycle = 7.00 if cold == "upper" else 6.43
        if lmhead_bytes_per_cycle is None:
            lmhead_bytes_per_cycle = 6.61

    spec = MODELS[model]
    hw = HW(clock_hz=clock_ghz * 1e9, mem_bytes_per_cycle=mem_bytes_per_cycle)
    db = MeasuredDB.load(measured_path)
    dev = {"int8_gemv": "gemmini", "lm_head_gemv": "gemmini"}
    p = project(spec, "decode", S=S, hw=hw, measured=db, device_map=dev,
                gemv_bytes_per_cycle=gemv_bytes_per_cycle,
                lmhead_bytes_per_cycle=lmhead_bytes_per_cycle)
    tiles = make_tile_model(gemv_bytes_per_cycle)
    lmc = lmhead_tile_cycles(lmhead_bytes_per_cycle)
    return [batch_point(p.costed, hw, b, tiles, lmhead_tile, lmc) for b in batches], p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--measured", default="out/llama-profile/measured_cycles_round5.json")
    ap.add_argument("--batches", default="1,16,32")
    ap.add_argument("--mem-bytes-per-cycle", type=float, default=8.0)
    ap.add_argument("--clock-ghz", type=float, default=1.0)
    ap.add_argument("--lmhead-tile", action="store_true",
                    help="cost lm_head from its OWN measured M=2048 tile (634,507 cyc, "
                         "6.61 B/cycle) instead of the n1-basis extrapolation")
    ap.add_argument("--gemv-bytes-per-cycle", type=float, default=None,
                    help="re-base the N=1/N=16 GEMV tile model on a COLD-DRAM "
                         "measured rate (macs/rate) instead of the measured "
                         "tiles (which include a warm-L2 tail)")
    ap.add_argument("--lmhead-bytes-per-cycle", type=float, default=None,
                    help="same override for the lm_head tile (implies "
                         "--lmhead-tile); default reuses --gemv-bytes-per-cycle")
    ap.add_argument("--cold", choices=("upper", "lower"), default=None,
                    help="shorthand: 'upper'=7.00, 'lower'=6.43 B/cycle GEMV, "
                         "both with 6.61 B/cycle lm_head (2026-09-12 audit, "
                         "out/loop/20260909-200541-dae1/agent_07.txt)")
    a = ap.parse_args(argv)

    lmhead_tile = a.lmhead_tile or a.lmhead_bytes_per_cycle is not None or a.cold is not None
    batches = [int(x) for x in a.batches.split(",")]
    pts, p = run(a.S, a.measured, batches, mem_bytes_per_cycle=a.mem_bytes_per_cycle,
                 clock_ghz=a.clock_ghz, lmhead_tile=lmhead_tile,
                 gemv_bytes_per_cycle=a.gemv_bytes_per_cycle,
                 lmhead_bytes_per_cycle=a.lmhead_bytes_per_cycle, cold=a.cold)
    clk = a.clock_ghz * 1e9
    gemv_bpc = a.gemv_bytes_per_cycle
    lmhead_bpc = a.lmhead_bytes_per_cycle
    if a.cold:
        if gemv_bpc is None:
            gemv_bpc = 7.00 if a.cold == "upper" else 6.43
        if lmhead_bpc is None:
            lmhead_bpc = 6.61
    tiles = make_tile_model(gemv_bpc)

    print(f"Llama-3.2-1B batched decode, S={a.S}, {a.clock_ghz:.2f} GHz, "
          f"{a.mem_bytes_per_cycle:g} B/cycle mbus")
    print(f"measured: {a.measured}")
    print(f"GEMV tile model: N=1 {tiles.n1:,.0f}, N=16 {tiles.n16:,.0f}, "
          f"slope {tiles.slope:,.1f} cyc/extra column (linear)")
    print()
    hdr = (f"{'B':>4} {'tile cyc':>10} {'step cyc':>14} {'cyc/tok':>12} "
           f"{'tok/s agg':>11} {'tok/s seq':>10} {'roofline agg':>13} {'gemv%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for pt in pts:
        flag = "*" if pt.estimated else " "
        print(f"{pt.batch:>4}{flag}{pt.tile_cycles:>10,.0f} {pt.step_cycles:>14,.0f} "
              f"{pt.cycles_per_token:>12,.0f} {pt.tok_s_aggregate(clk):>11.3f} "
              f"{pt.tok_s_per_seq(clk):>10.3f} "
              f"{pt.roofline_tok_s_aggregate(clk):>13.3f} "
              f"{100*pt.gemv_cycles/pt.step_cycles:>6.1f}%")
    print("\n* = tile cycles linearly extrapolated from the N=1/N=16 measurements "
          "(estimate, not measured)")
    base = pts[0]
    print(f"\nB=1 cross-check vs llama_project.py decode: "
          f"{p.cycles_per_token:,.0f} cyc/tok projected, batch model says "
          f"{base.cycles_per_token:,.0f}")
    for pt in pts[1:]:
        print(f"B={pt.batch}: {pt.tok_s_aggregate(clk)/base.tok_s_aggregate(clk):.2f}x "
              f"aggregate throughput, {pt.tok_s_per_seq(clk)/base.tok_s_per_seq(clk):.3f}x "
              f"per-sequence rate vs B=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
