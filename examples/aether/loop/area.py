"""Proxy area model for the AETHER hardware-search outer loop.

WHY A PROXY
------------
The hackathon proposal asks the outer hardware-search loop to Pareto-optimize
Saturn (RVV 1.0 vector unit: VLEN, DLEN) and Gemmini (systolic array: DIM,
scratchpad/accumulator capacity, dataflow) "under area/frequency constraints."
No synthesis flow is available in this environment (no confirmed Genus
license), so `estimate_area()` below is a closed-form PROXY: simple scaling
laws with hand-picked coefficients, not a synthesis result. `read_synth_area()`
is a stub for the day a real Genus/Yosys area report exists, returning the
same `AreaEstimate` type so the outer loop never has to change call sites.

WHAT WAS SEARCHED FOR GROUNDING (and what was/was not found)
--------------------------------------------------------------
Before picking coefficients, `docs/paper.txt` and the `repos/saturn`,
`repos/gemmini`, `repos/shuttle` checkouts were grepped for area/mm2/um2 and
process-node (TSMC/Intel 16/22nm/16nm/12nm/28nm) figures. Result:

  * `docs/paper.txt` is the CHIA framework paper (LLM-driven RTL loop on a
    MegaBOOM core). It never mentions Saturn/Gemmini/VLEN/DLEN/systolic at
    all. Its only area numbers (`docs/paper.txt:1159-1206`, ~24.2-25.2 mm^2
    on the Skywater 130nm PDK) are for an unrelated MegaBOOM case study and
    are NOT used below.
  * `repos/saturn` has no mm^2/um^2 figures and no process node anywhere.
    It DOES explicitly say the vector unit is a unified SIMD datapath, not a
    per-lane array (`repos/saturn/docs/system.adoc:28`), which is why the
    model below scales a single "datapath" term with DLEN rather than a
    "per-lane count" term. VLEN/DLEN/MLEN are defined at
    `repos/saturn/docs/system.adoc:52-55`.
  * `repos/gemmini` has no mm^2/um^2/process-node figures either, but DOES
    confirm the default architecture: 16x16 mesh, sp_capacity=256KB,
    acc_capacity=64KB, dataflow=BOTH
    (`repos/gemmini/src/main/scala/gemmini/GemminiConfigs.scala:34-50`,
    `repos/gemmini/src/main/scala/gemmini/Configs.scala:34-42`), and that
    scratchpad/accumulator row widths scale with DIM
    (`repos/gemmini/README.md:212`).
  * `repos/shuttle/README.md:15` only says Shuttle has "similar physical
    design complexity as Rocket" -- qualitative, no number.

None of the four numeric area coefficients below could therefore be sourced
from this repo. They are external, order-of-magnitude, industry-standard /
literature-anchored GUESSES, clearly flagged as such. Only the architectural
scaling exponents/structure (bits ~ 32*VLEN, PEs ~ DIM^2, SRAM ~ KB) and the
Gemmini/Saturn default parameter values are repo-grounded.

COEFFICIENT TABLE (all normalized to one generic "16nm-class" node --
TSMC 16FFC / Intel 16 -- chosen because it is the node the published Gemmini
accelerator paper (Genc et al., "Gemmini: Enabling Systematic Deep-Learning
Architecture Evaluation via Full-Stack Integration", DAC 2021) targets; no
in-repo source states a node, so this is a documentation choice, not a
finding)
--------------------------------------------------------------------------
| Symbol                       | Value            | Status    | Source |
|------------------------------|------------------|-----------|--------|
| SRAM_BIT_AREA_MM2             | 7.4e-8 mm^2/bit  | GUESS     | Industry-standard HD 6T bitcell figure commonly quoted for 16nm-class finFET nodes (~0.074 um^2/bit, e.g. TSMC 16FFC ISSCC bitcell disclosures). Not verified against a primary datasheet in this session; correct to within ~2x. |
| SRAM_PERIPHERY_OVERHEAD       | 1.8x             | GUESS     | Standard CACTI-style rule of thumb: SRAM macro area (bitcell array + decoders/sense-amps/mux/periphery) is typically 1.5-2x the raw bitcell array area for small-to-medium macros. |
| ACC_PERIPHERY_OVERHEAD        | 2.2x             | GUESS     | Gemmini's accumulator banks carry per-element adders (they accumulate partial sums), so periphery overhead is modeled higher than plain scratchpad SRAM. Magnitude is a guess. |
| GEMMINI_PE_AREA_MM2           | 1.8e-3 mm^2/PE   | GUESS     | Order-of-magnitude back-of-envelope from the published Gemmini paper's reported full-accelerator area for a 16x16 (256 PE) Intel-16 instance being on the order of a few tenths of an mm^2 for the array; NOT independently re-derived or verified in this session (repo contains no such figure -- confirmed absent from README/src). Treat as low-confidence. |
| DLEN_DATAPATH_AREA_PER_BIT_MM2| 4.0e-3 mm^2/bit  | GUESS     | Loosely anchored to published RVV vector-datapath area scaling (e.g. Cavalcante et al., "Ara: A 1 GHz+ Scalable and Energy-Efficient RISC-V Vector Processor", TVLSI 2020, GF22nm lane-area order of magnitude), scaled down because Saturn explicitly uses one unified SIMD datapath rather than N replicated lanes (system.adoc:28). Low confidence, order-of-magnitude only. |
| SHUTTLE_CORE_FIXED_MM2        | 0.45 mm^2        | GUESS     | No number in repos/shuttle; Rocket-class in-order scalar core order-of-magnitude at a 16nm-class node, informed only by the qualitative claim that Shuttle has "similar physical design complexity as Rocket" (repos/shuttle/README.md:15). |
| DATAFLOW_BOTH_OVERHEAD        | 1.08x            | GUESS     | Supporting both WS and OS dataflow (`Dataflow.BOTH`, the Gemmini default -- Configs.scala:38) needs extra muxing/control versus a single fixed dataflow; magnitude is a guess. |
| OTHER_GLUE_FRACTION           | 5% of subtotal   | GUESS     | Clock/reset/interconnect/TileLink glue logic not modeled per-block. |
| NUM_VRF_ARCH_REGS = 32        | 32               | GROUNDED  | RISC-V "V" (RVV) extension architectural constant: 32 vector registers, independent of this project. |
| Gemmini defaults: DIM=16,     |                  | GROUNDED  | repos/gemmini/src/main/scala/gemmini/Configs.scala:34-42 (meshRows=meshColumns=16, sp_capacity=256KB, acc_capacity=64KB, dataflow=BOTH). |
| sp=256KB, acc=64KB, WS+OS     |                  |           | |
| Saturn VLEN/DLEN are a config | (128..1024/      | GROUNDED  | repos/saturn/chipyard/SaturnConfigs.scala (e.g. line 92: GENV256D128ShuttleConfig -> VLEN=256, DLEN=128); confirms VLEN/DLEN naming and swept range, not a scaling law. |
| sweep, not one fixed default  | 64..256)         |           | |

Everything else in `estimate_area()` (the arithmetic combining these
coefficients with vlen/dlen/dim/capacities) is architecture-grounded scaling
*structure* (bits ~ 32*VLEN for VRF, PEs ~ DIM^2, SRAM area ~ capacity_kb)
applied to GUESS-level coefficients. Swap the six GUESS constants above for
real numbers (from a datasheet, a published paper you can pin a page/table
to, or `read_synth_area()` once synthesis is available) without touching the
rest of the module.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AreaEstimate:
    """Result of an area estimate (proxy or real synthesis), in mm^2."""

    total_mm2: float
    breakdown: dict[str, float]
    notes: list[str] = field(default_factory=list)


# --- Coefficients (see docstring table above for citations/status) ---------
SRAM_BIT_AREA_MM2 = 7.4e-8           # mm^2 per SRAM bit, ~0.074 um^2/bit
SRAM_PERIPHERY_OVERHEAD = 1.8        # macro-area / raw-bitcell-array-area
ACC_PERIPHERY_OVERHEAD = 2.2         # accumulator banks carry adders
GEMMINI_PE_AREA_MM2 = 1.8e-3         # mm^2 per systolic PE
DLEN_DATAPATH_AREA_PER_BIT_MM2 = 4.0e-3  # mm^2 per bit of SIMD datapath width
SHUTTLE_CORE_FIXED_MM2 = 0.45        # mm^2, fixed scalar-core cost
DATAFLOW_BOTH_OVERHEAD = 1.08        # extra mux/control for WS+OS support
OTHER_GLUE_FRACTION = 0.05           # fraction of subtotal for glue logic

NUM_VRF_ARCH_REGS = 32               # RVV: architectural vector register count

_ASSUMPTIONS_NOTE = (
    "Proxy area model (no synthesis flow available). SRAM bit-area, PE "
    "area, per-bit datapath area, and the fixed core area are external "
    "order-of-magnitude GUESSES normalized to a generic 16nm-class node -- "
    "not sourced from this repo (confirmed absent from docs/paper.txt and "
    "repos/{saturn,gemmini,shuttle}). Only the scaling structure (bits ~ "
    "32*VLEN, PEs ~ DIM^2, SRAM area ~ capacity_kb) and Gemmini/Saturn "
    "default parameters are repo-grounded. See module docstring for the "
    "full coefficient table and citations."
)


def estimate_area(params: dict[str, Any]) -> AreaEstimate:
    """Proxy area estimate (mm^2) for a Saturn+Gemmini+Shuttle configuration.

    Recognized keys in `params` (extras are tolerated and ignored):
      vlen            -- Saturn architectural vector length, bits
      dlen            -- Saturn SIMD datapath width, bits
      gemmini_dim     -- Gemmini systolic array dimension (DIM x DIM PEs)
      sp_capacity_kb  -- Gemmini scratchpad capacity, KB
      acc_capacity_kb -- Gemmini accumulator capacity, KB
      dataflow        -- Gemmini dataflow: "WS", "OS", or "BOTH"
    """
    vlen = float(params.get("vlen", 256))
    dlen = float(params.get("dlen", 128))
    dim = float(params.get("gemmini_dim", 16))
    sp_kb = float(params.get("sp_capacity_kb", 256))
    acc_kb = float(params.get("acc_capacity_kb", 64))
    dataflow = str(params.get("dataflow", "BOTH")).upper()

    # Saturn VRF: 32 architectural vector registers, each VLEN bits wide.
    vrf_bits = NUM_VRF_ARCH_REGS * vlen
    saturn_vrf = vrf_bits * SRAM_BIT_AREA_MM2 * SRAM_PERIPHERY_OVERHEAD

    # Saturn datapath: one unified SIMD datapath scaling with DLEN (Saturn is
    # explicitly not a per-lane architecture -- system.adoc:28).
    saturn_lanes = dlen * DLEN_DATAPATH_AREA_PER_BIT_MM2

    # Gemmini systolic array: PE count scales with DIM^2.
    dataflow_factor = DATAFLOW_BOTH_OVERHEAD if dataflow == "BOTH" else 1.0
    gemmini_pes = dim * dim * GEMMINI_PE_AREA_MM2 * dataflow_factor

    # Gemmini scratchpad / accumulator SRAM, linear in capacity.
    gemmini_sp = (sp_kb * 1024 * 8) * SRAM_BIT_AREA_MM2 * SRAM_PERIPHERY_OVERHEAD
    gemmini_acc = (acc_kb * 1024 * 8) * SRAM_BIT_AREA_MM2 * ACC_PERIPHERY_OVERHEAD

    # Shuttle host core: fixed cost, independent of vector/accelerator config.
    shuttle_core_fixed = SHUTTLE_CORE_FIXED_MM2

    subtotal = (
        saturn_vrf + saturn_lanes + gemmini_pes + gemmini_sp
        + gemmini_acc + shuttle_core_fixed
    )
    other = subtotal * OTHER_GLUE_FRACTION

    breakdown = {
        "saturn_vrf": saturn_vrf,
        "saturn_lanes": saturn_lanes,
        "gemmini_pes": gemmini_pes,
        "gemmini_sp": gemmini_sp,
        "gemmini_acc": gemmini_acc,
        "shuttle_core_fixed": shuttle_core_fixed,
        "other": other,
    }
    total = subtotal + other

    notes = [
        _ASSUMPTIONS_NOTE,
        f"inputs: vlen={vlen:.0f} dlen={dlen:.0f} gemmini_dim={dim:.0f} "
        f"sp_capacity_kb={sp_kb:.0f} acc_capacity_kb={acc_kb:.0f} "
        f"dataflow={dataflow}",
    ]
    return AreaEstimate(total_mm2=total, breakdown=breakdown, notes=notes)


def read_synth_area(report_path: str | Path) -> AreaEstimate:
    """Parse a Genus/Yosys area report, if one ever exists.

    Stub: same `AreaEstimate` return type as `estimate_area()` so the outer
    loop can swap proxy -> real area with no call-site changes. Currently
    understands two common report shapes:
      * Cadence Genus `report_area` text: a line like
        "Total area: 12345.6" (arbitrary units) or a summary table with a
        trailing numeric column per cell/instance.
      * Yosys `stat` JSON/text output with a "Chip area for module" line.

    Anything it cannot confidently parse is surfaced via `notes` rather than
    raising, so a malformed/absent report degrades gracefully.
    """
    path = Path(report_path)
    notes = [f"read_synth_area: parsed from {path}"]
    if not path.exists():
        return AreaEstimate(
            total_mm2=float("nan"),
            breakdown={},
            notes=notes + ["report file does not exist; no real area yet"],
        )

    text = path.read_text(errors="ignore")
    total = None

    m = re.search(r"Chip area for module[^:]*:\s*([\d.eE+-]+)", text)
    if m:
        total = float(m.group(1))
        notes.append("matched Yosys 'Chip area for module' line")

    if total is None:
        m = re.search(r"Total\s+area\s*[:=]\s*([\d.eE+-]+)", text, re.IGNORECASE)
        if m:
            total = float(m.group(1))
            notes.append("matched Genus-style 'Total area' line")

    if total is None:
        return AreaEstimate(
            total_mm2=float("nan"),
            breakdown={},
            notes=notes + [
                "could not find a recognized area line; extend the regexes "
                "here once a real report format is available"
            ],
        )

    notes.append(
        "units are whatever the tool reported (not necessarily mm^2 -- "
        "Genus/Yosys area units depend on the target library); caller "
        "should confirm/convert before comparing against estimate_area()"
    )
    return AreaEstimate(total_mm2=total, breakdown={"synth_total": total}, notes=notes)


def _self_test() -> None:
    vlens = [128, 256, 512]
    dlens = [64, 128, 256]
    dims = [4, 8, 16, 32]

    header = (
        f"{'vlen':>6} {'dlen':>6} {'dim':>4} {'total_mm2':>10}  "
        f"{'vrf':>8} {'lanes':>8} {'pes':>8} {'sp':>8} {'acc':>8} "
        f"{'core':>6} {'other':>7}"
    )
    print("=== estimate_area() sweep: VLEN x DLEN (gemmini_dim=16) ===")
    print(header)
    prev_total_by_dlen: dict[int, float] = {}
    for dlen in dlens:
        prev_total = None
        for vlen in vlens:
            est = estimate_area({"vlen": vlen, "dlen": dlen, "gemmini_dim": 16,
                                  "sp_capacity_kb": 256, "acc_capacity_kb": 64,
                                  "dataflow": "BOTH"})
            b = est.breakdown
            print(f"{vlen:6d} {dlen:6d} {16:4d} {est.total_mm2:10.4f}  "
                  f"{b['saturn_vrf']:8.4f} {b['saturn_lanes']:8.4f} "
                  f"{b['gemmini_pes']:8.4f} {b['gemmini_sp']:8.4f} "
                  f"{b['gemmini_acc']:8.4f} {b['shuttle_core_fixed']:6.2f} "
                  f"{b['other']:7.4f}")
            if prev_total is not None:
                assert est.total_mm2 > prev_total, (
                    f"area not monotonic in vlen at dlen={dlen}: "
                    f"{prev_total} -> {est.total_mm2}"
                )
            prev_total = est.total_mm2
        prev_total_by_dlen[dlen] = prev_total  # type: ignore[assignment]

    # Monotonic in dlen (fixed vlen/dim).
    prev = None
    for dlen in dlens:
        est = estimate_area({"vlen": 256, "dlen": dlen, "gemmini_dim": 16})
        if prev is not None:
            assert est.total_mm2 > prev, f"area not monotonic in dlen: {prev} -> {est.total_mm2}"
        prev = est.total_mm2

    print("\n=== estimate_area() sweep: Gemmini DIM (vlen=256, dlen=128) ===")
    print(f"{'dim':>4} {'total_mm2':>10} {'gemmini_pes':>12}")
    prev = None
    for dim in dims:
        est = estimate_area({"vlen": 256, "dlen": 128, "gemmini_dim": dim,
                              "sp_capacity_kb": 256, "acc_capacity_kb": 64})
        print(f"{dim:4d} {est.total_mm2:10.4f} {est.breakdown['gemmini_pes']:12.4f}")
        if prev is not None:
            assert est.total_mm2 > prev, f"area not monotonic in dim: {prev} -> {est.total_mm2}"
        prev = est.total_mm2

    # Monotonic in scratchpad/accumulator capacity.
    prev = None
    for sp_kb in [64, 128, 256, 512]:
        est = estimate_area({"sp_capacity_kb": sp_kb})
        if prev is not None:
            assert est.total_mm2 > prev, f"area not monotonic in sp_capacity_kb"
        prev = est.total_mm2

    prev = None
    for acc_kb in [16, 32, 64, 128]:
        est = estimate_area({"acc_capacity_kb": acc_kb})
        if prev is not None:
            assert est.total_mm2 > prev, f"area not monotonic in acc_capacity_kb"
        prev = est.total_mm2

    # Extra/unknown keys must be tolerated.
    est = estimate_area({"vlen": 256, "dlen": 128, "gemmini_dim": 16,
                          "totally_unknown_key": "ignored", "clock_ghz": 1.0})
    assert est.total_mm2 > 0

    # read_synth_area stub: missing file degrades gracefully.
    missing = read_synth_area("/nonexistent/path/to/report.rpt")
    assert missing.breakdown == {} and missing.total_mm2 != missing.total_mm2  # NaN

    print("\nAll monotonicity/sanity assertions passed.")
    print(f"\nDefault config estimate: {estimate_area({})}")


if __name__ == "__main__":
    _self_test()
