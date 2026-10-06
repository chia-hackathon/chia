## Round twelve: PPA, area first (this overrides "implement" above where they differ)

**Nothing is being added this round.** Every Zvvm cell is implemented and
passes, and the FP matrix unit already meets its timing goal. The job this
round is **area**. `${PPA_TOP_FP}`
(`${SATURN_SRC_PATH}/exu/fp/MatrixFPMultiplyPipe.scala`) is
${PPA_R35_FP_AREA} um2 when the judge synthesises it flattened -- 83% of the
whole IME overhead. Make it smaller: its flat area must come down to
**<= ${PPA_A_TARGET} um2** (stretch goal ${PPA_A_STRETCH} um2), and change
nothing else about what it computes or how many cycles it takes.

### What the judge measured on the tree you start from

Flat, standalone `${PPA_TOP_FP}`: ${PPA_R35_FP_AREA} um2, about 71,000 cells,
about 5,000 flops, longest path 25.98 ns. The same unit synthesised
hierarchically (each submodule on its own; this overstates the area, because
flattening lets the mapper optimise across submodule boundaries) is
798,093 um2 and 5,380 flops, made of:

| item | instances | area um2 (hier.) | where it comes from |
|---|---|---:|---|
| own registers | 5,263 enable flops + 117 plain flops | 160,384 | stage regs, S and -S (2 x 560 b), acc, `v0_egs` (2 x 128 b) |
| own combinational logic | | 219,507 | 7 x 80-bit chunk adders x 2, 1024-bit LZC + normaliser, AddRawFN halves, OFP8 path, muxes |
| `MulFullRawFN_1` (53 x 53, binary64) | 3 | 288,224 | `mulF_d`, `bsF_d`, `smF_d` |
| `MulFullRawFN` (24 x 24, binary32) | 3 | 61,781 | `mulF_s`, `bsF_s`, `smF_s` |
| `MulFullRawFN_2/_3` (binary16, bfloat16) | 2 + 2 | 14,564 | `bsF_h/bf`, `smF_h/bf` |
| `RoundRawFNToRecFN_*` | 14 | 38,488 | one per format per stage |
| `RoundAnyRawFNToRecFN_*` | 8 | 8,442 | int-to-fp and S-to-fmt_C, one per format |
| `RecFNToRecFN_*` | 7 | 6,703 | E8M0 scale conversions, bf16 widen/narrow |

`${PPA_TOP_INT}` (the int8 MAC, one 64 x 64 signed multiply plus a 3-input
64-bit add per product) is the other FU; the judge also measures it.

Facts about the design, as the judge and the source show them:

- The FU is an `IterativeFunctionalUnit` with **exactly one uop in flight**
  (`io.stall := valid`). Every stage register is written once per uop, in
  the stage counter `cnt`'s order, and the next uop does not start until the
  write has fired. One format (H / BF / S / D) is selected per uop.
- The FP critical path is `s0_p_sign -> fpw_cA_reg`, the 80-bit chunk carry,
  25.98 ns flat. The INT FU is at 29.66 ns in the full SoC, 0.6 ns from the
  SoC's longest path. Both FU delay gates (`${PPA_FP_DELAY}` / `${PPA_INT_DELAY}`
  ns) are the SoC's longest abc path (30.25 ns): abc's delay is noisy, an
  equivalent rewrite can move a flat path by up to +3.2 ns, so a gate tighter
  than the SoC path would reject equivalent designs. A flat INT path of
  ${PPA_INT_STRETCH} ns or less is reported as the stretch goal, not gated.
- `${PPA_TOP_INT}` retires one product per cycle (`last := Mux(op.matrix_last_k,
  io.write.fire, valid)`), so one more register on its path is one more
  cycle per k.

### The PPA judge (runs after S1 and S2 pass, every iteration)

The loop takes the Verilog of the design the directed programs just ran on.
Stage P is, in this order:

1. **Scope check** (no synthesis). Your changes may touch only
   `exu/fp/MatrixFPMultiplyPipe.scala`, `exu/int/MatrixMultiplyPipe.scala`,
   and new files under `exu/fp/` or `exu/int/`. Any other file, any change to
   the Spike model (`riscv-isa-sim`), any test or script, is rejected as a
   scope violation: not a verdict, nothing is measured. These constructs are
   forbidden in the code you add: `BlackBox`, `ExtModule`, `HasBlackBox*`,
   `Mem`, `SyncReadMem`, `asClock`, `withClock*`, `ClockGate*`, `annotate`,
   Verilog attributes, `printf`/`stop`. **Clock gating is not allowed.**
2. **Synthesis** on sky130_fd_sc_hd tt_025C_1v80 (yosys + abc, `-D 10000`):
   `${PPA_TOP_FP}` and `${PPA_TOP_INT}` flattened, each with all submodules.
   Wrapping a stage in its own Chisel `Module` hides nothing. The netlist must
   contain only sky130_fd_sc_hd cells, no latches, no unmapped cells, and the
   same top-level ports as the starting tree.
3. **Cycle suite**: nine bare-metal programs (int8 `vqmmacc`, FP16 and bf16
   `vfwmmacc`, OFP4 `vf8wmmacc`, MXFP8, MXINT8, `vfmmacc` FP32 and FP64, bf16
   C) on the non-cosim Verilator build, `rdcycle` deltas around the one IME
   instruction, deterministic. Every benchmark must take **no more cycles than
   the starting tree**. Fewer is fine.

The judge passes only when all of these hold:

- `${PPA_TOP_FP}` flat area <= ${PPA_A_TARGET} um2
- `${PPA_TOP_FP}` longest path <= ${PPA_FP_DELAY} ns
- `${PPA_TOP_INT}` longest path <= ${PPA_INT_DELAY} ns
- ${PPA_INT_AREA_RULE}
- the netlist sanity checks and the cycle suite above

Power is reported (flop count, enable-flop count, liberty leakage sum) and
**not** graded.

The iteration is clean only when **S1, S2 and stage P all pass**. The loop
keeps the clean iteration with the smallest FP area as the best one (the last
iteration is not automatically the best, and the loop will restore the best
tree if you end on a worse one). It stops when the stretch goal is met, or
after ${PPA_NO_PROGRESS} judged attempts in a row without an area improvement
larger than the measurement noise (${PPA_NOISE_NOTE}). The gate and Stage 3
then run on the best tree: the full directed sweep, the full RVV suite, a
full-SoC synthesis that checks nothing outside the two FUs grew, and
randomised cosim against the Spike model.

### Hard rules (in addition to the ones above)

- **No functional change.** Every result must stay bit-identical, including:
  - NaN and -0 handling
  - the MXFP scale path
  - accumulation order
  - rounding mode and tininess behaviour for every format
  - the SEW 32/64 paths

  Do not touch the Spike model (`riscv-isa-sim`), the tests, or anything
  outside the files listed above. The functional judges are simulation only: a
  claim that two things "compute the same" is worth nothing unless the tests
  agree.
- **No cycle regression.** The cycle suite compares exactly against the
  starting tree.
- **Keep the datapath in `${PPA_TOP_FP}` and `${PPA_TOP_INT}`.** Moving logic
  into the sequencer, the backend or another module outside the allowed files
  is a scope violation.
- **It is an `IterativeFunctionalUnit`. Keep it one.** Make it a multi-cycle
  unit (a stage counter or FSM with registers at the cut points), not a
  `PipelinedFunctionalUnit`. The pipelined path has a fixed write slot, no
  write backpressure, and single-element-group hazard reporting, and it
  would break the accumulator recurrences.

  The interface already tolerates any latency, provided you keep these:

  - Keep `io.stall := valid`, and keep `io.hazard.valid` high for the whole
    busy period, until the write has fired. Together these give the RAW
    protection on the accumulator group C and the late vat release
    (`pipelinedVatClear = false`). Dropping `valid` early re-opens a
    read-before-write on C.
  - Drive `io.write.valid` from a register. `io.write.ready` can be low
    (it is the iterative write arbiter plus the VRF hiccup buffer), so the
    write data must stay stable until `io.write.fire`.
  - The state carried across uops must update **exactly once per uop**, in
    the cycle its value is produced. Today every piece of it updates on
    `valid && !valid_prev`. The state is:
    - the group-sum `fpw_acc_reg` with its NaN/Inf/±0 trackers
    - `acc_reg`
    - `int_acc_reg`
    - `nan_out_reg`
    - the `v0_egs` scale cache

    If you register the inputs to those updates, move each update to the
    stage that produces its value. Do not update again while you wait for
    `write.ready`; that double-adds.
  - `op.*` (`rvd_data`, `frm`, format bits) is stable while `valid` is
    high. Only datapath intermediates need stage registers.
  - Do not use `valid && !last` stall tricks to overlap uops unless you
    replace the `valid_prev` edge detection with an explicit
    first-cycle flag.

### Your tools for this

`run_ppa_start(rebuild=True)` -> `run_ppa_wait(job_id)` runs the scope check
and the three syntheses (flat FP, flat INT, hierarchical FP for attribution)
on your current tree. At most ${PPA_RUNS} per turn, one job at a time, shared
with `run_directed_start`, `run_rvv_start` and `run_cycles_start`. It rebuilds
the tree unless you pass `rebuild=False` right after a `run_directed_start`
of the same tree. Expect the build plus about 5 minutes of synthesis.

`run_cycles_start(rebuild=True)` -> `run_cycles_wait(job_id)` runs the cycle
suite (at most ${PPA_CYCLE_RUNS} per turn, same sharing rules).

The PPA report gives you:

- a metric table: this tree, the starting tree, the best clean tree so far,
  the target
- the hierarchical submodule table (top 15, own register / logic split)
- the flat netlist's flops grouped by register-name prefix
- the critical path of each FU (start and end points are flattened net names,
  whose prefix is the instance path)
- the reasons for any failure

Before you finish, run `run_directed_start('all')`. Every directed program
must still pass. A smaller FU that computes something else is a regression.
