## Round eleven: timing (this overrides "implement" above where they differ)

**Nothing is being added this round.** Every Zvvm cell is already implemented
and passes. The job this round is **timing**. The widening FP matrix datapath in
`${TIMING_TOP}` (`${SATURN_SRC_PATH}/src/main/scala/exu/fp/MatrixFPMultiplyPipe.scala`)
does all of this in ONE clock cycle:

- W products
- the exact-sum 560-bit align/negate/add
- the 1024-bit leading-zero count and normalise
- rounding to fmt_C
- the MXFP block-scale multiply
- the accumulate

Measured the judge's way it takes **319 ns** (logic depth 624 cells, from `op_*` to `io_write_bits_data`). Baseline Saturn's own vector FMA unit (`FPFMAPipe`),
measured exactly the same way, takes 32.9 ns. Pipeline the FU so that its
longest path is **<= ${TIMING_TARGET_NS} ns**, and change nothing else about
what it computes.

### The timing judge (runs after S1 and S2 pass, every iteration)

The loop takes the Verilog of the design the directed programs just ran on.
It synthesises `${TIMING_TOP}` standalone and **flattened**, together with all
of its submodules. The flow is yosys + abc (`-D 10000`) on
sky130_fd_sc_hd tt_025C_1v80. The judge reports the longest path that abc maps,
from a register or input port to a register or output port.

The judge passes only when both of these hold:

- the longest path is at most ${TIMING_TARGET_NS} ns
- ${TIMING_AREA_RULE}

Because the synthesis is flattened, wrapping a stage in its own Chisel
`Module` hides nothing. A path through a submodule is timed end to end.

The iteration is clean only when **S1, S2 and timing all pass**. The gate and
Stage 3 then run unchanged: the full directed sweep, the full RVV suite, and
randomised cosim against the Spike model.

### Hard rules (in addition to the ones above)

- **No functional change.** Every result must stay bit-identical, including:
  - NaN and -0 handling
  - the MXFP scale path
  - accumulation order
  - the SEW 32/64 paths

  Do not touch the Spike model (`riscv-isa-sim`), the tests, or anything
  outside the RTL.
- **Keep the datapath in `${TIMING_TOP}`.** The judge synthesises only this
  module and its children. Moving logic into the sequencer, the backend or a
  new sibling module does not make the design faster; it only moves the path
  where the judge cannot see it. The area floor catches that, and it is
  treated as a failure.
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
- More cycles per uop is allowed and expected. A uop that is not
  `block_last` can retire as soon as its state update is done. Use as few
  stages as you need to meet the target.
- Keep the area guard in mind. Every stage boundary on a ~560-bit sum or
  ~1024-bit intermediate costs thousands of flops. Cut where the
  intermediate is narrow: after the LZC, and after rounding.

### Your tools for this

`run_timing_start(rebuild=True)` → `run_timing_wait(job_id)` runs the same
judge on your current tree. At most ${TIMING_RUNS} per turn, one job at a time,
shared with `run_directed_start` and `run_rvv_start`. It rebuilds the tree
unless you pass `rebuild=False` right after a `run_directed_start` of the same
tree. Expect the build plus about 5 minutes of synthesis.

The report gives you:

- the longest path
- its logic depth
- its start and end points: flattened net names, whose prefix is the
  instance path
- the gate mix and arrival profile along it

Use it to decide where the next register goes.

Before you finish, run `run_directed_start('all')`. Every directed program
must still pass. A faster FU that computes something else is a regression.
