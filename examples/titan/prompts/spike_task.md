Add the RISC-V Integrated Matrix Extension (Zvvm) to Spike. Four
instructions are already implemented and passing: `vmmacc.vv`, `vmtl.v`,
`vmts.v`, `vqmmacc.vv`. Round three adds four more: `vwmmacc.vv`,
`v8wmmacc.vv`, `vmttl.v`, `vmtts.v` -- plus the `vtype` fields they need.

Start with `read_spec`. The specification carries formal SAIL semantics for
each instruction; transcribe those. Spike's source is at `${SPIKE_SRC_PATH}`.

## Rounds two and three

Round two (`vqmmacc.vv`) is done and converged. Round three adds
`vwmmacc.vv`, `v8wmmacc.vv`, `vmttl.v`, `vmtts.v`. If a working model for
earlier instructions is already applied in the tree, extend it --
everything that passes today must still pass. The SAIL appendix is
normative and outranks the prose and the tests, for the new instructions as
much as the old.

Two facts that save a wrong turn here. `vmttl.v`/`vmtts.v` are
`vmtl.v`/`vmts.v` with two changes: bits 27:26 `0b01` not `0b00`, and offset
`(i % linesize) * LD + (i / linesize)` (div/mod swapped); `flat_idx =
tile_reg_idx(...)` is unchanged, `rs2 = 0` default LD is now
`VLEN/(SEW*LAMBDA)` not `LAMBDA*LMUL`. And `vm = 0` on funct6
`0x39`/`0x3a`/`0x3b` is not a don't-care: it decodes
`vfwimmacc.vv`/`vfqimmacc.vv`/`vf8wimmacc.vv` (out of scope) -- raise
illegal-instruction, don't leave it unhandled.

## Suggested order

1. **`vtype` first.** Add `lambda[2:0]`, `bs`, `altfmt_A` and `altfmt_B` at
   the high end of `vtype`, immediately below `vill`, and implement the
   `vsetvl` write rules including WARL clamping of `lambda`. Nothing else can
   be tested until a geometry can be requested and read back, and the test
   programs check exactly this before doing anything else — so it is also
   where you get feedback fastest.
2. **The tile load and store.** Their pseudocode is explicit and short. Get
   them exact before the arithmetic: if `vmtl.v` places the wrong elements,
   every multiply-accumulate result will be wrong for a reason that has
   nothing to do with the multiply.
3. **`vmmacc.vv`.** Derive the geometry (`M`, `K_eff`, `N`, `EMUL_C`) from
   `VLEN`, `SEW`, `LAMBDA`, `LMUL` and `VL` rather than special-casing sizes.
4. **The legality checks.** Illegal-instruction conditions are listed per
   instruction in the spec. A model that silently accepts an illegal encoding
   will let a hardware bug through later.

## Things the spec is specific about and easy to get wrong

- `EMUL_C` is independent of `LMUL`, and may be 16 — outside the range an
  ordinary RVV operand `EMUL` can take. The C group's base register aligns to
  `EMUL_C`, not to `LMUL`.
- The C tile's row stride is the physical edge `N_max = M` even when only
  `N < M` columns are active. The inactive ones are C tile tail elements,
  defined by the two-dimensional geometry rather than by `element_index >= VL`,
  and their fate depends on `vta`.
- The `VL` used for a multiply-accumulate is not the `VL` used to move a C
  accumulator tile.
- `lambda` written through `vsetvl` is a request, not an assignment.
- An order-preserving tile load reads `LAMBDA * LMUL` contiguous elements and
  then strides by the leading dimension; if `rs2` is zero the leading
  dimension defaults to `LAMBDA * LMUL`.
- In the SAIL body of `vmtl.v` / `vmts.v`, the loop variable `i` is the
  *sequential* tile element index. Memory uses it directly
  (`mem_off = (i / linesize) * LD + (i % linesize)`) and the register file
  uses `tile_reg_idx(i, LMUL, eff_lambda, elems_per_reg)`. The two indices
  are never composed. Call `tile_reg_idx`; do not substitute `flat_idx = i`
  on the grounds that it is the identity at LMUL = 1.
- `vmmacc.vv` reads and writes C through `mat_C_idx(i, j, N_max, EMUL_C,
  lambda, epr) = tile_reg_idx(i * N_max + j, EMUL_C, lambda, epr)`. The group
  multiplier there is `EMUL_C`, not `LMUL`. It collapses to `i * N_max + j`
  only when `EMUL_C = 1`.

**The SAIL appendix is normative and outranks the tests.** If a test fails
and the only fix you can see is to write something the SAIL does not say, do
not write it: transcribe the SAIL, record the disagreement with
`append_knowledge` (quoting the SAIL lines and the numbers that differ), and
say so in `finish`. Never add code that deviates from the SAIL to make a test
pass.

Write C++ only. Do not build, do not run Spike, do not read the hardware
implementation or the Titan reference model. Call `finish` when done.
