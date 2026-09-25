Add the RISC-V Integrated Matrix Extension (Zvvm) to Spike. Nine
instructions are already implemented and passing: `vmmacc.vv`, `vmtl.v`,
`vmts.v`, `vqmmacc.vv`, `vwmmacc.vv`, `v8wmmacc.vv`, `vmttl.v`, `vmtts.v`,
`vfmmacc.vv`. Round six adds three more: `vfwimmacc.vv`, `vfqimmacc.vv` and
`vf8wimmacc.vv` -- the microscaled integer-input, floating-point-accumulate
forms, plus the paired E8M0 block scales in `v0` that they read.

Start with `read_spec`. The specification carries formal SAIL semantics for
each instruction; transcribe those. Spike's source is at `${SPIKE_SRC_PATH}`.

## Rounds one through six

Rounds one through five are done and converged: `vmmacc.vv`, `vmtl.v`,
`vmts.v`, `vqmmacc.vv`, `vwmmacc.vv`, `v8wmmacc.vv`, `vmttl.v`, `vmtts.v`,
`vfmmacc.vv`. Round six adds three: `vfwimmacc.vv` (W=2), `vfqimmacc.vv`
(W=4) and `vf8wimmacc.vv` (W=8). If a working model for earlier
instructions is already applied in the tree, extend it -- everything that
passes today must still pass. The SAIL appendix is normative and outranks
the prose and the tests, for the new instructions as much as the old.

Two facts that save a wrong turn here. `vmttl.v`/`vmtts.v` are
`vmtl.v`/`vmts.v` with two changes: bits 27:26 `0b01` not `0b00`, and offset
`(i % linesize) * LD + (i / linesize)` (div/mod swapped); `flat_idx =
tile_reg_idx(...)` is unchanged, `rs2 = 0` default LD is now
`VLEN/(SEW*LAMBDA)` not `LAMBDA*LMUL`. And `vm = 0` on funct6
`0x39`/`0x3a`/`0x3b` is not a don't-care: it decodes
`vfwimmacc.vv`/`vfqimmacc.vv`/`vf8wimmacc.vv`, which is **this round's
work**. Those three share funct6 with `vwmmacc.vv`/`vqmmacc.vv`/
`v8wmmacc.vv` and are separated by `vm` alone, so mis-routing `vm` breaks
rounds one through three as well as failing round six.

Round six, transcribed from Sail `int_scaled_gemm` (5373-5410). There is
**no (G, psm, rnd)** on this path -- it never calls `get_fp_grouping` /
`get_fp_psm` / `get_fp_rnd` and has no `G` check, unlike `fp_gemm` at
5238-5242 (spec 1283-1287, 1645-1652). The loop nest is `j`/`i`/`s` with
**no LMUL step loop**; the block/step intersection and shortened groups of
spec 2030-2060 belong to `fp_scaled_gemm`, not here. `int_block_dot`
(5128-5147) is **exact and unbounded, no modular reduction**, and reads both
operands signed regardless of altfmt (5397 passes literal `true`/`true`).
Then exactly three roundings per (element, block) under `frm`: `int_to_fp`,
`fp_mul` by the block scale, `fp_add` into C. The paired E8M0 scales are
read from `v0` at pair width 16 -- low byte `scale_A` at `i*R+s`, high byte
`scale_B` at `j*R+s`, `R = LAMBDA*SEW/16` (2129-2170, Sail 5097-5122).
`bs` is `vtype[XLEN-5]` (0 -> 32, 1 -> 16), legality is `SEW*LAMBDA >= 16`
and at `bs=1` also `W*LMUL <= SEW` (Sail 5151-5158). The NaN-scale early
exit (5390-5392, 5403-5406) tests the **combined** scale after conversion
and multiplication, not the encoded bytes: spec 2021-2024 makes `+0 x +inf`
from two finite bytes a default NaN. E8M0 itself is restated in full at
1990-1993 -- bias 127, 2^-127..2^127, `0xFF` NaN, and **no zero, infinity or
subnormal encoding**: byte 0x00 is the ordinary finite value 2^-127.

Five facts about `vfmmacc.vv`, still in the regression. Titan discloses **G=1, psm=0, rnd=frm**
(spec 1771): the model must not fuse multiply-add or sum groups, so
`acc = fp_add(acc, fp_round_to_frm(fp_mul_exact(a,b)))` per increasing `k`,
two roundings per term under `frm`. `vtype.SEW` is the accumulator width,
as in round two; `altfmt_A`/`altfmt_B`/`altfmt` are ignored at SEW 32/64.
`vm = 0` on funct6 `0x14` is **reserved** -- raise illegal-instruction.
Its funct3 is **OPFVV (0x1), not OPIVV**, unlike every instruction so far.
NaNs canonicalise and `fflags` OR across active elements, but the directed
tests do not compare `fflags` either way.

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

**Independence is the point of this stage, and round seven makes it fragile.**
You are the third derivation of these semantics, after the SAIL you are
transcribing and the Titan reference model you must not read. That
independence is what makes an agreement between the three worth anything.

From round seven onwards the directed programs for the widening
floating-point instructions (`vfwmmacc.vv`, `vfqmmacc.vv`, `vf8wmmacc.vv`)
carry *embedded expected bytes*: the result of each multiply-accumulate,
precomputed by the Titan reference model and compared with `memcmp`, because
a narrow accumulator (OFP8, binary16, bfloat16) cannot be recomputed on the
DUT with baseline `rv64imafd` instructions. Those bytes will be visible to
you in any test source you happen to open.

Do not use them. Specifically:

- Do not read an expected-value blob and work backwards to the arithmetic
  that must have produced it. That turns three independent derivations into
  one derivation and two copies of it, and a mistake in the reference model
  then reaches the hardware wearing three separate endorsements.
- Do not tune rounding, ordering, or special-value handling until a golden
  comparison passes. Transcribe what the SAIL says; if the result disagrees
  with an embedded expectation, that disagreement is a *finding* — record it
  with `append_knowledge`, quoting the SAIL lines and both values, and say so
  in `finish`.
- **A disagreement with our golden bytes does not mean you are wrong.** The
  golden bytes come from the Titan reference model, and the Titan reference
  model can be the mistaken party — that is precisely why your derivation is
  kept independent of it. You are not the junior party to it; you are a
  second opinion whose whole value is that it was formed separately. So when
  your SAIL transcription and our expected value disagree, report the
  disagreement and stop. Do not resolve it by moving toward us. A Stage 0
  model that quietly converges on our expectations tells us nothing we did
  not already believe, and it destroys the only check we have on the
  reference model itself.
- The OFP8 (E4M3, E5M2) and OFP4 (E2M1) element encodings are defined by the
  OCP specifications the IME adoc cites normatively at lines 1908-1913, not
  by the IME adoc itself and not by analogy with IEEE 754. The OCP OFP8 v1.0
  specification is available next to the adoc as
  `specs/ime/ocp-ofp8-v1.0.txt` (listed by the spec tool): read it for E4M3 /
  E5M2. OCP MX v1.0 (E2M1 / OFP4) is NOT available; if you need a rule it
  states, stop and say so in `finish`. Do not infer one. E4M3 in particular
  has no infinity encoding and only one NaN pattern (S.1111.111), so an
  IEEE-shaped decode or overflow path is wrong there in a way that looks
  right.
