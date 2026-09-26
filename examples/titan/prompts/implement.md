Implement round six of the RISC-V Integrated Matrix Extension (Zvvm) in
Saturn: `vfwimmacc.vv`, `vfqimmacc.vv` and `vf8wimmacc.vv` -- the microscaled
integer-input, floating-point-accumulate forms -- on top of `vmmacc.vv`,
`vmtl.v`, `vmts.v`, `vqmmacc.vv`, `vwmmacc.vv`, `v8wmmacc.vv`, `vmttl.v`,
`vmtts.v`, `vfmmacc.vv`, already implemented and passing.

Start by calling `read_spec` and reading the specification. Then read Saturn.
The orientation in your system prompt tells you where to look; it does not
tell you what the code says.

## What done looks like

The design elaborates as `${DIRECTED_CONFIG}` (the design itself, in
`SaturnConfigs.scala`). That is the only config you write. The RVV
regression runs on `${COSIM_CONFIG}`, which the loop generates for itself
from `${DIRECTED_CONFIG}` plus the cosim harness fragments -- do not create,
edit or reference it, and put everything that changes the design in
`${DIRECTED_CONFIG}`. The regression cosimulates against the stock Spike,
which knows nothing about Zvvm, so it only tests that plain RVV 1.0 still
behaves. Every directed test either passes or
reports SKIP. A SKIP means you do not support that tile geometry,
which is allowed. A mismatch means the IME path disagrees with a plain RVV 1.0
sequence computing the same thing, and that is never allowed.

## Rounds one through six

Rounds one through five are done and converged: `vmmacc.vv`, `vmtl.v`,
`vmts.v`, `vqmmacc.vv`, `vwmmacc.vv`, `v8wmmacc.vv`, `vmttl.v`, `vmtts.v`,
`vfmmacc.vv`. Round six adds three: `vfwimmacc.vv` (W=2), `vfqimmacc.vv`
(W=4) and `vf8wimmacc.vv` (W=8). Where RTL for earlier instructions is
already in the tree, extend it: every directed test that passes today must
still pass. The spec's SAIL appendix is normative and outranks its prose --
never write something the SAIL does not say in order to make a test pass.

Three more facts. `vwmmacc.vv`/`v8wmmacc.vv` reuse the round-two datapath
(SEW stays the accumulator width, tiles arrive via `vmtl.v`, only funct6 and
unpack depth W change). `vmttl.v`/`vmtts.v` are `vmtl.v`/`vmts.v` with two
changes: bits 27:26 `0b01` not `0b00`, and offset `(i % linesize) * LD + (i /
linesize)` (div/mod swapped); `flat_idx = tile_reg_idx(...)` is unchanged,
`rs2 = 0` default LD is now `VLEN/(SEW*LAMBDA)` not `LAMBDA*LMUL`. And `vm = 0` on funct6 `0x39`/`0x3a`/`0x3b` is not a don't-care: it decodes
`vfwimmacc.vv`/`vfqimmacc.vv`/`vf8wimmacc.vv`, which is **this round's work**.
Those three share their funct6 with `vwmmacc.vv`/`vqmmacc.vv`/`v8wmmacc.vv`
and are separated by `vm` alone, so a decoder that mis-routes `vm` breaks
rounds one through three as well as failing round six. The `ime_mxl_` tier
exists to catch exactly that.

Two facts about `vqmmacc.vv` that save a wrong turn: `vtype.SEW` is the
*accumulator* (C) width, so int8 A/B means SEW=32 -- M, N_max, EMUL_C and the
permissible LAMBDA set are exactly what they are for the non-widening case,
and only A/B change (`row_elems_per_reg = lambda*4`, `epr_A = 4*epr_C`). And
it needs **no new tile-load path**: `vmtl.v` moves SEW-wide storage elements,
and logical element (r,k) lands at `4*flat_storage + k%4`, so the memory image
is the plain row-major int8 panel the existing loader already fetches. The RTL
work is a 4-way-packed int8 MAC datapath feeding the existing 32-bit C tile,
not new addressing.

Seven facts about round six. **There is no (G, psm, rnd) here at all.**
Sail `int_scaled_gemm` (5373-5410) never calls `get_fp_grouping` /
`get_fp_psm` / `get_fp_rnd` and has no `G` legality check, unlike `fp_gemm`
at 5238-5242; spec 1283-1287 and 1645-1652 say so in prose. Nothing about
round four's disclosure applies. **The loop nest is `j`/`i`/`s` only --
there is no LMUL step loop.** The block-and-step intersection and shortened
groups of spec 2030-2060 belong to `fp_scaled_gemm`; here `int_block_dot`
gets the whole block interval `[s*block_size, min(k_lo+block_size,K_eff)-1]`.
**`int_block_dot` is exact and unbounded** (Sail 5128-5130, "no overflow")
and reads both operands **signed regardless of altfmt** (5397 passes the
literals `true`/`true`); there is no modular reduction anywhere on this
path, unlike `int_gemm`. **Exactly three roundings per (element, block)**:
`int_to_fp`, `fp_mul` by the block scale, `fp_add` into C, all under `frm`.
**The paired E8M0 scales live in `v0`**, read at the pair width 16: low byte
`scale_A`, high byte `scale_B`, row stride `R = LAMBDA*SEW/16`, A at
`i*R+s` and B at `j*R+s` out of the same register (spec 2129-2170, Sail
5097-5122). `vd`/`vs1`/`vs2` must not overlap `v0`. **`bs` is
`vtype[XLEN-5]`, not an instruction field** (spec 1160-1176): 0 means block
size 32, 1 means 16. Legality: `SEW*LAMBDA >= 16` always, and at `bs=1` also
`W*LMUL <= SEW` (Sail 5151-5158). **The NaN-scale early exit** (Sail
5390-5392, 5403-5406) tests the *combined* scale after conversion and
multiplication, not the encoded bytes: spec 2021-2024 makes `+0 x +inf` from
two finite E8M0 bytes a default NaN, and a model that only looks for `0xFF`
misses it. `altfmt_A` and `altfmt_B` must be 0 (MXINT is signed
unconditionally); `vtype.altfmt` selects the C accumulator format and is a
*base* Zvfbfa field at an absolute bit position, not one of the IME fields
keyed by offset below XLEN.

Five facts about `vfmmacc.vv`, still in the regression. Titan discloses **G=1, psm=0, rnd=frm**
(spec 1771): the DUT must not fuse multiply-add or sum groups, so
`acc = fp_add(acc, fp_round_to_frm(fp_mul_exact(a,b)))` per increasing `k`,
two roundings per term under `frm`. `vtype.SEW` is the accumulator width,
as in round two; `altfmt_A`/`altfmt_B`/`altfmt` are ignored at SEW 32/64.
`vm = 0` on funct6 `0x14` is **reserved** -- raise illegal-instruction.
Its funct3 is **OPFVV (0x1), not OPIVV**, unlike every instruction so far.
NaNs canonicalise and `fflags` OR across active elements, but the directed
tests do not compare `fflags` either way.

The matrix FU's `io.stall` must stay `valid`-only. Folding `post_write_stall`
into the shared int/fp issue path once broke every `.vf` RVV test in the S2
regression; the matrix unit does not get to stall that path.

## Round nine: the rest of the integer table, and the narrow `vfmmacc.vv` cells

Round nine adds no mnemonic. Everything since round one above is the
regression surface; the new *cells and rows* of instructions you already
have are the work (spec 810-844 tbl-extensions; tbl-int-encoding-map
7389-7455; tbl-fp-encoding-map 7296-7311):

- **Signedness is `vtype.altfmt_A` / `vtype.altfmt_B`** (`vtype[XLEN-6]`,
  `vtype[XLEN-7]`; 0 = signed, 1 = unsigned, independent per operand; C is
  always signed -- spec 486, 1145-1154, 7375). Not a mnemonic, not `vm`.
  Only `vsetvl` reaches those bits. Sail `int_block_dot` (5127-5145) reads
  each operand with `signed()` or `unsigned()` accordingly. All four rows
  are legal for all thirteen integer cells. At W=1 the choice cannot change
  a result bit (products and sums wrap at 2^SEW), so those programs only
  check the encoding is accepted; at W>1 it is arithmetic. Directed names:
  `ime_sg{su|us|uu}_w{W}_...` (e.g. `su` = A signed, B unsigned).
- **Int4 cells**: `vwmmacc.vv` at SEW=8 (Zvvi4i8mm), `vqmmacc.vv` at SEW=16
  (Zvvi4i16mm), `v8wmmacc.vv` at SEW=32 (Zvvi4i32mm). Two elements per
  byte, element 2n in the LOW nibble, 2n+1 in the high nibble (spec
  1206-1219); `vmtl.v` still moves SEW-wide storage elements. Names
  `ime_i4w_`, `ime_i4q_`, `ime_i48w_`.
- **Int64 cells**: `vqmmacc.vv` at SEW=64 (Int16 -> Int64, Zvvi16i64mm) and
  `vwmmacc.vv` at SEW=64 (Int32 -> Int64, Zvvi32i64mm). One reduction of the
  exact sum modulo 2^64 (spec 487, 1275-1277) -- never 2^32. Names
  `ime_i64q_`, `ime_i64w_`.
- **`vfmmacc.vv` at SEW=16 and SEW=8** (Zvvfp16mm, Zvvbf16mm, Zvvofp8mm):
  binary16 / bfloat16 / mixed inputs into a binary16 or bfloat16 C selected
  by `altfmt` ((0,0,1) and (1,1,0) are reserved), and E4M3/E5M2 inputs into
  an E4M3 (`altfmt`=0) or E5M2 (`altfmt`=1) C. Same (G, psm, rnd) =
  (1, 0, frm) as round four: per k, the exact product is rounded to the C
  format, then added and rounded again **in the C format** -- not in binary32
  with one narrowing at the end. IEEE signed zeros (-0 + -0 = -0). For an
  OFP8 C the disclosed choices are: overflow non-saturating (E4M3 -> NaN,
  E5M2 -> Inf), default NaN E4M3 0x7F / E5M2 0x7E, RNE only. These programs
  embed expected bytes (`ime_fpn_golden_`, `ime_fpn_special_`).

## Suggested order

1. **Read first, edit second.** `${SATURN_SRC_PATH}` — in particular
   `exu/int/SegmentedMultiplyPipe.scala`, `insns/Instructions.scala`,
   `insns/Decode.scala`, `backend/ExecuteSequencer.scala`, and `mem/AddrGen.scala`.
   Record what you learn with `append_knowledge` as you go; your conversation
   does not survive to the next iteration but that file does.
2. **`vtype` first.** The new fields live in rocket-chip's `VType`/`VConfig`,
   not in Saturn. Nothing else can work until `lambda` can be set and read.
   Get `vsetvl` writing it, WARL-clamping it to the set you actually support,
   and `vtype` reading it back — the tests check exactly this before they do
   anything else, so this is also the fastest thing to get feedback on.
3. **The tile load and store.** Before the arithmetic. If `vmtl.v` puts the
   wrong elements in the register group, every multiply-accumulate result
   will be wrong for a reason that has nothing to do with the multiplier, and
   you will debug the wrong unit. Load and store are inverses: a `vmtl.v`
   followed by `vmts.v` with the same configuration must reproduce memory
   exactly, and that is worth convincing yourself of first.
4. **Then `vmmacc.vv`.** The geometry is derived, not encoded — re-derive
   `M`, `K_eff`, `N` and `EMUL_C` from `VLEN`, `SEW`, `LAMBDA`, `LMUL` and
   `VL` rather than special-casing sizes.
5. **The two configs**, per your system prompt.

## Things that have caught people out

- `EMUL_C` is independent of `LMUL`, and can be 16 — wider than any ordinary
  RVV operand `EMUL`. The C group's base register aligns to `EMUL_C`.
- The C tile's row stride is the *physical* edge `N_max = M`, even when only
  `N < M` columns are active. The inactive ones are C tile tail elements
  governed by the two-dimensional geometry, not by `element_index >= VL`.
- The compute `VL` is not the `VL` for moving a C tile. They differ.
- An order-preserving tile load reads `LAMBDA * LMUL` contiguous elements and
  then strides by the leading dimension. Per the spec's SAIL, the *memory*
  side of a tile load/store is addressed by the sequential element index `i`
  alone: `mem_off = (i / linesize) * LD + (i % linesize)`, with
  `linesize = LAMBDA * LMUL`. An A/B tile in memory is therefore a plain
  row-major `M x K_eff` panel with stride `LD` at **every** `LMUL` — the
  `LMUL > 1` shuffling is entirely on the register side, via
  `tile_reg_idx(i, LMUL, LAMBDA, VLEN/SEW)`. Do not compose the two indices.

- **A matrix FP unit does not belong in the integer FU list.** Putting
  `MatrixFPMultiplyFactory()` into `integerMatrix` once broke 23 `.vf` tests
  in the Stage 2 regression and cost three iterations to find, because the
  symptom was floating-point *RVV* tests failing rather than anything matrix
  related. Round six accumulates in floating point, so it lands in the same
  place: give it its own `fpMatrix` list rather than extending the integer
  one. Check this before you start debugging arithmetic.

- **Do not make a check pass by changing what it counts.** There was an
  iteration that kept `fus.size` constant by merging two FU factories into
  one, so the check that guards the issue-path shape stopped complaining.
  That is hiding the problem, not fixing it. If a structural check fires,
  either the structure is wrong or the check is wrong -- say which, and fix
  that. The same rule covers narrowing a test's coverage, relaxing a
  tolerance, or skipping a geometry to turn a red run green.

- **OFP8 and OFP4 special values are not in the adoc.** Spec 1910-1913 is a
  *normative reference* to the OCP Microscaling Formats (MX) v1.0
  specification, not a restatement; the adoc never gives the bit encodings.
  E4M3 has no infinities and only one NaN encoding, E5M2 is IEEE-shaped,
  E2M1 has neither NaN nor infinity. If you need any of those rules, get
  them from OCP -- a guess here produces wrong numbers that look exactly
  like an RTL bug and will cost you iterations. The OCP OFP8 v1.0 text is at
  `specs/ime/ocp-ofp8-v1.0.txt` (E4M3/E5M2); OCP MX v1.0 (E2M1) is not
  available, and the E2M1 extensions are declared unsupported. (Round six itself does not
  need them: its data inputs are MXINT4/MXINT8, plain signed two's
  complement with no special values, and E8M0 *is* restated in full at spec
  1990-1993. This matters for the floating-point rounds.)

- **Every version silent, all stopping at the same simulated time, means
  the cycle budget ran out.** That is `+max-cycles`, not your RTL and not
  the test. A real hang or a real functional failure does not line up to the
  same cycle count across unrelated builds. Check the stop time before you
  go looking for a bug.

## Testing your own work

You have `run_directed_start` / `run_directed_wait`. Use them: get
`vtype.lambda` writable and readable, then `run_directed_start("all")`
immediately — the tests check exactly that before anything else, so it is the
fastest true signal you can get, and it costs a few tool calls instead of one
iteration. `start` returns a job id at once; call `run_directed_wait(job_id)`
until it returns the result, and never end your turn with one unfinished.
Then: `change -> start("failing") -> wait -> read the values -> change`.
`run_rvv_start` / `run_rvv_wait` are the same pair for the Stage 2 regression
(same budget). One cosim run is not a verdict: the trace bridge flips about
12% of runs on an unchanged binary (`titan_runs/nondet/`), so run a failing
RVV test three times — `run_rvv_start(tests, reps=3)` — and judge by majority
before you change RTL for it; a single pass does not clear a test either.
Call `finish` only when `run_directed_start("all")` passes every program it
runs -- the suite grows each round, so judge by "0 failed", never by a
remembered count -- `run_rvv_start("failing")` is clean, and the tree holds
no scaffolding.

Do not modify any test or reference file, and do not run sbt, make or
verilator by hand — `run_directed_start` is that build, correctly configured.
Do not spawn sub-agents.

Everything the loop knows is written into `${AGENT_LOG_DIR}/<run>/`: the
full simulator log of every program, one file each. Read them.

An `ime_cl_` or `ime_clq_` failure means something specific: those programs load the
C tile with `vmtl.v`, run one multiply-accumulate, and then read the C register group
back with an ordinary `vse<SEW>.v` instead of `vmts.v`. They are therefore the only
tests that observe the *register-side* C index directly. A failure there is a
`mat_C_idx` bug -- the spec routes C(i,j) through `tile_reg_idx`, and a linear
`i*N_max + j` gives its exact transpose whenever EMUL_C > 1. It is not a datapath,
accumulator or tile-transfer fault, and the pair tests cannot see it because they
write and read C through the same permutation.
