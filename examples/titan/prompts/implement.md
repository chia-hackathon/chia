Implement round three of the RISC-V Integrated Matrix Extension (Zvvm) in
Saturn: `vwmmacc.vv` (W=2), `v8wmmacc.vv` (W=8), `vmttl.v`, `vmtts.v` -- on
top of `vmmacc.vv`, `vmtl.v`, `vmts.v`, `vqmmacc.vv`, already implemented
and passing.

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

## Rounds two and three

Round two (`vqmmacc.vv`) is done and converged. Round three adds
`vwmmacc.vv` (W=2), `v8wmmacc.vv` (W=8), and the transposing tile pair
`vmttl.v` / `vmtts.v`. Where RTL for earlier instructions is already in the
tree, extend it: every directed test that passes today must still pass. The
spec's SAIL appendix is normative and outranks its prose -- never write
something the SAIL does not say in order to make a test pass.

Three more facts. `vwmmacc.vv`/`v8wmmacc.vv` reuse the round-two datapath
(SEW stays the accumulator width, tiles arrive via `vmtl.v`, only funct6 and
unpack depth W change). `vmttl.v`/`vmtts.v` are `vmtl.v`/`vmts.v` with two
changes: bits 27:26 `0b01` not `0b00`, and offset `(i % linesize) * LD + (i /
linesize)` (div/mod swapped); `flat_idx = tile_reg_idx(...)` is unchanged,
`rs2 = 0` default LD is now `VLEN/(SEW*LAMBDA)` not `LAMBDA*LMUL`. And `vm =
0` on funct6 `0x39`/`0x3a`/`0x3b` is not a don't-care: it decodes
`vfwimmacc.vv`/`vfqimmacc.vv`/`vf8wimmacc.vv` (out of scope) -- raise
illegal-instruction, don't leave it unhandled.

Two facts about `vqmmacc.vv` that save a wrong turn: `vtype.SEW` is the
*accumulator* (C) width, so int8 A/B means SEW=32 -- M, N_max, EMUL_C and the
permissible LAMBDA set are exactly what they are for the non-widening case,
and only A/B change (`row_elems_per_reg = lambda*4`, `epr_A = 4*epr_C`). And
it needs **no new tile-load path**: `vmtl.v` moves SEW-wide storage elements,
and logical element (r,k) lands at `4*flat_storage + k%4`, so the memory image
is the plain row-major int8 panel the existing loader already fetches. The RTL
work is a 4-way-packed int8 MAC datapath feeding the existing 32-bit C tile,
not new addressing.

The matrix FU's `io.stall` must stay `valid`-only. Folding `post_write_stall`
into the shared int/fp issue path once broke every `.vf` RVV test in the S2
regression; the matrix unit does not get to stall that path.

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

## Testing your own work

You have `run_directed_start` / `run_directed_wait`. Use them: get
`vtype.lambda` writable and readable, then `run_directed_start("all")`
immediately — the tests check exactly that before anything else, so it is the
fastest true signal you can get, and it costs a few tool calls instead of one
iteration. `start` returns a job id at once; call `run_directed_wait(job_id)`
until it returns the result, and never end your turn with one unfinished.
Then: `change -> start("failing") -> wait -> read the values -> change`.
`run_rvv_start` / `run_rvv_wait` are the same pair for the Stage 2 regression
(same budget). Call `finish` only when `run_directed_start("all")` is 27/27,
`run_rvv_start("failing")` is clean, and the tree holds no scaffolding.

Do not modify any test or reference file, and do not run sbt, make or
verilator by hand — `run_directed_start` is that build, correctly configured.
Do not spawn sub-agents.

Everything the loop knows is written into `${AGENT_LOG_DIR}/<run>/`: the
full simulator log of every program, one file each. Read them.
