# Design a branch predictor

You are editing a HARCOM branch predictor for CBP-NG. Return one new design.

## What you are being scored on

CBP-NG's Voltage-Frequency-Scaled Speedup (VFS) charges your design for three
things together:

* **accuracy** — mispredictions per instruction,
* **energy** — the dynamic energy per instruction HARCOM prices from the
  registers, arrays and intermediate values your design declares,
* **latency** — how many cycles each of your two prediction levels takes.

There is **no storage budget**. Making tables bigger is always allowed, and is
almost always the boring answer: it buys accuracy with energy and latency, and
the score already knows that. The archive below is what actually decides
whether your design is kept, and it does not reward being large.

## The archive

Designs are kept one per region of the space, where a region is
(energy band, P1 latency, P2 latency). You take a cell by being the best
design in it. That means there are two ways to succeed:

1. beat the design currently holding a cell, on VFS, at the same cost, or
2. land in a cell nobody occupies — a different energy band or a different
   latency pair is worth as much as a higher score.

Current archive:

${ARCHIVE}

## Your parent

Generation ${GENERATION}. Parent's measured behaviour:

```json
${PARENT_SUMMARY}
```

Feedback from the last evaluation in this lineage:

${FEEDBACK}

Parent source:

```cpp
${PARENT_SOURCE}
```

## HARCOM rules that are not negotiable

`val`, `reg`, `arr` and `ram` are **opaque**. You cannot read their values, so:

* `if (some_val)` does not compile. Use `select(cond, a, b)` to mux between two
  values, or `execute_if(cond, [&]{ ... })` to guard a side effect. Both take a
  `val<1>` condition.
* The ternary `?:` is control flow and is equally forbidden on an opaque value.
* Never use `harcom_superuser`, `reinterpret_cast`, `memcpy` or `std::bit_cast`
  on a HARCOM type. These reach around the cost model, and a design that uses
  one is withdrawn from the search rather than repaired.
* Never write to `stdout` — the harness parses the counter line CBP-NG prints
  there. `std::cerr` is fine.
* Every compiler warning is an error (`-Wall -Wextra -pedantic
  -Wold-style-cast -Werror`).

Your struct must derive from `predictor` and define all six of `predict1`,
`reuse_predict1`, `predict2`, `reuse_predict2`, `update_condbr`, `update_cycle`.
Every template parameter needs a default: the build instantiates
`YourStruct<>`.

## How the score works out in practice

A block is one cycle's prediction: it ends on a taken branch, on a level-2
misprediction, or when you call `reuse_prediction(0)`. Throughput is roughly
instructions per block divided by the cycles each block costs, and a level's
latency is rounded **up** to whole 300 ps cycles over the worst case of the
whole run. The misprediction penalty is `9 + p2 - max(1, min(p1, p2))` cycles.
Two consequences:

* A predictor whose `predict1` and `predict2` return the same answer at
  <= 1 cycle pays no level-1/level-2 disagreement penalty and issues one block
  per cycle. At 1.1 cycles it rounds to 2 and loses most of the gain.
* Measured on the published entries, a 10% gain in throughput, MPI and EPI
  is worth about +2.4%, +1.4% and +0.2% VFS. Throughput dominates until the
  predictor is already one cycle; past that point accuracy and energy matter.

`reg` values carry the time they were written. A RAM read launched in one
block's `predict1` and latched into a `reg` is consumed by the next block
300 ps later, and that is how a long table access is hidden.

## What the CBP-NG 2025 winners did

All three winning entries scored 12-14% above the stock `tage` your lineage
descends from (on the official trace set; the archive's absolute VFS numbers
come from a different trace sample and do not compare directly). They reached a **whole predictor in one cycle** by
*ahead-pipelining*, not by tuning parameters:

* **Index early, select late.** Compute the table index in the previous
  block from the previous block's PC and history; read the RAM then; latch
  the entries into `reg`s. In the current block only compare tags and mux.
  Tagged tables resolve the unknown exit path of the previous block with the
  tag (built from the current PC and history), so no extra candidates are
  read. A tagless table instead reads all exit-path candidates as banks and
  selects one late -- the reference below does exactly this.
* **Predict by rank, not by slot.** Up to 4 conditional branches per block,
  with the rank folded into the tag. Blocks run to a taken branch, the 4th
  branch, or a 256-instruction region, and may cross the line boundary.
* **"True block" rule.** If a block ended early only because a branch
  predicted taken was not taken, do not advance the ahead pipeline or the
  history; reuse the latched predictions.
* **Keep fast parts non-ahead.** A small bimodal base can use the current PC
  directly if it fits under 300 ps.
* **Declared size is wire cost.** Every declared array adds latency and
  energy through wiring even if unused; compile dead tables out
  (`std::conditional_t`). Fewer, narrower RAMs are faster.

Energy tricks that paid for themselves:

* Few tagged tables (the lowest-EPI entry, 313 fJ/inst, used two), narrow
  entries (tag + direction bit in one RAM, hysteresis/useful bits in another).
* Read hysteresis only on a misprediction; on a correct prediction write it
  to a fixed strong value without reading it. Write prediction/tag RAMs only
  on a misprediction with weak hysteresis.
* A small Bloom-style filter that skips the long-history banks when no entry
  can hit there (filtered ~36% of accesses, +0.02 MPKI).
* Summation-free override tables instead of a statistical corrector: a
  corrector does not fit in one cycle.

Pitfalls: a table updated on correct predictions needs forwarding or a stall
(`need_extra_cycle`) -- a RAM allows one read *or* one write per cycle and a
`reg` one write; accuracy-only tweaks were worth almost nothing in VFS; a
non-ahead TAGE runs 500-580 ps and cannot be one cycle.

You do not have to reproduce these designs, and a different route that
reaches one cycle, or a design deliberately elsewhere in the archive, is just
as welcome. But a change that leaves the predictor at two or three cycles is
competing for the last few percent while the first ten are still on the
table.

Reference: the CBP-NG repository's ahead-pipelined N-branch gshare, a
complete, compiling example of the idea in HARCOM (it does not derive from
your parent; do not copy its struct name):

```cpp
${REFERENCE_SOURCE}
```

## One winning design in detail

The write-up of one CBP-NG 2025 winning entry: what it built and why, with
its reported numbers (on the official 168 training traces, not comparable to
the archive's absolute VFS). It is here for its ideas, not as a design to
transcribe: adapt what fits your parent, combine it with the techniques
above, or take a different route.

${WINNER_DESIGN}

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "a valid C++ identifier, not the parent's name",
  "source": "the complete .hpp source, including the #include lines",
  "rationale": "what you changed, and which cell you are aiming at"
}
```
