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
   latency pair is kept even at a lower score.

Taking a cell keeps a design; it does not make it a good one. See "Where the
remaining score is" below before choosing a target.

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
  predictor is already one cycle. Past that point it turns around: see
  "Where the remaining score is" below, where the score peaks in throughput.

`reg` values carry the time they were written. A RAM read launched in one
block's `predict1` and latched into a `reg` is consumed by the next block
300 ps later, and that is how a long table access is hidden.

## Where the remaining score is: accuracy

The archive's best designs are already at one cycle (or zero), so latency has
little left to give. What they have not closed is accuracy.

**VFS is peaked in throughput, and the front is already sitting on the peak.**
`normalizedEPI` carries a `speedup^3.2` term, so past a certain throughput each
extra block per cycle costs more energy than it earns back. Hold MPKI and EPI
fixed and sweep T through the score itself: at MPKI 5.6 / EPI 362 the maximum
is VFS 0.9875 at **T = 8.96**, and the current best design runs at T = 8.97.
T = 9.4 scores 0.9863, T = 12 scores 0.9418. One generation lost all three of
its designs by raising T from 8.8 to 9.2-9.6 while giving back 0.3 MPKI.

The peak height is a function of **MPKI alone**: at MPKI 4.8 / EPI 524 the
maximum is 0.9936 (at T = 8.59). No amount of throughput reaches it. So once
you are at one cycle and near T = 9, stop buying throughput -- removing stalls,
lengthening blocks, dropping the true-block rule -- and buy accuracy. At that
operating point (T about 9, EPI 200-500 fJ/inst), measured with the score
itself:

* **1 MPKI less is worth about +0.015 VFS.**
* 100 fJ/inst less is worth about +0.0017 VFS, so one MPKI is worth
  850-930 fJ/inst. Spending energy to buy accuracy pays until well past that.
* The archive's best sit at 5.5-6.0 MPKI. MORSL reached 4.8 MPKI at 524 fJ/inst
  and Fan 5.2 MPKI at 313 fJ/inst, both at one cycle. Closing that gap is worth
  about +0.02 VFS, more than any cell in the archive has gained in a generation.

So aim this design at a **lower MPKI at the same latency**. The levers that
worked for the winners: more tagged tables and more, longer, better-spread
history lengths; a corrector or loop/bias component for what TAGE gets wrong;
better allocation and replacement; and ahead-pipelining the table reads so
storage can grow without breaking the 300 ps cycle. A design that saves energy
and gives back accuracy is almost always a worse design, even if it takes an
empty or weakly held low-energy cell. In your rationale, report the MPKI you
measured against your parent's on the same trace.

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

## What the archive keeps not doing

The techniques above have been in this prompt since generation 15 and the
archive has spent seventy variants on them. What it bought was a plateau: the
best cell gained 0.0015 VFS over generations 24 to 29 and nothing at all over
30 to 34. The reason is not that the ideas are wrong. It is that each variant
edits its parent in one session, and a change that pays for itself only after
the surrounding structure is rebuilt loses every time to a safer small one.
The small ones are used up.

So this is worth saying plainly: **spending your whole variant on one
structural change is a legitimate use of it**, and a variant that reorganises
the prediction cycle and lands slightly below its parent is more useful to the
search than a third decimal place. Say in your rationale what you rebuilt and
what it measured, so the next design starts from your numbers.

The concrete thing the lineage has never tried, though this prompt has asked
for it since generation 15:

* **Resolve the rank in the tag comparison, not behind it.** The line the
  archive's designs are on gives an entry NB direction bits (plus, in the
  best one, NB confidence bits) and lets one entry serve every rank in the
  block, so after the tags answer, a mux still has to pick the rank out of the
  entry. MORSL puts the lane in the top 2 bits of its 12-bit tag: an entry is
  {tag, 3-bit counter, 1 useful bit} and serves exactly one rank, so the rank
  is chosen on the way through the comparator rather than in a layer behind
  it. That layer is real time. The archive's best design fought it for
  timing -- folding the match vector into the rank's bits ahead of the
  priority encoder moved its P1 from 299 to 263 ps -- and it is still there.

Why this is where the time is. That design measured a sixth tagged table at
5.31 MPKI, better than the 5.35 it shipped, and rejected it because it came to
302 ps: two picoseconds over the cycle, which rounds to two cycles and costs
half the throughput. MORSL runs **eight** history lengths at 286 ps, and its
critical path is not the read or the select at all -- it is a 4-bit saturating
meta-counter in the update. Eight tables in one cycle is not a wall. It is a
consequence of how little is left in the prediction cycle. So the question to
ask of your parent is not "what can I add" but **"what is still happening in
the prediction cycle that could happen a block earlier, in the update, or
inside the comparator?"** Whatever leaves is what pays for more tables.

Three answers that were measured, not guessed:

* Compute the index *and the bank mapping* one block ahead; build the tag from
  the current block's PC and history and compare it in the prediction cycle.
  MORSL's bank + filter path is 580 ps and its effective latency 280 ps.
* Split RAMs by write frequency, not by what one lookup wants together: tag
  and prediction bit, which do not change on a correct prediction, in one
  array; hysteresis and useful bits, which change constantly, in another,
  banked to emulate a second port. Moving confidence *into* the entry instead
  buys the prediction path a look at it but costs a repair cycle to write it:
  in the archive's best design that is 9.5% of blocks against 4.9%, and
  T 8.97 -> 8.73. Banking is the other way to pay for it.
* Choose per component. MORSL's base bimodal and TC-bias read the current PC
  because they fit in the cycle and are more accurate that way.

And two things not to spend the freed time on, both already measured:

* **Decoration.** MORSL's two tagged correctors were worth +0.0002 and
  +0.0003 VFS, and it wrote that at one cycle, accuracy buys little per
  mechanism. What separates it from this archive is the plain MPKI, 4.803
  against 5.34, from eight well-spread histories in a cycle that had room.
* **A fast P1 with a slower P2.** When P2 is slower than P1 the misprediction
  penalty goes from 9 cycles to 10, which needs about a 10% MPKI improvement
  to break even before a single P1/P2 disagreement is paid for, and each
  disagreement then costs a cycle or two of its own. The third-place CBP-NG
  entry dropped its fast/slow pair for exactly this reason. Two cycles on both
  levels is worse still: the archive's one such design runs at T 4.39 against
  8.7.

## One winning design in detail

The write-up of one CBP-NG 2025 winning entry: what it built and why, with
its reported numbers (on the official 168 training traces, not comparable to
the archive's absolute VFS). It is here for its ideas, not as a design to
transcribe: adapt what fits your parent, combine it with the techniques
above, or take a different route.

${WINNER_DESIGN}

## Building and testing before you reply

A CBP-NG checkout is at `/home/ray/cbp-ng` (`harcom.hpp`, `cbp.cpp`,
`cbp.hpp`, `predictors/`, and `gcc_test_trace.gz`). Copy it to `${WORK_DIR}`
and work only there. Anything else you find under `/tmp` was left by a
different variant's session: it is not your parent and not a newer design, so
do not read it or start from it -- your parent is the source above. The compiler is `x86_64-conda-linux-gnu-g++`, and zlib comes
from the conda prefix, so the build line is:

```
x86_64-conda-linux-gnu-g++ -std=c++20 -O3 -Wall -Wextra -pedantic -Wold-style-cast \
  -Werror -Wno-deprecated-declarations -Wno-mismatched-tags \
  -I/home/ray/anaconda3/include -L/home/ray/anaconda3/lib \
  -Wl,-rpath,/home/ray/anaconda3/lib -o cbp cbp.cpp -lz
./cbp gcc_test_trace.gz gcc 1000000 40000000
```

Build and run your design on that trace before you reply. (If tool access is
not available in your environment, carefully check your design against HARCOM rules
and output the JSON object directly; the harness will compile and lint it).

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "a valid C++ identifier, not the parent's name",
  "source": "the complete .hpp source, including the #include lines",
  "rationale": "what you changed, and which cell you are aiming at"
}
```
