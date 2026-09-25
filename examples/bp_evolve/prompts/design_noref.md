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
300 ps later.

## Where the remaining score is: accuracy

The archive's best designs are already at one cycle (or zero), so latency has
little left to give. What they have not closed is accuracy.

**VFS is peaked in throughput, and the front is already sitting on the peak.**
`normalizedEPI` carries a `speedup^3.2` term, so past a certain throughput each
extra block per cycle costs more energy than it earns back. Hold MPKI and EPI
fixed and sweep T through the score itself: at MPKI 5.6 / EPI 362 the maximum
is VFS 0.9875 at **T = 8.96**.
T = 9.4 scores 0.9863, T = 12 scores 0.9418. One generation lost all three of
its designs by raising T from 8.8 to 9.2-9.6 while giving back 0.3 MPKI.

The peak height is a function of **MPKI alone**: at MPKI 4.8 the maximum is
0.9936 (at T = 8.59). No amount of throughput reaches it. So once
you are at one cycle and near T = 9, stop buying throughput -- removing stalls,
lengthening blocks, dropping the true-block rule -- and buy accuracy. At that
operating point (T about 9, EPI 200-500 fJ/inst), measured with the score
itself:

* **1 MPKI less is worth about +0.015 VFS.**
* 100 fJ/inst less is worth about +0.0017 VFS, so one MPKI is worth
  850-930 fJ/inst. Spending energy to buy accuracy pays until well past that.
* The archive's best sit at 5.5-6.0 MPKI, so there is room.

So aim this design at a **lower MPKI at the same latency**. A design that saves energy
and gives back accuracy is almost always a worse design, even if it takes an
empty or weakly held low-energy cell. In your rationale, report the MPKI you
measured against your parent's on the same trace.

## How much to change

A change that pays for itself only after the surrounding structure is rebuilt
loses every time to a safer small one, and the small ones run out. So this is
worth saying plainly: **spending your whole variant on one structural change is
a legitimate use of it**, and a variant that reorganises the prediction cycle
and lands slightly below its parent is more useful to the search than a third
decimal place. Say in your rationale what you rebuilt and what it measured, so
the next design starts from your numbers.

## Building and testing before you reply

There is no CBP-NG checkout in this environment: your parent's source above is
the whole of what you have to work from, and `harcom.hpp`'s rules are the ones
listed above. Anything you find under `/tmp` was left by a different variant's
session: it is not your parent and not a newer design, so do not read it or
start from it. Check your design against the HARCOM rules by hand and output
the JSON object; the harness will compile it, lint it and hand you the
diagnostics if it does not build.

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "a valid C++ identifier, not the parent's name",
  "source": "the complete .hpp source, including the #include lines",
  "rationale": "what you changed, and which cell you are aiming at"
}
```
