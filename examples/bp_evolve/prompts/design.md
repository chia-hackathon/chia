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

## Where the interesting designs are

The two-level `predict1`/`predict2` interface exists so a fast level can supply
throughput while a slower level overrides it. That trade — spending P2 latency
to buy accuracy — is the least explored part of the space and the part where
the score's assumptions are least certain. Consider also: what sits in each
prediction stage, allocation and update policy, history length distribution,
a corrector or loop component, and whether the second level is worth having at
all for this design.

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "a valid C++ identifier, not the parent's name",
  "source": "the complete .hpp source, including the #include lines",
  "rationale": "what you changed, and which cell you are aiming at"
}
```
