# Choose the template parameters for one branch predictor

A MAP-Elites search is running over the template parameters of a TAGE branch
predictor. You are given one parent design and you return the parameters for
one child. You do not write or edit any source: the predictor is rendered from
a fixed template, so the algorithm is identical for every design in this sweep
and only the eight numbers below differ.

## What you are being scored on

CBP-NG's Voltage-Frequency-Scaled Speedup (VFS) charges the design for three
things together:

* **accuracy** — mispredictions per instruction,
* **energy** — dynamic energy per instruction, priced from the arrays and
  registers the rendered design declares,
* **latency** — cycles at each of the two prediction levels.

There is **no storage budget**. Bigger tables are always allowed, and are
usually the boring answer: they buy accuracy with energy and latency, and the
score already knows that.

## How a design is kept

The archive holds one design per region of the behaviour space, where a region
is (energy band, P1 latency, P2 latency). Your child is kept if it either

1. beats the design currently holding its cell on VFS, or
2. lands in a cell nobody occupies — a different energy band or a different
   latency pair is worth as much as a higher score.

So a child that is slightly worse than its parent but lands somewhere new is a
better move than one that is slightly better in the same cell.

## The parameters

Generation ${GENERATION}. Each value must be an integer inside its bound; a
value outside is clipped, which wastes the proposal.

```
${BOUNDS}
```

What each one does in the rendered predictor:

* `LOGLB` — log2 of the fetch block size in bytes. One block is predicted per
  cycle, and `LOGLINEINST = LOGLB - 2` is log2 of the instructions in a block.
* `NUMG` — number of tagged components.
* `LOGG` — log2 entries per tagged component.
* `LOGB` — log2 entries in the bimodal table.
* `TAGW` — tag width in bits. The stored tag is `TAGW - LOGLINEINST` bits.
* `GHIST` — longest global history length.
* `LOGP1` — log2 entries in the first-level gshare.
* `GHIST1` — first-level history length.

`LOGP1` and `GHIST1` size the first-level predictor. It exists at Tier 0 but
has no equivalent in the downstream simulators, so a child differing only in
those two is invisible below Tier 0.

## The parent

```
${PARENT}
```

## The archive

```
${ARCHIVE}
```

## What happened to the last child of this parent

```
${FEEDBACK}
```

## Reply

Reply with ONLY this JSON object, giving all eight parameters:

```json
{
  "params": {"LOGLB": 0, "NUMG": 0, "LOGG": 0, "LOGB": 0,
             "TAGW": 0, "GHIST": 0, "LOGP1": 0, "GHIST1": 0},
  "rationale": "what you are moving, and which cell you expect to land in"
}
```
