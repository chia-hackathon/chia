# Set the bounds of the search space

A MAP-Elites search is running over the template parameters of a TAGE branch
predictor. It proposes designs by perturbing those parameters and keeps the
best design in each region of the behaviour space.

You do not propose designs. You decide **how far the search is allowed to go**
on each parameter. Every generation you are shown where the search has got to,
and you may widen or narrow any bound.

## What the search is maximizing

CBP-NG's Voltage-Frequency-Scaled Speedup (VFS). It charges a design for
accuracy (mispredictions per instruction), energy (dynamic energy per
instruction, priced from the arrays and registers the design declares), and
latency (cycles per prediction level). There is no storage budget.

Your objective is the same as the search's: make the best achievable VFS as
high as possible. A bound that is holding the search back costs you score. A
bound that is letting it waste generations in a dead region also costs you
score.

A region is only dead if designs were evaluated there and scored badly. A
region no design has ever been placed in is not dead; nothing is known about
it. The occupancy table below is what tells the two apart, and it is the
reason it is there.

## The parameters

Generation ${GENERATION}. Current bounds:

```
${BOUNDS}
```

What each one means in the rendered predictor:

* `LOGLB` — log2 of the fetch block size in bytes. The predictor predicts one
  block per cycle, and `LOGLINEINST = LOGLB - 2` is log2 of the instructions in
  a block.
* `NUMG` — number of tagged components.
* `LOGG` — log2 entries per tagged component.
* `LOGB` — log2 entries in the bimodal table.
* `TAGW` — tag width in bits. The stored tag is `TAGW - LOGLINEINST` bits wide.
* `GHIST` — longest global history length.
* `LOGP1` — log2 entries in the first-level gshare.
* `GHIST1` — first-level history length.

`LOGP1` and `GHIST1` size the first-level predictor, which exists at Tier 0 but
has no equivalent in the downstream simulators, so changes confined to those two
are invisible below Tier 0.

## Where the search has got to

The archive holds the elites only -- the single best design in each occupied
cell of the behaviour space. It is a few designs out of every one evaluated,
selected for score, so it says where the search succeeded and nothing about
where it looked. Three designs are proposed per generation.

```
${ARCHIVE}
```

## Where the evaluated designs actually landed

Every design scored so far, elites and non-elites, summarised per parameter:
the range of values actually tried, and how many sat exactly on each bound.

A parameter whose designs pile up against a bound is telling you the search
kept trying to go further and could not. A parameter whose designs never come
near its bounds is telling you the mutation operator has never taken it there,
whatever the bounds allow. Neither is visible in the archive above.

```
${OCCUPANCY}
```

## Bounds you have already changed

```
${HISTORY}
```

## What will be rejected

One structural rule is enforced. `TAGW - LOGLINEINST` and `LOGB - LOGLINEINST`
are computed as unsigned integers in the renderer, so if `LOGLB - 2` reaches
`TAGW` or `LOGB` the subtraction wraps and the predictor runs with its tag
comparison silently disabled. Any bound set that permits that combination —
worst case, `LOGLB` at its ceiling with `TAGW` or `LOGB` at their floor — is
rejected whatever its rationale. If you want a higher `LOGLB` ceiling you must
raise the floors it depends on in the same reply.

Nothing else is filtered. Widening a bound is not an admission that the old one
was wrong, and leaving every bound alone is a legitimate answer.

## Reply

Reply with ONLY this JSON object. Use `"changes": []` to leave the bounds as
they are.

```json
{
  "changes": [
    {"param": "NAME", "low": 0, "high": 0, "why": "one sentence for this parameter"}
  ],
  "rationale": "what you are trying to unlock or cut off, and what evidence in the archive says so"
}
```
