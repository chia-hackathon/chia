# Translate a HARCOM predictor into a ChampSim branch module

The design below scored well on CBP-NG and is being promoted to ChampSim, which
models a full core. Your job is a **translation, not a redesign**.

## Why this is not a rebuild

The two simulators call a predictor differently:

* **HARCOM/CBP-NG** calls the predictor for *every instruction*, with a
  block-based, two-level, timed interface: `predict1` and `predict2` are called
  at the start of a block, and the second level may override the first.
* **ChampSim** calls a branch predictor *once per branch*:
  `predict_branch(ip)` returns taken/not-taken, and
  `last_branch_result(ip, target, taken, type)` updates.

So the block structure and the two-level override have to be collapsed into a
per-branch interface. The prediction the core actually acts on is the **second
level's**, since that is what CBP-NG lets override; a translation that returns
the first level's prediction will mispredict differently and fail the check
below.

## The check you must pass

The same predictor on the same branch stream must mispredict the same way. Your
translation is accepted only if its ChampSim MPKI lands within ${TOLERANCE} of
the Tier-0 measurement:

    Tier-0 MPKI = ${TIER0_MPKI}

A wider gap means the translation is wrong, and it is rejected — because a
mistranslation looks exactly like the cost-model disagreement this whole
experiment is trying to measure, and the two must not be confused.

Do not tune the design to hit the number. Translate it faithfully; the number
follows.

## What to write

A ChampSim branch module: a header defining `class evolved_bp` with the module
methods, self-contained, with all state as members. HARCOM's energy and latency
model does not exist here, so drop `val`/`reg`/`arr` and use plain integer types
and arrays with the same widths and geometry. Keep table sizes, history lengths,
hash functions, allocation policy and update policy identical.

## Source to translate — `struct ${STRUCT_NAME}`

```cpp
${HARCOM_SOURCE}
```

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "evolved_bp",
  "source": "the complete ChampSim branch module header",
  "rationale": "how the block-based two-level interface was collapsed to per-branch, and anything that could not be preserved exactly"
}
```
