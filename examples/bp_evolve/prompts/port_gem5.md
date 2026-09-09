# Translate a HARCOM predictor into a gem5 BPredUnit

The design below survived Tier 0 and Tier 1 and is being promoted to a gem5 Arm
O3 core, where the pipeline depth CBP-NG holds fixed is swept. Your job is a
**translation, not a redesign**.

## Why this is not a rebuild

* **HARCOM/CBP-NG** calls the predictor for *every instruction*, with a
  block-based, two-level, timed interface; `predict2` may override `predict1`.
* **gem5** calls a `BPredUnit` subclass per branch: `lookup(tid, pc, &bp_history)`
  returns the direction, with `update`, `squash`, `btbUpdate` and
  `uncondBranch` managing history across speculation.

Two things follow. First, the block structure and the override collapse into a
per-branch lookup, and the prediction the core acts on is the **second level's**.
Second — and this has no analogue in Tier 0 or Tier 1 — gem5 speculates, so
your history state must be checkpointed into `bp_history` at `lookup` and
restored on `squash`. A translation that updates global history in place will
work on short traces and drift on long ones.

This is the tier where an overriding predictor can actually be expressed:
neither ChampSim nor stock gem5 has one. If the design's second level is a real
override, say in your rationale how you represented it.

## The check you must pass

Tier-0 MPKI = ${TIER0_MPKI}. gem5 executes binaries rather than traces, so the
workload differs and MPKI will not match exactly — this tier is corroboration,
not the same measurement. Keep table sizes, history lengths, hash functions,
allocation and update policy identical, and let the number land where it lands.

## What to write

A self-contained gem5 branch predictor: the class declaration and definitions,
deriving from `BPredUnit`, with all state as members and a `BranchHistory`
struct carrying whatever `squash` needs to restore. Drop `val`/`reg`/`arr` for
plain integer types and arrays of the same widths and geometry.

## Source to translate — `struct ${STRUCT_NAME}`

```cpp
${HARCOM_SOURCE}
```

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "EvolvedBP",
  "source": "the complete gem5 branch predictor source",
  "rationale": "how the two-level override was represented, and what is checkpointed into bp_history for squash"
}
```
