# Repair a rejected predictor

Your design was rejected before it could be scored. This is repair round
${ROUND} of ${MAX_ROUNDS}; after that the design is dropped.

## What was wrong

```
${DIAGNOSTICS}
```

## Reminders

* HARCOM values are opaque — no `operator bool`, so no `if`, `while` or `?:`
  on one. `select(cond, a, b)` muxes; `execute_if(cond, [&]{...})` guards.
* Every template parameter needs a default; the build instantiates
  `${STRUCT_NAME}<>`.
* All six of `predict1`, `reuse_predict1`, `predict2`, `reuse_predict2`,
  `update_condbr`, `update_cycle` must be defined, or the struct is abstract.
* Warnings are errors. Silencing one with a `#pragma` withdraws the design.

Fix the cause, not the symptom. If a diagnostic points into `harcom.hpp`, the
mistake is still in your source — usually an opaque value used where a plain
integer was expected, or the reverse.

## The source

```cpp
${SOURCE}
```

## Reply

Reply with ONLY this JSON object:

```json
{
  "struct_name": "${STRUCT_NAME}",
  "source": "the complete corrected .hpp source",
  "rationale": "what the mistake was"
}
```
