# Llama-3.2-1B round10 projection — lm_head overlap-extrapolation test (negative result)

Generated against `measured_cycles_round10.json` (new in this task), which is
`measured_cycles_round9.json` plus one new documentation-only entry,
`llama-lmhead-fused-n1` (`baseline_cycles=648292`, `best_cycles=648292`,
`roofline_cycles=532232` — baseline never beaten). No existing entry was
touched; every other kernel is carried forward from round9 byte-for-byte.

`llama-lmhead-fused-n1` is not one of the kernel types in `loop/llama_model.py`'s
`Op` taxonomy (same as `llama-layer-fused-n1` in rounds 7-9) — it is a
whole-fused-region measurement used only to reason about the overlap model
below, and is **not** read as a per-op override by `llama_project.py`'s
`MeasuredDB.load`. This round did not re-run `llama_project.py`'s decode
projection (no new per-op measurement exists to feed it); this document is
about `--overlap`'s modelling assumption, not a new tok/s number.

## Round10 summary

Single run: `llama-lmhead-fused-n1`, run_id `20260917-155834-2887`, 4
iterations, wall-clock ~2026-09-17 15:58 -> ~2026-09-18 03:36 Taipei
(overnight), cost $2.3956.

| iter | cycles | note |
|---|---:|---|
| baseline | 648,292 | strictly sequential (GEMV -> rmsnorm -> quantise -> argmax, fenced) |
| 1 | 676,825 | denominator-only probe, schedule unchanged |
| 2 | 651,726 | fence moved + 80-unit interleaved schedule (the actual optimization attempt) |
| 3 | 689,999 | held-cycles probe on top of iter2's interleaved schedule |
| 4 | 698,428 | spin-wait scan probe, reverted to baseline serialised schedule |
| **best** | **648,292** | **= baseline; never beaten. Speedup 1.000x (+0.0%)** |
| roofline | 532,232 | best/roofline = 648,292/532,232 = 1.218x |

This is a clean negative result, not a failed search: `loop/kernels.py`'s own
notes for this kernel (`_LMHEAD_FUSED_NOTES`) state up front that "a measured
ZERO is a publishable answer" and predicted exactly this outcome ("exposure
ratio ~1.0 ... **This is the outcome the arithmetic predicts**").

## (a) lm_head's exposure ratio — the 13,785 / 12,772 / 27,151 numbers, reconciled

The task brief for this round quotes a headline "total overlappable Saturn
work is only 13,785 cycles" for lm_head. The two numbers actually measured
on-chip this round are:

* **iter1 (phase split, schedule untouched):** `stream=634,821 rms=7,296
  quant=2,496 argmax=2,980`, PROBE `total=647,593`. Excluding `stream`,
  serialised Saturn work = `rms+quant+argmax = 7,296+2,496+2,980 = 12,772`
  cycles, i.e. **12,772**, not 13,785. `12,772/647,593 = 1.97%` of the
  iteration total.
* **iter3 (held-cycles probe on the interleaved schedule):**
  `held_rms1=5,984 held_rms2=8,990 held_quant=6,552 held_amax1=2,093
  held_amax2=3,532`, sum = **27,151**.

**Which number this document uses, and why:** we use the iter1 figure,
**12,772 cycles (serialised Saturn, cold)**, as the "total overlappable Saturn
work" for lm_head, not the round brief's 13,785 and not iter1's own
`total=647,593` minus the *warm* stream reference (634,507, which would give
13,086 — also close but not the brief's number either). 12,772 is the
one actually printed by the kernel as the sum of the three Saturn phases
under the SAME cold-baseline methodology `llama-layer-fused-n1` used for its
own 28,372-cycle "serialised Saturn" denominator, so it is the only
apples-to-apples number to compare against that kernel's 0.611. The 13,785
figure in the task brief could not be reproduced from any single PROBE line
in `simlog_01.txt`-`simlog_04.txt`, `kernel_01.h`-`kernel_04.h`, or
`agent_01.txt`-`agent_04.txt`; it is closest to `12,772 + ~1,000` (a
cold-vs-warm stream delta or a rounding of a different partition), but no
exact derivation was found. **This is flagged as unreconciled** — treat
12,772 as the number backed by a citable PROBE line, and 13,785 as an
input given by the round brief that this document was not able to verify
against the raw simulator logs.

The 27,151-cycle `held_*` sum from iter3 is a **different measurement of a
different (interleaved) schedule**, not a like-for-like alternative to
12,772: it sums how long each of 80 Saturn units held/blocked the host once
the work was split up and issued inside the weight stream, which turned out
to be *inflated* relative to the serial cost (the phases are 16 K-blocks =
530 KiB apart, which exceeds the 512 KiB L2, so `X`/`xn`/`LG` are evicted and
re-fetched cold between visits — see `kernel_04.h`'s HYPOTHESIS and
`agent_04.txt`). It is evidence for *why* the interleaving didn't help, not
a corrected version of the 12,772-cycle denominator.

Either way — 12,772 (serialised) or 27,151 (interleaved, inflated) — lm_head's
overlappable Saturn work is a small, single-digit-to-low-double-digit
percentage of its own iteration total, versus `llama-layer-fused-n1`'s
Saturn share of **28,372/206,304 = 13.8%** of its own baseline (see
`loop/kernels.py` line ~4340 and `llama_project.py`'s `OVERLAP_NOTE_MD`).

## (b) Measured net benefit = 0 → exposure ratio ≈ 1.0

`llama-layer-fused-n1` achieved `exposure ratio = 17,346/28,372 = 0.611`
(38.9% of its serialised Saturn cost was hidden in the Gemmini DMA shadow).
lm_head's best achieved cycle count is the baseline itself — **0 cycles of
net benefit from any interleaving attempted this round** (iter2, the one
schedule that actually interleaved Saturn work into the stream, regressed to
651,726, worse than the printf-free serial baseline of 647,593). By the same
definition (`exposure = achieved_total - stream`, `exposure ratio =
exposure / serialised Saturn`), lm_head's exposure ratio is effectively
**~1.0**: essentially none of its Saturn-side work could be hidden behind
the Gemmini weight stream, in sharp contrast to layer-fused's 61.1%
(`DEFAULT_OVERLAP_FACTOR` in `loop/llama_project.py`).

## (c) What `--overlap` / `DEFAULT_OVERLAP_FACTOR` actually models

Read from `loop/llama_project.py`:

* `DEFAULT_OVERLAP_FACTOR = 17_346.0 / 28_372.0` (≈0.6114), computed and
  commented in place (lines ~100-122) as "the measured overlap efficiency"
  of `llama-layer-fused-n1`: `strictly sequential baseline 206,304 ->
  best overlapped 179,033`, of which `161,687` is "weight stream alone,
  Saturn serialised out" and `28,372` is "Saturn phases alone, serialised";
  `17,346` cycles of that Saturn cost remained exposed in the best kernel.
* `--overlap[=F]` (argparse help, same file, ~line 916): "model
  Gemmini/Saturn overlap: saturn operators keep only FACTOR of their cycles
  (default 0.611, the measured exposure ratio of llama-layer-fused-n1). OFF
  by default." It multiplies the cycles of every operator costed on
  `device="saturn"` by `F` and treats the removed cycles as hidden inside
  the Gemmini weight-DMA shadow of the *same layer*, capped so the model
  degenerates to `gemmini + F x saturn >= max(gemmini, saturn)`
  (`OVERLAP_NOTE_MD`, ~line 124-168).
* Its own docstring already lists the LM head as extrapolation-limit #2:
  "The LM head (about a fifth of decode cycles) streams 4 MiB of weights
  with no attention work to hide underneath it; its Saturn partner ops are
  only rmsnorm-sized. Applying `F` there is the weakest part of the model."
  This round supplies the first actual on-chip measurement confirming that
  stated weakness: exposure ratio ≈1.0 for lm_head, not 0.611.

`loop/kernels.py`'s own notes for this kernel (`_LMHEAD_FUSED_NOTES`,
~line 4462-4798) framed this exact experiment beforehand and said an
exposure ratio near 1.0 would mean "0.611 is a property of attention-shaped
companion work and does NOT extrapolate. `--overlap` must be restricted to
the attention phases, and the LM head — 20.5% of decode — gets no credit."
Round10's measurement bears that prediction out.

## (d) Recommendation for `loop/llama_project.py --overlap`

**No code was changed for this task — the following is a recommendation
only, to be acted on in a future round.**

A single global `DEFAULT_OVERLAP_FACTOR = 0.611` applied to every
Saturn-costed operator model-wide is now demonstrably wrong for at least one
major kernel family: it predicts ~61% of lm_head's Saturn work should hide
behind its 4 MiB weight stream, and the measured answer (this round,
`llama-lmhead-fused-n1`, run `20260917-155834-2887`) is ~0%. Recommend one of
the following, in order of preference:

1. **Make the overlap factor per-op-type / per-kernel-family, not a single
   constant.** Key it by the same category the companion Saturn work falls
   into (e.g. `layer-fused` / attention-shaped companion work -> 0.611;
   `lmhead-fused` / no-attention companion work -> ~1.0, i.e. no discount),
   and calibrate each family's factor from its own measured fused-kernel
   round the way `llama-layer-fused-n1` and now `llama-lmhead-fused-n1` were
   measured. `--overlap` would then look up the right `F` per operator by
   which fused-kernel family it belongs to, rather than applying one `F` to
   every `device="saturn"` operator uniformly.
2. **If per-op calibration is not feasible soon:** explicitly document
   `--overlap`'s current single-constant model as unreliable/not
   generalizable outside `llama-layer-fused-n1`, and require any new fused
   kernel family (lm_head, or whatever comes after it) to get its own
   measured overlap number — even if that number is "0.611 does not apply,
   use 1.0 for these ops" — before any projection is allowed to apply
   `--overlap` to that family's operators. In practice this means gating
   `--overlap`'s discount so it is applied ONLY to operators known to have
   attention-shaped companion work, and leaving lm_head-classified operators
   at the conservative (undiscounted) sequential cost until a dedicated
   factor is measured and wired in.

Either way, `--overlap`'s existing `OVERLAP_NOTE_MD` extrapolation-limits
section (limit #2) should be updated to say the LM head case is no longer
hypothetical — it is now measured, exposure ratio ≈1.0, `0.611` does not
apply — once someone is authorized to edit `loop/llama_project.py`.

## Reconciling 13,785 vs 12,772 (2026-09-18)

Both numbers are correct; they measure different things and the round-10 brief
conflated them.

| quantity | value | source |
|---|---:|---|
| fused baseline | 648,292 | run 20260917-155834-2887 baseline |
| warm-state GEMV-only best, same stream | 634,507 | `llama-q8-gemv-gemmini-lmhead` round-4 winner |
| difference (pre-round upper bound on overlap) | **13,785** | arithmetic, `round10-prep.md` |
| weight stream measured *inside* the fused kernel | 634,821 | `PROBE stream=`, simlog iter1 |
| Saturn phases alone (rms + quant + argmax) | **12,772** | `PROBE rms/quant/argmax`, simlog iter1 |
| probe total | 647,593 | `PROBE total=`, simlog iter1 |

The 13,785 gap is the Saturn phases (12,772) plus ~700 cycles of cold-L2 penalty
and fence, and against the baseline rather than the probe build
(648,292 - 634,821 = 13,471). So 12,772 is what could in principle be hidden,
13,785 is the ceiling the pre-round arithmetic placed on the whole round, and
the measured outcome was 0 of either.
