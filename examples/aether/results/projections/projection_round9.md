# Llama-3.2-1B round9 projection — layer-fused overlap update (continued)

Generated against `measured_cycles_round9.json` (new in this task), which is
`measured_cycles_round8.json` with the `llama-layer-fused-n1` entry's
`cycles`/`best_cycles` updated from round8's 179,585 to round9's improved
**179,033** (baseline and roofline for that entry are unchanged: 206,304 and
140,680 respectively). No other entry was touched — round9 produced no new
measurement for any of the other 14 kernel entries, so they are carried
forward from round8 byte-for-byte (confirmed: `diff` of the two projection
runs below is identical except for the `--measured` path in the header line).

All commands use `--device int8_gemv=gemmini --device lm_head_gemv=gemmini`,
same convention as round2-round8, at both 1.00 GHz and 500 MHz. Raw command
output for every run below is saved under
`out/llama-profile/proj_round9_decode_cold567_{1ghz,500mhz}.txt`.

```
python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round9.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --gemv-bytes-per-cycle 5.67 --clock-ghz 1.0

python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round9.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --gemv-bytes-per-cycle 5.67 --clock-ghz 0.5
```

## Round9 summary

Single run this round: `llama-layer-fused-n1`, run_id `20260916-195958-36b4`,
4 iterations, `--seed best` from round8's 179,585 (run `20260915-060820-a93a`
iter 1). Wall-clock ~2026-09-17 04:00-09:4x Taipei (same day), cost $6.4724.

| iter | cycles | note |
|---|---:|---|
| seed (179,585) | 179,585 | round8's best |
| 1 | 185,374 | regression (Gemmini-DMA operand warming; didn't help) |
| 2 | 180,107 | close, not a new best (half-grain units; exposure unchanged) |
| 3 | 183,430 | probe-only iteration (rdcycle instrumentation overhead), regression |
| **4** | **179,033** | **new overall best** |

| | cycles | vs. baseline | vs. round8 best |
|---|---:|---:|---:|
| baseline (sequential, unfused) | 206,304 | — | — |
| round8 best | 179,585 | -12.95% | — |
| **round9 best** | **179,033** | **-13.22%** | **-0.31%** |
| roofline (ideal fully-overlapped floor) | 140,680 | 179,033/140,680 = 1.273x | (round8: 1.277x) |

`(206304 - 179033) / 206304 = 0.132188` → **13.2% reduction** from overlap
(rounded from the exact 13.2188%), up from round8's 12.95%/13.0%. The
179,033-cycle point nudges further toward the 140,680-cycle roofline
(1.273x vs. round8's 1.277x) — a small, incremental gain (552 cycles) on top
of round8's larger jump, consistent with the loop approaching a local floor
for this specific technique on this kernel (see the new finding below for
why).

## Cold-GEMV decode with overlap — **RETRACTED uniform extrapolation, replaced with `--overlap`**

> **Retraction.** This section previously took the 13.2% cycle reduction
> measured on the single `llama-layer-fused-n1` kernel and applied it
> *uniformly* to the entire cold decode step's cycle count
> (`226,300,507 x (1 - 0.132188) = 196,386,200` -> 5.092 tok/s @ 1 GHz /
> 2.546 tok/s @ 500 MHz, +15.2%). **That extrapolation is wrong and is
> withdrawn.** It silently assumed every kernel in decode overlaps with
> Gemmini's weight stream at the same 13.2%/61.1%-exposure ratio as the one
> hand-fused, hand-tuned kernel that produced it — an assumption
> `loop/llama_project.py`'s own `OVERLAP_NOTE_MD` explicitly warns against
> (point 1: "One shape, one layer... the projection applies one scalar to
> all of them"; point 2: the LM head, ~20.5% of decode cycles, streams 4 MiB
> with *no* attention work to hide underneath it at all). The correct,
> measured way to apply this overlap to the whole-decode projection is the
> tool's own `--overlap` flag (`DEFAULT_OVERLAP_FACTOR = 17,346/28,372 =
> 0.611`, i.e. Saturn-costed operators keep 61.1% of their cycles, capped by
> the Gemmini cycles available to hide them under), run below with two
device-assignment scenarios, not a hand-applied uniform multiplier.

`--overlap` only discounts the cycles of operators whose `device` is **not**
`gemmini` (see `_apply_overlap` in `loop/llama_project.py`). Which decode
operators count as "saturn" for that purpose depends on device assignment:
`int8_gemv`/`lm_head_gemv` are pinned to `gemmini` (the round2-round9
convention, `--device int8_gemv=gemmini --device lm_head_gemv=gemmini`) in
both scenarios below; the two scenarios differ only in where `attn_scores`/
`attn_pv` are assigned:

- **Scenario A — attention pinned to `gemmini`** (the literal `DEVICE_MAP`
  default in `loop/llama_project.py`, i.e. attention ops are *not* eligible
  for the overlap discount): only the small elementwise ops (rmsnorm, rope,
  softmax, silu_mul, add, embedding) get discounted.
- **Scenario B — attention pinned to `saturn`**
  (`--device attn_scores=saturn --device attn_pv=saturn`, matching the
  actual device split the `llama-layer-fused-n1` kernel measured the 0.611
  factor on): attention's ~1.9%/1.1% MAC share also gets discounted.

Commands run (all against `measured_cycles_round9.json`, cold 5.67 B/cycle
GEMV rate, both clocks):

```
python -m loop.llama_project --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round9.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --gemv-bytes-per-cycle 5.67 --clock-ghz 1.0   # and --clock-ghz 0.5
# no attn override needed for Scenario A: gemmini is DEVICE_MAP's own default
# for attn_scores/attn_pv; add --overlap for the "after" column

python -m loop.llama_project --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round9.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --device attn_scores=saturn --device attn_pv=saturn \
  --gemv-bytes-per-cycle 5.67 --overlap --clock-ghz 1.0   # and --clock-ghz 0.5
```

| scenario | clock | before overlap | after `--overlap` | reduction | tok/s gain |
|---|---|---:|---:|---:|---:|
| A (attn on gemmini) | 1.00 GHz | 226.30M cyc/tok -> **4.419 tok/s** | 225.65M cyc/tok -> **4.432 tok/s** | 0.29% | +0.29% |
| A (attn on gemmini) | 500 MHz  | 226.30M cyc/tok -> **2.209 tok/s** | 225.65M cyc/tok -> **2.216 tok/s** | 0.29% | +0.29% |
| B (attn on saturn)  | 1.00 GHz | 226.30M cyc/tok -> **4.419 tok/s** | 223.05M cyc/tok -> **4.483 tok/s** | 1.43% | +1.45% |
| B (attn on saturn)  | 500 MHz  | 226.30M cyc/tok -> **2.209 tok/s** | 223.05M cyc/tok -> **2.242 tok/s** | 1.43% | +1.45% |

(Exact `Projection.cycles_per_token`: baseline 226,300,507; Scenario A
overlapped 225,649,300; Scenario B overlapped 223,053,277. Tok/s at each
clock is `clock_hz / cycles_per_token`, so 500 MHz is exactly half of 1 GHz.)

**N=1 tokens/sec under cold GEMV: overlap is worth 0.3-1.4% end to end, not
15.2%.** The gap between this and the 13.2% single-kernel number is exactly
the point `OVERLAP_NOTE_MD` makes: the fused kernel's *own* Saturn device
share (28,372 / 206,304 = 13.8% of that one kernel's cycles) is an order of
magnitude larger than the whole decode step's actual Saturn device share
(0.7-2.2% depending on where attention is assigned — see the per-device
table in any `--no-ops`-free run of the tool). Applying a ratio measured on
a 13.8%-Saturn kernel to a 0.7-2.2%-Saturn workload and expecting the same
percentage-of-total win is the arithmetic error the retracted extrapolation
made. `projection_round8.md`'s equivalent 14.9%/5.076-tok/s number carries
the identical error and is retracted there too.

## (corrected) Round9 finding: `sat_done` timing is geometry, not proof of hiding

> **Correction.** This section originally read `sat_done` finishing at
> 55-58% of `gemv_end` as evidence Saturn's compute is "now almost entirely
> overlapped/hidden." That reading is wrong: `sat_done < gemv_end` is a
> **geometric artifact** of how the kernel calls Saturn during the mvin
> stream, not a measurement of how much Saturn cost survives as exposed wall
> time. The stream-rate breakdown below (not `sat_done`/`gemv_end`) is what
> actually shows the exposure.

Round9's iter1-3 kernels added phase-split probes that round8 did not have:
`sat_done` (cycle at which Saturn's own attention/elementwise compute
finishes) vs. `gemv_end` (cycle at which the whole kernel, including
Gemmini's GEMV weight streaming, finishes):

| iter | sat_done | gemv_end | sat_done / gemv_end |
|---|---:|---:|---:|
| 1 | 106,708 | 185,332 | 57.6% |
| 2 | 99,956 | 179,890 | 55.6% |
| 3 | 104,365 | 183,228 | 57.0% |

**Why `sat_done` finishing early is geometric, not overlap.** The 1 MiB
weight stream is 4,096 `mvin` commands (1 MiB / (4 rows x 64 B/row)), and
Saturn's unit is called once every 16 `mvin`s — but only the first ~130 of
those calls do real attention/elementwise work; the rest are no-ops once
Saturn's fixed amount of work (one head's QK^T/softmax/PV) is exhausted.
Saturn's calls are therefore **necessarily** finished by roughly the
halfway point of the mvin stream regardless of how well its work is
overlapped — `sat_done` at ~56% of `gemv_end` is exactly what you'd see even
if none of Saturn's cost were successfully hidden, because it measures
*when Saturn runs out of work to do*, not *how much of Saturn's cost is
still exposed on the critical path*.

The actual exposure is visible only in the **mvin rate**, from the
round9-seed probe (`out/loop/20260915-060820-a93a/simlog_02.txt`, iter 2):
the first 2,080 mvins (while Saturn work is in flight) average **50.18
c/mvin**; the remaining 2,016 mvins (Saturn idle) average **39.12 c/mvin**;
a purely Saturn-free stream runs at **39.47 c/mvin** (161,687 / 4,096, the
`stream=` probe with Saturn units no-op'd). So:

```
179,033 (best kernel total)
  = 161,687 (pure weight stream, Saturn fully removed)
  + 17,346  (Saturn cost STILL exposed on the critical path, 9.7% of the total)
```

`17,346 / 28,372 = 0.611` is `DEFAULT_OVERLAP_FACTOR` — i.e. **61.1% of
Saturn's own compute survives as exposed wall time**, not "almost entirely
hidden." Round8's own iter6 probe had already found essentially the same
17,898-cycle exposed residual; round9's iter1-4 (DMA-warming, half-grain
units, host-dispatch probes) did not close it — iter4's 552-cycle gain came
from host-side scalar-recomputation overhead, not from hiding more of
Saturn's compute. **Practical implication: the ~61% Saturn exposure is an
achieved constant from four rounds of tuning, not a quantity trending to
zero** — see `OVERLAP_NOTE_MD` point 4 in `loop/llama_project.py`. The
critical path is shared between weight streaming, host command issue, and
the still-exposed 61% of Saturn's compute — it has not "shifted away" from
Saturn compute, because that compute was never shown to be fully hidden in
the first place.

iter3 measured the host-dispatch side of that remaining critical path
directly:

- `unit_host_cycles=48799` over `unit_calls=256` → **~190.6 host cycles per
  unit-call** (per hardware-unit invocation dispatched from the host to
  Saturn/Gemmini). Against a kernel total of 183,430 cycles, the 48,799
  cycles of host-side unit dispatch is **~26.6%** of the whole run — a
  non-trivial fraction, but note this figure includes the *entire* host wall
  time inside each unit (compute issue + whatever DMA/queue interaction
  happens synchronously within it), not purely idle dispatch overhead, so it
  overlaps with (rather than strictly adds to) the DMA-bound portion of the
  timeline.
- `mvin_stall_sampled=2558` over `mvin_samples=512` → **~5.0 stall cycles per
  sampled mvin2 issue** on average. iter4's kernel header interprets this
  precisely: at ~5 cycles/mvin the load-reservation-station / DMA issue path
  essentially never blocks the host (a full DMA round trip would cost tens
  of cycles here), so the host's mvin-issue path is *not* the dominant
  stall; instead the host has slack (~15-27 cycles per mvin, per iter4's own
  model) that is being spent on **scalar address/operand recomputation
  around each RoCC command**, not on waiting for the memory system. iter4's
  change (incrementally-updated 64-bit RoCC operands instead of per-mvin
  address rebuild) shaved 2,080 mvins' worth of that scalar overhead and
  produced the round's 552-cycle net improvement.
- Combined reading: mvin/DMA issue itself is **not** the bottleneck (only
  ~5 c/mvin of exposed stall), but the **host-side dispatch/bookkeeping**
  around each unit and each mvin (the ~190.6 c/unit-call and the scalar
  RoCC-operand-rebuild work identified in iter4) *is* a meaningful, still
  partially exposed cost. This grounds "weight streaming is the remaining
  bottleneck" more precisely as: **the DMA/weight-stream fabric itself has
  headroom (it isn't stalling much), but the host's per-command/per-unit
  dispatch overhead needed to keep that stream fed is what's now on the
  critical path** — batching/coalescing RoCC commands and further reducing
  scalar work per mvin (continuing iter4's direction) is one promising next
  lever. **Corrected note:** this does not mean Saturn-compute tuning is
  played out — per the stream-rate breakdown above, 61.1% of Saturn's own
  compute is still exposed (17,346 of 28,372 cycles), so a Saturn-side win
  and a host-dispatch win are both still on the table, not either/or.

## Limitations

Carried forward from round8 (still applicable, now with round9's data
layered on top):

- The 13.2% (13.2188% exact) overlap/retiling reduction was measured on
  **exactly one** fused kernel, `llama-layer-fused-n1`, which itself covers
  only **part of one transformer decode layer** — not a full layer, and not
  the full 16-layer decode step.
- Applying this ratio "to the whole decode step" assumes the same
  overlap/retiling technique, at the same overlap ratio, generalizes
  **uniformly** across: (a) the rest of this same layer's ops not covered by
  the fused kernel, (b) all 15 other decode layers, and (c) the `lm_head`
  kernel (which alone is 20.5% of decode cycles/token per the per-kernel-type
  table). None of these have been measured with this overlap/retiling
  technique applied — their actual achievable overlap ratio is **unmeasured**.
- It is plausible other kernels/layers overlap better or worse than 13.2%
  (e.g. `lm_head` has no following layer's attention/FFN work to overlap
  against, since it's the last op in the decode step) — this round's
  179,033-cycle point is still 1.273x its own roofline, so there is
  headroom for a larger overlap win on this one kernel too, in addition to
  the unmeasured cross-layer generalization question above.
- **Bottom line, corrected**: the 5.092 tok/s @ 1 GHz / 2.546 tok/s @ 500 MHz
  figures (and round8's 5.076/2.538) were never a defensible projection — see
  the retraction above. The measured, tool-native way to bound the effect
  is `--overlap`: **0.3% (attention pinned to gemmini) to 1.4% (attention
  pinned to saturn)**, i.e. 4.419 -> 4.432-4.483 tok/s @ 1 GHz / 2.209 ->
  2.216-2.242 tok/s @ 500 MHz, not +15.2%. This is because the fused
  kernel's own Saturn cycle share (13.8%) is 6-20x the whole decode step's
  actual Saturn cycle share (0.7-2.2%) — an overlap ratio cannot be moved
  from one workload to a workload with an order-of-magnitude-different
  device mix and preserve its whole-workload percentage effect.
- **New this round, corrected**: the sat_done/gemv_end split does **not**
  show Saturn's own compute is now mostly hidden — that reading was an
  artifact of `sat_done` marking when Saturn *runs out of assigned work*
  (a geometric property of the 16-mvins-per-Saturn-call schedule), not when
  its cost stops being exposed. The stream-rate breakdown shows 61.1% of
  Saturn's own compute (17,346 of 28,372 cycles) is still exposed — the same
  order of magnitude as round8's 17,898-cycle residual. Round9 iter1-3's
  Saturn-side/DMA-warming changes did regress or stay flat, and iter4's
  host-dispatch-overhead fix did move the needle (552 cycles) — that much is
  still true — but it does not follow that Saturn-side optimization is
  played out; four iterations of one technique (DMA-warming, grain-halving)
  failing is not the same as the ~61%-exposed Saturn cost being fundamentally
  unhideable. Both host-dispatch batching and further Saturn-side work
  remain open, and — as above — neither says anything about whether the
  whole decode step's other kernels behave the same way.

## Data sources

- `out/llama-profile/measured_cycles_round9.json` —
  `measured_cycles_round8.json` with the `llama-layer-fused-n1` entry's
  `cycles`/`best_cycles` updated from 179,585 to 179,033 (baseline 206,304
  and roofline 140,680 unchanged). All 14 other entries byte-for-byte
  identical to round8 — no round1-8 per-op measurement was altered.
- Raw tool output: `out/llama-profile/proj_round9_decode_cold567_1ghz.txt`,
  `out/llama-profile/proj_round9_decode_cold567_500mhz.txt`.
- Probe source: `out/loop/20260916-195958-36b4/simlog_01.txt` (sat_done,
  gemv_end, warm_cmds), `simlog_02.txt` (sat_done, gemv_end),
  `simlog_03.txt` (sat_done, gemv_end, unit_host_cycles, unit_calls,
  mvin_stall_sampled, mvin_samples) — all grepped directly from the run
  directory, verbatim.
