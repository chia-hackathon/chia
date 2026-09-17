# Llama-3.2-1B round8 projection — layer-fused overlap update + cold-GEMV decode

Generated against `measured_cycles_round8.json` (new in this task), which is
`measured_cycles_round7.json` with the `llama-layer-fused-n1` entry's
`cycles`/`best_cycles` updated from round7's 189,851 to round8's improved
**179,585** (baseline and roofline for that entry are unchanged: 206,304 and
140,680 respectively; see that file's `_README`-adjacent notes on the entry
itself). No other entry was touched — round8 produced no new measurement for
any of the other 14 kernel entries, so they are carried forward from round7
byte-for-byte.

All commands use `--device int8_gemv=gemmini --device lm_head_gemv=gemmini`,
same convention as round2-round7. Round7 only ran at **1.00 GHz**; round8
repeats the same cold command at **both 1.00 GHz and 500 MHz** (via
`--clock-ghz`), since the whole point of a clock projection is comparing
candidate clocks, not just picking one. Raw command output for every run
below is saved under
`out/llama-profile/proj_round8_decode_cold567_{1ghz,500mhz}.txt`.

```
# cold, round7/round8's own measured cold-GEMV rate (5.67 B/cycle) --
# same rate round7 used, since round8 did not re-measure the cold-DRAM GEMV
# rate; only the layer-fused-n1 overlap number changed this round.
python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round8.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --gemv-bytes-per-cycle 5.67 --clock-ghz 1.0

python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round8.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --gemv-bytes-per-cycle 5.67 --clock-ghz 0.5
```

## (new) Round8 finding: `llama-layer-fused-n1` improves further, 8.0% -> 13.0%

The `llama-layer-fused-n1` documentation-only measurement (see round7's
finding 1) got a new best-achieved point this round, from further
overlap/retiling on top of round7's iter2:

| | cycles | vs. baseline | vs. round7 best |
|---|---:|---:|---:|
| baseline (sequential, unfused) | 206,304 | — | — |
| round7 best (overlap/fusion, iter2) | 189,851 | -8.03% | — |
| **round8 best (further overlap/retiling)** | **179,585** | **-12.95%** | **-5.41%** |
| roofline (ideal fully-overlapped floor) | 140,680 | 179,585/140,680 = 1.277x | (round7: 1.349x) |

`(206304 - 179585) / 206304 = 0.12951` → **13.0% reduction** from overlap
(rounded from the exact 12.951%, same rounding convention round7 used for its
8.029% -> "8.0%"), up from round7's 8.03%/8.0%. The 179,585-cycle point is
also closer to the 140,680-cycle roofline than round7's was (1.277x vs.
1.349x), i.e. round8 closed part of the remaining overlap headroom without
eliminating it. As in round7, this kernel name is **not** one of the kernel
types in `loop/llama_model.py`'s Op taxonomy, so `MeasuredDB.load` does not
read it into any per-op override — the decode/prefill projections above are
byte-for-byte unaffected by updating this entry, and (as in round7) it is
used below purely as an input to a manual, whole-decode-step "what if"
calculation.

## Cold-GEMV decode, 13.0% overlap applied, at two clocks

> **RETRACTED (see `out/llama-profile/projection_round9.md`,
> "Cold-GEMV decode with overlap" section).** The 5.076/2.538 tok/s
> (+14.9%) figures below apply the `llama-layer-fused-n1` kernel's 13.0%
> cycle reduction *uniformly* to the whole decode step's cycle count. That
> extrapolation is invalid: it assumes every decode kernel overlaps with
> Gemmini's weight stream at the same ratio as one hand-fused, hand-tuned
> kernel whose own Saturn cycle share (13.8% of its total) is an order of
> magnitude larger than the whole decode step's actual Saturn cycle share
> (0.7-2.2%, depending on where attention is device-assigned). The correct,
> measured figure — using `loop/llama_project.py`'s `--overlap` flag against
> `measured_cycles_round9.json` — is **0.3-1.4%**, i.e. 4.419 -> 4.432-4.483
> tok/s @ 1 GHz / 2.209 -> 2.216-2.242 tok/s @ 500 MHz, not +14.9%. The rest
> of this section is kept for the historical record of round8's own
> `llama-layer-fused-n1` kernel result (179,585 cycles, -12.95% vs.
> baseline), which is unaffected by this retraction — only the "apply this
> ratio to the whole decode step" step below is wrong.

Round7 established the cold-GEMV rate (5.67 B/cycle, from the
`llama-q8-gemv-gemmini-n1` cold re-measurement) and the end-to-end cold
decode projection built from it (226.30M cycles/token). Round8 made no new
cold-GEMV measurement, so the same 5.67 B/cycle rate and the same 226.30M
cycles/token baseline (before overlap) carry forward unchanged — confirmed by
re-running the identical `--gemv-bytes-per-cycle 5.67` command above against
`measured_cycles_round8.json` (the per-op/per-device table is identical to
round7's, since the layer-fused-n1 entry is documentation-only and does not
feed the per-op projection). What's new is applying the improved 13.0%
overlap ratio (vs. round7's 8.0%) to that same cold baseline, and doing so at
both 1 GHz and 500 MHz instead of only 1 GHz:

| clock | before overlap (cold, 5.67 B/cycle GEMV) | after overlap (x(1-0.1295)) | tok/s gain |
|---|---:|---:|---:|
| **1.00 GHz** | 226.30M cyc/tok -> **4.419 tok/s** | 196.99M cyc/tok -> **5.076 tok/s** | +14.9% |
| **500 MHz**  | 226.30M cyc/tok -> **2.209 tok/s** | 196.99M cyc/tok -> **2.538 tok/s** | +14.9% |

(`llama_project.py` has no flag to apply an "overlap this fraction of every
kernel" adjustment directly — same limitation round5 and round7 hit — so,
following round7's precedent, the 13.0% ratio is applied by hand to the
tool's raw cold-decode cycles/token: `226,300,507 x (1 - 0.129513) =
196,991,704` cycles/token, exact values from the tool's internal
`Projection.cycles_per_token`, rounded to `226.30M` / `196.99M` above for
display. Tok/s at each clock is then `clock_hz / cycles_per_token`, so the
500 MHz column is exactly half the 1 GHz column's tok/s, as expected since
cycles/token doesn't depend on clock.)

**N=1 tokens/sec under cold GEMV, round8 overlap applied: 4.419 -> 5.076
tok/s @ 1 GHz (+14.9%), 2.209 -> 2.538 tok/s @ 500 MHz (+14.9%).** As in
round7, tok/s scales as the inverse of cycles/token, so a 12.95% cycle
reduction yields a slightly larger `1/(1-0.12951)-1 = 14.9%` tok/s gain, not
exactly 13.0% — same mechanism as round7's 8.03% -> 8.7%, just a larger
overlap ratio this round. The two clocks produce proportional tok/s (500 MHz
is exactly half of 1 GHz's number) because this projection changes only
cycles/token, never the clock itself.

## Limitations

This projection is a **hand-applied, optimistic upper-bound-style
extrapolation, not a measured result**, for the same reason round7's was, now
stated explicitly:

- The 13.0% (12.95% exact) overlap/retiling reduction was measured on
  **exactly one** fused kernel, `llama-layer-fused-n1`, which itself covers
  only **part of one transformer decode layer** (a subset of that layer's
  attention + FFN sub-ops fused/overlapped end-to-end) — not a full layer,
  and not the full 16-layer decode step.
- Applying this ratio "to the whole decode step" assumes the same
  overlap/retiling technique, at the same overlap ratio, generalizes
  **uniformly** across: (a) the rest of this same layer's ops not covered by
  the fused kernel, (b) all 15 other decode layers, and (c) the `lm_head`
  kernel (which alone is 20.5% of decode cycles/token per the per-kernel-type
  table above). None of these have been measured with this overlap/retiling
  technique applied — their actual achievable overlap ratio is **unmeasured**.
  Prior rounds' int8_gemv/lm_head_gemv entries used in the per-op table above
  are unrelated warm/cold GEMV rate measurements, not overlap measurements,
  and do not corroborate or contradict this assumption either way.
- It is plausible other kernels/layers overlap better than 13.0% (e.g. if
  they have more independent Saturn-side work to hide behind Gemmini's GEMV
  latency) or worse (e.g. `lm_head` has no following layer's attention/FFN
  work to overlap against, since it's the last op in the decode step) — this
  round's 179,585-cycle point is still 1.277x its own roofline, so there is
  headroom for a larger overlap win on this one kernel too, in addition to
  the unmeasured cross-layer generalization question above.
- **Bottom line, RETRACTED**: the 5.076 tok/s @ 1 GHz / 2.538 tok/s @ 500 MHz
  figures were never a defensible projection (see the retraction note at the
  top of this section) — treating them as any kind of bound, optimistic or
  otherwise, repeats the same error. The measured figure, from `--overlap`
  against `measured_cycles_round9.json`, is 0.3-1.4% (see
  `out/llama-profile/projection_round9.md`). Closing the remaining gap
  requires either measuring the same overlap technique on
  additional kernels (especially `lm_head`, given its 20.5% share) or
  fusing/measuring a full decode layer (all 16) end-to-end rather than the
  partial single-layer kernel measured so far.

## Data sources

- `out/llama-profile/measured_cycles_round8.json` —
  `measured_cycles_round7.json` with the `llama-layer-fused-n1` entry's
  `cycles`/`best_cycles` updated from 189,851 to 179,585 (baseline 206,304
  and roofline 140,680 unchanged). Diffed against round7's entries list: all
  14 other entries byte-for-byte identical, only the one layer-fused-n1 entry
  changed (`cycles`, `best_cycles`, `source`, `notes` fields) — no round1-7
  per-op measurement was altered.
- Raw tool output: `out/llama-profile/proj_round8_decode_cold567_1ghz.txt`,
  `proj_round8_decode_cold567_500mhz.txt`.
- `llama_project.py --help` (captured 2026-09-15): confirms `--clock-ghz`
  (default 1.0) is the general flag for projecting at an arbitrary clock, and
  that `--gemv-bytes-per-cycle`/`--cold` behave identically to round7 — round8
  reuses round7's 5.67 B/cycle cold-GEMV rate unchanged (no new cold-DRAM
  measurement was taken this round).
