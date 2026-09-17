# Llama-3.2-1B round7 projection — layer-fused overlap + cold-GEMV decode

Generated against `measured_cycles_round7.json` (new in this task), which is
`measured_cycles_round5.json` plus one new documentation-only entry,
`llama-layer-fused-n1` (see that file's `_README`-adjacent notes on the entry
itself). Round6 produced no new best (carried forward from round5 unchanged,
per `out/loop/ledger.md`), so round7 is compared against round5, not round6.

All commands use `--device int8_gemv=gemmini --device lm_head_gemv=gemmini`,
same convention as round2-round5. Clock is **1.00 GHz** throughout. Raw
command output for every run below is saved under
`out/llama-profile/proj_round7_decode_{warm,cold_upper,cold_lower,cold567}.txt`.

```
# warm (unchanged baseline, same as round5's decode number)
python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round7.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini

# cold, built-in shorthand presets (2026-09-12 audit rates, for comparison)
python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round7.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini --cold upper   # 7.00 B/cycle
python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round7.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini --cold lower  # 6.43 B/cycle

# cold, round7's own measured cold-GEMV rate (5.67 B/cycle, see below) --
# slower than either built-in --cold preset, so passed explicitly via
# --gemv-bytes-per-cycle rather than the coarser --cold {upper,lower} shorthand
python loop/llama_project.py --scenario decode --S 512 \
  --measured out/llama-profile/measured_cycles_round7.json \
  --device int8_gemv=gemmini --device lm_head_gemv=gemmini \
  --gemv-bytes-per-cycle 5.67
```

## (new) Round7 finding 1: `llama-layer-fused-n1` — overlap/fusion of a whole decode layer

A new documentation-only measurement, `llama-layer-fused-n1` (added to
`measured_cycles_round7.json`'s `entries`), fuses several decode-layer ops
end-to-end instead of running them strictly sequentially:

| | cycles | vs. baseline |
|---|---:|---:|
| baseline (sequential, unfused) | 206,304 | — |
| best achieved (round7 iter2, overlap/fusion) | 189,851 | **-8.03%** |
| roofline (ideal fully-overlapped floor) | 140,680 | 189,851/140,680 = 1.349x |

`(206304 - 189851) / 206304 = 0.08029` → **8.0% reduction** from overlap,
consistent with the task's framing. This kernel name is **not** one of the
kernel types in `loop/llama_model.py`'s Op taxonomy, so — like the round5
N=16 `int8_gemv_gemmini` point — `MeasuredDB.load` does not read it into any
per-op override; `llama_project.py`'s decode/prefill projections above are
byte-for-byte unaffected by adding this entry. It is used below purely as an
input to a manual, whole-decode-step "what if" calculation, the same pattern
round5 used for its N=16 batched-GEMV comparison (section (e) of
`projection_round5.md`).

## (new) Round7 finding 2: cold vs warm GEMV, and what it does to decode

Round7 also re-measured `llama-q8-gemv-gemmini-n1` under a cold scenario
(weights not resident / cache cold at the start of the GEMV):

| | cycles | B/cycle (1 MiB weights) |
|---|---:|---:|
| warm (unchanged since round2) | 132,424 (task quoted 132,435; matches to rounding) | 7.91-7.92 |
| **cold (round7)** | **184,837** (task quoted; `1,048,576 / 184,837 = 5.673` B/cycle) | **5.67** |

This 5.67 B/cycle cold rate is slower than *both* of `llama_project.py`'s
built-in `--cold {upper,lower}` presets (7.00 and 6.43 B/cycle, from the
2026-09-12 audit), so it was passed explicitly with
`--gemv-bytes-per-cycle 5.67` rather than via `--cold`.

### End-to-end decode (S=512, N=1 token), warm vs. cold

| scenario | cycles/token | tokens/s | gemmini share of decode cycles | saturn share of decode cycles |
|---|---:|---:|---:|---:|
| warm (baseline) | 159.73M | 6.260 | 94.8% | **5.2%** |
| cold, `--cold upper` (7.00 B/cycle) | 187.10M | 5.345 | 95.5% | 4.5% |
| cold, `--cold lower` (6.43 B/cycle) | 199.43M | 5.014 | 95.8% | 4.2% |
| **cold, round7 measured (5.67 B/cycle)** | **226.30M** | **4.419** | **96.3%** | **3.7%** |

(gemmini/saturn shares read directly off each run's `--- per device ---`
table; `int8_gemv` + `lm_head_gemv` both run on `gemmini` here because of
`--device ...=gemmini`, everything else — attention scores/pv, softmax,
rmsnorm, rope, add, embedding — runs on plain `saturn` RVV.)

**Why cold matters here, reasoned from the numbers above (not asserted):**
cold only makes the GEMV/lm_head weight fetch slower — it does not touch any
`saturn`-device op's cycle count, which is fixed at 8.36M cycles/token in
every row above. What changes is the denominator: as GEMV's rate drops from
7.91-7.92 B/cycle (warm) to 5.67 B/cycle (cold), `gemmini`'s cycles/token
climb from 151.38M to 217.94M, and total decode cycles/token climb from
159.73M to 226.30M. Since `saturn`'s absolute 8.36M cycles/token is unchanged,
its **share** of the (now larger) total shrinks: **5.2% (warm) -> 3.7% (cold,
round7 rate)** — a real, precisely-quantified drop, not just a rounding
artifact (it drops monotonically through the intermediate `--cold`
upper/lower presets too: 5.2% -> 4.5% -> 4.2% -> 3.7%). So in a cold-decode
world, Gemmini/the GEMV path dominates decode even more than in the warm
case (94.8% -> 96.3% of total decode cycles), and the plain-Saturn RVV
elementwise/attention work becomes a correspondingly smaller slice of the
wall clock — exactly the opposite of "cold makes Saturn matter more"; cold
makes the GEMV weight-fetch bottleneck matter more and everything else matter
less, in relative terms, even though nothing about those other ops changed.

### Applying the 8.0% layer-fusion overlap saving to the whole decode step, cold

The `llama-layer-fused-n1` measurement's 8.03% reduction was measured on one
fused decode layer under (implicitly) warm conditions. `llama_project.py` has
no flag to apply an "overlap this fraction of every kernel" adjustment
directly (same limitation round5 hit for batched decode), so — following
round5's precedent of computing such derived numbers by hand from the tool's
raw output — the ratio is applied to the **cold, round7-rate** end-to-end
decode projection above:

- Before overlap (cold, 5.67 B/cycle GEMV, run directly by the tool):
  **226.30M cycles/token -> 4.419 tokens/s**
- After overlap (`226.30M x (1 - 0.0803) = 208.13M` cycles/token, hand
  calculation on the tool's baseline output, same method as round5(e)):
  **208.13M cycles/token -> 4.805 tokens/s**

**N=1 tokens/sec under cold GEMV: 4.419 -> 4.805 tok/s, a +8.7% throughput
gain** (tok/s scales as the inverse of cycles/token, so an 8.03% cycle
reduction yields a very slightly larger 1/(1-0.0803)-1 = 8.7% tok/s gain, not
exactly 8.0%). This is a projection, not a new hardware measurement: it
assumes the 8.0% overlap ratio measured on one fused layer generalizes
uniformly to the rest of the decode step (all 16 layers + lm_head), which is
optimistic — the layer-fused kernel's own 189,851 cycles are still 1.349x its
140,680 roofline, so there is headroom for a larger overlap win in later
rounds, or a smaller one if 8.0% doesn't generalize past the one measured
layer.

## Data sources

- `out/llama-profile/measured_cycles_round7.json` — `measured_cycles_round5.json`
  plus the new `llama-layer-fused-n1` entry (baseline_cycles 206,304,
  roofline_cycles 140,680, best_cycles 189,851). Diffed against round5's
  entries list: first 14 entries byte-for-byte identical, 1 new entry
  appended — no round1-5 data was altered.
- Raw tool output: `out/llama-profile/proj_round7_decode_warm.txt`,
  `proj_round7_decode_cold_upper.txt`, `proj_round7_decode_cold_lower.txt`,
  `proj_round7_decode_cold567.txt`.
- `llama_project.py --help` (captured 2026-09-14): confirms `--cold
  {upper,lower}` is a shorthand for two specific 2026-09-12-audit GEMV rates
  (7.00 / 6.43 B/cycle) and that `--gemv-bytes-per-cycle` is the general
  escape hatch for "apply a COLD-DRAM measured rate" that doesn't match
  either preset — used here for round7's own 5.67 B/cycle finding.
