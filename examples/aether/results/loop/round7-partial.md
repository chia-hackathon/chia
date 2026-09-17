# Round 7 — partial results (extracted after weekly Claude usage cap)

## 1. Round 7 summary

Round 7 of the inner optimization loop started 2026-09-12/13 and was cut short by the
weekly Claude usage cap after **2 real sub-runs** (out of a larger planned batch).
Both runs completed all their scheduled iterations before the cap hit; nothing was
mid-iteration.

- `out/loop/20260912-191730-3a47/` — kernel **`llama-layer-fused-n1`** (new this
  round; fuses one Gemmini decode GEMV with one attention head's QK^T/softmax/
  probs@V on Saturn). Baseline 206,304 cycles, roofline (memory-bound, 8 B/cycle
  mbus) 140,680 cycles. 4 iterations, cost **$4.99** (`1.137207 + 2.236368 +
  0.737698 + 0.882153`).
- `out/loop/20260912-191830-e713/` — kernel **`llama-q8-gemv-gemmini-n1`**, a
  probe-only run seeded from the existing 132,424-cycle winner (baseline
  re-measured at 150,339). 2 iterations, cost **$2.54** (`cost.json` /
  `summary.txt`).
- Combined cost for the round so far: **~$7.54**.

Both runs deliberately spent iterations on measurement ("PROBE") rather than
score improvement, per the `kernels.py` guidance for these entries ("a
confident prediction that misses by 10k is worth more than a 200-cycle
tweak, because it closes a branch").

---

## 2. `llama-layer-fused-n1` (run `20260912-191730-3a47`)

Baseline (iter 0): **206,304** cycles. Roofline: **140,680** cycles (1.47x).

| iter | HYPOTHESIS (one-line) | result | vs baseline | verdict |
|---|---|---|---|---|
| 1 | Moving `gemmini_fence()` to after the *last* `loop_ws` issue and overlapping QK^T (4×128 rows, gaps 1-4), softmax (gap 5) and probs@V (gap 6) inside the 8 host-idle gaps between the 9 `loop_ws` issues will hide most of the ~14.6k Saturn cost; EXPECTED ~190,000. | **217,295** | +5.3% (nominal regression) | Hypothesis **SURVIVED**: real fence-completion time (`PROBE gap … f 190487`) was 190,487 — the scheduling win is real (~16k saved vs 206,304). The 217,295 total is inflated by ~26.8k of PROBE `printf`/HTIF overhead on top of that. |
| 2 | Move probs@V from gap 5 to **gap 8** (after the final `loop_ws` issue, before the single fence) so the ≥32k-cycle idle tail absorbs its ~28k cold cost, avoiding its 512 single-row V8 misses colliding with loop 5's weight stream on DRAM; EXPECTED ~175,000 (190,487 − loop 5's ~15k drag). | **189,851** | **−8.0% (NEW BEST)** | Expectation of ~175,000 **DIED** — only 636 cycles better than iter 1's real (probe-free) fence time of 190,487. But it is still the round's best score. |
| 3 | The Zicbop `prefetch.r` bursts issued between `loop_ws` calls act, when the L2 is cold, as a second sequential DRAM stream that interleaves with the weight stream and costs ~30 cycles/hint (page-switch) per hint (~1000 hints/burst × 8 bursts); deleting all bursts should save ~25k; EXPECTED ~165,000. | **189,924** | flat (+73 vs iter 2) | **DIED** — cold, the Zicbop burst is a NOP: it neither helps (unlike the warm case, −229 cycles) nor hurts. Three different schedules (pv@gap5, pv@gap8, no-prefetch) all land at 190k ± 0.3k. |
| 4 | Total time is set by a schedule-independent serial resource: the baseline's cold GEMV alone is intrinsically ~175-180k cycles (iter 1's 66k idle tail in loops 7+8 is *inherent* DRAM busy-time, not hidden slack), and the uncontended cold Saturn half is only ~26-30k; the 3× convergence at ~190k = GEMV floor + an additive Saturn/DMA-contention cost (~10k). This iteration reverts to the **strict sequential baseline schedule** and adds `rdcycle` timestamps at every `loop_ws` return and every phase boundary; EXPECTED ~206,304 + ~18k printf ≈ 224,000 (predicted fence≈178k, scores≈12k, softmax≈2k, pv≈14k). | **223,640** | +8.4% (deliberate probe regression) | Matches prediction almost exactly (223,640 vs ~224,000) — this iteration is a pure measurement pass, not a score attempt. |

### Iter-4 four-phase breakdown (clean sequential baseline + PROBE)

Note: the task brief expected this decomposition in iter 1, but the file that
actually carries the clean sequential 4-phase split is **iter 4** (kernel_04.h /
`simlog_04.txt`), because iter 4 is the "PROBE iteration" that deliberately
reverted to the strict back-to-back baseline schedule specifically to measure
this. Iter 1's own PROBE lines (`PROBE iss …` / `PROBE gap … f 190487`)
measure the *overlapped* schedule's loop_ws-issue and gap timestamps instead,
and are reported verbatim in section 4.

From `simlog_04.txt`: `PROBE f 186245 s 196526 m 198168 p 205864` (all
timestamps `rdcycle`-relative to kernel entry; total reported score 223,640
includes ~17.8k of PROBE `printf`/HTIF overhead not captured by `t_p`):

| phase | cycles | % of real (205,864) schedule |
|---|---|---|
| Gemmini GEMV (incl. terminal fence) | **186,245** | 90.5% |
| attn-scores (QK^T) | 196,526 − 186,245 = **10,281** | 5.0% |
| softmax | 198,168 − 196,526 = **1,642** | 0.8% |
| attn-pv (probs@V) | 205,864 − 198,168 = **7,696** | 3.7% |
| *(PROBE/printf overhead, not part of the schedule)* | 223,640 − 205,864 = 17,776 | — |

This confirms: GEMV completely dominates (>90%), and the Saturn tail
(scores+softmax+pv = 19,619 cycles) is close to — but larger than — the
"14,602 cycles, warm" figure quoted in `kernels.py`, consistent with iter 3/
agent 4's conclusion that K8/V8/kscale being cold adds real stall time on top
of the warm-harness numbers.

### Why iter 2 improved (206,304/217,295 → 189,851)

From `kernel_02.h` header and `agent_02.txt`: iter 1's PROBE data showed the
overlapped-schedule fence completing at 190,487 (already ~16k below baseline,
confirming the fence-move + gap-overlap hypothesis), but it also exposed a
finer machine model: loops 1-4 and 6 stream at the full ~7.7 B/cycle rate,
while loop 0, loop 5 (during probs@V) and the tail loops 7+8 (66k cycles, host
completely idle!) run at only ~3.8 B/cycle — i.e. cold Saturn work colliding
with the weight stream costs ~250-330 extra cycles per miss (scores 4×13.5k,
softmax 9.2k, pv 28.3k under contention). Iter 2's single change: **move
probs@V from gap 5 to gap 8** (after the 9th/last `loop_ws` issue, immediately
before the single terminal fence), so its 512 single-row V8 misses land in the
already-idle tail instead of interleaving with loop 5's weight stream on
DRAM — while keeping the fence-move and gaps 1-4 (QK^T) / gap 5 (softmax)
unchanged. Result: 189,851, a new best, though only 636 cycles better than
iter 1's real (190,487) fence time — iter 3 later explained why the gain was
so small (DRAM contention cost is additive/position-independent, not
"hideable" by relocation).

---

## 3. `llama-q8-gemv-gemmini-n1` probe run (`20260912-191830-e713`)

Seed 132,424 cycles (from run `20260908-202413-e8e8` iter 2); baseline
re-measured at 150,339; best stays 132,424 (speedup 1.135x / +11.9% over
baseline) — **this run did not attempt to beat the score**, per
`summary.txt`.

| iter | HYPOTHESIS (one-line) | result |
|---|---|---|
| 1 | Gemmini's StreamReader allows 16 Gets in flight but the L2 only has 12 MSHRs; beyond that L2 back-pressures the sbus and the StreamReader stalls in `waiting_for_dma_req_ready` — i.e. the Round-4 "rows-per-mvin" curve is an on-chip in-flight-depth effect, not DRAM latency. Predicted: cold mvin-only 1 MiB scan at rows=4 (8 in flight) ≈150k, rows=8/16 (16 in flight) ≈9% worse ≈163k, knee between 8-12 in flight. | **3,017,659** (huge regression — by design: this iteration appends 6 cold mvin-only scans, each preceded by a 1.5 MiB L2 flush, after the scored kernel, inside the timed region) |
| 2 | Wrap the winning 132,424-cycle kernel unchanged as `lq8_gemv_ws`, then run two follow-up probes: (a) a "pass-through" probe — issue 64 single-row mvins to unused scratchpad rows immediately after one `loop_ws` K-tile returns, to see whether `LoopMatmul` blocks host issue (≈8k ⇒ blocked, the 4-row-mvin lever is unusable; ≈hundreds ⇒ usable, worth ≈15k); (b) flush the L2 with 1.5 MiB unrelated DRAM traffic, then re-run the same kernel cold and print `PROBE cold_gemv_cycles`. Predicted cold ≈184.6k (5.68 B/cycle). | **660,409** |

### Iter 1 — in-flight-depth scan (all values from `simlog_01.txt`, cold, each preceded by an L2 flush)

| rows/mvin | in-flight | flush_cycles | cold_mvin_cycles | B/cycle |
|---|---|---|---|---|
| 3 (measured first — genuinely SoC-cold) | 6 | 274,021 | 166,067 | 6.31 |
| **4** | **8** | 276,279 | **158,109** | **6.63** ← optimum |
| 5 | 10 | 276,368 | 173,029 | 6.06 |
| 6 | 12 | 276,702 | 179,649 | 5.83 |
| 8 | 16 | 276,693 | 181,520 | 5.77 |
| 16 (= `loop_ws` shape) | 16 | 276,716 | 184,298 | 5.68 |

Conclusion (agent_02.txt): the MSHR/in-flight-depth **hypothesis SURVIVED** —
there is a real peak at 8-in-flight and monotonic degradation from 10 in-flight
up, independent of stride (matches lmhead's stride-2048 behavior at 512), so
it's an on-chip Gemmini→L2 effect, not a DRAM-locality effect. But it also
overturned the "cold ≈7.0-7.1 B/cycle" assumption from `kernels.py`: the actual
cold `loop_ws`-shape (16 rows/mvin) rate is only **5.68 B/cycle**, meaning
132,424 cycles' true warm-L2-tail credit is ~52k cycles, not the previously
assumed 17k. Iter 1's own EXPECTED (~2.2M) was itself wrong: it assumed
~8 B/cycle for the flush/probe streams, but the flush stream and cold B
stream both ran far slower (5.68-6.63 B/cycle), producing the observed
3,017,659.

### Iter 2 — warm/cold contrast (`simlog_02.txt`)

```
PROBE warm_gemv_cycles=132435
PROBE warm_bytes_per_cycle_x100=791
PROBE passthru_loop_issue_cycles=38
PROBE passthru_64mvin_accept_cycles=15126
PROBE passthru_loop_plus_mvins_total_cycles=15333
PROBE flush_cycles=274464
PROBE cold_gemv_cycles=184837
PROBE cold_bytes_per_cycle_x100=567
```

Confirmed: **warm_gemv_cycles = 132,435 @ 7.91 B/cycle** and
**cold_gemv_cycles = 184,837 @ 5.67 B/cycle**.

Context / what "warm" vs "cold" means here: "warm" is the *scored* kernel run
as usual, immediately after the benchmark harness fills matrix B in DRAM —
this leaves ~200-260 KiB of B resident (randomly) in the 512 KiB L2, and the
kernel's descending-K-tile order deliberately harvests that residue (worth
17,104 cycles per earlier Round-4/5 measurements). "Cold" is the identical
kernel run a second time in the same timed region, immediately after 1.5 MiB
of unrelated DRAM traffic has been DMA'd through to evict that residue from
L2 — i.e. it approximates what a real decode step sees (previous layer's
weights have just evicted everything), rather than the benchmark-artifact
warm state.

The pass-through probe found `passthru_loop_issue_cycles=38` (host regains
control almost immediately) but `passthru_64mvin_accept_cycles=15126` for 64
single-row mvins to spend into that slack — i.e. `LoopMatmul` does **not**
fully block host issue, but injecting mvins there is expensive per-mvin, so
the 4-row-mvin lever (potential ≈15k win from re-tiling B's mvin at 4 rows
instead of 16, since 4 rows/mvin hit 6.63 B/cycle vs 16 rows/mvin's 5.68
B/cycle) is a real but as-yet-unexploited opportunity (≈600 KiB ×
(1/5.68 − 1/6.63) ≈ 15k cycles, per agent_02.txt's arithmetic).

Bottom line the agent draws: the "cold ≈7.0-7.1 B/cycle" figure used
throughout `kernels.py` (and therefore in `llama-layer-fused-n1`'s roofline
discussion) is optimistic by ~19%; the true cold `loop_ws`-shape rate is
5.68 B/cycle, so a cold GEMV of this shape costs ≈184.6k, not ≈149-150k,
and the warm-tail credit inside 132,424 is ≈52k cycles, not ≈17k.

---

## 4. All PROBE lines (verbatim)

### Run `20260912-191730-3a47`

**iter 1** (`simlog_01.txt`):
```
PROBE ref_score_min_micro=-9160885
PROBE ref_score_max_micro=9122776
PROBE iss 152 221 16791 32461 47932 63838 77360 108302 124612
PROBE gap 195 14154 29894 45372 61276 73015 105628 108318 124628 f 190487
```

**iter 2** (`simlog_02.txt`):
```
PROBE ref_score_min_micro=-9160885
PROBE ref_score_max_micro=9122776
```

**iter 3** (`simlog_03.txt`):
```
PROBE ref_score_min_micro=-9160885
PROBE ref_score_max_micro=9122776
```

**iter 4** (`simlog_04.txt`):
```
PROBE ref_score_min_micro=-9160885
PROBE ref_score_max_micro=9122776
PROBE iss 2956 11600 33347 55019 76763 98555 120433
PROBE f 186245 s 196526 m 198168 p 205864
```

(`ref_score_min_micro` / `ref_score_max_micro` are printed by the sealed
harness's golden-model generator in every iteration, not by the agent's own
code; they are not related to the schedule experiments.)

Agent-transcript PROBE mentions (not simulator output, but the agent
describing its own instrumentation plan):
- agent_01.txt: "探针：`rdcycle` 记录 9 次发射返回时刻与每个间隙 Saturn 结束时刻…末尾打印两行 `PROBE iss …` / `PROBE gap … f <fence 完成>`…"; "…`PROBE iss` 各发射间距会直接给出是哪一种。"
- agent_04.txt: "…末尾打印 2 行 PROBE。"; "…用 ~18k printf 换取…真实拆分。"
- kernel_01.h: `// mbus time its 73 KiB of reads steal, minus PROBE printf cost).`; `// Timestamps (rdcycle) for the PROBE lines printed at the end of the kernel.`; `// PROBE: cycle offsets from kernel entry. One line per array, decimal.`
- kernel_04.h: `// CHANGE:     PROBE iteration: strictly sequential baseline schedule (with`; `//             each Saturn phase, printed as two PROBE lines at the end.`; `// PROBE SCHEDULE (this iteration): strictly sequential, like the baseline,`

### Run `20260912-191830-e713`

**iter 1** (`simlog_01.txt`):
```
PROBE rows_per_mvin=3
PROBE inflight=6
PROBE flush_cycles=274021
PROBE cold_mvin_cycles=166067
PROBE cold_bytes_per_cycle_x100=631
PROBE rows_per_mvin=4
PROBE inflight=8
PROBE flush_cycles=276279
PROBE cold_mvin_cycles=158109
PROBE cold_bytes_per_cycle_x100=663
PROBE rows_per_mvin=5
PROBE inflight=10
PROBE flush_cycles=276368
PROBE cold_mvin_cycles=173029
PROBE cold_bytes_per_cycle_x100=606
PROBE rows_per_mvin=6
PROBE inflight=12
PROBE flush_cycles=276702
PROBE cold_mvin_cycles=179649
PROBE cold_bytes_per_cycle_x100=583
PROBE rows_per_mvin=8
PROBE inflight=16
PROBE flush_cycles=276693
PROBE cold_mvin_cycles=181520
PROBE cold_bytes_per_cycle_x100=577
PROBE rows_per_mvin=16
PROBE inflight=16
PROBE flush_cycles=276716
PROBE cold_mvin_cycles=184298
PROBE cold_bytes_per_cycle_x100=568
```

**iter 2** (`simlog_02.txt`):
```
PROBE warm_gemv_cycles=132435
PROBE warm_bytes_per_cycle_x100=791
PROBE passthru_loop_issue_cycles=38
PROBE passthru_64mvin_accept_cycles=15126
PROBE passthru_loop_plus_mvins_total_cycles=15333
PROBE flush_cycles=274464
PROBE cold_gemv_cycles=184837
PROBE cold_bytes_per_cycle_x100=567
```

Agent-transcript / kernel-header PROBE mentions:
- agent_01.txt: "以 `PROBE ` 前缀逐行 printf。"; "…有价值的数字是各 PROBE 行。"; "下一轮请读取 `PROBE cold_mvin_cycles` 各行…"
- agent_02.txt: "打印 `PROBE cold_gemv_cycles`。"
- kernel_01.h: `// ROUND 7 -- PROBE ITERATION (the numbers matter more than the score).`; `//             flush stream, timed with rdcycle, printed as "PROBE ..." lines.`; `//             ~ 2.2M cycles.  The answer is in the PROBE lines, not the total.`; `// ROUND-7 PROBE SECTION.  Runs after the scored kernel, inside the timed`
- kernel_02.h: `// ROUND 7, ITERATION 2 -- PROBE ITERATION (the numbers matter, not the score).`; `//             Key line: PROBE cold_gemv_cycles ~ 184,600.`

---

## 5. Next steps for iter5-iter8 (`llama-layer-fused-n1`)

Grounded in `kernels.py`'s `_LAYER_FUSED_NOTES` (roofline 140,680; baseline
206,304; "quote the baseline, not the roofline"; "descending K order may no
longer be worth anything here — it is one line… flipping it is a clean, cheap
control experiment") and the iter1-4 results above:

1. **iter5 — retest descending-vs-ascending K order now that we know the
   real cold rate is 5.68 B/cycle, not 7.0-7.9.** `kernels.py` flags this as
   an untested "one-line experiment" once the L2 is genuinely cold — nobody
   in this run touched it (all 4 iterations kept K descending). Start from
   the 189,851 best (iter 2's schedule: fence after loop_ws#9, gaps 1-4 =
   QK^T, gap 5 = softmax, gap 8 = probs@V) and just flip K to ascending.
   Given the e713 finding that warm-tail credit was previously
   overestimated (17k assumed vs ~52k actual, but under a *different*,
   fully-cold harness that never gets any residue), the expected effect
   here is small, but it's a cheap, high-information control the round
   hasn't run.

2. **iter6 — apply the e713 4-row-mvin insight to `lf_gemv_gemmini`.** e713's
   iter 2 found the winning kernel's B-mvin shape (16 rows/mvin, matching
   `loop_ws`'s native shape) runs at only 5.68 B/cycle cold, while 4
   rows/mvin (8-in-flight, the knee of the in-flight-depth curve from e713
   iter 1) hits 6.63 B/cycle — worth an estimated ≈15k cycles on ≈600 KiB of
   B, per agent_02.txt's own arithmetic, *if* `LoopMatmul`'s hardware loop
   can be re-tiled to mvin B at 4 rows/call instead of 16 while keeping the
   same `ex`/accumulator shape. This is the single largest un-exploited
   lever surfaced across both runs and hasn't been tried anywhere in
   `llama-layer-fused-n1`. Caution: this changes `lf_gemv_gemmini`'s inner
   tiling, which the kernel header explicitly says not to do ("do not
   rewrite the four kernels' inner loops... change the SCHEDULE") — treat
   this as a candidate that needs sign-off against that constraint, or frame
   it as changing only the mvin granularity within the same weight-stationary
   structure rather than the loop nest itself.

3. **iter7 — retest the granularity of the QK^T split now that iter 4 has a
   trustworthy phase breakdown.** iter 4 shows scores=10,281, softmax=1,642,
   pv=7,696 cold-uncontended (vs `kernels.py`'s assumed warm total of
   14,602) — the Saturn half is bigger than assumed once genuinely cold.
   `kernels.py` suggests "splitting QK^T into 8 chunks of 64 rows (~1,030
   cycles each)" as an alternative to iter 1-2's 4×128-row split; given iter
   3 showed the Zicbop burst is a cold NOP (so there's no burst/Saturn-work
   competition for gap budget), try finer-grained QK^T chunks (8×64 instead
   of 4×128) spread across more of the 8 available gaps, freeing gap
   real estate to move some softmax/pv work earlier rather than jamming it
   all into the last 1-2 gaps. Since iter 2's placement change only bought
   636 cycles (contention cost is additive/position-independent per iter 3),
   the marginal value of pure re-scheduling looks close to exhausted — this
   iteration should be treated as a low-expectation confirmatory check, not
   a big swing.

4. **iter8 — measure K8/V8/kscale prefetch-early, per `kernels.py`'s own
   suggestion**: "Prefetching K8 and V8 *early*, while the first weight tile
   is still warm from nothing, is a legitimate idea." Given iter 4's
   breakdown shows the Saturn tail (scores+softmax+pv) is 19,619 cycles —
   larger than the docs' assumed 14,602 — and agent_03's finding that Saturn
   reads interleaving with the weight stream cause real page-switch stalls,
   try issuing the K8/V8 reads (Zicbop hints or small mvins) in gap 0/1
   (the earliest gaps, before any Saturn compute needs them) instead of
   only right before each compute phase, to get their ~72 KiB moved while
   the weight stream itself is least likely to be disrupted. Combine with
   whatever K-order iter5 settles on.

Overall: the layer-fused run has essentially exhausted cheap scheduling wins
(iter 2's placement change was worth only 636 cycles, iter 3's prefetch-removal
was a wash) and 189,851 (−8.0% vs 206,304) looks close to a local optimum for
"just reorder around a single fence." The two most promising remaining paths
are (a) the descending/ascending-K control experiment nobody ran, and (b) the
4-row-vs-16-row mvin retiling insight borrowed from the sibling `e713` run,
which is a genuinely new mechanism worth ≈15k cycles if it can be applied
without violating the "don't rewrite inner loops" constraint.
