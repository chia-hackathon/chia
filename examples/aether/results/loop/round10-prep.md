# Round 10 prep — `llama-lmhead-fused-n1`

Status: kernel registered, harness built, baseline measured (numbers below).
**Round 10 is NOT launched.** `out/loop/round10-launch.sh` and
`out/loop/round10-watchdog.sh` are written and syntax-checked, waiting for the
operator.

## Why this kernel exists

Round 9 closed `llama-layer-fused-n1`: 206,304 sequential -> 179,033
overlapped (-13.2%), decomposing into

    pure weight stream, Saturn serialised out   161,687
    Saturn cost serialised                       28,372
    Saturn cost still EXPOSED                    17,346
    exposure ratio  17,346 / 28,372            =  0.611

`loop/llama_project.py --overlap` extrapolates 0.611 to the whole decode step,
which is worth only 0.29-1.43% end to end because the fused kernel's Saturn
share (13.8%) is 6-20x a real decode token's (0.7-2.2%). That much is
arithmetic and it is already in the paper.

The **validity threat** is that 0.611 was measured with ATTENTION as the
companion work — 64 KiB of KV cache, 28k cycles, arithmetic-shaped, hiding
inside a 1 MiB stream. The largest Gemmini phase of a decode step is the LM
head: ~20.5% of decode cycles, 2048 x 128256 int8 = ~250 MiB for one token,
and **no attention anywhere near it**. The only concurrent work is the final
RMSNorm, the int8 quantisation of the next hidden state, and the running
argmax over the logits. There is no measurement either way, and the arithmetic
predicts 0.611 fails there.

This entry is that measurement. **A measured exposure of ~1.0 is a publishable
negative result** and closes the "overlap extrapolates" threat for good.

## Design

One timed region, COLD (2 MiB streamed through the 512 KiB L2 with Gemmini's
own DMA immediately before `read_cycles()`):

| piece | shape | seeded from | its own best, WARM |
|---|---|---|---|
| Gemmini LM-head GEMV | `1 x 2048 x 2048` int8, **4 MiB** of B | `llama-q8-gemv-gemmini-lmhead` Round-4 winner (hand-issued, 4-row mvin2, coprime chunk permutation, descending K) | 634,507 (6.61 B/cycle) |
| final RMSNorm | 2048 fp32 | `llama-rmsnorm` incumbent (RVV LMUL=1, never beaten) | 5,454 |
| int8 quantise | 2048 fp32 -> int8, RNE, clamp +-127 | new (asm: `vfmul.vf` / `vfcvt.x.f.v` / `vmax.vx` / `vmin.vx` / 2x `vnsrl.wi`) | — |
| running argmax | 2048 int32, lowest-index tie-break | new (asm: `vmax.vv` + `vredmax.vs`, then `vmseq.vx` + `vfirst.m`) | — |

### Timed region

`read_cycles()` ... `llama_lmhead_fused(...)` ... `read_cycles()`, printed as
`Cycles taken: N`. Everything is inside: the GEMV, all three Saturn phases,
and the final fence. Data generation, the golden models, the L2 flush and the
self-checks are all outside.

### Scaling, and why it is deliberately generous to Saturn

The real head is 2048 x 128256, so this is ONE 2048-column tile = 1/62.6 of it.

* the **running argmax is tile-matched and faithful**: a decode step computes
  the head tile by tile, and the natural software pipeline folds tile t-1's
  2048 logits into the running max while tile t's weights stream. `L = 2048`
  per 4 MiB of B is exactly the true ratio.
* the **RMSNorm and quantiser happen once per TOKEN, not once per tile**.
  Charging a whole token's worth of them to a single tile over-attributes the
  Saturn side by **62.6x**. That is on purpose: whatever exposure ratio comes
  out is an **UPPER BOUND on overlap opportunity**, so a near-1.0 result is
  conclusive rather than suggestive.

A full-vocabulary argmax (128,256 int32 = 513 KiB) was considered and
rejected: it would add 12.5% to the byte count and turn the experiment into a
bandwidth-contention measurement rather than an overlap measurement. Keeping
the Saturn side compute-shaped is what makes this ratio comparable with 0.611.
Bandwidth contention is a separate question and deserves its own entry.

### Independence (the thing that makes the benchmark mean anything)

If the argmax read `C`, or the GEMV read `xq`, a true data dependency would
serialise the halves and no schedule could ever win. Both are avoided the same
way `llama-layer-fused-n1` avoids quantising `probs` inside its timed region —
by supplying the operands separately, and the separation is physically honest:

* `LG` is a pre-supplied int32 logit vector standing for the **previous
  tile's** output. That is what a real streaming argmax consumes.
* `A` is the **current** token's already-quantised hidden state (an input);
  `xq` is the **next** one (an output). The rmsnorm/quantise pair never
  produces `A`.

The only dependency inside the region is `xn -> xq`.

### Self-check — four gates plus three checksums

1. **GEMV**: 8 sampled outputs (`j` = 0, 1, 15, 16, 17, 31, 32, 2047 —
   straddling the `DIM = 16` tile boundaries and including the last column),
   recomputed with a scalar int32 dot product in ONE fused pass over k,
   compared **exactly**.
2. **RMSNorm**: all 2048 against a double-precision scalar reference,
   1e-4 relative / 1e-5 absolute — the same gate `llama-rmsnorm` uses.
3. **int8 quantisation**: all 2048, two gates. Per element at most 1 LSB (the
   reference rounds half away from zero in double, RVV's `vfcvt.x.f` rounds
   half to even in fp32, so exact ties may legitimately differ by one); in
   aggregate at most 8 LSBs over the whole vector, which catches a systematic
   off-by-one or a wrong scale that the per-element gate would pass.
4. **running argmax**: value AND index compared **exactly**. The generator
   forces the maximum to be unique (every duplicate of the max is pushed one
   below it), so there is one right answer and no tolerance is warranted.

Plus, printed every run: `C checksum` over all 2048 int32 outputs,
`xn sum_micro`, `xq checksum`, and the argmax pair. A kernel that writes only
the sampled GEMV columns, or only half of `xn`, is caught by those even though
gate 1 is sampled.

`MISMATCH` + `exit(1)` on any failure, which scores the attempt zero.

### Roofline — `max`, not sum, because the two resources are independent

    MEMORY FLOOR — everything crosses the one 8 B/cycle mbus
      B weights (every byte exactly once)      4,194,304 B
      A  1 x 2048 int8                             2,048 B
      C  out 2048 int32 (+ write-allocate)        16,384 B
      X  2048 fp32                                 8,192 B
      G  2048 fp32                                 8,192 B
      xn store 2048 fp32 (+ write-allocate)       16,384 B
      xq store 2048 int8 (+ write-allocate)        4,096 B
      LG 2048 int32                                8,192 B
      amax (one line, write-allocate)                 64 B
      -------------------------------------------------------
      total                                    4,257,856 B
      memory floor = 4,257,856 / 8                = 532,232 cycles

    COMPUTE FLOOR — two different units, so max of the two
      Gemmini array: 4,194,304 MAC / 256 MAC per cycle   = 16,384 c
      Saturn (DLEN = 128 -> 4 fp32 or 4 int32 lanes/cycle):
          rmsnorm   reduce 512 + scale 1,024            =  1,536 c
                    (exactly `llama-rmsnorm`'s own roofline)
          quantise  6 element-ops x 2048 / 4            =  3,072 c
          argmax    2 passes x (load 512 + op 512)      =  2,048 c
          Saturn total                                  =  6,656 c
      compute floor = max(16,384, 6,656)                = 16,384 cycles

    roofline = max(532,232, 16,384) = 532,232 cycles — MEMORY binds by 32x.

Two consequences:

a) **In principle the Saturn work is free**: 6,656 cycles of arithmetic
   against 532,232 cycles of unavoidable DMA, 80x more shadow than work.
b) **The weight stream is 98.5% of the bytes.** The Saturn side's 63,552 B is
   1.5% of the traffic (~7,900 cycles of mbus time). Unlike
   `llama-layer-fused-n1` (6.5% of bytes), byte displacement is second-order
   here. If the Saturn cost stays exposed, it will not be because of bytes.

532,232 is unreachable: the measured rate for this exact 4 MiB stream is
6.61 B/cycle WARM and this harness runs cold.

## Baseline

**648,292 cycles, self-check PASSED**, run `20260917-145930-bf8a`
(2026-09-17, `--baseline-only`, `GENV256D128GemminiShuttleConfig`).
DB row: `iters(run_id='20260917-145930-bf8a', iter=0, passed=1)`.

Wall time for one measurement: **3,102 s (51.7 min)** end to end — Ray
connect, Gemmini collateral regeneration (~2.5 min), ELF build, boot, the
4 MiB + 2 MiB + 24 KiB data fill, the 2 MiB DMA L2 flush, the 648k-cycle timed
region, four golden models and four checks. `sim_timeout_seconds` is set to
**9000** (~2.9x margin, the same margin `llama-layer-fused-n1` carries).

Cheaper to simulate than `llama-layer-fused-n1` (3,434 s) despite 4x the
weights, because that entry's Saturn golden models are ~82k scalar MACs and
these are ~6k. Four iterations plus a baseline is roughly 4.5 hours of
simulator time.

| what | cycles |
|---|---|
| **BASELINE, strictly sequential, cold** | **648,292** |
| roofline (memory floor, 8 B/cycle) | 532,232 (1.218x) |
| the SAME 4 MiB GEMV alone, WARM, its own harness | 634,507 (6.61 B/c) |
| `llama-rmsnorm` alone, warm, its own harness | 5,454 |
| quantise + argmax alone | never measured |

Effective rate **6.47 B/cycle** against 6.61 B/cycle warm.

### The number that caps Round 10 before it starts

    648,292 (fused, cold, sequential)
  - 634,507 (weight stream alone, WARM)
  = 13,785  <- EVERYTHING else: the cold-L2 penalty on the stream itself,
               PLUS all three Saturn phases, PLUS the fence.

**Perfect overlap is worth at most 13,785 cycles (2.1%), and strictly less,**
because part of that 13,785 is bytes that must cross the mbus and no schedule
can recover them. Compare `llama-layer-fused-n1`, where the Saturn half alone
was 28,372 cycles on a 206,304-cycle baseline (13.8%). Here the whole
non-stream remainder is 2.1%.

That asymmetry is not a disappointment — **it is the finding**, and it is
visible before a single optimisation iteration. The companion work of a decode
step does not grow with the weight tile, so at LM-head scale there is almost
nothing to hide. Expect the exposure ratio to come out high, and expect the
honest conclusion to be that 0.611 is a property of attention-shaped companion
work and does not extrapolate to the 20.5% of decode cycles the LM head owns.

Round 10's job is to measure the split precisely and state the mechanism, not
to chase 13,785 cycles. Iteration 1's `PROBE stream=` is what splits the
13,785 into "cold penalty on the stream" (irreducible) and "serialised Saturn"
(the exposure denominator).

### A secondary result already in hand

The cold-L2 penalty at 4 MiB is small: 648,292 - 634,507 = 13,785 *total*,
whereas the 1 MiB GEMV paid 52,402 cycles for the same flush
(132,435 warm -> 184,837 cold). That is what the arithmetic predicts — the
warm tail a 512 KiB L2 can hold is at most 12% of 4 MiB but was 25-50% of
1 MiB — and it means the descending-K ordering trick is worth much less here
than the 17,104 cycles it was worth in the warm 1 MiB harness. Useful for the
paper independently of the overlap question: **the warm-tail benchmark
artifact shrinks as the weight tile grows, so the LM-head numbers on this board
are the least contaminated ones.**

## Files

* `repos/gemmini/software/gemmini-rocc-tests/include/llama_lmhead_fused_body.h`
  — SEALED harness body (main, generator, L2 flush, golden models, 4 checks).
* `repos/gemmini/software/gemmini-rocc-tests/bareMetalC/llama_lmhead_fused_n1.c`
  — SEALED wrapper; pins `LH_N = 1`, `LH_M = 2048`.
* `repos/gemmini/software/gemmini-rocc-tests/include/llama_lmhead_fused_n1_kernel.h`
  — the agent-owned file; the strictly-sequential baseline.
* `loop/kernels.py` — `LLAMA_LMHEAD_FUSED_N1` entry (`_LMHEAD_FUSED_CONTRACT`,
  `_LMHEAD_FUSED_NOTES`), added to `KERNELS`. Backup of the previous file:
  `loop/kernels.py.bak-round10prep`.
* `cluster/install-gemmini-kernels.sh` — the three new files added to `FILES`.
  **This must be re-run after any `chia up`**, or the build fails.

## Round 10 plan (4 iterations, in the notes)

1. **Establish the denominator.** Timestamp the phase boundaries inside the
   timed region of the unchanged baseline and print, in exactly
   `llama-layer-fused-n1` iter6's format:

       PROBE stream=NNNNNN rms=NNNNN quant=NNNNN argmax=NNNNN

   Then `exposure = best_total - stream`,
   `serialised Saturn = rms + quant + argmax`, and the ratio is the
   deliverable. This iteration costs nothing in score and turns one number
   into five.
2. **Move the fence.** Drop `gemmini_fence()` from the end of the GEMV,
   interleave the Saturn phases into the command stream (a `lhk_sat_unit` call
   every N mvin2 issues — the structure `llama-layer-fused-n1` converged on),
   fence once at the very end.
3. **Load distance** — issue the next Saturn unit's vector loads during the
   current one, so the window the in-order core is held for covers arithmetic
   rather than cold-miss latency. This is the one surviving model from Round 9
   and has never been tested anywhere.
4. **Ascending vs descending K**, cold. One line; disentangles the warm-tail
   credit from the schedule.

Retired-hypothesis list (do not re-test): Gemmini-DMA prefetch of the Saturn
operands, finer interleave grain, cutting host scalar operands, 8/16-row
mvins, Saturn taking a slice of the GEMV, `gemmini_loop_ws`. All six are in
the registry notes with their measured numbers.

## Traps hit in previous rounds — still live

* `cache.baseline_key()` hashes only the AGENT-OWNED file, not the sealed
  body. After editing `llama_lmhead_fused_body.h` you must pass `--no-cache`
  or delete the matching `baseline-*.pkl`, or you will be handed a stale
  number. (`--no-cache` also forces a full simulator rebuild — prefer deleting
  the pkl.)
* After `chia up`, re-run `cluster/install-gemmini-kernels.sh` or the build
  fails with a missing header.
* `/share1/saves/max410011/aether_ray` is >95% full (~6.2 GB free). Ray logs a
  warning every 10 s and object spilling would fail. Worth clearing before a
  4-iteration round.
