# Inner-loop iteration journal

> **Data sources & limitations**
> - Companion to `ledger.py` (which reports per-round/per-run only): one row
>   per `(run_id, iter)`, joining `out/loop/rounds.json`, `loop/aether.db`
>   (`runs`+`iters`), and `loop/kernels.py` (`roofline_cycles`, read-only).
> - Cost and cycle counts are always read from `iters` directly (`cost_usd`,
>   `fitness`/`mcycle`), never from `runs`'s rollup columns — a run that died
>   mid-flight never got its `finish_run()` rollup (same convention as
>   `ledger.py`).
> - `iters` has no per-iteration timestamp. Rows are ordered by
>   `(round start time, run's runs.started, iter number)` rather than by
>   on-disk kernel-file mtime (ledger.py's approach) — coarser, but immune to
>   mtimes going stale if a run directory is ever touched/copied after the
>   fact. No per-iteration duration is reported here for that reason.
> - The "agent note" column is best-effort text mined from `agent_NN.txt`
>   (preferred) or the `kernel_NN.h` header comment (fallback) via a
>   HYPOTHESIS-marker regex — not a structured DB column. Blank means neither
>   file existed or neither contained a recognizable HYPOTHESIS marker.
> - Regenerate with: `python loop/journal.py --rounds-file out/loop/rounds.json`


## smoke — smoke test of the inner loop before round1
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-softmax | 20260906-193727-ebbf | 0 | ok | 21,906 | 1.000x | 14.882x | ★ | $0.0000 |  |
| llama-softmax | 20260906-193727-ebbf | 1 | ok | 15,026 | 0.686x | 10.208x | ★ | $0.4301 | HYPOTHESIS: At LMUL=1 every vector op is a 2-beat chime but the FMA pipe is 4 stages deep, so the ~40-op fully dependent exp chain per strip stalls on every op; measured ≈ 342 cycles per 8-element... |
| llama-softmax | 20260906-193727-ebbf | 2 | ok | 4,312 | 0.197x | 2.929x | ★ | $0.2194 | Assembles cleanly with the `vfdiv` gone (one `vfmul.vf` in pass 3, scalar `fdiv` once); the LMUL=4 best kernel is otherwise untouched. |

**smoke summary** — iterations: 3, pass rate: 3/3 (100.0%), total cost: $0.6494

## round1 — config GENV256D128GemminiShuttleConfig, model claude-fable-5-1 (see out/loop/round1-runs.txt)
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 0 | ok | 150,339 | 1.000x | 1.147x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 1 | ok | 132,653 | 0.882x | 1.012x | ★ | $2.0505 | I'll start by loading the remote tool and reading the relevant parts of `gemmini.h`.The kernel compiles cleanly with the harness flags (12 RoCC custom instructions emitted, no errors). Tile... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 2 | build_failed | ? | n/a | n/a |  | $1.7869 | **Last turn's miss:** I predicted 115–125k and got 132,653. That number is telling: 1 MiB / 8 B per cycle = 131,072, and we landed 1.2% above it. The L2-residency trick and the sequential sweep... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 3 | ok | 134,585 | 0.895x | 1.027x |  | $0.8477 | The harness's real compile line evidently lacks `v` in `-march` (the header's `#error` fires), so I can't use C intrinsics. I'll emit the vector code as inline asm with `.option arch, +v`, and check... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 4 | ok | 133,800 | 0.890x | 1.021x |  | $0.5067 | **Last turn's miss:** predicted 116–118k, measured 134.6k (+1.9k vs v1) even though Gemmini moved only 7/8 of B. The self-check passed, so the CPU slice computed correctly — it was just slow. Two... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 5 | ok | 132,653 | 0.882x | 1.012x |  | $0.7453 | **Last turn's miss:** predicted ~124.5k under model (a), got 133.8k — model (a) is refuted. The CPU slice cannot have been the critical path this time (v3 bounded the vector path at ≤526 cycles/row,... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 6 | ok | 149,757 | 0.996x | 1.143x |  | $0.3552 | **Last turn's EXPECTED held** (v1 re-measured at exactly 132,653 — the machine is deterministic, so any change ≥ ~100 cycles is real signal, not noise).  Since another edit is requested, I'll use it... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 7 | ok | 142,221 | 0.946x | 1.085x |  | $0.8704 | **Last turn's EXPECTED missed badly — and that miss is the most informative result so far.** Ascending order cost **+17,104 cycles** (149,757 vs 132,653), with identical loops, bytes and mvouts. So... |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 8 | ok | 132,653 | 0.882x | 1.012x |  | $1.5833 | **Last turn's EXPECTED missed** (predicted 126–132.5k, got 142,221 = **+9,568**). Prediction said ">133.5k means the core's misses actively interfere" — they do, and badly: ~8 windows × ≤6k cycles of... |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 0 | ok | 51,280 | 1.000x | 8.346x | ★ | $0.0000 |  |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 1 | ok | 12,729 | 0.248x | 2.072x | ★ | $1.4631 | The sealed header isn't present locally; I'll assemble-check with a stub header.Macro-parameter name collision with the asm operand name; renaming the parameter.Kernel written and it compiles cleanly... |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 2 | ok | 10,472 | 0.204x | 1.704x | ★ | $0.5490 | **Last turn's miss:** I predicted 5.5–7k and got 12,729 ≈ 99 cycles per 4-row block. My block issues 64 arithmetic beats (8×4 int16 + 4×8 int32) plus ~11 beats of reduce, so the evidence says the... |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 3 | ok | 10,008 | 0.195x | 1.629x | ★ | $0.3536 | Last turn's prediction (~10k) landed: 10,472 ≈ 82 cycles/block against a model of 64 arithmetic beats + ~11 reduce beats per block, so the kernel is now arithmetic-pipe-bound with the loads hidden —... |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 4 | ok | 8,574 | 0.167x | 1.396x | ★ | $0.4238 | Last turn saved 464 cycles, not the 1,024 I predicted: the kernel is at 78 cycles/block against 56 arithmetic + 10 reduce beats, so ~12 cycles/block are stalls, not beats. That tells me the... |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 5 | ok | 10,081 | 0.197x | 1.641x |  | $0.5372 | Last turn's prediction (~8,900) was beaten: 8,574 ≈ 67 cycles/block, which matches the beat model exactly (32 mul + 24 int32-add + ~10 reduce = 66). The kernel is now at the floor of a **single... |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 6 | ok | 10,522 | 0.205x | 1.713x |  | $0.5746 | Last turn's result (10,081, a regression) confirms the second branch of my prediction: on this Saturn config **all** vector arithmetic — integer ALU, multiplier, FP, conversions — shares one in-order... |
| llama-attn-pv-int8 | 20260907-043140-2246 | 0 | ok | 22,856 | 1.000x | 5.580x | ★ | $0.0000 |  |
| llama-attn-pv-int8 | 20260907-043140-2246 | 1 | ok | 9,803 | 0.429x | 2.393x | ★ | $0.5413 | HYPOTHESIS: The per-row `vle32`/`vse32` of the int32 accumulator costs 2 strips × (16+16 beats on load/store pipes) per row and, worse, creates a load→vwadd→store→load dependency through memory every... |
| llama-attn-pv-int8 | 20260907-043140-2246 | 2 | ok | 6,276 | 0.275x | 1.532x | ★ | $0.2705 | Last turn's EXPECTED (12.5k single-pipe / 8.4k overlapped) bracketed the measured 9,803: 19.1 beats/row is below the 24-beat serialized figure, so the `vwmul` (8 beats) and `vwadd.wv` (16 beats)... |
| llama-attn-pv-int8 | 20260907-043140-2246 | 3 | ok | 4,999 | 0.219x | 1.220x | ★ | $0.2517 | Last turn's EXPECTED (5,800) vs measured 6,276: 24.5 cycles per pair, when the mul pipe (8+8 beats) and add pipe (16 beats) should have overlapped to 16. What that says about the machine: the single... |
| llama-attn-pv-int8 | 20260907-043140-2246 | 4 | ok | 4,992 | 0.218x | 1.219x | ★ | $0.6807 | Last turn's EXPECTED (4,600) vs measured 4,999: 39 cycles per 4-row block against a 32-beat pipe model — the pipes now overlap as intended, but ~7 cycles/block are still not explained by vector... |
| llama-attn-pv-int8 | 20260907-043140-2246 | 5 | ok | 8,508 | 0.372x | 2.077x |  | $0.6706 | Iter 4 taught me something important: 4,992 vs 4,999 — two different unroll depths / ring depths land on the **same 9.75 cycles per row**, so the register-hazard model was wrong and the bound is... |
| llama-attn-pv-int8 | 20260907-043140-2246 | 6 | ok | 4,853 | 0.212x | 1.185x | ★ | $0.3507 | Iter 5 taught two things: (1) scalar `lbu` misses on Shuttle are **not** free — they cost ~7 cycles each (≈+3,500), so scalar prefetch is off the table; (2) pulling lines into L1/L2 ahead of time did... |
| llama-silu-mul | 20260907-043150-7557 | 0 | ok | 361,260 | 1.000x | 11.760x | ★ | $0.0000 |  |
| llama-silu-mul | 20260907-043150-7557 | 1 | ok | 289,169 | 0.800x | 9.413x | ★ | $0.4764 | The directory isn't present on the kernel host — let me locate where the workspace actually lives.(The one remaining "m8" match is just the comment text.) The kernel is now the pristine algorithm at... |
| llama-silu-mul | 20260907-043150-7557 | 2 | ok | 91,474 | 0.253x | 2.978x | ★ | $0.2532 | Last turn's EXPECTED (~100k) missed badly: 289k cycles is ~35 cycles per element, while the whole Cephes chain at LMUL=4 is only ~30 ops × 8 beats / 32 elem ≈ 7.5 cycles/element. So the arithmetic... |
| llama-silu-mul | 20260907-043150-7557 | 3 | ok | 79,914 | 0.221x | 2.601x | ★ | $0.3904 | Last turn's EXPECTED (~95–100k) landed at 91.5k — the model is now confirmed: the iterative divider was ~200k cycles, and what remains is 357 cycles per 32-element strip against ~40 vector ops × 8... |
| llama-silu-mul | 20260907-043150-7557 | 4 | ok | 66,820 | 0.185x | 2.175x | ★ | $0.5713 | Last turn: EXPECTED ~77k, measured 79.9k — the beat model holds (312 cycles/strip vs 32 ops × 8 = 256 beats + ~56 of load/store/loop overhead that isn't fully hidden). Every arithmetic op removed is... |
| llama-silu-mul | 20260907-043150-7557 | 5 | ok | 66,811 | 0.185x | 2.175x | ★ | $0.3050 | Last turn: EXPECTED ~66k, measured 66.8k — model holds. The loop is now 28 arithmetic ops × 8 beats = 224 of the 261 cycles/strip; the remaining ~37/strip is load/store/loop overhead I'll look at... |
| llama-silu-mul | 20260907-043150-7557 | 6 | selfcheck_failed | 301 | 0.001x | 0.010x |  | $0.3121 | Last turn's EXPECTED (−2k) missed: removing `vfmv.v.f` + `vfsub`→`vfrsub` saved only 9 cycles. That tells me the broadcast wasn't occupying the FP pipe (it goes to the integer/permute side,... |
| llama-softmax | 20260907-043200-f0d7 | 0 | ok | 21,906 | 1.000x | 14.882x | ★ | $0.0000 |  |
| llama-softmax | 20260907-043200-f0d7 | 1 | ok | 2,174 | 0.099x | 1.477x | ★ | $1.3040 | Numerics verified in fp32 emulation: max relative error 6.4e-6 over [-20, 0] (16x under the 1e-4 gate). Assembles cleanly; the pass-2 loop has no whole-register spills.  Summary: replaced the LMUL=1,... |
| llama-softmax | 20260907-043200-f0d7 | 2 | ok | 1,975 | 0.090x | 1.342x | ★ | $0.5782 | Last turn's EXPECTED (~3,200) was pessimistic: the measured 2,174 matches the pure beat count almost exactly — 14 arithmetic ops × 8 beats × 16 groups = 1,792, plus 2 × 128 for passes 1/3, plus ~130... |
| llama-softmax | 20260907-043200-f0d7 | 3 | ok | 2,103 | 0.096x | 1.429x |  | $0.6466 | Assembles cleanly; the loop body is now 10 arithmetic vector ops (the `vsll`+`vadd` pair became one `vmacc.vx`), and the fp32 emulation confirms identical numerics (worst rel error 7.3e-6).  Summary:... |
| llama-softmax | 20260907-043200-f0d7 | 4 | ok | 1,752 | 0.080x | 1.190x | ★ | $0.4099 | Assembles cleanly: the `call rintf` is gone, replaced by `fcvt.w.s rne` + `fcvt.s.w` (numerics identical — same round-to-nearest-even integer).  Summary: removed the newlib `rintf` call from the... |

**round1 summary** — iterations: 35, pass rate: 33/35 (94.3%), total cost: $20.6599

## round2 — first round2 launch, aborted mid-run (session limit); see out/loop/round2-runs.txt
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260907-142155-86bf | -1 | ok | 132,653 | 0.882x | 1.012x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260907-142155-86bf | 0 | ok | 150,339 | 1.000x | 1.147x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260907-142155-86bf | 1 | selfcheck_failed | 134,019 | 0.891x | 1.022x |  | $2.7854 | Kernel written to `/tmp/aether-kernel/include/llama_q8_gemv_gemmini_n1_kernel.h`; it compiles cleanly under `-march=rv64gcv_zfh_zvfh -O2` with no vector-register spills and a tight inner loop... |
| llama-attn-scores-int8 | 20260907-142205-6c45 | -1 | ok | 8,574 | 0.167x | 1.396x | ★ | $0.0000 |  |
| llama-attn-scores-int8 | 20260907-142205-6c45 | 0 | ok | 51,280 | 1.000x | 8.346x |  | $0.0000 |  |
| llama-attn-pv-int8 | 20260907-142215-b093 | -1 | ok | 4,853 | 0.212x | 1.185x | ★ | $0.0000 |  |
| llama-attn-pv-int8 | 20260907-142215-b093 | 0 | ok | 22,856 | 1.000x | 5.580x |  | $0.0000 |  |
| llama-silu-mul | 20260907-142225-55f2 | -1 | ok | 66,811 | 0.185x | 2.175x | ★ | $0.0000 |  |
| llama-silu-mul | 20260907-142225-55f2 | 0 | ok | 361,260 | 1.000x | 11.760x |  | $0.0000 |  |
| llama-silu-mul | 20260907-142225-55f2 | 1 | ok | 59,008 | 0.163x | 1.921x | ★ | $1.2794 | I'll start by loading the remote tools and reading the sealed harness (check formula, data range) plus the DB history, since the data range and tolerance decide which sigmoid shortcuts are legal.Bug... |
| llama-silu-mul | 20260907-142225-55f2 | 2 | ok | 50,256 | 0.139x | 1.636x | ★ | $0.4676 | Last turn's EXPECTED (42–47k) missed: measured 59,008. Per strip that is 115 cycles for 22 vector instructions × 4 beats = 88 beats, i.e. ~27 cycles of bubbles per strip — consistent with the... |
| llama-softmax | 20260907-142235-7231 | -1 | ok | 1,752 | 0.080x | 1.190x | ★ | $0.0000 |  |
| llama-softmax | 20260907-142235-7231 | 0 | ok | 21,906 | 1.000x | 14.882x |  | $0.0000 |  |
| llama-softmax | 20260907-142235-7231 | 1 | ok | 1,630 | 0.074x | 1.107x | ★ | $1.1133 | **HYPOTHESIS:** Pass 2 is 11 serialized arithmetic ops × 8 beats × 16 groups = 1408 c; pass 1/3 are 128 c each; ~90 c overhead → 1752. Two of those 11 ops (`vfmul_vf r,c4` then `vfadd_vf ,c3`) exist... |
| llama-softmax | 20260907-142235-7231 | 2 | ok | 1,760 | 0.080x | 1.196x |  | $0.3113 | Last turn's EXPECTED (1624) vs measured 1630: the model is confirmed exactly — every serialized arithmetic op in pass 2 costs 8 beats × 16 groups = 128 c, and the 1630 kernel (10 ops) is the one I'm... |
| llama-q8-gemv-gemmini-lmhead | 20260907-144414-da36 | 0 | ok | 740,101 | 1.000x | 1.412x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260907-144414-da36 | 1 | ok | 691,515 | 0.934x | 1.319x | ★ | $1.4157 | Kernel replaced and compiles cleanly (`-march=rv64gcv_zfh_zvfh -O2`, exit 0). Tiling resolved from the macros is tile_I/J/K = 1/32/15 → 4 strips × 9 `loop_ws` calls (vs. 16 strips × 3 with 128 B rows... |

**round2 summary** — iterations: 17, pass rate: 16/17 (94.1%), total cost: $7.3727

## round2b — round2 restart 2026-09-08T11:00:03+08:00 (session limit reset, seeded from --seed best); see out/loop/round2-runs.txt
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | -1 | ok | 132,653 | 0.882x | 1.012x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 0 | ok | 150,339 | 1.000x | 1.147x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 1 | ok | 132,834 | 0.884x | 1.013x |  | $1.8401 | HYPOTHESIS: The hardware loop (`LoopMatmul`) only starts a loop's execute stage after that loop's whole A/B mvin has been *issued*, and the weight preload in WS mode costs ~DIM cycles per 16x16 B... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 2 | selfcheck_failed | 134,842 | 0.897x | 1.029x |  | $1.2898 | Last turn's EXPECTED missed badly: predicted ~125k, got 132,834 (+181 = just 3 more `loop_ws` issue overheads). Lesson: the execute stage is **not** exposed at the end of the stream — the ROB tracks... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 3 | ok | 140,285 | 0.933x | 1.070x |  | $1.4548 | Only `j = 511` (the last column, chunk 7, computed *last* and read back immediately after the fence) mismatched; the seven samples in chunk 0 (computed first) were right. So the arithmetic (pairing,... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 4 | ok | 132,653 | 0.882x | 1.012x |  | $0.5628 | Last turn's EXPECTED (~124k) missed by 16k, and that miss is the result of this round. Two data points on the same K-split:  - iter 2 (host slice overlapped, combine read C too early): **134,842** -... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 5 | ok | 132,653 | 0.882x | 1.012x |  | $0.2930 | Last turn's EXPECTED was 132,653 and the measurement was 132,653 — the model is now exact, and the run also re-confirms determinism (Round 1 iter 5 got the same number). The workspace already holds... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 6 | ok | 132,653 | 0.882x | 1.012x |  | $0.3500 | Last turn's EXPECTED (132,653) matched exactly again. The file in the workspace is already the best-known kernel (iter 5 was that file, byte-for-byte), so there is nothing to restore — I am... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 7 | ok | 132,424 | 0.881x | 1.010x | ★ | $0.4856 | Last turn's EXPECTED matched exactly (third time). I said I'd stop, but the loop keeps consuming runs on the same file, so I'll spend this turn on the one mechanism that is genuinely untested and... |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 8 | ok | 132,424 | 0.881x | 1.010x |  | $0.3590 | Last turn's EXPECTED had three branches; the measurement (132,424, **−229**, the best so far and deterministic — the v1 kernel reproduced 132,653 three times) matches neither the NOP branch (would be... |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | -1 | ok | 691,515 | 0.934x | 1.319x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 0 | ok | 740,101 | 1.000x | 1.412x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 1 | selfcheck_failed | ? | n/a | n/a |  | $1.7998 | I'll load the remote tools first, then inspect the hardware loop implementation and the experiment history before proposing a change.The kernel compiles cleanly with the new prefetch path (`rc=0`,... |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 2 | ok | 706,155 | 0.954x | 1.347x |  | $0.1998 | The failure is a trap (`tohost=1337` is the bare-metal trap-handler exit, before `MISMATCH` could print), not a wrong result — almost certainly an illegal-instruction trap because `mstatus.VS` is off... |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 3 | ok | 742,166 | 1.003x | 1.416x |  | $0.5466 | Last turn's EXPECTED missed badly: the host-side prefetch *cost* ~14.6k cycles (≈6 cycles per touched line) instead of saving ~20k. Lesson: the host's L1 miss path adds no usable MLP on this SoC —... |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 4 | ok | 704,084 | 0.951x | 1.343x |  | $1.5084 | The harness only echoed the cycle count, not my `CTR` lines — let me check whether the simulator log was kept anywhere on the remote machine before deciding the diagnostic was lost.Compiles cleanly... |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 5 | ok | 659,142 | 0.891x | 1.257x | ★ | $1.7806 | +12.6k instead of −55k: the L2-resident-tail model is dead. The spill traffic (36 × 4 KiB, ~2300 lines) cost ~5.5 cycles/line and the reordered reads gained *nothing*, so the just-written tail of B... |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 6 | ok | 738,754 | 0.998x | 1.409x |  | $0.8588 | Last turn's EXPECTED (~580k) missed: the hand-issued stream gained only 32k, not 110k. What that says about the machine: 4 MiB / 659k = 6.36 B/cycle, which matches a DMA with ~16 outstanding 64 B... |
| llama-attn-scores-int8 | 20260908-025850-2356 | -1 | ok | 8,574 | 0.167x | 1.396x | ★ | $0.0000 |  |
| llama-attn-scores-int8 | 20260908-025850-2356 | 0 | ok | 51,280 | 1.000x | 8.346x |  | $0.0000 |  |
| llama-attn-scores-int8 | 20260908-025850-2356 | 1 | ok | 12,138 | 0.237x | 1.976x |  | $2.8758 | Now writing the new kernel.Kernel rewritten and assembles cleanly (`-march=rv64gcv_zfh_zvfh -O2`, no warnings): K is now fetched with plain unit-stride `vle8` (e8m8) instead of `vlseg8e8`, each row's... |
| llama-attn-scores-int8 | 20260908-025850-2356 | 2 | ok | 8,233 | 0.161x | 1.340x | ★ | $2.1737 | Best-known kernel restored with a single isolated edit: the four-row K block (and the once-only `q` load over its 4×-replicated buffer) is now fetched with `vlseg8e64` at vl=4 — 4 sixty-four-byte... |
| llama-attn-scores-int8 | 20260908-025850-2356 | 3 | ok | 8,763 | 0.171x | 1.426x |  | $1.7648 | Last turn's EXPECTED (~6,000) missed: measured 8,233, only −5 c/pair. That taught me segmented-load cost on this machine is essentially per-byte — the segmenter is *not* the binding resource, so the... |
| llama-attn-scores-int8 | 20260908-025850-2356 | 4 | ok | 10,740 | 0.209x | 1.748x |  | $1.2170 | Last turn's EXPECTED (7,300–7,700) missed: 8,763 (+8 c/pair). Hoisting kscale two blocks ahead bought nothing, so the cold `vle32` was never an exposed stall; the +2 reduce ops and +1 load per pair... |
| llama-attn-pv-int8 | 20260908-025905-5427 | -1 | ok | 4,853 | 0.212x | 1.185x | ★ | $0.0000 |  |
| llama-attn-pv-int8 | 20260908-025905-5427 | 0 | ok | 22,856 | 1.000x | 5.580x |  | $0.0000 |  |
| llama-attn-pv-int8 | 20260908-025905-5427 | 1 | ok | 4,991 | 0.218x | 1.219x |  | $1.2817 | I'll start by checking the experiment database for what Round 1 already measured, so I don't repeat known-bad ideas.The kernel directory isn't materialised yet on that machine; I'll create it and... |
| llama-attn-pv-int8 | 20260908-025905-5427 | 2 | ok | 4,967 | 0.217x | 1.213x |  | $0.5300 | **Last turn's EXPECTED missed** (4991 vs predicted ~4500; the best is 4853). Lesson: dispatching `vle8`s whose destination is still being read by a pending `vwmul` *clogs* the in-order load... |
| llama-attn-pv-int8 | 20260908-025905-5427 | 3 | ok | 6,034 | 0.264x | 1.473x |  | $1.2295 | **Last turn's EXPECTED missed again** (4967 vs best 4853). Removing 384 scalar loads made it *slower*, so scalar loads don't contend with the vector stream; together with turn 1 this says: the scalar... |
| llama-attn-pv-int8 | 20260908-025905-5427 | 4 | ok | 5,416 | 0.237x | 1.322x |  | $1.1155 | **Turn 3's EXPECTED missed badly** (6034, +9 c/iteration), and that is the most informative result so far. It tells me Saturn resolves **WAR at instruction granularity**: `vwmul v12` of the next... |
| llama-silu-mul | 20260908-025919-3d8d | -1 | ok | 50,256 | 0.139x | 1.636x | ★ | $0.0000 |  |
| llama-silu-mul | 20260908-025919-3d8d | 0 | ok | 361,260 | 1.000x | 11.760x |  | $0.0000 |  |
| llama-silu-mul | 20260908-025919-3d8d | 1 | ok | 47,173 | 0.131x | 1.536x | ★ | $0.6222 | Assembles cleanly, no spills: the loop body is now 2 loads, 15 FP-pipe ops, 4 integer-pipe ops (`vmin.vx`, `vminu.vx`, `vsll.vi`, `vadd.vv`), 1 store — the `t` clamp moved off the FP pipe onto... |
| llama-silu-mul | 20260908-025919-3d8d | 2 | ok | 35,795 | 0.099x | 1.165x | ★ | $1.0935 | Assembles cleanly: the loop is now 2 loads, 17 FP-pipe ops, 5 integer-pipe ops (+ one compiler `vmv.v.v` copy on the integer pipe, which has ~90 beats of slack), 1 store, 3 constant groups, no... |
| llama-silu-mul | 20260908-025919-3d8d | 3 | ok | 35,774 | 0.099x | 1.165x | ★ | $0.7382 | Last turn's EXPECTED (~38k) landed at 35,795 — the hypothesis was confirmed: `vfrec7` was executing on the iterative div/sqrt unit at ~2 cycles/element with the FP sequencer stalled behind it. The... |
| llama-silu-mul | 20260908-025919-3d8d | 4 | ok | 37,012 | 0.102x | 1.205x |  | $1.1066 | Last turn's EXPECTED missed badly: removing 8 FP beats/strip saved 21 cycles total, not ~2,000. Per-strip time stayed at 139.7 whether the FP pipe needs 136 or 128 beats — so the FP pipe is **no... |
| llama-softmax | 20260908-025933-f639 | -1 | ok | 1,630 | 0.074x | 1.107x | ★ | $0.0000 |  |
| llama-softmax | 20260908-025933-f639 | 0 | ok | 21,906 | 1.000x | 14.882x |  | $0.0000 |  |
| llama-softmax | 20260908-025933-f639 | 1 | ok | 1,760 | 0.080x | 1.196x |  | $0.7180 | HYPOTHESIS: Pass 2 is bound by the single execute sequencer: 10 arithmetic ops × 8 beats × 16 groups = 1280 of the 1630 cycles. Two of those ops (`vsll` by 23 then `vadd` into `bits(p)`) build the... |
| llama-softmax | 20260908-025933-f639 | 2 | ok | 1,624 | 0.074x | 1.103x | ★ | $0.9787 | Last turn's EXPECTED (≈1500) missed by +260: `vmacc.vx` on this config costs ~24 beats per e32m4 group, i.e. the integer multiplier is **not** DLEN-rate (roughly one third of the FMA/ALU rate).... |
| llama-softmax | 20260908-025933-f639 | 3 | ok | 1,568 | 0.072x | 1.065x | ★ | $0.4729 | Last turn's EXPECTED (−60) missed: only −6. What that says about the machine: (1) the load sequencer is **in-order** too — my three constant `vle32`s were emitted right before the loop, so the loop's... |

**round2b summary** — iterations: 41, pass rate: 39/41 (95.1%), total cost: $31.2473

## round3 — launched 2026-09-09T04:24:12+08:00 Taipei, all --seed best --budget-usd 1000; ended ~2026-09-09T17:59+08:00 (out/loop/round3-launch.sh, out/loop/round3-runs.txt, out/loop/round3-watchdog.log). llama-q8-gemv-gemmini-n1 and llama-q8-gemv-gemmini-lmhead were each interrupted by a Claude session-limit 429 (n1 after 4 iters, lmhead after 3) and restarted with the remaining planned_iters, seed=best; the watchdog's auto-restart misparsed the API's 'resets 5pm America/Los_Angeles' message as local time, so a human manually restarted both at 2026-09-09T13:23+08:00 instead of waiting for the watchdog.
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | -1 | ok | 132,424 | 0.881x | 1.010x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | 0 | ok | 150,339 | 1.000x | 1.147x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | 1 | selfcheck_failed | 779,335 | 5.184x | 5.946x |  | $3.6266 | 我先加载远端工具和实验数据库工具，看看现有测量数据能否区分"DMA 受限"还是"execute 受限"这两种模型。编译通过，文件已写入 `/tmp/aether-kernel/include/llama_q8_gemv_gemmini_n1_kernel.h`。  本轮小结：这是一次刻意的测量迭代（选项 e/f）——kernel 只做 7 段 mvin-only 扫描并把每段周期数写进 8... |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | 2 | ok | 132,424 | 0.881x | 1.010x |  | $1.0845 | 测量成功——这次"失败"本来就是设计好的测量迭代（选项 e），侧信道把 7 段计时全部带了出来。先记录结论，再把正确的最优 kernel 恢复回去。  **测量结果（mvin-only，无计算）：**  \| 段 \| 请求形状 / 顺序 / L2 状态 \| cycles \| \|---\|---\|---\| \| P1 \| stride-512 16×64 B、K 降序、harness 刚填完（与真实... |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | 3 | ok | 133,147 | 0.886x | 1.016x |  | $1.4329 | 上一轮 EXPECTED（132,424 ± 噪声）精确命中，说明仿真是确定性的、恢复的 kernel 就是最优版本。本轮我把上一轮的测量数据再往下推一层：  - mvin-only 探针 P1 = 132,109 与真实 kernel 只差 315，所以任何 execute/mvout/loop 边界/A 操作数加载的微调最多只值 315 cycles——不值得动。 -... |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | 4 | ok | 132,424 | 0.881x | 1.010x |  | $0.3178 | **HYPOTHESIS:** 上一轮 EXPECTED 未命中（预期 ~130,400，实测 133,147，比最优差 +723）。它教会我的是：L2 对 hint 的取数并不"免费"——即便在开头 DRAM 看似空转的窗口里，hint 引入的 L2 分配会随机逐出尚未读取的暖行（脏行还要写回），再加上提前两个 tile 取回的行在被用到之前又有 ~20–40%... |
| llama-q8-gemv-gemmini-lmhead | 20260908-202428-217e | -1 | ok | 659,142 | 0.891x | 1.257x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260908-202428-217e | 0 | ok | 740,101 | 1.000x | 1.412x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260908-202428-217e | 1 | ok | 738,596 | 0.998x | 1.409x |  | $0.8565 | 编译通过。改动只有一处：`LQ8_B_ROWS` 4 → `DIM`，每条 B 的 mvin 从 256 B 变成 1024 B（16 行 × 64 B），让 2 条可跟踪的 mvin 命令足以填满 16 个 DMA 请求槽；流水线结构（ring 槽、执行与加载交错、累加器整段 K 归约后一次 mvout）不变。预期 ~400k... |
| llama-q8-gemv-gemmini-lmhead | 20260908-202428-217e | 2 | ok | 729,519 | 0.986x | 1.391x |  | $0.4753 | 上一轮 EXPECTED 落空（预测 ~400k，实测 738.6k ≈ 原始 `tiled_matmul_auto` 的 740k）。教训：1024 B/命令的 16 行、2 KiB 步长的请求序列，其后端吞吐与原始 loop_ws 完全一样（5.68 B/cycle），也就是说**16 行 × 2 KiB 步长的地址模式本身**才是 740k 级别的元凶，而 4 行分组之所以好... |
| llama-q8-gemv-gemmini-lmhead | 20260908-202428-217e | 3 | ok | 810,910 | 1.096x | 1.547x |  | $1.5685 | 编译通过（警告全部来自 `gemmini.h`，我的文件无警告）。本轮改动：把 4 条 512 列条带合并为**单遍 K 扫描**——因为每个输出块仅占 1 个累加器行（C_rows=1），128 个输出块只需 128 个 acc 行，于是 B 的每个 4 行 × 8 KiB 跨度在 32 条 mvin 内被完整顺序消费（与 n1 实测 8.4 cycle/行的请求流同构）；A 打包成 128... |
| llama-q8-gemm | 20260908-202443-5906 | 0 | ok | 146,737 | 1.000x | 1.120x | ★ | $0.0000 |  |
| llama-q8-gemm | 20260908-202443-5906 | 1 | ok | 138,892 | 0.947x | 1.060x | ★ | $2.2524 | Now the compile check against the pristine `/tmp/gq8` tree (my header first on the include path):Kernel written to `/tmp/aether-kernel/include/llama_q8_gemm_kernel.h` and it compiles cleanly... |
| llama-q8-gemm | 20260908-202443-5906 | 2 | ok | 140,692 | 0.959x | 1.073x |  | $0.7604 | 上一轮的 EXPECTED（≈139,000）与实测 138,892 几乎完全吻合——"DMA 约 8 B/cycle（每个 64 B 请求约 8 cycle）、A 读两次、B 读一次"这个机器模型成立。剩余的 ~7.8k cycles 由两部分组成：冷启动前奏（ex 要等 LdA 以 i-major 顺序装完 A 的 3/4 行带 + B(0,·)，≈27 KiB ≈ 3.4k... |
| llama-q8-gemm | 20260908-202443-5906 | 3 | ok | 139,137 | 0.948x | 1.062x |  | $2.8718 | 上一轮 EXPECTED（≈136.8k）落空：实测 140,692，比最佳内核**慢** 1.8k。这否定了"冷启动等待由 LdA 的 i-major 顺序主导、且 DMA 能在后续 loop 中建立领先量"的模型。能同时解释两次测量的模型是：**loop n+1 的装载实际上只能在 loop n 的 ex 期间进行**（第 n+2 个 LOOP_WS 要等 n 号 slot 释放，其... |
| llama-q8-gemm | 20260908-202443-5906 | 4 | ok | 138,892 | 0.947x | 1.060x |  | $0.4844 | 上一轮 EXPECTED（≈136.9k）落空：结果 139,137，比最佳恰好多 245 cycles——正好是"多一个 LOOP_WS"的代价。它教会我的是：**J 块边界处并不存在被 mvout 拖延的装载停顿**，slot/装载流水在 K=32 的均匀分块下已经完全覆盖；一个 LOOP_WS 的固定代价只有 ~250 cycles（8 个 loop ≈ 2k），而不是拟合出来的... |
| llama-q8-gemm | 20260908-202443-5906 | 5 | ok | 138,892 | 0.947x | 1.060x |  | $0.3626 | 上一轮没有提出 EXPECTED（我判断已到极限，只恢复了最佳内核），测量 138,892 与最佳完全一致，确认工作区状态正确。  本轮我再次把每一个可动的项穷举了一遍，结论没有变化：  - **loop 数（8 × ~250）**：J=8 时每个 loop 的 spad 占用为 (4+8)·K·16 ≤ 8192 行 ⇒ K ≤ 42，128 个 K tile 无论如何都要 4 个... |
| llama-q8-gemm | 20260908-202443-5906 | 6 | ok | 138,892 | 0.947x | 1.060x |  | $0.2790 | 上一轮 EXPECTED（138,892，不变）命中。它再次验证：工作区已是最佳内核，且结果可精确复现（噪声为 0），所以 138,892 与更好结果之间不存在可"碰运气"的测量抖动。  本轮我又核对了最后两个可能的漏洞，均已关闭：  1. **StC 能否提前搬出？** LoopMatmul 的 StC 只需 ex 到达最后一个 k 且已越过该 tile 即可发... |
| llama-attn-scores-int8 | 20260908-202458-85ab | -1 | ok | 8,233 | 0.161x | 1.340x | ★ | $0.0000 |  |
| llama-attn-scores-int8 | 20260908-202458-85ab | 0 | ok | 51,280 | 1.000x | 8.346x |  | $0.0000 |  |
| llama-attn-scores-int8 | 20260908-202458-85ab | 1 | ok | 8,443 | 0.165x | 1.374x |  | $1.8506 | 已写入并通过汇编检查（`ASSEMBLE_OK`）。  一句话总结：本轮仅做寄存器重分配——累加器搬进各块自己的 K4–K7，块 A/块 B 各用 v24–v27 / v28–v31 作临时寄存器，`REDUCE8` 先释放 v24–v27 再在 v28–v31 中收尾——以消除让乘法管线空转的 WAR 依赖链，指令与 beat 数不变，预期约 5,700 周期。 |
| llama-attn-scores-int8 | 20260908-202458-85ab | 2 | ok | 8,474 | 0.165x | 1.379x |  | $1.3119 | 已从最佳内核出发写入并通过汇编检查（`ASSEMBLE_OK`，乘法指令的源寄存器对为 v16/v9、v18/v11、v17/v10、v19/v12……全部落在不同 bank）。  一句话总结：本轮唯一改动是把 `qrep` 右旋一个 8 字节 chunk，使 q chunk c 落在 `v8+((c+1)&7)`，从而让每条 `vwmul/vwmacc` 的两个源寄存器不再同 bank；若... |
| llama-attn-scores-int8 | 20260908-202458-85ab | 3 | ok | 18,638 | 0.363x | 3.034x |  | $0.4518 | **上一轮的教训**：bank 旋转无效（8,474，+241，其中大部分是多出的 `memcpy`/建表开销），VRF bank 冲突假说被证伪。连同第 1 轮，两次只动寄存器/算术侧的改动都对 ~8.2k 毫无正面影响——这个数字对算术管线的调度**不敏感**。剩下唯一没被触碰过的资源是**内存路径**：32 KiB 的 K = 512 条 64 B 缓存行，8,233 周期正好是... |
| llama-attn-scores-int8 | 20260908-202458-85ab | 4 | ok | 12,935 | 0.252x | 2.105x |  | $1.1005 | **上一轮的教训**：标量触碰让周期翻倍（+10,405 ≈ 512 行 × 20.3 周期），而向量装载**毫无受益**——这说明 (a) Shuttle 的标量 miss 会阻塞流水线约 20 周期（L2 时延），(b) 向量 LSU **绕过 L1D 直接走自己的 TileLink 端口读 L2**。回头核对所有数据点：向量单元的总内存流量 = K 32 KiB + scratch 写... |
| llama-attn-pv-int8 | 20260908-202513-620b | -1 | ok | 4,853 | 0.212x | 1.185x | ★ | $0.0000 |  |
| llama-attn-pv-int8 | 20260908-202513-620b | 0 | ok | 22,856 | 1.000x | 5.580x |  | $0.0000 |  |
| llama-attn-pv-int8 | 20260908-202513-620b | 1 | ok | 7,501 | 0.328x | 1.831x |  | $1.3333 | 先查一下实验数据库，看看 Round 1 里哪些变体被试过、各自测了多少 cycle，避免重复发现。数据库不可用，直接看编译产物。先确认工具链和当前内核的反汇编，检查编译器是否在 inline asm 周围额外插入了东西。目录 `llama-attn-pv-int8` 还不存在（只有 scores-int8）。看一下相邻目录和 header... |
| llama-attn-pv-int8 | 20260908-202513-620b | 2 | ok | 4,916 | 0.215x | 1.200x |  | $1.3850 | **上一轮 EXPECTED 落空的教训：** 预取版本慢了 +2648 cycles（≈+20.7/iter），说明标量 `lbu` miss 会在 Shuttle 侧直接把流水线卡住（MSHR 争用 / nack-replay），且这种标量停顿几乎 1:1 反映到总 cycle 上——即向量单元基本与标量前端**同步运行，issue queue 很浅**，向量 pipe... |
| llama-attn-pv-int8 | 20260908-202513-620b | 3 | ok | 4,814 | 0.211x | 1.175x | ★ | $1.9709 | **上一轮 EXPECTED 落空的教训：** 软件流水后 4916（+63），向量单元看到的 beat 流完全一样、派发提前却毫无收益 → Saturn 的 issue queue 已足够深，派发时序不是瓶颈；多出的 63 ≈ 每迭代多一条 `j`/`beqz` 分支。剩余 757 cycles 只能来自：(a) 冷 miss 延迟被 WAR 限制的 ~24-cycle 加载余量暴露，(b)... |
| llama-silu-mul | 20260908-202528-f504 | -1 | ok | 35,774 | 0.099x | 1.165x | ★ | $0.0000 |  |
| llama-silu-mul | 20260908-202528-f504 | 0 | ok | 361,260 | 1.000x | 11.760x |  | $0.0000 |  |
| llama-silu-mul | 20260908-202528-f504 | 1 | ok | 34,695 | 0.096x | 1.129x | ★ | $0.6630 | 编辑已完成并通过汇编检查：循环体现在是 2 load + 4 int + **16 FP op**（无 `vmv.v.v` 拷贝）+ 1 store，`out = sgu + sgu·eps²` 由一条 `vfmadd.vv v28,v0,v28` 完成，数值与原 `sgu·(1+eps²)` 等价；预计约 33,700 cycles。 |
| llama-silu-mul | 20260908-202528-f504 | 2 | ok | 35,712 | 0.099x | 1.163x |  | $0.6536 | **上轮 EXPECTED 复盘：** 我预测省 8 beats/strip（≈2,048 cycles），实测只省了 1,079（≈4.2 cycles/strip），与 turn 3 的"16 beats 只兑现 12"同一模式。结论：FP pipe 现在是 16×8 = 128 beats/strip，但实测 135.5，仍有 ~7.5 cycles/strip 的**串行气泡**不随... |
| llama-silu-mul | 20260908-202528-f504 | 3 | ok | 33,177 | 0.092x | 1.080x | ★ | $0.6917 | 编辑完成并通过汇编检查：strip 头部现在是 `vle g, vle u → vfmul t → vfmul gu → …`，钳位的两条 int op（`vmax/vmin` on bits(m)）落在 Horner 链旁边并行执行，16 FP + 4 int op，无拷贝、无多余 `vsetvli`；预计约 33,000 cycles。 |
| llama-silu-mul | 20260908-202528-f504 | 4 | ok | 31,024 | 0.086x | 1.010x | ★ | $0.5225 | **上轮 EXPECTED 复盘：** 预测 ~33,000，实测 33,177 = 129.6 cycles/strip，与 16 FP op × 8 = 128 beats 只差 1.6 cycles/strip——机器模型已经对齐：**FP sequencer 的 beat 数就是全部成本**，int/load/store pipe 完全隐藏。再往下只能砍 FP op 数（16 → 15... |
| llama-softmax | 20260908-202543-fc79 | -1 | ok | 1,568 | 0.072x | 1.065x | ★ | $0.0000 |  |
| llama-softmax | 20260908-202543-fc79 | 0 | ok | 21,906 | 1.000x | 14.882x |  | $0.0000 |  |
| llama-softmax | 20260908-202543-fc79 | 1 | ok | 1,704 | 0.078x | 1.158x |  | $1.2208 | **HYPOTHESIS:** pass 2 上每组 32 元素要走 10 条 e32m4 算术指令（每条 8 beat），16 组 × 80 beat = 1280 c，加上 pass 1/3 各 128 c 与 ~30 c 的开销 ≈ 1568 c——测得值与"整数与浮点算术全部串行在同一个执行 sequencer"的模型完全吻合。其中 `vsll`(fx<<23) +... |
| llama-softmax | 20260908-202543-fc79 | 2 | ok | 1,555 | 0.071x | 1.056x | ★ | $2.5916 | 上一轮 **EXPECTED 落空**：预计 −128 c，实测 **+136 c**。这说明 Saturn 的整数乘法管线远非 DLEN 全吞吐——一条 e32m4 `vmacc` 占了约 24 beat（≈ 3 个 ALU 指令的时间），比被它替换的 `vsll`+`vadd`（16 beat）还贵。模型更新：**在这台机器上任何 `vmul/vmacc` 类整数乘法都要按 ~3× beat... |
| llama-softmax | 20260908-202543-fc79 | 3 | ok | 1,693 | 0.077x | 1.150x |  | $2.2292 | 上轮 EXPECTED 命中（预计 −6~10，实测 −13）：1568 → **1555**，k 链上的 fp↔int 搬运确实在关键路径上。现在 1555 = 1536（10 op × 8 beat × 16 组 + pass 1/3）+ **19 c** 的串行空转。  **HYPOTHESIS:** 剩下的空转主要在 pass 1→2 之间：`vfredmax` 尾部延迟 →... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | -1 | ok | 132,424 | 0.881x | 1.010x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 0 | ok | 150,339 | 1.000x | 1.147x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 1 | ok | 182,726 | 1.215x | 1.394x |  | $8.8021 | I'll load the remote tool first.Now the earlier rounds (a7d0 iters 3,4,6,7 and f7c4 iters 1,2,3,7) plus checking whether the run out_dirs with logs are accessible on disk.I have enough. Summary... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 2 | ok | 132,424 | 0.881x | 1.010x |  | $8.2210 | I'll start by loading the database tools and fetching the schema.Now the iters for all n1 runs.The `note` fields are terse. Let me check the `diff` column and search all text columns for the... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 3 | ok | 132,424 | 0.881x | 1.010x |  | $0.8170 | 上一轮 EXPECTED（保持 132,424）**精确命中**，没有失准；它证实的只是工作区里确实是最佳内核。本轮我在动手前又把最后一个未被 DB 覆盖的想法——"host 在 `loop_ws` 流式读冷行的同时，手工向空闲 spad 区插入热行 mvin + preload/compute，让 L2 hit 与 DRAM miss 在 DMA 请求流中交错"——按... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 4 | ok | 132,424 | 0.881x | 1.010x |  | $0.6585 | 上一轮 EXPECTED（132,424 不变）再次精确命中——工作区里就是最佳内核（md5 `32d252581f84a556256dfed8c8034979`，`cmp` 与保存的 v1 副本逐字节一致），无需恢复。  本轮我在动手前又核算了两条此前未在 DB 中出现的思路，结论都是负收益，因此仍然不改文件：  1. **命令内热/冷混合**（用 `stride_B = 33·M`... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 5 | ok | 132,424 | 0.881x | 1.010x |  | $0.3552 | 上一轮 EXPECTED（132,424 不变）第三次精确命中；工作区文件与最佳内核逐字节一致（`cmp` 无差异，md5 `32d252581f84a556256dfed8c8034979`），无需恢复。  本轮我又把剩余 1,352 cycles（132,424 − 131,072）逐项归因，寻找可切的"未被隐藏的 drain"：  - **每次 loop_ws 的 A mvin**：每个... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 6 | ok | 133,513 | 0.888x | 1.019x |  | $0.4364 | 上一轮 EXPECTED 再次命中，无新信息。但这轮我重新审视了 loop_ws 的**每个 loop 起始处的 A 装载**：LoopMatmul 的 ldA 单元把 240 B 的 A tile 切成 4 条单行 mvin（每条 ≤ 64 B、只有 1 个 DMA 请求），而 LoadController 只跟踪 2 条未完成命令——这 4 条命令每条独占一个 tracker 一整个... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 7 | selfcheck_failed | ? | n/a | n/a |  | $2.4242 | 上一轮 EXPECTED（≈131.5k）失准：实测 **+1,089**。它教会我的是：**host 触碰过的行对 Gemmini 反而更贵**——Rocket/Shuttle 对 load 取得的是 TIP 态，InclusiveCache 在 Gemmini 的 Get 到达时必须先 probe L1，一次 probe 往返（~40 cycles）× 28 条 A mvin ≈... |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 8 | selfcheck_failed | 133,587 | 0.889x | 1.019x |  | $0.8248 | 编译干净（0 条内核警告），反汇编确认在任何向量指令之前执行 `csrs mstatus, 0x600`（MSTATUS_VS），文件 md5 `2ebe77aad42d5674254b4344140239ba`。  **总结：** `tohost = 1337` 是 riscv-tests 的 trap 出口码——`crt.S` 只开了 FS\|XS 没开 `mstatus.VS`，v7 内核的... |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | -1 | ok | 659,142 | 0.891x | 1.257x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 0 | ok | 740,101 | 1.000x | 1.412x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 1 | ok | 666,484 | 0.901x | 1.271x |  | $1.3316 | 编译检查通过（仅 `gemmini.h` 自身的既有 warning，我的文件无错误/警告）。几何校验：`a_rows = 2048`，每 bank 4 个 512 行 slot，共 16 个 slot，最高地址 3·4096 + 2048 + 4·512 = 16384 = 恰好占满 scratchpad。  本轮变更：ring slot 改为按 bank 轮转（block g → bank... |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 2 | ok | 738,148 | 0.997x | 1.408x |  | $0.5002 | **上一轮教训**：bank 分离后 666484 ≈ 659142（+1%，多出的 A 副本 mvin），说明 **scratchpad 单端口的 DMA 写 / ex 读冲突根本不是瓶颈**——per-tile ~40 cycles 不来自 bank 端口。这一条退役。剩下两个候选：DMA 请求管线（命令跟踪器 2 槽 × 4 行命令 → 实际只有 4–8 个 64 B 请求在飞）或 ex... |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 3 | ok | 745,566 | 1.007x | 1.422x |  | $1.7126 | 编译通过。本轮变更：改用转置公式 C^T = B^T·A^T——B tile 作为带 `a_stride=64`、`a_transpose` 的流式操作数（每 tile 一条 compute，每 K block 只 preload 一次 A^T 权重），从而 B 可以按 DRAM 行方向用 stride-64 B 的 16 行 × 64 B mvin **纯顺序**流入... |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 4 | ok | 1,291,233 | 1.745x | 2.463x |  | $0.6614 | **上一轮教训**：完全顺序的 1 KiB DMA 流（转置公式，自检通过）得到 745k ≈ 16 行命令的 738k ≈ pristine 的 740k。三种截然不同的 DRAM 访问模式都落在 ~740k，只有 4 行 × 64 B 命令的 v3 是 659k。这排除了 DRAM 页局部性、bank 端口、命令跟踪器并发 三个假说。剩下两种可能：(A) 内存路径本身的吞吐上限就在... |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 5 | ok | 635,908 | 0.859x | 1.213x | ★ | $0.7745 | **上一轮结论**：DMA-only = 1291233 − 659142 = **632k cycles（6.64 B/cycle）**，预测（620k）命中。v3 全内核只比纯 DMA 流多 4%——计算/重叠已接近完美，**墙是内存路径本身**（对 4 MiB 冷流 ~6.6 B/c，很像 8 B 宽 mbus 的 ~83%）。此后只有两个方向有意义：改变必须走 DRAM... |

**round3 summary** — iterations: 57, pass rate: 54/57 (94.7%), total cost: $61.8682

## round4 — launched 2026-09-10T04:05:40+08:00 Taipei, all --seed best --budget-usd 1000; ended ~2026-09-10T12:55:24+08:00 (all three runs FINISHED ... completed=N of N, no restart -- unlike round3, no session-limit interruptions this round; out/loop/round4-launch.sh, out/loop/round4-runs.txt, out/loop/round4-watchdog.log). Two infra notes predating this round: (1) the roofline memory-bandwidth bug (mbus is 8 B/cycle, not 16) was fixed 2026-09-09, just before round4 launched -- see out/loop/README.md's 'Roofline correction (2026-09-09)' section; round4 is the first round measured/analyzed entirely under the corrected roofline. (2) Ray's temp/spill dir was migrated from local disk to /share1 on 2026-09-09 (see out/loop/migrate-ray-to-share1.sh). llama-q8-gemv-gemmini-n1 was run as a deliberate 2-iteration probe (not a full search) since round3 had already converged it to ~1.01x roofline -- see out/loop/README.md's 'Round 4 planning' section.
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | -1 | ok | 635,908 | 0.859x | 1.213x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 0 | ok | 740,101 | 1.000x | 1.412x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 1 | ok | 695,516 | 0.940x | 1.327x |  | $0.3211 | **HYPOTHESIS:** v6 以 4 行 × 64 B 的 mvin 发出权重流，ld 队列只有 nCmds = 2 个命令槽，因此在飞请求数 = 2 × 4 = 8，仅占 StreamReader 16 个槽位的一半；同时 16,384 条 mvin + 16,384 对 preload/compute ≈ 33,000 条 RoCC 命令，636k 周期内每 19... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 2 | ok | 6,614,734 | 8.938x | 12.617x |  | $0.9277 | 上一轮 **假设死亡**：16 行 × 64 B mvin 反而慢了 +59,608 周期（695,516）。这退掉了"请求池只填一半 / 主机发射率"这条分支——把在飞请求从 8 提到 16、命令数从 33k 降到 20k 都没帮忙，反而变差。说明真正的变量是 **一条命令内 16 个请求跨 32 KiB（stride 2048）的 DRAM/L2 访问模式**，而不是在飞深度；M=512 的... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 3 | ok | 690,466 | 0.933x | 1.317x |  | $1.1632 | 上一轮的 8 个探针数字没有回传到我这里（只拿到总和 6,614,734）。先查实验数据库，看运行日志是否保存了 stdout。编译通过，内核文件无告警，探针代码与 `stdio` 已全部移除。  **本轮改动：** 从 v6 最佳内核出发，仅把 `LQ8_B_ROWS` 4 → 8（mvin 命令 16,384 → 8,192，2 命令槽 × 8 行 = 16 个请求在飞，单命令跨度 16... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 4 | ok | 634,507 | 0.857x | 1.210x | ★ | $0.4917 | 上一轮 **假设死亡**：8 行 = 690,466，与 16 行的 695,516 几乎相同，而不是"两 bank 就好"的双峰。这说明 stride 2048 下 **跨度不是变量，"每条命令 ≥8 行"才是**——4 行 (636k) vs 8 行 (690k) vs 16 行 (695k)。同一条命令的所有行相隔 2 KiB，若 L2 bank 是按 ≥512 B 粒度交织的 4... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 5 | selfcheck_failed | ? | n/a | n/a |  | $0.8592 | 本轮结果 634,507（−1,401，仅 0.2%）——bank 假设 **实质死亡**：chunk 顺序几乎不影响吞吐，退掉"L2 bank 冲突"整条分支。  汇总模型：4 行 = 636k，8 行 = 690k，16 行 = 695k，顺序无关。一条命令的行数一旦 ≥ 8 就掉到 6.0 B/cycle，唯一合理解释是 DRAM row-buffer：8 KiB 的 DRAM... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 6 | ok | 634,907 | 0.858x | 1.211x |  | $0.5744 | 断言 `bytes_to_read === 0` 说明 `cols` 字段在 LoadController 中被截断为 0：硬件的 `mvin_cols_bits = log2Up(dma_maxbytes/sizeof(elem_t)+1) = 7` 位，1024 → 0。即 **`MAX_BLOCK_LEN·DIM = 64` 列是硬件硬上限**，不是软件保守——"沿 B... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 7 | ok | 1,335,959 | 1.805x | 2.548x |  | $0.6253 | 上一轮 **假设死亡**：mvout 交错 +400（噪声级），尾部不是杠杆。  把已有数据合起来做一次收敛的算术：升序 K 曾测 +17,104 ⇒ 暖尾 w 满足 w·(1/6.6 − 1/16) = 17,104 ⇒ w ≈ 192 KiB；于是 634,507 = 192K/16 + (4M − 192K)/r ⇒ **冷流速率 r ≈ 6.43 B/cycle**。上一轮探针总和... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 8 | ok | 1,299,397 | 1.756x | 2.478x |  | $0.4743 | 探针解码：总数 1,335,959 − 内核冷态 ≈ 647k − printf ≈ **16 行 × stride 512 扫 4 MiB ≈ 685–689k（≈6.1 B/cycle）**，与 stride 2048 的 16 行（695k）几乎相同，且明显差于 4 行（634.5k）。假设 **存活并加强**：请求形状的差异与 stride 无关，只与每条命令的行数（=在飞请求数 16... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 9 | ok | 637,154 | 0.861x | 1.215x |  | $0.7857 | 探针解码：1,299,397 − 内核冷态 ≈ 650k ⇒ **3 行命令（6 在飞）mvin-only ≈ 649k**，比 4 行的 632k 差 ~17k，也不是延迟界预测的 843k。假设 **死亡**：在飞曲线在 8（4 行）处达到峰值，两侧都变差；冷 DRAM 在这个 DRAMSim2/mbus 路径上对 Gemmini 的饱和速率就是 ~6.4–6.6 B/cycle（约 80%... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 10 | ok | 634,934 | 0.858x | 1.211x |  | $0.5438 | 637,154（+2.6k）。EXPECTED 未中，教训：execute 按置换序发出反而更慢——ex 队列并不是被"未就绪 execute 堵住"，而是 execute 顺序本身影响 spad 读取/依赖检查；顺序 tile 序（spad 行连续）就是最好的。稳态流水线没有可挤的余量了。  但这个失败让我重新核算了尾部：稳态每步是 128 load + 256 execute... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 11 | ok | 822,414 | 1.111x | 1.569x |  | $0.2577 | 634,934（+427）。EXPECTED 未中，教训：把最后一个 K 块的 execute 藏到 DMA 尾部下面没有任何收益 ⇒ 串行 epilogue 本身几乎不花时间（execute 命令入队极快，排空只由最后几条 load 的返回延迟决定），内核与 mvin-only 流之间的 ~2.5k 差不是排空，而是 execute 与 load 争用 spad... |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 12 | ok | 682,974 | 0.923x | 1.303x |  | $0.2675 | 探针解码：822,414 − ~646.5k（冷态内核 + printf）≈ **176k / 16,384 条 ≈ 10.7 周期/命令**。EXPECTED（X≈20）未中，教训：主机发射成本约 10–11 周期而非 20。核对：rows=2 内核 65k 条命令 × 10.7 ≈ 700k ≈ 观测的 655k+，确实是主机界；而现在的内核 49k 条 × 10.7 ≈ 526k <... |
| llama-q8-gemm | 20260909-200556-5de3 | -1 | ok | 138,892 | 0.947x | 1.060x | ★ | $0.0000 |  |
| llama-q8-gemm | 20260909-200556-5de3 | 0 | ok | 146,737 | 1.000x | 1.120x |  | $0.0000 |  |
| llama-q8-gemm | 20260909-200556-5de3 | 1 | ok | 137,391 | 0.936x | 1.048x | ★ | $2.0579 | 数据库不可用。先读硬件参数和 `gemmini_loop_ws` / `sp_tiled_matmul_ws` 的实现细节。The benchmark C file isn't present in this workspace (the harness reassembles it), so I'll compile-check the header with a minimal... |
| llama-q8-gemm | 20260909-200556-5de3 | 2 | ok | 137,246 | 0.935x | 1.047x | ★ | $1.2858 | 编译通过（无错误/警告，41 条 RoCC 指令）。  **本轮改动总结**：在最优内核（J=16 单遍 + 半高尾部拆分）基础上，把所有 LOOP_WS 的 scratchpad 半区改为显式指定（与硬件未复位的 `head_loop_id` 奇偶性无关），并让尾部 Q 命令通过 `B=NULL, b_spad_id=hp` 直接复用 P 已载入的 B 块（省掉 32 KiB 的 B... |
| llama-q8-gemm | 20260909-200556-5de3 | 3 | ok | 137,280 | 0.936x | 1.047x |  | $0.8151 | 编译通过（41 条 RoCC 指令）。工作区里原本就是最优内核，本轮改动只加了一处：主循环的第一条 LOOP_WS 强制为 `Kc = 1`（后续序列为 16×6、15，再接尾部 P/Q，半区奇偶交替不受影响）。  **本轮总结**：用一条 K=1 的首命令把冷启动从"等 8 KiB 到齐"缩短到"等 ~2 KiB"，预计 ≈136.5k（省 ~600–800）。若实测不变，说明冷启动并非由... |
| llama-q8-gemm | 20260909-200556-5de3 | 4 | ok | 137,246 | 0.935x | 1.047x |  | $0.5399 | 上一轮 EXPECTED 136.5k，实测 137280（+34）。教训：首条命令改成 K=1 完全没有缩短冷启动，说明冷启动并不由"A 首块 4 KiB + B 首行 4 KiB 到达时间"主导（RS 按 mvin 粒度追踪依赖，首个 compute 早就能开始），冷启动本身已经只有 DRAM 首访延迟量级。也就是说剩余 ~6.2k 里绝大部分是**末尾 32 KiB 的 C 排空**（8... |
| llama-q8-gemm | 20260909-200556-5de3 | 5 | ok | 137,246 | 0.935x | 1.047x |  | $0.3625 | 工作区里的文件就是最优内核（md5 `14a356df…`，与上一轮恢复后一致；第 4 次实测 137246 也证明它逐字相同、结果可复现）。  **HYPOTHESIS:** 剩余 6174 周期 ≈ 末尾 32 KiB C 排空（≥4096 @ 8 B/c，写路径实际 ~4.5–5k）+ DRAM 首访冷启动（<1k）。两者都是硬件结构决定的：排空粒度由 `LoopMatmul` 的... |
| llama-q8-gemm | 20260909-200556-5de3 | 6 | ok | 137,246 | 0.935x | 1.047x |  | $0.1968 | 工作区文件 md5 仍为 `14a356df…`（即上方最优内核，第 4、5 轮已两次复测 137246，完全可复现）。  **HYPOTHESIS:** 剩余 6174 周期 = 末尾 Q 的 32 tile C 排空（32 KiB，≥4096 @ 8 B/c）+ DRAM 首访冷启动（<1k）。前者由硬件 accumulator 基址只在 {0, ACC_ROWS/2} 间翻转、且仅在输出... |
| llama-q8-gemv-gemmini-n1 | 20260909-200611-94c0 | -1 | ok | 132,424 | 0.881x | 1.010x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260909-200611-94c0 | 0 | ok | 150,339 | 1.000x | 1.147x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260909-200611-94c0 | 1 | ok | 768,849 | 5.114x | 5.866x |  | $0.9403 | 我先用一个本地的小模拟量化一下"分块降序 vs 严格降序"在随机替换 L2 上的差异，再决定这一轮怎么花。**HYPOTHESIS:** 内核已在 132,424 = 1.010x 的 8 B/cycle 链路下限（mvin-only 探针 132,109 证明数据流本身就是全部成本）。我本地用随机替换 L2 模型做的模拟显示：tile 内部读取顺序（严格降序 vs 当前分块降序）只值 ~300... |
| llama-q8-gemv-gemmini-n1 | 20260909-200611-94c0 | 2 | ok | 132,839 | 0.884x | 1.013x |  | $0.8872 | 上一轮的 EXPECTED（scored ≈600k）与实测 768,849 相差 ~170k：说明 2 MiB 的 DMA flush 比 7 B/cycle 模型慢（或 printf 经 HTIF 更贵）。探针的 `PROBE` 行没有被转发给我——先查实验数据库看 stdout 是否被记录，再恢复最优内核。内核已恢复到最优版本并叠加了唯一一个改动（尾块 loop 移到 k=0... |

**round4 summary** — iterations: 26, pass rate: 25/26 (96.2%), total cost: $14.3770

## round5 — launched 2026-09-11T14:56:00+08:00 Taipei, ended ~2026-09-11T19:19:00+08:00 Taipei (~4h23m). Single run: llama-q8-gemv-gemmini-n16 (the pristine N=16 batched-decode GEMV tile, never previously optimized -- see measured_cycles_round4.json's int8_gemv_gemmini entry, N=16 additional_measurement, 168,367 cycles baseline), 10 iterations, run_id 20260911-065653-8115; baseline 168,367 -> best 142,458 cycles (1.087x the 131,072 roofline). Starting this round, the harness saves simlog_NN.txt per iteration, and the passing feedback includes a stdout tail.
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 0 | ok | 168,367 | 1.000x | 1.285x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 1 | ok | 144,521 | 0.858x | 1.103x | ★ | $1.4159 | 编译通过（`rc=0`）。切片序列：K_tiles=128，tile_K=15 → 9 个 loop（8×15 + 1×8），从 k=[113..128) 降到 [0..8)，最后一个 loop 带 C 指针做唯一一次 mvout，第一个 loop `ex_accumulate=false` 覆盖 accumulator，其余累加。  **总结**：已将 `tiled_matmul_auto`... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 2 | ok | 142,458 | 0.846x | 1.087x | ★ | $0.7294 | 上一轮 EXPECTED 135k，实测 144.5k，差了 ~9.5k。修正模型：N=1 的 132.4k 只覆盖 B 流；N=16 额外有 A 32 KiB（+4k，带内）、末尾 32 KiB mvout（走 L2 写分配，~2.5k）以及**最后一个 loop 的 ex 尾巴没被隐藏**（~5k）。ex 尾巴之所以长，是因为 ex 每 tile 远慢于 16 cycles：硬件 loop... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 3 | ok | 768,839 | 4.566x | 5.866x |  | $1.0795 | 编译通过。本轮是探针：一次调用内连跑 T1（最佳 kernel）、T2/T3/T4（纯 mvin：A+B 16-row / 仅 B 16-row / A+B 且 B 用 4-row 命令）、T5（再跑完整 kernel，保证 C 正确），并以 `PROBE` 前缀打印 5 个数字——目的是把剩余 ~4k 拆成 A 的真实 DMA 成本、ex/mvout 暴露量、以及 4-row... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 4 | ok | 158,708 | 0.943x | 1.211x |  | $0.6128 | 探针结果（T2–T5 起始 L2 状态相同）修正了模型： - **A 的真实成本 = T2 − T3 = 11,207 cycles**，不是按字节算的 4,096：A 是 16 行 × stride 2048 的 64 B 散读，每请求 ~22 cycles，而 B 顺序流 ~8-9 cycles/请求。这是最大的剩余杠杆（~7k 超额）。 - 完整 kernel（T5... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 5 | ok | 155,636 | 0.924x | 1.187x |  | $0.3319 | 上一轮假说死亡：158,708（+16k）。自检通过说明 `a_spad_id` 区域（1→0，2→spad half）与手工 A 布局都正确，但 **1 行/命令的 mvin 是灾难**：LoadController 只允许 2 条命令同时打开，1 行命令意味着只有 2 个请求在飞，A 流退化到 ~1 B/cycle。这反过来也解释了最佳 kernel 里 A 为何贵：硬件 loop 的... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 6 | ok | 142,609 | 0.847x | 1.088x |  | $0.7653 | 上一轮：155,636，假说死亡。4 行命令只比 1 行好 3k，说明 **手工 mvin 夹在硬件 loop 之间本身就贵 ~13k**：LoopMatmul 对非 loop 命令是"等所有已配置的 loop 完成后再放行"，每个 loop 前的手工 mvin 都强制流水线排空一次（16 × ~800 cycles）。结论：任何在 loop 之间插入的手工 Gemmini 命令都不可行；A... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 7 | ok | 143,005 | 0.849x | 1.091x |  | $0.8979 | 上一轮：142,609（+151）→ `prefetch.r` 在这颗 Shuttle 上就是 NOP，Zicbop 通道关闭。至此 A 的三条"绕开硬件 loop ldA"路径全部排除（手工 mvin 会排空流水线、Zicbop 无效、布局不可改）。  回到硬件 loop 的 ldA 本身：kt=8 时每个 loop 的 ldA 发 **2 条**16... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 8 | ok | 1,036,276 | 6.155x | 7.906x |  | $0.8617 | 上一轮 143,005（+547）：EXPECTED 139.8k 未中，假设 **DIED**。教训：A 的在位成本与"两条 ldA 命令占满 2 个打开槽"无关——每 loop 一条 A 命令也毫无改善，反而多付 16 次 loop 启动。所以 A 的开销是**每个请求本身的延迟**（16 个 stride-2048 的 64 B 请求在 DRAM 侧串行化，每个请求占池子 ~600... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 9 | ok | 145,220 | 0.863x | 1.108x |  | $0.7698 | 上一轮 PROBE 结果：A2048_cold 5593 / A512_cold 5787 / A64_cold 5647 / A2048_warm 3630。EXPECTED（2048 ≈ 11k、512/64 ≈ 4.5k）全部落空——"A 因 stride-2048 造成 DRAM 行缺失"这一假设 **DIED**。教训：这台机器的内存模型对访问 stride... |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 10 | ok | 168,980 | 1.004x | 1.289x |  | $0.5344 | 上一轮 145,220（+2,762）：EXPECTED −3k 未中，"L1D 脏行 probe"假设 **DIED**。教训：host 在 Gemmini 流期间做 512 条 L2 命中读非但没有收益，反而抢走 ~2.7k——host 的 L1D miss 与 Gemmini 的 DMA 在 sbus/L2 上是同一条路，任何 host 侧的内存触碰都是净损失（与 notes 里关闭 lb... |

**round5 summary** — iterations: 11, pass rate: 11/11 (100.0%), total cost: $7.9986

## round6 — launched 2026-09-11T19:58:00+08:00 Taipei, ended ~2026-09-11T22:21:00+08:00 Taipei (~2h23m). Single run: llama-q8-gemv-gemmini-n16 (continuation of round5's kernel, seeded from round5's best), 4 iterations, run_id 20260911-115840-5f95; seed 142458 cycles (from round5 iter 2) -> iterations 688048 (per-loop probe) and 1992395 (per-loop probe) regressed, best remained 142458 -- no new best this round (1.087x the 131,072 roofline, unchanged from round5).
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | -1 | ok | 142,458 | 0.846x | 1.087x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | 0 | ok | 168,367 | 1.000x | 1.285x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | 1 | ok | 688,048 | 4.087x | 5.249x |  | $0.8128 | 我先加载远端工具，读取 `gemmini_loop_ws` 的实现细节（slot/bank 布局与 `b_spad_id` 语义），再决定探针设计。内核已写入并通过 `-march=rv64gcv_zfh_zvfh -mabi=lp64d -O2` 的完整编译检查（无警告）。  **本轮摘要：** 这是一个探针轮——最佳内核结构原封不动（tile_K=8、K 降序、单次 mvout），在... |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | 2 | ok | 214,211 | 1.272x | 1.634x |  | $2.0318 | 上一轮 EXPECTED 与实测：探针本身按预期（`Cycles taken` 688k 是 3 次 pass + printf，不计分；scored pass `s_total=142,024`）。但逐 loop 数据推翻了我的模型：scored 状态下每 loop 从 5.9k 单调恶化到 10.8k，而**尾部 loop（10.8k/64 KiB = 6.07... |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | 3 | ok | 1,992,395 | 11.834x | 15.201x |  | $1.0695 | 上一轮 EXPECTED（≈138k）与实测 214,211 相差 +72k，自检通过说明 `b_spad_id` 语义与布局正确，但每条手工 mvin 的实际代价 ≈52 cycles（4096 条）。教训：`LoopMatmul` 的输出仲裁器优先于主机命令的直通路径——N=16 时 ex 队列经常满，循环展开器的 ex 命令堵在仲裁器头部，主机的 mvin 只能在空隙里挤进去；N=1 的... |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | 4 | ok | 268,314 | 1.594x | 2.047x |  | $1.0044 | 上一轮 EXPECTED **命中**：(d) 几乎逐 loop 复现了 (s)（141,213 vs 142,892；5.85k→10.7k 的同一条曲线），而 (r) 平坦 9.4k。假设 **SURVIVED**：那 ≈9k 的尾段惩罚完全由 harness 留下的「主机写入产生的脏行」状态决定——只要 L2/L1D 里是脏的填充行，冷段 miss 的逐出写回就会把 ld 从 6.97... |

**round6 summary** — iterations: 6, pass rate: 6/6 (100.0%), total cost: $4.9184

## round7 — launched 2026-09-13T11:17:00+08:00 Taipei; ended early at 2026-09-14T00:50:00+08:00 (2026-09-13T16:50:00Z) when the operator stopped the watchdog after it hit the weekly Claude usage cap at 2026-09-13T08:01:00+08:00 (2026-09-13T00:01:00Z) and then restarted 52 times, every restart returning HTTP 429 (weekly limit resets 2026-09-15T06:00:00Z). Two real runs completed before the cap: llama-layer-fused-n1 (new kernel, baseline 206,304, roofline 140,680), run_id 20260912-191730-3a47, 4 iterations -- 217295 (iter1, regression), 189851 (iter2, new best, -8.0% vs baseline), 189924 (iter3, ~flat), 223640 (iter4, regression); and llama-q8-gemv-gemmini-n1 in-flight-scan probe, run_id 20260912-191830-e713, 2 iterations -- 3017659 (iter1, in-flight scan, large regression) and 660409 (iter2, warm/cold contrast probe). After the cap, the watchdog auto-restarted the layer-fused run 26 times (out/loop/20260913-*/ and one 20260914-000202-*/ dir); each restart only re-ran the baseline probe (kind=baseline, cycles=206304, cost_usd=0.0) before hitting 429 on the first agent turn, so none of the 26 restarts produced any real iteration or cost -- no residual progress was lost. Total round7 cost across all 28 run rows: $7.536489 (iters table SUM(cost_usd)).
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-layer-fused-n1 | 20260912-191730-3a47 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260912-191730-3a47 | 1 | ok | 217,295 | 1.053x | 1.545x |  | $1.1372 | 我先读取远端 harness（sealed body、syscalls 实现）以确认 printf 代价与 `sp_tiled_matmul_ws` 的阻塞形状，再决定本轮改动。编辑完成并已通过交叉编译检查（`-march=rv64gcv_zfh_zvfh -mabi=lp64d -O2`，无警告；三段 Saturn 内核体逐字节未变）。  **本轮改动摘要**  -... |
| llama-layer-fused-n1 | 20260912-191730-3a47 | 2 | ok | 189,851 | 0.920x | 1.350x | ★ | $2.2364 | 上一轮 EXPECTED 190,000 —— 实际 fence 完成时刻是 **190,487**（探针打印本身花了 26.8k：217,295 − 190,487），所以「移 fence + 分块重叠」这个假设 **SURVIVED**：纯调度本身就比基线省 ~16k。探针还给出了更细的机器模型：loop 1-4、6 以 ~7.7 B/c 全速流，但 loop 0、loop 5（pv... |
| llama-layer-fused-n1 | 20260912-191730-3a47 | 3 | ok | 189,924 | 0.921x | 1.350x |  | $0.7377 | 上一轮 EXPECTED 175,000，实测 189,851 —— **DIED**：把 pv 从 loop 5 挪到尾部空档只省了 636 周期。这说明 Saturn 的 DRAM 流量代价与放置位置无关（是可加的：DRAM 是串行资源，Saturn 的 ~1200 条 cache line 每条打断权重流 ~30-45 周期），而 iter-1 里尾部 loop 7+8 的... |
| llama-layer-fused-n1 | 20260912-191730-3a47 | 4 | ok | 223,640 | 1.084x | 1.590x |  | $0.8822 | 上一轮 EXPECTED 165,000，实测 189,924（+73）—— **DIED**：Zicbop 突发在冷态下等于 NOP，既不伤也不帮。三个不同调度（pv 在 loop 5 / pv 在尾部 / 无预取）全部落在 190k±0.3k，说明总时长由一个与主机行为无关的串行资源决定，而 iter-1 尾部 66k 的解释只剩下一种：**基线的 GEMV 本身在冷态下就远不止「预期的... |
| llama-q8-gemv-gemmini-n1 | 20260912-191830-e713 | -1 | ok | 132,424 | 0.881x | 1.010x | ★ | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260912-191830-e713 | 0 | ok | 150,339 | 1.000x | 1.147x |  | $0.0000 |  |
| llama-q8-gemv-gemmini-n1 | 20260912-191830-e713 | 1 | ok | 3,017,659 | 20.072x | 23.023x |  | $0.9075 | HYPOTHESIS: 按 Round-7 简报，这是一次 **探针** 迭代。待验证机制：Gemmini StreamReader 允许 16 个 Get 在飞，而 L2 只有 12 个 MSHR，超过后 L2 反压 sbus，StreamReader 卡在 `waiting_for_dma_req_ready`——即 rows-per-mvin 曲线的劣化是片上效应，不是 DRAM... |
| llama-q8-gemv-gemmini-n1 | 20260912-191830-e713 | 2 | ok | 660,409 | 4.393x | 5.039x |  | $1.6356 | 上一轮 EXPECTED（≈2.2M）偏离实测 3.02M，原因很有教益：**冲刷流和探针流都远慢于 8 B/cycle**——1.5 MiB 连续 16 行 mvin 冲刷耗 276k cycles = 5.69 B/cycle，冷态 B 流 5.68–6.63 B/cycle。这修正了机器模型：  \| rows/mvin \| 在飞 \| 冷态 cycles \| B/cycle... |
| llama-layer-fused-n1 | 20260913-000137-2132 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-004138-786f | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-012139-c080 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-020140-53fc | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-024141-3644 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-032142-e3a6 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-040143-edab | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-044144-a893 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-052146-2e52 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-060147-fea2 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-064148-3da8 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-072149-c40c | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-080150-e953 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-084151-bb60 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-092152-8335 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-100153-1f7c | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-104154-43d8 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-112156-d66a | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-120157-3fa8 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-124158-4820 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-132159-ff8b | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-140200-51f5 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-144201-9d99 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-152202-4b7b | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-160203-35d3 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260913-164204-0fd2 | 0 | ok | 206,304 | 1.000x | 1.466x | ★ | $0.0000 |  |

**round7 summary** — iterations: 35, pass rate: 35/35 (100.0%), total cost: $7.5365

## round8 — launched 2026-09-15T14:08:00+08:00 Taipei, ended ~2026-09-15T19:52:00+08:00 Taipei (~5h44m). This is the operator-restarted continuation of round7's llama-layer-fused-n1 run, which was cut short by the weekly Claude usage cap after only 4 of its planned 8 iterations; round7+round8 together make up the full 8-iteration search for this kernel. Single run: llama-layer-fused-n1, run_id 20260915-060820-a93a, 4 iterations, seeded from round7's best (189851 cycles, run 20260912-191730-3a47 iter 2) with --seed best; baseline 206304, roofline 140680 -> iterations 179585 (iter1, new overall best), 198688 (iter2, regression), 199347 (iter3, regression), 181860 (iter4, close to best but not improving). Best cycles 179585 is a cumulative +13.0% improvement over baseline across round7+round8's 8 iterations combined.
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-layer-fused-n1 | 20260915-060820-a93a | -1 | ok | 189,851 | 0.920x | 1.350x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260915-060820-a93a | 0 | ok | 206,304 | 1.000x | 1.466x |  | $0.0000 |  |
| llama-layer-fused-n1 | 20260915-060820-a93a | 1 | ok | 179,585 | 0.870x | 1.277x | ★ | $9.1983 | I'll start by loading the MCP tools.Chain confirmed (691515 → 659142 → 635908 → 634507, all passed). All four patches applied with zero fuzz/offset.  ## (a) Deliverable... |
| llama-layer-fused-n1 | 20260915-060820-a93a | 2 | ok | 198,688 | 0.963x | 1.412x |  | $1.3301 | 已确认 remote 上的文件在编辑前就与 179,585 那版逐字节一致（md5 `a36188a2…`，本地另存为 `fused_best_179585.h`），在其之上仅做了入口处的探针改动（md5 `1c61486f…`，`-O2 -Wall` 编译无警告）。  一句话总结：本轮是纯探针迭代——命令流原封不动但 Saturn 单元置空，fence 后冷跑... |
| llama-layer-fused-n1 | 20260915-060820-a93a | 3 | ok | 199,347 | 0.966x | 1.417x |  | $1.6183 | 上一轮探针（丢弃）给出了决定性数据：`stream=161687 qk=16569 softmax=1613 pv=10190`。  HYPOTHESIS: 上轮 EXPECTED 是"stream≈160k+sat≈15k"——stream 部分命中（161.7k，1 MiB 冷流 6.49 B/cycle），但 Saturn 冷跑是 28.4k（QK^T 16.6k vs 暖态... |
| llama-layer-fused-n1 | 20260915-060820-a93a | 4 | ok | 181,860 | 0.882x | 1.293x |  | $2.5125 | 远端已恢复到 best-known kernel 并叠加本轮唯一改动（md5 `6548c649d902db354a41082ec7f38341`，编译检查通过，无警告；4-row mvin2 与 `LF_SAT_EVERY 16` 保持不变）。  **上一轮 EXPECTED 落空的教训：** 8-row mvin（16 个请求在飞）反而 +20k，说明内存路径不是... |

**round8 summary** — iterations: 6, pass rate: 6/6 (100.0%), total cost: $14.6592

## round9 — launched 2026-09-17T04:00:00+08:00 Taipei, ended ~2026-09-17T09:4x:00+08:00 Taipei (same day). This is the continuation of round7+round8's llama-layer-fused-n1 search, seeded --seed best from round8's best (179,585 cycles, run 20260915-060820-a93a iter 1). Single run: llama-layer-fused-n1, run_id 20260916-195958-36b4, 4 iterations, baseline 206304, roofline 140680 -> iterations 185374 (iter1, regression), 180107 (iter2, close but not a new best), 183430 (iter3, probe-only iteration, regression), 179033 (iter4, new overall best). Best cycles 179033 is a +13.2% improvement over baseline ((206304-179033)/206304 = 0.13219), down slightly from round8's 179585 (1.277x roofline) to 1.273x roofline. Round7+round8+round9 together make up 12 iterations total on this kernel. iter1-3 probes split Saturn compute completion (sat_done) from kernel end (gemv_end): iter1 106708/185332, iter2 99956/179890, iter3 104365/183228 -- Saturn's own compute finishes at roughly half the total cycle count, i.e. it is now mostly hidden behind Gemmini weight-stream / host dispatch, not exposed on the critical path. iter3 also measured host dispatch cost directly: unit_host_cycles=48799 over unit_calls=256 (~190.6 cycles/call) and mvin_stall_sampled=2558 over mvin_samples=512 (~5.0 stall cycles/sampled mvin).
| kernel | run_id | iter | status | cycles | vs baseline | vs roofline | new best? | cost | agent note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-layer-fused-n1 | 20260916-195958-36b4 | -1 | ok | 179,585 | 0.870x | 1.277x | ★ | $0.0000 |  |
| llama-layer-fused-n1 | 20260916-195958-36b4 | 0 | ok | 206,304 | 1.000x | 1.466x |  | $0.0000 |  |
| llama-layer-fused-n1 | 20260916-195958-36b4 | 1 | ok | 185,374 | 0.899x | 1.318x |  | $1.8298 | 我先确认远端环境的几个事实（`mvin3`/`config_ld` 的签名、harness 的 include 顺序），再写假设。编辑完成并通过交叉编译检查（`-march=rv64gcv_zfh_zvfh -O2`，无新告警）。  **本轮改动摘要（单一机制）：** 在 iter5 最佳内核（179,585）的基础上，用 Gemmini 自己的 DMA（`mvin3`、ld config... |
| llama-layer-fused-n1 | 20260916-195958-36b4 | 2 | ok | 180,107 | 0.873x | 1.280x |  | $0.9998 | **上一轮结论：假设 DIED。** 预测 172k，实测 185,374（比 iter5 差 +5.8k）。探针：`sat_done=106708`、`gemv_end=185332`、`warm_cmds=267`。这教会了我关于这台机器的两件事： 1. 把 68 KiB 的 Saturn 操作数搬进 Gemmini 的请求池，付出了完整的链路时间（≈+10.5k），而原先 Saturn... |
| llama-layer-fused-n1 | 20260916-195958-36b4 | 3 | ok | 183,430 | 0.889x | 1.304x |  | $2.1677 | **上一轮结论：假设 DIED（预测 168k，实测 180,107，与 iter5 持平，+522）。** 把 unit 粒度减半、总向量指令数不变，暴露成本一点没变——所以暴露成本**不是"unit 超出 DMA 缓冲深度"**，而是与向量工作的**总量**成正比。探针：`sat_done=99956`（2080 个 mvin 用了 100k，48 c/mvin），随后 2016 个... |
| llama-layer-fused-n1 | 20260916-195958-36b4 | 4 | ok | 179,033 | 0.868x | 1.273x | ★ | $1.4750 | **上一轮 EXPECTED（~181k）miss 了 2.4k，但探针给出了决定性的数据，它教会我的是：**  - `mvin_stall_sampled = 2,558 / 512` → 每条 `mvin2` 指令从 rdcycle 到 rdcycle 只花 **5 个周期**（含两条 rdcycle）。host 在 mvin 发射上**基本不停顿**——ld RS 从未把 host... |

**round9 summary** — iterations: 6, pass rate: 6/6 (100.0%), total cost: $6.4724

## Overall summary
- total iterations: 243
- overall pass rate: 234/243 (96.3%)
- total cost across all rounds: $177.7596
- cross-check vs ledger.json's sum of per-round `total_cost_usd`: $177.7596 — MATCHES
