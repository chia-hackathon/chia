# Inner-loop ledger

> **Data sources & limitations**
> - Every run's cost / best-cycles is recomputed from `iters` (`SUM(cost_usd)`,
>   `MIN(fitness)`), not read from `runs.cost_usd` / `runs.best_cycles` —
>   those columns are only rolled up by `finish_run()`, so a run that died
>   mid-flight (session limit, crash) would otherwise show `$0.00`.
> - `iters` has no per-iteration timestamp. Per-iteration duration is inferred
>   from the on-disk kernel file's mtime (`iters.kernel_path`), relative to
>   the previous iteration's mtime (or the run's `started` time for the first
>   iteration) — an approximation, and it breaks if the run directory is ever
>   touched/copied after the fact.
> - The appendix "note" column is `iters.note` (loop.py's own status string),
>   not a model-authored summary — the DB does not store one.
> - Regenerate with: `python loop/ledger.py --rounds-file out/loop/rounds.json`


## a. Round summary
| round | start | end | wall-clock | runs | iters | ok | fail | cost | model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 2026-09-06T19:37:28+00:00 | 2026-09-06T19:48:24+00:00 | 10m56s | 1 | 3 | 3 | 0 | $0.6494 | claude-fable-5-1 |
| round1 | 2026-09-07T04:31:20+00:00 | 2026-09-07T08:17:47+00:00 | 3h46m | 5 | 35 | 33 | 2 | $20.6599 | claude-fable-5-1 |
| round2 | 2026-09-07T14:21:56+00:00 | 2026-09-07T14:57:36.762205+00:00 | 35m41s | 6 | 17 | 16 | 1 | $7.3727 | claude-fable-5-1 |
| round2b | 2026-09-08T02:58:21+00:00 | 2026-09-08T07:19:49+00:00 | 4h21m | 6 | 41 | 39 | 2 | $31.2473 | claude-fable-5-1 |
| round3 | 2026-09-08T20:24:14+00:00 | 2026-09-09T09:58:14+00:00 | 13h34m | 9 | 57 | 54 | 3 | $61.8682 | claude-fable-5-1 |
| round4 | 2026-09-09T20:05:42+00:00 | 2026-09-10T04:50:33+00:00 | 8h44m | 3 | 26 | 25 | 1 | $14.3770 | claude-fable-5-1 |
| round5 | 2026-09-11T06:56:54+00:00 | 2026-09-11T11:19:47+00:00 | 4h22m | 1 | 11 | 11 | 0 | $7.9986 | claude-fable-5-1 |
| round6 | 2026-09-11T11:58:40+00:00 | 2026-09-11T14:20:24+00:00 | 2h21m | 1 | 6 | 6 | 0 | $4.9184 | claude-fable-5-1 |
| round7 | 2026-09-12T19:17:31+00:00 | 2026-09-13T16:42:08.648605+00:00 | 21h24m | 28 | 35 | 35 | 0 | $7.5365 | claude-fable-5-1 |
| round8 | 2026-09-15T06:08:21+00:00 | 2026-09-15T11:51:24+00:00 | 5h43m | 1 | 6 | 6 | 0 | $14.6592 | claude-fable-5-1 |
| round9 | 2026-09-16T19:59:59+00:00 | 2026-09-17T01:14:23+00:00 | 5h14m | 1 | 6 | 6 | 0 | $6.4724 | claude-fable-5-1 |
| round10 | 2026-09-17T15:58:35+00:00 | 2026-09-17T19:36:42+00:00 | 3h38m | 1 | 5 | 5 | 0 | $2.3956 | claude-fable-5-1 |

## b. Per-run detail
### smoke — smoke test of the inner loop before round1
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-softmax | 20260906-193727-ebbf | 2/2 | ? | 21,906 | 4,312 | 1,472 | 2.929x | $0.6494 | 10m56s |  |

### round1 — config GENV256D128GemminiShuttleConfig, model claude-fable-5-1 (see out/loop/round1-runs.txt)
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260907-043120-a7d0 | 8/8 | ? | 150,339 | 132,653 | 131,072 | 1.012x | $8.7460 | 3h46m |  |
| llama-attn-scores-int8 | 20260907-043130-fa24 | 6/6 | ? | 51,280 | 8,574 | 6,144 | 1.396x | $3.9013 | 2h12m |  |
| llama-attn-pv-int8 | 20260907-043140-2246 | 6/6 | ? | 22,856 | 4,853 | 4,096 | 1.185x | $2.7655 | 2h08m |  |
| llama-silu-mul | 20260907-043150-7557 | 6/6 | ? | 361,260 | 66,811 | 30,720 | 2.175x | $2.3084 | 55m13s |  |
| llama-softmax | 20260907-043200-f0d7 | 4/4 | ? | 21,906 | 1,752 | 1,472 | 1.190x | $2.9387 | 35m43s |  |

### round2 — first round2 launch, aborted mid-run (session limit); see out/loop/round2-runs.txt
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260907-142155-86bf | 1/8 | 132,653 | 150,339 | 132,653 | 131,072 | 1.012x | $2.7854 | 31m12s (ongoing/aborted) | aborted: session limit |
| llama-attn-scores-int8 | 20260907-142205-6c45 | 0/4 | 8,574 | 51,280 | 8,574 | 6,144 | 1.396x | $0.0000 | 0m00s (ongoing/aborted) | aborted: session limit |
| llama-attn-pv-int8 | 20260907-142215-b093 | 0/4 | 4,853 | 22,856 | 4,853 | 4,096 | 1.185x | $0.0000 | 0m00s (ongoing/aborted) | aborted: session limit |
| llama-silu-mul | 20260907-142225-55f2 | 2/4 | 66,811 | 361,260 | 50,256 | 30,720 | 1.636x | $1.7470 | 18m26s (ongoing/aborted) | aborted: session limit |
| llama-softmax | 20260907-142235-7231 | 2/3 | 1,752 | 21,906 | 1,630 | 1,472 | 1.107x | $1.4246 | 15m26s (ongoing/aborted) | aborted: session limit |
| llama-q8-gemv-gemmini-lmhead | 20260907-144414-da36 | 1/6 | ? | 740,101 | 691,515 | 524,288 | 1.319x | $1.4157 | 13m22s (ongoing/aborted) | aborted: session limit |

### round2b — round2 restart 2026-09-08T11:00:03+08:00 (session limit reset, seeded from --seed best); see out/loop/round2-runs.txt
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260908-025820-f7c4 | 8/8 | 132,653 | 150,339 | 132,424 | 131,072 | 1.010x | $6.6351 | 3h19m | seeded from best (/home/max410011_l/aether/out/loop/20260907-043120-a7d0/kernel_ |
| llama-q8-gemv-gemmini-lmhead | 20260908-025835-a0cc | 6/6 | 691,515 | 740,101 | 659,142 | 524,288 | 1.257x | $6.6940 | 4h21m | seeded from best (/home/max410011_l/aether/out/loop/20260907-144414-da36/kernel_ |
| llama-attn-scores-int8 | 20260908-025850-2356 | 4/4 | 8,574 | 51,280 | 8,233 | 6,144 | 1.340x | $8.0313 | 2h12m | seeded from best (/home/max410011_l/aether/out/loop/20260907-043130-fa24/kernel_ |
| llama-attn-pv-int8 | 20260908-025905-5427 | 4/4 | 4,853 | 22,856 | 4,853 | 4,096 | 1.185x | $4.1568 | 2h12m | seeded from best (/home/max410011_l/aether/out/loop/20260907-043140-2246/kernel_ |
| llama-silu-mul | 20260908-025919-3d8d | 4/4 | 50,256 | 361,260 | 35,774 | 30,720 | 1.165x | $3.5605 | 51m33s | seeded from best (/home/max410011_l/aether/out/loop/20260907-142225-55f2/kernel_ |
| llama-softmax | 20260908-025933-f639 | 3/3 | 1,630 | 21,906 | 1,568 | 1,472 | 1.065x | $2.1695 | 36m46s | seeded from best (/home/max410011_l/aether/out/loop/20260907-142235-7231/kernel_ |

### round3 — launched 2026-09-09T04:24:12+08:00 Taipei, all --seed best --budget-usd 1000; ended ~2026-09-09T17:59+08:00 (out/loop/round3-launch.sh, out/loop/round3-runs.txt, out/loop/round3-watchdog.log). llama-q8-gemv-gemmini-n1 and llama-q8-gemv-gemmini-lmhead were each interrupted by a Claude session-limit 429 (n1 after 4 iters, lmhead after 3) and restarted with the remaining planned_iters, seed=best; the watchdog's auto-restart misparsed the API's 'resets 5pm America/Los_Angeles' message as local time, so a human manually restarted both at 2026-09-09T13:23+08:00 instead of waiting for the watchdog.
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n1 | 20260908-202413-e8e8 | 4/12 | 132,424 | 150,339 | 132,424 | 131,072 | 1.010x | $6.4618 | 2h09m (ongoing/aborted) | aborted after 4 iters: session limit (429); restarted as 20260909-052323-9d90 |
| llama-q8-gemv-gemmini-n1 | 20260909-052323-9d90 | 8/8 | 132,424 | 150,339 | 132,424 | 131,072 | 1.010x | $22.5392 | 4h34m | manual restart 2026-09-09T13:23:22+08:00 of 20260908-202413-e8e8, seed=best |
| llama-q8-gemv-gemmini-lmhead | 20260908-202428-217e | 3/8 | 659,142 | 740,101 | 659,142 | 524,288 | 1.257x | $2.9003 | 2h11m (ongoing/aborted) | aborted after 3 iters: session limit (429); restarted as 20260909-052338-b80e |
| llama-q8-gemv-gemmini-lmhead | 20260909-052338-b80e | 5/5 | 659,142 | 740,101 | 635,908 | 524,288 | 1.213x | $4.9801 | 4h24m | manual restart 2026-09-09T13:23:37+08:00 of 20260908-202428-217e, seed=best |
| llama-q8-gemm | 20260908-202443-5906 | 6/6 | ? | 146,737 | 138,892 | 131,072 | 1.060x | $7.0106 | 2h27m | seed best unresolved |
| llama-attn-scores-int8 | 20260908-202458-85ab | 4/4 | 8,233 | 51,280 | 8,233 | 6,144 | 1.340x | $4.7148 | 2h05m | seeded from best (/home/max410011_l/aether/out/loop/20260908-025850-2356/kernel_ |
| llama-attn-pv-int8 | 20260908-202513-620b | 3/3 | 4,853 | 22,856 | 4,814 | 4,096 | 1.175x | $4.6892 | 2h00m | seeded from best (/home/max410011_l/aether/out/loop/20260907-043140-2246/kernel_ |
| llama-silu-mul | 20260908-202528-f504 | 4/4 | 35,774 | 361,260 | 31,024 | 30,720 | 1.010x | $2.5308 | 1h07m | seeded from best (/home/max410011_l/aether/out/loop/20260908-025919-3d8d/kernel_ |
| llama-softmax | 20260908-202543-fc79 | 3/3 | 1,568 | 21,906 | 1,555 | 1,472 | 1.056x | $6.0415 | 58m31s | seeded from best (/home/max410011_l/aether/out/loop/20260908-025933-f639/kernel_ |

### round4 — launched 2026-09-10T04:05:40+08:00 Taipei, all --seed best --budget-usd 1000; ended ~2026-09-10T12:55:24+08:00 (all three runs FINISHED ... completed=N of N, no restart -- unlike round3, no session-limit interruptions this round; out/loop/round4-launch.sh, out/loop/round4-runs.txt, out/loop/round4-watchdog.log). Two infra notes predating this round: (1) the roofline memory-bandwidth bug (mbus is 8 B/cycle, not 16) was fixed 2026-09-09, just before round4 launched -- see out/loop/README.md's 'Roofline correction (2026-09-09)' section; round4 is the first round measured/analyzed entirely under the corrected roofline. (2) Ray's temp/spill dir was migrated from local disk to /share1 on 2026-09-09 (see out/loop/migrate-ray-to-share1.sh). llama-q8-gemv-gemmini-n1 was run as a deliberate 2-iteration probe (not a full search) since round3 had already converged it to ~1.01x roofline -- see out/loop/README.md's 'Round 4 planning' section.
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-lmhead | 20260909-200541-dae1 | 12/12 | 635,908 | 740,101 | 634,507 | 524,288 | 1.210x | $7.2915 | 8h44m | seeded from best (/home/max410011_l/aether/out/loop/20260909-052338-b80e/kernel_ |
| llama-q8-gemm | 20260909-200556-5de3 | 6/6 | 138,892 | 146,737 | 137,246 | 131,072 | 1.047x | $5.2581 | 2h13m | seeded from best (/home/max410011_l/aether/out/loop/20260908-202443-5906/kernel_ |
| llama-q8-gemv-gemmini-n1 | 20260909-200611-94c0 | 2/2 | 132,424 | 150,339 | 132,424 | 131,072 | 1.010x | $1.8274 | 1h08m | deliberate 2-iter probe, not a full search (round3 had already converged this ke |

### round5 — launched 2026-09-11T14:56:00+08:00 Taipei, ended ~2026-09-11T19:19:00+08:00 Taipei (~4h23m). Single run: llama-q8-gemv-gemmini-n16 (the pristine N=16 batched-decode GEMV tile, never previously optimized -- see measured_cycles_round4.json's int8_gemv_gemmini entry, N=16 additional_measurement, 168,367 cycles baseline), 10 iterations, run_id 20260911-065653-8115; baseline 168,367 -> best 142,458 cycles (1.087x the 131,072 roofline). Starting this round, the harness saves simlog_NN.txt per iteration, and the passing feedback includes a stdout tail.
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n16 | 20260911-065653-8115 | 10/10 | ? | 168,367 | 142,458 | 131,072 | 1.087x | $7.9986 | 4h22m | seed best unresolved |

### round6 — launched 2026-09-11T19:58:00+08:00 Taipei, ended ~2026-09-11T22:21:00+08:00 Taipei (~2h23m). Single run: llama-q8-gemv-gemmini-n16 (continuation of round5's kernel, seeded from round5's best), 4 iterations, run_id 20260911-115840-5f95; seed 142458 cycles (from round5 iter 2) -> iterations 688048 (per-loop probe) and 1992395 (per-loop probe) regressed, best remained 142458 -- no new best this round (1.087x the 131,072 roofline, unchanged from round5).
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-q8-gemv-gemmini-n16 | 20260911-115840-5f95 | 4/4 | 142,458 | 168,367 | 142,458 | 131,072 | 1.087x | $4.9184 | 2h21m | seeded from best (/home/max410011_l/aether/out/loop/20260911-065653-8115/kernel_ |

### round7 — launched 2026-09-13T11:17:00+08:00 Taipei; ended early at 2026-09-14T00:50:00+08:00 (2026-09-13T16:50:00Z) when the operator stopped the watchdog after it hit the weekly Claude usage cap at 2026-09-13T08:01:00+08:00 (2026-09-13T00:01:00Z) and then restarted 52 times, every restart returning HTTP 429 (weekly limit resets 2026-09-15T06:00:00Z). Two real runs completed before the cap: llama-layer-fused-n1 (new kernel, baseline 206,304, roofline 140,680), run_id 20260912-191730-3a47, 4 iterations -- 217295 (iter1, regression), 189851 (iter2, new best, -8.0% vs baseline), 189924 (iter3, ~flat), 223640 (iter4, regression); and llama-q8-gemv-gemmini-n1 in-flight-scan probe, run_id 20260912-191830-e713, 2 iterations -- 3017659 (iter1, in-flight scan, large regression) and 660409 (iter2, warm/cold contrast probe). After the cap, the watchdog auto-restarted the layer-fused run 26 times (out/loop/20260913-*/ and one 20260914-000202-*/ dir); each restart only re-ran the baseline probe (kind=baseline, cycles=206304, cost_usd=0.0) before hitting 429 on the first agent turn, so none of the 26 restarts produced any real iteration or cost -- no residual progress was lost. Total round7 cost across all 28 run rows: $7.536489 (iters table SUM(cost_usd)).
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-layer-fused-n1 | 20260912-191730-3a47 | 4/8 | ? | 206,304 | 189,851 | 140,680 | 1.350x | $4.9934 | 3h10m (ongoing/aborted) | completed 4 of 8 iterations before session-limit stop; best 189851 (iter2) |
| llama-q8-gemv-gemmini-n1 | 20260912-191830-e713 | 2/2 | 132,424 | 150,339 | 132,424 | 131,072 | 1.010x | $2.5431 | 1h33m | in-flight-scan / warm-cold probe run, seeded from 132424 (run 20260908-202413-e8 |
| llama-layer-fused-n1 | 20260913-000137-2132 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m05s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, no agent turn ran, cost $0 |
| llama-layer-fused-n1 | 20260913-004138-786f | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-012139-c080 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-020140-53fc | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-024141-3644 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-032142-e3a6 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-040143-edab | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m06s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-044144-a893 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-052146-2e52 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-060147-fea2 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-064148-3da8 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-072149-c40c | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-080150-e953 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m05s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-084151-bb60 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-092152-8335 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m05s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-100153-1f7c | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-104154-43d8 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-112156-d66a | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-120157-3fa8 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m05s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-124158-4820 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-132159-ff8b | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-140200-51f5 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m03s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-144201-9d99 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m03s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-152202-4b7b | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-160203-35d3 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |
| llama-layer-fused-n1 | 20260913-164204-0fd2 | 0/4 | ? | 206,304 | 206,304 | 140,680 | 1.466x | $0.0000 | 0m04s (ongoing/aborted) | watchdog auto-restart after 429; baseline-only, cost $0 |

### round8 — launched 2026-09-15T14:08:00+08:00 Taipei, ended ~2026-09-15T19:52:00+08:00 Taipei (~5h44m). This is the operator-restarted continuation of round7's llama-layer-fused-n1 run, which was cut short by the weekly Claude usage cap after only 4 of its planned 8 iterations; round7+round8 together make up the full 8-iteration search for this kernel. Single run: llama-layer-fused-n1, run_id 20260915-060820-a93a, 4 iterations, seeded from round7's best (189851 cycles, run 20260912-191730-3a47 iter 2) with --seed best; baseline 206304, roofline 140680 -> iterations 179585 (iter1, new overall best), 198688 (iter2, regression), 199347 (iter3, regression), 181860 (iter4, close to best but not improving). Best cycles 179585 is a cumulative +13.0% improvement over baseline across round7+round8's 8 iterations combined.
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-layer-fused-n1 | 20260915-060820-a93a | 4/4 | 189,851 | 206,304 | 179,585 | 140,680 | 1.277x | $14.6592 | 5h43m | continuation of round7's run_id 20260912-191730-3a47, seeded from its best (1898 |

### round9 — launched 2026-09-17T04:00:00+08:00 Taipei, ended ~2026-09-17T09:4x:00+08:00 Taipei (same day). This is the continuation of round7+round8's llama-layer-fused-n1 search, seeded --seed best from round8's best (179,585 cycles, run 20260915-060820-a93a iter 1). Single run: llama-layer-fused-n1, run_id 20260916-195958-36b4, 4 iterations, baseline 206304, roofline 140680 -> iterations 185374 (iter1, regression), 180107 (iter2, close but not a new best), 183430 (iter3, probe-only iteration, regression), 179033 (iter4, new overall best). Best cycles 179033 is a +13.2% improvement over baseline ((206304-179033)/206304 = 0.13219), down slightly from round8's 179585 (1.277x roofline) to 1.273x roofline. Round7+round8+round9 together make up 12 iterations total on this kernel. iter1-3 probes split Saturn compute completion (sat_done) from kernel end (gemv_end): iter1 106708/185332, iter2 99956/179890, iter3 104365/183228 -- Saturn's own compute finishes at roughly half the total cycle count, i.e. it is now mostly hidden behind Gemmini weight-stream / host dispatch, not exposed on the critical path. iter3 also measured host dispatch cost directly: unit_host_cycles=48799 over unit_calls=256 (~190.6 cycles/call) and mvin_stall_sampled=2558 over mvin_samples=512 (~5.0 stall cycles/sampled mvin).
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-layer-fused-n1 | 20260916-195958-36b4 | 4/4 | 179,585 | 206,304 | 179,033 | 140,680 | 1.273x | $6.4724 | 5h14m | continuation of round8's run_id 20260915-060820-a93a, seeded from its best (1795 |

### round10 — launched 2026-09-17 15:58 Taipei, ended ~2026-09-18 03:36 Taipei (overnight). This round tested the new kernel llama-lmhead-fused-n1, asking whether the 'overlap ratio can be extrapolated' hypothesis from the llama-layer-fused-n1 work (round7-9; Saturn compute increasingly hidden behind Gemmini weight-stream / host dispatch, ~61% overlap assumed by DEFAULT_OVERLAP_FACTOR) generalizes to lm_head. Single run: llama-lmhead-fused-n1, run_id 20260917-155834-2887, 4 iterations, baseline 648292, roofline 532232 -> iterations 676825 (iter1, regression), 651726 (iter2, regression, close to baseline), 689999 (iter3, probe-only, regression), 698428 (iter4, regression). Baseline 648292 was NEVER beaten in this round; speedup 1.000x (+0.0%), a clear negative result. iter1 probe decomposed the Saturn pipeline into phases: stream=634821 rms=7296 quant=2496 argmax=2980 (total 647593); excluding 'stream', the overlappable Saturn-side work is only rms+quant+argmax=12772 cycles, ~2.0% of the iteration total -- far smaller than layer-fused's overlap-relevant share. iter2 probe measured host-side exposure (held_*) per stage instead of raw stage duration: held_rms1=5984 held_rms2=8990 held_quant=6552 held_amax1=2093 held_amax2=3532 (sum 27151, total 653221) -- larger than iter1's 12772, likely reflecting host/sync-boundary wall-clock exposure (and possibly double-counting split sub-stages) rather than raw Saturn compute time; not fully reconciled. iter3 swept a busy-wait spin parameter (0/80/250/600) before checking the Saturn pipeline, sampling ~32 periods each: spin0 period_sum=154598 cnt=32, spin80=154376 cnt=32, spin250=150380 cnt=32, spin600=145647 cnt=31 (iteration total 663023) -- tighter polling (higher spin) modestly reduces sampled sync-wait period, but iter4 (698428, applying findings from the spin scan) still did not beat baseline, confirming lm_head's exposure problem is structural, not a scheduling-tuning issue. Conclusion: lm_head's overlappable Saturn work is proportionally too small (and/or too exposed at host sync points) for the layer-fused overlap findings to transfer; a single global overlap factor should not be assumed across kernel families.
| kernel | run_id | iters (done/planned) | seed | baseline | best | roofline | best/roofline | cost | wall-clock | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-lmhead-fused-n1 | 20260917-155834-2887 | 4/4 | ? | 648,292 | 648,292 | 532,232 | 1.218x | $2.3956 | 3h38m | new kernel llama-lmhead-fused-n1, testing overlap-ratio extrapolation from layer |

## c. Per-kernel history (baseline -> round-by-round best)
| kernel | baseline | roofline | smoke | round1 | round2 | round2b | round3 | round4 | round5 | round6 | round7 | round8 | round9 | round10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-attn-pv-int8 | 22,856 | 4,096 | - | 4,853 (cum $2.7655) | 4,853 (cum $2.7655) | 4,853 (cum $6.9224) | 4,814 (cum $11.6115) | - | - | - | - | - | - | - |
| llama-attn-scores-int8 | 51,280 | 6,144 | - | 8,574 (cum $3.9013) | 8,574 (cum $3.9013) | 8,233 (cum $11.9327) | 8,233 (cum $16.6475) | - | - | - | - | - | - | - |
| llama-layer-fused-n1 | 206,304 | 140,680 | - | - | - | - | - | - | - | - | 206,304 (cum $4.9934) | 179,585 (cum $19.6526) | 179,033 (cum $26.1251) | - |
| llama-lmhead-fused-n1 | 648,292 | 532,232 | - | - | - | - | - | - | - | - | - | - | - | 648,292 (cum $2.3956) |
| llama-q8-gemm | 146,737 | 131,072 | - | - | - | - | 138,892 (cum $7.0106) | 137,246 (cum $12.2686) | - | - | - | - | - | - |
| llama-q8-gemv-gemmini-lmhead | 740,101 | 524,288 | - | - | 691,515 (cum $1.4157) | 659,142 (cum $8.1097) | 635,908 (cum $15.9902) | 634,507 (cum $23.2817) | - | - | - | - | - | - |
| llama-q8-gemv-gemmini-n1 | 150,339 | 131,072 | - | 132,653 (cum $8.7460) | 132,653 (cum $11.5313) | 132,424 (cum $18.1664) | 132,424 (cum $47.1673) | 132,424 (cum $48.9947) | - | - | 132,424 (cum $51.5378) | - | - | - |
| llama-q8-gemv-gemmini-n16 | 168,367 | 131,072 | - | - | - | - | - | - | 142,458 (cum $7.9986) | 142,458 (cum $12.9170) | - | - | - | - |
| llama-silu-mul | 361,260 | 30,720 | - | 66,811 (cum $2.3084) | 50,256 (cum $4.0554) | 35,774 (cum $7.6159) | 31,024 (cum $10.1468) | - | - | - | - | - | - | - |
| llama-softmax | 21,906 | 1,472 | 4,312 (cum $0.6494) | 1,752 (cum $3.5881) | 1,630 (cum $5.0127) | 1,568 (cum $7.1822) | 1,555 (cum $13.2237) | - | - | - | - | - | - | - |

## d. Appendix — every iteration
| run_id | iter | status | cycles | cost | duration | note |
| --- | --- | --- | --- | --- | --- | --- |
| 20260906-193727-ebbf | 0 | ok | 21,906 | $0.0000 | 0m00s | passed, cycles=21906, instret=3236 |
| 20260906-193727-ebbf | 1 | ok | 15,026 | $0.4301 | 1m18s | passed, cycles=15026, instret=981 |
| 20260906-193727-ebbf | 2 | ok | 4,312 | $0.2194 | 5m14s | passed, cycles=4312, instret=989 |
| 20260907-043120-a7d0 | 0 | ok | 150,339 | $0.0000 | 54m15s | passed, cycles=150339, instret=None |
| 20260907-043120-a7d0 | 1 | ok | 132,653 | $2.0505 | 9m14s | passed, cycles=132653, instret=None |
| 20260907-043120-a7d0 | 2 | build_failed | ? | $1.7869 | 27m20s | 6m4_t w1 = __riscv_vsext_vf2_i16m4(b1, vl);
      \|                   ^~~~~~~~~~
./include/llama_q8 |
| 20260907-043120-a7d0 | 3 | ok | 134,585 | $0.8477 | 2m16s | passed, cycles=134585, instret=None |
| 20260907-043120-a7d0 | 4 | ok | 133,800 | $0.5067 | 20m38s | passed, cycles=133800, instret=None |
| 20260907-043120-a7d0 | 5 | ok | 132,653 | $0.7453 | 21m32s | passed, cycles=132653, instret=None |
| 20260907-043120-a7d0 | 6 | ok | 149,757 | $0.3552 | 19m13s | passed, cycles=149757, instret=None |
| 20260907-043120-a7d0 | 7 | ok | 142,221 | $0.8704 | 23m26s | passed, cycles=142221, instret=None |
| 20260907-043120-a7d0 | 8 | ok | 132,653 | $1.5833 | 30m25s | passed, cycles=132653, instret=None |
| 20260907-043130-fa24 | 0 | ok | 51,280 | $0.0000 | 0m01s | passed, cycles=51280, instret=20524 |
| 20260907-043130-fa24 | 1 | ok | 12,729 | $1.4631 | 7m39s | passed, cycles=12729, instret=4349 |
| 20260907-043130-fa24 | 2 | ok | 10,472 | $0.5490 | 29m51s | passed, cycles=10472, instret=4666 |
| 20260907-043130-fa24 | 3 | ok | 10,008 | $0.3536 | 18m29s | passed, cycles=10008, instret=4538 |
| 20260907-043130-fa24 | 4 | ok | 8,574 | $0.4238 | 19m04s | passed, cycles=8574, instret=4282 |
| 20260907-043130-fa24 | 5 | ok | 10,081 | $0.5372 | 19m47s | passed, cycles=10081, instret=4474 |
| 20260907-043130-fa24 | 6 | ok | 10,522 | $0.5746 | 20m03s | passed, cycles=10522, instret=4342 |
| 20260907-043140-2246 | 0 | ok | 22,856 | $0.0000 | 0m01s | passed, cycles=22856, instret=16877 |
| 20260907-043140-2246 | 1 | ok | 9,803 | $0.5413 | 4m18s | passed, cycles=9803, instret=4657 |
| 20260907-043140-2246 | 2 | ok | 6,276 | $0.2705 | 21m28s | passed, cycles=6276, instret=3640 |
| 20260907-043140-2246 | 3 | ok | 4,999 | $0.2517 | 20m09s | passed, cycles=4999, instret=3007 |
| 20260907-043140-2246 | 4 | ok | 4,992 | $0.6807 | 21m55s | passed, cycles=4992, instret=2808 |
| 20260907-043140-2246 | 5 | ok | 8,508 | $0.6706 | 21m50s | passed, cycles=8508, instret=4190 |
| 20260907-043140-2246 | 6 | ok | 4,853 | $0.3507 | 20m03s | passed, cycles=4853, instret=3009 |
| 20260907-043150-7557 | 0 | ok | 361,260 | $0.0000 | 0m00s | passed, cycles=361260, instret=48180 |
| 20260907-043150-7557 | 1 | ok | 289,169 | $0.4764 | 11m41s | passed, cycles=289169, instret=12602 |
| 20260907-043150-7557 | 2 | ok | 91,474 | $0.2532 | 7m57s | passed, cycles=91474, instret=15676 |
| 20260907-043150-7557 | 3 | ok | 79,914 | $0.3904 | 10m23s | passed, cycles=79914, instret=11830 |
| 20260907-043150-7557 | 4 | ok | 66,820 | $0.5713 | 7m48s | passed, cycles=66820, instret=10034 |
| 20260907-043150-7557 | 5 | ok | 66,811 | $0.3050 | 6m38s | passed, cycles=66811, instret=9778 |
| 20260907-043150-7557 | 6 | selfcheck_failed | 301 | $0.3121 | 6m41s | simulator rc=255; the benchmark's self-check did not pass |
| 20260907-043200-f0d7 | 0 | ok | 21,906 | $0.0000 | 0m01s | passed, cycles=21906, instret=3236 |
| 20260907-043200-f0d7 | 1 | ok | 2,174 | $1.3040 | 9m54s | passed, cycles=2174, instret=564 |
| 20260907-043200-f0d7 | 2 | ok | 1,975 | $0.5782 | 6m53s | passed, cycles=1975, instret=605 |
| 20260907-043200-f0d7 | 3 | ok | 2,103 | $0.6466 | 7m16s | passed, cycles=2103, instret=590 |
| 20260907-043200-f0d7 | 4 | ok | 1,752 | $0.4099 | 7m20s | passed, cycles=1752, instret=550 |
| 20260907-142155-86bf | -1 | ok | 132,653 | $0.0000 | 0m04s | seed from /home/max410011_l/aether/out/loop/20260907-043120-a7d0/kernel_01.h (run 20260907-043120-a7 |
| 20260907-142155-86bf | 0 | ok | 150,339 | $0.0000 | 0m00s | passed, cycles=150339, instret=None |
| 20260907-142155-86bf | 1 | selfcheck_failed | 134,019 | $2.7854 | 31m08s | simulator rc=255; the benchmark's self-check did not pass |
| 20260907-142205-6c45 | -1 | ok | 8,574 | $0.0000 | 0m00s | seed from /home/max410011_l/aether/out/loop/20260907-043130-fa24/kernel_04.c (run 20260907-043130-fa |
| 20260907-142205-6c45 | 0 | ok | 51,280 | $0.0000 | 0m00s | passed, cycles=51280, instret=20524 |
| 20260907-142215-b093 | -1 | ok | 4,853 | $0.0000 | 0m00s | seed from /home/max410011_l/aether/out/loop/20260907-043140-2246/kernel_06.c (run 20260907-043140-22 |
| 20260907-142215-b093 | 0 | ok | 22,856 | $0.0000 | 0m00s | passed, cycles=22856, instret=16877 |
| 20260907-142225-55f2 | -1 | ok | 66,811 | $0.0000 | 0m00s | seed from /home/max410011_l/aether/out/loop/20260907-043150-7557/kernel_05.c (run 20260907-043150-75 |
| 20260907-142225-55f2 | 0 | ok | 361,260 | $0.0000 | 0m00s | passed, cycles=361260, instret=48180 |
| 20260907-142225-55f2 | 1 | ok | 59,008 | $1.2794 | 11m33s | passed, cycles=59008, instret=15414 |
| 20260907-142225-55f2 | 2 | ok | 50,256 | $0.4676 | 6m53s | passed, cycles=50256, instret=7734 |
| 20260907-142235-7231 | -1 | ok | 1,752 | $0.0000 | 0m00s | seed from /home/max410011_l/aether/out/loop/20260907-043200-f0d7/kernel_04.c (run 20260907-043200-f0 |
| 20260907-142235-7231 | 0 | ok | 21,906 | $0.0000 | 0m00s | passed, cycles=21906, instret=3236 |
| 20260907-142235-7231 | 1 | ok | 1,630 | $1.1133 | 9m29s | passed, cycles=1630, instret=550 |
| 20260907-142235-7231 | 2 | ok | 1,760 | $0.3113 | 5m56s | passed, cycles=1760, instret=535 |
| 20260907-144414-da36 | 0 | ok | 740,101 | $0.0000 | 0m04s | passed, cycles=740101, instret=None |
| 20260907-144414-da36 | 1 | ok | 691,515 | $1.4157 | 13m18s | passed, cycles=691515, instret=None |
| 20260908-025820-f7c4 | -1 | ok | 132,653 | $0.0000 | 0m05s | seed from /home/max410011_l/aether/out/loop/20260907-043120-a7d0/kernel_01.h (run 20260907-043120-a7 |
| 20260908-025820-f7c4 | 0 | ok | 150,339 | $0.0000 | 0m00s | passed, cycles=150339, instret=None |
| 20260908-025820-f7c4 | 1 | ok | 132,834 | $1.8401 | 31m36s | passed, cycles=132834, instret=None |
| 20260908-025820-f7c4 | 2 | selfcheck_failed | 134,842 | $1.2898 | 24m03s | simulator rc=255; the benchmark's self-check did not pass |
| 20260908-025820-f7c4 | 3 | ok | 140,285 | $1.4548 | 25m42s | passed, cycles=140285, instret=None |
| 20260908-025820-f7c4 | 4 | ok | 132,653 | $0.5628 | 21m13s | passed, cycles=132653, instret=None |
| 20260908-025820-f7c4 | 5 | ok | 132,653 | $0.2930 | 19m25s | passed, cycles=132653, instret=None |
| 20260908-025820-f7c4 | 6 | ok | 132,653 | $0.3500 | 19m42s | passed, cycles=132653, instret=None |
| 20260908-025820-f7c4 | 7 | ok | 132,424 | $0.4856 | 20m15s | passed, cycles=132424, instret=None |
| 20260908-025820-f7c4 | 8 | ok | 132,424 | $0.3590 | 19m26s | passed, cycles=132424, instret=None |
| 20260908-025835-a0cc | -1 | ok | 691,515 | $0.0000 | 0m04s | seed from /home/max410011_l/aether/out/loop/20260907-144414-da36/kernel_01.h (run 20260907-144414-da |
| 20260908-025835-a0cc | 0 | ok | 740,101 | $0.0000 | 0m00s | passed, cycles=740101, instret=None |
| 20260908-025835-a0cc | 1 | selfcheck_failed | ? | $1.7998 | 46m16s | simulator rc=255; the benchmark's self-check did not pass |
| 20260908-025835-a0cc | 2 | ok | 706,155 | $0.1998 | 25m24s | passed, cycles=706155, instret=None |
| 20260908-025835-a0cc | 3 | ok | 742,166 | $0.5466 | 35m30s | passed, cycles=742166, instret=None |
| 20260908-025835-a0cc | 4 | ok | 704,084 | $1.5084 | 40m54s | passed, cycles=704084, instret=None |
| 20260908-025835-a0cc | 5 | ok | 659,142 | $1.7806 | 42m02s | passed, cycles=659142, instret=None |
| 20260908-025835-a0cc | 6 | ok | 738,754 | $0.8588 | 36m59s | passed, cycles=738754, instret=None |
| 20260908-025850-2356 | -1 | ok | 8,574 | $0.0000 | 0m00s | seed from /home/max410011_l/aether/out/loop/20260907-043130-fa24/kernel_04.c (run 20260907-043130-fa |
| 20260908-025850-2356 | 0 | ok | 51,280 | $0.0000 | 0m00s | passed, cycles=51280, instret=20524 |
| 20260908-025850-2356 | 1 | ok | 12,138 | $2.8758 | 34m01s | passed, cycles=12138, instret=6895 |
| 20260908-025850-2356 | 2 | ok | 8,233 | $2.1737 | 29m36s | passed, cycles=8233, instret=4281 |
| 20260908-025850-2356 | 3 | ok | 8,763 | $1.7648 | 28m19s | passed, cycles=8763, instret=4732 |
| 20260908-025850-2356 | 4 | ok | 10,740 | $1.2170 | 22m55s | passed, cycles=10740, instret=4473 |
| 20260908-025905-5427 | -1 | ok | 4,853 | $0.0000 | 0m01s | seed from /home/max410011_l/aether/out/loop/20260907-043140-2246/kernel_06.c (run 20260907-043140-22 |
| 20260908-025905-5427 | 0 | ok | 22,856 | $0.0000 | 0m00s | passed, cycles=22856, instret=16877 |
| 20260908-025905-5427 | 1 | ok | 4,991 | $1.2817 | 39m39s | passed, cycles=4991, instret=3009 |
| 20260908-025905-5427 | 2 | ok | 4,967 | $0.5300 | 21m24s | passed, cycles=4967, instret=3014 |
| 20260908-025905-5427 | 3 | ok | 6,034 | $1.2295 | 28m17s | passed, cycles=6034, instret=3009 |
| 20260908-025905-5427 | 4 | ok | 5,416 | $1.1155 | 24m03s | passed, cycles=5416, instret=4044 |
| 20260908-025919-3d8d | -1 | ok | 50,256 | $0.0000 | 0m01s | seed from /home/max410011_l/aether/out/loop/20260907-142225-55f2/kernel_02.c (run 20260907-142225-55 |
| 20260908-025919-3d8d | 0 | ok | 361,260 | $0.0000 | 0m00s | passed, cycles=361260, instret=48180 |
| 20260908-025919-3d8d | 1 | ok | 47,173 | $0.6222 | 8m30s | passed, cycles=47173, instret=7732 |
| 20260908-025919-3d8d | 2 | ok | 35,795 | $1.0935 | 11m14s | passed, cycles=35795, instret=8499 |
| 20260908-025919-3d8d | 3 | ok | 35,774 | $0.7382 | 17m01s | passed, cycles=35774, instret=8241 |
| 20260908-025919-3d8d | 4 | ok | 37,012 | $1.1066 | 9m37s | passed, cycles=37012, instret=10546 |
| 20260908-025933-f639 | -1 | ok | 1,630 | $0.0000 | 0m00s | seed from /home/max410011_l/aether/out/loop/20260907-142235-7231/kernel_01.c (run 20260907-142235-72 |
| 20260908-025933-f639 | 0 | ok | 21,906 | $0.0000 | 0m00s | passed, cycles=21906, instret=3236 |
| 20260908-025933-f639 | 1 | ok | 1,760 | $0.7180 | 7m00s | passed, cycles=1760, instret=535 |
| 20260908-025933-f639 | 2 | ok | 1,624 | $0.9787 | 9m03s | passed, cycles=1624, instret=540 |
| 20260908-025933-f639 | 3 | ok | 1,568 | $0.4729 | 16m16s | passed, cycles=1568, instret=541 |
| 20260908-202413-e8e8 | -1 | ok | 132,424 | $0.0000 | 0m05s | seed from /home/max410011_l/aether/out/loop/20260908-025820-f7c4/kernel_07.h (run 20260908-025820-f7 |
| 20260908-202413-e8e8 | 0 | ok | 150,339 | $0.0000 | 0m00s | passed, cycles=150339, instret=None |
| 20260908-202413-e8e8 | 1 | selfcheck_failed | 779,335 | $3.6266 | 49m33s | simulator rc=255; the benchmark's self-check did not pass |
| 20260908-202413-e8e8 | 2 | ok | 132,424 | $1.0845 | 32m52s | passed, cycles=132424, instret=None |
| 20260908-202413-e8e8 | 3 | ok | 133,147 | $1.4329 | 26m45s | passed, cycles=133147, instret=None |
| 20260908-202413-e8e8 | 4 | ok | 132,424 | $0.3178 | 20m07s | passed, cycles=132424, instret=None |
| 20260909-052323-9d90 | -1 | ok | 132,424 | $0.0000 | 0m04s | seed from /home/max410011_l/aether/out/loop/20260908-202413-e8e8/kernel_02.h (run 20260908-202413-e8 |
| 20260909-052323-9d90 | 0 | ok | 150,339 | $0.0000 | 0m00s | passed, cycles=150339, instret=None |
| 20260909-052323-9d90 | 1 | ok | 182,726 | $8.8021 | 48m59s | passed, cycles=182726, instret=None |
| 20260909-052323-9d90 | 2 | ok | 132,424 | $8.2210 | 47m09s | passed, cycles=132424, instret=None |
| 20260909-052323-9d90 | 3 | ok | 132,424 | $0.8170 | 25m28s | passed, cycles=132424, instret=None |
| 20260909-052323-9d90 | 4 | ok | 132,424 | $0.6585 | 27m57s | passed, cycles=132424, instret=None |
| 20260909-052323-9d90 | 5 | ok | 132,424 | $0.3552 | 28m28s | passed, cycles=132424, instret=None |
| 20260909-052323-9d90 | 6 | ok | 133,513 | $0.4364 | 27m11s | passed, cycles=133513, instret=None |
| 20260909-052323-9d90 | 7 | selfcheck_failed | ? | $2.4242 | 31m10s | simulator rc=255; the benchmark's self-check did not pass |
| 20260909-052323-9d90 | 8 | selfcheck_failed | 133,587 | $0.8248 | 19m18s | simulator rc=255; the benchmark's self-check did not pass |
| 20260908-202428-217e | -1 | ok | 659,142 | $0.0000 | 0m04s | seed from /home/max410011_l/aether/out/loop/20260908-025835-a0cc/kernel_05.h (run 20260908-025835-a0 |
| 20260908-202428-217e | 0 | ok | 740,101 | $0.0000 | 0m00s | passed, cycles=740101, instret=None |
| 20260908-202428-217e | 1 | ok | 738,596 | $0.8565 | 52m31s | passed, cycles=738596, instret=None |
| 20260908-202428-217e | 2 | ok | 729,519 | $0.4753 | 36m54s | passed, cycles=729519, instret=None |
| 20260908-202428-217e | 3 | ok | 810,910 | $1.5685 | 42m20s | passed, cycles=810910, instret=None |
| 20260909-052338-b80e | -1 | ok | 659,142 | $0.0000 | 0m04s | seed from /home/max410011_l/aether/out/loop/20260908-025835-a0cc/kernel_05.h (run 20260908-025835-a0 |
| 20260909-052338-b80e | 0 | ok | 740,101 | $0.0000 | 0m00s | passed, cycles=740101, instret=None |
| 20260909-052338-b80e | 1 | ok | 666,484 | $1.3316 | 40m25s | passed, cycles=666484, instret=None |
| 20260909-052338-b80e | 2 | ok | 738,148 | $0.5002 | 35m49s | passed, cycles=738148, instret=None |
| 20260909-052338-b80e | 3 | ok | 745,566 | $1.7126 | 42m51s | passed, cycles=745566, instret=None |
| 20260909-052338-b80e | 4 | ok | 1,291,233 | $0.6614 | 55m10s | passed, cycles=1291233, instret=None |
| 20260909-052338-b80e | 5 | ok | 635,908 | $0.7745 | 55m44s | passed, cycles=635908, instret=None |
| 20260908-202443-5906 | 0 | ok | 146,737 | $0.0000 | 15m39s | passed, cycles=146737, instret=None |
| 20260908-202443-5906 | 1 | ok | 138,892 | $2.2524 | 9m48s | passed, cycles=138892, instret=None |
| 20260908-202443-5906 | 2 | ok | 140,692 | $0.7604 | 23m23s | passed, cycles=140692, instret=None |
| 20260908-202443-5906 | 3 | ok | 139,137 | $2.8718 | 31m08s | passed, cycles=139137, instret=None |
| 20260908-202443-5906 | 4 | ok | 138,892 | $0.4844 | 18m32s | passed, cycles=138892, instret=None |
| 20260908-202443-5906 | 5 | ok | 138,892 | $0.3626 | 17m13s | passed, cycles=138892, instret=None |
| 20260908-202443-5906 | 6 | ok | 138,892 | $0.2790 | 16m40s | passed, cycles=138892, instret=None |
| 20260908-202458-85ab | -1 | ok | 8,233 | $0.0000 | 0m01s | seed from /home/max410011_l/aether/out/loop/20260908-025850-2356/kernel_02.c (run 20260908-025850-23 |
| 20260908-202458-85ab | 0 | ok | 51,280 | $0.0000 | 0m00s | passed, cycles=51280, instret=20524 |
| 20260908-202458-85ab | 1 | ok | 8,443 | $1.8506 | 38m59s | passed, cycles=8443, instret=4281 |
| 20260908-202458-85ab | 2 | ok | 8,474 | $1.3119 | 25m10s | passed, cycles=8474, instret=4298 |
| 20260908-202458-85ab | 3 | ok | 18,638 | $0.4518 | 19m23s | passed, cycles=18638, instret=5047 |
| 20260908-202458-85ab | 4 | ok | 12,935 | $1.1005 | 24m11s | passed, cycles=12935, instret=5499 |
| 20260908-202513-620b | -1 | ok | 4,853 | $0.0000 | 0m01s | seed from /home/max410011_l/aether/out/loop/20260907-043140-2246/kernel_06.c (run 20260907-043140-22 |
| 20260908-202513-620b | 0 | ok | 22,856 | $0.0000 | 0m00s | passed, cycles=22856, instret=16877 |
| 20260908-202513-620b | 1 | ok | 7,501 | $1.3333 | 45m30s | passed, cycles=7501, instret=4394 |
| 20260908-202513-620b | 2 | ok | 4,916 | $1.3850 | 27m41s | passed, cycles=4916, instret=3141 |
| 20260908-202513-620b | 3 | ok | 4,814 | $1.9709 | 28m17s | passed, cycles=4814, instret=3009 |
| 20260908-202528-f504 | -1 | ok | 35,774 | $0.0000 | 0m01s | seed from /home/max410011_l/aether/out/loop/20260908-025919-3d8d/kernel_03.c (run 20260908-025919-3d |
| 20260908-202528-f504 | 0 | ok | 361,260 | $0.0000 | 0m00s | passed, cycles=361260, instret=48180 |
| 20260908-202528-f504 | 1 | ok | 34,695 | $0.6630 | 8m30s | passed, cycles=34695, instret=7985 |
| 20260908-202528-f504 | 2 | ok | 35,712 | $0.6536 | 7m49s | passed, cycles=35712, instret=8497 |
| 20260908-202528-f504 | 3 | ok | 33,177 | $0.6917 | 38m55s | passed, cycles=33177, instret=7987 |
| 20260908-202528-f504 | 4 | ok | 31,024 | $0.5225 | 7m16s | passed, cycles=31024, instret=8245 |
| 20260908-202543-fc79 | -1 | ok | 1,568 | $0.0000 | 0m01s | seed from /home/max410011_l/aether/out/loop/20260908-025933-f639/kernel_03.c (run 20260908-025933-f6 |
| 20260908-202543-fc79 | 0 | ok | 21,906 | $0.0000 | 0m00s | passed, cycles=21906, instret=3236 |
| 20260908-202543-fc79 | 1 | ok | 1,704 | $1.2208 | 10m09s | passed, cycles=1704, instret=526 |
| 20260908-202543-fc79 | 2 | ok | 1,555 | $2.5916 | 20m58s | passed, cycles=1555, instret=541 |
| 20260908-202543-fc79 | 3 | ok | 1,693 | $2.2292 | 22m52s | passed, cycles=1693, instret=568 |
| 20260909-200541-dae1 | -1 | ok | 635,908 | $0.0000 | 0m05s | seed from /home/max410011_l/aether/out/loop/20260909-052338-b80e/kernel_05.h (run 20260909-052338-b8 |
| 20260909-200541-dae1 | 0 | ok | 740,101 | $0.0000 | 0m00s | passed, cycles=740101, instret=None |
| 20260909-200541-dae1 | 1 | ok | 695,516 | $0.3211 | 34m56s | passed, cycles=695516, instret=None |
| 20260909-200541-dae1 | 2 | ok | 6,614,734 | $0.9277 | 39m24s | passed, cycles=6614734, instret=None |
| 20260909-200541-dae1 | 3 | ok | 690,466 | $1.1632 | 1h20m | passed, cycles=690466, instret=None |
| 20260909-200541-dae1 | 4 | ok | 634,507 | $0.4917 | 35m57s | passed, cycles=634507, instret=None |
| 20260909-200541-dae1 | 5 | selfcheck_failed | ? | $0.8592 | 37m19s | simulator rc=255; the benchmark's self-check did not pass |
| 20260909-200541-dae1 | 6 | ok | 634,907 | $0.5744 | 27m51s | passed, cycles=634907, instret=None |
| 20260909-200541-dae1 | 7 | ok | 1,335,959 | $0.6253 | 36m08s | passed, cycles=1335959, instret=None |
| 20260909-200541-dae1 | 8 | ok | 1,299,397 | $0.4743 | 40m29s | passed, cycles=1299397, instret=None |
| 20260909-200541-dae1 | 9 | ok | 637,154 | $0.7857 | 42m50s | passed, cycles=637154, instret=None |
| 20260909-200541-dae1 | 10 | ok | 634,934 | $0.5438 | 38m52s | passed, cycles=634934, instret=None |
| 20260909-200541-dae1 | 11 | ok | 822,414 | $0.2577 | 35m32s | passed, cycles=822414, instret=None |
| 20260909-200541-dae1 | 12 | ok | 682,974 | $0.2675 | 36m04s | passed, cycles=682974, instret=None |
| 20260909-200556-5de3 | -1 | ok | 138,892 | $0.0000 | 0m05s | seed from /home/max410011_l/aether/out/loop/20260908-202443-5906/kernel_01.h (run 20260908-202443-59 |
| 20260909-200556-5de3 | 0 | ok | 146,737 | $0.0000 | 0m00s | passed, cycles=146737, instret=None |
| 20260909-200556-5de3 | 1 | ok | 137,391 | $2.0579 | 24m34s | passed, cycles=137391, instret=None |
| 20260909-200556-5de3 | 2 | ok | 137,246 | $1.2858 | 20m55s | passed, cycles=137246, instret=None |
| 20260909-200556-5de3 | 3 | ok | 137,280 | $0.8151 | 19m28s | passed, cycles=137280, instret=None |
| 20260909-200556-5de3 | 4 | ok | 137,246 | $0.5399 | 18m17s | passed, cycles=137246, instret=None |
| 20260909-200556-5de3 | 5 | ok | 137,246 | $0.3625 | 17m18s | passed, cycles=137246, instret=None |
| 20260909-200556-5de3 | 6 | ok | 137,246 | $0.1968 | 16m37s | passed, cycles=137246, instret=None |
| 20260909-200611-94c0 | -1 | ok | 132,424 | $0.0000 | 0m03s | seed from /home/max410011_l/aether/out/loop/20260908-202413-e8e8/kernel_02.h (run 20260908-202413-e8 |
| 20260909-200611-94c0 | 0 | ok | 150,339 | $0.0000 | 0m00s | passed, cycles=150339, instret=None |
| 20260909-200611-94c0 | 1 | ok | 768,849 | $0.9403 | 22m35s | passed, cycles=768849, instret=None |
| 20260909-200611-94c0 | 2 | ok | 132,839 | $0.8872 | 27m52s | passed, cycles=132839, instret=None |
| 20260911-065653-8115 | 0 | ok | 168,367 | $0.0000 | 19m51s | passed, cycles=168367, instret=None |
| 20260911-065653-8115 | 1 | ok | 144,521 | $1.4159 | 4m06s | passed, cycles=144521, instret=None |
| 20260911-065653-8115 | 2 | ok | 142,458 | $0.7294 | 23m16s | passed, cycles=142458, instret=None |
| 20260911-065653-8115 | 3 | ok | 768,839 | $1.0795 | 24m05s | passed, cycles=768839, instret=None |
| 20260911-065653-8115 | 4 | ok | 158,708 | $0.6128 | 26m55s | passed, cycles=158708, instret=None |
| 20260911-065653-8115 | 5 | ok | 155,636 | $0.3319 | 20m57s | passed, cycles=155636, instret=None |
| 20260911-065653-8115 | 6 | ok | 142,609 | $0.7653 | 23m00s | passed, cycles=142609, instret=None |
| 20260911-065653-8115 | 7 | ok | 143,005 | $0.8979 | 24m00s | passed, cycles=143005, instret=None |
| 20260911-065653-8115 | 8 | ok | 1,036,276 | $0.8617 | 24m27s | passed, cycles=1036276, instret=None |
| 20260911-065653-8115 | 9 | ok | 145,220 | $0.7698 | 31m01s | passed, cycles=145220, instret=None |
| 20260911-065653-8115 | 10 | ok | 168,980 | $0.5344 | 22m00s | passed, cycles=168980, instret=None |
| 20260911-115840-5f95 | -1 | ok | 142,458 | $0.0000 | 0m06s | seed from /home/max410011_l/aether/out/loop/20260911-065653-8115/kernel_02.h (run 20260911-065653-81 |
| 20260911-115840-5f95 | 0 | ok | 168,367 | $0.0000 | 0m00s | passed, cycles=168367, instret=None |
| 20260911-115840-5f95 | 1 | ok | 688,048 | $0.8128 | 21m52s | passed, cycles=688048, instret=None |
| 20260911-115840-5f95 | 2 | ok | 214,211 | $2.0318 | 35m18s | passed, cycles=214211, instret=None |
| 20260911-115840-5f95 | 3 | ok | 1,992,395 | $1.0695 | 25m59s | passed, cycles=1992395, instret=None |
| 20260911-115840-5f95 | 4 | ok | 268,314 | $1.0044 | 38m08s | passed, cycles=268314, instret=None |
| 20260912-191730-3a47 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260912-191730-3a47 | 1 | ok | 217,295 | $1.1372 | 3m21s | passed, cycles=217295, instret=None |
| 20260912-191730-3a47 | 2 | ok | 189,851 | $2.2364 | 1h08m | passed, cycles=189851, instret=None |
| 20260912-191730-3a47 | 3 | ok | 189,924 | $0.7377 | 58m59s | passed, cycles=189924, instret=None |
| 20260912-191730-3a47 | 4 | ok | 223,640 | $0.8822 | 1h00m | passed, cycles=223640, instret=None |
| 20260912-191830-e713 | -1 | ok | 132,424 | $0.0000 | 0m04s | seed from /home/max410011_l/aether/out/loop/20260908-202413-e8e8/kernel_02.h (run 20260908-202413-e8 |
| 20260912-191830-e713 | 0 | ok | 150,339 | $0.0000 | 0m00s | passed, cycles=150339, instret=None |
| 20260912-191830-e713 | 1 | ok | 3,017,659 | $0.9075 | 22m48s | passed, cycles=3017659, instret=None |
| 20260912-191830-e713 | 2 | ok | 660,409 | $1.6356 | 47m59s | passed, cycles=660409, instret=None |
| 20260913-000137-2132 | 0 | ok | 206,304 | $0.0000 | 0m05s | passed, cycles=206304, instret=None |
| 20260913-004138-786f | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-012139-c080 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-020140-53fc | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-024141-3644 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-032142-e3a6 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-040143-edab | 0 | ok | 206,304 | $0.0000 | 0m06s | passed, cycles=206304, instret=None |
| 20260913-044144-a893 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-052146-2e52 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-060147-fea2 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-064148-3da8 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-072149-c40c | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-080150-e953 | 0 | ok | 206,304 | $0.0000 | 0m05s | passed, cycles=206304, instret=None |
| 20260913-084151-bb60 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-092152-8335 | 0 | ok | 206,304 | $0.0000 | 0m05s | passed, cycles=206304, instret=None |
| 20260913-100153-1f7c | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-104154-43d8 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-112156-d66a | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-120157-3fa8 | 0 | ok | 206,304 | $0.0000 | 0m05s | passed, cycles=206304, instret=None |
| 20260913-124158-4820 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-132159-ff8b | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-140200-51f5 | 0 | ok | 206,304 | $0.0000 | 0m03s | passed, cycles=206304, instret=None |
| 20260913-144201-9d99 | 0 | ok | 206,304 | $0.0000 | 0m03s | passed, cycles=206304, instret=None |
| 20260913-152202-4b7b | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-160203-35d3 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260913-164204-0fd2 | 0 | ok | 206,304 | $0.0000 | 0m04s | passed, cycles=206304, instret=None |
| 20260915-060820-a93a | -1 | ok | 189,851 | $0.0000 | 0m06s | seed from /home/max410011_l/aether/out/loop/20260912-191730-3a47/kernel_02.h (run 20260912-191730-3a |
| 20260915-060820-a93a | 0 | ok | 206,304 | $0.0000 | 0m00s | passed, cycles=206304, instret=None |
| 20260915-060820-a93a | 1 | ok | 179,585 | $9.1983 | 1h27m | passed, cycles=179585, instret=None |
| 20260915-060820-a93a | 2 | ok | 198,688 | $1.3301 | 1h05m | passed, cycles=198688, instret=None |
| 20260915-060820-a93a | 3 | ok | 199,347 | $1.6183 | 1h05m | passed, cycles=199347, instret=None |
| 20260915-060820-a93a | 4 | ok | 181,860 | $2.5125 | 1h07m | passed, cycles=181860, instret=None |
| 20260916-195958-36b4 | -1 | ok | 179,585 | $0.0000 | 0m05s | seed from /home/max410011_l/aether/out/loop/20260915-060820-a93a/kernel_01.h (run 20260915-060820-a9 |
| 20260916-195958-36b4 | 0 | ok | 206,304 | $0.0000 | 0m00s | passed, cycles=206304, instret=None |
| 20260916-195958-36b4 | 1 | ok | 185,374 | $1.8298 | 1h03m | passed, cycles=185374, instret=None |
| 20260916-195958-36b4 | 2 | ok | 180,107 | $0.9998 | 1h01m | passed, cycles=180107, instret=None |
| 20260916-195958-36b4 | 3 | ok | 183,430 | $2.1677 | 1h06m | passed, cycles=183430, instret=None |
| 20260916-195958-36b4 | 4 | ok | 179,033 | $1.4750 | 1h04m | passed, cycles=179033, instret=None |
| 20260917-155834-2887 | 0 | ok | 648,292 | $0.0000 | 0m04s | passed, cycles=648292, instret=None |
| 20260917-155834-2887 | 1 | ok | 676,825 | $0.7500 | 1m22s | passed, cycles=676825, instret=None |
| 20260917-155834-2887 | 2 | ok | 651,726 | $0.6624 | 54m15s | passed, cycles=651726, instret=None |
| 20260917-155834-2887 | 3 | ok | 689,999 | $0.3922 | 53m43s | passed, cycles=689999, instret=None |
| 20260917-155834-2887 | 4 | ok | 698,428 | $0.5909 | 54m28s | passed, cycles=698428, instret=None |
