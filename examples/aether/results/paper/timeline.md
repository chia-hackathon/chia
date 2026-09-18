# CHIA Inner-Loop Timeline — Reproducibility Appendix

Scope: the CHIA-based LLM-kernel optimization inner loop targeting the
Gemmini/Saturn RISC-V SoC (`GENV256D128GemminiShuttleConfig`,
model `claude-fable-5-1`), from the pre-round1 smoke test through
round9 (finished; this document reflects final round9 numbers).

Primary sources (read in this order, not re-derived): `out/loop/rounds.json`,
`out/loop/FINAL_REPORT.md`, `out/loop/ledger.md` / `ledger.json`,
`out/loop/round7-partial.md`, `out/loop/round7-prep.md`,
`out/loop/round{7,8,9}-watchdog.log`, `out/loop/round{8,9}-runs.txt`,
`out/loop/round9-launch.sh`, `loop/kernels.py` (read-only).
All costs/iteration counts for smoke–round6 are `ledger.md`'s authoritative
`SUM(cost_usd)` / `MIN(fitness)` rollups over `loop/aether.db`'s `iters`
table (not the `runs` table, which is only updated by a clean `finish_run()`
and would misreport any run that died mid-flight); round7–9 use the same
source. Roofline throughout uses the corrected 8 B/cycle mbus bandwidth
(see "Cross-cutting incidents" below) and a 1.00 GHz clock assumption unless
noted otherwise (§ audit correction flags this assumption as unverified —
RTL elaboration shows 500 MHz).

---

## smoke test

- **Date (UTC)**: 2026-09-06T19:37:28 → 19:48:24 (wall-clock ~10m56s).
  Source: `out/loop/rounds.json` (`name: "smoke"`), `ledger.md` §a.
- **Kernel**: `llama-softmax`.
- **Run**: `20260906-193727-ebbf`, planned_iters=2.
- **Iterations**: 3 (3 ok / 0 fail). **Cost**: $0.6494.
- **Findings**:
  - Smoke test of the inner loop itself, before round1 began.
  - `llama-softmax` improved 21,906 → 4,312 cycles.
- **Incidents**: none recorded.

---

## round1

- **Date (UTC)**: 2026-09-07T04:31:20 → 08:17:47 (wall-clock 3h46m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernels / runs** (5): `llama-q8-gemv-gemmini-n1` (`20260907-043120-a7d0`, planned 8),
  `llama-attn-scores-int8` (`20260907-043130-fa24`, planned 6),
  `llama-attn-pv-int8` (`20260907-043140-2246`, planned 6),
  `llama-silu-mul` (`20260907-043150-7557`, planned 6),
  `llama-softmax` (`20260907-043200-f0d7`, planned 4).
- **Iterations**: 35 (33 ok / 2 fail). **Cost**: $20.6599.
- **Findings** (source: `FINAL_REPORT.md` §1):
  - First full-board search; captured most of the round's own headroom in one pass.
  - `llama-softmax` 21,906 → 1,752 cycles (12.5x).
  - `llama-attn-scores-int8` 51,280 → 8,574 (6.0x).
  - `llama-silu-mul` 361,260 → 66,811 (5.4x).
  - `llama-attn-pv-int8` 22,856 → 4,853 (4.7x).
  - `llama-q8-gemv-gemmini-n1` 150,339 → 132,653 (already ~1.012x roofline at this
    stage, under the *pre-correction* 16 B/cycle roofline — see round4 note on the
    later bandwidth fix).
- **Incidents**: none recorded.

---

## round2

- **Date (UTC)**: 2026-09-07T14:21:55/56 → 14:57:36 (wall-clock ~35–36m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernels / runs** (6, all `note: "aborted: session limit"`):
  `llama-q8-gemv-gemmini-n1` (`20260907-142155-86bf`, planned 8),
  `llama-attn-scores-int8` (`20260907-142205-6c45`, planned 4),
  `llama-attn-pv-int8` (`20260907-142215-b093`, planned 4),
  `llama-silu-mul` (`20260907-142225-55f2`, planned 4),
  `llama-softmax` (`20260907-142235-7231`, planned 3),
  `llama-q8-gemv-gemmini-lmhead` (`20260907-144414-da36`, planned 6).
- **Iterations**: 17 (16 ok / 1 fail). **Cost**: $7.3727.
- **Findings** (`FINAL_REPORT.md` §1):
  - First round to introduce `llama-q8-gemv-gemmini-lmhead` (740,101 → 691,515 cycles).
  - `llama-silu-mul` → 50,256; `llama-softmax` → 1,630 before the interruption.
- **Incident**: **Claude session-limit 429 interruption #1.** All 6 runs launched
  together were cut off ~36 minutes in; only 4 of 6 runs reached any iteration.
  Source: `rounds.json` round2 note ("first round2 launch, aborted mid-run
  (session limit)"), `FINAL_REPORT.md` §4 row 1.

---

## round2b (round2 restart)

- **Date (UTC)**: 2026-09-08T02:58:20/21 → 07:19:49 (wall-clock 4h21m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernels / runs** (6, all `--seed best`, all continuations of round2's
  kernel set): `llama-q8-gemv-gemmini-n1` (`20260908-025820-f7c4`, 8),
  `llama-q8-gemv-gemmini-lmhead` (`20260908-025835-a0cc`, 6),
  `llama-attn-scores-int8` (`20260908-025850-2356`, 4),
  `llama-attn-pv-int8` (`20260908-025905-5427`, 4),
  `llama-silu-mul` (`20260908-025919-3d8d`, 4),
  `llama-softmax` (`20260908-025933-f639`, 3).
- **Iterations**: 41 (39 ok / 2 fail). **Cost**: $31.2473.
- **Findings** (`FINAL_REPORT.md` §1):
  - Restart of round2 after the session-limit reset (2026-09-08T11:00:03+08:00),
    seeded from `--seed best` so no progress was lost.
  - `llama-q8-gemv-gemmini-n1` → **132,424** cycles — this value was not beaten
    for the next five rounds (through round6), and is the value later revisited
    by the 2026-09-12 audit correction (see Cross-cutting incidents).
  - `llama-q8-gemv-gemmini-lmhead` → 659,142; `llama-attn-scores-int8` → **8,233**;
    `llama-silu-mul` → 35,774; `llama-softmax` → 1,568.
- **Incidents**: none new (this round *is* the recovery from round2's 429).

---

## round3

- **Date (UTC)**: 2026-09-08T20:24:12/14 → 2026-09-09T09:58:14 (wall-clock 13h34m,
  the most expensive round). Source: `rounds.json`, `ledger.md` §a.
- **Kernels / runs** (9 rows, all `--seed best --budget-usd 1000` except the new
  kernel): `llama-q8-gemv-gemmini-n1` (`20260908-202413-e8e8`, planned 12,
  aborted after 4 iters by 429, restarted as `20260909-052323-9d90` planned 8);
  `llama-q8-gemv-gemmini-lmhead` (`20260908-202428-217e`, planned 8, aborted
  after 3 iters by 429, restarted as `20260909-052338-b80e` planned 5);
  `llama-q8-gemm` (`20260908-202443-5906`, planned 6, new kernel this round);
  `llama-attn-scores-int8` (`20260908-202458-85ab`, 4);
  `llama-attn-pv-int8` (`20260908-202513-620b`, 3);
  `llama-silu-mul` (`20260908-202528-f504`, 4);
  `llama-softmax` (`20260908-202543-fc79`, 3).
- **Iterations**: 57 (54 ok / 3 fail). **Cost**: $61.8682 (41.5% of the total
  round1–round6 cost, per `FINAL_REPORT.md` §4 row 6).
- **Findings** (`FINAL_REPORT.md` §1):
  - Introduced `llama-q8-gemm` (prefill kernel): 146,737 → 138,892.
  - `llama-q8-gemv-gemmini-lmhead` → 635,908; `llama-attn-pv-int8` → **4,814**;
    `llama-silu-mul` → **31,024**; `llama-softmax` → **1,555**.
  - At round end, discovered the **roofline memory-bandwidth bug** (mbus is
    8 B/cycle, not the previously assumed 16) — see Cross-cutting incidents;
    every "distance to roofline" ratio computed before this point was restated.
- **Incidents**:
  - **Claude session-limit 429 interruption #2** (by the task brief's counting,
    this round counts as one occurrence even though it struck twice): `n1` was
    interrupted after 4 iterations, `lmhead` after 3, both resuming via
    `--seed best`.
  - **Watchdog bug #1 (timezone-parsing).** The watchdog's auto-restart logic
    misparsed the Anthropic API's "resets 5pm America/Los_Angeles" message as
    local (Taipei) time, so the automatic restart never fired correctly; a
    human manually restarted both interrupted runs at 2026-09-09T13:23+08:00
    instead. Source: `rounds.json` round3 note (verbatim: "the watchdog's
    auto-restart misparsed the API's 'resets 5pm America/Los_Angeles' message
    as local time, so a human manually restarted both at 2026-09-09T13:23+08:00
    instead of waiting for the watchdog"), `FINAL_REPORT.md` §4 row 2.

---

## round4

- **Date (UTC)**: 2026-09-09T20:05:40/42 → 2026-09-10T04:50:33 (wall-clock 8h44m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernels / runs** (3, all `--seed best --budget-usd 1000`):
  `llama-q8-gemv-gemmini-lmhead` (`20260909-200541-dae1`, planned 12);
  `llama-q8-gemm` (`20260909-200556-5de3`, planned 6);
  `llama-q8-gemv-gemmini-n1` (`20260909-200611-94c0`, planned 2 — a deliberate
  convergence-check probe, not a full search, since round3 had already
  converged this kernel to ~1.01x roofline).
- **Iterations**: 26 (25 ok / 1 fail). **Cost**: $14.3770.
- **Findings** (`rounds.json` round4 note, `FINAL_REPORT.md` §1):
  - First round measured/analyzed entirely under the **corrected 8 B/cycle
    roofline** (fixed just before this round launched — see Cross-cutting
    incidents).
  - Two infra events predate this round but gate its interpretation: (1) the
    roofline bandwidth fix (2026-09-09); (2) Ray's temp/spill directory
    migrated from local disk to `/share1` (2026-09-09,
    `out/loop/migrate-ray-to-share1.sh`), following the root-disk-full incident.
  - `llama-q8-gemv-gemmini-lmhead` → **634,507** (only 1.002x better — the
    hoped-for `LQ8_B_ROWS` 4→16 retiling did not pay off); `llama-q8-gemm` →
    **137,246**; the n1 probe confirmed 132,424 unchanged.
  - All three runs FINISHED cleanly — no interruptions this round, unlike round3.
- **Incidents**: none new this round (references the round3 disk-full/roofline
  events as pre-existing context).

---

## round5

- **Date (UTC)**: 2026-09-11T06:56:00/54 → 11:19:47 (wall-clock ~4h22m–4h23m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernel / run**: `llama-q8-gemv-gemmini-n16` (`20260911-065653-8115`, planned 10)
  — the pristine, never-before-optimized N=16 batched-decode GEMV tile.
- **Iterations**: 11 (11 ok / 0 fail). **Cost**: $7.9986.
- **Findings**:
  - Baseline 168,367 → best **142,458** cycles (1.087x the 131,072-cycle roofline).
  - Starting this round, the harness began saving `simlog_NN.txt` per iteration,
    and passing-iteration feedback started including a stdout tail — this is the
    **harness printf/simlog fix** (see Cross-cutting incidents; it actually
    landed just before this round, 2026-09-11).
- **Incidents**: none.

---

## round6

- **Date (UTC)**: 2026-09-11T11:58:00/40 → 14:20:24 (wall-clock ~2h21m–2h23m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernel / run**: `llama-q8-gemv-gemmini-n16` continuation, seeded from
  round5's best (`20260911-115840-5f95`, planned 4).
- **Iterations**: 6 (6 ok / 0 fail). **Cost**: $4.9184.
- **Findings**:
  - Seed 142,458 → all 4 iterations regressed (688,048 and 1,992,395 among the
    reported probe values) — best remained 142,458, **no new record**.
  - Interpreted as this kernel also having hit its ceiling; the round5+round6
    n16 work is treated as converged and the inner loop paused kernel-level
    search at this point (round7 introduces a new kernel instead of continuing
    n16).
- **Incidents**: none.

---

## round7

- **Date (UTC)**: 2026-09-12T19:17:31 → 2026-09-13T16:42:08 per `ledger.md`
  (21h24m wall-clock, 28 run rows); `rounds.json`'s narrative gives the same
  span as "launched 2026-09-13T11:17:00+08:00 Taipei; ended early at
  2026-09-14T00:50:00+08:00" when the operator manually stopped the watchdog.
- **Kernels / runs**: 28 run rows total, but only **2 real runs**:
  - `llama-layer-fused-n1` (**new kernel** this round — see kernel note below),
    run `20260912-191730-3a47`, planned 8, completed 4 of 8 before the cap:
    baseline 206,304, roofline 140,680 → iter1 217,295 (regression), iter2
    **189,851 (new best, −8.0% vs baseline)**, iter3 189,924 (flat), iter4
    223,640 (deliberate probe regression, for a 4-phase timing breakdown).
  - `llama-q8-gemv-gemmini-n1` in-flight-scan probe, run `20260912-191830-e713`,
    planned 2, both iterations deliberately probe-only (not score attempts):
    iter1 3,017,659 (in-flight-depth scan, 6 cold mvin-only sweeps), iter2
    660,409 (warm/cold contrast: warm 132,435 cyc @7.91 B/cycle vs cold
    184,837 cyc @5.67 B/cycle).
  - The remaining **26 run rows** (`20260913-000137-2132` through
    `20260913-164204-0fd2`, plus one `20260914-000202-*` directory) are
    watchdog auto-restarts of `llama-layer-fused-n1` after the cap hit; each
    only re-ran the baseline probe (`kind=baseline`, cycles=206,304,
    cost_usd=0.0) before immediately hitting another 429 — no real iteration
    or cost was produced by any of them.
- **Iterations**: 35 total (35 ok / 0 fail per `ledger.md` — the 26 restarts'
  baseline-only probes count as trivially "ok" 0-cost rows). **Cost**: $7.5365
  (`iters` table `SUM(cost_usd)` across all 28 run rows).
- **Findings** (`round7-partial.md`, `round7-prep.md`):
  - **New kernel `llama-layer-fused-n1`**: fuses one Gemmini decode GEMV
    (1x2048x512 int8, 1 MiB weights) together with one attention head's
    QK^T/softmax/probs@V on Saturn in a single timed region, attempting to
    overlap Saturn's ~14.6k-cycle (warm-basis) arithmetic tail inside the
    Gemmini weight-DMA's idle "gaps" between `loop_ws` issues, rather than
    running the four pieces back to back. Registered in `loop/kernels.py`
    around line 4224 (`LLAMA_LAYER_FUSED_N1 = Kernel(name="llama-layer-fused-n1", ...)`).
  - Moving the terminal `gemmini_fence()` and relocating `probs@V` into the
    idle tail after the last `loop_ws` issue recovered ~16k cycles
    (206,304 → 189,851), but iter3 showed the DRAM-contention cost is
    additive/position-independent, not further hideable by relocation alone.
  - The `llama-q8-gemv-gemmini-n1` probe overturned an earlier assumption:
    the true cold `loop_ws`-shape DMA rate is **5.68 B/cycle**, not the
    ~7.0–7.1 B/cycle assumed in `kernels.py`; the 132,424-cycle "best" carries
    ~52k cycles of warm-L2-tail credit, not the ~17k previously assumed. It
    also found a real in-flight-depth optimum at 8-in-flight (4 rows/mvin,
    6.63 B/cycle) vs the kernel's native 16-in-flight/16-rows shape
    (5.68 B/cycle) — an unexploited ~15k-cycle lever.
  - Prep work (`round7-prep.md`) derived the kernel's roofline: memory floor
    140,680 cycles (dominant), compute floor 11,712 cycles — an unreachable
    hard lower bound, since the SoC's measured cold weight-streaming rate is
    6.0–7.1 B/cycle, not the nominal 8.0.
- **Incidents**:
  - **Claude weekly-usage-cap 429 (a third 429-type event — see the
    discrepancy flagged in Cross-cutting incidents).** The cap hit at
    2026-09-13T08:01:00+08:00 (00:01:00Z); the reset was scheduled for
    2026-09-15T06:00:00Z. Source: `rounds.json` round7 note, `round7-watchdog.log`.
  - **Watchdog bug #2 (restart-loop burning wall-clock after a correctly
    detected cap).** After the cap, `round7-watchdog.sh` correctly detected
    the 429 each time but auto-restarted the `llama-layer-fused-n1` run
    **52 times** (26 of which produced a new `run_id` in the DB, since some
    restart attempts failed before even registering a run) over ~16.5 hours,
    every restart immediately hitting another 429 on its first agent turn and
    producing zero iterations/cost. `round7-watchdog.log`'s final line
    (2026-09-14T00:47:39+08:00): *"WATCHDOG STOPPED by operator: weekly usage
    limit hit 2026-09-13T08:01, resets 2026-09-15T06:00Z (09-15 14:00
    Taipei). 52 restarts all 429'd, 0 iterations gained."* This is a
    distinct failure mode from round3's bug: the watchdog's *rate-limit
    detection* worked correctly this time, but it lacked any backoff /
    cap-duration awareness, so it kept burning wall-clock in a tight
    ~30-minute restart loop for ~13 hours until a human intervened, instead
    of sleeping until the known reset time.

---

## round8

- **Date (UTC)**: 2026-09-15T06:08:20/21 → 11:51:24 (wall-clock ~5h43m–5h44m).
  Source: `rounds.json`, `ledger.md` §a.
- **Kernel / run**: `llama-layer-fused-n1`, run `20260915-060820-a93a`, planned 4,
  **operator-restarted continuation of round7's cut-short 8-iteration search**
  (round7 completed only 4 of 8; round7+round8 together make up the full
  8-iteration search for this kernel). Seeded from round7's best (189,851
  cycles, run `20260912-191730-3a47` iter 2) with `--seed best`.
- **Iterations**: 6 (6 ok / 0 fail per `ledger.md`; 4 of these are the
  numbered score-attempt iterations described below, the rest are
  baseline/seed bookkeeping rows). **Cost**: $14.6592.
- **Findings** (`rounds.json` round8 note, `ledger.md` §b):
  - baseline 206,304, roofline 140,680 → iter1 **179,585 (new overall best,
    −13.0% vs baseline, cumulative round7+round8 result)**, iter2 198,688
    (regression), iter3 199,347 (regression), iter4 181,860 (close to best,
    not improving).
  - This closes out the "layer-fused" search opened in round7 at a cumulative
    −13.0% improvement across the combined 8 iterations.
- **Incidents**:
  - **Watchdog bug #3 (false-positive rate-limit detection + stale planned
    count).** `round8-watchdog.log`'s final line
    (2026-09-15T20:03:05+08:00): *"WATCHDOG STOPPED: round8 completed 4/4
    (best 179585). Watchdog misfired RATE-LIMITED because its grep for '429'
    matched raylet disk-warning lines containing pid 429; PLANNED was still 8
    from the round7 copy. No restart wanted."* Two compounding bugs in one
    event: (a) the watchdog's `429`-detection grep matched an unrelated
    process ID ("429") inside a Ray raylet disk-space warning line rather
    than an actual HTTP 429 response, producing a false "RATE-LIMITED"
    status; (b) the watchdog script was copied from round7 without updating
    its hardcoded `PLANNED=8` iteration count, even though round8 only
    launched 4. Neither bug caused any data loss here — the run had already
    completed 4/4 by the time the watchdog misfired, so "no restart wanted" —
    but this is a **third distinct watchdog bug**, beyond the two the task
    brief's framing expects (see Cross-cutting incidents discrepancy note).

---

## round9

- **Date (UTC)**: launched 2026-09-16T19:59:57/59 (= 2026-09-17T04:00:00+08:00
  Taipei per `out/loop/rounds.json`'s round9 entry; `runs.started` in
  `out/loop/aether.db` confirms `2026-09-16T19:59:59+00:00`). Ended
  `runs.finished` = `2026-09-17T01:14:23+00:00` (≈2026-09-17T09:14+08:00
  Taipei), ~5h14m wall-clock. Round9 is now recorded in
  `out/loop/rounds.json` and reflected in `out/loop/ledger.md` / `ledger.json`
  and `out/loop/iterations.md` / `iterations.json` (regenerated via
  `loop/ledger.py` and `loop/journal.py` against the finished run).
- **Kernel / run**: `llama-layer-fused-n1`, run `20260916-195958-36b4`,
  `--iters 4 --seed best` (seeded from round8's best, 179,585 cycles, run
  `20260915-060820-a93a` iter 1). Only one kernel/run was launched this
  round, all 4 planned iterations completed.
- **Iterations**: 4 scored (4 ok / 0 fail), plus baseline/seed bookkeeping
  rows (6 total per `ledger.md`). **Cost**: $6.4724.
- **Findings** (`rounds.json` round9 note, `ledger.md` §b, `out/loop/20260916-195958-36b4/summary.txt`):
  - baseline 206,304, roofline 140,680 → iter1 **185,374** (regression;
    Gemmini-DMA operand warming, `warm_cmds=267`, didn't help — the exposed
    cost stayed ~13k, i.e. the warm compute itself, not cold-operand misses),
    iter2 **180,107** (halved unit grain; close but not a new best — exposure
    barely changed, ruling out "unit exceeds DMA backlog depth" as the
    mechanism), iter3 **183,430** (probe-only iteration on the unchanged
    179,585 kernel, regression from added `rdcycle` overhead — but yielded
    the round's key host-dispatch-cost data), iter4 **179,033 (new overall
    best, −13.2% vs baseline)**.
  - This continues the "layer-fused" search opened in round7 (189,851,
    −8.0%) and advanced in round8 (179,585, −13.0%) to a new cumulative
    result of 179,033 cycles, −13.2% vs. baseline, 1.273x roofline (down
    from round8's 1.277x). Round7+round8+round9 together total 12
    iterations on this kernel.
  - **New this round — phase-split probes (interpretation corrected).**
    iter1-3 added `sat_done` (Saturn compute completion) vs. `gemv_end`
    (kernel end) instrumentation: iter1 106,708/185,332 (57.6%), iter2
    99,956/179,890 (55.6%), iter3 104,365/183,228 (57.0%). **This does not
    mean Saturn's compute is mostly hidden.** `sat_done` finishing at
    ~56% of `gemv_end` is a geometric consequence of the mvin/Saturn-call
    schedule (4,096 mvins, one Saturn call per 16, only ~130 calls with real
    work), not a measurement of hidden cost. The mvin-rate breakdown (see
    `out/llama-profile/projection_round9.md`) shows the real exposure:
    179,033 = 161,687 (pure weight stream) + 17,346 cycles of Saturn cost
    still exposed — 61.1% of Saturn's own compute (17,346/28,372), about the
    same as round8's 17,898-cycle residual. iter1's operand-warming and
    iter2's unit-regrain attempts having little effect means those specific
    techniques didn't reduce the ~61% exposure — not that there was little
    exposure left to reduce.
  - **New this round — host-dispatch-cost probes.** iter3 also measured
    `unit_host_cycles=48799` over `unit_calls=256` (~190.6 host cycles per
    unit-call) and `mvin_stall_sampled=2558` over `mvin_samples=512` (~5.0
    stall cycles per sampled mvin). The near-zero mvin-issue stall rules out
    the DMA/mvin issue path itself as the dominant remaining cost; the
    larger unit-call host time, attributed by iter4 to per-mvin scalar
    address/operand recomputation rather than memory waits, is what iter4's
    change (incrementally-updated 64-bit RoCC operands, no address rebuild)
    targeted — the only change this round that produced a net improvement
    (552 cycles). See `out/llama-profile/projection_round9.md` for the full
    interpretation.
  - **Practical implication, corrected**: host command dispatch/batching is
    *a* promising remaining lever (iter4's 552-cycle gain came from there),
    but Saturn's own compute is not shown to have diminishing returns —
    61.1% of it is still exposed on the critical path, unreduced by round9's
    attempts. Also **retracted**: applying the fused kernel's 13.2% cycle
    reduction uniformly to the whole decode step (5.092 tok/s @ 1 GHz,
    +15.2%) is an invalid extrapolation — the fused kernel's Saturn cycle
    share (13.8%) is an order of magnitude larger than the whole decode
    step's actual Saturn share (0.7-2.2%). The measured, tool-native figure
    via `--overlap` is 0.3-1.4% (4.419 -> 4.432-4.483 tok/s @ 1 GHz). See
    `out/llama-profile/projection_round9.md`.
- **Incidents**: none — 4/4 iterations completed, no watchdog restarts or
  rate-limit events logged for this round.

---

## round10

- **Date (UTC)**: launched 2026-09-17T15:58:35+00:00 (= 2026-09-17 15:58
  Taipei local machine time per `out/loop/rounds.json`'s round10 entry;
  `runs.started` in `out/loop/aether.db` confirms
  `2026-09-17T15:58:35+00:00`). Ended `runs.finished` =
  `2026-09-17T19:36:42+00:00` (≈2026-09-18 03:36 local machine time),
  ~3h38m wall-clock (overnight). Round10 is now recorded in
  `out/loop/rounds.json` and reflected in `out/loop/ledger.md` / `ledger.json`
  and `out/loop/iterations.md` / `iterations.json` (regenerated via
  `loop/ledger.py` and `loop/journal.py` against the finished run).
- **Kernel / run**: **new kernel** `llama-lmhead-fused-n1`, run
  `20260917-155834-2887`, `--iters 4`. This is the LM-head-shaped
  counterpart to `llama-layer-fused-n1` (round7-9): one Gemmini LM-head GEMV
  (1x2048x2048 int8, 4 MiB of weights) fused, in one timed region, with the
  final RMSNorm, int8 quantisation of the next hidden state, and a running
  argmax over the logits — no attention work anywhere in it. Only one
  kernel/run was launched this round, all 4 planned iterations completed.
- **Purpose**: test whether `llama-layer-fused-n1`'s measured Saturn/Gemmini
  overlap exposure ratio (0.611, `DEFAULT_OVERLAP_FACTOR` in
  `loop/llama_project.py`) extrapolates to the LM head, which is ~20.5% of
  decode cycles and streams 4x the weight bytes of the layer-fused kernel's
  GEMV with none of its attention companion work.
- **Iterations**: 4 scored (4 ok / 0 fail), plus a baseline row (5 total per
  `ledger.md`). **Cost**: $2.3956.
- **Findings** (`rounds.json` round10 note, `ledger.md` §b,
  `out/loop/20260917-155834-2887/summary.txt`):
  - baseline 648,292, roofline 532,232 → iter1 **676,825** (denominator-only
    probe, schedule untouched), iter2 **651,726** (the actual optimization
    attempt: fence moved to after the last mvout, Saturn's five sub-phases
    interleaved into an 80-unit state machine inside the weight stream —
    regressed), iter3 **689,999** (probe-only, held-cycles instrumentation
    on iter2's interleaved schedule), iter4 **698,428** (probe-only,
    spin-wait scan, reverted to the baseline serialised schedule). **Best
    cycles remain 648,292 (the baseline) — never beaten. Speedup 1.000x
    (+0.0%), a clear negative result.**
  - **New this round — phase-split probe (iter1).** `PROBE stream=634,821
    rms=7,296 quant=2,496 argmax=2,980 total=647,593`. Serialised Saturn
    work (`rms+quant+argmax`) = **12,772 cycles**, only **1.97%** of the
    iteration total — versus `llama-layer-fused-n1`'s Saturn share of its
    own baseline, `28,372/206,304 = 13.8%`. lm_head's overlappable Saturn
    work is proportionally far smaller than layer-fused's to begin with.
  - **New this round — held-cycles probe (iter3), on iter2's interleaved
    schedule.** `held_rms1=5,984 held_rms2=8,990 held_quant=6,552
    held_amax1=2,093 held_amax2=3,532` (sum **27,151**), `total=653,221`.
    Exposure = 653,221 − 634,821 = 18,400, so the 80 host departures hid only
    27,151 − 18,400 = 8,751 cycles net (~109 cycles/departure) — far below
    the ~390-cycle queued-DMA shadow iter2's model assumed. Likely mechanism
    (per `kernel_04.h`/`agent_04.txt`, not independently re-derived): the
    interleaved phases are spaced 530 KiB of weight stream apart (> the
    512 KiB L2), so cold data is evicted and re-fetched between visits —
    stated as the most probable explanation, flagged as inference rather
    than an independently confirmed mechanism.
  - **New this round — spin-wait scan (iter4), on the reverted baseline
    serialised schedule.** `spin0_period_sum=154,598 spin80=154,376
    spin250=150,380 spin600=145,647` (cnt≈32 each), `total=663,023`.
    `period_sum` decreases monotonically as the scalar spin length
    increases — tighter/longer host busy-waiting modestly reduces
    synchronization latency/overhead — but this did not translate into
    iter4's overall total (698,428) beating the baseline: the effect exists
    but is far too small relative to lm_head's fundamental exposure problem.
  - **Conclusion**: lm_head's measured exposure ratio is effectively
    **~1.0** (baseline never beaten; best/roofline unchanged at 1.218x).
    `0.611` does **not** extrapolate to the LM head — exactly the outcome
    `loop/kernels.py`'s own pre-round notes for this kernel predicted
    ("exposure ratio ~1.0 ... this is the outcome the arithmetic
    predicts"). See `out/llama-profile/projection_round10.md` for the full
    interpretation and a recommendation that `loop/llama_project.py
    --overlap` should not apply one global overlap constant across kernel
    families.
- **Incidents**: none — 4/4 iterations completed, no watchdog restarts or
  rate-limit events logged for this round.

---

## Cross-cutting incidents

| # | Incident | Round(s) affected | Source |
|---|---|---|---|
| 1 | **Claude session-limit 429 interruptions.** The task brief's framing expects exactly two (round2, round3). This investigation confirms round2 (one launch, all 6 runs aborted ~36 min in) and round3 (two separate runs — `n1` after 4 iters, `lmhead` after 3 iters — both resumed via `--seed best`) as genuine session-limit 429s. **However, round7 also hit a real 429 wall** — a *weekly usage cap* (not a per-session limit) at 2026-09-13T08:01+08:00, resetting 2026-09-15T06:00Z. Whether this counts as a "third 429" depends on whether weekly-cap 429s are grouped with session-limit 429s; **this document flags it as a discrepancy against the brief's "two 429s" framing rather than silently omitting it.** | round2, round3, (round7 — flagged discrepancy) | `out/loop/rounds.json` (round2, round3, round7 notes), `out/loop/FINAL_REPORT.md` §4 row 1, `out/loop/round7-watchdog.log` |
| 2 | **Watchdog bugs.** The task brief's framing expects exactly two (round3 timezone bug, round7 restart-loop issue). This investigation found those two, **plus a third, previously undocumented watchdog bug in round8**: a false-positive 429 detection (the watchdog's grep for "429" matched an unrelated raylet process-ID string in a disk-space warning, not an actual HTTP 429) compounded by a stale hardcoded `PLANNED=8` iteration count copy-pasted from round7's watchdog script (round8 only planned 4). **This is a genuine discrepancy against the brief's "two watchdog bugs" expectation** — flagged explicitly rather than reconciled away, since it did not cause data loss (round8 had already finished 4/4 when the watchdog misfired) but is nonetheless a distinct bug. | round3, round7, round8 | round3: `out/loop/rounds.json` round3 note. round7: `out/loop/round7-watchdog.log` final line ("52 restarts all 429'd, 0 iterations gained"). round8: `out/loop/round8-watchdog.log` final line ("Watchdog misfired RATE-LIMITED because its grep for '429' matched raylet disk-warning lines containing pid 429; PLANNED was still 8 from the round7 copy"). |
| 3 | **Root disk full + migration to `/share1`.** Ray's temp/spill directory on local disk filled the root partition; migrated 2026-09-09. | pre-round4 (context for round4) | `out/loop/rounds.json` round4 note, `out/loop/migrate-ray-to-share1.sh`, `FINAL_REPORT.md` §4 row 3 |
| 4 | **Roofline bandwidth correction: 16 → 8 B/cycle.** Gemmini's weight/activation traffic goes over rocket-chip's mbus (`MemoryBusParams(beatBytes = 8)`), physically 8 B/cycle, not the previously assumed 16. Fixed 2026-09-09, before round4 launched. Doubled all memory-bound roofline cycle counts (n1/n16 gemv 65,536→131,072; lmhead 262,144→524,288; q8-gemm unchanged, compute-bound). `llama_project.py --mem-bytes-per-cycle` still defaults to the stale 16.0 — any rerun must pass `--mem-bytes-per-cycle 8.0` explicitly. | round3 (discovered at round end) → round4 onward (first round analyzed under the fix) | `out/loop/README.md` "Roofline correction (2026-09-09)", `FINAL_REPORT.md` §3(1), `out/llama-profile/projection_round3.md` |
| 5 | **Harness printf/simlog fix enabling reliable PROBE capture.** Landed before round5 (2026-09-11). Three changes: (1) `loop/loop.py` now saves each iteration's simulator log to `out/loop/<run_id>/simlog_NN.txt`; (2) `loop/llm.py`'s new `_benchmark_stdout_tail()` (line 212, called from line 284) attaches a stdout tail to *passing*-iteration feedback too (previously only failing iterations saw stdout); (3) added `loop/tests/test_llm_feedback.py` (4 passed). Before this fix, probe numbers printed via `printf` were lost once an iteration passed — several round4-and-earlier probes (e.g. `20260908-202413-e8e8` iter1, `20260909-200541-dae1` iters 2/7/8/11) have only narrative reconstructions in `agent_NN.txt`, not structured data. | pre-round5 (affects interpretability of round1-4 probe data retroactively) | `FINAL_REPORT.md` §3(8) and §4 row 4 |
| 6 | **Warm-L2 illusion / 2026-09-12 audit correction.** The `llama-q8-gemv-gemmini-n1` "best" of 132,424 cycles (round2b onward) is not a purely cold measurement — it includes ~200 KiB of harness-left warm L2 residue from the benchmark's own DRAM fill; true cold-ascending measurement is **149,757 cycles = 7.00 B/cycle** (upper bound). Related corrections in the same audit: (a) lm_head's cold rate is 6.61 B/cycle (not a >8 B/cycle n1-basis extrapolation, which was physically impossible); (b) the simulator uses the no-timing `mm_magic_t` DRAM model, not DRAMSim2 (never passes `+dramsim`/`+dramsim_ini_dir` — see `loop/nodes.py:156-168`, `chia/chipyard/verilator_run_node.py:513-519`) — so the true bottleneck for cold weight streaming is L2 MSHR occupancy/bank conflicts, not "DRAMSim2 cold-stream" behavior as originally described **[SUPERSEDED — see incident 7 below: the 2026-09-19 hardware sweep found the MSHR attribution was itself wrong; the real limit is memory-bus width]**; (c) the 1 GHz clock assumption was never validated — RTL elaboration shows sbus/pbus/fbus/mbus/cbus all running at 500 MHz (`out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`), yielding a revised cold-state N=1 decode projection of 5.345/5.014 tok/s @1GHz (upper/lower bound) or 2.673/2.507 tok/s @500MHz, versus the original warm-state 6.260 tok/s @1GHz. | round1-6 numbers reinterpreted; discovered 2026-09-12, after round6 and before round7 | `FINAL_REPORT.md` §7, `out/llama-profile/projection_final_v2.md` |
| 7 | **Hardware sweep correction (2026-09-19): the L2-MSHR bottleneck claim (incident 6b) was itself wrong.** A four-point outer-loop design-point sweep (`GENV256D128GemminiShuttleConfig` control, plus WideMbus/DeepMshr/WideDeep variants overriding `MemoryBusKey.beatBytes` and `InclusiveCacheKey.memCycles`) isolated mbus width from L2 MSHR count directly. Result: widening the mbus alone (8→16 B/cycle) captures 98.6% of the achievable gain (n1 cold 5.67→8.92 of 9.05 B/cycle); doubling the L2 MSHR file alone (12→24) changes the 4 MiB lm-head cold stream by 0.01% and the 1 MiB n1 cold stream by 6.5%. The control's own warm rate (7.92 B/cycle against an 8 B/cycle bus, 99.0% of roofline) shows there was never MSHR-shaped headroom — the in-flight-request scan that originally motivated the MSHR hypothesis was measuring queueing congestion at a narrow bus, not an MSHR shortage. **Corrected conclusion: decode is bounded by memory-bus width; the L2 MSHR file is not the binding constraint at either width.** This is the project's 5th documented correction. | (retroactively) round1-9 memory-bound analysis; measured 2026-09-19, after the loop concluded (round10) | `out/hw-sweep/README.md`, `out/paper/hardware_sweep.csv` |

---

## 2026-09-19 — Hardware sweep (outer loop, single dimension)

A four-point design-point sweep testing proposal step 4 (the hardware outer
loop), run manually rather than through `loop/` itself — see
`out/paper/methodology.md` §7 for the full method, verification approach,
and limitations, and `out/hw-sweep/README.md` for the complete write-up.

- **What was swept**: memory-bus width (`beatBytes` 8→16, i.e. 8→16 B/cycle)
  crossed with L2 MSHR count (12→24, via `InclusiveCacheKey.memCycles`
  40→88), a 2x2 factorial: control, WideMbus, DeepMshr, WideDeep.
- **Kernels**: `llama-q8-gemv-gemmini-n1` (cold, 1 MiB weight stream) and
  `llama-q8-gemv-gemmini-lmhead` (cold, 4 MiB weight stream) — the same
  kernel binaries used in round7/round9-10, unmodified, so results are
  directly comparable to the loop's own recorded cycle counts (the control
  point reproduces both exactly: n1 cold 184,837, lmhead 634,507).
- **Measured cold cycles / B-per-cycle / speedup vs. control**:

  | design point | mbus | L2 MSHR | n1 cycles | n1 B/c | n1 speedup | lmhead cycles | lmhead speedup |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | control | 8 | 12 | 184,837 | 5.67 | 1.000x | 634,507 | 1.000x |
  | DeepMshr | 8 | 24 | 173,633 | 6.04 | 1.065x | 634,580 | 0.9999x |
  | WideMbus | 16 | 12 | 117,526 | 8.92 | 1.573x | 472,616 | 1.343x |
  | WideDeep | 16 | 24 | 115,865 | 9.05 | 1.595x | 472,093 | 1.344x |

- **Finding — the standing MSHR hypothesis was inverted.** See incident 7
  above and `out/hw-sweep/README.md` §4. Widening the mbus alone gets 98.6%
  of the two-knob gain; deepening MSHRs alone gets 6.5% (n1) or ~0% (lmhead).
- **End-to-end decode projection** (`loop/llama_project.py --scenario
  decode --S 512`, cold): control 2.29 → WideDeep 3.47 tok/s @500 MHz
  (**1.51x**); @1 GHz, 4.58 → 6.93 tok/s.
- **Verification that the config fragments actually took effect**: the
  generated device tree's `sifive,mshr-count` (12/12/24/24) and the
  generated `TestHarness.sv`'s `SimDRAM .DATA_BITS()` (64/128/64/128)
  were both checked against the elaborated artefacts, not the source — see
  `out/hw-sweep/elaborate.*.log`.
- **Status**: proposal step 4 (hardware outer loop) partially delivered —
  one manually-driven dimension of a single design axis, not the automated
  loop, and not a Pareto front (no area/power data). See `out/paper-
  submission/README.md` §What is still missing, item 8 (updated).
- **Incidents**: none — all four elaborate/build/run cycles completed
  cleanly (`rc=0`, `PASSED`). A Ray raylet disk-space warning (5.4 GB free
  of 876 GB) was logged throughout but caused no task failure.

---

## Roll-up (through round10, final)

Computed by summing `ledger.md` §a's per-round rows (smoke through round10,
now that round10 has finished and is recorded in `rounds.json`/`ledger.md`).

| | value |
|---|---:|
| Rounds covered (incl. smoke) | 11 (smoke, round1, round2, round2b, round3, round4, round5, round6, round7, round8, round9, round10 — 12 named entries, "10 optimization rounds" per the task's round1-round10 framing plus the pre-round1 smoke test) |
| Runs | 63 (62 through round9 + 1 round10) |
| Iterations | 248 (243 through round9 + 5 round10) |
| Cost | **$180.1551** ($177.7596 through round9 + $2.3956 round10) |
| Wall-clock | **~73h58m** (~70h19m through round9 + 3h38m round10) |
| Overall best (llama-layer-fused-n1) | **179,033 cycles** (round9 iter4), −13.2% vs. baseline 206,304, 1.273x roofline (140,680) — unchanged by round10, which optimizes a different kernel |
| llama-lmhead-fused-n1 (round10, new kernel) | **648,292 cycles = baseline, never beaten**; speedup 1.000x (+0.0%), a clear negative result. Roofline 532,232 (1.218x). Measured exposure ratio ≈1.0 — `llama-layer-fused-n1`'s 0.611 overlap factor does NOT extrapolate to the LM head. |

Kernels touched across the whole loop (8 from round1-6, plus 2 new in
round7/round10):
`llama-q8-gemv-gemmini-n1`, `llama-q8-gemv-gemmini-n16`,
`llama-q8-gemv-gemmini-lmhead`, `llama-q8-gemm`, `llama-attn-scores-int8`,
`llama-attn-pv-int8`, `llama-silu-mul`, `llama-softmax`,
`llama-layer-fused-n1` (introduced round7, continued round8-9), and
`llama-lmhead-fused-n1` (introduced round10, single-round negative result).
