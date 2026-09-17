# PROBE line harvest

This document catalogues every `PROBE`-prefixed diagnostic line found under
`out/loop/`, plus prose-reported numbers from the pre-harness-fix era that
describe the same kind of measurement but were never captured in a `simlog`.

**Grep used (reproducible):**

```
grep -rn "PROBE" out/loop/*/simlog_*.txt
```

This returned **149 `PROBE` lines**, spanning **20 (run_id, iter) file pairs**
across **6 run_ids**: `20260911-065653-8115` (round5, `llama-q8-gemv-gemmini-n16`),
`20260911-115840-5f95` (round6, same kernel, continuation),
`20260912-191730-3a47` (round7, `llama-layer-fused-n1`),
`20260912-191830-e713` (round7, `llama-q8-gemv-gemmini-n1`, dedicated
in-flight-scan / warm-cold probe run), `20260915-060820-a93a` (round8,
`llama-layer-fused-n1`, continuation of round7's search), and
`20260916-195958-36b4` (round9, `llama-layer-fused-n1`, continuation of
round8's search). Run-to-kernel mapping is from `out/loop/rounds.json`.

**Caveat — pre-round5 era:** the harness only began saving `simlog_NN.txt`
starting round5 (2026-09-11). Iterations before that (rounds 1-4, and the
`smoke`/`round2`/`round2b`/`round3`/`round4` run dirs) have **no simlog
capture at all**. A `PROBE`/`LQ8PROBE`-style measurement from that era, if it
survives, exists only as **prose inside `agent_NN.txt`** describing what the
agent printed and expected — the actual printed numbers were never persisted
to a scoreable log, so these rows are far less precise/verifiable and are
flagged with source type `agent_NN.txt prose (pre-harness-fix, no simlog
capture)`. Two such run_ids were found: `20260909-200541-dae1` (round4,
`llama-q8-gemv-gemmini-lmhead`) and `20260909-200611-94c0` (round4,
`llama-q8-gemv-gemmini-n1`).

`out/loop/20260916-195958-36b4/` (round9, now finished) was read-only
exploration; its 19 `PROBE` lines across iters 1-4 are folded into sections
B and C below.

---

## A. DMA rate probes (warm/cold, B/cycle, rows-per-mvin and in-flight sweeps)

| run_id | iter | kernel | condition | value | source |
|---|---|---|---|---|---|
| 20260911-065653-8115 | 08 | llama-q8-gemv-gemmini-n16 | Region A read warm (no L2 flush), 32 hand-issued 16-row×64B mvin commands, stride 2048 (kernel's real A-access stride) | `A2048_warm=3630` cycles | out/loop/20260911-065653-8115/simlog_08.txt:3 (agent_08.txt) |
| 20260911-065653-8115 | 08 | llama-q8-gemv-gemmini-n16 | Region A read cold (after two B-only hardware-loop L2-flush streams), stride 2048 | `A2048_cold=5593` cycles | out/loop/20260911-065653-8115/simlog_08.txt:4 (agent_08.txt) |
| 20260911-065653-8115 | 08 | llama-q8-gemv-gemmini-n16 | Region A read cold, stride 512 (probe: does a different stride change the cold cost?) | `A512_cold=5787` cycles | out/loop/20260911-065653-8115/simlog_08.txt:5 (agent_08.txt) |
| 20260911-065653-8115 | 08 | llama-q8-gemv-gemmini-n16 | Region A read cold, stride 64 (contiguous) | `A64_cold=5647` cycles | out/loop/20260911-065653-8115/simlog_08.txt:6 (agent_08.txt) |
| 20260911-115840-5f95 | 03 | llama-q8-gemv-gemmini-n16 | Same best kernel replayed 3 ways in one invocation: (s)=scored/harness state, (r)=clean-cold replay, (d)=host deliberately re-dirties L2 before replay to mimic harness residue. Per-loop issue-gap totals across the 16-loop mvin stream. | `s_total=142892 r_total=154742 d_total=141213`; drain `s=36088 r=31412 d=35830` (16 per-loop gap values each, loop_00..loop_15, not all reproduced here — see file) | out/loop/20260911-115840-5f95/simlog_03.txt:3-21 (agent_03.txt) |
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | rows_per_mvin sweep on a cold 1 MiB mvin-only stream (L2 flushed via fixed 1.5 MiB stream first each time). rows_per_mvin=3, nominal in-flight = 2×rows (capped at 16) = 6 | `flush_cycles=274021`, `cold_mvin_cycles=166067`, `cold_bytes_per_cycle_x100=631` (6.31 B/cycle) | out/loop/20260912-191830-e713/simlog_01.txt:3-7 (agent_01.txt) |
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | rows_per_mvin=4, inflight=8 | `flush_cycles=276279`, `cold_mvin_cycles=158109`, `cold_bytes_per_cycle_x100=663` (6.63 B/cycle) — fastest point in this sweep | out/loop/20260912-191830-e713/simlog_01.txt:8-12 (agent_01.txt) |
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | rows_per_mvin=5, inflight=10 | `flush_cycles=276368`, `cold_mvin_cycles=173029`, `cold_bytes_per_cycle_x100=606` (6.06 B/cycle) | out/loop/20260912-191830-e713/simlog_01.txt:13-17 (agent_01.txt) |
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | rows_per_mvin=6, inflight=12 | `flush_cycles=276702`, `cold_mvin_cycles=179649`, `cold_bytes_per_cycle_x100=583` (5.83 B/cycle) | out/loop/20260912-191830-e713/simlog_01.txt:18-22 (agent_01.txt) |
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | rows_per_mvin=8, inflight=16 (nominal pool fully used by one command) | `flush_cycles=276693`, `cold_mvin_cycles=181520`, `cold_bytes_per_cycle_x100=577` (5.77 B/cycle) | out/loop/20260912-191830-e713/simlog_01.txt:23-27 (agent_01.txt) |
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | rows_per_mvin=16, inflight=16 (capped; one command fills entire 16-slot XactTracker pool) | `flush_cycles=276716`, `cold_mvin_cycles=184298`, `cold_bytes_per_cycle_x100=568` (5.68 B/cycle) — worst point | out/loop/20260912-191830-e713/simlog_01.txt:28-32 (agent_01.txt) |
| 20260912-191830-e713 | 02 | llama-q8-gemv-gemmini-n1 | Full winning GEMV kernel, warm L2 (harness-residual state, no flush) | `warm_gemv_cycles=132435`, `warm_bytes_per_cycle_x100=791` (7.91 B/cycle) | out/loop/20260912-191830-e713/simlog_02.txt:3-4 (agent_02.txt) |
| 20260912-191830-e713 | 02 | llama-q8-gemv-gemmini-n1 | Same full GEMV kernel, cold L2 (after the 1.5 MiB flush stream) | `cold_gemv_cycles=184837`, `cold_bytes_per_cycle_x100=567` (5.67 B/cycle) — the paper-relevant cold correction vs. the ~149-150k/7.0 B/cycle predicted by earlier rounds | out/loop/20260912-191830-e713/simlog_02.txt:8-10 (agent_02.txt) |
| **pre-harness-fix** 20260909-200541-dae1 | 02 | llama-q8-gemv-gemmini-lmhead | Prose-only (no simlog): 8 mvin-only probes inserted ahead of best kernel v6 — 4-row warm/cold, 2/8/16-row @ stride 2048, 16-row & 4-row @ a "fake" stride 512 (same bytes), 1-row pure sequential — testing whether the ~6.6 B/cycle ceiling is a request-shape artifact or the true cold-DRAM floor. EXPECTED values only (agent's predictions, not measured/printed to a captured log): 4-row cold ≈648k, warm ≈632k, 2-row ≈625k, 8-row ≈660k, 16-row ≈695k, 16-row@fake-512 ≈570k, 1-row ≈900k (host-issue-bound) | source type: **agent_NN.txt prose (pre-harness-fix, no simlog capture)** — EXPECTED, not confirmed-measured, numbers | out/loop/20260909-200541-dae1/agent_02.txt:5-9 |
| **pre-harness-fix** 20260909-200611-94c0 | 01 | llama-q8-gemv-gemmini-n1 | Prose-only (no simlog): best kernel run once as-is (~132.4k, "run1"), then a 2 MiB garbage-region DMA stream is used to flush ~512 KiB of L2, then a pure-cold rerun ("run2") — testing the never-directly-measured pure cold rate. EXPECTED run2 ≈148-150k (1 MiB / 7.0-7.1 B/cycle) | source type: **agent_NN.txt prose (pre-harness-fix, no simlog capture)** — EXPECTED only; agent_02.txt of the same run reports the actual scored total came back as 768,849 (includes flush+printf overhead, not a clean run2 number) and notes the PROBE stdout was never forwarded to the agent | out/loop/20260909-200611-94c0/agent_01.txt:3-5, agent_02.txt:1 |

**Cross-reference — `loop/kernels.py` rows-per-mvin table (read-only, not modified):**
`kernels.py` lines 899-905 hold an older, separate Round-4 table for the
**4 MiB lmhead stream** (not the same run as e713's 1 MiB n1-kernel sweep):

| rows per mvin | requests in flight | lmhead cycles | B/cycle |
|---|---|---|---|
| 3 (probe) | 6 | ~649,000 | 6.46 |
| 4 (optimum) | 8 | 634,507 | 6.61 |
| 5-6 | 10-12 | 682,974 | 6.14 |
| 8 | 16 | 690,466 | 6.07 |
| 16 | 16 (capped) | 695,516 | 6.03 |

This is a **different measurement** (4 MiB lmhead stream, pre-round5, no
simlog — itself only traceable to `kernels.py` prose, not a raw log) than
`20260912-191830-e713`'s 1 MiB sweep, so the absolute cycle counts differ
(634k vs 158-184k) as expected for a 4x-larger data volume. The **qualitative
curve matches**: both sweeps peak at rows=3-4 (inflight=6-8) and degrade
monotonically above that, and `kernels.py` (lines 4116-4123, per the
sub-agent that read it) explicitly cites run `20260912-191830-e713` and
reproduces its warm/cold GEMV numbers (132,435 / 184,837) verbatim — no
inconsistency found between `kernels.py`'s citation of e713 and the e713
simlog itself. `out/loop/round7-partial.md` (around lines 88-260) also
reproduces the e713 PROBE values verbatim, matching exactly.

---

## B. Phase-split probes

| run_id | iter | kernel | condition | value | source |
|---|---|---|---|---|---|
| 20260911-065653-8115 | 03 | llama-q8-gemv-gemmini-n16 | Exploratory decomposition of the kernel's residual cost into checkpoints: T1 = full kernel scored-state cycle count; T2 = mvin of A+B at 16-row shape; T3 = mvin of B-only at 16-row shape; T4 = mvin of A+B with B issued as 4-row commands; T5 = full kernel post-state. Values are cycles-at-checkpoint (not deltas). | `T1_full_scoredstate=142734`, `T2_mvin_AB_16row=153012`, `T3_mvin_Bonly_16row=141805`, `T4_mvin_AB_B4row=148560`, `T5_full_poststate=149764` | out/loop/20260911-065653-8115/simlog_03.txt:3-7 (agent_03.txt) |
| 20260911-065653-8115 | 10 | llama-q8-gemv-gemmini-n16 | Best kernel run unmodified; `rdcycle` taken at the last loop-issue point and again after the final fence, to expose the un-hidden "tail" after the last B-tile lands. `issue_done_at` = cycle when the final mvin issue completes; `stream_total` = cycle at fence; `tail_after_issue` = stream_total − issue_done_at (a derived tail, not independently clocked). | `stream_total=142073`, `issue_done_at=105790`, `tail_after_issue=36283` | out/loop/20260911-065653-8115/simlog_10.txt:3-5 (agent_10.txt) |
| 20260911-065653-8115 | 10 | llama-q8-gemv-gemmini-n16 | Same iteration, separate probe: C region (already correct) replayed with 32 hand-issued mvout commands timed alone, isolating pure mvout cost for a 32×64B (2 KiB) region. | `mvout32=2109` cycles | out/loop/20260911-065653-8115/simlog_10.txt:6 (agent_10.txt) |
| 20260911-115840-5f95 | 01 | llama-q8-gemv-gemmini-n16 | Best kernel (K-descending, tile_K=8, single mvout) run with per-loop issue timestamps recorded in 3 conditions in one invocation: scored run ("s"), per-loop-fence replay ("f"), plain replay ("r"). `s_gap`/`r_gap` = cycles since the previous loop's issue in that condition; `f_loop` = per-loop duration under per-loop-fence replay; `s_at`/`r_at` = cumulative timestamps; drain = tail after the last loop's issue to fence. 16 per-loop lines (loop_00..loop_15) recorded; totals shown here. | `s_total=142024`, `f_total=157503`, `r_total=154014`, `s_drain=36061`, `r_drain=30903` | out/loop/20260911-115840-5f95/simlog_01.txt:3-23 (agent_01.txt) |
| 20260911-115840-5f95 | 04 | llama-q8-gemv-gemmini-n16 | Best kernel with the first two K-descending slices swapped (loop_00 starts at kstart=112, loop_01 at kstart=120, vs. the previous strictly-descending order); `gap` = per-loop issue gap (same metric family as s_gap above); tests whether loop_0's extra gap is an L1D-probe effect (would move with the swap) or a pipeline-startup effect (would stay at loop_0). 16 per-loop lines recorded; totals/tail shown here. | `total=143125`, `drain=35883`, `mvout2=2097` (independent mvout timing of the C accumulator region, no longer L1D-resident) | out/loop/20260911-115840-5f95/simlog_04.txt:3-20 (agent_04.txt) |
| 20260912-191730-3a47 | 01 | llama-layer-fused-n1 | `rdcycle` timestamps (relative to kernel entry) at each of the 9 `loop_ws` dispatch returns (`iss`), plus the timestamp at the end of each inter-dispatch Saturn work "gap" (gap1-4 = 128-row QK^T slices, gap5 = softmax, gap6 = probs@V, gap0/7/8 = prefetch-only gaps), plus `f` = fence-completion timestamp. Purpose per agent_01.txt: recover the cold-weight-stream real rate and whether Saturn vector work stalls RoCC issue. | `iss = 152 221 16791 32461 47932 63838 77360 108302 124612`; `gap = 195 14154 29894 45372 61276 73015 105628 108318 124628`; `f=190487` | out/loop/20260912-191730-3a47/simlog_01.txt:6-7 (agent_01.txt, interpretation continued in agent_02.txt) |
| 20260912-191730-3a47 | 04 | llama-layer-fused-n1 | Reverted to the strict sequential baseline schedule (prefetch bursts restored); pure measurement iteration (not scored for search), adding `rdcycle` at the 9 `loop_ws` dispatch returns (`iss`), fence completion (`f`), and cumulative phase-end times for scores (`s`), softmax (`m`), and probs@V (`p`), each measured from kernel entry — isolating "cold-state GEMV inherent duration + uncontended Saturn per-phase cost." | `iss = 2956 11600 33347 55019 76763 98555 120433`; `f=186245 s=196526 m=198168 p=205864` (implied phase costs: scores ≈10.3k, softmax ≈1.6k, pv ≈7.7k) | out/loop/20260912-191730-3a47/simlog_04.txt:6-7 (agent_04.txt) |
| 20260915-060820-a93a | 02 | llama-layer-fused-n1 | Pure-probe iteration (command stream left byte-identical to the 179,585-cycle version, but Saturn compute units no-op'd); after fence, QK^T, softmax, and probs@V run cold/uncontended purely to isolate their standalone cost. `stream=` B-tile mvin streaming phase; `qk=` QK^T; `softmax=`; `pv=` probs@V — 4 sequential rdcycle segments. Purpose: determine whether the 15-20k gap vs. the scored total comes from B-stream rate or un-hidden cold-K8/V8 Saturn misses. | `stream=161687` (6.49 B/cycle for the 1 MiB stream), `qk=16569`, `softmax=1613`, `pv=10190` (interpreted in agent_03.txt: Saturn cold-run total 28.4k vs. ~15.0k warm — ~17.9k exposed, ~10.5k successfully hidden) | out/loop/20260915-060820-a93a/simlog_02.txt:6 (agent_02.txt, interpretation continued in agent_03.txt) |
| **round9** 20260916-195958-36b4 | 01 | llama-layer-fused-n1 | `sat_done` = cycle at which Saturn's own attention/elementwise compute finishes; `gemv_end` = cycle at which the whole kernel (incl. Gemmini GEMV weight streaming) finishes; `warm_cmds` = count of DMA-warming commands issued ahead of use (this iter's CHANGE, which regressed). Purpose: check whether warming Saturn's operands into L2 ahead of time hides more of Saturn's cost behind the GEMV stream. | `sat_done=106708`, `gemv_end=185332`, `warm_cmds=267` (sat_done/gemv_end = 57.6%) | out/loop/20260916-195958-36b4/simlog_01.txt:5-7 |
| **round9** 20260916-195958-36b4 | 02 | llama-layer-fused-n1 | Same `sat_done`/`gemv_end` split, this iter halving the Saturn unit grain (finer-grained interleaving with the mvin stream) instead of DMA-warming. | `sat_done=99956`, `gemv_end=179890` (sat_done/gemv_end = 55.6%) | out/loop/20260916-195958-36b4/simlog_02.txt:5-6 |
| **round9** 20260916-195958-36b4 | 03 | llama-layer-fused-n1 | Probe-only iteration on the unchanged iter-5/best kernel (no functional change): same `sat_done`/`gemv_end` split, plus new host-dispatch-cost probes — `unit_host_cycles`/`unit_calls` = total host wall time inside `lf_sat_unit` calls and the call count (rdcycle around every unit); `mvin_stall_sampled`/`mvin_samples` = sampled rdcycle-to-rdcycle span around every 8th `mvin2` issue and the sample count. Purpose: distinguish "host co-bottlenecked with DMA" (Model C) from "host has slack, DMA/vector traffic contends" (Model D). | `sat_done=104365`, `gemv_end=183228` (57.0%); `unit_host_cycles=48799`, `unit_calls=256` (~190.6 c/unit-call); `mvin_stall_sampled=2558`, `mvin_samples=512` (~5.0 c/sampled mvin) | out/loop/20260916-195958-36b4/simlog_03.txt:5-9 |

**Round9 interpretation, CORRECTED** (all 4 iterations of run
`20260916-195958-36b4`, seeded from round8's 179,585 best): `sat_done` sits
at only 55.6-57.6% of `gemv_end` in every probed iteration, but this is
**not** evidence Saturn's compute is "mostly hidden/overlapped." It is a
geometric artifact: the 1 MiB weight stream issues 4,096 `mvin`s, Saturn is
called once every 16 of them, and only the first ~130 calls have real
attention work to do (one head's QK^T/softmax/PV is a fixed, small amount
of work) — so Saturn's calls necessarily finish partway through the mvin
stream regardless of how much of its cost is actually hidden. `sat_done`
measures *when Saturn runs out of assigned work*, not *how much of its cost
survives as exposed wall time*.

The actual exposure is visible in the **mvin rate**, from the round9-seed
probe (`out/loop/20260915-060820-a93a/simlog_02.txt`): the first 2,080
mvins (Saturn active) average 50.18 c/mvin; the remaining 2,016 (Saturn
idle) average 39.12 c/mvin; a Saturn-free stream runs at 39.47 c/mvin
(161,687/4,096). So `179,033 = 161,687` (pure stream) `+ 17,346` (Saturn
cost still exposed, 9.7% of the total) — and `17,346/28,372 = 0.611`, i.e.
**61.1% of Saturn's own compute is still exposed**, essentially unchanged
from round8's own 17,898-cycle residual (iter6 `PROBE stream=` probe). This
is corroborated by iter2's near-null result from halving the unit grain
(179,585 -> 180,107, +522 cycles) — a genuinely-exposed 61% residual that
DMA-warming and grain-halving both failed to reduce, not a residual so small
that finer interleaving had nothing left to improve. iter3's host figures
add a second, independent finding: `mvin_stall_sampled` averages only ~5.0
cycles/sampled mvin, so the mvin/DMA issue path itself essentially never
blocks the host (ruling out DMA-issue stalls as dominant);
`unit_host_cycles/unit_calls` ≈190.6 host cycles per unit-call is the larger,
partially-exposed cost, and iter4 attributes it to scalar RoCC-operand
recomputation per command rather than to waiting on the memory system —
consistent with iter4's fix (incrementally-updated RoCC operands, no address
rebuild) being the only change this round that produced a net improvement
(179,033, -552 cycles vs. round8's best). Net: host command dispatch/batching
overhead is *a* promising remaining lever, but the ~61%-exposed Saturn cost
has not been shown to be near a floor either — four iterations of one
technique (DMA-warming, grain-halving) not working is not the same as
Saturn-side optimization being played out.

---

## C. Host-issue-cost probes

| run_id | iter | kernel | condition | value | source |
|---|---|---|---|---|---|
| 20260912-191830-e713 | 02 | llama-q8-gemv-gemmini-n1 | Isolating host RoCC-issue overhead underneath `loop_ws`/`LoopMatmul`: `passthru_loop_issue_cycles` = cycles for the host to hand off one K-tile `loop_ws` command and regain control; `passthru_64mvin_accept_cycles` = cycles for 64 hand-issued 1-row mvins (to an unused scratchpad row) to be accepted while `loop_ws` executes underneath; `passthru_loop_plus_mvins_total_cycles` = the combined total. Result shows `LoopMatmul` blocks passthrough issue much more than expected (~15.1k, not "a few hundred"), so extra hand-issued commands cannot be freely interleaved under a running `loop_ws`. | `passthru_loop_issue_cycles=38`, `passthru_64mvin_accept_cycles=15126`, `passthru_loop_plus_mvins_total_cycles=15333` | out/loop/20260912-191830-e713/simlog_02.txt:5-7 (agent_02.txt) |
| **pre-harness-fix** 20260909-200541-dae1 | 11 | llama-q8-gemv-gemmini-lmhead | Prose-only (no simlog): `lq8_probe_issue` — 16,384 hand-issued 1-row×16B mvin2 commands, all hitting L2, fence-bounded, timed with `rdcycle`; printed as `LQ8PROBE issue: ... per_cmd=%lu`. HYPOTHESIS: host-issue cost ≈20 cycles/command. This is an EXPECTED/hypothesis value, not a confirmed-measured one (no captured log). | source type: **agent_NN.txt prose (pre-harness-fix, no simlog capture)** — hypothesis ≈20 cycles/command; EXPECTED total ≈780k-975k depending on assumed per-command cost | out/loop/20260909-200541-dae1/agent_11.txt:16-18 |

---

## D. Other probes

| run_id | iter | kernel | condition | value | source |
|---|---|---|---|---|---|
| 20260912-191830-e713 | 01 | llama-q8-gemv-gemmini-n1 | L2-flush cost: a fixed 1.5 MiB unrelated-DRAM 16-row×64B mvin stream run before each rows_per_mvin sweep point, to force a cold L2 state. Recorded once per sweep point (6 instances this iter); constant within ~1% (274,021-276,716) regardless of the swept rows_per_mvin, confirming it is independent of the measured parameter. | `flush_cycles = 274021, 276279, 276368, 276702, 276693, 276716` | out/loop/20260912-191830-e713/simlog_01.txt:5,10,15,20,25,30 (agent_01.txt) |
| 20260912-191830-e713 | 02 | llama-q8-gemv-gemmini-n1 | Same L2-flush cost, reused ahead of the warm/cold full-GEMV comparison. | `flush_cycles=274464` | out/loop/20260912-191830-e713/simlog_02.txt:8 (agent_02.txt) |

---

## E. Uncategorized / meaning unclear

| run_id | iter | kernel | raw PROBE text | note |
|---|---|---|---|---|
| 20260912-191730-3a47 | 01, 02, 03, 04 | llama-layer-fused-n1 | `PROBE ref_score_min_micro=-9160885` / `PROBE ref_score_max_micro=9122776` (appears in every iter of this run) | condition unmeasured/unclear as a hardware measurement — no `agent_NN.txt` in this run ever discusses these lines; they appear to be reference-score correctness bounds emitted by the scoring harness itself rather than an agent-authored hardware probe, but this could not be confirmed from agent prose alone. Not treated as a DMA/phase/host-issue measurement. |
| 20260915-060820-a93a | 01, 02, 03, 04 | llama-layer-fused-n1 | Same `PROBE ref_score_min_micro=...` / `PROBE ref_score_max_micro=...` pair, identical values, every iter | Same as above — condition unmeasured/unclear; likely a harness-level correctness-bound printout, not an agent probe. |
| 20260916-195958-36b4 | 01, 02, 03, 04 | llama-layer-fused-n1 | Same `PROBE ref_score_min_micro=...` / `PROBE ref_score_max_micro=...` pair, identical values, every iter (round9) | Same as above — condition unmeasured/unclear; likely a harness-level correctness-bound printout, not an agent probe. |

---

## Summary of coverage

- 149 raw `PROBE` lines found via the grep above, across 20 (run_id, iter) simlog files and 6 run_ids (round5-round9).
- 2 additional pre-round5 run_ids (`20260909-200541-dae1`, `20260909-200611-94c0`, both round4) contain prose-only `PROBE`/`LQ8PROBE` mentions with no simlog capture; these are flagged throughout as `agent_NN.txt prose (pre-harness-fix, no simlog capture)` and their numbers are EXPECTED/hypothesis values from the agent, not confirmed measurements.
- Category D `flush_cycles` lines (7 instances) and the per-loop `loop_NN ...` lines within categories A/B are individually numerous (most of the 130 raw lines) but are grouped into single table rows per iteration/condition above rather than transcribed one-line-per-row, to keep the table readable; the exact per-loop breakdown is preserved in the cited simlog files.
