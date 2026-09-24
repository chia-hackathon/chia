# out/paper/ — Paper data appendix

Read-only reproducibility appendix for the CHIA inner-loop LLM-kernel
optimization experiments (Gemmini/Saturn RISC-V SoC,
`GENV256D128GemminiShuttleConfig`). Everything here is derived from
`loop/aether.db`, `out/loop/rounds.json`, `out/loop/*.md`/`*.json`,
`out/loop/<run_id>/{agent_NN.txt,kernel_NN.h,simlog_NN.txt}`, and
`loop/kernels.py` — all read-only against `out/loop/aether.db` (never opened
for writing; the DB was already fully populated by the finished loop run
itself), except that `out/loop/rounds.json` gained a round9 entry once that
round finished, and `loop/ledger.py`/`loop/journal.py` were then re-run
(read-only against the DB, writing only their own derived
`ledger.md`/`ledger.json`/`iterations.md`/`iterations.json` outputs) to fold
round9's final results into this appendix. Generated 2026-09-17 (round9
finished, final numbers throughout).

> **Update (2026-09-18, post-round10):** a Round 10 (`llama-lmhead-fused-n1`)
> ran after this appendix was generated and is a completed negative result
> (648,292 -> 648,292 cycles, 1.00x). It is not folded into the per-round
> narrative below, which still reflects round9 as the last completed round.
> The true final loop totals, including round10, are **63 runs, 248 DB
> rows, 159 LLM-invoking iterations (150 accepted/8 correctness-fail/1
> build-fail), $180.16, ~74 wall-clock hours** — see
> `results/loop/ledger.json`/`ledger.md` (round10-inclusive) and
> `../../README.md`'s Results summary / Aggregate loop statistics sections
> for the authoritative current numbers.

## Files

- **`experiments.csv`** (335 lines incl. header, 334 data rows) — one row
  per `(run_id, iter)` across every round from the pre-round1 `smoke` test
  through round9 (round9's single run, `20260916-195958-36b4`, is complete
  and its numbers are final). Columns: `round, kernel, run_id, iter, status,
  cycles, baseline_cycles, roofline_cycles, ratio_to_roofline, is_new_best,
  cost_usd, hypothesis_summary`.
  - Source: `loop/aether.db` (`runs`+`iters` tables, opened
    `mode=ro`), `out/loop/rounds.json`, `loop/kernels.py`
    (`KERNELS[k].roofline_cycles` — current, i.e. post-2026-09-09-fix
    values), and per-iteration `agent_NN.txt` / `kernel_NN.h` header
    `HYPOTHESIS:` markers.
  - Logic reused from `loop/journal.py` (`agent_note_for`,
    `extract_hypothesis`, `build_run_rows`'s baseline/roofline/new-best
    computation) rather than reimplemented; the helper script that adds
    the two raw columns journal.py doesn't expose (`baseline_cycles`,
    `roofline_cycles`) is kept at `out/paper/scratch/build_csvs.py` for
    inspection/reuse.
  - iter `-1` = seed (from `--seed best`), iter `0` = baseline (pristine
    kernel). `is_new_best` is True only where that iteration improved the
    running-minimum `fitness` within its own run.
  - **Reproduce:**
    `~/.conda/envs/chia_env/bin/python3 out/paper/scratch/build_csvs.py`
    (reads only; do not redirect its output back into `out/loop/`).

- **`kernels_final.csv`** (9 rows) — one row per distinct kernel:
  `kernel, baseline_cycles, final_best_cycles, roofline_cycles_corrected,
  ratio_final_to_roofline, verdict, final_run_id, final_iter`.
  `roofline_cycles_corrected` is the value in `loop/kernels.py` today
  (i.e. already reflecting the 2026-09-09 mbus 16→8 B/cycle fix — see
  `timeline.md`'s "roofline correction" entry and `methodology.md` §4).
  `verdict` is SOLVED / TERMINAL / ACTIVE:
  - `llama-layer-fused-n1` is the only ACTIVE kernel (round7-9; round9
    finished with a new best of 179,033 cycles, but the kernel is not
    declared SOLVED/TERMINAL — no round has yet closed the gap to its
    140,680 roofline, so future rounds could still resume the search).
  - `llama-q8-gemv-gemmini-n16` is marked **TERMINAL**, not the SOLVED its
    own `kernels.py` notes claim — overridden because
    `out/loop/FINAL_REPORT.md`'s round6 entry documents the loop as
    explicitly abandoning further search on it (four regressions, zero new
    bests) rather than declaring it at a proven floor. This is the one
    deliberate judgment call in this file; see `experiments.csv`'s round6
    rows for the underlying evidence.
  - All other kernels' verdicts (SOLVED for n1-gemv/lmhead/q8-gemm;
    TERMINAL for attn-scores/attn-pv/silu-mul/softmax) follow directly
    from `kernels.py`'s own "STATUS OF THIS KERNEL" language, cross-checked
    against `FINAL_REPORT.md` §2.

- **`probes.md`** — every `PROBE`-prefixed diagnostic line found under
  `out/loop/*/simlog_*.txt` (149 raw lines, 20 (run_id, iter) file pairs,
  6 run_ids, round5–round9), grouped into evidence rows across five categories:
  (A) DMA rate — warm/cold B/cycle, rows-per-mvin and in-flight-request
  sweeps (the centerpiece: run `20260912-191830-e713`'s full sweep); (B)
  phase-split (mvin-stream / compute / mvout / fence, and the fused
  kernel's Gemmini-stream vs Saturn-attention gap timeline); (C) host-issue
  cost per RoCC command; (D) other (L2 flush cost, etc.); (E) two
  uncategorized `ref_score_min/max_micro` lines nobody's agent note
  explains — left unexplained rather than guessed at. A handful of
  pre-round5 numbers exist only as prose in `agent_NN.txt` (no simlog
  capture existed yet); these are explicitly labeled "agent_NN.txt prose
  (pre-harness-fix)" rather than mixed in with the printf-captured rows.
  - **Reproduce:** `grep -rn "PROBE" out/loop/*/simlog_*.txt` (command is
    given again at the top of the file itself).

- **`timeline.md`** — one section per round (smoke, round1, round2,
  round2b, round3–round9), each with dates, kernel(s), run_id(s),
  iteration counts, cost, wall-clock, findings, and any incident, plus a
  "Cross-cutting incidents" table (429 session-limit interruptions,
  watchdog bugs, the disk-full migration to `/share1`, the 2026-09-09
  roofline correction, the 2026-09-11 printf/simlog harness fix, the
  2026-09-12 warm-L2 audit correction).
  - **Discrepancies found vs. the "two 429s, two watchdog bugs" framing
    given in the task brief** (documented in the file's own
    Cross-cutting-incidents section, not silently reconciled): round7 also
    hit a real (weekly-cap) 429, which may or may not count as a "third"
    depending on how you classify session-limit vs. weekly-cap throttling;
    and round8's watchdog had an undocumented third bug (its 429-detector
    regex false-matched a PID inside an unrelated Ray disk-space warning,
    compounded by a stale hardcoded `PLANNED=8` copied from round7's
    script) — caused no data loss (round8 had already finished 4/4 by
    then) but is a distinct bug from the two named in the brief.
  - Rollup through round9 (final): 62 runs, 243 iterations, $177.7596,
    ~70h19m wall-clock. Overall best: `llama-layer-fused-n1` at 179,033
    cycles (round9 iter4), −13.2% vs. baseline 206,304, 1.273x roofline.
  - Sources: `out/loop/rounds.json`, `out/loop/FINAL_REPORT.md`,
    `out/loop/ledger.md`/`ledger.json`, `out/loop/round7-partial.md`,
    `out/loop/round7-prep.md`, `out/loop/round{7,8,9}-watchdog.log`,
    `out/loop/round{8,9}-runs.txt`, `out/loop/round9-launch.sh`.

- **`methodology.md`** — the CHIA loop's experimental method: §1 iteration
  flow (build → simulate → self-check → score → LLM feedback →
  next-iteration; `--seed best` restart mechanism; iter -1/0 convention;
  `iters.status` vocabulary), §2 correctness self-check, §3 timing
  methodology (`mcycle`-delta measurement region; the 1 GHz cost-model
  clock vs. the RTL's actual 500 MHz elaboration, cited to
  `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`),
  §4 hardware configuration (`GENV256D128GemminiShuttleConfig`'s DIM/
  scratchpad/accumulator/DMA/queue parameters and the mbus 8 B/cycle
  bus-width chain, both cited to file:line in `loop/kernels.py` and
  `docker/add_coexist_config.py`), §5 cost accounting
  (`chia/models/claude.py`'s `total_cost_usd`), §6 threats to validity:
  (a) benchmark-harness warm-L2 residue (~200 KiB tail left in L2 by the
  harness's own setup code, inflating measured B/cycle), (b) `mm_magic_t`
  zero-latency DRAM model — the simulator never runs with `+dramsim`, so
  there is no DRAMSim2 timing model in play at all (cited to
  `chia/chipyard/verilator_run_node.py:513-514`), (c) the 1 GHz vs. 500 MHz
  clock mismatch, (d) `llama-layer-fused-n1` fuses only four sub-ops of one
  decode layer, not the full layer/model, (e) linear-extrapolation limits
  for the untested N=4/32/64 batch points (only N=1 and N=16 GEMV tiles
  were ever actually measured), (f) the fused kernel's 13.2% overlap
  reduction cannot be extrapolated uniformly to the whole decode step — its
  own Saturn device share (13.8%) is an order of magnitude larger than
  decode's actual Saturn share (0.7-2.2%); the round9/round8 projection
  docs' uniform-extrapolation figures (5.092/5.076 tok/s, +15.2%/+14.9%)
  were retracted for this reason and replaced with `--overlap`-measured
  figures of 0.3-1.4%.
  - Three items the writing agent could not verify in this checkout and
    marked "unclear from source" rather than asserting: the exact
    rocket-chip `MemoryBusParams(beatBytes = 8)` declaration (no
    rocket-chip/chipyard source tree exists in this checkout — the claim
    rests on in-repo comments plus the ~7.9 B/cycle measured ceiling as
    indirect corroboration); the code path that actually emits
    `ST_RETRY_EXHAUSTED` (declared in `loop/db.py`, no emitting call site
    found in `loop/loop.py`/`loop/nodes.py`); and what public model the
    internal codename `claude-fable-5-1` (`loop/constants.py:68`) maps to.

## Regenerating everything

All four artifacts were produced independently (different data sources,
no shared intermediate state besides the read-only DB/rounds.json/
kernels.py) and can be regenerated independently:

```bash
# experiments.csv + kernels_final.csv
~/.conda/envs/chia_env/bin/python3 out/paper/scratch/build_csvs.py

# probes.md — re-run the grep in the file's own header, then re-read the
# paired agent_NN.txt files for context (manual/LLM-assisted step, not a
# single script)
grep -rn "PROBE" out/loop/*/simlog_*.txt

# timeline.md and methodology.md are hand-written syntheses of
# out/loop/rounds.json, out/loop/*.md/*.json, loop/*.py, docker/*.py, and
# out/coexist/*.log — regenerate by re-reading those sources fresh, they
# are not script outputs.
```

## Known caveats (do not silently paper over these in the paper text)

1. **Round9 is now final.** Round9 finished 2026-09-17 (run
   `20260916-195958-36b4`, 4/4 iterations, best 179,033 cycles, $6.4724).
   `out/loop/rounds.json`, `ledger.md`/`ledger.json`, `iterations.md`/
   `iterations.json`, and every figure in `experiments.csv`, `probes.md`,
   `timeline.md`, and `out/llama-profile/projection_round9.md` reflect these
   final numbers — no round9 figure anywhere in this appendix is a live/
   in-flight snapshot any longer.
2. **One judgment call, not a measurement**: `kernels_final.csv`'s
   `llama-q8-gemv-gemmini-n16` verdict (TERMINAL, overriding its own
   `kernels.py` notes' "SOLVED" language) — see that file's entry above.
3. **Two uninterpreted probe lines** (`ref_score_min/max_micro` in
   `probes.md` category E) — flagged as unexplained rather than guessed.
4. **"Two 429s, two watchdog bugs" is an undercount** — see `timeline.md`'s
   Cross-cutting-incidents section for the third watchdog bug (round8) and
   the ambiguous round7 weekly-cap 429.
