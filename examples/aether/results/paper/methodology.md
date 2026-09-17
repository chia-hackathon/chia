# AETHER Inner Loop — Reproducibility Appendix: Experimental Methodology

This document describes, with file:line citations into the repository at
`/share1/saves/max410011/hackathon/aether`, how the CHIA-based inner
optimization loop measures, scores, and reports on RISC-V kernels for a
Saturn (RVV 1.0 vector core) + Gemmini (systolic-array RoCC accelerator)
SoC running a Verilator-based Chipyard/rocket-chip RTL simulation, as used
to optimize Llama-3.2-1B decode/prefill kernels. Where a claim could not be
verified directly against source in this checkout, it is marked "unclear
from source" rather than asserted.

---

## 1. CHIA inner loop process

**Unit of work.** One invocation of `run_inner()` (`loop/loop.py:336-589`)
optimizes exactly one kernel (one entry from the `kernels.py` registry) on
one already-built simulator artifact. The hardware configuration is fixed
for the whole inner loop; only the software kernel source changes between
iterations (`loop/loop.py:1-19` docstring diagram).

**Build → simulate → self-check → score → feedback loop.** Each iteration
`i` (`loop/loop.py:532-581`):
1. The agent edits the one file it owns (`kernel_bash` BashTool scoped to
   `AGENT_WORK_DIR`, `loop/loop.py:518-524`) and the driver reads the kernel
   bytes back off that workspace (`loop/loop.py:547`, dispatched to
   `nodes.read_kernel`, `loop/nodes.py:239-247`).
2. `_attempt()` (`loop/loop.py:215-223`) compiles the kernel against sealed
   collateral (`nodes.build_kernel`, `loop/nodes.py:129-148`) and, if the
   build succeeded, runs the ELF on the Verilator simulator
   (`nodes.run_kernel`, `loop/nodes.py:154-166`).
3. `nodes.measure()` (`loop/nodes.py:185-207`) classifies the attempt into a
   `Measurement(ok, kind, cycles, instret, detail)`: `build_failure` if the
   cross-compile failed, `sim_failure` if the simulator did not run or
   printed no cycle count, `incorrect` if the simulator run itself returned
   non-zero (the benchmark's internal self-check calls `exit(1)` on
   mismatch, so a failing self-check surfaces as `run.success == False`,
   `loop/nodes.py:200-202`), otherwise `ok` with a `cycles` value pulled from
   the log via a per-kernel regex (`k.cycle_re`, default
   `r"mcycle\s*=\s*(\d+)"`, `loop/kernels.py:222`).
4. The measurement is written to `history.json` (`loop/loop.py:418-420`),
   the kernel source is snapshotted to `out/loop/<run_id>/kernel_NN.*`
   (`loop/loop.py:548-549`), the simulator log is snapshotted to
   `simlog_NN.txt` (`loop/loop.py:552`), and the iteration is recorded in
   SQLite via `db.record_iter()` (`loop/loop.py:556-561`, schema at
   `loop/db.py:133-160` approx., diff-to-parent tracked via
   `db.make_diff`/`parent_id`).
5. If the attempt passed and beat the current best, it becomes the new
   `best`/`best_kernel` and is snapshotted to `kernel_best.*`
   (`loop/loop.py:564-571`).
6. Feedback for the next turn is built by `agent.format_feedback()`
   (`loop/llm.py:229-298`) and handed to the agent's next turn
   (`agent.next_turn`, `loop/loop.py:539`).

**Iteration budget.** The default iteration count after the baseline is
`MAX_ITERS = 10` (`loop/constants.py:77`), overridable per run with
`--iters` (`loop/loop.py:606-607`). A dollar budget can also be supplied
with `--budget-usd`; the loop checks accumulated `cost_usd` after recording
each iteration and stops early once the budget is met or exceeded
(`loop/loop.py:573-577`). There is no automatic early-stopping on
convergence (no new best for N iterations) — `out/loop/FINAL_REPORT.md:179`
explicitly lists the absence of automatic convergence detection as a
lesson-learned from round 3, where one kernel (`llama-q8-gemv-gemmini-n1`)
absorbed \$48.99 of spend without moving after round 2b.

**Baseline (iteration 0).** Before any agent turn, the pristine kernel
(`nodes.pristine_kernel_for()`) is measured once via `measure_baseline()`
(`loop/loop.py:226-252`), which is itself cached by `(simulator, kernel
bytes)` so a baseline already known is never re-simulated
(`loop/loop.py:235-247`, using `cache.baseline_key`/`cache.load_baseline`).
This baseline is recorded as iteration `0` (`loop/loop.py:440-448`) and is
always the speedup denominator (`InnerResult.speedup`,
`loop/loop.py:142-146`), regardless of whether the run is seeded. If the
baseline itself fails to pass, the run is aborted with
`status="baseline_failed"` and no LLM is invoked (`loop/loop.py:426-434`).

**Seed mechanism (`--seed`).** `--seed best` (alias `--seed-run RUN_ID` for
a specific run) starts the agent's editing session from a previously
optimized kernel rather than from the pristine source, while keeping
iteration 0 as the pristine baseline so speedups stay comparable across
seeded and unseeded runs (`loop/loop.py:259-267`, `465-467` comment).
`_seed_row()` (`loop/loop.py:270-287`) looks up, for `spec == "best"`, the
best-fitness passing iteration (`iter > 0`, `fitness IS NOT NULL`) across
all runs for the same `(kernel, config)`; for a specific `RUN_ID` it is
restricted to that run's own best iteration. `resolve_seed()`
(`loop/loop.py:290-329`) turns that DB row into kernel bytes, preferring
`iters.kernel_path` and falling back to `<out_dir>/kernel_best.*` on disk if
the DB row's path is missing, so a run whose rows never made it into the
store can still be seeded from. The seed's own measurement is recorded at a
reserved negative iteration number, `SEED_ITER = -1`
(`loop/loop.py:263-267`, written at `loop/loop.py:486-504`), distinct from
and never displacing the iteration-0 baseline row. If the seed kernel fails
its own self-check, the loop falls back silently to the pristine kernel as
the starting point (`loop/loop.py:489-492`). `out/loop/FINAL_REPORT.md:174`
documents `--seed best` being used operationally to resume six runs after a
Claude session-limit (HTTP 429) interruption in round 2 (round 2b) and two
runs interrupted mid-round-3.

**Retry / status vocabulary.** `loop/db.py:76-93` defines the `iters.status`
vocabulary: `ST_OK = "ok"`, `ST_BUILD_FAILED = "build_failed"`,
`ST_SELFCHECK_FAILED = "selfcheck_failed"`, `ST_SIM_FAILED = "sim_failed"`,
`ST_RETRY_EXHAUSTED = "retry_exhausted"`, with the first four disqualifying
an iteration from ever being sampled as a parent (`DEAD_STATUSES`,
`loop/db.py:83-84`). `status_from_kind()` (`loop/db.py:96-102`) maps a
`nodes.Measurement.kind` string onto this vocabulary via the table at
`loop/db.py:87-93` (`"ok"/"baseline" -> ST_OK`, `"build_failure" ->
ST_BUILD_FAILED`, `"incorrect" -> ST_SELFCHECK_FAILED`, `"sim_failure" ->
ST_SIM_FAILED`, `"retry_exhausted" -> ST_RETRY_EXHAUSTED`, with any other
kind falling through to `ST_OK` if `passed` else `ST_SIM_FAILED`). **Note:**
`nodes.Measurement.kind` as actually produced by `nodes.measure()`
(`loop/nodes.py:185-207`) only ever emits `"build_failure"`, `"sim_failure"`,
`"incorrect"`, or `"ok"` — the `"retry_exhausted"` kind and the
corresponding `ST_RETRY_EXHAUSTED` status are declared in the vocabulary but
no per-iteration retry-then-give-up code path that emits them was found in
`loop/loop.py` or `loop/nodes.py` in this checkout; this is marked
**unclear from source** — the status may be reserved for a caller (e.g. an
outer loop) not exercised by `run_inner()` itself, or for a manual/DB-side
annotation.

---

## 2. Correctness self-check mechanism

Correctness is enforced entirely *inside* the benchmark binary, not by the
loop: "the self-check inside the benchmark decides correctness; we only
read the numbers it prints" (`loop/nodes.py:186-187`). Concretely, for the
`llama-layer-fused-n1` Gemmini kernel (representative of the pattern used
across the `llama-*` entries), the sealed harness body
(`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h`)
generates every input from a fixed PRNG seed (`lf_rng`, seed
`0x243F6A8885A308D3`, lines 65-84 of that file), computes a scalar/golden
reference for each of the fused sub-operations, compares the kernel's
output against that reference to a stated tolerance, and calls
`exit(1)`/prints `MISMATCH` on any failure — this is documented explicitly
in the kernel's registry entry `check_desc`
(`loop/kernels.py:4252-4262`): GEMV compared exactly on 8 sampled outputs,
attention scores compared to 1e-4 relative on all 512 outputs, softmax
compared to 5e-3 relative plus a `sum(probs)` gate within 1e-3 of 1.0, and
`probs@V` compared as exact int32 (`pvacc`) / 1e-4 relative (`pvout`).
On the driver side, `nodes.run_kernel()`'s `RunResult.success` is what
actually reflects this: a non-zero simulator/program return code (from the
harness's `exit(1)`) makes `run.success == False`, which `nodes.measure()`
classifies as `kind="incorrect"` regardless of whether a cycle count could
still be parsed from the log (`loop/nodes.py:200-202`). The agent is told
explicitly in `loop/llm.py:222-226` ("Your kernel compiled but **failed the
self-check** — the result is wrong... a wrong answer scores nothing") and
is never shown or given access to the reference/golden-model code, since
`main_src` (e.g. `bareMetalC/llama_layer_fused_n1.c`) and the harness body
are sealed collateral outside the one file the agent may edit
(`loop/kernels.py:4227-4228`, `4252-4254`).

---

## 3. Timing methodology

**What region is measured.** Timing is a single-invocation wall-cycle count
taken with `read_cycles()` (an inline `rdcycle` instruction,
`repos/gemmini/software/gemmini-rocc-tests/include/gemmini_testutils.h:273-275`)
bracketing exactly one call to the kernel under optimization, e.g.
```
uint64_t start = read_cycles();
... call kernel ...
uint64_t end = read_cycles();
printf("Cycles taken: %lu\n", (unsigned long)(end - start));
```
(`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h:339-347`;
the same pattern appears in `llama_q8_gemv_gemmini_body.h:138-143` and
`llama_q8_gemv_gemmini_lmhead_body.h:151-156`). For the Saturn (non-Gemmini)
benchmarks the equivalent is the `stats()` macro in
`repos/saturn/benchmarks/common/util.h:95-102`, which reads `mcycle`/
`minstret` via `read_csr` before and after a code block. The loop's
`Kernel.cycle_re` per registry entry (default
`r"mcycle\s*=\s*(\d+)"`, `loop/kernels.py:222`, overridden to
`r"Cycles taken:\s*(\d+)"` for `llama-layer-fused-n1`,
`loop/kernels.py:4231`) is a regex applied against the simulator's captured
stdout to pull this one printed number out as the scored objective
(`nodes._extract`, `loop/nodes.py:168-178`).

**What "cycles" means.** It is the cycle count of *one kernel invocation*
under a cold or (for most `llama-*` Gemmini entries prior to
`llama-layer-fused-n1`) partially-warm cache/DRAM state set up by the
surrounding sealed harness — see §6(a) for the warm-L2 caveat on the
`llama-q8-gemv-gemmini-n1` "132,424" figure specifically. It is *not* an
end-to-end decode-token latency; per-token throughput numbers in
`out/loop/FINAL_REPORT.md` are a separate cost-model projection
(`loop/llama_project.py`, `loop/llama_batch_project.py`) built by summing
per-kernel roofline/measured cycles across the fixed operator graph of one
decoder layer, not measured directly end to end.

**Clock frequency assumption vs. RTL elaboration.** All roofline arithmetic
and the loop's own cost model assume a uniform **1.00 GHz** clock throughout
(`out/loop/FINAL_REPORT.md:14`, "時脈一律 1.00 GHz"). The actual RTL
elaboration of the simulator config, however, reports every relevant clock
domain at **500 MHz**: `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`
reads
```
Clock sbus_0: using diplomatically specified frequency of 500.0.
Clock pbus_0: using diplomatically specified frequency of 500.0.
Clock fbus_0: using diplomatically specified frequency of 500.0.
Clock mbus_0: using diplomatically specified frequency of 500.0.
Clock cbus_0: using diplomatically specified frequency of 500.0.
```
i.e. sbus/pbus/fbus/mbus/cbus all elaborate at 500 MHz, not 1 GHz. Verilator
itself is cycle-accurate and clock-frequency-agnostic (it counts RTL clock
edges, not wall time), so the reported *cycle counts* from the simulator are
unaffected by this discrepancy — but converting those cycle counts to a
tok/s (or any wall-clock) figure at "1 GHz" is a factor-of-2-optimistic
assumption relative to what this SoC's own elaborated clock tree reports.
`out/loop/FINAL_REPORT.md:258-261` and its companion table at
`out/loop/FINAL_REPORT.md:269-276` (also `out/llama-profile/projection_final_v2.md`
§2) already carry an explicit 500 MHz column for this reason; see §5/§6(c)
below.

---

## 4. Hardware configuration

**Config identity.** The loop's default simulator config is
`GENV256D128GemminiShuttleConfig` (`loop/constants.py:29`, package
`chipyard`, `loop/constants.py:30`), built once per inner-loop session by
`nodes.build_simulator()` via `ChiselBuildNode(... target=BuildTarget.VERILATOR ...)`
(`loop/nodes.py:95-111`).

**Scala config chain (source of the class).** The config is assembled in
`docker/add_coexist_config.py:41-47`:
```
class GENV256D128GemminiShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, saturn.common.VectorParams.genParams) ++
  new gemmini.DefaultGemminiConfig ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)
```
i.e. Saturn RVV at VLEN=256/DLEN=128 stacked with Gemmini's
`DefaultGemminiConfig` on one Shuttle tile, with `WithSystemBusWidth(128)`
and `WithShuttleTileBeatBytes(16)` overriding only the sbus width and tile
beat bytes (`docker/add_coexist_config.py:15-18` comment). `loop/hwconfig.py:110-113`
documents this same class as the loop's "already-built AETHER baseline."

**mbus (DRAM-facing bus) width.** `loop/kernels.py:106-112` and
`loop/kernels.py:1161-1174` both assert that DRAM sits behind the rocket-chip
`MemoryBusKey => MemoryBusParams(beatBytes = 8)` (i.e. 8 B/cycle), and that
neither `docker/add_coexist_config.py`'s config chain nor chipyard's
`AbstractConfig` overrides that beat width — only the sbus width (128 bit)
and the Shuttle tile beat bytes (16 B) are overridden. **This specific claim
about `MemoryBusParams(beatBytes = 8)` could not be independently verified
against a local rocket-chip/chipyard source checkout**: this repository's
`repos/` tree contains `saturn`, `shuttle`, `gemmini`, and `llama.cpp`
checkouts but no `rocket-chip` or `chipyard` source tree (chipyard lives on
a separate container image at `CHIPYARD_PATH = "/home/ray/chipyard"`,
`loop/constants.py:31`, not present in this filesystem) — **unclear from
source; the claim rests on the in-repo comments at `loop/kernels.py:106-112`,
`loop/kernels.py:1161-1174`, and `docker/add_coexist_config.py:15-18`, and
is corroborated only indirectly** by the measured cold-DRAM throughput
ceiling of ~7.9-7.94 B/cycle (`loop/kernels.py:118-121`, and see §6(a))
being close to, but never exceeding, 8 B/cycle. This bandwidth correction
(from an earlier, incorrect assumption of 16 B/cycle to 8 B/cycle) is
recorded as a mid-project bug fix on 2026-09-09
(`out/loop/FINAL_REPORT.md:77-83`; also documented in `out/loop/README.md`
"Roofline correction (2026-09-09)", not independently re-verified in this
review).

**GENV256D128GemminiShuttleConfig elaborated parameters.** Two
near-duplicate summaries appear in `loop/kernels.py`, both attributed to the
elaborated hardware (the second citing the Chisel source files directly):

- `loop/kernels.py:951-955`: "Elaborated `GENV256D128GemminiShuttleConfig`
  (= `gemmini.DefaultGemminiConfig` + `WithSystemBusWidth(128)` +
  `WithShuttleTileBeatBytes(16)`): DIM 16, sp 256 KiB / 4 banks, acc 64 KiB /
  2 banks, dma_maxbytes 64, max_in_flight_mem_reqs 16, ld/st/ex queue length
  8/2/8, reservation-station entries ld 8 / st 4 / ex 16."
- `loop/kernels.py:1161-1174` ("ground truth, from
  `gemmini/src/main/scala/gemmini/Configs.scala`, `DMA.scala`,
  `LoadController.scala`"): DIM 16x16 int8 (256 MAC/cycle peak), scratchpad
  256 KB / 4 banks, accumulator 64 KB (`ACC_ROWS = 1024`), `dma_buswidth`
  128 bit = 16 B/cycle (the sbus/DMA-port width, distinct from the 8 B/cycle
  mbus), `dma_maxbytes` 64 B (4 beats), `max_in_flight_mem_reqs` 16, TLB 4
  entries, outstanding load commands `nCmds = max_in_flight_mem_reqs / DIM +
  1 = 2`.

Supporting RTL detail on the request-shape constraints (`nCmds = 2`,
`XactTracker` pool of 16, per-row serialization inside `LoadController`) is
cited to specific Gemmini Scala source lines at `loop/kernels.py:906-923`
(`LoadController.scala:30-31,74,102-106,164-170`; `DMA.scala:135`;
`XactTracker.scala:56-71`; `Scratchpad.scala:208`), which line up with the
checked-out `repos/gemmini/src/main/scala/gemmini/` tree present in this
repository (not individually re-verified line-by-line in this review; the
citations are to files that do exist at `repos/gemmini/src/main/scala/gemmini/`).

**Roofline derivation.** The method is written out in full at the top of
`loop/kernels.py:20-135` ("ROOFLINE METHOD"), summarized here with its
formula citations:

- General (Saturn/vector) formula:
  `roofline_cycles = max(compute floor, memory floor)` (`loop/kernels.py:103`).
  - Compute floor: derived per-kernel from an arithmetic-intensity table
    (`loop/kernels.py:31-52`) — e.g. int8×int8 widening MAC into int16 at
    8 MAC/cycle is the key rate (`loop/kernels.py:42`, "The int8 MAC rate is
    **8/cycle, not 4/cycle**"), so an int8 dot product's compute floor is
    `MACs / 8` (`loop/kernels.py:55`).
  - Memory floor: `memory_floor = max(bytes_read, bytes_written) / 16`
    (`loop/kernels.py:94`), reflecting independent 16 B/cycle load, store,
    and arithmetic DLEN-wide pipes on Saturn (`loop/kernels.py:92-93`).
- Gemmini-specific formula (`loop/kernels.py:113-121`):
  `compute floor = MACs / 256` (DIM×DIM = 16×16 = 256 MAC/cycle peak,
  `loop/kernels.py:117`), `memory floor = max(bytes in, bytes out) / 8`
  (8, not the sbus's 16, because DRAM traffic crosses the 8 B/cycle mbus,
  `loop/kernels.py:118`).
- Worked numeric examples appear throughout the file, e.g.
  `loop/kernels.py:752,758` (`compute floor = 1,048,576/8 = 131,072`;
  `memory floor = max(65,664,128) = 65,664`) and
  `loop/kernels.py:1322,1332` (`compute floor = 33,554,432/256 = 131,072`;
  `memory floor = max(81,920,8,192) = 81,920`).
- A closing sanity rule (`loop/kernels.py:123-135`) forbids any `llama-*`
  registry entry from carrying a `roofline_cycles` above its own best
  measured cycle count.

---

## 5. Cost accounting

**Mechanism.** Cost/token accounting flows from the Claude Code CLI's own
end-of-turn `result` event. `chia/models/claude.py:1078-1084` (in the
separate `chia` package at `/share1/saves/max410011/hackathon/chia`) reads
`event["total_cost_usd"]` and the `usage` sub-dict
(`input_tokens`/`output_tokens`/`cache_creation_input_tokens`/
`cache_read_input_tokens`/`num_turns`) directly off that CLI event and pushes
them into the profiler as call metadata (`loop/costs.py:3-7` docstring,
citing `chia/models/claude.py ~L1065-1098`). **This means `cost_usd` is not
computed by the loop from a static per-model price table in this
repository** — it is whatever the underlying `claude` CLI process itself
reports as the turn's cost; no separate Anthropic pricing table was found
in `loop/` or in the referenced `chia/models/claude.py` region. This is
consistent with the loop's own comment that `cost_usd`/tokens are "read off
the chia profiler" (`loop/costs.py:1`).

`loop/costs.py`'s `CostMeter` (`loop/costs.py:33-98`) is a cursor over the
profiler's event stream: `poll()` (`loop/costs.py:72-93`) sums every
cost-bearing event seen since the previous poll (waiting up to 20s for a
delayed event) and returns per-iteration usage; `CostMeter.add()`
(`loop/costs.py:96-98`) accumulates a running total across iterations. The
per-run total is written to `out/loop/<run_id>/cost.json`
(`loop/loop.py:586`) and to the `runs`/`iters` tables' `cost_usd`,
`in_tokens`, `out_tokens` columns (`loop/db.py:122-135` region;
`db.record_iter`, `loop/db.py:405-436`).

**LLM model.** A single model string, `LLM_MODEL = "claude-fable-5-1"`
(`loop/constants.py:68`), is used for the entire loop (`agent.make_llm()`,
`loop/llm.py:126-135`, constructs one `ClaudeCodeLLM(model=LLM_MODEL, ...,
resume_session=True, ...)` that is reused — via `--resume` on the same
underlying CLI session — across all iterations of one `run_inner()` call,
`loop/llm.py:1-6` docstring). This model string is recorded once per run
in the `runs.llm_model` column (`db.start_run`, `loop/db.py:396-404`) and
is stamped into each `InnerResult.summary()` (`loop/loop.py:163`). There is
no per-iteration model override discovered anywhere in `loop/` — the
column exists at the granularity of a run, not an iteration. **The string
`"claude-fable-5-1"` is an internal/codename model identifier; nothing in
this checkout (including `chia/models/claude.py`) resolves it to a specific
publicly-documented Anthropic model id, so which released model this
corresponds to is unclear from source.** `LLM_EXTRA_CLI_ARGS = ["--effort",
"high"]` (`loop/constants.py:69`) is passed to every turn. The kernel
registry's own roofline notes independently label the model in use as
"Fable 5.1" (`loop/kernels.py:126`, "Round-1 sanity constraint (2026-09-07,
model Fable 5.1)"), consistent with the constants.py string but not further
disambiguating it.

---

## 6. Threats to validity

### (a) Benchmark harness warm-L2 residue

The headline `llama-q8-gemv-gemmini-n1` "best" figure of **132,424 cycles**
(`out/loop/FINAL_REPORT.md:44`) is not a purely cold-cache measurement.
`out/loop/FINAL_REPORT.md:101-104` states the measured 7.92 B/cycle "混了約
200 KiB 的 harness 暖 L2 尾巴" (mixes in roughly 200 KiB of harness-warmed
L2 tail), and that a genuinely cold, ascending-order measurement instead
gives **149,757 cyc = 7.00 B/cycle**. §7's audit correction
(`out/loop/FINAL_REPORT.md:249-251`) restates this as item (1): "n1 GEMV's
'warm' measurement contains ~200 KiB of harness warm-L2 tail... cold
ascending measurement is 149,757 cyc = 7.00 B/cycle." Mechanistically, this
happens because the harness fills the weight buffer `B` in DRAM
*immediately before* calling the timed kernel (`llama_layer_fused_body.h:96-100`
describes exactly this pattern for the later, harness-fixed benchmark: "Every
earlier Gemmini entry inherited a warm L2 tail: the harness fills B in DRAM
immediately before calling the kernel, so ~200-260 KiB of the 1 MiB is still
resident and descending-K order harvests it" —
`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h:96-100`).
Because the tail of `B` is still resident in L2 from the fill, a kernel that
streams `B` in descending-K (or otherwise tail-first) order gets a partial
cache hit on that tail, inflating the *achieved* B/cycle figure relative to
a fully-cold run. This matters directly for interpreting "achieved
B/cycle" numbers throughout §2/§3 of `out/loop/FINAL_REPORT.md`: the
99%-of-mbus-peak framing of `llama-q8-gemv-gemmini-n1` (7.92/8 B/cycle,
`out/loop/FINAL_REPORT.md:44,101`) is a warm-state number; the cold-state
ceiling is 7.00 B/cycle (87.5% of nominal peak), a materially different
headroom conclusion. The `llama-layer-fused-n1` kernel introduced later
(§6(d) below) was explicitly built to close this gap by streaming 2 MiB of
unrelated data through the L2 with Gemmini's own DMA immediately before the
timed region, forcing a genuinely cold start
(`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h:96-104`).

### (b) `mm_magic_t` zero-latency DRAM model, not DRAMSim2

The simulator invocation never enables a timing-accurate DRAM model. In
`chia`'s `VerilatorRunNode.run()`, the `+dramsim`/`+dramsim_ini_dir=`
plusargs are appended to the simulator's argv **only if**
`self._dramsim_ini_dir` is set (`chia/chipyard/verilator_run_node.py:513-514`,
confirmed by direct read: `if self._dramsim_ini_dir: argv +=
["+dramsim", f"+dramsim_ini_dir={self._dramsim_ini_dir}"]`). The loop's own
`nodes.run_kernel()` (`loop/nodes.py:154-166`) constructs its
`VerilatorRunNode.run.chia_remote(...)` call without ever supplying a
`dramsim_ini_dir` argument, so that branch is never taken and `+dramsim` is
never passed. Consequently every simulated run uses the simulator's default,
untimed DRAM abstraction, `mm_magic_t` — a bandwidth-limited but
zero-additional-latency memory model, not a cycle/row/bank-accurate DRAMSim2
timing model. `out/loop/FINAL_REPORT.md:90-99` documents this as a
mid-project correction: an earlier working hypothesis (agent transcript
`out/loop/20260909-200541-dae1/agent_01.txt` and `.../agent_07.txt`)
attributed the ~6.6 B/cycle cold-weight-stream ceiling to "DRAMSim2 cold-flow
rate," which was subsequently falsified by checking the actual invocation
(`out/loop/20260909-052323-9d90/agent_01.txt`); the real bottleneck was
identified as L2 MSHR occupancy and bank conflicts under the stride-2048
weight-streaming access pattern (`out/loop/20260909-200541-dae1/agent_04.txt`),
not a modeled DRAM timing effect. **Implication for validity:** the cost
model's memory-bound roofline numbers assume a bandwidth-limited-but-
latency-free DRAM, which is optimistic relative to any timing-accurate DRAM
model a reader might otherwise assume was in use; the true bottleneck
mechanism (L2 MSHR/bank contention) is itself a property of this specific
RTL microarchitecture and cache configuration, not a fundamental physical
DRAM limit, so its 6.6-7.9 B/cycle ceilings should not be read as DRAM
device bandwidth figures.

### (c) 1 GHz clock assumption vs. 500 MHz RTL elaboration

As detailed in §3 above, all reported tok/s and cycles-to-time conversions
in `out/loop/FINAL_REPORT.md` use a uniform 1.00 GHz assumption
(`out/loop/FINAL_REPORT.md:14`), while the actual elaborated design reports
every relevant clock (sbus/pbus/fbus/mbus/cbus) at 500 MHz
(`out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`,
verified present and reading exactly "using diplomatically specified
frequency of 500.0" for all five buses in this review). This is a **2x**
optimistic bias in any *wall-clock-derived* throughput figure (tok/s,
seconds); it has no effect on the raw *cycle counts* reported by Verilator,
since Verilator counts RTL clock edges rather than modeling absolute time.
The audit correction in `out/loop/FINAL_REPORT.md:258-261` and its
before/after table at `out/loop/FINAL_REPORT.md:269-276` (also
`out/llama-profile/projection_final_v2.md` §2, file confirmed present at
`out/llama-profile/projection_final_v2.md`) list both a 1 GHz and 500 MHz
column going forward, e.g. warm-state N=1 decode: 6.260 tok/s @1GHz vs.
3.130 tok/s @500MHz; cold-state upper bound: 5.345 vs 2.673 tok/s; cold-state
lower bound: 5.014 vs 2.507 tok/s.

### (d) `llama-layer-fused-n1` covers only part of one decode layer

The `llama-layer-fused-n1` kernel (`loop/kernels.py:4224-4262`) fuses
exactly four sub-operations of a single Llama-3.2-1B decoder layer at
decode batch N=1: one `1×2048×512` int8 GEMV on Gemmini (1 MiB of weights,
matching the standalone `llama-q8-gemv-gemmini-n1` shape) together with, on
Saturn, one attention head's `QK^T` scores, softmax, and `probs@V`
(matching the standalone `llama-attn-scores-int8` / `llama-softmax` /
`llama-attn-pv-int8` shapes) — see the shape comment at
`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h:70-80`
and the registry `objective` string at `loop/kernels.py:4238-4243` ("one
Llama-3.2-1B decoder-layer slice at decode N=1 ... TOGETHER with one
attention head's QK^T, softmax and probs@V"). Explicitly **not** covered by
this fused kernel, and measured only as separate standalone kernels
elsewhere in the registry (`loop/kernels.py`'s `KERNELS` dict,
`loop/kernels.py:4272-4283`, listing `LLAMA_RMSNORM`, `LLAMA_ROPE`,
`LLAMA_ADD`, `LLAMA_SILU_MUL` as distinct entries): RMSNorm, RoPE, the
residual add, the SiLU-gated MLP (up/gate/down projections and the SiLU
nonlinearity), and — within attention itself — every head beyond the single
head modeled here (a real decode layer for this model has multiple
attention heads; this kernel's Saturn phase computes exactly one head's
QK^T/softmax/PV, `LF_D = 64` head dim,
`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h:83`).
It also does not include the output projection, KV-cache write-back, or any
cross-layer/model-level work (embedding, final RMSNorm, LM head). Treating
this kernel's measured cycle count as representative of "one decode layer"
end to end would therefore substantially undercount total layer cost; it
should be read strictly as "the DMA-bound GEMV phase plus one attention
head's compute phase, measured together to test whether they can be
software-pipelined," per its own stated purpose
(`repos/gemmini/software/gemmini-rocc-tests/include/llama_layer_fused_body.h:20-29`).

### (e) Linear extrapolation limits for untested batch sizes (N=4/32/64)

The batch-decode projection table in `out/loop/FINAL_REPORT.md:202-208`
reports GEMV-tile cycle figures for batch sizes N=1, 4, 16, 32, and 64, but
only **N=1** (132,424 cyc) and **N=16** (142,458 cyc) correspond to kernels
actually built, run, and self-check-passed on the simulator (`
out/loop/FINAL_REPORT.md:44-45`, round 5's `llama-q8-gemv-gemmini-n16`
result). The N=4, N=32, and N=64 rows are explicitly flagged with a `*` in
that table and a footnote: "GEMV tile cycles 為 N=1/N=16 兩點的線性外推（斜率
668.9 cyc/column），估計值" — i.e. GEMV-tile cycles for N=4/32/64 are a
**linear extrapolation** from the two measured points (N=1, N=16) using a
slope of 668.9 cycles per additional batch column, and are explicitly
labeled estimates, not measurements (`out/loop/FINAL_REPORT.md:210`; the
same 668.9 cyc/column slope, derived from exactly the two measured points
`(1, 132424)` and `(16, 142458)`, is independently stated in
`out/llama-profile/projection_final.md:45`, confirmed present in this
review). No N=32 or N=64 GEMV tile was ever compiled or simulated in this
project. This limits the reliability of any downstream conclusion drawn
from the N=32/N=64 rows (e.g. the "B=32 marginal-gain" and "KV cache 16.8 MB"
discussion in `out/loop/FINAL_REPORT.md:212-214`): a linear model assumes
the per-mvin-command/queue-depth behavior characterized at N=1 and N=16 (see
§4's `nCmds=2`, 16-deep `XactTracker` discussion,
`loop/kernels.py:906-923`) continues to hold at 2x and 4x the largest
measured batch, which is an extrapolation of a microarchitectural request-
scheduling nonlinearity (`loop/kernels.py:119-120`, "非單調" / "太多在飛請求
反而讓 DRAM row buffer 打架" — the rows-per-command curve is explicitly
noted to be non-monotonic in that scan) rather than a first-principles
physical bound, and `out/loop/FINAL_REPORT.md:222` itself lists "實測 N=32
GEMV tile 以取代線性外推" (measure the actual N=32 GEMV tile to replace the
linear extrapolation) as recommended future work.

### (f) The fused kernel's overlap ratio cannot be extrapolated uniformly to the whole decode step

Round8 and round9's projection documents (`out/llama-profile/
projection_round8.md`, `projection_round9.md`) originally applied
`llama-layer-fused-n1`'s measured cycle reduction (13.0%/13.2% vs. its own
sequential baseline) *uniformly* to the entire cold decode step's
cycles/token, yielding 5.076/5.092 tok/s @ 1 GHz (+14.9%/+15.2%). This
extrapolation is invalid and has been retracted in both documents: the
fused kernel's own Saturn-device cycle share is 28,372/206,304 = 13.8% of
that one kernel's total, whereas the whole decode step's actual Saturn
device share is only 0.7-2.2% (depending on where attention is device-
assigned — `attn_scores`/`attn_pv` can be modeled on either `gemmini` or
`saturn`; see `DEVICE_MAP` in `loop/llama_project.py`). A cycle-reduction
ratio measured on a kernel whose relevant device is 13.8% of the total
cannot be applied to a workload where that device is an order of magnitude
smaller a share (0.7-2.2%) and still be expected to reproduce anywhere near
the same whole-workload percentage effect. `loop/llama_project.py` now
exposes a measured, tool-native `--overlap` flag (`DEFAULT_OVERLAP_FACTOR =
17,346/28,372 = 0.611`, the fused kernel's own measured Saturn-exposure
ratio, capped by available Gemmini cycles to hide under) instead of the
hand-applied uniform multiplier; running it against
`measured_cycles_round9.json` gives a measured whole-decode effect of
0.3% (attention device-pinned to `gemmini`) to 1.4% (attention pinned to
`saturn`, matching the fused kernel's own device split) — not 13-15%. A
second, related correction: round9's `sat_done`/`gemv_end` phase-split
probes (55.6-57.6%) were originally read as evidence Saturn's compute is
"mostly hidden"; that reading conflated *when Saturn's fixed, small amount
of assigned work runs out* (a geometric property of the mvin-stream
schedule: 4,096 mvins, one Saturn call per 16, real work in only the first
~130 calls) with *how much of Saturn's cost survives as exposed wall time*.
The mvin-rate breakdown shows 61.1% of Saturn's own compute (17,346 of
28,372 cycles) is still exposed, essentially unchanged from round8's
17,898-cycle residual — see `out/paper/probes.md` and
`out/llama-profile/projection_round9.md` for the corrected derivation.

---

## Sources not independently re-verified in this review

- The Gemmini Scala line-number citations inside `loop/kernels.py` comments
  (e.g. `LoadController.scala:30-31,74,102-106,164-170`) were checked only
  for the existence of the cited files under `repos/gemmini/src/main/scala/gemmini/`,
  not for the exact line contents.
- `out/loop/README.md`'s "Roofline correction (2026-09-09)" section, cited
  by `out/loop/FINAL_REPORT.md:82`, was not independently opened in this
  review; the roofline-formula citations in §4 above instead trace directly
  to `loop/kernels.py`.
- `out/llama-profile/projection_round3.md`'s "2026-09-09 roofline 修正"
  section (also cited at `out/loop/FINAL_REPORT.md:83`) was not opened in
  this review.
- Individual `agent_NN.txt` transcripts cited by `out/loop/FINAL_REPORT.md`
  (e.g. `out/loop/20260909-200541-dae1/agent_04.txt`,
  `out/loop/20260909-052323-9d90/agent_01.txt`) were not independently
  re-read in this review; their content is reported here only as quoted/
  summarized by `out/loop/FINAL_REPORT.md`, which is itself cited inline.
