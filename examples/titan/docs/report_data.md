# Titan — report source material

Evidence base for a written report on **Titan**: the A³ Workshop CHIA Hackathon Proposal 2, in which the CHIA agentic loop was used to implement the RISC-V **IME (Zvvm) matrix extension** on the **Saturn** vector unit in a Chipyard tree, targeting edge-LLM **INT8 GEMM**. Two rounds converged: round 1 `vmmacc.vv` / `vmtl.v` / `vmts.v`, round 2 `vqmmacc.vv`.

**This is source material, not the report.** It is dense on purpose, every number carries the path it came from, and disagreements between sources are shown rather than resolved silently.

Assembled 2026-09-17 (snapshot as of 04:28) by re-reading the artifacts. A machine-readable companion sits beside this file at `/share1/saves/max410011/hackathon/titan_runs/report_data.json` (per-run totals, every LLM call, all trace marker events, the operation-cost table and the r9 anomaly record). Nothing here is taken on trust from a summary; where a claim rests only on a human-written note, that is stated.

## Root paths

| what | path |
|---|---|
| run artifacts, ledger, one-off experiments | `/share1/saves/max410011/hackathon/titan_runs/` |
| the loop itself | `/share1/saves/max410011/hackathon/chia/examples/titan/` |
| converged design diffs | `titan_runs/{r12_rtl_a3.diff, r13_round2.diff}` |
| Stage M model diffs | `titan_runs/{r3_stageM_spike_model.diff, r6_stageM_spike_model_sail.diff}` |

## Headline numbers

| | |
|---|---|
| runs | **16 run directories** covering 13 numbered runs r1…r13 (r7, r10 and r13 were each relaunched after a failure); 15 have a non-empty trace, `r10_0912_0254` has a 0-event one |
| total wall clock | **41 h 07 m** |
| total LLM cost | **$751.13** (43 agent sessions, 39 of them real) |
| total output tokens | 6,636,804; total turns 3,708 |
| instructions delivered | **4 of 15** in the Zvvm v0.9.0 spec |
| final verification, round 1 | directed 27/27 · S2 0 failing / 150 · Gate 200/200 · S3 64/64 |
| final verification, round 2 | model 34/34 · directed 34/34 · S2 0 failing / 150 · Gate 244/244 · S3 64/64 |
| cost of the two converged runs | **$17.61** combined ($0.65 + $16.96) |

The last line is the shape of the whole project: **$733 was spent discovering what the loop and the judge had to be**, and $18 was spent using them.

## Reading order for the three things that most need care

1. **§3.4** — run r9's "iterations" never happened; the ledger's note for that run is wrong, and the correction turns r9 into a clean repeatability experiment on the S2 judge.
2. **§7.2** — the PPA figures are a *structural RTL proxy*, not synthesis, and the primary comparison is `baseline_ref`, not `baseline`.
3. **§1.4 and §4.1** — the "841-test regression" was never what the converged runs actually ran (they ran a 150-test stride sample), and the one 841-test run in the corpus is the r8 structural failure, so its 159 s wall is not the cost of a real regression.

## Caveats that apply to the whole document

1. **Line numbers in a moving file.** `/share1/saves/max410011/hackathon/chia/examples/titan/constants.py` was **edited while this document was being assembled** (529 → 572 lines on 2026-09-17; a `.bak-r14` now exists, i.e. round-3 preparation is under way). Every `constants.py` citation below was re-resolved against the 572-line version and spot-checked by symbol (`COSIM_CONFIG:71`, `retireWidth 1:93`, `A gate whose harness…:108`, `S2_COSIM_SCALA:119`, `SYNTH_CONFIG:135`, `BASELINE_CONFIG:142`, `MODEL_MAX_SKIP_FRACTION:297`, `MAX_ITERS:310`, `REGRESSION_SAMPLE:342`, `AGENT_RUNS_PER_ITER:437`, `DISALLOWED_TOOLS:491`, `SIMLOG_TAIL_LINES:510`, `SKY130_COL_PATH:513`). **If a citation does not resolve, grep the symbol name rather than trusting the number** — the file is live.
2. **The PPA work is also live.** `titan_runs/ppa/` was created and extended *during* this assembly (04:02 → 04:13 on 2026-09-17), gaining a third config partway through. §7.2 reflects the state at 04:13 and names `results.json` as canonical.
3. **Round three is already running.** A run **`r14_0917_0425`** started at 04:25 on 2026-09-17, while this document was being assembled, and `/share1/saves/max410011/hackathon/titan_runs/round3_design.md` (04:19) sets out its scope — a survey of all 15 IME v0.9.0 instructions and a batching decision. **Nothing about r14 or round three is covered below**; §7.1's "4 of 15 implemented" is the state as of the end of round two. `titan_runs/.current_run` reads `r14_0917_0425`.
4. **`*.py.bak-rN` means "the code as run by run N"**, snapshotted minutes after run N ended (verified: `titan_loop.py.bak-r3` mtime 09-07 21:03, r3 ended 21:01). **No `.bak-r1` or `.bak-r2` exists**, so r2-era code is inferable only.
5. **The profiler log is not monotonic in `ts`**, and `dispatch` as well as `complete` is emitted *worker-side at execution time*. A task Ray never schedules therefore logs nothing at all until it eventually runs. This is load-bearing for §5.2.
6. **Three things the ledger gets wrong or omits**, all evidenced below: the r9 note attributes to the agent work it never did (§3.4, §5.7b); `r10_0912_0254` has a `ledger_notes.json` entry but no `ledger.md` row (§5.7b); and `constants.py`'s own comment calling `machine_vsetvl-0`/`vsetivli-0` pre-existing pristine failures is contradicted by the baseline measurement (§7.4).
## 1. What was built

Spec pinned throughout: **Zvvm v0.9.0** (`/share1/saves/max410011/hackathon/chia/examples/titan/specs/ime/instructions.json:3`, consumed by `.../titan/ime_encodings.py:37`).
Target parameterisation: **VLEN=256, DLEN=128, Shuttle host, INT8→INT32 GEMM** (see config table below; every trace's `run_start` carries `"vlen": 256`).

### 1.1 The four instructions

Encodings are extracted programmatically from `specs/ime/instructions.json`; the pinned bit values are re-asserted as a human cross-check in `ime_encodings.py:check_round_one_bits` (lines 217–238) and `check_round_two_bits` (lines 241–279). All paths below are relative to `/share1/saves/max410011/hackathon/chia/examples/titan/`.

| Mnemonic | Round | opcode[6:0] | funct3[14:12] | funct6[31:26] | vm[25] | other fixed bits | semantics | cite |
|---|---|---|---|---|---|---|---|---|
| `vmmacc.vv vd,vs1,vs2` | 1 | `0x57` OP-V | `0x0` OPIVV | `0x38` (56) | `1` pinned | — | integer same-width matrix multiply-accumulate: C tile += A tile × B tile at vtype SEW | `ime_encodings.py:224-229`; `specs/ime/instructions.json:601-669` |
| `vmtl.v vd,(rs1),rs2[,Lλ][,vm]` | 1 | `0x07` LOAD | `0b111` | bits[27:26]=`00` (order-preserving), bit28=`1` (mew) | field | `lambda`[31:29] 3-bit immediate | order-preserving 2-D tile **load** into vd, geometry from (SEW, LMUL, λ) | `ime_encodings.py:231-238`; `specs/ime/instructions.json:677-745`, field detail `:702-714` |
| `vmts.v vs3,(rs1),rs2[,Lλ][,vm]` | 1 | `0x27` STORE | `0b111` | same shape as `vmtl.v` | field | `lambda`[31:29] | order-preserving 2-D tile **store** from vs3, inverse of `vmtl.v` | `ime_encodings.py:233-238`; `specs/ime/instructions.json:771-841`, field detail `:796-808` |
| `vqmmacc.vv vd,vs1,vs2` | 2 | `0x57` OP-V | `0x0` OPIVV | `0x3a` (58) | `1` pinned (vm=0 aliases a *different* instruction, `vfqimmacc.vv`) | — | **Int8×Int8 quad-widening MACC into Int32** (Zvvi8i32mm) — the shape llama.cpp INT8 GEMM needs; SEW here is the *accumulator* width; A/B tiles still move via round-1 `vmtl.v`/`vmts.v` | `ime_encodings.py:260-267`; `specs/ime/instructions.json:1053-1121` |

Encoding-family facts worth carrying into the report:
* The family also defines `vwmmacc.vv` (funct6 `0x39`) and `v8wmmacc.vv` (funct6 `0x3B`) — **not implemented** (`ime_encodings.py:41-51`).
* Signed/unsigned operand mixes ride `vtype.altfmt_A`/`altfmt_B` rather than separate mnemonics; checked structurally by `check_round_two_bits` (`ime_encodings.py:270-279`) and `check_mx_split` (`:301-314`).
* `check_llvm_drift` (`ime_encodings.py:282-298`) records that LLVM ≥23.1 (its 0.1 draft) is **bit-identical to v0.9.0** for the round-one three, so real mnemonics would be safe. The harness nonetheless emits raw `.insn` words (`ime_encodings.py:184-185`); `MARCH_ROUND_ONE_CLANG = "rv64gcv_zvvmm0p1_zvvmtls0p1"` (`ime_encodings.py:186`) is defined but not what is used.

### 1.2 Config names

All in `/share1/saves/max410011/hackathon/chia/examples/titan/constants.py`.

| purpose | python name | value | defined | consumed |
|---|---|---|---|---|
| S1 directed + synth design config | `SYNTH_CONFIG`, **imported as `DIRECTED_CONFIG`** | `TitanV256D128ShuttleConfig` | `constants.py:135-136` | `titan_loop.py:66`; `_build_directed`/`_run_directed`/`_gate` |
| S2 cosim harness config (loop-owned) | `COSIM_CONFIG` | `TitanS2CosimConfig` | `constants.py:71` (env `TITAN_COSIM_CONFIG`) | `titan_loop.py:435,543,716`; generated by `_write_s2_cosim_config`, `titan_loop.py:529-543` |
| generated S2 Scala | `S2_COSIM_SCALA` | `class TitanS2CosimConfig extends Config(WithCospike ++ WithTraceIO ++ WithShuttleDebugROB ++ WithShuttleRetireWidth(1) ++ new TitanV256D128ShuttleConfig)` | `constants.py:119-131`, written to `generators/chipyard/src/main/scala/config/TitanS2Cosim.scala` (`constants.py:86-88`) | confirmed live in `titan_runs/r12_0914_0341/20260914_042028_impl_cosim_build_attempt1.stdout.txt:35` (`--legacy-configs chipyard:TitanS2CosimConfig`) |
| pristine-tree preflight baseline | `BASELINE_CONFIG` | `GENV256D128ShuttleConfig` | `constants.py:142-143` | preflight only |

Load-bearing: `S2_COSIM_SCALA` **extends `SYNTH_CONFIG`**, so S2 judges the identical design S1 just passed — the loop refuses to allow a hand-written parallel cosim config as a second place the design is specified (`constants.py:104-114`). `WithShuttleRetireWidth(1)` narrows S2 to a 1-wide host because of the Shuttle DebugROB DPI ordering hazard (§5); S1 still runs dual-issue on `DIRECTED_CONFIG`.

### 1.3 The final diffs

All under `/share1/saves/max410011/hackathon/titan_runs/`. Counting: bytes/lines via `wc`, files via `grep -c '^+++ '`, +/- via `grep -c` minus header lines (no `diffstat` available).

| diff | bytes | lines | files | net +/- | content |
|---|---|---|---|---|---|
| `r3_stageM_spike_model.diff` | 33,899 | 798 | 11 (all riscv-isa-sim) | +646 / −8 | round-1 Spike model, r3 attempt (later found tuned to the tests — §5) |
| `r6_stageM_spike_model_sail.diff` | 33,591 | 794 | same 11 | +642 / −8 | round-1 Spike model, Sail-cross-checked successor; a refinement, not a rewrite |
| `r12_rtl_a3.diff` | 96,453 | 2,035 | 27 (11 Spike + 16 RTL) | +1,521 / −53 | **converged round 1** |
| `r13_round2.diff` | 107,591 | 2,236 | 27 (11 Spike + 16 RTL) | +1,722 / −53 | **converged round 2**; adds `riscv/insns/vqmmacc_vv.h`, no new RTL files |

Provenance of the converged round-1 diff, from `titan_runs/a3_verify/results.json`:
`diff_source = r8_0910_0521/20260910_112121_impl_diff_attempt6.diff` → `out_diff = r12_rtl_a3.diff`.
That is, r12's starting point is **r8 attempt 6's RTL with the human "A3" fix applied** — the two-register capture/accumulate rewrite of `MatrixMultiplyPipe` in `titan_runs/a3_verify/20260914_013732_MatrixMultiplyPipe.a3.scala`.

Residue worth noting: `r12_rtl_a3.diff` carries a stray generated `generators/saturn/src/main/scala/exu/int/MatrixMultiplyPipe.scala.bak` (new file, +111 lines, `r12_rtl_a3.diff:529-534`), absent from `r13_round2.diff`. `collect_diff` picked up agent debug residue in r12; it did not recur in r13.

### 1.4 What each stage verifies, and the counts

Stage code: `_model_stage` (`titan_loop.py:1037-1135`), directed suite (`ime_tests.py:718-767`), `_gate` (`titan_loop.py:996-1006`), `_run_regression`/`_run_s2` (`titan_loop.py:556-733`), `_stress` (`titan_loop.py:1140-1180`).

| stage | judges what against what | round 1 (r12_0914_0341) | round 2 (r13_0915_2118) | evidence |
|---|---|---|---|---|
| **Stage M** | agent's Spike model vs `rvv_ref.py`, cheap tier (`full_vl_only=True`) — no RTL exists yet | **27/27** on reseed | **34/34**, converged in 1 iteration | `r12_.../20260914_034746_model_reseed.json` = `{"total":27,"counts":{"pass":27},"failing":[]}`; `r13_.../20260915_213549_model_attempt1.json` = `{"total":34,"counts":{"pass":34}}`; trace `model_iter {'attempt':1,'pass':34}` |
| **S1 directed** | RTL vs `rvv_ref.py`, cheap tier, on `DIRECTED_CONFIG` (dual-issue) | **27/27** | **34/34**, converged in 1 iteration | `r12_.../20260914_041813_impl_directed_attempt1.json`; also `..._035022_impl_resume_directed_attempt0.json` 27/27; `r13_.../20260915_220429_impl_directed_attempt1.json` = `{"total":34,"counts":{"pass":34}}` |
| **S2 cosim regression** | RTL vs **stock** Spike, lockstep cosim on riscv-vector-tests, **stride-sampled to 150** | **0 failing / 150** | **0 failing / 150** | sample size: `titan_runs/tools/next_run.sh:16` sets `TITAN_REGRESSION_SAMPLE=${SAMPLE:-150}` (constants' own default is 0 = all, `constants.py:342`). r12 trace event `rtl_resume_regression {'attempt':0,'cosim_build_ok':True,'failing':0}`; **no `regression_failure` event in either r12 or r13's trace**, which is how a zero is recorded. r13's S2 occupies 22:06:44→22:24:07 between `..._impl_cosim_build_attempt1.stdout.txt` and the gate. |
| **Gate** (full S1 sweep incl. partial-VL) | every legal (SEW, λ, LMUL, VL) combination, `full_vl_only=False` | **200/200** | **244/244** | `r12_.../20260914_044010_gate.json` = `{"total":200,"counts":{"pass":200},"failing":[]}`; `r13_.../20260915_222621_gate.json` = `{"total":244,"counts":{"pass":244},"failing":[]}` |
| **S3 lockstep stress sweep** | randomised pool of `STRESS_TEST_CASES_PER_GEOM = 64` (`constants.py:323-324`) cosim'd RTL vs the agent's **own** Spike model until deadline or divergence | **64/64**, no divergence | **64/64**, no divergence | `r12_.../20260914_044652_summary.json` `"stress_done": 64`, no `stress_failure` key; `r13_.../20260915_223330_summary.json` same; trace `stress_pool_filled {'count': 64}` in both |

**The "841" number, and what it actually is.** `841` is the *nominal* size of riscv-vector-tests (`constants.py:330,336`; `titan_loop.py:402,550,563,669,675`) and the count in r8's structural false failure ("841/841 failing in ~2.5 min"). **Neither converged run cosimmed 841 tests** — both used the 150 sample. The one near-full run is the side investigation `titan_runs/a3_verify/results.json`:

| a3_verify block | tests run | failing | seconds | failing names |
|---|---|---|---|---|
| `directed` | 27 | 0 | build 165.7 | — |
| `s2_sample` | 150 | 1 | 1,238.6 (build 133.3) | `machine_vsetivli-0` |
| `s2_full` | **839** | **2** | **5,821.3** | `machine_vsetivli-0`, `machine_vsetvli-0` |

839, not 841, because 2 were already excluded as pre-existing baseline failures. Both remaining failures are the same signature — `wdata mismatch reg 14 5 != 3000000000000005` — i.e. **IME vtype fields that stock Spike does not model**, subsequently added to `rvv_baseline_failures.json` as known-not-a-regression (§7).

**Two numbers that disagree, both real.** The knowledge note handed to the r12 RTL agent says "S2 sample 150: 149 pass" (`r12_0914_0341/work/knowledge_rtl.md:1441`), i.e. 1 failure; r12's own S2 reports 0/150. The difference is that `rvv_baseline_failures.json` grew from 2 to 3 entries (adding `machine_vsetivli-0`) in between. Trust both, in sequence.

### 1.5 Files touched by the converged diffs

RTL file list is **identical** in `r12_rtl_a3.diff` and `r13_round2.diff` (round 2 adds logic inside existing files, no new RTL file). Source: diff `+++` headers, and `a3_verify/results.json` `diff.rtl_files` / `diff.model_files`.

| repo | files |
|---|---|
| Saturn — config | `generators/saturn/chipyard/SaturnConfigs.scala` |
| Saturn — frontend | `frontend/Dispatch.scala`, `frontend/EarlyDecode.scala`, `frontend/PipelinedFaultCheck.scala` |
| Saturn — insns | `insns/MatrixInstructions.scala` |
| Saturn — backend | `backend/ExecuteSequencer.scala` |
| Saturn — exu | `exu/int/MatrixMultiplyPipe.scala` (**the IME MACC functional unit**), plus r12-only stray `MatrixMultiplyPipe.scala.bak` |
| Saturn — mem | `mem/AddrGen.scala`, `mem/LoadOrderBuffer.scala`, `mem/Mem.scala` |
| Saturn — common | `common/Bundles.scala`, `common/Parameters.scala` |
| rocket-chip | `rocket/CSR.scala` (vtype CSR λ/bs/altfmt fields), `rocket/RocketCore.scala` |
| shuttle | `exu/Core.scala` |
| Spike (`toolchains/riscv-tools/riscv-isa-sim/`) | `Makefile.in`, `riscv/riscv.mk.in`, `disasm/isa_parser.cc`, `riscv/isa_parser.h`, `riscv/encoding.h`, `riscv/vector_unit.{cc,h}`, `riscv/insns/vsetvl.h`, `riscv/insns/vmmacc_vv.h`, `riscv/insns/vmtl_v.h`, `riscv/insns/vmts_v.h`, + round 2 `riscv/insns/vqmmacc_vv.h` |

Note the rocket-chip footprint: the `vtype` CSR is defined in rocket-chip, not Saturn, which is why `CHIPYARD_DIFF_SUBMODULES` has to include it (`constants.py:57-66`).
## 2. Method

Unless a path starts with `/`, it is relative to **`/share1/saves/max410011/hackathon/chia/examples/titan/`**.

### 2.1 Loop structure

Four stages, stated in the module docstring `titan_loop.py:6-24`:

| stage | what runs | judge | function |
|---|---|---|---|
| **M** — model | agent implements Zvvm in Spike (`riscv-isa-sim`) | the S1 directed programs, run on Spike | `_model_stage`, `titan_loop.py:1037` |
| **S1** — directed | agent implements Zvvm RTL in Saturn; paired IME/RVV-1.0 self-checking programs | `rvv_ref.py`, embedded in the test programs | `_iterate`, `titan_loop.py:800` |
| **S2** — regression | Saturn's own `riscv-vector-tests`, lockstep cosim vs **stock** Spike | ships with Saturn; passed before anyone touched it | `_run_s2`, `titan_loop.py:694`, called inside `_iterate` |
| **Gate** | full directed sweep including partial-N geometries | same self-checking programs, exhaustive | `_gate`, `titan_loop.py:996` |
| **S3** — stress | randomised tile geometries, lockstep RTL vs the Stage-M Spike model | the Stage-M model | `_stress`, `titan_loop.py:1140` |

**S1 and S2 are one loop, not two stages.** `_iterate` docstring, `titan_loop.py:800-810`:
> "S2 is inside this loop rather than after it: a directed pass that broke plain RVV is not a pass."

Per attempt: edit → build `DIRECTED_CONFIG` (no cosim harness — S1 must not depend on Spike, `titan_loop.py:786`) → run directed → if clean, build `COSIM_CONFIG` once and run the RVV regression sample. Returns `True` only when both are clean **in the same attempt**.

Budgets (`constants.py`):

| constant | value | meaning | line |
|---|---|---|---|
| `MAX_ITERS` | 60 | S1/S2 combined loop per run | `constants.py:310` |
| `MODEL_MAX_ITERS` | 25 | Stage M | `constants.py:315` |
| `DEBUG_MAX_ITERS` | 5 | re-entry loop if the Gate fails after S1/S2 passed | `constants.py:316` |
| `LLM_MAX_CONSECUTIVE_FAILURES` | 2 | consecutive *infra* LLM failures before abort | `titan_loop.py:115` |
| `AGENT_RUNS_PER_ITER` | 6 | self-serve `run_directed_start`/`run_rvv_start` calls per turn, one shared pool | `constants.py:437` |
| `MODEL_MAX_SKIP_FRACTION` | 0.5 | Stage M fails if >50% of programs report SKIP (a model that declines everything must not "pass") | `constants.py:297` |
| `REGRESSION_ALERT_DELTA` | 3 | drop vs the run's own best that triggers a REGRESSION block in the prompt | `constants.py:469` |
| `STRESS_TEST_HOURS` / `STRESS_TEST_CASES_PER_GEOM` | 24 / 64 | S3 deadline and pool fill | `constants.py:280-282` |
| `REGRESSION_SAMPLE` | 0 (=all) in code, **150** in practice | S2 stride sample | `constants.py:342`; overridden by `TITAN_REGRESSION_SAMPLE=${SAMPLE:-150}` in `/share1/saves/max410011/hackathon/titan_runs/tools/next_run.sh:16` |

Note the practical consequence: **the runs as actually launched never used the code default.** Every number in §3 comes from a 150-test S2 sample, not an 841-test one.

**Convergence.** `run()` (`titan_loop.py:1439`) is strictly sequential and short-circuits: Stage M ok → `_iterate` ok (S1+S2) → if `_gate` fails, one `_iterate` re-entry with `label="gate"` and `DEBUG_MAX_ITERS` → if `stress=True`, `_stress` runs to deadline with no divergence. `result["converged"]=True` only after all of it (`titan_loop.py:1651`).

**Seeding / resuming.** Four composable CLI flags (`main()`, `titan_loop.py:1724-1753`), all splitting a `collect_diff` output along `_MODEL_PATH_PREFIX` = `SPIKE_SRC_REL` (`_split_diff`, `titan_loop.py:1218-1225`):

| flag | behaviour | function |
|---|---|---|
| `--model-diff` | reapply Spike half, rebuild, must pass outright or the run errors — "skip the agent" | `_reseed_model`, `titan_loop.py:1335` |
| `--model-seed` | reapply Spike half, rebuild, run the suite (expected to fail on newly-added instructions), hand the failures to the model agent as its opening turn | `_seed_model`, `titan_loop.py:1379` |
| `--rtl-diff` | reapply the non-Spike half to a freshly-reset tree, then **build and judge it before the first LLM call** ("attempt 0"), so the agent's first prompt carries a real S1/S2 result rather than a claim | `_rtl_resume`, `titan_loop.py:1238-1330` |
| `--rtl-diff-note` | raw text appended to the first prompt, when the attempt-0 pre-run is undesirable | — |

Round 2 was seeded this way on top of round 1: `implement.md:22-28` and `spike_task.md:7-13` both carry explicit "## Round two" sections telling the agent to *extend*, not re-derive. The trace confirms: `r13_0915_2118` `model_seed {'source': '.../r12_rtl_a3.diff', 'total': 34, 'pass': 27, 'failing': 7, 'trap': 7}` — i.e. seeding reproduced round 1's 27 and left exactly the 7 new `vqmmacc` programs trapping.

### 2.2 "Judge precedes defendant"

Stated verbatim, `titan_loop.py:12-24`:
> "**Every judge predates the defendant it judges.** rvv_ref.py is written by a human before any RTL exists; riscv-vector-tests shipped with Saturn and passed before anyone touched it; and the Spike model runs *first*, so its author cannot have seen the RTL it will later judge — that independence is a property of the ordering, not of a rule in a prompt that an agent might work around."

And, on harness ownership, verified verbatim at `constants.py:108-109`:
> "3. **The loop writes it, not the agent.** A gate whose harness the graded party maintains is not a gate."

That sits in the comment block above `S2_COSIM_SCALA` (`constants.py:64-116`). `constants.py:70`: "r9: the loop now OWNS this config instead of asking the agent for it." The motivating incident is r8's DebugROB race (§5.5), found precisely because the harness had been hand-written and was second-guessable.

Restated for the Stage-M agent, `prompts/spike_system.md:65-66`:
> "Your model will later be used to judge that implementation. A judge that was derived from the defendant judges nothing."

The judge's own soundness was audited before it was allowed to convict: `rvv_baseline_failures.json` + `.README.md` record an exclusion list of tests that fail against a **pristine** Saturn / stock Spike for reasons unrelated to Zvvm (see §7.4).

### 2.3 Two-agent separation

| | Stage M (model) agent | S1/S2 (RTL) agent |
|---|---|---|
| system prompt | `prompts/spike_system.md` | `prompts/system.md` |
| task prompt | `prompts/spike_task.md` | `prompts/implement.md` |
| debug prompt | `prompts/spike_debug.md` | `prompts/debug.md` |
| builder | `llm.make_model_llm()`, `llm.py:154` | `llm.make_llm()`, `llm.py:120` |
| session | `resume_session=True`, kept for the whole stage (`llm.py:156-161`) | **fresh session every turn** (`RESUME_SESSION` defaults `"0"`, `llm.py:60-63`) |
| may edit | Spike source (`SPIKE_SRC_PATH`) | `generators/saturn/`, `generators/rocket-chip/` (VType/VConfig only), `SaturnConfigs.scala` (`system.md:210-215`) |
| self-serve build/test | **none** — `_tools("model")` is called with `with_run_directed` defaulting `False` (`titan_loop.py:1533`); only the driver builds and tests it, once per turn | `RunDirectedTool` inserted (`titan_loop.py:1497-1503`) |
| prohibition | RULE #2, `spike_system.md:60-79`: do not read/open/grep anything under `generators/saturn/`; do not read `rvv_ref.py`, `ime_tests.py`, `ime_stress.py`, `sim_check.py` | `system.md:222-227`: may not edit `rvv_ref.py`, `ime_tests.py`, `ime_stress.py`, `ime_encodings.py`, `specs/` — "These are the judge. They were written before your implementation existed… If you believe one of them is wrong, say so via finish and stop — do not change it." |

Both agents' `BashTool` has `work_dir=CHIPYARD_PATH` (`titan_loop.py:1472-1473`). The separation is enforced by prompt instruction **plus ordering** — `_model_stage` docstring, `titan_loop.py:1043-1046`: "An agent cannot copy an implementation that has not been written yet, so the prompt's rule against reading the RTL is backed by the fact that there is nothing there to read."

*Uncertain:* whether the Titan test-generator files (which live under `chia/examples/titan/`, not under `CHIPYARD_PATH`) are even reachable from the agents' bash tool. No mount/copy step making them visible inside the container was found, but it could not be ruled out by static reading.

### 2.4 What the agent could and could not see

**Exposed every turn** (`StatusTool.read_status`, `tools.py:83`; `KnowledgeTool.read_knowledge`, `tools.py:653`):
* Directed-test status **grouped by outcome / EMUL_C / LAMBDA / SEW / LMUL**, not per-instruction — deliberately (`tools.py:1-13`).
* Failing RVV regression test **names only**, capped at `RVV_STATUS_NAMES` = 40 (`constants.py:447`; section built in `StatusTool._rvv_section`, `tools.py:96-118`).
* Its own persistent notes, `work/knowledge_{tag}.md`.
* On mismatch, `format_directed_failure` (`helpers.py:491-570`): one line per failing test; WARL classification (`skip` vs `bad_geometry`) with an explanation of which is a real failure; the LAMBDA the DUT actually supports; and **numeric evidence** — up to `MAX_EVIDENCE_TESTS` = 6 (`constants.py:504`) failures with actual mismatching values and `TITAN CDUMP` vs `TITAN CREF` tile dumps, bounded by `MAX_EVIDENCE_CHARS` = 4000 (`constants.py:401`), plus a path to full per-test simulator logs under `${AGENT_LOG_DIR}/<run>/iter<N>/`.
* Iteration orientation (`_orient`, `titan_loop.py:761-794`): iteration number/budget, `git diff --stat` of the current tree **RTL half only** (split via `_split_diff` so the RTL agent never sees the model's diffstat, `titan_loop.py:783-789`), its notes, the result, and a REGRESSION block if the run just got worse than its own best.

**Hidden**: the reference model's *source* (`rvv_ref.py`), the test generators (`ime_tests.py`, `ime_stress.py`, `sim_check.py`), the encoding/spec derivation (`ime_encodings.py`), and — for Stage M — the RTL. Feedback is limited to what the self-checking programs print at runtime; never the golden model's code.

**Feedback size history** (`helpers.py:495-503`): pre-r5 the message pasted whole build logs and simulator tails, and with `resume_session=True` that was replayed every turn — *"145M cached tokens over five calls"* (also `llm.py:3-9,44-52`). r5 cut this to bounded per-test evidence plus a log path.

### 2.5 Tool inventory, and when each arrived

| tool | class / line | what it does | introduced |
|---|---|---|---|
| `bash` | `BashTool` (chia base), instantiated `titan_loop.py:1472` | shell, `work_dir=CHIPYARD_PATH`, 300 s timeout | earliest snapshot |
| `read_spec` | `SpecTool.read_spec`, `tools.py:38-63` | lists spec docs under `specs/ime/` for the agent to open itself | earliest |
| `read_status` | `StatusTool.read_status`, `tools.py:68-118` | directed results by geometry + failing RVV names | earliest; `bad_geometry` classification added **r6** |
| `run_directed` (blocking) | `RunDirectedTool.run_directed`, `tools.py.bak-r8:208` | build + test the current tree mid-turn | **r6** — present in `titan_loop.py.bak-r7:932` and referenced by `system.md.bak-r6` rule 2; absent from `tools.py.bak-r5`. No `tools.py.bak-r6` exists, so the exact commit is inferred from the prompt/loop snapshots, not seen directly |
| `run_directed_start` / `_wait` / `_status` | `RunDirectedTool`, `tools.py:211-556` | non-blocking start/poll pair + per-turn job/budget listing | **r9** — `tools.py.bak-r8` still has only the blocking form; `tools.py.bak-r9b` has the triple |
| `run_rvv_start` / `run_rvv_wait` | `RunDirectedTool`, `tools.py:328-372` | self-serve S2 regression run, same shared budget/job table | **r13** — `tools.py.bak-r9b` has no `run_rvv*`; `tools.py.bak-r13` is the first snapshot with it, and is byte-identical to the current `tools.py`. **Caveat:** the traces show `agent_rvv` events already in `r10_0912_0302` and `r11_0912_0535`, so the capability was live by r10; the `.bak` label lags the actual introduction. Trust the traces: `run_rvv` shipped **by r10**, and `tools.py.bak-r13` is simply the earliest surviving snapshot containing it. |
| `append_knowledge` / `read_knowledge` | `KnowledgeTool`, `tools.py:643-676` | durable cross-iteration notebook, survives session resets | earliest |
| `finish` | `FinishTool`, `tools.py:678-704` | declares the turn's edits complete; loop then builds and grades | earliest |

**Budget plumbing**: `work/agent_budget_{tag}` is a plain integer counter reset each iteration (`RunDirectedTool.reset_budget`/`_bump`, `tools.py:555-565`; `titan_loop.py:1497-1502`), paired with `work/agent_budget_{tag}.jobs.json`, both on the head node so the driver's `drain()` can see jobs the tool-server actor started (`tools.py:245-253`). `tag` ∈ {`model`, `rtl`}; cap `AGENT_RUNS_PER_ITER` = 6, **shared** between `run_directed_start` and `run_rvv_start`. Only the RTL agent gets the tool at all (`with_run_directed=True` only for `tag="rtl"`, `titan_loop.py:1533`). Both are advertised in the task prompt as the primary iteration-speedup mechanism (`implement.md:84-99`, `debug.md:20-26`).

### 2.6 Prompt/loop anti-patterns that had to be patched

| anti-pattern | fix | evidence / run |
|---|---|---|
| **SKIP amnesty** — geometry mismatches (DUT answering an illegal WARL λ) were all classified as harmless `skip` | added `bad_geometry` outcome; `classify_run` calls every illegal round-up `bad_geometry`, downgraded to `skip` only when whole-run evidence excuses it (`reconcile_geometry`) | `helpers.py:1-34`: "cost five iterations: *every* geometry disagreement was called a skip… granted amnesty fifteen times per run." 0 occurrences of `bad_geometry` in `helpers.py.bak-r5`, 13 in `helpers.py.bak-r6` → **fixed at r6** |
| **Uncontrolled sub-agent spawning** — the RTL agent used the CLI's `Task` tool, putting edits and spend outside the loop's accounting | `DISALLOWED_TOOLS=["Task"]` **and** prompt rule 7 "Do not spawn sub-agents. No `Task`, no delegation…" | `constants.py:491`: "in r5 the agent used it freely, which put edits and token spend outside every accounting the loop keeps." Rule added `system.md.bak-r6:259` → **fixed at r6** |
| **Backgrounded MCP call graded an instrumented tree** — a `run_directed` call over 120 s was silently backgrounded by the `claude` CLI; in `-p` mode the model then ended its turn with an unfinished job and the loop graded whatever was in the tree | three parts: (1) non-blocking `run_directed_start`/`_wait` so no single tool call approaches 120 s (`tools.py:211-253`); (2) prompt rule against ending a turn with instrumentation in the tree or a job unfinished (`debug.md:20-24`, `system.md:274-281`); (3) `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS` raised to 1,800,000 ms in the CLI subprocess env (`constants.py:477`, `llm.py:33-45`) | `tools.py:211-253` names the incident: "exactly what happened in r8 iteration 3: the agent installed a printf instrumented `MatrixMultiplyPipe`… The loop then graded the instrumented tree: 26/27 passing became 27 mismatch." Prompt line appears at `debug.md.bak-r9b` → **fixed at r9** |
| **`resume_session` replay cost** — full transcript replay every turn, ~$23/iteration, no new information in the replay | sessions fresh by default; continuity moved to small explicit artifacts (knowledge notes, tree diffstat, log path) | `llm.py:1-9,44-52` → **fixed at r5** |
| **Agent-authored S2 cosim harness** could drift from or weaken the design under test | loop generates and owns `TitanS2CosimConfig`, extending the agent's own `SYNTH_CONFIG` | `constants.py:64-116`, "r9: the loop now OWNS this config" → **fixed at r9** |
| **Sail-vs-test authority** (Stage M) — agent tempted to bend the Spike model until a wrong test passes | RULE: "The executable SAIL appendix is normative… If a test fails and the only way you can see to make it pass is to write something the SAIL does not say… **stop**. Do not write it," with an `append_knowledge`/`finish` reporting protocol | `spike_system.md:32-59`. Present in `.bak-r13`, **absent from `.bak-r6`** → introduced between r6 and r13; exact run **uncertain** (no intermediate snapshot). Circumstantially r7, which is when the judge moved to Sail (§5.4) |

Two of these land together in the snapshot trail (the async tool split and the "no instrumentation left in the tree" line, both between r8 and r9b) and the loop's own comments trace both to the same r8-iteration-3 incident — internally consistent.
## 3. Results table

### 3.1 Per-run ledger

Source of truth: `/share1/saves/max410011/hackathon/titan_runs/ledger.md`, regenerated by
`/share1/saves/max410011/hackathon/titan_runs/tools/ledger.py`, which shells out to
`/share1/saves/max410011/hackathon/titan_runs/tools/cost_report.py` and parses
`<run>/trace/ChiaProfileCollector.log`. Notes come from
`/share1/saves/max410011/hackathon/titan_runs/ledger_notes.json` (author's own, in Chinese; translated below).
I independently re-parsed all 13 traces (script: scratchpad `agg.py`) and **reproduce every ledger number exactly**.

Runs are listed in chronological order (the ledger file sorts lexically, which interleaves r1/r10).

| # | run dir | wall | LLM calls | cost | out tokens | converged | note (translated from `ledger_notes.json`) |
|---|---|---|---|---|---|---|---|
| 1 | `r1_0907_1231` | 0:28:41 | 1 | $11.66 | 78,093 | no | (no note) — Stage M first attempt |
| 2 | `r2_0907_1327` | 4:14:18 | 1 | $11.44 | 82,810 | no | (no note) — 4h14m wall for a single 23-min LLM call: Ray placement-group deadlock |
| 3 | `r3_0907_1742` | 3:14:47 | 5 | $110.81 | 772,218 | no | Stage M converged in 3 iterations 27/27; S1 both iterations all-trap (cosim config bug) |
| 4 | `r4_0907_2120` | 1:55:08 | 5 | $50.56 | 475,806 | no | Model reused; S1 5 iterations stuck at 12 mismatch / 15 skip; 6th hit the rate-limit quota |
| 5 | `r5_0908_1057` | 3:03:34 | 5 | $113.84 | 690,497 | no | Continues r4; 5 iterations, 12/15 unmoved; diagnosed: feedback carried no numbers, `rvv_ref` layout wrong |
| 6 | `r6_0909_0440` | 8:06:38 | 5 | $126.81 | 1,374,000 | no | S1 rebuilt (numeric feedback, `run_directed`, fresh session); 8/27 pass; includes 81 min rate-limit wait |
| 7 | `r7_0909_1935` | 0:04:19 | 0 | $0.00 | 0 | no | Died at startup: root filesystem full |
| 8 | `r7_0909_2306` | 5:52:18 | 5 | $107.45 | 1,249,008 | no | Judge switched to Sail; scratch moved to /share1; continues r6; 2→16→16→16→25→25 pass |
| 9 | `r8_0910_0521` | 6:07:36 | 6 | $120.67 | 1,229,079 | no (manual stop) | Continues r7; iter2 26, iter3 regressed to 27 mismatch (tool backgrounding), iter4 recovered, **iter6 S1 27/27**; S2 then reported 841/841 failing — structural; stopped by hand |
| 10 | `r9_0912_0137` | 1:11:06 | 4 | $0.00 | 0 | no (manual stop) | Continues r8 27/27; S2 sample of 150 gave 21 failures (all floating-point); agent had no tools, 3 blind iterations 21→21→23; stopped by hand. **See §3.4 — the trace contradicts this note.** |
| 11 | `r10_0912_0254` | — (empty trace) | 0 | $0.00 | 0 | no | Credentials expired, run was a no-op; stopped |
| 12 | `r10_0912_0302` | 2:01:52 | 1 | $37.28 | 226,679 | no (manual stop) | Credentials fixed; attempt 0 S2 sample 23 failing, iter1 22 (agent self-tested with `run_rvv`); found the S2 judge was linked against the IME-modified Spike, suspected ABI skew; stopped to run a discriminating experiment |
| 13 | `r11_0912_0535` | 2:21:03 | 2 | $43.00 | 332,560 | no (manual stop) | Stock-Spike judge, `vsetvl-0` excluded; S2 sample 24→23; agent self-tested 2 iterations, `.vf` still failing; stopped to run RTL bisect |
| 14 | `r12_0914_0341` | 1:05:23 | 1 | $0.65 | 3,745 | **yes** | Start point = the A3-corrected diff; **whole loop converged**: directed 27/27, S2 sample 0 failing, Gate 200/200, S3 lockstep 64/64 |
| 15 | `r13_0915_2100` | 0:05:33 | 0 | $0.00 | 0 | no | Round-2 first submit; Ray session log file vanished → `apply_diff` crashed; cluster restarted |
| 16 | `r13_0915_2118` | 1:14:47 | 2 | $16.96 | 122,309 | **yes** | **Round 2 `vqmmacc.vv` converged first try**: Stage M 1 iteration 34/34, S1 1 iteration 34/34, S2 sample 0 failing, Gate 244/244, S3 64/64 |

**Totals** (`ledger.md` footer, reproduced by my re-parse): **41 h 07 m wall, $751.13, 43 LLM calls**
(sum of out tokens = 6,636,804; note `r10_0912_0254`'s trace is a 0-byte/0-event file so it contributes nothing).

Caveat: "LLM calls" counts *agent sessions* (one `prompt` remote-function call per loop iteration), **not**
API requests. Each session is a multi-turn agentic session; cache-read tokens per session run to tens of
millions (e.g. `r10_0912_0302` attempt 1: `cache_read_input_tokens = 54,161,530` for a single call —
`r10_0912_0302/trace/ChiaProfileCollector.log`). Raw `input_tokens` is always ~10–200: essentially all
context is served from cache.

### 3.2 Cost and wall clock by stage

Derived by bucketing each trace's `section_start` intervals and attributing each LLM `complete` event to the
section containing its completion timestamp (script: scratchpad `stage.py`; sections are named
`model`, `impl:iterN`, `gate` in the trace).
"Prologue" = run start → first `section_start`, i.e. `reset_chipyard` + model reseed + RTL resume + the
first build/directed baseline; no LLM runs there by construction.

| bucket | wall | cost | share of cost |
|---|---|---|---|
| Stage M (`model` sections) | 7:15:29 | $105.50 | 14.0% |
| S1 + S2 (`impl:iterN` sections) | 31:38:45 | $645.61 | 86.0% |
| Gate + S3 (`gate` section) | 0:17:57 | $0.00 | 0% |
| prologue (reset/seed/resume/baseline build) | 1:54:57 | $0.00 | 0% |
| **total** | **41:07:08** | **$751.11** | |

Two structural facts fall out of this:

* **The Gate and Stage 3 sweep cost no LLM money at all.** They are pure compute: 8:34 in `r12_0914_0341`
  (200 gate tests + 64-case stress pool) and 9:22 in `r13_0915_2118` (244 + 64).
  Files: `r12_0914_0341/20260914_044010_gate.json` (`{"total": 200, "counts": {"pass": 200}, "failing": []}`),
  `r13_0915_2118/20260915_222621_gate.json` (`{"total": 244, "counts": {"pass": 244}, "failing": []}`).
* **86% of the money went into the RTL implementation stage**, and inside that stage the S2 cosim
  regression is what the agent could never close without human-side debugging (r8→r11).

Per-run stage split (only runs that contain more than one bucket):

| run | Stage M wall / cost | impl wall / cost | gate wall / cost |
|---|---|---|---|
| `r1_0907_1231` | 0:28:40 / $11.66 | — | — |
| `r2_0907_1327` | 4:14:17 / $11.44 | — | — |
| `r3_0907_1742` | 2:18:23 / $77.84 | 0:56:23 / $32.96 | — |
| `r12_0914_0341` | — (reseeded) | 0:26:03 / $0.65 | 0:08:34 / $0.00 |
| `r13_0915_2118` | 0:14:08 / $4.56 | 0:45:33 / $12.41 | 0:09:22 / $0.00 |

`r4`–`r11` are impl-only runs (the Stage M model was reseeded from a stored diff, which is why they have no
`model` section): their whole cost is S1/S2.

### 3.3 Per-iteration LLM cost

Every LLM call in every trace, in order, with cost / output tokens / session duration. (`func` is always
`prompt`.) Extracted from each `<run>/trace/ChiaProfileCollector.log` `complete` event's
`extra.total_cost_usd` / `extra.output_tokens` / `exec_time_s`.

| run | # | stage | cost | out tok | session duration |
|---|---|---|---|---|---|
| r1_0907_1231 | 1 | model | $11.66 | 78,093 | 0:25:25 |
| r2_0907_1327 | 1 | model | $11.44 | 82,810 | 0:23:21 |
| r3_0907_1742 | 1 | model | $18.63 | 93,566 | 0:27:31 |
| r3_0907_1742 | 2 | model | $30.24 | 131,122 | 0:37:40 |
| r3_0907_1742 | 3 | model | $28.97 | 370,286 | 1:12:10 |
| r3_0907_1742 | 4 | impl:iter1 | $14.82 | 79,337 | 0:19:33 |
| r3_0907_1742 | 5 | impl:iter2 | $18.14 | 97,907 | 0:23:21 |
| r4_0907_2120 | 1 | impl:iter1 | $19.85 | 174,851 | 0:40:33 |
| r4_0907_2120 | 2 | impl:iter2 | $2.94 | 6,929 | 0:02:00 |
| r4_0907_2120 | 3 | impl:iter3 | $7.69 | 92,628 | 0:19:10 |
| r4_0907_2120 | 4 | impl:iter4 | $3.23 | 19,854 | 0:04:15 |
| r4_0907_2120 | 5 | impl:iter5 | $16.85 | 181,544 | 0:37:53 |
| r5_0908_1057 | 1 | impl:iter1 | $42.83 | 255,606 | 1:00:28 |
| r5_0908_1057 | 2 | impl:iter2 | $21.37 | 139,235 | 0:32:15 |
| r5_0908_1057 | 3 | impl:iter3 | $22.81 | 103,045 | 0:25:16 |
| r5_0908_1057 | 4 | impl:iter4 | $11.96 | 83,864 | 0:19:53 |
| r5_0908_1057 | 5 | impl:iter5 | $14.88 | 108,747 | 0:29:04 |
| r6_0909_0440 | 1 | impl:iter1 | $30.42 | 287,461 | 1:06:41 |
| r6_0909_0440 | 2 | impl:iter2 | $26.67 | 319,697 | 1:18:03 |
| r6_0909_0440 | 3 | impl:iter3 | $31.59 | 309,393 | 1:16:57 |
| r6_0909_0440 | 4 | impl:iter4 | $22.34 | 269,086 | 1:11:59 |
| r6_0909_0440 | 5 | impl:iter5 | $15.79 | 188,363 | 0:44:31 |
| r7_0909_2306 | 1 | impl:iter1 | $24.26 | 251,087 | 1:01:55 |
| r7_0909_2306 | 2 | impl:iter2 | $27.26 | 288,249 | 1:16:58 |
| r7_0909_2306 | 3 | impl:iter3 | $20.99 | 290,247 | 1:11:24 |
| r7_0909_2306 | 4 | impl:iter4 | $21.47 | 221,213 | 1:02:06 |
| r7_0909_2306 | 5 | impl:iter5 | $13.47 | 198,212 | 0:53:17 |
| r8_0910_0521 | 1 | impl:iter1 | $17.06 | 184,385 | 0:51:46 |
| r8_0910_0521 | 2 | impl:iter2 | $21.90 | 246,742 | 1:06:47 |
| r8_0910_0521 | 3 | impl:iter3 | $13.38 | 161,795 | 0:39:03 |
| r8_0910_0521 | 4 | impl:iter4 | $12.93 | 156,277 | 0:43:36 |
| r8_0910_0521 | 5 | impl:iter5 | $30.02 | 251,857 | 1:11:51 |
| r8_0910_0521 | 6 | impl:iter6 | $25.38 | 228,023 | 0:57:55 |
| r9_0912_0137 | 1–4 | impl:iter1–4 | $0.00 ×4 | 0 ×4 | 0:00:04 / 0:00:03 / 0:00:03 / 0:00:03 |
| r10_0912_0302 | 1 | impl:iter1 | $37.28 | 226,679 | 1:05:28 |
| r11_0912_0535 | 1 | impl:iter1 | $14.59 | 122,207 | 0:39:51 |
| r11_0912_0535 | 2 | impl:iter2 | $28.41 | 210,353 | 0:53:41 |
| r12_0914_0341 | 1 | impl:iter1 | $0.65 | 3,745 | 0:03:27 |
| r13_0915_2118 | 1 | model | $4.56 | 45,566 | 0:10:35 |
| r13_0915_2118 | 2 | impl:iter1 | $12.41 | 76,743 | 0:23:19 |

Excluding the four r9 zero-cost no-ops: **39 real LLM sessions**, min $0.65 (`r12` iter1, 3:27 — a
confirm-nothing-to-do session), median $18.63 (mean $19.26), max $42.83 (`r5_0908_1057` iter1, 1:00:28).
The single most expensive *output* session is `r3_0907_1742` call 3 at 370,286 output tokens for $28.97.

### 3.4 Discrepancy: run r9's four "iterations" never happened

The ledger note for `r9_0912_0137` says the agent made blind edits over three rounds with no tools and the
failure count moved 21→21→23. **The artifacts do not support this**, and the difference matters for the
defect narrative:

| evidence | file |
|---|---|
| All four `prompt` calls cost $0.00, produced 0 tokens, and lasted 3–4 s | `r9_0912_0137/trace/ChiaProfileCollector.log` |
| All four agent transcripts are 15 bytes and contain only `success=False` | `r9_0912_0137/20260912_014613_impl_llm_attempt1.md` … `_attempt4.md` |
| All four captured diffs are **byte-identical** (md5 `3c90d18eab3b17d0019306dd9aac7af3`, 2043 lines, 96,796 bytes, 27 files) and identical to the resumed r8 diff `r8_0910_0521/20260910_112121_impl_diff_attempt6.diff` | `r9_0912_0137/*_impl_diff_attempt{1,2,3,4}.diff` |

So the credential expiry that the ledger blames on `r10_0912_0254` was **already silently active in r9**:
the loop ran four full build+directed+regression cycles (1:07:09 of compute) against an unchanged design.

That makes r9 an accidental, and unusually clean, **repeatability experiment on the S2 judge**: three
regression passes over one bit-identical design gave

| attempt | failing | "simulator died" among them | file |
|---|---|---|---|
| 1 | 21 | 7 | `r9_0912_0137/20260912_020559_impl_regression_attempt1.json` |
| 2 | 21 | 1 | `r9_0912_0137/20260912_022513_impl_regression_attempt2.json` |
| 3 | 23 | 6 | `r9_0912_0137/20260912_024816_impl_regression_attempt3.json` |

- Union of failing tests across the three runs: **23**; stable intersection: **19**. Flaky four:
  `machine_vfwadd_vf-1`, `machine_vmfeq_vf-0`, `machine_vfslide1up_vf-0`, `machine_vfsub_vf-1`.
- Even the *same* test gives a different divergence signature run to run. `machine_vfadd_vf-2`:
  attempt 1 `diverged after 1863 committed instructions … 7f8000007f800000 != 7fc000007fc00000`;
  attempt 2 `diverged after 2091 committed instructions … 3f8ccccd40490fdb != 3f8cccce40490fdc`;
  attempt 3 `the simulator died (exit -11) after 30960 committed instructions -- no verdict`.
- **Every** one of the 23 is a `.vf` (vector–scalar) test except `machine_vsetvl-0`. Full union:
  `vfadd_vf-2, vfdiv_vf-2, vfmacc_vf-2, vfmadd_vf-3, vfmax_vf-3, vfmin_vf-2, vfmsac_vf-2, vfmsub_vf-3,
  vfnmsac_vf-0, vfnmsub_vf-1, vfrdiv_vf-1, vfrsub_vf-3, vfsgnjn_vf-1, vfslide1up_vf-0, vfsub_vf-1,
  vfwadd_vf-1, vfwmacc_vf-0, vfwsub_vf-0, vmfeq_vf-0, vmfge_vf-0, vmfgt_vf-2, vmfle_vf-3, vsetvl-0`
  (all prefixed `machine_`).

This is the sharpest available evidence for the `.vf` defect being real and operand-path-shaped, and it also
shows the judge itself carried a nondeterministic `exit -11` failure mode that inflated counts by up to 6.

I trust the artifacts over the ledger note here: three independent artifact classes agree, and the ledger
note was written from memory during a manual stop.

### 3.5 Caveats on these numbers

* `<run>/resources.csv` is **not run-scoped** — it appears to be a snapshot of a shared appending monitor
  log, so e.g. `r7_0909_1935/resources.csv` has 11,471 samples for a 4-minute run. Do not compute per-run
  averages from it. It is still usable for absolute host load at a given timestamp (columns:
  `ts,load1,load5,mem_used_gb,CPU_used,chipyard_used,verilator_run_used,llm_used,riscv_build_used,head_local_used,database_used,CPU_total,jobs_running`).
* `jobs_running` is 0 in every sample of every run — that column never worked.
* The event type `llm_failed` never appears in any of the 13 traces (the r9 failures were recorded as
  ordinary zero-cost `complete` events, which is exactly why they went unnoticed). `agent_directed` appears
  54 times and `agent_rvv` 9 times across all runs — these are the agent's own in-session tool invocations.
  `rate_limit_wait` appears exactly once (`r6_0909_0440`, `wait_seconds = 4871.92`, i.e. **81.2 min**,
  `retries: 6`, `reset_time 2026-09-09T00:00:00+00:00`).
* Section attribution assigns an LLM call to the section that contains its *completion* timestamp. Because
  a section ends only when the next one starts, an `impl:iterN` span includes the following build + directed
  + regression time, not just the LLM session. Compare the "session duration" column of §3.3 to the "wall"
  of §3.2 to separate the two.
## 4. Measured cost of each operation

All numbers parsed from `<run>/trace/ChiaProfileCollector.log` in the 13 run dirs under `/share1/saves/max410011/hackathon/titan_runs/`, using `dispatch`/`complete` pairs keyed on `call_id` + `func` + `exec_time_s`, and the marker events `section_start/end`, `stress_pool_filled`, `regression_failure`. `/share1/saves/max410011/hackathon/titan_runs/tools/cost_report.py` was read for schema only.

Remote-function vocabulary:

| `func` | what one call is |
|---|---|
| `build_spike` | one riscv-isa-sim rebuild |
| `build_saturn` | **one combined** Chisel elaborate + firtool + Verilator compile |
| `verilator_run_remote` | one directed test against an already-built sim binary |
| `cosim_run` | one RVV cosim test |
| `prompt` | one LLM agent iteration |

**Elaborate and verilate cannot be separated from these traces** — they are one `build_saturn` call. Confirmed by matching a `build_saturn` `complete.ts` to the exact mtime of `r12_0914_0341/20260914_035007_impl_resume_build_attempt0.stdout.txt` (both `1789329007`).

### 4.1 Operation table

| operation | n samples | min | median | max | parallelism | source |
|---|---|---|---|---|---|---|
| Spike build | 21 | 0.487 s (cache hit) | ~45.1 s | 54.21 s | 1 | min `r13_0915_2118` call `02000000ffff_000002`; max `r7_0909_2306` call `04000000ffff_000003`; the ~41–47 s band spans r1,r2,r4,r5,r6,r8,r9,r10,r12 |
| Saturn elaborate+verilate | 118 | 22.37 s (incremental) | **139.56 s** | 176.53 s | 1 | min `r8_0910_0521` call `be23a20e8c1e_000617`; max same run call `..._000646`; the ~134–147 s band spans nearly every run |
| one directed test | 2,080 | 0.533 s | **3.593 s** | 127.19 s (outlier) | 1 standalone; 8-way inside the gate | min `r3_0907_1742` `04000000ffff_000151`; median `r7_0909_2306` `04000000ffff_000211`; max `r7_0909_2306` `04000000ffff_000088` |
| one cosim RVV test | 2,545 | 0.799 s | **6.332 s** all-context / **1.44 s** inside the bulk regression | 842.91 s (outlier) | 1 standalone; 8-way inside the regression | min `r10_0912_0302` `008c3629fddd_000086`; median `r12_0914_0341` `0e000000ffff_000233`; max `r13_0915_2118` `02000000ffff_000302`; bulk-regression band from r8's 841 (median 1.44 s, min 1.20, max 1.95) |
| **full 841-test RVV regression** | 1 | — | **159.0 s wall** (Σ exec = 1,217.5 s → ×7.66 speedup) | — | **8** workers | `r8_0910_0521`, attempt 6: first `cosim_run` dispatch ts 1789010779.77 → last complete 1789010938.79, then `regression_failure {attempt:6, count:841}` at 1789010938.81 |
| **200-test gate** (round 1) | 1 | — | **112.12 s wall** (Σ exec = 791.9 s → ×7.06) | — | **8** | `r12_0914_0341` `section_start{gate}` 1789331897.91 → `section_end{gate, pass:200}` 1789332010.03 |
| **244-test gate** (round 2) | 1 | — | **134.38 s wall** (Σ exec = 1,030.9 s → ×7.67) | — | **8** | `r13_0915_2118` `section_start{gate}` 1789482247.22 → `section_end{gate, pass:244}` 1789482381.60 |
| **64-case Stage 3 sweep** | 2 | 263.2 s (r12) | — | 291.1 s (r13) | **3** workers but Σ exec/wall ≈ 0.97–0.98, i.e. **≈1× effective** | window `stress_pool_filled{count:64}` → `run_end`, in `r12_0914_0341` and `r13_0915_2118`. Pool construction (64 `build_ime_test` + `pool_add`) precedes this and is excluded |
| **near-full 839-test S2, sequential side-run** | 1 | — | **5,821.3 s** | — | as configured in that driver | `/share1/saves/max410011/hackathon/titan_runs/a3_verify/results.json` `s2_full` |
| **150-test S2 sample, side-run** | 1 | — | **1,238.6 s** (+133.3 s cosim build) | — | as above | `a3_verify/results.json` `s2_sample` |
| **27-test directed, side-run** | 1 | — | build 165.7 s | — | — | `a3_verify/results.json` `directed` |

> **Warning — do not quote 159 s as "the cost of a full RVV regression."** The only full-841 run in the corpus is r8 attempt 6, which is *the Shuttle DebugROB structural failure itself* (§5): every one of the 841 tests aborted at the first bootrom instruction. 159 s is the cost of 841 tests **failing instantly**, which is exactly why `constants.py:100` describes it as "841/841 failing in 2.5 minutes" — 159 s is 2.65 min, an independent confirmation of that comment, and a confirmation that the run measured nothing about the design.
>
> For a **healthy** full sweep, use `a3_verify`: 839 tests in **5,821.3 s sequential** = 6.94 s/test. At the loop's 8-way fan-out that projects to roughly **12–13 minutes**, not 2.65. The 150-test sample the loop actually runs took 1,238.6 s sequential in the same driver.

Derived per-unit figures worth quoting:
* The gates are the clean parallel measurements: 200 tests in 112.12 s and 244 in 134.38 s, both at ×7.0–7.7 effective speedup on 8 workers — i.e. **~0.55 s of wall per directed test** once fan-out is counted.
* A single S1 iteration's irreducible compute floor is roughly **build (140 s) + directed (27 × 3.6 s ÷ 8 ≈ 12 s) ≈ 2.5 min**; adding the cosim build and the 150-sample S2 puts a full S1+S2 attempt at ~**17 min** of compute — consistent with the observed `impl:iterN` spans minus LLM session time in §3.2/§3.3.

### 4.2 Reliability notes

* **Spike / Saturn build minima are cache hits**, not cold builds (0.49 s, 0.50 s Spike; 22.37 s, 67.18 s, 79–82 s Saturn). The tight ~40–55 s / ~130–177 s clusters are the representative numbers.
* **"One cosim test" is strongly context-dependent.** The all-context median (6.33 s) mixes ad-hoc agent debugging runs — which appear to carry extra overhead, plausibly waveform/trace dumping — with bulk regression runs at 1.44 s. Quote whichever context you mean; do not use 6.33 s as "the" cosim cost.
* The 841-regression, both gates and the 64-sweep are each **single observed instances**. There is no second full-841 run in the corpus (the recurring `regression_failure` counts of 21–24 are residual-subset runs, not full sweeps). Treat the wall times as anecdotal.
* **Parallelism = 8** is from counting distinct `worker_id`s dispatched to inside each window, corroborated by `resources.csv`'s `verilator_run_used` peaking at exactly **7.2** (= 8 × 0.9 resource units) in `r8_0910_0521`, `r12_0914_0341`, `r13_0915_2118`. Corroboration only: `resources.csv` is not run-scoped (§3.5).
* **The Stage 3 sweep's ≈1× effective concurrency is a real finding**, not an artifact: Σ`exec_time_s` ≈ wall despite 3 workers. That phase is scheduled essentially serially and is the obvious cheap win if S3 is ever widened.
* Two unexplained outliers, both real elapsed execution (dispatch→complete gap equals `exec_time_s`, so not queueing):
  * `cosim_run` **842.91 s** — `r13_0915_2118` call `02000000ffff_000302`, dispatch 1789481236.07 → complete 1789482079.01. ~130× the aggregate median, no error marker at that `call_id`. Root cause not visible in the trace.
  * `verilator_run_remote` **127.19 s** — `r7_0909_2306` call `04000000ffff_000088`, ~35× median, likewise unexplained.

### 4.3 LLM iteration distribution

43 `prompt` completes across 13 runs; model recorded as **`claude-opus-4-7`** in every `complete.extra`. Total **$751.10**. Excluding the 4 anomalous r9 zero-cost rows (§3.4): **n = 39**.

| metric | min | median | max | total |
|---|---|---|---|---|
| wall time | 120.997 s (`r4` att 2) | **2,433.63 s** (`r4` att 1) | 4,683.35 s (`r6` att 2) | 101,479 s = 28 h 11 m |
| cost | $0.65 (`r12` att 1) | **$18.63** (`r3` att 1) | $42.83 (`r5` att 1) | $751.10 |
| output tokens | 3,745 (`r12` att 1) | **174,851** (`r4` att 1) | 370,286 (`r3` att 3) | 6,636,804 |
| turns | 8 (`r4` att 4) | **100** (`r4` att 1 / `r13` att 2) | 217 (`r10` att 1) | 3,708 |

Full per-attempt table (run, attempt, wall_s, cost, output tokens, turns):

```
r1_0907_1231   1  1525.54   $11.66    78,093   77
r2_0907_1327   1  1401.13   $11.44    82,810   87
r3_0907_1742   1  1651.74   $18.63    93,566   99
r3_0907_1742   2  2260.56   $30.24   131,122  115
r3_0907_1742   3  4330.25   $28.97   370,286   44
r3_0907_1742   4  1173.90   $14.82    79,337  138
r3_0907_1742   5  1401.22   $18.14    97,907   71
r4_0907_2120   1  2433.63   $19.85   174,851  100
r4_0907_2120   2   120.997   $2.94     6,929   12
r4_0907_2120   3  1150.56    $7.69    92,628   19
r4_0907_2120   4   255.16    $3.23    19,854    8
r4_0907_2120   5  2273.62   $16.85   181,544   28
r5_0908_1057   1  3628.27   $42.83   255,606  181
r5_0908_1057   2  1935.39   $21.37   139,235   54
r5_0908_1057   3  1516.73   $22.81   103,045   48
r5_0908_1057   4  1193.62   $11.96    83,864   20
r5_0908_1057   5  1744.04   $14.88   108,747   97
r6_0909_0440   1  4001.27   $30.42   287,461  164
r6_0909_0440   2  4683.35   $26.67   319,697  131
r6_0909_0440   3  4617.98   $31.59   309,393  150
r6_0909_0440   4  4319.31   $22.34   269,086  109
r6_0909_0440   5  2671.78   $15.79   188,363   82
r7_0909_2306   1  3715.69   $24.26   251,087  113
r7_0909_2306   2  4618.11   $27.26   288,249  121
r7_0909_2306   3  4284.86   $20.99   290,247  101
r7_0909_2306   4  3726.50   $21.47   221,213  115
r7_0909_2306   5  3197.61   $13.47   198,212   73
r8_0910_0521   1  3106.90   $17.06   184,385  105
r8_0910_0521   2  4007.80   $21.90   246,742  121
r8_0910_0521   3  2343.02   $13.38   161,795   86
r8_0910_0521   4  2616.33   $12.93   156,277   78
r8_0910_0521   5  4311.60   $30.02   251,857  161
r8_0910_0521   6  3475.39   $25.38   228,023  128
r9_0912_0137   1     4.78    $0.00         0    1   <- no-op
r9_0912_0137   2     3.71    $0.00         0    1   <- no-op
r9_0912_0137   3     3.46    $0.00         0    1   <- no-op
r9_0912_0137   4     3.71    $0.00         0    1   <- no-op
r10_0912_0302  1  3928.92   $37.28   226,679  217
r11_0912_0535  1  2391.97   $14.59   122,207  110
r11_0912_0535  2  3221.92   $28.41   210,353  181
r12_0914_0341  1   207.53    $0.65     3,745   12
r13_0915_2118  1   635.39    $4.56    45,566   52
r13_0915_2118  2  1399.53   $12.41    76,743  100
```

The four r9 rows completed in 3.5–4.8 s with `cost_usd: 0`, no output tokens, and `num_turns: 1`. **No `llm_failed` marker exists in any of the 13 logs** — the type never occurs — so the trace alone cannot label them errors; the corroborating artifacts (§3.4) show they performed no edit at all.

### 4.4 The one-off PPA-proxy A/B (2026-09-17)

An A/B RTL-elaboration comparison was run **after** all 13 loop runs, on 2026-09-17, driver `/share1/saves/max410011/hackathon/titan_runs/tools/ppa_rtl_ab.py` (Ray job `titan-ppa-rtl-ab-1`), outputs in `/share1/saves/max410011/hackathon/titan_runs/ppa/`. Cost, from `ppa/build_ab.json`:

| build | config | wall | generated SV | job record |
|---|---|---|---|---|
| baseline | `GENV256D128ShuttleConfig`, pristine tree | 163.7 s | 667 files, 21,109,111 B | `ppa/build_ab.json` |
| ime | `TitanV256D128ShuttleConfig` + `ppa/r13_round2_rtl_only.diff` | 160.0 s | 671 files, 21,292,329 B | `ppa/build_ab.json`; job total **324.3 s**, `ppa/BUILD_DONE` = `ok 324.3s 2026-09-17T04:07:52` |
| baseline_ref | `REFV256D128ShuttleConfig`, pristine tree | 164.6 s | 669 files, 21,111,047 B | `ppa/build_ab.baseline_ref.json`; job total **164.8 s**, `ppa/BUILD_DONE.baseline_ref` = `ok 164.8s 2026-09-17T04:13:34` |

These three numbers (163.7 / 160.0 / 164.6 s cold full builds) are the cleanest available measurement of a Saturn elaborate+verilate, and they sit above the loop's in-run median of 139.56 s — the loop's builds benefit from a warm sbt/cache that a fresh `reset_chipyard` does not have. See §7.2 for what the comparison found.
## 5. Defect narratives

Nine defects in the order they were hit, plus five smaller ones (§5.10). Costs from `ledger.md` and the traces (§3).

**Reading the `.bak` trail**: `*.py.bak-rN` is *the code as run by run N*, snapshotted minutes after run N ended (verified: `titan_loop.py.bak-r3` mtime 09-07 21:03, r3 ended 21:01). **No `.bak-r1`/`.bak-r2` exist**, so r2-era code is inferable only.

**One trace subtlety that matters below**: profiler `dispatch` and `complete` are *both* emitted worker-side at execution time (`is_remote: true`, `worker_ip` on both). A task Ray never schedules therefore logs **nothing** until it eventually runs — and the log file is not monotonic in `ts`.

| # | defect | class | runs | wall lost | $ |
|---|---|---|---|---|---|
| 5.1 | S1 built the directed image with the **cosim** config | loop harness | r3 (S1 half) | ~0:56 | $32.96 |
| 5.2 | `spike_run` dispatched outside the placement group | loop scheduling | r2 | 3:50:11 dead | $0 |
| 5.3 | SKIP amnesty + evidence-free feedback | loop grading | r4, r5 | 4:58:42 | $164.40 † |
| 5.4 | `rvv_ref` vs Sail tile layout; Spike model fitted to the broken tests | **judge** | r4, r5, r6 invalidated | ~13:05 | $291.21 † |
| 5.5 | Shuttle DebugROB `retireWidth` | cosim harness (upstream) | r8's S2 | ~1:10 | ~$25 |
| 5.6 | `.vf` regression: matrix FU stalling the shared issue path | **design** | r9, r10, r11 | 5:34:01 | $80.28 |
| 5.7a | root filesystem full | infrastructure | r7_0909_1935 | 0:04:19 | $0 |
| 5.7b | expired container credentials → silent no-ops | infrastructure | **r9** + r10_0912_0254 | 1:11:06 | $0 |
| 5.8 | r13 `apply_diff` crash | infrastructure | r13_0915_2100 | 0:05:33 | $0 |

**† 5.3 and 5.4 overlap and must not be added**: both were live during r4 and r5, so r4+r5 = $164.40 is counted in each row. The union of runs affected is r4+r5+r6 = **$291.21**. §5.9 de-duplicates.

### 5.1 r3 — S1 built the directed image with the cosim config (27/27 trap, twice)

| | |
|---|---|
| **symptom** | Stage M converged cleanly (3 rounds, 27/27). S1 then returned 27/27 `trap` in attempt 1 and a byte-identical 27/27 `trap` in attempt 2, **across two different RTL diffs**. |
| **evidence** | `r3_0907_1742/20260907_202624_impl_directed_attempt1.json` and `..._205218_impl_directed_attempt2.json` are each exactly `{"total": 27, "counts": {"trap": 27}}`. `r3_0907_1742/work/status_rtl.md:4-30` shows `trap=27` with no LAMBDA/SEW/LMUL breakdown — a uniform-failure signature, i.e. harness, not design. |
| **mis-localisation** | `r3_0907_1742/work/knowledge_rtl.md:88-102` — the agent hypothesised a WARL-clamp bug in its own `vtype.lambda` decode and spent attempt 2 on a "LAMBDA always 2" experiment. Both iterations were unwinnable regardless of what it wrote. |
| **root cause** | `titan_loop.py.bak-r3:227-228` — S1's directed image was built with `COSIM_CONFIG`: <br>`artifact = get(nodes.build_saturn.options(**pg_opts).chia_remote(COSIM_CONFIG, extension=EXTENSION))` <br>The cosim config links the container image's **stock** `libriscv`, which has never heard of Zvvm. The directed programs are self-checking and need no golden model, but cospike aborts on the first `vtype` write carrying a lambda field — before any program can print its verdict. The original assumption is still recorded at `nodes.py:200-202`: *"the cosim config is the LLM's to write."* |
| **fix — shipped in r4** | `titan_loop.py.bak-r4:59` imports `SYNTH_CONFIG as DIRECTED_CONFIG`; `:250-251` builds S1 with `DIRECTED_CONFIG`; S2 gets its own `COSIM_CONFIG` build only after directed passes (`:261-272`). Comment at `titan_loop.py.bak-r4:243-249`: <br>&gt; *"Directed tests run on the design WITHOUT the cosim harness. The cosim config links the image's stock libriscv, which has never heard of Zvvm: the first vtype write with a lambda field diverges from that Spike, cospike aborts, and every program classifies as 'trap' before it can print a verdict (r3_0907_1742 burned two S1 iterations, 27/27 trap each, exactly this way)."* <br>The same diff adds `_dump_sim_logs` (`:145-158`) because r3 left no way to see *why* 27/27 trapped. |
| **cost** | r3's S1 half: 0:56:23, 2 LLM iterations, **$32.96**. Stage M in the same run was real work (27/27 in 3 iterations, $77.84), so the run is not a total loss. |

### 5.2 r2 — `spike_run` dispatched outside the placement group (3h50m deadlock)

| | |
|---|---|
| **symptom** | 4:14:18 wall, 1 LLM call, $11.44, `converged: false`. The Stage-M LLM turn finished normally (`r2_0907_1327/20260907_134953_model_llm_attempt1.md`, `Cost: $11.4399 \| Duration: 1399.8s`). Everything after it hung. |
| **localisation** | `r2_0907_1327/trace/ChiaProfileCollector.log`, 127 events: `complete build_spike` 13:50:34 → `run_end` 17:40:45. **One gap, 13,811 s = 230.2 min, zero events in between.** Then at 17:40:47 all 27 `spike_run` pairs fire and finish in 6–9 ms each. |
| **careful reading of the trace** | Every function in the run is balanced (`spike_run dispatch=27 complete=27`) — there are **no unmatched dispatches**. But the file is **non-monotonic in `ts`**: those 27 pairs sit *after* `run_end` in file order, with `ts` 17:40:47 > run_end 17:40:45. Because both event types are emitted worker-side at execution time, a task Ray never schedules logs nothing — which is precisely why the 230-minute window is empty. The tasks were submitted by the driver at ~13:50:34, sat unschedulable for 3h50m, and were logged only once teardown released the placement group. The correct statement is **"never *scheduled*, hence never logged until the end"**, not "never dispatched". |
| **root cause — verbatim in the code, twice** | `titan_loop.py:1015-1021` (`_run_on_spike` docstring): <br>&gt; *"Dispatched **inside** the chipyard placement group: the group reserves the build node's whole `chipyard` resource, so a `spike_run` scheduled outside it can never be placed (**r2_0907_1327 sat on 27 pending tasks for four hours**). The bundle carries 4 CPUs, so two runs overlap (chipyard 0.4 each)."* <br>`titan_loop.py:304-313`: <br>&gt; *"`build_saturn` needs `chipyard` and there is exactly one; the loop's placement group has reserved it for the whole run… `pg_opts` is a scheduling strategy and nothing else -- so the actor is placed in the bundle but reserves none of its `chipyard`. **Dispatching *outside* the group is the deadlock** (see `_run_on_spike`: r2 sat on 27 pending tasks for four hours that way)."* <br>Mechanically: r2's `_run_on_spike` omitted `pg_opts`, so its 27 `spike_run` tasks (`@ChiaFunction(resources={"chipyard": 0.4}, num_cpus=1)`, `nodes.py:250`) asked for `chipyard` from the node's default pool, which the `STRICT_PACK` PG `[{"CPU": 4, "chipyard": 1}]` had emptied. They sat `PENDING_NODE_ASSIGNMENT`; **Ray never errors on this**. The "27" in the comment matches the trace exactly. |
| **a wrong inference, recorded** | An earlier analysis pass attributed this to the bundle having `CPU: 1`, reading the `# CPU 4, not 1` comment at `titan_loop.py:1448-1450` as a bug-fix note. It is not: the bundle was always `CPU: 4`, and that comment explains why 4 CPUs are needed for two overlapping `spike_run`s. |
| **fix — shipped in r3** | `refs = {name: nodes.spike_run.options(**pg_opts).chia_remote(...)}` — present at `titan_loop.py.bak-r3:290`, the earliest surviving snapshot, unchanged through current `titan_loop.py:1023`. The pre-fix line is reconstructed from the post-mortem comments, not diffed (no `.bak-r2`). |
| **cost** | 4:14:18 wall of which **3:50:11 is dead**; $11.44, 82,810 out-tokens, 10.1M cache-read. The LLM work completed before the stall, so no money was lost — only a day. `ledger.md`'s `note` column for r2 is **blank**: the whole diagnosis comes from code comments, not the ledger. |

### 5.3 SKIP amnesty and evidence-free feedback (r4, r5 — ten dead iterations)

Two defects that compounded: the loop told the agent the wrong **score**, and gave it no **numbers**.

#### 5.3a SKIP amnesty

| | |
|---|---|
| **symptom** | S1 frozen at `mismatch=12 / skip=15` of 27 for **all 5 rounds of r4 and all 5 of r5** — nine consecutive JSON checkpoints, identical: `r4_0907_2120/20260907_222317_impl_directed_attempt4.json`, `..._230344_..._attempt5.json`, and every one of r5's six from `20260908_110445_impl_resume_directed_attempt0.json` to `20260908_140451_impl_directed_attempt5.json`. |
| **evidence** | `r4_0907_2120/work/status_rtl.md` → `LAMBDA=1: skip=7`. The DUT never once honoured a LAMBDA=1 request; all 7 were amnestied. |
| **root cause** | `helpers.py.bak-r5:80-85` — `classify_run`'s `_SKIP_RE` branch turned **every** `TITAN SKIP` into `Outcome("skip", …)` unconditionally; `helpers.py.bak-r5:59-60` excludes `"skip"` from `Outcome.failed`. The feedback then told the agent in writing they were "not counted against you". |
| **spec basis** | IME v0.9.0 *"Writing vtype.lambda"*, quoted at `helpers.py:27-30`: *"the implementation shall select the largest supported nonzero lambda value that is less than or equal to the requested value; if no supported value is less than or equal to the request, it shall select the smallest supported nonzero lambda value."* Rounding **up** is legal in exactly one case, and that is a property of the run as a whole, not of one program. |
| **self-narration** | `helpers.py:21-24`: *"The mistake that actually happened is the mirror image, and cost five iterations: **every** geometry disagreement was called a skip, so a DUT that answered LAMBDA=4 to a request for LAMBDA=2 was granted amnesty fifteen times per run and told in writing that it was 'not counted against you'."* The "fifteen times per run" is literally the `"skip": 15` frozen in all nine JSONs. |
| **why nothing caught it** | `MODEL_MAX_SKIP_FRACTION` (`constants.py:297`, default 0.5) — *"A model that clamps LAMBDA everywhere passes every test by declining every one of them"* — guards **Stage M only**. A 15/27 ≈ 0.56 skip rate in S1 tripped nothing. |
| **fix** | Authored as `titan_runs/patch-r5/fixes.diff` (2026-09-08 13:00, during r5's tail), first live in **r6**. New failing outcome kind `bad_geometry` split out from `skip` (`helpers.py:142-178`); `reconcile_geometry()` (`:210-258`) downgrades round-ups that whole-run evidence excuses — *"Deliberately conservative in the direction that costs iterations rather than the direction that hides bugs"* (`:221-223`); `geometry_support()` (`:261-297`) surfaces per-(VLEN,SEW) supported lambdas — *"they say '12 mismatch, 15 skip' as though the same amount of work had been done"* (`:266-267`). Agent-facing legend at `helpers.py:517-529`. 0 occurrences of `bad_geometry` in `helpers.py.bak-r5`, 13 in `.bak-r6`. |
| **post-fix form** | `r9_0912_0137/work/status_rtl.md` carries the `skip` vs `bad_geometry` legend, the per-(VLEN,SEW) support table, and the line *"Every declined lambda is a tile geometry this run did not test. Legal, but untested is not tested."* |

#### 5.3b Feedback with no numbers in it

| | |
|---|---|
| **symptom** | Feedback carried coordinates, never values: `ime_sew8_lam4_lmul1_n8: mismatch at C[0,1]` and nothing else. |
| **root cause, three layers** | (1) the program never *emitted* a value — `ime_tests.py.bak-r5:393` prints only `TITAN FAIL row=%d col=%d …`; (2) `helpers.py.bak-r5:76-85` therefore built a coordinate-only `Outcome`; (3) `helpers.py.bak-r5:186-220` `format_directed_failure` pasted `kind`/`detail` plus a raw 60-line Verilator tail. |
| **the 145M-token figure, independently verified** | `helpers.py:497-503`: *"Shrunk, in r5, from 'everything we have'… The old message pasted a 300-line simulator tail per failure and up to 100kB of build output; with `resume_session` on, all of it was replayed every turn (**145M cached tokens over five calls**) and none of it contained a single wrong *value*."* Confirmed: `tools/cost_report.py r5_0908_1057` → `cache_read = 145,384,536` over 5 calls (r4: 56,038,809). See also `constants.py:~420` and `helpers.py:481-482` (*"the coordinate alone was what ten iterations of r4/r5 had, and it was not enough"*). |
| **fix (r6)** | (a) the program prints evidence before its verdict: `TITAN DIFF r=%d c=%d exp=0x.. got=0x..` plus `TITAN CDUMP`/`TITAN CREF` tile dumps (`ime_tests.py.bak-r6:41,356-,580-586`); (b) `_DIFF_RE`/`_CDUMP_RE` capture into `Outcome.diffs`/`.dump` (`helpers.py.bak-r6:57-63,134-139`); (c) `_evidence()` + rewritten `format_directed_failure` carry real hex exp/got, budgeted by `MAX_EVIDENCE_TESTS`=6 / `MAX_EVIDENCE_CHARS` (`constants.py:504`, `helpers.py:442-464,490-570`); (d) `SIMLOG_TAIL_LINES` 60 → 200 (`constants.py:510`) — without it the DIFF/CDUMP lines, printed *after* the FAIL line, would be truncated; (e) raw log tail default **0**, *"and the zero is the point"* (`constants.py:~420`); (f) logs copied into the chipyard tree so the agent reads them itself — `constants.py:~444`: *"Until r5 they were written to TITAN_LOG_ROOT on the head, which the agent's container cannot see -- so the loop had to paste, and pasting is what made an iteration cost $23."*; (g) the mid-turn `run_directed` tool (`tools.py:211+`, wiring `titan_loop.py.bak-r7:228-309`) — *"The single most expensive property of the loop up to r5 was that a change could only be evaluated by ending the turn."* |
| **proof it worked** | `r6_0909_0440/20260909_055608_impl_llm_attempt1.md:971` → `TITAN DIFF r=0 c=1 exp=0x98 got=0xff` with full CDUMP/CREF. That is the same iteration in which the agent derives the 1/8-writeback signature (§6.2). |
| **cost (5.3a + 5.3b)** | r4 **1:55:08 / 5 calls / $50.56** (the 6th round hit the quota) + r5 **3:03:34 / 5 calls / $113.84** = **4:58:42, $164.40**, with the score structurally unable to move. Recovery run r6: 8:06:38 / $126.81 (including 81 min of quota wait) → 8/27 pass — unstuck, not converged. |
| **same-period fix** | `constants.py:491` — `DISALLOWED_TOOLS = ["Task"]`: *"in r5 the agent used it freely, which put edits and token spend outside every accounting the loop keeps, and gave the sub-agents none of the context the system prompt carries."* |

### 5.4 `rvv_ref` vs Sail tile layout — and a Spike model fitted to the broken tests (r5 → r7)

The most consequential defect in the corpus, because it was in **the judge**.

| | |
|---|---|
| **symptom** | r4, r5 and r6 all reused the same Stage-M Spike diff (`summary.json.model_source` = `titan_runs/r3_stageM_spike_model.diff`, digest `5037a57e7bc3`) while S1 refused to converge. The agent "fixed" its addressing every round to no effect — it was chasing a judge that was itself wrong. |
| **localisation, step 1 — the agent saw it** | `r5_0908_1057/work/knowledge_rtl.md:19-22`: *"Reference (rvv_ref.py) says register position of C[i,j] = i\*N_max + j (plain flat row-major over EMUL_C registers)… SPEC's SAIL uses mat_C_idx(i,j) = tile_reg_idx(i\*N_max+j, EMUL_C, LAMBDA, epr), which for EMUL_C > 1 SHUFFLES the position… **The test is judged by rvv_ref's model, NOT the SPEC's — so RTL must use reference-style flat i\*N_max+j for C.**"* The agent identified the divergence correctly and then, reasonably and disastrously, wrote to the judge rather than to the spec. |
| **localisation, step 2 — the decisive experiment** | `tools/verify_model_diff.py` → `titan_runs/verify_r6_model/verify.json`, a zero-LLM A/B of both Stage-M diffs against the **corrected** tests. Its stated contract: *"NEW diff + NEW tests → must be 27/27; OLD diff + NEW tests → must fail exactly the geometries the change moved."* <br><br>| model diff | verdict vs corrected tests | seconds | <br>|---|---|---| <br>| `r6_stageM_spike_model_sail.diff` | `{"total":27,"counts":{"pass":27}}`, `ok: true` | 48.8 | <br>| `r3_stageM_spike_model.diff` | `ok: false` — `RuntimeError: … reseeded model does not pass the directed programs: {'mismatch': 19, 'pass': 8}` | 47.0 | <br><br>**The r3 model scored 27/27 against the old tests and 8/27 against the corrected ones.** |
| **the exact discriminator** | The `By LMUL` breakdown in `verify.json`'s `old_r3_prose.status` is `LMUL=1: pass=8` / `LMUL=2: mismatch=8` / `LMUL=4: mismatch=7` / `LMUL=8: mismatch=4` — a clean 8 + 19 split on **LMUL ≥ 2**. It is *not* `EMUL_C > 1`: `By EMUL_C` reads `EMUL_C=1: mismatch=3, pass=2`, and three named failures are EMUL_C=1 (`ime_sew16_lam4_lmul2_n4`, `ime_sew16_lam4_lmul4_n4`, `ime_sew64_lam2_lmul2_n2`). Every one of the 19 fails at `C[0,0]`. |
| **root cause A — the reference** | `rvv_ref.py.bak-r6:234-244`: `def c_element_index(i, j, geom): return i * geom.n_max + j`. The A and B tiles were already correctly permuted; only the C accessor was flat. |
| **root cause B — the model was fitted to the tests, knowingly** | `r3_stageM_spike_model.diff` → `riscv/insns/vmmacc_vv.h:110` `const reg_t c_flat = i * n_max + j;`, with the comment above it: *"…The tests supply mat_a_tile / mat_b_tile in a layout-A-serialized… format that only yields correct results with the prose interpretation (contiguous load), so we follow the prose here."* Same in `insns/vmtl_v.h:95` and `insns/vmts_v.h:82` (`const reg_t flat_idx = i;`), **each acknowledging "The SAIL literally passes i through tile_reg_idx(…)" and choosing the tests anyway.** Two mutually consistent wrong artefacts are undetectable from inside the loop. |
| **why the "judge precedes defendant" ordering did not save this** | The ordering (§2.2) guarantees the model agent cannot read the RTL. It does not stop the model agent reading the **tests** — and the tests embedded the defect. This is the single clearest limit of the principle as implemented. |
| **fix** | `rvv_ref.py:394-419` transcribes `tile_reg_idx` literally; `:461-479` composes it as `return tile_reg_idx(c_sequential_index(i, j, geom), geom.emul_c, geom.lam, geom.elems_per_reg)`. Model side, in `r6_stageM_spike_model_sail.diff`: `vmmacc_vv.h:109` → `c_flat = tile_reg_idx(i*n_max + j, emul_c, lambda, epr);`, `vmtl_v.h:95` / `vmts_v.h:82` → `flat_idx = tile_reg_idx(i, lmul, eff_lambda, elems_per_reg);`, with the new comment *"The group multiplier is EMUL_C…, not LMUL. At EMUL_C = 1 this collapses to the row-major i\*N_max+j; at EMUL_C > 1 it does not, and **the SAIL is normative**."* Plus the standing prompt rule, `spike_system.md:32-59`: *"The executable SAIL appendix is normative… If a test fails and the only way you can see to make it pass is to write something the SAIL does not say… **stop**. Do not write it."* And in `rvv_ref.py:15-19`, a self-test that *"derives every geometry rule twice by independent routes and asserts the two agree"* — plus the header's standing warning, `rvv_ref.py:1-33`: *"**If this file is wrong, everything downstream is wrong and nothing will say so.**"* |
| **shipped in** | **r7_0909_2306** — the first run whose `summary.json.model_source` is `r6_stageM_spike_model_sail.diff` (digest `aef5a8df596b`). `titan_runs/r7_seed_notes.md:3`: *"READ THIS FIRST: the tile layout definition changed after these notes were written. The test reference (rvv_ref.py) and the Spike golden model were corrected to match the IME v0.9.0 spec's Sail appendix. **Four claims in the notes below are now WRONG** and are marked SUPERSEDED inline."* r7's S1 then moved 2 → 16 → 16 → 16 → 25 → 25 pass. |
| **uncertain** | No snapshot captures the `rvv_ref.py` edit itself (`.bak-r6` was deliberately kept as the pre-fix reference). Bracketed between `.bak-r6` (09-09 13:48) and the `verify_r6_model` run (13:53–13:55). |
| **cost** | The RTL work of r4, r5 and r6 was graded against a wrong judge: **$50.56 + $113.84 + $126.81 = $291.21**, ~13 h. Not all waste — r6 found the real `PipelinedFaultCheck` vl-clobber bug (§6.2) — but every score those runs reported was meaningless, which is why r7 restarts from a 2/27 baseline (`r7_0909_2306` `rtl_resume_directed {attempt: 0, mismatch: 25, pass: 2}`) after r6 had "reached" 8/27. |

### 5.5 r8 — Shuttle DebugROB `retireWidth` (841/841 in 2.5 minutes)

| | |
|---|---|
| **symptom** | r8 attempt 6 reached **S1 27/27** — round one's first clean directed pass — and S2 immediately reported **841 of 841 failing**, all with a PC mismatch at the very first bootrom instruction. |
| **localisation** | The timing is the tell. `r8_0910_0521/trace/ChiaProfileCollector.log:9748` → `{"type":"regression_failure","attempt":6,"count":841}`; first `cosim_run` dispatch `ts=1789010779.77` → failure `ts=1789010938.81` = **159 s**. I independently measured all 841 `cosim_run` calls: dispatched 11:26:19.77, all complete by 11:28:58.79, per-test `exec_time_s` min **1.20** / median **1.44** / max **1.95 s**. A design bug does not fail 841 heterogeneous tests in identical time. `20260910_112858_impl_regression_attempt6.json` is exactly 841 names. |
| **root cause — verbatim** | `constants.py:93-103`: <br>&gt; *"**retireWidth 1.** Shuttle's DebugROB support calls the `debug_rob` DPI once per retire lane -- `retireWidth` separate `popTrace` instances reading one shared C++ deque, in an order Verilator does not define. With the default 2-wide core the two lanes routinely pop the pair out of order, so cospike sees the second instruction first and aborts with `PC mismatch spike 10000 != DUT 10004` -- at the *first bootrom instruction*, before any test code runs. That is why r8's S2 reported 841/841 failing in 2.5 minutes. **It reproduces on a pristine tree with Saturn's own `GENV256D128ShuttleCosimConfig`, so it is a harness bug, not anything an agent did.** Rocket cosim is unaffected (retireWidth is 1 there, one popTrace)."* <br>My measured 159 s = 2.65 min corroborates the "2.5 minutes" independently. The agent-authored `TitanV256D128ShuttleCosimConfig` (`20260910_112121_impl_diff_attempt6.diff:25-33`) structurally mirrors the pristine `GENV256D128ShuttleCosimConfig` context in the same diff (`:6-8`): both carry `WithShuttleDebugROB`, neither overrides retireWidth. |
| **uncertain** | The literal `PC mismatch spike 10000 != DUT 10004` string is **not** preserved in r8's files — no S2 simlogs were kept before the manual stop. Trusted from the `constants.py` comment, structurally corroborated by the 841 count and the 159 s. |
| **diagnosis experiment** | `titan_runs/r9_verify/`: `20260910_120447_verify_subset.simlogs.txt` and `..._121031_...` both report **"0 failing regression test(s)"** on a subset; `..._132448_...` reports **125 failing** on a larger run but with *genuine* divergences (`diverged after 7001 committed instructions: … wdata mismatch reg 5 …`) instead of a first-instruction PC mismatch. The fix turned "841/841 dead in 2.5 min" into a real signal. |
| **fix — shipped in r9** | Absent from `constants.py.bak-r9`, present in `.bak-r9b`. Three parts (`constants.py:89-130`): (1) the loop **generates** the harness, `S2_COSIM_SCALA` with `new shuttle.common.WithShuttleRetireWidth(1) ++`; (2) it **extends `SYNTH_CONFIG`** — *"S2 must judge the *same* design the directed suite just passed… r8 proved the agent will let the two drift"* (`:104-107`); (3) **the loop writes it, not the agent** — *"A gate whose harness the graded party maintains is not a gate"* (`:108-109`), cf. `constants.py:68` *"r9: the loop now OWNS this config instead of asking the agent for it."* Writer: `titan_loop.py.bak-r9b:372-390` `_write_s2_cosim_config()`; the drop path `S2_COSIM_SCALA_REL` was chosen to dodge submodule `git add -N .` contamination (`constants.py:73-87`). |
| **acknowledged cost of the fix** | `constants.py:111-118`: *"S2 grades the vector unit on a 1-wide host, so a bug that only appears when two instructions commit in the same cycle is outside this gate. S1 (directed) still runs on the 2-wide DIRECTED_CONFIG… The principled fix is upstream -- Shuttle should drain its DebugROB through one popTrace with an ordered multi-pop DPI, or pass a real `has_wb` the way RocketCore does."* |
| **companion fixes** | `helpers.py.bak-r9b:687-696` — `format_regression_failure` now carries reasons, not just names: *"r8 shipped only the names, so an agent told '841 tests fail' had no way to see that all 841 died at the same instruction."* `constants.py:342` (`REGRESSION_SAMPLE`): *"a cosim of one test is 5-20s… the full suite is roughly an hour of wall clock per iteration that reaches S2. **r8 never noticed because all 841 died in 0.6s each.**"* |
| **cost** | r8: 6:07:36, 6 calls, $120.67. S1 legitimately reached 27/27 in round 6; the S2 verdict was worthless. `r9_verify` was operator-driven, $0. |

This defect is the origin of the "a gate whose harness the graded party maintains is not a gate" principle (§2.2) — found *because* the harness was hand-written and therefore suspect.

### 5.6 The `.vf` regression: the matrix FU stalling the shared issue path (r9 → r12)

The longest-running real design defect, and the one the agent never closed.

#### 5.6a Symptom per round

| stage | judge / config | result | set |
|---|---|---|---|
| pristine baseline (`titan_runs/rvv_baseline/20260910_190750_rvv_baseline_summary.json`) | unmodified tree, stock Spike, full 841 | **840/841 pass** | only `machine_vfcvt_f_x_v-0` (1-ulp) |
| r9 att.1 (`r9_0912_0137/20260912_020559_impl_regression_attempt1.json`) | r8's 27/27 diff | 21/150 | 20 `.vf`-family + `vsetvl-0` |
| r9 att.2/3 | same | 21 → **23** | **the agent never ran; this movement is cosim flakiness, not edits** (§5.7b, §3.4) |
| r10_0912_0302 | credentials fixed, agent gains `run_rvv` | 23 → **22** | fixed `vsetvl-0` + `vfrsub_vf-3` via a Shuttle `Core.scala` vtype/lambda change |
| r11_0912_0535 | judge switched to **stock** Spike | 24 → **23** | fixed `vsetivli-0`; **22 `.vf` unchanged**, signature `DUT last trap cycle=1199 pc=0x800000bc cause=2`. Agent's "scalar-FP RAW hazard" hypothesis wrong; budget exhausted |

The 20 `.vf` members: `vfadd_vf-2, vfdiv_vf-2, vfmacc_vf-2, vfmadd_vf-3, vfmax_vf-3, vfmin_vf-2, vfmsac_vf-2, vfmsub_vf-3, vfnmsac_vf-0, vfnmsub_vf-1, vfrdiv_vf-1, vfrsub_vf-3, vfsgnjn_vf-1, vfwadd_vf-1, vfwmacc_vf-0, vfwsub_vf-0, vmfeq_vf-0, vmfge_vf-0, vmfgt_vf-2, vmfle_vf-3` (all `machine_`-prefixed). Full 23-member union with the flaky four at §3.4.

#### 5.6b The judge red herring, ruled out with numbers

* `tools/s2_judge_experiment.py` → `s2_judge/A_rtl_only_stock_spike.json` (RTL-only diff, freshly built **stock** Spike, digest `6d78935237dd6593`) fails **9/10**; `B_full_diff_ime_spike.json` fails the same 9/10 ⇒ same failures under either Spike ⇒ **real RTL breakage, not a judge artefact**.
* `tools/s2_judge_c.py` → `s2_judge_c/s2_judge_c.json`: the identical rebuilt-libriscv process on a **fully pristine** tree → **0/10 failing** ⇒ the rebuild is a valid judge.
* `r11_0912_0535/work/knowledge_rtl.md:1255` states the conclusion to the agent: *"the `.vf` failures are caused by your RTL diff, not by the judge."*
* The `vsetvl` family is *legitimately* excluded — see §7.4.

#### 5.6c File-level bisect — produced no localisation

`tools/rtl_bisect.py` → `titan_runs/rtl_bisect_0914_0047/bisect.json`. (`rtl_bisect_0757/` is **empty** — an aborted earlier attempt.) 16 candidate RTL files, 3 discriminating tests (`machine_vfadd_vf-2`, `machine_vmfeq_vf-0`, `machine_vfsgnjn_vf-1`). **8 arms; only one built:**

| arm | files | build s | test s | outcome |
|---|---|---|---|---|
| `a_all_rtl` | 16 | 149.8 | 38.9 | **fails 3/3** — control reproduces |
| `b_saturn_only` | 13 | 31.4 | — | build_failed |
| `c_rocketchip_shuttle_only` | 3 | 42.7 | — | build_failed |
| `d_saturn_minus_frontend` | 10 | 31.0 | — | build_failed |
| `e_saturn_minus_execseq_mmp` | 10 | 21.6 | — | build_failed |
| `f_saturn_minus_addrgen_loadseq` | 12 | 21.1 | — | build_failed |
| `drop_SaturnConfigs.scala` | 15 | 41.8 | — | build_failed |
| `drop_ExecuteSequencer.scala` | 15 | 79.0 | — | build_failed |
| **total** | | **418.4 s** | **38.9 s** | |

`bisect.json.meta.verdict`: *"smallest failing set = a_all_rtl … largest passing set = none observed … inconclusive builds: 7."* The Zvvm additions are too cross-coupled (MatrixMacc decode threaded through Dispatch / EarlyDecode / ExecuteSequencer / SaturnConfigs) for file-level bisection. **Note for the report**: the ledger says r11 was stopped "to run RTL bisect"; the bisect was run and **failed to bisect**.

#### 5.6d A/B hypothesis test — the actual localisation

`tools/ab_hypothesis.py` applies the **full** diff (so it always compiles), then one *reverse* line. `ab_experiment/results.json`:

| arm | one-line change | 3 `.vf` tests |
|---|---|---|
| `C_full_diff` (control) | — | **fail 3/3** |
| `A_no_integerMatrix` | `Parameters.scala`: `) ++ integerMatrix` → `)` | **pass 3/3** |
| `A2_stall_neutralised` | `MatrixMultiplyPipe.scala`: `io.stall := valid \|\| post_write_stall` → `io.stall := false.B` | **pass 3/3** |
| `A3_stall_valid_only` | same line → `io.stall := valid` | **pass 3/3** |
| `B_revert_vtype` | revert the Shuttle `Core.scala` IME vtype/lambda block | **fail 3/3, and worse** — PC mismatch after ~200 instructions (boot broken) |

`meta.verdict`: *"Hypothesis A CONFIRMED, B REFUTED … Minimal IME-preserving fix = A3: delete post_write_stall from MatrixMultiplyPipe."* Reconfirmed in `ab_experiment/run2/` and `run3/`. **Semantic reverts succeeded where file reverts could not even compile.**

#### 5.6e Root cause — the full propagation chain

`titan_runs/r12_0914_0341/work/knowledge_rtl.md:1439`, verbatim:

> *"Why: `common/Parameters.scala` adds `integerMatrix` to `integerFUs` unconditionally, so the matrix FU sits in the always-instantiated "int" ExecutionUnit. Its stall reaches `ExecutionUnit.scala:75 io.iss.ready := !Mux1H(fu_sel, stalls)` and then `Backend.scala:230 dis_stall`, and under `VectorIssueStructure.Shared` the int sequencer stalls the fp sequencer. That perturbed the late-arriving `.vf` scalar (it reaches Saturn one stage later than every other scalar, via EarlyDecode io.read_frs1 -> Frontend pfc.io.s1.rs1 -> PipelinedFaultCheck:165), and **element 0 of every `.vf` op came out as if the vector operand were 0**."*

Offending code, `generators/saturn/src/main/scala/exu/int/MatrixMultiplyPipe.scala:104-110` (a new file from the r8 diff):

```scala
// Post-write stall as before -- gives previous write time to drain from
// the ll_write hiccup path before the next cell's rvd read fires.
val post_write_stall = RegInit(false.B)
when (op.matrix_last_k && io.write.fire) { post_write_stall := true.B }
  .otherwise                             { post_write_stall := false.B }
io.stall := valid || post_write_stall
```

registered into the shared group at `generators/saturn/src/main/scala/common/Parameters.scala:145-150`:

```scala
def integerFUs(idivDoesImul: Boolean = false) = integerALUs ++ Seq(
  IntegerDivideFactory(idivDoesImul),
  PermuteUnitFactory,
) ++ integerMatrix
```

So: a stall the agent added to the **matrix** functional unit propagated through the shared issue structure and corrupted the **floating-point** scalar operand path. Nothing about it is visible from the directed IME tests, which is why S1 stayed at a clean 27/27 throughout.

**The irony**: `post_write_stall` was itself an agent fix for an earlier symptom, and the knowledge file records both its introduction and its refutation — *"Added `io.stall := valid || post_write_stall` (1 extra stall cycle after each write). Result: BYTE-IDENTICAL failure to original"* (`knowledge_rtl.md:342`), *"**REFUTED**: Extra stall between iters does NOT fix the bug"* (`:344`), then later *"This took us from 3 → 2 fails"* (`:509`). It was kept because it appeared to help, and it was the cause of a larger problem elsewhere.

#### 5.6f Fix and verification

`tools/make_fixed_diff.py` deletes the register and its `when` block and rewrites the last line to `io.stall := valid`, emitting `titan_runs/r12_rtl_a3.diff` (2,035 lines, 27 files; `a3_verify/results.json.diff.checks` all true: `rtl_stall_valid`, `rtl_no_post_write_stall`, `files_match_r8`). Reconstructed body at `a3_verify/20260914_013732_MatrixMultiplyPipe.a3.scala`.

`a3_verify/results.json`: directed **27/27**; S2 sample 150 → **1** failure (`machine_vsetivli-0`, the expected exclusion); S2 full 839 → **2** (`vsetivli-0` + `vsetvli-0`); **zero `.vf` failures**.

The r12 operator note adds two guardrails: *"Do NOT 'fix' this by forcing lambda to 0 — the IME spec requires a configuration instruction in the IME-legal domain to select the largest supported nonzero lambda (preserve-or-initialize). S3's IME-model lockstep judges those fields."* and *"if your first run_directed and run_rvv come back clean, you are done — call finish. Do not invent work."* **r12 then converged the whole loop in one LLM call, $0.65.**

And it stuck: r13's impl agent wrote, unprompted, *"Kept `io.stall := valid` (valid-only) -- folding post-write-stall broke S2 in a prior round."* (§6.7). The `knowledge_rtl.md` mechanism carried the landmine across five runs.

#### 5.6g Cost

| item | wall | $ |
|---|---|---|
| r9_0912_0137 | 1:11:06 | $0.00 † |
| r10_0912_0254 | — (0-byte trace) | $0.00 |
| r10_0912_0302 | 2:01:52 | $37.28 |
| r11_0912_0535 | 2:21:03 | $43.00 |
| **ledgered subtotal** | **5:34:01** | **$80.28** |
| `rtl_bisect_0914_0047` | 457.3 s (~9 min), no result | $0 |
| `ab_experiment` runs 1–3 | ~20–25 min | $0 |
| `a3_verify` | 7,362.7 s (2:02:43); S2 full is 5,821.3 s of it | $0 |
| **out-of-loop forensics** | **~2:32** | **$0.00** |

† r9's $0.00 is real, not a logging gap — §5.7b. **Three paid agent rounds (5h34m, $80.28) failed to localise it; ~2.5 h of unattended scripted experiments found and fixed it.**

### 5.7 Two infrastructure failures

#### 5.7a Root filesystem full (r7_0909_1935)

| | |
|---|---|
| **symptom** | 0:04:19, 0 LLM calls, $0. `ledger_notes.json`: *"啟動即失敗：根碟滿"* ("failed at startup: root disk full"). |
| **localisation** | `r7_0909_1935/trace/ChiaProfileCollector.log` (250 events, monotonic): `verilator_run_remote dispatch=27 complete=4`; every other function balanced. All 27 dispatched at **19:43:26**; 4 completed at 19:44:11 / :21 / :30 / :31 (45.4 / 55.3 / 63.7 / 65.3 s); the last event in the run is 19:44:33.87. **So the build succeeded and the run died mid-directed-suite with 23 simulations in flight** — `20260909_194325_impl_resume_build_attempt0.stdout.txt` ends normally at `make[1]: Leaving directory …TitanV256D128ShuttleConfig`. Corroborating host load: `r7_0909_1935/resources.csv` shows `load1` peaking at **795.4** and `mem_used_gb` at **249.6**, the highest in the corpus by a wide margin (compare r6: 79.0 / 74.5). |
| **uncertain** | No `ENOSPC` string survives (`resmon.err` empty, nothing in the build stdout), so the exact errno path is unconfirmed. The unmatched-dispatch count is the hard evidence. |
| **root cause** | Ray's temp dir and every node's `TMPDIR` were on the head's **root filesystem**, which filled. `cluster.yaml:9`: *"r7 起改到 /share1（root fs 877G 已滿）"* ("from r7 onward moved to /share1 — root fs, 877G, was full"). Two independent pointers at the small local root: `cluster.yaml.bak-r7:140` `ray start --head … --temp-dir=/tmp/titan_ray`, and `titan_loop.py.bak-r7:82-83` `SIM_WORK_DIR = "/tmp/titan-sim"` / `BUILD_WORK_DIR = "/tmp/titan-build"`; `preflight.py.bak-r7` hardcoded `"/tmp/titan-preflight"` in ~10 places. r6 alone ran 8h06m of builds and cosims into this. |
| **fix — `cluster.yaml.bak-r7` → `.bak-r8`, shipped in r7_0909_2306** | `--temp-dir=/share1/saves/max410011/titan_ray` (`cluster.yaml:184`); `export TMPDIR=/share1/saves/max410011/titan_scratch/node_tmp` on **every** node type and head setup (`:35,44,53,86,108,129,180`); bind-mounts on all worker containers (`:66-67, 93-94, 115-116, 136-137, 151-152`); chipyard tree relocated (`:97`). Code side: `titan_loop.py:88-92` now `os.path.join(tempfile.gettempdir(), "titan-sim")`, inheriting `$TMPDIR`; `preflight.py` de-hardcoded. |
| **two secondary hazards fixed in the same edit** (`cluster.yaml:10-15`) | (i) the head runs as one uid and the build/cosim/riscv containers as uid 1000 (`ray`), so sharing one `titan_ray` makes head-created `session_*/logs/events` unreadable and crashes the containers' core workers — hence the separate `titan_ray_ct` mount (the `llm` container runs `--user $(id -u)` and *can* share); (ii) *"Ray 的 plasma socket 是 AF_UNIX，全長不能超過 107 bytes"* — the path is deliberately short, with `titan_scratch/ray_tmp` left as a findability symlink. |
| **discrepancy** | Ledger 0:04:19 vs `resources.csv` activity 19:35:39–19:44:39 (~9 min). Trust the ledger — it is `max(ts)−min(ts)` over profiler events; resmon's span includes pre/post-run sampling, and (§3.5) `resources.csv` is not run-scoped anyway. |
| **cost** | 0:04:19, $0 — plus a restart mid-way through the r6→r7 chain. |

#### 5.7b Expired container credentials → silent no-op iterations — **and the ledger blames the agent**

This is the most consequential correction in the whole document.

| | |
|---|---|
| **the ledger says (r9)** | `ledger.md` row for `r9_0912_0137`: *"接續 r8 27/27；S2 抽樣 150 支 21 失敗（全浮點），agent 無工具盲改 3 輪 21→21→23，手動停止"* — "the agent had no tools, blind-edited 3 rounds 21→21→23". |
| **the artefacts say** | The r9 agent **never ran**. Four transcripts, each **15 bytes**, containing only `success=False`: `r9_0912_0137/20260912_014613_impl_llm_attempt1.md` and attempts 2, 3, 4 (`helpers.py:344` writes that field as the transcript's first line). All four captured diffs are byte-identical, **md5 `3c90d18eab3b17d0019306dd9aac7af3`** — the same md5 as `r8_0910_0521/20260910_112121_impl_diff_attempt6.diff`, i.e. **literally r8's final tree, unmodified, four times**. Profiler `extra` on each `prompt`: `{"cost_usd": 0, "duration_s": 0.05, "num_turns": 1, "model": "claude-opus-4-7"}` — **the CLI exited in 50 ms**; the Ray-side `exec_time_s` of 3.4–4.8 s is container spin-up. `work/agent_budget_rtl` = `0`, `work/agent_budget_rtl.jobs.json` = `{}`. **r9 is the only one of the 13 runs whose transcripts say `success=False`** — every other `*_llm_attempt*.md` in every run dir reads `success=True` and is 12 KB–469 KB. |
| **verdict** | The credential expiry began in **r9_0912_0137, one run before `r10_0912_0254`**. r9 burned **1:11:06** of cluster time on four full build + S2-regression cycles against an untouched tree (450 `cosim_run` calls, 4:28:03 aggregate exec) with zero agent participation. The ledger's `21→21→23` is **cosim nondeterminism on an unchanged tree**, not edits — `20260912_020559_impl_regression_attempt1.json` contains `"reason": "the simulator died (exit -11) after 34217 committed instructions -- no verdict"`. **The ledger's `$0.00` for r9 is literally correct and is *evidence of the bug*, not a measurement gap.** Recommend correcting `ledger_notes.json` for r9. Full repeatability analysis at §3.4. |
| **r10_0912_0254** | Died earlier still: its `trace/ChiaProfileCollector.log` is **0 bytes** — not even `run_start`. Only a pre-seeded `work/knowledge_rtl.md`; `resmon.err` empty. Because `tools/ledger.py` skips runs whose `cost_report` emits no `wall:` line, **`r10_0912_0254` has a `ledger_notes.json` entry but no row in `ledger.md`** — a real gap in the published ledger. |
| **the credential** | `cluster.yaml:70` bind-mounts the **host's** `${HOME}/.claude` into the `llm` container (`-v ${HOME}/.claude:/home/ray/.claude`), so the CLI's OAuth session / `.credentials.json` comes from the host. When the host token expired, the container's `claude` could not authenticate. |
| **why it was silent** | `titan_loop.py:108-114`: <br>&gt; *"How many LLM turns may come back `success=False` in a row before the run gives up. A CLI that fails instantly -- a bad flag, **a missing credential**, a model name the backend does not serve -- does not raise: it returns a result object with `success=False` and an empty transcript, and the loop cheerfully builds and judges an untouched tree for every remaining iteration. Two in a row is not a flake; it is a broken configuration, and continuing costs a build per iteration for nothing."* <br>There is no `llm_failed` event anywhere in the 13 traces; the type never occurs. |
| **fix — operational for r10** | `cluster-logs/down-0912_0256.log` shows teardown ~2 min after the run dir was created; `cluster-logs/up-0912_0301.log:41` shows the `llm` container restarted from `ghcr.io/ucb-bar/chia-claude-code:latest`. `r10_0912_0302` then ran normally (`20260912_043352_impl_llm_attempt1.md`, 307,592 B, `success=True`). Separately, `cluster.yaml.bak-r10` vs current differs by exactly **one line** — the `/home/ray/.claude` mount reverting from a static snapshot `/share1/saves/max410011/titan_claude_home` back to live `${HOME}/.claude`, edited between 03:00 and 03:02. The causal chain (stale token snapshot → live-refreshed host dir) is **inferred, not stated in any artefact**. |
| **code guard — shipped in r13, three runs late** | `LLM_MAX_CONSECUTIVE_FAILURES = 2` (`titan_loop.py:115`), predicate `_llm_call_failed()` (`:118-125` — *"`helpers.dump_llm` writes this same field at the top of every transcript, which is how the failed runs were identified after the fact"*), enforced at `:886-901` (RTL) and `:1064-1087` (model), emitting `llm_failed` then `run_abort`. **Pinning**: `grep -c _llm_call_failed` = **0** in `titan_loop.py.bak-r9`, `.bak-r9b`, `.bak-r10` *and* `.bak-r13`; present only in the live file (mtime 2026-09-15 20:49, 11 minutes before r13_0915_2100 started). **So r10-retry, r11 and r12 all ran unprotected**; the guard arrived as part of the Sep 15 hardening pass, ~3 days and 3 runs after the symptom. |
| **cost** | r9: 1:11:06, four wasted build+S2 cycles, **$0.00**. r10_0912_0254: minutes, $0, **unledgered**. A failure that costs nothing is a failure nothing notices. |

### 5.8 r13_0915_2100 — Ray session-log-missing → `apply_diff` crash

| | |
|---|---|
| **symptom** | 0:05:33, 0 LLM calls, $0. `ledger_notes.json`: *"第二輪首送，Ray session log 檔遺失導致 apply_diff 崩潰，重啟叢集"*. |
| **localisation** | `r13_0915_2100/trace/ChiaProfileCollector.log` (17 events): `run_start` 21:01:07 → `reset_chipyard` (0.22 s) → `build_spike` completes 21:01:48 (41.3 s) → `stock_spike` digest `6d78935237dd` 21:01:49 → **289.2 s (4m49s) gap with zero Ray events** → `run_end` 21:06:38 → instant cleanup (all <10 ms). **No `apply_diff` dispatch event exists.** Unlike r2's gap — which ended in a 27-task burst — this one contains nothing at all, placing the crash in **local driver Python between `build_spike` and the next dispatch**, where `apply_diff` runs `subprocess.run(["git","apply",...])` (`nodes.py:79-91`, `:121-134`). |
| **uncertain** | **No traceback survives anywhere**: `resmon.err` empty, `work/` empty, `20260915_210638_model_seed.diff` and `20260915_210639_summary.json` both written by the crash handler at run_end with no error field, `cluster-logs/down-0915_2113.log` a routine teardown. "Ray session log file missing" is attested **only** by the human `ledger_notes.json` note — there are no `session_latest`/`session_dir` hits anywhere in `chia/` or the titan example. Circumstantial support: the cluster had been up ~40 h continuously (since r12, Sep 14 04:47), and `cluster.yaml:71-72` sets `--log-opt max-size=50m --log-opt max-file=2` on the `llm` container, so a >40 h-old session's logs are plausible rotation candidates. |
| **fix** | Cluster restart only. `cluster-logs/down-0915_2113.log` → `cluster-logs/up-0915_2117.log` bracket the failure and the retry. **Both r13 attempts ran byte-identical code** (live `titan_loop.py` mtime Sep 15 20:49, before both), matching *"重啟叢集"* literally. `r13_0915_2118` then converged round 2 in one pass. |
| **cost** | 0:05:33, $0. Retry: 1:14:47, 2 calls, $16.96. |

### 5.9 Five smaller defects

| # | defect | evidence | cost |
|---|---|---|---|
| a | **MCP auto-backgrounding graded an instrumented tree.** The CLI moves any MCP call still running after 120 s to a background task and returns; in `-p` mode the model ends its turn and the session dies with the job still running. | `constants.py:477`: *"(r8 iteration 3: an instrumented tree was graded, 26/27 -> 27 mismatch)"*. Fixed twice over: `RUN_DIRECTED_WAIT_CAP = 100` (poll, never approach 120 s) and `CLI_MCP_AUTO_BACKGROUND_MS = "1800000"` via the LLM task's `runtime_env` (`:462-469`). Transcript evidence at §6.5. | one round of r8; ~$26 and two iterations |
| b | **`Task` sub-agents escaped accounting** (r5) | `constants.py:491`; fixed by `DISALLOWED_TOOLS = ["Task"]` | part of r5's $113.84 |
| c | **Pre-existing baseline failures billed to the agent** | `constants.py:342-356`: *"build-tests.sh prunes the suites Saturn does not implement, which is why this was assumed empty -- but r9 measured it and it is not… would otherwise be reported as a regression the agent caused, every iteration, forever."* Fixed by `rvv_baseline_failures.json` plus a `regression_baseline_missing` event so a missing file is visible, not silent. Baseline measurement: `rvv_baseline/20260910_190750_rvv_baseline_summary.json`, 840/841 pristine | folded into r9–r11 |
| d | **Hardcoded `/scratch` DB root** | `constants.py` `_default_db_root()`: *"riscv_extensions hardcodes `/scratch/vext-db`; on this cluster `/scratch` does not exist at all, and the failure surfaces only at the very end of a long build, as a PermissionError from `makedirs`."* Fixed by a per-node first-writable-candidate probe | — |
| e | **The loop's own harness leaking into the reseed diff** | `constants.py:73-87` — the generated S2 cosim config had to live under `generators/chipyard`, not beside `SaturnConfigs.scala`, because `collect_diff` runs `git add -N .` per submodule, so a file dropped in `generators/saturn` would be ignored by the superproject and staged by the Saturn pass anyway, *"putting the loop's own harness into the diff that reseeds the run."* | — |

### 5.10 What the defect list says as a whole

Attribution de-duplicated by run (each run's cost assigned once, to the defect that made it unproductive):

| run(s) | dominant defect | cost | share of $751.13 |
|---|---|---|---|
| r1, r2, r3 | Stage M bring-up; §5.2 deadlock; §5.1 cosim config | $133.91 | 17.8% |
| r4, r5, r6 | §5.3 amnesty + evidence-free feedback, then §5.4 judge defect invalidating all three | **$291.21** | **38.8%** |
| r7 (both), r8 | genuine RTL convergence on a corrected judge, ending in §5.5 | $228.12 | 30.4% |
| r9, r10 (both), r11 | §5.7b silent no-ops, then the §5.6 `.vf` hunt | $80.28 | 10.7% |
| r12, r13 (both) | **the two converged runs** | **$17.61** | **2.3%** |

Three readings, all defensible from this table:

1. **The single most expensive defect was in the judge** (§5.4, $291.21, 38.8%). It did not make the loop fail — it made the loop *succeed at the wrong thing* and report clean scores while doing so. It was caught by an out-of-band check (`tools/verify_model_diff.py`), not by the loop. And the mechanism that was supposed to prevent exactly this — "judge precedes defendant" — did not, because the model agent could read the *tests*, and the tests carried the defect.
2. **Roughly 87% of the spend predates the apparatus being correct.** The runs after the last judge/harness defect was fixed (r12, r13) cost $17.61 and converged twice, the second on a brand-new instruction in a single iteration each for model and RTL.
3. **Infrastructure defects cost almost no money and a great deal of time** (§5.2, §5.7a, §5.7b, §5.8: ~$11 attributable, ~5:20 of dead wall), and two of the four were *silent*: r2 produced no error, and r9's credential failure produced four `$0.00` "successful" iterations. **The loop's instrumentation was good at recording cost and bad at recording absence of work.**

**One aggregate not to quote.** A "roughly $424 and 22 hours went to harness/judge/reference-model defects" figure was produced during this analysis by summing r2, r3, r4, r5, r6, r7a, r8, r9, r10, r11. **It does not reconcile**: those runs sum to **$614.41**, not $424. Use the de-duplicated table above, which sums exactly to $751.13.

Three items worth escalating to whoever writes the report: (i) the r9 ledger note attributes to the agent work the agent demonstrably never did — the diff md5 is r8's, four times over; (ii) `r10_0912_0254` exists in `ledger_notes.json` but in no `ledger.md` row; (iii) the `_llm_call_failed` guard postdates the failure it describes by three runs, so r10-retry, r11 and r12 all ran unprotected.
## 6. What the agent did well and badly

Corpus: 39 real LLM sessions with metadata across 13 run dirs (plus r9's 4 no-ops), **$751, 3,708 turns, ~28 agent-hours** (§4.3). All paths under `/share1/saves/max410011/hackathon/titan_runs/`.

### 6.1 Master table — every impl attempt

`new` = lines this attempt added on top of the previous attempt's diff (`diff prev this`, counting `>` lines). For **attempt 1 of each run** this is inflated by ~790 lines of Spike-model files entering the impl diff for the first time — these are not the impl agent's work; such rows are marked `*`.

| run | att | cost | dur | turns | tools | directed before→after | new | characterisation |
|---|---|---|---|---|---|---|---|---|
| r3 | 1 | $14.82 | 1172 s | 138 | 137 | — → 27 trap | 696 | added IME `vtype` fields to `CSR.scala`; nothing decodes yet |
| r3 | 2 | $18.14 | 1399 s | 71 | 70 | 27 trap → 27 trap | 94 | clamped WARL λ to 2 to force SKIP verdicts; zero progress |
| r4 | 1 | $19.85 | 2432 s | 100 | 150 | — → (no json) | 1043 | **GOOD (honesty)**: couldn't find the tree, refused to invent — *"I want to flag this clearly rather than fabricate paths and line numbers."* |
| r4 | 2 | $2.94 | 119 s | 12 | 11 | → 15 skip/12 trap | 10 | one-line Chisel width fix (`AddrGen.scala:114` out-of-range bit select). Cheap, correct |
| r4 | 3 | $7.69 | 1149 s | 19 | 18 | 12 trap → (nb) | 354 | correctly diagnosed "vmmacc declared but never wired to an FU factory"; built the FU |
| r4 | 4 | $3.23 | 254 s | 8 | 7 | → 12 mismatch | 10 | zero-width `Fill(0,…)` elaboration fix. Tight and right |
| r4 | 5 | $16.85 | 2272 s | 28 | 27 | 12 mm → 12 mm | 28 | tile_reg_idx inverse rework; byte-identical failures |
| r5 | 1 | $42.83 | 3627 s | 181 | **257** | 12 mm → 12 mm | 843* | most tool calls in corpus; no score movement |
| r5 | 2 | $21.37 | 1934 s | 54 | 53 | 12 mm → 12 mm | 37 | chaining-clear hazard hypothesis; **byte-identical** failures |
| r5 | 3 | $22.81 | 1515 s | 48 | 47 | 12 mm → 12 mm | 73 | re-architected accumulator; still byte-identical |
| r5 | 4 | $11.96 | 1192 s | 20 | 19 | 12 mm → 12 mm | 12 | *"cycle counts are identical too across all four iterations"* — noticed the invariance, didn't escape it |
| r5 | 5 | $14.88 | 1743 s | 97 | 95 | 12 mm → 12 mm | 7 | **whole run: 5 iters, $114, 400 turns, score literally unchanged** |
| r6 | 1 | $30.42 | 4000 s | 164 | 163 | 12mm/15skip → 27 mm | 810* | **GOOD — the 1/8-writeback derivation** (§6.2) |
| r6 | 2 | $26.67 | 4682 s | 131 | 130 | 27 mm → 20 mm/7 pass | **18** | **GOOD — root cause found from that signature; 18 lines → 7 passes** |
| r6 | 3 | $31.59 | 4617 s | 150 | 149 | 20 mm → 20 mm | **0** | $31.59 and 150 turns for **zero net code change** |
| r6 | 4 | $22.34 | 4318 s | 109 | 108 | 20 mm → 19 mm | 26 | **BAD — all 4 `run_directed` calls on the single test `ime_sew32_lam1_lmul1_n8`** |
| r6 | 5 | $15.79 | 2670 s | 82 | 81 | 19 mm → 19 mm | 19 | localised the bug in notes, applied no fix |
| r7 | 1 | $24.26 | 3714 s | 113 | 112 | 25 mm → 11 mm | 827* | **GOOD**: sequential-layout rework, 25→11 in one iteration |
| r7 | 2 | $27.26 | 4617 s | 121 | 120 | 11 mm → 11 mm | 3 | 4 of 5 test runs on one test; 3 lines changed |
| r7 | 3 | $20.99 | 4283 s | 101 | 100 | 11 mm → 11 mm | **0** | **BAD — 71 min, 101 turns, 4× `run_directed` on one test, zero net change** |
| r7 | 4 | $21.47 | 3725 s | 115 | 114 | 11 mm → **2 mm** | **34** | **GOOD**: broadened back to 3 tests + `all`; 34 lines → 9 tests fixed |
| r7 | 5 | $13.47 | 3196 s | 73 | 72 | 2 mm → 2 mm | 29 | narrowed to the last 2 tests; no movement |
| r8 | 1 | $17.06 | 3106 s | 105 | 104 | 2 mm → 2 mm | 794* | no movement |
| r8 | 2 | $21.90 | 4007 s | 121 | 120 | 2 mm → **1 mm** | **8** | 8 lines, one test fixed. Efficient |
| r8 | 3 | $13.38 | 2337 s | 86 | 85 | 1 mm → **27 mm** | 130 | **BAD — diagnostic RTL shipped** (§6.4/§6.5) |
| r8 | 4 | $12.93 | 2615 s | 78 | 77 | 27 mm → 1 mm | 3 | a whole iteration spent undoing r8 att 3 |
| r8 | 5 | $30.02 | 4310 s | 161 | 160 | 1 mm → 1 mm | **0** | $30 / 161 turns / zero net change, all on one test |
| r8 | 6 | $25.38 | 3474 s | 128 | 127 | 1 mm → **27/27 pass** | 81 | converged round 1 |
| r9 | 1–4 | none | — | — | **0** | 27/27 → 27/27 | 0 (all four diffs md5-identical) | **agent never ran** (§3.4, §6.5) |
| r10 | 1 | **$37.28** | 3928 s | **217** | 216 | 27/27 → 27/27 | 835* | most expensive iteration in the corpus; S2 RVV 23→21, root cause not found |
| r11 | 1 | $14.59 | 2390 s | 110 | 109 | 27/27 → 27/27 | 794* | S2 got *worse*: *"Change made things WORSE: vfadd_vf-2 diverges at 321 committed instructions vs 167"* |
| r11 | 2 | $28.41 | 3221 s | 181 | 180 | 27/27 → 27/27 | 10 | 10 lines; S2 24→23; 22 `.vf` still failing |
| r12 | 1 | **$0.65** | 205 s | 12 | 11 | 27/27 → 27/27 | 794* | nothing to do; recognised it in 12 turns and finished. Cheapest iteration |
| **r13** | 1 | $12.41 | 1398 s | 100 | 99 | **27/34 → 34/34** | 1200* | **BEST — round-2 feature landed in one shot** (§6.7) |

Model attempts: r1 a1 $11.66 / 77 turns; r2 a1 $11.44 / 87; r3 a1 $18.63 / 99 (27 skip), a2 $30.24 / 115 (19 mismatch), a3 $28.97 / 44 (**27/27**); **r13 model a1 $4.56 / 635 s / 52 turns / 51 tool calls: 27/34 → 34/34**.

### 6.2 GOOD — the 1/8-writeback derivation (the best reasoning in the corpus)

`r6_0909_0440/20260909_055608_impl_llm_attempt1.md`, line 16, repeated in its `append_knowledge` call at line 2111:

> 粗略統計每個幾何下實際發生的 vmmacc 迭代數，均約為 `M*N*K / 8`——寫入次數只有理論值的 1/8 左右（SEW=64 M=2 那個特殊情況是 1/4）。
> *("Roughly counting the vmmacc iterations that actually occur under each geometry, all are about M\*N\*K/8 — the number of writes is only about 1/8 of the theoretical value (the SEW=64 M=2 special case is 1/4).")*

It then built an explicit table (line 2111) of `writes`, `iters` and `iters/(M*N*K)` across six geometries, self-corrected an arithmetic slip mid-note (*"Ratio 8/64 = 1/8. Fixed."*), and concluded:

> "So consistently 1/8, EXCEPT sew64_lam2_lmul1 = 1/4. … Perhaps the '1/8' bug is really 'one dLenB worth of iterations.'"

**The payoff is in the next iteration**, `r6_0909_0440/20260909_091933_impl_llm_attempt2.md` line 848:

> "FOUND IT! The bug is in `PipelinedFaultCheck.scala`. The tile ops have `umop = rs2` (which is 11 for our tests), which happens to equal `lumopMask = 0b01011 = 11`, so the code reduces vl to `ceil(vl/8)`."

The invariant extracted in attempt 1 (**exactly 1/8**) literally names the defect (`vl := ceil(vl/8)`). Cost of the fix: **18 new diff lines**, 27 mismatch → 20 mismatch / 7 pass. And it stuck: the finding becomes the permanent header of every downstream `knowledge_rtl.md`, e.g. `r7_0909_2306/work/knowledge_rtl.md:18` — *"Fixed bug: PipelinedFaultCheck vl clobber (root cause of the old '1/8 of iterations fire' symptom) … CONFIRMED FIXED, don't re-investigate."*

Secondary good marks in the same knowledge file: an explicit **ruled-out list with the reason each was excluded** (*"matched-fail experiments produced byte-identical failures — NOT the cause"*), and a tooling fact that saved later runs real time — *"Chisel `printf` is suppressed by the Verilator build here … don't rely on it."*

### 6.3 BAD — whole iterations burned on one failing test

| attempt | file | `run_directed` calls | net diff change | score |
|---|---|---|---|---|
| r7 a3 | `r7_0909_2306/20260910_025725_impl_llm_attempt3.md` | **4× `"tests": "ime_sew32_lam1_lmul1_n8"`, nothing else** | **0 lines** | 11 mm → 11 mm |
| r6 a4 | `r6_0909_0440/20260909_115825_impl_llm_attempt4.md` | **4× the same single test** | 26 | 20 → 19 mm |
| r8 a5 | `r8_0910_0521/20260910_101924_impl_llm_attempt5.md` | 3× `ime_sew32_lam1_lmul8_n8` + 1× `all` | **0 lines** | 1 mm → 1 mm |

r7 a3 is the sharpest: **$20.99, 101 turns, 71 minutes, one test, zero code change.** r6 a4's own summary is candid that every experiment hit the same wall — *"均以字節相同的失敗簽名結束"* ("all ended with a byte-identical failure signature"). The escape, when it came, came from *widening*: r7 a4 broadened back to 3 tests + `all` and turned **34 lines into 9 tests fixed**.

Once the budget mechanism exists the agent does report it, e.g. `r9_0912_0137/work/knowledge_rtl.md`: *"Budget: 4/4 run_directed calls used this iter (all four returned the same wrong values). Fix applied but not run-verified this turn."*

### 6.4 BAD — debug instrumentation left in the shipped tree

Worse than a stray print: it is **result-corrupting** instrumentation. `r8_0910_0521/20260910_081502_impl_diff_attempt3.diff:515`:

```
+  // DIAG: pack Cat(a_ext[15:0], b_ext[15:0]) as 32-bit value.
+  // For sew=32 result, we write this diag value instead of the sum.
+  val diag_val = Cat(a_ext.asUInt(15, 0), b_ext.asUInt(15, 0))
+  val write_val = Mux(op.matrix_last_k, Cat(0.U(32.W), diag_val), new_acc)
```

This deliberately writes an operand-pair probe instead of the accumulator on every `last_k` — i.e. it guarantees every test fails. The same diff also adds a brand-new untracked **117-line `generators/saturn/src/main/scala/exu/int/MatrixMultiplyPipe.scala.bak`**.

That `.bak` then **leaked across five subsequent runs**: present in r8 a3–a6, all four r9 attempts, r10 a1, r11 a1–a2, r12 a1, and even r13's inherited `20260915_213549_rtl_resume.diff`. It survives **10 consecutive accepted diffs** because no iteration was ever asked to inspect its own working tree. No `printf`/`dontTouch`/`$display` appears in any diff — consistent with the agent's own note that Chisel `printf` is inert in this build, which is exactly *why* it reached for a corrupting write instead.

### 6.5 BAD — r8 attempt 3: one backgrounded tool call cost ~$26 and two iterations

Tail of `r8_0910_0521/20260910_081451_impl_llm_attempt3.md`:

```
[Tool Call: mcp__..._run_directed]  Args: {"tests": "ime_sew32_lam1_lmul8_n8"}
[Tool Result]
MCP tool "..._run_directed" is still running after 120s. It was moved to the background
as task keyg8tjtt and keeps running; you'll receive a notification with the result...
[Response]
Waiting for the diagnostic run to complete.
[Metadata] Cost: $13.3773 | Duration: 2336.5s | Turns: 86 | ...
```

The iteration **ended on that sentence**. Two calls earlier it had installed the diagnostic (`[Tool Result] {"result":"Diagnostic version installed\n"}`). The harness then snapshotted a tree containing the corrupting `diag_val` write plus the `.bak` scaffold: `20260910_082019_impl_directed_attempt3.json` = `{"mismatch": 27}`, down from `{"mismatch": 1, "pass": 26}`. Attempt 4 ($12.93, 78 turns) was spent entirely reverting 3 lines. This single incident is the origin of three separate loop fixes (§2.6).

Contributing factor visible in the same transcript: the agent repeatedly used the local `Read` tool on chipyard paths that only exist behind the MCP shell — *"File does not exist. Note: your current working directory is /share1/saves/max410011/titan_ray/…"* — burning turns on a tool/environment mismatch its own notes had already documented.

### 6.6 BAD — r10/r11: good self-test discipline, still cannot close `.vf`

Nuance worth keeping: the agent *did* use the self-test tools correctly and *did* make measurable progress; it simply never reached the root cause, across three iterations and ~$80.

* **r10** `r10_0912_0302/20260912_043352_impl_llm_attempt1.md` (**$37.28, 217 turns, 216 tool calls** — corpus maximum), header summary: directed 27/27; `machine_vsetvl-0` and `machine_vfrsub_vf-3` fixed; *"其餘 21 個 FP 測試仍失敗，mismatch 統一表現為「scalar wdata 端顯示前個向量元素 FP 值」，疑為 Shuttle 3-wide retire trace 的 wdata 別名問題"* — i.e. it correctly characterised the uniform signature and correctly suspected the retire-trace path, then handed it forward in knowledge.
* **r11 a1** `r11_0912_0535/20260912_063916_impl_llm_attempt1.md:2749` — its change made things *worse*, and it said so: *"Change made things WORSE: vfadd_vf-2 diverges at 321 committed instructions vs 167"*.
* **r11 a2** `r11_0912_0535/20260912_075341_impl_llm_attempt2.md` ($28.41, 181 turns, 10 net lines): S2 24 → 23 fails, *"其余 22 个 `.vf` 测试仍失败 … 这一轮没能定位根因"* ("did not localise the root cause this round"), and an accurate budget report: *"Budget: 5/6 starts used: rv1, rv2 (rebuild=true), rv3 (failing list), rd4, rd5."* `r11_0912_0535/work/agent_budget_rtl.jobs.json` corroborates.

Net: **good instrumentation discipline, ~3 tests recovered for ~$80 across 3 iterations, root cause never found.** The `.vf` class was closed out of band, by a human-run A/B hypothesis experiment — **not** by the agent, and not by the file-level RTL bisect, which build-failed on 7 of its 8 arms and produced no localisation at all (§5.6c, §5.6d).

### 6.7 BEST — r13_0915_2118: one model iteration + one impl iteration, 34/34 both

Round 2 (`vqmmacc.vv`) landed in **$16.96 total and 152 turns**, against round 1's six iterations / $121 in r8 alone.

* **Model**: `r13_0915_2118/20260915_213502_model_llm_attempt1.md` — $4.56, 635 s, 52 turns, 51 tool calls. `20260915_212425_model_seed.json` 27 pass / 7 trap → `20260915_213549_model_attempt1.json` **34 pass**.
* **Impl**: `r13_0915_2118/20260915_220155_impl_llm_attempt1.md` — $12.41, 1,398 s, 100 turns, 99 tool calls. attempt-0 27 pass / 7 trap → `20260915_220429_impl_directed_attempt1.json` **34 pass**, S2 clean.

Four checkable reasons it went well:

1. **The seed prompt did the framing, not the agent.** `r13_0915_2118/20260915_212425_model_seed_note.md` opens *"You are extending a model that already works… Extend the code that is there -- do not re-derive it, do not restructure it"*, states the score up front (*"{'pass': 27, 'trap': 7}, 7 of 34 failing"*), says **which failures are its own** (*"Those failures are the new instruction's tests"*), and points at raw evidence rather than a summary: *"the full logs are on disk, read them yourself … Grep it before you guess; nothing in this message is a summary you cannot check."* Contrast r3–r7, where the agent had to discover the geometry, the tile layout and the test contract itself.
2. **It transcribed the spec instead of inferring from mismatch data.** Model agent: *"按照 SAIL 规范 `integrated-matrix-v0.9.0.adoc` 中 `vqmmacc.vv` 的形式化语义 … 内循环直接照抄 SAIL `int_gemm`/`int_block_dot`"* and explicitly *"**没有修改** 已经工作的 `vmmacc.vv`、`vmtl.v`、`vmts.v` 或 `vtype`/WARL 逻辑"*. The impl agent scoped itself to **three files** (`MatrixInstructions.scala`, `MatrixMultiplyPipe.scala`, `ExecuteSequencer.scala`) and stated its non-changes.
3. **It used inherited knowledge as a constraint.** From the impl finish summary: *"Kept `io.stall := valid` (valid-only) -- folding post-write-stall broke S2 in a prior round."* That is the §5.6 landmine being actively avoided — the `knowledge_rtl.md` mechanism finally paying for itself.
4. **It escalated verification correctly and cleaned up.** `r13_0915_2118/work/agent_budget_rtl.jobs.json` shows exactly **3 starts in the right order**: `rd1` (`failing` only → *"All 7 directed test(s) run (failing) are clean"*), then `rd2` (`all` → *"All 34 … are clean"*), then RVV. Narrow → full → regression, the exact opposite of r7 a3. And at line 2621 it ran `git status`, spotted the five-run-old stray, and removed it:
   ```
   rm -f /home/ray/chipyard/generators/saturn/src/main/scala/exu/int/MatrixMultiplyPipe.scala.bak
   ```
   closing with *"工作树：已清理干净，无 `.bak`、无调试代码、无未完成的更改"*. `r13_0915_2118/best_rtl.diff` contains **0** references to `.bak`, versus 2 in the inherited `rtl_resume.diff`. **r13 is where the r8-a3 leftover finally dies.**

### 6.8 Signal vs noise, quantified

* Across 30 impl attempts, **4 produced literally zero net diff change** (r6 a3, r7 a3, r8 a5, and all of r9) while costing **$82.60 and 412 turns**.
* **The highest-yield iterations were the smallest.** r8 a2: 8 lines → 1 test. r6 a2: 18 lines → 7 tests. r7 a4: 34 lines → 9 tests. r4 a4: 10 lines unblocked elaboration.
* **The most expensive iterations moved least.** The three priciest — r5 a1 $42.83, r10 a1 $37.28, r8 a5 $30.02 — delivered zero score change, 2 RVV tests, and zero net code change respectively.
* **r5 is the clearest write-off**: 5 iterations, **$113.84, 400 turns, 471 tool calls**, directed score `{mismatch:12, skip:15}` at *every* checkpoint from `impl_resume_directed_attempt0.json` to `impl_directed_attempt5.json`. Its own attempt-4 note names the trap — *"all of my iterations 1-3 haven't budged the failure pattern"*, *"the 12 failures are byte-identical AND cycle counts are identical too"* — and it kept going anyway. **Recognising a stuck loop and escalating is the capability that was missing**; the operator supplied it externally by re-seeding the reference for r7.
* **Diff hygiene degrades monotonically without an explicit cleanup step** — see the `.bak` leak in §6.4. r13 is the only iteration that ran `git status` for hygiene, and the only one that caught it.
## 7. Open items

### 7.1 Remaining instructions — 11 of 15

Spec source: `/share1/saves/max410011/hackathon/chia/examples/titan/specs/ime/instructions.json` — **15 entries**, cross-checked against `integrated-matrix-v0.9.0.adoc` §"Instructions (in alphabetical order)" (adoc lines 4619–7150). Scope authority: `constants.py:160` (`ROUND_ONE_INSNS`), `:173` (`ROUND_TWO_INSNS`), `:177` (`ALL_INSNS`), `:211` (`INSNS = _scope_insns()`).

`ime_encodings.py` does not hand-encode instructions — it data-drives generically off `instructions.json` for all 15 (`ime_encodings.py:34-38`), but only 4 have real `asm()` emission templates (`ime_encodings.py:161-178`; anything else raises `EncodingError` at `:177-178`).

| status | count | instructions | evidence |
|---|---|---|---|
| implemented + tested | **4** | `vmmacc.vv`, `vmtl.v`, `vmts.v`, `vqmmacc.vv` | emitted/decoded `ime_tests.py:222,228,239-240,243,255-256`; decode sequence asserted `ime_tests.py:889,893,903` |
| encoding-only self-check, never emitted | **2** | `vwmmacc.vv`, `v8wmmacc.vv` | `ime_encodings.py:269` — "the rest of the integer widening family, **reported but not implemented**"; funct6 spot-checked only, `:269-279` |
| not touched at all | **9** | `vmttl.v`, `vmtts.v` (transposing tile load/store); `vfmmacc.vv`, `vfwmmacc.vv`, `vfqmmacc.vv`, `vf8wmmacc.vv`, `vfwimmacc.vv`, `vfqimmacc.vv`, `vf8wimmacc.vv` (whole FP-matrix / microscaled-input family) | zero grep hits in `ime_tests.py` / `ime_stress.py` |

**11 remaining**, across three untouched sub-extension families: the rest of `Zvvmm` integer widening, the transposing tile-move family `Zvvmttls`, and the entire FP-matrix family `Zvvfmm` plus microscaled variants. `ime_stress.py:1-17` states outright "there are four instructions" — the stress harness has no structural path to the other 11.

### 7.2 PPA — no synthesis; an RTL-elaboration proxy exists as of 2026-09-17

**Confirmed: no synthesis tool, no PDK, and therefore no area/power/timing numbers in the physical sense exist in this deployment.** The evidence is documented in `/share1/saves/max410011/hackathon/titan_runs/ppa/README.md` §1, which records the feasibility checks:

| requirement | check | result |
|---|---|---|
| synthesis binary | `which genus yosys openroad innovus abc dc_shell` on head **and** inside `chia-chisel-build-aether-cosim:local` | all missing |
| Hammer | `python -c "import hammer"` both places | `ModuleNotFoundError` |
| Sky130 PDK | `find / -iname "*sky130*"` | only chipyard's example YAML templates; no `sky130A`, no libs |
| VLSI worker | `cluster.yaml`, node type `vlsi` | `num_workers: 0`, `image: # FILL — genus + hammer + cacti + sky130 PDK image` |
| PDK path constant | `constants.py:513` | `SKY130_COL_PATH = …("TITAN_SKY130_COL_PATH", "#FILL")`, env var unset |
| tool/tech configs | `examples/sky130_vlsi/tools-chia.yml`, `tech-sky130.yml` | unfilled templates (`genus_bin: "/path/to/cadence/GENUS"`) |
| docker images | `docker images` | no genus/sky130/vlsi image on the host |

The loop admits the gap in-line: `titan_loop.py:105` `PPA_SYNTH_WIRED = False`, with the comment at `:94-104` — deferred until "Stage 3 produces a design worth measuring… Left explicit rather than silently dead so the next reader knows it is a gap, not an oversight." `synth_node.py:34,74` (`run_shuttle_tile_synthesis`, Hammer + Cadence Genus) is never called from `nodes.py`. The README's conclusion: *"`synth_node.py`'s never having been wired into the loop is not an oversight in the driver — the cluster has no `vlsi` worker to dispatch to and nothing to run on it… **No PPA numbers were fabricated.**"* Had the flow existed, the constraint would have been `examples/sky130_vlsi/design.yml`: **10 ns period (100 MHz), 0.5 ns uncertainty, top `ShuttleTile`** (`constants.SYNTH_VLSI_TOP`).

**What does exist**, produced 2026-09-17 04:02–04:13 (after all 13 loop runs, and still being extended while this document was written): a **structural RTL-proxy A/B** by static parsing of the SystemVerilog the existing Chisel/firtool flow already emits. Driver `/share1/saves/max410011/hackathon/titan_runs/tools/ppa_rtl_ab.py` (Ray job `titan-ppa-rtl-ab-1`), analyser `tools/ppa_rtl_metrics.py`, raw SV in `ppa/rtl/{baseline,baseline_ref,ime}/`, canonical output `ppa/results.json`, method and provenance in `ppa/README.md`. (`ppa/metrics.json` is the earlier two-config version, superseded by `results.json`.)

Three configs were elaborated through the identical `nodes.build_saturn` path with identical flags, each from a fresh `reset_chipyard`:

| label | config | tree | build wall | generated SV |
|---|---|---|---|---|
| `baseline` | `GENV256D128ShuttleConfig` | pristine | 163.7 s | 667 files, 21,109,111 B |
| **`baseline_ref`** | **`REFV256D128ShuttleConfig`** | pristine | 164.6 s | 669 files, 21,111,047 B |
| `ime` | `TitanV256D128ShuttleConfig` | pristine + `ppa/r13_round2_rtl_only.diff` | 160.0 s | 671 files, 21,292,329 B |

`r13_round2_rtl_only.diff` is the RTL half of `r13_round2.diff` — the 15 `generators/**` files (saturn 12, rocket-chip 2, shuttle 1), Spike files removed since the model is not elaborated: **+844 / −45 lines of Chisel**.

**The primary comparison is `baseline_ref` vs `ime`**, not `baseline` vs `ime` — `results.json` names it `PRIMARY_baseline_ref_vs_ime` and explains why: both are `WithShuttleVectorUnit(256, 128, VectorParams.refParams)` + `WithSystemBusWidth(128)` + `WithShuttleTileBeatBytes(16)` + `WithNShuttleCores(1)` + `AbstractConfig`, i.e. **parameter-matched**, so the delta isolates the IME RTL hunks and nothing else. The `GEN…` baseline uses different vector params; comparing against it produces spurious **negative** flop-bit deltas (−653 whole-design, −1,914 on `ShuttleTile`) which are an artifact of the parameter mismatch, not a smaller design. **Quote the `baseline_ref` column.**

Whole-design, `baseline_ref` → `ime`:

| metric | baseline_ref | ime | delta | % |
|---|---|---|---|---|
| `.sv` files | 668 | 670 | +2 | +0.30% |
| total SV lines | 185,097 | 186,236 | +1,139 | +0.62% |
| modules | 681 | 683 | +2 | +0.29% |
| **flop bits** (Σ unique modules) | 415,684 | 416,535 | **+851** | **+0.21%** |
| **multiply sites** | 20 | 22 | **+2** | **+10.0%** |
| add sites | 1,828 | 1,859 | +31 | +1.70% |
| SRAM macros / bits | 8 / 1,533,824 | 8 / 1,533,824 | 0 / 0 | 0% |

Hierarchy rollups (with instance multiplicity), `baseline_ref` → `ime`:

| subtree | metric | baseline_ref | ime | delta | % |
|---|---|---|---|---|---|
| `ShuttleTile` (intended synth top) | flop bits | 71,719 | 72,570 | +851 | **+1.19%** |
| | multiply sites | 39 | 42 | +3 | +7.69% |
| | add sites | 2,254 | 2,290 | +36 | +1.60% |
| | lines | 87,769 | 88,842 | +1,073 | +1.22% |
| `SaturnShuttleUnit` (the vector unit) | flop bits | 28,952 | 29,803 | +851 | **+2.94%** |
| | multiply sites | 33 | 36 | +3 | +9.09% |
| | add sites | 1,730 | 1,766 | +36 | +2.08% |
| `VectorBackend` | flop bits | 16,368 | 17,083 | +715 | **+4.37%** |
| | multiply sites | 24 | 25 | +1 | +4.17% |
| `VectorMemUnit` | flop bits | 7,935 | 8,071 | +136 | +1.71% |
| | multiply sites | 7 | 9 | +2 | +28.6% |
| `MatrixMultiplyPipe` | — | **absent** | flop bits **558**, mult sites 1, add sites 4, 173 lines, 19 reg decls | new module | — |

Read as a one-line summary: **the entire IME extension costs +851 flop bits (+1.19% of `ShuttleTile`), +2 multiply sites and +31 add sites, with zero change to SRAM.** All 851 added flop bits sit inside `SaturnShuttleUnit` — the tile's non-vector logic is untouched — and 558 of them are the new `MatrixMultiplyPipe` itself.

Module-level delta (from the earlier two-config `metrics.json`; the module sets are the same): new in IME — `MatrixMultiplyPipe`, `Arbiter2_VectorWrite`, `Arbiter8_ScalarWrite`, `DCEQueue_{4,5,9}`, `IssueQueue_3`; removed — `Arbiter7_ScalarWrite`, `DCEQueue_{8,12}`; 28 changed, including `AddrGen`, `CSRFile`, `EarlyVectorDecode`, `ExecuteSequencer`, `LoadOrderBuffer`, `PipelinedFaultCheck`, `ShuttleCore`, `VectorBackend`.

**Do not quote these as area.** `ppa/README.md` §5 and `results.json`'s own `IMPORTANT` field are explicit: register bits do not map linearly to cell area; `*` counts are *multiply sites*, not multiplier area (one `*` in Verilog can be any width); **there is no timing or power number at all — WNS/Fmax/power are absent, not zero**; and un-synthesised RTL contains logic a synthesiser would delete while missing sharing it would find. Treat the deltas as *an upper-bound sketch of where the design grew*. What the A/B is genuinely good for: both builds used the same tree, tools, flags and container, and neither includes the cosim harness, so the comparison is internally consistent even though the absolute numbers are not physical.

Toolchain provenance, from `results.json` `provenance` (worth quoting in the report because it pins the whole project): host gpuserv4 (<HEAD_IP>); chipyard `1.14.0-19-g4ab72313`; saturn `dfe75de`, rocket-chip `70430823f`, shuttle `622f08b`; Chisel 6.7.0 / Scala 2.13 / sbt 1.8.2 / OpenJDK 20.0.2; **CIRCT firtool-1.75.0 (LLVM 19.0.0git)**; Verilator 5.022.

**Open**: real PPA needs a licensed Cadence Genus install, the `sky130_col` collateral tree, and a container image carrying both. `synth_node.py` needs no fix; it needs a worker.

### 7.3 Image pinning

**Confirmed** (`/share1/saves/max410011/hackathon/chia/examples/titan/cluster.yaml`): every active image uses a **mutable tag, no digest anywhere** (zero `sha256`/`digest` hits):

| image | line | `pull_before_run` |
|---|---|---|
| `ghcr.io/ucb-bar/chia-claude-code:latest` | L61 | False (L63) |
| `chia-chisel-build-aether-cosim:local` | L88 | False (L90) |
| `ghcr.io/ucb-bar/chia-verilator-run:latest` | L110 | False (L112) |
| `ghcr.io/ucb-bar/chia-riscv-cross:latest` | L131 | False (L133) |

`pull_before_run: False` mitigates accidental re-pull drift but **not** a rebuild or retag of the same tag on the host. The `py_modules` shadowing documented at `constants.py:1-19` ships the head's *current* `chia` package over whatever is baked into the worker image — but only the `chia` package: the Claude Code CLI, Verilator, riscv-gnu-toolchain and the chipyard/Saturn toolchain in those images stay silently mutable, with no version log.

**The r10_0912_0254 credential incident.** `ledger_notes.json`: `"憑證失效，啟動即空跑，停"` (credentials invalid, ran empty at startup, stopped); the run dir has a 0-byte `trace/ChiaProfileCollector.log` and no LLM artifacts. Diffing `cluster.yaml.bak-r10` against the current file shows exactly **one** changed line: the `llm` node's `/home/ray/.claude` mount reverted from a static snapshot `/share1/saves/max410011/titan_claude_home` (in `.bak-r8b`/`.bak-r10`) back to the live `${HOME}/.claude` (in `.bak-r7`/`.bak-r8` and current); the file timestamps place that edit between the failed run's teardown (03:00) and the fixed retry `r10_0912_0302`'s start (03:02). The causal chain (stale token snapshot → live-refreshed host dir) is **inferred, not stated in any artifact**.

**Correction to the ledger's blame allocation** (§3.4): the same silent-no-op failure mode was **already active in r9_0912_0137**, one run earlier — four zero-cost `prompt` calls, 15-byte transcripts, byte-identical diffs. The ledger attributes r9's non-progress to agent behaviour instead. Whatever the root cause, it went undetected for a full run because the loop records a failed LLM call as an ordinary zero-cost `complete` and the `llm_failed` event type is never emitted.

**Unresolved risk today**: no image digest pinning; no institutionalised comment or guard against reintroducing the static-credential-snapshot pattern (unlike the `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS` r8-iter3 note at `cluster.yaml` L54-59, which *is* documented in-line); and no loop-side detector for a zero-token, zero-cost, one-turn "iteration".

### 7.4 The `machine_vsetvl-0` / `machine_vsetivli-0` exclusions

Mechanism: `titan_loop.py:601-607` reads `REGRESSION_BASELINE_PATH` (`constants.py:356`) and filters `suite = [(n,e) for n,e in suite if n not in known_bad]`. The file is `/share1/saves/max410011/hackathon/chia/examples/titan/rvv_baseline_failures.json`:

```json
["machine_vfcvt_f_x_v-0", "machine_vsetivli-0", "machine_vsetvl-0"]
```

**The code comment overstates the case and the README gets it right.** `constants.py:301-302` describes these as tests that "ALREADY fail on a pristine Saturn". That is true only of the first: the raw baseline run `/share1/saves/max410011/hackathon/titan_runs/rvv_baseline/20260910_190750_rvv_baseline_summary.json` shows the pristine tree failing **only** `machine_vfcvt_f_x_v-0` (`n_failing: 1`). `machine_vsetvl-0` and `machine_vsetivli-0` **pass** on a pristine tree. Trust the baseline JSON over the constants comment.

The exclusion is nonetheless **principled**, for a different and stronger reason, stated in `rvv_baseline_failures.README.md`:

* `machine_vfcvt_f_x_v-0` — pristine Saturn already differs from stock Spike by 1 ulp (measured 9/10 in `rvv_baseline`). A genuine pre-existing baseline failure.
* `machine_vsetvl-0` — IME v0.9.0 requires an IME-legal `vsetvli` to write a **non-zero λ into `vtype[62:60]`** (preserve-or-initialize). Stock Spike does not model the IME field and always returns 0, so **this test cannot pass against stock Spike for any spec-correct IME implementation**. The semantics of the IME `vtype` fields are instead judged by **S3, lockstep against the IME Spike model** (decision recorded 9/12).
* `machine_vsetivli-0` (added 9/14) — same class. After the A3 fix this was the only remaining failure in the 150-sample S2; the divergence is `spike x14 = 0x5` vs `DUT x14 = 0x3000000000000005`, i.e. exactly `vtype[62:60] = 0b011`, the maximum legal λ for VLEN=256/SEW=8. Any `vsetvl`-family test that writes `vtype` back to a scalar register and compares it will behave this way.

So the principle is: **a test that a correct implementation cannot pass against this judge is moved to a judge that can adjudicate it, not deleted.** `titan_runs/r12_seed/knowledge_rtl.md:1254` confirms the exclusion was made specifically to stop the agent "fixing" the symptom by breaking spec compliance — an earlier agent change that forced λ=0 was discarded.

**Gap found by this pass**: the vector-test suite contains a fourth test of the same family, **`machine_vsetvli-0`** (riscv-vector-tests `Makefrag:1078-1081`), which `titan_runs/a3_verify/results.json` (`s2_full`) shows failing with the **identical divergence signature** — yet it is **not** in `rvv_baseline_failures.json`. Whether that is a live gap or simply not sampled since is not resolved by the artifacts. The S2 150-sample would only surface it on a run where the stride happened to select it.

### 7.5 The S2/S3 cospike header-ABI issue

| claim | status | evidence |
|---|---|---|
| The S2 judge linked an **IME-modified `libriscv.so`** against stock cospike headers | **confirmed** | `titan_loop.py:216-232` docstring; digest mismatch demonstrated in `/share1/saves/max410011/hackathon/titan_runs/s2_judge/inspect.txt` (installed `aef5a8df…` vs rebuilt-from-source stock) |
| That ABI contamination caused r10's `.vf` / vsetvl "regressions" | **suspected → experimentally refuted** | the A/B/C experiment (`tools/s2_judge_experiment.py`, `tools/s2_judge_c.py`): stock and IME-contaminated Spike fail the *same* 9/10 tests identically, and pristine RTL passes 10/10 under either. `r11_0912_0535/work/knowledge_rtl.md:1255`: "the `.vf` failures are caused by your RTL diff, not by the judge." |
| Fix: the judge always relinks a freshly built **stock** Spike | **confirmed, applied before r11** | `tools/s2_stock_spike_patch.py` patches `titan_loop.py:1521-1536,431-435,713-716`; `r11_0912_0535/trace/ChiaProfileCollector.log` line 8 logs `stock_spike {digest: 6d78935237dd}`; the same digest recurs in r12 and both r13 traces. `ledger.md` r11: "stock Spike 裁判、排除 vsetvl-0" |
| Real root cause of the `.vf` regression | **confirmed, fixed in r12, unrelated to Spike** | §5.6 |

Net: the ABI hardening is real and permanent judge hygiene, but it was a **red herring** for the bug it was diagnosed alongside. What remains latent is that the header/ABI coupling is mitigated by always rebuilding stock Spike rather than by a version check — nothing detects a future header drift; it would resurface as unexplained cosim divergence.

### 7.6 Other explicitly-noted limitations in the code

| file:line | note |
|---|---|
| `chia/examples/titan/db_node.py:93` | test-ELF staging under `DB_ROOT/tests/<set>/` is hand-staged from the chipyard image's prebuilt binaries, not a build-from-source node |
| `chia/examples/titan/ime_encodings.py:269` | `vwmmacc.vv` / `v8wmmacc.vv` "reported but not implemented" |
| `chia/examples/titan/ime_tests.py:629` | for EMUL_C=16 tile geometries there is no single-instruction accumulator move (LMUL=16 is an illegal vtype) — explicitly *"Out of scope for round one"* |
| `chia/examples/titan/nodes.py:84` | `apply_diff` diff-seeding: *"TODO: this should be offered with bypassing."* |

`prompts/*.md` contain **zero** TODO/FIXME/limitation markers.

### 7.7 Summary of what is not done

1. **11 of 15 instructions** (§7.1) — the headline functional gap.
2. **No physical PPA** (§7.2); only a structural RTL proxy, and it needs a licensed toolchain to become real.
3. **No image digest pinning**, and no detector for a silent zero-token iteration (§7.3).
4. **`machine_vsetvli-0` possibly missing from the exclusion list** (§7.4) — the one unresolved correctness question in the judge configuration.
5. **Header/ABI drift is mitigated, not detected** (§7.5).
6. **S2 runs a 150-test stride sample, not the full suite** (§1.4, §2.1) in every loop run; the only near-full sweep is the offline `a3_verify` 839-test run.
7. **S2 is judged on a 1-wide retire host** (`WithShuttleRetireWidth(1)`, `constants.py:89-114`): a bug appearing only when two instructions commit in the same cycle is outside that gate. S1 still runs dual-issue. The principled fix is upstream in Shuttle.
8. **S2 itself is nondeterministic** (§3.4): identical designs produced 21/21/23 failures with a recurring `exit -11` simulator death. Nothing in the loop accounts for judge flakiness.
