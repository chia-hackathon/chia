# AETHER hardware sweep: the memory system is the decode bottleneck

**Outer-loop design-point experiment, measured — not projected.**
Four hardware design points, one fixed kernel source, one RTL simulator each.

Everything below is a Verilator run of the *same* probe kernel
`out/loop/20260912-191830-e713/kernel_02.h` (Round 7, iteration 2) on
`llama-q8-gemv-gemmini-n1` (1x2048x512 int8 GEMV, B = 2048*512 = 1,048,576 B).
That kernel measures the GEMV three times in one run: warm (B resident in L2),
a 1.5 MiB DRAM flush, then the identical GEMV **cold**.
All four runs self-checked `PASSED (8 sampled outputs)`.

The baseline run **reproduces Round 7 exactly** (cold 184,837 / warm 132,435),
so the three new points are directly comparable.

---

## 1. The design points

| config | mbus | L2 MSHRs | elaborate |
|---|---|---|---|
| `GENV256D128GemminiShuttleConfig` (control) | 8 B/cyc (`DATA_BITS(64)`) | 12 | (pre-existing) |
| `GENV256D128GemminiShuttleWideMbusConfig` | **16 B/cyc** (`DATA_BITS(128)`) | 12 | rc=0, 110 s |
| `GENV256D128GemminiShuttleDeepMshrConfig` | 8 B/cyc | **24** | rc=0, 50 s |
| `GENV256D128GemminiShuttleWideDeepConfig` | **16 B/cyc** | **24** | rc=0, 51 s |

Verified from the elaborated artefacts, not from the source:

* `sifive,mshr-count` in the generated `.dts`: 12 / 12 / **24** / **24**
  (`out/hw-sweep/elaborate.*.log`, e.g. `elaborate.GENV256D128GemminiShuttleDeepMshrConfig.log:128`).
* `SimDRAM` instantiation `.DATA_BITS()` in `gen-collateral/TestHarness.sv:174`:
  64 / **128** / 64 / **128** — i.e. the AXI4 memory port and the DRAM model
  really did follow the memory bus.
* `elaborate.summary` — all three new points elaborate `rc=0`.

## 2. How the fragments are written

Appended to `generators/gemmini/chipyard/GemminiConfigs.scala` (package
`chipyard`) by `docker/add_coexist_config.py --sweep`; the literal text lives in
`docker/add_coexist_config.py:64-120` (`SWEEP_CONFIG_TEXT`) and a copy is in
`out/hw-sweep/sweep_configs.scala.txt`.

### Wider memory bus

```scala
class WithAetherMemoryBusWidth(bits: Int) extends Config((site, here, up) => {
  case freechips.rocketchip.subsystem.MemoryBusKey =>
    up(freechips.rocketchip.subsystem.MemoryBusKey).copy(beatBytes = bits / 8)
})
```

* `MemoryBusKey` defaults to `MemoryBusParams(beatBytes = 8)` and *nothing* in
  the AETHER chain overrode it — `rocket-chip/src/main/scala/subsystem/Configs.scala:50`.
  `WithSystemBusWidth(128)` only touches `SystemBusKey`
  (`chipyard/src/main/scala/config/fragments/SubsystemFragments.scala:17-19`).
* No other knob has to move: `WithDefaultMemPort` sets the AXI4 master's
  `beatBytes = site(MemoryBusKey).beatBytes`
  (`rocket-chip/.../subsystem/Configs.scala:267-273`), and chipyard's
  harness binder hands that width straight to testchipip's `SimDRAM`
  (`DATA_BITS` param, `testchipip/.../dram/SimDRAM.scala:13`), whose
  `mm_magic_t` backend is width-generic. `WithNMemoryChannels(1)` stays as-is —
  channels and beat width are independent.
* `WithEdgeDataBits(n)` (`Configs.scala:217-220`) would also work but drags
  `ExtIn` along with it, so a dedicated fragment is cleaner.

### Deeper L2 MSHR file

```scala
class WithAetherL2MemCycles(cycles: Int) extends Config((site, here, up) => {
  case freechips.rocketchip.subsystem.InclusiveCacheKey =>
    up(freechips.rocketchip.subsystem.InclusiveCacheKey).copy(memCycles = cycles)
})
```

`chipyard`'s `WithInclusiveCache` exposes the MSHR count only *indirectly*, as
`outerLatencyCycles` -> `InclusiveCacheParams.memCycles`
(`rocket-chip-inclusive-cache/design/craft/inclusivecache/src/Configs.scala:50-68`).
The count itself is derived:

```
all_mshrs = 2 + max(if (dirReg) 3 else 2, ceil(memCycles / blockBeats))
blockBeats = cache.blockBytes / cache.beatBytes = 64 / sbus.beatBytes = 64/16 = 4
```
(`.../inclusivecache/src/Parameters.scala:295-302`, `:46`.)
So `memCycles = 40` -> `2 + 10 = 12` (the control's `sifive,mshr-count = <12>`)
and `memCycles = 88` -> `2 + 22 = **24**`. Overriding `InclusiveCacheKey` in
place is preferable to re-running `new WithInclusiveCache(outerLatencyCycles=88)`,
which would also re-derive `sets` and re-install the coherence manager.

Both fragments must sit **left of** `chipyard.config.AbstractConfig` (which is
where `WithInclusiveCache` and the default `MemoryBusKey` come from).

## 3. Measured: `llama-q8-gemv-gemmini-n1`, 1 MiB of B

Rate = 1,048,576 B / cycles. Source: `out/hw-sweep/simlog.llama-q8-gemv-gemmini-n1.<CONFIG>.txt`.

### Cold (B flushed out of L2 — the decode-relevant case)

| design point | mbus | MSHR | cold cycles | B/cycle | speedup |
|---|---|---|---|---|---|
| control | 8 | 12 | 184,837 | 5.67 | 1.000x |
| DeepMshr | 8 | 24 | 173,633 | 6.04 | **1.065x** |
| WideMbus | 16 | 12 | 117,526 | 8.92 | **1.573x** |
| WideDeep | 16 | 24 | 115,865 | 9.05 | **1.595x** |

### Warm (B resident in the 512 KB L2)

| design point | warm cycles | B/cycle | speedup |
|---|---|---|---|
| control | 132,435 | 7.92 | 1.000x |
| DeepMshr | 129,845 | 8.08 | 1.020x |
| WideMbus | 99,507 | 10.54 | 1.331x |
| WideDeep | 99,311 | 10.56 | 1.334x |

### The 1.5 MiB DRAM-flush probe (a pure streaming-bandwidth meter)

| design point | flush cycles | relative |
|---|---|---|
| control | 274,464 | 1.000x |
| DeepMshr | 259,048 | 1.060x |
| WideMbus | 175,411 | 1.565x |
| WideDeep | 167,211 | 1.641x |


## 3b. Measured: `llama-q8-gemv-gemmini-lmhead`, 4 MiB of B

Second kernel, different cold characteristic: B is 2048x2048 = 4,194,304 B,
8x the 512 KB L2, so *every* run is a cold stream — no probe instrumentation
needed. Source: the best-known lmhead kernel,
`out/loop/20260909-200541-dae1/kernel_04.h` (634,507 cycles on the control,
which this sweep **reproduces to the cycle**). Logs:
`out/hw-sweep/simlog.llama-q8-gemv-gemmini-lmhead.<CONFIG>.txt`, all `PASSED`.

| design point | mbus | MSHR | cycles | B/cycle | speedup |
|---|---|---|---|---|---|
| control | 8 | 12 | 634,507 | 6.61 | 1.000x |
| DeepMshr | 8 | 24 | 634,580 | 6.61 | **0.9999x** |
| WideMbus | 16 | 12 | 472,616 | 8.87 | **1.343x** |
| WideDeep | 16 | 24 | 472,093 | 8.88 | **1.344x** |

This is the cleaner of the two experiments and it is unambiguous: **doubling
the L2 MSHR file changes a 4 MiB cold weight stream by 73 cycles out of
634,507 (0.01%) — nothing.** Doubling the memory bus buys 1.34x. The two
kernels also converge on the same post-widening ceiling (8.87-9.05 B/cycle),
which is further evidence that at 16 B/cycle the binding constraint is no
longer the bus.

## 4. The prediction was wrong — and wrong in the informative direction

The standing hypothesis (from the in-flight-request scan: 6/8/10/12/16
in-flight -> 6.31/6.63/6.06/5.83/5.78 B/cyc, best at 8 < MSHR=12) was:

> the limit is the MSHR count x latency bandwidth-delay product, not the mbus
> width; **widening the mbus alone should do almost nothing**, deepening the
> MSHRs alone should reach ~8 B/cyc, and only both together should give ~2x.

Measured, the ordering is **exactly inverted**:

* **Widening the mbus alone: +57% cold (5.67 -> 8.92 B/cyc)** — it captures
  98.6% of the best point's gain on its own.
* **Deepening the MSHRs alone: +6.5% on n1 (5.67 -> 6.04 B/cyc) and +0.01% on
  lmhead** — nowhere near 8, and on the bigger cold stream it is exactly zero.
* **Both together: +59.5% (9.05 B/cyc)** — *not* 2x, and only 1.4% better than
  widening alone.

The mechanism is visible in the control's own numbers: the warm rate was
7.92 B/cyc against an 8 B/cyc memory bus — **99.0% of the mbus roofline**. The
L2->DRAM link was already saturated, so there was no MSHR-shaped headroom to
recover; more MSHRs only buy the ~6% that queueing behind a full bus costs.
Conversely, once the bus is 16 B/cyc, 12 MSHRs are still enough to keep it
~66% busy, and going to 24 adds only 1.4%. The in-flight scan that motivated
the MSHR hypothesis was measuring congestion *at the 8 B/cyc bus*, not an MSHR
shortage.

**The correct statement for the paper: decode on this SoC is bounded by the
memory-bus *width*, and the L2 MSHR file is not the binding constraint at
either width.** The one caveat is that the new ceiling is not itself the bus:
16 B/cyc buys only 9.05 B/cyc cold (57% of the bus), so at 16 B the binding
constraint has moved elsewhere (Gemmini's StreamReader / L2 occupancy); a
32 B/cyc point would likely show much less than another 1.57x.

## 5. End-to-end decode projection (Llama-3.2-1B, S=512, cold)

`loop/llama_project.py --scenario decode --S 512 --gemv-bytes-per-cycle G
--lmhead-bytes-per-cycle L --clock-ghz F` (decode is 99.0% Gemmini and
1.25 GB read/token, so tok/s is essentially linear in the streaming rate):

| design point | n1 cold B/cyc | lmhead B/cyc | Mcyc/token | tok/s @500 MHz | tok/s @1 GHz |
|---|---|---|---|---|---|
| control | 5.67 | 6.61 | 218.6 | 2.29 | 4.575 |
| DeepMshr | 6.04 | 6.61 | 208.1 | 2.40 | 4.81 |
| **WideMbus** | **8.92** | **8.87** | **145.9** | **3.43** | **6.854** |
| WideDeep (both knobs) | 9.05 | 8.88 | 144.3 | 3.47 | 6.93 |

(`--gemv-bytes-per-cycle` from the n1 cold probe, `--lmhead-bytes-per-cycle`
from the lmhead run above, per design point. Reproduce the headline row:
`python loop/llama_project.py --scenario decode --S 512
--gemv-bytes-per-cycle 8.92 --lmhead-bytes-per-cycle 8.87
--mem-bytes-per-cycle 8` -> 6.854 tok/s @1 GHz; the control uses
`--gemv-bytes-per-cycle 5.67 --lmhead-bytes-per-cycle 6.61` -> 4.575.)

**Headline: widening the memory bus alone is the story.** WideMbus alone
takes the end-to-end decode projection from 2.29 to 3.43 tok/s at 500 MHz
(4.575 to 6.854 tok/s at 1 GHz) — a **1.50x** speedup — with L2 MSHRs left
at the control's 12. Isolating each knob's share of the *combined*
(WideDeep) saving over control: the bus alone recovers **97.6%** of the
saving on n1 and **99.7%** on lm_head; doubling MSHRs alone recovers only
**16.2%** on n1 and **-0.04%** (i.e. slightly worse) on lm_head. The two
knobs are not additive — combining them (WideDeep) buys only a sliver more
than the bus alone. Doubling the memory bus alone (WideMbus, both kernels)
is therefore the correct headline; the extra MSHRs are not worth their
area on their own.

For completeness, the **both-knobs** design point (WideDeep: wider bus +
deeper MSHRs together) reaches a **1.51x** end-to-end speedup (218.6 ->
144.3 Mcycles/token; 3.47 tok/s @500 MHz, 6.93 tok/s @1 GHz; 1.57x on the
n1 GEMV alone, 1.34x on the lm-head) — only marginally better than
widening the bus alone.

**Caveat:** the DRAM backend in these RTL sims (`testchipip`'s `SimDRAM`,
`mm_magic_t`) is untimed/width-generic behavioral memory, not a timed DRAM
model with row/bank/refresh contention — see the paper for the resulting
scope limits on this projection.

## 6. Reproduce

```
# 1. inject the three configs (idempotent)
python3 docker/add_coexist_config.py \
    /home/ray/chipyard/generators/gemmini/chipyard/GemminiConfigs.scala --sweep
# 2. elaborate (container aether-coexist; logs land in out/coexist/hw-sweep)
docker exec -d aether-coexist bash /work-out/hw_sweep_elab_only.sh
# 3. measure (Ray: chisel build -> riscv build -> verilator run, no LLM)
python3 out/hw-sweep/measure.py llama-q8-gemv-gemmini-n1 \
    $PWD/out/loop/20260912-191830-e713/kernel_02.h <CONFIG>
```

...and the same with `llama-q8-gemv-gemmini-lmhead` +
`out/loop/20260909-200541-dae1/kernel_04.h`.

Artefacts in this directory: `elaborate.*.log`, `elaborate.summary`,
`simlog.llama-q8-gemv-gemmini-{n1,lmhead}.*.txt`, `measure.n1*.log`,
`measure.lmhead.*.log`, `measure.py`, `sweep_configs.scala.txt`.

Caveat on timings: each measurement rebuilt its own simulator
(`nodes.build_simulator`, 150-600 s, serialized on the single `chipyard` Ray
resource); each Verilator run took 20-40 min. A Ray raylet warned that its
session directory was >95% full (5.4 GB free of 876 GB) throughout; no task
failed, but that node is close to needing a cleanup.
