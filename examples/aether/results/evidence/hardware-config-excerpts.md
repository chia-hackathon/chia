# Hardware configuration — excerpts from the elaboration/build logs

These excerpts are pulled read-only from the (unversioned, multi-thousand-line)
Chisel elaboration and build logs under `aether/out/coexist/` — those full logs
are **not** copied into this package (build noise, large, not reproducible
without the exact container image); only the lines relevant to bus width, L2
MSHR count, and address mapping are extracted here, each labeled with its
original source path and line number(s) in the `aether` working tree this
package was cut from.

Config under elaboration: `GENV256D128GemminiShuttleConfig` (see
`../paper/methodology.md` §4 for the full Scala config chain).

## 1. Clock domains — all elaborate at 500 MHz, not the 1 GHz assumed by the cost model

Source: `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:216-220`

```
Clock sbus_0: using diplomatically specified frequency of 500.0.
Clock pbus_0: using diplomatically specified frequency of 500.0.
Clock fbus_0: using diplomatically specified frequency of 500.0.
Clock mbus_0: using diplomatically specified frequency of 500.0.
Clock cbus_0: using diplomatically specified frequency of 500.0.
```

Every bus relevant to the memory path (system bus, peripheral bus, front bus,
**memory bus**, control bus) is elaborated at 500 MHz. Verilator itself counts
RTL clock edges (cycle-accurate, clock-frequency-agnostic), so this does not
affect any raw cycle count — but every cycles-to-tok/s conversion in
`results/loop/FINAL_REPORT.md` and `results/projections/projection_final*.md`
that assumes 1.00 GHz is a **2x-optimistic** wall-clock/throughput figure
relative to what this SoC's own elaborated clock tree reports.

## 2. L2 MSHR count = 12

Source: `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:299`
(inside the devicetree `cache-controller@2010000` node, lines 290-300):

```
			cache-block-size = <64>;
			cache-level = <2>;
			cache-sets = <1024>;
			cache-size = <524288>;
			cache-unified;
			compatible = "sifive,inclusivecache0", "cache";
			next-level-cache = <&L20 &L23>;
			reg = <0x2010000 0x1000>;
			reg-names = "control";
			sifive,mshr-count = <12>;
```

This is the RTL-elaborated confirmation that the shared L2 (`InclusiveCache`,
512 KiB, 1024 sets, 64 B lines) has exactly **12 MSHRs**. This is the L2
resource the loop's kernel authors identified as the actual DMA-throughput
bottleneck for large weight streams (`results/loop/FINAL_REPORT.md` §3(2)/(3)),
not DRAM bandwidth or a DRAMSim2 timing effect — see `../paper/methodology.md`
§6(b) for why no timing-accurate DRAM model was even in the loop (the
simulator runs the untimed `mm_magic_t` DRAM abstraction).

## 3. L2 client map (who shares those 12 MSHRs)

Source: `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:180-186`
(also reproduced verbatim as its own file in the original run at
`out/paper-submission/03-evidence/hwconfig/elaborate-l2-clientmap.txt`):

```
L2 InclusiveCache Client Map:
	0 <= debug
	1 <= serial_tl_0_0
	2 <= serial_tl_0_1
	3 <= serial_tl_0_2
	4 <= serial_tl_0_3
	5 <= stream-reader
```

`stream-reader` (client 5) is Gemmini's own DMA/`StreamReader` port into the
L2 — the path every `mvin`/weight-stream byte takes.

## 4. Generated address map

Source: `out/coexist/elaborate.GENV256D128GemminiShuttleConfig.log:382-393`:

```
Generated Address Map
	       0 -     1000 ARWX  debug-controller@0
	    1000 -     2000 ARW   boot-address-reg@1000
	    3000 -     4000 ARWX  error-device@3000
	   10000 -    20000  R X  rom@10000
	  100000 -   101000 ARW   clock-gater@100000
	  110000 -   111000 ARW   tile-reset-setter@110000
	 2000000 -  2010000 ARW   clint@2000000
	 2010000 -  2011000 ARW   cache-controller@2010000
	 8000000 -  8010000 ARWXC memory@8000000
	 c000000 - 10000000 ARW   interrupt-controller@c000000
	10020000 - 10021000 ARW   serial@10020000
	80000000 - 90000000 ARWXC memory@80000000
```

Main DRAM-backed memory region is `0x80000000`-`0x90000000` (256 MiB,
`ARWXC` — cacheable), matching the devicetree `memory@80000000` node
(`reg = <0x80000000 0x10000000>`, same log file, line ~278). A small
16 KiB scratch memory region also exists at `0x8000000` (`memory@8000000`,
disabled by default). `cache-controller@2010000` is the L2 InclusiveCache's
own control/MSHR-count MMIO window referenced in §2 above.

## 5. mbus (DRAM-facing bus) width — 8 B/cycle, not the 16 B/cycle sbus width

This specific number (`MemoryBusKey => MemoryBusParams(beatBytes = 8)`) is
**not** a line the elaboration log prints directly — it is a rocket-chip/
chipyard default that this repository's checkout cannot independently
re-derive from Chisel source (no `rocket-chip`/`chipyard` source tree is
present under `aether/repos/`; chipyard lives on a separate container image).
It is documented, with the same caveat, in `loop/kernels.py:106-112` and
`loop/kernels.py:1161-1174`, and in `../paper/methodology.md` §4. What *is*
independently corroborated from measurement is that the achieved cold-DRAM
weight-stream throughput never exceeds ~7.9-7.94 B/cycle and converges toward
this 8 B/cycle ceiling from below (see `../paper/methodology.md` §6(a) and
`results/loop/FINAL_REPORT.md` §3(2)) — consistent with, but not direct RTL
proof of, an 8 B/cycle mbus. The elaborated **sbus** width and Shuttle tile
beat bytes, by contrast, are directly visible in the Scala config chain
(`docker/add_coexist_config.py:41-47`, reproduced in
`../paper/methodology.md` §4): `WithSystemBusWidth(128)` (128 bit = 16 B/cycle)
and `WithShuttleTileBeatBytes(16)`.

## Sources not reproduced here

The full `out/coexist/*.log` files (elaboration logs ~1,500-3,300 lines each,
build logs ~1,000-2,000 lines each, plus `cosim_*.log`/`.err.gz` cosimulation
traces) are build/tooling noise around the four excerpts above and were not
copied into this package. They exist, read-only, in the original
`aether/out/coexist/` directory this snapshot was cut from, if line-by-line
re-verification is needed.
