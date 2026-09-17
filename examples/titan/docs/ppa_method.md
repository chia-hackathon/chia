# Titan IME vs. Saturn baseline — structural RTL comparison

## TL;DR — this is NOT a PPA run

**No synthesis tool and no PDK are installed in this deployment.** Nothing in
`titan_runs/ppa/` is an area, timing or power number in the physical sense:
there is no cell area, no worst negative slack, no achieved frequency and no
power figure here, because none was measured. What is reported instead is a
**structural RTL-proxy comparison** derived by static parsing of the
SystemVerilog the existing Chisel/firtool flow already emits. Every number is
accompanied by the exact rule used to obtain it (§4), so it can be reproduced
or challenged.

## 1. Feasibility: evidence that no real flow exists

The Titan example *does* ship a synthesis node — `examples/titan/synth_node.py`
(`run_shuttle_tile_synthesis`, resources `{"VLSI": 1, "Syn": 0}`), which wraps
`examples/sky130_vlsi/hammer_syn_node.py` (`Sky130SynNode`) around
`chia/chia/vlsi/hammer.py` (`HammerNode`: one `hammer-vlsi` CLI call per
action). The intended flow is therefore **Hammer + Cadence Genus on SkyWater
Sky130**, top module `ShuttleTile`. None of it is installed:

| Requirement | Check performed | Result |
|---|---|---|
| Synthesis binary | `which genus yosys openroad innovus abc dc_shell` on the head **and** inside `chia-chisel-build-aether-cosim:local` | all MISSING |
| Hammer | `python -c "import hammer"` on the head and in the build image | `ModuleNotFoundError: No module named 'hammer'` |
| Sky130 PDK | `find / -iname "*sky130*"` on host and in the build image | only chipyard's *example YAML templates* (`vlsi/example-sky130.yml` etc.); no `sky130A`, no libs, no LEF/LIB |
| VLSI worker | `examples/titan/cluster.yaml`, node type `vlsi` | `num_workers: 0`, `image:  # FILL — genus + hammer + cacti + sky130 PDK image` |
| PDK path constant | `examples/titan/constants.py:470` | `SKY130_COL_PATH = os.environ.get("TITAN_SKY130_COL_PATH", "#FILL")`; the env var is unset |
| Tool/tech configs | `examples/sky130_vlsi/tools-chia.yml`, `tech-sky130.yml` | unfilled templates: `synthesis.genus.genus_bin: "/path/to/cadence/GENUS"`, `technology.sky130.basepath: "/path/to/base/"` |
| Docker images | `docker images` on gpuserv4 | no genus / sky130 / vlsi image exists on the host at all |

`synth_node.py`'s never having been wired into the loop is therefore not an
oversight in the driver: the cluster has **no `vlsi` worker to dispatch to and
nothing to run on it**. Standing this up requires a licensed Cadence Genus
install, the `sky130_col` collateral tree, `hammer-vlsi`, and a container image
carrying all three. **`synth_node.py` was read and needed no fix — it was left
untouched.** No PPA numbers were fabricated.

For the record, had the flow existed the constraint would have been the one in
`examples/sky130_vlsi/design.yml`: clocks `clock` and `clock_uncore`, **period
10 ns (100 MHz), uncertainty 0.5 ns**, `vlsi_top = ShuttleTile`
(`constants.SYNTH_VLSI_TOP`), Genus version 211, `sky130_scl` stackup.

## 2. What was produced instead, and against what

Three designs were elaborated through the loop's own build path
(`nodes.build_saturn` → `chia.chipyard.ChiselBuildNode`,
`BuildTarget.VERILATOR`, `collect_generated_src=True`) on the idle Titan Ray
cluster, with identical tools, flags and container for all three, and the
emitted `gen-collateral` SystemVerilog was compared structurally.

| label | config | tree |
|---|---|---|
| `ime` | `TitanV256D128ShuttleConfig` (`constants.SYNTH_CONFIG` / `DIRECTED_CONFIG`) | pristine + `ppa/r13_round2_rtl_only.diff` |
| `baseline_ref` | `REFV256D128ShuttleConfig` | pristine (`nodes.reset_chipyard`) |
| `baseline` | `GENV256D128ShuttleConfig` (`constants.BASELINE_CONFIG`) | pristine |

### The A/B pair had to be corrected — read this before using any number

The task named `constants.BASELINE_CONFIG` (`GENV256D128ShuttleConfig`) as the
A/B partner for `SYNTH_CONFIG`. **It is not parameter-matched**, and using it
alone would have produced a misleading result:

```
GENV256D128ShuttleConfig  : WithShuttleVectorUnit(256, 128, VectorParams.genParams)
TitanV256D128ShuttleConfig: WithShuttleVectorUnit(256, 128, VectorParams.refParams)
```

`genParams = dspParams.copy(issStructure = Split, vlifqEntries = 16,
vlrobEntries = 16, vliqEntries = 4, vsiqEntries = 6)`, whereas `refParams`
keeps `issStructure = Shared` and the case-class defaults `vlifqEntries = 8`,
`vlrobEntries = 4` (`generators/saturn/src/main/scala/common/Parameters.scala`).
The observable consequence: the baseline's `LoadOrderBuffer` elaborates 16
entries (2201 flop bits) and the Titan design's 8 (747 flop bits) — a 1454-bit
*reduction* that has **nothing to do with IME**. That was the first symptom
that exposed the mismatch. `BASELINE_CONFIG` is the right baseline for the
*functional* RVV regression it was introduced for (issue structure and queue
depths do not change architectural results); it is the wrong baseline for a
cost comparison.

`REFV256D128ShuttleConfig` was therefore added as the **primary** baseline. It
is fragment-for-fragment identical to `TitanV256D128ShuttleConfig` —
`WithShuttleVectorUnit(256, 128, VectorParams.refParams)` +
`WithSystemBusWidth(128)` + `WithShuttleTileBeatBytes(16)` +
`WithNShuttleCores(1)` + `AbstractConfig` — so it is `TitanV256D128ShuttleConfig`
on a pristine tree, and **every difference between `baseline_ref` and `ime` is
attributable to the IME RTL hunks and to nothing else.** The
`baseline` (GEN) comparison is kept in `results.json` as a secondary data
point, explicitly flagged as not parameter-matched.

Neither build carries the cosim harness (no `WithCospike` / `WithTraceIO` /
`WithShuttleDebugROB`), so no measurement collateral inflates either side.

### The design under test

`ppa/r13_round2_rtl_only.diff` is the RTL half of `titan_runs/r13_round2.diff`:
the 15 `generators/**` files (saturn 12, rocket-chip 2, shuttle 1), **+844 /
−45 lines of Chisel**, with the 12 `toolchains/riscv-tools/riscv-isa-sim/**`
Spike-model files removed, since the Spike model is not elaborated into
hardware. It was applied with `nodes.apply_diff` (reported `applied`).

## 3. Provenance

| | |
|---|---|
| Date | 2026-09-17 |
| Host | gpuserv4, <HEAD_IP>, Ubuntu 22.04.5 |
| Cluster | Titan Ray head `<HEAD_IP>:6380`, job server `http://127.0.0.1:8266` |
| Ray jobs | `titan-ppa-rtl-ab-1` (baseline + ime), `titan-ppa-rtl-ref-1` (baseline_ref) |
| Build container | `chia-chisel-build-aether-cosim:local` (`titan-chisel-build-max410011_l-0`) |
| Chipyard | `1.14.0-19-g4ab72313` |
| Submodules | saturn `dfe75de`, rocket-chip `70430823f`, shuttle `622f08b` (all reset to the pinned commit before each build) |
| Chisel / Scala / sbt / JDK | 6.7.0 / 2.13 / 1.8.2 / OpenJDK 20.0.2-internal |
| FIRRTL compiler | **CIRCT firtool-1.75.0** (LLVM 19.0.0git) |
| Verilator | 5.022 2024-02-24 (conda-forge) |
| Synthesis tool / PDK / clock constraint | **none — nothing was synthesised or timed** |
| Seed / effort | not applicable (no synthesis); all three elaborations used the identical `nodes.build_saturn` code path and make flags |

## 4. Method — how each number is obtained

`tools/ppa_rtl_metrics.py` parses the `.sv`/`.v` files collected from
`gen-collateral` (firtool emits exactly one module per file):

* **`n_sv_files` / `total_sv_lines`** — file count and newline count over all
  collected `.sv`/`.v` files.
* **`n_modules`** — matches of `^\s*module\s+(\w+)\s*[(#]`.
* **`flop_bits`** — per module, every `reg [hi:lo] name;` declaration without
  an array dimension contributes `hi-lo+1` bits; a bare `reg name;`
  contributes 1. This is firtool's flop emission style, so this is a
  **register-bit count** — the largest single sequential-area driver, and the
  most defensible area proxy available without a library. Comments are
  stripped first.
* **`mem_array_bits`** — the same declarations *with* an array dimension
  (`reg [w] name [d];`), i.e. memories that stayed as inferred RTL arrays.
  Zero in every config here: Saturn's memories are extracted to macros.
* **`mult_ops` / `add_ops`** — occurrences of binary `*` and `+` in the module
  body after comment stripping (`**`, `*/`, `/*`, `++` excluded). A proxy for
  multiplier/adder *inference sites*, **not** a datapath-accurate count: one
  `*` in Verilog can be any width, so read it as "how many multiply sites
  exist", never as "how much multiplier area".
* **`n_sram_macros` / `sram_total_bits`** — parsed from the flow's own
  `*.top.mems.conf` macro manifest (`name … depth … width … ports …`);
  `bits = depth × width`. These are black-box SRAM macros, not synthesised
  logic, and in a real flow they would dominate area.
* **hierarchy roll-ups** — instantiation edges are recovered from
  `^\s+(Child)\s+(inst)\s*\(` lines where `Child` is a known module name, then
  aggregated from a named root **with multiplicity** (a module instantiated 4×
  counts 4×). This yields per-hierarchy figures for `ShuttleTile` (the
  intended synthesis top), `SaturnShuttleUnit` (the vector unit),
  `VectorBackend`, `VectorMemUnit`, and the new `MatrixMultiplyPipe`.
  Roll-ups, not per-module-name diffs, are the trustworthy view: firtool
  renumbers deduplicated modules (`DCEQueue_8` → `DCEQueue_9` etc.) between
  builds, so comparing same-named modules across configs is meaningless.

Reproduce with:

```
# builds (Ray) -- writes ppa/rtl/{baseline,ime}/ and ppa/rtl/baseline_ref/
ray job submit --address http://127.0.0.1:8266 --no-wait \
  -- bash -c "cd chia/examples/titan && python titan_runs/tools/ppa_rtl_ab.py"
TITAN_PPA_ONLY=baseline_ref  # ... same, for the third config
# metrics + results.json
python titan_runs/tools/ppa_assemble.py
```

## 5. Results

See `results.json` (machine-readable; `comparisons.PRIMARY_baseline_ref_vs_ime`
is the one to quote) and §6 below for the headline table. Raw SystemVerilog is
kept under `rtl/{baseline,baseline_ref,ime}/`.

## 6. Caveats — what these numbers cannot tell you

1. **They are not PPA numbers.** Register bits do not map linearly to cell
   area; `*` counts do not map to multiplier area; and there is **no timing or
   power figure here at all** — WNS, Fmax and power are *absent*, not zero.
2. Un-synthesised RTL still contains logic a synthesiser would delete
   (constant propagation, unused-output pruning) and none of the sharing a
   synthesiser would find. Treat the deltas as an upper-bound sketch of *where
   the design grew*, not as area.
3. `mult_ops` is especially weak for this design: the IME's arithmetic cost
   lives in the width and count of the MAC operations the `MatrixMultiplyPipe`
   issues over many cycles, which a `*`-site count cannot see. An iterative
   unit that reuses one multiplier looks "free" here and is not.
4. The GEN-vs-IME comparison in `results.json` mixes the IME change with the
   genParams/refParams microarchitecture difference (§2) and must not be
   quoted as an IME cost.
5. All three builds used the same tree, tools, flags and container, and each
   was preceded by `nodes.reset_chipyard`; the chipyard tree was reset again
   after the last build. The comparison is internally consistent even though
   the absolute numbers are not physical.

---

## 7. Headline table (generated from results.json)

### Whole design — `baseline_ref` (REFV256D128ShuttleConfig, pristine) vs `ime` (TitanV256D128ShuttleConfig + IME RTL diff)

| metric | baseline_ref | ime | delta | % |
|---|---:|---:|---:|---:|
| generated .sv/.v files | 668 | 670 | +2 | +0.30% |
| total generated Verilog lines | 185,097 | 186,236 | +1,139 | +0.61% |
| modules | 681 | 683 | +2 | +0.29% |
| register bits (sum over unique modules) | 415,684 | 416,535 | +851 | +0.20% |
| inferred RTL memory bits | 0 | 0 | +0 | n/a |
| multiply sites (`*`) | 20 | 22 | +2 | +10.00% |
| add sites (`+`) | 1,828 | 1,859 | +31 | +1.70% |
| SRAM macros | 8 | 8 | +0 | +0.00% |
| SRAM macro bits | 1,533,824 | 1,533,824 | +0 | +0.00% |

### By hierarchy (instantiation roll-up, multiplicity-weighted)

| root | metric | baseline_ref | ime | delta | % |
|---|---|---:|---:|---:|---:|
| ShuttleTile (intended synth top) | flop_bits | 71,719 | 72,570 | +851 | +1.19% |
| ShuttleTile (intended synth top) | mult_ops | 39 | 42 | +3 | +7.69% |
| ShuttleTile (intended synth top) | add_ops | 2,254 | 2,290 | +36 | +1.60% |
| ShuttleTile (intended synth top) | lines | 87,769 | 88,842 | +1,073 | +1.22% |
| ShuttleTile (intended synth top) | module_instances | 668 | 669 | +1 | +0.15% |
| SaturnShuttleUnit (vector unit) | flop_bits | 28,952 | 29,803 | +851 | +2.94% |
| SaturnShuttleUnit (vector unit) | mult_ops | 33 | 36 | +3 | +9.09% |
| SaturnShuttleUnit (vector unit) | add_ops | 1,730 | 1,766 | +36 | +2.08% |
| SaturnShuttleUnit (vector unit) | lines | 42,945 | 43,864 | +919 | +2.14% |
| SaturnShuttleUnit (vector unit) | module_instances | 366 | 367 | +1 | +0.27% |
| VectorBackend | flop_bits | 16,368 | 17,083 | +715 | +4.37% |
| VectorBackend | mult_ops | 24 | 25 | +1 | +4.17% |
| VectorBackend | add_ops | 1,607 | 1,630 | +23 | +1.43% |
| VectorBackend | lines | 30,167 | 30,751 | +584 | +1.94% |
| VectorBackend | module_instances | 328 | 329 | +1 | +0.30% |
| VectorMemUnit | flop_bits | 7,935 | 8,071 | +136 | +1.71% |
| VectorMemUnit | mult_ops | 7 | 9 | +2 | +28.57% |
| VectorMemUnit | add_ops | 78 | 89 | +11 | +14.10% |
| VectorMemUnit | lines | 7,485 | 7,636 | +151 | +2.02% |
| VectorMemUnit | module_instances | 17 | 17 | +0 | +0.00% |
| MatrixMultiplyPipe (NEW matrix pipe) | flop_bits | 0 | 558 | +558 | new |
| MatrixMultiplyPipe (NEW matrix pipe) | mult_ops | 0 | 1 | +1 | new |
| MatrixMultiplyPipe (NEW matrix pipe) | add_ops | 0 | 4 | +4 | new |
| MatrixMultiplyPipe (NEW matrix pipe) | lines | 0 | 173 | +173 | new |
| MatrixMultiplyPipe (NEW matrix pipe) | module_instances | 0 | 1 | +1 | new |

New modules in `ime`: `Arbiter2_VectorWrite`, `Arbiter8_ScalarWrite`, `MatrixMultiplyPipe`. Gone from `baseline_ref`: `Arbiter7_ScalarWrite` (the two `Arbiter*` changes are arity changes — one extra write port — not new logic blocks).

**Reading of the primary result.** The IME hunks add **+851 register bits (+1.19%) to `ShuttleTile`**, of which **558 bits (66%) are the new `MatrixMultiplyPipe`** itself; the remaining ~293 bits are state added to the vector backend (`ExecuteSequencer` tile-walk state), the memory unit (`page_carry` on `IFQEntry`), and the rocket-chip CSR/decode paths. The design grows by **+1,139 lines of generated Verilog (+0.62%)**, **+2 multiply sites** and **+31 add sites**, adds **one module** net, and uses the **same 8 SRAM macros / 1,533,824 SRAM bits** as the baseline — the IME added no memory macro. Wall clock: 3 elaborations, 163.7 s + 160.0 s + 164.8 s, 489 s of build time across two Ray jobs totalling ~500 s.

No timing or power number accompanies these, and none can be produced on this deployment (§1).

For contrast, the *unmatched* GEN baseline would have reported -653 register bits (-0.16%) — i.e. the IME design looking **cheaper** than the baseline — purely because of the genParams/refParams queue-depth difference. That is why §2's correction matters.


### Tool warnings

All three builds completed cleanly (`build_ok: true`). The only diagnostics are
Verilator lint on the generated Verilog — `Warning-WIDTHEXPAND` (166),
`WIDTHCONCAT` (24), `WIDTHTRUNC` (14), `UNSIGNED` (2) — plus a cosmetic sbt
`implicit.relative.glob` notice. The IME build emits **3 more warning lines
than the parameter-matched baseline (221 vs 218)**, all of the same width-lint
kind; none is an error and none is suppressed. Full text in
`{baseline,baseline_ref,ime}_build.stderr.txt`.
