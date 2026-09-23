# SRAM instance counts (NOT definition counts)

`.top.mems.conf` lists unique SRAM *definitions*. Summing it directly is wrong,
and so is counting `_ext` instantiations flat: each `_ext` sits inside a wrapper
module (e.g. `mem`, `mem_0`, `array_0`) that is itself instantiated several times.
Counts below are **hierarchical instance counts from `ChipTop`** (module graph of
gen-collateral/*.sv, each child count multiplied by its parent's). They agree with
Yosys's elaborated netlist (27 macro instances per config).

| macro             | geometry | control | WideMbus | what it is (by capacity match to Table 1) |
|-------------------|----------|---------|----------|-------------------------------------------|
| cc_dir_ext        | 1024x152 | 1       | 1        | L2 directory |
| cc_banks_0_ext    | 8192x64  | 8       | 8        | L2 data, 512 KiB |
| mem_ext           | 4096x128 | 4       | 5        | Gemmini scratchpad banks (4 = 256 KiB) + in WideMbus one more |
| mem_0_ext         | 512x512  | 2       | 2        | Gemmini accumulator, 64 KiB |
| mem_1_ext         | 8192x64  | 1       | 0        | replaced by a 4096x128 mem_ext in WideMbus |
| tag_array_ext     | 64x160   | 1       | 1        | |
| data_arrays_0_ext | 512x512  | 1       | 1        | |
| tag_array_0_ext   | 16x88    | 4       | 4        | (CACTI cannot model; excluded from area) |
| array_0_ext       | 128x256  | 4       | 4        | |
| l2_tlb_ram_0_ext  | 512x45   | 1       | 1        | |
| **instances**     |          | **27**  | **27**   | |

Total SRAM capacity is IDENTICAL: **967.8 KiB** in both configs.
The WideMbus config replaces one 8192x64 macro with one 4096x128 macro (same 64 KiB).

## Instance-weighted CACTI 7 totals (tag_array_0_ext excluded: CACTI cannot model it)

| Node | control (mm^2) | WideMbus (mm^2) | delta   | delta % |
|------|----------------|-----------------|---------|---------|
| 90nm | 12.6117        | 12.6260         | +0.0143 | +0.11%  |
| 45nm | 3.1522         | 3.1558          | +0.0036 | +0.11%  |
| 32nm | 1.5949         | 1.5967          | +0.0018 | +0.11%  |
| 22nm | 0.7541         | 0.7549          | +0.0009 | +0.11%  |

An earlier version of this file counted `_ext` instantiations flat (17 per config,
731.2 KiB, +0.16%); that undercounted the Gemmini scratchpad/accumulator and
other wrapped macros. The absolute delta (+0.0036 mm^2 @45nm) is unchanged; only
the denominator was wrong.
