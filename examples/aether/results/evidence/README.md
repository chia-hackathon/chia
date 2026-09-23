# Evidence

Raw artifacts backing claims in the AETHER write-up. Everything here is output
of a tool, not prose.

- `hardware-config-excerpts.md` — elaborated device tree / clock frequency.
- `membus/` — first-hand proof that the DRAM-facing memory bus elaborates at
  64 bits (8 B/cycle), not 128. Four independent sources:
  `xbar-module-names.txt` (generated crossbars: `mbus …a32d64…` vs
  `sbus …a32d128…`), `chiptop-axi4-port.txt` (AXI4 memory port is `[63:0]`
  data with `[7:0]` strb), `testharness-simdram.txt`
  (`SimDRAM #(.DATA_BITS(64))`, whose `mm_magic_t` backend moves
  `DATA_BITS/8` bytes per beat), and `config-defaults.txt`
  (rocket-chip's `MemoryBusKey => MemoryBusParams(beatBytes = 8)` alongside
  `WithSystemBusWidth`, which sets only `SystemBusKey`).
  Produced inside the chisel-build container from
  `sims/verilator/generated-src/…GENV256D128GemminiShuttleConfig/`.

- `cacti/` — SRAM macro area for the two sweep configurations, from CACTI 7
  (HewlettPackard/cacti @ 1ffd8df). Not a synthesis result: an analytic model,
  SRAM only, no logic, no place & route.
  - `*.mems.conf` — the elaborated SRAM lists both configurations produce.
  - `instance-counts.md` — **read this first.** `.top.mems.conf` lists SRAM
    *definitions*, not instances: `cc_banks_0_ext` is instantiated 8 times, wrapped macros (e.g. the Gemmini
    scratchpad `mem_ext`, 4×) must be counted hierarchically, and
    the wide configuration replaces one 8192×64 macro with one 4096×128.
    Summing the file directly gives −18%; weighting by instantiation count
    gives **+0.11%**, with total SRAM capacity unchanged at 967.8 KiB.
  - `sram-area.md`, `results.json` — per-macro and total area at 90/45/32/22 nm
    (CACTI 7 rejects anything coarser than 90 nm).
  - `raw/<node>/<config>/<sram>.{cfg,out}` — CACTI inputs and outputs verbatim.
    `raw/130nm/` holds only `.cfg` files and no `.out`: CACTI 7 aborts with
    `Feature size must be <= 90 nm`, so the 130 nm default in CHIA's
    `cacti_runner.py` cannot be used at all.
  - `tag_array_0_ext` (16×88 = 176 B) is below what CACTI can model and is
    excluded from both totals; no analytical fallback was substituted for it.
