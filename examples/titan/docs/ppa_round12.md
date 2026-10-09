# Round 12 PPA (r37 design, measured in r38)

Flow: yosys/sv2v, sky130_fd_sc_hd tt_025C_1v80, `abc -D 10000` (same as ppa2..ppa8). abc post-mapping path estimates, not sign-off STA; no power measured (flop count only).

## Per-FU judge (flattened standalone closure)

| metric | r35 | r37 | change |
|---|---:|---:|---:|
| MatrixFPMultiplyPipe area (um2) | 561,908 | 286,940 | -48.9% |
| MatrixMultiplyPipe area (um2) | 102,314 | 88,556 | -13.4% |
| FP FU abc delay (ns) | 25.98 | 20.53 | |
| INT FU abc delay (ns) | 31.37 | 12.74 | |
| FP FU flops | 4,997 | 2,616 | |
| cycles, 9 benchmarks | reference | identical | 0 |

## Full SoC (SaturnShuttleUnit, hierarchical; P-SOC at the r38 Gate)

| metric | baseline (no IME) | r35 | r37 |
|---|---:|---:|---:|
| total area (um2) | 3,201,433 | 4,160,822 | 3,655,423 |
| IME overhead vs baseline | - | +959,389 (+29.97%) | +453,990 (+14.18%) |
| flops | 33,508 | 39,983 | 36,942 |
| SoC longest abc path | 30.87 ns (Compactor_4) | 30.25 ns (Compactor_4) | 30.25 ns (Compactor_4) |
| IME longest path | - | 23.70 ns | 21.36 ns |

## MatrixFPMultiplyPipe hierarchy (top rows)

| module | inst | own each (um2) | flops each |
|---|---:|---:|---:|
| MatrixFPMultiplyPipe | 1 | 188,202 | 2616 |
| MulFullRawFN | 1 | 95,742 | 0 |
| RoundAnyRawFNToRecFN_ie11_is55_oe11_os53 | 1 | 4,916 | 0 |
| RoundAnyRawFNToRecFN_ie8_is24_oe5_os11 | 2 | 1,401 | 0 |
| RoundAnyRawFNToRecFN_ie11_is55_oe8_os24 | 1 | 2,529 | 0 |
| RoundAnyRawFNToRecFN_ie8_is26_oe8_os24 | 1 | 2,460 | 0 |
| RoundAnyRawFNToRecFN_ie11_is55_oe5_os11 | 1 | 1,539 | 0 |
| RoundAnyRawFNToRecFN_ie11_is55_oe8_os8 | 1 | 1,369 | 0 |

## Cross-boundary timing check (flattened VectorBackend, own NLDM STA, loads clamped to 20 fF)

Longest path from outside an IME FU into its registers: INT 9.2 -> 12.6 ns, FP 9.2 -> 10.6 ns; longest path ending in IME registers: INT 26.4 -> 22.3 ns, FP 29.3 -> 26.1 ns; module longest path (FP FMA, not IME) 33.2 -> 34.3 ns. Approximate absolute values; same flow for both.
