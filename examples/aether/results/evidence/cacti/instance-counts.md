# SRAM instance counts (NOT definition counts)

`.top.mems.conf` lists unique SRAM *definitions*. Summing it directly is wrong:
`cc_banks_0_ext` is instantiated 8 times, and the two mem-channel buffers are
deduplicated differently between the two configs.

Counted by grepping module instantiations in gen-collateral/*.sv:

| macro            | geometry   | control | WideMbus |
|------------------|------------|---------|----------|
| cc_dir_ext       | 1024x152   | 1       | 1        |
| cc_banks_0_ext   | 8192x64    | 8       | 8        |
| mem_ext          | 4096x128   | 1       | 2        |
| mem_0_ext        | 512x512    | 1       | 1        |
| tag_array_ext    | 64x160     | 1       | 1        |
| data_arrays_0_ext| 512x512    | 1       | 1        |
| tag_array_0_ext  | 16x88      | 1       | 1        |
| array_0_ext      | 128x256    | 1       | 1        |
| l2_tlb_ram_0_ext | 512x45     | 1       | 1        |
| mem_1_ext        | 8192x64    | 1       | 0        |

Total SRAM capacity is IDENTICAL: 731.2 KiB in both configs.
The WideMbus config replaces {1x4096x128 + 1x8192x64} with {2x4096x128}.

## Instance-weighted CACTI 7 totals (tag_array_0_ext excluded: CACTI cannot model it)

| Node | control (mm^2) | WideMbus (mm^2) | delta   | delta % |
|------|----------------|-----------------|---------|---------|
| 90nm | 9.1459         | 9.1602          | +0.0143 | +0.16%  |
| 45nm | 2.2860         | 2.2896          | +0.0036 | +0.16%  |
| 32nm | 1.1566         | 1.1584          | +0.0018 | +0.16%  |
| 22nm | 0.5469         | 0.5478          | +0.0009 | +0.16%  |

Reproduce: see run_area.py / mkreport.py; instance counts from
  docker exec aether-coexist grep -rhE '^\s*<macro>\s+\w+\s*\(' <gen-collateral>/*.sv | wc -l
