#!/usr/bin/env python3
"""Parse a `stat -top ChipTop -width` yosys output and compute:
  - FF bit count ($dff_N, $aldff_N -> sum N*count)
  - latch bit count ($dlatch_N -> sum N*count), reported separately
  - memory port cells ($memrd_v2*, $memwr_v2* -> not gates, reported separately)
  - SRAM blackbox macro instances (*_ext -> reported separately, 0 gates)
  - combinational gate bit count (everything else -> sum N*count)
  - instance counts for the same buckets (unweighted)
"""
import re
import sys

def parse(path):
    ff_bits = ff_inst = 0
    latch_bits = latch_inst = 0
    memport_bits = memport_inst = 0
    ext_inst = 0
    comb_bits = comb_inst = 0
    total_cells = None
    in_cells_section = False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"\s*(\d+)\s+cells$", line)
            if m:
                total_cells = int(m.group(1))
                in_cells_section = True
                continue
            if not in_cells_section:
                continue
            if line.strip() == "":
                # end of the cell-type breakdown for this module
                in_cells_section = False
                continue
            m = re.match(r"\s*(\d+)\s+(\S+)$", line)
            if not m:
                continue
            count = int(m.group(1))
            name = m.group(2)

            wm = re.match(r"(\$a?l?dff\w*?)_(\d+)$", name)
            # handle $dff_N / $aldff_N (also $dffe_N, $sdff_N etc if present)
            mdff = re.match(r"\$(a?l?s?dffe?)_(\d+)$", name)
            if mdff and mdff.group(1) in ("dff", "aldff", "adff", "sdff", "dffe", "aldffe", "adffe", "sdffe"):
                bits = int(mdff.group(2))
                ff_bits += bits * count
                ff_inst += count
                continue

            mlatch = re.match(r"\$dlatch\w*_(\d+)$", name)
            if mlatch:
                bits = int(mlatch.group(1))
                latch_bits += bits * count
                latch_inst += count
                continue

            mmem = re.match(r"\$mem(rd|wr)(_v2)?", name)
            if mmem or name.startswith("$mem"):
                # these are generic (no _N width suffix typically); still count instances
                memport_inst += count
                continue

            if name.endswith("_ext"):
                ext_inst += count
                continue

            # anything else with a _N suffix -> combinational, weight by width
            mgen = re.match(r"(\$\w+)_(\d+)$", name)
            if mgen:
                bits = int(mgen.group(2))
                comb_bits += bits * count
                comb_inst += count
            else:
                # no width suffix (e.g. $check, $tribuf, plain names) -> count as instance, 1 bit each
                comb_bits += count
                comb_inst += count
    return dict(total_cells=total_cells, ff_bits=ff_bits, ff_inst=ff_inst,
                latch_bits=latch_bits, latch_inst=latch_inst,
                memport_inst=memport_inst, ext_inst=ext_inst,
                comb_bits=comb_bits, comb_inst=comb_inst)

if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = parse(path)
        print(path)
        for k, v in r.items():
            print(f"  {k}: {v}")
