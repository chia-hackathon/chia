#!/usr/bin/env python3
"""Extract port-only declarations for top-level SRAM *_ext modules from the
*.top.mems.v behavioral memory model, and emit them as yosys blackboxes.

We only keep the 10 top-level _ext modules that are actually instantiated by
the SoC RTL (ChipTop/DigitalTop hierarchy). We deliberately drop every
recursive `split_*_ext` / bit-sliced leaf module in top.mems.v: those are
Verilator-only behavioral RAM models (reg array + always block) used only for
simulation, not synthesizable SRAM macros. Treating the top-level _ext module
as a blackbox (ports only, no body) matches the "SRAM area already done with
CACTI, do not re-synthesize it" instruction.
"""
import re
import sys

TOPLEVEL_EXT_MODULES = [
    "cc_dir_ext",
    "cc_banks_0_ext",
    "mem_ext",
    "mem_0_ext",
    "tag_array_ext",
    "data_arrays_0_ext",
    "tag_array_0_ext",
    "array_0_ext",
    "l2_tlb_ram_0_ext",
    "mem_1_ext",
]


def extract(text, modname):
    # Find "module <modname>(" ... up to the matching ");" that closes the
    # port list (first ");" after the opening, since ports here are simple
    # input/output decls with no nested parens).
    m = re.search(r"\bmodule\s+" + re.escape(modname) + r"\s*\(", text)
    if not m:
        return None
    start = m.start()
    close = text.find(");", m.end())
    if close == -1:
        return None
    header = text[start:close + 2]
    return header + "\nendmodule\n"


def main():
    if len(sys.argv) != 3:
        print("usage: make_blackbox_stub.py <top.mems.v> <out.sv>", file=sys.stderr)
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    with open(src) as f:
        text = f.read()
    chunks = []
    missing = []
    for name in TOPLEVEL_EXT_MODULES:
        stub = extract(text, name)
        if stub is None:
            missing.append(name)
            continue
        chunks.append("(* blackbox *)\n" + stub)
    if missing:
        print(f"WARNING: could not extract: {missing}", file=sys.stderr)
    with open(out, "w") as f:
        f.write("// Auto-generated blackbox stubs for SRAM *_ext macros (ports only).\n")
        f.write("// SRAM area/gate count is NOT included here; done separately via CACTI.\n\n")
        f.write("\n".join(chunks))
    print(f"wrote {len(chunks)} blackbox stubs to {out} (missing: {missing})")


if __name__ == "__main__":
    main()
