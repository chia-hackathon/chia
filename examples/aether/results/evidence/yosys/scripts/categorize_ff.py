#!/usr/bin/env python3
"""Bucket $dff/$aldff cell instance counts by top-level hierarchical path
segment (best-effort submodule attribution; instance counts only, not
bit-weighted -- flip-flop cells here are the auto-derived process/register
cells from read_slang elaboration, and their names retain the original
FIRRTL/Chisel hierarchical instance path).
"""
import re
import sys
from collections import Counter

CATEGORIES = [
    ("Gemmini (accelerator)", re.compile(r"\bgemmini\b")),
    ("Saturn core (scalar+FP+vector, excl. Gemmini)", re.compile(r"\b(core|fp_pipe|vector_unit)\b")),
    ("L2 / coherence (coh_wrapper)", re.compile(r"\bcoh_wrapper\b")),
    ("System/Mem/Periph/Front bus (sbus/mbus/pbus/fbus/cbus)", re.compile(r"\b(sbus|mbus|pbus|fbus|cbus)\b")),
    ("Serial TL link", re.compile(r"\bserial_tl_domain\b")),
    ("Debug module (tlDM)", re.compile(r"\btlDM\b")),
    ("Tile infra (prci/buffer/clock, not core/gemmini)", re.compile(r"\btile_prci_domain\b")),
]

def categorize(name):
    for label, pat in CATEGORIES:
        if pat.search(name):
            return label
    return "Other / top-level glue"

def main(path):
    c = Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c[categorize(line)] += 1
    total = sum(c.values())
    print(f"{path}  (total FF instances: {total})")
    for label, _ in CATEGORIES + [("Other / top-level glue", None)]:
        print(f"  {label}: {c.get(label,0)}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
