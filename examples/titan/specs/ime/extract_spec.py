#!/usr/bin/env python3
"""Derive machine-readable IME encodings from the Zvvm v0.9.0 asciidoc source.

The spec is a draft and will move.  Nothing in Titan hand-copies an opcode:
`instructions.json` is regenerated from the adoc by this script, and
`encodings.py` asserts the two agree.  Re-run after every spec bump:

    python specs/ime/extract_spec.py <path-to-integrated-matrix.adoc>

Authoritative source as of 2026-09-02:
    riscv/integrated-matrix-extension @ origin/zvvm-review-fixes
    tag v0.9.0 = 5a2d0f65 (2026-08-23), src/unpriv/integrated-matrix.adoc
    NOTE: that tag is *not* an ancestor of main; main carries an older text.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADOC = os.path.join(
    HERE, "..", "..", "..", "..", "..", "titan", "ime-spec",
    "integrated-matrix-v0.9.0.adoc",
)
OUT_JSON = os.path.join(HERE, "instructions.json")

SPEC_VERSION = "0.9.0"
SPEC_COMMIT = "5a2d0f65"

# --- wavedrom field grammar -------------------------------------------------
# Fields carry optional `attr:` and `type:` keys in any order, so pick the
# braces apart key by key rather than positionally.
_BLOCK = re.compile(r"\{([^{}]*(?:\[[^\]]*\][^{}]*)*)\}")
_BITS = re.compile(r"bits:\s*(\d+)")
_NAME = re.compile(r"name:\s*('[^']*'|0x[0-9a-fA-F]+|\d+|\w+)")
_ATTR = re.compile(r"attr:\s*\[([^\]]*)\]")


def _name_token(tok: str):
    tok = tok.strip()
    if tok.startswith("'") and tok.endswith("'"):
        return "field", tok[1:-1]
    if tok.startswith("0x"):
        return "const", int(tok, 16)
    if re.fullmatch(r"\d+", tok):
        return "const", int(tok)
    return "field", tok


def parse_wavedrom(block: str):
    """Return [{lsb,width,kind,name,value,attr}, ...] ordered lsb-first."""
    fields, lsb = [], 0
    for m in _BLOCK.finditer(block):
        blk = m.group(1)
        mb, mn = _BITS.search(blk), _NAME.search(blk)
        if not (mb and mn):
            continue
        width = int(mb.group(1))
        kind, val = _name_token(mn.group(1))
        ma = _ATTR.search(blk)
        attr = (ma.group(1) if ma else "").replace("'", "").strip() or None
        fields.append({
            "lsb": lsb,
            "width": width,
            "kind": kind,
            "name": val if kind == "field" else None,
            "value": val if kind == "const" else None,
            "attr": attr,
        })
        lsb += width
    if lsb != 32:
        raise ValueError(f"wavedrom does not cover 32 bits (got {lsb})")
    return fields


def extract(adoc_path: str) -> dict:
    lines = io.open(adoc_path, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("=== Instructions (in alphabetical"))
    end = next(i for i, l in enumerate(lines)
               if i > start and l.startswith("=== Microscaling subextensions"))
    body = lines[start:end]

    heads = [(m.group(1), i) for i, l in enumerate(body)
             if (m := re.match(r"^==== (\S+)$", l))]
    heads.append((None, len(body)))

    insns = {}
    for (name, i), (_, j) in zip(heads[:-1], heads[1:]):
        chunk = "\n".join(body[i:j])
        wd = re.search(r"Encoding::\n\[wavedrom.*?\n\.\.\.\.\n(.*?)\n\.\.\.\.",
                       chunk, re.S)
        if not wd:
            print(f"warning: no encoding block for {name}", file=sys.stderr)
            continue
        fields = parse_wavedrom(wd.group(1))
        syn = re.search(r"Synopsis::\n(.+)", chunk)
        mne = re.search(r"Mnemonic::\n(.+)", chunk)
        insns[name] = {
            "synopsis": syn.group(1).strip() if syn else None,
            "mnemonic": (mne.group(1).strip().replace("_", "") if mne else None),
            "extensions": sorted(set(re.findall(r"^\|(Zvvm\w*)", chunk, re.M))),
            "fields": fields,
            "constants": {
                (f["attr"] or f"bits[{f['lsb']}:{f['lsb'] + f['width'] - 1}]"): f["value"]
                for f in fields if f["kind"] == "const"
            },
            "operands": [f["name"] for f in fields if f["kind"] == "field"],
        }
    return insns


def main(argv):
    adoc = argv[1] if len(argv) > 1 else DEFAULT_ADOC
    if not os.path.exists(adoc):
        sys.exit(f"spec not found: {adoc}\nusage: extract_spec.py <adoc>")
    insns = extract(adoc)
    doc = {
        "spec": "Zvvm Family of Integrated Matrix Extensions",
        "version": SPEC_VERSION,
        "commit": SPEC_COMMIT,
        "source": os.path.basename(adoc),
        "instructions": insns,
    }
    with io.open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {OUT_JSON}: {len(insns)} instructions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
