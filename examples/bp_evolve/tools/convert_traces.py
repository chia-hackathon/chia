#!/usr/bin/env python3
"""Build the trace converter and convert a CBP-NG trace set for ChampSim.

Tiers 0 and 1 only mean anything together if they see the same branches.  They
read different file formats, so one of them has to be translated, and
translating the *trace* is far safer than translating the predictor: the result
is checkable against ground truth, because cbp-ng reports its own instruction
and branch totals for every trace it runs.

``--verify`` does exactly that check.  It runs the stock ``cbp`` binary on the
original and compares its counts against the converter's over the same window.
Instruction and branch totals must match exactly; the conditional-branch total
is allowed to differ by one, because cbp-ng does not count the conditional
branch that straddles its warmup boundary and the converter has no boundary.

Usage:
    convert_traces.py --cbp-root ~/cbp-ng --in <dir-or-file> --out <dir>
                      [--verify] [--max-instructions N] [--jobs N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER_SRC = os.path.join(HERE, "cbp2champsim.cpp")


def build_converter(out_path: str) -> str:
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, "-std=c++17", "-O2", "-o", out_path, CONVERTER_SRC, "-lz"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot build the converter:\n{r.stderr}")
    return out_path


def parse_counts(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "," in line:
            k, v = line.split(",", 1)
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out


def convert_one(binary: str, src: str, dst: str, max_instructions: int) -> dict:
    tmp = dst + ".partial"
    cmd = [binary, src, tmp]
    if max_instructions:
        cmd.append(str(max_instructions))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return {"trace": src, "ok": False, "error": r.stderr.strip()[:300]}
    # Rename only on success, so an interrupted run leaves no half-written
    # trace that the next sweep would happily simulate.
    os.replace(tmp, dst)
    counts = parse_counts(r.stdout)
    counts.update({"trace": src, "out": dst, "ok": True})
    return counts


def verify_one(cbp_root: str, binary: str, src: str, counts: dict,
               warmup: int = 100_000) -> str:
    """Compare against what cbp-ng says about the same trace."""
    cbp = os.path.join(cbp_root, "cbp")
    if not os.path.exists(cbp):
        return "cbp binary not built; skipped verification"
    r = subprocess.run(
        [cbp, src, "verify", str(warmup), "100000000", "--format", "csv"],
        capture_output=True, text=True, cwd=cbp_root)
    if r.returncode != 0 or "," not in r.stdout:
        return f"cbp did not run: {r.stderr.strip()[:200]}"
    fields = r.stdout.strip().splitlines()[0].split(",")
    cbp_instr, cbp_br, cbp_cond = int(fields[1]), int(fields[2]), int(fields[3])

    # cbp-ng measures from just after the warmup; the converter counts the
    # whole file.  Re-count the warmup prefix and subtract.
    pre = parse_counts(subprocess.run(
        [binary, src, os.devnull, str(cbp_instr and warmup + 1)],
        capture_output=True, text=True).stdout)

    d_instr = counts["instructions"] - pre["instructions"]
    d_br = counts["branches"] - pre["branches"]
    d_cond = counts["conditional"] - pre["conditional"]

    problems = []
    if d_instr != cbp_instr:
        problems.append(f"instructions {d_instr} vs cbp {cbp_instr}")
    if d_br != cbp_br:
        problems.append(f"branches {d_br} vs cbp {cbp_br}")
    if abs(d_cond - cbp_cond) > 1:
        problems.append(f"conditional {d_cond} vs cbp {cbp_cond}")
    return "OK" if not problems else "MISMATCH: " + "; ".join(problems)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="a CBP-NG trace or a directory of them")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cbp-root", default=os.path.expanduser("~/cbp-ng"))
    ap.add_argument("--max-instructions", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--verify", action="store_true",
                    help="check the counts against cbp-ng's own")
    ap.add_argument("--force", action="store_true",
                    help="reconvert traces that already exist in --out")
    args = ap.parse_args(argv)

    if os.path.isdir(args.src):
        srcs = sorted(os.path.join(args.src, f) for f in os.listdir(args.src)
                      if f.endswith(".gz"))
    else:
        srcs = [args.src]
    if not srcs:
        print(f"no .gz traces under {args.src}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    binary = build_converter(os.path.join(args.out, ".cbp2champsim"))

    jobs = []
    for src in srcs:
        stem = os.path.basename(src)
        for suffix in (".champsimtrace.gz", ".gz"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        dst = os.path.join(args.out, f"{stem}.champsimtrace.gz")
        if os.path.exists(dst) and not args.force:
            print(f"skip  {stem} (already converted)")
            continue
        jobs.append((src, dst))

    if not jobs:
        print("nothing to do")
        return 0

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(convert_one, binary, src, dst,
                               args.max_instructions): src
                   for src, dst in jobs}
        for fut in concurrent.futures.as_completed(futures):
            c = fut.result()
            name = os.path.basename(c["trace"])
            if not c["ok"]:
                failures += 1
                print(f"FAIL  {name}: {c['error']}")
                continue
            line = (f"ok    {name}: {c['instructions']:,} instructions, "
                    f"{c['branches']:,} branches "
                    f"({c['conditional']:,} conditional)")
            if args.verify:
                line += "  |  " + verify_one(args.cbp_root, binary,
                                             c["trace"], c)
            print(line)

    os.remove(binary)
    if failures:
        print(f"\n{failures}/{len(jobs)} traces failed to convert",
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
