#!/usr/bin/env python3
"""Measure the ported predictor's conditional MPKI directly, no simulator.

The cross-tier comparison rests on one claim: the ChampSim and gem5 ports run
the *same algorithm* as the HARCOM original, and the MPKI gap between them is
the block-versus-branch interface rather than a translation error.  That claim
was previously untestable in isolation -- the only way to see the port's MPKI
was to build ChampSim and simulate, which measures the port and the core model
together and takes ten minutes.

This renders ``ports/tage_core.h.in`` at a chosen parameter set through the
same ``string.Template`` path the loop uses, compiles it against
``tools/tage_selfcheck.cpp``, and runs it over a CBP-NG trace.  Seconds, and
the number it prints is attributable to the algorithm alone.

    tools/port_selfcheck.py --trace ~/bp_evolve_data/cbp_traces/gcc_1.gz
    tools/port_selfcheck.py --trace a.gz b.gz c.gz --params 6,8,11,12,11,100,14,6

Compare its output against Tier 0's MPKI for the same design and traces.  Give
it the *same trace set* Tier 0 scored on: Tier-0 MPKI is the unweighted mean of
the per-trace rates (vfs.aggregate averages ``mpi`` over traces, it does not
pool mispredicts over instructions), so the ``conditional_mpki`` printed last
here is that same unweighted mean and nothing else is comparable to it.

The core is rendered and compiled once and then run once per trace, and the
per-trace runs go out --jobs at a time.  They are independent -- each builds its
own predictor state from its own trace -- so the only reason this was ever
serial is that it used to be given four short traces.  It sustains 6.7 MIPS per
core -- roughly 36x HARCOM, since there is no library between it and the
rendered core -- so the 126-trace inner set at full length is about ten
core-minutes, and well under a minute at --jobs 16.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import pathlib
import subprocess
import sys
import tempfile
from string import Template

HERE = pathlib.Path(__file__).resolve().parent
PORTS = HERE.parent / "ports"

# Kept in step with agents.PORTED_PARAMS deliberately rather than imported:
# this tool must run from a bare checkout, with no chia and no Ray on the path.
ORDER = ("LOGLB", "NUMG", "LOGG", "LOGB", "TAGW", "GHIST", "LOGP1", "GHIST1")
PORTED = ("LOGLB", "NUMG", "LOGG", "LOGB", "TAGW", "GHIST")
DEFAULTS = (6, 8, 11, 12, 11, 100, 14, 6)


def render_core(params: dict[str, int]) -> str:
    return Template((PORTS / "tage_core.h.in").read_text()).substitute(
        {k: str(params[k]) for k in PORTED})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, nargs="+",
                    help="CBP-NG trace(s) (.gz); give the set Tier 0 scored on")
    ap.add_argument("--params", default=",".join(str(v) for v in DEFAULTS),
                    help=f"comma-separated {','.join(ORDER)}")
    ap.add_argument("--warmup", type=int, default=1_000_000,
                    help="warmup instructions (match BPE_WARMUP_INSTRUCTIONS)")
    ap.add_argument("--sim", type=int, default=0,
                    help="measured instructions per trace, 0 = to end of trace "
                         "(match BPE_SIM_INSTRUCTIONS). Any value past the "
                         "longest trace means the same thing as 0, since the "
                         "reader throws at EOF.")
    ap.add_argument("--jobs", type=int, default=0,
                    help="traces to run at once (0 = one per CPU). The runs "
                         "are independent; only the compile is shared.")
    ap.add_argument("--core", default=None,
                    help="use this rendered core instead of ports/tage_core.h.in "
                         "(for A/B against a modified algorithm)")
    ap.add_argument("--keep", action="store_true", help="keep the build directory")
    args = ap.parse_args()

    values = [int(x) for x in args.params.split(",")]
    if len(values) != len(ORDER):
        print(f"--params needs {len(ORDER)} values ({','.join(ORDER)})", file=sys.stderr)
        return 2
    params = dict(zip(ORDER, values))

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="port_selfcheck_"))
    core = (pathlib.Path(args.core).read_text() if args.core
            else render_core(params))
    (tmp / "tage_core_rendered.h").write_text(core)

    binary = tmp / "tage_selfcheck"
    build = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I", str(tmp), "-I", str(HERE),
         "-o", str(binary), str(HERE / "tage_selfcheck.cpp"), "-lz"],
        capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        return 1

    def one(trace: str):
        return trace, subprocess.run(
            [str(binary), trace, str(args.warmup), str(args.sim)],
            capture_output=True, text=True)

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(args.trace)))
    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        # Results are collected in argument order, not completion order: the
        # per-trace lines are the audit trail for the aggregate, and a set that
        # reorders itself run to run cannot be diffed against a previous one.
        # (The mean would be the same either way; the log would not.)
        done = list(pool.map(one, args.trace))

    rates = []
    for trace, run in done:
        sys.stderr.write(run.stderr)
        if run.returncode != 0:
            return run.returncode
        fields = dict(line.split(",", 1) for line in run.stdout.splitlines()
                      if "," in line)
        rate = float(fields["conditional_mpki"])
        rates.append(rate)
        print(f"trace,{pathlib.Path(trace).name},"
              f"{fields['instructions']},{fields['conditional']},"
              f"{fields['mispredicts']},{rate:.6f}")

    # Exactly one conditional_mpki line, and it is the aggregate -- callers
    # parse for that key, and a per-trace line carrying it too would let the
    # first trace masquerade as the whole set.
    print(f"n_traces,{len(rates)}")
    print(f"conditional_mpki,{sum(rates) / len(rates):.6f}")
    if args.keep:
        print(f"# build kept in {tmp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
