#!/usr/bin/env python3
"""Round 11 timing judge: standalone, flattened gate-level synthesis of the
matrix FP FU and a pass/fail on its longest combinational path.

Written (2026-09-30) before the round-11 RTL agent exists -- "judges before
defendants" -- and calibrated on the r31 tree it is meant to convict.

Flow (identical liberty / abc script / -D to titan_runs/tools/ppa2_syn.py, the
ppa2..ppa7 flow, so every number is comparable to those reports):

    sv2v  (only the dependency closure of the top, not all ~700 files)
    yosys synth -top TOP -flatten ; dfflibmap ; abc -liberty -D 10000 (+stime -p)
    stat -liberty

Two deliberate differences from ppa2_syn.py, both about what is *measured*,
not how the netlist is mapped:

* ``-flatten``.  ppa2..ppa7 synthesised hierarchically, so abc timed each
  module's own logic and never saw a path that crosses a submodule boundary.
  A judge built that way is trivially gamed -- wrap each stage of the exact-sum
  chain in its own Chisel ``Module`` and every module is short while the real
  register-to-register path is not.  Flattened, abc sees every path from a
  register (or input port) to a register (or output port) inside the FU.
* ``stime -p`` instead of ``stime``.  Same mapping (stime only reports); ``-p``
  additionally prints the critical path gate by gate plus its start and end
  points by *name*, which is the feedback the RTL agent needs.

Caveats the verdict inherits (same as ppa7 README sec. 6): no sign-off STA,
typical corner, no wire load, no SDC; paths that leave the FU are not seen;
abc's number is a mapped-logic-depth estimate, pessimistic for wide ripple
adders.  It is deterministic for an identical netlist (measured: two runs of
r31, bit-identical Delay), which is what a judge needs.

Usage (host, conda env; never /tmp -- TMPDIR is set to <out>/tmp):
    timing_judge.py RTL_DIR OUT_DIR [--top MatrixFPMultiplyPipe]
                    [--target-ns 40] [--area-base UM2] [--area-max-growth 0.5]
                    [--area-min-ratio 0.8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

LIB = os.environ.get(
    "PPA2_LIB",
    "/share1/saves/max410011/opt/orfs/flow/platforms/sky130hd/lib/"
    "sky130_fd_sc_hd__tt_025C_1v80.lib")
YOSYS = os.environ.get("PPA2_YOSYS", "/share1/saves/max410011/opt/eda/bin/yosys")
SV2V = os.environ.get("PPA2_SV2V", "/share1/saves/max410011/opt/sv2v-Linux/sv2v")

#: ppa2_syn.py's abc script with ``stime`` -> ``stime -p`` (report only).
ABC_SCRIPT = ("+strash;&get -n;&fraig -x;&put;scorr;dc2;dretime;"
              "retime -D {D};strash;&get -n;&dch -f;&nf {D};&put;"
              "stime -p")
PERIOD_PS = 10000          # abc -D, as ppa2..ppa7 (100 MHz, design.yml)

_FLOP_RE = re.compile(r"sky130_fd_sc_hd__(?:e?df\w*|dl\w*|sdf\w*)")


# --------------------------------------------------------------------------
# RTL closure: firtool writes one module per <Module>.sv, so the files a top
# needs are the transitive closure of module names mentioned in it.
# --------------------------------------------------------------------------
def rtl_files(rtl_dir: str) -> Dict[str, str]:
    """{module name: path} for every generated .sv/.v (the TestHarness-only
    memory model excluded, as ppa2_syn.collect)."""
    out = {}
    for fn in sorted(os.listdir(rtl_dir)):
        if fn.endswith((".sv", ".v")) and not fn.endswith(".model.mems.v"):
            out[os.path.splitext(fn)[0]] = os.path.join(rtl_dir, fn)
    return out


def closure_names(texts: Dict[str, str], tops: Sequence[str]) -> List[str]:
    """Module names needed to elaborate ``tops``, from {module: source}.
    Over-inclusion is harmless; omission makes yosys fail loudly
    (``hierarchy -check``)."""
    missing = [t for t in tops if t not in texts]
    if missing:
        raise FileNotFoundError(
            f"top module(s) {missing} not found among the generated "
            f"<Module>.sv files")
    ident = re.compile(r"\b([A-Za-z_][A-Za-z0-9_$]*)\b")
    seen, todo = set(), list(tops)
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for tok in set(ident.findall(texts[mod])):
            if tok in texts and tok not in seen:
                todo.append(tok)
    return sorted(seen)


def closure_files(files: Sequence[Sequence[str]], tops: Sequence[str]
                  ) -> List[Sequence[str]]:
    """The same over ``(filename, contents)`` pairs -- the shape of
    ``BuildArtifact.generated_src_files`` -- so the loop can ship only the
    FU's closure (tens of files) to the synthesis task, not all ~700."""
    by_mod = {os.path.splitext(fn)[0]: (fn, txt) for fn, txt in files
              if fn.endswith((".sv", ".v"))
              and not fn.endswith(".model.mems.v")}
    keep = closure_names({m: t for m, (_, t) in by_mod.items()}, tops)
    return [by_mod[m] for m in keep]


def closure(rtl_dir: str, tops: Sequence[str]) -> List[str]:
    """Paths of the files needed to elaborate ``tops`` in ``rtl_dir``."""
    files = rtl_files(rtl_dir)
    texts = {}
    for mod, path in files.items():
        with open(path, errors="replace") as fh:
            texts[mod] = fh.read()
    return sorted(files[m] for m in closure_names(texts, tops))


# --------------------------------------------------------------------------
# Log / stat parsing -- pure functions, unit-tested in test_timing_judge.py
# --------------------------------------------------------------------------
_HDR = re.compile(r"^\d+(?:\.\d+)*\. Extracting gate netlist of module "
                  r"`\\?([^']+)'")
_WL = re.compile(r"ABC: WireLoad.*?Gates =\s*(\d+).*?Area =\s*([\d.]+)"
                 r".*?Delay =\s*([\d.]+) ps")
_PATH = re.compile(r"^ABC: Path\s*(\d+)\s+--\s+(\d+)\s*:\s*(\d+)\s+(\d+)\s+"
                   r"(\S+)\s+A =\s*([\d.]+)\s+Df =\s*([\d.]+)")
_SE = re.compile(r"Start-point = (\S+) \((.*?)\)\.\s+End-point = (\S+) "
                 r"\((.*?)\)\.")


def parse_abc(log_text: str) -> List[Dict[str, object]]:
    """Every liberty-mapped abc run in the log, most critical first.

    The first abc pass inside ``synth`` maps to generic gates and prints no
    WireLoad line, so only the liberty pass contributes -- the attribution
    bug ppa5_abc_attr.py fixed in ppa2_syn.py's regex is avoided the same
    way (each Delay line belongs to the most recent module header)."""
    runs: List[Dict[str, object]] = []
    cur_mod: Optional[str] = None
    cur: Optional[Dict[str, object]] = None
    for line in log_text.splitlines():
        m = _HDR.match(line)
        if m:
            cur_mod, cur = m.group(1), None
            continue
        m = _WL.search(line)
        if m:
            cur = {"module": cur_mod, "gates": int(m.group(1)),
                   "abc_area": float(m.group(2)),
                   "delay_ps": float(m.group(3)), "path": [],
                   "start": None, "end": None}
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = _PATH.match(line)
        if m:
            cur["path"].append({"i": int(m.group(1)), "gate": m.group(5),
                                "arrival_ps": float(m.group(7))})
            continue
        m = _SE.search(line)
        if m:
            cur["start"], cur["end"] = m.group(2), m.group(4)
            cur["start_kind"] = "input/reg" if m.group(1).startswith("pi") \
                else m.group(1)
            cur["end_kind"] = "output/reg" if m.group(3).startswith("po") \
                else m.group(3)
    runs.sort(key=lambda r: r["delay_ps"], reverse=True)
    return runs


def parse_stat(stat_text: str, top: str) -> Dict[str, object]:
    """Area / cells / flops of ``top`` from ``stat -liberty -top`` output."""
    res: Dict[str, object] = {"area_um2": None, "cells": None, "flops": None}
    tail = stat_text.split("Design hierarchy")[-1] \
        if "Design hierarchy" in stat_text else stat_text
    m = (re.search(r"Chip area for (?:top )?module\s+'?\\?%s'?:\s*([\d.]+)"
                   % re.escape(top), stat_text)
         or re.search(r"Area for module\s+\\?%s:\s*([\d.]+)"
                      % re.escape(top), tail))
    if m:
        res["area_um2"] = float(m.group(1))
    # flattened: a single per-module block; its first "Number of cells"
    mc = re.findall(r"(?m)^\s+Number of cells:?\s+(\d+)", tail) or \
        re.findall(r"(?m)^\s+(\d+)\s+[\d.Ee+\-]+\s+cells\s*$", tail)
    if mc:
        res["cells"] = int(mc[0])
    flops = 0
    for line in tail.splitlines():
        toks = line.split()
        cell = next((t for t in toks if "sky130_fd_sc_hd__" in t), None)
        if not cell or not _FLOP_RE.search(cell):
            continue
        nums = [t for t in toks if re.fullmatch(r"\d+", t)]
        if nums:
            flops += int(nums[0])
    res["flops"] = flops
    return res


def _owner(sig: Optional[str]) -> str:
    """Instance path of a flattened signal name: ``\\u_add.x [3]`` -> u_add;
    a name without a dot is the top module's own register or port."""
    if not sig:
        return "?"
    s = sig.lstrip("\\").split(" [")[0]
    if s.startswith("$"):
        return "(yosys-generated net in top)"
    return s.rsplit(".", 1)[0] if "." in s else "(top)"


def decide(meas: Dict[str, object], target_ps: float,
           area_base: Optional[float] = None,
           area_max_growth: Optional[float] = None,
           area_min_ratio: Optional[float] = None) -> Dict[str, object]:
    """Pure pass/fail.  ``meas`` = {"delay_ps", "area_um2", ...}.

    Pass iff the longest path is <= target AND (when a base area is given)
    the FU area is within [base*min_ratio, base*(1+max_growth)].  The lower
    bound is the relocation guard: an FU that shrank by a fifth has not been
    pipelined, its datapath has moved somewhere this judge does not look."""
    reasons: List[str] = []
    delay = meas.get("delay_ps")
    if delay is None:
        return {"pass": False, "timing_pass": False, "area_pass": False,
                "reasons": ["no abc delay found (synthesis failed?)"],
                "target_ps": target_ps}
    timing_pass = float(delay) <= float(target_ps)
    if not timing_pass:
        reasons.append(f"longest path {delay / 1000:.2f} ns > target "
                       f"{target_ps / 1000:.2f} ns")
    area = meas.get("area_um2")
    area_pass = True
    growth = None
    if area_base:
        if area is None:
            area_pass = False
            reasons.append("area not found in stat output")
        else:
            growth = float(area) / float(area_base) - 1.0
            if area_max_growth is not None and growth > area_max_growth:
                area_pass = False
                reasons.append(f"area {area:,.0f} um2 is {growth:+.1%} vs "
                               f"base {area_base:,.0f} (limit "
                               f"+{area_max_growth:.0%})")
            if area_min_ratio is not None and \
                    float(area) < float(area_base) * area_min_ratio:
                area_pass = False
                reasons.append(f"area {area:,.0f} um2 is below "
                               f"{area_min_ratio:.0%} of base "
                               f"{area_base:,.0f}: the datapath has left "
                               f"the FU")
    return {"pass": timing_pass and area_pass, "timing_pass": timing_pass,
            "area_pass": area_pass, "reasons": reasons,
            "target_ps": float(target_ps), "delay_ps": float(delay),
            "slack_ps": float(target_ps) - float(delay),
            "area_um2": area, "area_base_um2": area_base,
            "area_growth": growth, "area_max_growth": area_max_growth,
            "area_min_ratio": area_min_ratio}


def summarize_run(run: Dict[str, object]) -> Dict[str, object]:
    path = run.get("path") or []
    gates = Counter(p["gate"].replace("sky130_fd_sc_hd__", "")
                    for p in path if p["i"] > 0)
    return {"module": run.get("module"), "delay_ps": run.get("delay_ps"),
            "logic_depth": max(0, len(path) - 1), "gates": run.get("gates"),
            "start": run.get("start"), "end": run.get("end"),
            "start_owner": _owner(run.get("start")),
            "end_owner": _owner(run.get("end")),
            "path_gate_mix": dict(gates.most_common(8)),
            # the arrival profile, every ~10% of the path, so a reader can
            # see where along the chain the time goes
            "arrival_profile_ps": [round(p["arrival_ps"], 1) for p in
                                   path[::max(1, len(path) // 10)]]}


def format_feedback(verdict: Dict[str, object],
                    crit: Dict[str, object], top: str,
                    report_path: str = "") -> str:
    """The message the RTL agent reads (and the loop dumps)."""
    if verdict.get("infra_failure") or verdict.get("harness_error"):
        rs = "; ".join(verdict.get("reasons") or [])
        return (f"## Timing judge (round 11): HARNESS FAILURE, not a "
                f"verdict\n\n- {rs}\n- Nothing about your design was "
                f"measured; do not change the RTL because of this.  "
                f"{('Report: ' + report_path) if report_path else ''}\n")
    ok = "PASS" if verdict.get("pass") else "FAIL"
    lines = [f"## Timing judge (round 11): {ok}", ""]
    t = verdict.get("target_ps", 0) / 1000
    d = (verdict.get("delay_ps") or 0) / 1000
    lines.append(f"- `{top}` standalone, flattened, sky130_fd_sc_hd tt_025C_1v80, "
                 f"abc -D {PERIOD_PS}: longest register/port-to-register/port "
                 f"path **{d:.2f} ns** vs target **{t:.2f} ns** "
                 f"(slack {t - d:+.2f} ns).")
    if verdict.get("area_um2") is not None:
        g = verdict.get("area_growth")
        lines.append(f"- FU area {verdict['area_um2']:,.0f} um2"
                     + (f" ({g:+.1%} vs round-10 base "
                        f"{verdict['area_base_um2']:,.0f}; allowed "
                        f"{-(1 - (verdict.get('area_min_ratio') or 0)):+.0%} "
                        f"to +{(verdict.get('area_max_growth') or 0):.0%})"
                        if g is not None else ""))
    if crit:
        lines += [f"- critical path: logic depth {crit['logic_depth']} cells",
                  f"  - start: `{crit.get('start')}`  (in {crit['start_owner']})",
                  f"  - end:   `{crit.get('end')}`  (in {crit['end_owner']})",
                  f"  - gate mix on the path: {crit.get('path_gate_mix')}",
                  f"  - arrival profile (ps, ~every 10% of the path): "
                  f"{crit.get('arrival_profile_ps')}"]
    for r in verdict.get("reasons") or []:
        lines.append(f"- FAIL reason: {r}")
    if report_path:
        lines.append(f"- full report: {report_path} (timing.json, yosys.log "
                     f"-- grep 'ABC: Path' for the gate-by-gate path)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------
def _sv2v(files: Sequence[str], dest: str) -> List[str]:
    os.makedirs(dest, exist_ok=True)
    for f in os.listdir(dest):
        os.remove(os.path.join(dest, f))
    r = subprocess.run([SV2V, "--write=" + dest, "-DSYNTHESIS", *files],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("sv2v failed:\n" + (r.stdout + r.stderr)[-4000:])
    return sorted(os.path.join(dest, f) for f in os.listdir(dest)
                  if f.endswith(".v"))


# ---- isolation / watchdog / serialisation (round 34 fix) -------------------
#: r34: yosys's internal abc wrote output.blif and yosys never read it back
#: when run as a child of a Ray worker (killed at 3600 s; the identical run
#: from a shell takes ~6 min).  Exact trigger unproven, so yosys now runs
#: fully detached from the worker's process state.
TIMEOUT_S = int(os.environ.get("TITAN_TIMING_TIMEOUT_S", "900"))
STALL_S = int(os.environ.get("TITAN_TIMING_STALL_S", "120"))
LOCK_PATH = os.environ.get(
    "TITAN_TIMING_LOCK",
    "/share1/saves/max410011/titan_scratch/timing_judge.lock")
LOCK_WAIT_S = int(os.environ.get("TITAN_TIMING_LOCK_WAIT_S", "3600"))
#: Only these survive into yosys's environment (no RAY_*, OMP_*, ...).
_ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL",
             "LD_LIBRARY_PATH", "PPA2_LIB")
REASON_STALL = "abc handoff stall"


def clean_env(environ: Dict[str, str], tmp: str) -> Dict[str, str]:
    """Scrubbed environment for yosys: whitelist + TMPDIR under ``tmp``."""
    env = {k: v for k, v in environ.items()
           if k in _ENV_KEEP or k.startswith("LC_")}
    env.update(TMPDIR=tmp, TMP=tmp, TEMP=tmp)
    # ROOT CAUSE of the r34 hang (bisected 2026-09-30): with TERM unset --
    # as in a Ray worker started from ssh/nohup -- yosys-abc sits in
    # pselect() on its stdin after the script and yosys blocks in read()
    # on abc's stdout forever.  A real TERM (xterm) fixes it; TERM=dumb does not.
    env["TERM"] = "xterm"
    return env


def _isolate() -> None:                      # runs in the child, pre-exec
    import signal
    for sig in range(1, signal.NSIG):
        if sig in (signal.SIGKILL, signal.SIGSTOP):
            continue
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError, RuntimeError):
            pass
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    if os.environ.get("TITAN_TIMING_RAISE_NOFILE", "1") != "0":
        # A Ray worker's soft RLIMIT_NOFILE can be 1024 (shell: 1048576);
        # yosys/abc inherit it.  Raise it to the hard limit.
        import resource
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (OSError, ValueError):
            pass


def stall_decision(blif_since: Optional[float], last_growth: float,
                   now: float, stall_s: float = STALL_S) -> bool:
    """Pure watchdog step.  ``blif_since`` is when abc's output.blif was
    first seen (None: absent), ``last_growth`` when yosys.log last grew.
    Stalled iff the blif has been there, and the log silent, for more than
    ``stall_s`` -- the clock starts at the later of the two, because the log
    is legitimately silent for the whole (~4 min) abc run *before* the blif
    appears."""
    return blif_since is not None and now - max(blif_since, last_growth) \
        > stall_s


def is_infra_failure(res: Optional[Dict[str, object]]) -> bool:
    """True when a judge result is a harness failure, not a verdict."""
    v = (res or {}).get("verdict") or {}
    return bool(v.get("infra_failure") or v.get("harness_error"))


class _Lock:
    """Cluster-wide (one host: head_local) mutex on timing synthesis."""

    def __init__(self, path: str, wait_s: float):
        self.path, self.wait_s, self.fh, self.waited = (os.path.abspath(path), wait_s,
                                                    None, 0.0)

    def __enter__(self):
        import fcntl
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a+")
        t0 = time.time()
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() - t0 > self.wait_s:
                    self.fh.close()
                    raise TimeoutError(
                        f"timing lock {self.path} busy for {self.wait_s:.0f}s")
                time.sleep(3)
        self.waited = round(time.time() - t0, 1)
        return self

    def __exit__(self, *a):
        try:
            self.fh.close()          # closing drops the flock
        except OSError:
            pass


def _tail(path: str, n: int = 1500) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def _run_yosys(ys: str, log: str, tmp: str, cwd: str, nice: int,
               timeout_s: float, stall_s: float) -> Dict[str, object]:
    """Run yosys detached; returns {"rc", "infra": reason|None}."""
    import glob
    import signal
    try:
        os.remove(log)
    except OSError:
        pass
    proc = subprocess.Popen(
        ["nice", "-n", str(nice), YOSYS, "-l", log, "-s", ys],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=clean_env(os.environ, tmp), cwd=cwd,
        start_new_session=True, preexec_fn=_isolate)
    t0 = last_growth = time.time()
    prev, infra, blif_since = -1, None, None
    while proc.poll() is None:
        time.sleep(2)
        now = time.time()
        if now - t0 > timeout_s:
            infra = f"timeout after {timeout_s:.0f}s"
        else:
            try:
                size = os.path.getsize(log)
            except OSError:
                size = 0
            blif = bool(glob.glob(os.path.join(tmp, "yosys-abc-*",
                                               "output.blif")))
            if size != prev:
                last_growth, prev = now, size
            if not blif:
                blif_since = None
            elif blif_since is None:
                blif_since = now
            if stall_decision(blif_since, last_growth, now, stall_s):
                infra = REASON_STALL
        if infra:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
            break
    return {"rc": proc.returncode, "infra": infra}


def synth(rtl_dir: str, top: str, out_dir: str, *, flatten: bool = True,
          nice: int = 19, timeout_s: Optional[float] = None,
          stall_s: Optional[float] = None) -> Dict[str, object]:
    """sv2v + yosys; returns {"rc", "wall_s", "log", "stat", "infra"}.  All
    temporaries (abc temp dirs, the liberty scl cache) go under
    ``out_dir/tmp`` via TMPDIR and are removed afterwards.  ``infra`` is a
    reason string when the harness (not the design) failed: timeout, abc
    handoff stall, lock wait timeout -- else None."""
    timeout_s = TIMEOUT_S if timeout_s is None else timeout_s
    stall_s = STALL_S if stall_s is None else stall_s
    out_dir, rtl_dir = os.path.abspath(out_dir), os.path.abspath(rtl_dir)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    files = _sv2v(closure(rtl_dir, [top]), os.path.join(out_dir, "sv2v"))
    macros = sorted(os.path.basename(f)[:-2] for f in files
                    if f.endswith("_ext.v"))
    ys = os.path.join(out_dir, "synth.ys")
    stat = os.path.join(out_dir, "stat.txt")
    with open(ys, "w") as f:
        for p in files:
            f.write(f"read_verilog {p}\n")
        f.write(f"hierarchy -top {top} -check\n")
        for m in macros:
            f.write(f"blackbox {m}\n")
        f.write(f"synth -top {top}{' -flatten' if flatten else ''}\n")
        f.write(f"dfflibmap -liberty {LIB}\n")
        f.write(f"abc -liberty {LIB} -D {PERIOD_PS} -script \"{ABC_SCRIPT}\"\n")
        f.write("setundef -zero\nopt_clean -purge\n")
        f.write(f"tee -o {stat} stat -liberty {LIB} -top {top}\n")
    log = os.path.join(out_dir, "yosys.log")
    if os.path.exists(log):                    # keep a retry's first log
        os.replace(log, os.path.join(out_dir, "yosys.prev.log"))
    waited, infra, rc = 0.0, None, -9
    t0 = time.time()
    try:
        with _Lock(LOCK_PATH, LOCK_WAIT_S) as lk:
            waited = lk.waited
            t0 = time.time()
            r = _run_yosys(ys, log, tmp, out_dir, nice, timeout_s, stall_s)
            rc, infra = r["rc"], r["infra"]
    except TimeoutError as exc:
        infra = f"timing lock wait timeout: {exc}"
    wall = round(time.time() - t0, 1)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(os.path.join(out_dir, "sv2v"), ignore_errors=True)
    if infra is None and rc is not None and rc < 0:
        infra = f"yosys killed by signal {-rc}"
    return {"rc": rc, "wall_s": wall, "lock_wait_s": waited, "log": log,
            "stat": stat, "n_rtl_files": len(files), "infra": infra,
            "log_tail": _tail(log) if infra else ""}


def judge(rtl_dir: str, out_dir: str, *, top: str = "MatrixFPMultiplyPipe",
          target_ns: float, area_base: Optional[float] = None,
          area_max_growth: Optional[float] = None,
          area_min_ratio: Optional[float] = None,
          flatten: bool = True, reparse: bool = False) -> Dict[str, object]:
    """Synthesize, parse, decide.  Writes ``out_dir/timing.json`` and
    ``out_dir/feedback.md``; returns the same dict (with ``feedback``).
    ``reparse`` skips synthesis and re-judges an existing ``yosys.log`` /
    ``stat.txt`` in ``out_dir`` (e.g. with a different target)."""
    out_dir = os.path.abspath(out_dir)
    if reparse:
        s = {"rc": 0, "wall_s": None, "reparsed": True,
             "log": os.path.join(out_dir, "yosys.log"),
             "stat": os.path.join(out_dir, "stat.txt")}
    else:
        s = synth(rtl_dir, top, out_dir, flatten=flatten)
    txt = open(s["log"], errors="replace").read() \
        if os.path.exists(s["log"]) else ""
    runs = parse_abc(txt)
    st = parse_stat(open(s["stat"], errors="replace").read(), top) \
        if os.path.exists(s["stat"]) else {"area_um2": None}
    crit = summarize_run(runs[0]) if runs else {}
    meas = {"delay_ps": runs[0]["delay_ps"] if runs else None,
            "area_um2": st.get("area_um2")}
    if s["rc"] != 0:
        meas["delay_ps"] = None
    verdict = decide(meas, target_ns * 1000.0, area_base, area_max_growth,
                     area_min_ratio)
    if s["rc"] != 0:
        verdict["reasons"].insert(0, f"yosys rc={s['rc']} (see {s['log']})")
    if s.get("infra"):
        # A harness failure is not a timing FAIL: flag it so the loop can
        # retry once and, failing that, tell the agent it is not a verdict.
        verdict["infra_failure"] = True
        verdict["infra_reason"] = s["infra"]
        verdict["reasons"].insert(
            0, f"HARNESS FAILURE (not a verdict on the design): {s['infra']}")
    res = {"top": top, "rtl_dir": rtl_dir, "flatten": flatten,
           "liberty": LIB, "period_ps": PERIOD_PS, "abc_script": ABC_SCRIPT,
           "synth": s, "stat": st, "critical": crit,
           "modules_top5": [{"module": r["module"], "delay_ps": r["delay_ps"]}
                            for r in runs[:5]],
           "verdict": verdict}
    res["feedback"] = format_feedback(verdict, crit, top, report_path=out_dir)
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(out_dir, "feedback.md"), "w") as f:
        f.write(res["feedback"])
    return res


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rtl_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--top", default="MatrixFPMultiplyPipe")
    ap.add_argument("--target-ns", type=float, required=True)
    ap.add_argument("--area-base", type=float, default=None)
    ap.add_argument("--area-max-growth", type=float, default=None)
    ap.add_argument("--area-min-ratio", type=float, default=None)
    ap.add_argument("--hier", action="store_true",
                    help="hierarchical (ppa2..ppa7 style), for comparison "
                         "only -- not a valid judge")
    ap.add_argument("--reparse", action="store_true",
                    help="do not synthesize; re-judge OUT_DIR/yosys.log")
    a = ap.parse_args(list(argv) if argv is not None else None)
    res = judge(a.rtl_dir, a.out_dir, top=a.top, target_ns=a.target_ns,
                area_base=a.area_base, area_max_growth=a.area_max_growth,
                area_min_ratio=a.area_min_ratio, flatten=not a.hier,
                reparse=a.reparse)
    print(res["feedback"])
    print(json.dumps({"verdict": res["verdict"],
                      "wall_s": res["synth"]["wall_s"],
                      "stat": res["stat"]}, indent=1, default=str))
    return 0 if res["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
