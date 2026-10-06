#!/usr/bin/env python3
"""Round 12 PPA judge ("stage P"): area / delay of the IME datapath FUs.

Derived from ``timing_judge.py`` (round 11) and sharing its hardened
machinery rather than copying it: the sv2v closure walk, the abc / stat
parsers, the isolated yosys subprocess (TERM=xterm, scrubbed env, process
group kill, stall watchdog, ``fcntl`` lock) and the ``infra_failure`` verdict
convention all live in ``timing_judge`` and are imported from there.  What is
new here:

* four synthesis runs, same frozen flow as ppa2..ppa8 / timing_judge
  (sv2v, ``synth``, ``dfflibmap``, ``abc -liberty -D 10000``, sky130 hd tt):

  =======  ==========================  ==============================
  P-FP     MatrixFPMultiplyPipe flat   binding: area, delay
  P-INT    MatrixMultiplyPipe   flat   binding: area, delay
  P-ATTR   MatrixFPMultiplyPipe hier   feedback only (submodule table)
  P-SOC    SaturnShuttleUnit    hier   Gate only, binding, memory-checked
  =======  ==========================  ==============================

* the pass criteria are *configuration* (``--ppa-config`` JSON, see
  ``DEFAULT_CONFIG``): calibration fills in the numbers (r35 INT area, the
  noise band, the final area target) without touching code.  The round-11
  0.8x area floor is gone -- shrinking is the goal.  The relocation guard is
  the file allowlist (``ppa_scope``) plus the Gate SoC check.
* netlist sanity on the written netlist: only ``sky130_fd_sc_hd__`` cells, no
  latches, no unmapped ``$`` cells, same top-level port list and widths as
  the reference (a stubbed-out multiplier shows up as a port or cell change).
* a flow stamp (sha256 of this script, timing_judge.py, the liberty and the
  abc script, plus the yosys / sv2v versions) in every verdict; when the
  configuration pins an expected stamp, a mismatch is a harness failure.
* non-gating power proxies: flop count, enable-flop count, liberty leakage
  sum.  (True dynamic power needs a gate-level activity flow the host does not
  have -- plan section 3.7.)

Same caveats as ppa7 section 6: abc's number is a mapped-logic-depth
estimate, not sign-off STA, typical corner, no wire load; relative, same-flow
only.

Usage (host, conda env; never /tmp):
    ppa_judge.py RTL_DIR OUT_DIR --ppa-config cfg.json [--mode iter|soc]
                 [--reparse]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import timing_judge as tj

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, object] = {
    "fp_top": "MatrixFPMultiplyPipe",
    "int_top": "MatrixMultiplyPipe",
    "soc_top": "SaturnShuttleUnit",
    # --- FP FU (binding) --------------------------------------------------
    "a_target_um2": 450000.0,          # -20% of r35 flat (561,908)
    "a_stretch_um2": 400000.0,         # loop stops here (calibrated: C2/C3)
    "r35_fp_area_um2": 561907.664,     # flat; stage-T record of r35
    "r35_fp_delay_ns": 25.98,
    "fp_delay_max_ns": 30.25,          # = SoC longest abc path (ppa8); abc noise up to +3.2 ns (J2/M4)
    # --- INT FU (binding) -------------------------------------------------
    "int_delay_max_ns": 30.25,         # required, = SoC longest abc path (r35 INT flat 31.374 ns)
    "int_delay_stretch_ns": 27.0,      # non-gating stretch / reported only
    "r35_int_area_um2": 102314.4,      # calibration C1
    "int_area_max_ratio": 1.05,
    "r35_int_delay_ns": 31.374,
    # --- noise ------------------------------------------------------------
    # progress / SoC-guard band: max(frac * reference, 2 * sigma)
    "noise_area_frac": 0.01,
    "noise_area_sigma_um2": None,      # calibration C3 fills this in
    "noise_delay_ns": 0.5,
    # --- netlist sanity: {top: {port: [dir, width]}}; {} = not checked ------
    "ref_ports": {},
    # --- Gate (P-SOC) -----------------------------------------------------
    "soc": {
        "enabled": True,
        "noise_area_frac": 0.005,            # SoC total / outside-FU band
        "r35_total_um2": 4160821.808,        # ppa8, hierarchical
        "r35_outside_fu_um2": 3259970.32,    # ppa8: total minus both FU subtrees
        "r35_modules_um2": {},               # {module: local area}
        "max_delay_ns": 30.25,
        "ime_module_max_delay_ns": 27.0,
        "min_free_gb": 40.0,
        "mem_wait_s": 1800,
        "timeout_s": 3600,
    },
    # --- cycle suite --------------------------------------------------------
    "cycles": {"enabled": True, "reference": None, "reference_path": None},
    # --- flow stamp pins (subset of flow_stamp() keys); None = record only ----
    "expected_stamp": None,
    # --- loop policy ----------------------------------------------------------
    "max_iters": 8,
    "cost_cap_usd": 250.0,
    "no_progress_stop": 3,
    "runs_per_turn": 3,
}


class ConfigError(ValueError):
    pass


def _merge(base: Dict[str, object], over: Dict[str, object], path: str = ""
           ) -> Dict[str, object]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise ConfigError(f"unknown ppa config key {path}{k!r}")
        if isinstance(base[k], dict) and isinstance(v, dict) \
                and k not in ("ref_ports", "r35_modules_um2", "reference"):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(src: Optional[object] = None) -> Dict[str, object]:
    """Config from a path, a dict, or None (defaults).  Unknown keys are an
    error (a typo must not silently fall back to a default target)."""
    over: Dict[str, object] = {}
    if isinstance(src, dict):
        over = src
    elif src:
        with open(str(src), encoding="utf-8") as fh:
            over = json.load(fh)
        if not isinstance(over, dict):
            raise ConfigError("ppa config must be a JSON object")
    cfg = _merge(DEFAULT_CONFIG, over)
    ref = cfg["cycles"].get("reference")
    path = cfg["cycles"].get("reference_path")
    if ref is None and path:
        with open(str(path), encoding="utf-8") as fh:
            cfg["cycles"]["reference"] = json.load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, object]) -> None:
    def pos(name, allow_none=False):
        v = cfg.get(name)
        if v is None and allow_none:
            return
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            raise ConfigError(f"{name} must be a positive number, got {v!r}")
    for n in ("a_target_um2", "a_stretch_um2", "fp_delay_max_ns",
              "int_delay_max_ns", "int_area_max_ratio", "noise_delay_ns",
              "max_iters", "cost_cap_usd", "runs_per_turn"):
        pos(n)
    for n in ("r35_int_area_um2", "r35_fp_area_um2", "r35_fp_delay_ns",
              "r35_int_delay_ns", "noise_area_sigma_um2"):
        pos(n, allow_none=True)
    if not 0 <= float(cfg["noise_area_frac"]) < 0.5:
        raise ConfigError("noise_area_frac must be in [0, 0.5)")
    if float(cfg["a_stretch_um2"]) > float(cfg["a_target_um2"]):
        raise ConfigError("a_stretch_um2 must not exceed a_target_um2")
    pos("int_delay_stretch_ns", allow_none=True)
    if cfg.get("int_delay_stretch_ns") is not None and \
            float(cfg["int_delay_stretch_ns"]) > float(cfg["int_delay_max_ns"]):
        raise ConfigError("int_delay_stretch_ns must not exceed "
                          "int_delay_max_ns")
    if int(cfg["no_progress_stop"]) < 0:
        raise ConfigError("no_progress_stop must be >= 0")


def noise_band_um2(cfg: Dict[str, object], ref_area: float) -> float:
    """max(frac * ref, 2 * sigma): the area difference below which two
    different-but-equivalent netlists are not distinguishable."""
    band = float(cfg["noise_area_frac"]) * float(ref_area)
    sigma = cfg.get("noise_area_sigma_um2")
    if sigma:
        band = max(band, 2.0 * float(sigma))
    return band


# --------------------------------------------------------------------------
# flow stamp
# --------------------------------------------------------------------------
_STAMP_CACHE: Dict[str, str] = {}


def _sha256_file(path: str) -> str:
    key = f"f:{path}"
    if key not in _STAMP_CACHE:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            _STAMP_CACHE[key] = h.hexdigest()
        except OSError:
            _STAMP_CACHE[key] = "unreadable"
    return _STAMP_CACHE[key]


def _version(cmd: Sequence[str]) -> str:
    key = "v:" + " ".join(cmd)
    if key not in _STAMP_CACHE:
        try:
            r = subprocess.run(list(cmd), capture_output=True, text=True,
                               timeout=30, env=dict(os.environ, TERM="xterm"))
            _STAMP_CACHE[key] = ((r.stdout or r.stderr).strip()
                                 .splitlines() or ["?"])[0]
        except (OSError, subprocess.SubprocessError):
            _STAMP_CACHE[key] = "unavailable"
    return _STAMP_CACHE[key]


def flow_stamp() -> Dict[str, str]:
    """Identity of the frozen flow.  Every verdict records it."""
    return {
        "ppa_judge_sha256": _sha256_file(os.path.abspath(__file__)),
        "timing_judge_sha256": _sha256_file(os.path.abspath(tj.__file__)),
        "liberty_sha256": _sha256_file(tj.LIB),
        "abc_script_sha256": hashlib.sha256(
            (tj.ABC_SCRIPT + f"|D={tj.PERIOD_PS}").encode()).hexdigest(),
        "yosys_version": _version([tj.YOSYS, "-V"]),
        "sv2v_version": _version([tj.SV2V, "--version"]),
    }


def stamp_mismatch(stamp: Dict[str, str],
                   expected: Optional[Dict[str, str]]) -> List[str]:
    """Keys of ``expected`` that ``stamp`` disagrees with ([] = ok)."""
    return [k for k, v in (expected or {}).items() if stamp.get(k) != v]


# --------------------------------------------------------------------------
# liberty leakage (power proxy)
# --------------------------------------------------------------------------
_LEAK_CACHE: Dict[str, Dict[str, float]] = {}
_CELL_RE = re.compile(r'^\s*cell\s*\(\s*"?([\w$]+)"?\s*\)')
_LEAK_RE = re.compile(r"^\s*cell_leakage_power\s*:\s*([\d.eE+\-]+)\s*;")


def liberty_leakage(lib_path: Optional[str] = None) -> Dict[str, float]:
    """{cell: leakage in nW} (the liberty's ``leakage_power_unit : 1nW``)."""
    lib_path = lib_path or tj.LIB
    if lib_path not in _LEAK_CACHE:
        out: Dict[str, float] = {}
        cur = None
        try:
            with open(lib_path, errors="replace") as fh:
                for line in fh:
                    m = _CELL_RE.match(line)
                    if m:
                        cur = m.group(1)
                        continue
                    m = _LEAK_RE.match(line)
                    if m and cur and cur not in out:
                        out[cur] = float(m.group(1))
        except OSError:
            pass
        _LEAK_CACHE[lib_path] = out
    return _LEAK_CACHE[lib_path]


# --------------------------------------------------------------------------
# stat parsing (flat cell histogram, hierarchical tree)
# --------------------------------------------------------------------------
_CELL_LINE = re.compile(r"^\s+(\d+)\s+[\d.Ee+\-]+\s+(\S+)\s*$")
_SEQ_RE = re.compile(r"sky130_fd_sc_hd__(?:e?df\w*|dl\w*|sdf\w*)$")
_ENFLOP_RE = re.compile(r"sky130_fd_sc_hd__edf\w*$")
_LATCH_RE = re.compile(r"sky130_fd_sc_hd__dl\w*$|\$_?(?:dlatch|DLATCH|SR)\w*",
                       re.I)


def parse_cell_hist(stat_text: str) -> Dict[str, int]:
    """{cell type: count} of the first (flat) module block of a
    ``stat -liberty`` dump.  Submodule rows are not cells; in a flattened
    design there are none."""
    hist: Dict[str, int] = {}
    for line in stat_text.splitlines():
        if "Chip area for" in line:
            break
        m = _CELL_LINE.match(line)
        if m and ("sky130_fd_sc_hd__" in m.group(2) or m.group(2).startswith(
                ("$", "\\$"))):
            hist[m.group(2)] = hist.get(m.group(2), 0) + int(m.group(1))
    return hist


def power_proxies(hist: Dict[str, int],
                  leak: Optional[Dict[str, float]] = None
                  ) -> Dict[str, object]:
    """Non-gating power proxies from a cell histogram."""
    leak = liberty_leakage() if leak is None else leak
    flops = sum(n for c, n in hist.items() if _SEQ_RE.match(c))
    enable = sum(n for c, n in hist.items() if _ENFLOP_RE.match(c))
    total_nw = sum(n * leak.get(c, 0.0) for c, n in hist.items())
    return {"flops": flops, "enable_flops": enable,
            "leakage_nW": round(total_nw, 3),
            "leakage_cells_unknown": sorted(c for c in hist
                                            if c not in leak)[:5]}


def parse_hier_stat(stat_text: str) -> Dict[str, Dict[str, object]]:
    """{module: {area, seq, flops, subs}} -- *local* figures (excluding
    submodules) per ``=== module ===`` block of a hierarchical
    ``stat -liberty`` dump (yosys >= 0.4x layout)."""
    mods: Dict[str, Dict[str, object]] = {}
    cur = None
    insub = False
    for line in stat_text.splitlines():
        m = re.match(r"=== (.+?) ===", line)
        if m:
            cur = m.group(1)
            if cur.startswith("design hierarchy"):
                cur = None
                continue
            mods[cur] = {"area": 0.0, "seq": 0.0, "flops": 0, "subs": {}}
            insub = False
            continue
        if cur is None:
            continue
        m = re.match(r"\s+Chip area for (?:top )?module '\\?(\S+)':\s*([\d.]+)",
                     line)
        if m:
            if m.group(1) in mods:
                mods[m.group(1)]["area"] = float(m.group(2))
            continue
        m = re.match(r"\s+of which used for sequential elements:\s*([\d.]+)",
                     line)
        if m:
            mods[cur]["seq"] = float(m.group(1))
            continue
        if re.match(r"\s+\d+\s+-\s+submodules\s*$", line):
            insub = True
            continue
        m = re.match(r"\s+(\d+)\s+-\s+(\S+)\s*$", line)
        if insub and m:
            mods[cur]["subs"][m.group(2)] = int(m.group(1))
            continue
        m = _CELL_LINE.match(line)
        if m and _SEQ_RE.match(m.group(2)):
            mods[cur]["flops"] += int(m.group(1))
    return mods


def _norm_name(n: str) -> str:
    return n.lstrip("\\")


def rollup(mods: Dict[str, Dict[str, object]], top: str
           ) -> Dict[str, Tuple[float, float, int]]:
    """{module: (total area, total seq area, total flops)} below each module
    (instances multiplied through)."""
    memo: Dict[str, Tuple[float, float, int]] = {}

    def tot(m: str) -> Tuple[float, float, int]:
        if m in memo:
            return memo[m]
        memo[m] = (0.0, 0.0, 0)          # cycle guard
        d = mods.get(m)
        if d is None:
            return memo[m]
        a, s, f = float(d["area"]), float(d["seq"]), int(d["flops"])
        for c, n in d["subs"].items():
            ca, cs, cf = tot(_norm_name(c))
            a, s, f = a + n * ca, s + n * cs, f + n * cf
        memo[m] = (a, s, f)
        return memo[m]
    for m in mods:
        tot(m)
    if top not in mods:
        raise KeyError(f"top {top} not in stat blocks")
    return memo


def instance_counts(mods: Dict[str, Dict[str, object]], top: str
                    ) -> Dict[str, int]:
    """{module: number of instances in the tree under ``top``} (top = 1)."""
    counts: Dict[str, int] = {}

    def walk(m: str, mult: int, depth: int = 0) -> None:
        counts[m] = counts.get(m, 0) + mult
        if depth > 64:
            return
        for c, n in (mods.get(m) or {"subs": {}})["subs"].items():
            walk(_norm_name(c), mult * n, depth + 1)
    walk(top, 1)
    return counts


def descendants(mods: Dict[str, Dict[str, object]], top: str) -> set:
    return set(instance_counts(mods, top))


def submodule_table(mods: Dict[str, Dict[str, object]], top: str,
                    n: int = 15) -> Dict[str, object]:
    """The hierarchical attribution table: per module *type*, its instance
    count, own (excluding children) area split seq / comb, and its
    contribution to the top's total.  Rows sum to the total (no double
    counting), sorted by contribution."""
    counts = instance_counts(mods, top)
    roll = rollup(mods, top)
    total = roll[top][0]
    rows = []
    for m, cnt in counts.items():
        d = mods.get(m)
        if d is None:
            continue
        own, seq = float(d["area"]), float(d["seq"])
        rows.append({"module": m, "instances": cnt, "own_each_um2": own,
                     "seq_each_um2": seq, "comb_each_um2": own - seq,
                     "flops_each": int(d["flops"]),
                     "contribution_um2": cnt * own,
                     "share": (cnt * own / total) if total else 0.0,
                     "subtree_each_um2": roll[m][0]})
    rows.sort(key=lambda r: r["contribution_um2"], reverse=True)
    return {"top": top, "total_um2": total, "total_flops": roll[top][2],
            "rows": rows[:n], "n_modules": len(rows)}


# --------------------------------------------------------------------------
# netlist parsing and sanity (yosys write_verilog -noattr, flat)
# --------------------------------------------------------------------------
_KEYWORDS = {"module", "endmodule", "input", "output", "inout", "wire", "reg",
             "assign", "always", "initial", "if", "else", "begin", "end",
             "parameter", "localparam", "case", "endcase", "generate",
             "endgenerate", "function", "endfunction", "task", "endtask",
             "integer", "genvar", "posedge", "negedge", "default"}
_PORT_DECL = re.compile(
    r"^\s*(input|output|inout)\s+(?:\[(\d+):(\d+)\]\s+)?(\\\S+|[^\s;]+)\s*;")
_INST_HEAD = re.compile(
    r"^\s*(\\?\$?[A-Za-z_][\w$.]*)\s+(\\\S+|[^\s(;]+)\s*\(\s*$")
_Q_CONN = re.compile(r"\.Q\((.*)\)\s*,?\s*$")


def flop_groups(names: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """Group register-bit names by base name and by name prefix.
    ``\\fpw_acc_reg [12]`` -> base ``fpw_acc_reg``; prefix drops a trailing
    ``_reg`` and any trailing digits (``s7_addb_bf`` stays, ``acc_3`` -> acc)."""
    base_n: Dict[str, int] = {}
    prefix_n: Dict[str, int] = {}
    for raw in names:
        n = _norm_name(raw.strip())
        n = re.sub(r"\s*\[\d+\]\s*$", "", n)
        if not n or n.startswith(("_", "$")) or re.fullmatch(r"_\d+_", n):
            n = "(unnamed)"
        base_n[n] = base_n.get(n, 0) + 1
        pre = re.sub(r"(?:_reg)?(?:_?\d+)?$", "", n) or n
        prefix_n[pre] = prefix_n.get(pre, 0) + 1
    return {"base": base_n, "prefix": prefix_n}


def parse_netlist(text: str) -> Dict[str, object]:
    """Ports, cell-type histogram and flop Q-net names of a flat netlist."""
    ports: Dict[str, List[object]] = {}
    hist: Dict[str, int] = {}
    qnames: List[str] = []
    cur_flop = False
    in_inst = False
    for line in text.splitlines():
        if in_inst:
            if cur_flop:
                m = _Q_CONN.search(line.strip())
                if m:
                    qnames.append(m.group(1).strip())
            if line.strip().startswith(");"):
                in_inst = False
            continue
        m = _PORT_DECL.match(line)
        if m:
            width = (abs(int(m.group(2)) - int(m.group(3))) + 1
                     if m.group(2) is not None else 1)
            ports[_norm_name(m.group(4))] = [m.group(1), width]
            continue
        m = _INST_HEAD.match(line)
        if m and m.group(1) not in _KEYWORDS:
            ctype = m.group(1)
            hist[ctype] = hist.get(ctype, 0) + 1
            cur_flop = bool(_SEQ_RE.match(ctype))
            in_inst = True
    return {"ports": ports, "cells": hist, "flop_q": qnames,
            "has_always_latch": bool(re.search(r"always_latch|\bdlatch\b",
                                               text, re.I))}


def netlist_sanity(info: Dict[str, object],
                   ref_ports: Optional[Dict[str, List[object]]] = None
                   ) -> Dict[str, object]:
    """§3.3 netlist checks.  ``issues`` empty = sane."""
    issues: List[str] = []
    hist: Dict[str, int] = info["cells"]
    foreign = {c: n for c, n in hist.items()
               if not c.startswith("sky130_fd_sc_hd__")}
    unmapped = {c: n for c, n in foreign.items() if c.lstrip("\\").startswith("$")}
    other = {c: n for c, n in foreign.items() if c not in unmapped}
    if unmapped:
        issues.append(f"unmapped yosys cells in the netlist: "
                      f"{dict(sorted(unmapped.items())[:6])}")
    if other:
        issues.append(f"cells that are not sky130_fd_sc_hd (black boxes / "
                      f"submodules in a flat netlist): "
                      f"{dict(sorted(other.items())[:6])}")
    latches = {c: n for c, n in hist.items() if _LATCH_RE.search(c)}
    if latches or info.get("has_always_latch"):
        issues.append(f"latches in the netlist: {latches or 'always_latch'}")
    if not hist:
        issues.append("no cells found in the netlist (empty / unparsable)")
    if ref_ports:
        got = info["ports"]
        missing = sorted(set(ref_ports) - set(got))
        extra = sorted(set(got) - set(ref_ports))
        changed = sorted(p for p in set(got) & set(ref_ports)
                         if [got[p][0], got[p][1]] !=
                         [ref_ports[p][0], ref_ports[p][1]])
        if missing or extra or changed:
            issues.append(
                "top-level port list differs from the reference: "
                f"missing {missing[:6]}, extra {extra[:6]}, "
                f"changed {changed[:6]}")
    return {"ok": not issues, "issues": issues,
            "ports_checked": bool(ref_ports)}


# --------------------------------------------------------------------------
# decisions (pure)
# --------------------------------------------------------------------------
def decide_unit_fp(cfg: Dict[str, object], meas: Dict[str, object]
                   ) -> Dict[str, object]:
    reasons: List[str] = []
    area, delay = meas.get("area_um2"), meas.get("delay_ps")
    ok_area = ok_delay = False
    if area is None or delay is None:
        reasons.append("FP FU: no area/delay found (synthesis failed?)")
    else:
        ok_area = float(area) <= float(cfg["a_target_um2"])
        if not ok_area:
            reasons.append(f"FP area {area:,.0f} um2 > target "
                           f"{float(cfg['a_target_um2']):,.0f} um2")
        lim = float(cfg["fp_delay_max_ns"]) * 1000.0
        ok_delay = float(delay) <= lim
        if not ok_delay:
            reasons.append(f"FP longest path {delay / 1000:.2f} ns > "
                           f"{lim / 1000:.2f} ns (= the SoC longest abc path, ppa8)")
    san = meas.get("sanity") or {"ok": True, "issues": []}
    for i in san.get("issues") or []:
        reasons.append(f"FP netlist sanity: {i}")
    return {"pass": ok_area and ok_delay and bool(san.get("ok", True)),
            "area_pass": ok_area, "delay_pass": ok_delay,
            "sanity_pass": bool(san.get("ok", True)), "reasons": reasons}


def decide_unit_int(cfg: Dict[str, object], meas: Dict[str, object]
                    ) -> Dict[str, object]:
    reasons: List[str] = []
    area, delay = meas.get("area_um2"), meas.get("delay_ps")
    ok_area = ok_delay = False
    area_checked = cfg.get("r35_int_area_um2") is not None
    if area is None or delay is None:
        reasons.append("INT FU: no area/delay found (synthesis failed?)")
    else:
        lim = float(cfg["int_delay_max_ns"]) * 1000.0
        ok_delay = float(delay) <= lim
        if not ok_delay:
            reasons.append(f"INT longest path {delay / 1000:.2f} ns > "
                           f"{lim / 1000:.2f} ns")
        if area_checked:
            cap = float(cfg["int_area_max_ratio"]) * float(
                cfg["r35_int_area_um2"])
            ok_area = float(area) <= cap
            if not ok_area:
                reasons.append(
                    f"INT area {area:,.0f} um2 > {cap:,.0f} um2 "
                    f"({float(cfg['int_area_max_ratio']):.2f} x r35 "
                    f"{float(cfg['r35_int_area_um2']):,.0f})")
        else:
            ok_area = True            # not configured yet: reported only
    san = meas.get("sanity") or {"ok": True, "issues": []}
    for i in san.get("issues") or []:
        reasons.append(f"INT netlist sanity: {i}")
    return {"pass": ok_area and ok_delay and bool(san.get("ok", True)),
            "area_pass": ok_area, "area_checked": area_checked,
            "delay_pass": ok_delay,
            "sanity_pass": bool(san.get("ok", True)), "reasons": reasons}


def decide_iter(cfg: Dict[str, object], fp: Dict[str, object],
                int_: Dict[str, object]) -> Dict[str, object]:
    """Combined P verdict of one iteration from the two binding units."""
    dfp = decide_unit_fp(cfg, fp)
    dint = decide_unit_int(cfg, int_)
    a_fp = fp.get("area_um2")
    return {"pass": dfp["pass"] and dint["pass"], "fp": dfp, "int": dint,
            "reasons": dfp["reasons"] + dint["reasons"],
            "a_fp_um2": a_fp, "a_int_um2": int_.get("area_um2"),
            "fp_delay_ps": fp.get("delay_ps"),
            "int_delay_ps": int_.get("delay_ps"),
            "stretch_met": bool(dfp["pass"] and dint["pass"]
                                and a_fp is not None
                                and float(a_fp) <= float(cfg["a_stretch_um2"]))}


def decide_soc(cfg: Dict[str, object], meas: Dict[str, object]
               ) -> Dict[str, object]:
    """Gate-only SoC checks.  ``meas``: total_um2, outside_fu_um2,
    modules {name: local area} outside the FU subtrees, new_modules,
    longest_ps, ime_longest_ps."""
    soc = cfg["soc"]
    reasons: List[str] = []
    if meas.get("total_um2") is None:
        return {"pass": False, "reasons": ["P-SOC: no total area found"]}
    base_total = soc.get("r35_total_um2")
    # total-level band stays 0.5% (1% of 4.16M um2 would hide ~42k of
    # relocated logic); the 1% band is the per-FU no-progress key
    noise = noise_band_um2(
        dict(cfg, noise_area_frac=soc.get("noise_area_frac", 0.005)),
        base_total or meas["total_um2"])
    # the plan's total bound: r35 total minus what the two FUs saved (same
    # hierarchical flow), plus noise == outside-FU area did not grow
    base_out = soc.get("r35_outside_fu_um2")
    if base_out is not None and meas.get("outside_fu_um2") is not None:
        if float(meas["outside_fu_um2"]) > float(base_out) + noise:
            reasons.append(
                f"SoC area outside the two FUs grew: "
                f"{meas['outside_fu_um2']:,.0f} > r35 {float(base_out):,.0f}"
                f" + noise {noise:,.0f} (logic relocated?)")
    if base_total is not None and float(meas["total_um2"]) > \
            float(base_total) + noise:
        reasons.append(f"SoC total {meas['total_um2']:,.0f} um2 exceeds the "
                       f"r35 total {float(base_total):,.0f} + noise "
                       f"{noise:,.0f}")
    ref_mods = soc.get("r35_modules_um2") or {}
    grew = []
    for m, area in (meas.get("modules") or {}).items():
        ref = ref_mods.get(m)
        if ref is None:
            continue
        band = noise_band_um2(cfg, max(float(ref), 1.0))
        if float(area) > float(ref) + max(band, 1.0):
            grew.append(f"{m} {area:,.0f} > {float(ref):,.0f}")
    if grew:
        reasons.append("modules outside the two FUs grew beyond the noise "
                       "band: " + "; ".join(grew[:6]))
    if meas.get("new_modules"):
        reasons.append("new modules outside the FU subtrees: "
                       f"{sorted(meas['new_modules'])[:6]}")
    lim = (float(soc["max_delay_ns"]) + float(cfg["noise_delay_ns"])) * 1000
    if meas.get("longest_ps") is not None and float(meas["longest_ps"]) > lim:
        reasons.append(f"SoC longest abc path {meas['longest_ps'] / 1000:.2f}"
                       f" ns > {lim / 1000:.2f} ns")
    ilim = float(soc["ime_module_max_delay_ns"]) * 1000
    if meas.get("ime_longest_ps") is not None and \
            float(meas["ime_longest_ps"]) > ilim:
        reasons.append("an IME module's path "
                       f"{float(meas['ime_longest_ps']) / 1000:.2f} ns > "
                       f"{ilim / 1000:.2f} ns")
    return {"pass": not reasons, "reasons": reasons}


# --------------------------------------------------------------------------
# progress / best tracking (pure; the loop owns one of these)
# --------------------------------------------------------------------------
class PpaTracker:
    """Best-clean and no-progress bookkeeping for the PPA loop.

    * ``best``: the fully clean iteration with the smallest A_FP (ties: the
      earliest) -- the last iteration is not automatically the best.
    * progress is measured on the *frontier* = the smallest A_FP of any
      judged attempt (clean or not): an attempt that gets area down but trips
      another check is still progress toward the goal.  An attempt improves
      only when it beats the frontier by more than the noise band.
    * only judged attempts count; infra failures and scope rejections never
      reach :meth:`observe`.
    """

    def __init__(self, cfg: Dict[str, object]):
        self.cfg = cfg
        self.limit = int(cfg["no_progress_stop"])
        self.frontier: Optional[float] = None
        self.streak = 0
        self.best: Optional[Dict[str, object]] = None
        self.history: List[Dict[str, object]] = []

    def observe(self, attempt: int, a_fp: Optional[float], clean: bool,
                **extra) -> Dict[str, object]:
        rec = {"attempt": attempt, "a_fp_um2": a_fp, "clean": bool(clean),
               **extra}
        improved = False
        if a_fp is not None:
            if self.frontier is None:
                improved = True
            else:
                improved = a_fp < self.frontier - noise_band_um2(
                    self.cfg, self.frontier)
            if self.frontier is None or a_fp < self.frontier:
                self.frontier = float(a_fp)
        # an attempt with no area figure (synthesis failed) is not progress
        self.streak = 0 if improved else self.streak + 1
        new_best = False
        if clean and a_fp is not None and (
                self.best is None or a_fp < self.best["a_fp_um2"]):
            self.best, new_best = rec, True
        rec.update(improved=improved, new_best=new_best, streak=self.streak)
        self.history.append(rec)
        stretch = bool(clean and a_fp is not None and
                       float(a_fp) <= float(self.cfg["a_stretch_um2"]))
        return {"improved": improved, "new_best": new_best,
                "streak": self.streak, "stretch": stretch,
                "stop_no_progress": bool(self.limit and
                                         self.streak >= self.limit)}


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def _read(path: str) -> str:
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def judge_unit(rtl_dir: str, out_dir: str, top: str, *, flatten: bool = True,
               ref_ports: Optional[Dict[str, List[object]]] = None,
               reparse: bool = False, timeout_s: Optional[float] = None
               ) -> Dict[str, object]:
    """One synthesis + parse of ``top``.  Returns the unit record (no
    pass/fail: that is :func:`decide_iter`)."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    netlist = os.path.join(out_dir, "netlist.v")
    if reparse:
        s = {"rc": 0, "wall_s": None, "reparsed": True, "infra": None,
             "log": os.path.join(out_dir, "yosys.log"),
             "stat": os.path.join(out_dir, "stat.txt")}
    else:
        extra = [f"write_verilog -noattr {netlist}"] if flatten else []
        s = tj.synth(rtl_dir, top, out_dir, flatten=flatten,
                     timeout_s=timeout_s, extra_ys=extra)
    log, stat_txt = _read(s["log"]), _read(s["stat"])
    runs = tj.parse_abc(log)
    unit: Dict[str, object] = {
        "top": top, "flatten": flatten, "synth": s, "dir": out_dir,
        "delay_ps": runs[0]["delay_ps"] if runs and s["rc"] == 0 else None,
        "critical": tj.summarize_run(runs[0]) if runs else {},
        "modules_top5": [{"module": r["module"], "delay_ps": r["delay_ps"]}
                         for r in runs[:5]],
        "infra": s.get("infra"),
    }
    st = tj.parse_stat(stat_txt, top) if stat_txt else {"area_um2": None}
    unit["area_um2"] = st.get("area_um2")
    unit["cells"] = st.get("cells")
    if flatten:
        hist = parse_cell_hist(stat_txt)
        unit["cell_hist_top"] = dict(sorted(hist.items(),
                                            key=lambda kv: -kv[1])[:12])
        unit.update(power_proxies(hist))
        nl_text = _read(netlist)
        info = parse_netlist(nl_text) if nl_text else {
            "ports": {}, "cells": {}, "flop_q": [], "has_always_latch": False}
        unit["ports"] = info["ports"]
        sanity = netlist_sanity(info, ref_ports)
        # stat's own histogram catches unmapped cells even if the netlist
        # write failed
        sanity_stat = netlist_sanity({"ports": info["ports"], "cells": hist,
                                      "has_always_latch": False}, None)
        if not nl_text:
            sanity["issues"].append("netlist.v missing (write_verilog "
                                    "did not run)")
        for i in sanity_stat["issues"]:
            if i not in sanity["issues"]:
                sanity["issues"].append(i)
        sanity["ok"] = not sanity["issues"]
        unit["sanity"] = sanity
        groups = flop_groups(info["flop_q"])
        unit["flop_prefix_top"] = dict(sorted(
            groups["prefix"].items(), key=lambda kv: -kv[1])[:15])
        unit["flop_base_top"] = dict(sorted(
            groups["base"].items(), key=lambda kv: -kv[1])[:15])
    else:
        mods = parse_hier_stat(stat_txt) if stat_txt else {}
        try:
            unit["table"] = submodule_table(mods, top) if mods else None
        except KeyError:
            unit["table"] = None
    return unit


def judge_iter(rtl_dir: str, out_dir: str, cfg: Dict[str, object], *,
               reparse: bool = False) -> Dict[str, object]:
    """Stage P for one iteration: P-FP, P-INT (binding), P-ATTR (feedback).
    Writes ``out_dir/ppa.json`` and ``feedback.md``; returns the result with
    ``verdict`` and ``feedback``."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = flow_stamp()
    refp = cfg.get("ref_ports") or {}
    fp_top, int_top = str(cfg["fp_top"]), str(cfg["int_top"])
    fp = judge_unit(rtl_dir, os.path.join(out_dir, "fp"), fp_top,
                    ref_ports=refp.get(fp_top), reparse=reparse)
    intu = judge_unit(rtl_dir, os.path.join(out_dir, "int"), int_top,
                      ref_ports=refp.get(int_top), reparse=reparse) \
        if not fp.get("infra") else {"infra": "skipped (P-FP failed)"}
    attr = None
    if not (fp.get("infra") or intu.get("infra")):
        attr = judge_unit(rtl_dir, os.path.join(out_dir, "attr"), fp_top,
                          flatten=False, reparse=reparse)
        if attr.get("infra"):
            attr = {"infra": attr["infra"], "table": None}   # feedback only
    verdict: Dict[str, object]
    infra = fp.get("infra") or intu.get("infra")
    mism = stamp_mismatch(stamp, cfg.get("expected_stamp"))
    if infra or mism:
        why = (f"HARNESS FAILURE (not a verdict on the design): {infra}"
               if infra else
               f"HARNESS FAILURE (not a verdict on the design): flow stamp "
               f"mismatch on {mism} -- the judge's frozen flow changed")
        verdict = {"pass": False, "infra_failure": True,
                   "infra_reason": infra or f"stamp mismatch {mism}",
                   "reasons": [why], "a_fp_um2": None, "a_int_um2": None}
    else:
        for u in (fp, intu):
            if u["synth"]["rc"] != 0:
                u["delay_ps"] = None
                u["area_um2"] = None if not u.get("area_um2") else \
                    u["area_um2"]
        verdict = decide_iter(cfg, fp, intu)
        for u, name in ((fp, "P-FP"), (intu, "P-INT")):
            if u["synth"]["rc"] != 0:
                verdict["reasons"].insert(
                    0, f"{name}: yosys rc={u['synth']['rc']} (see "
                       f"{u['synth'].get('log')})")
    res = {"cfg_targets": {k: cfg[k] for k in (
               "a_target_um2", "a_stretch_um2", "fp_delay_max_ns",
               "int_delay_max_ns", "int_area_max_ratio", "r35_int_area_um2")},
           "fp": fp, "int": intu, "attr": attr, "flow_stamp": stamp,
           "verdict": verdict}
    res["feedback"] = format_feedback(res, cfg, report_path=out_dir)
    with open(os.path.join(out_dir, "ppa.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(out_dir, "feedback.md"), "w") as f:
        f.write(res["feedback"])
    return res


# ---- Gate: full SoC ------------------------------------------------------
def wait_for_memory(min_gb: float, max_wait_s: float, poll_s: float = 30.0,
                    avail=None, sleep=time.sleep) -> Tuple[bool, Optional[float]]:
    """Block until ``min_gb`` GB are available (MemAvailable) or the wait
    runs out.  Returns (ok, last reading).  An unreadable /proc is treated as
    "cannot verify" = not ok."""
    if avail is None:
        import hostload
        avail = hostload.avail_gb
    t0 = time.time()
    last = avail()
    while last is None or last < min_gb:
        if time.time() - t0 >= max_wait_s:
            return False, last
        sleep(poll_s)
        last = avail()
    return True, last


def soc_measurements(stat_text: str, log_text: str, cfg: Dict[str, object]
                     ) -> Dict[str, object]:
    """Everything decide_soc needs, from a hierarchical SoC stat + log."""
    soc_top = str(cfg["soc_top"])
    fps, ints = str(cfg["fp_top"]), str(cfg["int_top"])
    mods = parse_hier_stat(stat_text)
    roll = rollup(mods, soc_top)
    counts = instance_counts(mods, soc_top)
    total = roll[soc_top][0]
    fu_mods = set()
    inside_fu_area = 0.0
    for fu in (fps, ints):
        if fu in mods:
            fu_mods |= descendants(mods, fu)
            inside_fu_area += counts.get(fu, 0) * roll[fu][0]
    outside = total - inside_fu_area
    ref_mods = (cfg["soc"].get("r35_modules_um2") or {})
    local = {m: float(d["area"]) for m, d in mods.items()
             if m not in fu_mods}
    new_mods = [m for m in local if ref_mods and m not in ref_mods
                and m in counts]
    runs = tj.parse_abc(log_text)
    longest = runs[0]["delay_ps"] if runs else None
    ime = [r["delay_ps"] for r in runs if _norm_name(str(r["module"])) in fu_mods]
    return {"total_um2": total, "outside_fu_um2": outside,
            "fu_um2": inside_fu_area, "modules": local,
            "new_modules": new_mods,
            "longest_ps": longest,
            "longest_module": runs[0]["module"] if runs else None,
            "ime_longest_ps": max(ime) if ime else None,
            "total_flops": roll[soc_top][2],
            "fu_tables": {fu: submodule_table(mods, fu, 8)
                          for fu in (fps, ints) if fu in mods}}


def soc_reference(stat_text: str, log_text: str, cfg: Dict[str, object]
                  ) -> Dict[str, object]:
    """The ``soc`` config block's reference fields, measured from a
    reference (unmodified r35) SoC synthesis: paste into --ppa-config."""
    m = soc_measurements(stat_text, log_text, cfg)
    return {"r35_total_um2": m["total_um2"],
            "r35_outside_fu_um2": m["outside_fu_um2"],
            "r35_modules_um2": {k: round(v, 3)
                                for k, v in sorted(m["modules"].items())}}


def ports_of_netlist(text: str) -> Dict[str, List[object]]:
    """{port: [dir, width]} of a flat netlist: the ``ref_ports`` entry for a
    reference (r35) FU, for --ppa-config."""
    return parse_netlist(text)["ports"]


def judge_soc(rtl_dir: str, out_dir: str, cfg: Dict[str, object], *,
              reparse: bool = False, avail=None) -> Dict[str, object]:
    """P-SOC (Gate only): hierarchical SaturnShuttleUnit, the ppa8 flow."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    soc = cfg["soc"]
    stamp = flow_stamp()
    verdict: Dict[str, object]
    meas: Dict[str, object] = {}
    s: Dict[str, object] = {}
    if reparse:
        s = {"rc": 0, "wall_s": None, "reparsed": True, "infra": None,
             "log": os.path.join(out_dir, "yosys.log"),
             "stat": os.path.join(out_dir, "stat.txt")}
    else:
        ok, got = wait_for_memory(float(soc["min_free_gb"]),
                                  float(soc["mem_wait_s"]), avail=avail)
        if not ok:
            s = {"rc": None, "infra": f"insufficient free memory for the "
                 f"SoC synthesis: {got} GB < {soc['min_free_gb']} GB after "
                 f"{soc['mem_wait_s']} s"}
        else:
            s = tj.synth(rtl_dir, str(cfg["soc_top"]), out_dir, flatten=False,
                         all_files=True, timeout_s=float(soc["timeout_s"]))
    if s.get("infra"):
        verdict = {"pass": False, "infra_failure": True,
                   "infra_reason": s["infra"],
                   "reasons": [f"HARNESS FAILURE (not a verdict on the "
                               f"design): {s['infra']}"]}
    else:
        stat_txt, log = _read(s["stat"]), _read(s["log"])
        if s["rc"] != 0 or not stat_txt:
            verdict = {"pass": False, "reasons": [
                f"P-SOC: yosys rc={s['rc']} / no stat output "
                f"(see {s['log']})"]}
        else:
            try:
                meas = soc_measurements(stat_txt, log, cfg)
                verdict = decide_soc(cfg, meas)
            except KeyError as exc:
                verdict = {"pass": False,
                           "reasons": [f"P-SOC: cannot parse stat: {exc}"]}
    mism = stamp_mismatch(stamp, cfg.get("expected_stamp"))
    if mism and not verdict.get("infra_failure"):
        verdict = {"pass": False, "infra_failure": True,
                   "infra_reason": f"stamp mismatch {mism}",
                   "reasons": [f"HARNESS FAILURE: flow stamp mismatch on "
                               f"{mism}"]}
    res = {"mode": "soc", "synth": s, "meas": {k: v for k, v in meas.items()
                                                if k != "modules"},
           "flow_stamp": stamp, "verdict": verdict}
    res["feedback"] = format_soc_feedback(res, cfg, out_dir)
    with open(os.path.join(out_dir, "ppa_soc.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(out_dir, "feedback.md"), "w") as f:
        f.write(res["feedback"])
    return res


# --------------------------------------------------------------------------
# feedback text
# --------------------------------------------------------------------------
def _k(v, nd=0) -> str:
    return "-" if v is None else f"{float(v):,.{nd}f}"


def _ns(ps) -> str:
    return "-" if ps is None else f"{float(ps) / 1000:.2f}"


def _pct(now, ref) -> str:
    if now is None or not ref:
        return ""
    return f" ({(float(now) / float(ref) - 1) * 100:+.1f}%)"


def metric_table(res: Dict[str, object], cfg: Dict[str, object],
                 best: Optional[Dict[str, object]] = None) -> str:
    """The metric table: this iteration vs r35 / best clean so far / target."""
    fp, intu, v = res.get("fp") or {}, res.get("int") or {}, \
        res.get("verdict") or {}
    best = best or {}

    def stat(ok):
        return "-" if ok is None else ("ok" if ok else "FAIL")
    dfp, dint = v.get("fp") or {}, v.get("int") or {}
    r35i = cfg.get("r35_int_area_um2")
    rows = [
        ("FP area, flat (um2)", _k(fp.get("area_um2")) +
         _pct(fp.get("area_um2"), cfg["r35_fp_area_um2"]),
         _k(cfg["r35_fp_area_um2"]), _k(best.get("a_fp_um2")),
         f"<= {_k(cfg['a_target_um2'])} (stretch {_k(cfg['a_stretch_um2'])})",
         stat(dfp.get("area_pass"))),
        ("FP longest path (ns)", _ns(fp.get("delay_ps")),
         "-" if cfg.get("r35_fp_delay_ns") is None
         else f"{cfg['r35_fp_delay_ns']:.2f}", "-",
         f"<= {cfg['fp_delay_max_ns']:.2f}", stat(dfp.get("delay_pass"))),
        ("INT area, flat (um2)", _k(intu.get("area_um2")) +
         _pct(intu.get("area_um2"), r35i), _k(r35i), "-",
         ("<= %s (%.2f x r35)" % (_k(float(r35i) * float(
             cfg["int_area_max_ratio"])), cfg["int_area_max_ratio"]))
         if r35i else "(not gated: r35 INT area not calibrated)",
         stat(dint.get("area_pass") if dint.get("area_checked") else None)),
        ("INT longest path (ns)", _ns(intu.get("delay_ps")),
         "-" if cfg.get("r35_int_delay_ns") is None
         else f"{cfg['r35_int_delay_ns']:.2f}", "-",
         f"<= {cfg['int_delay_max_ns']:.2f}" + (
             f" (stretch {cfg['int_delay_stretch_ns']:.2f}, not gated)"
             if cfg.get("int_delay_stretch_ns") else ""),
         stat(dint.get("delay_pass"))),
        ("FP flops / enable flops (info)",
         f"{fp.get('flops', '-')} / {fp.get('enable_flops', '-')}", "-", "-",
         "report only", "-"),
        ("FP leakage, liberty sum (nW, info)", _k(fp.get("leakage_nW"), 1),
         "-", "-", "report only", "-"),
    ]
    lines = ["| metric | now | r35 | best clean so far | target | status |",
             "|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def submodule_table_text(table: Optional[Dict[str, object]]) -> str:
    if not table or not table.get("rows"):
        return "(hierarchical attribution not available this iteration)\n"
    lines = [f"Hierarchical (P-ATTR, feedback only -- hierarchical numbers "
             f"overstate the flat area): total {_k(table['total_um2'])} um2, "
             f"{table['total_flops']} flops, {table['n_modules']} module "
             f"types.", "",
             "| module | inst | own um2 each (seq / comb) | contribution um2 "
             "| share |", "|---|---:|---|---:|---:|"]
    for r in table["rows"]:
        lines.append(f"| {r['module']} | {r['instances']} | "
                     f"{_k(r['own_each_um2'])} ({_k(r['seq_each_um2'])} / "
                     f"{_k(r['comb_each_um2'])}) | "
                     f"{_k(r['contribution_um2'])} | {r['share']:.1%} |")
    return "\n".join(lines) + "\n"


def format_feedback(res: Dict[str, object], cfg: Dict[str, object],
                    best: Optional[Dict[str, object]] = None,
                    report_path: str = "") -> str:
    """The stage-P message the RTL agent reads."""
    v = res.get("verdict") or {}
    if v.get("infra_failure") or v.get("harness_error"):
        rs = "; ".join(v.get("reasons") or [])
        return (f"## PPA judge (round 12): HARNESS FAILURE, not a verdict\n\n"
                f"- {rs}\n- Nothing about your design was measured; do not "
                f"change the RTL because of this.  "
                f"{('Report: ' + report_path) if report_path else ''}\n")
    ok = "PASS" if v.get("pass") else "FAIL"
    lines = [f"## PPA judge (round 12): {ok}", "",
             "sky130_fd_sc_hd tt_025C_1v80, yosys + abc -D 10000, standalone "
             "FUs flattened (a stage wrapped in its own Module hides "
             "nothing). Areas are mapped-cell area; delays are abc's "
             "mapped-logic estimate (relative, not sign-off).", "",
             metric_table(res, cfg, best)]
    fp = res.get("fp") or {}
    crit = fp.get("critical") or {}
    if crit:
        lines += ["### FP critical path (flat)",
                  f"- logic depth {crit.get('logic_depth')} cells",
                  f"  - start: `{crit.get('start')}` (in "
                  f"{crit.get('start_owner')})",
                  f"  - end:   `{crit.get('end')}` (in "
                  f"{crit.get('end_owner')})",
                  f"  - gate mix: {crit.get('path_gate_mix')}",
                  f"  - arrival profile (ps): {crit.get('arrival_profile_ps')}",
                  ""]
    icrit = (res.get("int") or {}).get("critical") or {}
    if icrit:
        lines += ["### INT critical path (flat)",
                  f"- logic depth {icrit.get('logic_depth')} cells, start "
                  f"`{icrit.get('start')}`, end `{icrit.get('end')}`", ""]
    attr = res.get("attr") or {}
    lines += ["### FP submodule table", submodule_table_text(
        attr.get("table")), ""]
    pre = fp.get("flop_prefix_top") or {}
    if pre:
        lines += ["### FP flops by register-name prefix (flat netlist, top "
                  "15)",
                  ", ".join(f"{k}: {n}" for k, n in pre.items()), ""]
    for tag, u in (("FP", fp), ("INT", res.get("int") or {})):
        san = (u.get("sanity") or {}).get("issues") or []
        for i in san:
            lines.append(f"- {tag} netlist sanity FAIL: {i}")
    for r in v.get("reasons") or []:
        lines.append(f"- FAIL reason: {r}")
    if report_path:
        lines.append(f"- full report: {report_path} (ppa.json, fp/ int/ "
                     f"attr/ yosys.log, netlist.v)")
    return "\n".join(lines) + "\n"


def format_soc_feedback(res: Dict[str, object], cfg: Dict[str, object],
                        report_path: str = "") -> str:
    v = res.get("verdict") or {}
    if v.get("infra_failure"):
        return ("## PPA Gate (P-SOC): HARNESS FAILURE, not a verdict\n\n- "
                + "; ".join(v.get("reasons") or []) + "\n")
    m = res.get("meas") or {}
    soc = cfg["soc"]
    lines = [f"## PPA Gate, full SoC (P-SOC): "
             f"{'PASS' if v.get('pass') else 'FAIL'}", "",
             f"- SoC total {_k(m.get('total_um2'))} um2 (r35 "
             f"{_k(soc.get('r35_total_um2'))}{_pct(m.get('total_um2'), soc.get('r35_total_um2'))})",
             f"- outside the two FUs {_k(m.get('outside_fu_um2'))} um2 "
             f"(r35 {_k(soc.get('r35_outside_fu_um2'))})",
             f"- FU subtrees {_k(m.get('fu_um2'))} um2, longest abc path "
             f"{_ns(m.get('longest_ps'))} ns in {m.get('longest_module')}, "
             f"longest IME-module path {_ns(m.get('ime_longest_ps'))} ns"]
    for r in v.get("reasons") or []:
        lines.append(f"- FAIL reason: {r}")
    if report_path:
        lines.append(f"- full report: {report_path}")
    return "\n".join(lines) + "\n"


def is_infra_failure(res: Optional[Dict[str, object]]) -> bool:
    return tj.is_infra_failure(res)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rtl_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ppa-config", default=None,
                    help="JSON config (see DEFAULT_CONFIG); defaults if absent")
    ap.add_argument("--mode", choices=("iter", "soc"), default="iter")
    ap.add_argument("--reparse", action="store_true",
                    help="do not synthesize; re-judge the logs in OUT_DIR "
                         "(iter: fp/ int/ attr/ subdirs)")
    a = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(a.ppa_config)
    res = (judge_iter if a.mode == "iter" else judge_soc)(
        a.rtl_dir, a.out_dir, cfg, reparse=a.reparse)
    print(res["feedback"])
    return 0 if res["verdict"].get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
