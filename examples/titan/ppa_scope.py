"""Round 12 scope / construct checker for the PPA round (pure, no cluster).

"Judges before defendants": the PPA judge is only worth anything if the area
it measures cannot be bought by moving or deleting logic where the judge does
not look.  So before a single synthesis is paid for, the diff the agent left
behind is compared with the tree it started from (the ``--rtl-diff`` r35 hunks
plus the ``--model-diff`` Spike model) and rejected -- as a *scope violation,
which is not a verdict* -- unless it

* touches only the two FU files (``exu/fp/MatrixFPMultiplyPipe.scala``,
  ``exu/int/MatrixMultiplyPipe.scala``) and brand-new files under
  ``exu/fp/`` / ``exu/int/`` (a new file is reachable only from the files
  that are allowed to change, because nothing else in the tree changed);
* leaves every riscv-isa-sim hunk byte-identical (the "model frozen" check:
  a model edit would let the agent redefine what "correct" means);
* adds none of the forbidden constructs: black boxes / external modules,
  memories, clock tricks, annotations, Verilog attributes, printf side
  channels.  Only lines the agent *added* relative to the reference are
  scanned (r35's own text is not re-judged), with comments stripped.

A diff is a ``collect_diff`` output: ``diff --git a/<path> b/<path>`` sections
rooted at the chipyard root.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SATURN_SCALA = "generators/saturn/src/main/scala/"
#: The two files the agent may edit.
ALLOWED_FILES = (SATURN_SCALA + "exu/fp/MatrixFPMultiplyPipe.scala",
                 SATURN_SCALA + "exu/int/MatrixMultiplyPipe.scala")
#: Directories where brand-new files may appear.
ALLOWED_NEW_DIRS = (SATURN_SCALA + "exu/fp/", SATURN_SCALA + "exu/int/")
#: The Spike model lives here (constants.SPIKE_SRC_REL, restated so this
#: module imports nothing from the loop).
MODEL_PREFIX = "toolchains/riscv-tools/riscv-isa-sim"

#: name -> regex on a *comment-stripped* added line.
FORBIDDEN_CODE = {
    "BlackBox/ExtModule": re.compile(
        r"\bBlackBox\b|\bHasBlackBox\w*|\bExtModule\b|\bBlackBoxInline\b"),
    "memory (Mem/SyncReadMem)": re.compile(
        r"\bSyncReadMem\b|(?<![\w.])Mem\s*[\(\[]|\bSeqMem\b|\bMemBase\b"),
    "clock tricks (asClock/withClock/ClockGate)": re.compile(
        r"\basClock\b|\bwithClock\w*|\bClockGate\w*|\bClockDivider\w*|"
        r"\bClock\s*\(\s*\)|\bClockMux\w*"),
    "annotation": re.compile(
        r"\bannotate\b|\bChiselAnnotation\b|\baddAttribute\b|"
        r"\bchisel3\.experimental\.(?:annotate|ChiselAnnotation)"),
    "print/stop side channel": re.compile(r"(?<![\w.])(?:printf|stop)\s*\("),
}
#: Verilog attributes are text inside a string (``"(* keep *)"``), so this
#: one runs on the raw added line.
FORBIDDEN_RAW = {
    "Verilog attribute": re.compile(r"\(\*\s*\w+"),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def split_sections(diff: str) -> Dict[str, str]:
    """{path: section text} of a ``diff --git`` stream, in order.  The path is
    the b-side (new) path, or the a-side one for a deletion."""
    out: Dict[str, str] = {}
    cur: Optional[str] = None
    buf: List[str] = []
    for line in (diff or "").splitlines(keepends=True):
        if line.startswith("diff --git "):
            if cur is not None:
                out[cur] = "".join(buf)
            parts = line.split()
            a = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") \
                else ""
            b = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") \
                else ""
            cur = b or a or f"?{len(out)}"
            buf = [line]
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "".join(buf)
    return out


def _norm(section: str) -> str:
    """A section minus its ``index`` line (the blob hashes follow the content,
    so identical hunks always agree, but an ``index`` line that only differs
    in abbreviation length must not read as an edit)."""
    return "".join(ln for ln in section.splitlines(keepends=True)
                   if not ln.startswith("index "))


def touched_paths(cur: Dict[str, str], ref: Dict[str, str]) -> List[str]:
    """Paths whose section is new, gone or different relative to ``ref``."""
    out = []
    for path in sorted(set(cur) | set(ref)):
        if path not in ref or path not in cur or \
                _norm(cur[path]) != _norm(ref[path]):
            out.append(path)
    return out


def _added_lines(section: str) -> List[str]:
    return [ln[1:].rstrip("\n") for ln in section.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")]


def _strip_comments(lines: Iterable[str]) -> List[str]:
    """Drop ``//`` and ``/* */`` comments (block state carried across the
    given lines; good enough for added lines, which are mostly whole blocks)."""
    out, in_block = [], False
    for ln in lines:
        if not in_block and re.match(r"\s*\*(?:\s|$)", ln):
            # a javadoc-style continuation line whose opener is context, not
            # an added line: Scala code does not start a line with `*`
            out.append("")
            continue
        res, i = [], 0
        while i < len(ln):
            if in_block:
                j = ln.find("*/", i)
                if j < 0:
                    i = len(ln)
                else:
                    in_block, i = False, j + 2
            elif ln.startswith("/*", i):
                in_block, i = True, i + 2
            elif ln.startswith("//", i):
                i = len(ln)
            else:
                res.append(ln[i])
                i += 1
        out.append("".join(res))
    return out


def _strip_strings(line: str) -> str:
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)


def new_added_lines(cur_section: str, ref_section: Optional[str]
                    ) -> List[str]:
    """Added lines of ``cur_section`` that ``ref_section`` does not add
    (multiset difference: moving a line does not make it new)."""
    cur = _added_lines(cur_section)
    if not ref_section:
        return cur
    pool: Dict[str, int] = {}
    for ln in _added_lines(ref_section):
        pool[ln] = pool.get(ln, 0) + 1
    out = []
    for ln in cur:
        if pool.get(ln, 0) > 0:
            pool[ln] -= 1
        else:
            out.append(ln)
    return out


def forbidden_constructs(lines: Sequence[str]) -> List[Dict[str, str]]:
    """[{construct, line}] for every forbidden construct in ``lines``."""
    hits = []
    code = _strip_comments(lines)
    for raw, c in zip(lines, code):
        c_nostr = _strip_strings(c)
        for name, rx in FORBIDDEN_CODE.items():
            if rx.search(c_nostr):
                hits.append({"construct": name, "line": raw.strip()[:160]})
        for name, rx in FORBIDDEN_RAW.items():
            if rx.search(c):
                hits.append({"construct": name, "line": raw.strip()[:160]})
    return hits


def classify_path(path: str, in_ref: bool) -> Optional[str]:
    """None when editing ``path`` is allowed, else the reason."""
    if path.startswith(MODEL_PREFIX):
        return "riscv-isa-sim (Spike model) is frozen"
    if path in ALLOWED_FILES:
        return None
    if not in_ref and path.startswith(ALLOWED_NEW_DIRS) \
            and path.endswith(".scala"):
        return None
    if in_ref and path.startswith(ALLOWED_NEW_DIRS):
        return ("only the two FU files may be edited; new files go in "
                "exu/fp|int/ but existing ones there are not in scope")
    return "outside the allowed files (the two FU .scala files and new " \
           "files under exu/fp|int/)"


def check_scope(cur_diff: str, ref_diff: str) -> Dict[str, object]:
    """The whole check.  ``cur_diff`` is the tree as graded, ``ref_diff`` the
    tree the round started from (r35 RTL hunks + the r31 model hunks, joined).

    Returns {"ok", "violations": [{path, why}], "forbidden": [{path,
    construct, line}], "touched": [...], "new_files": [...],
    "model_frozen": bool, "model_sha256": str, "reasons": [...]}.
    ``ok`` False means *scope violation, not a verdict*."""
    cur, ref = split_sections(cur_diff), split_sections(ref_diff)
    touched = touched_paths(cur, ref)
    violations, forbidden, new_files = [], [], []
    for path in touched:
        why = classify_path(path, in_ref=path in ref)
        if path not in ref and path in cur:
            new_files.append(path)
        if why:
            violations.append({"path": path, "why": why})
            continue
        if path in cur:
            for hit in forbidden_constructs(
                    new_added_lines(cur[path], ref.get(path))):
                forbidden.append(dict(hit, path=path))
    model_cur = "".join(_norm(t) for p, t in cur.items()
                        if p.startswith(MODEL_PREFIX))
    model_ref = "".join(_norm(t) for p, t in ref.items()
                        if p.startswith(MODEL_PREFIX))
    frozen = model_cur == model_ref
    reasons = [f"{v['path']}: {v['why']}" for v in violations]
    reasons += [f"{h['path']}: forbidden construct [{h['construct']}]: "
                f"{h['line']}" for h in forbidden]
    if not frozen and not any(v["path"].startswith(MODEL_PREFIX)
                              for v in violations):
        reasons.append("the Spike model changed (model digest not frozen)")
    return {"ok": not reasons, "violations": violations,
            "forbidden": forbidden, "touched": touched,
            "new_files": new_files, "model_frozen": frozen,
            "model_sha256": sha256_text(model_cur), "reasons": reasons}


def closure_scan(files: Sequence[Sequence[str]]) -> List[Dict[str, str]]:
    """Scan the generated Verilog closure ((name, text) pairs) for the
    constructs that must not survive elaboration: attributes that steer
    synthesis, black-box markers, clock-gate cells.  The Scala check is the
    primary one; this catches what a generator indirection could hide."""
    rx = re.compile(r"\(\*\s*(?:keep|dont_touch|blackbox|black_box|syn_\w+|"
                    r"preserve)\b|\bblackbox\b|\bClockGate\w*|\bICG\w*\b",
                    re.I)
    hits = []
    for name, text in files:
        for m in rx.finditer(text):
            hits.append({"file": name, "match": m.group(0)})
            break
    return hits


def format_scope_rejection(res: Dict[str, object], attempt: int) -> str:
    """Agent-facing text for a scope violation."""
    lines = [f"# Iteration {attempt}: SCOPE VIOLATION (not a verdict)\n",
             "The PPA round only accepts changes confined to "
             "`exu/fp/MatrixFPMultiplyPipe.scala`, "
             "`exu/int/MatrixMultiplyPipe.scala` and new files under "
             "`exu/fp/` or `exu/int/`; the Spike model is frozen and a fixed "
             "list of constructs is forbidden. Nothing was synthesised or "
             "simulated for this tree. Undo the following and finish again:\n"]
    lines += [f"- {r}" for r in res.get("reasons") or []]
    return "\n".join(lines) + "\n"
