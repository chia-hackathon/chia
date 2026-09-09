"""The lint gate: reject illegal or dishonest HARCOM before anything compiles.

Two kinds of failure sit between a generated predictor and a usable score, and
they want different treatment.

**Illegal C++.** HARCOM's ``val``/``reg``/``arr``/``ram`` are opaque -- they
have no ``operator bool`` and no way to read the value out -- so ``if (pred)``
is a compile error, not a wrong answer.  The compiler already catches these.
What it does not do is explain them: a missing ``select()`` surfaces as pages of
template substitution failure, which is a poor thing to hand back to an agent
as a repair prompt.  :func:`prelint` catches the common shapes textually first
and says what to write instead.

**Dishonest C++.** This is the one that matters.  A predictor that reaches
around HARCOM -- ``harcom_superuser`` to read a private value, a ``#pragma``
to silence the warnings the design is supposed to be judged under -- still
compiles, still runs, and still produces a VFS score.  It is just no longer a
score about hardware.  Those cost the loop nothing to check and everything to
miss, so they are hard failures with no repair round offered: the design is
withdrawn, not fixed.

The gate never calls an LLM.  It is the framework's ruling, not the agent's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LintFinding:
    """One violation, with the line that triggered it and what to do instead."""
    rule: str
    line_no: int
    line: str
    message: str
    fix: str
    fatal: bool          # True = withdraw the design; False = offer a repair round

    def render(self) -> str:
        tag = "FATAL" if self.fatal else "error"
        return (f"{tag} [{self.rule}] line {self.line_no}: {self.message}\n"
                f"    {self.line.strip()}\n"
                f"  fix: {self.fix}")


@dataclass
class LintResult:
    findings: list[LintFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def withdrawn(self) -> bool:
        """True when at least one finding is a cheat rather than a mistake.

        A withdrawn design is not repaired.  Offering a repair round on
        ``harcom_superuser`` would be teaching the agent that the rule is
        negotiable.
        """
        return any(f.fatal for f in self.findings)

    def render(self) -> str:
        return "\n".join(f.render() for f in self.findings)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
#
# Each rule is (name, compiled pattern, message, fix, fatal).  Patterns run
# against source with comments and string literals blanked out, so a rule name
# mentioned in a comment does not trip its own rule.

_RULES: list[tuple[str, re.Pattern, str, str, bool]] = [
    # -- cheats: withdraw ---------------------------------------------------
    ("superuser",
     re.compile(r"\bharcom_superuser\b"),
     "harcom_superuser has friend access to val's private members. Using it "
     "reads a value the hardware could not read, so the energy and latency "
     "HARCOM reports stop describing anything buildable.",
     "There is no legitimate use in a predictor. Express the decision with "
     "select() instead.",
     True),
    ("warning-suppression",
     re.compile(r"#\s*pragma\s+GCC\s+diagnostic\s+(ignored|push|pop)"
                r"|#\s*pragma\s+clang\s+diagnostic"
                r"|\[\[\s*gnu::\s*diagnose"),
     "The design is scored under -Wall -Wextra -Werror. Suppressing a "
     "diagnostic hides the thing being measured.",
     "Fix the warning. [[maybe_unused]] on a genuinely unused parameter is "
     "allowed and is not a suppression pragma.",
     True),
    ("raw-cast",
     re.compile(r"\breinterpret_cast\s*<|\bmemcpy\s*\(|\bstd::bit_cast\s*<"),
     "Casting or copying the bytes of an opaque type reads a value the "
     "hardware could not read.",
     "Use select(), concat(), or the arr/ram accessors.",
     True),
    ("reads-outside-world",
     re.compile(r"#\s*include\s*<(fstream|filesystem)>"
                r"|\bstd::(cin|ifstream|ofstream|filesystem)\b"
                r"|\b(fopen|freopen|popen|system|getenv|open)\s*\("),
     "A predictor that opens a file or reads the environment can be handed the "
     "answer out of band, and the score stops being about prediction.",
     "A predictor sees only what predict1/predict2/update_condbr are passed.",
     True),
    # std::cerr is fine -- the shipped tage.hpp prints its history lengths at
    # construction, and the harness never reads stderr.  stdout is not: cbp.cpp
    # writes the CSV counter line there, and run_cbp parses it.  A stray print
    # does not corrupt a design, it corrupts the *measurement*, which is worse
    # because it still looks like a number.
    ("writes-stdout",
     re.compile(r"\bstd::cout\b|\b(printf|puts|fputs\s*\(\s*[^,]+,\s*stdout)\s*\("),
     "cbp.cpp writes the per-trace counter line to stdout and the harness "
     "parses it. Anything else printed there is parsed as measurement.",
     "Use std::cerr for diagnostics; the harness ignores it.",
     True),

    # -- mistakes: repairable ------------------------------------------------
]

_COMMENT_OR_STRING = re.compile(
    r"//[^\n]*"                    # line comment
    r"|/\*.*?\*/"                  # block comment
    r"|\"(?:\\.|[^\"\\])*\""       # string literal
    r"|'(?:\\.|[^'\\])*'",         # char literal
    re.S,
)


def _blank(src: str) -> str:
    """Blank comments and literals, preserving line numbers and length.

    Newlines survive so a finding's line number still points at real source.
    """
    def repl(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return _COMMENT_OR_STRING.sub(repl, src)


# Declarations that introduce an opaque name.  HARCOM's own headers declare
# them regularly enough that a pattern is sound: the type is always written
# out, because there is no other way to give a register a width.
_OPAQUE_DECL = re.compile(
    r"\b(?:val|reg|valt|arr|ram)\s*<[^;{}()]*?>\s*&?\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?=[;,=\[){])"
)
# `auto x = select(...)` and `auto x = something.fo1()` are opaque too, and are
# the two idioms the shipped predictors actually use.
_OPAQUE_AUTO = re.compile(
    r"\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*?"
    r"\b(?:select|concat|hard)\s*\(|\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*?\.fo1\s*\("
)

_CONDITION = re.compile(r"\b(if|while)\s*\(")


def opaque_names(source: str) -> set[str]:
    """Identifiers declared with a HARCOM type in ``source``.

    Used to decide whether a condition depends on a value the hardware cannot
    read.  Deliberately over-narrow: it only recognises explicit declarations,
    so an opaque value reached through a helper's return type is missed and the
    compiler catches it instead.  A missed violation costs one build; a
    false one would reject a legal design, which is worse.
    """
    blanked = _blank(source)
    names = {m.group(1) for m in _OPAQUE_DECL.finditer(blanked)}
    for m in _OPAQUE_AUTO.finditer(blanked):
        names.add(m.group(1) or m.group(2))
    return {n for n in names if n}


def _condition_text(line: str, start: int) -> str:
    """The text inside the parentheses opened at ``start``, to end of line.

    Conditions in these headers do not span lines; one that did would simply
    not be checked, which is the safe direction.
    """
    depth = 0
    for i in range(start, len(line)):
        if line[i] == "(":
            depth += 1
        elif line[i] == ")":
            depth -= 1
            if depth == 0:
                return line[start + 1:i]
    return line[start + 1:]


def branch_on_value(source: str) -> list[LintFinding]:
    """Flag `if`/`while` conditions that depend on an opaque value.

    This is the mistake the proposal names: HARCOM forbids branching on a
    value, so an ordinary ``if`` must become a multiplexer.  The compiler does
    reject it -- ``val`` has no ``operator bool`` -- but as a wall of template
    substitution failure.  Naming the variable and the fix is the difference
    between a repair round that works and one that guesses.
    """
    names = opaque_names(source)
    if not names:
        return []
    word = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b")
    findings = []
    lines = source.splitlines()
    for i, bline in enumerate(_blank(source).splitlines()):
        for m in _CONDITION.finditer(bline):
            cond = _condition_text(bline, m.end() - 1)
            hit = word.search(cond)
            if not hit:
                continue
            findings.append(LintFinding(
                rule="branch-on-value", line_no=i + 1,
                line=lines[i] if i < len(lines) else "",
                message=f"`{hit.group(1)}` is a HARCOM value, and HARCOM values "
                        f"are opaque -- there is no operator bool, so control "
                        f"flow cannot depend on one.",
                fix="select(cond, a, b) muxes between two values; "
                    "execute_if(cond, [&]{ ... }) guards a side effect. Both "
                    "take a val<1> condition.",
                fatal=False))
    return findings


def prelint(source: str, *, struct_name: str | None = None) -> LintResult:
    """Check ``source`` before handing it to a compiler.

    ``struct_name``, when given, is additionally required to exist and to
    derive from ``predictor`` -- the build passes it as ``-DPREDICTOR=``, and a
    name that is not there fails at link time with a message about
    ``branch_predictor`` that says nothing about the real mistake.
    """
    result = LintResult()
    blanked = _blank(source)
    lines = source.splitlines()
    blanked_lines = blanked.splitlines()

    for name, pattern, message, fix, fatal in _RULES:
        for i, bline in enumerate(blanked_lines):
            if pattern.search(bline):
                result.findings.append(LintFinding(
                    rule=name, line_no=i + 1,
                    line=lines[i] if i < len(lines) else "",
                    message=message, fix=fix, fatal=fatal))

    result.findings.extend(branch_on_value(source))

    if struct_name is not None:
        decl = re.search(
            rf"\bstruct\s+{re.escape(struct_name)}\b\s*:\s*([^{{]*)\{{", blanked)
        if not decl:
            result.findings.append(LintFinding(
                rule="missing-predictor-base", line_no=1,
                line=f"(no `struct {struct_name} : predictor` found)",
                message=f"The build compiles with -DPREDICTOR={struct_name}<>, "
                        f"so a struct of exactly that name must exist.",
                fix=f"Declare `template<...> struct {struct_name} : predictor "
                    f"{{ ... }};` with defaults for every template parameter.",
                fatal=False))
        elif "predictor" not in decl.group(1):
            result.findings.append(LintFinding(
                rule="missing-predictor-base", line_no=1 + blanked[:decl.start()].count("\n"),
                line=decl.group(0).strip(),
                message=f"{struct_name} does not derive from `predictor`, so it "
                        f"does not implement predict1/predict2/update_condbr.",
                fix="struct {name} : predictor {{ ... }}".format(name=struct_name),
                fatal=False))

    return result


# The four pure-virtual members of cbp.hpp's `predictor`, plus update_cycle.
REQUIRED_METHODS = (
    "predict1", "reuse_predict1", "predict2", "reuse_predict2",
    "update_condbr", "update_cycle",
)


def missing_methods(source: str) -> list[str]:
    """Which of :data:`REQUIRED_METHODS` the source never defines.

    A pure-virtual left unimplemented is an abstract class, which fails at the
    ``branch_predictor pred;`` definition in ``cbp.cpp`` -- far from the source
    of the mistake.  Naming them here costs nothing.
    """
    blanked = _blank(source)
    return [m for m in REQUIRED_METHODS
            if not re.search(rf"\b{m}\s*\(", blanked)]
