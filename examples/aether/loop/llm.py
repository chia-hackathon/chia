"""The optimizer agent: one resumable Claude Code session across all iterations.

A single ``ClaudeCodeLLM`` instance is built once and reused, so every feedback
call ``--resume``s the same conversation — the agent remembers which strategies
it already tried and what they measured. It acts on the kernel through a
``BashTool`` deployed into the riscv_build container.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
from pathlib import Path

from chia.base.ChiaFunction import get
from chia.base.llm_call import QueryResult
from chia.base.tools.BashTool import BashTool
from chia.models.claude import ClaudeCodeLLM

from constants import (
    AGENT_WORK_DIR,
    LLM_EXTRA_CLI_ARGS,
    LLM_MODEL,
    LLM_RESOURCE,
    LLM_SYSTEM_MESSAGE,
    LLM_TIMEOUT_SECONDS,
    MAX_FEEDBACK_CHARS,
)
from context import CONFIG, K
import db
from nodes import Measurement

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
logger = logging.getLogger("aether.llm")

#: Name the DB tool registers under; the agent sees `<name>_query` / `<name>_schema`.
DB_TOOL_NAME = "aether_db"

#: The read-only experiment-DB tool is ON by default; AETHER_DB_TOOL=0 disables
#: it (e.g. when Ray cannot host another MCP actor).
DB_TOOL_ENABLED = os.environ.get("AETHER_DB_TOOL", "1") not in ("0", "false", "no")

_db_tool = None          # spawned lazily on the first agent turn
_db_tool_failed = False


def db_tool():
    """The agent's read-only SQL view of its own experiment history.

    Spawned once per driver process (CHIA MCP tools are Ray actors, so this
    costs an actor, not a per-turn dispatch) and stopped at interpreter exit.
    Returns None when disabled or when spawning fails — a missing history tool
    must never take the optimization loop down with it.
    """
    global _db_tool, _db_tool_failed
    if not DB_TOOL_ENABLED or _db_tool_failed:
        return None
    if _db_tool is None:
        try:
            _db_tool = db.spawn_query_tool(DB_TOOL_NAME)
            atexit.register(stop_db_tool)
        except Exception as e:                      # noqa: BLE001
            _db_tool_failed = True
            logger.warning("experiment-DB query tool unavailable (%s: %s) — "
                           "continuing without it", type(e).__name__, e)
            return None
    return _db_tool


def stop_db_tool() -> None:
    """Tear the DB tool's MCP actor down. Idempotent; safe at exit."""
    global _db_tool
    tool, _db_tool = _db_tool, None
    if tool is not None:
        try:
            tool.stop()
        except Exception as e:                      # noqa: BLE001
            logger.warning("failed to stop %s: %s", DB_TOOL_NAME, e)


def _tools(kernel_bash: BashTool) -> list:
    """Every tool the agent gets this turn."""
    return [t for t in (kernel_bash, db_tool()) if t is not None]


def _fence(rel: str) -> str:
    """Markdown code-fence language for the kernel file (asm vs C)."""
    return "asm" if rel.endswith(".S") or rel.endswith(".s") else "c"


def _load_prompt(name: str, **subs: str) -> str:
    """Read prompts/<name>, substituting ${KEY} (not str.format — the text
    contains literal braces in code snippets)."""
    text = (PROMPTS_DIR / name).read_text()
    for key, val in subs.items():
        text = text.replace("${" + key + "}", val)
    return text


def task_prompt() -> str:
    """The optimization brief for the SELECTED kernel.

    The template in prompts/optimize.md is generic; everything benchmark- or
    target-specific (contract, self-check, roofline arithmetic, the
    already-ruled-out table) comes from the registry entry.
    """
    k = K()
    return _load_prompt(
        "optimize.md",
        WORK_DIR=AGENT_WORK_DIR,
        KERNEL_REL=k.kernel_rel,
        SIM_CONFIG=CONFIG(),
        OBJECTIVE=k.objective,
        CHECK_DESC=k.check_desc,
        CONTRACT=k.contract,
        TARGET_NOTES=k.target_notes,
        KERNEL_NOTES=k.notes,
    )


def make_llm() -> ClaudeCodeLLM:
    """Build the optimizer once; reuse it so sessions thread across iterations."""
    return ClaudeCodeLLM(
        model=LLM_MODEL,
        system_message=LLM_SYSTEM_MESSAGE,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        logging_name="aether_optimizer",
        resume_session=True,
        projects_cwd=None,
        extra_cli_args=LLM_EXTRA_CLI_ARGS,
    )


def _run(llm: ClaudeCodeLLM, prompt: str, kernel_bash: BashTool) -> QueryResult:
    return get(llm.prompt.options(resources={"llm": LLM_RESOURCE})
               .chia_remote(llm, prompt, _tools(kernel_bash)))


def first_turn(llm, kernel_bash: BashTool, baseline: Measurement,
               kernel: bytes | None = None) -> QueryResult:
    """Iteration 1: the task plus the measured baseline to beat.

    *kernel* (the pristine source, when given) is inlined into the prompt so
    the agent has it in hand even before its first `kernel_bash` call — it
    still has to use `kernel_bash` to write an edit back, but it never has to
    "find" the file just to read it.
    """
    kernel_rel = K().kernel_rel
    instret = (f" (minstret = {baseline.instret})"
               if baseline.instret is not None else "")
    intro = (
        f"The unmodified kernel currently measures **{baseline.cycles} cycles**"
        f"{instret} and passes its self-check. "
        f"That is the number to beat.\n\n"
    )
    body = task_prompt()
    if db_tool() is not None:
        body += "\n" + db.QUERY_TOOL_BLURB.format(name=DB_TOOL_NAME) + "\n"
    if kernel is not None:
        body += (
            f"\n# Current contents of `{kernel_rel}`\n\n"
            "This is exactly what `cat "
            f"{AGENT_WORK_DIR}/{kernel_rel}` (via the `kernel_bash` tool) "
            "would show you right now:\n\n"
            f"```{_fence(kernel_rel)}\n"
            f"{kernel.decode('utf-8', errors='replace')}\n```\n"
        )
    return _run(llm, intro + body, kernel_bash)


def next_turn(llm, kernel_bash: BashTool, feedback: str) -> QueryResult:
    """Later iterations: the measurement from the agent's last edit."""
    return _run(llm, feedback, kernel_bash)


def _ledger(history: list[dict] | None) -> str:
    """Compact scoreboard of every attempt so far — the agent sees its own
    experiment log, not just the last number."""
    rows = [h for h in (history or []) if h.get("iter", 0) > 0]
    if not rows:
        return ""
    lines = ["\n\nYour attempts so far (cycles is the objective; "
             "instructions retired is NOT):",
             "",
             "| iter | cycles | instret | result |",
             "|---|---|---|---|"]
    base = next((h.get("cycles") for h in (history or [])
                 if h.get("iter") == 0), None)
    for h in rows:
        c = h.get("cycles")
        vs = (f"{(base - c) / base * 100:+.1f}% vs baseline"
              if c and base else h.get("kind", ""))
        lines.append(
            f"| {h['iter']} | {c or 'n/a'} | {h.get('instret') or 'n/a'} | "
            f"{h.get('kind') if h.get('kind') != 'ok' else vs} |")
    return "\n".join(lines)


# Lines that look like simulator/harness framework noise rather than
# something the benchmark itself printed (Verilator/cospike/HTIF banners,
# core trace lines, etc.) -- filtered out of the passing-run stdout tail.
_SIM_NOISE_RE = re.compile(
    r"verilator|cospike|htif|chipyard|firesim|dromajo|plusargs|"
    r"\bvcd\b|copyright|\$finish|uvm_|dpi|elf loaded|"
    r"core\s+\d+:|mstatus|mcause|\bpriv\b|\bcsr\b|riscv-|spike",
    re.IGNORECASE,
)


def _benchmark_stdout_tail(sim_log: str, n_lines: int = 60,
                           limit: int = 4096) -> str:
    """Best-effort extraction of the benchmark's own printed lines from the
    raw simulator log for a PASSING run: take the last `n_lines` lines and
    drop ones that look like simulator/harness framework noise, then cap the
    result to `limit` bytes. If filtering leaves nothing, fall back to the
    raw (non-blank) tail lines so a probe is never silently dropped."""
    lines = (sim_log or "").splitlines()[-n_lines:]
    kept = [ln for ln in lines if ln.strip() and not _SIM_NOISE_RE.search(ln)]
    if not kept:
        kept = [ln for ln in lines if ln.strip()]
    text = "\n".join(kept)
    if len(text.encode("utf-8", errors="replace")) > limit:
        text = f"...[truncated]...\n{text[-limit:]}"
    return text


def format_feedback(m: Measurement, best: int | None, baseline: int,
                    build_stderr: str = "", sim_log: str = "",
                    history: list[dict] | None = None,
                    best_kernel: bytes | None = None) -> str:
    """Turn a Measurement into the next prompt."""
    def tail(text: str, n: int = MAX_FEEDBACK_CHARS) -> str:
        text = text or ""
        return text if len(text) <= n else f"...[truncated]...\n{text[-n:]}"

    if m.kind == "build_failure":
        return ("Your kernel did not compile. Fix it.\n\n"
                f"```\n{tail(build_stderr or m.detail)}\n```")

    if m.kind == "incorrect":
        got = f"It ran for mcycle={m.cycles}, " if m.cycles else "It ran, "
        return (f"Your kernel compiled but **failed the self-check** — the result "
                f"is wrong. {got}but a wrong answer scores nothing.\n\n"
                f"Simulator output:\n```\n{tail(sim_log, 3000)}\n```\n\n"
                "Find the correctness bug and fix it, then keep optimizing.")

    if m.kind == "sim_failure":
        return (f"The run did not produce a usable measurement: {m.detail}\n\n"
                f"```\n{tail(sim_log, 3000)}\n```")

    k = K()
    delta = (baseline - m.cycles) / baseline * 100
    verdict = "FASTER than" if delta > 0 else ("SLOWER than" if delta < 0
                                               else "identical to")
    instret = (f" (instructions retired = {m.instret})"
               if m.instret is not None else "")
    line = (f"Measured: **{m.cycles} cycles**{instret}, "
            f"self-check passed. That is {abs(delta):.1f}% {verdict} the "
            f"{baseline}-cycle baseline")
    if k.roofline_cycles:
        over = (m.cycles - k.roofline_cycles) / k.roofline_cycles * 100
        line += (f", and {over:+.1f}% above the ~{k.roofline_cycles}-cycle "
                 f"DLEN roofline")
    line += "."

    regressed = best is not None and m.cycles >= best
    if regressed:
        line += (f" Your best so far is still **{best}** — this change did not "
                 f"help, so it is discarded.")
    else:
        line += " **That is your best so far.**"

    line += _ledger(history)

    if regressed and best_kernel is not None:
        line += (
            "\n\nThe workspace still holds your (worse) last edit. **Restore the "
            f"best-known kernel below into `{AGENT_WORK_DIR}/{k.kernel_rel}` via "
            "`kernel_bash` first**, then apply exactly one new idea on top of it:\n\n"
            f"```\n{best_kernel.decode('utf-8', errors='replace')}\n```\n")

    stdout_tail = _benchmark_stdout_tail(sim_log)
    if stdout_tail:
        line += (
            "\n\n=== benchmark stdout (tail) ===\n"
            "(your `printf` probe output, if any, is here -- prefix each "
            "probe line with `PROBE ` to make it easy to find)\n"
            f"```\n{stdout_tail}\n```"
        )

    line += (
        "\n\nNext turn: first state your `HYPOTHESIS:` / `CHANGE:` / `EXPECTED:` "
        "lines (and, if last turn's EXPECTED missed, say what that taught you "
        "about the machine), then make **one** isolated edit. Remember: only "
        "beats in the vector pipes and un-hidden pipeline drain cost cycles — "
        "cutting instruction count for its own sake is not the objective."
    )
    return line
